using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// ONNX Gather operation: take slices from data along an axis using indices.
/// output[i][j][k] = data[index[i][j][k]][j][k] (for axis=0)
///
/// Simplified version for common cases: 1D indices gathering along axis 0.
/// For the full ONNX Gather spec with arbitrary axis, use the params-buffer variant.
/// </summary>
public class GatherKernel : IDisposable
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int>? _gatherAxis0Kernel;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _gatherElementsKernel;
    private MemoryBuffer1D<int, Stride1D.Dense>? _lastElementsParams;
    private readonly List<MemoryBuffer1D<int, Stride1D.Dense>> _oldElementsParams = new();

    public GatherKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Gather along axis 0: output[i, :] = data[indices[i], :].
    /// data: [dataRows, innerSize], indices: [numIndices], output: [numIndices, innerSize].
    /// One thread per output element.
    /// </summary>
    private static void GatherAxis0Impl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<int, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> output,
        int innerSize, int dummy)
    {
        int outRow = idx / innerSize;
        int col = idx % innerSize;
        int srcRow = indices[outRow];
        int srcIdx = srcRow * innerSize + col;
        output[idx] = (srcIdx >= 0 && srcIdx < data.Length) ? data[srcIdx] : 0f;
    }

    /// <summary>
    /// Gather along axis 0. data: [dataRows, innerSize], indices: [numIndices].
    /// Output: [numIndices, innerSize].
    /// </summary>
    public void GatherAxis0(ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<int, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> output,
        int numIndices, int innerSize)
    {
        EnsureLoaded();
        _gatherAxis0Kernel!(numIndices * innerSize, data, indices, output, innerSize, 0);
    }

    // ── Float-index variant for NLP models (token IDs stored as float) ──

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int>? _gatherAxis0FloatKernel;

    /// <summary>
    /// Gather along axis 0 with float indices (cast to int internally).
    /// Used for embedding lookups where token IDs are stored as float32.
    /// data: [dataRows, innerSize], indices: [numIndices] (float), output: [numIndices, innerSize].
    /// </summary>
    private static void GatherAxis0FloatImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> output,
        int innerSize, int dataRows)
    {
        int outRow = idx / innerSize;
        int col = idx % innerSize;
        int srcRow = (int)indices[outRow];
        if (srcRow < 0) srcRow += dataRows;
        int srcIdx = srcRow * innerSize + col;
        output[idx] = (srcRow >= 0 && srcRow < dataRows && srcIdx >= 0 && srcIdx < data.Length)
            ? data[srcIdx] : 0f;
    }

    /// <summary>
    /// Gather along axis 0 with float indices.
    /// data: [dataRows, innerSize], indices: [numIndices] (float token IDs).
    /// Output: [numIndices, innerSize].
    /// </summary>
    public void GatherAxis0Float(ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> output,
        int numIndices, int innerSize, int dataRows)
    {
        EnsureLoaded();
        _gatherAxis0FloatKernel!(numIndices * innerSize, data, indices, output, innerSize, dataRows);
    }

    // ── Generic-axis Gather with float indices (GPU) ──

    // ── Generic-axis Gather with float indices (GPU) ──

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _gatherGenericFloatKernel;
    private MemoryBuffer1D<int, Stride1D.Dense>? _lastGenericParams;
    // Old per-call params buffers, retired here and disposed in Dispose() — NOT inline. On WebGPU/WebGL a
    // dispatch batches into an un-submitted command encoder; destroying the previous call's params buffer
    // inline (while its Gather dispatch is still pending in the batch) makes the later Queue.Submit fail
    // "[Buffer] used in submit while destroyed" (this is the DAv3-518 RoPE node-177 bug — the RoPE dynamic-shape
    // subgraph issues several GatherGenericFloat calls that batch together). Same deferred-disposal pattern as
    // ElementWiseKernels.BroadcastBinaryOpND's _oldStridesBufs. Each call still gets a FRESH buffer (no
    // write-after-read hazard from reusing one), we just free the old ones at a safe point.
    private readonly List<MemoryBuffer1D<int, Stride1D.Dense>> _oldGenericParams = new();

    /// <summary>
    /// General Gather along any axis with float indices.
    /// params[0]=numIdx, params[1]=innerSize, params[2]=outerSize, params[3]=axisSize.
    /// Total output elements = outerSize * numIdx * innerSize.
    /// </summary>
    private static void GatherGenericFloatImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int numIdx = p[0];
        int innerSize = p[1];
        int axisSize = p[3];
        int sliceSize = numIdx * innerSize;

        int outerIdx = idx / sliceSize;
        int rem = idx % sliceSize;
        int idxIdx = rem / innerSize;
        int innerIdx = rem % innerSize;

        int srcRow = (int)indices[idxIdx];
        if (srcRow < 0) srcRow += axisSize;

        int srcLinear = (outerIdx * axisSize + srcRow) * innerSize + innerIdx;
        output[idx] = (srcRow >= 0 && srcRow < axisSize && srcLinear >= 0 && srcLinear < data.Length)
            ? data[srcLinear] : 0f;
    }

    /// <summary>
    /// Gather along arbitrary axis with float indices.
    /// data: [..., axisSize, ...], indices: [numIdx] (float).
    /// Output: [..., numIdx, ...] with axis dimension replaced.
    /// </summary>
    public void GatherGenericFloat(ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> output,
        int numIdx, int innerSize, int outerSize, int axisSize)
    {
        EnsureLoaded();
        var packed = new[] { numIdx, innerSize, outerSize, axisSize };
        ArrayView1D<int, Stride1D.Dense> paramsView;
        if (Graph.GraphExecutor.UseCaptureParamSlots)
        {
            // CUDA-graph capture: stable per-forward slot (no per-call cuMemAlloc — illegal mid-capture).
            paramsView = CaptureParamArena.Shared(_accelerator).RentStableSlot(packed);
        }
        else
        {
            // Retire the previous buffer for deferred disposal (see _oldGenericParams) instead of destroying it
            // inline — its Gather dispatch may still be pending in an un-submitted WebGPU command batch.
            if (_lastGenericParams != null) _oldGenericParams.Add(_lastGenericParams);
            _lastGenericParams = _accelerator.Allocate1D(packed);
            paramsView = _lastGenericParams.View;
        }
        _gatherGenericFloatKernel!(outerSize * numIdx * innerSize, data, indices, output, paramsView);
    }

    /// <summary>
    /// ONNX GatherElements: an ELEMENT-WISE gather along one axis.
    /// <c>output[i,j,k] = data[i, indices[i,j,k], k]</c> for axis 1, and so on. Indices have the SAME
    /// shape as the output, which is what distinguishes this from Gather - Gather takes whole slices,
    /// GatherElements picks one element per output position.
    /// </summary>
    /// <remarks>
    /// ⚠️ This exists because the operator had NO GPU path at all. It computed the gather only when
    /// BOTH data and indices happened to be readable as host values, and otherwise fell through to
    /// <c>output = data</c> - copying the input and IGNORING the indices. That is a silently wrong answer
    /// of exactly the right shape, and relative-position attention (ZipVoice's text encoder uses four of
    /// them) computes its indices at runtime, so the wrong branch was the only one that ever ran there.
    /// <para>
    /// Flattened to [outer, axis, inner] so the kernel needs four scalars instead of a stride array:
    /// outer is the product of the dims before the axis, inner the product of those after. Non-axis dims
    /// agree between data and indices by the ONNX spec, so only the axis length differs.
    /// </para>
    /// </remarks>
    private static void GatherElementsImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int idxAxis = p[0];    // axis length of indices/output
        int inner = p[1];      // product of dims after the axis
        int dataAxis = p[2];   // axis length of data (may differ from idxAxis)

        int plane = idxAxis * inner;
        int outerIdx = idx / plane;
        int rem = idx - outerIdx * plane;
        int innerIdx = rem - (rem / inner) * inner;

        int g = (int)indices[idx];
        if (g < 0) g += dataAxis;

        int srcLinear = (outerIdx * dataAxis + g) * inner + innerIdx;
        output[idx] = (g >= 0 && g < dataAxis && srcLinear >= 0 && srcLinear < data.Length)
            ? data[srcLinear] : 0f;
    }

    /// <summary>
    /// GatherElements along <paramref name="axis"/>, with indices held on the GPU as floats.
    /// </summary>
    /// <param name="outer">Product of the output dims BEFORE the axis.</param>
    /// <param name="idxAxis">Length of the axis in indices/output.</param>
    /// <param name="inner">Product of the output dims AFTER the axis.</param>
    /// <param name="dataAxis">Length of the axis in data.</param>
    public void GatherElements(ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> output,
        int outer, int idxAxis, int inner, int dataAxis)
    {
        EnsureLoaded();
        var packed = new[] { idxAxis, inner, dataAxis, outer };
        ArrayView1D<int, Stride1D.Dense> paramsView;
        if (Graph.GraphExecutor.UseCaptureParamSlots)
        {
            // CUDA-graph capture: a stable per-forward slot, since cuMemAlloc mid-capture is illegal.
            paramsView = CaptureParamArena.Shared(_accelerator).RentStableSlot(packed);
        }
        else
        {
            // Retire rather than dispose: the dispatch may still be pending in an un-submitted WebGPU
            // command batch, and freeing a buffer it reads makes the GPU read zeros.
            if (_lastElementsParams != null) _oldElementsParams.Add(_lastElementsParams);
            _lastElementsParams = _accelerator.Allocate1D(packed);
            paramsView = _lastElementsParams.View;
        }
        _gatherElementsKernel!(outer * idxAxis * inner, data, indices, output, paramsView);
    }

    private void EnsureLoaded()
    {
        _gatherAxis0Kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int>(GatherAxis0Impl);
        _gatherAxis0FloatKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int>(GatherAxis0FloatImpl);
        _gatherGenericFloatKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(GatherGenericFloatImpl);
        _gatherElementsKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(GatherElementsImpl);
    }

    public void Dispose()
    {
        _lastGenericParams?.Dispose();
        _lastGenericParams = null;
        foreach (var b in _oldGenericParams) b.Dispose();
        _oldGenericParams.Clear();
        _lastElementsParams?.Dispose();
        _lastElementsParams = null;
        foreach (var b in _oldElementsParams) b.Dispose();
        _oldElementsParams.Clear();
    }
}
