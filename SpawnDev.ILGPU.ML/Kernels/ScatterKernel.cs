using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU scatter for ONNX <c>ScatterElements</c> and <c>ScatterND</c>, written as a GATHER.
/// </summary>
/// <remarks>
/// ⚠️ Neither operator had a GPU path. Both read their inputs with <c>TryGetInputValues</c>, which returns
/// COMPILE-TIME CONSTANTS ONLY, and when that returned null they copied <c>data</c> to the output and
/// RETURNED - silently discarding every update. ScatterND's own comment called it "fall back to identity".
/// A real model computes its indices and updates at runtime, so that was the only path either ever took:
/// both were no-ops emitting a correctly shaped, plausible tensor. Found by
/// <c>tools/audit-operator-support.cs</c>.
///
/// <para>
/// ⚠️ THESE ARE INVERTED ON PURPOSE. The obvious kernel is one thread per UPDATE writing
/// <c>output[computedIndex]</c>, and it works on CUDA, OpenCL, CPU, WebGPU and Wasm - but NOT on WebGL,
/// where compute is emulated with transform feedback and an invocation can only write its OWN output slot.
/// MEASURED: the scatter-style kernel passed 5 of 6 backends and put an update value on an untouched
/// element on WebGL. Removing the mid-loop early returns did not help, because the problem is the
/// data-dependent WRITE itself, not the branch.
/// </para>
/// <para>
/// So each thread owns one OUTPUT element and SEARCHES for an index that targets it. Every write is to the
/// thread's own slot, which every backend can express, and no backend-specific branching enters the
/// library. Cost: ScatterElements searches only along the index tensor's scatter axis (small - it is the
/// number of updates per row, not the whole tensor); ScatterND searches all update tuples, which is
/// O(numUpdates) per output element and is the one to revisit if a model ever scatters a large index set.
/// </para>
/// <para>
/// Duplicate indices resolve to the LAST match, matching onnxruntime. ONNX leaves the order unspecified
/// for <c>reduction="none"</c>, so this is within spec and, unlike a racing scatter, it is deterministic.
/// Reductions (add/mul/min/max) are rejected by the operators rather than silently treated as "none".
/// </para>
/// </remarks>
public class ScatterKernel : IDisposable
{
    /// <summary>Highest tensor rank the packed parameter layout supports.</summary>
    public const int MaxRank = 8;

    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _scatterElementsKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _scatterNDKernel;

    private MemoryBuffer1D<int, Stride1D.Dense>? _lastParams;
    private readonly List<MemoryBuffer1D<int, Stride1D.Dense>> _oldParams = new();

    public ScatterKernel(Accelerator accelerator) => _accelerator = accelerator;

    // ScatterElements params:
    //   [0]=rank [1]=axis [2]=idxAxisLen
    //   [3..]=data dims  [11..]=idx dims  [19..]=data strides  [27..]=idx strides
    private const int EDataDim = 3;
    private const int EIdxDim = EDataDim + MaxRank;
    private const int EDataStride = EIdxDim + MaxRank;
    private const int EIdxStride = EDataStride + MaxRank;
    private const int EParamLen = EIdxStride + MaxRank;

    /// <summary>One thread per OUTPUT element; finds the index entry (if any) that targets it.</summary>
    private static void ScatterElementsImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> updates,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int rank = p[0], axis = p[1], idxAxisLen = p[2];

        // Walk the non-axis dims to find where this output element sits in the INDEX tensor. An index
        // tensor may be smaller than data (a subset scatter), so a coordinate outside it means this
        // element can never be targeted.
        int basePos = 0;
        int reachable = 1;
        int axisCoord = 0;
        for (int d = 0; d < rank; d++)
        {
            int coord = (idx / p[EDataStride + d]) % p[EDataDim + d];
            if (d == axis) { axisCoord = coord; continue; }
            if (coord >= p[EIdxDim + d]) reachable = 0;
            else basePos += coord * p[EIdxStride + d];
        }

        float value = output[idx];   // the caller has already copied `data` here
        if (reachable == 1)
        {
            for (int j = 0; j < idxAxisLen; j++)
            {
                int pos = basePos + j * p[EIdxStride + axis];
                int target = (int)indices[pos];
                if (target < 0) target += p[EDataDim + axis];
                if (target == axisCoord) value = updates[pos];   // last match wins, as ORT does
            }
        }
        output[idx] = value;
    }

    /// <summary>ScatterElements along <paramref name="axis"/>. Output must already hold a copy of data.</summary>
    public void ScatterElements(ArrayView1D<float, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> updates,
        ArrayView1D<float, Stride1D.Dense> output,
        int[] indicesShape, int[] dataShape, int axis, int outputCount)
    {
        EnsureLoaded();
        int rank = Math.Min(dataShape.Length, MaxRank);
        var packed = new int[EParamLen];
        packed[0] = rank;
        packed[1] = axis;
        packed[2] = axis < indicesShape.Length ? indicesShape[axis] : 0;

        for (int d = 0; d < rank; d++)
        {
            packed[EDataDim + d] = dataShape[d];
            packed[EIdxDim + d] = d < indicesShape.Length ? indicesShape[d] : 1;
        }
        int ds = 1, ixs = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            packed[EDataStride + d] = ds;
            ds *= dataShape[d];
            packed[EIdxStride + d] = ixs;
            ixs *= d < indicesShape.Length ? indicesShape[d] : 1;
        }

        _scatterElementsKernel!(outputCount, indices, updates, output, RentParams(packed));
    }

    // ScatterND params: [0]=indexDepth [1]=sliceSize [2]=numUpdates [3..]=data dims covered by the tuple
    private const int NDDimBase = 3;

    /// <summary>One thread per OUTPUT element; scans the update tuples for one addressing it.</summary>
    private static void ScatterNDImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> updates,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int depth = p[0], sliceSize = p[1], numUpdates = p[2];

        int flatLeading = idx / sliceSize;   // which slice this element belongs to
        int e = idx - flatLeading * sliceSize;

        float value = output[idx];   // already a copy of `data`
        for (int u = 0; u < numUpdates; u++)
        {
            int flatU = 0;
            int ok = 1;
            for (int d = 0; d < depth; d++)
            {
                int dim = p[NDDimBase + d];
                int c = (int)indices[u * depth + d];
                if (c < 0) c += dim;
                if (c < 0 || c >= dim) ok = 0;
                else flatU = flatU * dim + c;
            }
            if (ok == 1 && flatU == flatLeading) value = updates[u * sliceSize + e];
        }
        output[idx] = value;
    }

    /// <summary>ScatterND. Output must already hold a copy of data.</summary>
    public void ScatterND(ArrayView1D<float, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> updates,
        ArrayView1D<float, Stride1D.Dense> output,
        int[] dataShape, int indexDepth, int numUpdates, int sliceSize, int outputCount)
    {
        EnsureLoaded();
        var packed = new int[NDDimBase + MaxRank];
        packed[0] = indexDepth;
        packed[1] = sliceSize;
        packed[2] = numUpdates;
        for (int d = 0; d < indexDepth && d < MaxRank; d++)
            packed[NDDimBase + d] = d < dataShape.Length ? dataShape[d] : 1;

        _scatterNDKernel!(outputCount, indices, updates, output, RentParams(packed));
    }

    private ArrayView1D<int, Stride1D.Dense> RentParams(int[] packed)
    {
        if (Graph.GraphExecutor.UseCaptureParamSlots)
        {
            // CUDA-graph capture: a stable per-forward slot, since cuMemAlloc mid-capture is illegal.
            return CaptureParamArena.Shared(_accelerator).RentStableSlot(packed);
        }
        // Retire rather than dispose: the dispatch may still be pending in an un-submitted WebGPU command
        // batch, and freeing a buffer it reads makes the GPU read zeros.
        if (_lastParams != null) _oldParams.Add(_lastParams);
        _lastParams = _accelerator.Allocate1D(packed);
        return _lastParams.View;
    }

    private void EnsureLoaded()
    {
        _scatterElementsKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(ScatterElementsImpl);
        _scatterNDKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(ScatterNDImpl);
    }

    public void Dispose()
    {
        _lastParams?.Dispose();
        _lastParams = null;
        foreach (var b in _oldParams) b.Dispose();
        _oldParams.Clear();
    }
}
