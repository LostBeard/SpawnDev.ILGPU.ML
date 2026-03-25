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
public class GatherKernel
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int>? _gatherAxis0Kernel;

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
        output[idx] = data[srcRow * innerSize + col];
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
        output[idx] = data[srcRow * innerSize + col];
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

    private void EnsureLoaded()
    {
        _gatherAxis0Kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int>(GatherAxis0Impl);
        _gatherAxis0FloatKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int>(GatherAxis0FloatImpl);
    }
}
