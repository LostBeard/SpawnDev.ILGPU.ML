using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU reduction kernels for sum, mean, max, min along an axis.
///
/// All reductions are decomposed into (outerSize, reduceSize, innerSize) where:
/// - outerSize = product of dims before the reduce axis
/// - reduceSize = size of the reduce axis
/// - innerSize = product of dims after the reduce axis
///
/// One thread per output element (outerSize * innerSize).
/// Sequential reduction over reduceSize — fine for small-to-medium reduce dims.
/// </summary>
public class ReductionKernels
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int>? _reduceSumKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int>? _reduceMeanKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int>? _reduceMaxKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int>? _reduceMinKernel;

    public ReductionKernels(Accelerator accelerator) => _accelerator = accelerator;

    private static void ReduceSumImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int outerSize, int reduceSize, int innerSize)
    {
        int outer = idx / innerSize;
        int inner = idx % innerSize;
        float sum = 0f;
        int baseIdx = outer * reduceSize * innerSize + inner;
        for (int r = 0; r < reduceSize; r++)
            sum += input[baseIdx + r * innerSize];
        output[idx] = sum;
    }

    private static void ReduceMeanImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int outerSize, int reduceSize, int innerSize)
    {
        int outer = idx / innerSize;
        int inner = idx % innerSize;
        float sum = 0f;
        int baseIdx = outer * reduceSize * innerSize + inner;
        for (int r = 0; r < reduceSize; r++)
            sum += input[baseIdx + r * innerSize];
        output[idx] = sum / reduceSize;
    }

    private static void ReduceMaxImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int outerSize, int reduceSize, int innerSize)
    {
        int outer = idx / innerSize;
        int inner = idx % innerSize;
        int baseIdx = outer * reduceSize * innerSize + inner;
        float max = input[baseIdx];
        for (int r = 1; r < reduceSize; r++)
        {
            float v = input[baseIdx + r * innerSize];
            if (v > max) max = v;
        }
        output[idx] = max;
    }

    private static void ReduceMinImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int outerSize, int reduceSize, int innerSize)
    {
        int outer = idx / innerSize;
        int inner = idx % innerSize;
        int baseIdx = outer * reduceSize * innerSize + inner;
        float min = input[baseIdx];
        for (int r = 1; r < reduceSize; r++)
        {
            float v = input[baseIdx + r * innerSize];
            if (v < min) min = v;
        }
        output[idx] = min;
    }

    // ── Public API ──

    public void ReduceSum(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int outerSize, int reduceSize, int innerSize)
    {
        EnsureLoaded();
        _reduceSumKernel!(outerSize * innerSize, input, output, outerSize, reduceSize, innerSize);
    }

    public void ReduceMean(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int outerSize, int reduceSize, int innerSize)
    {
        EnsureLoaded();
        _reduceMeanKernel!(outerSize * innerSize, input, output, outerSize, reduceSize, innerSize);
    }

    public void ReduceMax(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int outerSize, int reduceSize, int innerSize)
    {
        EnsureLoaded();
        _reduceMaxKernel!(outerSize * innerSize, input, output, outerSize, reduceSize, innerSize);
    }

    public void ReduceMin(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int outerSize, int reduceSize, int innerSize)
    {
        EnsureLoaded();
        _reduceMinKernel!(outerSize * innerSize, input, output, outerSize, reduceSize, innerSize);
    }

    private void EnsureLoaded()
    {
        var a = _accelerator;
        _reduceSumKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int>(ReduceSumImpl);
        _reduceMeanKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int>(ReduceMeanImpl);
        _reduceMaxKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int>(ReduceMaxImpl);
        _reduceMinKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int>(ReduceMinImpl);
    }
}
