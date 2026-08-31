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

    // ── whole-tensor (single-output) reduction ──
    //
    // ⚠️ The one-thread-per-output design above is documented as "fine for small-to-medium reduce
    // dims", and a WHOLE-TENSOR reduction is neither: outerSize == innerSize == 1 launches exactly ONE
    // thread to walk every element. DynamicQuantizeLinear does that TWICE per call (min and max), and on
    // ZipVoice's int8 flow-matching decoder it measured **61.2% of all GPU time** - 4,802 ms across 350
    // calls, with the widest feed_forward3 tensors taking ~83 ms EACH. One thread on an RTX 4070.
    //
    // The fix is a two-stage reduction: PartialCount threads each stride through the tensor, then one
    // thread folds the partials. Deliberately no atomics and no shared memory, so it behaves identically
    // on every backend including WebGL (which has no atomics at all).
    //
    // ⚠️ ONLY min and max get this treatment, and that is a correctness decision rather than an
    // oversight: max/min are exactly reassociative, so splitting the traversal cannot change the answer by
    // even one bit - the ZipVoice render was verified BIT-IDENTICAL before and after (max abs diff 0 over
    // 76,544 samples). Sum and Mean are NOT reassociative in floating point, so giving them the same
    // treatment would silently change every existing numeric expectation in the suite. If ReduceSum ever
    // needs this, it needs its own tolerance discussion first.
    private const int PartialCount = 1024;
    private const int TwoStageThreshold = 4096;   // below this, one thread is cheaper than two launches

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int>? _maxPartialKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int>? _minPartialKernel;
    // ⚠️ Allocated in EnsureLoaded (kernel-load time), NEVER per call. A per-call device allocation is
    // what made Conv1D impossible to CUDA-graph-capture - a mid-capture cuMemAlloc is an UNCATCHABLE
    // access violation - and this operator sits inside every quantised graph we would want to capture.
    private MemoryBuffer1D<float, Stride1D.Dense>? _maxPartials;
    private MemoryBuffer1D<float, Stride1D.Dense>? _minPartials;

    private static void MaxPartialImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> partials,
        int count, int stride)
    {
        // A thread with no elements must contribute a value that cannot change the result, so it echoes
        // element 0 rather than a sentinel - a float.MinValue would be indistinguishable from real data on
        // a backend that flushes it, and NaN would poison the fold.
        if (idx >= count) { partials[idx] = input[0]; return; }
        float m = input[idx];
        for (int i = idx + stride; i < count; i += stride)
        {
            float v = input[i];
            if (v > m) m = v;
        }
        partials[idx] = m;
    }

    private static void MinPartialImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> partials,
        int count, int stride)
    {
        if (idx >= count) { partials[idx] = input[0]; return; }
        float m = input[idx];
        for (int i = idx + stride; i < count; i += stride)
        {
            float v = input[i];
            if (v < m) m = v;
        }
        partials[idx] = m;
    }

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
        // Whole-tensor reduction: one output element means one thread, which is the pathological case
        // this class's own summary warns about. Split it.
        if (outerSize * innerSize == 1 && reduceSize >= TwoStageThreshold)
        {
            var partials = _maxPartials!.View;
            _maxPartialKernel!(PartialCount, input, partials, reduceSize, PartialCount);
            _reduceMaxKernel!(1, partials, output, 1, PartialCount, 1);
            return;
        }
        _reduceMaxKernel!(outerSize * innerSize, input, output, outerSize, reduceSize, innerSize);
    }

    public void ReduceMin(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int outerSize, int reduceSize, int innerSize)
    {
        EnsureLoaded();
        if (outerSize * innerSize == 1 && reduceSize >= TwoStageThreshold)
        {
            var partials = _minPartials!.View;
            _minPartialKernel!(PartialCount, input, partials, reduceSize, PartialCount);
            _reduceMinKernel!(1, partials, output, 1, PartialCount, 1);
            return;
        }
        _reduceMinKernel!(outerSize * innerSize, input, output, outerSize, reduceSize, innerSize);
    }

    // ── ReduceProd ──

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int>? _reduceProdKernel;

    private static void ReduceProdImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int outerSize, int reduceSize, int innerSize)
    {
        int outer = idx / innerSize;
        int inner = idx % innerSize;
        int baseIdx = outer * reduceSize * innerSize + inner;
        float prod = 1f;
        for (int r = 0; r < reduceSize; r++)
            prod *= input[baseIdx + r * innerSize];
        output[idx] = prod;
    }

    public void ReduceProd(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int outerSize, int reduceSize, int innerSize)
    {
        _reduceProdKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int>(ReduceProdImpl);
        _reduceProdKernel(outerSize * innerSize, input, output, outerSize, reduceSize, innerSize);
    }

    private void EnsureLoaded()
    {
        var a = _accelerator;
        _maxPartialKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int>(MaxPartialImpl);
        _minPartialKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int>(MinPartialImpl);
        // Allocated HERE, with the kernels, so no device allocation can ever happen mid-graph-capture.
        _maxPartials ??= a.Allocate1D<float>(PartialCount);
        _minPartials ??= a.Allocate1D<float>(PartialCount);
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
