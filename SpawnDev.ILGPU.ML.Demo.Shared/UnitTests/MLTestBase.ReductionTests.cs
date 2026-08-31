using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task ReduceSum_LastAxis() => await RunTest(async accelerator =>
    {
        // Reduce [10, 384] along axis 1 → [10]
        int outer = 10, reduce = 384, inner = 1;
        var input = RandomFloats(outer * reduce, seed: 70);
        var expected = new float[outer];
        for (int o = 0; o < outer; o++)
        {
            float sum = 0;
            for (int r = 0; r < reduce; r++) sum += input[o * reduce + r];
            expected[o] = sum;
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(outer);
        var red = new ReductionKernels(accelerator);
        red.ReduceSum(inBuf.View, outBuf.View, outer, reduce, inner);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, outer), expected, reduce * 1e-5f, "ReduceSum last axis: ");
    });

    [TestMethod]
    public async Task ReduceMean_MiddleAxis() => await RunTest(async accelerator =>
    {
        // Reduce [6, 1370, 64] along axis 1 → [6, 64]
        int outer = 6, reduce = 1370, inner = 64;
        var input = RandomFloats(outer * reduce * inner, seed: 71);
        var expected = new float[outer * inner];
        for (int o = 0; o < outer; o++)
            for (int i = 0; i < inner; i++)
            {
                float sum = 0;
                for (int r = 0; r < reduce; r++)
                    sum += input[o * reduce * inner + r * inner + i];
                expected[o * inner + i] = sum / reduce;
            }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(outer * inner);
        var red = new ReductionKernels(accelerator);
        red.ReduceMean(inBuf.View, outBuf.View, outer, reduce, inner);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, outer * inner), expected, reduce * 1e-5f, "ReduceMean middle axis: ");
    });

    [TestMethod]
    public async Task ReduceMax_MatchesCpu() => await RunTest(async accelerator =>
    {
        int outer = 10, reduce = 100, inner = 1;
        var input = RandomFloats(outer * reduce, seed: 72, scale: 10f);
        var expected = new float[outer];
        for (int o = 0; o < outer; o++)
        {
            float max = float.MinValue;
            for (int r = 0; r < reduce; r++) max = MathF.Max(max, input[o * reduce + r]);
            expected[o] = max;
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(outer);
        var red = new ReductionKernels(accelerator);
        red.ReduceMax(inBuf.View, outBuf.View, outer, reduce, inner);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, outer), expected, 1e-6f, "ReduceMax: ");
    });

    /// <summary>
    /// WHOLE-TENSOR min/max - <c>outerSize == innerSize == 1</c> - across the two-stage threshold.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠️ Every other reduction test in this file uses a handful of elements with
    /// <c>outerSize &gt; 1</c>, so none of them ever exercised the case where there is exactly ONE output
    /// element. That case used to launch exactly ONE GPU thread to walk the entire tensor - the class
    /// summary's own "fine for small-to-medium reduce dims" caveat, taken past its limit.
    /// </para>
    /// <para>
    /// It was not a theoretical cost. <c>DynamicQuantizeLinear</c> reduces min AND max over its whole
    /// input, and on ZipVoice's int8 flow-matching decoder that measured <b>61.2% of all GPU time</b> -
    /// 4,802 ms over 350 calls, up to ~83 ms for a single call, on an RTX 4070. The two-stage form took it
    /// to 131 ms (36.7x) and the decoder from 23.1 s to 5.4 s, with the rendered audio BIT-IDENTICAL.
    /// </para>
    /// <para>
    /// Sizes here straddle the threshold deliberately so BOTH paths are covered, and the large sizes are
    /// not multiples of the partial count - a stride loop that mishandles the ragged tail would pass on a
    /// round number and fail here.
    /// </para>
    /// </remarks>
    [TestMethod]
    public async Task Reduce_WholeTensorMinMax_BothPathsMatchCpu() => await RunTest(async accelerator =>
    {
        // 4096 is the two-stage threshold and 1024 the partial count, so: comfortably single-stage,
        // just below the threshold, exactly at it, and two ragged sizes well above.
        int[] sizes = { 64, 4095, 4096, 100_003, 262_144 + 7 };
        var red = new ReductionKernels(accelerator);

        foreach (int n in sizes)
        {
            var input = RandomFloats(n, seed: 4242 + n % 97);
            // Guarantee the extremes are not at index 0 - a reduction that simply echoed its first
            // element, or that dropped its ragged tail, would otherwise be indistinguishable from correct.
            input[n / 3] = 12345.5f;
            input[(2 * n) / 3] = -9876.25f;

            float expMax = input[0], expMin = input[0];
            for (int i = 1; i < n; i++)
            {
                if (input[i] > expMax) expMax = input[i];
                if (input[i] < expMin) expMin = input[i];
            }

            using var inBuf = accelerator.Allocate1D(input);
            using var maxBuf = accelerator.Allocate1D<float>(1);
            using var minBuf = accelerator.Allocate1D<float>(1);

            red.ReduceMax(inBuf.View, maxBuf.View, 1, n, 1);
            red.ReduceMin(inBuf.View, minBuf.View, 1, n, 1);
            await accelerator.SynchronizeAsync();

            await AssertCloseGpu(accelerator, maxBuf.View, new[] { expMax }, 1e-4f,
                $"whole-tensor ReduceMax n={n}: ");
            await AssertCloseGpu(accelerator, minBuf.View, new[] { expMin }, 1e-4f,
                $"whole-tensor ReduceMin n={n}: ");
        }
    });

    /// <summary>
    /// Back-to-back whole-tensor reductions must not interfere.
    /// </summary>
    /// <remarks>
    /// The two-stage path writes into a REUSED scratch buffer held for the accelerator's lifetime (per-call
    /// device allocation is what makes a graph uncapturable). Reused scratch is the classic place for one
    /// reduction to read another's leftovers, so this runs several in a row on different data and checks
    /// every answer rather than just the last.
    /// </remarks>
    [TestMethod]
    public async Task Reduce_WholeTensorMinMax_ReusedScratchStaysCorrect() => await RunTest(async accelerator =>
    {
        var red = new ReductionKernels(accelerator);
        const int n = 50_000;
        for (int round = 0; round < 4; round++)
        {
            var input = RandomFloats(n, seed: 900 + round);
            float target = 1000f + round;          // distinct per round
            input[n / 2 + round] = target;
            input[n / 4 + round] = -target;

            float expMax = input[0], expMin = input[0];
            for (int i = 1; i < n; i++)
            {
                if (input[i] > expMax) expMax = input[i];
                if (input[i] < expMin) expMin = input[i];
            }

            using var inBuf = accelerator.Allocate1D(input);
            using var maxBuf = accelerator.Allocate1D<float>(1);
            using var minBuf = accelerator.Allocate1D<float>(1);
            red.ReduceMax(inBuf.View, maxBuf.View, 1, n, 1);
            red.ReduceMin(inBuf.View, minBuf.View, 1, n, 1);
            await accelerator.SynchronizeAsync();
            await AssertCloseGpu(accelerator, maxBuf.View, new[] { expMax }, 1e-4f,
                $"reused-scratch ReduceMax round {round}: ");
            await AssertCloseGpu(accelerator, minBuf.View, new[] { expMin }, 1e-4f,
                $"reused-scratch ReduceMin round {round}: ");
        }
    });
}
