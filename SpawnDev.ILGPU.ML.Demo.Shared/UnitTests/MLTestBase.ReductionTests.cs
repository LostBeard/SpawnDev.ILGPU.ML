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
}
