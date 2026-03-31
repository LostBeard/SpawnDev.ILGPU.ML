using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task Sigmoid_MatchesCpu() => await RunTest(async accelerator =>
    {
        int count = 500;
        var input = RandomFloats(count, seed: 60, scale: 5f);
        var expected = new float[count];
        for (int i = 0; i < count; i++)
            expected[i] = 1f / (1f + MathF.Exp(-input[i]));

        using var buf = accelerator.Allocate1D((float[])input.Clone());
        var act = new ActivationKernels(accelerator);
        act.SigmoidInPlace(buf.View, count);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, buf.View.SubView(0, count), expected, 1e-5f, "Sigmoid: ");
    });

    [TestMethod]
    public async Task Tanh_MatchesCpu() => await RunTest(async accelerator =>
    {
        int count = 500;
        var input = RandomFloats(count, seed: 61, scale: 5f);
        var expected = new float[count];
        for (int i = 0; i < count; i++)
            expected[i] = MathF.Tanh(input[i]);

        using var buf = accelerator.Allocate1D((float[])input.Clone());
        var act = new ActivationKernels(accelerator);
        act.TanhInPlace(buf.View, count);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, buf.View.SubView(0, count), expected, 1e-5f, "Tanh: ");
    });

    [TestMethod]
    public async Task SiLU_MatchesCpu() => await RunTest(async accelerator =>
    {
        int count = 500;
        var input = RandomFloats(count, seed: 62, scale: 5f);
        var expected = new float[count];
        for (int i = 0; i < count; i++)
        {
            float sig = 1f / (1f + MathF.Exp(-input[i]));
            expected[i] = input[i] * sig;
        }

        using var buf = accelerator.Allocate1D((float[])input.Clone());
        var act = new ActivationKernels(accelerator);
        act.SiLUInPlace(buf.View, count);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, buf.View.SubView(0, count), expected, 1e-5f, "SiLU: ");
    });

    [TestMethod]
    public async Task HardSigmoid_MatchesCpu() => await RunTest(async accelerator =>
    {
        int count = 500;
        var input = RandomFloats(count, seed: 63, scale: 10f);
        var expected = new float[count];
        for (int i = 0; i < count; i++)
            expected[i] = MathF.Max(0f, MathF.Min(1f, input[i] / 6f + 0.5f));

        using var buf = accelerator.Allocate1D((float[])input.Clone());
        var act = new ActivationKernels(accelerator);
        act.HardSigmoidInPlace(buf.View, count);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, buf.View.SubView(0, count), expected, 1e-5f, "HardSigmoid: ");
    });

    [TestMethod]
    public async Task HardSwish_MatchesCpu() => await RunTest(async accelerator =>
    {
        int count = 500;
        var input = RandomFloats(count, seed: 64, scale: 10f);
        var expected = new float[count];
        for (int i = 0; i < count; i++)
        {
            float hs = MathF.Max(0f, MathF.Min(1f, input[i] / 6f + 0.5f));
            expected[i] = input[i] * hs;
        }

        using var buf = accelerator.Allocate1D((float[])input.Clone());
        var act = new ActivationKernels(accelerator);
        act.HardSwishInPlace(buf.View, count);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, buf.View.SubView(0, count), expected, 1e-5f, "HardSwish: ");
    });
}
