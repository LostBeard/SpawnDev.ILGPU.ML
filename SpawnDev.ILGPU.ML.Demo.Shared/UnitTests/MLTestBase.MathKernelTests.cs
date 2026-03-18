using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task Sqrt_MatchesCpu() => await RunTest(async accelerator =>
    {
        int count = 500;
        var input = RandomFloats(count, seed: 200, scale: 10f);
        for (int i = 0; i < count; i++) input[i] = MathF.Abs(input[i]) + 0.001f; // positive

        var expected = new float[count];
        for (int i = 0; i < count; i++) expected[i] = MathF.Sqrt(input[i]);

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(count);
        var ew = new ElementWiseKernels(accelerator);
        ew.Sqrt(inBuf.View, outBuf.View, count);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, count);
        AssertClose(expected, actual, 1e-5f, "Sqrt: ");
    });

    [TestMethod]
    public async Task Exp_MatchesCpu() => await RunTest(async accelerator =>
    {
        int count = 500;
        var input = RandomFloats(count, seed: 201, scale: 5f);
        var expected = new float[count];
        for (int i = 0; i < count; i++) expected[i] = MathF.Exp(input[i]);

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(count);
        var ew = new ElementWiseKernels(accelerator);
        ew.Exp(inBuf.View, outBuf.View, count);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, count);
        AssertClose(expected, actual, 1e-3f, "Exp: "); // Exp amplifies errors
    });

    [TestMethod]
    public async Task Div_MatchesCpu() => await RunTest(async accelerator =>
    {
        int count = 500;
        var a = RandomFloats(count, seed: 202, scale: 5f);
        var b = RandomFloats(count, seed: 203, scale: 5f);
        for (int i = 0; i < count; i++) if (MathF.Abs(b[i]) < 0.01f) b[i] = 1f; // avoid div by ~0

        var expected = new float[count];
        for (int i = 0; i < count; i++) expected[i] = a[i] / b[i];

        using var aBuf = accelerator.Allocate1D(a);
        using var bBuf = accelerator.Allocate1D(b);
        using var outBuf = accelerator.Allocate1D<float>(count);
        var ew = new ElementWiseKernels(accelerator);
        ew.Div(aBuf.View, bBuf.View, outBuf.View, count);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, count);
        AssertClose(expected, actual, 1e-4f, "Div: ");
    });

    [TestMethod]
    public async Task Erf_MatchesCpu() => await RunTest(async accelerator =>
    {
        int count = 500;
        var input = RandomFloats(count, seed: 204, scale: 3f);

        // CPU erf reference (same Abramowitz & Stegun approximation)
        var expected = new float[count];
        for (int i = 0; i < count; i++)
        {
            float x = input[i];
            float ax = MathF.Abs(x);
            float t = 1f / (1f + 0.3275911f * ax);
            float t2 = t * t; float t3 = t2 * t; float t4 = t3 * t; float t5 = t4 * t;
            float erfAbs = 1f - (0.254829592f * t - 0.284496736f * t2 + 1.421413741f * t3
                - 1.453152027f * t4 + 1.061405429f * t5) * MathF.Exp(-ax * ax);
            expected[i] = x < 0f ? -erfAbs : erfAbs;
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(count);
        var ew = new ElementWiseKernels(accelerator);
        ew.Erf(inBuf.View, outBuf.View, count);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, count);
        AssertClose(expected, actual, 1e-5f, "Erf: ");
    });

    [TestMethod]
    public async Task Abs_MatchesCpu() => await RunTest(async accelerator =>
    {
        int count = 500;
        var input = RandomFloats(count, seed: 205, scale: 10f);
        var expected = new float[count];
        for (int i = 0; i < count; i++) expected[i] = MathF.Abs(input[i]);

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(count);
        var ew = new ElementWiseKernels(accelerator);
        ew.Abs(inBuf.View, outBuf.View, count);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, count);
        AssertClose(expected, actual, 0f, "Abs: ");
    });
}
