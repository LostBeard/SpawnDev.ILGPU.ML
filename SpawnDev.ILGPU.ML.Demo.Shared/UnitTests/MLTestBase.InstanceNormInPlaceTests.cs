using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Verifies the in-place InstanceNorm kernel (NormalizationKernels.InstanceNormInPlace) produces byte-identical
/// results to the regular two-buffer InstanceNorm — it just writes back into the same buffer (a single read_write
/// binding, WebGPU-legal), saving the output feature map (a 256 MiB buffer in the SD VAE). Checked on every
/// backend, including WebGPU/WebGL where binding one buffer as both in and out of a two-param kernel is illegal
/// (the in-place kernel uses a single binding, so it's valid).
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task InstanceNormInPlace_MatchesTwoBuffer_AllBackends() => await RunTest(async accelerator =>
    {
        const int N = 1, C = 8, H = 12, W = 12, spatial = H * W;
        int n = N * C * spatial;
        var rng = new Random(811);
        var x = new float[n]; for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 6 - 3);
        var scale = new float[C]; var bias = new float[C];
        for (int c = 0; c < C; c++) { scale[c] = (float)(rng.NextDouble() * 1.5 + 0.5); bias[c] = (float)(rng.NextDouble() * 0.4 - 0.2); }

        var norm = new SpawnDev.ILGPU.ML.Kernels.NormalizationKernels(accelerator);
        using var scaleB = accelerator.Allocate1D(scale);
        using var biasB = accelerator.Allocate1D(bias);

        // Reference: regular two-buffer InstanceNorm.
        using var refIn = accelerator.Allocate1D(x);
        using var refOut = accelerator.Allocate1D<float>(n);
        norm.InstanceNorm(refIn.View, refOut.View, scaleB.View, biasB.View, N, C, spatial);
        await accelerator.SynchronizeAsync();
        var expected = await refOut.CopyToHostAsync<float>(0, n);

        // In-place: normalize a copy of x in its own buffer.
        using var data = accelerator.Allocate1D(x);
        norm.InstanceNormInPlace(data.View, scaleB.View, biasB.View, N, C, spatial);
        await accelerator.SynchronizeAsync();
        var got = await data.CopyToHostAsync<float>(0, n);

        float worst = 0;
        for (int i = 0; i < n; i++) worst = MathF.Max(worst, MathF.Abs(got[i] - expected[i]));
        if (worst > 1e-5f)
            throw new Exception($"in-place InstanceNorm diverged from two-buffer (worst |Δ|={worst:E3}) on {BackendName}");
        Console.WriteLine($"[InstanceNormInPlace] in-place matches two-buffer (worst |Δ|={worst:E3}) on {BackendName}");
    });
}
