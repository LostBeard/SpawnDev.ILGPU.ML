using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Approach-(i) precision-AWARE GroupNorm: generic <see cref="PrecisionAwareKernels.GroupNorm"/> reads low-p
/// activations, accumulates mean/var in fp32, normalizes, writes low precision DIRECTLY (no fp32 activation
/// temp). The affine weight/bias stay fp32. Verified vs a CPU fp32 reference (reading the SAME quantized input
/// the kernel sees) for Half and bf16 on EVERY backend. GroupNorm is heavy in the SD VAE decoder.
/// </summary>
public abstract partial class MLTestBase
{
    // CPU fp32 GroupNorm matching GroupNormKernel exactly. xq is the already-quantized (fp16/bf16→float) input.
    private static float[] GroupNormCpu(float[] xq, float[] weight, float[] bias, int B, int C, int S, int G, float eps)
    {
        var o = new float[xq.Length];
        int channelsPerGroup = C / G;
        int groupSize = channelsPerGroup * S;
        for (int batch = 0; batch < B; batch++)
            for (int group = 0; group < G; group++)
            {
                int groupBase = batch * C * S + group * channelsPerGroup * S;
                float sum = 0f;
                for (int gi = 0; gi < groupSize; gi++) sum += xq[groupBase + gi];
                float mean = sum / groupSize;
                float varSum = 0f;
                for (int gi = 0; gi < groupSize; gi++) { float d = xq[groupBase + gi] - mean; varSum += d * d; }
                float invStd = 1f / MathF.Sqrt(varSum / groupSize + eps);
                for (int cg = 0; cg < channelsPerGroup; cg++)
                {
                    int channel = group * channelsPerGroup + cg;
                    for (int s = 0; s < S; s++)
                    {
                        int idx = groupBase + cg * S + s;
                        o[idx] = weight[channel] * (xq[idx] - mean) * invStd + bias[channel];
                    }
                }
            }
        return o;
    }

    [TestMethod]
    public async Task PrecisionAwareGroupNorm_Half_MatchesFp32_AllBackends() => await RunTest(async accelerator =>
    {
        const int B = 1, C = 8, S = 16, G = 2;
        const float eps = 1e-5f;
        int n = B * C * S;
        var rng = new Random(201);
        var x = new float[n]; for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 6 - 3);
        var weight = new float[C]; var bias = new float[C];
        for (int c = 0; c < C; c++) { weight[c] = (float)(rng.NextDouble() * 1.5 + 0.5); bias[c] = (float)(rng.NextDouble() * 0.4 - 0.2); }

        var xh = new global::ILGPU.Half[n]; for (int i = 0; i < n; i++) xh[i] = (global::ILGPU.Half)x[i];
        var xq = new float[n]; for (int i = 0; i < n; i++) xq[i] = (float)xh[i];
        var expected = GroupNormCpu(xq, weight, bias, B, C, S, G, eps);

        using var inBuf = accelerator.Allocate1D(xh);
        using var outBuf = accelerator.Allocate1D<global::ILGPU.Half>(n);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);
        var pa = new PrecisionAwareKernels(accelerator);
        pa.GroupNorm<global::ILGPU.Half>(inBuf.View, outBuf.View, wBuf.View, bBuf.View, B, C, S, G, eps);
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<global::ILGPU.Half>(0, n);

        for (int i = 0; i < n; i++)
        {
            float g = (float)got[i];
            if (MathF.Abs(g - expected[i]) > MathF.Max(1.5e-2f, MathF.Abs(expected[i]) * 1.5e-2f))
                throw new Exception($"Half precision-aware GroupNorm @{i}: got {g}, want {expected[i]} on {BackendName}");
        }
        Console.WriteLine($"[PrecisionAwareGroupNorm] Half read+fp32-stats+write matches fp32 on {BackendName}");
    });

    [TestMethod]
    public async Task PrecisionAwareGroupNorm_BFloat16_MatchesFp32_AllBackends() => await RunTest(async accelerator =>
    {
        const int B = 1, C = 8, S = 16, G = 2;
        const float eps = 1e-5f;
        int n = B * C * S;
        var rng = new Random(203);
        var x = new float[n]; for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 6 - 3);
        var weight = new float[C]; var bias = new float[C];
        for (int c = 0; c < C; c++) { weight[c] = (float)(rng.NextDouble() * 1.5 + 0.5); bias[c] = (float)(rng.NextDouble() * 0.4 - 0.2); }

        var xb = new global::ILGPU.BFloat16[n]; for (int i = 0; i < n; i++) xb[i] = (global::ILGPU.BFloat16)x[i];
        var xq = new float[n]; for (int i = 0; i < n; i++) xq[i] = (float)xb[i];
        var expected = GroupNormCpu(xq, weight, bias, B, C, S, G, eps);

        using var inBuf = accelerator.Allocate1D(xb);
        using var outBuf = accelerator.Allocate1D<global::ILGPU.BFloat16>(n);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);
        var pa = new PrecisionAwareKernels(accelerator);
        pa.GroupNorm<global::ILGPU.BFloat16>(inBuf.View, outBuf.View, wBuf.View, bBuf.View, B, C, S, G, eps);
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<global::ILGPU.BFloat16>(0, n);

        for (int i = 0; i < n; i++)
        {
            float g = (float)got[i];
            if (MathF.Abs(g - expected[i]) > MathF.Max(5e-2f, MathF.Abs(expected[i]) * 5e-2f))
                throw new Exception($"bf16 precision-aware GroupNorm @{i}: got {g}, want {expected[i]} on {BackendName}");
        }
        Console.WriteLine($"[PrecisionAwareGroupNorm] bf16 read+fp32-stats+write matches fp32 on {BackendName}");
    });
}
