using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// First approach-(i) precision-AWARE op: a single generic SiLU kernel (<see cref="PrecisionAwareKernels.SiLU"/>)
/// that reads low precision, computes fp32, writes low precision DIRECTLY (no fp32 temp). Verifies it matches
/// a CPU fp32 SiLU reference for Half and bf16 (the low-p I/O paths) on EVERY backend — proving the
/// read-low-p / fp32-compute / write-low-p op pattern (via PrecisionConvert) works end-to-end, the building
/// block for fp16/bf16 VAE/decoder activations with no fp32 temps.
/// </summary>
public abstract partial class MLTestBase
{
    private static float[] SiLUCpu(float[] x)
    {
        var o = new float[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            o[i] = v > 30f ? v : v < -30f ? 0f : v / (1f + MathF.Exp(-v));
        }
        return o;
    }

    [TestMethod]
    public async Task PrecisionAwareSiLU_Half_MatchesFp32_AllBackends() => await RunTest(async accelerator =>
    {
        const int n = 513;
        var rng = new Random(41);
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 12 - 6);
        var expected = SiLUCpu(x);

        var xh = new global::ILGPU.Half[n]; for (int i = 0; i < n; i++) xh[i] = (global::ILGPU.Half)x[i];
        using var inBuf = accelerator.Allocate1D(xh);
        using var outBuf = accelerator.Allocate1D<global::ILGPU.Half>(n);
        var pa = new PrecisionAwareKernels(accelerator);
        pa.SiLU<global::ILGPU.Half>(inBuf.View, outBuf.View, n);
        await accelerator.SynchronizeAsync();
        var gotH = await outBuf.CopyToHostAsync<global::ILGPU.Half>(0, n);

        for (int i = 0; i < n; i++)
        {
            float got = (float)gotH[i];
            // fp16 quantizes both input and output; ~2^-10 relative + small abs floor.
            if (MathF.Abs(got - expected[i]) > MathF.Max(8e-3f, MathF.Abs(expected[i]) * 8e-3f))
                throw new Exception($"Half precision-aware SiLU @{i}: got {got}, want {expected[i]} (in {x[i]}) on {BackendName}");
        }
        Console.WriteLine($"[PrecisionAwareSiLU] Half read+fp32-compute+write matches fp32 SiLU on {BackendName}");
    });

    [TestMethod]
    public async Task PrecisionAwareSiLU_BFloat16_MatchesFp32_AllBackends() => await RunTest(async accelerator =>
    {
        const int n = 513;
        var rng = new Random(43);
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 12 - 6);
        var expected = SiLUCpu(x);

        var xb = new global::ILGPU.BFloat16[n]; for (int i = 0; i < n; i++) xb[i] = (global::ILGPU.BFloat16)x[i];
        using var inBuf = accelerator.Allocate1D(xb);
        using var outBuf = accelerator.Allocate1D<global::ILGPU.BFloat16>(n);
        var pa = new PrecisionAwareKernels(accelerator);
        pa.SiLU<global::ILGPU.BFloat16>(inBuf.View, outBuf.View, n);
        await accelerator.SynchronizeAsync();
        var gotB = await outBuf.CopyToHostAsync<global::ILGPU.BFloat16>(0, n);

        for (int i = 0; i < n; i++)
        {
            float got = (float)gotB[i];
            if (MathF.Abs(got - expected[i]) > MathF.Max(3e-2f, MathF.Abs(expected[i]) * 3e-2f))
                throw new Exception($"bf16 precision-aware SiLU @{i}: got {got}, want {expected[i]} (in {x[i]}) on {BackendName}");
        }
        Console.WriteLine($"[PrecisionAwareSiLU] bf16 read+fp32-compute+write matches fp32 SiLU on {BackendName}");
    });
}
