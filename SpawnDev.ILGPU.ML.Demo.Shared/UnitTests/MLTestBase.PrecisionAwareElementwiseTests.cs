using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Approach-(i) precision-AWARE elementwise ops: generic <see cref="PrecisionAwareKernels.Add"/> and
/// <see cref="PrecisionAwareKernels.Mul"/> read low-precision operands, compute fp32, write low precision
/// DIRECTLY (no fp32 temp). Verified vs a CPU fp32 reference for Half and bf16 on EVERY backend.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task PrecisionAwareAdd_Half_MatchesFp32_AllBackends() => await RunTest(async accelerator =>
    {
        const int n = 517;
        var rng = new Random(101);
        var a = new float[n]; var b = new float[n];
        for (int i = 0; i < n; i++) { a[i] = (float)(rng.NextDouble() * 8 - 4); b[i] = (float)(rng.NextDouble() * 8 - 4); }

        var ah = new global::ILGPU.Half[n]; var bh = new global::ILGPU.Half[n];
        for (int i = 0; i < n; i++) { ah[i] = (global::ILGPU.Half)a[i]; bh[i] = (global::ILGPU.Half)b[i]; }
        using var aBuf = accelerator.Allocate1D(ah);
        using var bBuf = accelerator.Allocate1D(bh);
        using var outBuf = accelerator.Allocate1D<global::ILGPU.Half>(n);
        var pa = new PrecisionAwareKernels(accelerator);
        pa.Add<global::ILGPU.Half>(aBuf.View, bBuf.View, outBuf.View, n);
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<global::ILGPU.Half>(0, n);

        for (int i = 0; i < n; i++)
        {
            // Reference reads the same fp16-quantized operands the kernel sees.
            float expected = (float)ah[i] + (float)bh[i];
            float g = (float)got[i];
            if (MathF.Abs(g - expected) > MathF.Max(8e-3f, MathF.Abs(expected) * 8e-3f))
                throw new Exception($"Half precision-aware Add @{i}: got {g}, want {expected} on {BackendName}");
        }
        Console.WriteLine($"[PrecisionAwareAdd] Half low-p add matches fp32 on {BackendName}");
    });

    [TestMethod]
    public async Task PrecisionAwareAdd_BFloat16_MatchesFp32_AllBackends() => await RunTest(async accelerator =>
    {
        const int n = 517;
        var rng = new Random(103);
        var a = new float[n]; var b = new float[n];
        for (int i = 0; i < n; i++) { a[i] = (float)(rng.NextDouble() * 8 - 4); b[i] = (float)(rng.NextDouble() * 8 - 4); }

        var ab = new global::ILGPU.BFloat16[n]; var bb = new global::ILGPU.BFloat16[n];
        for (int i = 0; i < n; i++) { ab[i] = (global::ILGPU.BFloat16)a[i]; bb[i] = (global::ILGPU.BFloat16)b[i]; }
        using var aBuf = accelerator.Allocate1D(ab);
        using var bBuf = accelerator.Allocate1D(bb);
        using var outBuf = accelerator.Allocate1D<global::ILGPU.BFloat16>(n);
        var pa = new PrecisionAwareKernels(accelerator);
        pa.Add<global::ILGPU.BFloat16>(aBuf.View, bBuf.View, outBuf.View, n);
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<global::ILGPU.BFloat16>(0, n);

        for (int i = 0; i < n; i++)
        {
            float expected = (float)ab[i] + (float)bb[i];
            float g = (float)got[i];
            if (MathF.Abs(g - expected) > MathF.Max(3e-2f, MathF.Abs(expected) * 3e-2f))
                throw new Exception($"bf16 precision-aware Add @{i}: got {g}, want {expected} on {BackendName}");
        }
        Console.WriteLine($"[PrecisionAwareAdd] bf16 low-p add matches fp32 on {BackendName}");
    });

    [TestMethod]
    public async Task PrecisionAwareMul_Half_MatchesFp32_AllBackends() => await RunTest(async accelerator =>
    {
        const int n = 517;
        var rng = new Random(105);
        var a = new float[n]; var b = new float[n];
        for (int i = 0; i < n; i++) { a[i] = (float)(rng.NextDouble() * 4 - 2); b[i] = (float)(rng.NextDouble() * 4 - 2); }

        var ah = new global::ILGPU.Half[n]; var bh = new global::ILGPU.Half[n];
        for (int i = 0; i < n; i++) { ah[i] = (global::ILGPU.Half)a[i]; bh[i] = (global::ILGPU.Half)b[i]; }
        using var aBuf = accelerator.Allocate1D(ah);
        using var bBuf = accelerator.Allocate1D(bh);
        using var outBuf = accelerator.Allocate1D<global::ILGPU.Half>(n);
        var pa = new PrecisionAwareKernels(accelerator);
        pa.Mul<global::ILGPU.Half>(aBuf.View, bBuf.View, outBuf.View, n);
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<global::ILGPU.Half>(0, n);

        for (int i = 0; i < n; i++)
        {
            float expected = (float)ah[i] * (float)bh[i];
            float g = (float)got[i];
            if (MathF.Abs(g - expected) > MathF.Max(8e-3f, MathF.Abs(expected) * 8e-3f))
                throw new Exception($"Half precision-aware Mul @{i}: got {g}, want {expected} on {BackendName}");
        }
        Console.WriteLine($"[PrecisionAwareMul] Half low-p mul matches fp32 on {BackendName}");
    });

    [TestMethod]
    public async Task PrecisionAwareMul_BFloat16_MatchesFp32_AllBackends() => await RunTest(async accelerator =>
    {
        const int n = 517;
        var rng = new Random(107);
        var a = new float[n]; var b = new float[n];
        for (int i = 0; i < n; i++) { a[i] = (float)(rng.NextDouble() * 4 - 2); b[i] = (float)(rng.NextDouble() * 4 - 2); }

        var ab = new global::ILGPU.BFloat16[n]; var bb = new global::ILGPU.BFloat16[n];
        for (int i = 0; i < n; i++) { ab[i] = (global::ILGPU.BFloat16)a[i]; bb[i] = (global::ILGPU.BFloat16)b[i]; }
        using var aBuf = accelerator.Allocate1D(ab);
        using var bBuf = accelerator.Allocate1D(bb);
        using var outBuf = accelerator.Allocate1D<global::ILGPU.BFloat16>(n);
        var pa = new PrecisionAwareKernels(accelerator);
        pa.Mul<global::ILGPU.BFloat16>(aBuf.View, bBuf.View, outBuf.View, n);
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<global::ILGPU.BFloat16>(0, n);

        for (int i = 0; i < n; i++)
        {
            float expected = (float)ab[i] * (float)bb[i];
            float g = (float)got[i];
            if (MathF.Abs(g - expected) > MathF.Max(3e-2f, MathF.Abs(expected) * 3e-2f))
                throw new Exception($"bf16 precision-aware Mul @{i}: got {g}, want {expected} on {BackendName}");
        }
        Console.WriteLine($"[PrecisionAwareMul] bf16 low-p mul matches fp32 on {BackendName}");
    });
}
