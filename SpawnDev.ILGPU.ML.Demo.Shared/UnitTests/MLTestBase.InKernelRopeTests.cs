using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

// Needs SpawnDev.ILGPU >= 5.4.34 (accurate WebGPU / WebGL sin / cos).
public abstract partial class MLTestBase
{
    /// <summary>
    /// The in-kernel rotary embedding (<see cref="SkinTokensKernels.Rope"/>: MathF.Cos / MathF.Sin evaluated on the device, used by the SkinTokens / GGUF paths) against the
    /// host-table one (<see cref="SkinTokensKernels.RopeTables"/>) at positions 0..599 over head width 128, theta 1e6. Bound 2e-5. SpawnDev.ILGPU 5.4.33's WebGPU / WebGL
    /// sin / cos were 3.8e-5 off at these angles (no argument reduction in D3D's native functions); 5.4.34 reduces them (Cody-Waite + Cephes).
    /// </summary>
    [TestMethod]
    public async Task QwenEdit_InKernelRope_MatchesHostTables() => await RunTest(async accelerator =>
    {
        var u = new SkinTokensKernels(accelerator);
        var rng = new Random(29);
        const int Rows = 600, H = 4, D = 128;
        var buf = new float[Rows * H * D]; for (int i = 0; i < buf.Length; i++) buf[i] = (float)(rng.NextDouble() * 2 - 1);
        var inv = new float[D / 2]; for (int k = 0; k < D / 2; k++) inv[k] = 1f / MathF.Pow(1_000_000f, 2f * k / D);
        var (cos, sin) = SkinTokensKernels.AxesRopeTables(Enumerable.Range(0, Rows).ToArray(), Rows, 1, inv, D / 2, D / 2);
        using var a = accelerator.Allocate1D(buf); using var b = accelerator.Allocate1D(buf);
        using var dInv = accelerator.Allocate1D(inv); using var dCos = accelerator.Allocate1D(cos); using var dSin = accelerator.Allocate1D(sin);
        u.Rope(a.View, dInv.View, Rows, H * D, 0, H, D, 0);
        u.RopeTables(b.View, dCos.View, dSin.View, Rows, H * D, 0, H, D);
        var ga = await a.View.CopyToHostAsync(); var gb = await b.View.CopyToHostAsync();
        double worst = 0; for (int i = 0; i < ga.Length; i++) worst = Math.Max(worst, Math.Abs(ga[i] - gb[i]));
        var report = $"{accelerator.AcceleratorType}: in-kernel sin / cos vs host tables, positions 0..{Rows - 1}: worst {worst:E2}";
        if (!(worst < 2e-5)) throw new Exception(report);
        return report;
    });
}
