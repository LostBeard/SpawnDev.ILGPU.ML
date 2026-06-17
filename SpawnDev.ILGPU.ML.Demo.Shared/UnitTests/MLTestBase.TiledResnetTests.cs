using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Tiling;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Proves the tiled-decode COMPOSITION: a full VAE resnet block (GroupNorm→SiLU→Conv3×3→GroupNorm→SiLU→Conv3×3 +
/// residual) run via <see cref="TiledVaeOps"/> on a <see cref="TiledFeatureMap"/> is byte-near the same block run
/// full-resolution with the same kernels — on every backend. This composes the three verified primitives
/// (global-stat GroupNorm + halo-refresh conv + pointwise per-tile) into the actual up-block forward shape, so
/// the only thing left for the full decoder is wiring the VAE's own weights. (Seam-free by construction: every
/// GroupNorm uses global stats.) inC==outC here (identity shortcut).
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task TiledResnet_MatchesFull_AllBackends() => await RunTest(async accelerator =>
    {
        const int C = 64, G = 32, H = 20, W = 24, rows = 2, cols = 2, halo = 1;
        const float eps = 1e-6f;
        int sp = H * W, n = C * sp;
        var rng = new Random(921);
        var x = new float[n]; for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 4 - 2);
        float[] g1 = Rand(C, rng, 0.5f, 1f), b1 = Rand(C, rng, -0.2f, 0.2f);
        float[] g2 = Rand(C, rng, 0.5f, 1f), b2 = Rand(C, rng, -0.2f, 0.2f);
        float[] W1 = Rand(C * C * 9, rng, -0.1f, 0.1f), CB1 = Rand(C, rng, -0.05f, 0.05f);
        float[] W2 = Rand(C * C * 9, rng, -0.1f, 0.1f), CB2 = Rand(C, rng, -0.05f, 0.05f);

        var gn = new GroupNormKernel(accelerator); var act = new ActivationKernels(accelerator);
        var conv = new Conv2DKernel(accelerator); var ew = new ElementWiseKernels(accelerator);
        using var g1B = accelerator.Allocate1D(g1); using var b1B = accelerator.Allocate1D(b1);
        using var g2B = accelerator.Allocate1D(g2); using var b2B = accelerator.Allocate1D(b2);
        using var w1B = accelerator.Allocate1D(W1); using var cb1B = accelerator.Allocate1D(CB1);
        using var w2B = accelerator.Allocate1D(W2); using var cb2B = accelerator.Allocate1D(CB2);

        // ── Full reference (same kernels, no tiling). ──
        float[] expected;
        {
            using var h = accelerator.Allocate1D(x);
            using var t = accelerator.Allocate1D<float>(n);
            gn.Forward(h.View, t.View, g1B.View, b1B.View, 1, C, sp, G, eps);
            act.SiLUInPlace(t.View, n);
            using var c1 = accelerator.Allocate1D<float>(n);
            conv.ForwardPadded(t.View, w1B.View, cb1B.View, c1.View, C, H, W, C, 3, 3, 1, 1, 1, 1, 1, 1, 1);
            gn.Forward(c1.View, t.View, g2B.View, b2B.View, 1, C, sp, G, eps);
            act.SiLUInPlace(t.View, n);
            using var c2 = accelerator.Allocate1D<float>(n);
            conv.ForwardPadded(t.View, w2B.View, cb2B.View, c2.View, C, H, W, C, 3, 3, 1, 1, 1, 1, 1, 1, 1);
            using var xB = accelerator.Allocate1D(x);
            ew.AddInPlace(c2.View, xB.View, n);            // + residual
            await accelerator.SynchronizeAsync();
            expected = await c2.CopyToHostAsync<float>(0, n);
        }

        // ── Tiled (TiledVaeOps). ──
        using var ops = new TiledVaeOps(accelerator);
        var map = TiledFeatureMap.FromFull(x, C, H, W, rows, cols, halo);
        var xMap = TiledFeatureMap.FromFull(x, C, H, W, rows, cols, halo);   // saved residual
        await ops.GroupNorm(map, g1B.View, b1B.View, C, G, eps);
        await ops.SiLU(map, C);
        map = await ops.Conv3x3(map, w1B.View, cb1B.View, C, C);
        await ops.GroupNorm(map, g2B.View, b2B.View, C, G, eps);
        await ops.SiLU(map, C);
        map = await ops.Conv3x3(map, w2B.View, cb2B.View, C, C);
        await ops.AddInPlace(map, xMap, C);
        var got = map.ToFull();

        float worst = 0;
        for (int i = 0; i < n; i++) worst = MathF.Max(worst, MathF.Abs(got[i] - expected[i]));
        if (worst > 3e-3f)
            throw new Exception($"tiled resnet diverged from full (worst |Δ|={worst:E3}) on {BackendName}");
        Console.WriteLine($"[TiledResnet] tiled GroupNorm→SiLU→Conv→GroupNorm→SiLU→Conv+res == full (worst |Δ|={worst:E3}) on {BackendName}");
    });

    private static float[] Rand(int n, Random rng, float lo, float hi)
    {
        var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * (hi - lo) + lo); return a;
    }
}
