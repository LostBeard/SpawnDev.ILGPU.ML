using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// The heart of seam-free tiled VAE decode: a GroupNorm computed across spatial TILES with GLOBAL per-group stats
/// is byte-near a single full-resolution GroupNorm. Composes the verified primitives — per-group partial stats
/// (InstanceNormPartialStats over each tile's group slice), combine the partials into global mean/var, apply the
/// global stats per tile (InstanceNormApplyWithStats, groups dim), then the per-CHANNEL γ/β (the same apply with
/// mean=0/invStd=1 = a per-channel affine). The group-reshape [C,H,W]→[G,(C/G)·H·W] is a contiguous reinterpret
/// (a group's channels are contiguous). Verified per backend vs GroupNormKernel.Forward. (No halo: GroupNorm is
/// pointwise once the global stats are known.)
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task TiledGroupNorm_GlobalStatsMatchFull_AllBackends() => await RunTest(async accelerator =>
    {
        const int C = 64, G = 32, H = 24, W = 20;          // 64 ch, 32 groups (2 ch/group), spatial 24×20
        const float eps = 1e-6f;
        int sp = H * W, n = C * sp, cpg = C / G;
        var rng = new Random(919);
        var x = new float[n]; for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 6 - 3);
        var gamma = new float[C]; var beta = new float[C];
        for (int c = 0; c < C; c++) { gamma[c] = (float)(rng.NextDouble() * 1.5 + 0.5); beta[c] = (float)(rng.NextDouble() * 0.4 - 0.2); }

        var norm = new NormalizationKernels(accelerator);
        using var gB = accelerator.Allocate1D(gamma); using var bB = accelerator.Allocate1D(beta);

        // ── Reference: a single full GroupNorm (group stats over the full spatial + per-channel affine). ──
        float[] expected;
        using (var inB = accelerator.Allocate1D(x))
        using (var outB = accelerator.Allocate1D<float>(n))
        {
            var gn = new GroupNormKernel(accelerator);
            gn.Forward(inB.View, outB.View, gB.View, bB.View, 1, C, sp, G, eps);
            await accelerator.SynchronizeAsync();
            expected = await outB.CopyToHostAsync<float>(0, n);
        }

        // ── Tiled: split spatial into 2 row-bands; per-tile core is [C, bandH, W] contiguous (channel-major). ──
        int h0 = H / 2, h1 = H - h0;
        var tileData = new[] { Extract(x, C, H, W, 0, h0), Extract(x, C, H, W, h0, h1) };
        var tileH = new[] { h0, h1 };
        var tBufs = tileData.Select(d => accelerator.Allocate1D(d)).ToArray();

        // 1) per-tile, per-group partial stats over the group slice [G, cpg*bandSpatial].
        var gSum = new double[G]; var gSq = new double[G]; long gCount = 0;
        for (int t = 0; t < 2; t++)
        {
            int bandSp = tileH[t] * W, groupSpan = cpg * bandSp;
            using var sums = accelerator.Allocate1D<float>(G); using var sqs = accelerator.Allocate1D<float>(G);
            norm.InstanceNormPartialStats(tBufs[t].View, sums.View, sqs.View, 1, G, groupSpan);
            await accelerator.SynchronizeAsync();
            var s = await sums.CopyToHostAsync<float>(0, G); var q = await sqs.CopyToHostAsync<float>(0, G);
            for (int g = 0; g < G; g++) { gSum[g] += s[g]; gSq[g] += q[g]; }
            gCount += groupSpan;
        }
        // 2) global per-group mean/invStd.
        var means = new float[G]; var inv = new float[G];
        for (int g = 0; g < G; g++)
        {
            double mean = gSum[g] / gCount, var = gSq[g] / gCount - mean * mean;
            means[g] = (float)mean; inv[g] = (float)(1.0 / Math.Sqrt(var + eps));
        }
        using var meansB = accelerator.Allocate1D(means); using var invB = accelerator.Allocate1D(inv);
        var onesG = new float[G]; var zerosG = new float[G]; for (int g = 0; g < G; g++) onesG[g] = 1f;
        var onesC = new float[C]; var zerosC = new float[C]; for (int c = 0; c < C; c++) onesC[c] = 1f;
        using var onesGB = accelerator.Allocate1D(onesG); using var zerosGB = accelerator.Allocate1D(zerosG);
        using var onesCB = accelerator.Allocate1D(onesC); using var zerosCB = accelerator.Allocate1D(zerosC);

        // 3) per-tile: apply global per-group norm, then per-channel γ/β (= apply-with-stats, mean=0 invStd=1).
        var outTiles = new float[2][];
        for (int t = 0; t < 2; t++)
        {
            int bandSp = tileH[t] * W, groupSpan = cpg * bandSp;
            norm.InstanceNormApplyWithStats(tBufs[t].View, onesGB.View, zerosGB.View, meansB.View, invB.View, 1, G, groupSpan);
            norm.InstanceNormApplyWithStats(tBufs[t].View, gB.View, bB.View, zerosCB.View, onesCB.View, 1, C, bandSp);
            await accelerator.SynchronizeAsync();
            outTiles[t] = await tBufs[t].CopyToHostAsync<float>(0, C * bandSp);
        }
        foreach (var b in tBufs) b.Dispose();

        // recombine the 2 bands → [C,H,W]
        var got = new float[n];
        Insert(got, outTiles[0], C, H, W, 0, h0);
        Insert(got, outTiles[1], C, H, W, h0, h1);

        float worst = 0;
        for (int i = 0; i < n; i++) worst = MathF.Max(worst, MathF.Abs(got[i] - expected[i]));
        if (worst > 2e-3f)
            throw new Exception($"tiled GroupNorm diverged from full (worst |Δ|={worst:E3}) on {BackendName}");
        Console.WriteLine($"[TiledGroupNorm] 2-tile global-stat GroupNorm == full GroupNorm (worst |Δ|={worst:E3}) on {BackendName}");
    });

    // Extract a row-band [C, bandH, W] (channel-major) from full [C,H,W].
    private static float[] Extract(float[] full, int C, int H, int W, int y0, int bandH)
    {
        var o = new float[C * bandH * W];
        for (int c = 0; c < C; c++)
            for (int yy = 0; yy < bandH; yy++)
                Array.Copy(full, (long)c * H * W + (y0 + yy) * W, o, (long)c * bandH * W + yy * W, W);
        return o;
    }
    private static void Insert(float[] full, float[] band, int C, int H, int W, int y0, int bandH)
    {
        for (int c = 0; c < C; c++)
            for (int yy = 0; yy < bandH; yy++)
                Array.Copy(band, (long)c * bandH * W + yy * W, full, (long)c * H * W + (y0 + yy) * W, W);
    }
}
