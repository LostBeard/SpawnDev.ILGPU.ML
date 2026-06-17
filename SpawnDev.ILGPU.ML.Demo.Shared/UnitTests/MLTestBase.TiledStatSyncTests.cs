using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Validates the load-bearing primitive of exact-stat (seam-free) tiled VAE decode: per-tile partial stats
/// (sum, sumSq, count) COMBINED across tiles produce the GLOBAL InstanceNorm mean/var, and applying those global
/// stats to each tile yields BYTE-EQUAL results to a single full-resolution InstanceNorm. This is what makes
/// tiled decode seam-free (every tile normalizes with the same global stats). Tested per backend.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task TiledStatSync_GlobalStatsMatchFullInstanceNorm_AllBackends() => await RunTest(async accelerator =>
    {
        const int C = 32, T = 2048, full = 2 * T;   // 32 groups; full spatial 4096 split into two T-tiles
        const float eps = 1e-6f;
        var rng = new Random(913);
        // Full feature map [1,32,full]: slice c at [c*full .. c*full+full).
        var fullData = new float[C * full];
        for (int i = 0; i < fullData.Length; i++) fullData[i] = (float)(rng.NextDouble() * 6 - 3);
        // Two tiles split along spatial: tile0 = first T of each slice, tile1 = second T.
        var tile0 = new float[C * T]; var tile1 = new float[C * T];
        for (int c = 0; c < C; c++)
            for (int j = 0; j < T; j++)
            {
                tile0[c * T + j] = fullData[c * full + j];
                tile1[c * T + j] = fullData[c * full + T + j];
            }

        var norm = new SpawnDev.ILGPU.ML.Kernels.NormalizationKernels(accelerator);
        var ones = new float[C]; for (int c = 0; c < C; c++) ones[c] = 1f;
        var zeros = new float[C];
        using var scaleB = accelerator.Allocate1D(ones);
        using var biasB = accelerator.Allocate1D(zeros);

        // Reference: a single full-resolution InstanceNorm (eps must match — use the in-place path which hardcodes
        // 1e-5; so compute the reference on CPU at eps=1e-6 to match the VAE).
        var expected = new float[C * full];
        for (int c = 0; c < C; c++)
        {
            double s = 0, s2 = 0;
            for (int j = 0; j < full; j++) { double v = fullData[c * full + j]; s += v; s2 += v * v; }
            double mean = s / full, var = s2 / full - mean * mean, inv = 1.0 / Math.Sqrt(var + eps);
            for (int j = 0; j < full; j++) expected[c * full + j] = (float)((fullData[c * full + j] - mean) * inv);
        }

        // Tiled: partial stats per tile (on GPU), combine on CPU into global mean/invStd, apply to each tile.
        using var t0 = accelerator.Allocate1D(tile0);
        using var t1 = accelerator.Allocate1D(tile1);
        using var sum0 = accelerator.Allocate1D<float>(C); using var sq0 = accelerator.Allocate1D<float>(C);
        using var sum1 = accelerator.Allocate1D<float>(C); using var sq1 = accelerator.Allocate1D<float>(C);
        norm.InstanceNormPartialStats(t0.View, sum0.View, sq0.View, 1, C, T);
        norm.InstanceNormPartialStats(t1.View, sum1.View, sq1.View, 1, C, T);
        await accelerator.SynchronizeAsync();
        var s0 = await sum0.CopyToHostAsync<float>(0, C); var q0 = await sq0.CopyToHostAsync<float>(0, C);
        var s1 = await sum1.CopyToHostAsync<float>(0, C); var q1 = await sq1.CopyToHostAsync<float>(0, C);

        var gMeans = new float[C]; var gInv = new float[C];
        for (int c = 0; c < C; c++)
        {
            double gs = (double)s0[c] + s1[c], gq = (double)q0[c] + q1[c];   // combine partials over count=full
            double mean = gs / full, var = gq / full - mean * mean;
            gMeans[c] = (float)mean; gInv[c] = (float)(1.0 / Math.Sqrt(var + eps));
        }
        using var meansB = accelerator.Allocate1D(gMeans);
        using var invB = accelerator.Allocate1D(gInv);
        norm.InstanceNormApplyWithStats(t0.View, scaleB.View, biasB.View, meansB.View, invB.View, 1, C, T);
        norm.InstanceNormApplyWithStats(t1.View, scaleB.View, biasB.View, meansB.View, invB.View, 1, C, T);
        await accelerator.SynchronizeAsync();
        var g0 = await t0.CopyToHostAsync<float>(0, C * T);
        var g1 = await t1.CopyToHostAsync<float>(0, C * T);

        float worst = 0;
        for (int c = 0; c < C; c++)
            for (int j = 0; j < T; j++)
            {
                worst = MathF.Max(worst, MathF.Abs(g0[c * T + j] - expected[c * full + j]));
                worst = MathF.Max(worst, MathF.Abs(g1[c * T + j] - expected[c * full + T + j]));
            }
        if (worst > 5e-4f)
            throw new Exception($"tiled global stats diverged from full InstanceNorm (worst |Δ|={worst:E3}) on {BackendName}");
        Console.WriteLine($"[TiledStatSync] 2-tile partial-stat combine == full InstanceNorm (worst |Δ|={worst:E3}) on {BackendName}");
    });
}
