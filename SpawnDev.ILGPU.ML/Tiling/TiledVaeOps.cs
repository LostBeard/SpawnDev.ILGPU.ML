using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;

namespace SpawnDev.ILGPU.ML.Tiling;

/// <summary>
/// The tiled-decode building blocks: each op processes a <see cref="TiledFeatureMap"/> ONE tile on the GPU at a
/// time (the CPU-backed tiles are the offload that bounds GPU peak), reusing the standard kernels. 3×3 SAME convs
/// refresh halos then run pad-0 on the padded tile; pointwise ops (SiLU, Add) run on cores; GroupNorm does the
/// two-pass global-stat sync (verified primitives). Composing these reproduces a VAE up-block's forward exactly
/// but tiled + seam-free. Plan: Plans/exact-tiled-vae-decode-2026-06-16.md (steps 3-4).
/// </summary>
public sealed class TiledVaeOps : IDisposable
{
    private readonly Accelerator _acc;
    private readonly Conv2DKernel _conv;
    private readonly ActivationKernels _act;
    private readonly NormalizationKernels _norm;
    private readonly ElementWiseKernels _ew;

    public TiledVaeOps(Accelerator acc)
    {
        _acc = acc; _conv = new Conv2DKernel(acc); _act = new ActivationKernels(acc);
        _norm = new NormalizationKernels(acc); _ew = new ElementWiseKernels(acc);
    }

    /// <summary>3×3 SAME conv (pad 1) tiled: refresh halos, then per-tile pad-0 conv on the padded tile → a new
    /// map with <paramref name="outC"/> channels (core stays same spatial size).</summary>
    public async Task<TiledFeatureMap> Conv3x3(TiledFeatureMap inMap, ArrayView1D<float, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> b, int inC, int outC)
    {
        inMap.RefreshHalos();
        var outMap = TiledFeatureMap.Allocate(outC, inMap.Height, inMap.Width, inMap.Rows, inMap.Cols, inMap.Halo);
        for (int r = 0; r < inMap.Rows; r++)
            for (int c = 0; c < inMap.Cols; c++)
            {
                int ph = inMap.PaddedH(r), pw = inMap.PaddedW(c), oh = ph - 2, ow = pw - 2;
                using var inB = _acc.Allocate1D(inMap.Tile(r, c));
                using var outB = _acc.Allocate1D<float>(outC * oh * ow);
                _conv.ForwardPadded(inB.View, w, b, outB.View, inC, ph, pw, outC, 3, 3, 1, 0, 0, 0, 0, 1, 1);
                await _acc.SynchronizeAsync();
                outMap.WriteCore(r, c, await outB.CopyToHostAsync<float>(0, outC * oh * ow));
            }
        return outMap;
    }

    /// <summary>1×1 conv (resnet shortcut / channel change), halo-free: per-tile conv on the core.</summary>
    public async Task<TiledFeatureMap> Conv1x1(TiledFeatureMap inMap, ArrayView1D<float, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> b, int inC, int outC)
    {
        var outMap = TiledFeatureMap.Allocate(outC, inMap.Height, inMap.Width, inMap.Rows, inMap.Cols, inMap.Halo);
        for (int r = 0; r < inMap.Rows; r++)
            for (int c = 0; c < inMap.Cols; c++)
            {
                int ch = inMap.CoreH(r), cw = inMap.CoreW(c);
                using var inB = _acc.Allocate1D(inMap.ReadCore(r, c));
                using var outB = _acc.Allocate1D<float>(outC * ch * cw);
                _conv.ForwardPadded(inB.View, w, b, outB.View, inC, ch, cw, outC, 1, 1, 1, 0, 0, 0, 0, 1, 1);
                await _acc.SynchronizeAsync();
                outMap.WriteCore(r, c, await outB.CopyToHostAsync<float>(0, outC * ch * cw));
            }
        return outMap;
    }

    /// <summary>SiLU (x·sigmoid(x)) in place on the cores (pointwise — no halo needed).</summary>
    public async Task SiLU(TiledFeatureMap map, int C)
    {
        for (int r = 0; r < map.Rows; r++)
            for (int c = 0; c < map.Cols; c++)
            {
                int cnt = C * map.CoreH(r) * map.CoreW(c);
                using var buf = _acc.Allocate1D(map.ReadCore(r, c));
                _act.SiLUInPlace(buf.View, cnt);
                await _acc.SynchronizeAsync();
                map.WriteCore(r, c, await buf.CopyToHostAsync<float>(0, cnt));
            }
    }

    /// <summary>Residual add (a += b) in place on cores (a and b must share the grid + channels).</summary>
    public async Task AddInPlace(TiledFeatureMap a, TiledFeatureMap b, int C)
    {
        for (int r = 0; r < a.Rows; r++)
            for (int c = 0; c < a.Cols; c++)
            {
                int cnt = C * a.CoreH(r) * a.CoreW(c);
                using var aB = _acc.Allocate1D(a.ReadCore(r, c));
                using var bB = _acc.Allocate1D(b.ReadCore(r, c));
                _ew.AddInPlace(aB.View, bB.View, cnt);
                await _acc.SynchronizeAsync();
                a.WriteCore(r, c, await aB.CopyToHostAsync<float>(0, cnt));
            }
    }

    /// <summary>GroupNorm in place with GLOBAL per-group stats across tiles (seam-free), then per-channel γ/β.
    /// Two passes: (1) accumulate per-group partial sum/sumSq over every tile's core; (2) apply the global stats
    /// + γ/β to each tile. groups divides C; eps matches the model (VAE = 1e-6).</summary>
    public async Task GroupNorm(TiledFeatureMap map, ArrayView1D<float, Stride1D.Dense> gamma,
        ArrayView1D<float, Stride1D.Dense> beta, int C, int groups, float eps)
    {
        int cpg = C / groups;
        // Pass 1: global per-group sum/sumSq/count.
        var gSum = new double[groups]; var gSq = new double[groups]; long gCount = 0;
        for (int r = 0; r < map.Rows; r++)
            for (int c = 0; c < map.Cols; c++)
            {
                int coreSp = map.CoreH(r) * map.CoreW(c), groupSpan = cpg * coreSp;
                using var buf = _acc.Allocate1D(map.ReadCore(r, c));
                using var sums = _acc.Allocate1D<float>(groups); using var sqs = _acc.Allocate1D<float>(groups);
                _norm.InstanceNormPartialStats(buf.View, sums.View, sqs.View, 1, groups, groupSpan);
                await _acc.SynchronizeAsync();
                var s = await sums.CopyToHostAsync<float>(0, groups); var q = await sqs.CopyToHostAsync<float>(0, groups);
                for (int g = 0; g < groups; g++) { gSum[g] += s[g]; gSq[g] += q[g]; }
                gCount += groupSpan;
            }
        var means = new float[groups]; var inv = new float[groups];
        for (int g = 0; g < groups; g++)
        {
            double mean = gSum[g] / gCount, var = gSq[g] / gCount - mean * mean;
            means[g] = (float)mean; inv[g] = (float)(1.0 / Math.Sqrt(var + eps));
        }
        var onesG = new float[groups]; var zerosG = new float[groups]; for (int g = 0; g < groups; g++) onesG[g] = 1f;
        var zerosC = new float[C]; var onesCinv = new float[C]; for (int i = 0; i < C; i++) onesCinv[i] = 1f;
        using var meansB = _acc.Allocate1D(means); using var invB = _acc.Allocate1D(inv);
        using var onesGB = _acc.Allocate1D(onesG); using var zerosGB = _acc.Allocate1D(zerosG);
        using var zerosCB = _acc.Allocate1D(zerosC); using var onesCinvB = _acc.Allocate1D(onesCinv);

        // Pass 2: per tile — global per-group normalize, then per-channel γ/β (apply-with-stats, mean=0/invStd=1).
        for (int r = 0; r < map.Rows; r++)
            for (int c = 0; c < map.Cols; c++)
            {
                int coreSp = map.CoreH(r) * map.CoreW(c), groupSpan = cpg * coreSp;
                using var buf = _acc.Allocate1D(map.ReadCore(r, c));
                _norm.InstanceNormApplyWithStats(buf.View, onesGB.View, zerosGB.View, meansB.View, invB.View, 1, groups, groupSpan);
                _norm.InstanceNormApplyWithStats(buf.View, gamma, beta, zerosCB.View, onesCinvB.View, 1, C, coreSp);
                await _acc.SynchronizeAsync();
                map.WriteCore(r, c, await buf.CopyToHostAsync<float>(0, C * coreSp));
            }
    }

    public void Dispose()
    {
        (_conv as IDisposable)?.Dispose();
        (_act as IDisposable)?.Dispose();
        _norm.Dispose();
        (_ew as IDisposable)?.Dispose();
    }
}
