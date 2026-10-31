using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.StableDiffusion;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// Token merging for Stable Diffusion's self-attention (<see cref="SdUNet.TokenMerge"/>): ToMe for SD (Bolya and Hoffman, "Token Merging for Fast Stable Diffusion", CVPRW 2023;
/// github.com/dbolya/tomesd, MIT), the algorithm of <c>bipartite_soft_matching_random2d</c>. The tokens of a self-attention at a merged resolution are split into destinations
/// (one per <see cref="Sx"/> x <see cref="Sy"/> window: its top-left token, or one drawn at random) and sources; each source is matched to its most similar destination
/// (cosine similarity), the <c>r = (int)(tokens * Ratio)</c> best-matched sources are averaged into their destinations, and after the attention every merged source takes its
/// destination's output.
/// <para>
/// Two placements. <see cref="MergeKeysAndValues"/> = false is ComfyUI's TomePatchModel node: only the queries are merged (the similarity of the normalised hidden states the
/// attention projects; keys and values keep every token), so the saving is the attention's query rows. True is tomesd's own patch: the similarity of the block's input, and
/// queries, keys and values all merged (attention over (N - r)^2).
/// </para>
/// <para>
/// A layer is merged when its downsampling from the latent, ceil(sqrt(latent tokens / tokens)), is at most <see cref="MaxDownsample"/> (1 = the full-resolution layers only,
/// both originals' default). ComfyUI's defaults: ratio 0.3, queries only, random destinations; tomesd's: ratio 0.5, all three merged, random destinations. Random destinations
/// draw from <see cref="Rng"/> (null = a shared System.Random): the originals draw from torch's generator, so only <see cref="RandomDestinations"/> = false is reproducible
/// against them.
/// </para>
/// </summary>
public sealed record SdTokenMerge(float Ratio = 0.3f, bool MergeKeysAndValues = false, int MaxDownsample = 1, int Sx = 2, int Sy = 2, bool RandomDestinations = true, Random? Rng = null);

/// <summary>How <see cref="SdTokenDownsample"/> samples the keys and values.</summary>
public enum SdKvDownsampling
{
    /// <summary>torch's "nearest": source index floor(i * in / out) (the window's first token).</summary>
    Nearest,
    /// <summary>torch's "nearest-exact": floor((i + 0.5) * in / out) (for a factor of 2, the window's last token).</summary>
    NearestExact,
    /// <summary>torch's "area" (adaptive average pooling): the mean of the window [floor(i * in / out), ceil((i + 1) * in / out)).</summary>
    Area,
}

/// <summary>
/// Token downsampling (ToDo: Smith, Saxena and Saha, "ToDo: Token Downsampling for Efficient Generation of High-Resolution Images", IJCAI 2024): the self-attention keeps every
/// query and attends to keys and values sampled on a grid <see cref="Factor"/> times coarser per side (attention cost / Factor^2), at the full-resolution layers; the second
/// resolution uses <see cref="Factor2"/> (1 = untouched). The paper samples the nearest token; <see cref="SdKvDownsampling.Area"/> averages the window instead.
/// No similarity search and no unmerging: the cheapest of the token-reduction methods.
/// </summary>
public sealed record SdTokenDownsample(int Factor = 2, int Factor2 = 1, SdKvDownsampling Mode = SdKvDownsampling.Nearest);

/// <summary>
/// The GPU side of <see cref="SdTokenMerge"/> and <see cref="SdTokenDownsample"/>: every step on the device (no host readback). The similarity is an fp32 GEMM in row chunks;
/// the sort of the sources by score is a rank count (rank = how many sources score higher, ties by index); each destination's merged sources are gathered by a count, a prefix
/// sum and a fill. The merged rows are [unmerged sources in score order, then the destinations], tomesd's order, so each token's row follows from its rank.
/// </summary>
internal sealed class SdTokenReducer : IDisposable
{
    private readonly Accelerator _acc;
    private readonly MatMulKernel _mm;
    private readonly Dictionary<string, MemoryBuffer1D<float, Stride1D.Dense>> _f = new();
    private readonly Dictionary<string, MemoryBuffer1D<int, Stride1D.Dense>> _i = new();
    private readonly List<IDisposable> _retired = new();
    private readonly Dictionary<(int, int, int, int), (MemoryBuffer1D<int, Stride1D.Dense> Src, MemoryBuffer1D<int, Stride1D.Dense> Dst, MemoryBuffer1D<int, Stride1D.Dense> Role, int Ns, int Nd)> _fixed = new();
    private readonly List<(MemoryBuffer1D<int, Stride1D.Dense> Src, MemoryBuffer1D<int, Stride1D.Dense> Dst, MemoryBuffer1D<int, Stride1D.Dense> Role)> _random = new();
    private int _randomCall;
    private static readonly Random SharedRng = new();

    /// <summary>Most similarity rows per GEMM launch: bounds the scratch (rows x destinations floats) and the length of one dispatch.</summary>
    public int ScoreChunkRows { get; set; } = 2048;

    public SdTokenReducer(Accelerator accelerator) { _acc = accelerator; _mm = new MatMulKernel(accelerator); }

    private FV F(string name, long n)
    {
        if (!_f.TryGetValue(name, out var b) || b.Length < n) { if (b != null) _retired.Add(b); _f[name] = b = _acc.Allocate1D<float>(Math.Max(n, 64)); }
        return b.View.SubView(0, n);
    }
    private IV I(string name, long n)
    {
        if (!_i.TryGetValue(name, out var b) || b.Length < n) { if (b != null) _retired.Add(b); _i[name] = b = _acc.Allocate1D<int>(Math.Max(n, 64)); }
        return b.View.SubView(0, n);
    }

    /// <summary>Called at the start of every prediction: random destination tables are drawn again, one buffer set per merged layer call.</summary>
    public void BeginPrediction() => _randomCall = 0;

    /// <summary>
    /// The merge plan of one layer: h x w tokens; null when nothing is merged (r = 0, or a side of 1, as the originals). Destination choice: deterministic tables are uploaded once
    /// per shape; random ones into a buffer per layer call of the prediction (a buffer a queued dispatch may still read is never written).
    /// </summary>
    public Plan? Prepare(SdTokenMerge tm, int h, int w)
    {
        int N = h * w, r = (int)(N * tm.Ratio);
        if (r <= 0 || h == 1 || w == 1) return null;
        int sx = tm.Sx, sy = tm.Sy, hsy = h / sy, wsx = w / sx;
        if (hsy == 0 || wsx == 0) return null;
        int Nd = hsy * wsx, Ns = N - Nd;
        r = Math.Min(r, Ns);
        IV src, dst, role;
        if (!tm.RandomDestinations)
        {
            var key = (h, w, sx, sy);
            if (!_fixed.TryGetValue(key, out var t))
            {
                var (s, d, ro) = Tables(h, w, sx, sy, null);
                t = (Upload(s), Upload(d), Upload(ro), Ns, Nd);
                _fixed[key] = t;
            }
            src = t.Src.View; dst = t.Dst.View; role = t.Role.View;
        }
        else
        {
            var (s, d, ro) = Tables(h, w, sx, sy, tm.Rng ?? SharedRng);
            if (_randomCall == _random.Count) _random.Add((Upload(s), Upload(d), Upload(ro)));
            var slot = _random[_randomCall];
            if (slot.Src.Length < s.Length || slot.Dst.Length < d.Length || slot.Role.Length < ro.Length)
            {
                _retired.Add(slot.Src); _retired.Add(slot.Dst); _retired.Add(slot.Role);
                slot = (Upload(s), Upload(d), Upload(ro));
                _random[_randomCall] = slot;
            }
            else { slot.Src.View.SubView(0, s.Length).CopyFromCPU(s); slot.Dst.View.SubView(0, d.Length).CopyFromCPU(d); slot.Role.View.SubView(0, ro.Length).CopyFromCPU(ro); }
            _randomCall++;
            src = slot.Src.View.SubView(0, Ns); dst = slot.Dst.View.SubView(0, Nd); role = slot.Role.View.SubView(0, N);
        }
        return new Plan(N, Ns, Nd, r, src, dst, role);
    }

    private MemoryBuffer1D<int, Stride1D.Dense> Upload(int[] a) { var b = _acc.Allocate1D<int>(Math.Max(a.Length, 1)); b.View.SubView(0, a.Length).CopyFromCPU(a); return b; }

    /// <summary>
    /// tomesd's partition: one destination per sy x sx window (the top-left token, or a random one), every other token a source, including the rows and columns past the last
    /// whole window. Sources and destinations in token order; role[t] = source index, or -(destination index) - 1.
    /// </summary>
    internal static (int[] Src, int[] Dst, int[] Role) Tables(int h, int w, int sx, int sy, Random? rng)
    {
        int hsy = h / sy, wsx = w / sx;
        var isDst = new bool[h * w];
        for (int by = 0; by < hsy; by++)
            for (int bx = 0; bx < wsx; bx++)
            {
                int k = rng == null ? 0 : rng.Next(sy * sx);
                isDst[(by * sy + k / sx) * w + bx * sx + k % sx] = true;
            }
        var src = new List<int>(); var dst = new List<int>(); var role = new int[h * w];
        for (int t = 0; t < h * w; t++)
            if (isDst[t]) { role[t] = -dst.Count - 1; dst.Add(t); }
            else { role[t] = src.Count; src.Add(t); }
        return (src.ToArray(), dst.ToArray(), role);
    }

    public sealed record Plan(int N, int Ns, int Nd, int R, IV Src, IV Dst, IV Role)
    {
        /// <summary>Rows after the merge: the unmerged sources, then the destinations.</summary>
        public int M => N - R;
    }

    /// <summary>
    /// The matching: cosine similarity of the metric rows (<paramref name="metric"/> [N, stride], columns [off, off + C)) between every source and every destination, each
    /// source's best destination and score, the sources ranked by score, and each destination's list of merged sources.
    /// </summary>
    public void Match(Plan p, FV metric, int stride, int off, int C)
    {
        var inv = F("inv", p.N);
        LoadKernels();
        _rowInvNorm!(p.N, metric, inv, stride, off, C);
        var a = F("a", (long)p.Ns * C); var bT = F("bT", (long)C * p.Nd);
        _gatherNorm!((int)((long)p.Ns * C), metric, inv, p.Src, a, stride, off, C, 0, p.Ns);
        _gatherNorm!((int)((long)p.Nd * C), metric, inv, p.Dst, bT, stride, off, C, 1, p.Nd);
        var nodeMax = F("nodeMax", p.Ns); var nodeIdx = I("nodeIdx", p.Ns);
        int chunk = Math.Max(1, Math.Min(p.Ns, ScoreChunkRows));
        var scores = F("scores", (long)chunk * p.Nd);
        for (int r0 = 0; r0 < p.Ns; r0 += chunk)
        {
            int rows = Math.Min(chunk, p.Ns - r0);
            var aRows = a.SubView((long)r0 * C, (long)rows * C);
            var s = scores.SubView(0, (long)rows * p.Nd);
            if (_acc.AcceleratorType == AcceleratorType.WebGPU) _mm.MatMulTiled16(aRows, bT, s, rows, C, p.Nd);   // the register-blocked GEMM spills on WebGPU (MatMulKernel)
            else _mm.MatMul(aRows, bT, s, rows, C, p.Nd);
            _rowArgMax!(rows, s, nodeMax, nodeIdx, r0, p.Nd);
        }
        var rank = I("rank", p.Ns); var invRank = I("invRank", p.Ns);
        _rank!(p.Ns, nodeMax, rank, p.Ns);
        _invRank!(p.Ns, rank, invRank);
        var cnt = I("cnt", p.Nd); var ofs = I("ofs", p.Nd); var members = I("members", Math.Max(p.R, 1));
        _count!(p.Nd, rank, nodeIdx, cnt, p.R, p.Ns);
        _prefix!(1, cnt, ofs, p.Nd);
        _fill!(p.Nd, rank, nodeIdx, ofs, members, p.R, p.Ns);
    }

    /// <summary>
    /// The merged rows of <paramref name="x"/> [N, stride] columns [off, off + width): dst [M, width] with the unmerged sources (in score order) and then each destination
    /// averaged with its merged sources (tomesd's scatter_reduce "mean", the destination included).
    /// </summary>
    public void Merge(Plan p, FV x, int stride, int off, int width, FV dst)
    {
        LoadKernels();
        _merge!((int)((long)p.M * width), x, dst, stride, off, width, p.Src, p.Dst, I("invRank", p.Ns), I("cnt", p.Nd), I("ofs", p.Nd), I("members", Math.Max(p.R, 1)), p.Ns, p.R);
    }

    /// <summary>Back to N rows: <paramref name="dst"/> [N, C] row t = <paramref name="merged"/> row of t's representative (itself, or the destination it was merged into).</summary>
    public void Unmerge(Plan p, FV merged, FV dst, int C)
    {
        LoadKernels();
        _unmerge!((int)((long)p.N * C), merged, dst, p.Role, I("rank", p.Ns), I("nodeIdx", p.Ns), C, p.Ns, p.R);
    }

    /// <summary>
    /// ToDo's keys and values: from the fused q|k|v rows of an h x w grid, k and v of the oh x ow grid (sampled or averaged per <paramref name="mode"/>) as [oh * ow, 2C] (k | v).
    /// </summary>
    public void DownsampleKv(FV qkv, FV kv, int h, int w, int oh, int ow, int C, SdKvDownsampling mode)
    {
        LoadKernels();
        _downKv!((int)((long)oh * ow * 2 * C), qkv, kv, h, w, oh, ow, C, (int)mode);
    }

    // ---------------------------------------------------------------------------------------------------
    // kernels
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, FV, int, int, int>? _rowInvNorm;
    private Action<Index1D, FV, FV, IV, FV, int, int, int, int, int>? _gatherNorm;
    private Action<Index1D, FV, FV, IV, int, int>? _rowArgMax;
    private Action<Index1D, FV, IV, int>? _rank;
    private Action<Index1D, IV, IV>? _invRank;
    private Action<Index1D, IV, IV, IV, int, int>? _count;
    private Action<Index1D, IV, IV, int>? _prefix;
    private Action<Index1D, IV, IV, IV, IV, int, int>? _fill;
    private Action<Index1D, FV, FV, int, int, int, IV, IV, IV, IV, IV, IV, int, int>? _merge;
    private Action<Index1D, FV, FV, IV, IV, IV, int, int, int>? _unmerge;
    private Action<Index1D, FV, FV, int, int, int, int, int, int>? _downKv;

    private void LoadKernels()
    {
        if (_rowInvNorm != null) return;
        _rowInvNorm = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int>(RowInvNormImpl);
        _gatherNorm = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, FV, int, int, int, int, int>(GatherNormImpl);
        _rowArgMax = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, int, int>(RowArgMaxImpl);
        _rank = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, int>(RankImpl);
        _invRank = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(InvRankImpl);
        _count = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, int, int>(CountImpl);
        _prefix = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, int>(PrefixImpl);
        _fill = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV, int, int>(FillImpl);
        _merge = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int, IV, IV, IV, IV, IV, IV, int, int>(MergeImpl);
        _unmerge = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, IV, IV, int, int, int>(UnmergeImpl);
        _downKv = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int, int, int, int>(DownKvImpl);
    }

    // 1 / |row| of x[t, off .. off + C) (torch: metric / metric.norm(dim=-1, keepdim=True), no epsilon)
    private static void RowInvNormImpl(Index1D t, FV x, FV inv, int stride, int off, int C)
    {
        long b = (long)(int)t * stride + off;
        float s = 0f;
        for (int c = 0; c < C; c++) { float v = x[b + c]; s += v * v; }
        inv[t] = 1f / MathF.Sqrt(s);
    }

    // the normalised metric rows of the tokens in idx: transpose 0 -> o[k, c] (the GEMM's A), 1 -> o[c, k] with leading dimension n (its B)
    private static void GatherNormImpl(Index1D i, FV x, FV inv, IV idx, FV o, int stride, int off, int C, int transpose, int n)
    {
        int ii = i, k = ii / C, c = ii % C, t = idx[k];
        float v = x[(long)t * stride + off + c] * inv[t];
        if (transpose == 0) o[ii] = v;
        else o[(long)c * n + k] = v;
    }

    // per source row of the score chunk: the best destination (the first maximum, as torch's max) and its score
    private static void RowArgMaxImpl(Index1D r, FV s, FV nodeMax, IV nodeIdx, int r0, int Nd)
    {
        long b = (long)(int)r * Nd;
        float best = s[b]; int bi = 0;
        for (int j = 1; j < Nd; j++) { float v = s[b + j]; if (v > best) { best = v; bi = j; } }
        nodeMax[r0 + r] = best; nodeIdx[r0 + r] = bi;
    }

    // the position of source i in the descending order of the scores (argsort(descending=True)): how many score higher, ties by index
    private static void RankImpl(Index1D i, FV v, IV rank, int n)
    {
        float vi = v[i]; int k = 0;
        for (int j = 0; j < n; j++) { float vj = v[j]; if (vj > vi || (vj == vi && j < i)) k++; }
        rank[i] = k;
    }

    private static void InvRankImpl(Index1D i, IV rank, IV invRank) => invRank[rank[i]] = i;

    // destination j: how many of the r best sources chose it
    private static void CountImpl(Index1D j, IV rank, IV nodeIdx, IV cnt, int r, int Ns)
    {
        int k = 0;
        for (int i = 0; i < Ns; i++) if (rank[i] < r && nodeIdx[i] == j) k++;
        cnt[j] = k;
    }

    // exclusive prefix sum of the counts (one thread: a few thousand additions)
    private static void PrefixImpl(Index1D t, IV cnt, IV ofs, int n)
    {
        int s = 0;
        for (int j = 0; j < n; j++) { ofs[j] = s; s += cnt[j]; }
    }

    // destination j's merged sources, in source order
    private static void FillImpl(Index1D j, IV rank, IV nodeIdx, IV ofs, IV members, int r, int Ns)
    {
        int k = ofs[j];
        for (int i = 0; i < Ns; i++) if (rank[i] < r && nodeIdx[i] == j) members[k++] = i;
    }

    // merged row m: m < Ns - r -> the unmerged source of rank m + r; else destination j = m - (Ns - r), the mean of itself and its merged sources
    private static void MergeImpl(Index1D i, FV x, FV o, int stride, int off, int width, IV src, IV dst, IV invRank, IV cnt, IV ofs, IV members, int Ns, int r)
    {
        int ii = i, m = ii / width, c = ii % width, unm = Ns - r;
        if (m < unm) { o[ii] = x[(long)src[invRank[m + r]] * stride + off + c]; return; }
        int j = m - unm, n = cnt[j], b = ofs[j];
        float s = x[(long)dst[j] * stride + off + c];
        for (int k = 0; k < n; k++) s += x[(long)src[members[b + k]] * stride + off + c];
        o[ii] = s / (n + 1);
    }

    // token t's row of the merged output: a destination's own row, an unmerged source's (rank - r), a merged source's destination row
    private static void UnmergeImpl(Index1D i, FV merged, FV o, IV role, IV rank, IV nodeIdx, int C, int Ns, int r)
    {
        int ii = i, t = ii / C, c = ii % C, ro = role[t], unm = Ns - r;
        int row;
        if (ro < 0) row = unm + (-ro - 1);
        else { int k = rank[ro]; row = k >= r ? k - r : unm + nodeIdx[ro]; }
        o[ii] = merged[(long)row * C + c];
    }

    // ToDo: k | v of the oh x ow grid from the fused q|k|v [h * w, 3C] (k at column C, v at 2C): nearest (0), nearest-exact (1) or area (2), torch's F.interpolate index rules
    private static void DownKvImpl(Index1D i, FV qkv, FV kv, int h, int w, int oh, int ow, int C, int mode)
    {
        int ii = i, C2 = 2 * C, col = ii % C2, p = ii / C2, ox = p % ow, oy = p / ow;
        float sh = (float)h / oh, sw = (float)w / ow;
        long stride = 3L * C; int qc = C + col;
        if (mode == 2)
        {
            // adaptive_avg_pool2d's integer window: [floor(i * in / out), ceil((i + 1) * in / out))
            int y0 = oy * h / oh, y1 = ((oy + 1) * h + oh - 1) / oh, x0 = ox * w / ow, x1 = ((ox + 1) * w + ow - 1) / ow;
            float s = 0f;
            for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) s += qkv[(long)(y * w + x) * stride + qc];
            kv[ii] = s / ((y1 - y0) * (x1 - x0));
            return;
        }
        int sy = mode == 1 ? (int)MathF.Floor((oy + 0.5f) * sh) : (int)MathF.Floor(oy * sh);
        int sx = mode == 1 ? (int)MathF.Floor((ox + 0.5f) * sw) : (int)MathF.Floor(ox * sw);
        sy = sy < h - 1 ? sy : h - 1; sx = sx < w - 1 ? sx : w - 1;
        kv[ii] = qkv[(long)(sy * w + sx) * stride + qc];
    }

    public void Dispose()
    {
        foreach (var b in _f.Values) b.Dispose();
        foreach (var b in _i.Values) b.Dispose();
        foreach (var t in _fixed.Values) { t.Src.Dispose(); t.Dst.Dispose(); t.Role.Dispose(); }
        foreach (var t in _random) { t.Src.Dispose(); t.Dst.Dispose(); t.Role.Dispose(); }
        foreach (var r in _retired) r.Dispose();
        _f.Clear(); _i.Clear(); _fixed.Clear(); _random.Clear(); _retired.Clear();
    }
}
