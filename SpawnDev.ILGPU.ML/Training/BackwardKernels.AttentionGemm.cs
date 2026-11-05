using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Training;

using FV = ArrayView1D<float, Stride1D.Dense>;

public sealed partial class BackwardKernels
{
    /// <summary>
    /// The attention backward as batched GEMMs over materialised score matrices (default on; false = the blocked kernels, the A/B control).
    /// Per chunk of query heads: S = Q K^T, the row statistics from S, P = exp(S * scale - lse) under the mask, dP = dO V^T,
    /// dS = P (dP - D) * scale, dQ = dS K, dK = dS^T Q, dV = P^T dO; grouped-query heads are summed into their kv head. fp32 throughout.
    /// MEASURED 2026-11-04 (llama3.1:8b, 1,600 tokens): the blocked kernels took 9.24 s of a 21.7 s step for ~42 GFLOP of work (~4.5 GFLOPS);
    /// as GEMMs the same work runs on the register-blocked batched matmul.
    /// </summary>
    public static bool UseGemmAttentionBackward { get; set; } = true;
    /// <summary>Bytes the three [heads, SeqQ, SeqKV] score buffers may take together; the heads are processed in chunks that fit.</summary>
    public static long GemmAttentionScoreBudgetBytes { get; set; } = 384L << 20;

    private MatMulKernel? _attnMm;
    private MemoryBuffer1D<float, Stride1D.Dense>? _gQh, _gdOh, _gdQh, _gKh, _gKhT, _gVhT, _gdKh, _gdVh, _gS, _gdP, _gT;
    private readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _gRetired = new();
    private Action<Index1D, FV, FV, PackDims>? _gPack;
    private Action<Index1D, FV, FV, FV, FV, AttnDims, int>? _gStats;
    private Action<Index1D, FV, FV, AttnDims, int>? _gP;
    private Action<Index1D, FV, FV, FV, AttnDims, int>? _gDs;
    private Action<KernelConfig, FV, FV, int, int, int>? _gTranspose;
    private Action<Index1D, FV, FV, PackDims>? _gScatterQ;
    private Action<Index1D, FV, FV, PackDims>? _gScatterKv;

    /// <summary>Head packing scalars: sequence length, heads of the strided tensor, head dim, first query head of the chunk, query heads per
    /// source head (1 for q / dO, H / KvH for k / v), transpose (0: [c, i, e], 1: [c, e, i]).</summary>
    public struct PackDims
    {
        /// <summary>Rows (sequence positions).</summary>
        public int S;
        /// <summary>Heads in the strided [S, Hs, hd] tensor.</summary>
        public int Hs;
        /// <summary>Head dim.</summary>
        public int Hd;
        /// <summary>First query head of the chunk.</summary>
        public int H0;
        /// <summary>Query heads per source head.</summary>
        public int G;
        /// <summary>1: write [c, e, i].</summary>
        public int Transpose;
        /// <summary>Query heads in the chunk.</summary>
        public int Hc;
    }

    // n plus an eighth, in 4 KiB steps: sequences of varying length re-grow a buffer a few times, not once per new maximum. MEASURED 2026-11-05:
    // the earlier next-power-of-two allocated up to twice the need (the 2,560-token gate / up dy pack: 134 MB for 73), and on a 12 GB card a
    // 2,560-token llama3.1:8b step ran out of memory headroom. The trainer frees replaced buffers after every step (ReleaseRetired).
    internal static long GrowCapacity(long n) => (n + n / 8 + 1023) / 1024 * 1024;

    private static MemoryBuffer1D<float, Stride1D.Dense> Grow(Accelerator acc, ref MemoryBuffer1D<float, Stride1D.Dense>? b, long n,
        List<MemoryBuffer1D<float, Stride1D.Dense>> retired)
    {
        if (b == null || b.Length < n)
        {
            if (b != null) retired.Add(b);   // a queued launch may still read it
            b = acc.Allocate1D<float>(GrowCapacity(n));
        }
        return b;
    }

    private void AttentionGemm(FV q, FV k, FV v, FV outp, FV dout, FV dq, FV dk, FV dv, FV stats, AttnDims d)
    {
        int H = d.Heads, g = d.Heads / d.KvHeads, Sq = d.SeqQ, Skv = d.SeqKV, hd = d.HeadDim;
        // heads per chunk: a multiple of g (a kv head's query heads stay in one chunk), at most H, three score buffers within the budget
        long perHead = 3L * Sq * Skv * sizeof(float);
        int hc = (int)Math.Max(1, Math.Min(H, GemmAttentionScoreBudgetBytes / Math.Max(1, perHead)));
        hc = Math.Max(g, hc / g * g);
        while (H % hc != 0) hc -= g;   // equal chunks (H is a multiple of g, so this stops at g at worst)
        long qn = (long)hc * Sq * hd, kn = (long)hc * Skv * hd, sn = (long)hc * Sq * Skv;
        var Qh = Grow(_acc, ref _gQh, qn, _gRetired); var dOh = Grow(_acc, ref _gdOh, qn, _gRetired); var dQh = Grow(_acc, ref _gdQh, qn, _gRetired);
        var Kh = Grow(_acc, ref _gKh, kn, _gRetired); var KhT = Grow(_acc, ref _gKhT, kn, _gRetired); var VhT = Grow(_acc, ref _gVhT, kn, _gRetired);
        var dKh = Grow(_acc, ref _gdKh, kn, _gRetired); var dVh = Grow(_acc, ref _gdVh, kn, _gRetired);
        // the score buffers are sized by the budget once (hc is chosen to fit it), so they never re-grow
        long scoreCap = Math.Max(sn, GemmAttentionScoreBudgetBytes / 3 / sizeof(float));
        var S = Grow(_acc, ref _gS, scoreCap, _gRetired); var dP = Grow(_acc, ref _gdP, scoreCap, _gRetired); var T = Grow(_acc, ref _gT, scoreCap, _gRetired);
        _attnMm ??= new MatMulKernel(_acc);
        _gPack ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, PackDims>(PackImpl);
        _gStats ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, AttnDims, int>(GemmStatsImpl);
        _gP ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, AttnDims, int>(GemmPImpl);
        _gDs ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, AttnDims, int>(GemmDsImpl);
        _gTranspose ??= _acc.LoadStreamKernel<FV, FV, int, int, int>(BatchTransposeImpl);
        _gScatterQ ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, PackDims>(ScatterQImpl);
        _gScatterKv ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, PackDims>(ScatterKvImpl);
        int tilesQ = (Sq + 31) / 32, tilesK = (Skv + 31) / 32;
        for (int h0 = 0; h0 < H; h0 += hc)
        {
            var pq = new PackDims { S = Sq, Hs = H, Hd = hd, H0 = h0, G = 1, Hc = hc };
            var pk = new PackDims { S = Skv, Hs = d.KvHeads, Hd = hd, H0 = h0, G = g, Hc = hc };
            _gPack((int)qn, q, Qh.View, pq);
            _gPack((int)qn, dout, dOh.View, pq);
            _gPack((int)kn, k, Kh.View, pk);
            pk.Transpose = 1;
            _gPack((int)kn, k, KhT.View, pk);
            _gPack((int)kn, v, VhT.View, pk);
            _attnMm.BatchedMatMul(Qh.View, KhT.View, S.View, hc, Sq, hd, Skv);          // S = Q K^T
            _gStats(hc * Sq, S.View, dout, outp, stats, d, h0);                         // lse and D per row
            _gP((int)sn, S.View, stats, d, h0);                                          // S -> P (masked)
            _attnMm.BatchedMatMul(dOh.View, VhT.View, dP.View, hc, Sq, hd, Skv);        // dP = dO V^T
            _gDs((int)sn, dP.View, S.View, stats, d, h0);                                // dP -> dS = P (dP - D) * scale
            _attnMm.BatchedMatMul(dP.View, Kh.View, dQh.View, hc, Sq, Skv, hd);         // dQ = dS K
            _gTranspose(new KernelConfig(hc * tilesQ * tilesK, 256), dP.View, T.View, Sq, Skv, tilesK);   // T = dS^T
            _attnMm.BatchedMatMul(T.View, Qh.View, dKh.View, hc, Skv, Sq, hd);          // dK = dS^T Q (per query head)
            _gTranspose(new KernelConfig(hc * tilesQ * tilesK, 256), S.View, T.View, Sq, Skv, tilesK);    // T = P^T
            _attnMm.BatchedMatMul(T.View, dOh.View, dVh.View, hc, Skv, Sq, hd);         // dV = P^T dO
            _gScatterQ((int)qn, dQh.View, dq, pq);
            pk.Transpose = 0;
            _gScatterKv((int)(kn / g), dKh.View, dk, pk);
            _gScatterKv((int)(kn / g), dVh.View, dv, pk);
        }
    }

    // dst[idx] for idx over the chunk: [c, i, e] (Transpose 0) or [c, e, i] (1); source head (H0 + c) / G of the strided [S, Hs, Hd] tensor
    private static void PackImpl(Index1D idx, FV src, FV dst, PackDims p)
    {
        int per = p.S * p.Hd;
        int c = idx / per, rem = idx - c * per;
        int i, e;
        if (p.Transpose == 0) { i = rem / p.Hd; e = rem - i * p.Hd; }
        else { e = rem / p.S; i = rem - e * p.S; }
        int sh = (p.H0 + c) / p.G;
        dst[idx] = src[((long)i * p.Hs + sh) * p.Hd + e];
    }

    // one thread per (chunk head c, query row i): the log-sum-exp of the allowed scaled scores of row i (from S), and D = dO . O
    private static void GemmStatsImpl(Index1D r, FV s, FV dout, FV outp, FV stats, AttnDims d, int h0)
    {
        int c = r / d.SeqQ, i = r - c * d.SeqQ, h = h0 + c;
        long rowBase = (long)(int)r * d.SeqKV;
        float m = float.NegativeInfinity;
        for (int j = 0; j < d.SeqKV; j++) if (Allowed(i, j, d)) m = MathF.Max(m, s[rowBase + j] * d.Scale);
        float sum = 0f;
        if (m != float.NegativeInfinity)
            for (int j = 0; j < d.SeqKV; j++) if (Allowed(i, j, d)) sum += MathF.Exp(s[rowBase + j] * d.Scale - m);
        stats[h * d.SeqQ + i] = m == float.NegativeInfinity ? 0f : m + MathF.Log(sum);
        float dd = 0f;
        long ob = ((long)i * d.Heads + h) * d.HeadDim;
        for (int e = 0; e < d.HeadDim; e++) dd += dout[ob + e] * outp[ob + e];
        stats[d.Heads * d.SeqQ + h * d.SeqQ + i] = dd;
    }

    private static void GemmPImpl(Index1D idx, FV s, FV stats, AttnDims d, int h0)
    {
        long r = idx / d.SeqKV;
        int j = (int)(idx - r * d.SeqKV);
        int c = (int)(r / d.SeqQ), i = (int)(r - (long)c * d.SeqQ);
        s[idx] = Allowed(i, j, d) ? MathF.Exp(s[idx] * d.Scale - stats[(h0 + c) * d.SeqQ + i]) : 0f;
    }

    private static void GemmDsImpl(Index1D idx, FV dp, FV p, FV stats, AttnDims d, int h0)
    {
        long r = idx / d.SeqKV;
        int c = (int)(r / d.SeqQ), i = (int)(r - (long)c * d.SeqQ);
        dp[idx] = p[idx] * (dp[idx] - stats[d.Heads * d.SeqQ + (h0 + c) * d.SeqQ + i]) * d.Scale;
    }

    // [batch, R, C] -> [batch, C, R] by 32 x 32 tiles through shared memory (33-float rows: no bank conflicts); grid = batch * tilesR * tilesC
    private static void BatchTransposeImpl(FV src, FV dst, int R, int C, int tilesC)
    {
        var sh = SharedMemory.Allocate<float>(32 * 33);
        int tilesR = (R + 31) / 32, perBatch = tilesR * tilesC;
        int b = Grid.IdxX / perBatch, t = Grid.IdxX - b * perBatch;
        int r0 = (t / tilesC) * 32, c0 = (t % tilesC) * 32;
        int tx = Group.IdxX & 31, ty = Group.IdxX >> 5;
        long bb = (long)b * R * C;
        for (int k = 0; k < 4; k++)
        {
            int rr = ty + 8 * k, r = r0 + rr, cc = c0 + tx;
            sh[rr * 33 + tx] = r < R && cc < C ? src[bb + (long)r * C + cc] : 0f;
        }
        Group.Barrier();
        for (int k = 0; k < 4; k++)
        {
            int cc = ty + 8 * k, col = c0 + cc, r = r0 + tx;
            if (col < C && r < R) dst[bb + (long)col * R + r] = sh[tx * 33 + cc];
        }
    }

    // dq[(i H + H0 + c) hd + e] = dQh[c, i, e]
    private static void ScatterQImpl(Index1D idx, FV src, FV dq, PackDims p)
    {
        int per = p.S * p.Hd;
        int c = idx / per, rem = idx - c * per;
        int i = rem / p.Hd, e = rem - i * p.Hd;
        dq[((long)i * p.Hs + p.H0 + c) * p.Hd + e] = src[idx];
    }

    // one thread per (kv head of the chunk, key j, e): the sum of its G query heads' rows -> dk[(j KvH + kvh) hd + e]
    private static void ScatterKvImpl(Index1D idx, FV src, FV dkv, PackDims p)
    {
        int per = p.S * p.Hd;
        int kl = idx / per, rem = idx - kl * per;
        int j = rem / p.Hd, e = rem - j * p.Hd;
        float s = 0f;
        for (int gg = 0; gg < p.G; gg++) s += src[(long)(kl * p.G + gg) * per + rem];
        int kvh = p.H0 / p.G + kl;
        dkv[((long)j * p.Hs + kvh) * p.Hd + e] = s;
    }

    private void DisposeAttentionGemm()
    {
        foreach (var b in new[] { _gQh, _gdOh, _gdQh, _gKh, _gKhT, _gVhT, _gdKh, _gdVh, _gS, _gdP, _gT }) b?.Dispose();
        _gQh = _gdOh = _gdQh = _gKh = _gKhT = _gVhT = _gdKh = _gdVh = _gS = _gdP = _gT = null;
        foreach (var b in _gRetired) b.Dispose();
        _gRetired.Clear();
    }
}
