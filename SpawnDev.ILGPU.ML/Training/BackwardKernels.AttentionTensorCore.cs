using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Training;

using FV = ArrayView1D<float, Stride1D.Dense>;
using HV16 = ArrayView1D<ushort, Stride1D.Dense>;

public sealed partial class BackwardKernels
{
    /// <summary>
    /// The GEMM attention backward's five products on tensor cores (CUDA sm_75+; default on; false = the fp32 batched matmul, the A/B control).
    /// Per query head: S = Q K^T and dP = dO V^T with K / V as the 16-bit operand, dQ = dS K, dK = dS^T Q, dV = P^T dO with K^T / Q^T / dO^T as
    /// the 16-bit operand; the sequence axes are zero-padded to multiples of 32 (the GEMM's K tile). sm_80+: bf16 operands. sm_75: f16 with a
    /// device-side power-of-two scale on the gradient tensors (dO, dS), exact to undo. MEASURED 2026-11-05 (llama3.1:8b, 1,600 tokens): the fp32
    /// GEMMs ran ~3.4 TFLOP per step at ~3.6 TFLOPS on an RTX 4070 (0.94 s) and 2.8 s on an RTX 2060.
    /// </summary>
    public static bool UseTensorCoreAttentionBackward { get; set; } = true;

    /// <summary>16-bit packing scalars (see <see cref="Pack16Impl"/>).</summary>
    public struct Pack16Dims
    {
        /// <summary>Rows of the source (sequence length) and the padded row count written.</summary>
        public int S, Sp;
        /// <summary>Heads of the strided [S, Hs, Hd] source, head dim, first query head, query heads per source head.</summary>
        public int Hs, Hd, H0, G;
        /// <summary>1: write [c, e, i] (transposed), 0: [c, i, e].</summary>
        public int Transpose;
        /// <summary>1: f16 (sm_75), 0: bf16.</summary>
        public int Half;
        /// <summary>1: multiply by sc[0] before rounding.</summary>
        public int UseScale;
    }

    private MemoryBuffer1D<ushort, Stride1D.Dense>? _tKb, _tVb, _tKTb, _tQTb, _tdOTb;
    private readonly List<MemoryBuffer1D<ushort, Stride1D.Dense>> _tRetired16 = new();
    private MemoryBuffer1D<float, Stride1D.Dense>? _tS, _tdP, _tT, _tScOne, _tScDo, _tScDs, _tScDv;
    private MemoryBuffer1D<int, Stride1D.Dense>? _tMaxDo, _tMaxDs;
    private Action<Index1D, FV, HV16, Pack16Dims, FV>? _pack16;
    private Action<KernelConfig, FV, FV, FV, FV, AttnDims, int, int>? _tStats;
    private Action<Index1D, FV, FV, AttnDims, int, int>? _tP;
    private Action<Index1D, FV, FV, FV, AttnDims, int, int>? _tDs;
    private Action<KernelConfig, FV, FV, int, int, int, int>? _tTranspose;
    private Action<Index1D, FV, FV>? _tScaleSet;
    private Action<KernelConfig, ArrayView1D<Trellis2.F4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>,
        FV, FV, ArrayView1D<int, Stride1D.Dense>, int, int, int, int, int, int, int>? _tBf16;
    private Action<KernelConfig, ArrayView1D<Trellis2.F4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, FV, FV, int, int, int>? _tF16;

    private static MemoryBuffer1D<ushort, Stride1D.Dense> Grow16(Accelerator acc, ref MemoryBuffer1D<ushort, Stride1D.Dense>? b, long n,
        List<MemoryBuffer1D<ushort, Stride1D.Dense>> retired)
    {
        if (b == null || b.Length < n)
        {
            if (b != null) retired.Add(b);
            long cap = 1;
            while (cap < n) cap <<= 1;
            b = acc.Allocate1D<ushort>(cap);
        }
        return b;
    }

    private static FV At(FV v, long off) => v.SubView(off, v.Length - off);
    private static HV16 At(HV16 v, long off) => v.SubView(off, v.Length - off);

    private bool TensorCoreAttentionEligible(AttnDims d)
        => UseTensorCoreAttentionBackward && d.HeadDim % 32 == 0 && Kernels.CudaCaps.TuringTensorCores(_acc);

    // C[M, N] = A[M, K] * W[N, K]^T on tensor cores; A fp32 (rounded in the kernel), W 16-bit. turing: f16 with C = (A * sc[0]) W^T * sc[1].
    private void Mma(bool turing, FV A, HV16 W, FV C, FV sc, int M, int K, int N)
    {
        int tiles = ((M + Trellis2.MmaGemm.BM - 1) / Trellis2.MmaGemm.BM) * ((N + Trellis2.MmaGemm.BN - 1) / Trellis2.MmaGemm.BN);
        var a4 = A.SubView(0, (long)M * K).Cast<float, Trellis2.F4>();
        var w4 = W.SubView(0, (long)N * K).Cast<ushort, Trellis2.U4>();
        if (turing)
        {
            _tF16 ??= _acc.LoadStreamKernel<ArrayView1D<Trellis2.F4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, FV, FV, int, int, int>(
                Trellis2.MmaGemm.KernelF16Sm75Scaled);
            _tF16(new KernelConfig(tiles, Trellis2.MmaGemm.Threads), a4, w4, sc, C, M, K, N);
        }
        else
        {
            _tBf16 ??= _acc.LoadStreamKernel<ArrayView1D<Trellis2.F4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>,
                FV, FV, ArrayView1D<int, Stride1D.Dense>, int, int, int, int, int, int, int>(Trellis2.MmaGemm.KernelBf16Dense);
            _noBias ??= _acc.Allocate1D<BFloat16>(8);
            _noNbr ??= _acc.Allocate1D<int>(1);
            _tBf16(new KernelConfig(tiles, Trellis2.MmaGemm.Threads), a4, w4, _noBias.View, C, C, _noNbr.View, M, K, N, 0, 0, 0, 0);
        }
    }

    // a power-of-two scale for f16 from max|x| over count values (device-side): sc = { s, 1/s }
    private void DeviceScale(FV x, long count, MemoryBuffer1D<int, Stride1D.Dense> maxBits, FV sc)
    {
        _zeroInt ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>>(ZeroIntImpl);
        _absMaxK ??= _acc.LoadStreamKernel<FV, ArrayView1D<int, Stride1D.Dense>, int>(AbsMaxImpl);
        _scaleK ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, FV>(ScaleFromMaxImpl);
        _zeroInt(1, maxBits.View);
        _absMaxK(new KernelConfig((int)Math.Max(1, Math.Min(1024, (count + 255) / 256)), 256), x, maxBits.View, (int)count);
        _scaleK(1, maxBits.View, sc);
    }

    private void AttentionTensorCore(FV q, FV k, FV v, FV outp, FV dout, FV dq, FV dk, FV dv, FV stats, AttnDims d)
    {
        bool turing = !Kernels.CudaCaps.TensorCores(_acc) || Kernels.FusedDequantMatMul.ForceSm75Mma;
        int H = d.Heads, g = d.Heads / d.KvHeads, Sq = d.SeqQ, Skv = d.SeqKV, hd = d.HeadDim;
        int Sqp = (Sq + 31) / 32 * 32, Skvp = (Skv + 31) / 32 * 32;
        long perHead = 3L * Sqp * Skvp * sizeof(float);
        int hc = (int)Math.Max(1, Math.Min(H, GemmAttentionScoreBudgetBytes / Math.Max(1, perHead)));
        hc = Math.Max(g, hc / g * g);
        while (H % hc != 0) hc -= g;
        long qn = (long)hc * Sq * hd, kn = (long)hc * Skv * hd;
        long sn = (long)hc * Sq * Skvp, tn = (long)hc * Skvp * Sqp;
        long scoreCap = Math.Max(Math.Max(sn, tn), GemmAttentionScoreBudgetBytes / 3 / sizeof(float));
        var Qh = Grow(_acc, ref _gQh, qn, _gRetired); var dOh = Grow(_acc, ref _gdOh, qn, _gRetired); var dQh = Grow(_acc, ref _gdQh, qn, _gRetired);
        var dKh = Grow(_acc, ref _gdKh, kn, _gRetired); var dVh = Grow(_acc, ref _gdVh, kn, _gRetired);
        var S = Grow(_acc, ref _tS, scoreCap, _gRetired); var dP = Grow(_acc, ref _tdP, scoreCap, _gRetired); var T = Grow(_acc, ref _tT, scoreCap, _gRetired);
        var Kb = Grow16(_acc, ref _tKb, (long)hc * Skvp * hd, _tRetired16); var Vb = Grow16(_acc, ref _tVb, (long)hc * Skvp * hd, _tRetired16);
        var KTb = Grow16(_acc, ref _tKTb, (long)hc * hd * Skvp, _tRetired16);
        var QTb = Grow16(_acc, ref _tQTb, (long)hc * hd * Sqp, _tRetired16); var dOTb = Grow16(_acc, ref _tdOTb, (long)hc * hd * Sqp, _tRetired16);
        _tScOne ??= _acc.Allocate1D(new[] { 1f, 1f });
        _tScDo ??= _acc.Allocate1D<float>(2); _tScDs ??= _acc.Allocate1D<float>(2); _tScDv ??= _acc.Allocate1D<float>(2);
        _tMaxDo ??= _acc.Allocate1D<int>(1); _tMaxDs ??= _acc.Allocate1D<int>(1);
        _gPack ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, PackDims>(PackImpl);
        _pack16 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, HV16, Pack16Dims, FV>(Pack16Impl);
        _tStats ??= _acc.LoadStreamKernel<FV, FV, FV, FV, AttnDims, int, int>(TcStatsImpl);
        _tP ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, AttnDims, int, int>(TcPImpl);
        _tDs ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, AttnDims, int, int>(TcDsImpl);
        _tTranspose ??= _acc.LoadStreamKernel<FV, FV, int, int, int, int>(TransposePadImpl);
        _tScaleSet ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(InvOnlyScaleImpl);
        _gScatterQ ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, PackDims>(ScatterQImpl);
        _gScatterKv ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, PackDims>(ScatterKvImpl);
        int half = turing ? 1 : 0;
        for (int h0 = 0; h0 < H; h0 += hc)
        {
            var pq = new PackDims { S = Sq, Hs = H, Hd = hd, H0 = h0, G = 1, Hc = hc };
            var pk = new PackDims { S = Skv, Hs = d.KvHeads, Hd = hd, H0 = h0, G = g, Hc = hc };
            _gPack((int)qn, q, Qh.View, pq);
            _gPack((int)qn, dout, dOh.View, pq);
            // dO's scale (Turing): dP's A operand and dV's W operand
            if (turing) { DeviceScale(dOh.View, qn, _tMaxDo, _tScDo.View); _tScaleSet(1, _tScDo.View, _tScDv.View); }
            var p16k = new Pack16Dims { S = Skv, Sp = Skvp, Hs = d.KvHeads, Hd = hd, H0 = h0, G = g, Half = half };
            _pack16((int)((long)hc * Skvp * hd), k, Kb.View, p16k, _tScOne.View);
            _pack16((int)((long)hc * Skvp * hd), v, Vb.View, p16k, _tScOne.View);
            p16k.Transpose = 1;
            _pack16((int)((long)hc * Skvp * hd), k, KTb.View, p16k, _tScOne.View);
            var p16q = new Pack16Dims { S = Sq, Sp = Sqp, Hs = H, Hd = hd, H0 = h0, G = 1, Half = half, Transpose = 1 };
            _pack16((int)((long)hc * Sqp * hd), q, QTb.View, p16q, _tScOne.View);
            p16q.UseScale = turing ? 1 : 0;
            _pack16((int)((long)hc * Sqp * hd), dout, dOTb.View, p16q, _tScDo.View);
            for (int c = 0; c < hc; c++)                                               // S = Q K^T  [Sq, Skvp]
                Mma(turing, At(Qh.View, (long)c * Sq * hd), At(Kb.View, (long)c * Skvp * hd), At(S.View, (long)c * Sq * Skvp), _tScOne.View, Sq, hd, Skvp);
            _tStats(new KernelConfig(hc * Sq, StatsThreads), S.View, dout, outp, stats, d, h0, Skvp);   // lse and D per row
            _tP((int)sn, S.View, stats, d, h0, Skvp);                                   // S -> P (masked, padding 0)
            for (int c = 0; c < hc; c++)                                               // dP = dO V^T  [Sq, Skvp]
                Mma(turing, At(dOh.View, (long)c * Sq * hd), At(Vb.View, (long)c * Skvp * hd), At(dP.View, (long)c * Sq * Skvp), _tScDo.View, Sq, hd, Skvp);
            _tDs((int)sn, dP.View, S.View, stats, d, h0, Skvp);                         // dP -> dS = P (dP - D) * scale
            if (turing) DeviceScale(dP.View, sn, _tMaxDs, _tScDs.View);
            for (int c = 0; c < hc; c++)                                               // dQ = dS K = dS (K^T)^T  [Sq, hd]
                Mma(turing, At(dP.View, (long)c * Sq * Skvp), At(KTb.View, (long)c * hd * Skvp), At(dQh.View, (long)c * Sq * hd), _tScDs.View, Sq, Skvp, hd);
            int tilesR = Sqp / 32, tilesC = Skvp / 32;
            _tTranspose(new KernelConfig(hc * tilesR * tilesC, 256), dP.View, T.View, Sq, Skvp, Sqp, tilesC);   // T = dS^T [Skvp, Sqp]
            for (int c = 0; c < hc; c++)                                               // dK = dS^T Q = T (Q^T)^T  [Skv, hd]
                Mma(turing, At(T.View, (long)c * Skvp * Sqp), At(QTb.View, (long)c * hd * Sqp), At(dKh.View, (long)c * Skv * hd), _tScDs.View, Skv, Sqp, hd);
            _tTranspose(new KernelConfig(hc * tilesR * tilesC, 256), S.View, T.View, Sq, Skvp, Sqp, tilesC);    // T = P^T
            for (int c = 0; c < hc; c++)                                               // dV = P^T dO = T (dO^T)^T  [Skv, hd]
                Mma(turing, At(T.View, (long)c * Skvp * Sqp), At(dOTb.View, (long)c * hd * Sqp), At(dVh.View, (long)c * Skv * hd), _tScDv.View, Skv, Sqp, hd);
            _gScatterQ((int)qn, dQh.View, dq, pq);
            _gScatterKv((int)(kn / g), dKh.View, dk, pk);
            _gScatterKv((int)(kn / g), dVh.View, dv, pk);
        }
    }

    // dst for idx over the chunk: [c, i, e] or (Transpose) [c, e, i] with i < Sp; rows i >= S are zero. Source head (H0 + c) / G of [S, Hs, Hd].
    private static void Pack16Impl(Index1D idx, FV src, HV16 dst, Pack16Dims p, FV sc)
    {
        int per = p.Sp * p.Hd;
        int c = idx / per, rem = idx - c * per;
        int i, e;
        if (p.Transpose == 0) { i = rem / p.Hd; e = rem - i * p.Hd; }
        else { e = rem / p.Sp; i = rem - e * p.Sp; }
        float v = 0f;
        if (i < p.S)
        {
            v = src[((long)i * p.Hs + (p.H0 + c) / p.G) * p.Hd + e];
            if (p.UseScale != 0) v *= sc[0];
        }
        if (p.Half != 0) dst[idx] = F32ToF16Bits(v);
        else
        {
            uint b = (uint)Interop.FloatAsInt(v);
            b += 0x7FFFu + ((b >> 16) & 1u);
            dst[idx] = (ushort)(b >> 16);
        }
    }

    // out = { 1, sc[1] }: A unscaled, output multiplied by sc's inverse (dV = P^T (dO s)^T / s)
    private static void InvOnlyScaleImpl(Index1D i, FV sc, FV o) { o[0] = 1f; o[1] = sc[1]; }

    private const int StatsThreads = 128;

    // one group per (chunk head c, query row i): the row's max and sum of exp over the allowed keys (coalesced strided loops, shared-memory
    // trees in a fixed order: deterministic), and D = dO . O. MEASURED 2026-11-05: a thread per row walking its own 1,600-wide row twice was
    // most of FusedAttentionGrad's time once the GEMMs moved to tensor cores. S rows have stride Skvp; keys j >= SeqKV are padding.
    private static void TcStatsImpl(FV s, FV dout, FV outp, FV stats, AttnDims d, int h0, int stride)
    {
        var sh = SharedMemory.Allocate<float>(StatsThreads);
        int r = Grid.IdxX, tid = Group.IdxX;
        int c = r / d.SeqQ, i = r - c * d.SeqQ, h = h0 + c;
        long rowBase = (long)r * stride;
        float m = float.NegativeInfinity;
        for (int j = tid; j < d.SeqKV; j += StatsThreads) if (Allowed(i, j, d)) m = MathF.Max(m, s[rowBase + j] * d.Scale);
        sh[tid] = m;
        Group.Barrier();
        for (int w = StatsThreads / 2; w > 0; w >>= 1)
        {
            if (tid < w) sh[tid] = MathF.Max(sh[tid], sh[tid + w]);
            Group.Barrier();
        }
        float rowMax = sh[0];
        Group.Barrier();
        float sum = 0f;
        if (rowMax != float.NegativeInfinity)
            for (int j = tid; j < d.SeqKV; j += StatsThreads) if (Allowed(i, j, d)) sum += MathF.Exp(s[rowBase + j] * d.Scale - rowMax);
        sh[tid] = sum;
        Group.Barrier();
        for (int w = StatsThreads / 2; w > 0; w >>= 1)
        {
            if (tid < w) sh[tid] += sh[tid + w];
            Group.Barrier();
        }
        float rowSum = sh[0];
        Group.Barrier();
        long ob = ((long)i * d.Heads + h) * d.HeadDim;
        float dd = 0f;
        for (int e = tid; e < d.HeadDim; e += StatsThreads) dd += dout[ob + e] * outp[ob + e];
        sh[tid] = dd;
        Group.Barrier();
        for (int w = StatsThreads / 2; w > 0; w >>= 1)
        {
            if (tid < w) sh[tid] += sh[tid + w];
            Group.Barrier();
        }
        if (tid == 0)
        {
            stats[h * d.SeqQ + i] = rowMax == float.NegativeInfinity ? 0f : rowMax + MathF.Log(rowSum);
            stats[d.Heads * d.SeqQ + h * d.SeqQ + i] = sh[0];
        }
    }

    private static void TcPImpl(Index1D idx, FV s, FV stats, AttnDims d, int h0, int stride)
    {
        int r = idx / stride, j = idx - r * stride;
        int c = r / d.SeqQ, i = r - c * d.SeqQ;
        s[idx] = j < d.SeqKV && Allowed(i, j, d) ? MathF.Exp(s[idx] * d.Scale - stats[(h0 + c) * d.SeqQ + i]) : 0f;
    }

    private static void TcDsImpl(Index1D idx, FV dp, FV p, FV stats, AttnDims d, int h0, int stride)
    {
        int r = idx / stride;
        int c = r / d.SeqQ, i = r - c * d.SeqQ;
        dp[idx] = p[idx] * (dp[idx] - stats[d.Heads * d.SeqQ + (h0 + c) * d.SeqQ + i]) * d.Scale;
    }

    // [batch, R, C] -> [batch, C, Rp] by 32 x 32 tiles; columns r in [R, Rp) of the output are written as zero. grid = batch * (Rp/32) * tilesC
    private static void TransposePadImpl(FV src, FV dst, int R, int C, int Rp, int tilesC)
    {
        var sh = SharedMemory.Allocate<float>(32 * 33);
        int tilesR = Rp / 32, perBatch = tilesR * tilesC;
        int b = Grid.IdxX / perBatch, t = Grid.IdxX - b * perBatch;
        int r0 = (t / tilesC) * 32, c0 = (t % tilesC) * 32;
        int tx = Group.IdxX & 31, ty = Group.IdxX >> 5;
        long sb = (long)b * R * C, db = (long)b * C * Rp;
        for (int k = 0; k < 4; k++)
        {
            int rr = ty + 8 * k, r = r0 + rr, cc = c0 + tx;
            sh[rr * 33 + tx] = r < R && cc < C ? src[sb + (long)r * C + cc] : 0f;
        }
        Group.Barrier();
        for (int k = 0; k < 4; k++)
        {
            int cc = ty + 8 * k, col = c0 + cc, r = r0 + tx;
            if (col < C && r < Rp) dst[db + (long)col * Rp + r] = sh[tx * 33 + cc];
        }
    }

    private void DisposeAttentionTensorCore()
    {
        foreach (var b in new[] { _tKb, _tVb, _tKTb, _tQTb, _tdOTb }) b?.Dispose();
        _tKb = _tVb = _tKTb = _tQTb = _tdOTb = null;
        foreach (var b in _tRetired16) b.Dispose();
        _tRetired16.Clear();
        foreach (var b in new[] { _tS, _tdP, _tT, _tScOne, _tScDo, _tScDs, _tScDv }) b?.Dispose();
        _tS = _tdP = _tT = _tScOne = _tScDo = _tScDs = _tScDv = null;
        _tMaxDo?.Dispose(); _tMaxDs?.Dispose(); _tMaxDo = _tMaxDs = null;
    }
}
