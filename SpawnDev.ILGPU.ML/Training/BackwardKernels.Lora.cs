using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Training;

using FV = ArrayView1D<float, Stride1D.Dense>;
using F4V = ArrayView1D<Trellis2.F4, Stride1D.Dense>;

/// <summary>
/// The fused LoRA projection (2026-11-05): <c>y = yBase + scale (x A_T) B_T</c> and its gradients, for the LoraLinear / LoraLinearGrad ops that
/// <see cref="GraphAutograd.AddLora"/> emits. MEASURED (llama3.1:8b, 1,600 tokens, rank 16 on q/k/v/o, RTX 4070, Nsight Systems): the four-op
/// form (MatMul, MatMul, Mul, Add) and its autograd (Mul, Gemm transA with a materialised 1600 x 4096 transpose, MatMulGradInput through a
/// transposed f32 B, Add) took ~290 ms of a 2.29 s step: the rank-16 products ran on a 16 x 16 tiled kernel with 100 groups (K = 4096 long,
/// ~1 TFLOPS), and three full-size elementwise passes. Here every product is a split-K dot with an ordered reduction (deterministic), and the
/// elementwise passes are folded into the products. fp32 throughout (the same precision as the four-op form).
/// </summary>
public sealed partial class BackwardKernels
{
    private const int LrRows = 32, LrSlice = 256, LcCols = 256, LcSlice = 64;

    private MemoryBuffer1D<float, Stride1D.Dense>? _lrPart, _lrDxa, _lrNone, _lrAR;
    private Action<Index1D, FV, FV, int, int>? _lrTrans;
    private Action<KernelConfig, F4V, F4V, FV, int, int, int, int>? _lrRowDot;
    private Action<KernelConfig, FV, FV, FV, int, int, int, int>? _lrColDot;
    private Action<Index1D, F4V, F4V, F4V, F4V, int, int, int, int, float>? _lrCombine;
    private Action<Index1D, FV, FV, int, int, float, int, int, int>? _lrReduce;
    private Action<Index1D, FV, FV, FV, int, int, int, float>? _lrRowDotNaive;
    private Action<Index1D, FV, FV, FV, int, int, int, int, float>? _lrColDotNaive;
    private Action<Index1D, FV, FV, FV, FV, int, int, int, float>? _lrCombineNaive;

    /// <summary>Whether the fused LoRA kernels support this rank (a multiple of 8: the dot kernels accumulate 8 rank columns at a time).</summary>
    public static bool LoraRankSupported(int rank) => rank > 0 && rank % 8 == 0;

    private static F4V Vec4(FV v, long n) => v.SubView(0, n).Cast<float, Trellis2.F4>();

    private bool LoraShared => _acc.AcceleratorType != AcceleratorType.WebGL && _acc.MaxNumThreadsPerGroup >= 256;

    private void LoraLoad()
    {
        if (LoraShared)
        {
            _lrRowDot ??= _acc.LoadStreamKernel<F4V, F4V, FV, int, int, int, int>(LoraRowDotImpl);
            _lrColDot ??= _acc.LoadStreamKernel<FV, FV, FV, int, int, int, int>(LoraColDotImpl);
            _lrReduce ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, float, int, int, int>(LoraReduceImpl);
        }
        else
        {
            _lrRowDotNaive ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, float>(LoraRowDotNaiveImpl);
            _lrColDotNaive ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, int, float>(LoraColDotNaiveImpl);
            _lrCombineNaive ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, int, int, int, float>(LoraCombineNaiveImpl);
        }
        if (LoraShared) _lrCombine ??= _acc.LoadAutoGroupedStreamKernel<Index1D, F4V, F4V, F4V, F4V, int, int, int, int, float>(LoraCombineImpl);
        _lrTrans ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int>(LoraTransposeImpl);
    }

    // A_T [F, R] -> A [R, F]: the dot and combine kernels read Q rank-major, so a warp's lanes (consecutive k) read consecutive addresses
    // (MEASURED 2026-11-05: reading A_T [F, R] directly, the x A_T dot ran at 220 us and the dx combine near 3x its memory time)
    private FV LoraRankMajor(FV aT, int F, int R)
    {
        var aR = Grow(_acc, ref _lrAR, (long)F * R, _gRetired);
        _lrTrans!(F * R, aT, aR.View, F, R);
        return aR.View;
    }

    private static void LoraTransposeImpl(Index1D i, FV aT, FV aR, int F, int R)
    {
        int r = i / F, f = i - r * F;
        aR[i] = aT[(long)f * R + r];
    }

    // out[m, r] = scale * sum_k P[m, k] Q[r, k] over K (Q rank-major [R, K])
    private void LoraRowDot(FV P, FV Q, FV outp, int M, int K, int R, float scale)
    {
        if (!LoraShared || K % 4 != 0) { _lrRowDotNaive!(M * R, P, Q, outp, K, R, 1, scale); return; }
        int slices = (K + LrSlice - 1) / LrSlice;
        var part = Grow(_acc, ref _lrPart, (long)slices * M * R, _gRetired);
        _lrRowDot!(new KernelConfig(((M + LrRows - 1) / LrRows) * slices, 256), Vec4(P, (long)M * K), Vec4(Q, (long)R * K), part.View, M, K, R, slices);
        _lrReduce!(M * R, part.View, outp, M * R, slices, scale, 0, R, 1);
    }

    // out[m, n] = (hasBase ? base[m, n] : 0) + scale * sum_r P[m, r] Q[r, n]
    private void LoraCombine(FV P, FV Q, FV bas, FV outp, int M, int N, int R, bool hasBase, float scale)
    {
        _lrNone ??= _acc.Allocate1D<float>(4);   // the unused base (WebGPU forbids one buffer in two bindings)
        if (!LoraShared || N % 4 != 0)
        {
            // one output per thread (WebGL captures one store per thread)
            _lrCombineNaive!(M * N, P, Q, hasBase ? bas : _lrNone.View, outp, N, R, hasBase ? 1 : 0, scale);
            return;
        }
        _lrCombine!((M + 3) / 4 * (N / 4), Vec4(P, (long)M * R), Vec4(Q, (long)R * N), hasBase ? Vec4(bas, (long)M * N) : Vec4(_lrNone.View, 4), Vec4(outp, (long)M * N),
            M, N / 4, R, hasBase ? 1 : 0, scale);
    }

    // sum_m P[m, r] Q[m, n] over M, times scale: written [R, N] (transpose 0) or [N, R] (transpose 1)
    private void LoraColDot(FV P, FV Q, FV outp, int M, int N, int R, int transpose, float scale)
    {
        if (!LoraShared) { _lrColDotNaive!(R * N, P, Q, outp, M, N, R, transpose, scale); return; }
        int slices = (M + LcSlice - 1) / LcSlice;
        var part = Grow(_acc, ref _lrPart, (long)slices * R * N, _gRetired);
        _lrColDot!(new KernelConfig(((N + LcCols - 1) / LcCols) * slices, LcCols), P, Q, part.View, M, N, R, slices);
        _lrReduce!(R * N, part.View, outp, R * N, slices, scale, transpose, R, N);
    }

    /// <summary>
    /// LoraLinear forward: <c>xa = x A_T</c> [M, R] and <c>y = yBase + scale xa B_T</c> [M, N]. x [M, F], A_T [F, R], B_T [R, N].
    /// </summary>
    public void LoraForward(FV x, FV aT, FV bT, FV yBase, FV y, FV xa, int M, int F, int N, int R, float scale)
    {
        LoraLoad();
        LoraRowDot(x, LoraRankMajor(aT, F, R), xa, M, F, R, 1f);
        LoraCombine(xa, bT, yBase, y, M, N, R, true, scale);
    }

    /// <summary>
    /// LoraLinearGrad: from dy [M, N] and the forward's x [M, F] and xa [M, R]: dB_T = scale xa^T dy [R, N], dxa = scale dy B_T^T [M, R],
    /// dA_T = x^T dxa [F, R], and (when <paramref name="needDx"/>) dx = dxa A_T^T [M, F].
    /// </summary>
    public void LoraBackward(FV x, FV aT, FV bT, FV dy, FV xa, FV dx, FV daT, FV dbT, int M, int F, int N, int R, float scale, bool needDx, FV accumulate = default)
    {
        LoraLoad();
        var dxa = Grow(_acc, ref _lrDxa, (long)M * R, _gRetired);
        LoraColDot(xa, dy, dbT, M, N, R, 0, scale);
        LoraRowDot(dy, bT, dxa.View, M, N, R, scale);
        LoraColDot(dxa.View, x, daT, M, F, R, 1, 1f);
        // dx = accumulate + dxa A (the running gradient sum folded in: the same value as the graph's Add)
        if (needDx) LoraCombine(dxa.View, LoraRankMajor(aT, F, R), accumulate.IsValid ? accumulate : dxa.View, dx, M, F, R, accumulate.IsValid, 1f);
    }

    // One group = 32 rows x one K slice of 256; thread (row = tid / 8, kl = tid % 8) takes 4 consecutive k per step (16-byte loads; a row's 8
    // lanes read 128 contiguous bytes), k4 = kl, kl + 8, ... of the slice, for 8 rank columns at a time (the row's P values are re-read per
    // chunk of 8 from L1 / L2); then the 8 k-lanes are summed in order through shared memory. part[slice, m, r]. Rows past M write nothing.
    private static void LoraRowDotImpl(F4V P, F4V Q, FV part, int M, int K, int R, int slices)
    {
        var sh = SharedMemory.Allocate<float>(LrRows * 64);   // [row][r of the chunk][k-lane]
        int gi = Grid.IdxX, rb = gi / slices, sl = gi - rb * slices;
        int tid = Group.IdxX, row = tid >> 3, kl = tid & 7;
        int m = rb * LrRows + row;
        int K4 = K >> 2;
        int k0 = sl * (LrSlice / 4), k1 = Math.Min(K4, k0 + LrSlice / 4);
        long pb = (long)m * K4;
        for (int rc = 0; rc < R; rc += 8)
        {
            float a0 = 0f, a1 = 0f, a2 = 0f, a3 = 0f, a4 = 0f, a5 = 0f, a6 = 0f, a7 = 0f;
            if (m < M)
            {
                long q0 = (long)rc * K4;
                for (int k = k0 + kl; k < k1; k += 8)
                {
                    var p = P[pb + k];
                    long qb = q0 + k;
                    var q = Q[qb]; a0 += p.X * q.X; a0 += p.Y * q.Y; a0 += p.Z * q.Z; a0 += p.W * q.W;
                    q = Q[qb + K4]; a1 += p.X * q.X; a1 += p.Y * q.Y; a1 += p.Z * q.Z; a1 += p.W * q.W;
                    q = Q[qb + 2 * (long)K4]; a2 += p.X * q.X; a2 += p.Y * q.Y; a2 += p.Z * q.Z; a2 += p.W * q.W;
                    q = Q[qb + 3 * (long)K4]; a3 += p.X * q.X; a3 += p.Y * q.Y; a3 += p.Z * q.Z; a3 += p.W * q.W;
                    q = Q[qb + 4 * (long)K4]; a4 += p.X * q.X; a4 += p.Y * q.Y; a4 += p.Z * q.Z; a4 += p.W * q.W;
                    q = Q[qb + 5 * (long)K4]; a5 += p.X * q.X; a5 += p.Y * q.Y; a5 += p.Z * q.Z; a5 += p.W * q.W;
                    q = Q[qb + 6 * (long)K4]; a6 += p.X * q.X; a6 += p.Y * q.Y; a6 += p.Z * q.Z; a6 += p.W * q.W;
                    q = Q[qb + 7 * (long)K4]; a7 += p.X * q.X; a7 += p.Y * q.Y; a7 += p.Z * q.Z; a7 += p.W * q.W;
                }
            }
            int b = row * 64 + kl;
            sh[b] = a0; sh[b + 8] = a1; sh[b + 16] = a2; sh[b + 24] = a3; sh[b + 32] = a4; sh[b + 40] = a5; sh[b + 48] = a6; sh[b + 56] = a7;
            Group.Barrier();
            int rb8 = row * 64 + kl * 8;   // thread (row, kl) now owns rank column rc + kl
            float s = sh[rb8] + sh[rb8 + 1] + sh[rb8 + 2] + sh[rb8 + 3] + sh[rb8 + 4] + sh[rb8 + 5] + sh[rb8 + 6] + sh[rb8 + 7];
            if (m < M) part[((long)sl * M + m) * R + rc + kl] = s;
            Group.Barrier();
        }
    }

    // One group = 256 columns n x one M slice of 64 rows; thread n sums its column over the slice for 8 rank columns at a time, with the slice's
    // P[m, rc..rc+7] staged in shared memory (a broadcast read). part[slice, r, n].
    private static void LoraColDotImpl(FV P, FV Q, FV part, int M, int N, int R, int slices)
    {
        var ps = SharedMemory.Allocate<float>(LcSlice * 8);
        int gi = Grid.IdxX, cb = gi / slices, sl = gi - cb * slices;
        int tid = Group.IdxX, n = cb * LcCols + tid;
        int m0 = sl * LcSlice, rows = Math.Min(M, m0 + LcSlice) - m0;
        for (int rc = 0; rc < R; rc += 8)
        {
            for (int e = tid; e < LcSlice * 8; e += LcCols)
            {
                int mm = e >> 3, j = e & 7;
                ps[e] = mm < rows ? P[(long)(m0 + mm) * R + rc + j] : 0f;
            }
            Group.Barrier();
            float a0 = 0f, a1 = 0f, a2 = 0f, a3 = 0f, a4 = 0f, a5 = 0f, a6 = 0f, a7 = 0f;
            if (n < N)
            {
                int mm = 0;
                long qr = (long)m0 * N + n;
                for (; mm + 4 <= rows; mm += 4, qr += 4 * (long)N)
                {
                    // four rows' loads in flight; the sums stay in row order
                    float qa = Q[qr], qb = Q[qr + N], qc = Q[qr + 2 * (long)N], qd = Q[qr + 3 * (long)N];
                    int pb = mm * 8;
                    a0 += ps[pb] * qa; a1 += ps[pb + 1] * qa; a2 += ps[pb + 2] * qa; a3 += ps[pb + 3] * qa;
                    a4 += ps[pb + 4] * qa; a5 += ps[pb + 5] * qa; a6 += ps[pb + 6] * qa; a7 += ps[pb + 7] * qa;
                    a0 += ps[pb + 8] * qb; a1 += ps[pb + 9] * qb; a2 += ps[pb + 10] * qb; a3 += ps[pb + 11] * qb;
                    a4 += ps[pb + 12] * qb; a5 += ps[pb + 13] * qb; a6 += ps[pb + 14] * qb; a7 += ps[pb + 15] * qb;
                    a0 += ps[pb + 16] * qc; a1 += ps[pb + 17] * qc; a2 += ps[pb + 18] * qc; a3 += ps[pb + 19] * qc;
                    a4 += ps[pb + 20] * qc; a5 += ps[pb + 21] * qc; a6 += ps[pb + 22] * qc; a7 += ps[pb + 23] * qc;
                    a0 += ps[pb + 24] * qd; a1 += ps[pb + 25] * qd; a2 += ps[pb + 26] * qd; a3 += ps[pb + 27] * qd;
                    a4 += ps[pb + 28] * qd; a5 += ps[pb + 29] * qd; a6 += ps[pb + 30] * qd; a7 += ps[pb + 31] * qd;
                }
                for (; mm < rows; mm++, qr += N)
                {
                    float q = Q[qr];
                    int pb = mm * 8;
                    a0 += ps[pb] * q; a1 += ps[pb + 1] * q; a2 += ps[pb + 2] * q; a3 += ps[pb + 3] * q;
                    a4 += ps[pb + 4] * q; a5 += ps[pb + 5] * q; a6 += ps[pb + 6] * q; a7 += ps[pb + 7] * q;
                }
                long o = ((long)sl * R + rc) * N + n;
                part[o] = a0; part[o + N] = a1; part[o + 2 * (long)N] = a2; part[o + 3 * (long)N] = a3;
                part[o + 4 * (long)N] = a4; part[o + 5 * (long)N] = a5; part[o + 6 * (long)N] = a6; part[o + 7 * (long)N] = a7;
            }
            Group.Barrier();
        }
    }

    // out = scale * sum over slices in order; i indexes [rows, cols] = [R, N] for the column dots (transpose 1 writes out[n, r]) and [M, R] for
    // the row dots (cols = 1 marks "no transpose" there: transpose is 0)
    private static void LoraReduceImpl(Index1D i, FV part, FV outp, int count, int slices, float scale, int transpose, int R, int N)
    {
        float s = 0f;
        for (int k = 0; k < slices; k++) s += part[(long)k * count + i];
        long o = i;
        if (transpose != 0) { int r = i / N, n = i - r * N; o = (long)n * R + r; }
        outp[o] = s * scale;
    }

    // 4 rows x 4 consecutive columns per thread (16-byte loads and stores), eight ranks per step (R % 8 == 0): each Q load serves four rows.
    // MEASURED 2026-11-05 (RTX 4070, 1600 x 4096): one row per thread re-read B from L2 for every output row (~420 MB of L2 traffic) and ran at
    // ~200 us. out[m, 4 n4 + c] = (hasBase ? base : 0) + scale * sum_r P[m, r] Q[r, 4 n4 + c] (the sum in rank order).
    private static void LoraCombineImpl(Index1D idx, F4V P, F4V Q, F4V bas, F4V outp, int M, int N4, int R, int hasBase, float scale)
    {
        int mg = idx / N4, n4 = idx - mg * N4;
        int m0 = mg * 4;
        int R4 = R >> 2;
        long pb0 = (long)Math.Min(m0 + 0, M - 1) * R4;
        long pb1 = (long)Math.Min(m0 + 1, M - 1) * R4;
        long pb2 = (long)Math.Min(m0 + 2, M - 1) * R4;
        long pb3 = (long)Math.Min(m0 + 3, M - 1) * R4;
        float s00 = 0f, s01 = 0f, s02 = 0f, s03 = 0f;
        float s10 = 0f, s11 = 0f, s12 = 0f, s13 = 0f;
        float s20 = 0f, s21 = 0f, s22 = 0f, s23 = 0f;
        float s30 = 0f, s31 = 0f, s32 = 0f, s33 = 0f;
        for (int r = 0; r < R; r += 8)
        {
            long qb = (long)r * N4 + n4;
            var q0 = Q[qb]; var q1 = Q[qb + N4]; var q2 = Q[qb + 2 * (long)N4]; var q3 = Q[qb + 3 * (long)N4];
            var q4 = Q[qb + 4 * (long)N4]; var q5 = Q[qb + 5 * (long)N4]; var q6 = Q[qb + 6 * (long)N4]; var q7 = Q[qb + 7 * (long)N4];
            var pa0 = P[pb0 + (r >> 2)]; var pc0 = P[pb0 + (r >> 2) + 1];
            var pa1 = P[pb1 + (r >> 2)]; var pc1 = P[pb1 + (r >> 2) + 1];
            var pa2 = P[pb2 + (r >> 2)]; var pc2 = P[pb2 + (r >> 2) + 1];
            var pa3 = P[pb3 + (r >> 2)]; var pc3 = P[pb3 + (r >> 2) + 1];
            s00 += pa0.X * q0.X; s01 += pa0.X * q0.Y; s02 += pa0.X * q0.Z; s03 += pa0.X * q0.W;
            s00 += pa0.Y * q1.X; s01 += pa0.Y * q1.Y; s02 += pa0.Y * q1.Z; s03 += pa0.Y * q1.W;
            s00 += pa0.Z * q2.X; s01 += pa0.Z * q2.Y; s02 += pa0.Z * q2.Z; s03 += pa0.Z * q2.W;
            s00 += pa0.W * q3.X; s01 += pa0.W * q3.Y; s02 += pa0.W * q3.Z; s03 += pa0.W * q3.W;
            s00 += pc0.X * q4.X; s01 += pc0.X * q4.Y; s02 += pc0.X * q4.Z; s03 += pc0.X * q4.W;
            s00 += pc0.Y * q5.X; s01 += pc0.Y * q5.Y; s02 += pc0.Y * q5.Z; s03 += pc0.Y * q5.W;
            s00 += pc0.Z * q6.X; s01 += pc0.Z * q6.Y; s02 += pc0.Z * q6.Z; s03 += pc0.Z * q6.W;
            s00 += pc0.W * q7.X; s01 += pc0.W * q7.Y; s02 += pc0.W * q7.Z; s03 += pc0.W * q7.W;
            s10 += pa1.X * q0.X; s11 += pa1.X * q0.Y; s12 += pa1.X * q0.Z; s13 += pa1.X * q0.W;
            s10 += pa1.Y * q1.X; s11 += pa1.Y * q1.Y; s12 += pa1.Y * q1.Z; s13 += pa1.Y * q1.W;
            s10 += pa1.Z * q2.X; s11 += pa1.Z * q2.Y; s12 += pa1.Z * q2.Z; s13 += pa1.Z * q2.W;
            s10 += pa1.W * q3.X; s11 += pa1.W * q3.Y; s12 += pa1.W * q3.Z; s13 += pa1.W * q3.W;
            s10 += pc1.X * q4.X; s11 += pc1.X * q4.Y; s12 += pc1.X * q4.Z; s13 += pc1.X * q4.W;
            s10 += pc1.Y * q5.X; s11 += pc1.Y * q5.Y; s12 += pc1.Y * q5.Z; s13 += pc1.Y * q5.W;
            s10 += pc1.Z * q6.X; s11 += pc1.Z * q6.Y; s12 += pc1.Z * q6.Z; s13 += pc1.Z * q6.W;
            s10 += pc1.W * q7.X; s11 += pc1.W * q7.Y; s12 += pc1.W * q7.Z; s13 += pc1.W * q7.W;
            s20 += pa2.X * q0.X; s21 += pa2.X * q0.Y; s22 += pa2.X * q0.Z; s23 += pa2.X * q0.W;
            s20 += pa2.Y * q1.X; s21 += pa2.Y * q1.Y; s22 += pa2.Y * q1.Z; s23 += pa2.Y * q1.W;
            s20 += pa2.Z * q2.X; s21 += pa2.Z * q2.Y; s22 += pa2.Z * q2.Z; s23 += pa2.Z * q2.W;
            s20 += pa2.W * q3.X; s21 += pa2.W * q3.Y; s22 += pa2.W * q3.Z; s23 += pa2.W * q3.W;
            s20 += pc2.X * q4.X; s21 += pc2.X * q4.Y; s22 += pc2.X * q4.Z; s23 += pc2.X * q4.W;
            s20 += pc2.Y * q5.X; s21 += pc2.Y * q5.Y; s22 += pc2.Y * q5.Z; s23 += pc2.Y * q5.W;
            s20 += pc2.Z * q6.X; s21 += pc2.Z * q6.Y; s22 += pc2.Z * q6.Z; s23 += pc2.Z * q6.W;
            s20 += pc2.W * q7.X; s21 += pc2.W * q7.Y; s22 += pc2.W * q7.Z; s23 += pc2.W * q7.W;
            s30 += pa3.X * q0.X; s31 += pa3.X * q0.Y; s32 += pa3.X * q0.Z; s33 += pa3.X * q0.W;
            s30 += pa3.Y * q1.X; s31 += pa3.Y * q1.Y; s32 += pa3.Y * q1.Z; s33 += pa3.Y * q1.W;
            s30 += pa3.Z * q2.X; s31 += pa3.Z * q2.Y; s32 += pa3.Z * q2.Z; s33 += pa3.Z * q2.W;
            s30 += pa3.W * q3.X; s31 += pa3.W * q3.Y; s32 += pa3.W * q3.Z; s33 += pa3.W * q3.W;
            s30 += pc3.X * q4.X; s31 += pc3.X * q4.Y; s32 += pc3.X * q4.Z; s33 += pc3.X * q4.W;
            s30 += pc3.Y * q5.X; s31 += pc3.Y * q5.Y; s32 += pc3.Y * q5.Z; s33 += pc3.Y * q5.W;
            s30 += pc3.Z * q6.X; s31 += pc3.Z * q6.Y; s32 += pc3.Z * q6.Z; s33 += pc3.Z * q6.W;
            s30 += pc3.W * q7.X; s31 += pc3.W * q7.Y; s32 += pc3.W * q7.Z; s33 += pc3.W * q7.W;
        }
        if (m0 + 0 < M)
        {
            long o = (long)(m0 + 0) * N4 + n4;
            Trellis2.F4 v;
            if (hasBase != 0)
            {
                var bv = bas[o];
                v.X = bv.X + s00 * scale; v.Y = bv.Y + s01 * scale; v.Z = bv.Z + s02 * scale; v.W = bv.W + s03 * scale;
            }
            else { v.X = s00 * scale; v.Y = s01 * scale; v.Z = s02 * scale; v.W = s03 * scale; }
            outp[o] = v;
        }
        if (m0 + 1 < M)
        {
            long o = (long)(m0 + 1) * N4 + n4;
            Trellis2.F4 v;
            if (hasBase != 0)
            {
                var bv = bas[o];
                v.X = bv.X + s10 * scale; v.Y = bv.Y + s11 * scale; v.Z = bv.Z + s12 * scale; v.W = bv.W + s13 * scale;
            }
            else { v.X = s10 * scale; v.Y = s11 * scale; v.Z = s12 * scale; v.W = s13 * scale; }
            outp[o] = v;
        }
        if (m0 + 2 < M)
        {
            long o = (long)(m0 + 2) * N4 + n4;
            Trellis2.F4 v;
            if (hasBase != 0)
            {
                var bv = bas[o];
                v.X = bv.X + s20 * scale; v.Y = bv.Y + s21 * scale; v.Z = bv.Z + s22 * scale; v.W = bv.W + s23 * scale;
            }
            else { v.X = s20 * scale; v.Y = s21 * scale; v.Z = s22 * scale; v.W = s23 * scale; }
            outp[o] = v;
        }
        if (m0 + 3 < M)
        {
            long o = (long)(m0 + 3) * N4 + n4;
            Trellis2.F4 v;
            if (hasBase != 0)
            {
                var bv = bas[o];
                v.X = bv.X + s30 * scale; v.Y = bv.Y + s31 * scale; v.Z = bv.Z + s32 * scale; v.W = bv.W + s33 * scale;
            }
            else { v.X = s30 * scale; v.Y = s31 * scale; v.Z = s32 * scale; v.W = s33 * scale; }
            outp[o] = v;
        }
    }

    // the per-element forms (no shared memory: WebGL)
    private static void LoraRowDotNaiveImpl(Index1D i, FV P, FV Q, FV outp, int K, int R, int qLayout, float scale)
    {
        int m = i / R, r = i - m * R;
        long pb = (long)m * K;
        float s = 0f;
        for (int k = 0; k < K; k++) s += P[pb + k] * (qLayout == 0 ? Q[(long)k * R + r] : Q[(long)r * K + k]);
        outp[i] = s * scale;
    }

    private static void LoraCombineNaiveImpl(Index1D i, FV P, FV Q, FV bas, FV outp, int N, int R, int hasBase, float scale)
    {
        int m = i / N, n = i - m * N;
        long pb = (long)m * R;
        float s = 0f;
        for (int r = 0; r < R; r++) s += P[pb + r] * Q[(long)r * N + n];
        float v = s * scale;
        outp[i] = hasBase != 0 ? bas[i] + v : v;
    }

    private static void LoraColDotNaiveImpl(Index1D i, FV P, FV Q, FV outp, int M, int N, int R, int transpose, float scale)
    {
        int r, n;
        if (transpose != 0) { n = i / R; r = i - n * R; } else { r = i / N; n = i - r * N; }
        float s = 0f;
        for (int m = 0; m < M; m++) s += P[(long)m * R + r] * Q[(long)m * N + n];
        outp[i] = s * scale;
    }

    private void DisposeLora()
    {
        _lrPart?.Dispose(); _lrDxa?.Dispose(); _lrNone?.Dispose(); _lrAR?.Dispose();
        _lrPart = _lrDxa = _lrNone = _lrAR = null;
    }
}
