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
    private Action<KernelConfig, F4V, F4V, FV, int, int, int, int>? _lrRowDot, _lrRowDot16;
    private Action<KernelConfig, FV, FV, FV, int, int, int, int>? _lrColDot, _lrColDot16;
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
            _lrRowDot16 ??= _acc.LoadStreamKernel<F4V, F4V, FV, int, int, int, int>(LoraRowDot16Impl);
            _lrColDot16 ??= _acc.LoadStreamKernel<FV, FV, FV, int, int, int, int>(LoraColDot16Impl);
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
        (R % 16 == 0 ? _lrRowDot16! : _lrRowDot!)(new KernelConfig(((M + LrRows - 1) / LrRows) * slices, 256), Vec4(P, (long)M * K), Vec4(Q, (long)R * K), part.View, M, K, R, slices);
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
        (R % 16 == 0 ? _lrColDot16! : _lrColDot!)(new KernelConfig(((N + LcCols - 1) / LcCols) * slices, LcCols), P, Q, part.View, M, N, R, slices);
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
                    var q = Q[qb]; a0 = float.MultiplyAddEstimate(p.X, q.X, a0); a0 = float.MultiplyAddEstimate(p.Y, q.Y, a0); a0 = float.MultiplyAddEstimate(p.Z, q.Z, a0); a0 = float.MultiplyAddEstimate(p.W, q.W, a0);
                    q = Q[qb + K4]; a1 = float.MultiplyAddEstimate(p.X, q.X, a1); a1 = float.MultiplyAddEstimate(p.Y, q.Y, a1); a1 = float.MultiplyAddEstimate(p.Z, q.Z, a1); a1 = float.MultiplyAddEstimate(p.W, q.W, a1);
                    q = Q[qb + 2 * (long)K4]; a2 = float.MultiplyAddEstimate(p.X, q.X, a2); a2 = float.MultiplyAddEstimate(p.Y, q.Y, a2); a2 = float.MultiplyAddEstimate(p.Z, q.Z, a2); a2 = float.MultiplyAddEstimate(p.W, q.W, a2);
                    q = Q[qb + 3 * (long)K4]; a3 = float.MultiplyAddEstimate(p.X, q.X, a3); a3 = float.MultiplyAddEstimate(p.Y, q.Y, a3); a3 = float.MultiplyAddEstimate(p.Z, q.Z, a3); a3 = float.MultiplyAddEstimate(p.W, q.W, a3);
                    q = Q[qb + 4 * (long)K4]; a4 = float.MultiplyAddEstimate(p.X, q.X, a4); a4 = float.MultiplyAddEstimate(p.Y, q.Y, a4); a4 = float.MultiplyAddEstimate(p.Z, q.Z, a4); a4 = float.MultiplyAddEstimate(p.W, q.W, a4);
                    q = Q[qb + 5 * (long)K4]; a5 = float.MultiplyAddEstimate(p.X, q.X, a5); a5 = float.MultiplyAddEstimate(p.Y, q.Y, a5); a5 = float.MultiplyAddEstimate(p.Z, q.Z, a5); a5 = float.MultiplyAddEstimate(p.W, q.W, a5);
                    q = Q[qb + 6 * (long)K4]; a6 = float.MultiplyAddEstimate(p.X, q.X, a6); a6 = float.MultiplyAddEstimate(p.Y, q.Y, a6); a6 = float.MultiplyAddEstimate(p.Z, q.Z, a6); a6 = float.MultiplyAddEstimate(p.W, q.W, a6);
                    q = Q[qb + 7 * (long)K4]; a7 = float.MultiplyAddEstimate(p.X, q.X, a7); a7 = float.MultiplyAddEstimate(p.Y, q.Y, a7); a7 = float.MultiplyAddEstimate(p.Z, q.Z, a7); a7 = float.MultiplyAddEstimate(p.W, q.W, a7);
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
                    a0 = float.MultiplyAddEstimate(ps[pb], qa, a0); a1 = float.MultiplyAddEstimate(ps[pb + 1], qa, a1); a2 = float.MultiplyAddEstimate(ps[pb + 2], qa, a2); a3 = float.MultiplyAddEstimate(ps[pb + 3], qa, a3);
                    a4 = float.MultiplyAddEstimate(ps[pb + 4], qa, a4); a5 = float.MultiplyAddEstimate(ps[pb + 5], qa, a5); a6 = float.MultiplyAddEstimate(ps[pb + 6], qa, a6); a7 = float.MultiplyAddEstimate(ps[pb + 7], qa, a7);
                    a0 = float.MultiplyAddEstimate(ps[pb + 8], qb, a0); a1 = float.MultiplyAddEstimate(ps[pb + 9], qb, a1); a2 = float.MultiplyAddEstimate(ps[pb + 10], qb, a2); a3 = float.MultiplyAddEstimate(ps[pb + 11], qb, a3);
                    a4 = float.MultiplyAddEstimate(ps[pb + 12], qb, a4); a5 = float.MultiplyAddEstimate(ps[pb + 13], qb, a5); a6 = float.MultiplyAddEstimate(ps[pb + 14], qb, a6); a7 = float.MultiplyAddEstimate(ps[pb + 15], qb, a7);
                    a0 = float.MultiplyAddEstimate(ps[pb + 16], qc, a0); a1 = float.MultiplyAddEstimate(ps[pb + 17], qc, a1); a2 = float.MultiplyAddEstimate(ps[pb + 18], qc, a2); a3 = float.MultiplyAddEstimate(ps[pb + 19], qc, a3);
                    a4 = float.MultiplyAddEstimate(ps[pb + 20], qc, a4); a5 = float.MultiplyAddEstimate(ps[pb + 21], qc, a5); a6 = float.MultiplyAddEstimate(ps[pb + 22], qc, a6); a7 = float.MultiplyAddEstimate(ps[pb + 23], qc, a7);
                    a0 = float.MultiplyAddEstimate(ps[pb + 24], qd, a0); a1 = float.MultiplyAddEstimate(ps[pb + 25], qd, a1); a2 = float.MultiplyAddEstimate(ps[pb + 26], qd, a2); a3 = float.MultiplyAddEstimate(ps[pb + 27], qd, a3);
                    a4 = float.MultiplyAddEstimate(ps[pb + 28], qd, a4); a5 = float.MultiplyAddEstimate(ps[pb + 29], qd, a5); a6 = float.MultiplyAddEstimate(ps[pb + 30], qd, a6); a7 = float.MultiplyAddEstimate(ps[pb + 31], qd, a7);
                }
                for (; mm < rows; mm++, qr += N)
                {
                    float q = Q[qr];
                    int pb = mm * 8;
                    a0 = float.MultiplyAddEstimate(ps[pb], q, a0); a1 = float.MultiplyAddEstimate(ps[pb + 1], q, a1); a2 = float.MultiplyAddEstimate(ps[pb + 2], q, a2); a3 = float.MultiplyAddEstimate(ps[pb + 3], q, a3);
                    a4 = float.MultiplyAddEstimate(ps[pb + 4], q, a4); a5 = float.MultiplyAddEstimate(ps[pb + 5], q, a5); a6 = float.MultiplyAddEstimate(ps[pb + 6], q, a6); a7 = float.MultiplyAddEstimate(ps[pb + 7], q, a7);
                }
                long o = ((long)sl * R + rc) * N + n;
                part[o] = a0; part[o + N] = a1; part[o + 2 * (long)N] = a2; part[o + 3 * (long)N] = a3;
                part[o + 4 * (long)N] = a4; part[o + 5 * (long)N] = a5; part[o + 6 * (long)N] = a6; part[o + 7 * (long)N] = a7;
            }
            Group.Barrier();
        }
    }

    // LoraRowDotImpl with 16 rank columns per pass (R % 16 == 0): the row's P values are read once for rank 16 instead of twice. Each
    // accumulator sums the same products in the same order as the 8-column form: bit-identical. The k-lane reduction runs in two halves
    // through the same 8 KB of shared memory.
    private static void LoraRowDot16Impl(F4V P, F4V Q, FV part, int M, int K, int R, int slices)
    {
        var sh = SharedMemory.Allocate<float>(LrRows * 64);
        int gi = Grid.IdxX, rb = gi / slices, sl = gi - rb * slices;
        int tid = Group.IdxX, row = tid >> 3, kl = tid & 7;
        int m = rb * LrRows + row;
        int K4 = K >> 2;
        int k0 = sl * (LrSlice / 4), k1 = Math.Min(K4, k0 + LrSlice / 4);
        long pb = (long)m * K4;
        for (int rc = 0; rc < R; rc += 16)
        {
            float a0 = 0f; float a1 = 0f; float a2 = 0f; float a3 = 0f; float a4 = 0f; float a5 = 0f; float a6 = 0f; float a7 = 0f; float a8 = 0f; float a9 = 0f; float a10 = 0f; float a11 = 0f; float a12 = 0f; float a13 = 0f; float a14 = 0f; float a15 = 0f;
            if (m < M)
            {
                long q0 = (long)rc * K4;
                for (int k = k0 + kl; k < k1; k += 8)
                {
                    var p = P[pb + k];
                    long qb = q0 + k;
                    var q = Q[qb]; a0 = float.MultiplyAddEstimate(p.X, q.X, a0); a0 = float.MultiplyAddEstimate(p.Y, q.Y, a0); a0 = float.MultiplyAddEstimate(p.Z, q.Z, a0); a0 = float.MultiplyAddEstimate(p.W, q.W, a0);
                    q = Q[qb + K4]; a1 = float.MultiplyAddEstimate(p.X, q.X, a1); a1 = float.MultiplyAddEstimate(p.Y, q.Y, a1); a1 = float.MultiplyAddEstimate(p.Z, q.Z, a1); a1 = float.MultiplyAddEstimate(p.W, q.W, a1);
                    q = Q[qb + 2 * (long)K4]; a2 = float.MultiplyAddEstimate(p.X, q.X, a2); a2 = float.MultiplyAddEstimate(p.Y, q.Y, a2); a2 = float.MultiplyAddEstimate(p.Z, q.Z, a2); a2 = float.MultiplyAddEstimate(p.W, q.W, a2);
                    q = Q[qb + 3 * (long)K4]; a3 = float.MultiplyAddEstimate(p.X, q.X, a3); a3 = float.MultiplyAddEstimate(p.Y, q.Y, a3); a3 = float.MultiplyAddEstimate(p.Z, q.Z, a3); a3 = float.MultiplyAddEstimate(p.W, q.W, a3);
                    q = Q[qb + 4 * (long)K4]; a4 = float.MultiplyAddEstimate(p.X, q.X, a4); a4 = float.MultiplyAddEstimate(p.Y, q.Y, a4); a4 = float.MultiplyAddEstimate(p.Z, q.Z, a4); a4 = float.MultiplyAddEstimate(p.W, q.W, a4);
                    q = Q[qb + 5 * (long)K4]; a5 = float.MultiplyAddEstimate(p.X, q.X, a5); a5 = float.MultiplyAddEstimate(p.Y, q.Y, a5); a5 = float.MultiplyAddEstimate(p.Z, q.Z, a5); a5 = float.MultiplyAddEstimate(p.W, q.W, a5);
                    q = Q[qb + 6 * (long)K4]; a6 = float.MultiplyAddEstimate(p.X, q.X, a6); a6 = float.MultiplyAddEstimate(p.Y, q.Y, a6); a6 = float.MultiplyAddEstimate(p.Z, q.Z, a6); a6 = float.MultiplyAddEstimate(p.W, q.W, a6);
                    q = Q[qb + 7 * (long)K4]; a7 = float.MultiplyAddEstimate(p.X, q.X, a7); a7 = float.MultiplyAddEstimate(p.Y, q.Y, a7); a7 = float.MultiplyAddEstimate(p.Z, q.Z, a7); a7 = float.MultiplyAddEstimate(p.W, q.W, a7);
                    q = Q[qb + 8 * (long)K4]; a8 = float.MultiplyAddEstimate(p.X, q.X, a8); a8 = float.MultiplyAddEstimate(p.Y, q.Y, a8); a8 = float.MultiplyAddEstimate(p.Z, q.Z, a8); a8 = float.MultiplyAddEstimate(p.W, q.W, a8);
                    q = Q[qb + 9 * (long)K4]; a9 = float.MultiplyAddEstimate(p.X, q.X, a9); a9 = float.MultiplyAddEstimate(p.Y, q.Y, a9); a9 = float.MultiplyAddEstimate(p.Z, q.Z, a9); a9 = float.MultiplyAddEstimate(p.W, q.W, a9);
                    q = Q[qb + 10 * (long)K4]; a10 = float.MultiplyAddEstimate(p.X, q.X, a10); a10 = float.MultiplyAddEstimate(p.Y, q.Y, a10); a10 = float.MultiplyAddEstimate(p.Z, q.Z, a10); a10 = float.MultiplyAddEstimate(p.W, q.W, a10);
                    q = Q[qb + 11 * (long)K4]; a11 = float.MultiplyAddEstimate(p.X, q.X, a11); a11 = float.MultiplyAddEstimate(p.Y, q.Y, a11); a11 = float.MultiplyAddEstimate(p.Z, q.Z, a11); a11 = float.MultiplyAddEstimate(p.W, q.W, a11);
                    q = Q[qb + 12 * (long)K4]; a12 = float.MultiplyAddEstimate(p.X, q.X, a12); a12 = float.MultiplyAddEstimate(p.Y, q.Y, a12); a12 = float.MultiplyAddEstimate(p.Z, q.Z, a12); a12 = float.MultiplyAddEstimate(p.W, q.W, a12);
                    q = Q[qb + 13 * (long)K4]; a13 = float.MultiplyAddEstimate(p.X, q.X, a13); a13 = float.MultiplyAddEstimate(p.Y, q.Y, a13); a13 = float.MultiplyAddEstimate(p.Z, q.Z, a13); a13 = float.MultiplyAddEstimate(p.W, q.W, a13);
                    q = Q[qb + 14 * (long)K4]; a14 = float.MultiplyAddEstimate(p.X, q.X, a14); a14 = float.MultiplyAddEstimate(p.Y, q.Y, a14); a14 = float.MultiplyAddEstimate(p.Z, q.Z, a14); a14 = float.MultiplyAddEstimate(p.W, q.W, a14);
                    q = Q[qb + 15 * (long)K4]; a15 = float.MultiplyAddEstimate(p.X, q.X, a15); a15 = float.MultiplyAddEstimate(p.Y, q.Y, a15); a15 = float.MultiplyAddEstimate(p.Z, q.Z, a15); a15 = float.MultiplyAddEstimate(p.W, q.W, a15);
                }
            }
            {
                int b = row * 64 + kl;
                sh[b + 0] = a0; sh[b + 8] = a1; sh[b + 16] = a2; sh[b + 24] = a3; sh[b + 32] = a4; sh[b + 40] = a5; sh[b + 48] = a6; sh[b + 56] = a7;
                Group.Barrier();
                int rb8 = row * 64 + kl * 8;
                float s = sh[rb8] + sh[rb8 + 1] + sh[rb8 + 2] + sh[rb8 + 3] + sh[rb8 + 4] + sh[rb8 + 5] + sh[rb8 + 6] + sh[rb8 + 7];
                if (m < M) part[((long)sl * M + m) * R + rc + 0 + kl] = s;
                Group.Barrier();
            }
            {
                int b = row * 64 + kl;
                sh[b + 0] = a8; sh[b + 8] = a9; sh[b + 16] = a10; sh[b + 24] = a11; sh[b + 32] = a12; sh[b + 40] = a13; sh[b + 48] = a14; sh[b + 56] = a15;
                Group.Barrier();
                int rb8 = row * 64 + kl * 8;
                float s = sh[rb8] + sh[rb8 + 1] + sh[rb8 + 2] + sh[rb8 + 3] + sh[rb8 + 4] + sh[rb8 + 5] + sh[rb8 + 6] + sh[rb8 + 7];
                if (m < M) part[((long)sl * M + m) * R + rc + 8 + kl] = s;
                Group.Barrier();
            }
        }
    }

    // LoraColDotImpl with 16 rank columns per pass (R % 16 == 0): the column's Q values are read once for rank 16. Bit-identical.
    private static void LoraColDot16Impl(FV P, FV Q, FV part, int M, int N, int R, int slices)
    {
        var ps = SharedMemory.Allocate<float>(LcSlice * 16);
        int gi = Grid.IdxX, cb = gi / slices, sl = gi - cb * slices;
        int tid = Group.IdxX, n = cb * LcCols + tid;
        int m0 = sl * LcSlice, rows = Math.Min(M, m0 + LcSlice) - m0;
        for (int rc = 0; rc < R; rc += 16)
        {
            for (int e = tid; e < LcSlice * 16; e += LcCols)
            {
                int mm = e >> 4, j = e & 15;
                ps[e] = mm < rows ? P[(long)(m0 + mm) * R + rc + j] : 0f;
            }
            Group.Barrier();
            float a0 = 0f; float a1 = 0f; float a2 = 0f; float a3 = 0f; float a4 = 0f; float a5 = 0f; float a6 = 0f; float a7 = 0f; float a8 = 0f; float a9 = 0f; float a10 = 0f; float a11 = 0f; float a12 = 0f; float a13 = 0f; float a14 = 0f; float a15 = 0f;
            if (n < N)
            {
                int mm = 0;
                long qr = (long)m0 * N + n;
                for (; mm + 4 <= rows; mm += 4, qr += 4 * (long)N)
                {
                    float qa = Q[qr], qb = Q[qr + N], qc = Q[qr + 2 * (long)N], qd = Q[qr + 3 * (long)N];
                    int pb = mm * 16;
                    a0 = float.MultiplyAddEstimate(ps[pb + 0], qa, a0); a1 = float.MultiplyAddEstimate(ps[pb + 1], qa, a1); a2 = float.MultiplyAddEstimate(ps[pb + 2], qa, a2); a3 = float.MultiplyAddEstimate(ps[pb + 3], qa, a3); a4 = float.MultiplyAddEstimate(ps[pb + 4], qa, a4); a5 = float.MultiplyAddEstimate(ps[pb + 5], qa, a5); a6 = float.MultiplyAddEstimate(ps[pb + 6], qa, a6); a7 = float.MultiplyAddEstimate(ps[pb + 7], qa, a7); a8 = float.MultiplyAddEstimate(ps[pb + 8], qa, a8); a9 = float.MultiplyAddEstimate(ps[pb + 9], qa, a9); a10 = float.MultiplyAddEstimate(ps[pb + 10], qa, a10); a11 = float.MultiplyAddEstimate(ps[pb + 11], qa, a11); a12 = float.MultiplyAddEstimate(ps[pb + 12], qa, a12); a13 = float.MultiplyAddEstimate(ps[pb + 13], qa, a13); a14 = float.MultiplyAddEstimate(ps[pb + 14], qa, a14); a15 = float.MultiplyAddEstimate(ps[pb + 15], qa, a15);
                    a0 = float.MultiplyAddEstimate(ps[pb + 16], qb, a0); a1 = float.MultiplyAddEstimate(ps[pb + 17], qb, a1); a2 = float.MultiplyAddEstimate(ps[pb + 18], qb, a2); a3 = float.MultiplyAddEstimate(ps[pb + 19], qb, a3); a4 = float.MultiplyAddEstimate(ps[pb + 20], qb, a4); a5 = float.MultiplyAddEstimate(ps[pb + 21], qb, a5); a6 = float.MultiplyAddEstimate(ps[pb + 22], qb, a6); a7 = float.MultiplyAddEstimate(ps[pb + 23], qb, a7); a8 = float.MultiplyAddEstimate(ps[pb + 24], qb, a8); a9 = float.MultiplyAddEstimate(ps[pb + 25], qb, a9); a10 = float.MultiplyAddEstimate(ps[pb + 26], qb, a10); a11 = float.MultiplyAddEstimate(ps[pb + 27], qb, a11); a12 = float.MultiplyAddEstimate(ps[pb + 28], qb, a12); a13 = float.MultiplyAddEstimate(ps[pb + 29], qb, a13); a14 = float.MultiplyAddEstimate(ps[pb + 30], qb, a14); a15 = float.MultiplyAddEstimate(ps[pb + 31], qb, a15);
                    a0 = float.MultiplyAddEstimate(ps[pb + 32], qc, a0); a1 = float.MultiplyAddEstimate(ps[pb + 33], qc, a1); a2 = float.MultiplyAddEstimate(ps[pb + 34], qc, a2); a3 = float.MultiplyAddEstimate(ps[pb + 35], qc, a3); a4 = float.MultiplyAddEstimate(ps[pb + 36], qc, a4); a5 = float.MultiplyAddEstimate(ps[pb + 37], qc, a5); a6 = float.MultiplyAddEstimate(ps[pb + 38], qc, a6); a7 = float.MultiplyAddEstimate(ps[pb + 39], qc, a7); a8 = float.MultiplyAddEstimate(ps[pb + 40], qc, a8); a9 = float.MultiplyAddEstimate(ps[pb + 41], qc, a9); a10 = float.MultiplyAddEstimate(ps[pb + 42], qc, a10); a11 = float.MultiplyAddEstimate(ps[pb + 43], qc, a11); a12 = float.MultiplyAddEstimate(ps[pb + 44], qc, a12); a13 = float.MultiplyAddEstimate(ps[pb + 45], qc, a13); a14 = float.MultiplyAddEstimate(ps[pb + 46], qc, a14); a15 = float.MultiplyAddEstimate(ps[pb + 47], qc, a15);
                    a0 = float.MultiplyAddEstimate(ps[pb + 48], qd, a0); a1 = float.MultiplyAddEstimate(ps[pb + 49], qd, a1); a2 = float.MultiplyAddEstimate(ps[pb + 50], qd, a2); a3 = float.MultiplyAddEstimate(ps[pb + 51], qd, a3); a4 = float.MultiplyAddEstimate(ps[pb + 52], qd, a4); a5 = float.MultiplyAddEstimate(ps[pb + 53], qd, a5); a6 = float.MultiplyAddEstimate(ps[pb + 54], qd, a6); a7 = float.MultiplyAddEstimate(ps[pb + 55], qd, a7); a8 = float.MultiplyAddEstimate(ps[pb + 56], qd, a8); a9 = float.MultiplyAddEstimate(ps[pb + 57], qd, a9); a10 = float.MultiplyAddEstimate(ps[pb + 58], qd, a10); a11 = float.MultiplyAddEstimate(ps[pb + 59], qd, a11); a12 = float.MultiplyAddEstimate(ps[pb + 60], qd, a12); a13 = float.MultiplyAddEstimate(ps[pb + 61], qd, a13); a14 = float.MultiplyAddEstimate(ps[pb + 62], qd, a14); a15 = float.MultiplyAddEstimate(ps[pb + 63], qd, a15);
                }
                for (; mm < rows; mm++, qr += N)
                {
                    float q = Q[qr];
                    int pb = mm * 16;
                    a0 = float.MultiplyAddEstimate(ps[pb + 0], q, a0); a1 = float.MultiplyAddEstimate(ps[pb + 1], q, a1); a2 = float.MultiplyAddEstimate(ps[pb + 2], q, a2); a3 = float.MultiplyAddEstimate(ps[pb + 3], q, a3); a4 = float.MultiplyAddEstimate(ps[pb + 4], q, a4); a5 = float.MultiplyAddEstimate(ps[pb + 5], q, a5); a6 = float.MultiplyAddEstimate(ps[pb + 6], q, a6); a7 = float.MultiplyAddEstimate(ps[pb + 7], q, a7); a8 = float.MultiplyAddEstimate(ps[pb + 8], q, a8); a9 = float.MultiplyAddEstimate(ps[pb + 9], q, a9); a10 = float.MultiplyAddEstimate(ps[pb + 10], q, a10); a11 = float.MultiplyAddEstimate(ps[pb + 11], q, a11); a12 = float.MultiplyAddEstimate(ps[pb + 12], q, a12); a13 = float.MultiplyAddEstimate(ps[pb + 13], q, a13); a14 = float.MultiplyAddEstimate(ps[pb + 14], q, a14); a15 = float.MultiplyAddEstimate(ps[pb + 15], q, a15);
                }
                long o = ((long)sl * R + rc) * N + n;
                part[o + 0 * (long)N] = a0; part[o + 1 * (long)N] = a1; part[o + 2 * (long)N] = a2; part[o + 3 * (long)N] = a3; part[o + 4 * (long)N] = a4; part[o + 5 * (long)N] = a5; part[o + 6 * (long)N] = a6; part[o + 7 * (long)N] = a7; part[o + 8 * (long)N] = a8; part[o + 9 * (long)N] = a9; part[o + 10 * (long)N] = a10; part[o + 11 * (long)N] = a11; part[o + 12 * (long)N] = a12; part[o + 13 * (long)N] = a13; part[o + 14 * (long)N] = a14; part[o + 15 * (long)N] = a15;
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
            s00 = float.MultiplyAddEstimate(pa0.X, q0.X, s00); s01 = float.MultiplyAddEstimate(pa0.X, q0.Y, s01); s02 = float.MultiplyAddEstimate(pa0.X, q0.Z, s02); s03 = float.MultiplyAddEstimate(pa0.X, q0.W, s03);
            s00 = float.MultiplyAddEstimate(pa0.Y, q1.X, s00); s01 = float.MultiplyAddEstimate(pa0.Y, q1.Y, s01); s02 = float.MultiplyAddEstimate(pa0.Y, q1.Z, s02); s03 = float.MultiplyAddEstimate(pa0.Y, q1.W, s03);
            s00 = float.MultiplyAddEstimate(pa0.Z, q2.X, s00); s01 = float.MultiplyAddEstimate(pa0.Z, q2.Y, s01); s02 = float.MultiplyAddEstimate(pa0.Z, q2.Z, s02); s03 = float.MultiplyAddEstimate(pa0.Z, q2.W, s03);
            s00 = float.MultiplyAddEstimate(pa0.W, q3.X, s00); s01 = float.MultiplyAddEstimate(pa0.W, q3.Y, s01); s02 = float.MultiplyAddEstimate(pa0.W, q3.Z, s02); s03 = float.MultiplyAddEstimate(pa0.W, q3.W, s03);
            s00 = float.MultiplyAddEstimate(pc0.X, q4.X, s00); s01 = float.MultiplyAddEstimate(pc0.X, q4.Y, s01); s02 = float.MultiplyAddEstimate(pc0.X, q4.Z, s02); s03 = float.MultiplyAddEstimate(pc0.X, q4.W, s03);
            s00 = float.MultiplyAddEstimate(pc0.Y, q5.X, s00); s01 = float.MultiplyAddEstimate(pc0.Y, q5.Y, s01); s02 = float.MultiplyAddEstimate(pc0.Y, q5.Z, s02); s03 = float.MultiplyAddEstimate(pc0.Y, q5.W, s03);
            s00 = float.MultiplyAddEstimate(pc0.Z, q6.X, s00); s01 = float.MultiplyAddEstimate(pc0.Z, q6.Y, s01); s02 = float.MultiplyAddEstimate(pc0.Z, q6.Z, s02); s03 = float.MultiplyAddEstimate(pc0.Z, q6.W, s03);
            s00 = float.MultiplyAddEstimate(pc0.W, q7.X, s00); s01 = float.MultiplyAddEstimate(pc0.W, q7.Y, s01); s02 = float.MultiplyAddEstimate(pc0.W, q7.Z, s02); s03 = float.MultiplyAddEstimate(pc0.W, q7.W, s03);
            s10 = float.MultiplyAddEstimate(pa1.X, q0.X, s10); s11 = float.MultiplyAddEstimate(pa1.X, q0.Y, s11); s12 = float.MultiplyAddEstimate(pa1.X, q0.Z, s12); s13 = float.MultiplyAddEstimate(pa1.X, q0.W, s13);
            s10 = float.MultiplyAddEstimate(pa1.Y, q1.X, s10); s11 = float.MultiplyAddEstimate(pa1.Y, q1.Y, s11); s12 = float.MultiplyAddEstimate(pa1.Y, q1.Z, s12); s13 = float.MultiplyAddEstimate(pa1.Y, q1.W, s13);
            s10 = float.MultiplyAddEstimate(pa1.Z, q2.X, s10); s11 = float.MultiplyAddEstimate(pa1.Z, q2.Y, s11); s12 = float.MultiplyAddEstimate(pa1.Z, q2.Z, s12); s13 = float.MultiplyAddEstimate(pa1.Z, q2.W, s13);
            s10 = float.MultiplyAddEstimate(pa1.W, q3.X, s10); s11 = float.MultiplyAddEstimate(pa1.W, q3.Y, s11); s12 = float.MultiplyAddEstimate(pa1.W, q3.Z, s12); s13 = float.MultiplyAddEstimate(pa1.W, q3.W, s13);
            s10 = float.MultiplyAddEstimate(pc1.X, q4.X, s10); s11 = float.MultiplyAddEstimate(pc1.X, q4.Y, s11); s12 = float.MultiplyAddEstimate(pc1.X, q4.Z, s12); s13 = float.MultiplyAddEstimate(pc1.X, q4.W, s13);
            s10 = float.MultiplyAddEstimate(pc1.Y, q5.X, s10); s11 = float.MultiplyAddEstimate(pc1.Y, q5.Y, s11); s12 = float.MultiplyAddEstimate(pc1.Y, q5.Z, s12); s13 = float.MultiplyAddEstimate(pc1.Y, q5.W, s13);
            s10 = float.MultiplyAddEstimate(pc1.Z, q6.X, s10); s11 = float.MultiplyAddEstimate(pc1.Z, q6.Y, s11); s12 = float.MultiplyAddEstimate(pc1.Z, q6.Z, s12); s13 = float.MultiplyAddEstimate(pc1.Z, q6.W, s13);
            s10 = float.MultiplyAddEstimate(pc1.W, q7.X, s10); s11 = float.MultiplyAddEstimate(pc1.W, q7.Y, s11); s12 = float.MultiplyAddEstimate(pc1.W, q7.Z, s12); s13 = float.MultiplyAddEstimate(pc1.W, q7.W, s13);
            s20 = float.MultiplyAddEstimate(pa2.X, q0.X, s20); s21 = float.MultiplyAddEstimate(pa2.X, q0.Y, s21); s22 = float.MultiplyAddEstimate(pa2.X, q0.Z, s22); s23 = float.MultiplyAddEstimate(pa2.X, q0.W, s23);
            s20 = float.MultiplyAddEstimate(pa2.Y, q1.X, s20); s21 = float.MultiplyAddEstimate(pa2.Y, q1.Y, s21); s22 = float.MultiplyAddEstimate(pa2.Y, q1.Z, s22); s23 = float.MultiplyAddEstimate(pa2.Y, q1.W, s23);
            s20 = float.MultiplyAddEstimate(pa2.Z, q2.X, s20); s21 = float.MultiplyAddEstimate(pa2.Z, q2.Y, s21); s22 = float.MultiplyAddEstimate(pa2.Z, q2.Z, s22); s23 = float.MultiplyAddEstimate(pa2.Z, q2.W, s23);
            s20 = float.MultiplyAddEstimate(pa2.W, q3.X, s20); s21 = float.MultiplyAddEstimate(pa2.W, q3.Y, s21); s22 = float.MultiplyAddEstimate(pa2.W, q3.Z, s22); s23 = float.MultiplyAddEstimate(pa2.W, q3.W, s23);
            s20 = float.MultiplyAddEstimate(pc2.X, q4.X, s20); s21 = float.MultiplyAddEstimate(pc2.X, q4.Y, s21); s22 = float.MultiplyAddEstimate(pc2.X, q4.Z, s22); s23 = float.MultiplyAddEstimate(pc2.X, q4.W, s23);
            s20 = float.MultiplyAddEstimate(pc2.Y, q5.X, s20); s21 = float.MultiplyAddEstimate(pc2.Y, q5.Y, s21); s22 = float.MultiplyAddEstimate(pc2.Y, q5.Z, s22); s23 = float.MultiplyAddEstimate(pc2.Y, q5.W, s23);
            s20 = float.MultiplyAddEstimate(pc2.Z, q6.X, s20); s21 = float.MultiplyAddEstimate(pc2.Z, q6.Y, s21); s22 = float.MultiplyAddEstimate(pc2.Z, q6.Z, s22); s23 = float.MultiplyAddEstimate(pc2.Z, q6.W, s23);
            s20 = float.MultiplyAddEstimate(pc2.W, q7.X, s20); s21 = float.MultiplyAddEstimate(pc2.W, q7.Y, s21); s22 = float.MultiplyAddEstimate(pc2.W, q7.Z, s22); s23 = float.MultiplyAddEstimate(pc2.W, q7.W, s23);
            s30 = float.MultiplyAddEstimate(pa3.X, q0.X, s30); s31 = float.MultiplyAddEstimate(pa3.X, q0.Y, s31); s32 = float.MultiplyAddEstimate(pa3.X, q0.Z, s32); s33 = float.MultiplyAddEstimate(pa3.X, q0.W, s33);
            s30 = float.MultiplyAddEstimate(pa3.Y, q1.X, s30); s31 = float.MultiplyAddEstimate(pa3.Y, q1.Y, s31); s32 = float.MultiplyAddEstimate(pa3.Y, q1.Z, s32); s33 = float.MultiplyAddEstimate(pa3.Y, q1.W, s33);
            s30 = float.MultiplyAddEstimate(pa3.Z, q2.X, s30); s31 = float.MultiplyAddEstimate(pa3.Z, q2.Y, s31); s32 = float.MultiplyAddEstimate(pa3.Z, q2.Z, s32); s33 = float.MultiplyAddEstimate(pa3.Z, q2.W, s33);
            s30 = float.MultiplyAddEstimate(pa3.W, q3.X, s30); s31 = float.MultiplyAddEstimate(pa3.W, q3.Y, s31); s32 = float.MultiplyAddEstimate(pa3.W, q3.Z, s32); s33 = float.MultiplyAddEstimate(pa3.W, q3.W, s33);
            s30 = float.MultiplyAddEstimate(pc3.X, q4.X, s30); s31 = float.MultiplyAddEstimate(pc3.X, q4.Y, s31); s32 = float.MultiplyAddEstimate(pc3.X, q4.Z, s32); s33 = float.MultiplyAddEstimate(pc3.X, q4.W, s33);
            s30 = float.MultiplyAddEstimate(pc3.Y, q5.X, s30); s31 = float.MultiplyAddEstimate(pc3.Y, q5.Y, s31); s32 = float.MultiplyAddEstimate(pc3.Y, q5.Z, s32); s33 = float.MultiplyAddEstimate(pc3.Y, q5.W, s33);
            s30 = float.MultiplyAddEstimate(pc3.Z, q6.X, s30); s31 = float.MultiplyAddEstimate(pc3.Z, q6.Y, s31); s32 = float.MultiplyAddEstimate(pc3.Z, q6.Z, s32); s33 = float.MultiplyAddEstimate(pc3.Z, q6.W, s33);
            s30 = float.MultiplyAddEstimate(pc3.W, q7.X, s30); s31 = float.MultiplyAddEstimate(pc3.W, q7.Y, s31); s32 = float.MultiplyAddEstimate(pc3.W, q7.Z, s32); s33 = float.MultiplyAddEstimate(pc3.W, q7.W, s33);
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
        for (int k = 0; k < K; k++) s = float.MultiplyAddEstimate(P[pb + k], (qLayout == 0 ? Q[(long)k * R + r] : Q[(long)r * K + k]), s);
        outp[i] = s * scale;
    }

    private static void LoraCombineNaiveImpl(Index1D i, FV P, FV Q, FV bas, FV outp, int N, int R, int hasBase, float scale)
    {
        int m = i / N, n = i - m * N;
        long pb = (long)m * R;
        float s = 0f;
        for (int r = 0; r < R; r++) s = float.MultiplyAddEstimate(P[pb + r], Q[(long)r * N + n], s);
        float v = s * scale;
        outp[i] = hasBase != 0 ? bas[i] + v : v;
    }

    private static void LoraColDotNaiveImpl(Index1D i, FV P, FV Q, FV outp, int M, int N, int R, int transpose, float scale)
    {
        int r, n;
        if (transpose != 0) { n = i / R; r = i - n * R; } else { r = i / N; n = i - r * N; }
        float s = 0f;
        for (int m = 0; m < M; m++) s = float.MultiplyAddEstimate(P[(long)m * R + r], Q[(long)m * N + n], s);
        outp[i] = s * scale;
    }

    /// <summary>
    /// Free the scratch buffers a growth replaced (the attention, LoRA and transposed-weight scratch), kept until now because a queued launch could
    /// still read them. Call only when the device has finished every launch issued so far. Returns the bytes freed. Added 2026-11-05: they were
    /// freed only at Dispose, so every growth during a session left its predecessor allocated.
    /// </summary>
    public long ReleaseRetired()
    {
        long bytes = 0;
        foreach (var b in _gRetired) { bytes += b.LengthInBytes; b.Dispose(); }
        _gRetired.Clear();
        foreach (var b in _tRetired16) { bytes += b.LengthInBytes; b.Dispose(); }
        _tRetired16.Clear();
        foreach (var b in _retiredWt) { bytes += b.LengthInBytes; b.Dispose(); }
        _retiredWt.Clear();
        return bytes;
    }

    private void DisposeLora()
    {
        _lrPart?.Dispose(); _lrDxa?.Dispose(); _lrNone?.Dispose(); _lrAR?.Dispose();
        _lrPart = _lrDxa = _lrNone = _lrAR = null;
    }
}
