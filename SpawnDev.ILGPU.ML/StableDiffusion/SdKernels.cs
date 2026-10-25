using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.StableDiffusion;

using FV = ArrayView1D<float, Stride1D.Dense>;
using HV = ArrayView1D<global::ILGPU.Half, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// The kernels of the Stable Diffusion UNet, CLIP text encoder and VAE that the shared libraries do not have, all on channel-last fp32 activations [pixels, C]
/// (the layout the gathered-row convolution GEMM reads): GroupNorm with an optional SiLU, GEGLU, the convolution neighbour tables (plain, strided, nearest x2),
/// channel concatenation for the skip connections, a row-vector add, causal attention for the 77-token text encoder, and the bounded convolution / linear launches.
/// Every launch is bounded: a convolution or linear layer is split into row chunks of at most <see cref="MaxFlopsPerDispatch"/> (a long launch stalls the compositor and
/// trips the OS watchdog), and the GroupNorm reductions run as chunked partial sums.
/// </summary>
public sealed class SdKernels
{
    private readonly Accelerator _acc;
    private readonly SparseKernels _sk;
    private MemoryBuffer1D<float, Stride1D.Dense>? _partial, _stats;
    private MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>? _noBias;

    /// <summary>The most multiply-adds x 2 one convolution or linear dispatch may cover (default 2^34: ~10 ms on a mid-range desktop GPU's portable GEMM); larger ones run as row chunks.</summary>
    public static long MaxFlopsPerDispatch { get; set; } = 1L << 34;

    public SdKernels(Accelerator accelerator)
    {
        _acc = accelerator; _sk = new SparseKernels(accelerator);
    }

    public SparseKernels Sparse => _sk;

    // ---------------------------------------------------------------------------------------------------
    // GroupNorm over channel-last x [rows, C]: group g = channels [g * C / G, (g + 1) * C / G). Two chunked reductions (the mean, then the centred sum of squares:
    // no cancellation), then the elementwise part with fp16 gamma / beta and an optional SiLU.
    // ---------------------------------------------------------------------------------------------------
    private const int GnThreads = 256, GnChunkElems = 16384;
    private Action<KernelConfig, FV, FV, FV, int, int, int, int, int, int>? _gnPartial;
    private static void GnPartialImpl(FV x, FV stats, FV partial, int C, int cg, int rows, int rowsPerChunk, int chunks, int centred)
    {
        int g = Grid.IdxX / chunks, chunk = Grid.IdxX - g * chunks, t = Group.IdxX;
        var sh = SharedMemory.Allocate<float>(GnThreads);
        int r0 = chunk * rowsPerChunk, r1 = Math.Min(rows, r0 + rowsPerChunk);
        int n = (r1 - r0) * cg;
        float mean = centred != 0 ? stats[2 * g] : 0f;
        float s = 0f;
        for (int i = t; i < n; i += GnThreads)
        {
            int r = r0 + i / cg, c = g * cg + i % cg;
            float v = x[(long)r * C + c] - mean;
            s += centred != 0 ? v * v : v;
        }
        sh[t] = s;
        Group.Barrier();
        for (int st = GnThreads / 2; st > 0; st >>= 1) { if (t < st) sh[t] += sh[t + st]; Group.Barrier(); }
        if (t == 0) partial[Grid.IdxX] = sh[0];
    }
    private Action<Index1D, FV, FV, int, int, float, int>? _gnFinal;
    private static void GnFinalImpl(Index1D g, FV partial, FV stats, int chunks, int count, float eps, int centred)
    {
        float s = 0f;
        for (int c = 0; c < chunks; c++) s += partial[g * chunks + c];
        if (centred == 0) stats[2 * g] = s / count;
        else stats[2 * g + 1] = 1f / MathF.Sqrt(s / count + eps);
    }
    private Action<Index1D, FV, FV, FV, HV, HV, int, int, int>? _gnApply;
    private static void GnApplyImpl(Index1D i, FV x, FV o, FV stats, HV gamma, HV beta, int C, int cg, int silu)
    {
        int c = (int)(i % C), g = c / cg;
        float v = (x[i] - stats[2 * g]) * stats[2 * g + 1] * PrecisionConvert.ConvertToSingle(gamma[c]) + PrecisionConvert.ConvertToSingle(beta[c]);
        if (silu != 0) v = v / (1f + MathF.Exp(-v));
        o[i] = v;
    }

    /// <summary>o = GroupNorm(x) * gamma + beta (then SiLU when <paramref name="silu"/>), x and o channel-last [rows, C]; <paramref name="o"/> may not alias <paramref name="x"/>.</summary>
    public void GroupNorm(FV x, FV o, HV gamma, HV beta, int rows, int C, int groups, float eps, bool silu)
    {
        int cg = C / groups;
        int rowsPerChunk = Math.Max(1, GnChunkElems / cg);
        int chunks = (rows + rowsPerChunk - 1) / rowsPerChunk;
        long need = (long)groups * chunks;
        if (_partial == null || _partial.Length < need) { _acc.Fence(); _partial?.Dispose(); _partial = _acc.Allocate1D<float>(Math.Max(need, 4096)); }
        if (_stats == null || _stats.Length < 2L * groups) { _acc.Fence(); _stats?.Dispose(); _stats = _acc.Allocate1D<float>(Math.Max(2L * groups, 256)); }
        _gnPartial ??= _acc.LoadStreamKernel<FV, FV, FV, int, int, int, int, int, int>(GnPartialImpl);
        _gnFinal ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, float, int>(GnFinalImpl);
        _gnApply ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, HV, HV, int, int, int>(GnApplyImpl);
        int count = rows * cg;
        var cfg = new KernelConfig((int)need, GnThreads);
        _gnPartial(cfg, x, _stats.View, _partial.View, C, cg, rows, rowsPerChunk, chunks, 0);
        _gnFinal(groups, _partial.View, _stats.View, chunks, count, eps, 0);
        _gnPartial(cfg, x, _stats.View, _partial.View, C, cg, rows, rowsPerChunk, chunks, 1);
        _gnFinal(groups, _partial.View, _stats.View, chunks, count, eps, 1);
        _gnApply((int)((long)rows * C), x, o, _stats.View, gamma, beta, C, cg, silu ? 1 : 0);
    }

    // ---------------------------------------------------------------------------------------------------
    // GEGLU: h [rows, 2F] = (value | gate) -> o [rows, F] = value * gelu(gate)   (diffusers GEGLU, exact erf GELU)
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, FV, int>? _geglu;
    private static void GegluImpl(Index1D i, FV h, FV o, int F)
    {
        long r = i / F, f = i % F;
        o[i] = h[r * 2 * F + f] * DiTKernels.Activate(h[r * 2 * F + F + f], 3);
    }
    public void Geglu(FV h, FV o, int rows, int F)
    {
        _geglu ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int>(GegluImpl);
        _geglu((int)((long)rows * F), h, o, F);
    }

    // ---------------------------------------------------------------------------------------------------
    // Convolution neighbour table (row stride 27, the layout of SparseKernels.ConvTaps): output pixel m = oy * outW + ox, tap t = ky * k + kx reads the pixel
    // (iy, ix) = (oy * stride - pad + ky, ox * stride - pad + kx) of a picture of size (inH * up, inW * up); up = 2 is the nearest x2 upsampling in front of the
    // convolution (the pixel (iy / 2, ix / 2) of the stored picture). Outside the picture: -1 (zero padding). padEnd = the padding on the bottom / right edge
    // (the VAE encoder's downsampler pads (0, 1, 0, 1) and convolves with no padding).
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, IV, int, int, int, int, int, int, int, int>? _table;
    private static void TableImpl(Index1D idx, IV nbr, int inH, int inW, int outW, int k, int stride, int pad, int up, int dummy)
    {
        int taps = k * k;
        int t = idx % taps, m = idx / taps;
        int ox = m % outW, oy = m / outW;
        int ky = t / k, kx = t - ky * k;
        int iy = oy * stride - pad + ky, ix = ox * stride - pad + kx;
        int H = inH * up, W = inW * up;
        nbr[m * 27 + t] = (iy >= 0 && iy < H && ix >= 0 && ix < W) ? (iy / up) * inW + ix / up : -1;
    }

    /// <summary>Fill a neighbour table for a k x k convolution (k = 1 or 3): output size (outH, outW) for an input of (inH, inW).</summary>
    public void BuildTable(IV nbr, int inH, int inW, int outH, int outW, int k, int stride, int pad, int up)
    {
        _table ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, int, int, int, int, int, int, int, int>(TableImpl);
        _table(outH * outW * k * k, nbr, inH, inW, outW, k, stride, pad, up, 0);
    }

    /// <summary>
    /// y [M, cout] = conv(x) + bias through the neighbour table (weight in the gather layout [cout][taps][cin], fp16), split into row chunks of bounded work.
    /// <paramref name="exact"/> keeps the activations fp32 (the portable tiled GEMM, never the tensor cores, which round them to fp16): the VAE needs it (SDXL's
    /// VAE activations overflow fp16).
    /// </summary>
    public void Conv(FV x, IV nbr, HV weight, HV bias, FV y, int M, int cin, int cout, int taps, bool exact = false)
    {
        long perRow = 2L * taps * cin * cout;
        int chunk = (int)Math.Max(64, Math.Min(M, MaxFlopsPerDispatch / Math.Max(1, perRow)));
        for (int m0 = 0; m0 < M; m0 += chunk)
        {
            int rows = Math.Min(chunk, M - m0);
            var nb = nbr.SubView((long)m0 * 27, (long)rows * 27);
            var yo = y.SubView((long)m0 * cout, (long)rows * cout);
            if (exact && _sk.ConvTapsExactOk(rows, taps, cin, cout, x.Length / cin)) _sk.ConvTapsExact(x, nb, weight, yo, rows, cin, cout, taps, false, bias);
            else _sk.ConvTaps(x, nb, weight, bias, yo, rows, cin, cout, taps);
        }
    }

    /// <summary>
    /// o [M, N] = act(x W^T + b) (+ res), the linear layer split into row chunks of bounded work. <paramref name="res"/> may be <paramref name="o"/> (an in-place residual).
    /// <paramref name="exact"/> keeps the activations fp32 (no tensor cores).
    /// </summary>
    public void Linear(FV x, HV weight, HV? bias, FV o, int M, int K, int N, int act = 0, FV? res = null, bool exact = false)
    {
        long perRow = 2L * K * N;
        int chunk = (int)Math.Max(64, Math.Min(M, MaxFlopsPerDispatch / Math.Max(1, perRow)));
        if (chunk >= M) { _sk.Linear(x, weight, bias, o, M, K, N, act, res, exact); return; }
        for (int m0 = 0; m0 < M; m0 += chunk)
        {
            int rows = Math.Min(chunk, M - m0);
            _sk.Linear(x.SubView((long)m0 * K, (long)rows * K), weight, bias, o.SubView((long)m0 * N, (long)rows * N), rows, K, N, act,
                res.HasValue ? res.Value.SubView((long)m0 * N, (long)rows * N) : null, exact);
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // channel concatenation (the UNet's skip connections): o [rows, ca + cb] = (a [rows, ca] | b [rows, cb])
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, FV, FV, int, int>? _concat;
    private static void ConcatImpl(Index1D i, FV a, FV b, FV o, int ca, int cb)
    {
        int C = ca + cb;
        long r = i / C; int c = (int)(i % C);
        o[i] = c < ca ? a[r * ca + c] : b[r * cb + c - ca];
    }
    public void ConcatChannels(FV a, FV b, FV o, int rows, int ca, int cb)
    {
        _concat ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int>(ConcatImpl);
        _concat((int)((long)rows * (ca + cb)), a, b, o, ca, cb);
    }

    // dst [rows, W] columns [col, col + C) = src [rows, C] (SDXL's two text towers side by side)
    private Action<Index1D, FV, FV, int, int, int>? _copyCols;
    private static void CopyColumnsImpl(Index1D i, FV src, FV dst, int C, int W, int col)
    {
        long r = i / C; int c = (int)(i % C);
        dst[r * W + col + c] = src[i];
    }
    public void CopyColumns(FV src, FV dst, int rows, int C, int W, int col)
    {
        _copyCols ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int>(CopyColumnsImpl);
        _copyCols((int)((long)rows * C), src, dst, C, W, col);
    }

    // x [rows, C] += v [C] (the ResBlock's time embedding projection)
    private Action<Index1D, FV, FV, int>? _addRow;
    private static void AddRowImpl(Index1D i, FV x, FV v, int C) => x[i] = x[i] + v[(int)(i % C)];
    public void AddRowVector(FV x, FV v, int rows, int C)
    {
        _addRow ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int>(AddRowImpl);
        _addRow((int)((long)rows * C), x, v, C);
    }

    // ---------------------------------------------------------------------------------------------------
    // LoRA merge: W[rowOff + n, k] += scale * sum_r up[n, r] * down[r, k'] on an fp16 weight. A convolution weight is held in the gather layout [cout][taps][cin]
    // while the LoRA's down is [rank][cin][taps]: k = t * cin + ci of the stored row reads k' = ci * taps + t.
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, HV, HV, HV, int, int, int, int, int, int, float>? _mergeLora;
    private static void MergeLoraImpl(Index1D i, HV w, HV up, HV down, int K, int rank, int rowOff, int cols, int taps, int cin, float scale)
    {
        int n = i / K, k = i % K;
        int kk = taps > 1 ? (k % cin) * taps + k / cin : k;
        float acc = 0f;
        for (int r = 0; r < rank; r++) acc += PrecisionConvert.ConvertToSingle(up[(long)n * rank + r]) * PrecisionConvert.ConvertToSingle(down[(long)r * K + kk]);
        long wi = (long)(rowOff + n) * cols + k;
        w[wi] = PrecisionConvert.ConvertFromSingle<global::ILGPU.Half>(PrecisionConvert.ConvertToSingle(w[wi]) + scale * acc);
    }
    public void MergeLora(HV w, HV up, HV down, int N, int K, int rank, int rowOff, int cols, int taps, int cin, float scale)
    {
        _mergeLora ??= _acc.LoadAutoGroupedStreamKernel<Index1D, HV, HV, HV, int, int, int, int, int, int, float>(MergeLoraImpl);
        _mergeLora((int)((long)N * K), w, up, down, K, rank, rowOff, cols, taps, cin, scale);
    }

    // fp16 -> fp32 (reading a small weight back to the host)
    private Action<Index1D, HV, FV>? _h2f;
    private static void HalfToFloatImpl(Index1D i, HV src, FV dst) => dst[i] = PrecisionConvert.ConvertToSingle(src[i]);
    public void HalfToFloat(HV src, FV dst)
    {
        _h2f ??= _acc.LoadAutoGroupedStreamKernel<Index1D, HV, FV>(HalfToFloatImpl);
        _h2f((int)src.Length, src, dst);
    }

    // x += y (elementwise)
    private Action<Index1D, FV, FV>? _addInPlace;
    private static void AddInPlaceImpl(Index1D i, FV x, FV y) => x[i] = x[i] + y[i];
    public void AddInPlace(FV x, FV y, long count)
    {
        _addInPlace ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(AddInPlaceImpl);
        _addInPlace((int)count, x, y);
    }

    // ---------------------------------------------------------------------------------------------------
    // text encoder: token + position embedding, causal attention over the (short) prompt, quick GELU
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, IV, HV, HV, FV, int>? _embed;
    private static void EmbedImpl(Index1D i, IV ids, HV tok, HV pos, FV o, int C)
    {
        int t = i / C, c = i % C;
        o[i] = PrecisionConvert.ConvertToSingle(tok[(long)ids[t] * C + c]) + PrecisionConvert.ConvertToSingle(pos[(long)t * C + c]);
    }
    /// <summary>o [T, C] = token_embedding[ids[t]] + position_embedding[t].</summary>
    public void Embed(IV ids, HV tokenTable, HV positionTable, FV o, int T, int C)
    {
        _embed ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, HV, HV, FV, int>(EmbedImpl);
        _embed(T * C, ids, tokenTable, positionTable, o, C);
    }

    // causal self attention, one thread per (query, head): qkv [T, 3C] (q | k | v), o [T, C]. T is the prompt length (77): two passes over the keys, no score buffer.
    private Action<Index1D, FV, FV, int, int, int, float>? _causal;
    private static void CausalImpl(Index1D idx, FV qkv, FV o, int T, int H, int D, float scale)
    {
        int h = idx % H, i = idx / H;
        int C = H * D;
        long qb = (long)i * 3 * C + h * D;
        float mx = -1e30f;
        for (int j = 0; j <= i; j++)
        {
            long kb = (long)j * 3 * C + C + h * D;
            float s = 0f;
            for (int d = 0; d < D; d++) s += qkv[qb + d] * qkv[kb + d];
            mx = MathF.Max(mx, s * scale);
        }
        float sum = 0f;
        long ob = (long)i * C + h * D;
        for (int d = 0; d < D; d++) o[ob + d] = 0f;
        for (int j = 0; j <= i; j++)
        {
            long kb = (long)j * 3 * C + C + h * D, vb = kb + C;
            float s = 0f;
            for (int d = 0; d < D; d++) s += qkv[qb + d] * qkv[kb + d];
            float p = MathF.Exp(s * scale - mx);
            sum += p;
            for (int d = 0; d < D; d++) o[ob + d] += p * qkv[vb + d];
        }
        float inv = 1f / sum;
        for (int d = 0; d < D; d++) o[ob + d] *= inv;
    }
    public void CausalAttention(FV qkv, FV o, int T, int H, int D)
    {
        _causal ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int, float>(CausalImpl);
        _causal(T * H, qkv, o, T, H, D, 1f / MathF.Sqrt(D));
    }

    // quick GELU (OpenAI CLIP): x * sigmoid(1.702 x), in place
    private Action<Index1D, FV>? _quickGelu;
    private static void QuickGeluImpl(Index1D i, FV x) { float v = x[i]; x[i] = v / (1f + MathF.Exp(-1.702f * v)); }
    public void QuickGeluInPlace(FV x, long count)
    {
        _quickGelu ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV>(QuickGeluImpl);
        _quickGelu((int)count, x);
    }

    // exact GELU in place (OpenCLIP)
    private Action<Index1D, FV>? _gelu;
    private static void GeluImpl(Index1D i, FV x) => x[i] = DiTKernels.Activate(x[i], 3);
    public void GeluInPlace(FV x, long count)
    {
        _gelu ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV>(GeluImpl);
        _gelu((int)count, x);
    }

    // ---------------------------------------------------------------------------------------------------
    // single-head attention as two GEMMs (the VAE mid block: one 512-wide head over every latent pixel, beyond the flash kernels' widths): K and V^T go to fp16
    // "weights", the scores of a chunk of queries are one GEMM, a row softmax, and P V is the second GEMM. The chunk bounds the score buffer.
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, HV, int, int, int>? _toHalfRows;
    private static void ToHalfRowsImpl(Index1D i, FV src, HV dst, int stride, int off, int C)
    {
        long r = i / C; int c = (int)(i % C);
        dst[i] = PrecisionConvert.ConvertFromSingle<global::ILGPU.Half>(src[r * stride + off + c]);
    }
    private Action<Index1D, FV, HV, int, int, int, int>? _toHalfT;
    private static void ToHalfTImpl(Index1D i, FV src, HV dst, int N, int stride, int off, int C)
    {
        // dst [C, N] = src[n, off + c]^T
        int n = (int)(i % N); long c = i / N;
        dst[i] = PrecisionConvert.ConvertFromSingle<global::ILGPU.Half>(src[(long)n * stride + off + c]);
    }
    private Action<Index1D, FV, int, float>? _softmaxRows;
    private static void SoftmaxRowsImpl(Index1D row, FV s, int N, float scale)
    {
        long b = (long)(int)row * N;
        float mx = float.NegativeInfinity;
        for (int j = 0; j < N; j++) mx = MathF.Max(mx, s[b + j]);
        float sum = 0f;
        for (int j = 0; j < N; j++) { float e = MathF.Exp((s[b + j] - mx) * scale); s[b + j] = e; sum += e; }
        float inv = 1f / sum;
        for (int j = 0; j < N; j++) s[b + j] *= inv;
    }
    private Action<Index1D, FV, FV, int, int>? _copyRows;
    private static void CopyRowsImpl(Index1D i, FV src, FV dst, int stride, int C)
    {
        long r = i / C; int c = (int)(i % C);
        dst[i] = src[r * stride + c];
    }

    /// <summary>
    /// o [N, C] = softmax(q k^T / sqrt(C)) v for one head of width C, q / k / v read from qkv [N, 3C]. <paramref name="kHalf"/> ([N, C]) and <paramref name="vT"/> ([C, N]) are fp16 scratch,
    /// <paramref name="q"/> [chunk, C] and <paramref name="scores"/> [chunk, N] fp32 scratch; queries go in chunks of <paramref name="chunk"/>.
    /// </summary>
    public void SingleHeadAttention(FV qkv, FV o, int N, int C, HV kHalf, HV vT, FV q, FV scores, int chunk)
    {
        _toHalfRows ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, HV, int, int, int>(ToHalfRowsImpl);
        _toHalfT ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, HV, int, int, int, int>(ToHalfTImpl);
        _softmaxRows ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, int, float>(SoftmaxRowsImpl);
        _copyRows ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int>(CopyRowsImpl);
        _toHalfRows((int)((long)N * C), qkv, kHalf, 3 * C, C, C);
        _toHalfT((int)((long)N * C), qkv, vT, N, 3 * C, 2 * C, C);
        float scale = 1f / MathF.Sqrt(C);
        for (int r0 = 0; r0 < N; r0 += chunk)
        {
            int rows = Math.Min(chunk, N - r0);
            var qc = q.SubView(0, (long)rows * C);
            _copyRows((int)((long)rows * C), qkv.SubView((long)r0 * 3 * C, (long)rows * 3 * C), qc, 3 * C, C);
            var sc = scores.SubView(0, (long)rows * N);
            Linear(qc, kHalf, null, sc, rows, C, N);
            _softmaxRows(rows, sc, N, scale);
            Linear(sc, vT, null, o.SubView((long)r0 * C, (long)rows * C), rows, N, C);
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // host helpers
    // ---------------------------------------------------------------------------------------------------

    /// <summary>
    /// The UNet's timestep embedding as diffusers computes it (flip_sin_to_cos, downscale_freq_shift 0): [cos(t f_j), sin(t f_j)], f_j = exp(-ln(10000) j / half), with the
    /// frequencies and the argument rounded to float32 like torch (at t near 1000 a double computation moves the phase by ~1e-4).
    /// </summary>
    public static float[] TimestepEmbedding(double t, int dim)
    {
        int half = dim / 2;
        var e = new float[dim];
        float tf = (float)t;
        for (int j = 0; j < half; j++)
        {
            float exponent = (float)(-Math.Log(10000.0)) * j / half;
            float freq = MathF.Exp(exponent);
            float a = tf * freq;
            e[j] = MathF.Cos(a);
            e[half + j] = MathF.Sin(a);
        }
        return e;
    }
}
