using ILGPU;
using ILGPU.Algorithms;
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
    // up > 1: the convolution reads a nearest-upsampled image of the OUTPUT's size (outH, outW), source row floor(y * inH / outH) computed in fp32 as torch's
    // F.interpolate(mode="nearest", size=...) does: 2x for the usual blocks (y / 2 exactly), any size for ComfyUI's Upsample(output_shape = the next skip's shape).
    private static void TableImpl(Index1D idx, IV nbr, int inH, int inW, int outW, int k, int stride, int pad, int up, int outH)
    {
        int taps = k * k;
        int t = idx % taps, m = idx / taps;
        int ox = m % outW, oy = m / outW;
        int ky = t / k, kx = t - ky * k;
        int iy = oy * stride - pad + ky, ix = ox * stride - pad + kx;
        int H = up > 1 ? outH : inH, W = up > 1 ? outW : inW;
        bool inside = iy >= 0 && iy < H && ix >= 0 && ix < W;
        int sy = up > 1 ? XMath.Min((int)XMath.Floor(iy * ((float)inH / H)), inH - 1) : iy;
        int sx = up > 1 ? XMath.Min((int)XMath.Floor(ix * ((float)inW / W)), inW - 1) : ix;
        nbr[m * 27 + t] = inside ? sy * inW + sx : -1;
    }

    /// <summary>Fill a neighbour table for a k x k convolution (k = 1 or 3): output size (outH, outW) for an input of (inH, inW).</summary>
    public void BuildTable(IV nbr, int inH, int inW, int outH, int outW, int k, int stride, int pad, int up)
    {
        _table ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, int, int, int, int, int, int, int, int>(TableImpl);
        _table(outH * outW * k * k, nbr, inH, inW, outW, k, stride, pad, up, outH);
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

    // dst [rows, C] = src [rows, W] columns [col, col + C) (PAG: the value block of a fused q|k|v)
    private Action<Index1D, FV, FV, int, int, int>? _copyColsFrom;
    private static void CopyColumnsFromImpl(Index1D i, FV src, FV dst, int C, int W, int col)
    {
        long r = i / C; int c = (int)(i % C);
        dst[i] = src[r * W + col + c];
    }
    public void CopyColumnsFrom(FV src, FV dst, int rows, int C, int W, int col)
    {
        _copyColsFrom ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int>(CopyColumnsFromImpl);
        _copyColsFrom((int)((long)rows * C), src, dst, C, W, col);
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

    // ---------------------------------------------------------------------------------------------------
    // Model merge: w = w * a + o * b on fp16 weights, computed in fp32 (ComfyUI's add_patches: strength_model, strength_patch). Launched in chunks of at most
    // BlendChunk elements so no launch grows with the tensor (SDXL's token table is 63 M).
    // ---------------------------------------------------------------------------------------------------
    private const int BlendChunk = 1 << 24;
    private Action<Index1D, HV, HV, float, float>? _blend;
    private static void BlendImpl(Index1D i, HV w, HV o, float a, float b)
        => w[i] = PrecisionConvert.ConvertFromSingle<global::ILGPU.Half>(PrecisionConvert.ConvertToSingle(w[i]) * a + PrecisionConvert.ConvertToSingle(o[i]) * b);
    public void Blend(HV w, HV o, float a, float b)
    {
        if (w.Length != o.Length) throw new ArgumentException($"blend of {w.Length} and {o.Length} elements");
        _blend ??= _acc.LoadAutoGroupedStreamKernel<Index1D, HV, HV, float, float>(BlendImpl);
        // OpenCL (SpawnDev.ILGPU 5.4.13-local.1): a launch on a SubView at offset >= 2^24 elements wrote outside its range (garbage in other buffers); one launch
        // was exact. Reported to SpawnDev.ILGPU; gated by Sd21_MergeBlocksAndClip_MatchComfyFormulas (SD 2.1's 50.6 M-element token table takes 3 chunks).
        for (long at = 0; at < w.Length; at += BlendChunk)
        {
            int n = (int)Math.Min(BlendChunk, w.Length - at);
            _blend(n, w.SubView(at, n), o.SubView(at, n), a, b);
        }
    }

    // channel-first src [C, rows] into columns [colOff, colOff + C) of channel-last dst [rows, dstC]: the inpainting UNet's input is the latent and the
    // concatenated mask + masked-image latent side by side, assembled here with no host copy
    private Action<Index1D, FV, FV, int, int, int, int>? _toLastCols;
    private static void ToLastColsImpl(Index1D i, FV src, FV dst, int rows, int C, int dstC, int colOff)
    {
        int p = i % rows, c = i / rows;
        dst[(long)p * dstC + colOff + c] = src[i];
    }
    public void ChannelFirstToLastColumns(FV src, FV dst, int rows, int C, int dstC, int colOff)
    {
        _toLastCols ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int, int>(ToLastColsImpl);
        _toLastCols((int)((long)rows * C), src, dst, rows, C, dstC, colOff);
    }

    // ---------------------------------------------------------------------------------------------------
    // FreeU (ComfyUI's FreeU / FreeU_V2) on channel-last [rows = H * W, C] activations.
    // Backbone: the first `cols` channels of h times s (V1), or times (s - 1) * normMean[p] + 1 (V2: normMean = the per-pixel channel mean of h, min-max normalised
    // over the pixels). Skip: ComfyUI's Fourier_filter(x, threshold = 1, scale): fftn over (H, W), fftshift, the 2 x 2 window [c - 1, c + 1) around the centre times
    // scale, inverse. That window holds the frequencies {-1, 0} x {-1, 0}, so the filter is x + (scale - 1) / (H W) * sum over those (k, l) of
    // Re(X[k, l] e^{2 pi i (k m / H + l n / W)}): four DFT coefficients per channel and an elementwise correction, exact without an FFT. A side of 1 has one frequency.
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, int, int, float>? _scaleCols;
    private static void ScaleColsImpl(Index1D i, FV x, int C, int cols, float s)
    {
        int ii = i; long p = ii / cols; int c = ii % cols;
        x[p * C + c] = x[p * C + c] * s;
    }
    public void FreeUBackbone(FV x, int rows, int C, int cols, float s)
    {
        _scaleCols ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, int, int, float>(ScaleColsImpl);
        _scaleCols((int)((long)rows * cols), x, C, cols, s);
    }

    private Action<Index1D, FV, FV, int>? _pixelMean;
    private static void PixelMeanImpl(Index1D p, FV x, FV mean, int C)
    {
        float acc = 0f;
        for (int c = 0; c < C; c++) acc += x[(long)(int)p * C + c];
        mean[p] = acc / C;
    }
    private Action<Index1D, FV, FV, int>? _minMax;
    private static void MinMaxImpl(Index1D i, FV mean, FV mm, int rows)
    {
        float lo = mean[0], hi = mean[0];
        for (int p = 1; p < rows; p++) { float v = mean[p]; lo = v < lo ? v : lo; hi = v > hi ? v : hi; }
        mm[0] = lo; mm[1] = hi;
    }
    private Action<Index1D, FV, FV, FV, int, int, float>? _scaleColsV2;
    private static void ScaleColsV2Impl(Index1D i, FV x, FV mean, FV mm, int C, int cols, float s)
    {
        int ii = i; long p = ii / cols; int c = ii % cols;
        float m = (mean[p] - mm[0]) / (mm[1] - mm[0]);
        x[p * C + c] = x[p * C + c] * ((s - 1f) * m + 1f);
    }
    /// <summary>FreeU_V2's backbone scaling. <paramref name="scratch"/> holds rows + 2 floats.</summary>
    public void FreeUBackboneV2(FV x, int rows, int C, int cols, float s, FV scratch)
    {
        _pixelMean ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int>(PixelMeanImpl);
        _minMax ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int>(MinMaxImpl);
        _scaleColsV2 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, float>(ScaleColsV2Impl);
        var mean = scratch.SubView(0, rows); var mm = scratch.SubView(rows, 2);
        _pixelMean(rows, x, mean, C);
        _minMax(1, mean, mm, rows);
        _scaleColsV2((int)((long)rows * cols), x, mean, mm, C, cols, s);
    }

    // coefficient f = (kIdx, lIdx) with k = -kIdx, l = -lIdx: X[c, f] = sum_p x[p, c] e^{-2 pi i (k m / H + l n / W)} -> coef[(c * 4 + f) * 2 + {re, im}]
    private Action<Index1D, FV, FV, int, int, int>? _fourierCoef;
    private static void FourierCoefImpl(Index1D i, FV x, FV coef, int H, int W, int C)
    {
        int ii = i, c = ii / 4, f = ii % 4, k = -(f >> 1), l = -(f & 1);
        float re = 0f, im = 0f;
        for (int m = 0; m < H; m++)
            for (int n = 0; n < W; n++)
            {
                float th = -6.28318530718f * ((float)(k * m) / H + (float)(l * n) / W);
                float v = x[((long)m * W + n) * C + c];
                re += v * XMath.Cos(th); im += v * XMath.Sin(th);
            }
        coef[ii * 2] = re; coef[ii * 2 + 1] = im;
    }
    private Action<Index1D, FV, FV, int, int, int, float>? _fourierApply;
    private static void FourierApplyImpl(Index1D i, FV x, FV coef, int H, int W, int C, float s)
    {
        int ii = i, c = ii % C; long p = ii / C; int m = (int)(p / W), n = (int)(p % W);
        float add = 0f;
        for (int f = 0; f < 4; f++)
        {
            int k = -(f >> 1), l = -(f & 1);
            if ((k != 0 && H == 1) || (l != 0 && W == 1)) continue;      // a side of 1 has only frequency 0 (the window holds it once)
            float th = 6.28318530718f * ((float)(k * m) / H + (float)(l * n) / W);
            float re = coef[(c * 4 + f) * 2], im = coef[(c * 4 + f) * 2 + 1];
            add += re * XMath.Cos(th) - im * XMath.Sin(th);
        }
        x[i] = x[i] + (s - 1f) / (H * W) * add;
    }
    /// <summary>ComfyUI's Fourier_filter(x, threshold 1, scale) in place on channel-last [H * W, C]. <paramref name="scratch"/> holds C * 8 floats.</summary>
    public void FreeUFourier(FV x, int H, int W, int C, float scale, FV scratch)
    {
        _fourierCoef ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int>(FourierCoefImpl);
        _fourierApply ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int, float>(FourierApplyImpl);
        var coef = scratch.SubView(0, (long)C * 8);
        _fourierCoef(C * 4, x, coef, H, W, C);
        _fourierApply((int)((long)H * W * C), x, coef, H, W, C, scale);
    }

    // ---------------------------------------------------------------------------------------------------
    // Resize channel-last [inH * inW, C] -> [outH * outW, C] as torch's F.interpolate(size=...) with align_corners=False, no antialias (ComfyUI's common_upscale):
    // mode 0 bicubic (A = -0.75, border-clamped taps), 1 bilinear, 2 nearest-exact. Source coordinate (dst + 0.5) * in / out - 0.5 (bilinear clamps it at 0).
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, FV, int, int, int, int, int, int>? _resize;
    private static float Cubic1(float x) => ((1.25f * x - 2.25f) * x) * x + 1f;                     // |x| <= 1, A = -0.75
    private static float Cubic2(float x) => ((-0.75f * x + 3.75f) * x - 6f) * x + 3f;               // 1 < |x| < 2
    private static void ResizeImpl(Index1D i, FV src, FV dst, int inH, int inW, int outH, int outW, int C, int mode)
    {
        int ii = i, c = ii % C, p = ii / C, ox = p % outW, oy = p / outW;
        float sh = (float)inH / outH, sw = (float)inW / outW;
        float v;
        if (mode == 2)
        {
            int sy = XMath.Min((int)XMath.Floor((oy + 0.5f) * sh), inH - 1), sx = XMath.Min((int)XMath.Floor((ox + 0.5f) * sw), inW - 1);
            v = src[((long)sy * inW + sx) * C + c];
        }
        else if (mode == 1)
        {
            float ry = XMath.Max((oy + 0.5f) * sh - 0.5f, 0f), rx = XMath.Max((ox + 0.5f) * sw - 0.5f, 0f);
            int y0 = (int)ry, x0 = (int)rx;
            int y1 = XMath.Min(y0 + 1, inH - 1), x1 = XMath.Min(x0 + 1, inW - 1);
            float ty = ry - y0, tx = rx - x0;
            v = (1 - ty) * ((1 - tx) * src[((long)y0 * inW + x0) * C + c] + tx * src[((long)y0 * inW + x1) * C + c])
              + ty * ((1 - tx) * src[((long)y1 * inW + x0) * C + c] + tx * src[((long)y1 * inW + x1) * C + c]);
        }
        else
        {
            float ry = (oy + 0.5f) * sh - 0.5f, rx = (ox + 0.5f) * sw - 0.5f;
            int fy = (int)XMath.Floor(ry), fx = (int)XMath.Floor(rx);
            float ty = ry - fy, tx = rx - fx;
            float wy0 = Cubic2(ty + 1f), wy1 = Cubic1(ty), wy2 = Cubic1(1f - ty), wy3 = Cubic2(2f - ty);
            float wx0 = Cubic2(tx + 1f), wx1 = Cubic1(tx), wx2 = Cubic1(1f - tx), wx3 = Cubic2(2f - tx);
            v = 0f;
            for (int a = 0; a < 4; a++)
            {
                int yy = XMath.Clamp(fy - 1 + a, 0, inH - 1);
                float wy = a == 0 ? wy0 : a == 1 ? wy1 : a == 2 ? wy2 : wy3;
                long row = (long)yy * inW;
                int xa = XMath.Clamp(fx - 1, 0, inW - 1), xb = XMath.Clamp(fx, 0, inW - 1), xc = XMath.Clamp(fx + 1, 0, inW - 1), xd = XMath.Clamp(fx + 2, 0, inW - 1);
                v += wy * (wx0 * src[(row + xa) * C + c] + wx1 * src[(row + xb) * C + c] + wx2 * src[(row + xc) * C + c] + wx3 * src[(row + xd) * C + c]);
            }
        }
        dst[i] = v;
    }
    public void Resize(FV src, FV dst, int inH, int inW, int outH, int outW, int C, int mode)
    {
        _resize ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int, int, int, int>(ResizeImpl);
        _resize((int)((long)outH * outW * C), src, dst, inH, inW, outH, outW, C, mode);
    }

    // fp16 -> fp32 (reading a small weight back to the host)
    private Action<Index1D, HV, FV>? _h2f;
    private static void HalfToFloatImpl(Index1D i, HV src, FV dst) => dst[i] = PrecisionConvert.ConvertToSingle(src[i]);
    public void HalfToFloat(HV src, FV dst)
    {
        _h2f ??= _acc.LoadAutoGroupedStreamKernel<Index1D, HV, FV>(HalfToFloatImpl);
        _h2f((int)src.Length, src, dst);
    }

    // ReLU in place; x = relu(x + y); TAESD's input clamp tanh(x * scale / 3) * 3; x = a * x + b
    private Action<Index1D, FV>? _relu;
    private static void ReluImpl(Index1D i, FV x) => x[i] = MathF.Max(0f, x[i]);
    public void ReluInPlace(FV x, long count)
    {
        _relu ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV>(ReluImpl);
        _relu((int)count, x);
    }
    private Action<Index1D, FV, FV>? _addRelu;
    private static void AddReluImpl(Index1D i, FV x, FV y) => x[i] = MathF.Max(0f, x[i] + y[i]);
    public void AddReluInPlace(FV x, FV y, long count)
    {
        _addRelu ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(AddReluImpl);
        _addRelu((int)count, x, y);
    }
    private Action<Index1D, FV, float>? _taesdIn;
    private static void TaesdInImpl(Index1D i, FV x, float scale) => x[i] = MathF.Tanh(x[i] * scale / 3f) * 3f;
    public void TaesdIn(FV x, long count, float scale)
    {
        _taesdIn ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, float>(TaesdInImpl);
        _taesdIn((int)count, x, scale);
    }
    private Action<Index1D, FV, float, float>? _affine;
    private static void AffineImpl(Index1D i, FV x, float a, float b) => x[i] = a * x[i] + b;
    public void Affine(FV x, long count, float a, float b)
    {
        _affine ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, float, float>(AffineImpl);
        _affine((int)count, x, a, b);
    }

    // x += a * y (the ControlNet residuals, times their strength)
    private Action<Index1D, FV, FV, float>? _axpy;
    private static void AxpyImpl(Index1D i, FV x, FV y, float a) => x[i] = x[i] + a * y[i];
    public void AxpyInPlace(FV x, FV y, float a, long count)
    {
        _axpy ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, float>(AxpyImpl);
        _axpy((int)count, x, y, a);
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
