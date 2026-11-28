using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Anima;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;
using BV = ArrayView1D<BFloat16, Stride1D.Dense>;

/// <summary>
/// The kernels Anima needs beyond <see cref="Trellis2.DiTKernels"/> / <see cref="Wan.WanKernels"/> (tools/anima-ref/ANIMA-SPEC.md):
/// the Cosmos patch layout (feature order (c, p_t, p_h, p_w) in, (p_h, p_w, p_t, c) out), rotate-half RoPE (dims j and j + D/2, not the adjacent
/// pairs of <c>WanKernels.RopeApply</c>; interleaved pairing costs 0.39 relRMS on block 0), the timestep sinusoid (cos first), and a bf16 row gather.
/// </summary>
public sealed class AnimaKernels
{
    private readonly Accelerator _acc;
    /// <summary>Kernels on <paramref name="accelerator"/> (compiled on first use).</summary>
    public AnimaKernels(Accelerator accelerator) => _acc = accelerator;

    /// <summary>Latent channels of the DiT, the padding-mask channel appended after them (all zeros for a full image), the patch.</summary>
    public const int LatentChannels = 16, PatchFeatures = (LatentChannels + 1) * 4, OutFeatures = LatentChannels * 4;

    // feat[n, f], f = c * 4 + iph * 2 + ipw (c slowest; c == 16 is the padding mask = 0); x = [16, T, H, W]
    private Action<Index1D, FV, FV, int, int, int>? _patchify;
    private static void PatchifyImpl(Index1D i, FV x, FV feat, int T, int H, int W)
    {
        int f = i % PatchFeatures, n = i / PatchFeatures;
        int c = f / 4, r = f % 4, iph = r / 2, ipw = r % 2;
        int hp = H / 2, wpN = W / 2;
        int t = n / (hp * wpN), rem = n % (hp * wpN), y = rem / wpN, xx = rem % wpN;
        feat[i] = c == LatentChannels ? 0f : x[(((long)c * T + t) * H + 2 * y + iph) * W + 2 * xx + ipw];
    }
    /// <summary>[16, T, H, W] latent -> [N, 68] patch features, N = T * H/2 * W/2 (token (t, hp, wp) at (t * H/2 + hp) * W/2 + wp).</summary>
    public void Patchify(FV latent, FV feat, int T, int H, int W)
    {
        _patchify ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int>(PatchifyImpl);
        _patchify(T * (H / 2) * (W / 2) * PatchFeatures, latent, feat, T, H, W);
    }

    // v[c, t, 2hp + iph, 2wp + ipw] = y[L, (iph * 2 + ipw) * 16 + c]
    private Action<Index1D, FV, FV, int, int, int>? _unpatchify;
    private static void UnpatchifyImpl(Index1D i, FV y, FV v, int T, int H, int W)
    {
        int xx = i % W, rest = i / W, yy = rest % H; rest /= H;
        int t = rest % T, c = rest / T;
        int hp = yy / 2, iph = yy % 2, wp = xx / 2, ipw = xx % 2;
        long token = ((long)t * (H / 2) + hp) * (W / 2) + wp;
        v[i] = y[token * OutFeatures + (iph * 2 + ipw) * LatentChannels + c];
    }
    /// <summary>[N, 64] final-layer output -> [16, T, H, W] (the order (p_h, p_w, p_t, c), c fastest: NOT the inverse of the patch order).</summary>
    public void Unpatchify(FV y, FV v, int T, int H, int W)
    {
        _unpatchify ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int>(UnpatchifyImpl);
        _unpatchify(LatentChannels * T * H * W, y, v, T, H, W);
    }

    // Rotate-half RoPE, IN PLACE: for every (token n, head h, j < D/2): (x[j], x[j + D/2]) -> (x[j] c - x[j + D/2] s, x[j + D/2] c + x[j] s),
    // the vector at buf[n * stride + off + h * D]; phases = [N, D/2, 2] (cos, sin) per token.
    private Action<Index1D, FV, FV, int, int, int, int>? _ropeHalf;
    private static void RopeHalfImpl(Index1D i, FV buf, FV phases, int stride, int off, int H, int D)
    {
        int half = D / 2;
        int j = i % half, rest = i / half, h = rest % H, n = rest / H;
        long b = (long)n * stride + off + (long)h * D;
        long p = ((long)n * half + j) * 2;
        float c = phases[p], s = phases[p + 1];
        float x0 = buf[b + j], x1 = buf[b + j + half];
        buf[b + j] = x0 * c - x1 * s;
        buf[b + j + half] = x1 * c + x0 * s;
    }
    /// <summary>Rotate-half RoPE on <paramref name="N"/> tokens x <paramref name="H"/> heads of width <paramref name="D"/>, in place.</summary>
    public void RopeHalf(FV buf, FV phases, int N, int stride, int off, int H, int D)
    {
        _ropeHalf ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int, int>(RopeHalfImpl);
        _ropeHalf(N * H * (D / 2), buf, phases, stride, off, H, D);
    }

    // t_proj[k] = cos(sigma * f_k), t_proj[1024 + k] = sin(sigma * f_k), f_k = exp(-ln(10000) * k / 1024)
    private Action<Index1D, FV, float, int>? _sinusoid;
    private static void SinusoidImpl(Index1D i, FV o, float sigma, int half)
    {
        int k = i % half;
        float f = MathF.Exp(-9.210340371976184f * k / half);            // ln(10000)
        float a = sigma * f;
        o[i] = i < half ? MathF.Cos(a) : MathF.Sin(a);
    }
    /// <summary>The timestep sinusoid of width <paramref name="dim"/> (cos half first) for timestep = sigma (NOT sigma * 1000: 1.27 relRMS).</summary>
    public void TimestepSinusoid(FV o, float sigma, int dim)
    {
        _sinusoid ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, float, int>(SinusoidImpl);
        _sinusoid(dim, o, sigma, dim / 2);
    }

    // dst[i] = i < count ? src[i] : 0     (one dispatch, so the zero fill and the copy cannot reorder)
    private Action<Index1D, FV, FV, long>? _copyPadded;
    private static void CopyPaddedImpl(Index1D i, FV src, FV dst, long count) => dst[i] = i < count ? src[i] : 0f;
    /// <summary>The first <paramref name="count"/> values of <paramref name="src"/> into <paramref name="dst"/>, the rest of <paramref name="dst"/> zeroed.</summary>
    public void CopyPadded(FV src, FV dst, long count)
    {
        _copyPadded ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, long>(CopyPaddedImpl);
        _copyPadded((int)dst.Length, src, dst, count);
    }

    // o[m, c] = table[ids[m], c] (bf16 -> fp32)
    private Action<Index1D, BV, IV, FV, int>? _gather;
    private static void GatherImpl(Index1D i, BV table, IV ids, FV o, int C)
    {
        int m = i / C, c = i % C;
        o[i] = PrecisionConvert.ConvertToSingle(table[(long)ids[m] * C + c]);
    }
    /// <summary>Rows <paramref name="ids"/> of a bf16 [rows, C] table, as fp32 [M, C].</summary>
    public void GatherRows(BV table, IV ids, FV o, int M, int C)
    {
        _gather ??= _acc.LoadAutoGroupedStreamKernel<Index1D, BV, IV, FV, int>(GatherImpl);
        _gather(M * C, table, ids, o, C);
    }

    // ---------------------------------------------------------------------------------------------------
    // ControlNet-LLLite (AnimaLLLite): the conditioning trunk's convolutions and norms, and each module's small elementwise steps
    // ---------------------------------------------------------------------------------------------------

    // y[o, oy, ox] = bias[o] + sum_{c, ky, kx} w[o, c, ky, kx] * x[c, oy * s - p + ky, ox * s - p + kx]   (zero padding; channel-first)
    private Action<Index1D, FV, BV, BV, FV, int, int, int, int, int, int, int, int>? _conv;
    private static void ConvImpl(Index1D i, FV x, BV w, BV bias, FV y, int cin, int H, int W, int Ho, int Wo, int k, int s, int p)
    {
        int ox = i % Wo, rest = i / Wo, oy = rest % Ho, o = rest / Ho;
        float acc = PrecisionConvert.ConvertToSingle(bias[o]);
        long wb = (long)o * cin * k * k;
        for (int c = 0; c < cin; c++)
        {
            long xb = (long)c * H * W, wc = wb + (long)c * k * k;
            for (int ky = 0; ky < k; ky++)
            {
                int iy = oy * s - p + ky;
                if (iy < 0 || iy >= H) continue;
                for (int kx = 0; kx < k; kx++)
                {
                    int ix = ox * s - p + kx;
                    if (ix < 0 || ix >= W) continue;
                    acc = float.MultiplyAddEstimate(PrecisionConvert.ConvertToSingle(w[wc + ky * k + kx]), x[xb + (long)iy * W + ix], acc);
                }
            }
        }
        y[i] = acc;
    }
    /// <summary>
    /// Direct 2-D convolution, channel-first ([cin, H, W] -> [cout, Ho, Wo]), square kernel <paramref name="k"/>, stride, zero padding, bf16 weights
    /// [cout, cin, k, k] (torch layout) and bias. For the LLLite conditioning trunk (runs once per picture). Returns (Ho, Wo).
    /// </summary>
    public (int Ho, int Wo) Conv2d(FV x, BV w, BV bias, FV y, int cin, int H, int W, int cout, int k, int stride, int pad)
    {
        int Ho = (H + 2 * pad - k) / stride + 1, Wo = (W + 2 * pad - k) / stride + 1;
        _conv ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, BV, BV, FV, int, int, int, int, int, int, int, int>(ConvImpl);
        _conv(cout * Ho * Wo, x, w, bias, y, cin, H, W, Ho, Wo, k, stride, pad);
        return (Ho, Wo);
    }

    // x[c, p] = x[c, p] * gamma[c] + beta[c], then SiLU when silu != 0, IN PLACE (channel-first, HW values per channel)
    private Action<Index1D, FV, BV, BV, int, int>? _chanAffine;
    private static void ChannelAffineImpl(Index1D i, FV x, BV gamma, BV beta, int HW, int silu)
    {
        int c = i / HW;
        float v = x[i] * PrecisionConvert.ConvertToSingle(gamma[c]) + PrecisionConvert.ConvertToSingle(beta[c]);
        x[i] = silu != 0 ? v / (1f + MathF.Exp(-v)) : v;
    }
    /// <summary>Per-channel affine (bf16 gamma, beta) and optional SiLU on a channel-first [C, HW] tensor, in place.</summary>
    public void ChannelAffine(FV x, BV gamma, BV beta, int C, int HW, bool silu)
    {
        _chanAffine ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, BV, BV, int, int>(ChannelAffineImpl);
        _chanAffine(C * HW, x, gamma, beta, HW, silu ? 1 : 0);
    }

    // dst[r, c] = src[r, c] + v[c]
    private Action<Index1D, FV, BV, FV, int>? _addRow;
    private static void AddRowImpl(Index1D i, FV src, BV v, FV dst, int C) => dst[i] = src[i] + PrecisionConvert.ConvertToSingle(v[i % C]);
    /// <summary>dst = src + v broadcast over the rows ([rows, C] + bf16 [C]).</summary>
    public void AddRow(FV src, BV v, FV dst, int rows, int C)
    {
        _addRow ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, BV, FV, int>(AddRowImpl);
        _addRow(rows * C, src, v, dst, C);
    }

    // o[r, :ca] = a[r], o[r, ca:] = b[r]
    private Action<Index1D, FV, FV, FV, int, int>? _concat;
    private static void ConcatImpl(Index1D i, FV a, FV b, FV o, int ca, int cb)
    {
        int w = ca + cb, r = i / w, c = i % w;
        o[i] = c < ca ? a[(long)r * ca + c] : b[(long)r * cb + c - ca];
    }
    /// <summary>Row-wise concatenation [rows, ca] ++ [rows, cb] -> [rows, ca + cb].</summary>
    public void Concat(FV a, FV b, FV o, int rows, int ca, int cb)
    {
        _concat ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int>(ConcatImpl);
        _concat(rows * (ca + cb), a, b, o, ca, cb);
    }

    // m[r, j] = SiLU(m[r, j] * (1 + gb[r, j]) + gb[r, D + j]), IN PLACE
    private Action<Index1D, FV, FV, int>? _filmSilu;
    private static void FilmSiluImpl(Index1D i, FV m, FV gb, int D)
    {
        int r = i / D, j = i % D;
        long g = (long)r * 2 * D + j;
        float v = float.MultiplyAddEstimate(m[i], 1f + gb[g], gb[g + D]);
        m[i] = v / (1f + MathF.Exp(-v));
    }
    /// <summary>FiLM then SiLU, in place: m = SiLU(m * (1 + gamma) + beta), (gamma, beta) = the two halves of each row of gb [rows, 2D].</summary>
    public void FilmSilu(FV m, FV gb, int rows, int D)
    {
        _filmSilu ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int>(FilmSiluImpl);
        _filmSilu(rows * D, m, gb, D);
    }

    // o = x + s * y
    private Action<Index1D, FV, FV, FV, float>? _addScaled;
    private static void AddScaledImpl(Index1D i, FV x, FV y, FV o, float s) => o[i] = float.MultiplyAddEstimate(s, y[i], x[i]);
    /// <summary>o = x + s * y (three separate buffers).</summary>
    public void AddScaled(FV x, FV y, FV o, float s, int count)
    {
        _addScaled ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, float>(AddScaledImpl);
        _addScaled(count, x, y, o, s);
    }

    // o = [rgb * 2 - 1 (zeroed where mask >= 0.5 when maskedInput), mask >= 0.5 ? 1 : -1]   (rgb [3, H, W] in [0, 1], mask [H, W])
    private Action<Index1D, FV, FV, FV, int, int>? _inpaintPic;
    private static void InpaintPictureImpl(Index1D i, FV rgb, FV mask, FV o, int HW, int maskedInput)
    {
        int c = i / HW, p = i % HW;
        bool hole = mask[p] >= 0.5f;
        o[i] = c == 3 ? (hole ? 1f : -1f) : (hole && maskedInput != 0 ? 0f : rgb[i] * 2f - 1f);
    }
    /// <summary>
    /// The 4-channel inpainting picture of an LLLite (kohya's _build_inpaint_cond_image): rgb [3, H, W] in [0, 1] (ComfyUI's IMAGE range) to [-1, 1],
    /// zeroed (= mid grey) where the mask marks the part to repaint when <paramref name="maskedInput"/>, plus the binarised mask as a 4th channel in {-1, 1}
    /// (1 = repaint). MEASURED against ComfyUI (tools/anima-ref/check_lllite.py): black instead of 0 in the hole costs 3.4e-2 on block 0, a {0, 1} mask
    /// channel 1.2e-1, an inverted mask 1.2e-1.
    /// </summary>
    public void InpaintPicture(FV rgb, FV mask, FV o, int H, int W, bool maskedInput = true)
    {
        _inpaintPic ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int>(InpaintPictureImpl);
        _inpaintPic(4 * H * W, rgb, mask, o, H * W, maskedInput ? 1 : 0);
    }

    /// <summary>
    /// The DiT's rotate-half phase table [N, 64, 2] for a T x H/2 x W/2 token grid (head dim 128): pairs 0..21 temporal (t * theta_t^(-2j/44)),
    /// 22..42 rows, 43..63 columns (index * theta_hw^(-2k/42), theta_hw = 10000 * 4^(42/40)). Computed in double, stored as float (the taps match it
    /// to 1.2e-7). theta_t is not observable with one frame; 10000 (rope scale 1) is used, as the reference does.
    /// </summary>
    public static float[] DitRopePhases(int T, int hp, int wp)
    {
        const int pairs = 64, tPairs = 22, hPairs = 21;
        double thetaHw = 10000.0 * Math.Pow(4.0, 42.0 / 40.0), thetaT = 10000.0;
        int n = T * hp * wp;
        var o = new float[(long)n * pairs * 2];
        for (int t = 0; t < T; t++)
            for (int y = 0; y < hp; y++)
                for (int x = 0; x < wp; x++)
                {
                    long b = ((long)(t * hp + y) * wp + x) * pairs * 2;
                    for (int j = 0; j < pairs; j++)
                    {
                        double a = j < tPairs ? t * Math.Pow(thetaT, -2.0 * j / 44.0)
                            : j < tPairs + hPairs ? y * Math.Pow(thetaHw, -2.0 * (j - tPairs) / 42.0)
                            : x * Math.Pow(thetaHw, -2.0 * (j - tPairs - hPairs) / 42.0);
                        o[b + 2 * j] = (float)Math.Cos(a);
                        o[b + 2 * j + 1] = (float)Math.Sin(a);
                    }
                }
        return o;
    }

    /// <summary>The adapter's rotate-half phase table [count, 32, 2] (head dim 64, base 10000, positions 0..count-1; HF Llama style).</summary>
    public static float[] AdapterRopePhases(int count)
    {
        const int pairs = 32;
        var o = new float[(long)count * pairs * 2];
        for (int p = 0; p < count; p++)
            for (int k = 0; k < pairs; k++)
            {
                double a = p * Math.Pow(10000.0, -2.0 * k / 64.0);
                o[((long)p * pairs + k) * 2] = (float)Math.Cos(a);
                o[((long)p * pairs + k) * 2 + 1] = (float)Math.Sin(a);
            }
        return o;
    }
}
