using ILGPU;
using ILGPU.Runtime;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;
using BV = ILGPU.Runtime.ArrayView1D<ILGPU.BFloat16, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.Ltx2;

/// <summary>
/// LTX-2 kernels the other models do not have (diffusers' LTX-2, Apache-2.0): the "split" rotary embedding with a frequency table per head, the RMS norm without a weight,
/// and the text connectors' register fill.
/// </summary>
public sealed class Ltx2Kernels
{
    private readonly Accelerator _acc;
    public Ltx2Kernels(Accelerator accelerator) => _acc = accelerator;

    private Action<Index1D, FV, FV, int, float>? _rms;
    private static void RmsImpl(Index1D row, FV x, FV o, int C, float eps)
    {
        long b = (long)(int)row * C;
        float ss = 0f;
        for (int c = 0; c < C; c++) { float v = x[b + c]; ss += v * v; }
        float inv = 1f / MathF.Sqrt(ss / C + eps);
        for (int c = 0; c < C; c++) o[b + c] = x[b + c] * inv;
    }
    /// <summary>o[r, :] = x[r, :] / sqrt(mean(x[r, :]^2) + eps)  (torch RMSNorm without elementwise affine; x and o may be the same view).</summary>
    public void RmsNormPlain(FV x, FV o, int rows, int C, float eps)
    {
        _rms ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, float>(RmsImpl);
        _rms(rows, x, o, C, eps);
    }

    private Action<Index1D, FV, FV, FV, int, int, int, int>? _rope;
    private static void RopeImpl(Index1D i, FV buf, FV cos, FV sin, int stride, int off, int H, int D)
    {
        int half = D / 2;
        int k = i % half, rest = i / half, h = rest % H, n = rest / H;
        long t = (long)n * H * half + h * half + k;
        float c = cos[t], s = sin[t];
        long b = (long)n * stride + off + h * D;
        float x0 = buf[b + k], x1 = buf[b + half + k];
        buf[b + k] = x0 * c - x1 * s;
        buf[b + half + k] = x1 * c + x0 * s;
    }
    /// <summary>
    /// In place "split" rotary embedding (diffusers' apply_split_rotary_emb): each D-wide head vector of rows [n, stride] from column off is two halves, and pair k of head h
    /// turns by <c>cos[n, h D/2 + k]</c>, <c>sin[..]</c>: tables [rows, H D / 2] (the frequency list cut per head).
    /// </summary>
    public void RopeSplit(FV buf, FV cos, FV sin, int rows, int stride, int off, int H, int D)
    {
        if (cos.Length < (long)rows * H * (D / 2) || sin.Length < (long)rows * H * (D / 2)) throw new ArgumentException($"tables need rows x {H * (D / 2)} values");
        _rope ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, int>(RopeImpl);
        _rope(rows * H * (D / 2), buf, cos, sin, stride, off, H, D);
    }

    private Action<Index1D, FV, BV, FV, int, int, int>? _fill;
    private static void FillImpl(Index1D i, FV input, BV registers, FV o, int n, int C, int R)
    {
        int p = i / C, c = i % C;
        o[i] = p < n ? input[(long)p * C + c] : PrecisionConvert.ConvertToSingle(registers[(long)(p % R) * C + c]);
    }
    /// <summary>o [T, C]: rows 0..n-1 the input's rows, row p &gt;= n the learnable register p mod R (registers [R, C]) — the connectors' padding (diffusers LTX2ConnectorTransformer1d).</summary>
    public void FillRegisters(FV input, BV registers, FV o, int n, int T, int C, int R)
    {
        _fill ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, BV, FV, int, int, int>(FillImpl);
        _fill((int)((long)T * C), input, registers, o, n, C, R);
    }

    // (the keyframe-guide self-attention bias runs in Trellis2.DiTKernels.BiasedAttention: the flash kernel with a two-group score bias)

    // ───────────────────────── the model call's packing and the sampler, on the device ─────────────────────────
    // packed [B, P]: per item the video latent [128, nv] row-major (nv = F H W tokens), then the audio latent [8, Ta, 16]; aOff = 128 nv

    private Action<Index1D, FV, FV, int, int>? _unpackV;
    private static void UnpackVideoImpl(Index1D i, FV packed, FV tok, int nv, int P)
    {
        int c = i % 128, r = i / 128, t = r % nv, b = r / nv;
        tok[i] = packed[(long)b * P + (long)c * nv + t];
    }
    private Action<Index1D, FV, FV, int, int, int>? _unpackA;
    private static void UnpackAudioImpl(Index1D i, FV packed, FV tok, int Ta, int P, int aOff)
    {
        int ch = i % 128, r = i / 128, f = r % Ta, b = r / Ta;
        int c = ch / 16, m = ch % 16;
        tok[i] = packed[(long)b * P + aOff + ((long)c * Ta + f) * 16 + m];
    }
    /// <summary>The packed latents [B, P] as the transformer's tokens: video [B nv, 128] (token t, feature = channel), audio [B Ta, 128] (token = frame, feature c 16 + bin).</summary>
    public void UnpackTokens(FV packed, FV video, FV audio, int B, int nv, int Ta)
    {
        int P = 128 * nv + 128 * Ta;
        _unpackV ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int>(UnpackVideoImpl);
        _unpackA ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int>(UnpackAudioImpl);
        _unpackV(B * nv * 128, packed, video, nv, P);
        _unpackA(B * Ta * 128, packed, audio, Ta, P, 128 * nv);
    }

    private Action<Index1D, FV, FV, FV, FV, int, int>? _denV;
    private static void DenoiseVideoImpl(Index1D i, FV packed, FV vel, FV sigma, FV den, int nv, int P)
    {
        int per = 128 * nv, b = i / per, r = i % per, c = r / nv, t = r % nv;
        long p = (long)b * P + r;
        den[p] = packed[p] - sigma[b] * vel[((long)b * nv + t) * 128 + c];
    }
    private Action<Index1D, FV, FV, FV, FV, int, int, int>? _denA;
    private static void DenoiseAudioImpl(Index1D i, FV packed, FV vel, FV sigma, FV den, int Ta, int P, int aOff)
    {
        int per = 128 * Ta, b = i / per, r = i % per, c = r / (Ta * 16), f = (r / 16) % Ta, m = r % 16;
        long p = (long)b * P + aOff + r;
        den[p] = packed[p] - sigma[b] * vel[((long)b * Ta + f) * 128 + c * 16 + m];
    }
    /// <summary>The denoised packed latents [B, P]: <c>x - sigma v</c> with the item's sigma (sigma [B]), from the transformer's velocities in token layout.</summary>
    public void DenoisedPacked(FV packed, FV videoVel, FV audioVel, FV sigma, FV den, int B, int nv, int Ta)
    {
        int P = 128 * nv + 128 * Ta;
        _denV ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, int, int>(DenoiseVideoImpl);
        _denA ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, int, int, int>(DenoiseAudioImpl);
        _denV(B * 128 * nv, packed, videoVel, sigma, den, nv, P);
        _denA(B * 128 * Ta, packed, audioVel, sigma, den, Ta, P, 128 * nv);
    }

    private Action<Index1D, FV, FV, FV, FV, int, int, int, int>? _blend;
    private static void BlendRowsImpl(Index1D i, FV x, FV clean, FV frameMask, FV o, int P, int nv, int hw, int videoLen)
    {
        int k = i % P;
        float m = k < videoLen ? frameMask[(k % nv) / hw] : 1f;       // the audio latent is free (mask 1)
        o[i] = m * x[k] + (1f - m) * clean[k];
    }
    /// <summary>
    /// The model input of a masked sampling step, into every row of o [rows, P]: <c>m x + (1 - m) clean</c> with m the latent frame's denoise mask (frameMask [F]; audio 1).
    /// MEASURED 2026-11-17 against ComfyUI's guided calls (strength 0.6: 5.7e-8): the sampler keeps its own state x and the model sees this blend.
    /// </summary>
    public void BlendRows(FV x, FV clean, FV frameMask, FV o, int rows, int P, int nv, int hw)
    {
        _blend ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, int, int, int, int>(BlendRowsImpl);
        _blend(rows * P, x, clean, frameMask, o, P, nv, hw, 128 * nv);
    }

    private Action<Index1D, FV, FV, int, int, float, float, float>? _euler;
    private static void EulerCfgImpl(Index1D k, FV x, FV den, int P, int B, float cfg, float invSigma, float dt)
    {
        float u = den[k], c = den[(long)(B - 1) * P + k];              // one row: c = u, so d = u
        float d = u + (c - u) * cfg;
        x[k] = x[k] + (x[k] - d) * invSigma * dt;
    }
    private Action<Index1D, FV, FV, FV, FV, int, int, float, float, float, int, int, int>? _eulerMasked;
    private static void EulerCfgMaskedImpl(Index1D k, FV x, FV den, FV clean, FV frameMask, int P, int B, float cfg, float invSigma, float dt, int nv, int hw, int videoLen)
    {
        float u = den[k], c = den[(long)(B - 1) * P + k];
        float d = u + (c - u) * cfg;
        float m = k < videoLen ? frameMask[(k % nv) / hw] : 1f;
        d = m * d + (1f - m) * clean[k];
        x[k] = x[k] + (x[k] - d) * invSigma * dt;
    }
    /// <summary>
    /// One Euler step of the sampler's state x [P] on the guided denoised latent: <c>D = uncond + cfg (cond - uncond)</c> (den [B, P], rows [uncond, cond]; one row: D = den),
    /// <c>x += (x - D) / sigma (sigma_next - sigma)</c> (invSigma = 1 / sigma, dt = sigma_next - sigma: divided on the host). With a mask (frameMask [F] not default):
    /// <c>D = m D + (1 - m) clean</c> first (MEASURED against ComfyUI's guided run: the state after a step 1.3e-7, the final latent 4.5e-8).
    /// </summary>
    public void EulerCfgStep(FV x, FV den, int P, int B, float cfg, float invSigma, float dt, FV clean = default, FV frameMask = default, int nv = 0, int hw = 0)
    {
        if (frameMask.Length == 0)
        {
            _euler ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, float, float, float>(EulerCfgImpl);
            _euler(P, x, den, P, B, cfg, invSigma, dt);
            return;
        }
        _eulerMasked ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, int, int, float, float, float, int, int, int>(EulerCfgMaskedImpl);
        _eulerMasked(P, x, den, clean, frameMask, P, B, cfg, invSigma, dt, nv, hw, 128 * nv);
    }

    // ───────────────────────── LoRA ─────────────────────────

    private Action<Index1D, FV, FV, float>? _axpy;
    private static void AxpyImpl(Index1D i, FV x, FV y, float a) => x[i] = float.MultiplyAddEstimate(a, y[i], x[i]);
    /// <summary>x += a y over count values (a LoRA's scaled update into a Linear's output).</summary>
    public void AxpyInPlace(FV x, FV y, float a, int count)
    {
        _axpy ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, float>(AxpyImpl);
        _axpy(count, x, y, a);
    }

    private Action<Index1D, FV, int>? _act;
    private static void ActImpl(Index1D i, FV x, int act) => x[i] = Trellis2.DiTKernels.Activate(x[i], act);
    /// <summary>In place activation (<see cref="Trellis2.DiTKernels.Act"/>): the one a fused Linear would apply, after a LoRA update was added.</summary>
    public void ActivateInPlace(FV x, int count, int act)
    {
        _act ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, int>(ActImpl);
        _act(count, x, act);
    }

    // ───────────────────────── video VAE decoder ─────────────────────────

    // the 3x3x3 neighbour table of the LTX-2 decoder's (non-causal) convolutions: REFLECT padding in space (index -1 -> 1, H -> H - 2), REPLICATE in time, row stride 27
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, int, int, int>? _reflect;
    private static void ReflectTableImpl(Index1D m, ArrayView1D<int, Stride1D.Dense> nbr, int T, int H, int W)
    {
        int hw = H * W, t = m / hw, p = m % hw, h = p / W, w = p % W;
        for (int kk = 0; kk < 27; kk++)
        {
            int tt = t + kk / 9 - 1, hh = h + (kk / 3) % 3 - 1, ww = w + kk % 3 - 1;
            tt = tt < 0 ? 0 : (tt >= T ? T - 1 : tt);
            hh = hh < 0 ? -hh : (hh >= H ? 2 * H - 2 - hh : hh);
            ww = ww < 0 ? -ww : (ww >= W ? 2 * W - 2 - ww : ww);
            nbr[m * 27 + kk] = (tt * H + hh) * W + ww;
        }
    }
    /// <summary>The neighbour table [T H W, 27] for 3x3x3 convolutions with reflect padding in space and replicate padding in time (diffusers' LTX-2 decoder, causal = false). H, W &gt;= 2.</summary>
    public void ReflectTable(ArrayView1D<int, Stride1D.Dense> nbr, int T, int H, int W)
    {
        if (H < 2 || W < 2) throw new ArgumentException($"reflect padding needs at least 2 x 2 positions, got {H} x {W}");
        _reflect ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, int, int, int>(ReflectTableImpl);
        _reflect(T * H * W, nbr, T, H, W);
    }

    private Action<Index1D, BV, BV, int, int>? _gatherBf16;
    private static void GatherBf16Impl(Index1D i, BV src, BV dst, int cin, int taps)
    {
        int c = i % cin, rest = i / cin, k = rest % taps, o = rest / taps;                    // dst [o][k][c] <- src [o][c][k]
        dst[i] = src[((long)o * cin + c) * taps + k];
    }
    /// <summary>A torch Conv3d weight [cout][cin][taps] (bf16) into the gather layout [cout][taps][cin] the convolution GEMMs read.</summary>
    public void GatherLayoutBf16(BV src, BV dst, int cout, int cin, int taps)
    {
        _gatherBf16 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, BV, BV, int, int>(GatherBf16Impl);
        _gatherBf16((int)((long)cout * taps * cin), src, dst, cin, taps);
    }

    private Action<Index1D, FV, FV, int, float>? _pixelSilu;
    private static void PixelSiluImpl(Index1D row, FV x, FV o, int C, float eps)
    {
        long b = (long)(int)row * C;
        float s = 0f;
        for (int c = 0; c < C; c++) { float v = x[b + c]; s += v * v; }
        float inv = 1f / MathF.Sqrt(s / C + eps);
        for (int c = 0; c < C; c++) { float v = x[b + c] * inv; o[b + c] = v / (1f + MathF.Exp(-v)); }
    }
    /// <summary>o = SiLU(pixel norm(x)) per row of C channels (the LTX-2 decoder's resnet, no timestep modulation). x and o are different buffers.</summary>
    public void PixelNormSilu(FV x, FV o, int rows, int C, float eps)
    {
        _pixelSilu ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, float>(PixelSiluImpl);
        _pixelSilu(rows, x, o, C, eps);
    }

    // ───────────────────────── audio VAE decoder + vocoder (channel-last [positions, C]) ─────────────────────────

    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, int, int, int, int>? _causal2d;
    private static void Causal2dImpl(Index1D p, ArrayView1D<int, Stride1D.Dense> nbr, int T, int M, int kh, int kw)
    {
        int t = p / M, m = p % M;
        for (int a = 0; a < kh; a++)
            for (int b = 0; b < kw; b++)
            {
                int tt = t + a - (kh - 1), mm = m + b - (kw - 1) / 2;          // time: all padding before (causal); mel: symmetric; zeros
                nbr[p * 27 + a * kw + b] = tt >= 0 && mm >= 0 && mm < M ? tt * M + mm : -1;
            }
    }
    /// <summary>The neighbour table [T M, 27] of a kh x kw convolution over (time, mel) positions, causal in time (diffusers' LTX2AudioCausalConv2d, causality_axis "height"), zero padding.</summary>
    public void CausalTable2d(ArrayView1D<int, Stride1D.Dense> nbr, int T, int M, int kh, int kw)
    {
        _causal2d ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, int, int, int, int>(Causal2dImpl);
        _causal2d(T * M, nbr, T, M, kh, kw);
    }

    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, int, int, int>? _same1d;
    private static void Same1dImpl(Index1D l, ArrayView1D<int, Stride1D.Dense> nbr, int L, int k, int d)
    {
        for (int j = 0; j < k; j++) { int s = l + (j - (k - 1) / 2) * d; nbr[l * 27 + j] = s >= 0 && s < L ? s : -1; }
    }
    /// <summary>The neighbour table [L, 27] of a 1-D convolution of odd kernel k, dilation d, "same" zero padding.</summary>
    public void SameTable1d(ArrayView1D<int, Stride1D.Dense> nbr, int L, int k, int d)
    {
        if (k > 27 || k % 2 == 0) throw new ArgumentException($"odd kernels up to 27, got {k}");
        _same1d ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, int, int, int>(Same1dImpl);
        _same1d(L, nbr, L, k, d);
    }

    private Action<Index1D, FV, FV, FV, FV, int, int, int>? _denormAudio;
    private static void DenormAudioImpl(Index1D i, FV z, FV mean, FV std, FV o, int C, int T, int M)
    {
        int c = i % C, p = i / C, t = p / M, m = p % M;                                       // o [t * M + m, c] <- z [c, t, m]; statistics per packed feature c * M + m
        int f = c * M + m;
        o[i] = z[((long)c * T + t) * M + m] * std[f] + mean[f];
    }
    /// <summary>The audio latent z [C, T, M] (ComfyUI's [8, Ta, 16]) denormalised per packed feature (c M + m: <c>z * std + mean</c>) into channel-last [T M, C].</summary>
    public void DenormAudioLatent(FV z, FV mean, FV std, FV o, int C, int T, int M)
    {
        _denormAudio ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, int, int, int>(DenormAudioImpl);
        _denormAudio(C * T * M, z, mean, std, o, C, T, M);
    }

    private Action<Index1D, FV, FV, int, int, int>? _up2;
    private static void Up2Impl(Index1D i, FV x, FV o, int T, int M, int C)
    {
        int c = i % C, p = i / C, M2 = 2 * M, t2 = p / M2, m2 = p % M2;
        o[i] = x[((long)(t2 / 2) * M + m2 / 2) * C + c];
    }
    /// <summary>Nearest x2 upsampling of channel-last [T M, C] in both axes: [2T 2M, C].</summary>
    public void NearestUp2(FV x, FV o, int T, int M, int C)
    {
        _up2 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int>(Up2Impl);
        _up2(4 * T * M * C, x, o, T, M, C);
    }

    private Action<Index1D, FV, FV, float>? _leaky;
    private static void LeakyImpl(Index1D i, FV x, FV o, float slope) { float v = x[i]; o[i] = v >= 0f ? v : v * slope; }
    /// <summary>o = leaky_relu(x, slope); x and o different buffers.</summary>
    public void LeakyRelu(FV x, FV o, int count, float slope)
    {
        _leaky ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, float>(LeakyImpl);
        _leaky(count, x, o, slope);
    }

    private Action<Index1D, FV, FV, FV, FV>? _mean3;
    private static void Mean3Impl(Index1D i, FV a, FV b, FV c, FV o) => o[i] = (a[i] + b[i] + c[i]) / 3f;
    /// <summary>o = (a + b + c) / 3 (torch.mean of the three resblock outputs).</summary>
    public void Mean3(FV a, FV b, FV c, FV o, int count)
    {
        _mean3 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV>(Mean3Impl);
        _mean3(count, a, b, c, o);
    }

    private Action<Index1D, FV>? _tanh;
    private static void TanhImpl(Index1D i, FV x) { float v = x[i]; x[i] = v > 15f ? 1f : v < -15f ? -1f : MathF.Tanh(v); }
    /// <summary>In place tanh.</summary>
    public void TanhInPlace(FV x, int count)
    {
        _tanh ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV>(TanhImpl);
        _tanh(count, x);
    }

    private Action<Index1D, FV, BV, BV, FV, int, int, int, int, int, int>? _convT;
    private static void ConvT1dImpl(Index1D i, FV x, BV w, BV bias, FV o, int Lin, int cin, int cout, int k, int s, int p)
    {
        int co = i % cout, l = i / cout;
        float acc = PrecisionConvert.ConvertToSingle(bias[co]);
        for (int j = 0; j < k; j++)
        {
            int num = l + p - j;                                                                 // l = in s - p + j
            if (num < 0 || num % s != 0) continue;
            int li = num / s;
            if (li >= Lin) continue;
            for (int ci = 0; ci < cin; ci++)
                acc = float.MultiplyAddEstimate(x[(long)li * cin + ci], PrecisionConvert.ConvertToSingle(w[((long)ci * cout + co) * k + j]), acc);
        }
        o[i] = acc;
    }
    /// <summary>
    /// torch ConvTranspose1d (weight [cin, cout, k] bf16, bias bf16, stride s, padding p) on channel-last x [Lin, cin] -> o [Lout, cout], Lout = (Lin - 1) s - 2 p + k; fp32,
    /// exact products.
    /// </summary>
    public void ConvTranspose1d(FV x, BV w, BV bias, FV o, int Lin, int cin, int cout, int k, int s, int p)
    {
        int Lout = (Lin - 1) * s - 2 * p + k;
        _convT ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, BV, BV, FV, int, int, int, int, int, int>(ConvT1dImpl);
        _convT(Lout * cout, x, w, bias, o, Lin, cin, cout, k, s, p);
    }

    private Action<Index1D, FV, FV, int, int>? _transposeIn;
    private static void TransposeInImpl(Index1D i, FV spec, FV o, int T, int M)
    {
        int ch = i % (2 * M), t = i / (2 * M), s = ch / M, m = ch % M;                        // o [t, s M + m] <- spec [t M + m, s] (decoder output channel-last)
        o[i] = spec[((long)t * M + m) * 2 + s];
    }
    /// <summary>The decoder's spectrogram, channel-last [T M, 2], as the vocoder's input [T, 2 M] (channel s M + m: diffusers' transpose(2, 3).flatten(1, 2)).</summary>
    public void SpectrogramToVocoder(FV spec, FV o, int T, int M)
    {
        _transposeIn ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int>(TransposeInImpl);
        _transposeIn(T * 2 * M, spec, o, T, M);
    }

    /// <summary>
    /// The split-RoPE tables for one axis of normalised positions: frequency j of F is <c>theta^(j / (F - 1)) pi / 2</c> (computed in float64, used as float32), the angle
    /// <c>(2 frac - 1) freq</c> in float32, cos / sin evaluated accurately; <paramref name="pad"/> leading entries per row are cos 1 / sin 0. Rows [rows, pad + F].
    /// </summary>
    public static (float[] Cos, float[] Sin) SplitRopeTables(float[] frac, int F, int pad, double theta)
    {
        var freq = new float[F];
        for (int j = 0; j < F; j++) freq[j] = (float)(Math.Pow(theta, F == 1 ? 0.0 : (double)j / (F - 1)) * Math.PI / 2.0);
        int W = pad + F, rows = frac.Length;
        var c = new float[rows * W]; var s = new float[rows * W];
        for (int r = 0; r < rows; r++)
        {
            for (int j = 0; j < pad; j++) { c[r * W + j] = 1f; s[r * W + j] = 0f; }
            float g = frac[r] * 2f - 1f;
            for (int j = 0; j < F; j++) { float a = g * freq[j]; c[r * W + pad + j] = (float)Math.Cos(a); s[r * W + pad + j] = (float)Math.Sin(a); }
        }
        return (c, s);
    }
}
