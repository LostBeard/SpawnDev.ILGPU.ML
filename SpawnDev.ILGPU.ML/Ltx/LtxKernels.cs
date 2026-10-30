using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Ltx;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// The kernels of the LTX-Video DiT that <see cref="Trellis2.DiTKernels"/> and <see cref="Wan.WanKernels"/> do not have: the RMS norm with NO affine weight fused with the
/// adaLN modulation <c>x * rsqrt(mean(x^2) + eps) * (1 + scale) + shift</c> (LTX modulates an RMS norm, Wan a LayerNorm), plus the host-side RoPE phase table.
/// The rotation itself reuses <see cref="Wan.WanKernels.RopeApply"/>: LTX rotates adjacent pairs of the WHOLE 2048-wide q / k vector with a per-token phase per pair (not per
/// head), which is the Wan kernel with one head of width 2048.
/// </summary>
public sealed class LtxKernels
{
    private readonly Accelerator _acc;
    public LtxKernels(Accelerator accelerator) => _acc = accelerator;

    private const int NormGroup = 256;

    // o[n, c] = x[n, c] * rsqrt(mean_c(x[n]^2) + eps) * (1 + mod[scaleOff + c]) + mod[shiftOff + c]       (RMSNorm without weight, then the adaLN modulation)
    private Action<KernelConfig, FV, FV, FV, int, int, int, float, int>? _modGroup;
    private static void RmsNormModulateGroupImpl(FV x, FV o, FV mod, int shiftOff, int scaleOff, int C, float eps, int silu)
    {
        int row = Grid.IdxX, t = Group.IdxX;
        long b = (long)row * C;
        var sh = SharedMemory.Allocate<float>(NormGroup);
        float s = 0f;
        for (int c = t; c < C; c += NormGroup) { float v = x[b + c]; s += v * v; }
        sh[t] = s;
        Group.Barrier();
        for (int st = NormGroup / 2; st > 0; st >>= 1) { if (t < st) sh[t] += sh[t + st]; Group.Barrier(); }
        float inv = 1f / MathF.Sqrt(sh[0] / C + eps);
        for (int c = t; c < C; c += NormGroup)
        {
            float v = x[b + c] * inv * (1f + mod[scaleOff + c]) + mod[shiftOff + c];
            o[b + c] = silu != 0 ? v / (1f + MathF.Exp(-v)) : v;
        }
    }
    private Action<Index1D, FV, FV, FV, int, int, int, float, int>? _mod;
    private static void RmsNormModulateImpl(Index1D row, FV x, FV o, FV mod, int shiftOff, int scaleOff, int C, float eps, int silu)
    {
        long b = (long)(int)row * C;
        float s = 0f;
        for (int c = 0; c < C; c++) { float v = x[b + c]; s += v * v; }
        float inv = 1f / MathF.Sqrt(s / C + eps);
        for (int c = 0; c < C; c++)
        {
            float v = x[b + c] * inv * (1f + mod[scaleOff + c]) + mod[shiftOff + c];
            o[b + c] = silu != 0 ? v / (1f + MathF.Exp(-v)) : v;
        }
    }
    /// <summary>
    /// RMS-normalise each of <paramref name="rows"/> rows of width <paramref name="C"/> (no weight), then <c>* (1 + mod[scaleOff..]) + mod[shiftOff..]</c>, then optionally SiLU (the LTX VAE's
    /// resnet: pixel norm, timestep modulation, swish). <paramref name="x"/> and <paramref name="o"/> are different buffers.
    /// </summary>
    public void RmsNormModulate(FV x, FV o, FV mod, int shiftOff, int scaleOff, int rows, int C, float eps, bool silu = false)
    {
        if (_acc.MaxGroupSize.X >= NormGroup)
        {
            _modGroup ??= _acc.LoadStreamKernel<FV, FV, FV, int, int, int, float, int>(RmsNormModulateGroupImpl);
            _modGroup(new KernelConfig(rows, NormGroup), x, o, mod, shiftOff, scaleOff, C, eps, silu ? 1 : 0);
            return;
        }
        _mod ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, float, int>(RmsNormModulateImpl);
        _mod(rows, x, o, mod, shiftOff, scaleOff, C, eps, silu ? 1 : 0);
    }

    // ---------------------------------------------------------------------------------------------------
    // VAE decoder kernels. Activations are channel-last fp32 [voxels, C], voxel = (t * H + h) * W + w.
    // ---------------------------------------------------------------------------------------------------

    // o[i] = a[i] + b[i]   (the per-resnet scale_shift_table plus the timestep embedding)
    private Action<Index1D, FV, FV, FV>? _add;
    private static void AddImpl(Index1D i, FV a, FV b, FV o) => o[i] = a[i] + b[i];
    public void Add(FV a, FV b, FV o, int count)
    {
        _add ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV>(AddImpl);
        _add(count, a, b, o);
    }

    // channel-first latent [C, vox] -> channel-last [vox, C], denormalised, then mixed with noise like the pipeline's decode_noise_scale:
    // o[v, c] = (1 - s) * (z[c, v] * std[c] / scaling + mean[c]) + s * noise[c, v]
    private Action<Index1D, FV, FV, FV, FV, FV, int, int, float, float>? _latentToVoxels;
    private static void LatentToVoxelsImpl(Index1D i, FV z, FV o, FV mean, FV std, FV noise, int vox, int C, float scale, float noiseScale)
    {
        int c = i % C, v = i / C;
        long src = (long)c * vox + v;
        o[i] = (1f - noiseScale) * (z[src] * std[c] / scale + mean[c]) + noiseScale * noise[src];
    }
    private Action<Index1D, FV, FV, FV, FV, int, int, float>? _latentToVoxelsPlain;
    private static void LatentToVoxelsPlainImpl(Index1D i, FV z, FV o, FV mean, FV std, int vox, int C, float scale)
    {
        int c = i % C, v = i / C;
        o[i] = z[(long)c * vox + v] * std[c] / scale + mean[c];
    }
    /// <param name="noise">
    /// standard normal noise shaped like <paramref name="z"/> when <paramref name="noiseScale"/> is not 0; null otherwise (a separate kernel without it: WebGPU binds every view
    /// read-write, so the latent cannot stand in for the noise - it failed with "storage buffer aliasing" in the browser).
    /// </param>
    public void LatentToVoxels(FV z, FV o, FV mean, FV std, FV? noise, int vox, int C, float scalingFactor, float noiseScale)
    {
        if (noiseScale == 0f || noise == null)
        {
            if (noiseScale != 0f) throw new ArgumentException("a decode noise scale needs the noise", nameof(noise));
            _latentToVoxelsPlain ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, int, int, float>(LatentToVoxelsPlainImpl);
            _latentToVoxelsPlain(vox * C, z, o, mean, std, vox, C, scalingFactor);
            return;
        }
        _latentToVoxels ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, FV, int, int, float, float>(LatentToVoxelsImpl);
        _latentToVoxels(vox * C, z, o, mean, std, noise.Value, vox, C, scalingFactor, noiseScale);
    }

    // x[i] += dt * v[i]   (the flow-matching Euler step, on the device)
    private Action<Index1D, FV, FV, float>? _axpy;
    private static void AxpyImpl(Index1D i, FV x, FV v, float dt) => x[i] += dt * v[i];
    public void Axpy(FV x, FV v, float dt, int count)
    {
        _axpy ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, float>(AxpyImpl);
        _axpy(count, x, v, dt);
    }

    // The 3x3x3 table of the LTX decoder's (non-causal) convolutions: zero padding in space, REPLICATE padding in time (the first / last frame is repeated), row stride 27.
    private Action<Index1D, IV, int, int, int>? _replicateTable;
    private static void ReplicateTableImpl(Index1D m, IV nbr, int T, int H, int W)
    {
        int hw = H * W, t = m / hw, p = m % hw, h = p / W, w = p % W;
        for (int kk = 0; kk < 27; kk++)
        {
            int tt = t + kk / 9 - 1, hh = h + (kk / 3) % 3 - 1, ww = w + kk % 3 - 1;
            tt = tt < 0 ? 0 : (tt >= T ? T - 1 : tt);
            nbr[m * 27 + kk] = hh >= 0 && hh < H && ww >= 0 && ww < W ? (tt * H + hh) * W + ww : -1;
        }
    }
    public void ReplicateTable(IV nbr, int T, int H, int W)
    {
        _replicateTable ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, int, int, int>(ReplicateTableImpl);
        _replicateTable(T * H * W, nbr, T, H, W);
    }

    // LTXVideoUpsampler3d with the residual shortcut. conv [T*H*W, 8 * C] (channel = ((c * 2 + s0) * 2 + s1) * 2 + s2) and x [T*H*W, Cin] become the doubled grid
    // [(2T - 1) * 2H * 2W, C] (the first doubled frame is dropped): o = conv shuffled + x shuffled the same way (Cin / 8 channels) and tiled Cout / (Cin / 8) times along the channel axis.
    private Action<Index1D, FV, FV, FV, int, int, int, int, int>? _upShuffle;
    private static void UpShuffleImpl(Index1D i, FV conv, FV x, FV o, int H, int W, int C, int Cin, int T)
    {
        int c = i % C, rest = i / C, Wo = 2 * W, Ho = 2 * H;
        int w2 = rest % Wo; rest /= Wo;
        int h2 = rest % Ho; int f = rest / Ho + 1;                    // the doubled frame index in the full 2T grid (frame 0 is dropped)
        int t = f / 2, s0 = f % 2, h = h2 / 2, s1 = h2 % 2, w = w2 / 2, s2 = w2 % 2;
        int s = s0 * 4 + s1 * 2 + s2;
        long vox = ((long)t * H + h) * W + w;
        o[i] = conv[vox * (8 * C) + c * 8 + s] + x[vox * Cin + (c % (Cin / 8)) * 8 + s];
    }
    public void UpShuffle(FV conv, FV x, FV o, int T, int H, int W, int C, int Cin)
    {
        _upShuffle ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, int, int>(UpShuffleImpl);
        _upShuffle((2 * T - 1) * 4 * H * W * C, conv, x, o, H, W, C, Cin, T);
    }

    // head output [F * H * W, 3 * 16] (channel c * 16 + pA * 4 + pB) -> video [3, F, 4H, 4W] channel-first: pixel (y = 4h + pB, x = 4w + pA)
    private Action<Index1D, FV, FV, int, int, int>? _unpatchifyVideo;
    private static void UnpatchifyVideoImpl(Index1D i, FV head, FV video, int F, int H, int W)
    {
        int Wo = 4 * W, Ho = 4 * H;
        int x = i % Wo, y = (i / Wo) % Ho, f = (i / (Wo * Ho)) % F, c = i / (Wo * Ho * F);
        int pA = x % 4, w = x / 4, pB = y % 4, h = y / 4;
        video[i] = head[((long)(f * H + h) * W + w) * 48 + c * 16 + pA * 4 + pB];
    }
    public void UnpatchifyVideo(FV head, FV video, int F, int H, int W)
    {
        _unpatchifyVideo ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int>(UnpatchifyVideoImpl);
        _unpatchifyVideo(3 * F * 16 * H * W, head, video, F, H, W);
    }

    // ---------------------------------------------------------------------------------------------------
    // host helpers
    // ---------------------------------------------------------------------------------------------------

    /// <summary>
    /// The RoPE phase table of <c>LTXVideoRotaryPosEmbed</c> for a latent grid of <paramref name="frames"/> x <paramref name="h"/> x <paramref name="w"/> tokens (patch 1x1x1, tokens in
    /// (frame, row, column) order): [N, dim / 2, 2] (cos, sin), the layout <see cref="Wan.WanKernels.RopeApply"/> reads with one head of width <paramref name="dim"/>.
    /// The reference computes everything in float32, so this does too (a double table differs by up to ~1e-3 rad at the high frequencies, which the reference does not have).
    /// <c>dim / 6</c> frequencies per axis, entry (j, axis) at index 3 * j + axis, each repeated for the two members of a pair; the <c>dim % 6</c> leading entries are padding (cos 1, sin 0).
    /// Positions: axis value = index * scale / base, then <c>angle = freq_j * (value * 2 - 1)</c>. <paramref name="scaleT"/>, <paramref name="scaleH"/>, <paramref name="scaleW"/> is the
    /// pipeline's <c>rope_interpolation_scale</c> (temporal VAE ratio / frame rate, spatial ratio, spatial ratio); base sizes 20 frames, 2048, 2048; theta 10000.
    /// </summary>
    public static float[] RopePhases(int frames, int h, int w, int dim, float scaleT, float scaleH, float scaleW, float baseFrames = 20f, float baseH = 2048f, float baseW = 2048f, float theta = 10000f)
    {
        int nf = dim / 6, pad = dim % 6, pairs = dim / 2;
        if (pad % 2 != 0) throw new ArgumentException("the RoPE width must leave an even padding (dim % 6 even)", nameof(dim));
        var lin = new float[nf];
        float step = nf > 1 ? 1f / (nf - 1) : 0f;                       // torch.linspace(0, 1, nf) in float32: the upper half is computed from the end
        for (int i = 0; i < nf; i++) lin[i] = i < nf / 2 ? i * step : 1f - (nf - 1 - i) * step;
        var freq = new float[nf];
        for (int i = 0; i < nf; i++) freq[i] = MathF.Pow(theta, lin[i]) * MathF.PI / 2f;
        var gf = new float[frames]; var gh = new float[h]; var gw = new float[w];
        for (int i = 0; i < frames; i++) gf[i] = i * scaleT * 1f / baseFrames;
        for (int i = 0; i < h; i++) gh[i] = i * scaleH * 1f / baseH;
        for (int i = 0; i < w; i++) gw[i] = i * scaleW * 1f / baseW;
        int N = frames * h * w;
        var t = new float[(long)N * pairs * 2];
        int n = 0;
        for (int f = 0; f < frames; f++)
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++, n++)
                {
                    long o = (long)n * pairs * 2;
                    for (int p = 0; p < pad / 2; p++) { t[o + 2 * p] = 1f; t[o + 2 * p + 1] = 0f; }
                    for (int k = 0; k < 3 * nf; k++)
                    {
                        int jf = k / 3, axis = k % 3;
                        float g = axis == 0 ? gf[f] : axis == 1 ? gh[i] : gw[j];
                        float a = freq[jf] * (g * 2f - 1f);
                        int pair = pad / 2 + k;
                        t[o + 2 * pair] = MathF.Cos(a);
                        t[o + 2 * pair + 1] = MathF.Sin(a);
                    }
                }
        return t;
    }
}
