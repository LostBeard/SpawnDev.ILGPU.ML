using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Wan;

using FV = ArrayView1D<float, Stride1D.Dense>;
using BV = ArrayView1D<BFloat16, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// The kernels of the Wan 2.2 DiT that the TRELLIS.2 flow transformer's <see cref="Trellis2.DiTKernels"/> do not have: the RMS norm over a whole row (Wan
/// normalises q and k across all heads together, TRELLIS per head), RoPE applied from a precomputed phase table, the patch gather and its inverse
/// (the patch embedding is a Conv3d with kernel = stride = (1, 2, 2), i.e. a linear layer on patches), and a broadcast add for the head modulation.
/// Like the DiT kernels these are the correctness lane first: one tensor layout, no tricks.
/// </summary>
public sealed class WanKernels
{
    private readonly Accelerator _acc;
    public WanKernels(Accelerator accelerator) => _acc = accelerator;

    // ---------------------------------------------------------------------------------------------------
    // x[n, off + c] = x[n, off + c] * rsqrt(mean_c(x^2) + eps) * gamma[c]      over C columns of every row, IN PLACE (WanRMSNorm).
    // ---------------------------------------------------------------------------------------------------
    private const int NormGroup = 256;
    private Action<KernelConfig, FV, BV, int, int, int, float>? _rmsGroup;
    private static void RowRmsNormGroupImpl(FV buf, BV gamma, int stride, int off, int C, float eps)
    {
        int row = Grid.IdxX, t = Group.IdxX;
        long b = (long)row * stride + off;
        var sh = SharedMemory.Allocate<float>(NormGroup);
        float s = 0f;
        for (int c = t; c < C; c += NormGroup) { float v = buf[b + c]; s += v * v; }
        sh[t] = s;
        Group.Barrier();
        for (int st = NormGroup / 2; st > 0; st >>= 1) { if (t < st) sh[t] += sh[t + st]; Group.Barrier(); }
        float inv = 1f / MathF.Sqrt(sh[0] / C + eps);
        for (int c = t; c < C; c += NormGroup) buf[b + c] = buf[b + c] * inv * PrecisionConvert.ConvertToSingle(gamma[c]);
    }
    private Action<Index1D, FV, BV, int, int, int, float>? _rms;
    private static void RowRmsNormImpl(Index1D row, FV buf, BV gamma, int stride, int off, int C, float eps)
    {
        long b = (long)(int)row * stride + off;
        float s = 0f;
        for (int c = 0; c < C; c++) { float v = buf[b + c]; s += v * v; }
        float inv = 1f / MathF.Sqrt(s / C + eps);
        for (int c = 0; c < C; c++) buf[b + c] = buf[b + c] * inv * PrecisionConvert.ConvertToSingle(gamma[c]);
    }
    /// <summary>RMS-normalise C columns starting at <paramref name="off"/> of each of <paramref name="rows"/> rows (row stride <paramref name="stride"/>), scaled by the bf16 gamma[C].</summary>
    public void RowRmsNorm(FV buf, BV gamma, int rows, int stride, int off, int C, float eps)
    {
        if (_acc.MaxGroupSize.X >= NormGroup)
        {
            _rmsGroup ??= _acc.LoadStreamKernel<FV, BV, int, int, int, float>(RowRmsNormGroupImpl);
            _rmsGroup(new KernelConfig(rows, NormGroup), buf, gamma, stride, off, C, eps);
            return;
        }
        _rms ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, BV, int, int, int, float>(RowRmsNormImpl);
        _rms(rows, buf, gamma, stride, off, C, eps);
    }

    // ---------------------------------------------------------------------------------------------------
    // RoPE from a phase table: phases = [N, D/2, 2] (cos, sin); every (token, head) vector at buf[n*stride + off + h*D ..) has its adjacent pairs
    // (x[2p], x[2p+1]) rotated, IN PLACE (the complex multiply of rope_apply).
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, FV, int, int, int, int>? _rope;
    private static void RopeApplyImpl(Index1D i, FV buf, FV phases, int stride, int off, int H, int D)
    {
        int pairs = D / 2;
        int p = i % pairs, h = (i / pairs) % H, n = i / (pairs * H);
        long b = (long)n * stride + off + h * D + 2 * p;
        long pi = ((long)n * pairs + p) * 2;
        float c = phases[pi], s = phases[pi + 1];
        float x0 = buf[b], x1 = buf[b + 1];
        buf[b] = x0 * c - x1 * s;
        buf[b + 1] = x0 * s + x1 * c;
    }
    public void RopeApply(FV buf, FV phases, int N, int stride, int off, int H, int D)
    {
        _rope ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int, int>(RopeApplyImpl);
        _rope(N * H * (D / 2), buf, phases, stride, off, H, D);
    }

    // ---------------------------------------------------------------------------------------------------
    // patch gather: latent [Cin, Fz, h, w] -> patches [N, Cin*4], N = Fz * (h/2) * (w/2), token order (f, i, j), column (c, kh, kw)
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, FV, int, int, int, int>? _patchify;
    private static void PatchifyImpl(Index1D i, FV latent, FV patches, int Cin, int Fz, int h, int w)
    {
        int cols = Cin * 4, n = i / cols, col = i % cols;
        int c = col / 4, kh = (col % 4) / 2, kw = col % 2;
        int hp = h / 2, wp = w / 2;
        int f = n / (hp * wp), rem = n % (hp * wp), pi = rem / wp, pj = rem % wp;
        patches[i] = latent[(((long)c * Fz + f) * h + 2 * pi + kh) * w + 2 * pj + kw];
    }
    public void Patchify(FV latent, FV patches, int Cin, int Fz, int h, int w)
    {
        _patchify ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int, int>(PatchifyImpl);
        _patchify(Fz * (h / 2) * (w / 2) * Cin * 4, latent, patches, Cin, Fz, h, w);
    }

    // unpatchify: head output [N, 4*Cout] with column ((q*2 + r) * Cout + c) -> latent [Cout, Fz, h, w] (height 2*i + q, width 2*j + r)
    private Action<Index1D, FV, FV, int, int, int, int>? _unpatchify;
    private static void UnpatchifyImpl(Index1D i, FV headOut, FV latent, int Cout, int Fz, int h, int w)
    {
        int x = i % w, y = (i / w) % h, f = (i / (w * h)) % Fz, c = i / (w * h * Fz);
        int hp = h / 2, wp = w / 2;
        int n = f * hp * wp + (y / 2) * wp + x / 2;
        int q = y % 2, r = x % 2;
        latent[i] = headOut[(long)n * (4 * Cout) + (q * 2 + r) * Cout + c];
    }
    public void Unpatchify(FV headOut, FV latent, int Cout, int Fz, int h, int w)
    {
        _unpatchify ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int, int>(UnpatchifyImpl);
        _unpatchify(Cout * Fz * h * w, headOut, latent, Cout, Fz, h, w);
    }

    // out[i] = a[i] + b[i % period]   (the head's [2, C] modulation table plus the time embedding e)
    private Action<Index1D, BV, FV, FV, int>? _addBroadcast;
    private static void AddBroadcastImpl(Index1D i, BV a, FV b, FV o, int period) => o[i] = PrecisionConvert.ConvertToSingle(a[i]) + b[i % period];
    public void AddBroadcastBf16(BV a, FV b, FV o, int count, int period)
    {
        _addBroadcast ??= _acc.LoadAutoGroupedStreamKernel<Index1D, BV, FV, FV, int>(AddBroadcastImpl);
        _addBroadcast(count, a, b, o, period);
    }

    // ===================================================================================================
    // VAE kernels. Activations are channel-last [voxels, C] fp32 with voxel = (t * H + h) * W + w.
    // ===================================================================================================

    // F.normalize(x, dim=channels) * sqrt(C) * gamma (the VAE's RMS_norm), per voxel, IN PLACE; gamma fp32.
    private Action<KernelConfig, FV, FV, int>? _chanRmsGroup;
    private static void ChannelRmsGroupImpl(FV buf, FV gamma, int C)
    {
        int row = Grid.IdxX, t = Group.IdxX;
        long b = (long)row * C;
        var sh = SharedMemory.Allocate<float>(NormGroup);
        float s = 0f;
        for (int c = t; c < C; c += NormGroup) { float v = buf[b + c]; s += v * v; }
        sh[t] = s;
        Group.Barrier();
        for (int st = NormGroup / 2; st > 0; st >>= 1) { if (t < st) sh[t] += sh[t + st]; Group.Barrier(); }
        float inv = MathF.Sqrt((float)C) / MathF.Max(MathF.Sqrt(sh[0]), 1e-12f);
        for (int c = t; c < C; c += NormGroup) buf[b + c] = buf[b + c] * inv * gamma[c];
    }
    private Action<Index1D, FV, FV, int>? _chanRms;
    private static void ChannelRmsImpl(Index1D row, FV buf, FV gamma, int C)
    {
        long b = (long)(int)row * C;
        float s = 0f;
        for (int c = 0; c < C; c++) { float v = buf[b + c]; s += v * v; }
        float inv = MathF.Sqrt((float)C) / MathF.Max(MathF.Sqrt(s), 1e-12f);
        for (int c = 0; c < C; c++) buf[b + c] = buf[b + c] * inv * gamma[c];
    }
    public void ChannelRmsNorm(FV buf, FV gamma, int rows, int C)
    {
        if (_acc.MaxGroupSize.X >= NormGroup)
        {
            _chanRmsGroup ??= _acc.LoadStreamKernel<FV, FV, int>(ChannelRmsGroupImpl);
            _chanRmsGroup(new KernelConfig(rows, NormGroup), buf, gamma, C);
            return;
        }
        _chanRms ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int>(ChannelRmsImpl);
        _chanRms(rows, buf, gamma, C);
    }

    // Neighbour tables (row stride 27, the gather GEMM's). Input frames are laid out [cache frames][chunk frames]; an output frame t of a causal temporal kernel of 3 taps
    // reads input frames cnt + t - 2 + dt (dt = 0..2), a negative frame is the zero padding. `cnt` = the cached frames in front (0, 1 or 2).
    private Action<Index1D, IV, int, int, int>? _causalTable;
    private static void CausalTableImpl(Index1D m, IV nbr, int H, int W, int cnt)
    {
        int hw = H * W, t = m / hw, p = m % hw, h = p / W, w = p % W;
        for (int kk = 0; kk < 27; kk++)
        {
            int dt = kk / 9, hh = h + (kk / 3) % 3 - 1, ww = w + kk % 3 - 1, c = cnt + t - 2 + dt;
            nbr[m * 27 + kk] = c >= 0 && hh >= 0 && hh < H && ww >= 0 && ww < W ? (c * H + hh) * W + ww : -1;
        }
    }
    /// <summary>3x3x3 causal convolution table over T output frames of H x W voxels.</summary>
    public void CausalTable(IV nbr, int T, int H, int W, int cnt)
    {
        _causalTable ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, int, int, int>(CausalTableImpl);
        _causalTable(T * H * W, nbr, H, W, cnt);
    }

    private Action<Index1D, IV, int, int>? _timeTable;
    private static void TimeTableImpl(Index1D m, IV nbr, int hw, int cnt)
    {
        int t = m / hw, p = m % hw;
        for (int dt = 0; dt < 3; dt++) { int c = cnt + t - 2 + dt; nbr[m * 27 + dt] = c >= 0 ? c * hw + p : -1; }
    }
    /// <summary>(3,1,1) causal temporal convolution table (3 taps).</summary>
    public void TimeTable(IV nbr, int T, int H, int W, int cnt)
    {
        _timeTable ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, int, int>(TimeTableImpl);
        _timeTable(T * H * W, nbr, H * W, cnt);
    }

    private Action<Index1D, IV, int, int>? _upTable;
    private static void UpTableImpl(Index1D m, IV nbr, int Hin, int Win)
    {
        int Ho = 2 * Hin, Wo = 2 * Win, hw = Ho * Wo, t = m / hw, p = m % hw, h = p / Wo, w = p % Wo;
        for (int kk = 0; kk < 9; kk++)
        {
            int hh = h + kk / 3 - 1, ww = w + kk % 3 - 1;
            nbr[m * 27 + kk] = hh >= 0 && hh < Ho && ww >= 0 && ww < Wo ? (t * Hin + hh / 2) * Win + ww / 2 : -1;     // nearest-exact x2, then the 3x3 taps on the upsampled grid
        }
    }
    /// <summary>Nearest-neighbour x2 followed by a 3x3 convolution, as one gather table (9 taps): input [T, Hin, Win], output [T, 2Hin, 2Win].</summary>
    public void UpTable(IV nbr, int T, int Hin, int Win)
    {
        _upTable ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, int, int>(UpTableImpl);
        _upTable(T * 4 * Hin * Win, nbr, Hin, Win);
    }

    // DupUp3D (the up-block shortcut): channel repeat_interleave then a channel-to-space regroup; output [T*ft - dropFrames, H*fs, W*fs, outC]
    private Action<Index1D, FV, FV, int, int, int, int, int, int, int, int>? _dupUp;
    private static void DupUpImpl(Index1D i, FV x, FV o, int H, int W, int inC, int outC, int ft, int fs, int repeats, int dropFrames)
    {
        int oc = i % outC, rest = i / outC, Wo = W * fs, Ho = H * fs;
        int col = rest % Wo; rest /= Wo;
        int row = rest % Ho; int f = rest / Ho + dropFrames;
        int a = f % ft, t = f / ft, b = row % fs, h = row / fs, d = col % fs, w = col / fs;
        int cprime = ((oc * ft + a) * fs + b) * fs + d;
        o[i] = x[((long)(t * H + h) * W + w) * inC + cprime / repeats];
    }
    public void DupUp(FV x, FV o, int T, int H, int W, int inC, int outC, int ft, int fs, int dropFrames)
    {
        _dupUp ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int, int, int, int, int, int>(DupUpImpl);
        int repeats = outC * ft * fs * fs / inC;
        _dupUp((T * ft - dropFrames) * H * fs * W * fs * outC, x, o, H, W, inC, outC, ft, fs, repeats, dropFrames);
    }

    // upsample3d's time doubling: conv output [T*HW, 2c] -> [2T*HW, c], output frame 2t + j takes channels j*c .. (j+1)*c of input frame t
    private Action<Index1D, FV, FV, int, int>? _timeDouble;
    private static void TimeDoubleImpl(Index1D i, FV x, FV o, int hw, int c)
    {
        int ch = i % c, rest = i / c, p = rest % hw, f = rest / hw, t = f / 2, j = f % 2;
        o[i] = x[((long)(t * hw + p)) * (2 * c) + j * c + ch];
    }
    public void TimeDouble(FV x, FV o, int T, int hw, int c)
    {
        _timeDouble ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int>(TimeDoubleImpl);
        _timeDouble(2 * T * hw * c, x, o, hw, c);
    }

    // The VAE's single-head spatial attention over N voxels of one frame, head width D: qkv [N, 3D] (q | k | v), scale 1/sqrt(D)
    private Action<Index1D, FV, FV, int, int, float>? _attnScores;
    private static void AttnScoresImpl(Index1D idx, FV qkv, FV s, int N, int D, float scale)
    {
        int i = idx / N, j = idx % N;
        long qb = (long)i * 3 * D, kb = (long)j * 3 * D + D;
        float acc = 0f;
        for (int d = 0; d < D; d++) acc += qkv[qb + d] * qkv[kb + d];
        s[idx] = acc * scale;
    }
    private Action<Index1D, FV, int>? _softmaxRows;
    private static void SoftmaxRowsImpl(Index1D i, FV s, int N)
    {
        long b = (long)(int)i * N;
        float mx = s[b];
        for (int j = 1; j < N; j++) mx = MathF.Max(mx, s[b + j]);
        float sum = 0f;
        for (int j = 0; j < N; j++) { float e = MathF.Exp(s[b + j] - mx); s[b + j] = e; sum += e; }
        float inv = 1f / sum;
        for (int j = 0; j < N; j++) s[b + j] *= inv;
    }
    private Action<Index1D, FV, FV, FV, int, int>? _attnOut;
    private static void AttnOutImpl(Index1D idx, FV p, FV qkv, FV o, int N, int D)
    {
        int i = idx / D, d = idx % D;
        long pb = (long)i * N, vb = 2L * D + d;
        float acc = 0f;
        for (int j = 0; j < N; j++) acc += p[pb + j] * qkv[(long)j * 3 * D + vb];
        o[idx] = acc;
    }
    /// <summary>scores [N, N] scratch, o [N, D].</summary>
    public void SpatialAttention(FV qkv, FV scores, FV o, int N, int D)
    {
        _attnScores ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, float>(AttnScoresImpl);
        _softmaxRows ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, int>(SoftmaxRowsImpl);
        _attnOut ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int>(AttnOutImpl);
        _attnScores(N * N, qkv, scores, N, D, 1f / MathF.Sqrt(D));
        _softmaxRows(N, scores, N);
        _attnOut(N * D, scores, qkv, o, N, D);
    }

    // the decoder's last step: head output [F*H*W, 12] (channel c * 4 + r * 2 + q) -> video [3, F, 2H, 2W] channel-first, clamped to [-1, 1]
    private Action<Index1D, FV, FV, int, int, int>? _unpatchVideo;
    private static void UnpatchVideoImpl(Index1D i, FV head, FV video, int F, int H, int W)
    {
        int x = i % (2 * W), y = (i / (2 * W)) % (2 * H), f = (i / (4 * H * W)) % F, c = i / (4 * H * W * F);
        int q = y % 2, r = x % 2;
        float v = head[((long)(f * H + y / 2) * W + x / 2) * 12 + c * 4 + r * 2 + q];
        video[i] = MathF.Max(-1f, MathF.Min(1f, v));
    }
    public void UnpatchifyVideo(FV head, FV video, int F, int H, int W)
    {
        _unpatchVideo ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int>(UnpatchVideoImpl);
        _unpatchVideo(3 * F * 4 * H * W, head, video, F, H, W);
    }

    // ---------------------------------------------------------------------------------------------------
    // host helpers
    // ---------------------------------------------------------------------------------------------------

    /// <summary>sinusoidal_embedding_1d(freq_dim, t): [cos(t * f_j), sin(t * f_j)], f_j = 10000^(-j / half), computed in double like the reference.</summary>
    public static float[] SinusoidalEmbedding(double t, int dim = 256)
    {
        int half = dim / 2;
        var e = new float[dim];
        for (int j = 0; j < half; j++)
        {
            double a = t * Math.Pow(10000.0, -(double)j / half);
            e[j] = (float)Math.Cos(a);
            e[half + j] = (float)Math.Sin(a);
        }
        return e;
    }

    /// <summary>
    /// The 3-D RoPE phase table of Wan: head dim D split into (frame, height, width) pair counts [D/2 - 2 * (D/2/3), D/2/3, D/2/3] (22, 21, 21 for D = 128),
    /// pair j of an axis with 2m pairs rotating by position * 10000^(-2j / (2m)); tokens in (f, i, j) order. [N, D/2, 2] (cos, sin), angles in double.
    /// </summary>
    public static float[] RopePhases3D(int Fz, int hp, int wp, int D)
    {
        int pairs = D / 2, per = pairs / 3, pf = pairs - 2 * per;     // pair counts: frame, height, width
        int dimF = 2 * pf, dimH = 2 * per, dimW = 2 * per;
        var fF = new double[pf]; var fH = new double[per]; var fW = new double[per];
        for (int j = 0; j < pf; j++) fF[j] = Math.Pow(10000.0, -(2.0 * j) / dimF);
        for (int j = 0; j < per; j++) { fH[j] = Math.Pow(10000.0, -(2.0 * j) / dimH); fW[j] = Math.Pow(10000.0, -(2.0 * j) / dimW); }
        int N = Fz * hp * wp;
        var t = new float[(long)N * pairs * 2];
        int n = 0;
        for (int f = 0; f < Fz; f++)
            for (int i = 0; i < hp; i++)
                for (int j = 0; j < wp; j++, n++)
                {
                    long o = (long)n * pairs * 2;
                    for (int p = 0; p < pairs; p++)
                    {
                        double a = p < pf ? f * fF[p] : p < pf + per ? i * fH[p - pf] : j * fW[p - pf - per];
                        t[o + 2 * p] = (float)Math.Cos(a);
                        t[o + 2 * p + 1] = (float)Math.Sin(a);
                    }
                }
        return t;
    }
}
