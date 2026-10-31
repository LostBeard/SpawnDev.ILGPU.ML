using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Wan;

using FV = ArrayView1D<float, Stride1D.Dense>;
using BV = ArrayView1D<BFloat16, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;
using UV = ArrayView1D<uint, Stride1D.Dense>;

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
    // MagCache's residual: res = hid - res (res held the blocks' input), hid += res (a skipped call), and per token |new| / |old| (its calibration ratio)
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, FV>? _residualFrom, _addRows;
    private static void ResidualFromImpl(Index1D i, FV hid, FV res) => res[i] = hid[i] - res[i];
    private static void AddRowsImpl(Index1D i, FV hid, FV res) => hid[i] += res[i];
    /// <summary><paramref name="res"/> (the blocks' input) becomes the residual <paramref name="hid"/> - input.</summary>
    public void ResidualFrom(FV hid, FV res, long n)
    {
        _residualFrom ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(ResidualFromImpl);
        _residualFrom((int)n, hid, res);
    }
    /// <summary><paramref name="hid"/> += <paramref name="res"/> (a call that reuses the cached residual instead of running the blocks).</summary>
    public void AddResidual(FV hid, FV res, long n)
    {
        _addRows ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(AddRowsImpl);
        _addRows((int)n, hid, res);
    }
    private Action<Index1D, FV, FV, FV, int>? _normRatio;
    private static void NormRatioImpl(Index1D t, FV a, FV b, FV o, int C)
    {
        long r = (long)(int)t * C;
        float sa = 0f, sb = 0f;
        for (int c = 0; c < C; c++) { float x = a[r + c], y = b[r + c]; sa += x * x; sb += y * y; }
        o[t] = MathF.Sqrt(sa) / MathF.Sqrt(sb);
    }
    /// <summary>o[t] = |a[t]| / |b[t]| over the C columns of each of <paramref name="rows"/> rows (torch: a.norm(dim=-1) / b.norm(dim=-1)).</summary>
    public void NormRatio(FV a, FV b, FV o, int rows, int C)
    {
        _normRatio ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int>(NormRatioImpl);
        _normRatio(rows, a, b, o, C);
    }

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

    // ---- encoder side ----------------------------------------------------------------------------------------------------------------

    // Resample downsample2d: ZeroPad2d((0, 1, 0, 1)) then a 3x3 convolution of stride 2: output (i, j) taps input rows 2i + kh, columns 2j + kw (kk = kh * 3 + kw), a tap past the edge is the zero pad
    private Action<Index1D, IV, int, int>? _downTable;
    private static void DownTableImpl(Index1D m, IV nbr, int Hin, int Win)
    {
        int Ho = Hin / 2, Wo = Win / 2, hw = Ho * Wo, t = m / hw, p = m % hw, i = p / Wo, j = p % Wo;
        for (int kk = 0; kk < 9; kk++)
        {
            int r = 2 * i + kk / 3, c = 2 * j + kk % 3;
            nbr[m * 27 + kk] = r < Hin && c < Win ? (t * Hin + r) * Win + c : -1;
        }
    }
    /// <summary>Stride-2 3x3 convolution table (9 taps): input [T, Hin, Win], output [T, Hin/2, Win/2].</summary>
    public void DownTable(IV nbr, int T, int Hin, int Win)
    {
        _downTable ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, int, int>(DownTableImpl);
        _downTable(T * (Hin / 2) * (Win / 2), nbr, Hin, Win);
    }

    // AvgDown3D (the down-block shortcut): x [T, H, W, cin] channel-last; the temporal axis is padded with padT zero frames in front, then every (ft x fs x fs) cell is flattened into the channels
    // (merged channel (c * ft + a) * fs * fs + b * fs + d) and groups of `group` consecutive merged channels are averaged: output [T', H/fs, W/fs, cout], T' = (T + padT) / ft.
    private Action<Index1D, FV, FV, int, int, int, int, int, int, int, int, int>? _avgDown;
    private static void AvgDownImpl(Index1D i, FV x, FV o, int T, int H, int W, int cin, int cout, int ft, int fs, int padT, int group)
    {
        int Ho = H / fs, Wo = W / fs, oc = i % cout, rest = i / cout;
        int j = rest % Wo; rest /= Wo;
        int r = rest % Ho; int tp = rest / Ho;
        int factor = ft * fs * fs;
        float acc = 0f;
        for (int g = 0; g < group; g++)
        {
            int mc = oc * group + g, c = mc / factor, q = mc % factor;
            int a = q / (fs * fs), b = (q / fs) % fs, d = q % fs;
            int f = tp * ft + a - padT;
            if (f >= 0) acc += x[(((long)f * H + r * fs + b) * W + j * fs + d) * cin + c];
        }
        o[i] = acc / group;
    }
    public void AvgDown(FV x, FV o, int T, int H, int W, int cin, int cout, int ft, int fs)
    {
        _avgDown ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int, int, int, int, int, int, int>(AvgDownImpl);
        int padT = (ft - T % ft) % ft, group = cin * ft * fs * fs / cout, Tp = (T + padT) / ft;
        _avgDown(Tp * (H / fs) * (W / fs) * cout, x, o, T, H, W, cin, cout, ft, fs, padT, group);
    }

    // the encoder's first step: image [3, F, H, W] channel-first -> [F * H/2 * W/2, 12] channel-last, channel c * 4 + r * 2 + q (q: row subpixel, r: column subpixel)
    private Action<Index1D, FV, FV, int, int, int>? _patchifyImage;
    private static void PatchifyImageImpl(Index1D i, FV img, FV o, int F, int H, int W)
    {
        int ch = i % 12, p = i / 12, Wh = W / 2, Hh = H / 2;
        int j = p % Wh, ii = (p / Wh) % Hh, f = p / (Wh * Hh);
        int c = ch / 4, r = (ch % 4) / 2, q = ch % 2;
        o[i] = img[(((long)c * F + f) * H + 2 * ii + q) * W + 2 * j + r];
    }
    public void PatchifyImage(FV img, FV o, int F, int H, int W)
    {
        _patchifyImage ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int>(PatchifyImageImpl);
        _patchifyImage(F * (H / 2) * (W / 2) * 12, img, o, F, H, W);
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

    // ===================================================================================================
    // umT5 encoder kernels: attention WITHOUT the 1/sqrt(d) scaling, plus a per-head relative-position bias read from a bucket table
    // ===================================================================================================
    private Action<Index1D, FV, BV, IV, FV, int, int, int>? _t5Scores;
    private static void T5ScoresImpl(Index1D idx, FV qkv, BV bias, IV bucket, FV s, int L, int H, int D)
    {
        int j = idx % L, i = (idx / L) % L, h = idx / (L * L);
        long qb = (long)i * 3 * H * D + h * D, kb = (long)j * 3 * H * D + H * D + h * D;
        float acc = 0f;
        for (int d = 0; d < D; d++) acc += qkv[qb + d] * qkv[kb + d];
        s[idx] = acc + PrecisionConvert.ConvertToSingle(bias[bucket[j - i + L - 1] * H + h]);
    }
    private Action<Index1D, FV, FV, FV, int, int, int>? _t5Out;
    private static void T5OutImpl(Index1D idx, FV p, FV qkv, FV o, int L, int H, int D)
    {
        int hd = H * D, i = idx / hd, c = idx % hd, h = c / D;
        float acc = 0f;
        long pb = ((long)h * L + i) * L, vb = 2L * hd + c;
        for (int j = 0; j < L; j++) acc += p[pb + j] * qkv[(long)j * 3 * hd + vb];
        o[idx] = acc;
    }
    /// <summary>qkv [L, 3*H*D] (q | k | v), bias bf16 [numBuckets, H], bucket [2L-1] (index j - i + L - 1), scores scratch [H*L*L], o [L, H*D].</summary>
    public void T5Attention(FV qkv, BV bias, IV bucket, FV scores, FV o, int L, int H, int D)
    {
        _t5Scores ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, BV, IV, FV, int, int, int>(T5ScoresImpl);
        _softmaxRows ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, int>(SoftmaxRowsImpl);
        _t5Out ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int>(T5OutImpl);
        _t5Scores(H * L * L, qkv, bias, bucket, scores, L, H, D);
        _softmaxRows(H * L, scores, L);
        _t5Out(L * H * D, scores, qkv, o, L, H, D);
    }

    // fp16 storage for the VAE's feature caches, as pairs packed in a uint (no sub-word writes: every thread owns whole words, so it is exact on every backend including WebGPU)
    private static uint FloatToHalfBits(float f)
    {
        uint x = Interop.FloatAsInt(f);
        uint sign = (x >> 16) & 0x8000u, exp = (x >> 23) & 0xFFu, mant = x & 0x7FFFFFu;
        if (exp == 255u) return sign | 0x7C00u | (mant != 0u ? 0x200u : 0u);               // inf / nan
        int e = (int)exp - 112;                                                              // exponent rebiased 127 -> 15
        if (e >= 31) return sign | 0x7C00u;                                                  // beyond 65504: inf (activations never get there)
        if (e <= 0)
        {
            if (e < -10) return sign;                                                        // underflow to zero
            mant |= 0x800000u;
            int shift = 14 - e;                                                              // 14 .. 24
            uint h = mant >> shift, rem = mant & ((1u << shift) - 1u), mid = 1u << (shift - 1);
            if (rem > mid || (rem == mid && (h & 1u) != 0u)) h++;
            return sign | h;
        }
        uint r = ((uint)e << 10) | (mant >> 13), low = mant & 0x1FFFu;
        if (low > 0x1000u || (low == 0x1000u && (r & 1u) != 0u)) r++;                        // round to nearest even (a carry into the exponent is right)
        return sign | r;
    }
    private static float HalfBitsToFloat(uint h)
    {
        uint sign = (h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u;
        if (e == 0u) { float v = m * 5.9604644775390625e-8f; return sign != 0u ? -v : v; }
        if (e == 31u) return Interop.IntAsFloat(sign | 0x7F800000u | (m << 13));
        return Interop.IntAsFloat(sign | ((e + 112u) << 23) | (m << 13));
    }
    private Action<Index1D, FV, UV>? _packHalf2;
    private static void PackHalf2Impl(Index1D i, FV src, UV dst) => dst[i] = FloatToHalfBits(src[2 * i]) | (FloatToHalfBits(src[2 * i + 1]) << 16);
    /// <summary>dst[i] = the halves of src[2i] and src[2i+1] packed in one uint (round to nearest even); <paramref name="words"/> = count / 2.</summary>
    public void PackHalf2(FV src, UV dst, int words)
    {
        _packHalf2 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, UV>(PackHalf2Impl);
        _packHalf2(words, src, dst);
    }
    private Action<Index1D, UV, FV>? _unpackHalf2;
    private static void UnpackHalf2Impl(Index1D i, UV src, FV dst) { uint u = src[i]; dst[2 * i] = HalfBitsToFloat(u & 0xFFFFu); dst[2 * i + 1] = HalfBitsToFloat(u >> 16); }
    /// <summary>The inverse of <see cref="PackHalf2"/>: src[i] to dst[2i], dst[2i+1].</summary>
    public void UnpackHalf2(UV src, FV dst, int words)
    {
        _unpackHalf2 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, UV, FV>(UnpackHalf2Impl);
        _unpackHalf2(words, src, dst);
    }

    private Action<Index1D, FV, FV>? _mulInPlace;
    private static void MulInPlaceImpl(Index1D i, FV a, FV b) => a[i] *= b[i];
    /// <summary>a[i] *= b[i] (the gated feed-forward: fc1(x) * gelu(gate(x))).</summary>
    public void MulInPlace(FV a, FV b, int count)
    {
        _mulInPlace ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(MulInPlaceImpl);
        _mulInPlace(count, a, b);
    }

    /// <summary>
    /// T5RelativeEmbedding._relative_position_bucket for a bidirectional sequence of <paramref name="L"/> tokens: bucket of (key - query) at index (key - query + L - 1).
    /// The reference computes the logarithmic buckets in float32 (a 1-ulp difference moves a bucket at the exact ratios 2, 4, 8: relative distance 16, 32, 64), so the same float32 steps are used.
    /// </summary>
    public static int[] T5RelativeBuckets(int L, int numBuckets = 32, int maxDist = 128)
    {
        int half = numBuckets / 2, maxExact = half / 2;
        float denom = (float)Math.Log((double)maxDist / maxExact);
        var t = new int[2 * L - 1];
        for (int rel = -(L - 1); rel <= L - 1; rel++)
        {
            int b = rel > 0 ? half : 0, r = Math.Abs(rel);
            int large = r == 0 ? 0 : maxExact + (int)(MathF.Log((float)r / maxExact) / denom * (half - maxExact));
            large = Math.Min(large, half - 1);
            t[rel + L - 1] = b + (r < maxExact ? r : large);
        }
        return t;
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
