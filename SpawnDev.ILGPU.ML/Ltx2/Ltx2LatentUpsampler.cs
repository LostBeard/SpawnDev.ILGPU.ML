using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Ltx2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using Buf = MemoryBuffer1D<float, Stride1D.Dense>;

/// <summary>
/// The LTX-2 x2 spatial latent upscaler (<c>ltx-2-spatial-upscaler-x2-1.0</c>: diffusers' LTX2LatentUpsamplerModel, Apache-2.0; the file's config: mid 1024, 4 blocks per stage,
/// dims 3, spatial x2, rational resampler, which at x2 is a per-frame 3x3 convolution to 4 x 1024 channels and a 2 x 2 pixel shuffle, its blur skipped at stride 1): a
/// model-space latent [128, F, h, w] becomes [128, F, 2h, 2w]. The latent is denormalised with the checkpoint VAE's statistics, then initial 3x3x3 convolution, GroupNorm(32),
/// SiLU, four resblocks (<c>conv, GroupNorm, SiLU, conv, GroupNorm, + input, SiLU</c>), the resampler, four more resblocks, a 3x3x3 convolution back to 128 (zero padding);
/// optionally AdaIN against the input (<see cref="UpsampleAsync"/>'s adainFactor), and normalised back. The same design as LTX-Video 0.9.8's upscaler
/// (<see cref="Ltx.LtxLatentUpsampler"/>) at twice the width, on the file's exact bf16 weights (<see cref="SparseKernels.ConvTapsBf16Exact"/>).
/// </summary>
public sealed class Ltx2LatentUpsampler : IDisposable
{
    public const int LatentChannels = 128;
    private const float GroupNormEps = 1e-5f;

    private readonly Accelerator _acc;
    private readonly StreamedModelWeights _sw;
    private readonly SparseKernels _sk;
    private readonly SdKernels _sd;
    private readonly Ltx.LtxKernels _lk;
    private readonly Ltx2Kernels _l;
    private readonly ActivationPool _temps;
    private readonly int _mid, _blocks;
    private readonly float[] _mean, _std;
    private readonly Dictionary<string, (MemoryBuffer1D<BFloat16, Stride1D.Dense> W, int Taps)> _convs = new();
    private readonly Dictionary<string, Buf> _norms = new();
    private readonly Dictionary<(int Taps, int T, int H, int W), MemoryBuffer1D<int, Stride1D.Dense>> _tables = new();

    /// <summary>Test hook: (stage name, activation view, frames, height, width, channels); valid only during the call.</summary>
    public Action<string, FV, int, int, int, int>? Tap { get; set; }

    private Ltx2LatentUpsampler(Accelerator acc, StreamedModelWeights w, int mid, int blocks, float[] mean, float[] std)
    {
        _acc = acc; _sw = w; _mid = mid; _blocks = blocks; _mean = mean; _std = std;
        _sk = new SparseKernels(acc); _sd = new SdKernels(acc); _lk = new Ltx.LtxKernels(acc); _l = new Ltx2Kernels(acc); _temps = new ActivationPool(acc);
    }

    /// <summary>
    /// Load the upscaler (<paramref name="upscalerFile"/>) and the latent statistics of the LTX-2 checkpoint's video VAE (<paramref name="checkpointFile"/>:
    /// <c>vae.per_channel_statistics.{mean,std}-of-means</c>; only those two vectors are read).
    /// </summary>
    public static Task<Ltx2LatentUpsampler> OpenAsync(Accelerator accelerator, IModelSource source, string repo, string upscalerFile, string checkpointFile, CancellationToken ct = default)
        => OpenAsync(accelerator, source, repo, upscalerFile, source, repo, checkpointFile, ct);

    /// <summary>The same with the upscaler and the checkpoint in different places (ComfyUI keeps latent_upscale_models and checkpoints apart).</summary>
    public static async Task<Ltx2LatentUpsampler> OpenAsync(Accelerator accelerator, IModelSource upscalerSource, string upscalerRepo, string upscalerFile,
        IModelSource checkpointSource, string checkpointRepo, string checkpointFile, CancellationToken ct = default)
    {
        var mean = await ReadBf16VectorAsync(checkpointSource, checkpointRepo, checkpointFile, "vae.per_channel_statistics.mean-of-means", ct);
        var std = await ReadBf16VectorAsync(checkpointSource, checkpointRepo, checkpointFile, "vae.per_channel_statistics.std-of-means", ct);
        var w = await StreamedModelWeights.OpenAsync(accelerator, upscalerSource, upscalerRepo, upscalerFile, "initial_conv.weight", _ => -1, 0, 0, null, ct);
        try
        {
            var b = w.Base;
            int mid = b.Shape("initial_conv.weight")[0], blocks = 0;
            while (b.Has($"res_blocks.{blocks}.conv1.weight")) blocks++;
            if (!b.Has("upsampler.conv.weight") || b.Shape("upsampler.conv.weight")[0] != 4 * mid || b.Shape("upsampler.conv.weight").Length != 4)
                throw new NotSupportedException($"'{upscalerFile}': expected the x2 rational resampler (upsampler.conv [{4 * mid}, {mid}, 3, 3])");
            var u = new Ltx2LatentUpsampler(accelerator, w, mid, blocks, mean, std);
            await u.PrepareAsync();
            return u;
        }
        catch { w.Dispose(); throw; }
    }

    private static async Task<float[]> ReadBf16VectorAsync(IModelSource source, string repo, string file, string name, CancellationToken ct)
    {
        await using var s = await source.OpenAsync(repo, file, ct);
        var lenBytes = new byte[8]; await s.ReadExactlyAsync(lenBytes, ct);
        long hl = BitConverter.ToInt64(lenBytes);
        if (hl <= 0 || hl > 100_000_000) throw new InvalidDataException($"'{file}': header size {hl}");
        var header = new byte[hl]; await s.ReadExactlyAsync(header, ct);
        using var doc = System.Text.Json.JsonDocument.Parse(header);
        if (!doc.RootElement.TryGetProperty(name, out var t)) throw new InvalidDataException($"'{file}' has no {name}");
        if (t.GetProperty("dtype").GetString() != "BF16") throw new NotSupportedException($"{name}: {t.GetProperty("dtype").GetString()}, expected BF16");
        long a = t.GetProperty("data_offsets")[0].GetInt64(), e = t.GetProperty("data_offsets")[1].GetInt64();
        var raw = new byte[e - a];
        s.Seek(8 + hl + a, SeekOrigin.Begin); await s.ReadExactlyAsync(raw, ct);
        var o = new float[raw.Length / 2];
        for (int i = 0; i < o.Length; i++) o[i] = BitConverter.Int32BitsToSingle(BitConverter.ToUInt16(raw, 2 * i) << 16);
        return o;
    }

    // convolutions into the gather layout (bf16, exact); GroupNorm parameters to fp32 (exact)
    private async Task PrepareAsync()
    {
        var b = _sw.Base;
        // upsampler.blur_down.kernel: the rational resampler's anti-alias blur, a no-op at x2 (stride 1)
        foreach (var n in b.Names.Where(n => n.EndsWith(".weight", StringComparison.Ordinal)).ToList())
        {
            var s = b.Shape(n);
            if (s.Length == 5 || s.Length == 4)
            {
                int taps = s.Length == 5 ? s[2] * s[3] * s[4] : s[2] * s[3];
                var g = _acc.Allocate1D<BFloat16>((long)s[0] * taps * s[1]);
                _l.GatherLayoutBf16(b.BF16(n), g.View, s[0], s[1], taps);
                _convs[n[..^".weight".Length]] = (g, taps);
            }
            else if (s.Length == 1 && n.Contains("norm", StringComparison.Ordinal))
            {
                string p = n[..^".weight".Length];
                _norms[p + ".w"] = await ToF32(b.BF16(n)); _norms[p + ".b"] = await ToF32(b.BF16(p + ".bias"));
            }
        }
        await _acc.SynchronizeAsync();
    }

    private async Task<Buf> ToF32(ArrayView1D<BFloat16, Stride1D.Dense> v)
    {
        var u = await v.Cast<BFloat16, ushort>().CopyToHostAsync();
        var f = new float[u.Length]; for (int i = 0; i < u.Length; i++) f[i] = BitConverter.Int32BitsToSingle(u[i] << 16);
        return _acc.Allocate1D(f);
    }

    private Buf Tmp(long n) => _temps.Get(n);
    private void Drop(Buf x) => _temps.Drop(x);
    private void Flush() => _temps.Flush();

    private MemoryBuffer1D<int, Stride1D.Dense> Table(int taps, int T, int H, int W)
    {
        if (_tables.TryGetValue((taps, T, H, W), out var t)) return t;
        t = _acc.Allocate1D<int>((long)T * H * W * 27);
        if (taps == 27) _lk.ZeroTable3(t.View, T, H, W); else _lk.ZeroTable2(t.View, T, H, W);
        _tables[(taps, T, H, W)] = t;
        return t;
    }

    private Buf Conv(string name, Buf x, int T, int H, int W, int cin)
    {
        var (w, taps) = _convs[name];
        int rows = T * H * W, cout = (int)(w.Length / taps / cin);
        if (!_sk.ConvTapsBf16ExactOk(rows, taps, cin, cout, rows)) throw new NotSupportedException($"{name}: {T} x {H} x {W} x {cin} beyond one exact GEMM");
        var y = Tmp((long)rows * cout);
        _sk.ConvTapsBf16Exact(x.View, Table(taps, T, H, W).View, w.View, _sw.Base.BF16(name + ".bias"), y.View, rows, cin, cout, taps);
        return y;
    }

    private Buf Norm(string name, Buf x, int rows, int C, bool silu)
    {
        var o = Tmp((long)rows * C);
        _sd.GroupNorm(x.View, o.View, _norms[name + ".w"].View, _norms[name + ".b"].View, rows, C, 32, GroupNormEps, silu);
        return o;
    }

    private Buf ResBlock(string p, Buf x, int T, int H, int W)
    {
        int rows = T * H * W, C = _mid;
        var c1 = Conv(p + "conv1", x, T, H, W, C);
        var n1 = Norm(p + "norm1", c1, rows, C, silu: true); Drop(c1);
        var c2 = Conv(p + "conv2", n1, T, H, W, C); Drop(n1);
        var n2 = Norm(p + "norm2", c2, rows, C, silu: false); Drop(c2);
        var o = Tmp((long)rows * C);
        _lk.AddSilu(n2.View, x.View, o.View, rows * C);
        Drop(n2); Drop(x); Flush();
        return o;
    }

    /// <summary>
    /// Upsample the model-space latent <paramref name="z"/> ([128, frames, h, w] channel-first) to [128, frames, 2h, 2w] in model space. <paramref name="adainFactor"/>: 1 = each
    /// channel takes the input's mean and (unbiased) standard deviation, 0 = the network's output as it is (ComfyUI's LTXVLatentUpsampler: MEASURED 4.8e-6 with 0, 0.44 with 1).
    /// </summary>
    public async Task<float[]> UpsampleAsync(float[] z, int frames, int h, int w, float adainFactor = 0f)
    {
        int T = frames, H = h, W = w, vox = T * H * W;
        if (z.Length != (long)LatentChannels * vox) throw new ArgumentException("the latent must be [128, frames, h, w]");
        using var zIn = _acc.Allocate1D(z); using var mean = _acc.Allocate1D(_mean); using var std = _acc.Allocate1D(_std);
        var f = await NetworkAsync(zIn.View, mean.View, std.View, T, H, W);
        var up = await f.View.CopyToHostAsync();
        Drop(f); Flush();
        H *= 2; W *= 2;
        int vox2 = T * H * W;
        var o = new float[(long)LatentChannels * vox2];
        for (int ch = 0; ch < LatentChannels; ch++)
        {
            double rMean = 0, rSd = 1, uMean = 0, uSd = 1;
            if (adainFactor != 0f)
            {
                double rs = 0, rq = 0, us = 0, uq = 0;
                for (int v = 0; v < vox; v++) { double r = z[(long)ch * vox + v] * _std[ch] + _mean[ch]; rs += r; rq += r * r; }
                for (int v = 0; v < vox2; v++) { double a = up[(long)v * LatentChannels + ch]; us += a; uq += a * a; }
                rMean = rs / vox; rSd = Math.Sqrt(Math.Max(0, (rq - rs * rMean) / (vox - 1)));
                uMean = us / vox2; uSd = Math.Sqrt(Math.Max(0, (uq - us * uMean) / (vox2 - 1)));
            }
            for (int v = 0; v < vox2; v++)
            {
                double a = up[(long)v * LatentChannels + ch];
                double blended = adainFactor == 0f ? a : a + adainFactor * ((a - uMean) / uSd * rSd + rMean - a);
                o[(long)ch * vox2 + v] = (float)((blended - _mean[ch]) / _std[ch]);
            }
        }
        return o;
    }

    // the network on a device latent [128, T, H, W] channel-first (read, not modified) -> [T * 2H * 2W, 128] channel-last, denormalised (pooled: the caller Drops it), drained
    private async Task<Buf> NetworkAsync(FV z, FV mean, FV std, int T, int H, int W)
    {
        int vox = T * H * W;
        var x0 = Tmp((long)vox * LatentChannels);
        _lk.LatentToVoxels(z, x0.View, mean, std, null, vox, LatentChannels, 1f, 0f);
        var c = Conv("initial_conv", x0, T, H, W, LatentChannels); Drop(x0);
        var x = Norm("initial_norm", c, vox, _mid, silu: true); Drop(c); Flush();
        Tap?.Invoke("initial", x.View, T, H, W, _mid);
        for (int j = 0; j < _blocks; j++) x = ResBlock($"res_blocks.{j}.", x, T, H, W);
        var u = Conv("upsampler.conv", x, T, H, W, _mid); Drop(x);
        x = Tmp((long)vox * 4 * _mid);
        _lk.PixelShuffle2(u.View, x.View, T, H, W, _mid);
        Drop(u); Flush();
        H *= 2; W *= 2;
        for (int j = 0; j < _blocks; j++) x = ResBlock($"post_upsample_res_blocks.{j}.", x, T, H, W);
        var f = Conv("final_conv", x, T, H, W, _mid); Drop(x); Flush();
        Tap?.Invoke("final", f.View, T, H, W, LatentChannels);
        await _acc.SynchronizeAsync();
        return f;
    }

    /// <summary>
    /// <see cref="UpsampleAsync"/> on the device (rule 4: a device latent in, a device latent out): <paramref name="z"/> [128, frames, h, w] channel-first (read, not modified);
    /// returns [128, frames, 2h, 2w] in model space in a buffer the caller owns. AdaIN (when <paramref name="adainFactor"/> is not 0) on the device in float32.
    /// </summary>
    public async Task<Buf> UpsampleDeviceAsync(FV z, int frames, int h, int w, float adainFactor = 0f)
    {
        int vox = frames * h * w, vox2 = frames * 4 * h * w;
        if (z.Length != (long)LatentChannels * vox) throw new ArgumentException("the latent must be [128, frames, h, w]");
        using var mean = _acc.Allocate1D(_mean); using var std = _acc.Allocate1D(_std);
        var f = await NetworkAsync(z, mean.View, std.View, frames, h, w);
        using var rs = _acc.Allocate1D<float>(2 * LatentChannels); using var us = _acc.Allocate1D<float>(2 * LatentChannels);
        if (adainFactor != 0f)
        {
            _lk.ChannelStats(z, rs.View, mean.View, std.View, LatentChannels, vox, vox, 1, affine: true);
            _lk.ChannelStats(f.View, us.View, mean.View, std.View, LatentChannels, vox2, 1, LatentChannels, affine: false);
        }
        var o = _acc.Allocate1D<float>((long)LatentChannels * vox2);
        _lk.AdainApply(f.View, rs.View, us.View, mean.View, std.View, o.View, LatentChannels, vox2, adainFactor);
        await _acc.SynchronizeAsync();                                                   // f back to the pool; mean, std, rs, us freed
        Drop(f); Flush();
        return o;
    }

    public void Dispose()
    {
        _temps.Dispose();
        foreach (var t in _tables.Values) t.Dispose();
        foreach (var (w, _) in _convs.Values) w.Dispose();
        foreach (var n in _norms.Values) n.Dispose();
        _tables.Clear(); _convs.Clear(); _norms.Clear();
        _sw.Dispose();
    }
}
