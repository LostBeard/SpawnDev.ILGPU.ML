using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Ltx;

using FV = ArrayView1D<float, Stride1D.Dense>;
using Buf = MemoryBuffer1D<float, Stride1D.Dense>;

/// <summary>
/// The LTX-Video 0.9.8 spatial latent upscaler (<c>ltxv-spatial-upscaler-0.9.8</c>, diffusers' <c>LTXLatentUpsamplerModel</c> inside <c>LTXLatentUpsamplePipeline</c>): a model-space
/// latent [128, F, h, w] becomes [128, F, 2h, 2w], the start of the distilled recipe's second pass. The latent is denormalised (z * std + mean), then a 3x3x3 convolution to 512
/// channels, GroupNorm(32) and SiLU, four resblocks (<c>conv, GroupNorm, SiLU, conv, GroupNorm, + input, SiLU</c>), a per-frame 3x3 convolution to 4 x 512 channels with a 2x2
/// pixel shuffle, four more resblocks, a 3x3x3 convolution back to 128; the convolutions pad with zeros (not causal). Then AdaIN (each channel's mean and standard deviation
/// moved to the input's, blended by <c>adainFactor</c>; the recipe uses 1) and the normalisation back to model space. Same scheme as <see cref="LtxVae"/> (channel-last fp32
/// activations, fp16 weights, gathered-row implicit GEMMs). Weights: <c>tools/ltx-harness/reference/export_upsampler_f16.py</c> (253 MB).
/// </summary>
public sealed class LtxLatentUpsampler : IDisposable
{
    private readonly Accelerator _acc;
    private readonly GpuWeights _w;
    private readonly SparseKernels _sk;
    private readonly SdKernels _sd;
    private readonly LtxKernels _lk;
    private readonly ActivationPool _temps;
    private readonly Dictionary<(int Taps, int T, int H, int W), MemoryBuffer1D<int, Stride1D.Dense>> _tables = new();
    private readonly int _mid, _blocks;

    private const int LatentChannels = 128;
    private const float GroupNormEps = 1e-5f;

    /// <summary>Test hook: (stage name, activation view, frames, height, width, channels); valid only during the call.</summary>
    public Action<string, FV, int, int, int, int>? Tap { get; set; }

    public LtxLatentUpsampler(Accelerator accelerator, GpuWeights weights)
    {
        _acc = accelerator; _w = weights; _temps = new ActivationPool(accelerator); _sk = new SparseKernels(accelerator); _sd = new SdKernels(accelerator); _lk = new LtxKernels(accelerator);
        _mid = int.Parse(weights.Metadata.GetValueOrDefault("mid_channels", "512"));
        _blocks = int.Parse(weights.Metadata.GetValueOrDefault("num_blocks_per_stage", "4"));
        if (weights.Metadata.GetValueOrDefault("dims", "3") != "3") throw new NotSupportedException("LtxLatentUpsampler: the 3D upsampler (dims 3) only");
    }

    // temporaries: see ActivationPool (large activations recycled by size, small ones freed after a fence)
    private Buf Tmp(long n) => _temps.Get(n);
    private void Drop(Buf b) => _temps.Drop(b);
    private void Flush() => _temps.Flush();

    private MemoryBuffer1D<int, Stride1D.Dense> Table(int taps, int T, int H, int W)
    {
        if (_tables.TryGetValue((taps, T, H, W), out var t)) return t;
        t = _acc.Allocate1D<int>((long)T * H * W * 27);                                // ConvTaps reads rows of stride 27 whatever the tap count
        if (taps == 27) _lk.ZeroTable3(t.View, T, H, W); else _lk.ZeroTable2(t.View, T, H, W);
        _tables[(taps, T, H, W)] = t;
        return t;
    }

    private Buf Conv(string name, Buf x, int T, int H, int W, int cin, int cout, int taps = 27)
    {
        var y = Tmp((long)T * H * W * cout);
        _sk.ConvTaps(x.View, Table(taps, T, H, W).View, _w.F16(name + ".weight"), _w.F16(name + ".bias"), y.View, T * H * W, cin, cout, taps);
        return y;
    }

    private Buf Norm(string name, Buf x, int rows, int C, bool silu)
    {
        var o = Tmp((long)rows * C);
        _sd.GroupNorm(x.View, o.View, _w.F16(name + ".weight"), _w.F16(name + ".bias"), rows, C, 32, GroupNormEps, silu);
        return o;
    }

    /// <summary>One resblock (consumes <paramref name="x"/>).</summary>
    private Buf ResBlock(string p, Buf x, int T, int H, int W)
    {
        int rows = T * H * W, C = _mid;
        var c1 = Conv(p + "conv1", x, T, H, W, C, C);
        var n1 = Norm(p + "norm1", c1, rows, C, silu: true); Drop(c1);
        var c2 = Conv(p + "conv2", n1, T, H, W, C, C); Drop(n1);
        var n2 = Norm(p + "norm2", c2, rows, C, silu: false); Drop(c2);
        var o = Tmp((long)rows * C);
        _lk.AddSilu(n2.View, x.View, o.View, rows * C);
        Drop(n2); Drop(x);
        Flush();
        return o;
    }

    private void Tapped(string name, Buf x, int T, int H, int W, int C) => Tap?.Invoke(name, x.View, T, H, W, C);

    /// <summary>
    /// Upsample the model-space latent <paramref name="z"/> ([128, <paramref name="frames"/>, <paramref name="h"/>, <paramref name="w"/>] channel-first) to [128, frames, 2h, 2w] in model
    /// space. <paramref name="adainFactor"/>: 1 = each channel takes the input's mean and standard deviation (the 0.9.8 recipe), 0 = the upsampler's output as it is.
    /// </summary>
    public async Task<float[]> UpsampleAsync(float[] z, int frames, int h, int w, float adainFactor = 1f)
    {
        int T = frames, H = h, W = w, vox = T * H * W;
        if (z.Length != (long)LatentChannels * vox) throw new ArgumentException("the latent must be [128, frames, h, w]");
        var meanH = await _w.F32("latents_mean").CopyToHostAsync(); var stdH = await _w.F32("latents_std").CopyToHostAsync();

        using var zIn = _acc.Allocate1D(z);
        var x0 = Tmp((long)vox * LatentChannels);
        _lk.LatentToVoxels(zIn.View, x0.View, _w.F32("latents_mean"), _w.F32("latents_std"), null, vox, LatentChannels, 1f, 0f);
        var c = Conv("initial_conv", x0, T, H, W, LatentChannels, _mid); Drop(x0);
        var x = Norm("initial_norm", c, vox, _mid, silu: true); Drop(c); Flush();
        Tapped("initial", x, T, H, W, _mid);
        for (int j = 0; j < _blocks; j++) x = ResBlock($"res_blocks.{j}.", x, T, H, W);
        Tapped("res", x, T, H, W, _mid);

        var u = Conv("upsampler.0", x, T, H, W, _mid, 4 * _mid, taps: 9); Drop(x);
        x = Tmp((long)vox * 4 * _mid);
        _lk.PixelShuffle2(u.View, x.View, T, H, W, _mid);
        Drop(u); Flush();
        H *= 2; W *= 2;
        Tapped("upsampled", x, T, H, W, _mid);
        for (int j = 0; j < _blocks; j++) x = ResBlock($"post_upsample_res_blocks.{j}.", x, T, H, W);
        Tapped("post", x, T, H, W, _mid);
        var f = Conv("final_conv", x, T, H, W, _mid, LatentChannels); Drop(x); Flush();
        Tapped("final", f, T, H, W, LatentChannels);
        await _acc.SynchronizeAsync();
        var up = await f.View.CopyToHostAsync();                                          // [vox2, 128] channel-last, denormalised
        Drop(f); Flush();

        // AdaIN against the denormalised input (torch.std_mean: the unbiased standard deviation), then back to model space, channel-first
        int vox2 = T * H * W;
        var o = new float[(long)LatentChannels * vox2];
        for (int ch = 0; ch < LatentChannels; ch++)
        {
            double rs = 0, rq = 0, us = 0, uq = 0;
            for (int v = 0; v < vox; v++) { double r = z[(long)ch * vox + v] * stdH[ch] + meanH[ch]; rs += r; rq += r * r; }
            for (int v = 0; v < vox2; v++) { double a = up[(long)v * LatentChannels + ch]; us += a; uq += a * a; }
            double rMean = rs / vox, rSd = Math.Sqrt(Math.Max(0, (rq - rs * rMean) / (vox - 1)));
            double uMean = us / vox2, uSd = Math.Sqrt(Math.Max(0, (uq - us * uMean) / (vox2 - 1)));
            for (int v = 0; v < vox2; v++)
            {
                double a = up[(long)v * LatentChannels + ch];
                double adain = (a - uMean) / uSd * rSd + rMean;
                double blended = a + adainFactor * (adain - a);                           // torch.lerp(latents, result, factor)
                o[(long)ch * vox2 + v] = (float)((blended - meanH[ch]) / stdH[ch]);
            }
        }
        return o;
    }

    public void Dispose()
    {
        _temps.Dispose();
        foreach (var t in _tables.Values) t.Dispose();
        _tables.Clear();
    }
}
