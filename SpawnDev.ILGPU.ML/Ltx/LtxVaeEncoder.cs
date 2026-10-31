using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Ltx;

using FV = ArrayView1D<float, Stride1D.Dense>;
using Buf = MemoryBuffer1D<float, Stride1D.Dense>;

/// <summary>
/// The LTX-Video 0.9.8 VAE encoder (<c>LTXVideoEncoder3d</c> with the 0.9.5 down blocks): video [3, F, 32 h, 32 w] in [-1, 1] (F = 8 n + 1) becomes the normalised latent
/// [128, (F - 1) / 8 + 1, h, w] that LTXConditionPipeline puts in the conditioned latent frames (image-to-video: one frame; video conditioning: a clip).
/// The 4x4 pixel patch to 48 channels, a causal 3x3x3 convolution to 128 channels, then four stages of (resnets, residual space-to-depth downsampler): 4 resnets at 128 then 2x2 spatial
/// to 256, 6 at 256 then 2x temporal to 512, 6 at 512 then 2x2x2 to 1024, 2 at 1024 then 2x2x2 to 2048; a mid block of 2 resnets at 2048; pixel norm, SiLU and a convolution to
/// 128 means + one log-variance. A resnet is <c>pixel norm, SiLU, conv, pixel norm, SiLU, conv, + input</c> (no timestep conditioning, no channel change).
/// Every convolution is CAUSAL: the temporal padding is two copies of the first frame in front, the spatial one zeros; a temporal downsampler first repeats the first frame once more.
/// Same implementation scheme as <see cref="LtxVae"/> (channel-last fp32 activations, fp16 weights, gathered-row implicit GEMMs). Weights: <c>tools/ltx-harness/reference/export_vae_encoder_f16.py</c>.
/// </summary>
public sealed class LtxVaeEncoder : IDisposable
{
    private readonly Accelerator _acc;
    private readonly GpuWeights _w;
    private readonly SparseKernels _sk;
    private readonly LtxKernels _lk;
    private readonly DiTKernels _dk;
    private readonly ActivationPool _temps;
    private readonly Dictionary<(int T, int H, int W, int Shift), MemoryBuffer1D<int, Stride1D.Dense>> _tables = new();
    private Buf? _zeros;

    private const float PixelNormEps = 1e-8f;

    /// <summary>(resnets, output channels, temporal stride, spatial stride) of the four down stages; the stage's resnets run at the previous width.</summary>
    private static readonly (int Resnets, int Cout, int St, int Ss)[] Stages = { (4, 256, 1, 2), (6, 512, 2, 1), (6, 1024, 2, 2), (2, 2048, 2, 2) };

    /// <summary>Test hook: (stage name, activation view, frames, height, width, channels); valid only during the call.</summary>
    public Action<string, FV, int, int, int, int>? Tap { get; set; }

    public LtxVaeEncoder(Accelerator accelerator, GpuWeights weights)
    {
        _acc = accelerator; _w = weights; _temps = new ActivationPool(accelerator); _sk = new SparseKernels(accelerator); _lk = new LtxKernels(accelerator); _dk = new DiTKernels(accelerator);
    }

    // temporaries: see ActivationPool (large activations recycled by size, small ones freed after a fence)
    private Buf Tmp(long n) => _temps.Get(n);
    private void Drop(Buf b) => _temps.Drop(b);
    private void Flush() => _temps.Flush();

    private MemoryBuffer1D<int, Stride1D.Dense> Table(int T, int H, int W, int shift)
    {
        if (_tables.TryGetValue((T, H, W, shift), out var t)) return t;
        t = _acc.Allocate1D<int>((long)(T + shift) * H * W * 27);
        _lk.CausalTable(t.View, T, H, W, shift);
        _tables[(T, H, W, shift)] = t;
        return t;
    }

    /// <summary>A causal 3x3x3 convolution of the source grid T x H x W (with <paramref name="shift"/> extra copies of the first frame in front): [(T + shift) H W, cout].</summary>
    private Buf Conv(string name, Buf x, int T, int H, int W, int cin, int cout, int shift = 0)
    {
        int m = (T + shift) * H * W;
        var y = Tmp((long)m * cout);
        _sk.ConvTaps(x.View, Table(T, H, W, shift).View, _w.F16(name + ".weight"), _w.F16(name + ".bias"), y.View, m, cin, cout, 27);
        return y;
    }

    /// <summary>Pixel norm (RMS over the channels, no weight) then SiLU: the modulated norm kernel with a zero modulation.</summary>
    private Buf NormSilu(Buf x, int rows, int C)
    {
        var o = Tmp((long)rows * C);
        _lk.RmsNormModulate(x.View, o.View, _zeros!.View, 0, 0, rows, C, PixelNormEps, silu: true);
        return o;
    }

    /// <summary>One resnet (consumes <paramref name="x"/>).</summary>
    private Buf Resnet(string p, Buf x, int T, int H, int W, int C)
    {
        int rows = T * H * W;
        var a = NormSilu(x, rows, C);
        var c1 = Conv(p + "conv1", a, T, H, W, C, C); Drop(a); Flush();
        var b = NormSilu(c1, rows, C); Drop(c1); Flush();
        var c2 = Conv(p + "conv2", b, T, H, W, C, C); Drop(b); Flush();
        _dk.GateResidual(c2.View, x.View, default, -1, rows * C, C);                                          // c2 += x
        Drop(x);
        Flush();
        return c2;
    }

    /// <summary>The residual downsampler (consumes <paramref name="x"/>): [T, H, W, cin] -> [(T + st - 1) / st, H / ss, W / ss, cout].</summary>
    private Buf Downsample(string p, Buf x, ref int T, ref int H, ref int W, int cin, int cout, int st, int ss)
    {
        int S = st * ss * ss;
        var y = Conv(p, x, T, H, W, cin, cout / S, shift: st - 1);
        int To = (T + st - 1) / st, Ho = H / ss, Wo = W / ss;
        var o = Tmp((long)To * Ho * Wo * cout);
        _lk.DownShuffle(y.View, x.View, o.View, T, H, W, cout, cin, st, ss, ss);
        Drop(y); Drop(x);
        T = To; H = Ho; W = Wo;
        Flush();
        return o;
    }

    private void Tapped(string name, Buf x, int T, int H, int W, int C) => Tap?.Invoke(name, x.View, T, H, W, C);

    /// <summary>
    /// Encode <paramref name="video"/> ([3, <paramref name="frames"/>, <paramref name="height"/>, <paramref name="width"/>] channel-first in [-1, 1]; frames 8 n + 1, sides multiples of 32) to the
    /// normalised latent [128, (frames - 1) / 8 + 1, height / 32, width / 32]: <c>(z - latents_mean) / latents_std</c> where z is the posterior mean, or with <paramref name="noise"/>
    /// (standard normal, shaped like the latent) a sample of it (<c>mean + exp(logvar / 2) * noise</c>, what the reference pipeline draws with its generator).
    /// </summary>
    public async Task<float[]> EncodeAsync(float[] video, int frames, int height, int width, float[]? noise = null)
    {
        if ((frames - 1) % 8 != 0 || frames < 1) throw new ArgumentException("the frame count must be 8n + 1");
        if (height % 32 != 0 || width % 32 != 0) throw new ArgumentException("height and width must be multiples of 32");
        if (video.Length != 3L * frames * height * width) throw new ArgumentException("the video must be [3, frames, height, width]");
        int T = frames, H = height / LtxVae.Patch, W = width / LtxVae.Patch;
        int latentLength = 128 * ((frames - 1) / 8 + 1) * (height / 32) * (width / 32);
        if (noise != null && noise.Length != latentLength) throw new ArgumentException("the sampling noise must be shaped like the latent");
        _zeros ??= _acc.Allocate1D(new float[2048]);

        using var vIn = _acc.Allocate1D(video);
        var x0 = Tmp((long)T * H * W * 48);
        _lk.PatchifyVideo(vIn.View, x0.View, T, H, W);
        var x = Conv("conv_in", x0, T, H, W, 48, 128); Drop(x0); Flush();
        Tapped("conv_in", x, T, H, W, 128);

        int c = 128;
        for (int s = 0; s < Stages.Length; s++)
        {
            var (resnets, cout, st, ss) = Stages[s];
            for (int j = 0; j < resnets; j++) x = Resnet($"down_blocks.{2 * s}.res_blocks.{j}.", x, T, H, W, c);
            x = Downsample($"down_blocks.{2 * s + 1}.conv", x, ref T, ref H, ref W, c, cout, st, ss);
            c = cout;
            Tapped($"down_{s}", x, T, H, W, c);
        }
        for (int j = 0; j < 2; j++) x = Resnet($"down_blocks.8.res_blocks.{j}.", x, T, H, W, c);
        Tapped("mid_block", x, T, H, W, c);

        int vox = T * H * W;
        var n = NormSilu(x, vox, c); Drop(x);
        var head = Conv("conv_out", n, T, H, W, c, 129); Drop(n); Flush();
        Tapped("conv_out", head, T, H, W, 129);
        using var nIn = noise != null ? _acc.Allocate1D(noise) : null;
        var latent = Tmp(128L * vox);
        _lk.Posterior(head.View, latent.View, _w.F32("latents_mean"), _w.F32("latents_std"), nIn?.View, vox);
        Drop(head);
        await _acc.SynchronizeAsync();
        var result = await latent.View.CopyToHostAsync();
        Drop(latent); Flush();
        return result;
    }

    public void Dispose()
    {
        _temps.Dispose();
        foreach (var t in _tables.Values) t.Dispose();
        _tables.Clear();
        _zeros?.Dispose(); _zeros = null;
    }
}
