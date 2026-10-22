using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.Wan;

namespace SpawnDev.ILGPU.ML.Ltx;

using FV = ArrayView1D<float, Stride1D.Dense>;
using Buf = MemoryBuffer1D<float, Stride1D.Dense>;

/// <summary>
/// The LTX-Video 0.9.8 VAE decoder (<c>AutoencoderKLLTXVideo</c> with timestep conditioning, 128 latent channels, 32x spatial and 8x temporal compression): the latent [128, Fz, h, w]
/// becomes video [3, 8 (Fz - 1) + 1, 32 h, 32 w] in [-1, 1]. Denormalise (z * std + mean), a 3x3x3 convolution to 1024 channels, a mid block of five resnets, then three stages of
/// (residual 2x2x2 upsampler that drops its first doubled frame, five resnets), a modulated pixel norm and a convolution to 3 x 4 x 4 channels that is unpatchified (the 4x4 spatial patch).
/// A resnet is <c>pixel norm, * (1 + scale) + shift, SiLU, conv, pixel norm, * (1 + scale) + shift, SiLU, conv, + input</c> where the four shift / scale vectors come from a table plus the
/// block's embedding of the (scaled) decode timestep. The convolutions are NOT causal: the temporal padding repeats the first / last frame, the spatial one is zeros.
/// Implemented like the Wan VAE: channel-last fp32 activations [voxels, C], fp16 weights, every convolution a gathered-row implicit GEMM (<see cref="SparseKernels.ConvTaps"/>) over
/// neighbour tables built on the device and shared by all convolutions of a shape. The whole clip is decoded at once. Weights: <c>tools/ltx-harness/reference/export_vae_f16.py</c>.
/// </summary>
public sealed class LtxVae : IDisposable
{
    private readonly Accelerator _acc;
    private readonly GpuWeights _w;
    private readonly SparseKernels _sk;
    private readonly LtxKernels _lk;
    private readonly DiTKernels _dk;
    private readonly List<Buf> _live = new();
    private readonly List<Buf> _drop = new();
    private readonly Dictionary<(int T, int H, int W), MemoryBuffer1D<int, Stride1D.Dense>> _tables = new();

    public const int LatentChannels = 128, Patch = 4;
    private const float PixelNormEps = 1e-8f;

    /// <summary>Test hook: (stage name, activation view, frames, height, width, channels); valid only during the call.</summary>
    public Action<string, FV, int, int, int, int>? Tap { get; set; }

    public LtxVae(Accelerator accelerator, GpuWeights weights)
    {
        _acc = accelerator; _w = weights; _sk = new SparseKernels(accelerator); _lk = new LtxKernels(accelerator); _dk = new DiTKernels(accelerator);
    }

    // temporaries are queued and freed together after a fence: on WebGPU a buffer must outlive the dispatches that read it
    private Buf Tmp(long n) { var b = _acc.Allocate1D<float>(n); _live.Add(b); return b; }
    private void Drop(Buf b) { _drop.Add(b); }
    private void Flush()
    {
        if (_drop.Count == 0) return;
        _acc.Fence();
        foreach (var b in _drop) { _live.Remove(b); b.Dispose(); }
        _drop.Clear();
    }

    private MemoryBuffer1D<int, Stride1D.Dense> Table(int T, int H, int W)
    {
        if (_tables.TryGetValue((T, H, W), out var t)) return t;
        t = _acc.Allocate1D<int>((long)T * H * W * 27);
        _lk.ReplicateTable(t.View, T, H, W);
        _tables[(T, H, W)] = t;
        return t;
    }

    private Buf Conv(string name, Buf x, int T, int H, int W, int cin, int cout)
    {
        var y = Tmp((long)T * H * W * cout);
        _sk.ConvTaps(x.View, Table(T, H, W).View, _w.F16(name + ".weight"), _w.F16(name + ".bias"), y.View, T * H * W, cin, cout, 27);
        return y;
    }

    /// <summary>The block's embedding of the scaled timestep: sinusoid (256, cos then sin), Linear, SiLU, Linear -> [dim] (PixArtAlphaCombinedTimestepSizeEmbeddings without size embeddings).</summary>
    private Buf TimeEmbed(string prefix, double scaledT, int dim)
    {
        var s = Tmp(256); s.View.CopyFromCPU(WanKernels.SinusoidalEmbedding(scaledT, 256));
        var h1 = Tmp(dim); _sk.LinearF16(s.View, _w.F16(prefix + "0.weight"), _w.F16(prefix + "0.bias"), h1.View, 1, 256, dim);
        var h2 = Tmp(dim); _dk.Silu(h1.View, h2.View, dim);
        var e = Tmp(dim); _sk.LinearF16(h2.View, _w.F16(prefix + "2.weight"), _w.F16(prefix + "2.bias"), e.View, 1, dim, dim);
        Drop(s); Drop(h1); Drop(h2);
        return e;
    }

    /// <summary>One resnet (consumes <paramref name="x"/>): temb is the block's [4C] embedding.</summary>
    private Buf Resnet(string p, Buf x, Buf temb, int T, int H, int W, int C)
    {
        int rows = T * H * W;
        var mod = Tmp(4L * C); _lk.Add(_w.F32(p + "scale_shift_table"), temb.View, mod.View, 4 * C);       // shift_1, scale_1, shift_2, scale_2
        var a = Tmp((long)rows * C); _lk.RmsNormModulate(x.View, a.View, mod.View, 0 * C, 1 * C, rows, C, PixelNormEps, silu: true);
        var c1 = Conv(p + "conv1", a, T, H, W, C, C); Drop(a);
        var b = Tmp((long)rows * C); _lk.RmsNormModulate(c1.View, b.View, mod.View, 2 * C, 3 * C, rows, C, PixelNormEps, silu: true); Drop(c1);
        var c2 = Conv(p + "conv2", b, T, H, W, C, C); Drop(b);
        _dk.GateResidual(c2.View, x.View, default, -1, rows * C, C);                                          // c2 += x
        Drop(x); Drop(mod);
        Flush();
        return c2;
    }

    private Buf Resnets(string block, Buf x, Buf temb, int T, int H, int W, int C, int count)
    {
        for (int j = 0; j < count; j++) x = Resnet($"{block}.res_blocks.{j}.", x, temb, T, H, W, C);
        return x;
    }

    /// <summary>The residual 2x2x2 upsampler (consumes <paramref name="x"/>): [T, H, W, cin] -> [2T - 1, 2H, 2W, cin / 2].</summary>
    private Buf Upsample(string p, Buf x, ref int T, ref int H, ref int W, int cin)
    {
        int cout = cin / 2;
        var y = Conv(p + "conv", x, T, H, W, cin, cin * 4);                                                    // cin * 8 / upscale_factor 2 channels, 8 sub-positions of cout each
        var o = Tmp((long)(2 * T - 1) * 4 * H * W * cout);
        _lk.UpShuffle(y.View, x.View, o.View, T, H, W, cout, cin);
        Drop(y); Drop(x);
        T = 2 * T - 1; H *= 2; W *= 2;
        Flush();
        return o;
    }

    private void Tapped(string name, Buf x, int T, int H, int W, int C) => Tap?.Invoke(name, x.View, T, H, W, C);

    /// <summary>
    /// Decode the latent <paramref name="z"/> ([128, Fz, h, w] channel-first, the model-space latent the DiT predicts, before denormalisation; add the decode noise yourself) with the
    /// decode timestep <paramref name="timestep"/> (the 0.9.8 recipe: 0.05). Returns the video [3, 8 (Fz - 1) + 1, 32 h, 32 w] channel-first, nominally in [-1, 1] (not clamped).
    /// </summary>
    public async Task<float[]> DecodeAsync(float[] z, int Fz, int h, int w, float timestep = 0.05f, float noiseScale = 0f, float[]? noise = null)
    {
        int T = Fz, H = h, W = w, vox = T * H * W;
        if (z.Length != (long)LatentChannels * vox) throw new ArgumentException("the latent must be [128, Fz, h, w]");
        if (noiseScale != 0f && (noise == null || noise.Length != z.Length)) throw new ArgumentException("a decode noise scale needs standard normal noise shaped like the latent");
        var scaleHost = await _w.F32("timestep_scale_multiplier").CopyToHostAsync();
        double tScaled = timestep * scaleHost[0];

        using var zIn = _acc.Allocate1D(z);
        using var nIn = noiseScale != 0f ? _acc.Allocate1D(noise!) : null;
        var x0 = Tmp((long)vox * LatentChannels);
        _lk.LatentToVoxels(zIn.View, x0.View, _w.F32("latents_mean"), _w.F32("latents_std"), nIn?.View ?? zIn.View, vox, LatentChannels, 1f, noiseScale);
        var x = Conv("conv_in", x0, T, H, W, LatentChannels, 1024); Drop(x0); Flush();
        Tapped("conv_in", x, T, H, W, 1024);

        // mid block: five resnets at 1024 channels
        var temb = TimeEmbed("up_blocks.0.time_embedder.", tScaled, 4 * 1024);
        x = Resnets("up_blocks.0", x, temb, T, H, W, 1024, 5); Drop(temb); Flush();
        Tapped("mid_block", x, T, H, W, 1024);

        // three stages: (upsampler at up_blocks.{1,3,5}, five resnets at up_blocks.{2,4,6})
        int c = 1024;
        for (int stage = 0; stage < 3; stage++)
        {
            x = Upsample($"up_blocks.{2 * stage + 1}.", x, ref T, ref H, ref W, c);
            c /= 2;
            temb = TimeEmbed($"up_blocks.{2 * stage + 2}.time_embedder.", tScaled, 4 * c);
            x = Resnets($"up_blocks.{2 * stage + 2}", x, temb, T, H, W, c, 5); Drop(temb); Flush();
            Tapped($"up_{stage}", x, T, H, W, c);
        }

        // output: pixel norm modulated by (last_scale_shift_table + embedding), SiLU, conv to 48 channels, unpatchify
        int rows = T * H * W;
        var lastT = TimeEmbed("last_time_embedder.", tScaled, 2 * c);
        var mod = Tmp(2L * c); _lk.Add(_w.F32("last_scale_shift_table"), lastT.View, mod.View, 2 * c);        // shift, scale
        var n = Tmp((long)rows * c); _lk.RmsNormModulate(x.View, n.View, mod.View, 0, c, rows, c, PixelNormEps, silu: true);
        Drop(x); Drop(lastT); Drop(mod); Flush();
        var head = Conv("conv_out", n, T, H, W, c, 3 * Patch * Patch); Drop(n); Flush();
        Tapped("conv_out", head, T, H, W, 3 * Patch * Patch);
        var video = Tmp(3L * T * Patch * H * Patch * W);
        _lk.UnpatchifyVideo(head.View, video.View, T, H, W);
        Drop(head);
        await _acc.SynchronizeAsync();
        var result = await video.View.CopyToHostAsync();
        Drop(video); Flush();
        return result;
    }

    public void Dispose()
    {
        foreach (var b in _live.ToArray()) b.Dispose();
        _live.Clear(); _drop.Clear();
        foreach (var t in _tables.Values) t.Dispose();
        _tables.Clear();
    }
}
