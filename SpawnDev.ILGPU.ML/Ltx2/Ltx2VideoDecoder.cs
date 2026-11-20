using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Ltx2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using BV = ArrayView1D<BFloat16, Stride1D.Dense>;
using Buf = MemoryBuffer1D<float, Stride1D.Dense>;

/// <summary>
/// The LTX-2 video VAE decoder (the checkpoint's <c>vae.decoder</c>: diffusers' LTX2 video decoder, Apache-2.0, with the checkpoint's own config: causal_decoder false, no timestep
/// conditioning, decoder blocks res_x 5 / compress_all x2 alternating): the latent [128, Fz, h, w] becomes video [3, 8 (Fz - 1) + 1, 32 h, 32 w] in [-1, 1]. Denormalise
/// (z * std-of-means + mean-of-means), a 3x3x3 convolution to 1024 channels, five resnets, then three stages of (residual 2x2x2 upsampler that drops its first doubled frame,
/// five resnets) down to 128 channels, pixel norm, SiLU, a convolution to 3 x 4 x 4 channels, unpatchified. A resnet is <c>pixel norm, SiLU, conv, pixel norm, SiLU, conv, + input</c>.
/// The convolutions pad by REFLECTION in space (LTX-Video 0.9.8's decoder padded with zeros) and by repeating the first / last frame in time.
/// <para>Channel-last fp32 activations; every convolution a gathered implicit GEMM on the checkpoint's exact bf16 weights (<see cref="SparseKernels.ConvTapsBf16Exact"/>).</para>
/// </summary>
public sealed class Ltx2VideoDecoder : IDisposable
{
    public const int LatentChannels = 128, Patch = 4, ResnetsPerBlock = 5;
    private const float PixelNormEps = 1e-8f;
    private const string P = "vae.decoder.";

    private readonly Accelerator _acc;
    private readonly StreamedModelWeights _sw;
    private readonly SparseKernels _sk;
    private readonly Ltx.LtxKernels _lk;
    private readonly Ltx2Kernels _l;
    private readonly DiTKernels _dk;
    private readonly ActivationPool _temps;
    private readonly Dictionary<string, MemoryBuffer1D<BFloat16, Stride1D.Dense>> _convs = new();
    private readonly Dictionary<(int T, int H, int W), MemoryBuffer1D<int, Stride1D.Dense>> _tables = new();
    private readonly Buf _mean, _std;

    /// <summary>Test hook: (stage name, activation view, frames, height, width, channels); valid only during the call.</summary>
    public Action<string, FV, int, int, int, int>? Tap { get; set; }

    private Ltx2VideoDecoder(Accelerator accelerator, StreamedModelWeights weights, Buf mean, Buf std)
    {
        _acc = accelerator; _sw = weights; _mean = mean; _std = std;
        _temps = new ActivationPool(accelerator); _sk = new SparseKernels(accelerator); _lk = new Ltx.LtxKernels(accelerator); _l = new Ltx2Kernels(accelerator); _dk = new DiTKernels(accelerator);
    }

    /// <summary>Load the video decoder from an LTX-2 checkpoint (only <c>vae.decoder.*</c> and the latent statistics are read).</summary>
    public static async Task<Ltx2VideoDecoder> OpenAsync(Accelerator accelerator, IModelSource source, string repo, string file, CancellationToken ct = default)
    {
        static int BlockOf(string n) => n.StartsWith(P, StringComparison.Ordinal) || n.StartsWith("vae.per_channel_statistics.", StringComparison.Ordinal) ? -1 : StreamedModelWeights.HostOnly;
        var w = await StreamedModelWeights.OpenAsync(accelerator, source, repo, file, P + "conv_in.conv.weight", BlockOf, 0, 0, null, ct);
        try
        {
            var b = w.Base;
            static float[] F(ushort[] u) { var o = new float[u.Length]; for (int i = 0; i < u.Length; i++) o[i] = BitConverter.Int32BitsToSingle(u[i] << 16); return o; }
            async Task<float[]> Stat(string name) => F((await b.BF16("vae.per_channel_statistics." + name).Cast<BFloat16, ushort>().CopyToHostAsync()));
            var mean = accelerator.Allocate1D(await Stat("mean-of-means"));
            var std = accelerator.Allocate1D(await Stat("std-of-means"));
            var d = new Ltx2VideoDecoder(accelerator, w, mean, std);
            await d.PrepareAsync();
            return d;
        }
        catch { w.Dispose(); throw; }
    }

    // every 3x3x3 convolution weight into the gather layout (bf16, exact); the checkpoint's copies are freed
    private async Task PrepareAsync()
    {
        var b = _sw.Base;
        var names = b.Names.Where(n => n.StartsWith(P, StringComparison.Ordinal) && n.EndsWith(".conv.weight", StringComparison.Ordinal)).ToList();
        foreach (var n in names)
        {
            var s = b.Shape(n);
            if (s.Length != 5 || s[2] != 3 || s[3] != 3 || s[4] != 3) throw new InvalidDataException($"{n}: [{string.Join(", ", s)}], expected a 3x3x3 Conv3d weight");
            var g = _acc.Allocate1D<BFloat16>((long)s[0] * 27 * s[1]);
            _l.GatherLayoutBf16(b.BF16(n), g.View, s[0], s[1], 27);
            _convs[n[..^".weight".Length]] = g;
        }
        await _acc.SynchronizeAsync();
        foreach (var n in names) b.Remove(n)?.Dispose();
    }

    private Buf Tmp(long n) => _temps.Get(n);
    private void Drop(Buf x) => _temps.Drop(x);
    private void Flush() => _temps.Flush();

    private MemoryBuffer1D<int, Stride1D.Dense> Table(int T, int H, int W)
    {
        if (_tables.TryGetValue((T, H, W), out var t)) return t;
        t = _acc.Allocate1D<int>((long)T * H * W * 27);
        _l.ReflectTable(t.View, T, H, W);
        _tables[(T, H, W)] = t;
        return t;
    }

    private Buf Conv(string name, Buf x, int T, int H, int W, int cin, int cout)
    {
        int rows = T * H * W;
        if (!_sk.ConvTapsBf16ExactOk(rows, 27, cin, cout, rows)) throw new NotSupportedException($"{name}: {T} x {H} x {W} x {cin} is beyond one exact GEMM (decode a smaller clip or tile it)");
        var y = Tmp((long)rows * cout);
        _sk.ConvTapsBf16Exact(x.View, Table(T, H, W).View, _convs[P + name + ".conv"].View, _sw.Base.BF16(P + name + ".conv.bias"), y.View, rows, cin, cout, 27);
        return y;
    }

    // one resnet (consumes x)
    private Buf Resnet(string p, Buf x, int T, int H, int W, int C)
    {
        int rows = T * H * W;
        var a = Tmp((long)rows * C); _l.PixelNormSilu(x.View, a.View, rows, C, PixelNormEps);
        var c1 = Conv(p + "conv1", a, T, H, W, C, C); Drop(a); Flush();
        var b = Tmp((long)rows * C); _l.PixelNormSilu(c1.View, b.View, rows, C, PixelNormEps); Drop(c1); Flush();
        var c2 = Conv(p + "conv2", b, T, H, W, C, C); Drop(b); Flush();
        _dk.GateResidual(c2.View, x.View, default, -1, rows * C, C);                                          // c2 += x
        Drop(x); Flush();
        return c2;
    }

    // the residual 2x2x2 upsampler (consumes x): [T, H, W, cin] -> [2T - 1, 2H, 2W, cin / 2]
    private Buf Upsample(string p, Buf x, ref int T, ref int H, ref int W, int cin)
    {
        int cout = cin / 2;
        var y = Conv(p + "conv", x, T, H, W, cin, cin * 4);
        var o = Tmp((long)(2 * T - 1) * 4 * H * W * cout);
        _lk.UpShuffle(y.View, x.View, o.View, T, H, W, cout, cin);
        Drop(y); Drop(x);
        T = 2 * T - 1; H *= 2; W *= 2;
        Flush();
        return o;
    }

    /// <summary>
    /// Decode <paramref name="z"/> ([128, Fz, h, w] channel-first, the model-space latent the transformer works in). Returns the video [3, 8 (Fz - 1) + 1, 32 h, 32 w] channel-first,
    /// nominally in [-1, 1] (not clamped).
    /// </summary>
    public async Task<float[]> DecodeAsync(float[] z, int Fz, int h, int w)
    {
        var video = await DecodeToDeviceAsync(z, Fz, h, w);
        var result = await video.View.SubView(0, 3L * (8 * (Fz - 1) + 1) * 32 * h * 32 * w).CopyToHostAsync();
        Drop(video); Flush();
        return result;
    }

    /// <summary>
    /// <see cref="DecodeAsync"/> to RGBA bytes, one array per frame (converted on the device and read back a frame at a time: no float array of the whole clip on the host;
    /// a 768 x 512 x 121 clip is 566 MB of floats). The conversion is <see cref="Wan.WanVideo.GetFrameRgba"/>'s: round((v + 1) * 127.5) clamped.
    /// </summary>
    public async Task<byte[][]> DecodeRgbaFramesAsync(float[] z, int Fz, int h, int w)
    {
        var video = await DecodeToDeviceAsync(z, Fz, h, w);
        int frames = 8 * (Fz - 1) + 1, pixels = 32 * h * 32 * w;                                    // 32 x 32 pixels per latent position (the 4 x 4 patch after three 2x upsamplers)
        var o = new byte[frames][];
        using var packed = _acc.Allocate1D<int>(pixels);
        for (int f = 0; f < frames; f++)
        {
            _lk.FrameRgba(video.View, packed.View, frames, pixels, f);
            var ints = await packed.View.CopyToHostAsync();
            o[f] = new byte[pixels * 4];
            Buffer.BlockCopy(ints, 0, o[f], 0, o[f].Length);
        }
        Drop(video); Flush();
        return o;
    }

    /// <summary>
    /// <see cref="DecodeAsync"/> into a device buffer the caller owns: frames [8 (Fz - 1) + 1, H, W, 3] channel-last in 0..1 = clamp((x + 1) / 2, 0, 1) (ComfyUI's IMAGE),
    /// converted on the device (<see cref="Ltx.LtxKernels.FramesNhwc01"/>); nothing is read back to the host.
    /// </summary>
    public async Task<MemoryBuffer1D<float, Stride1D.Dense>> DecodeDeviceFramesAsync(float[] z, int Fz, int h, int w)
    {
        var video = await DecodeToDeviceAsync(z, Fz, h, w);
        int frames = 8 * (Fz - 1) + 1, pixels = 32 * h * 32 * w;
        var o = _acc.Allocate1D<float>(3L * frames * pixels);
        _lk.FramesNhwc01(video.View, o.View, frames, pixels, 0);
        await _acc.SynchronizeAsync();                                                              // the decoded video goes back to the pool below: drain first
        Drop(video); Flush();
        return o;
    }

    // the decoded video [3, 8 (Fz - 1) + 1, 32 h, 32 w] channel-first on the device (a pooled buffer: the caller Drops it)
    private async Task<Buf> DecodeToDeviceAsync(float[] z, int Fz, int h, int w)
    {
        int T = Fz, H = h, W = w, vox = T * H * W;
        if (z.Length != (long)LatentChannels * vox) throw new ArgumentException("the latent must be [128, Fz, h, w]");
        using var zIn = _acc.Allocate1D(z);
        var x0 = Tmp((long)vox * LatentChannels);
        _lk.LatentToVoxels(zIn.View, x0.View, _mean.View, _std.View, null, vox, LatentChannels, 1f, 0f);
        var x = Conv("conv_in", x0, T, H, W, LatentChannels, 1024); Drop(x0); Flush();
        Tap?.Invoke("conv_in", x.View, T, H, W, 1024);
        int c = 1024;
        for (int j = 0; j < ResnetsPerBlock; j++) x = Resnet($"up_blocks.0.res_blocks.{j}.", x, T, H, W, c);
        Tap?.Invoke("up_blocks.0", x.View, T, H, W, c);
        for (int stage = 0; stage < 3; stage++)
        {
            x = Upsample($"up_blocks.{2 * stage + 1}.", x, ref T, ref H, ref W, c);
            c /= 2;
            for (int j = 0; j < ResnetsPerBlock; j++) x = Resnet($"up_blocks.{2 * stage + 2}.res_blocks.{j}.", x, T, H, W, c);
            Tap?.Invoke($"up_blocks.{2 * stage + 2}", x.View, T, H, W, c);
        }
        int rows = T * H * W;
        var n = Tmp((long)rows * c); _l.PixelNormSilu(x.View, n.View, rows, c, PixelNormEps); Drop(x); Flush();
        var head = Conv("conv_out", n, T, H, W, c, 3 * Patch * Patch); Drop(n); Flush();
        var video = Tmp(3L * T * Patch * H * Patch * W);
        _lk.UnpatchifyVideo(head.View, video.View, T, H, W);
        Drop(head);
        await _acc.SynchronizeAsync();                                                               // zIn is disposed on return: nothing queued may still read it
        Flush();
        return video;
    }

    public void Dispose()
    {
        _temps.Dispose();
        foreach (var t in _tables.Values) t.Dispose();
        _tables.Clear();
        foreach (var g in _convs.Values) g.Dispose();
        _convs.Clear();
        _mean.Dispose(); _std.Dispose();
        _sw.Dispose();
    }
}
