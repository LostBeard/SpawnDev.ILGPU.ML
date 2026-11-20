using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Ltx2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using Buf = MemoryBuffer1D<float, Stride1D.Dense>;

/// <summary>
/// The LTX-2 video VAE encoder (the checkpoint's <c>vae.encoder</c>; diffusers' LTX2 video encoder, Apache-2.0, with the checkpoint's config: encoder blocks res_x 4 / space x2 /
/// res_x 6 / time x2 / res_x 6 / all x2 / res_x 2 / all x2 / res_x 2, causal, zero spatial padding, pixel norm, log-variance "uniform"): video [3, F, 32 h, 32 w] in [-1, 1]
/// (F = 8 n + 1; a picture is F = 1) becomes the normalised latent [128, (F - 1) / 8 + 1, h, w]. The same design as LTX-Video 0.9.8's encoder (<see cref="Ltx.LtxVaeEncoder"/>):
/// the 4x4 pixel patch to 48 channels, a causal 3x3x3 convolution to 128, the down stages (resnets, residual space-to-depth downsamplers), a mid stage, pixel norm, SiLU and a
/// convolution to 128 means + one log-variance; the mean (or a sample with given noise), normalised with the checkpoint's latent statistics. Exact bf16 convolutions.
/// </summary>
public sealed class Ltx2VideoEncoder : IDisposable
{
    public const int LatentChannels = 128, Patch = 4;
    private const float PixelNormEps = 1e-8f;
    private const string P = "vae.encoder.";
    private static readonly (int Resnets, int Cout, int St, int Ss)[] Stages = { (4, 256, 1, 2), (6, 512, 2, 1), (6, 1024, 2, 2), (2, 2048, 2, 2) };

    private readonly Accelerator _acc;
    private readonly StreamedModelWeights _sw;
    private readonly SparseKernels _sk;
    private readonly Ltx.LtxKernels _lk;
    private readonly Ltx2Kernels _l;
    private readonly DiTKernels _dk;
    private readonly ActivationPool _temps;
    private readonly Dictionary<string, MemoryBuffer1D<BFloat16, Stride1D.Dense>> _convs = new();
    private readonly Dictionary<(int T, int H, int W, int Shift), MemoryBuffer1D<int, Stride1D.Dense>> _tables = new();
    private readonly Buf _mean, _std, _zeros;

    /// <summary>Test hook: (stage name, activation view, frames, height, width, channels); valid only during the call.</summary>
    public Action<string, FV, int, int, int, int>? Tap { get; set; }

    private Ltx2VideoEncoder(Accelerator acc, StreamedModelWeights w, Buf mean, Buf std)
    {
        _acc = acc; _sw = w; _mean = mean; _std = std; _zeros = acc.Allocate1D(new float[2048]);
        _temps = new ActivationPool(acc); _sk = new SparseKernels(acc); _lk = new Ltx.LtxKernels(acc); _l = new Ltx2Kernels(acc); _dk = new DiTKernels(acc);
    }

    /// <summary>Load the video encoder from an LTX-2 checkpoint (only <c>vae.encoder.*</c> and the latent statistics are read).</summary>
    public static async Task<Ltx2VideoEncoder> OpenAsync(Accelerator accelerator, IModelSource source, string repo, string file, CancellationToken ct = default)
    {
        static int BlockOf(string n) => n.StartsWith(P, StringComparison.Ordinal) || n.StartsWith("vae.per_channel_statistics.", StringComparison.Ordinal) ? -1 : StreamedModelWeights.HostOnly;
        var w = await StreamedModelWeights.OpenAsync(accelerator, source, repo, file, P + "conv_in.conv.weight", BlockOf, 0, 0, null, ct);
        try
        {
            var b = w.Base;
            async Task<Buf> Stat(string name)
            {
                var u = await b.BF16("vae.per_channel_statistics." + name).Cast<BFloat16, ushort>().CopyToHostAsync();
                var o = new float[u.Length]; for (int i = 0; i < u.Length; i++) o[i] = BitConverter.Int32BitsToSingle(u[i] << 16);
                return accelerator.Allocate1D(o);
            }
            var e = new Ltx2VideoEncoder(accelerator, w, await Stat("mean-of-means"), await Stat("std-of-means"));
            await e.PrepareAsync();
            return e;
        }
        catch { w.Dispose(); throw; }
    }

    private async Task PrepareAsync()
    {
        var b = _sw.Base;
        var names = b.Names.Where(n => n.StartsWith(P, StringComparison.Ordinal) && n.EndsWith(".conv.weight", StringComparison.Ordinal)).ToList();
        foreach (var n in names)
        {
            var s = b.Shape(n);
            if (s.Length != 5 || s[2] * s[3] * s[4] != 27) throw new InvalidDataException($"{n}: [{string.Join(", ", s)}], expected a 3x3x3 Conv3d weight");
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

    private MemoryBuffer1D<int, Stride1D.Dense> Table(int T, int H, int W, int shift)
    {
        if (_tables.TryGetValue((T, H, W, shift), out var t)) return t;
        t = _acc.Allocate1D<int>((long)(T + shift) * H * W * 27);
        _lk.CausalTable(t.View, T, H, W, shift);
        _tables[(T, H, W, shift)] = t;
        return t;
    }

    // a causal 3x3x3 convolution of the grid T x H x W (with shift extra copies of the first frame in front): [(T + shift) H W, cout]
    private Buf Conv(string name, Buf x, int T, int H, int W, int cin, int shift = 0)
    {
        var w = _convs[P + name + ".conv"];
        int m = (T + shift) * H * W, cout = (int)(w.Length / 27 / cin);
        if (!_sk.ConvTapsBf16ExactOk(m, 27, cin, cout, (long)T * H * W)) throw new NotSupportedException($"{name}: {T} x {H} x {W} x {cin} beyond one exact GEMM");
        var y = Tmp((long)m * cout);
        _sk.ConvTapsBf16Exact(x.View, Table(T, H, W, shift).View, w.View, _sw.Base.BF16(P + name + ".conv.bias"), y.View, m, cin, cout, 27);
        return y;
    }

    private Buf NormSilu(Buf x, int rows, int C)
    {
        var o = Tmp((long)rows * C);
        _l.PixelNormSilu(x.View, o.View, rows, C, PixelNormEps);
        return o;
    }

    private Buf Resnet(string p, Buf x, int T, int H, int W, int C)
    {
        int rows = T * H * W;
        var a = NormSilu(x, rows, C);
        var c1 = Conv(p + "conv1", a, T, H, W, C); Drop(a); Flush();
        var b = NormSilu(c1, rows, C); Drop(c1); Flush();
        var c2 = Conv(p + "conv2", b, T, H, W, C); Drop(b); Flush();
        _dk.GateResidual(c2.View, x.View, default, -1, rows * C, C);
        Drop(x); Flush();
        return c2;
    }

    private Buf Downsample(string p, Buf x, ref int T, ref int H, ref int W, int cin, int cout, int st, int ss)
    {
        var y = Conv(p, x, T, H, W, cin, shift: st - 1);
        int To = (T + st - 1) / st, Ho = H / ss, Wo = W / ss;
        var o = Tmp((long)To * Ho * Wo * cout);
        _lk.DownShuffle(y.View, x.View, o.View, T, H, W, cout, cin, st, ss, ss);
        Drop(y); Drop(x);
        T = To; H = Ho; W = Wo;
        Flush();
        return o;
    }

    /// <summary>
    /// Encode <paramref name="video"/> ([3, frames, height, width] channel-first in [-1, 1]; frames 8 n + 1, sides multiples of 32) to the normalised latent
    /// [128, (frames - 1) / 8 + 1, height / 32, width / 32]: the posterior mean, or with <paramref name="noise"/> (standard normal, shaped like the latent) a sample.
    /// </summary>
    public async Task<float[]> EncodeAsync(float[] video, int frames, int height, int width, float[]? noise = null)
    {
        using var z = await EncodeCoreAsync(video, null, frames, height, width, noise);
        return await z.View.CopyToHostAsync();
    }

    /// <summary>
    /// <see cref="EncodeAsync(float[], int, int, int, float[])"/> on the device (rule 4): <paramref name="video"/> [3, frames, height, width] channel-first in [-1, 1] (read, not
    /// modified); returns the latent [128, (frames - 1) / 8 + 1, height / 32, width / 32] in a buffer the caller owns.
    /// </summary>
    public Task<Buf> EncodeDeviceAsync(FV video, int frames, int height, int width, float[]? noise = null) => EncodeCoreAsync(null, video, frames, height, width, noise);

    /// <summary><see cref="EncodeDeviceAsync"/> from ComfyUI's IMAGE: <paramref name="images"/> [frames, height, width, 3] channel-last in 0..1 on the device (x * 2 - 1).</summary>
    public async Task<Buf> EncodeFrames01Async(FV images, int frames, int height, int width, float[]? noise = null)
    {
        if (images.Length != 3L * frames * height * width) throw new ArgumentException($"the IMAGE must be [{frames}, {height}, {width}, 3]");
        using var cf = _acc.Allocate1D<float>(3L * frames * height * width);
        (_ik ??= new ImageTensorKernels(_acc)).FromImage01(images, cf.View, frames, height * width);
        return await EncodeCoreAsync(null, cf.View, frames, height, width, noise);           // drains before returning: cf may go
    }
    private ImageTensorKernels? _ik;

    // a host video or a device one (read only) -> the latent in a new buffer the caller owns
    private async Task<Buf> EncodeCoreAsync(float[]? video, FV? videoDev, int frames, int height, int width, float[]? noise)
    {
        if ((frames - 1) % 8 != 0 || frames < 1) throw new ArgumentException("the frame count must be 8n + 1");
        if (height % 32 != 0 || width % 32 != 0) throw new ArgumentException("height and width must be multiples of 32");
        if ((videoDev is { } vd0 ? vd0.Length : video!.Length) != 3L * frames * height * width) throw new ArgumentException("the video must be [3, frames, height, width]");
        int T = frames, H = height / Patch, W = width / Patch;
        int latentLength = 128 * ((frames - 1) / 8 + 1) * (height / 32) * (width / 32);
        if (noise != null && noise.Length != latentLength) throw new ArgumentException("the sampling noise must be shaped like the latent");
        using var vIn = videoDev == null ? _acc.Allocate1D(video!) : null;
        var x0 = Tmp((long)T * H * W * 48);
        _lk.PatchifyVideo(videoDev ?? vIn!.View, x0.View, T, H, W);
        var x = Conv("conv_in", x0, T, H, W, 48); Drop(x0); Flush();
        Tap?.Invoke("conv_in", x.View, T, H, W, 128);
        int c = 128;
        for (int s = 0; s < Stages.Length; s++)
        {
            var (resnets, cout, st, ss) = Stages[s];
            for (int j = 0; j < resnets; j++) x = Resnet($"down_blocks.{2 * s}.res_blocks.{j}.", x, T, H, W, c);
            x = Downsample($"down_blocks.{2 * s + 1}.conv", x, ref T, ref H, ref W, c, cout, st, ss);
            c = cout;
            Tap?.Invoke($"down_{s}", x.View, T, H, W, c);
        }
        for (int j = 0; j < 2; j++) x = Resnet($"down_blocks.8.res_blocks.{j}.", x, T, H, W, c);
        int vox = T * H * W;
        var n = NormSilu(x, vox, c); Drop(x);
        var head = Conv("conv_out", n, T, H, W, c); Drop(n); Flush();
        using var nIn = noise != null ? _acc.Allocate1D(noise) : null;
        var latent = _acc.Allocate1D<float>(128L * vox);
        _lk.Posterior(head.View, latent.View, _mean.View, _std.View, nIn?.View, vox);
        Drop(head);
        await _acc.SynchronizeAsync();                                                   // the uploads are disposed on return
        Flush();
        return latent;
    }

    public void Dispose()
    {
        _temps.Dispose();
        foreach (var t in _tables.Values) t.Dispose();
        foreach (var g in _convs.Values) g.Dispose();
        _tables.Clear(); _convs.Clear();
        _mean.Dispose(); _std.Dispose(); _zeros.Dispose();
        _sw.Dispose();
    }
}
