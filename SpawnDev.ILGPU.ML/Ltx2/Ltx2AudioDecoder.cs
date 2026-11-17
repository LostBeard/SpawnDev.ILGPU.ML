using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Ltx2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using Buf = MemoryBuffer1D<float, Stride1D.Dense>;

/// <summary>
/// LTX-2's audio decode (the checkpoint's <c>audio_vae.decoder</c> and <c>vocoder</c>; diffusers' LTX2AudioDecoder and LTX2Vocoder, Apache-2.0, with the checkpoint's config):
/// the audio latent [8, Ta, 16] is denormalised per packed feature, decoded to a stereo mel spectrogram [2, 4 Ta - 3, 64] (conv_in to 512 channels, two resnets, then three
/// levels of three resnets 512 -> 256 -> 128 with nearest x2 upsampling + a causal convolution that drops its first time row between them, pixel norm, SiLU, conv_out; every 2-D
/// convolution causal in time, pixel norms epsilon 1e-6), and the HiFi-GAN vocoder turns it into a stereo waveform at 24 kHz (240 samples per spectrogram frame).
/// Channel-last fp32 activations; convolutions are exact gathered GEMMs on the checkpoint's bf16 weights.
/// </summary>
public sealed class Ltx2AudioDecoder : IDisposable
{
    public const int LatentChannels = 8, LatentMel = 16, Mel = 64, SampleRate = 24000, HopSamples = 240;
    private const float PixelEps = 1e-6f, Slope = 0.1f, FinalSlope = 0.01f;
    private const string A = "audio_vae.decoder.", V = "vocoder.";
    private static readonly int[] UpRates = { 6, 5, 2, 2, 2 }, UpKernels = { 16, 15, 8, 4, 4 }, ResKernels = { 3, 7, 11 }, ResDilations = { 1, 3, 5 };

    private readonly Accelerator _acc;
    private readonly StreamedModelWeights _sw;
    private readonly SparseKernels _sk;
    private readonly Ltx2Kernels _l;
    private readonly DiTKernels _dk;
    private readonly ActivationPool _temps;
    private readonly Dictionary<string, (MemoryBuffer1D<BFloat16, Stride1D.Dense> W, int Taps)> _convs = new();
    private readonly List<IDisposable> _tables = new();
    private readonly Buf _mean, _std;

    /// <summary>Test hook: ("spectrogram" [T M, 2] channel-last) during the call.</summary>
    public Action<string, FV, int, int>? Tap { get; set; }

    private Ltx2AudioDecoder(Accelerator accelerator, StreamedModelWeights weights, Buf mean, Buf std)
    {
        _acc = accelerator; _sw = weights; _mean = mean; _std = std;
        _temps = new ActivationPool(accelerator); _sk = new SparseKernels(accelerator); _l = new Ltx2Kernels(accelerator); _dk = new DiTKernels(accelerator);
    }

    /// <summary>Load the audio decoder and vocoder from an LTX-2 checkpoint (only <c>audio_vae.decoder.*</c>, its statistics and <c>vocoder.*</c> are read).</summary>
    public static async Task<Ltx2AudioDecoder> OpenAsync(Accelerator accelerator, IModelSource source, string repo, string file, CancellationToken ct = default)
    {
        static int BlockOf(string n) => n.StartsWith(A, StringComparison.Ordinal) || n.StartsWith(V, StringComparison.Ordinal) || n.StartsWith("audio_vae.per_channel_statistics.", StringComparison.Ordinal)
            ? -1 : StreamedModelWeights.HostOnly;
        var w = await StreamedModelWeights.OpenAsync(accelerator, source, repo, file, A + "conv_in.conv.weight", BlockOf, 0, 0, null, ct);
        try
        {
            var b = w.Base;
            async Task<float[]> Stat(string name)
            {
                var u = await b.BF16("audio_vae.per_channel_statistics." + name).Cast<BFloat16, ushort>().CopyToHostAsync();
                var o = new float[u.Length]; for (int i = 0; i < u.Length; i++) o[i] = BitConverter.Int32BitsToSingle(u[i] << 16); return o;
            }
            var d = new Ltx2AudioDecoder(accelerator, w, accelerator.Allocate1D(await Stat("mean-of-means")), accelerator.Allocate1D(await Stat("std-of-means")));
            await d.PrepareAsync();
            return d;
        }
        catch { w.Dispose(); throw; }
    }

    // every (non-transposed) convolution weight into the gather layout [cout][taps][cin]
    private async Task PrepareAsync()
    {
        var b = _sw.Base;
        var names = b.Names.Where(n => n.EndsWith(".weight", StringComparison.Ordinal) && (n.StartsWith(A, StringComparison.Ordinal) && n.EndsWith(".conv.weight", StringComparison.Ordinal)
            || n.StartsWith(V, StringComparison.Ordinal) && !n.StartsWith(V + "ups.", StringComparison.Ordinal))).ToList();
        foreach (var n in names)
        {
            var s = b.Shape(n);
            int taps = s.Length == 4 ? s[2] * s[3] : s.Length == 3 ? s[2] : throw new InvalidDataException($"{n}: [{string.Join(", ", s)}]");
            var g = _acc.Allocate1D<BFloat16>((long)s[0] * taps * s[1]);
            _l.GatherLayoutBf16(b.BF16(n), g.View, s[0], s[1], taps);
            _convs[n[..^".weight".Length]] = (g, taps);
        }
        await _acc.SynchronizeAsync();
        foreach (var n in names) b.Remove(n)?.Dispose();
    }

    private Buf Tmp(long n) => _temps.Get(n);
    private void Drop(Buf x) => _temps.Drop(x);
    private void Flush() => _temps.Flush();

    private MemoryBuffer1D<int, Stride1D.Dense> Table(int rows, Action<ArrayView1D<int, Stride1D.Dense>> fill)
    {
        var t = _acc.Allocate1D<int>((long)rows * 27); _tables.Add(t); fill(t.View); return t;
    }

    // a gathered convolution: name without ".weight"; x [rows, cin] -> new [rows, cout]
    private Buf Conv(string name, Buf x, MemoryBuffer1D<int, Stride1D.Dense> nbr, int rows, int cin)
    {
        var (w, taps) = _convs[name];
        int cout = (int)(w.Length / taps / cin);
        if (!_sk.ConvTapsBf16ExactOk(rows, taps, cin, cout, rows)) throw new NotSupportedException($"{name}: {rows} x {taps} x {cin} beyond one exact GEMM");
        var y = Tmp((long)rows * cout);
        var bias = _sw.Base.Has(name + ".bias") ? _sw.Base.BF16(name + ".bias") : (ArrayView1D<BFloat16, Stride1D.Dense>?)null;
        _sk.ConvTapsBf16Exact(x.View, nbr.View, w.View, bias, y.View, rows, cin, cout, taps);
        return y;
    }

    // the VAE resnet (consumes x): pixel norm, SiLU, conv1, pixel norm, SiLU, conv2, + (nin_shortcut(x) or x)
    private Buf Resnet(string p, Buf x, MemoryBuffer1D<int, Stride1D.Dense> t3, MemoryBuffer1D<int, Stride1D.Dense> t1, int rows, int cin, int cout)
    {
        var a = Tmp((long)rows * cin); _l.PixelNormSilu(x.View, a.View, rows, cin, PixelEps);
        var h = Conv(p + "conv1.conv", a, t3, rows, cin); Drop(a); Flush();
        var b = Tmp((long)rows * cout); _l.PixelNormSilu(h.View, b.View, rows, cout, PixelEps); Drop(h); Flush();
        var h2 = Conv(p + "conv2.conv", b, t3, rows, cout); Drop(b); Flush();
        if (cin != cout) { var sc = Conv(p + "nin_shortcut.conv", x, t1, rows, cin); _dk.GateResidual(h2.View, sc.View, default, -1, rows * cout, cout); Drop(sc); }
        else _dk.GateResidual(h2.View, x.View, default, -1, rows * cout, cout);
        Drop(x); Flush();
        return h2;
    }

    /// <summary>
    /// Decode the audio latent <paramref name="z"/> ([8, Ta, 16] as ComfyUI's [1, 8, Ta, 16]) to a stereo waveform [2, 240 (4 Ta - 3)] at 24 kHz (channel-first: left then right).
    /// </summary>
    public async Task<float[]> DecodeAsync(float[] z, int Ta)
    {
        if (z.Length != LatentChannels * Ta * LatentMel) throw new ArgumentException($"the latent must be [8, {Ta}, 16]");
        int T = Ta, M = LatentMel;
        using var zIn = _acc.Allocate1D(z);
        var x = Tmp((long)T * M * LatentChannels);
        _l.DenormAudioLatent(zIn.View, _mean.View, _std.View, x.View, LatentChannels, T, M);
        var t3 = Table(T * M, v => _l.CausalTable2d(v, T, M, 3, 3)); var t1 = Table(T * M, v => _l.CausalTable2d(v, T, M, 1, 1));
        var h = Conv(A + "conv_in.conv", x, t3, T * M, LatentChannels); Drop(x); Flush();
        int c = 512;
        h = Resnet(A + "mid.block_1.", h, t3, t1, T * M, c, c);
        h = Resnet(A + "mid.block_2.", h, t3, t1, T * M, c, c);
        foreach (int level in new[] { 2, 1, 0 })
        {
            int cout = 128 << level;
            for (int j = 0; j < 3; j++) { h = Resnet($"{A}up.{level}.block.{j}.", h, t3, t1, T * M, c, cout); c = cout; }
            if (level == 0) break;
            // nearest x2, causal conv on the doubled grid, drop the first time row
            var u = Tmp(4L * T * M * c); _l.NearestUp2(h.View, u.View, T, M, c); Drop(h); Flush();
            int T2 = 2 * T, M2 = 2 * M;
            var tu = Table(T2 * M2, v => _l.CausalTable2d(v, T2, M2, 3, 3));
            var y = Conv($"{A}up.{level}.upsample.conv.conv", u, tu, T2 * M2, c); Drop(u); Flush();
            T = T2 - 1; M = M2;
            h = Tmp((long)T * M * c); h.View.SubView(0, (long)T * M * c).CopyFrom(y.View.SubView((long)M * c, (long)T * M * c)); Drop(y); Flush();
            t3 = Table(T * M, v => _l.CausalTable2d(v, T, M, 3, 3)); t1 = Table(T * M, v => _l.CausalTable2d(v, T, M, 1, 1));
        }
        var n = Tmp((long)T * M * c); _l.PixelNormSilu(h.View, n.View, T * M, c, PixelEps); Drop(h); Flush();
        var spec = Conv(A + "conv_out.conv", n, t3, T * M, c); Drop(n); Flush();             // [T M, 2]
        int target = 4 * Ta - 3;
        if (T != target || M != Mel) throw new InvalidOperationException($"decoded [{T}, {M}], expected [{target}, {Mel}]");
        Tap?.Invoke("spectrogram", spec.View, T * M, 2);

        // vocoder: [T, 128] (channel s 64 + m)
        var v0 = Tmp((long)T * 2 * M); _l.SpectrogramToVocoder(spec.View, v0.View, T, M); Drop(spec); Flush();
        int L = T, ch = 2 * M;
        var tk7 = Table(L, v => _l.SameTable1d(v, L, 7, 1));
        var cur = Conv(V + "conv_pre", v0, tk7, L, ch); Drop(v0); Flush(); ch = 1024;
        for (int i = 0; i < UpRates.Length; i++)
        {
            var a = Tmp((long)L * ch); _l.LeakyRelu(cur.View, a.View, L * ch, Slope); Drop(cur); Flush();
            int s = UpRates[i], k = UpKernels[i], p = (k - s) / 2, cout = ch / 2, Lo = (L - 1) * s - 2 * p + k;
            var up = Tmp((long)Lo * cout);
            _l.ConvTranspose1d(a.View, _sw.Base.BF16($"{V}ups.{i}.weight"), _sw.Base.BF16($"{V}ups.{i}.bias"), up.View, L, ch, cout, k, s, p); Drop(a); Flush();
            L = Lo; ch = cout;
            var outs = new Buf[3];
            for (int r = 0; r < 3; r++)
            {
                string rb = $"{V}resblocks.{i * 3 + r}.";
                var xr = Tmp((long)L * ch); xr.View.SubView(0, (long)L * ch).CopyFrom(up.View.SubView(0, (long)L * ch));
                for (int j = 0; j < ResDilations.Length; j++)
                {
                    int kr = ResKernels[r], dl = ResDilations[j];
                    var td = Table(L, v => _l.SameTable1d(v, L, kr, dl)); var t1d = Table(L, v => _l.SameTable1d(v, L, kr, 1));
                    var a1 = Tmp((long)L * ch); _l.LeakyRelu(xr.View, a1.View, L * ch, Slope);
                    var c1 = Conv($"{rb}convs1.{j}", a1, td, L, ch); Drop(a1); Flush();
                    var a2 = Tmp((long)L * ch); _l.LeakyRelu(c1.View, a2.View, L * ch, Slope); Drop(c1); Flush();
                    var c2 = Conv($"{rb}convs2.{j}", a2, t1d, L, ch); Drop(a2); Flush();
                    _dk.GateResidual(xr.View, c2.View, default, -1, L * ch, ch); Drop(c2); Flush();
                }
                outs[r] = xr;
            }
            Drop(up);
            cur = Tmp((long)L * ch); _l.Mean3(outs[0].View, outs[1].View, outs[2].View, cur.View, L * ch);
            foreach (var o in outs) Drop(o);
            Flush();
        }
        var fa = Tmp((long)L * ch); _l.LeakyRelu(cur.View, fa.View, L * ch, FinalSlope); Drop(cur); Flush();
        var tk7b = Table(L, v => _l.SameTable1d(v, L, 7, 1));
        var wav = Conv(V + "conv_post", fa, tk7b, L, ch); Drop(fa); Flush();                // [L, 2]
        _l.TanhInPlace(wav.View, L * 2);
        await _acc.SynchronizeAsync();
        var host = await wav.View.SubView(0, (long)L * 2).CopyToHostAsync();
        Drop(wav); Flush();
        foreach (var t in _tables) t.Dispose();
        _tables.Clear();
        var o2 = new float[host.Length];
        for (int l = 0; l < L; l++) { o2[l] = host[2 * l]; o2[L + l] = host[2 * l + 1]; }
        return o2;
    }

    public void Dispose()
    {
        _temps.Dispose();
        foreach (var t in _tables) t.Dispose();
        _tables.Clear();
        foreach (var (w, _) in _convs.Values) w.Dispose();
        _convs.Clear();
        _mean.Dispose(); _std.Dispose();
        _sw.Dispose();
    }
}
