using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Trellis2;
using System.Text.Json;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.AceStep;

/// <summary>
/// The Oobleck audio VAE's decoder (diffusers' AutoencoderOobleck; ACE-Step 1.5's <c>ace_1.5_vae</c>): latent [64, T] at 25 frames per second to stereo
/// audio [2, 1920 T] at 48 kHz. Layout read from the file (<c>decoder.layers.*</c>):
/// <list type="bullet">
/// <item>layers.0: Conv1d k7 64 -> 2048 (pad 3);</item>
/// <item>layers.1 .. 5: decoder blocks, each Snake, ConvTranspose1d (kernel 2 s, stride s, padding ceil(s / 2); s = 10, 6, 4, 4, 2), then three residual
/// units (Snake, Conv1d k7 dilated 1 / 3 / 9 with padding 3 d, Snake, Conv1d k1; added to the unit's input);</item>
/// <item>layers.6: Snake; layers.7: Conv1d k7 128 -> 2, no bias.</item>
/// </list>
/// Every convolution is weight-normalised in the file (<c>weight_g</c>, <c>weight_v</c>: weight = g v / |v| over all but the first dimension); the weights
/// are fused to fp32 once at load. Snake: x + sin^2(exp(alpha) x) / (exp(beta) + 1e-9) per channel (log-scale parameters, as stored).
/// </summary>
public sealed class OobleckDecoder : IDisposable
{
    /// <summary>Audio samples per latent frame (10 x 6 x 4 x 4 x 2).</summary>
    public int HopLength { get; }
    /// <summary>Latent channels the decoder takes.</summary>
    public int LatentChannels { get; }
    /// <summary>Audio channels it writes (2: stereo).</summary>
    public int AudioChannels { get; }

    private readonly Accelerator _acc;
    private readonly Conv1DKernel _conv;
    private readonly ConvTranspose1DKernel _convT;
    private readonly Action<Index1D, FV, FV, FV, FV, int> _snake;
    private readonly Action<Index1D, FV, FV> _add;
    private readonly Dictionary<string, Conv> _convs = new();
    private readonly Dictionary<string, (MemoryBuffer1D<float, Stride1D.Dense> A, MemoryBuffer1D<float, Stride1D.Dense> InvB)> _snakes = new();
    private readonly (int Stride, int InC, int OutC)[] _blocks;

    private sealed record Conv(MemoryBuffer1D<float, Stride1D.Dense> W, MemoryBuffer1D<float, Stride1D.Dense> B, int D0, int D1, int K);

    private OobleckDecoder(Accelerator acc, (int, int, int)[] blocks, int latentChannels, int audioChannels)
    {
        _acc = acc; _blocks = blocks; LatentChannels = latentChannels; AudioChannels = audioChannels;
        HopLength = blocks.Aggregate(1, (h, b) => h * b.Item1);
        _conv = new Conv1DKernel(acc); _convT = new ConvTranspose1DKernel(acc);
        _snake = acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, int>(SnakeImpl);
        _add = acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(AddImpl);
    }

    /// <summary>Load the decoder from a safetensors file holding <paramref name="prefix"/>layers.* (ComfyUI's ace_1.5_vae: "decoder.").</summary>
    public static async Task<OobleckDecoder> LoadAsync(Accelerator accelerator, IModelSource source, string repo, string file, string prefix = "decoder.", CancellationToken ct = default)
    {
        GpuWeights w;
        await using (var stream = await source.OpenAsync(repo, file, ct))
        {
            using var empty = JsonDocument.Parse("{}");
            w = await GpuWeights.LoadAsync(accelerator, stream, empty.RootElement, file, n => n.StartsWith(prefix, StringComparison.Ordinal), ct);
        }
        try
        {
            string L(string rest) => prefix + "layers." + rest;
            var blocks = new List<(int, int, int)>();
            for (int i = 1; w.Has(L($"{i}.layers.1.weight_v")); i++)
            {
                var s = w.Shape(L($"{i}.layers.1.weight_v"));                         // ConvTranspose1d [inC, outC, 2 stride]
                blocks.Add((s[2] / 2, s[0], s[1]));
            }
            if (blocks.Count == 0) throw new InvalidDataException($"'{file}': no Oobleck decoder blocks under {prefix}layers.");
            int latent = w.Shape(L("0.weight_v"))[1];
            int last = blocks.Count + 2;
            int audio = w.Shape(L($"{last}.weight_v"))[0];
            var d = new OobleckDecoder(accelerator, blocks.ToArray(), latent, audio);
            try
            {
                await d.AddConvAsync(w, L("0"), "in");
                for (int i = 1; i <= blocks.Count; i++)
                {
                    await d.AddSnakeAsync(w, L($"{i}.layers.0"), $"b{i}.snake");
                    await d.AddConvAsync(w, L($"{i}.layers.1"), $"b{i}.up");
                    for (int r = 0; r < 3; r++)
                    {
                        string u = L($"{i}.layers.{r + 2}.layers.");
                        await d.AddSnakeAsync(w, u + "0", $"b{i}.r{r}.s0");
                        await d.AddConvAsync(w, u + "1", $"b{i}.r{r}.c0");
                        await d.AddSnakeAsync(w, u + "2", $"b{i}.r{r}.s1");
                        await d.AddConvAsync(w, u + "3", $"b{i}.r{r}.c1");
                    }
                }
                await d.AddSnakeAsync(w, L($"{blocks.Count + 1}"), "out.snake");
                await d.AddConvAsync(w, L($"{last}"), "out");
                return d;
            }
            catch { d.Dispose(); throw; }
        }
        finally { await accelerator.SynchronizeAsync(); w.Dispose(); }
    }

    private static async Task<float[]> HostAsync(GpuWeights w, string name)
    {
        var v = await w.BF16(name).CopyToHostAsync();
        var f = new float[v.Length];
        for (int i = 0; i < f.Length; i++) f[i] = (float)v[i];
        return f;
    }

    // weight = g * v / |v|, the norm over every dimension but the first (torch.nn.utils.weight_norm, dim 0), in double then fp32
    private async Task AddConvAsync(GpuWeights w, string name, string key)
    {
        var shape = w.Shape(name + ".weight_v");
        var v = await HostAsync(w, name + ".weight_v");
        var g = await HostAsync(w, name + ".weight_g");
        int d0 = shape[0], per = v.Length / d0;
        var fused = new float[v.Length];
        for (int o = 0; o < d0; o++)
        {
            double ss = 0;
            for (int j = 0; j < per; j++) { double x = v[o * per + j]; ss += x * x; }
            double scale = g[o] / Math.Sqrt(ss);
            for (int j = 0; j < per; j++) fused[o * per + j] = (float)(v[o * per + j] * scale);
        }
        // no bias (the last conv): explicit zeros [outC], the same sum, and no zero-length binding
        var bias = _acc.Allocate1D(w.Has(name + ".bias") ? await HostAsync(w, name + ".bias") : new float[shape[0]]);
        _convs[key] = new Conv(_acc.Allocate1D(fused), bias, shape[0], shape[1], shape[2]);
    }

    private async Task AddSnakeAsync(GpuWeights w, string name, string key)
    {
        var alpha = await HostAsync(w, name + ".alpha");
        var beta = await HostAsync(w, name + ".beta");
        var a = alpha.Select(x => (float)Math.Exp(x)).ToArray();
        var invB = beta.Select(x => (float)(1.0 / (Math.Exp(x) + 1e-9))).ToArray();
        _snakes[key] = (_acc.Allocate1D(a), _acc.Allocate1D(invB));
    }

    /// <summary>
    /// Decode <paramref name="latent"/> [<see cref="LatentChannels"/>, <paramref name="frames"/>] (channel-major, as ComfyUI's [1, 64, T] latent) into a new
    /// buffer [<see cref="AudioChannels"/>, <see cref="HopLength"/> x <paramref name="frames"/>] the caller owns.
    /// </summary>
    public async Task<MemoryBuffer1D<float, Stride1D.Dense>> DecodeAsync(FV latent, int frames, CancellationToken ct = default)
    {
        long maxElems = (long)_convs["in"].D0 * frames;
        int len = frames;
        foreach (var (s, _, outC) in _blocks) { len *= s; maxElems = Math.Max(maxElems, (long)outC * len); }
        var a = _acc.Allocate1D<float>(maxElems); var b = _acc.Allocate1D<float>(maxElems); var c = _acc.Allocate1D<float>(maxElems);
        var output = _acc.Allocate1D<float>((long)AudioChannels * frames * HopLength);
        try
        {
            int L = frames, ch = LatentChannels;
            ConvForward("in", latent, a.View, ch, L, 3, 1); ch = _convs["in"].D0;           // a: [2048, T]
            for (int i = 0; i < _blocks.Length; i++)
            {
                var (s, inC, outC) = _blocks[i];
                Snake($"b{i + 1}.snake", a.View, b.View, inC, L);
                var up = _convs[$"b{i + 1}.up"];
                _convT.Forward(b.View.SubView(0, (long)inC * L), up.W.View, up.B.View, a.View.SubView(0, (long)outC * L * s),
                    1, inC, L, outC, up.K, s, (s + 1) / 2, (s + 1) / 2);
                L *= s; ch = outC;                                                          // a: block output before the residual units
                for (int r = 0; r < 3; r++)
                {
                    int dil = r == 0 ? 1 : r == 1 ? 3 : 9;
                    Snake($"b{i + 1}.r{r}.s0", a.View, b.View, ch, L);
                    ConvForward($"b{i + 1}.r{r}.c0", b.View, c.View, ch, L, 3 * dil, dil);
                    Snake($"b{i + 1}.r{r}.s1", c.View, b.View, ch, L);
                    ConvForward($"b{i + 1}.r{r}.c1", b.View, c.View, ch, L, 0, 1);
                    _add((int)((long)ch * L), a.View.SubView(0, (long)ch * L), c.View.SubView(0, (long)ch * L));
                }
            }
            Snake("out.snake", a.View, b.View, ch, L);
            ConvForward("out", b.View, output.View, ch, L, 3, 1);
            await _acc.SynchronizeAsync();
            return output;
        }
        catch { await _acc.SynchronizeAsync(); output.Dispose(); throw; }
        finally { await _acc.SynchronizeAsync(); a.Dispose(); b.Dispose(); c.Dispose(); }
    }

    private void ConvForward(string key, FV x, FV y, int inC, int L, int pad, int dilation)
    {
        var cv = _convs[key];
        _conv.Forward(x.SubView(0, (long)inC * L), cv.W.View, cv.B.View, y.SubView(0, (long)cv.D0 * L), inC, L, cv.D0, cv.K, 1, pad, dilation);
    }

    private void Snake(string key, FV x, FV y, int ch, int L)
    {
        var (a, invB) = _snakes[key];
        _snake((int)((long)ch * L), x.SubView(0, (long)ch * L), y.SubView(0, (long)ch * L), a.View, invB.View, L);
    }

    private static void SnakeImpl(Index1D i, FV x, FV y, FV a, FV invB, int len)
    {
        int c = i / len;
        float v = x[i];
        float s = MathF.Sin(a[c] * v);
        y[i] = v + invB[c] * s * s;
    }

    private static void AddImpl(Index1D i, FV acc, FV x) => acc[i] += x[i];

    public void Dispose()
    {
        foreach (var cv in _convs.Values) { cv.W.Dispose(); cv.B.Dispose(); }
        foreach (var (a, b) in _snakes.Values) { a.Dispose(); b.Dispose(); }
        _convs.Clear(); _snakes.Clear();
        _conv.Dispose(); _convT.Dispose();
    }
}
