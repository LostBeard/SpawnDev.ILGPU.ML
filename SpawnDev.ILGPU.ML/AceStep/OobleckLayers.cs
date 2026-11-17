using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Trellis2;
using System.Text.Json;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.AceStep;

/// <summary>
/// The pieces the Oobleck VAE's encoder and decoder share: weight-normalised convolutions fused to fp32 at load (weight = g v / |v| over all but the first dimension,
/// torch.nn.utils.weight_norm dim 0), Snake activations x + sin^2(exp(alpha) x) / (exp(beta) + 1e-9) (log-scale parameters as stored), and the residual unit
/// (Snake, Conv1d k7 dilated d with padding 3 d, Snake, Conv1d k1, added to its input). Convolutions run on ML's gather-form Conv1D / ConvTranspose1D kernels.
/// </summary>
internal sealed class OobleckLayers : IDisposable
{
    private readonly Accelerator _acc;
    private readonly Conv1DKernel _conv;
    private readonly ConvTranspose1DKernel _convT;
    private readonly Action<Index1D, FV, FV, FV, FV, int> _snake;
    private readonly Action<Index1D, FV, FV> _add;
    private readonly Dictionary<string, Conv> _convs = new();
    private readonly Dictionary<string, (MemoryBuffer1D<float, Stride1D.Dense> A, MemoryBuffer1D<float, Stride1D.Dense> InvB)> _snakes = new();

    internal sealed record Conv(MemoryBuffer1D<float, Stride1D.Dense> W, MemoryBuffer1D<float, Stride1D.Dense> B, int D0, int D1, int K);

    public OobleckLayers(Accelerator acc)
    {
        _acc = acc;
        _conv = new Conv1DKernel(acc); _convT = new ConvTranspose1DKernel(acc);
        _snake = acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, int>(SnakeImpl);
        _add = acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(AddImpl);
    }

    public Conv this[string key] => _convs[key];

    /// <summary>The tensors under <paramref name="prefix"/> of a safetensors file (bf16 as ComfyUI stores them), to read layers from; the caller disposes them after loading.</summary>
    public static async Task<GpuWeights> ReadAsync(Accelerator acc, IModelSource source, string repo, string file, string prefix, CancellationToken ct)
    {
        await using var stream = await source.OpenAsync(repo, file, ct);
        using var empty = JsonDocument.Parse("{}");
        return await GpuWeights.LoadAsync(acc, stream, empty.RootElement, file, n => n.StartsWith(prefix, StringComparison.Ordinal), ct);
    }

    private static async Task<float[]> HostAsync(GpuWeights w, string name)
    {
        var v = await w.BF16(name).CopyToHostAsync();
        var f = new float[v.Length];
        for (int i = 0; i < f.Length; i++) f[i] = (float)v[i];
        return f;
    }

    // weight = g * v / |v| over every dimension but the first, in double then fp32
    public async Task AddConvAsync(GpuWeights w, string name, string key)
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
        // a conv without bias (the decoder's last): explicit zeros over its output channels (shape[0] for a Conv1d), the same sum, and no zero-length binding
        var bias = _acc.Allocate1D(w.Has(name + ".bias") ? await HostAsync(w, name + ".bias") : new float[shape[0]]);
        _convs[key] = new Conv(_acc.Allocate1D(fused), bias, shape[0], shape[1], shape[2]);
    }

    public async Task AddSnakeAsync(GpuWeights w, string name, string key)
    {
        var alpha = await HostAsync(w, name + ".alpha");
        var beta = await HostAsync(w, name + ".beta");
        var a = alpha.Select(x => (float)Math.Exp(x)).ToArray();
        var invB = beta.Select(x => (float)(1.0 / (Math.Exp(x) + 1e-9))).ToArray();
        _snakes[key] = (_acc.Allocate1D(a), _acc.Allocate1D(invB));
    }

    /// <summary>The 3 residual units of a block (dilations 1, 3, 9) under <paramref name="prefix"/> + "{index}.layers.{0..3}", keyed <paramref name="key"/>.r{n}.</summary>
    public async Task AddResidualUnitsAsync(GpuWeights w, Func<int, string> unitPrefix, string key)
    {
        for (int r = 0; r < 3; r++)
        {
            string u = unitPrefix(r);
            await AddSnakeAsync(w, u + "0", $"{key}.r{r}.s0");
            await AddConvAsync(w, u + "1", $"{key}.r{r}.c0");
            await AddSnakeAsync(w, u + "2", $"{key}.r{r}.s1");
            await AddConvAsync(w, u + "3", $"{key}.r{r}.c1");
        }
    }

    /// <summary>Conv1d of x [inC, L] into y [outC, L'] (L' = (L + 2 pad - dilation (k - 1) - 1) / stride + 1).</summary>
    public void Conv1d(string key, FV x, FV y, int inC, int L, int pad, int dilation = 1, int stride = 1)
    {
        var cv = _convs[key];
        int outL = (L + 2 * pad - dilation * (cv.K - 1) - 1) / stride + 1;
        _conv.Forward(x.SubView(0, (long)inC * L), cv.W.View, cv.B.View, y.SubView(0, (long)cv.D0 * outL), inC, L, cv.D0, cv.K, stride, pad, dilation);
    }

    /// <summary>ConvTranspose1d (kernel 2 s, stride s, padding ceil(s / 2)) of x [inC, L] into y [outC, s L].</summary>
    public void Upsample(string key, FV x, FV y, int inC, int L, int s)
    {
        var up = _convs[key];
        _convT.Forward(x.SubView(0, (long)inC * L), up.W.View, up.B.View, y.SubView(0, (long)up.D1 * L * s), 1, inC, L, up.D1, up.K, s, (s + 1) / 2, (s + 1) / 2);
    }

    public void Snake(string key, FV x, FV y, int ch, int L)
    {
        var (a, invB) = _snakes[key];
        _snake((int)((long)ch * L), x.SubView(0, (long)ch * L), y.SubView(0, (long)ch * L), a.View, invB.View, L);
    }

    /// <summary>The 3 residual units on <paramref name="x"/> [ch, L] in place; <paramref name="t1"/>, <paramref name="t2"/> are scratch of the same size.</summary>
    public void ResidualUnits(string key, FV x, FV t1, FV t2, int ch, int L)
    {
        for (int r = 0; r < 3; r++)
        {
            int dil = r == 0 ? 1 : r == 1 ? 3 : 9;
            Snake($"{key}.r{r}.s0", x, t1, ch, L);
            Conv1d($"{key}.r{r}.c0", t1, t2, ch, L, 3 * dil, dil);
            Snake($"{key}.r{r}.s1", t2, t1, ch, L);
            Conv1d($"{key}.r{r}.c1", t1, t2, ch, L, 0);
            _add((int)((long)ch * L), x.SubView(0, (long)ch * L), t2.SubView(0, (long)ch * L));
        }
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
