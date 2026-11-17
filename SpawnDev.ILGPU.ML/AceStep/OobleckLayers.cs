using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Trellis2;
using System.Text.Json;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;
using HV = ILGPU.Runtime.ArrayView1D<ILGPU.Half, ILGPU.Stride1D.Dense>;
using IV = ILGPU.Runtime.ArrayView1D<int, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.AceStep;

/// <summary>
/// The pieces the Oobleck VAE's encoder and decoder share, on CHANNEL-LAST activations [L, C]: weight-normalised convolutions fused at load (weight = g v / |v| over all but
/// the first dimension, torch.nn.utils.weight_norm dim 0), Snake activations x + sin^2(exp(alpha) x) / (exp(beta) + 1e-9) (log-scale parameters as stored), and the residual
/// unit (Snake, Conv1d k7 dilated d with padding 3 d, Snake, Conv1d k1, added to its input).
/// <para>Convolutions are implicit GEMMs on the library's tiled GEMM (<see cref="SparseKernels.ConvTapsExact"/>): a neighbour table gives every output row its input rows (padding
/// = -1), fp32 activations throughout, each fp32 weight split into its high and low fp16 halves (two GEMMs: the weight's full 22-bit mantissa), the fp32 bias added after.
/// ConvTranspose1d (kernel 2 s, stride s) runs polyphase: output phase r = ox mod s only meets 2 taps (k0 = (r + pad) mod s and k0 + s), so each phase is a 2-tap GEMM whose rows
/// are scattered to positions q s + r. A conv whose taps x channels are not a multiple of the GEMM's k step (the encoder's first, 2 channels) runs on <see cref="Conv1DKernel"/>.</para>
/// </summary>
internal sealed class OobleckLayers : IDisposable
{
    private const int TableStride = 27;                     // the tiled GEMM's gathered rows read nbr[m * 27 + tap]

    private readonly Accelerator _acc;
    private readonly SparseKernels _sk;
    private readonly Conv1DKernel _conv;
    private readonly Action<Index1D, FV, FV, FV, FV, int> _snake;
    private readonly Action<Index1D, FV, FV> _add;
    private readonly Action<Index1D, FV, FV, int> _addBias;
    private readonly Action<Index1D, FV, FV, FV, int, int, int> _scatterPhase;
    private readonly Action<Index1D, FV, FV, int, int> _transpose;
    private readonly Dictionary<string, Conv> _convs = new();
    private readonly Dictionary<string, (MemoryBuffer1D<float, Stride1D.Dense> A, MemoryBuffer1D<float, Stride1D.Dense> InvB)> _snakes = new();
    private readonly Dictionary<string, MemoryBuffer1D<int, Stride1D.Dense>> _tables = new();
    private MemoryBuffer1D<float, Stride1D.Dense>? _phase;
    private readonly List<IDisposable> _retired = new();

    /// <param name="W">fp32 as stored after the weight-norm fusion: Conv1d [out, in, k], ConvTranspose1d [in, out, k] (the fallback kernel's layout).</param>
    /// <param name="Hi">The GEMM weight's high fp16 half, gather layout: Conv1d [out][k][in]; ConvTranspose1d per phase [s][out][2][in].</param>
    internal sealed record Conv(MemoryBuffer1D<float, Stride1D.Dense> W, MemoryBuffer1D<float, Stride1D.Dense> B, MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>? Hi,
        MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>? Lo, int InC, int OutC, int K, bool Transposed);

    public OobleckLayers(Accelerator acc)
    {
        _acc = acc;
        _sk = new SparseKernels(acc); _conv = new Conv1DKernel(acc);
        _snake = acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, int>(SnakeImpl);
        _add = acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(AddImpl);
        _addBias = acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int>(AddBiasImpl);
        _scatterPhase = acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int>(ScatterPhaseImpl);
        _transpose = acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int>(TransposeImpl);
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

    // the high / low fp16 halves of fp32 values: hi = half(v), lo = half(v - hi)
    private (MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>, MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>) Split(float[] v)
    {
        var hi = new global::ILGPU.Half[v.Length]; var lo = new global::ILGPU.Half[v.Length];
        for (int i = 0; i < v.Length; i++) { hi[i] = (global::ILGPU.Half)v[i]; lo[i] = (global::ILGPU.Half)(v[i] - (float)hi[i]); }
        return (_acc.Allocate1D(hi), _acc.Allocate1D(lo));
    }

    /// <summary>A Conv1d (or, <paramref name="transposed"/>, a ConvTranspose1d) with weight norm: fused once, laid out for the GEMM.</summary>
    public async Task AddConvAsync(GpuWeights w, string name, string key, bool transposed = false)
    {
        var shape = w.Shape(name + ".weight_v");
        var v = await HostAsync(w, name + ".weight_v");
        var g = await HostAsync(w, name + ".weight_g");
        int d0 = shape[0], k = shape[2], per = v.Length / d0;
        var fused = new float[v.Length];
        for (int o = 0; o < d0; o++)
        {
            double ss = 0;
            for (int j = 0; j < per; j++) { double x = v[o * per + j]; ss += x * x; }
            double scale = g[o] / Math.Sqrt(ss);
            for (int j = 0; j < per; j++) fused[o * per + j] = (float)(v[o * per + j] * scale);
        }
        int inC = transposed ? shape[0] : shape[1], outC = transposed ? shape[1] : shape[0];
        // no bias (the decoder's last conv): explicit zeros, the same sum
        var bias = _acc.Allocate1D(w.Has(name + ".bias") ? await HostAsync(w, name + ".bias") : new float[outC]);
        MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>? hi = null, lo = null;
        if (!transposed)
        {
            if (k * inC % TiledGemm.BK == 0)
            {
                var gather = new float[fused.Length];                                            // [out][in][k] -> [out][k][in]
                for (int o = 0; o < outC; o++) for (int i = 0; i < inC; i++) for (int t = 0; t < k; t++) gather[((long)o * k + t) * inC + i] = fused[((long)o * inC + i) * k + t];
                (hi, lo) = Split(gather);
            }
        }
        else
        {
            int s = k / 2;                                                                       // kernel 2 s: each output phase meets taps k0 and k0 + s
            int pad = (s + 1) / 2;
            var phases = new float[(long)s * outC * 2 * inC];                                   // [s][out][2][in]
            for (int r = 0; r < s; r++)
            {
                int k0 = (r + pad) % s;
                for (int o = 0; o < outC; o++) for (int tap = 0; tap < 2; tap++) for (int i = 0; i < inC; i++)
                    phases[(((long)r * outC + o) * 2 + tap) * inC + i] = fused[((long)i * outC + o) * k + k0 + tap * s];
            }
            (hi, lo) = Split(phases);
        }
        _convs[key] = new Conv(_acc.Allocate1D(fused), bias, hi, lo, inC, outC, k, transposed);
    }

    public async Task AddSnakeAsync(GpuWeights w, string name, string key)
    {
        var alpha = await HostAsync(w, name + ".alpha");
        var beta = await HostAsync(w, name + ".beta");
        var a = alpha.Select(x => (float)Math.Exp(x)).ToArray();
        var invB = beta.Select(x => (float)(1.0 / (Math.Exp(x) + 1e-9))).ToArray();
        _snakes[key] = (_acc.Allocate1D(a), _acc.Allocate1D(invB));
    }

    /// <summary>The 3 residual units of a block (dilations 1, 3, 9), from the tensors <paramref name="unitPrefix"/>(n) + "0" .. "3", keyed <paramref name="key"/>.r{n}.</summary>
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

    // the neighbour table of a conv: output row m reads input rows table[m * 27 + tap] (-1 = padding), built once per shape
    private IV Table(string id, int rows, Func<int, int, int> source, int taps)
    {
        if (_tables.TryGetValue(id, out var t)) return t.View;
        var nbr = new int[(long)rows * TableStride];
        Array.Fill(nbr, -1);
        for (int m = 0; m < rows; m++) for (int k = 0; k < taps; k++) nbr[(long)m * TableStride + k] = source(m, k);
        var buf = _acc.Allocate1D(nbr);
        _tables[id] = buf;
        return buf.View;
    }

    /// <summary>Conv1d of channel-last x [L, inC] into y [L', outC], L' = (L + 2 pad - dilation (k - 1) - 1) / stride + 1.</summary>
    public void Conv1d(string key, FV x, FV y, int L, int pad, int dilation = 1, int stride = 1)
    {
        var cv = _convs[key];
        int k = cv.K, outL = (L + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        if (cv.Hi == null) throw new InvalidOperationException($"{key}: {k} x {cv.InC} is not a multiple of the GEMM's k step; use Conv1dChannelMajor");
        var nbr = Table($"c{L}.{k}.{pad}.{dilation}.{stride}", outL, (m, t) => { int src = m * stride - pad + t * dilation; return src >= 0 && src < L ? src : -1; }, k);
        _sk.ConvTapsExact(x, nbr, cv.Hi.View, y, outL, cv.InC, cv.OutC, k, false);
        _sk.ConvTapsExact(x, nbr, cv.Lo!.View, y, outL, cv.InC, cv.OutC, k, true);
        _addBias(outL * cv.OutC, y.SubView(0, (long)outL * cv.OutC), cv.B.View, cv.OutC);
    }

    /// <summary>Conv1d of channel-MAJOR x [inC, L] into channel-major y (the direct kernel: convs too narrow for the GEMM).</summary>
    public void Conv1dChannelMajor(string key, FV x, FV y, int L, int pad, int dilation = 1, int stride = 1)
    {
        var cv = _convs[key];
        int outL = (L + 2 * pad - dilation * (cv.K - 1) - 1) / stride + 1;
        _conv.Forward(x.SubView(0, (long)cv.InC * L), cv.W.View, cv.B.View, y.SubView(0, (long)cv.OutC * outL), cv.InC, L, cv.OutC, cv.K, stride, pad, dilation);
    }

    /// <summary>ConvTranspose1d (kernel 2 s, stride s, padding ceil(s / 2)) of channel-last x [L, inC] into y [s L, outC]: s 2-tap GEMMs, one per output phase.</summary>
    public void Upsample(string key, FV x, FV y, int L, int s)
    {
        var cv = _convs[key];
        int pad = (s + 1) / 2;
        long phaseElems = (long)L * cv.OutC;
        if (_phase == null || _phase.Length < phaseElems) { if (_phase != null) _retired.Add(_phase); _phase = _acc.Allocate1D<float>(phaseElems); }
        long per = (long)cv.OutC * 2 * cv.InC;
        for (int r = 0; r < s; r++)
        {
            int k0 = (r + pad) % s, off = (r + pad - k0) / s;                                   // tap k0 reads input q + off, tap k0 + s reads q + off - 1
            var nbr = Table($"u{L}.{s}.{r}", L, (q, t) => { int src = q + off - t; return src >= 0 && src < L ? src : -1; }, 2);
            _sk.ConvTapsExact(x, nbr, cv.Hi!.View.SubView(r * per, per), _phase.View, L, cv.InC, cv.OutC, 2, false);
            _sk.ConvTapsExact(x, nbr, cv.Lo!.View.SubView(r * per, per), _phase.View, L, cv.InC, cv.OutC, 2, true);
            _scatterPhase((int)phaseElems, _phase.View, y, cv.B.View, cv.OutC, s, r);
        }
    }

    public void Snake(string key, FV x, FV y, int C, int L)
    {
        var (a, invB) = _snakes[key];
        _snake((int)((long)C * L), x.SubView(0, (long)C * L), y.SubView(0, (long)C * L), a.View, invB.View, C);
    }

    /// <summary>The 3 residual units on channel-last <paramref name="x"/> [L, C] in place; <paramref name="t1"/>, <paramref name="t2"/> are scratch of the same size.</summary>
    public void ResidualUnits(string key, FV x, FV t1, FV t2, int C, int L)
    {
        for (int r = 0; r < 3; r++)
        {
            int dil = r == 0 ? 1 : r == 1 ? 3 : 9;
            Snake($"{key}.r{r}.s0", x, t1, C, L);
            Conv1d($"{key}.r{r}.c0", t1, t2, L, 3 * dil, dil);
            Snake($"{key}.r{r}.s1", t2, t1, C, L);
            Conv1d($"{key}.r{r}.c1", t1, t2, L, 0);
            _add((int)((long)C * L), x.SubView(0, (long)C * L), t2.SubView(0, (long)C * L));
        }
    }

    /// <summary>dst [cols, rows] = src [rows, cols] transposed (channel-major &lt;-&gt; channel-last at the nets' edges).</summary>
    public void Transpose(FV src, FV dst, int rows, int cols) => _transpose((int)((long)rows * cols), src, dst, rows, cols);

    private static void SnakeImpl(Index1D i, FV x, FV y, FV a, FV invB, int C)
    {
        int c = i % C;
        float v = x[i];
        float s = MathF.Sin(a[c] * v);
        y[i] = v + invB[c] * s * s;
    }

    private static void AddImpl(Index1D i, FV acc, FV x) => acc[i] += x[i];
    private static void AddBiasImpl(Index1D i, FV y, FV b, int C) => y[i] += b[i % C];

    // phase rows q -> output rows q s + r, plus the bias
    private static void ScatterPhaseImpl(Index1D i, FV phase, FV y, FV b, int C, int s, int r)
    {
        int q = i / C, c = i % C;
        y[((long)q * s + r) * C + c] = phase[i] + b[c];
    }

    private static void TransposeImpl(Index1D i, FV src, FV dst, int rows, int cols)
    {
        int r = i / cols, c = i % cols;
        dst[(long)c * rows + r] = src[i];
    }

    public void Dispose()
    {
        foreach (var cv in _convs.Values) { cv.W.Dispose(); cv.B.Dispose(); cv.Hi?.Dispose(); cv.Lo?.Dispose(); }
        foreach (var (a, b) in _snakes.Values) { a.Dispose(); b.Dispose(); }
        foreach (var t in _tables.Values) t.Dispose();
        _phase?.Dispose(); foreach (var r in _retired) r.Dispose();
        _convs.Clear(); _snakes.Clear(); _tables.Clear(); _retired.Clear();
        _conv.Dispose();
    }
}
