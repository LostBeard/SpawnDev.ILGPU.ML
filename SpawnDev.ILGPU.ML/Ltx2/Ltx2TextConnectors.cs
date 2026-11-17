using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.Ltx2;

/// <summary>
/// LTX-2's text side after Gemma (diffusers' LTX2TextConnectors, Apache-2.0): the text projection (188160 -> 3840, no bias) of <see cref="Gemma3TextEncoder"/>'s features,
/// then the video and the audio connector: the n real tokens followed by learnable registers up to 1024 rows, two pre-norm transformer blocks (30 x 128 heads, no mask,
/// split RoPE on positions p / 4096), a final RMS norm. The conditioning is [1024, 7680] = video connector | audio connector.
/// Weights: the LTX-2 checkpoint's <c>text_embedding_projection.aggregate_embed</c> and <c>model.diffusion_model.{video,audio}_embeddings_connector</c> (BF16).
/// </summary>
public sealed class Ltx2TextConnectors : IDisposable
{
    public const int Width = 3840, Heads = 30, HeadDim = 128, Blocks = 2, Registers = 128, Tokens = 1024, RopeBase = 4096, Inter = 4 * Width;
    public const float Eps = 1e-6f;
    public const double RopeTheta = 10_000.0;
    private const string Projection = "text_embedding_projection.aggregate_embed.weight";
    private static readonly string[] Prefix = { "model.diffusion_model.video_embeddings_connector.", "model.diffusion_model.audio_embeddings_connector." };

    private readonly Accelerator _acc;
    private readonly StreamedModelWeights _w;
    private readonly DiTKernels _k;
    private readonly Ltx2Kernels _l;
    private readonly SkinTokensKernels _u;
    private readonly Flux2.Flux2Kernels _f;
    private readonly List<IDisposable> _owned = new();
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _proj, _x, _xn, _q, _kv, _part, _attn, _hid, _scores, _cos, _sin, _cond;

    /// <summary>Called with ("projection" [n, 3840] | "video" / "audio" [1024, 3840], the connector outputs) as <see cref="Run"/> goes.</summary>
    public Action<string, FV, int, int>? Probe;

    /// <summary>CUDA: fp32-class activations through the tensor-core GEMMs (on: the ComfyUI reference runs float32).</summary>
    public bool PreciseActivations { get => _k.PreciseActivations; set => _k.PreciseActivations = value; }

    private Ltx2TextConnectors(Accelerator accelerator, StreamedModelWeights weights)
    {
        _acc = accelerator; _w = weights;
        _k = new DiTKernels(accelerator) { PreciseActivations = true }; _l = new Ltx2Kernels(accelerator); _u = new SkinTokensKernels(accelerator); _f = new Flux2.Flux2Kernels(accelerator);
        long T = Tokens;
        _proj = Alloc(T * Width); _x = Alloc(T * Width); _xn = Alloc(T * Width); _q = Alloc(T * Width); _kv = Alloc(T * 2 * Width); _part = Alloc(T * Width);
        _attn = Alloc(T * Width); _hid = Alloc(T * Inter); _scores = Alloc((long)Heads * T * T);                       // the reference attention (non-CUDA backends) materialises the scores
        _cos = Alloc(T * (Width / 2)); _sin = Alloc(T * (Width / 2)); _cond = Alloc(T * 2 * Width);
        var frac = new float[Tokens];
        for (int p = 0; p < Tokens; p++) frac[p] = p / (float)RopeBase;
        var (c, s) = Ltx2Kernels.SplitRopeTables(frac, Width / 2, 0, RopeTheta);           // 1920 frequencies, cut 30 x 64 per head
        _cos.View.CopyFromCPU(c); _sin.View.CopyFromCPU(s);
    }

    private MemoryBuffer1D<float, Stride1D.Dense> Alloc(long n) { var b = _acc.Allocate1D<float>(Math.Max(1, n)); _owned.Add(b); return b; }

    /// <summary>Open the LTX-2 checkpoint and load the text projection and both connectors (about 2.9 GB as stored; the rest of the file is not read).</summary>
    public static async Task<Ltx2TextConnectors> OpenAsync(Accelerator accelerator, IModelSource source, string repo, string file, CancellationToken ct = default)
    {
        static int BlockOf(string n) => n == Projection ? -1 : n.StartsWith(Prefix[0], StringComparison.Ordinal) ? 0 : n.StartsWith(Prefix[1], StringComparison.Ordinal) ? 1 : StreamedModelWeights.HostOnly;
        var w = await StreamedModelWeights.OpenAsync(accelerator, source, repo, file, Projection, BlockOf, 2, long.MaxValue, null, ct);
        try
        {
            foreach (var p in Prefix)
                if (!w.HeaderNames.Contains(p + "learnable_registers") || !w.HeaderNames.Contains(p + $"transformer_1d_blocks.{Blocks - 1}.ff.net.2.weight"))
                    throw new InvalidDataException($"'{file}' has no LTX-2 text connector at {p}");
            return new Ltx2TextConnectors(accelerator, w);
        }
        catch { w.Dispose(); throw; }
    }

    private void Lin(GpuWeights w, FV x, string name, FV o, int M, int K, int N, DiTKernels.Act act = DiTKernels.Act.None)
    {
        float s = _w.ActivationScale(w, name);
        if (s != 1f) throw new NotSupportedException($"{name}: a scaled matrix in the text connectors");    // LTX-2's connectors are BF16
        string bias = name[..^"weight".Length] + "bias";
        _k.Linear(x, _w.MatrixExact(w, name), w.Has(bias) ? w.BF16(bias) : null, o, M, K, N, act);
    }

    /// <summary>
    /// The conditioning [1024, 7680] (video | audio) for <paramref name="features"/> [n, 188160] (<see cref="Gemma3TextEncoder.EncodeAsync"/>). Valid until the next call.
    /// </summary>
    public async Task<FV> RunAsync(FV features, int n, CancellationToken ct = default)
    {
        if (n < 1 || n > Tokens) throw new ArgumentOutOfRangeException(nameof(n), $"1 to {Tokens} tokens");
        Lin(_w.Base, features, Projection, _proj.View, n, Gemma3TextEncoder.FeatureWidth, Width);
        Probe?.Invoke("projection", _proj.View, n, Width);
        for (int m = 0; m < 2; m++)
        {
            var (w, streamed) = await _w.BlockAsync(m, ct);
            try { Connector(w, Prefix[m], n); }
            finally { if (streamed) { await _acc.SynchronizeAsync(); w.Dispose(); } }
            Probe?.Invoke(m == 0 ? "video" : "audio", _x.View, Tokens, Width);
            _f.CopyCols(_x.View, _cond.View, Tokens, Width, Width, 2 * Width, m * Width);
        }
        return _cond.View;
    }

    private void Connector(GpuWeights w, string p, int n)
    {
        const int T = Tokens, C = Width;
        _l.FillRegisters(_proj.View, w.BF16(p + "learnable_registers"), _x.View, n, T, C, Registers);
        for (int b = 0; b < Blocks; b++)
        {
            string a = $"{p}transformer_1d_blocks.{b}.";
            _l.RmsNormPlain(_x.View, _xn.View, T, C, Eps);
            Lin(w, _xn.View, a + "attn1.to_q.weight", _q.View, T, C, C);
            Lin(w, _xn.View, a + "attn1.to_k.weight", _part.View, T, C, C); _f.CopyCols(_part.View, _kv.View, T, C, C, 2 * C, 0);
            Lin(w, _xn.View, a + "attn1.to_v.weight", _part.View, T, C, C); _f.CopyCols(_part.View, _kv.View, T, C, C, 2 * C, C);
            _u.RmsNormHeads(_q.View, w.BF16(a + "attn1.q_norm.weight"), T, C, 0, 1, C, Eps);          // over all heads at once, with weights
            _u.RmsNormHeads(_kv.View, w.BF16(a + "attn1.k_norm.weight"), T, 2 * C, 0, 1, C, Eps);
            _l.RopeSplit(_q.View, _cos.View, _sin.View, T, C, 0, Heads, HeadDim);
            _l.RopeSplit(_kv.View, _cos.View, _sin.View, T, 2 * C, 0, Heads, HeadDim);
            _k.Attention(_q.View, C, 0, _kv.View, 2 * C, 0, C, _scores.View, _attn.View, T, T, Heads, HeadDim);
            Lin(w, _attn.View, a + "attn1.to_out.0.weight", _part.View, T, C, C);
            _k.GateResidual(_x.View, _part.View, _xn.View, -1, T * C, C);
            _l.RmsNormPlain(_x.View, _xn.View, T, C, Eps);
            Lin(w, _xn.View, a + "ff.net.0.proj.weight", _hid.View, T, C, Inter, DiTKernels.Act.GeluTanh);
            Lin(w, _hid.View, a + "ff.net.2.weight", _part.View, T, Inter, C);
            _k.GateResidual(_x.View, _part.View, _xn.View, -1, T * C, C);
        }
        _l.RmsNormPlain(_x.View, _x.View, T, C, Eps);
    }

    public void Dispose()
    {
        foreach (var o in _owned) o.Dispose();
        _owned.Clear();
        _w.Dispose();
    }
}
