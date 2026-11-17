using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;
using System.Text.Json;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;
using BV = ILGPU.Runtime.ArrayView1D<ILGPU.BFloat16, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.AceStep;

/// <summary>
/// ACE-Step 1.5's condition encoder (<c>AceStepConditionEncoder</c>, the <c>encoder.*</c> tensors of ComfyUI's acestep_v1.5_turbo): the DiT's cross-attention tokens
/// [lyrics | timbre | text] (batch 1: every token valid, so ACE-Step's pack_sequences is a plain concatenation).
/// <list type="bullet">
/// <item>text: text_projector (Linear 1024 -> 2048, no bias) of the caption's Qwen3-0.6B states;</item>
/// <item>lyrics: lyric_encoder = embed_tokens (Linear 1024 -> 2048 + bias) of the lyric token embeddings, 8 encoder layers, norm;</item>
/// <item>timbre: timbre_encoder = embed_tokens (Linear 64 -> 2048 + bias) of the reference latent [T, 64], 4 encoder layers, norm, frame 0's row (one token).</item>
/// </list>
/// An encoder layer: h += o(attn(rope(qk_norm(q, k, v of RMS(h))))); h += down(silu(gate(RMS'(h))) up(..)) - bidirectional GQA 16 / 8 x 128, full attention in every
/// layer as ComfyUI runs it (<see cref="SlidingWindow"/>), RoPE theta 1e6 on positions 0 .. L - 1.
/// </summary>
public sealed class AceStepConditionEncoder : IDisposable
{
    public const int Dim = 2048, Heads = 16, KvHeads = 8, HeadDim = 128, Mlp = 6144, TextWidth = 1024, LatentChannels = 64, Window = 128;
    private const float Eps = 1e-6f, RopeTheta = 1e6f;

    private readonly Accelerator _acc;
    private readonly GpuWeights _w;
    private readonly DiTKernels _k;
    private readonly SkinTokensKernels _u;
    private readonly AceStepKernels _a;
    private readonly int _maxTokens, _lyricLayers, _timbreLayers;
    private readonly List<IDisposable> _buffers = new();
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _h, _n, _q, _kk, _v, _att, _o, _g, _up, _gu, _scores, _cos, _sin;

    /// <summary>
    /// The even layers' bidirectional window: -1 (default) = full attention in every layer, as ComfyUI's implementation runs it (MEASURED 2026-11-17 on the 350 lyric tokens:
    /// full attention 7.5e-5, ACE-Step's own |i - j| &lt;= 128 window 0.47; AceStep_ConditionEncoder_MatchesComfyUiTap). <see cref="Window"/> gives ACE-Step's own code's behaviour.
    /// </summary>
    public int SlidingWindow { get; set; } = -1;

    /// <summary>CUDA: fp32-class activations through the tensor-core GEMMs; on by default (the reference is float32).</summary>
    public bool PreciseActivations { get => _k.PreciseActivations; set => _k.PreciseActivations = value; }

    private AceStepConditionEncoder(Accelerator acc, GpuWeights w, int maxTokens, int lyricLayers, int timbreLayers)
    {
        _acc = acc; _w = w; _maxTokens = maxTokens; _lyricLayers = lyricLayers; _timbreLayers = timbreLayers;
        _k = new DiTKernels(acc) { PreciseActivations = true }; _u = new SkinTokensKernels(acc); _a = new AceStepKernels(acc);
        MemoryBuffer1D<float, Stride1D.Dense> A(long n) { var b = acc.Allocate1D<float>(Math.Max(1, n)); _buffers.Add(b); return b; }
        long N = maxTokens, kv = KvHeads * HeadDim;
        _h = A(N * Dim); _n = A(N * Dim); _q = A(N * Dim); _kk = A(N * kv); _v = A(N * kv); _att = A(N * Dim); _o = A(N * Dim);
        _g = A(N * Mlp); _up = A(N * Mlp); _gu = A(N * Mlp); _scores = A((long)Heads * N * N); _cos = A(N * HeadDim / 2); _sin = A(N * HeadDim / 2);
        int half = HeadDim / 2;
        var c = new float[N * half]; var s = new float[N * half];
        for (int k = 0; k < half; k++)
        {
            float inv = 1f / MathF.Pow(RopeTheta, 2f * k / HeadDim);
            for (int p = 0; p < N; p++) { float ang = p * inv; c[p * half + k] = (float)Math.Cos(ang); s[p * half + k] = (float)Math.Sin(ang); }
        }
        _cos.View.CopyFromCPU(c); _sin.View.CopyFromCPU(s);
    }

    /// <summary>Load the condition encoder from ComfyUI's acestep_v1.5_turbo file; buffers sized for sequences up to <paramref name="maxTokens"/> (the longer of lyrics and reference frames).</summary>
    public static async Task<AceStepConditionEncoder> LoadAsync(Accelerator accelerator, IModelSource source, string repo, string file, int maxTokens, CancellationToken ct = default)
    {
        GpuWeights w;
        await using (var stream = await source.OpenAsync(repo, file, ct))
        {
            using var empty = JsonDocument.Parse("{}");
            w = await GpuWeights.LoadAsync(accelerator, stream, empty.RootElement, file, n => n.StartsWith("encoder.", StringComparison.Ordinal), ct);
        }
        try
        {
            int Count(string p) { int n = 0; while (w.Has($"{p}.layers.{n}.input_layernorm.weight")) n++; return n; }
            return new AceStepConditionEncoder(accelerator, w, maxTokens, Count("encoder.lyric_encoder"), Count("encoder.timbre_encoder"));
        }
        catch { w.Dispose(); throw; }
    }

    private BV B(string name) => _w.BF16(name);
    private void Lin(FV x, string name, FV o, int rows, int K, int N, bool bias = false) => _k.Linear(x, B(name + ".weight"), bias ? B(name + ".bias") : null, o, rows, K, N);

    // one encoder layer on _h [L, D]
    private void Layer(string b, int L, int index)
    {
        int D = Dim, kv = KvHeads * HeadDim;
        _u.RmsNorm(_h.View, _n.View, B(b + "input_layernorm.weight"), L, D, Eps);
        Lin(_n.View, b + "self_attn.q_proj", _q.View, L, D, D);
        Lin(_n.View, b + "self_attn.k_proj", _kk.View, L, D, kv);
        Lin(_n.View, b + "self_attn.v_proj", _v.View, L, D, kv);
        _u.RmsNormHeads(_q.View, B(b + "self_attn.q_norm.weight"), L, D, 0, Heads, HeadDim, Eps);
        _u.RmsNormHeads(_kk.View, B(b + "self_attn.k_norm.weight"), L, kv, 0, KvHeads, HeadDim, Eps);
        _u.RopeTables(_q.View, _cos.View, _sin.View, L, D, 0, Heads, HeadDim);
        _u.RopeTables(_kk.View, _cos.View, _sin.View, L, kv, 0, KvHeads, HeadDim);
        _a.Gqa(_q.View, D, _kk.View, kv, _v.View, kv, _scores.View, _att.View, L, L, Heads, HeadDim, Heads / KvHeads, index % 2 == 0 ? SlidingWindow : -1);
        Lin(_att.View, b + "self_attn.o_proj", _o.View, L, D, D);
        _k.GateResidual(_h.View, _o.View, _o.View, -1, L * D, D);
        _u.RmsNorm(_h.View, _n.View, B(b + "post_attention_layernorm.weight"), L, D, Eps);
        Lin(_n.View, b + "mlp.gate_proj", _g.View, L, D, Mlp);
        Lin(_n.View, b + "mlp.up_proj", _up.View, L, D, Mlp);
        _a.SiluMul(_g.View, _up.View, _gu.View, L * Mlp);
        Lin(_gu.View, b + "mlp.down_proj", _o.View, L, Mlp, D);
        _k.GateResidual(_h.View, _o.View, _o.View, -1, L * D, D);
    }

    /// <summary>
    /// The DiT's condition tokens into <paramref name="output"/> [lyrics + 1 + textTokens, 2048]: lyric embeddings <paramref name="lyrics"/> [lyricTokens, 1024], the
    /// reference latent <paramref name="refer"/> [referFrames, 64] (time-major; text-to-music: ACE-Step's silence latent), caption states <paramref name="text"/> [textTokens, 1024].
    /// </summary>
    public void Encode(FV text, int textTokens, FV lyrics, int lyricTokens, FV refer, int referFrames, FV output)
    {
        if (lyricTokens > _maxTokens || referFrames > _maxTokens) throw new ArgumentException($"{lyricTokens} lyric tokens / {referFrames} reference frames: the encoder was sized for {_maxTokens}");
        int D = Dim;
        // lyrics -> rows 0 .. lyricTokens - 1
        Lin(lyrics, "encoder.lyric_encoder.embed_tokens", _h.View, lyricTokens, TextWidth, D, bias: true);
        for (int l = 0; l < _lyricLayers; l++) Layer($"encoder.lyric_encoder.layers.{l}.", lyricTokens, l);
        _u.RmsNorm(_h.View, output.SubView(0, (long)lyricTokens * D), B("encoder.lyric_encoder.norm.weight"), lyricTokens, D, Eps);
        // timbre -> row lyricTokens (frame 0 after the final norm)
        Lin(refer, "encoder.timbre_encoder.embed_tokens", _h.View, referFrames, LatentChannels, D, bias: true);
        for (int l = 0; l < _timbreLayers; l++) Layer($"encoder.timbre_encoder.layers.{l}.", referFrames, l);
        _u.RmsNorm(_h.View, output.SubView((long)lyricTokens * D, D), B("encoder.timbre_encoder.norm.weight"), 1, D, Eps);
        // text -> the last textTokens rows
        Lin(text, "encoder.text_projector", output.SubView((long)(lyricTokens + 1) * D, (long)textTokens * D), textTokens, TextWidth, D);
    }

    public void Dispose()
    {
        foreach (var b in _buffers) b.Dispose();
        _buffers.Clear();
        _w.Dispose();
    }
}
