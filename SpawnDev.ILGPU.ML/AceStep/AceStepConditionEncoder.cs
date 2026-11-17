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
    private readonly AceStepEncoderLayers _layers;
    private readonly SkinTokensKernels _u;
    private readonly int _maxTokens, _lyricLayers, _timbreLayers;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _h, _cos, _sin;

    /// <summary>
    /// The even layers' bidirectional window: -1 (default) = full attention in every layer, as ComfyUI's implementation runs it (MEASURED 2026-11-17 on the 350 lyric tokens:
    /// full attention 7.5e-5, ACE-Step's own |i - j| &lt;= 128 window 0.47; AceStep_ConditionEncoder_MatchesComfyUiTap). <see cref="Window"/> gives ACE-Step's own code's behaviour.
    /// </summary>
    public int SlidingWindow { get; set; } = -1;

    /// <summary>CUDA: fp32-class activations through the tensor-core GEMMs; on by default (the reference is float32).</summary>
    public bool PreciseActivations { get => _layers.Kernels.PreciseActivations; set => _layers.Kernels.PreciseActivations = value; }

    private AceStepConditionEncoder(Accelerator acc, GpuWeights w, int maxTokens, int lyricLayers, int timbreLayers)
    {
        _acc = acc; _w = w; _maxTokens = maxTokens; _lyricLayers = lyricLayers; _timbreLayers = timbreLayers;
        _layers = new AceStepEncoderLayers(acc, w, maxTokens); _u = new SkinTokensKernels(acc);
        _h = acc.Allocate1D<float>((long)maxTokens * Dim);
        var (c, s) = AceStepEncoderLayers.RopeTables(Enumerable.Range(0, maxTokens).ToArray(), RopeTheta);
        _cos = acc.Allocate1D(c); _sin = acc.Allocate1D(s);
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
    private void Lin(FV x, string name, FV o, int rows, int K, int N, bool bias = false) => _layers.Kernels.Linear(x, B(name + ".weight"), bias ? B(name + ".bias") : null, o, rows, K, N);
    private void Layer(string b, int L, int index) => _layers.Run(b, _h.View, L, _cos.View, _sin.View, index % 2 == 0 ? SlidingWindow : -1);

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
        _layers.Dispose(); _h.Dispose(); _cos.Dispose(); _sin.Dispose();
        _w.Dispose();
    }
}
