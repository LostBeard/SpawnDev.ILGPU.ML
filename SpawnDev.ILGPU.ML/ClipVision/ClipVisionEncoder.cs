using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.ClipVision;

using FV = ArrayView1D<float, Stride1D.Dense>;
using HV = ArrayView1D<global::ILGPU.Half, Stride1D.Dense>;

/// <summary>
/// A CLIP vision tower in Hugging Face's <c>CLIPVisionModelWithProjection</c> layout (transformers' <c>modeling_clip.py</c>, Apache-2.0), loaded from ComfyUI's
/// <c>clip_vision_h.safetensors</c> (OpenCLIP ViT-H/14: width 1280, 32 layers, 16 heads of 80, MLP 5120 with the exact GELU, 224 px in 14 px patches = 256 + 1 tokens,
/// projection to 1024; fp16 weights kept fp16). What ComfyUI's CLIPVisionEncode returns, for Wan 2.1 i2v and the other templates that read it:
/// <code>
/// x = [class | patch_embedding(pixels)] + position_embedding        [257, 1280]
/// x = pre_layrnorm(x)
/// 32 x { x += out_proj(attn(q, k, v of layer_norm1(x)));  x += fc2(gelu(fc1(layer_norm2(x)))) }
/// last_hidden_state = x after the last layer,  penultimate_hidden_states = x after the one before
/// image_embeds = visual_projection(post_layernorm(last_hidden_state[0]))                             [1024]
/// </code>
/// Every linear runs fp32 on the exact fp16 weights (<see cref="SparseKernels.PreciseActivations"/>; ComfyUI computes in fp32). Input: CLIP-normalised pixels NCHW [3, 224, 224].
/// </summary>
public sealed class ClipVisionEncoder : IDisposable
{
    private readonly Accelerator _acc;
    private readonly GpuWeights _w;
    private readonly SdKernels _sd;
    private readonly SparseKernels _sk;
    private readonly DiTKernels _k;
    private readonly Flux2Kernels _f;
    private readonly List<IDisposable> _owned = new();
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _patches, _x, _n, _qkv, _part, _a, _h, _hid, _last, _pen, _pooled, _embeds, _scores;
    private readonly MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense> _zeroBias;

    /// <summary>Hidden width (ViT-H: 1280).</summary>
    public int Width { get; }
    /// <summary>Encoder layers (ViT-H: 32).</summary>
    public int Layers { get; }
    /// <summary>Attention heads (ViT-H: 16 of 80).</summary>
    public int Heads { get; }
    /// <summary>Patch side in pixels (14).</summary>
    public int Patch { get; }
    /// <summary>Input side in pixels (224).</summary>
    public int ImageSize { get; }
    /// <summary>Tokens: the class token plus one per patch (257).</summary>
    public int Tokens { get; }
    /// <summary>Width of <see cref="ImageEmbeds"/> (ViT-H: 1024; 0 without a projection).</summary>
    public int ProjectionWidth { get; }
    /// <summary>MLP width (ViT-H: 5120).</summary>
    public int MlpWidth { get; }
    /// <summary>LayerNorm epsilon (Hugging Face's CLIP default).</summary>
    public const float Eps = 1e-5f;

    /// <summary>The last layer's output [Tokens, Width] (CLIP_VISION_OUTPUT.last_hidden_state).</summary>
    public FV LastHiddenState => _last.View;
    /// <summary>The output of the layer before the last [Tokens, Width] (CLIP_VISION_OUTPUT.penultimate_hidden_states: what Wan 2.1 i2v reads).</summary>
    public FV PenultimateHiddenState => _pen.View;
    /// <summary>The projected class token [ProjectionWidth] (CLIP_VISION_OUTPUT.image_embeds).</summary>
    public FV ImageEmbeds => _embeds.View;
    /// <summary>Checkpoints for tests: "embeddings" (with positions), "pre_layrnorm", "layer{n}" (1-based), "pooled" (post_layernorm of the class token).</summary>
    public Action<string, FV, int, int>? Probe;

    private ClipVisionEncoder(Accelerator acc, GpuWeights w)
    {
        _acc = acc; _w = w;
        var pe = w.Shape("vision_model.embeddings.patch_embedding.weight");               // [C, 3, p, p]
        Width = pe[0]; Patch = pe[2];
        Tokens = w.Shape("vision_model.embeddings.position_embedding.weight")[0];
        int side = (int)Math.Round(Math.Sqrt(Tokens - 1));
        if (side * side != Tokens - 1 || pe[1] != 3 || pe[3] != Patch) throw new InvalidDataException($"position embedding of {Tokens} rows: not a square grid + class token");
        ImageSize = side * Patch;
        while (w.Has($"vision_model.encoder.layers.{Layers}.layer_norm1.weight")) Layers++;
        MlpWidth = w.Shape("vision_model.encoder.layers.0.mlp.fc1.weight")[0];
        ProjectionWidth = w.Has("visual_projection.weight") ? w.Shape("visual_projection.weight")[0] : 0;
        // the file carries no head count: the known CLIP vision widths (ViT-B 12 x 64, ViT-L 16 x 64, OpenCLIP ViT-H 16 x 80, ViT-bigG 16 x 104)
        Heads = Width switch { 768 => 12, 1024 => 16, 1280 => 16, 1664 => 16, _ => throw new NotSupportedException($"CLIP vision width {Width}: unknown head count") };
        _sd = new SdKernels(acc); _sk = _sd.Sparse; _sk.PreciseActivations = true; _k = new DiTKernels(acc); _f = new Flux2Kernels(acc);
        long T = Tokens, C = Width;
        _patches = Alloc((T - 1) * 3 * Patch * Patch); _x = Alloc(T * C); _n = Alloc(T * C); _qkv = Alloc(T * 3 * C); _part = Alloc(T * C); _a = Alloc(T * C); _h = Alloc(T * C);
        _hid = Alloc(T * MlpWidth); _last = Alloc(T * C); _pen = Alloc(T * C); _pooled = Alloc(C); _embeds = Alloc(Math.Max(1, ProjectionWidth));
        _scores = Alloc(_k.UsesFlashAttention(Width / Heads) ? 1 : (long)Heads * T * T);
        _zeroBias = _acc.Allocate1D<global::ILGPU.Half>(Math.Max(C, ProjectionWidth)); _zeroBias.MemSetToZero(); _owned.Add(_zeroBias);
    }

    private MemoryBuffer1D<float, Stride1D.Dense> Alloc(long n) { var b = _acc.Allocate1D<float>(Math.Max(1, n)); _owned.Add(b); return b; }

    /// <summary>Load a CLIP vision file (ComfyUI's clip_vision_*.safetensors: Hugging Face names under <c>vision_model.</c>, <c>visual_projection</c>) from a model source.</summary>
    public static async Task<ClipVisionEncoder> LoadAsync(Accelerator accelerator, IModelSource source, string repo, string file, CancellationToken ct = default)
    {
        await using var stream = await source.OpenAsync(repo, file, ct);
        using var empty = JsonDocument.Parse("{}");
        // position_ids is an I64 index buffer (arange), not a weight
        var w = await GpuWeights.LoadAsync(accelerator, stream, empty.RootElement, $"{repo}/{file}", n => !n.EndsWith("position_ids", StringComparison.Ordinal) && !n.StartsWith("text_model.", StringComparison.Ordinal), ct);
        try { return new ClipVisionEncoder(accelerator, w); }
        catch { w.Dispose(); throw; }
    }

    // ─────────────────────────────── kernels ───────────────────────────────

    private Action<Index1D, FV, FV, int, int>? _patchify;
    // NCHW pixels -> patch rows [(S/p)^2, 3 p p] in the conv weight's (c, py, px) order
    private static void PatchifyImpl(Index1D i, FV px, FV o, int S, int p)
    {
        int K = 3 * p * p, f = i % K, tok = i / K, g = S / p;
        int c = f / (p * p), py = (f / p) % p, pxx = f % p, ty = tok / g, tx = tok % g;
        o[i] = px[(long)c * S * S + (long)(ty * p + py) * S + tx * p + pxx];
    }

    private Action<Index1D, FV, FV, HV, HV, int>? _embed;
    // x[0] = class + pos[0];  x[t] = patch[t - 1] + pos[t]
    private static void EmbedImpl(Index1D i, FV patchEmb, FV x, HV cls, HV pos, int C)
    {
        int t = i / C, c = i % C;
        float v = t == 0 ? (float)cls[c] : patchEmb[(long)(t - 1) * C + c];
        x[i] = v + (float)pos[i];
    }

    private void Lin(FV x, string name, FV o, int M, int K, int N)
        => _sk.LinearF16(x, _w.F16(name + ".weight"), _w.Has(name + ".bias") ? _w.F16(name + ".bias") : _zeroBias.View.SubView(0, N), o, M, K, N);

    /// <summary>Run the tower on CLIP-normalised pixels NCHW [3, ImageSize, ImageSize]; the results are in <see cref="LastHiddenState"/>, <see cref="PenultimateHiddenState"/>, <see cref="ImageEmbeds"/>.</summary>
    public void Encode(FV pixels)
    {
        int S = ImageSize, p = Patch, T = Tokens, C = Width, D = C / Heads, P = T - 1, K = 3 * p * p;
        if (pixels.Length < 3L * S * S) throw new ArgumentException($"pixels: 3 x {S} x {S} values");
        _patchify ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int>(PatchifyImpl);
        _embed ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, HV, HV, int>(EmbedImpl);
        _patchify(P * K, pixels, _patches.View, S, p);
        Lin(_patches.View, "vision_model.embeddings.patch_embedding", _h.View, P, K, C);               // conv with kernel = stride, no bias
        _embed(T * C, _h.View, _x.View, _w.F16("vision_model.embeddings.class_embedding"), _w.F16("vision_model.embeddings.position_embedding.weight"), C);
        Probe?.Invoke("embeddings", _x.View, T, C);
        _sk.LayerNorm(_x.View, _n.View, _w.F16("vision_model.pre_layrnorm.weight"), _w.F16("vision_model.pre_layrnorm.bias"), T, C, Eps);
        _x.View.SubView(0, (long)T * C).CopyFrom(_n.View.SubView(0, (long)T * C));
        Probe?.Invoke("pre_layrnorm", _x.View, T, C);
        for (int l = 0; l < Layers; l++)
        {
            string b = $"vision_model.encoder.layers.{l}.";
            _sk.LayerNorm(_x.View, _n.View, _w.F16(b + "layer_norm1.weight"), _w.F16(b + "layer_norm1.bias"), T, C, Eps);
            foreach (var (proj, off) in new[] { ("q_proj", 0), ("k_proj", C), ("v_proj", 2 * C) })
            {
                Lin(_n.View, b + "self_attn." + proj, _part.View, T, C, C);
                _f.CopyCols(_part.View, _qkv.View, T, C, C, 3 * C, off);
            }
            _k.Attention(_qkv.View, 3 * C, 0, _qkv.View, 3 * C, C, 2 * C, _scores.View, _a.View, T, T, Heads, D);
            Lin(_a.View, b + "self_attn.out_proj", _h.View, T, C, C);
            _sd.AddInPlace(_x.View, _h.View, (long)T * C);
            _sk.LayerNorm(_x.View, _n.View, _w.F16(b + "layer_norm2.weight"), _w.F16(b + "layer_norm2.bias"), T, C, Eps);
            Lin(_n.View, b + "mlp.fc1", _hid.View, T, C, MlpWidth);
            _sd.GeluInPlace(_hid.View, (long)T * MlpWidth);
            Lin(_hid.View, b + "mlp.fc2", _h.View, T, MlpWidth, C);
            _sd.AddInPlace(_x.View, _h.View, (long)T * C);
            Probe?.Invoke($"layer{l + 1}", _x.View, T, C);
            if (l == Layers - 2) _pen.View.SubView(0, (long)T * C).CopyFrom(_x.View.SubView(0, (long)T * C));
        }
        _last.View.SubView(0, (long)T * C).CopyFrom(_x.View.SubView(0, (long)T * C));
        _sk.LayerNorm(_x.View.SubView(0, C), _pooled.View, _w.F16("vision_model.post_layernorm.weight"), _w.F16("vision_model.post_layernorm.bias"), 1, C, Eps);
        Probe?.Invoke("pooled", _pooled.View, 1, C);
        if (ProjectionWidth > 0) Lin(_pooled.View, "visual_projection", _embeds.View, 1, C, ProjectionWidth);
    }

    /// <summary>Frees the buffers and the weights.</summary>
    public void Dispose()
    {
        foreach (var o in _owned) o.Dispose();
        _owned.Clear();
        _w.Dispose();
    }
}
