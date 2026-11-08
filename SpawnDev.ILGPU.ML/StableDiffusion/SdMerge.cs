using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.StableDiffusion;

/// <summary>A merged tensor's weights: <c>W = W_self * Self + W_other * Other</c> (ComfyUI's add_patches: strength_model, strength_patch).</summary>
public readonly record struct SdMergeFactors(float Self, float Other);

/// <summary>What a merge did: tensors changed, and this model's tensors the other model lacks or holds with another shape (left as they were).</summary>
public sealed record SdMergeReport(int Merged, IReadOnlyList<string> NoCounterpart);

/// <summary>
/// The factor functions of ComfyUI's merge nodes, for <see cref="SdUNet.MergeAsync"/> and <see cref="SdTextEncoder.MergeAsync"/>. A factor function gets the tensor's
/// internal name and returns the factors, or null to leave the tensor alone. UNet names are the LDM names without <c>model.diffusion_model.</c>
/// (<c>input_blocks.4.1.transformer_blocks.0.attn1.qkv.weight</c>: the q | k | v and k | v projections are fused, under the same block prefix);
/// text encoder names are <c>l. / h. / g.</c> + <c>tok, pos, layers.N.{ln1, qkv, out, ln2, fc1, fc2}, final_ln, proj</c>.
/// </summary>
public static class SdMerge
{
    /// <summary>ModelMergeSimple / CLIPMergeSimple: <c>W = W1 * ratio + W2 * (1 - ratio)</c> (ratio 1 keeps this model).</summary>
    public static Func<string, SdMergeFactors?> Simple(float ratio) => _ => new SdMergeFactors(ratio, 1f - ratio);

    /// <summary>ModelAdd / CLIPAdd: <c>W = W1 + W2</c>.</summary>
    public static Func<string, SdMergeFactors?> Add() => _ => new SdMergeFactors(1f, 1f);

    /// <summary>ModelSubtract / CLIPSubtract: <c>W = (W1 - W2) * multiplier</c>.</summary>
    public static Func<string, SdMergeFactors?> Subtract(float multiplier) => _ => new SdMergeFactors(multiplier, -multiplier);

    /// <summary>
    /// ModelMergeBlocks / ModelMergeSD1 / ModelMergeSDXL: each tensor takes the ratio of the LONGEST prefix it starts with (<c>"input"</c>, <c>"middle"</c>, <c>"out"</c>
    /// for ModelMergeBlocks; <c>"time_embed."</c>, <c>"label_emb."</c>, <c>"input_blocks.4."</c>, <c>"middle_block.1."</c>, <c>"out."</c> ... for the per-block nodes),
    /// the FIRST entry's ratio when none matches, as ComfyUI does; then <c>W = W1 * ratio + W2 * (1 - ratio)</c>.
    /// </summary>
    public static Func<string, SdMergeFactors?> Blocks(IReadOnlyList<KeyValuePair<string, float>> prefixRatios)
    {
        if (prefixRatios.Count == 0) throw new ArgumentException("at least one prefix ratio", nameof(prefixRatios));
        var list = prefixRatios.ToArray();
        return name =>
        {
            float ratio = list[0].Value; int best = 0;
            foreach (var (prefix, r) in list)
                if (prefix.Length > best && name.StartsWith(prefix, StringComparison.Ordinal)) { ratio = r; best = prefix.Length; }
            return new SdMergeFactors(ratio, 1f - ratio);
        };
    }

    /// <summary>
    /// UNetSelfAttentionMultiply (<paramref name="attn"/> "attn1") / UNetCrossAttentionMultiply ("attn2"): the weight and bias of every <c>{attn}.to_q / to_k / to_v /
    /// to_out.0</c> times q / k / v / out, as ComfyUI's attention_multiply (key suffix match). For <see cref="SdUNet.ScaleAsync"/>.
    /// </summary>
    public static Func<string, float?> UNetAttentionMultiply(string attn, float q, float k, float v, float output) => name =>
    {
        foreach (var (proj, f) in new[] { ("to_q", q), ("to_k", k), ("to_v", v), ("to_out.0", output) })
            if (name.EndsWith($"{attn}.{proj}.weight", StringComparison.Ordinal) || name.EndsWith($"{attn}.{proj}.bias", StringComparison.Ordinal)) return f;
        return null;
    };

    /// <summary>
    /// UNetTemporalAttentionMultiply: <c>attn1.to_out.0</c> times self_temporal when the name contains <c>.time_stack.</c> (SVD), else self_structural; <c>attn2.to_out.0</c>
    /// likewise with the cross factors. For SD 1.x / SDXL only the structural factors apply (no time_stack blocks).
    /// </summary>
    public static Func<string, float?> UNetTemporalAttentionMultiply(float selfStructural, float selfTemporal, float crossStructural, float crossTemporal) => name =>
    {
        bool temporal = name.Contains(".time_stack.", StringComparison.Ordinal);
        if (name.EndsWith("attn1.to_out.0.bias", StringComparison.Ordinal) || name.EndsWith("attn1.to_out.0.weight", StringComparison.Ordinal))
            return temporal ? selfTemporal : selfStructural;
        if (name.EndsWith("attn2.to_out.0.bias", StringComparison.Ordinal) || name.EndsWith("attn2.to_out.0.weight", StringComparison.Ordinal))
            return temporal ? crossTemporal : crossStructural;
        return null;
    };

    /// <summary>
    /// CLIPAttentionMultiply: every text encoder layer's self-attention q / k / v / out projection (weight and bias) times q / k / v / out (ComfyUI matches
    /// <c>self_attn.{q,k,v,out}_proj</c>; here the normalised names <c>{l.|h.|g.}layers.N.{q,k,v,out}.{weight,bias}</c>). For <see cref="SdTextEncoder.ScaleAsync"/>.
    /// </summary>
    public static Func<string, float?> CLIPAttentionMultiply(float q, float k, float v, float output) => name =>
    {
        int at = name.IndexOf("layers.", StringComparison.Ordinal);
        if (at < 0) return null;
        int dot = name.IndexOf('.', at + 7);
        if (dot < 0) return null;
        return name[(dot + 1)..] switch
        {
            "q.weight" or "q.bias" => q,
            "k.weight" or "k.bias" => k,
            "v.weight" or "v.bias" => v,
            "out.weight" or "out.bias" => output,
            _ => null,
        };
    };

    /// <summary>
    /// Scale tensors in place by file-level name: <c>W = W * factor</c> (ComfyUI's <c>add_patches({key: (None,)}, 0.0, factor)</c>). A tensor fused into another
    /// (q | k | v) scales only its own rows of the fused tensor. Same ordering and restore rules as a merge (<see cref="SdPatchState"/>).
    /// </summary>
    internal static async Task<int> ScaleAsync(Accelerator acc, SdKernels k, SdPatchState state, Func<string, float?> factor, bool restorable, CancellationToken ct)
    {
        var w = state.Weights;
        int scaled = 0;
        foreach (var name in w.SourceNames.ToList())
        {
            ct.ThrowIfCancellationRequested();
            if (factor(name) is not { } f || f == 1f) continue;
            if (w.Locate(name) is not { } loc) continue;
            var view = w[loc.Name];
            var shape = w.Shape(loc.Name);
            long rowLen = shape.Length == 0 || shape[0] == 0 ? view.Length : view.Length / shape[0];
            state.BeforeMerge(loc.Name, restorable);
            k.ScaleHalf(view.SubView(loc.RowOffset * rowLen, loc.Rows * rowLen), f);
            scaled++;
        }
        await acc.SynchronizeAsync();
        state.AfterMerge();
        return scaled;
    }

    internal static async Task<SdMergeReport> RunAsync(Accelerator acc, SdKernels k, SdPatchState state, SdWeights other, Func<string, SdMergeFactors?> factors,
        bool restorable, CancellationToken ct)
    {
        var w = state.Weights;
        if (ReferenceEquals(w, other)) throw new ArgumentException("a model cannot be merged with its own weights: load the second model separately");
        int merged = 0;
        var none = new List<string>();
        foreach (var name in w.Names.ToList())
        {
            ct.ThrowIfCancellationRequested();
            if (factors(name) is not { } f) continue;
            if (!other.Has(name) || !w.Shape(name).AsSpan().SequenceEqual(other.Shape(name))) { none.Add(name); continue; }
            if (f.Self == 1f && f.Other == 0f) continue;
            state.BeforeMerge(name, restorable);
            k.Blend(w[name], other[name], f.Self, f.Other);
            merged++;
        }
        await acc.SynchronizeAsync();
        state.AfterMerge();
        return new SdMergeReport(merged, none);
    }
}
