using System.Text.RegularExpressions;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.Wan;

namespace SpawnDev.ILGPU.ML.Flux2;

/// <summary>
/// FLUX.2 transformer files in Black Forest Labs' own layout, the ones ComfyUI's templates load (Comfy-Org/flux2-klein <c>split_files/diffusion_models/flux-2-klein-4b.safetensors</c>,
/// bf16, and black-forest-labs/FLUX.2-klein-4b-fp8 <c>flux-2-klein-4b-fp8.safetensors</c>: F8_E4M3 matrices with one fp32 <c>weight_scale</c> each), rewritten into this library's
/// (diffusers) names at load. MEASURED against our export of black-forest-labs/FLUX.2-klein-4B (host, 2026-11-13, every tensor): the bf16 file is the same model, 148 of 149 tensors
/// bit-identical by name and <c>final_layer.adaLN_modulation.1</c> bit-identical to <c>norm_out.linear</c> with its two halves exchanged (shift / scale order); in the 8-bit file
/// value * weight_scale is within 2.7e-2 relative RMS of the bf16 weights (the format's own rounding) and its 69 bf16 tensors are bit-identical. The 8-bit matrices stay 8-bit
/// on the device (4.1 GB instead of 7.75) and are decoded one at a time into the bf16 scratch just before their GEMM (<see cref="Flux2Weights.Matrix"/>); the activation
/// <c>input_scale</c>s are not used (the GEMM runs on bf16 weights and fp32 activations).
/// </summary>
public static class Flux2ComfyFormat
{
    /// <summary>True when the transformer's tensors carry Black Forest Labs' names (double_blocks / single_blocks).</summary>
    public static bool IsBflTransformer(GpuWeights w) => w.Has("double_blocks.0.img_attn.qkv.weight");

    private static readonly (Regex From, string To)[] Map = new (string, string)[]
    {
        (@"^img_in\.", "x_embedder."), (@"^txt_in\.", "context_embedder."),
        (@"^time_in\.in_layer\.", "time_guidance_embed.timestep_embedder.linear_1."), (@"^time_in\.out_layer\.", "time_guidance_embed.timestep_embedder.linear_2."),
        (@"^(double_stream_modulation_img|double_stream_modulation_txt|single_stream_modulation)\.lin\.", "$1.linear."),
        (@"^final_layer\.linear\.", "proj_out."), (@"^final_layer\.adaLN_modulation\.1\.", "norm_out.linear."),
        (@"^double_blocks\.(\d+)\.img_attn\.qkv\.", "transformer_blocks.$1.attn.qkv."), (@"^double_blocks\.(\d+)\.txt_attn\.qkv\.", "transformer_blocks.$1.attn.add_qkv."),
        (@"^double_blocks\.(\d+)\.img_attn\.proj\.", "transformer_blocks.$1.attn.to_out."), (@"^double_blocks\.(\d+)\.txt_attn\.proj\.", "transformer_blocks.$1.attn.to_add_out."),
        (@"^double_blocks\.(\d+)\.img_attn\.norm\.query_norm\.scale$", "transformer_blocks.$1.attn.norm_q.weight"),
        (@"^double_blocks\.(\d+)\.img_attn\.norm\.key_norm\.scale$", "transformer_blocks.$1.attn.norm_k.weight"),
        (@"^double_blocks\.(\d+)\.txt_attn\.norm\.query_norm\.scale$", "transformer_blocks.$1.attn.norm_added_q.weight"),
        (@"^double_blocks\.(\d+)\.txt_attn\.norm\.key_norm\.scale$", "transformer_blocks.$1.attn.norm_added_k.weight"),
        (@"^double_blocks\.(\d+)\.img_mlp\.0\.", "transformer_blocks.$1.ff.linear_in."), (@"^double_blocks\.(\d+)\.img_mlp\.2\.", "transformer_blocks.$1.ff.linear_out."),
        (@"^double_blocks\.(\d+)\.txt_mlp\.0\.", "transformer_blocks.$1.ff_context.linear_in."), (@"^double_blocks\.(\d+)\.txt_mlp\.2\.", "transformer_blocks.$1.ff_context.linear_out."),
        (@"^single_blocks\.(\d+)\.linear1\.", "single_transformer_blocks.$1.attn.to_qkv_mlp_proj."), (@"^single_blocks\.(\d+)\.linear2\.", "single_transformer_blocks.$1.attn.to_out."),
        (@"^single_blocks\.(\d+)\.norm\.query_norm\.scale$", "single_transformer_blocks.$1.attn.norm_q.weight"),
        (@"^single_blocks\.(\d+)\.norm\.key_norm\.scale$", "single_transformer_blocks.$1.attn.norm_k.weight"),
    }.Select(t => (new Regex(t.Item1, RegexOptions.Compiled | RegexOptions.CultureInvariant), t.Item2)).ToArray();

    /// <summary>This library's name for a Black Forest Labs transformer tensor, or null for one it does not use.</summary>
    public static string? OurTransformerName(string name)
    {
        foreach (var (from, to) in Map) if (from.IsMatch(name)) return from.Replace(name, to);
        return null;
    }

    // ---- the text encoder: ComfyUI's qwen_3_4b.safetensors (Comfy-Org/flux2-klein split_files/text_encoders), Hugging Face Qwen3-4B names, bf16, all 36 layers ----

    /// <summary>
    /// The load filter for a FLUX.2 stage file: in a Hugging Face Qwen3 file (<c>model.*</c> names) only the embedding and the first <see cref="Flux2TextEncoder.Layers"/> layers are
    /// read (the pipeline reads the hidden states after layers 9, 18 and 27: the last 9 layers, the final norm and any head never run). Every other name passes.
    /// </summary>
    public static bool TextEncoderFilter(string name)
    {
        if (!name.StartsWith("model.", StringComparison.Ordinal) && !name.StartsWith("lm_head.", StringComparison.Ordinal)) return true;
        if (name == "model.embed_tokens.weight") return true;
        if (!name.StartsWith("model.layers.", StringComparison.Ordinal)) return false;                   // model.norm, lm_head
        int end = name.IndexOf('.', "model.layers.".Length);
        return end > 0 && int.TryParse(name.AsSpan("model.layers.".Length, end - "model.layers.".Length), out int l) && l < Flux2TextEncoder.Layers;
    }

    /// <summary>True when the text encoder's tensors carry Hugging Face Qwen3 names.</summary>
    public static bool IsHfQwen3(GpuWeights w) => w.Has("model.embed_tokens.weight");

    /// <summary>
    /// Rewrite a Hugging Face Qwen3 text encoder into this library's layout in place: <c>model.</c> dropped, q | k | v fused into <c>self_attn.qkv_proj</c> and gate | up into
    /// <c>mlp.gate_up_proj</c> (rows concatenated in that order, as our export does: export_flux2.py), the rest renamed. Each fused group is built and its sources freed before the next,
    /// so the peak is one group above the file.
    /// </summary>
    public static async Task AdaptTextEncoderAsync(Accelerator accelerator, GpuWeights w)
    {
        async Task Retire(IDisposable? old) { if (old == null) return; await accelerator.SynchronizeAsync(); old.Dispose(); }
        async Task Fuse(string ours, params string[] sources)
        {
            foreach (var s in sources) if (!w.IsBF16(s)) throw new NotSupportedException($"{s}: expected bf16 (the 8-bit Qwen files are not supported yet)");
            long total = 0; foreach (var s in sources) total += (long)w.Shape(s)[0] * w.Shape(s)[1];
            var dst = accelerator.Allocate1D<BFloat16>(total);
            long off = 0; int rows = 0;
            foreach (var s in sources) { long c = (long)w.Shape(s)[0] * w.Shape(s)[1]; dst.View.SubView(off, c).CopyFrom(w.BF16(s)); off += c; rows += w.Shape(s)[0]; }
            int cols = w.Shape(sources[0])[1];
            foreach (var s in sources) await Retire(w.Remove(s));
            await Retire(w.SetBF16(ours, dst, new[] { rows, cols }));
        }
        for (int l = 0; l < Flux2TextEncoder.Layers; l++)
        {
            string p = $"model.layers.{l}.", o = $"layers.{l}.";
            if (!w.Has(p + "self_attn.q_proj.weight")) throw new InvalidDataException($"{p}: missing (the encoder needs layers 0 to {Flux2TextEncoder.Layers - 1})");
            await Fuse(o + "self_attn.qkv_proj.weight", p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight");
            await Fuse(o + "mlp.gate_up_proj.weight", p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight");
        }
        foreach (var name in w.Names.Where(n => n.StartsWith("model.", StringComparison.Ordinal)).ToList()) w.Rename(name, name["model.".Length..]);
        await accelerator.SynchronizeAsync();
    }

    private const string WeightScale = ".weight_scale", InputScale = ".input_scale";

    /// <summary>
    /// Rewrite a Black Forest Labs transformer into this library's names in place: tensors renamed (no copy), <c>final_layer.adaLN_modulation.1</c> stored with its halves exchanged,
    /// the scales removed. Returns the decode scale of every 8-bit matrix, by our name (empty for the bf16 file), for <see cref="Flux2Weights"/>.
    /// </summary>
    public static async Task<Dictionary<string, float>> AdaptTransformerAsync(Accelerator accelerator, GpuWeights w)
    {
        async Task Retire(IDisposable? old) { if (old == null) return; await accelerator.SynchronizeAsync(); old.Dispose(); }
        var scales = new Dictionary<string, float>();
        foreach (var name in w.Names.ToList())
        {
            if (name.EndsWith(InputScale, StringComparison.Ordinal)) { await Retire(w.Remove(name)); continue; }
            if (name.EndsWith(WeightScale, StringComparison.Ordinal))
            {
                string matrix = name[..^WeightScale.Length] + ".weight";
                var ours = OurTransformerName(matrix) ?? throw new InvalidDataException($"{name}: the scale of a tensor this library does not know");
                if (!w.IsF32(name)) throw new InvalidDataException($"{name}: expected an fp32 scalar");
                scales[ours] = (await w.F32(name).CopyToHostAsync())[0];
                await Retire(w.Remove(name));
                continue;
            }
            var to = OurTransformerName(name) ?? throw new InvalidDataException($"{name}: not a FLUX.2 transformer tensor this library knows");
            w.Rename(name, to);
        }
        foreach (var m in w.Names.Where(w.IsF8))
            if (!scales.ContainsKey(m)) throw new InvalidDataException($"{m}: an F8_E4M3 matrix without its weight_scale");
        // shift / scale: Black Forest Labs' order is the reverse of the diffusers order this library runs
        const string Final = "norm_out.linear.weight";
        var shape = w.Shape(Final);
        long count = (long)shape[0] * shape[1], half = count / 2;
        var swapped = accelerator.Allocate1D<BFloat16>(count);
        if (w.IsF8(Final))
        {
            using var dec = accelerator.Allocate1D<BFloat16>(count);
            WanComfyFormat.DecodeF8E4M3(accelerator, w.F8Words(Final), dec.View, scales[Final]);
            swapped.View.SubView(0, half).CopyFrom(dec.View.SubView(half, half)); swapped.View.SubView(half, half).CopyFrom(dec.View.SubView(0, half));
            await accelerator.SynchronizeAsync();
            scales.Remove(Final);
        }
        else
        {
            var src = w.BF16(Final);
            swapped.View.SubView(0, half).CopyFrom(src.SubView(half, half)); swapped.View.SubView(half, half).CopyFrom(src.SubView(0, half));
        }
        await Retire(w.SetBF16(Final, swapped, shape));
        await accelerator.SynchronizeAsync();
        return scales;
    }
}
