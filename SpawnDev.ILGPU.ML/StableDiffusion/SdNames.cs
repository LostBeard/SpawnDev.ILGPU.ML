using System.Text.RegularExpressions;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.StableDiffusion;

/// <summary>
/// Tensor-name maps between the layouts Stable Diffusion files come in and the loader's internal names (the original LDM names without a component prefix; text
/// encoders normalised to <c>tok / pos / layers.N.{ln1,q,k,v,qkv,out,ln2,fc1,fc2} / final_ln / proj</c> under a tower prefix <c>l.</c>, <c>h.</c> or <c>g.</c>).
/// Layouts: LDM single files (ComfyUI / CivitAI), diffusers (<c>down_blocks.*</c>, <c>encoder.down_blocks.*</c>), Hugging Face transformers CLIP (<c>text_model.*</c>),
/// OpenCLIP (<c>transformer.resblocks.*</c>).
/// </summary>
internal static class SdNames
{
    // ---------------------------------------------------------------------------------------------------
    // text encoders
    // ---------------------------------------------------------------------------------------------------
    private static readonly Regex HfLayer = new(@"^encoder\.layers\.(\d+)\.(.+)$", RegexOptions.Compiled);
    private static readonly Regex OcLayer = new(@"^transformer\.resblocks\.(\d+)\.(.+)$", RegexOptions.Compiled);

    /// <summary>Hugging Face CLIPTextModel names (after <c>text_model.</c>) -> the normalised names, or null for tensors the encoder does not read.</summary>
    public static string? MapHfClip(string n, string p)
    {
        switch (n)
        {
            case "embeddings.token_embedding.weight": return p + "tok";
            case "embeddings.position_embedding.weight": return p + "pos";
            case "final_layer_norm.weight": return p + "final_ln.weight";
            case "final_layer_norm.bias": return p + "final_ln.bias";
        }
        var m = HfLayer.Match(n);
        if (!m.Success) return null;
        string l = $"{p}layers.{m.Groups[1].Value}.", rest = m.Groups[2].Value;
        string? r = rest switch
        {
            "self_attn.q_proj.weight" => "q.weight", "self_attn.q_proj.bias" => "q.bias",
            "self_attn.k_proj.weight" => "k.weight", "self_attn.k_proj.bias" => "k.bias",
            "self_attn.v_proj.weight" => "v.weight", "self_attn.v_proj.bias" => "v.bias",
            "self_attn.out_proj.weight" => "out.weight", "self_attn.out_proj.bias" => "out.bias",
            "layer_norm1.weight" => "ln1.weight", "layer_norm1.bias" => "ln1.bias",
            "layer_norm2.weight" => "ln2.weight", "layer_norm2.bias" => "ln2.bias",
            "mlp.fc1.weight" => "fc1.weight", "mlp.fc1.bias" => "fc1.bias",
            "mlp.fc2.weight" => "fc2.weight", "mlp.fc2.bias" => "fc2.bias",
            _ => null,
        };
        return r == null ? null : l + r;
    }

    /// <summary>OpenCLIP text names (after <c>model.</c>) -> the normalised names, or null. The text projection is applied as x @ P, so it is stored transposed ([out, in]).</summary>
    public static string? MapOpenClip(string n, string p, out SdLayout layout)
    {
        layout = SdLayout.AsIs;
        switch (n)
        {
            case "token_embedding.weight": return p + "tok";
            case "positional_embedding": return p + "pos";
            case "ln_final.weight": return p + "final_ln.weight";
            case "ln_final.bias": return p + "final_ln.bias";
            case "text_projection": layout = SdLayout.Transpose; return p + "proj.weight";
        }
        var m = OcLayer.Match(n);
        if (!m.Success) return null;
        string l = $"{p}layers.{m.Groups[1].Value}.", rest = m.Groups[2].Value;
        string? r = rest switch
        {
            "ln_1.weight" => "ln1.weight", "ln_1.bias" => "ln1.bias",
            "ln_2.weight" => "ln2.weight", "ln_2.bias" => "ln2.bias",
            "attn.in_proj_weight" => "qkv.weight", "attn.in_proj_bias" => "qkv.bias",
            "attn.out_proj.weight" => "out.weight", "attn.out_proj.bias" => "out.bias",
            "mlp.c_fc.weight" => "fc1.weight", "mlp.c_fc.bias" => "fc1.bias",
            "mlp.c_proj.weight" => "fc2.weight", "mlp.c_proj.bias" => "fc2.bias",
            _ => null,
        };
        return r == null ? null : l + r;
    }

    /// <summary>
    /// A standalone text-encoder file's tensor (ComfyUI's CLIPLoader / DualCLIPLoader: <c>clip_l.safetensors</c>, <c>clip_g.safetensors</c>, a diffusers <c>text_encoder</c>):
    /// HF names anywhere after a prefix (<c>text_model.*</c>, plus the top-level <c>text_projection.weight</c> of CLIPTextModelWithProjection, already [out, in]) or OpenCLIP
    /// names. Returns the normalised name under <paramref name="p"/> or null.
    /// </summary>
    public static string? MapClipFile(string n, string p, out SdLayout layout)
    {
        layout = SdLayout.AsIs;
        int i = n.IndexOf("text_model.", StringComparison.Ordinal);
        if (i >= 0) return MapHfClip(n[(i + "text_model.".Length)..], p);
        if (n == "text_projection.weight" || n.EndsWith(".text_projection.weight", StringComparison.Ordinal)) return p + "proj.weight";
        foreach (var pre in new[] { "cond_stage_model.model.", "conditioner.embedders.1.model.", "conditioner.embedders.0.model.", "model.", "" })
            if (n.StartsWith(pre, StringComparison.Ordinal))
            {
                var r = MapOpenClip(n[pre.Length..], p, out layout);
                if (r != null) return r;
            }
        return null;
    }

    // ---------------------------------------------------------------------------------------------------
    // UNet
    // ---------------------------------------------------------------------------------------------------
    private static readonly Regex InBlock = new(@"^input_blocks\.(\d+)\.(\d+)\.(.+)$", RegexOptions.Compiled);
    private static readonly Regex OutBlock = new(@"^output_blocks\.(\d+)\.(\d+)\.(.+)$", RegexOptions.Compiled);
    private static readonly Regex MidBlock = new(@"^middle_block\.(\d)\.(.+)$", RegexOptions.Compiled);
    private static readonly (string Ldm, string Diff)[] ResnetParts = { ("in_layers.0", "norm1"), ("in_layers.2", "conv1"), ("emb_layers.1", "time_emb_proj"), ("out_layers.0", "norm2"), ("out_layers.3", "conv2"), ("skip_connection", "conv_shortcut") };

    private static string Swap(string rest, bool toDiffusers)
    {
        foreach (var (a, b) in ResnetParts)
        {
            var (from, to) = toDiffusers ? (a, b) : (b, a);
            if (rest == from || rest.StartsWith(from + ".", StringComparison.Ordinal)) return to + rest[from.Length..];
        }
        return rest;
    }

    private static readonly (string Ldm, string Diff)[] UNetTop = { ("time_embed.0", "time_embedding.linear_1"), ("time_embed.2", "time_embedding.linear_2"),
        ("label_emb.0.0", "add_embedding.linear_1"), ("label_emb.0.2", "add_embedding.linear_2"), ("out.0", "conv_norm_out"), ("out.2", "conv_out"), ("input_blocks.0.0", "conv_in") };

    /// <summary>
    /// The diffusers UNet2DConditionModel module path of an LDM UNet module path (SD 1.x / 2.x / SDXL: two resnets per level, so input block k is level (k-1)/3, slot (k-1)%3),
    /// or null for a name with no counterpart.
    /// </summary>
    public static string? LdmToDiffusersUNet(string m)
    {
        foreach (var (l, d) in UNetTop) if (m == l) return d;
        var g = InBlock.Match(m);
        if (g.Success)
        {
            int k = int.Parse(g.Groups[1].Value), sub = int.Parse(g.Groups[2].Value);
            string rest = g.Groups[3].Value;
            if (k == 0) return null;
            int i = (k - 1) / 3, j = (k - 1) % 3;
            if (j == 2) return rest.StartsWith("op", StringComparison.Ordinal) ? $"down_blocks.{i}.downsamplers.0.conv{rest[2..]}" : null;
            return sub == 0 ? $"down_blocks.{i}.resnets.{j}.{Swap(rest, true)}" : $"down_blocks.{i}.attentions.{j}.{rest}";
        }
        g = MidBlock.Match(m);
        if (g.Success)
        {
            int sub = int.Parse(g.Groups[1].Value);
            string rest = g.Groups[2].Value;
            return sub switch { 0 => $"mid_block.resnets.0.{Swap(rest, true)}", 1 => $"mid_block.attentions.0.{rest}", 2 => $"mid_block.resnets.1.{Swap(rest, true)}", _ => null };
        }
        g = OutBlock.Match(m);
        if (g.Success)
        {
            int k = int.Parse(g.Groups[1].Value), sub = int.Parse(g.Groups[2].Value);
            string rest = g.Groups[3].Value;
            int i = k / 3, j = k % 3;
            if (sub == 0) return $"up_blocks.{i}.resnets.{j}.{Swap(rest, true)}";
            if (rest == "conv" || rest.StartsWith("conv.", StringComparison.Ordinal)) return $"up_blocks.{i}.upsamplers.0.{rest}";
            return $"up_blocks.{i}.attentions.{j}.{rest}";
        }
        return null;
    }

    private static readonly Regex DDown = new(@"^down_blocks\.(\d+)\.(resnets|attentions|downsamplers)\.(\d+)\.(.+)$", RegexOptions.Compiled);
    private static readonly Regex DUp = new(@"^up_blocks\.(\d+)\.(resnets|attentions|upsamplers)\.(\d+)\.(.+)$", RegexOptions.Compiled);
    private static readonly Regex DMid = new(@"^mid_block\.(resnets|attentions)\.(\d+)\.(.+)$", RegexOptions.Compiled);

    /// <summary>
    /// The LDM name of a diffusers UNet tensor name (the inverse of <see cref="LdmToDiffusersUNet"/>). <paramref name="upHasAttention"/> says whether up block i has
    /// attention modules (its upsampler is then module 2 of output block 3i+2, else module 1). Null for a name with no counterpart.
    /// </summary>
    public static string? DiffusersToLdmUNet(string n, Func<int, bool> upHasAttention)
    {
        foreach (var (l, d) in UNetTop) if (n.StartsWith(d + ".", StringComparison.Ordinal)) return l + n[d.Length..];
        var g = DDown.Match(n);
        if (g.Success)
        {
            int i = int.Parse(g.Groups[1].Value), j = int.Parse(g.Groups[3].Value);
            string kind = g.Groups[2].Value, rest = g.Groups[4].Value;
            return kind switch
            {
                "resnets" => $"input_blocks.{3 * i + j + 1}.0.{Swap(rest, false)}",
                "attentions" => $"input_blocks.{3 * i + j + 1}.1.{rest}",
                _ => rest.StartsWith("conv", StringComparison.Ordinal) ? $"input_blocks.{3 * (i + 1)}.0.op{rest[4..]}" : null,
            };
        }
        g = DMid.Match(n);
        if (g.Success)
        {
            int j = int.Parse(g.Groups[2].Value);
            string rest = g.Groups[3].Value;
            return g.Groups[1].Value == "attentions" ? $"middle_block.1.{rest}" : $"middle_block.{(j == 0 ? 0 : 2)}.{Swap(rest, false)}";
        }
        g = DUp.Match(n);
        if (g.Success)
        {
            int i = int.Parse(g.Groups[1].Value), j = int.Parse(g.Groups[3].Value);
            string kind = g.Groups[2].Value, rest = g.Groups[4].Value;
            return kind switch
            {
                "resnets" => $"output_blocks.{3 * i + j}.0.{Swap(rest, false)}",
                "attentions" => $"output_blocks.{3 * i + j}.1.{rest}",
                _ => $"output_blocks.{3 * i + 2}.{(upHasAttention(i) ? 2 : 1)}.{rest}",
            };
        }
        return null;
    }

    // ---------------------------------------------------------------------------------------------------
    // VAE
    // ---------------------------------------------------------------------------------------------------
    private static readonly Regex VEncDown = new(@"^encoder\.down_blocks\.(\d+)\.(resnets|downsamplers)\.(\d+)\.(.+)$", RegexOptions.Compiled);
    private static readonly Regex VDecUp = new(@"^decoder\.up_blocks\.(\d+)\.(resnets|upsamplers)\.(\d+)\.(.+)$", RegexOptions.Compiled);
    private static readonly Regex VMid = new(@"^(encoder|decoder)\.mid_block\.(resnets|attentions)\.(\d+)\.(.+)$", RegexOptions.Compiled);

    private static string VaeResnet(string rest) => rest.StartsWith("conv_shortcut", StringComparison.Ordinal) ? "nin_shortcut" + rest["conv_shortcut".Length..] : rest;

    private static string? VaeAttention(string rest) => rest switch
    {
        _ when rest.StartsWith("group_norm.", StringComparison.Ordinal) => "norm." + rest["group_norm.".Length..],
        _ when rest.StartsWith("to_q.", StringComparison.Ordinal) => "q." + rest[5..],
        _ when rest.StartsWith("to_k.", StringComparison.Ordinal) => "k." + rest[5..],
        _ when rest.StartsWith("to_v.", StringComparison.Ordinal) => "v." + rest[5..],
        _ when rest.StartsWith("to_out.0.", StringComparison.Ordinal) => "proj_out." + rest[9..],
        _ when rest.StartsWith("query.", StringComparison.Ordinal) => "q." + rest[6..],        // the older diffusers attention names
        _ when rest.StartsWith("key.", StringComparison.Ordinal) => "k." + rest[4..],
        _ when rest.StartsWith("value.", StringComparison.Ordinal) => "v." + rest[6..],
        _ when rest.StartsWith("proj_attn.", StringComparison.Ordinal) => "proj_out." + rest[10..],
        _ => null,
    };

    /// <summary>The LDM name of a diffusers AutoencoderKL tensor name (<paramref name="levels"/> = the resolution levels, 4), or the name itself when it already is one.</summary>
    public static string? DiffusersToLdmVae(string n, int levels)
    {
        if (n.StartsWith("encoder.conv_norm_out.", StringComparison.Ordinal)) return "encoder.norm_out." + n["encoder.conv_norm_out.".Length..];
        if (n.StartsWith("decoder.conv_norm_out.", StringComparison.Ordinal)) return "decoder.norm_out." + n["decoder.conv_norm_out.".Length..];
        var g = VEncDown.Match(n);
        if (g.Success)
        {
            int i = int.Parse(g.Groups[1].Value), j = int.Parse(g.Groups[3].Value);
            string rest = g.Groups[4].Value;
            return g.Groups[2].Value == "resnets" ? $"encoder.down.{i}.block.{j}.{VaeResnet(rest)}" : $"encoder.down.{i}.downsample.{rest}";
        }
        g = VDecUp.Match(n);
        if (g.Success)
        {
            int i = levels - 1 - int.Parse(g.Groups[1].Value), j = int.Parse(g.Groups[3].Value);
            string rest = g.Groups[4].Value;
            return g.Groups[2].Value == "resnets" ? $"decoder.up.{i}.block.{j}.{VaeResnet(rest)}" : $"decoder.up.{i}.upsample.{rest}";
        }
        g = VMid.Match(n);
        if (g.Success)
        {
            string side = g.Groups[1].Value, rest = g.Groups[4].Value;
            int j = int.Parse(g.Groups[3].Value);
            if (g.Groups[2].Value == "attentions") { var a = VaeAttention(rest); return a == null ? null : $"{side}.mid.attn_1.{a}"; }
            return $"{side}.mid.block_{j + 1}.{VaeResnet(rest)}";
        }
        return n;
    }

    // ---------------------------------------------------------------------------------------------------
    // fusion (one GEMM for q|k|v and k|v), shared by the checkpoint and the component loaders
    // ---------------------------------------------------------------------------------------------------
    public static async Task FuseUNetAsync(Accelerator acc, SdWeights unet)
    {
        foreach (var n in unet.Names.Where(n => n.EndsWith(".attn1.to_q.weight", StringComparison.Ordinal)).ToList())
        {
            string b = n[..^"to_q.weight".Length];
            await unet.FuseAsync(acc, b + "qkv.weight", b + "to_q.weight", b + "to_k.weight", b + "to_v.weight");
            string b2 = b[..^"attn1.".Length] + "attn2.";
            await unet.FuseAsync(acc, b2 + "kv.weight", b2 + "to_k.weight", b2 + "to_v.weight");
        }
    }

    public static async Task FuseVaeAsync(Accelerator acc, SdWeights vae)
    {
        foreach (var part in new[] { "encoder.mid.attn_1.", "decoder.mid.attn_1." })
        {
            await vae.FuseAsync(acc, part + "qkv.weight", part + "q.weight", part + "k.weight", part + "v.weight");
            await vae.FuseAsync(acc, part + "qkv.bias", part + "q.bias", part + "k.bias", part + "v.bias");
        }
    }

    public static async Task FuseTextAsync(Accelerator acc, SdWeights text)
    {
        foreach (var n in text.Names.Where(n => n.EndsWith(".q.weight", StringComparison.Ordinal)).ToList())
        {
            string b = n[..^"q.weight".Length];
            await text.FuseAsync(acc, b + "qkv.weight", b + "q.weight", b + "k.weight", b + "v.weight");
            await text.FuseAsync(acc, b + "qkv.bias", b + "q.bias", b + "k.bias", b + "v.bias");
        }
    }
}
