using System.Text.RegularExpressions;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.SafeTensors;

namespace SpawnDev.ILGPU.ML.StableDiffusion;

/// <summary>The model family of a checkpoint (read from its tensors).</summary>
public enum SdFamily { SD1, SD2, SDXL }

/// <summary>What the UNet predicts.</summary>
public enum SdPrediction { Epsilon, V }

/// <summary>Which parts of a checkpoint to load (and the overrides of what the loader infers).</summary>
public sealed class SdCheckpointOptions
{
    public bool LoadUNet { get; set; } = true;
    public bool LoadTextEncoder { get; set; } = true;
    public bool LoadVae { get; set; } = true;
    /// <summary>The prediction type; null = inferred as ComfyUI does (a <c>v_pred</c> tensor, or for SD 2.x the spread of one late norm bias).</summary>
    public SdPrediction? Prediction { get; set; }
}

/// <summary>
/// A Stable Diffusion checkpoint in the single-file format ComfyUI and CivitAI use (one .safetensors with the UNet, the text encoder(s) and the VAE under their original LDM
/// names: <c>model.diffusion_model.*</c>, <c>cond_stage_model.*</c> / <c>conditioner.embedders.*</c>, <c>first_stage_model.*</c>): SD 1.x, SD 2.x and SDXL base. One call
/// streams the file once, front to back (browser OPFS and HTTP streams work), converts every tensor to fp16 on the device and builds <see cref="UNet"/>, <see cref="TextEncoder"/>
/// and <see cref="Vae"/>. The caller keeps the sampler, the noise, CFG and the latents; <see cref="LatentScale"/> and <see cref="Prediction"/> say how to drive the models.
/// <see cref="Dispose"/> frees everything.
/// </summary>
public sealed class SdCheckpoint : IDisposable
{
    public SdFamily Family { get; }
    public SdPrediction Prediction { get; }
    /// <summary>The VAE latent scale: sampler latent = VAE latent x this (0.18215 SD 1.x / 2.x, 0.13025 SDXL).</summary>
    public float LatentScale => Family == SdFamily.SDXL ? 0.13025f : 0.18215f;
    public SdUNet? UNet { get; }
    public SdTextEncoder? TextEncoder { get; }
    public SdVae? Vae { get; }
    /// <summary>Device bytes of the loaded weights.</summary>
    public long WeightBytes => (_unet?.Bytes ?? 0) + (_text?.Bytes ?? 0) + (_vae?.Bytes ?? 0);

    private readonly SdWeights? _unet, _text, _vae;

    private SdCheckpoint(SdFamily family, SdPrediction prediction, SdWeights? unet, SdWeights? text, SdWeights? vae, Accelerator acc)
    {
        Family = family; Prediction = prediction; _unet = unet; _text = text; _vae = vae;
        if (unet != null) UNet = new SdUNet(acc, unet);
        if (text != null) TextEncoder = new SdTextEncoder(acc, text, family);
        if (vae != null) Vae = new SdVae(acc, vae);
    }

    /// <summary>Load <paramref name="file"/> of <paramref name="repo"/> from a model source (hub, models host, OPFS, a local folder).</summary>
    public static async Task<SdCheckpoint> LoadAsync(Accelerator accelerator, IModelSource source, string repo, string file, SdCheckpointOptions? options = null,
        Action<string, double>? onProgress = null, CancellationToken cancellationToken = default)
    {
        await using var stream = await source.OpenAsync(repo, file, cancellationToken);
        return await LoadAsync(accelerator, stream, $"{repo}/{file}", options, onProgress, cancellationToken);
    }

    /// <summary>Load a checkpoint from a stream positioned at its start (read once, front to back; seeking is used only to skip when the stream can).</summary>
    public static async Task<SdCheckpoint> LoadAsync(Accelerator accelerator, Stream stream, string name, SdCheckpointOptions? options = null,
        Action<string, double>? onProgress = null, CancellationToken cancellationToken = default)
    {
        options ??= new SdCheckpointOptions();
        var lenBytes = new byte[8];
        await stream.ReadExactlyAsync(lenBytes, cancellationToken);
        long headerSize = BitConverter.ToInt64(lenBytes);
        if (headerSize <= 0 || headerSize > 100_000_000) throw new InvalidDataException($"'{name}': not a safetensors file (header size {headerSize})");
        var headerBytes = new byte[headerSize];
        await stream.ReadExactlyAsync(headerBytes, cancellationToken);
        long pos = 8 + headerSize, length = -1;
        try { if (stream.CanSeek) length = stream.Length - pos; } catch (NotSupportedException) { }
        var (tensors, _) = SafeTensorsFileReader.ParseHeader(name, headerBytes, pos, length);
        var names = new HashSet<string>(tensors.Select(t => t.Name));
        var family = DetectFamily(name, names, tensors);
        long total = tensors.Count == 0 ? 1 : tensors.Max(t => t.DataOffset + t.DataLength);

        var unet = options.LoadUNet ? new SdWeights() : null;
        var text = options.LoadTextEncoder && HasTextEncoder(names, family) ? new SdWeights() : null;
        var vae = options.LoadVae && names.Any(n => n.StartsWith("first_stage_model.", StringComparison.Ordinal)) ? new SdWeights() : null;
        try
        {
            SdTensorStreamer.Target? Route(string n)
            {
                if (n.StartsWith("model.diffusion_model.", StringComparison.Ordinal))
                    return unet == null ? null : new SdTensorStreamer.Target(unet, n["model.diffusion_model.".Length..], SdLayout.ConvGather);
                if (n.StartsWith("first_stage_model.", StringComparison.Ordinal))
                    return vae == null ? null : new SdTensorStreamer.Target(vae, n["first_stage_model.".Length..], SdLayout.ConvGather);
                if (text == null) return null;
                string? local = null; SdLayout layout = SdLayout.AsIs;
                if (family == SdFamily.SD1 && n.StartsWith("cond_stage_model.transformer.text_model.", StringComparison.Ordinal))
                    local = MapHfClip(n["cond_stage_model.transformer.text_model.".Length..], "l.");
                else if (family == SdFamily.SD2 && n.StartsWith("cond_stage_model.model.", StringComparison.Ordinal))
                    local = MapOpenClip(n["cond_stage_model.model.".Length..], "h.", out layout);
                else if (family == SdFamily.SDXL && n.StartsWith("conditioner.embedders.0.transformer.text_model.", StringComparison.Ordinal))
                    local = MapHfClip(n["conditioner.embedders.0.transformer.text_model.".Length..], "l.");
                else if (family == SdFamily.SDXL && n.StartsWith("conditioner.embedders.1.model.", StringComparison.Ordinal))
                    local = MapOpenClip(n["conditioner.embedders.1.model.".Length..], "g.", out layout);
                return local == null ? null : new SdTensorStreamer.Target(text, local, layout);
            }
            using (var streamer = new SdTensorStreamer(accelerator))
                await streamer.StreamAsync(stream, tensors, pos, Route, p => onProgress?.Invoke("weights", (double)p / total), cancellationToken);
            onProgress?.Invoke("weights", 1.0);

            // fuse the projections that run as one GEMM: q | k | v of every self-attention, k | v of every cross-attention
            if (unet != null)
            {
                foreach (var n in unet.Names.Where(n => n.EndsWith(".attn1.to_q.weight", StringComparison.Ordinal)).ToList())
                {
                    string b = n[..^"to_q.weight".Length];
                    await unet.FuseAsync(accelerator, b + "qkv.weight", b + "to_q.weight", b + "to_k.weight", b + "to_v.weight");
                    string b2 = b[..^"attn1.".Length] + "attn2.";
                    await unet.FuseAsync(accelerator, b2 + "kv.weight", b2 + "to_k.weight", b2 + "to_v.weight");
                }
                if (!unet.Has("input_blocks.0.0.weight")) throw new InvalidDataException($"'{name}': no UNet (model.diffusion_model.*) in this file");
            }
            if (vae != null)
                foreach (var part in new[] { "encoder.mid.attn_1.", "decoder.mid.attn_1." })
                {
                    await vae.FuseAsync(accelerator, part + "qkv.weight", part + "q.weight", part + "k.weight", part + "v.weight");
                    await vae.FuseAsync(accelerator, part + "qkv.bias", part + "q.bias", part + "k.bias", part + "v.bias");
                }
            if (text != null)
                foreach (var n in text.Names.Where(n => n.EndsWith(".q.weight", StringComparison.Ordinal)).ToList())
                {
                    string b = n[..^"q.weight".Length];
                    await text.FuseAsync(accelerator, b + "qkv.weight", b + "q.weight", b + "k.weight", b + "v.weight");
                    await text.FuseAsync(accelerator, b + "qkv.bias", b + "q.bias", b + "k.bias", b + "v.bias");
                }

            var prediction = options.Prediction ?? (names.Contains("v_pred") ? SdPrediction.V : SdPrediction.Epsilon);
            if (options.Prediction == null && family == SdFamily.SD2 && unet != null)
                prediction = await Sd2PredictionAsync(accelerator, unet);
            return new SdCheckpoint(family, prediction, unet, text, vae, accelerator);
        }
        catch
        {
            unet?.Dispose(); text?.Dispose(); vae?.Dispose();
            throw;
        }
    }

    /// <summary>ComfyUI's test for an SD 2.x v-prediction model (the 768-v checkpoints): the spread of one late layer-norm bias, above 0.09 for the v models.</summary>
    private static async Task<SdPrediction> Sd2PredictionAsync(Accelerator acc, SdWeights unet)
    {
        const string key = "output_blocks.11.1.transformer_blocks.0.norm1.bias";
        if (!unet.Has(key)) return SdPrediction.Epsilon;
        var hv = unet[key];
        using var f = acc.Allocate1D<float>(hv.Length);
        new SdKernels(acc).HalfToFloat(hv, f.View);
        await acc.SynchronizeAsync();
        var v = await f.View.CopyToHostAsync();
        double mean = v.Average(x => (double)x), var = v.Average(x => (x - mean) * (x - mean));
        return Math.Sqrt(var) > 0.09 ? SdPrediction.V : SdPrediction.Epsilon;
    }

    private static SdFamily DetectFamily(string name, HashSet<string> names, List<SafeTensorInfo> tensors)
    {
        if (names.Any(n => n.StartsWith("conditioner.embedders.1.model.", StringComparison.Ordinal))) return SdFamily.SDXL;
        if (names.Any(n => n.StartsWith("conditioner.embedders.0.model.", StringComparison.Ordinal)))
            throw new NotSupportedException($"'{name}': an SDXL refiner checkpoint (one OpenCLIP bigG tower); only SDXL base is supported");
        if (names.Contains("cond_stage_model.model.ln_final.weight") || names.Any(n => n.StartsWith("cond_stage_model.model.transformer.", StringComparison.Ordinal))) return SdFamily.SD2;
        if (names.Any(n => n.StartsWith("cond_stage_model.transformer.", StringComparison.Ordinal))) return SdFamily.SD1;
        // a UNet-only file: the cross-attention width says the family
        var k = tensors.FirstOrDefault(t => t.Name.StartsWith("model.diffusion_model.", StringComparison.Ordinal) && t.Name.EndsWith(".attn2.to_k.weight", StringComparison.Ordinal));
        if (k != null) return k.Shape[1] switch { 768 => SdFamily.SD1, 1024 => SdFamily.SD2, 2048 => SdFamily.SDXL, _ => throw new NotSupportedException($"'{name}': cross-attention width {k.Shape[1]} is not SD 1.x / 2.x / SDXL") };
        if (names.Any(n => n.StartsWith("down_blocks.", StringComparison.Ordinal) || n.StartsWith("unet.down_blocks.", StringComparison.Ordinal)))
            throw new NotSupportedException($"'{name}': diffusers-format tensor names; this loader reads the single-file (LDM) layout ComfyUI and CivitAI use");
        throw new NotSupportedException($"'{name}': not a Stable Diffusion checkpoint (no model.diffusion_model.* tensors)");
    }

    private static bool HasTextEncoder(HashSet<string> names, SdFamily family) => family switch
    {
        SdFamily.SD1 => names.Contains("cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"),
        SdFamily.SD2 => names.Contains("cond_stage_model.model.token_embedding.weight"),
        _ => names.Contains("conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight") && names.Contains("conditioner.embedders.1.model.token_embedding.weight"),
    };

    private static readonly Regex HfLayer = new(@"^encoder\.layers\.(\d+)\.(.+)$", RegexOptions.Compiled);
    private static readonly Regex OcLayer = new(@"^transformer\.resblocks\.(\d+)\.(.+)$", RegexOptions.Compiled);

    /// <summary>Hugging Face CLIPTextModel names (after <c>text_model.</c>) -> the normalised names, or null for tensors the encoder does not read.</summary>
    internal static string? MapHfClip(string n, string p)
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
    internal static string? MapOpenClip(string n, string p, out SdLayout layout)
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

    public void Dispose()
    {
        UNet?.Dispose(); TextEncoder?.Dispose(); Vae?.Dispose();
        _unet?.Dispose(); _text?.Dispose(); _vae?.Dispose();
    }
}
