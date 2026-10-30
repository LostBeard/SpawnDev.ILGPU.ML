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
    public SdPrediction Prediction { get; private set; }
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
        Family = family; Prediction = prediction; _unet = unet; _text = text; _vae = vae; _acc = acc;
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
                    local = SdNames.MapHfClip(n["cond_stage_model.transformer.text_model.".Length..], "l.");
                else if (family == SdFamily.SD2 && n.StartsWith("cond_stage_model.model.", StringComparison.Ordinal))
                    local = SdNames.MapOpenClip(n["cond_stage_model.model.".Length..], "h.", out layout);
                else if (family == SdFamily.SD2 && n.StartsWith("conditioner.embedders.0.model.", StringComparison.Ordinal))     // the sgm layout (sd_turbo)
                    local = SdNames.MapOpenClip(n["conditioner.embedders.0.model.".Length..], "h.", out layout);
                else if (family == SdFamily.SDXL && n.StartsWith("conditioner.embedders.0.transformer.text_model.", StringComparison.Ordinal))
                    local = SdNames.MapHfClip(n["conditioner.embedders.0.transformer.text_model.".Length..], "l.");
                else if (family == SdFamily.SDXL && n.StartsWith("conditioner.embedders.1.model.", StringComparison.Ordinal))
                    local = SdNames.MapOpenClip(n["conditioner.embedders.1.model.".Length..], "g.", out layout);
                return local == null ? null : new SdTensorStreamer.Target(text, local, layout);
            }
            using (var streamer = new SdTensorStreamer(accelerator))
                await streamer.StreamAsync(stream, tensors, pos, Route, p => onProgress?.Invoke("weights", (double)p / total), cancellationToken);
            onProgress?.Invoke("weights", 1.0);

            // fuse the projections that run as one GEMM: q | k | v of every self-attention, k | v of every cross-attention
            if (unet != null)
            {
                await SdNames.FuseUNetAsync(accelerator, unet);
                if (!unet.Has("input_blocks.0.0.weight")) throw new InvalidDataException($"'{name}': no UNet (model.diffusion_model.*) in this file");
            }
            if (vae != null) await SdNames.FuseVaeAsync(accelerator, vae);
            if (text != null) await SdNames.FuseTextAsync(accelerator, text);

            var ck = new SdCheckpoint(family, SdPrediction.Epsilon, unet, text, vae, accelerator);
            if (ck.UNet != null) ck.UNet.VPredTensor = names.Contains("v_pred");
            ck.Prediction = options.Prediction ?? (ck.UNet != null ? await ck.UNet.InferPredictionAsync() : names.Contains("v_pred") ? SdPrediction.V : SdPrediction.Epsilon);
            return ck;
        }
        catch
        {
            unet?.Dispose(); text?.Dispose(); vae?.Dispose();
            throw;
        }
    }

    private static SdFamily DetectFamily(string name, HashSet<string> names, List<SafeTensorInfo> tensors)
    {
        if (names.Any(n => n.StartsWith("conditioner.embedders.1.model.", StringComparison.Ordinal))) return SdFamily.SDXL;
        if (names.Any(n => n.StartsWith("conditioner.embedders.0.model.", StringComparison.Ordinal)))
        {
            // Stability's sgm layout puts one OpenCLIP tower at embedders.0: ViT-H (1024 wide) = SD 2.x (sd_turbo.safetensors), bigG (1280) = the SDXL refiner (ComfyUI: SD20 / SDXLRefiner)
            var lnf = tensors.FirstOrDefault(t => t.Name == "conditioner.embedders.0.model.ln_final.weight");
            if (lnf != null && lnf.Shape[0] == 1024) return SdFamily.SD2;
            throw new NotSupportedException($"'{name}': an SDXL refiner checkpoint (one OpenCLIP bigG tower); only SDXL base is supported");
        }
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
        SdFamily.SD2 => names.Contains("cond_stage_model.model.token_embedding.weight") || names.Contains("conditioner.embedders.0.model.token_embedding.weight"),
        _ => names.Contains("conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight") && names.Contains("conditioner.embedders.1.model.token_embedding.weight"),
    };

    private readonly Accelerator _acc;

    /// <summary>
    /// Merge LoRAs into the loaded UNet and text encoder(s), replacing any set merged before (the originals of patched tensors are kept on the device, so this restores them
    /// first: call it again with other strengths, or with an empty list to remove every LoRA). Each LoRA's <see cref="SdLoraSpec.Strength"/> scales its UNet modules,
    /// <see cref="SdLoraSpec.ClipStrength"/> its text-encoder modules. The report lists the keys that matched nothing in this checkpoint (a LoRA for another family).
    /// </summary>
    public async Task<SdLoraReport> ApplyLorasAsync(IReadOnlyList<SdLoraSpec> loras, CancellationToken cancellationToken = default)
    {
        // each component merges its own modules (and keeps its own originals); a key is unmatched only when neither took it
        var u = UNet != null ? await UNet.ApplyLorasAsync(loras, cancellationToken) : null;
        var t = TextEncoder != null ? await TextEncoder.ApplyLorasAsync(loras, cancellationToken) : null;
        var rows = new List<(string, int, IReadOnlyList<string>)>();
        for (int i = 0; i < loras.Count; i++)
        {
            var ur = u?.Loras[i]; var tr = t?.Loras[i];
            IReadOnlyList<string> unmatched = ur == null ? tr!.Value.Unmatched : tr == null ? ur.Value.Unmatched : ur.Value.Unmatched.Intersect(tr.Value.Unmatched).ToList();
            rows.Add((loras[i].Name, (ur?.Patched ?? 0) + (tr?.Patched ?? 0), unmatched));
        }
        return new SdLoraReport(rows);
    }

    /// <summary>
    /// Write the loaded components' current weights (merges and LoRAs included) as one single-file checkpoint in the LDM layout (ComfyUI's CheckpointSave):
    /// <c>model.diffusion_model.*</c>, the text encoders under their family's names, <c>first_stage_model.*</c>, and ComfyUI's <c>v_pred</c> marker for a v-prediction
    /// model (<see cref="SdSaveOptions.VPrediction"/>, default <see cref="Prediction"/>). Components not loaded are not written. Streams front to back.
    /// </summary>
    public Task SaveAsync(Stream target, SdSaveOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new SdSaveOptions();
        var list = new List<SdSafeTensorsWriter.Out>();
        if (UNet != null) list.AddRange(SdSafeTensorsWriter.Ldm(UNet.Weights, "model.diffusion_model."));
        if (TextEncoder != null) list.AddRange(SdSafeTensorsWriter.Text(TextEncoder.Weights, Family));
        if (Vae != null) list.AddRange(SdSafeTensorsWriter.Ldm(Vae.Weights, "first_stage_model."));
        return SdSafeTensorsWriter.WriteAsync(_acc, target, list, options, options.VPrediction ?? Prediction == SdPrediction.V, cancellationToken);
    }

    public void Dispose()
    {
        UNet?.Dispose(); TextEncoder?.Dispose(); Vae?.Dispose();
        _unet?.Dispose(); _text?.Dispose(); _vae?.Dispose();
    }
}
