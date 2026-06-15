using System.IO;

namespace SpawnDev.ILGPU.ML.GGUF;

/// <summary>
/// A parsed gemma4 multimodal projector ("mmproj") GGUF. This is the small companion file to the text
/// decoder GGUF (e.g. <c>mmproj-gemma-4-12B-it-bf16.gguf</c>, ~167 MB, <c>general.architecture=clip</c>,
/// <c>general.type=mmproj</c>). Gemma 4 12B is the ENCODER-FREE "Unified" variant: there is no SigLIP
/// vision tower and no Conformer audio encoder (<c>clip.vision.block_count=0</c>,
/// <c>clip.audio.block_count=0</c>). Raw image patches and raw audio waveform frames are projected
/// DIRECTLY into the LLM's embedding space by the lightweight linear layers held here.
///
/// Tensors (the only 11):
///   v.patch_embd.weight [6912,3840] + v.patch_embd.bias [3840]   — vision 48x48x3 patch -> 3840
///   v.patch_norm.1.{weight,bias} [6912]                          — LayerNorm pre-projection
///   v.patch_norm.2.{weight,bias} [3840]                          — LayerNorm post-projection
///   v.patch_norm.3.{weight,bias} [3840]                          — LayerNorm post-position-embedding
///   v.position_embd.weight [3840,1120,2]                         — factorized 2D pos (tbl_x | tbl_y)
///   mm.input_projection.weight [3840,3840]   (bf16)              — vision -> LLM embed
///   mm.a.input_projection.weight [640,3840]  (bf16)              — audio  -> LLM embed
///
/// See <c>Plans/gemma4-multimodal-bringup.md</c> for the full forward spec.
/// </summary>
public sealed class MmprojModel
{
    /// <summary>The underlying parsed GGUF (full RawData loaded — the file is small enough to hold in memory).</summary>
    public GGUFModel Gguf { get; }

    private MmprojModel(GGUFModel gguf) => Gguf = gguf;

    /// <summary>Parse an mmproj GGUF from a local file path (reads the whole file into memory).</summary>
    public static MmprojModel Load(string path)
    {
        var bytes = File.ReadAllBytes(path);
        return Load(bytes);
    }

    /// <summary>Parse an mmproj GGUF from a full in-memory byte[].</summary>
    public static MmprojModel Load(byte[] bytes)
    {
        var gguf = GGUFParser.Parse(bytes);
        var arch = gguf.GetMetadataString("general.architecture");
        if (arch != "clip")
            throw new InvalidOperationException($"Not an mmproj GGUF: general.architecture='{arch}' (expected 'clip').");
        var model = new MmprojModel(gguf);
        model.Validate();
        return model;
    }

    // ── clip metadata ──────────────────────────────────────────────────────────────────────────────

    public bool HasVisionEncoder => Gguf.GetMetadataInt("clip.has_vision_encoder", 0) != 0;
    public bool HasAudioEncoder => Gguf.GetMetadataInt("clip.has_audio_encoder", 0) != 0;

    /// <summary>Vision projector type, e.g. "gemma4uv" (Unified encoder-free vision).</summary>
    public string? VisionProjectorType => Gguf.GetMetadataString("clip.vision.projector_type");
    /// <summary>Audio projector type, e.g. "gemma4ua" (Unified encoder-free raw-waveform audio).</summary>
    public string? AudioProjectorType => Gguf.GetMetadataString("clip.audio.projector_type");

    /// <summary>Nominal image size from metadata (224). NOTE the gemma4uv runtime uses free-aspect resize to
    /// multiples of the effective 48px patch, so this value is informational, not a hard square crop.</summary>
    public int VisionImageSize => (int)Gguf.GetMetadataInt("clip.vision.image_size", 224);
    /// <summary>Stored patch size (16). The gemma4uv runtime multiplies this by n_merge=3 → an effective 48px patch.</summary>
    public int VisionPatchSize => (int)Gguf.GetMetadataInt("clip.vision.patch_size", 16);
    public int VisionEmbeddingLength => (int)Gguf.GetMetadataInt("clip.vision.embedding_length", 3840);
    public int VisionProjectionDim => (int)Gguf.GetMetadataInt("clip.vision.projection_dim", 3840);
    public float[] VisionImageMean => Gguf.GetMetadataFloatArray("clip.vision.image_mean") ?? new[] { 0f, 0f, 0f };
    public float[] VisionImageStd => Gguf.GetMetadataFloatArray("clip.vision.image_std") ?? new[] { 1f, 1f, 1f };

    /// <summary>RAW waveform frame length fed to mm.a.input_projection (640 samples @ 16 kHz = 40 ms).
    /// Derived from the projection weight's input dim (ne0) — authoritative. NOTE: the metadata field
    /// <c>clip.audio.num_mel_bins</c> is a MISNOMER here (it reads 128 and is unused — gemma4ua is raw
    /// waveform, no mel); <c>clip.audio.embedding_length</c> (640) matches but the weight is the source of truth.</summary>
    public int AudioFrameLength =>
        GetTensorShape("mm.a.input_projection.weight")?[0] ?? (int)Gguf.GetMetadataInt("clip.audio.embedding_length", 640);
    public int AudioProjectionDim => (int)Gguf.GetMetadataInt("clip.audio.projection_dim", 3840);

    // ── tensors ────────────────────────────────────────────────────────────────────────────────────

    /// <summary>Look up a tensor descriptor by exact name, or null if absent.</summary>
    public GGUFTensorInfo? FindTensor(string name) => Gguf.Tensors.FirstOrDefault(t => t.Name == name);

    /// <summary>Get a tensor's data dequantized to float32 (bf16/f16/quant all widened to f32), or null if absent.</summary>
    public float[]? GetTensorF32(string name)
    {
        var t = FindTensor(name);
        return t == null ? null : Gguf.GetTensorFloat32(t);
    }

    /// <summary>Get a tensor's GGUF dimensions (fastest-varying first, as stored), or null if absent.</summary>
    public int[]? GetTensorShape(string name) => FindTensor(name)?.Shape;

    private void Validate()
    {
        // Fail loud at load if the expected projector tensors are missing — a silently-absent projection
        // weight would surface much later as a shape mismatch deep in the forward pass.
        var required = new List<string>();
        if (HasVisionEncoder)
            required.AddRange(new[] { "v.patch_embd.weight", "v.patch_embd.bias", "v.patch_norm.1.weight",
                "v.patch_norm.2.weight", "v.patch_norm.3.weight", "v.position_embd.weight", "mm.input_projection.weight" });
        if (HasAudioEncoder)
            required.Add("mm.a.input_projection.weight");

        var missing = required.Where(n => FindTensor(n) == null).ToList();
        if (missing.Count > 0)
            throw new InvalidOperationException(
                $"mmproj GGUF is missing expected tensor(s): {string.Join(", ", missing)}. " +
                $"Present: {string.Join(", ", Gguf.Tensors.Select(t => t.Name))}");
    }
}
