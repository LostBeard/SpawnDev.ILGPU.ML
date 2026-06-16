using SpawnDev.ILGPU.ML.GGUF;

namespace SpawnDev.ILGPU.ML.Multimodal;

/// <summary>
/// The raw gemma4 "Unified" projector weights (vision <c>gemma4uv</c> + audio <c>gemma4ua</c>), decoupled
/// from their GGUF source. Both the CPU reference (<see cref="Gemma4MultimodalProjector"/>) and the GPU port
/// (<see cref="Gemma4MultimodalProjectorGpu"/>) are built from this container, so a unit test can construct
/// synthetic weights at small dims and run the EXACT same forward on both paths to verify equivalence —
/// no GGUF file required.
///
/// All weight arrays are in raw ggml layout (the matmul indexes <c>W[o,i] = data[o*in + i]</c> directly,
/// no transpose). Null arrays mean that modality's projector is absent.
/// </summary>
public sealed class Gemma4ProjectorWeights
{
    /// <summary>Embedding dim of the LLM (projection output), 3840 for gemma4 12B.</summary>
    public int EmbedDim { get; init; }
    /// <summary>Flattened patch length the vision path consumes (6912 = 48*48*3).</summary>
    public int PatchLen { get; init; }
    /// <summary>Raw audio frame length (640 samples).</summary>
    public int AudioFrameLen { get; init; }
    /// <summary>Position-embedding table length (1120) — caps the resized grid to this many patches per axis.</summary>
    public int PosTableLen { get; init; }

    // Vision weights.
    public float[]? PatchEmbdW { get; init; }   // [out=EmbedDim, in=PatchLen]
    public float[]? PatchEmbdB { get; init; }   // [EmbedDim]
    public float[]? PatchNorm1W { get; init; }  // [PatchLen]
    public float[]? PatchNorm1B { get; init; }  // [PatchLen]
    public float[]? PatchNorm2W { get; init; }  // [EmbedDim]
    public float[]? PatchNorm2B { get; init; }  // [EmbedDim]
    public float[]? PatchNorm3W { get; init; }  // [EmbedDim]
    public float[]? PatchNorm3B { get; init; }  // [EmbedDim]
    public float[]? PosEmbd { get; init; }       // [EmbedDim * PosTableLen * 2] (ne0=EmbedDim, ne1=pos, ne2=axis)
    public float[]? MmInputProjW { get; init; }  // [out=EmbedDim, in=EmbedDim]

    // Audio weights.
    public float[]? MmAInputProjW { get; init; } // [out=EmbedDim, in=AudioFrameLen]

    /// <summary>True if the vision projector (gemma4uv) weights are present.</summary>
    public bool HasVision => PatchEmbdW != null;
    /// <summary>True if the audio projector (gemma4ua) weights are present.</summary>
    public bool HasAudio => MmAInputProjW != null;

    /// <summary>Pull the projector weights out of a parsed mmproj GGUF (all dequantized to f32). Vision and
    /// audio are independent — a single mmproj can carry either or both.</summary>
    public static Gemma4ProjectorWeights FromMmproj(MmprojModel mm)
    {
        bool v = mm.HasVisionEncoder;
        return new Gemma4ProjectorWeights
        {
            EmbedDim = mm.VisionProjectionDim,                              // 3840
            PatchLen = mm.GetTensorShape("v.patch_embd.weight")?[0] ?? 6912, // ne0 = in = 6912
            AudioFrameLen = mm.AudioFrameLength,                            // 640
            PosTableLen = mm.GetTensorShape("v.position_embd.weight")?[1] ?? 1120, // ne1 = positions

            PatchEmbdW = v ? mm.GetTensorF32("v.patch_embd.weight") : null,
            PatchEmbdB = v ? mm.GetTensorF32("v.patch_embd.bias") : null,
            PatchNorm1W = v ? mm.GetTensorF32("v.patch_norm.1.weight") : null,
            PatchNorm1B = v ? mm.GetTensorF32("v.patch_norm.1.bias") : null,
            PatchNorm2W = v ? mm.GetTensorF32("v.patch_norm.2.weight") : null,
            PatchNorm2B = v ? mm.GetTensorF32("v.patch_norm.2.bias") : null,
            PatchNorm3W = v ? mm.GetTensorF32("v.patch_norm.3.weight") : null,
            PatchNorm3B = v ? mm.GetTensorF32("v.patch_norm.3.bias") : null,
            PosEmbd = v ? mm.GetTensorF32("v.position_embd.weight") : null,
            MmInputProjW = v ? mm.GetTensorF32("mm.input_projection.weight") : null,

            MmAInputProjW = mm.HasAudioEncoder ? mm.GetTensorF32("mm.a.input_projection.weight") : null,
        };
    }
}
