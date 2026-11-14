using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Flux1;

using BV = ArrayView1D<BFloat16, Stride1D.Dense>;

/// <summary>
/// The FLUX.1 transformer's weights from a file in Black Forest Labs' layout: ComfyUI's all-in-one checkpoints (<c>flux1-dev-fp8.safetensors</c>: the transformer under
/// <c>model.diffusion_model.</c>, 8-bit floats with no scales, biases and norms included) or a bare transformer file (<c>flux1-dev.safetensors</c>, bf16, no prefix).
/// The 12B transformer is larger than most cards (11.9 GB as 8-bit floats), so its 57 blocks are split by <see cref="StreamedModelWeights"/>: as many as
/// <c>residentBytes</c> allows stay on the device, the rest are read from the file for every forward pass.
/// </summary>
public sealed class Flux1Weights : IDisposable
{
    public const int DoubleBlocks = 19, SingleBlocks = 38;
    private readonly StreamedModelWeights _s;

    /// <summary>Whether the model has a guidance embedder (FLUX.1 dev: guidance-distilled; schnell has none).</summary>
    public bool HasGuidance { get; }
    public long ResidentBytes => _s.ResidentBytes;
    /// <summary>How many of the 57 blocks stay on the device; the others are read from the file on every forward pass.</summary>
    public int ResidentBlocks => _s.ResidentBlocks;
    /// <summary>The bytes a forward pass reads from the file (the streamed blocks).</summary>
    public long StreamedBytesPerForward => _s.StreamedBytesPerForward;
    /// <summary>The non-block tensors (embedders, final layer).</summary>
    public GpuWeights Base => _s.Base;

    private Flux1Weights(StreamedModelWeights s) { _s = s; HasGuidance = s.Base.Has("guidance_in.in_layer.weight"); }

    /// <summary>Block index 0..18 = double_blocks.k, 19..56 = single_blocks.k - 19.</summary>
    public static string BlockPrefix(int k) => k < DoubleBlocks ? $"double_blocks.{k}." : $"single_blocks.{k - DoubleBlocks}.";

    private static int BlockOf(string n)
    {
        if (n.StartsWith("double_blocks.", StringComparison.Ordinal)) return int.Parse(n.AsSpan(14, n.IndexOf('.', 14) - 14));
        if (n.StartsWith("single_blocks.", StringComparison.Ordinal)) return DoubleBlocks + int.Parse(n.AsSpan(14, n.IndexOf('.', 14) - 14));
        return -1;
    }

    /// <summary>Open the transformer of <paramref name="file"/>: the non-block tensors, then blocks in order while they fit in <paramref name="residentBytes"/> (long.MaxValue keeps all).</summary>
    public static async Task<Flux1Weights> OpenAsync(Accelerator accelerator, IModelSource source, string repo, string file, long residentBytes, CancellationToken ct = default)
        => new Flux1Weights(await StreamedModelWeights.OpenAsync(accelerator, source, repo, file, "double_blocks.0.img_attn.qkv.weight", BlockOf, DoubleBlocks + SingleBlocks, residentBytes, null, ct));

    /// <summary>Block <paramref name="k"/>'s tensors: the resident set, or a fresh load the caller disposes after the device ran it (<paramref name="streamed"/> true).</summary>
    public Task<(GpuWeights Weights, bool Streamed)> BlockAsync(int k, CancellationToken ct = default) => _s.BlockAsync(k, ct);

    /// <summary>A matrix [N, K] of <paramref name="w"/> as bf16 (8-bit floats decoded into the shared scratch, valid until the next call).</summary>
    public BV Matrix(GpuWeights w, string name) => _s.Matrix(w, name);

    /// <summary>True for an fp8_scaled file (see <see cref="StreamedModelWeights.HasScales"/>).</summary>
    public bool HasScales => _s.HasScales;
    /// <summary><see cref="StreamedModelWeights.MatrixExact"/>: the unscaled exact decode, paired with <see cref="ActivationScale"/>.</summary>
    public BV MatrixExact(GpuWeights w, string name) => _s.MatrixExact(w, name);
    /// <summary><see cref="StreamedModelWeights.ActivationScale"/>.</summary>
    public float ActivationScale(GpuWeights w, string name) => _s.ActivationScale(w, name);

    public void Dispose() => _s.Dispose();
}
