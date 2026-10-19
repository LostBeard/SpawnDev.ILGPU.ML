using SpawnDev.ILGPU.ML.Flux3;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;

namespace SpawnDev.ILGPU.ML.Demo.Services;

/// <summary>
/// Writes the full scene caption FLUX 3 Image Klein draws from, with a small local language model (Qwen3 4B) and Black Forest Labs' upsampler instructions (<see cref="Flux3PromptUpsampler"/>).
/// The model is loaded for the call and freed after it, so it never shares the GPU with FLUX 3's transformer (the caller releases that first). Shared by the Image Generation and Text to 3D pages.
/// </summary>
public static class CaptionWriter
{
    // Q4_0, not Q4_K_M: the Q4_K GEMV keeps Group.Barrier inside a data-dependent loop, which WebGPU's WGSL validation rejects (Plans/webgpu-q4k-gemv-uniform-barrier-2026-10-19.md);
    // llama.cpp's Q4_0 files also hold some tensors as Q4_1 (fused kernel since 5.3.22) and Q6_K.
    public const string Repo = "unsloth/Qwen3-4B-GGUF", File = "Qwen3-4B-Q4_0.gguf";
    public const string Label = "Qwen3 4B", SizeText = "2.4 GB";

    /// <param name="status">receives progress text ("Qwen3 4B — downloading 12%") as the model loads and writes</param>
    public static async Task<string> WriteAsync(global::ILGPU.Runtime.Accelerator accelerator, HubModelSource source, string request, int width, int height, Action<string> status)
    {
        Stream? open = null; GgufTextGenerationPipeline? llm = null;
        try
        {
            var download = new Progress<ModelDownloadProgress>(p =>
                status(p.Fraction is { } f ? $"{Label} — downloading {f:P0}" : $"{Label} — downloading {p.BytesReceived / 1048576} MiB"));
            open = await source.OpenAsync(Repo, File, download);
            llm = await GgufTextGenerationPipeline.CreateFromStreamAsync(accelerator, open, maxSeqLen: 4096,
                onProgress: (stage, pct) => status($"{Label} — {stage} {pct}%"));
            llm.EnableWebGPUDecodeCapture = true;
            SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.EnableWebGPUGemv = true;
            status($"{Label} — writing the caption");
            var caption = await Flux3PromptUpsampler.FromPipeline(llm).ExpandAsync(request, width, height);
            return string.IsNullOrWhiteSpace(caption) ? request : caption;
        }
        finally
        {
            await accelerator.SynchronizeAsync();
            llm?.Dispose(); open?.Dispose();
        }
    }
}
