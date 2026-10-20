using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;

namespace SpawnDev.ILGPU.ML.Flux3;

/// <summary>
/// Writes the full scene caption FLUX 3 Image Klein draws from, with a small local language model (Qwen3 4B) and Black Forest Labs' upsampler instructions (<see cref="Flux3PromptUpsampler"/>).
/// The model is loaded for the call and freed after it, so it never shares the GPU with FLUX 3's transformer (the caller releases that first). Shared by the browser demo's Image Generation and Text to 3D pages
/// and by the desktop examples.
/// </summary>
public static class Flux3CaptionWriter
{
    // Q4_0, not Q4_K_M: the Q4_K GEMV keeps Group.Barrier inside a data-dependent loop, which WebGPU's WGSL validation rejects (Plans/webgpu-q4k-gemv-uniform-barrier-2026-10-19.md);
    // llama.cpp's Q4_0 files also hold some tensors as Q4_1 (fused kernel since 5.3.22) and Q6_K.
    public const string Repo = "unsloth/Qwen3-4B-GGUF", File = "Qwen3-4B-Q4_0.gguf";
    public const string Label = "Qwen3 4B", SizeText = "2.4 GB";

    /// <summary>Minimum time between two caption-text callbacks: a render per token would slow decoding on the single WASM thread (the chat page throttles to about 4 a second for the same reason).</summary>
    public const int CaptionTextIntervalMs = 200;

    /// <summary>Opens a model file (repo, path) reporting the download's progress: <c>HubModelSource.OpenAsync</c> and <c>HttpClientModelSource.OpenAsync</c> both fit.</summary>
    public delegate Task<Stream> OpenModel(string repo, string file, IProgress<ModelDownloadProgress> progress);

    /// <param name="open">where the model file comes from</param>
    /// <param name="status">receives progress text ("Qwen3 4B — downloading 12%") as the model loads and writes</param>
    /// <param name="progress">receives 0 to 100 while the model downloads and loads (100 when it is ready), so a page can drive its progress bar from it; not called while the caption is written</param>
    /// <param name="captionText">receives the caption as it is written (the whole text so far, at most every <see cref="CaptionTextIntervalMs"/> ms, and the finished caption last)</param>
    public static async Task<string> WriteAsync(global::ILGPU.Runtime.Accelerator accelerator, OpenModel open, string request, int width, int height, Action<string> status,
        Action<double>? progress = null, Action<string>? captionText = null)
    {
        Stream? stream = null; GgufTextGenerationPipeline? llm = null;
        try
        {
            var download = new Progress<ModelDownloadProgress>(p =>
            {
                status(p.Fraction is { } f ? $"{Label} — downloading {f:P0}" : $"{Label} — downloading {p.BytesReceived / 1048576} MiB");
                if (p.Fraction is { } fraction) progress?.Invoke(Math.Min(100, fraction * 100));
            });
            // generation reads only the last position's logits: without this the prefill computes the 152k-wide LM head for every prompt row (the 3.7k-character upsampler prompt is about 900 rows: a ~550 MB logits tensor)
            SpawnDev.ILGPU.ML.GGUF.GGUFGraphBuilder.EnableLastPositionLogits = true;
            stream = await open(Repo, File, download);
            llm = await GgufTextGenerationPipeline.CreateFromStreamAsync(accelerator, stream, maxSeqLen: 4096,
                onProgress: (stage, pct) => { status($"{Label} — {stage} {pct}%"); progress?.Invoke(Math.Min(100, pct)); });
            llm.EnableWebGPUDecodeCapture = true;
            SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.EnableWebGPUGemv = true;
            progress?.Invoke(100);
            status($"{Label} — writing the caption");
            var clock = System.Diagnostics.Stopwatch.StartNew();
            long lastShown = -CaptionTextIntervalMs;
            var caption = await Flux3PromptUpsampler.FromPipeline(llm).ExpandAsync(request, width, height, (tokens, textSoFar) =>
            {
                if (clock.ElapsedMilliseconds - lastShown >= CaptionTextIntervalMs)
                {
                    lastShown = clock.ElapsedMilliseconds;
                    captionText?.Invoke(textSoFar);
                    status($"{Label} — writing the caption ({tokens} tokens)");
                }
                return Task.CompletedTask;
            });
            caption = string.IsNullOrWhiteSpace(caption) ? request : caption;
            captionText?.Invoke(caption);
            return caption;
        }
        finally
        {
            await accelerator.SynchronizeAsync();
            llm?.Dispose(); stream?.Dispose();
        }
    }
}
