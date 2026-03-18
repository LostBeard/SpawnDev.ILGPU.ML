using SpawnDev.BlazorJS;

namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// Download ONNX models from HuggingFace Hub or any URL with browser-side caching.
/// Models are cached in OPFS so they only download once.
///
/// Usage:
/// <code>
/// var hub = new ModelHub(js);
/// hub.OnProgress += (received, total) => UpdateProgressBar(received, total);
///
/// // From HuggingFace
/// var modelBytes = await hub.LoadAsync("onnx-community/mobilenetv2-12", "mobilenetv2-12.onnx");
///
/// // From any URL
/// var weightsBytes = await hub.LoadFromUrlAsync("https://example.com/weights_fp16.bin");
/// </code>
/// </summary>
public class ModelHub : IDisposable
{
    private readonly ModelCache _cache;

    /// <summary>
    /// HuggingFace Hub base URL. Default: https://huggingface.co
    /// </summary>
    public string HuggingFaceBaseUrl { get; set; } = "https://huggingface.co";

    /// <summary>
    /// Fired during download with (bytesReceived, totalBytes).
    /// totalBytes may be -1 if Content-Length is not provided.
    /// </summary>
    public event Action<long, long>? OnProgress;

    public ModelHub(BlazorJSRuntime js)
    {
        _cache = new ModelCache(js);
        _cache.OnDownloadProgress += (received, total) => OnProgress?.Invoke(received, total);
    }

    /// <summary>
    /// Load a file from a HuggingFace repository.
    /// Cached in OPFS after first download.
    /// </summary>
    /// <param name="repoId">Repository ID (e.g., "onnx-community/mobilenetv2-12")</param>
    /// <param name="filename">File within the repo (e.g., "mobilenetv2-12.onnx" or "onnx/model_quantized.onnx")</param>
    /// <param name="revision">Git revision (default: "main")</param>
    public Task<byte[]> LoadAsync(string repoId, string filename, string revision = "main")
    {
        var url = $"{HuggingFaceBaseUrl}/{repoId}/resolve/{revision}/{filename}";
        var cacheKey = $"hf_{repoId.Replace('/', '_')}_{filename.Replace('/', '_')}";
        return _cache.GetOrFetchAsync(url, cacheKey);
    }

    /// <summary>
    /// Load a file from any URL with caching.
    /// </summary>
    public Task<byte[]> LoadFromUrlAsync(string url, string? cacheKey = null)
    {
        return _cache.GetOrFetchAsync(url, cacheKey);
    }

    /// <summary>
    /// Load multiple files from a HuggingFace repo (e.g., weights + manifest + graph).
    /// Downloads happen concurrently.
    /// </summary>
    public async Task<Dictionary<string, byte[]>> LoadMultipleAsync(
        string repoId, string[] filenames, string revision = "main")
    {
        var tasks = filenames.Select(f => new
        {
            Filename = f,
            Task = LoadAsync(repoId, f, revision)
        }).ToArray();

        await Task.WhenAll(tasks.Select(t => t.Task));

        return tasks.ToDictionary(t => t.Filename, t => t.Task.Result);
    }

    /// <summary>
    /// Check if a model file is already cached locally.
    /// </summary>
    public Task<bool> IsCachedAsync(string repoId, string filename, string revision = "main")
    {
        var cacheKey = $"hf_{repoId.Replace('/', '_')}_{filename.Replace('/', '_')}";
        return _cache.IsCachedAsync("", cacheKey);
    }

    /// <summary>
    /// List all cached models with sizes.
    /// </summary>
    public Task<List<(string Key, long SizeBytes)>> ListCachedAsync()
    {
        return _cache.ListCachedAsync();
    }

    /// <summary>
    /// Get total cache size in bytes.
    /// </summary>
    public Task<long> GetCacheSizeAsync()
    {
        return _cache.GetCacheSizeAsync();
    }

    /// <summary>
    /// Clear all cached models.
    /// </summary>
    public Task ClearCacheAsync()
    {
        return _cache.ClearAllAsync();
    }

    /// <summary>
    /// Well-known model repository IDs for quick access.
    /// </summary>
    public static class KnownModels
    {
        public const string MobileNetV2 = "onnxmodelzoo/mobilenetv2-12";
        public const string DepthAnythingV2Small = "onnx-community/depth-anything-v2-small";
        public const string MoveNetLightning = "Xenova/movenet-singlepose-lightning";
        public const string SuperResolution = "onnxmodelzoo/super-resolution-10";
        public const string RMBG14 = "briaai/RMBG-1.4";
    }

    /// <summary>
    /// Well-known filenames within model repos.
    /// </summary>
    public static class KnownFiles
    {
        public const string OnnxModel = "onnx/model.onnx";
        public const string OnnxModelFP16 = "onnx/model_fp16.onnx";
        public const string OnnxModelQuantized = "onnx/model_quantized.onnx";
        public const string OnnxModelQ4 = "onnx/model_q4.onnx";
        public const string OnnxModelQ4F16 = "onnx/model_q4f16.onnx";
    }

    public void Dispose()
    {
        _cache.Dispose();
    }
}
