using SpawnDev.BlazorJS;

namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// Download ONNX models from HuggingFace Hub or any URL with browser-side OPFS caching.
/// Models are cached locally so they only download once — subsequent loads are instant.
/// <para>
/// For API access (search, metadata, file listing) without caching, use
/// <see cref="HuggingFaceClient"/> directly.
/// </para>
/// <code>
/// var hub = new ModelHub(js);
/// hub.OnProgress += (received, total) =&gt; UpdateProgressBar(received, total);
///
/// // Load from HuggingFace (cached in OPFS after first download)
/// var bytes = await hub.LoadAsync(KnownModels.SqueezeNet, KnownFiles.OnnxModel);
/// var session = InferenceSession.CreateFromFile(accelerator, bytes);
///
/// // Or load any model by repo ID
/// var bytes2 = await hub.LoadAsync("onnx-community/depth-anything-v2-small", "onnx/model.onnx");
///
/// // Load from any URL (also cached)
/// var bytes3 = await hub.LoadFromUrlAsync("https://example.com/model.onnx");
///
/// // Check cache status
/// Console.WriteLine($"Cache size: {await hub.GetCacheSizeAsync() / 1024 / 1024} MB");
/// var cached = await hub.ListCachedAsync();
/// foreach (var (key, size) in cached) Console.WriteLine($"  {key}: {size / 1024 / 1024} MB");
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
    /// <param name="filename">File within the repo (e.g., "model.onnx" or "onnx/model.onnx")</param>
    /// <param name="revision">Git revision (default: "main")</param>
    public Task<byte[]> LoadAsync(string repoId, string filename, string revision = "main")
    {
        var url = $"{HuggingFaceBaseUrl}/{repoId}/resolve/{revision}/{filename}";
        var cacheKey = $"hf_{repoId.Replace('/', '_')}_{revision}_{filename.Replace('/', '_')}";
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
        var cacheKey = $"hf_{repoId.Replace('/', '_')}_{revision}_{filename.Replace('/', '_')}";
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

    // ═══════════════════════════════════════════════════════════
    //  Well-Known Models
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Well-known HuggingFace repository IDs for quick access.
    /// Use with <see cref="LoadAsync"/> and <see cref="KnownFiles"/>.
    /// <code>
    /// var bytes = await hub.LoadAsync(KnownModels.SqueezeNet, KnownFiles.OnnxModel);
    /// </code>
    /// </summary>
    public static class KnownModels
    {
        // ── Classification ──
        /// <summary>SqueezeNet 1.1 — Fast 1000-class ImageNet classification (4.8 MB)</summary>
        public const string SqueezeNet = "onnxmodelzoo/squeezenet1.1-7";
        /// <summary>MobileNetV2 — Accurate 1000-class ImageNet classification (14 MB)</summary>
        public const string MobileNetV2 = "onnxmodelzoo/mobilenetv2-12";
        /// <summary>EfficientNet-Lite0 — 1000-class ImageNet, TFLite format (18 MB)</summary>
        public const string EfficientNetLite0 = "litert-community/efficientnet-lite0";

        // ── Depth Estimation ──
        /// <summary>Depth Anything V2 Small — Monocular depth estimation, 26M params (95 MB)</summary>
        public const string DepthAnythingV2Small = "onnx-community/depth-anything-v2-small";

        // ── Object Detection ──
        /// <summary>YOLOv8 Nano — 80-class COCO object detection (13 MB)</summary>
        public const string YOLOv8n = "salim4n/yolov8n-detect-onnx";

        // ── Face Detection ──
        /// <summary>BlazeFace — MediaPipe face detector, TFLite format (228 KB)</summary>
        public const string BlazeFace = "litert-community/blaze-face";

        // ── Pose Estimation ──
        /// <summary>MoveNet Lightning — Fast 17-keypoint pose detection (9 MB)</summary>
        public const string MoveNetLightning = "Xenova/movenet-singlepose-lightning";

        // ── Style Transfer (ONNX Model Zoo on HuggingFace) ──
        /// <summary>Fast Neural Style — Mosaic tile artistic style (6.5 MB)</summary>
        public const string StyleMosaic = "onnxmodelzoo/mosaic-9";
        /// <summary>Fast Neural Style — Bright candy-colored style (6.5 MB)</summary>
        public const string StyleCandy = "onnxmodelzoo/candy-9";
        /// <summary>Fast Neural Style — Impressionist rain scene (6.5 MB)</summary>
        public const string StyleRainPrincess = "onnxmodelzoo/rain-princess-9";
        /// <summary>Fast Neural Style — Abstract cubist style (6.5 MB)</summary>
        public const string StyleUdnie = "onnxmodelzoo/udnie-9";
        /// <summary>Fast Neural Style — Pointillist dot painting (6.5 MB)</summary>
        public const string StylePointilism = "onnxmodelzoo/pointilism-9";

        // ── Super Resolution (ONNX Model Zoo on HuggingFace) ──
        /// <summary>ESPCN 3x — Sub-pixel CNN super resolution (236 KB)</summary>
        public const string SuperResolution = "onnxmodelzoo/super-resolution-10";

        // ── Text / NLP ──
        /// <summary>DistilBERT SST-2 — Sentiment analysis (257 MB)</summary>
        public const string DistilBertSST2 = "Xenova/distilbert-base-uncased-finetuned-sst-2-english";
        /// <summary>GPT-2 — Text generation (548 MB)</summary>
        public const string GPT2 = "onnxmodelzoo/gpt2-10";

        // ── Speech ──
        /// <summary>Whisper Tiny — Speech-to-text, encoder+decoder (226 MB)</summary>
        public const string WhisperTiny = "onnx-community/whisper-tiny";

        // ── Multimodal ──
        /// <summary>CLIP ViT-B/32 — Vision+text encoder for zero-shot classification (606 MB combined)</summary>
        public const string CLIPVitB32 = "Xenova/clip-vit-base-patch32";

        // ── Text-to-Speech ──
        /// <summary>SpeechT5 TTS — Text-to-speech encoder+decoder (587 MB)</summary>
        public const string SpeechT5TTS = "Xenova/speecht5_tts";
        /// <summary>SpeechT5 HiFi-GAN — Vocoder for SpeechT5 (55 MB)</summary>
        public const string SpeechT5HiFiGAN = "Xenova/speecht5_hifigan";

        // ── Background Removal ──
        /// <summary>RMBG 1.4 — Background removal (BRIA AI)</summary>
        public const string RMBG14 = "briaai/RMBG-1.4";
    }

    /// <summary>
    /// Well-known filenames within HuggingFace model repos.
    /// Most ONNX models on HuggingFace follow the <c>onnx/model.onnx</c> convention.
    /// </summary>
    public static class KnownFiles
    {
        /// <summary>Standard ONNX model path in HuggingFace repos</summary>
        public const string OnnxModel = "onnx/model.onnx";
        /// <summary>FP16 quantized ONNX model</summary>
        public const string OnnxModelFP16 = "onnx/model_fp16.onnx";
        /// <summary>INT8 quantized ONNX model</summary>
        public const string OnnxModelQuantized = "onnx/model_quantized.onnx";
        /// <summary>Q4 quantized ONNX model</summary>
        public const string OnnxModelQ4 = "onnx/model_q4.onnx";
        /// <summary>Q4F16 quantized ONNX model</summary>
        public const string OnnxModelQ4F16 = "onnx/model_q4f16.onnx";
        /// <summary>Vision model for CLIP</summary>
        public const string OnnxVisionModel = "onnx/vision_model.onnx";
        /// <summary>Text model for CLIP</summary>
        public const string OnnxTextModel = "onnx/text_model.onnx";
        /// <summary>Encoder model (e.g., Whisper, SpeechT5)</summary>
        public const string OnnxEncoderModel = "onnx/encoder_model.onnx";
        /// <summary>Decoder model merged (e.g., SpeechT5)</summary>
        public const string OnnxDecoderModelMerged = "onnx/decoder_model_merged.onnx";
    }

    public void Dispose()
    {
        _cache.Dispose();
    }
}
