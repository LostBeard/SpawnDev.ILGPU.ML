using System.Text.Json;
using System.Text.Json.Serialization;

namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// Client for the HuggingFace Hub API. Search models, get metadata, list files,
/// and construct download URLs for the HuggingFace CDN.
/// <para>
/// Uses <see cref="HttpClient"/> for all requests — works on all platforms
/// (Blazor WASM, desktop, server). For browser-side caching, pair with
/// <see cref="ModelHub"/> which adds OPFS-backed persistent storage.
/// </para>
/// <code>
/// var hf = new HuggingFaceClient(httpClient);
///
/// // Search for ONNX image classification models
/// var results = await hf.SearchModelsAsync("squeezenet", pipelineTag: "image-classification", library: "onnx");
///
/// // Get detailed model info
/// var info = await hf.GetModelInfoAsync("onnx-community/squeezenet1.1-7");
/// Console.WriteLine($"{info.Id}: {info.Downloads} downloads, {info.Likes} likes");
///
/// // List files in a repo
/// var files = await hf.ListRepoFilesAsync("onnx-community/squeezenet1.1-7");
/// foreach (var f in files) Console.WriteLine($"  {f.Path} ({f.Size} bytes)");
///
/// // Get a CDN download URL
/// var url = HuggingFaceClient.GetDownloadUrl("onnx-community/squeezenet1.1-7", "model.onnx");
///
/// // Load directly into an InferenceSession
/// var session = await InferenceSession.CreateFromFileAsync(accelerator, httpClient, url);
/// </code>
/// </summary>
public class HuggingFaceClient
{
    private readonly HttpClient _http;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    /// <summary>HuggingFace Hub base URL.</summary>
    public string BaseUrl { get; set; } = "https://huggingface.co";

    /// <summary>HuggingFace API base URL.</summary>
    public string ApiUrl => $"{BaseUrl}/api";

    /// <summary>
    /// Optional authentication token for accessing private/gated models.
    /// Public models do not require a token.
    /// </summary>
    public string? AuthToken { get; set; }

    public HuggingFaceClient(HttpClient http, string? authToken = null)
    {
        _http = http;
        AuthToken = authToken;
    }

    // ═══════════════════════════════════════════════════════════
    //  Search & Discovery
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Search for models on HuggingFace Hub.
    /// </summary>
    /// <param name="query">Free-text search query (e.g., "squeezenet", "depth estimation")</param>
    /// <param name="author">Filter by author/organization (e.g., "onnx-community")</param>
    /// <param name="pipelineTag">Filter by task (e.g., "image-classification", "depth-estimation", "text-generation")</param>
    /// <param name="library">Filter by library/framework (e.g., "onnx", "transformers", "gguf")</param>
    /// <param name="tags">Additional tags to filter by</param>
    /// <param name="sort">Sort field: "downloads", "likes", "lastModified", "trending"</param>
    /// <param name="direction">Sort direction: -1 for descending, 1 for ascending</param>
    /// <param name="limit">Maximum results to return (default: 20, max: 100)</param>
    /// <param name="full">If true, return full model info including siblings (files)</param>
    public async Task<HFModelInfo[]> SearchModelsAsync(
        string? query = null,
        string? author = null,
        string? pipelineTag = null,
        string? library = null,
        string[]? tags = null,
        string sort = "downloads",
        int direction = -1,
        int limit = 20,
        bool full = false)
    {
        var queryParams = new List<string>();
        if (!string.IsNullOrEmpty(query)) queryParams.Add($"search={Uri.EscapeDataString(query)}");
        if (!string.IsNullOrEmpty(author)) queryParams.Add($"author={Uri.EscapeDataString(author)}");
        if (!string.IsNullOrEmpty(pipelineTag)) queryParams.Add($"pipeline_tag={Uri.EscapeDataString(pipelineTag)}");
        if (!string.IsNullOrEmpty(library)) queryParams.Add($"library={Uri.EscapeDataString(library)}");
        if (tags != null)
        {
            foreach (var tag in tags)
                queryParams.Add($"tags={Uri.EscapeDataString(tag)}");
        }
        queryParams.Add($"sort={Uri.EscapeDataString(sort)}");
        queryParams.Add($"direction={direction}");
        queryParams.Add($"limit={limit}");
        if (full) queryParams.Add("full=true");

        var url = $"{ApiUrl}/models?{string.Join("&", queryParams)}";
        var json = await GetStringAsync(url);
        return JsonSerializer.Deserialize<HFModelInfo[]>(json, JsonOptions) ?? Array.Empty<HFModelInfo>();
    }

    /// <summary>
    /// Get detailed information about a specific model.
    /// Returns full metadata including file listings, tags, card data, etc.
    /// </summary>
    /// <param name="repoId">Repository ID (e.g., "onnx-community/squeezenet1.1-7")</param>
    public async Task<HFModelInfo> GetModelInfoAsync(string repoId)
    {
        var url = $"{ApiUrl}/models/{repoId}";
        var json = await GetStringAsync(url);
        return JsonSerializer.Deserialize<HFModelInfo>(json, JsonOptions)
            ?? throw new InvalidOperationException($"Failed to parse model info for '{repoId}'");
    }

    /// <summary>
    /// List all files in a model repository.
    /// </summary>
    /// <param name="repoId">Repository ID (e.g., "onnx-community/squeezenet1.1-7")</param>
    /// <param name="revision">Git revision (default: "main")</param>
    /// <param name="path">Subdirectory path to list (e.g., "onnx" to list only files in the onnx/ folder)</param>
    public async Task<HFRepoFile[]> ListRepoFilesAsync(string repoId, string revision = "main", string? path = null)
    {
        var url = $"{ApiUrl}/models/{repoId}/tree/{revision}";
        if (!string.IsNullOrEmpty(path)) url += $"/{path}";
        var json = await GetStringAsync(url);
        return JsonSerializer.Deserialize<HFRepoFile[]>(json, JsonOptions) ?? Array.Empty<HFRepoFile>();
    }

    // ═══════════════════════════════════════════════════════════
    //  Download
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Get the direct CDN download URL for a file in a HuggingFace repository.
    /// This URL can be passed directly to <see cref="InferenceSession.CreateFromFileAsync"/>.
    /// </summary>
    /// <param name="repoId">Repository ID (e.g., "onnx-community/squeezenet1.1-7")</param>
    /// <param name="filename">File path within the repo (e.g., "model.onnx" or "onnx/model.onnx")</param>
    /// <param name="revision">Git revision (default: "main")</param>
    /// <param name="baseUrl">HuggingFace base URL (default: "https://huggingface.co")</param>
    public static string GetDownloadUrl(string repoId, string filename, string revision = "main", string baseUrl = "https://huggingface.co")
    {
        return $"{baseUrl}/{repoId}/resolve/{revision}/{filename}";
    }

    /// <summary>
    /// Download a file from a HuggingFace repository.
    /// For cached downloads, use <see cref="ModelHub.LoadAsync"/> instead.
    /// </summary>
    /// <param name="repoId">Repository ID</param>
    /// <param name="filename">File path within the repo</param>
    /// <param name="revision">Git revision (default: "main")</param>
    /// <param name="onProgress">Progress callback: (bytesReceived, totalBytes). totalBytes is -1 if unknown.</param>
    public async Task<byte[]> DownloadFileAsync(string repoId, string filename, string revision = "main",
        Action<long, long>? onProgress = null)
    {
        var url = GetDownloadUrl(repoId, filename, revision, BaseUrl);
        return await DownloadAsync(url, onProgress);
    }

    /// <summary>
    /// Download a file from any URL with progress reporting.
    /// </summary>
    public async Task<byte[]> DownloadAsync(string url, Action<long, long>? onProgress = null)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        // Only add auth header when token is set — avoids CORS preflight in browsers
        if (!string.IsNullOrEmpty(AuthToken))
            request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", AuthToken);

        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();

        var totalBytes = response.Content.Headers.ContentLength ?? -1;
        using var stream = await response.Content.ReadAsStreamAsync();

        var chunks = new List<byte[]>();
        var buffer = new byte[81920];
        long received = 0;
        int bytesRead;

        while ((bytesRead = await stream.ReadAsync(buffer, 0, buffer.Length)) > 0)
        {
            var chunk = new byte[bytesRead];
            Array.Copy(buffer, chunk, bytesRead);
            chunks.Add(chunk);
            received += bytesRead;
            onProgress?.Invoke(received, totalBytes);
        }

        var output = new byte[received];
        int offset = 0;
        foreach (var chunk in chunks)
        {
            Array.Copy(chunk, 0, output, offset, chunk.Length);
            offset += chunk.Length;
        }
        return output;
    }

    // ═══════════════════════════════════════════════════════════
    //  Helpers
    // ═══════════════════════════════════════════════════════════

    private async Task<string> GetStringAsync(string url)
    {
        if (!string.IsNullOrEmpty(AuthToken))
        {
            // Authenticated request — custom headers are fine (server-side / desktop)
            using var request = new HttpRequestMessage(HttpMethod.Get, url);
            request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", AuthToken);
            request.Headers.Accept.ParseAdd("application/json");
            using var response = await _http.SendAsync(request);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStringAsync();
        }
        else
        {
            // Unauthenticated — use simple GET to avoid CORS preflight in browsers.
            // Custom headers (Accept, Authorization) trigger preflight OPTIONS requests
            // that HuggingFace API rejects with 401 from browser origins.
            return await _http.GetStringAsync(url);
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  Well-Known Pipeline Tags (HuggingFace task taxonomy)
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Common HuggingFace pipeline tags for filtering model searches.
    /// </summary>
    public static class Tasks
    {
        // Computer Vision
        public const string ImageClassification = "image-classification";
        public const string ObjectDetection = "object-detection";
        public const string ImageSegmentation = "image-segmentation";
        public const string DepthEstimation = "depth-estimation";
        public const string ImageToImage = "image-to-image";

        // NLP
        public const string TextClassification = "text-classification";
        public const string TextGeneration = "text-generation";
        public const string TokenClassification = "token-classification";
        public const string QuestionAnswering = "question-answering";
        public const string Translation = "translation";
        public const string Summarization = "summarization";
        public const string FillMask = "fill-mask";
        public const string FeatureExtraction = "feature-extraction";

        // Audio
        public const string AutomaticSpeechRecognition = "automatic-speech-recognition";
        public const string AudioClassification = "audio-classification";
        public const string TextToSpeech = "text-to-speech";

        // Multimodal
        public const string ZeroShotImageClassification = "zero-shot-image-classification";
        public const string ImageTextToText = "image-text-to-text";
    }

    /// <summary>
    /// Common library/framework tags for filtering model searches.
    /// </summary>
    public static class Libraries
    {
        public const string ONNX = "onnx";
        public const string Transformers = "transformers";
        public const string GGUF = "gguf";
        public const string TensorFlow = "tensorflow";
        public const string PyTorch = "pytorch";
        public const string SafeTensors = "safetensors";
    }
}

// ═══════════════════════════════════════════════════════════
//  Response Models
// ═══════════════════════════════════════════════════════════

/// <summary>
/// Model information returned by the HuggingFace Hub API.
/// </summary>
public class HFModelInfo
{
    /// <summary>Full repository ID (e.g., "onnx-community/squeezenet1.1-7")</summary>
    [JsonPropertyName("id")]
    public string Id { get; set; } = "";

    /// <summary>Same as Id for models.</summary>
    [JsonPropertyName("modelId")]
    public string? ModelId { get; set; }

    /// <summary>Author or organization (e.g., "onnx-community")</summary>
    [JsonPropertyName("author")]
    public string? Author { get; set; }

    /// <summary>Latest commit SHA</summary>
    [JsonPropertyName("sha")]
    public string? Sha { get; set; }

    /// <summary>Last modification timestamp</summary>
    [JsonPropertyName("lastModified")]
    public string? LastModified { get; set; }

    /// <summary>Whether this is a private repository</summary>
    [JsonPropertyName("private")]
    public bool Private { get; set; }

    /// <summary>Whether this is a gated model (requires agreement)</summary>
    [JsonPropertyName("gated")]
    public object? Gated { get; set; }

    /// <summary>Primary task/pipeline tag (e.g., "image-classification")</summary>
    [JsonPropertyName("pipeline_tag")]
    public string? PipelineTag { get; set; }

    /// <summary>All tags (task, library, dataset, license, etc.)</summary>
    [JsonPropertyName("tags")]
    public string[]? Tags { get; set; }

    /// <summary>Total download count</summary>
    [JsonPropertyName("downloads")]
    public long Downloads { get; set; }

    /// <summary>Downloads in the last 30 days</summary>
    [JsonPropertyName("downloadsLastMonth")]
    public long DownloadsLastMonth { get; set; }

    /// <summary>Like count</summary>
    [JsonPropertyName("likes")]
    public int Likes { get; set; }

    /// <summary>Primary library/framework (e.g., "onnx", "transformers")</summary>
    [JsonPropertyName("library_name")]
    public string? LibraryName { get; set; }

    /// <summary>Creation timestamp</summary>
    [JsonPropertyName("createdAt")]
    public string? CreatedAt { get; set; }

    /// <summary>
    /// File listings (only populated for detail queries or when full=true in search).
    /// Each entry has rfilename (relative path) and optional size/LFS info.
    /// </summary>
    [JsonPropertyName("siblings")]
    public HFSibling[]? Siblings { get; set; }

    /// <summary>Model card content (markdown)</summary>
    [JsonPropertyName("cardData")]
    public JsonElement? CardData { get; set; }

    /// <summary>Model configuration</summary>
    [JsonPropertyName("config")]
    public JsonElement? Config { get; set; }

    /// <summary>Whether this model is disabled</summary>
    [JsonPropertyName("disabled")]
    public bool Disabled { get; set; }

    /// <summary>Check if this model has ONNX files available.</summary>
    public bool HasOnnx => Tags?.Contains("onnx") == true
        || Siblings?.Any(s => s.Filename?.EndsWith(".onnx") == true) == true;

    /// <summary>Check if this model has GGUF files available.</summary>
    public bool HasGGUF => Tags?.Contains("gguf") == true
        || Siblings?.Any(s => s.Filename?.EndsWith(".gguf") == true) == true;

    /// <summary>Get all ONNX files in this model's repository.</summary>
    public HFSibling[] GetOnnxFiles() =>
        Siblings?.Where(s => s.Filename?.EndsWith(".onnx") == true).ToArray() ?? Array.Empty<HFSibling>();

    /// <summary>Get all GGUF files in this model's repository.</summary>
    public HFSibling[] GetGGUFFiles() =>
        Siblings?.Where(s => s.Filename?.EndsWith(".gguf") == true).ToArray() ?? Array.Empty<HFSibling>();

    public override string ToString() => $"{Id} ({PipelineTag ?? "unknown"}, {Downloads:N0} downloads)";
}

/// <summary>
/// File entry in a model's sibling list (returned with model detail queries).
/// </summary>
public class HFSibling
{
    /// <summary>Relative file path (e.g., "model.onnx", "onnx/model_quantized.onnx")</summary>
    [JsonPropertyName("rfilename")]
    public string? Filename { get; set; }

    /// <summary>File size in bytes (may be null for non-LFS files in search results)</summary>
    [JsonPropertyName("size")]
    public long? Size { get; set; }

    /// <summary>LFS metadata (pointer info) if the file is stored in Git LFS</summary>
    [JsonPropertyName("lfs")]
    public HFLfsInfo? Lfs { get; set; }

    /// <summary>
    /// Effective file size — uses LFS size if available, falls back to Size.
    /// </summary>
    public long EffectiveSize => Lfs?.Size ?? Size ?? 0;

    /// <summary>Format file size as a human-readable string.</summary>
    public string SizeFormatted
    {
        get
        {
            var bytes = EffectiveSize;
            if (bytes < 1024) return $"{bytes} B";
            if (bytes < 1024 * 1024) return $"{bytes / 1024.0:F1} KB";
            if (bytes < 1024 * 1024 * 1024) return $"{bytes / (1024.0 * 1024):F1} MB";
            return $"{bytes / (1024.0 * 1024 * 1024):F2} GB";
        }
    }

    public override string ToString() => $"{Filename} ({SizeFormatted})";
}

/// <summary>
/// Git LFS metadata for a file.
/// </summary>
public class HFLfsInfo
{
    [JsonPropertyName("size")]
    public long Size { get; set; }

    [JsonPropertyName("sha256")]
    public string? Sha256 { get; set; }

    [JsonPropertyName("pointerSize")]
    public int PointerSize { get; set; }
}

/// <summary>
/// File entry returned by the tree/files listing API.
/// </summary>
public class HFRepoFile
{
    /// <summary>"file" or "directory"</summary>
    [JsonPropertyName("type")]
    public string Type { get; set; } = "";

    /// <summary>Git object ID</summary>
    [JsonPropertyName("oid")]
    public string? Oid { get; set; }

    /// <summary>File size in bytes</summary>
    [JsonPropertyName("size")]
    public long Size { get; set; }

    /// <summary>File path relative to repo root (e.g., "onnx/model.onnx")</summary>
    [JsonPropertyName("path")]
    public string Path { get; set; } = "";

    /// <summary>LFS metadata if stored in Git LFS</summary>
    [JsonPropertyName("lfs")]
    public HFLfsInfo? Lfs { get; set; }

    /// <summary>Whether this is a file (not a directory)</summary>
    public bool IsFile => Type == "file";

    /// <summary>Whether this is a directory</summary>
    public bool IsDirectory => Type == "directory";

    /// <summary>Effective file size — uses LFS size if available.</summary>
    public long EffectiveSize => Lfs?.Size ?? Size;

    public override string ToString() => $"{Path} ({Type}, {EffectiveSize} bytes)";
}
