using SpawnDev.SpawnJS;

namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// Serves models over plain HTTP from the SpawnDev hub, cached in OPFS - <b>no WebTorrent</b>.
/// </summary>
/// <remarks>
/// <para>
/// The default <see cref="IModelSource"/>. A model is fetched once through the hub's <c>/hf</c> web seed
/// straight into OPFS (see <see cref="OpfsModelCache"/>: the payload never enters the .NET/WASM managed
/// heap), and every later open is served from that cache with no network at all. Interrupted downloads
/// resume; truncated ones are never served.
/// </para>
/// <para>
/// ⚠️ Always the hub, never huggingface.co directly. The hub caches, supplies the CORS headers a browser
/// will accept, and keeps us out of HuggingFace's rate limiter.
/// </para>
/// <code>
/// var source = new HubModelSource(js);
/// var pipe = await DepthEstimationPipeline.CreateFromHubAsync(accelerator, source, repoId);
/// </code>
/// </remarks>
public class HubModelSource : IModelSource, IDisposable
{
    private readonly OpfsModelCache _cache;
    private readonly bool _ownsCache;
    private readonly HttpClient? _http;

    /// <summary>Fired during a download with (bytesReceived, totalBytes); totalBytes is -1 when unknown.</summary>
    public event Action<long, long>? OnProgress;

    /// <summary>Create a source backed by its own OPFS cache.</summary>
    /// <param name="js">The SpawnJS runtime - used for <c>fetch</c> and <c>navigator.storage</c> in any scope.</param>
    /// <param name="http">
    /// Optional, and needed ONLY by <see cref="OpenForInspectionAsync"/>. Every other path uses
    /// <c>fetch</c> so the payload stays JS-side; inspection deliberately does not, because it reads a few
    /// hundred KB of structure and must NOT cache the file.
    /// </param>
    public HubModelSource(SpawnJSRuntime js, HttpClient? http = null)
    {
        _cache = new OpfsModelCache(js);
        _ownsCache = true;
        _http = http;
    }

    /// <summary>Create a source over an existing cache, so several sources share one OPFS store.</summary>
    public HubModelSource(OpfsModelCache cache)
    {
        _cache = cache ?? throw new ArgumentNullException(nameof(cache));
        _ownsCache = false;
    }

    /// <summary>The OPFS cache backing this source, for listing, sizing and eviction.</summary>
    public IModelStore Store => _cache;

    /// <summary>
    /// Downloads running right now through this source - for a cache/status page.
    /// </summary>
    /// <remarks>Only meaningful if this source is SHARED (registered in DI). A page that constructs its own
    /// source sees only its own downloads, which is exactly the wrong answer for a cache UI.</remarks>
    public IReadOnlyList<ActiveModelDownload> ActiveDownloads => _cache.Downloader.ActiveDownloads;

    /// <summary>Raised when a download starts, progresses or ends. Marshal to the UI thread yourself.</summary>
    public event Action? ActiveDownloadsChanged
    {
        add => _cache.Downloader.ActiveDownloadsChanged += value;
        remove => _cache.Downloader.ActiveDownloadsChanged -= value;
    }

    /// <summary>Git revision requested from the hub. The hub's web seed serves the default revision.</summary>
    public string Revision { get; set; } = "main";

    /// <summary>
    /// True when the browser storage backing this source is usable (a secure context on a supporting
    /// browser). False means nothing can be cached and every load re-downloads.
    /// </summary>
    /// <remarks>A cache UI needs this to tell "storage unavailable" apart from "nothing cached yet" - the
    /// listing is empty either way, and only one of them is worth warning the user about.</remarks>
    public Task<bool> IsAvailableAsync() => _cache.IsAvailableAsync();

    /// <inheritdoc/>
    public Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
        => OpenAsync(repoId, filePath, null, cancellationToken);

    /// <summary>
    /// Open a model file, reporting progress for THIS call only.
    /// </summary>
    /// <remarks>
    /// ⚠️ Prefer this over the <see cref="OnProgress"/> event when the source is shared (the normal case -
    /// it should be a DI singleton). The event fires for every download through the source, including ones
    /// started by other pages, and a subscriber that outlives the page leaks. A per-call
    /// <see cref="IProgress{T}"/> is scoped to the caller and needs no unsubscribe.
    /// </remarks>
    public async Task<Stream> OpenAsync(string repoId, string filePath,
        IProgress<ModelDownloadProgress>? progress, CancellationToken cancellationToken = default)
    {
        var url = HuggingFaceClient.GetDownloadUrl(repoId, filePath, Revision);
        return await _cache.OpenOrDownloadAsync(url, CacheKey(repoId, filePath), Combine(progress), cancellationToken)
            .ConfigureAwait(false);
    }

    /// <summary>Feed both the per-call sink and the source-wide <see cref="OnProgress"/> event.</summary>
    private IProgress<ModelDownloadProgress>? Combine(IProgress<ModelDownloadProgress>? progress)
    {
        if (OnProgress == null) return progress;
        return new RelayProgress(p =>
        {
            progress?.Report(p);
            OnProgress?.Invoke(p.BytesReceived, p.TotalBytes);
        });
    }

    /// <summary>Synchronous <see cref="IProgress{T}"/> - unlike <see cref="Progress{T}"/> it does not post
    /// to a SynchronizationContext, so reports arrive in order and before the download returns.</summary>
    private sealed class RelayProgress : IProgress<ModelDownloadProgress>
    {
        private readonly Action<ModelDownloadProgress> _onReport;
        public RelayProgress(Action<ModelDownloadProgress> onReport) => _onReport = onReport;
        public void Report(ModelDownloadProgress value) => _onReport(value);
    }

    /// <inheritdoc/>
    public async Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
    {
        // Small file, so the whole-file read is the point - but still go through the cache rather than
        // re-fetching on every call, and still read it from the OPFS stream rather than a second download.
        var stream = await OpenAsync(repoId, filePath, cancellationToken).ConfigureAwait(false);
        await using (stream.ConfigureAwait(false))
        {
            using var ms = new MemoryStream();
            await stream.CopyToAsync(ms, cancellationToken).ConfigureAwait(false);
            return ms.ToArray();
        }
    }

    /// <summary>
    /// Open an OLLAMA model layer from the hub - the twin of <see cref="OpenAsync"/> for models addressed
    /// as <c>model:tag/layer</c> rather than repo/path.
    /// </summary>
    /// <remarks>
    /// Deliberately NOT part of <see cref="IModelSource"/>: Ollama uses a different addressing scheme, and
    /// bending it into repo/path coordinates would be a lie that every implementer then has to honour.
    /// <c>HubModelStream</c> keeps its Ollama entry point outside the interface for the same reason.
    /// <para>
    /// The hub's <c>/ollama</c> web seed resolves the registry manifest and serves the layer blob, so this
    /// is the same download-into-OPFS path as everything else - resumable, cancellable, cached.
    /// </para>
    /// </remarks>
    /// <param name="model">Ollama model name, e.g. <c>gemma4</c>.</param>
    /// <param name="tag">Tag, e.g. <c>12b</c>.</param>
    /// <param name="layer"><c>model</c> (GGUF weights) | <c>projector</c> (mmproj) | <c>params</c> | <c>template</c> | <c>license</c>.</param>
    /// <param name="cancellationToken">Cancels the download; a cancelled one stays resumable.</param>
    public Task<Stream> OpenOllamaAsync(string model, string tag, string layer,
        CancellationToken cancellationToken = default)
        => OpenOllamaAsync(model, tag, layer, null, cancellationToken);

    /// <summary>Open an Ollama layer, reporting progress for THIS call only. See <see cref="OpenAsync"/>.</summary>
    public async Task<Stream> OpenOllamaAsync(string model, string tag, string layer,
        IProgress<ModelDownloadProgress>? progress, CancellationToken cancellationToken = default)
    {
        var url = $"{HuggingFaceClient.HubBaseUrl.TrimEnd('/')}/ollama/{model.Trim('/')}/{tag.Trim('/')}/{layer.Trim('/')}";
        return await _cache.OpenOrDownloadAsync(url, OllamaCacheKey(model, tag, layer), Combine(progress), cancellationToken)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// Open a model for STRUCTURE INSPECTION: a seekable range stream that fetches only the bytes actually
    /// read, and caches nothing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠️ Deliberately NOT <see cref="OpenAsync"/>. That downloads and caches the whole file, which is right
    /// for loading a model and wrong for inspecting one: <c>ModelInspectorHelper</c> seeks past every weight
    /// blob, so inspecting a multi-GB checkpoint should cost a few hundred KB. Routing inspection through
    /// the cache would download gigabytes to read a graph, and fill the cache with models nobody loaded.
    /// </para>
    /// <para>
    /// This is the one path that uses <see cref="HttpClient"/> rather than <c>fetch</c>, so its bytes do
    /// enter the managed heap - acceptable precisely because it reads structure, never weights. Supply an
    /// <c>HttpClient</c> to the constructor to use it.
    /// </para>
    /// <para>
    /// Uses <c>HEAD</c> for the size, falling back to a 0-0 range GET. Both work against the hub as of
    /// 2026-09-14; before that HEAD returned 405 and only the range probe worked.
    /// </para>
    /// </remarks>
    /// <returns>
    /// The concrete <see cref="HttpRangeStream"/>, not just a <see cref="Stream"/>, so a caller can assert
    /// on <see cref="HttpRangeStream.BytesFetched"/>. "Structure only" is a claim about how much crossed the
    /// wire, and a regression that quietly starts reading weight blobs still returns perfectly correct
    /// structure - without that counter the property is untestable.
    /// </returns>
    public async Task<HttpRangeStream> OpenForInspectionAsync(string repoId, string filePath,
        CancellationToken cancellationToken = default)
    {
        if (_http == null)
            throw new InvalidOperationException(
                "OpenForInspectionAsync needs an HttpClient. Pass one to the HubModelSource constructor - " +
                "inspection reads structure over ranged HTTP instead of caching the whole model.");

        var url = HuggingFaceClient.GetDownloadUrl(repoId, filePath, Revision);
        var size = await ProbeSizeAsync(_http, url, cancellationToken).ConfigureAwait(false);
        if (size <= 0)
            throw new IOException($"Could not determine the size of {url}; cannot open a range stream over it.");
        return new HttpRangeStream(_http, url, size);
    }

    /// <summary>Total size of a URL: a 0-0 range GET reading <c>Content-Range</c>, with HEAD as a fallback.</summary>
    /// <remarks>
    /// ⚠️ Range probe FIRST, deliberately. It works against every version of the hub, whereas HEAD only
    /// started working with the 2026-09-14 proxy fix - and an undeployed hub answers HEAD with 405, which
    /// costs a wasted round trip and logs a console error on every inspection. Both are one request, so
    /// there is nothing to gain by trying the narrower one first. HEAD stays as a fallback for an origin
    /// that refuses ranges.
    /// </remarks>
    private static async Task<long> ProbeSizeAsync(HttpClient http, string url, CancellationToken ct)
    {
        try
        {
            using var req = new HttpRequestMessage(HttpMethod.Get, url);
            req.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(0, 0);
            using var res = await http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
            if (res.IsSuccessStatusCode)
            {
                // A 206's Content-Length is the PART (1 byte) - only Content-Range carries the file size.
                var total = res.Content.Headers.ContentRange?.Length;
                if (total is > 0) return total.Value;
                if (res.StatusCode == System.Net.HttpStatusCode.OK && res.Content.Headers.ContentLength is > 0 and var full)
                    return full;
            }
        }
        catch (HttpRequestException) { /* fall through to HEAD */ }

        using var head = new HttpRequestMessage(HttpMethod.Head, url);
        using var headRes = await http.SendAsync(head, ct).ConfigureAwait(false);
        headRes.EnsureSuccessStatusCode();
        return headRes.Content.Headers.ContentLength ?? -1;
    }

    /// <summary>The OPFS cache key for a repo file. Stable across runs, and distinct per revision.</summary>
    public string CacheKey(string repoId, string filePath)
        => $"hf_{repoId.Replace('/', '_')}_{Revision}_{filePath.Replace('/', '_')}";

    /// <summary>The OPFS cache key for an Ollama layer.</summary>
    public static string OllamaCacheKey(string model, string tag, string layer)
        => $"ollama_{model.Replace('/', '_')}_{tag.Replace('/', '_')}_{layer.Replace('/', '_')}";

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_ownsCache) _cache.Dispose();
        GC.SuppressFinalize(this);
    }
}
