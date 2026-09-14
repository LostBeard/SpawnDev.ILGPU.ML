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

    /// <summary>Fired during a download with (bytesReceived, totalBytes); totalBytes is -1 when unknown.</summary>
    public event Action<long, long>? OnProgress;

    /// <summary>Create a source backed by its own OPFS cache.</summary>
    /// <param name="js">The SpawnJS runtime - used for <c>fetch</c> and <c>navigator.storage</c> in any scope.</param>
    public HubModelSource(SpawnJSRuntime js)
    {
        _cache = new OpfsModelCache(js);
        _ownsCache = true;
    }

    /// <summary>Create a source over an existing cache, so several sources share one OPFS store.</summary>
    public HubModelSource(OpfsModelCache cache)
    {
        _cache = cache ?? throw new ArgumentNullException(nameof(cache));
        _ownsCache = false;
    }

    /// <summary>The OPFS cache backing this source, for listing, sizing and eviction.</summary>
    public IModelStore Store => _cache;

    /// <summary>Git revision requested from the hub. The hub's web seed serves the default revision.</summary>
    public string Revision { get; set; } = "main";

    /// <inheritdoc/>
    public async Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
    {
        var url = HuggingFaceClient.GetDownloadUrl(repoId, filePath, Revision);
        var progress = OnProgress == null
            ? null
            : new Progress<ModelDownloadProgress>(p => OnProgress?.Invoke(p.BytesReceived, p.TotalBytes));
        return await _cache.OpenOrDownloadAsync(url, CacheKey(repoId, filePath), progress, cancellationToken)
            .ConfigureAwait(false);
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

    /// <summary>The OPFS cache key for a repo file. Stable across runs, and distinct per revision.</summary>
    public string CacheKey(string repoId, string filePath)
        => $"hf_{repoId.Replace('/', '_')}_{Revision}_{filePath.Replace('/', '_')}";

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_ownsCache) _cache.Dispose();
        GC.SuppressFinalize(this);
    }
}
