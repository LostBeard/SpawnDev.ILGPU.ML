namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// Serves models over HTTP from the SpawnDev hub into an <see cref="IResumableModelStore"/>, using
/// <see cref="HttpClient"/> - the desktop counterpart to <see cref="HubModelSource"/>.
/// </summary>
/// <remarks>
/// <para>
/// Same job, same store contract, different transport: <see cref="HubModelSource"/> uses the browser's
/// <c>fetch</c> so the payload can stay in JS, which is meaningless on desktop. Pair this with
/// <see cref="FileModelStore"/> and a console app gets resumable, cancellable, cached model delivery with
/// no WebTorrent and no browser.
/// </para>
/// <para>
/// ⚠️ Always the hub, never huggingface.co directly - it caches, and it keeps us out of HuggingFace's rate
/// limiter.
/// </para>
/// </remarks>
public class HttpClientModelSource : IModelSource
{
    /// <summary>Checkpoint the resume point every this many bytes.</summary>
    private const long CheckpointInterval = 8L * 1024 * 1024;

    private readonly HttpClient _http;
    private readonly IResumableModelStore _store;
    private readonly Dictionary<string, SemaphoreSlim> _keyGates = new();

    /// <summary>Copy buffer / write size. 16 MiB matches ILGPU's stream chunk default.</summary>
    public int BufferSize { get; set; } = 16 * 1024 * 1024;

    /// <summary>Minimum gap between progress reports. The final report is always sent.</summary>
    public int ProgressIntervalMs { get; set; } = 100;

    /// <summary>Git revision requested from the hub.</summary>
    public string Revision { get; set; } = "main";

    /// <summary>The store being filled, for listing, sizing and eviction.</summary>
    public IModelStore Store => _store;

    /// <summary>Create a source over <paramref name="http"/> filling <paramref name="store"/>.</summary>
    public HttpClientModelSource(HttpClient http, IResumableModelStore store)
    {
        _http = http ?? throw new ArgumentNullException(nameof(http));
        _store = store ?? throw new ArgumentNullException(nameof(store));
    }

    /// <summary>Create a source over a <see cref="FileModelStore"/> at the default per-user location.</summary>
    public HttpClientModelSource(HttpClient http) : this(http, FileModelStore.Default()) { }

    /// <inheritdoc/>
    public Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
        => OpenAsync(repoId, filePath, null, cancellationToken);

    /// <summary>Open a model file, reporting progress for this call.</summary>
    public async Task<Stream> OpenAsync(string repoId, string filePath,
        IProgress<ModelDownloadProgress>? progress, CancellationToken cancellationToken = default)
    {
        var url = HuggingFaceClient.GetDownloadUrl(repoId, filePath, Revision);
        var key = CacheKey(repoId, filePath);

        var gate = GetGate(key);
        await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var state = await _store.GetStateAsync(key, cancellationToken).ConfigureAwait(false);
            if (state.Complete && (state.SourceRef == url || string.IsNullOrEmpty(state.SourceRef)))
            {
                progress?.Report(new ModelDownloadProgress(state.BytesWritten, state.BytesWritten, false));
                return await OpenOrThrowAsync(key, cancellationToken).ConfigureAwait(false);
            }

            long resumeFrom = !state.Complete && state.SourceRef == url ? state.BytesWritten : 0;
            await DownloadAsync(url, key, resumeFrom, state.ETag, progress, cancellationToken).ConfigureAwait(false);
            return await OpenOrThrowAsync(key, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            gate.Release();
        }
    }

    /// <inheritdoc/>
    public async Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
    {
        var stream = await OpenAsync(repoId, filePath, cancellationToken).ConfigureAwait(false);
        await using (stream.ConfigureAwait(false))
        {
            using var ms = new MemoryStream();
            await stream.CopyToAsync(ms, cancellationToken).ConfigureAwait(false);
            return ms.ToArray();
        }
    }

    /// <summary>The store key for a repo file. Stable across runs, and distinct per revision.</summary>
    public string CacheKey(string repoId, string filePath)
        => $"hf_{repoId.Replace('/', '_')}_{Revision}_{filePath.Replace('/', '_')}";

    private async Task<Stream> OpenOrThrowAsync(string key, CancellationToken ct)
        => await _store.OpenReadAsync(key, ct).ConfigureAwait(false)
           ?? throw new IOException($"Store has no readable entry for '{key}' after a successful download.");

    private async Task DownloadAsync(string url, string key, long resumeFrom, string? knownETag,
        IProgress<ModelDownloadProgress>? progress, CancellationToken ct)
    {
        using var req = new HttpRequestMessage(HttpMethod.Get, url);
        if (resumeFrom > 0)
        {
            req.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(resumeFrom, null);
            // If the origin's copy changed it answers 200 with the whole body instead of 206, and we start
            // over rather than splicing two different files together.
            if (!string.IsNullOrEmpty(knownETag))
                req.Headers.TryAddWithoutValidation("If-Range", knownETag);
        }

        // ResponseHeadersRead: do not buffer the body. Without it HttpClient reads the WHOLE model into
        // memory before this method sees a single byte, which defeats streaming entirely.
        using var res = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
        if (!res.IsSuccessStatusCode)
            throw new HttpRequestException(
                $"Model download failed: {(int)res.StatusCode} {res.ReasonPhrase} for {url}");

        var resumed = resumeFrom > 0 && res.StatusCode == System.Net.HttpStatusCode.PartialContent;
        if (resumeFrom > 0 && !resumed) resumeFrom = 0;   // origin ignored Range, or If-Range failed

        long total = res.Content.Headers.ContentRange?.Length
                     ?? (res.Content.Headers.ContentLength is { } len ? resumeFrom + len : -1);
        var etag = res.Headers.ETag?.Tag;

        await _store.SetStateAsync(key, url, total, resumeFrom, false, etag, ct).ConfigureAwait(false);

        long written = resumeFrom;
        var lastReport = Environment.TickCount64 - ProgressIntervalMs;
        var lastCheckpoint = written;
        progress?.Report(new ModelDownloadProgress(written, total, resumed));

        var body = await res.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        await using (body.ConfigureAwait(false))
        {
            var dest = await _store.OpenWriteAsync(key, resumeFrom, ct).ConfigureAwait(false);
            await using (dest.ConfigureAwait(false))
            {
                var buffer = new byte[BufferSize];
                try
                {
                    while (true)
                    {
                        int read = await body.ReadAsync(buffer, ct).ConfigureAwait(false);
                        if (read == 0) break;
                        await dest.WriteAsync(buffer.AsMemory(0, read), ct).ConfigureAwait(false);
                        written += read;

                        if (written - lastCheckpoint >= CheckpointInterval)
                        {
                            await dest.FlushAsync(ct).ConfigureAwait(false);
                            await _store.SetStateAsync(key, url, total, written, false, etag, ct).ConfigureAwait(false);
                            lastCheckpoint = written;
                        }

                        var now = Environment.TickCount64;
                        if (now - lastReport >= ProgressIntervalMs)
                        {
                            progress?.Report(new ModelDownloadProgress(written, total, resumed));
                            lastReport = now;
                        }
                    }
                }
                catch
                {
                    // Record what landed so the next attempt resumes. CancellationToken.None on purpose:
                    // this bookkeeping is exactly what a CANCELLED download needs, and the cancelled token
                    // would skip it and throw away the resume point.
                    try
                    {
                        await dest.FlushAsync(CancellationToken.None).ConfigureAwait(false);
                        await _store.SetStateAsync(key, url, total, written, false, etag, CancellationToken.None)
                            .ConfigureAwait(false);
                    }
                    catch { /* the original failure is the one worth reporting */ }
                    throw;
                }

                await dest.FlushAsync(ct).ConfigureAwait(false);
            }
        }

        // A short read is a truncated download, not a model.
        if (total > 0 && written != total)
        {
            await _store.SetStateAsync(key, url, total, written, false, etag, ct).ConfigureAwait(false);
            throw new IOException(
                $"Model download truncated for {url}: got {written} of {total} bytes. " +
                "The partial entry was kept and the next attempt will resume it.");
        }

        await _store.SetStateAsync(key, url, total > 0 ? total : written, written, true, etag, ct).ConfigureAwait(false);
        progress?.Report(new ModelDownloadProgress(written, total > 0 ? total : written, resumed));
    }

    private SemaphoreSlim GetGate(string key)
    {
        lock (_keyGates)
        {
            if (!_keyGates.TryGetValue(key, out var gate))
                _keyGates[key] = gate = new SemaphoreSlim(1, 1);
            return gate;
        }
    }
}
