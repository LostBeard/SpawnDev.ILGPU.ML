using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// Downloads a file over HTTP into an <see cref="IResumableModelStore"/>, resuming an interrupted transfer
/// and never letting the payload touch the .NET/WASM managed heap.
/// </summary>
/// <remarks>
/// <para>
/// This is the TRANSPORT, and it is deliberately a separate object from the store. It knows about
/// <c>fetch</c>, ranges, ETags and retries; it knows nothing about OPFS. The store knows about durability;
/// it knows nothing about HTTP. Fusing the two is how model delivery ended up welded to one transport in
/// the first place - a torrent or file-system downloader drops in against the same store.
/// </para>
/// <para>
/// 🔴 The payload goes <c>fetch</c> -&gt; <see cref="Uint8Array"/> -&gt; store, and only a chunk's
/// <c>length</c> (a number) is ever read into .NET. Chunks are coalesced into one JS-side staging buffer
/// before each store write, because <c>fetch</c> delivers ~20 KB chunks while OPFS wants far larger ones
/// (MEASURED: 64 KiB reads ~85 MB/s, 16 MiB ~1560-1990 MB/s).
/// </para>
/// </remarks>
public class HttpModelDownloader
{
    /// <summary>Checkpoint the resume point every this many bytes, so an interrupted transfer resumes near where it stopped.</summary>
    private const long CheckpointInterval = 8L * 1024 * 1024;

    private readonly SpawnJSRuntime _js;
    private readonly IResumableModelStore _store;
    private readonly Dictionary<string, SemaphoreSlim> _keyGates = new();

    /// <summary>
    /// Coalesce incoming <c>fetch</c> chunks into a JS-side buffer of this size before writing to the
    /// store. 0 writes each chunk straight through. Default 16 MiB - see the class remarks.
    /// </summary>
    public int WriteBufferSize { get; set; } = 16 * 1024 * 1024;

    /// <summary>Minimum gap between progress reports. The final report is always sent.</summary>
    public int ProgressIntervalMs { get; set; } = 100;

    /// <summary>Chunks delivered by <c>fetch</c> during the most recent download (diagnostic).</summary>
    public long LastDownloadChunks { get; private set; }

    /// <summary>Store writes issued during the most recent download (diagnostic).</summary>
    public long LastDownloadWrites { get; private set; }

    /// <summary>Bytes transferred during the most recent download, excluding anything carried by a resume.</summary>
    public long LastDownloadBytes { get; private set; }

    /// <summary>Create a downloader that fills <paramref name="store"/>.</summary>
    public HttpModelDownloader(SpawnJSRuntime js, IResumableModelStore store)
    {
        _js = js ?? throw new ArgumentNullException(nameof(js));
        _store = store ?? throw new ArgumentNullException(nameof(store));
    }

    /// <summary>
    /// Return a readable stream for <paramref name="key"/>, downloading from <paramref name="url"/> only if
    /// the store does not already hold it complete. Concurrent calls for the same key do not race: the
    /// second waits and is then served from the store rather than downloading again.
    /// </summary>
    public async Task<Stream> GetOrDownloadAsync(string url, string key,
        IProgress<ModelDownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        var gate = GetGate(key);
        await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var state = await _store.GetStateAsync(key, cancellationToken).ConfigureAwait(false);

            // Hit: complete, and either from this URL or from nowhere. An empty SourceRef means the entry
            // was handed to the store directly (a torrent stream, a picked file); re-downloading it over
            // HTTP because we cannot prove its origin would defeat the point of storing it.
            if (state.Complete && (state.SourceRef == url || string.IsNullOrEmpty(state.SourceRef)))
            {
                progress?.Report(new ModelDownloadProgress(state.BytesWritten, state.BytesWritten, false));
                return await OpenOrThrowAsync(key, cancellationToken).ConfigureAwait(false);
            }

            // Resume only a partial that was for THIS url.
            long resumeFrom = !state.Complete && state.SourceRef == url ? state.BytesWritten : 0;

            await DownloadAsync(url, key, resumeFrom, state.ETag, progress, cancellationToken).ConfigureAwait(false);
            return await OpenOrThrowAsync(key, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            gate.Release();
        }
    }

    private async Task<Stream> OpenOrThrowAsync(string key, CancellationToken ct)
        => await _store.OpenReadAsync(key, ct).ConfigureAwait(false)
           ?? throw new IOException($"Store has no readable entry for '{key}' after a successful download.");

    private async Task DownloadAsync(string url, string key, long resumeFrom, string? knownETag,
        IProgress<ModelDownloadProgress>? progress, CancellationToken ct)
    {
        var headers = new Dictionary<string, string>();
        if (resumeFrom > 0)
        {
            headers["Range"] = $"bytes={resumeFrom}-";
            // If the origin's copy changed since our partial was written it answers 200 with the whole body
            // instead of 206, and we start over rather than splicing two different files together.
            if (!string.IsNullOrEmpty(knownETag)) headers["If-Range"] = knownETag!;
        }

        // 🔴 Cancellation must reach the NETWORK, not just this loop. Checking the token between chunks
        // leaves the HTTP request itself running: the browser keeps pulling the body, the connection and
        // bandwidth stay committed, and a cancel during a multi-GB model frees nothing. An AbortController
        // wired to the token aborts the fetch itself, and also unblocks a `reader.Read()` that is parked on
        // a stalled origin - which a token check between chunks can never do, because it never gets to run.
        using var abort = new AbortController();
        using var abortSignal = abort.Signal;
        using var abortReg = ct.Register(() => { try { abort.Abort(); } catch { /* already gone */ } });

        // Fetch from the RUNTIME, not from `window`: there is no `window` in a worker, and a model SHOULD be
        // loadable from a worker. SpawnJSRuntime.Fetch calls fetch() on whatever the global scope is.
        Response response;
        try
        {
            response = await _js.Fetch(url, new FetchOptions
            {
                Headers = headers.Count > 0 ? headers : null,
                Signal = abortSignal,
            }).ConfigureAwait(false);
        }
        catch (Exception ex) when (ct.IsCancellationRequested)
        {
            // An aborted fetch surfaces as a JS AbortError. Callers cancel with a token and expect the
            // token's exception, so translate rather than leaking the transport's error type.
            throw new OperationCanceledException($"Model download cancelled: {url}", ex, ct);
        }
        using var _response = response;

        // fetch() does NOT throw on 404/500 - it resolves with Ok=false and an ERROR BODY. Nothing is opened
        // or written before this check, so a failed response can never reach the store.
        if (!response.Ok)
            throw new HttpRequestException(
                $"Model download failed: {(int)response.Status} {response.StatusText} for {url}");

        var status = (int)response.Status;
        var resumed = resumeFrom > 0 && status == 206;
        if (resumeFrom > 0 && status != 206)
            resumeFrom = 0;   // origin ignored the Range, or If-Range failed: take the full body

        long total;
        string? etag;
        using (var respHeaders = response.Headers)
        {
            total = ParseTotalBytes(respHeaders, resumeFrom);
            etag = NullIfEmpty(respHeaders.Get("etag"));
        }

        await _store.SetStateAsync(key, url, total, resumeFrom, false, etag, ct).ConfigureAwait(false);

        long received = resumeFrom;   // pulled off the network
        long written = resumeFrom;    // durable in the store (lags `received` while buffering)

        // Report ONCE up front, as soon as the total is known and before a single byte is read. Without
        // this a caller cannot render a progress bar at all until the first interval elapses - and for a
        // file that finishes faster than ProgressIntervalMs the ONLY report was the final 100%, which is
        // indistinguishable from no progress reporting at all. It also hands the UI the total immediately.
        // Report ONCE up front, as soon as the total is known and before a single byte is read. Without
        // this a caller cannot render a progress bar at all until the first interval elapses - and for a
        // file that finishes faster than ProgressIntervalMs the ONLY report was the final 100%, which is
        // indistinguishable from no progress reporting at all. It also hands the UI the total immediately.
        progress?.Report(new ModelDownloadProgress(received, total, resumed));

        // Deliberately NOT `TickCount64`: seeding with "now" suppressed the first in-loop report for a
        // whole interval. Seeding a full interval in the past means the next chunk reports immediately, so
        // a short download still produces real intermediate progress.
        var lastReport = Environment.TickCount64 - ProgressIntervalMs;
        var lastCheckpoint = written;
        long chunkCount = 0, writeCount = 0;
        Uint8Array? writeBuffer = null;
        long bufferFill = 0;

        var dest = await _store.OpenWriteAsync(key, resumeFrom, ct).ConfigureAwait(false);
        await using (dest.ConfigureAwait(false))
        {
            using var body = response.Body ?? throw new InvalidOperationException($"Response for {url} had no body.");
            using var reader = body.GetReader();
            try
            {
                while (true)
                {
                    if (ct.IsCancellationRequested)
                    {
                        await reader.Cancel().ConfigureAwait(false);
                        ct.ThrowIfCancellationRequested();
                    }

                    using var result = await reader.Read().ConfigureAwait(false);
                    if (result.Done) break;

                    // 🔴 The chunk stays a JS Uint8Array from here to the store. Never ReadBytes() it.
                    using var chunk = result.Value;
                    if (chunk == null) continue;
                    var chunkLength = chunk.Length;
                    if (chunkLength == 0) continue;
                    chunkCount++;

                    if (WriteBufferSize <= 0 || chunkLength >= WriteBufferSize)
                    {
                        await FlushAsync().ConfigureAwait(false);   // keep byte order
                        await WriteChunkAsync(dest, chunk, ct).ConfigureAwait(false);
                        writeCount++;
                        written += chunkLength;
                    }
                    else
                    {
                        if (bufferFill + chunkLength > WriteBufferSize) await FlushAsync().ConfigureAwait(false);
                        // TypedArray.set: a JS-side copy into the staging buffer, no managed array involved.
                        writeBuffer ??= new Uint8Array(WriteBufferSize);
                        writeBuffer.Set(chunk, bufferFill);
                        bufferFill += chunkLength;
                    }
                    received += chunkLength;

                    // Checkpoint only what is DURABLE - `written`, never `received`. A resume trusts this.
                    if (written - lastCheckpoint >= CheckpointInterval)
                    {
                        await dest.FlushAsync(ct).ConfigureAwait(false);
                        await _store.SetStateAsync(key, url, total, written, false, etag, ct).ConfigureAwait(false);
                        lastCheckpoint = written;
                    }

                    var now = Environment.TickCount64;
                    if (progress != null && now - lastReport >= ProgressIntervalMs)
                    {
                        progress.Report(new ModelDownloadProgress(received, total, resumed));
                        lastReport = now;
                    }
                }

                await FlushAsync().ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                // Record only what actually landed. The staging buffer is deliberately NOT flushed: a
                // failure mid-fill leaves its tail indeterminate, and writing it would corrupt the resume.
                // CancellationToken.None on purpose - this bookkeeping is exactly what a CANCELLED download
                // needs, so passing the cancelled token would skip it and throw away the resume point.
                try
                {
                    await dest.FlushAsync(CancellationToken.None).ConfigureAwait(false);
                    await _store.SetStateAsync(key, url, total, written, false, etag, CancellationToken.None)
                        .ConfigureAwait(false);
                }
                catch { /* the original failure is the one worth reporting */ }

                // An aborted read surfaces as a JS AbortError; callers cancelled with a token and expect
                // the token's exception.
                if (ct.IsCancellationRequested && ex is not OperationCanceledException)
                    throw new OperationCanceledException($"Model download cancelled: {url}", ex, ct);
                throw;
            }
            finally
            {
                writeBuffer?.Dispose();
            }

            await dest.FlushAsync(ct).ConfigureAwait(false);

            async Task FlushAsync()
            {
                if (bufferFill == 0 || writeBuffer == null) return;
                if (bufferFill == WriteBufferSize)
                {
                    await WriteChunkAsync(dest, writeBuffer, ct).ConfigureAwait(false);
                }
                else
                {
                    // SubArray is a VIEW onto the same storage - writes the filled prefix, copies nothing.
                    using var view = writeBuffer.SubArray(0, bufferFill);
                    await WriteChunkAsync(dest, view, ct).ConfigureAwait(false);
                }
                writeCount++;
                written += bufferFill;
                bufferFill = 0;
            }
        }

        LastDownloadChunks = chunkCount;
        LastDownloadWrites = writeCount;
        LastDownloadBytes = received - resumeFrom;

        // A short read is a truncated download, not a model. Leave the entry incomplete so the next attempt
        // resumes it, rather than caching it and failing much later inside a proto reader.
        if (total > 0 && written != total)
        {
            await _store.SetStateAsync(key, url, total, written, false, etag, ct).ConfigureAwait(false);
            throw new IOException(
                $"Model download truncated for {url}: got {written} of {total} bytes. " +
                "The partial entry was kept and the next attempt will resume it.");
        }

        await _store.SetStateAsync(key, url, total > 0 ? total : written, written, true, etag, ct).ConfigureAwait(false);
        progress?.Report(new ModelDownloadProgress(received, total > 0 ? total : written, resumed));
    }

    /// <summary>
    /// Write a JS-side chunk to the store's stream without a managed copy when the stream supports it.
    /// </summary>
    /// <remarks>An <c>IJSWriteStream</c> takes the <see cref="Uint8Array"/> directly. Anything else is a
    /// non-JS store (a desktop file store), where a managed copy is the only option and is correct.</remarks>
    private static async Task WriteChunkAsync(Stream dest, Uint8Array chunk, CancellationToken ct)
    {
        if (dest is SpawnDev.SpawnJS.Toolbox.IJSWriteStream js)
        {
            await js.WriteUint8ArrayAsync(chunk, ct).ConfigureAwait(false);
            return;
        }
        await dest.WriteAsync(chunk.ReadBytes(), ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Total file size from the response headers. For a 206 that is the total after the '/' in
    /// <c>Content-Range: bytes A-B/TOTAL</c>; a 206's Content-Length is the PART length, never the file size.
    /// </summary>
    private static long ParseTotalBytes(Headers headers, long resumeFrom)
    {
        var contentRange = headers.Get("content-range");
        if (!string.IsNullOrEmpty(contentRange))
        {
            var slash = contentRange.LastIndexOf('/');
            if (slash >= 0 && long.TryParse(contentRange[(slash + 1)..].Trim(), out var fromRange))
                return fromRange;
        }
        var contentLength = headers.Get("content-length");
        if (!string.IsNullOrEmpty(contentLength) && long.TryParse(contentLength, out var len))
            return resumeFrom + len;
        return -1;
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

    private static string? NullIfEmpty(string? value) => string.IsNullOrEmpty(value) ? null : value;
}
