using System.Text;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.SpawnJS.Toolbox;

namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// Progress for a model download. <see cref="TotalBytes"/> is -1 when the server did not report a size.
/// </summary>
/// <param name="BytesReceived">Bytes present in the cache entry so far, including bytes carried over by a resume.</param>
/// <param name="TotalBytes">Total size of the file, or -1 when unknown.</param>
/// <param name="Resumed">True when this download continued a previously interrupted one rather than starting at 0.</param>
public readonly record struct ModelDownloadProgress(long BytesReceived, long TotalBytes, bool Resumed)
{
    /// <summary>Completion fraction in [0,1], or null when the total size is unknown.</summary>
    public double? Fraction => TotalBytes > 0 ? Math.Clamp((double)BytesReceived / TotalBytes, 0d, 1d) : null;
}

/// <summary>
/// An OPFS-backed model cache that downloads over plain HTTP and hands back a seekable
/// <see cref="OPFSStream"/> - <b>without the payload ever entering the .NET/WASM managed heap</b>.
/// <para>
/// This is the delivery path for <c>InferenceSession.CreateFromOnnxStreamAsync</c> /
/// <c>CreateFromGGUFStreamAsync</c>, which take a plain seekable <see cref="Stream"/> and detect
/// <c>IJSReadStream</c> for the JS-&gt;GPU zero-copy weight upload. <see cref="OPFSStream"/> is exactly that,
/// and it needs no WebWorkers dependency: it uses a <c>FileSystemSyncAccessHandle</c> automatically when it
/// happens to be running in a dedicated worker, and an async handle everywhere else.
/// </para>
/// <para>
/// ⚠️ <b>Why this class exists instead of <see cref="ModelCache"/>.</b> <c>ModelCache</c> reads each fetch
/// chunk with <c>Uint8Array.ReadBytes()</c>, accumulates the whole file in a <c>List&lt;byte[]&gt;</c>,
/// concatenates that into one more <c>byte[]</c>, and then copies it back to JS to write OPFS - so a 1.7 GB
/// checkpoint lands on the single-threaded WASM managed heap <b>twice</b> before it is cached. That is the
/// reason a torrent had to be the blessed delivery path for weights, and it is a plain bug rather than a
/// property of HTTP. Here the chunk stays a <see cref="Uint8Array"/> from <c>fetch</c> to OPFS and only its
/// <c>length</c> (a number) is ever read into .NET.
/// </para>
/// </summary>
/// <remarks>
/// Durability rules, which are the whole difference between a cache and a liability:
/// <list type="bullet">
/// <item>A non-OK response is never written. <c>fetch</c> resolves (it does not throw) on 404/500 and hands
/// back an <i>error body</i>; caching that is how a 15-byte 404 page once became a permanently-cached
/// "model" that failed identically on every later run.</item>
/// <item>An entry is only servable when its sidecar says <c>complete</c> <b>and</b> the file's real size
/// matches the recorded total. A truncated download is therefore resumed, never served.</item>
/// <item>A resume sends <c>If-Range</c> with the stored ETag. If the server's copy changed it answers 200
/// with the whole body instead of 206, and the entry is restarted from zero rather than being spliced
/// together from two different files.</item>
/// </list>
/// </remarks>
public class OpfsModelCache : IModelStore, IDisposable
{
    /// <summary>Sidecar suffix holding an entry's download state. Kept beside the data file, not inside it.</summary>
    private const string MetaSuffix = ".meta";

    /// <summary>Flush the sidecar every this many bytes so an interrupted download resumes near where it stopped.</summary>
    private const long MetaFlushInterval = 8L * 1024 * 1024;

    private readonly SpawnJSRuntime _js;
    private readonly Dictionary<string, SemaphoreSlim> _keyGates = new();
    private FileSystemDirectoryHandle? _cacheDir;
    private bool _initialized;

    /// <summary>Name of the OPFS subdirectory holding cached models.</summary>
    public string CacheDirectoryName { get; set; } = "ilgpu-ml-models";

    /// <summary>
    /// Minimum gap between <see cref="IProgress{T}"/> reports while downloading. The final report is always
    /// sent regardless. Reporting every chunk is a JS crossing per chunk for no benefit to a progress bar.
    /// </summary>
    public int ProgressIntervalMs { get; set; } = 100;

    /// <summary>
    /// Coalesce incoming <c>fetch</c> chunks into a JS-side buffer of this size and write THAT to OPFS,
    /// instead of writing every chunk as it arrives. Set to 0 to write each chunk straight through.
    /// </summary>
    /// <remarks>
    /// A <c>fetch</c> body arrives in whatever chunks the network and browser choose - typically tens of KB,
    /// far below the size at which OPFS writes are efficient. The buffer is a single
    /// <see cref="Uint8Array"/> allocated once per download and filled with <c>TypedArray.set</c>, so the
    /// coalescing happens entirely in JS and the payload still never touches the managed heap.
    /// <para>
    /// Use <see cref="LastDownloadChunks"/> / <see cref="LastDownloadWrites"/> to see what a given source
    /// actually delivered and how many OPFS writes it cost.
    /// </para>
    /// </remarks>
    public int WriteBufferSize { get; set; } = 4 * 1024 * 1024;

    /// <summary>Chunks delivered by <c>fetch</c> during the most recent download (diagnostic).</summary>
    public long LastDownloadChunks { get; private set; }

    /// <summary>OPFS writes issued during the most recent download (diagnostic).</summary>
    public long LastDownloadWrites { get; private set; }

    /// <summary>Bytes transferred during the most recent download, excluding anything carried over by a resume.</summary>
    public long LastDownloadBytes { get; private set; }

    /// <summary>Create a cache over the OPFS directory named by <see cref="CacheDirectoryName"/>.</summary>
    /// <param name="js">The SpawnJS runtime, used for <c>fetch</c> and <c>navigator.storage</c> in any scope.</param>
    public OpfsModelCache(SpawnJSRuntime js) => _js = js ?? throw new ArgumentNullException(nameof(js));

    /// <summary>True when OPFS is usable here (a secure context on a browser that supports it).</summary>
    public async Task<bool> IsAvailableAsync()
    {
        await EnsureInitializedAsync().ConfigureAwait(false);
        return _cacheDir != null;
    }

    /// <summary>
    /// True when <paramref name="cacheKey"/> is cached AND complete AND its size matches what was recorded.
    /// A partially downloaded entry reports false - it is resumable, not usable.
    /// </summary>
    public async Task<bool> IsCompleteAsync(string cacheKey)
    {
        await EnsureInitializedAsync().ConfigureAwait(false);
        if (_cacheDir == null) return false;
        var meta = await ReadMetaAsync(cacheKey).ConfigureAwait(false);
        if (meta is not { Complete: true }) return false;
        var size = await GetEntrySizeAsync(cacheKey).ConfigureAwait(false);
        return size >= 0 && (meta.Total < 0 || meta.Total == size);
    }

    /// <summary>
    /// Open a complete cache entry as a seekable JS-side stream, or null when it is absent or incomplete.
    /// Never touches the network. The caller disposes the stream.
    /// </summary>
    public async Task<OPFSStream?> OpenCachedAsync(string cacheKey, CancellationToken ct = default)
    {
        if (!await IsCompleteAsync(cacheKey).ConfigureAwait(false)) return null;
        return await OpenEntryAsync(cacheKey, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Return a seekable JS-side stream over <paramref name="url"/>, downloading and caching it first if
    /// needed. An interrupted previous attempt is resumed rather than restarted.
    /// </summary>
    /// <param name="url">
    /// Source URL. Point this at the hub (<c>/hf/{repoId}/{filePath}</c>) rather than at huggingface.co
    /// directly - the hub is what supplies the CORS headers a browser will accept, caches the file, and keeps
    /// us out of HuggingFace's rate limiter.
    /// </param>
    /// <param name="cacheKey">Explicit cache key; derived from the URL path when omitted.</param>
    /// <param name="progress">Optional progress sink, throttled by <see cref="ProgressIntervalMs"/>.</param>
    /// <param name="ct">Cancels the download. A cancelled download leaves a resumable partial entry.</param>
    public async Task<OPFSStream> OpenOrDownloadAsync(string url, string? cacheKey = null,
        IProgress<ModelDownloadProgress>? progress = null, CancellationToken ct = default)
    {
        cacheKey ??= UrlToCacheKey(url);
        await EnsureInitializedAsync().ConfigureAwait(false);
        if (_cacheDir == null)
            throw new InvalidOperationException(
                "OPFS is not available, so no model can be cached or streamed. OPFS needs a secure context " +
                "(https or localhost) on a browser that supports it.");

        var gate = GetGate(cacheKey);
        await gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            var meta = await ReadMetaAsync(cacheKey).ConfigureAwait(false);
            var onDisk = await GetEntrySizeAsync(cacheKey).ConfigureAwait(false);

            // Cache hit: complete, the file really is the size we recorded, and it came from this URL - or
            // from nowhere. An empty Url means the entry was handed to PutAsync (a torrent stream, a picked
            // file, another store) rather than fetched here; re-downloading it over HTTP because we cannot
            // prove its origin would defeat the point of storing it.
            if (meta is { Complete: true } && (meta.Url == url || meta.Url.Length == 0)
                && onDisk >= 0 && (meta.Total < 0 || meta.Total == onDisk))
            {
                progress?.Report(new ModelDownloadProgress(onDisk, onDisk, false));
                return await OpenEntryAsync(cacheKey, ct).ConfigureAwait(false);
            }

            // Resume only when the partial we hold was for THIS url. meta.Received is the last count we
            // actually flushed; anything past it in the file was never confirmed, so it is discarded.
            long resumeFrom = 0;
            if (meta is { Complete: false } && meta.Url == url && onDisk > 0)
                resumeFrom = Math.Max(0, Math.Min(onDisk, meta.Received));

            await DownloadAsync(url, cacheKey, resumeFrom, meta?.ETag, progress, ct).ConfigureAwait(false);
            return await OpenEntryAsync(cacheKey, ct).ConfigureAwait(false);
        }
        finally
        {
            gate.Release();
        }
    }

    // ──────────────────────────────────────────────────────────────────────────────────────────────
    //  IModelStore - store bytes by key, from ANY source
    // ──────────────────────────────────────────────────────────────────────────────────────────────

    /// <inheritdoc/>
    Task<bool> IModelStore.ExistsAsync(string key, CancellationToken cancellationToken) => IsCompleteAsync(key);

    /// <inheritdoc/>
    async Task<Stream?> IModelStore.OpenReadAsync(string key, CancellationToken cancellationToken)
        => await OpenCachedAsync(key, cancellationToken).ConfigureAwait(false);

    /// <inheritdoc/>
    async Task<IReadOnlyList<ModelStoreEntry>> IModelStore.ListAsync(CancellationToken cancellationToken)
    {
        var entries = await ListCachedAsync().ConfigureAwait(false);
        return entries.ConvertAll(e => new ModelStoreEntry(e.Key, e.SizeBytes, e.Complete));
    }

    /// <inheritdoc/>
    Task<long> IModelStore.GetTotalSizeAsync(CancellationToken cancellationToken) => GetCacheSizeAsync();

    /// <inheritdoc/>
    Task IModelStore.RemoveAsync(string key, CancellationToken cancellationToken) => RemoveAsync(key);

    /// <summary>
    /// Store <paramref name="source"/> under <paramref name="key"/> and mark it complete - the
    /// transport-agnostic way in. See <see cref="IModelStore.PutAsync"/>.
    /// </summary>
    /// <remarks>
    /// 🔴 The copy is <see cref="Stream.CopyToAsync(Stream, int, CancellationToken)"/> on purpose, not a
    /// hand-rolled read/write loop. <c>JSReadStreamBase</c> overrides it to pump <c>Uint8Array</c> chunks
    /// JS-side whenever the destination is an <c>IJSWriteStream</c> - which an <c>OPFSStream</c> opened for
    /// writing is - so a JS-side source (a torrent piece stream, a <c>BlobStream</c>, another OPFS entry)
    /// never lands a byte on the managed heap. Any other <c>Stream</c> falls back to the standard managed
    /// path and still works. Writing the loop by hand here would silently opt every JS source out of that.
    /// </remarks>
    public async Task PutAsync(string key, Stream source, IProgress<ModelDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        await EnsureInitializedAsync().ConfigureAwait(false);
        if (_cacheDir == null)
            throw new InvalidOperationException(
                "OPFS is not available, so nothing can be stored. OPFS needs a secure context (https or " +
                "localhost) on a browser that supports it.");

        long expected = -1;
        try { if (source.CanSeek) expected = source.Length - source.Position; } catch { /* unknowable */ }

        var gate = GetGate(key);
        await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            // Clear any prior state first: a stale sidecar saying "complete" must never survive a put that
            // then fails partway, or the truncated result would be served as a finished model.
            await RemoveAsync(key).ConfigureAwait(false);

            long written;
            var dest = await OPFSStream.OpenPath(_cacheDir, key, FileMode.Create, FileAccess.Write,
                cancellationToken: cancellationToken).ConfigureAwait(false);
            await using (dest.ConfigureAwait(false))
            {
                await source.CopyToAsync(dest, WriteBufferSize > 0 ? WriteBufferSize : 81920, cancellationToken)
                    .ConfigureAwait(false);
                await dest.FlushAsync(cancellationToken).ConfigureAwait(false);
                written = dest.Length;
            }

            if (expected >= 0 && written != expected)
                throw new IOException(
                    $"Storing '{key}' copied {written} bytes but the source reported {expected}.");

            await WriteMetaAsync(key, new CacheEntryMeta
            {
                Url = "",               // no origin: these bytes were handed to us, not fetched by us
                Total = written,
                Received = written,
                Complete = true,
                ETag = null,
            }).ConfigureAwait(false);

            progress?.Report(new ModelDownloadProgress(written, written, false));
        }
        finally
        {
            gate.Release();
        }
    }

    /// <summary>Delete a cache entry and its sidecar.</summary>
    public async Task RemoveAsync(string cacheKey)
    {
        await EnsureInitializedAsync().ConfigureAwait(false);
        if (_cacheDir == null) return;
        try { await _cacheDir.RemoveEntry(cacheKey); } catch { /* absent */ }
        try { await _cacheDir.RemoveEntry(cacheKey + MetaSuffix); } catch { /* absent */ }
    }

    /// <summary>Every cached entry with its real on-disk size and whether it is complete.</summary>
    public async Task<List<(string Key, long SizeBytes, bool Complete)>> ListCachedAsync()
    {
        var result = new List<(string, long, bool)>();
        await EnsureInitializedAsync().ConfigureAwait(false);
        if (_cacheDir == null) return result;

        var entries = await _cacheDir.ValuesList();
        foreach (var entry in entries)
        {
            using var handle = entry;
            if (handle.Name.EndsWith(MetaSuffix, StringComparison.Ordinal)) continue;
            if (handle is not FileSystemFileHandle fileHandle) continue;
            long size;
            try { size = await fileHandle.GetSize(); } catch { continue; }
            result.Add((handle.Name, size, await IsCompleteAsync(handle.Name).ConfigureAwait(false)));
        }
        return result;
    }

    /// <summary>Total bytes held by the cache, sidecars included.</summary>
    public async Task<long> GetCacheSizeAsync()
    {
        await EnsureInitializedAsync().ConfigureAwait(false);
        if (_cacheDir == null) return 0;

        long total = 0;
        var entries = await _cacheDir.ValuesList();
        foreach (var entry in entries)
        {
            using var handle = entry;
            if (handle is not FileSystemFileHandle fileHandle) continue;
            try { total += await fileHandle.GetSize(); } catch { /* raced with a delete */ }
        }
        return total;
    }

    /// <summary>Delete every cached model.</summary>
    public async Task ClearAllAsync()
    {
        await EnsureInitializedAsync().ConfigureAwait(false);
        if (_cacheDir == null) return;

        var entries = await _cacheDir.ValuesList();
        foreach (var entry in entries)
        {
            using var handle = entry;
            try { await _cacheDir.RemoveEntry(handle.Name); } catch { /* raced with a delete */ }
        }
    }

    /// <summary>Derive a flat, filesystem-safe cache key from a URL's path.</summary>
    public static string UrlToCacheKey(string url)
    {
        var uri = new Uri(url);
        var key = uri.AbsolutePath.TrimStart('/').Replace('/', '_').Replace('\\', '_');
        if (key.Length > 200) key = key[^200..];
        return key;
    }

    // ──────────────────────────────────────────────────────────────────────────────────────────────
    //  Download
    // ──────────────────────────────────────────────────────────────────────────────────────────────

    private async Task DownloadAsync(string url, string cacheKey, long resumeFrom, string? knownETag,
        IProgress<ModelDownloadProgress>? progress, CancellationToken ct)
    {
        var headers = new Dictionary<string, string>();
        if (resumeFrom > 0)
        {
            headers["Range"] = $"bytes={resumeFrom}-";
            // If the server's copy changed since our partial was written, it answers 200 with the whole body
            // instead of 206 and we start over - rather than splicing two different files together.
            if (!string.IsNullOrEmpty(knownETag)) headers["If-Range"] = knownETag!;
        }

        // Fetch from the RUNTIME, not from `window`: there is no `window` in a worker, and a model SHOULD be
        // loaded from a worker. SpawnJSRuntime.Fetch calls fetch() on whatever the global scope is.
        using var response = headers.Count > 0
            ? await _js.Fetch(url, new FetchOptions { Headers = headers }).ConfigureAwait(false)
            : await _js.Fetch(url).ConfigureAwait(false);

        // fetch() does NOT throw on 404/500 - it resolves with Ok=false and an ERROR BODY. Nothing is opened
        // or written before this check, so a failed response can never reach the cache.
        if (!response.Ok)
            throw new HttpRequestException(
                $"Model download failed: {(int)response.Status} {response.StatusText} for {url}");

        var status = (int)response.Status;
        var resumed = resumeFrom > 0 && status == 206;
        if (resumeFrom > 0 && status != 206)
        {
            // Server ignored the Range, or If-Range failed because the file changed: take the full body.
            resumeFrom = 0;
        }

        long total;
        string? etag;
        using (var respHeaders = response.Headers)
        {
            total = ParseTotalBytes(respHeaders, resumeFrom);
            etag = NullIfEmpty(respHeaders.Get("etag"));
        }

        var meta = new CacheEntryMeta { Url = url, Total = total, Received = resumeFrom, Complete = false, ETag = etag };
        await WriteMetaAsync(cacheKey, meta).ConfigureAwait(false);

        long received = resumeFrom;   // bytes pulled off the network
        long written = resumeFrom;    // bytes actually handed to OPFS (lags `received` while buffering)
        var lastReport = Environment.TickCount64;
        var lastMetaFlush = written;
        long chunkCount = 0, writeCount = 0;
        Uint8Array? writeBuffer = null;
        long bufferFill = 0;

        // OpenOrCreate + SetLength(resumeFrom) rather than FileMode.Append: it makes the truncation explicit,
        // so any unconfirmed bytes past the last flushed count are dropped instead of being kept and counted.
        var stream = await OPFSStream.OpenPath(_cacheDir!, cacheKey, FileMode.OpenOrCreate, FileAccess.Write,
            cancellationToken: ct).ConfigureAwait(false);
        await using (stream.ConfigureAwait(false))
        {
            stream.SetLength(resumeFrom);
            stream.Seek(resumeFrom, SeekOrigin.Begin);

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

                    // 🔴 The chunk stays a JS Uint8Array from here to OPFS. Never ReadBytes() it - that is
                    // the copy that put whole checkpoints on the managed heap in ModelCache.
                    using var chunk = result.Value;
                    if (chunk == null) continue;
                    var chunkLength = chunk.Length;
                    if (chunkLength == 0) continue;
                    chunkCount++;

                    if (WriteBufferSize <= 0 || chunkLength >= WriteBufferSize)
                    {
                        // Already big enough to write on its own. Drain anything pending first so the
                        // bytes reach the file in order.
                        await FlushWriteBufferAsync().ConfigureAwait(false);
                        await stream.WriteUint8ArrayAsync(chunk, ct).ConfigureAwait(false);
                        writeCount++;
                        written += chunkLength;
                    }
                    else
                    {
                        if (bufferFill + chunkLength > WriteBufferSize)
                            await FlushWriteBufferAsync().ConfigureAwait(false);
                        // TypedArray.set: a JS-side copy into the staging buffer. No managed array involved.
                        writeBuffer ??= new Uint8Array(WriteBufferSize);
                        writeBuffer.Set(chunk, bufferFill);
                        bufferFill += chunkLength;
                    }
                    received += chunkLength;

                    // Only record progress that is actually ON DISK - `written`, never `received`. Buffered
                    // bytes are not durable, and the sidecar is what a resume trusts.
                    if (written - lastMetaFlush >= MetaFlushInterval)
                    {
                        await stream.FlushAsync(ct).ConfigureAwait(false);
                        meta.Received = written;
                        await WriteMetaAsync(cacheKey, meta).ConfigureAwait(false);
                        lastMetaFlush = written;
                    }

                    var now = Environment.TickCount64;
                    if (progress != null && now - lastReport >= ProgressIntervalMs)
                    {
                        progress.Report(new ModelDownloadProgress(received, total, resumed));
                        lastReport = now;
                    }
                }

                // Everything received has now been handed to OPFS.
                await FlushWriteBufferAsync().ConfigureAwait(false);
            }
            catch
            {
                // Record only what actually landed, so the next attempt resumes instead of restarting.
                // The staging buffer is deliberately NOT flushed here: a failure mid-fill means its tail is
                // indeterminate, and writing it would corrupt the resume point.
                try
                {
                    await stream.FlushAsync(CancellationToken.None).ConfigureAwait(false);
                    meta.Received = written;
                    await WriteMetaAsync(cacheKey, meta).ConfigureAwait(false);
                }
                catch { /* the original failure is the one worth reporting */ }
                throw;
            }
            finally
            {
                writeBuffer?.Dispose();
            }

            await stream.FlushAsync(ct).ConfigureAwait(false);

            // Drain the staging buffer into the file, in order. Writes the exact filled prefix, never the
            // whole buffer - SubArray is a VIEW onto the same storage, so this copies nothing.
            async Task FlushWriteBufferAsync()
            {
                if (bufferFill == 0 || writeBuffer == null) return;
                if (bufferFill == WriteBufferSize)
                {
                    await stream.WriteUint8ArrayAsync(writeBuffer, ct).ConfigureAwait(false);
                }
                else
                {
                    using var view = writeBuffer.SubArray(0, bufferFill);
                    await stream.WriteUint8ArrayAsync(view, ct).ConfigureAwait(false);
                }
                writeCount++;
                written += bufferFill;
                bufferFill = 0;
            }
        }

        LastDownloadChunks = chunkCount;
        LastDownloadWrites = writeCount;
        LastDownloadBytes = received - resumeFrom;

        // A short read is a truncated download, not a model. Leave the entry incomplete so the next call
        // resumes it, and say so - rather than caching it and failing much later inside a proto reader.
        if (total > 0 && written != total)
        {
            meta.Received = written;
            await WriteMetaAsync(cacheKey, meta).ConfigureAwait(false);
            throw new IOException(
                $"Model download truncated for {url}: got {written} of {total} bytes. " +
                "The partial entry was kept and the next attempt will resume it.");
        }

        meta.Received = written;
        meta.Total = total > 0 ? total : written;
        meta.Complete = true;
        await WriteMetaAsync(cacheKey, meta).ConfigureAwait(false);
        progress?.Report(new ModelDownloadProgress(received, meta.Total, resumed));
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

    // ──────────────────────────────────────────────────────────────────────────────────────────────
    //  Entry + sidecar plumbing
    // ──────────────────────────────────────────────────────────────────────────────────────────────

    private Task<OPFSStream> OpenEntryAsync(string cacheKey, CancellationToken ct) =>
        OPFSStream.OpenPath(_cacheDir!, cacheKey, FileMode.Open, FileAccess.Read, cancellationToken: ct);

    private async Task<long> GetEntrySizeAsync(string cacheKey)
    {
        try
        {
            using var handle = await _cacheDir!.GetFileHandle(cacheKey);
            return await handle.GetSize();
        }
        catch { return -1; }
    }

    /// <summary>
    /// The sidecar is a handful of <c>key=value</c> lines rather than JSON: it is internal, it is about a
    /// hundred bytes, and a hand-rolled format has no trimming or source-generator contract to keep in step.
    /// </summary>
    private sealed class CacheEntryMeta
    {
        public string Url = "";
        public long Total = -1;
        public long Received;
        public bool Complete;
        public string? ETag;
    }

    private async Task<CacheEntryMeta?> ReadMetaAsync(string cacheKey)
    {
        try
        {
            using var handle = await _cacheDir!.GetFileHandle(cacheKey + MetaSuffix);
            using var file = await handle.GetFile();
            var text = await file.Text();
            if (string.IsNullOrWhiteSpace(text)) return null;

            var meta = new CacheEntryMeta();
            foreach (var line in text.Split('\n'))
            {
                var eq = line.IndexOf('=');
                if (eq <= 0) continue;
                var name = line[..eq].Trim();
                var value = line[(eq + 1)..].Trim('\r', ' ');
                switch (name)
                {
                    case "url": meta.Url = value; break;
                    case "total": meta.Total = long.TryParse(value, out var t) ? t : -1; break;
                    case "received": meta.Received = long.TryParse(value, out var r) ? r : 0; break;
                    case "complete": meta.Complete = value == "1"; break;
                    case "etag": meta.ETag = NullIfEmpty(value); break;
                }
            }
            return meta;
        }
        catch { return null; }
    }

    private async Task WriteMetaAsync(string cacheKey, CacheEntryMeta meta)
    {
        var text = new StringBuilder()
            .Append("v=1\n")
            .Append("url=").Append(meta.Url).Append('\n')
            .Append("total=").Append(meta.Total).Append('\n')
            .Append("received=").Append(meta.Received).Append('\n')
            .Append("complete=").Append(meta.Complete ? '1' : '0').Append('\n')
            .Append("etag=").Append(meta.ETag ?? "").Append('\n')
            .ToString();

        // ~100 bytes of metadata: the managed-heap rule is about BULK data, and this is the metadata
        // exception it names. The model payload itself never takes this path.
        using var handle = await _cacheDir!.GetFileHandle(cacheKey + MetaSuffix, create: true);
        using var writable = await handle.CreateWritable();
        using var bytes = new Uint8Array(Encoding.UTF8.GetBytes(text));
        await writable.Write(bytes);
        await writable.Close();
    }

    private SemaphoreSlim GetGate(string cacheKey)
    {
        lock (_keyGates)
        {
            if (!_keyGates.TryGetValue(cacheKey, out var gate))
                _keyGates[cacheKey] = gate = new SemaphoreSlim(1, 1);
            return gate;
        }
    }

    private static string? NullIfEmpty(string? value) => string.IsNullOrEmpty(value) ? null : value;

    private async Task EnsureInitializedAsync()
    {
        if (_initialized) return;
        _initialized = true;
        try
        {
            using var navigator = _js.Get<Navigator>("navigator");
            using var storage = navigator.Storage;
            using var root = await storage.GetDirectory();
            _cacheDir = await root.GetDirectoryHandle(CacheDirectoryName, create: true);
        }
        catch
        {
            _cacheDir = null; // OPFS unavailable (insecure context or unsupported browser)
        }
    }

    /// <summary>Release the cached OPFS directory handle. Streams already handed out are unaffected.</summary>
    public void Dispose() => _cacheDir?.Dispose();
}
