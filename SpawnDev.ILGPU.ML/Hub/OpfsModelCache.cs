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
public class OpfsModelCache : IResumableModelStore, IDisposable
{
    /// <summary>Sidecar suffix holding an entry's download state. Kept beside the data file, not inside it.</summary>
    private const string MetaSuffix = ".meta";

    /// <summary>Flush the sidecar every this many bytes so an interrupted download resumes near where it stopped.</summary>
    private const long MetaFlushInterval = 8L * 1024 * 1024;

    private readonly SpawnJSRuntime _js;
    private readonly Dictionary<string, SemaphoreSlim> _keyGates = new();
    private FileSystemDirectoryHandle? _cacheDir;
    private bool _initialized;
    private readonly SemaphoreSlim _dirGate = new(1, 1);

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
    /// <para>
    /// 16 MiB, matching ILGPU's <c>MemoryBuffer.DefaultStreamChunkSizeInBytes</c>. MEASURED 2026-09-14 on a
    /// 192 MiB OPFS file across three browser lanes (MB/s): 64 KiB 75/87/86, 1 MiB 577/591/612, 4 MiB
    /// 1290/1320/1291, <b>16 MiB 1559/1986/1981</b> - a consistent 1.21x-1.53x over 4 MiB. Above 16 MiB the
    /// sweep is noise rather than signal, so this is the largest size with a trustworthy gain. It matters
    /// most for <see cref="PutAsync"/> (OPFS-to-OPFS, disk-bound); a network download is bounded by the
    /// network long before the write size matters, and progress granularity is unaffected either way
    /// because progress is reported from bytes RECEIVED, not bytes written.
    /// </para>
    /// </remarks>
    public int WriteBufferSize { get; set; } = 16 * 1024 * 1024;


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
    /// What the browser says this origin's storage quota and usage are, or null when it cannot say (no <c>navigator.storage</c>).
    /// </summary>
    /// <remarks>Read <see cref="BrowserStorageEstimate"/> before using the number as a budget: Chrome reports <c>usage + 10 GiB</c>, not the real limit.</remarks>
    public async Task<BrowserStorageEstimate?> GetStorageEstimateAsync()
    {
        try
        {
            using var navigator = _js.Get<Navigator>("navigator");
            using var storage = navigator.Storage;
            var est = await storage.Estimate().ConfigureAwait(false);
            return new BrowserStorageEstimate(est.Quota, est.Usage);
        }
        catch { return null; }   // no StorageManager here (insecure context / not a browser)
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
        var stream = await OpenEntryAsync(cacheKey, ct).ConfigureAwait(false);
        await TouchAsync(cacheKey).ConfigureAwait(false);
        return stream;
    }

    /// <summary>Record that a complete entry was just used (the order <see cref="TrimAsync"/> evicts in). Never fails the open it follows.</summary>
    private async Task TouchAsync(string cacheKey)
    {
        try
        {
            var meta = await ReadMetaAsync(cacheKey).ConfigureAwait(false);
            if (meta is not { Complete: true }) return;
            meta.Used = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            await WriteMetaAsync(cacheKey, meta).ConfigureAwait(false);
        }
        catch { /* the order of eviction is a preference; the open already succeeded */ }
    }

    /// <summary>
    /// Evict entries until the model bytes this cache holds are at most <paramref name="maxBytes"/>: partial downloads first, then complete
    /// entries least recently used first. Entries in <paramref name="keep"/> and entries downloading right now are never evicted, so the total
    /// can stay above <paramref name="maxBytes"/> when those alone exceed it. Returns (bytes held afterwards, bytes freed).
    /// </summary>
    /// <remarks>
    /// The sizes are the cache's own files, not <c>navigator.storage.estimate()</c>: MEASURED 2026-11-24 (Lore, Chrome on Windows, after a
    /// full row of Studio examples) the origin's OPFS held 110 files / 42.25 GB while <c>estimate()</c> reported usage 3.60 GB of a 14.3 GB
    /// quota. A cache that plans against that estimate keeps growing until the browser clears the whole origin (it did, mid-session).
    /// </remarks>
    public async Task<(long StoredBytes, long FreedBytes)> TrimAsync(long maxBytes, IEnumerable<string>? keep = null, CancellationToken ct = default)
    {
        if (maxBytes < 0) throw new ArgumentOutOfRangeException(nameof(maxBytes));
        await EnsureInitializedAsync().ConfigureAwait(false);
        if (_cacheDir == null) return (0, 0);
        var pinned = new HashSet<string>(keep ?? System.Array.Empty<string>(), StringComparer.Ordinal);
        foreach (var d in Downloader.ActiveDownloads) pinned.Add(d.Key);

        var entries = new List<(string Key, long Size, bool Complete, long Used)>();
        var handles = await WithDirectoryAsync(d => d.ValuesList()).ConfigureAwait(false);
        foreach (var entry in handles)
        {
            using var handle = entry;
            if (handle.Name.EndsWith(MetaSuffix, StringComparison.Ordinal) || handle is not FileSystemFileHandle file) continue;
            long size;
            try { size = await file.GetSize(); } catch { continue; }   // raced with a delete
            CacheEntryMeta? meta;
            try { meta = await ReadMetaAsync(handle.Name).ConfigureAwait(false); }
            catch
            {
                // a sidecar that cannot be read right now says nothing about the model: keep it (counted, never evicted), as a held file would be
                pinned.Add(handle.Name);
                entries.Add((handle.Name, size, true, long.MaxValue));
                continue;
            }
            bool complete = meta is { Complete: true } && (meta.Total < 0 || meta.Total == size);
            entries.Add((handle.Name, size, complete, complete ? meta!.Used : 0));
        }
        long stored = entries.Sum(e => e.Size), freed = 0;
        foreach (var e in entries.Where(e => !pinned.Contains(e.Key))
                     .OrderBy(e => e.Complete).ThenBy(e => e.Used).ThenBy(e => e.Key, StringComparer.Ordinal))
        {
            if (stored <= maxBytes) break;
            ct.ThrowIfCancellationRequested();
            await RemoveAsync(e.Key).ConfigureAwait(false);
            stored -= e.Size; freed += e.Size;
        }
        return (stored, freed);
    }

    /// <summary>
    /// Return a seekable JS-side stream over <paramref name="url"/>, downloading and caching it first if
    /// needed. An interrupted previous attempt is resumed rather than restarted.
    /// </summary>
    /// <param name="url">
    /// Source URL. Point this at the models host (<see cref="HuggingFaceClient.GetDownloadUrl"/>, i.e.
    /// <c>https://models.spawndev.com/{repoId}/resolve/main/{filePath}</c>) rather than at huggingface.co: it serves the CORS headers a
    /// browser accepts and answers Range with a 206 from the first byte, and the download falls back to Hugging Face by itself
    /// (same path layout) when the host cannot give the file.
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
        var dir = _cacheDir;

        // The HTTP transport lives in HttpModelDownloader, which drives this class only through
        // IResumableModelStore. This method is the convenience wrapper over "store + HTTP"; it is NOT a
        // second implementation of the download, because a parallel implementation is exactly how one path
        // ends up with a fix or a guard the other lacks.
        try
        {
            return (OPFSStream)await Downloader.GetOrDownloadAsync(url, cacheKey, progress, ct).ConfigureAwait(false);
        }
        catch (Exception ex) when (IsStorageGone(ex))
        {
            // The browser cleared this origin's storage while the page was open (2026-11-24, Lore, the live site in Chrome: the
            // profile's OPFS went to 0 entries mid-session; every model load after that failed within 2 s with "A requested file or
            // directory could not be found at the time an operation was processed", downloads included). The cached directory handle
            // points at a directory that no longer exists, so every operation through it throws NotFoundError. Open the directory
            // again (it is recreated) and retry once: the entry is gone, so it downloads. A cache that loses its files must cost a
            // download, never the run.
            if (!await ReopenDirectoryAsync(dir).ConfigureAwait(false)) throw;
            return (OPFSStream)await Downloader.GetOrDownloadAsync(url, cacheKey, progress, ct).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// True when <paramref name="ex"/> (or an inner exception) says the stored files are gone: OPFS's NotFoundError (the entry or the cache directory
    /// itself no longer exists), the FileNotFoundException OPFSStream throws for a missing entry, or <see cref="StoredFileLostException"/>.
    /// </summary>
    internal static bool IsStorageGone(Exception? ex)
    {
        for (var e = ex; e != null; e = e.InnerException)
            if (e is JSException { Name: "NotFoundError" } or StoredFileLostException or FileNotFoundException) return true;
        return false;
    }

    /// <summary>A partial file the store was asked to continue no longer holds the bytes it was to continue from.</summary>
    internal sealed class StoredFileLostException(string message) : IOException(message);

    /// <summary>
    /// Replace the cached OPFS directory handle after an operation through <paramref name="stale"/> found the directory gone: open the directory again
    /// (created when missing). Several loads can fail on the same stale handle at once: the first one reopens, the others use what it opened. The stale
    /// handle is not disposed here, because another operation may still be inside a call on it. False when OPFS is not available any more.
    /// </summary>
    private async Task<bool> ReopenDirectoryAsync(FileSystemDirectoryHandle stale)
    {
        await _dirGate.WaitAsync().ConfigureAwait(false);
        try
        {
            if (ReferenceEquals(_cacheDir, stale)) _cacheDir = await OpenDirectoryAsync().ConfigureAwait(false);
            return _cacheDir != null;
        }
        finally { _dirGate.Release(); }
    }

    /// <summary>Run an operation on the cache directory; when the browser cleared the directory behind the handle, reopen it and run the operation once more.</summary>
    private async Task<T> WithDirectoryAsync<T>(Func<FileSystemDirectoryHandle, Task<T>> operation)
    {
        var dir = _cacheDir!;
        try { return await operation(dir).ConfigureAwait(false); }
        catch (Exception ex) when (IsStorageGone(ex))
        {
            if (!await ReopenDirectoryAsync(dir).ConfigureAwait(false)) throw;
            return await operation(_cacheDir!).ConfigureAwait(false);
        }
    }

    private HttpModelDownloader? _downloader;

    /// <summary>
    /// The HTTP transport filling this store. Exposed so callers can read its diagnostics
    /// (<see cref="HttpModelDownloader.LastDownloadChunks"/> and friends) or tune
    /// <see cref="HttpModelDownloader.WriteBufferSize"/>.
    /// </summary>
    public HttpModelDownloader Downloader => _downloader ??= new HttpModelDownloader(_js, this)
    {
        WriteBufferSize = WriteBufferSize,
        ProgressIntervalMs = ProgressIntervalMs,
    };

    /// <summary>Chunks delivered by <c>fetch</c> during the most recent download (diagnostic).</summary>
    public long LastDownloadChunks => Downloader.LastDownloadChunks;

    /// <summary>OPFS writes issued during the most recent download (diagnostic).</summary>
    public long LastDownloadWrites => Downloader.LastDownloadWrites;

    /// <summary>Bytes transferred during the most recent download, excluding anything carried by a resume.</summary>
    public long LastDownloadBytes => Downloader.LastDownloadBytes;

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

    /// <inheritdoc/>
    Task IModelStore.ClearAsync(CancellationToken cancellationToken) => ClearAllAsync();

    /// <inheritdoc/>
    public async Task<ModelStoreState> GetStateAsync(string key, CancellationToken cancellationToken = default)
    {
        await EnsureInitializedAsync().ConfigureAwait(false);
        if (_cacheDir == null) return default;

        var meta = await ReadMetaAsync(key).ConfigureAwait(false);
        var onDisk = await GetEntrySizeAsync(key).ConfigureAwait(false);
        if (onDisk < 0 && meta == null) return default;

        // Confirmed bytes are the LESSER of what the sidecar acknowledged and what is actually on disk.
        // Either can lead the other after a crash, and a resume must trust only what both agree on.
        var written = meta == null ? Math.Max(0, onDisk) : Math.Max(0, Math.Min(onDisk < 0 ? 0 : onDisk, meta.Received));
        var complete = meta is { Complete: true } && onDisk >= 0 && (meta.Total < 0 || meta.Total == onDisk);
        return new ModelStoreState(onDisk >= 0, complete, complete ? onDisk : written,
            meta?.Total ?? -1, meta?.Url, meta?.ETag);
    }

    /// <inheritdoc/>
    public async Task<Stream> OpenWriteAsync(string key, long startOffset, CancellationToken cancellationToken = default)
    {
        if (startOffset < 0) throw new ArgumentOutOfRangeException(nameof(startOffset));
        await EnsureInitializedAsync().ConfigureAwait(false);
        if (_cacheDir == null)
            throw new InvalidOperationException(
                "OPFS is not available, so nothing can be written. It needs a secure context (https or " +
                "localhost) on a browser that supports it.");

        var stream = await OPFSStream.OpenPath(_cacheDir, key, FileMode.OpenOrCreate, FileAccess.Write,
            cancellationToken: cancellationToken).ConfigureAwait(false);
        // Truncate rather than merely seek: anything past the confirmed resume point was never
        // acknowledged, and keeping it would bury unverified bytes inside a file that later says complete.
        // And never LENGTHEN: when the file vanished after the caller read its state (the browser cleared the origin), OpenOrCreate made an
        // empty one, and extending it to the resume point would leave zeros that later count as downloaded bytes.
        if (stream.Length < startOffset)
        {
            long held = stream.Length;
            await stream.DisposeAsync().ConfigureAwait(false);
            throw new StoredFileLostException($"'{key}' holds {held} bytes but was to continue at {startOffset}: the stored file is gone or was cut, so the download must restart.");
        }
        stream.SetLength(startOffset);
        stream.Seek(startOffset, SeekOrigin.Begin);
        return stream;
    }

    /// <inheritdoc/>
    public Task SetStateAsync(string key, string sourceRef, long totalBytes, long bytesWritten, bool complete,
        string? etag, CancellationToken cancellationToken = default)
        => WriteMetaAsync(key, new CacheEntryMeta
        {
            Url = sourceRef ?? "",
            Total = totalBytes,
            Received = bytesWritten,
            Complete = complete,
            ETag = etag,
            Used = complete ? DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() : 0,   // a finished download counts as a use
        });

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
            // only the OPEN recovers from a cleared directory: nothing of the source has been read yet
            var dest = await WithDirectoryAsync(d => OPFSStream.OpenPath(d, key, FileMode.Create, FileAccess.Write,
                cancellationToken: cancellationToken)).ConfigureAwait(false);
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

        var entries = await WithDirectoryAsync(d => d.ValuesList()).ConfigureAwait(false);
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
        var entries = await WithDirectoryAsync(d => d.ValuesList()).ConfigureAwait(false);
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

        var entries = await WithDirectoryAsync(d => d.ValuesList()).ConfigureAwait(false);
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
        public long Used;   // unix ms of the last open (0 = never recorded); TrimAsync evicts the oldest first
    }

    /// <summary>
    /// Reads the sidecar; null means there is none (NotFoundError). Any other failure is retried while it is a lock held elsewhere (NoModificationAllowedError:
    /// another tab or worker holds the file) and otherwise THROWN: reading it as absent made a complete model look unrecorded, and the downloader then
    /// started over from byte 0 (the desktop store did exactly that, Lore 2026-11-26).
    /// </summary>
    private async Task<CacheEntryMeta?> ReadMetaAsync(string cacheKey)
    {
        string? text = null;
        for (int attempt = 1, delay = 10; ; attempt++, delay = Math.Min(delay * 2, 500))
        {
            try
            {
                using var handle = await _cacheDir!.GetFileHandle(cacheKey + MetaSuffix);
                using var file = await handle.GetFile();
                text = await file.Text();
                break;
            }
            catch (JSException e) when (e.Name == "NotFoundError") { return null; }
            catch (JSException e) when (e.Name == "NoModificationAllowedError" && attempt < 10) { await Task.Delay(delay).ConfigureAwait(false); }
        }
        if (string.IsNullOrWhiteSpace(text)) return null;
        {

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
                    case "used": meta.Used = long.TryParse(value, out var u) ? u : 0; break;
                }
            }
            return meta;
        }
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
            .Append("used=").Append(meta.Used).Append('\n')
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
        await _dirGate.WaitAsync().ConfigureAwait(false);
        try
        {
            if (_initialized) return;
            _cacheDir = await OpenDirectoryAsync().ConfigureAwait(false);
            _initialized = true;   // after the open: a concurrent first caller waits for the handle instead of reading "OPFS is not available"
        }
        finally { _dirGate.Release(); }
    }

    /// <summary>The cache directory (created when missing), or null when OPFS is unavailable (insecure context or unsupported browser).</summary>
    private async Task<FileSystemDirectoryHandle?> OpenDirectoryAsync()
    {
        try
        {
            using var navigator = _js.Get<Navigator>("navigator");
            using var storage = navigator.Storage;
            using var root = await storage.GetDirectory();
            return await root.GetDirectoryHandle(CacheDirectoryName, create: true);
        }
        catch { return null; }
    }

    /// <summary>Release the cached OPFS directory handle. Streams already handed out are unaffected.</summary>
    public void Dispose() => _cacheDir?.Dispose();
}
