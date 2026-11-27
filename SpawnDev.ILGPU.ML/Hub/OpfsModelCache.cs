using System.Globalization;
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

    /// <summary>Suffix of the directory that holds an entry's bytes as committed part files (<see cref="PartBytes"/>).</summary>
    private const string PartsSuffix = ".parts";

    /// <summary>Suffix of the swap file Chrome keeps beside a file while a <c>FileSystemWritableFileStream</c> on it is open.</summary>
    private const string SwapSuffix = ".crswap";

    /// <summary>
    /// An entry is stored as a directory of part files of this many bytes, each committed (closed) as soon as it is full. Default 128 MiB.
    /// </summary>
    /// <remarks>
    /// 🔴 WHY PARTS (2026-11-27, the live /video demo, TJ's OS drive nearly filled). Outside a dedicated worker OPFS can only be written through
    /// <c>createWritable()</c>, and Chrome keeps everything written there in a swap file (<c>name.crswap</c>) until <c>close()</c>: the real file stays at
    /// 0 bytes for the whole download. MEASURED on the live site with one file per entry: a browser that ended mid-download left the real file at 0 bytes,
    /// a sidecar saying 1.02 GB were received, and the 1.02 GB swap file on disk as an orphan nothing ever removed; the retry started the file again at
    /// byte 0. A resume of a partial that did get committed copies the whole partial file into a new swap file first (<c>keepExistingData</c>), so a
    /// 6 GB partial needs 12 GB. Either way every interrupted attempt of a multi-GB model costs its bytes again, on the network and on the disk.
    /// With parts, what is on disk under a part's name is always complete (the commit is a rename), an interruption loses at most the part being written,
    /// a resume continues after the last committed part without copying anything, and the only swap file is one part long. A complete entry is read as
    /// one Blob over its parts (no copy), so a model never exists twice.
    /// </remarks>
    public long PartBytes { get; set; } = 128L * 1024 * 1024;

    /// <summary>
    /// Swap files older than this are left-overs of a session that ended while it wrote (nothing else removes them) and are deleted when the cache opens
    /// (<see cref="SweepOrphansAsync()"/>). The age keeps the swap file of a write another tab is doing right now. Default 10 minutes.
    /// </summary>
    public TimeSpan OrphanSwapMinAge { get; set; } = TimeSpan.FromMinutes(10);


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
    public async Task<Stream?> OpenCachedAsync(string cacheKey, CancellationToken ct = default)
    {
        await EnsureInitializedAsync().ConfigureAwait(false);
        if (_cacheDir == null) return null;
        var meta = await ReadMetaAsync(cacheKey).ConfigureAwait(false);
        if (meta is not { Complete: true }) return null;
        Stream stream;
        long single = await GetSingleFileSizeAsync(cacheKey).ConfigureAwait(false);
        if (single >= 0)
        {
            // an entry stored as one file (written before parts)
            if (meta.Total >= 0 && meta.Total != single) return null;
            stream = await OPFSStream.OpenPath(_cacheDir!, cacheKey, FileMode.Open, FileAccess.Read, cancellationToken: ct).ConfigureAwait(false);
        }
        else
        {
            var files = await ReadPartFilesAsync(cacheKey).ConfigureAwait(false);
            if (files == null) return null;
            try
            {
                long sum = 0;
                foreach (var f in files) sum += f.Size;
                if (meta.Total >= 0 && meta.Total != sum) return null;
                // one Blob over the part files: a view, nothing is copied or read until a range is asked for
                stream = new BlobStream(files.Count == 0 ? new Blob() : new Blob((IEnumerable<Blob>)files));
            }
            finally { foreach (var f in files) f.Dispose(); }
        }
        await TouchAsync(cacheKey, meta).ConfigureAwait(false);
        return stream;
    }

    /// <summary>The key this instance last recorded a use for, and the value it wrote (see <see cref="TouchAsync"/>).</summary>
    private string? _lastTouchedKey;
    private long _lastTouchedUsed;

    /// <summary>How long a repeated use of the same entry may go unrecorded (bounds the drift when another tab records uses in between).</summary>
    private const long RepeatTouchWindowMs = 60_000;

    /// <summary>
    /// Record that a complete entry was just used (the order <see cref="TrimAsync"/> evicts in). Never fails the open it follows.
    /// The write is skipped when this instance's last recorded use was THIS entry, the stored value is still the one it wrote, and that was under
    /// <see cref="RepeatTouchWindowMs"/> ago: the entry is already the most recent, so the order cannot change. A model read per layer reopens its file
    /// every layer (the Wan umT5 encoder 25 times a prompt), and rewriting the sidecar was about half of each reopen. Any other entry used in between
    /// makes the next use of this one write again, so a cache hit always counts (RangeStreamModelSource_CacheCapEvictsLeastRecentlyUsedOutsideTheSet).
    /// </summary>
    private async Task TouchAsync(string cacheKey, CacheEntryMeta meta)
    {
        try
        {
            long now = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            if (cacheKey == _lastTouchedKey && meta.Used == _lastTouchedUsed && now - meta.Used is >= 0 and < RepeatTouchWindowMs) return;
            meta.Used = now;
            await WriteMetaAsync(cacheKey, meta).ConfigureAwait(false);
            _lastTouchedKey = cacheKey; _lastTouchedUsed = now;
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
        foreach (var (name, size) in await ListEntriesAsync().ConfigureAwait(false))
        {
            CacheEntryMeta? meta;
            try { meta = await ReadMetaAsync(name).ConfigureAwait(false); }
            catch
            {
                // a sidecar that cannot be read right now says nothing about the model: keep it (counted, never evicted), as a held file would be
                pinned.Add(name);
                entries.Add((name, size, true, long.MaxValue));
                continue;
            }
            bool complete = meta is { Complete: true } && (meta.Total < 0 || meta.Total == size);
            entries.Add((name, size, complete, complete ? meta!.Used : 0));
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
    public async Task<Stream> OpenOrDownloadAsync(string url, string? cacheKey = null,
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
            return await Downloader.GetOrDownloadAsync(url, cacheKey, progress, ct).ConfigureAwait(false);
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
            return await Downloader.GetOrDownloadAsync(url, cacheKey, progress, ct).ConfigureAwait(false);
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
        bool single = await GetSingleFileSizeAsync(key).ConfigureAwait(false) >= 0;
        var onDisk = await GetEntrySizeAsync(key).ConfigureAwait(false);
        if (onDisk < 0 && meta == null) return default;

        // What a download may continue from is the COMMITTED parts, never the sidecar's count: a part exists under its name only once it was closed
        // (the commit is a rename), while the sidecar is written per received segment and runs ahead of the disk by up to one part - and by the whole
        // download for a single file, whose bytes sat in a swap file until close (see PartBytes). Without a sidecar the source of the bytes is unknown,
        // so nothing continues. An incomplete single file (an interrupted download of an earlier version) is replaced, not continued.
        var written = meta == null || single ? 0 : Math.Max(0, onDisk);
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

        return await OpenPartedWriteAsync(_cacheDir, key, startOffset).ConfigureAwait(false);
    }

    /// <summary>
    /// The writer of an entry's part files, continuing after the committed parts that make up exactly <paramref name="startOffset"/> bytes. Everything else
    /// in the entry's directory goes first: parts past that point were never acknowledged, and swap files are left-overs of a session that ended while it
    /// wrote. The entry's single file of an earlier version is removed (it is only ever replaced).
    /// </summary>
    private async Task<Stream> OpenPartedWriteAsync(FileSystemDirectoryHandle cacheDir, string key, long startOffset)
    {
        try { await cacheDir.RemoveEntry(key).ConfigureAwait(false); } catch { /* no single file */ }
        var partsDir = await cacheDir.GetDirectoryHandle(key + PartsSuffix, create: true).ConfigureAwait(false);
        try
        {
            var sizes = new Dictionary<string, long>(StringComparer.Ordinal);
            var names = new List<string>();
            foreach (var entry in await partsDir.ValuesList().ConfigureAwait(false))
            {
                using var handle = entry;
                names.Add(handle.Name);
                if (handle is FileSystemFileHandle file && IsPartName(handle.Name))
                    try { sizes[handle.Name] = await file.GetSize().ConfigureAwait(false); } catch { /* raced with a delete */ }
            }
            long sum = 0; int keep = 0;
            while (sum < startOffset && sizes.TryGetValue(PartName(keep), out long size) && size > 0 && sum + size <= startOffset) { sum += size; keep++; }
            // Never continue past what is really there: when the parts vanished after the caller read the state (the browser cleared the origin), writing
            // on at the resume point would leave a hole that later counts as downloaded bytes.
            if (sum != startOffset)
                throw new StoredFileLostException($"'{key}' holds {sum} committed bytes at a part boundary but was to continue at {startOffset}: the stored parts are gone or were cut, so the download must restart.");
            foreach (var name in names)
            {
                if (IsPartName(name) && int.Parse(name, CultureInfo.InvariantCulture) < keep) continue;
                try { await partsDir.RemoveEntry(name, recursive: true).ConfigureAwait(false); } catch { /* held by a live writer elsewhere */ }
            }
            return new PartedWriteStream(partsDir, keep, startOffset, Math.Max(1, PartBytes));
        }
        catch { partsDir.Dispose(); throw; }
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
            var dest = await WithDirectoryAsync(d => OpenPartedWriteAsync(d, key, 0)).ConfigureAwait(false);
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
        try { await _cacheDir.RemoveEntry(cacheKey + PartsSuffix, recursive: true); } catch { /* absent */ }
        try { await _cacheDir.RemoveEntry(cacheKey + MetaSuffix); } catch { /* absent */ }
    }

    /// <summary>Every cached entry with its real on-disk size and whether it is complete.</summary>
    public async Task<List<(string Key, long SizeBytes, bool Complete)>> ListCachedAsync()
    {
        var result = new List<(string, long, bool)>();
        await EnsureInitializedAsync().ConfigureAwait(false);
        if (_cacheDir == null) return result;

        foreach (var (name, size) in await ListEntriesAsync().ConfigureAwait(false))
            result.Add((name, size, await IsCompleteAsync(name).ConfigureAwait(false)));
        return result;
    }

    /// <summary>Total bytes held by the cache: every file in it, sidecars, parts and swap files included.</summary>
    public async Task<long> GetCacheSizeAsync()
    {
        await EnsureInitializedAsync().ConfigureAwait(false);
        if (_cacheDir == null) return 0;

        long total = 0;
        var entries = await WithDirectoryAsync(d => d.ValuesList()).ConfigureAwait(false);
        foreach (var entry in entries)
        {
            using var handle = entry;
            try
            {
                if (handle is FileSystemFileHandle fileHandle) total += await fileHandle.GetSize();
                else if (handle is FileSystemDirectoryHandle sub)
                    foreach (var inner in await sub.ValuesList().ConfigureAwait(false))
                    {
                        using var h = inner;
                        if (h is FileSystemFileHandle f) try { total += await f.GetSize(); } catch { /* raced with a delete */ }
                    }
            }
            catch { /* raced with a delete */ }
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
            try { await _cacheDir.RemoveEntry(handle.Name, recursive: true); } catch { /* raced with a delete */ }
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

    private static string PartName(int index) => index.ToString("D6", CultureInfo.InvariantCulture);

    private static bool IsPartName(string name)
    {
        if (name.Length != 6) return false;
        foreach (char c in name) if (c < '0' || c > '9') return false;
        return true;
    }

    /// <summary>The size of an entry stored as ONE file (entries written before parts), or -1 when there is no such file.</summary>
    private async Task<long> GetSingleFileSizeAsync(string cacheKey)
    {
        try
        {
            using var handle = await _cacheDir!.GetFileHandle(cacheKey);
            return await handle.GetSize();
        }
        catch { return -1; }
    }

    /// <summary>
    /// The committed parts of an entry in order, as File objects the caller disposes; null when the entry has no parts directory. Parts count from
    /// index 0 up to the first one missing or empty: a part under its name is always whole (see <see cref="PartBytes"/>), an empty one was created and
    /// never committed.
    /// </summary>
    private async Task<List<SpawnDev.SpawnJS.JSObjects.File>?> ReadPartFilesAsync(string cacheKey)
    {
        FileSystemDirectoryHandle dir;
        try { dir = await _cacheDir!.GetDirectoryHandle(cacheKey + PartsSuffix).ConfigureAwait(false); }
        catch { return null; }
        // ONE listing, then every part's File at once. It was two awaited calls per part in sequence: MEASURED 2026-11-27, a complete 48-part entry
        // (the Wan umT5 file is 48 parts of 128 MiB, and its encoder reopens it once per layer, 25 times a prompt) took 24.7 ms to list and 54 ms to open
        // (Benchmark_OpfsModelCache_ReopenAPartedEntry).
        var files = new List<SpawnDev.SpawnJS.JSObjects.File>();
        using (dir)
        {
            var byIndex = new Dictionary<int, FileSystemFileHandle>();
            try
            {
                foreach (var entry in await dir.ValuesList().ConfigureAwait(false))
                {
                    if (entry is FileSystemFileHandle fh && IsPartName(fh.Name)) byIndex[int.Parse(fh.Name, CultureInfo.InvariantCulture)] = fh;
                    else entry.Dispose();
                }
                var ordered = new List<FileSystemFileHandle>();
                for (int i = 0; byIndex.TryGetValue(i, out var h); i++) ordered.Add(h);           // consecutive from 0: the first missing index ends the entry
                var got = await Task.WhenAll(ordered.Select(async h =>
                {
                    try { return await h.GetFile().ConfigureAwait(false); } catch { return null; }  // raced with a delete
                })).ConfigureAwait(false);
                bool ended = false;
                foreach (var f in got)
                {
                    if (ended || f == null || f.Size <= 0) { ended = true; f?.Dispose(); continue; }   // an empty part was created and never committed
                    files.Add(f);
                }
            }
            finally { foreach (var h in byIndex.Values) h.Dispose(); }
        }
        return files;
    }

    /// <summary>The bytes an entry really holds: its single file, else the sum of its committed parts; -1 when it has neither.</summary>
    private async Task<long> GetEntrySizeAsync(string cacheKey)
    {
        long single = await GetSingleFileSizeAsync(cacheKey).ConfigureAwait(false);
        if (single >= 0) return single;
        var files = await ReadPartFilesAsync(cacheKey).ConfigureAwait(false);
        if (files == null) return -1;
        long sum = 0;
        foreach (var f in files) { sum += f.Size; f.Dispose(); }
        return sum;
    }

    /// <summary>Every entry with the bytes it holds: single files (sidecars and swap files are not entries) and parts directories.</summary>
    private async Task<List<(string Key, long Size)>> ListEntriesAsync()
    {
        var result = new List<(string, long)>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var handles = await WithDirectoryAsync(d => d.ValuesList()).ConfigureAwait(false);
        foreach (var entry in handles)
        {
            using var handle = entry;
            string name = handle.Name;
            if (handle is FileSystemFileHandle file)
            {
                if (name.EndsWith(MetaSuffix, StringComparison.Ordinal) || name.EndsWith(SwapSuffix, StringComparison.Ordinal)) continue;
                try { long size = await file.GetSize(); if (seen.Add(name)) result.Add((name, size)); } catch { /* raced with a delete */ }
            }
            else if (name.EndsWith(PartsSuffix, StringComparison.Ordinal))
            {
                string key = name[..^PartsSuffix.Length];
                long size = await GetEntrySizeAsync(key).ConfigureAwait(false);
                if (seen.Add(key)) result.Add((key, Math.Max(0, size)));
            }
        }
        return result;
    }

    /// <summary>
    /// Delete the swap files a session left behind when it ended while it wrote (a closed browser, a crash): Chrome keeps a
    /// <c>FileSystemWritableFileStream</c>'s bytes in <c>name.crswap</c> until close, and nothing removes the file of a writer that is gone. Before parts a
    /// swap file held a whole interrupted download (MEASURED: 1.02 GB left beside a finished 13.4 GB model set). Only files older than
    /// <see cref="OrphanSwapMinAge"/> go, so a write another tab is doing now keeps its swap file. Runs when the cache opens; returns the bytes freed.
    /// </summary>
    public async Task<long> SweepOrphansAsync()
    {
        await EnsureInitializedAsync().ConfigureAwait(false);
        return _cacheDir == null ? 0 : await SweepOrphansAsync(_cacheDir).ConfigureAwait(false);
    }

    private async Task<long> SweepOrphansAsync(FileSystemDirectoryHandle cacheDir)
    {
        long freed = 0;
        long cutoff = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - (long)OrphanSwapMinAge.TotalMilliseconds;
        async Task SweepAsync(FileSystemDirectoryHandle dir, bool recurse)
        {
            foreach (var entry in await dir.ValuesList().ConfigureAwait(false))
            {
                using var handle = entry;
                if (handle is FileSystemFileHandle file)
                {
                    if (!handle.Name.EndsWith(SwapSuffix, StringComparison.Ordinal)) continue;
                    try
                    {
                        long size, modified;
                        using (var f = await file.GetFile().ConfigureAwait(false)) { size = f.Size; modified = f.LastModified; }
                        if (modified > cutoff) continue;
                        await dir.RemoveEntry(handle.Name).ConfigureAwait(false);
                        freed += size;
                    }
                    catch { /* held by a live writer, or gone already */ }
                }
                else if (recurse && handle is FileSystemDirectoryHandle sub && handle.Name.EndsWith(PartsSuffix, StringComparison.Ordinal))
                    await SweepAsync(sub, false).ConfigureAwait(false);
            }
        }
        try { await SweepAsync(cacheDir, true).ConfigureAwait(false); } catch { /* a sweep never fails the cache */ }
        return freed;
    }

    /// <summary>
    /// Writes an entry as part files: each part is its own file, committed (closed) when it reaches the part size and when the stream is disposed.
    /// See <see cref="PartBytes"/> for why. The payload stays in JS: a chunk is written as it is, or as subarray views when it crosses a part boundary.
    /// </summary>
    private sealed class PartedWriteStream : JSWriteStreamBase
    {
        private FileSystemDirectoryHandle? _dir;
        private readonly long _partBytes;
        private int _index;
        private long _length, _partFill;
        private FileSystemFileHandle? _partHandle;
        private FileSystemHandleWritableStream? _part;
        private bool _closed;

        public PartedWriteStream(FileSystemDirectoryHandle dir, int nextIndex, long startOffset, long partBytes)
        {
            _dir = dir; _index = nextIndex; _length = startOffset; _partBytes = partBytes;
        }

        public override bool CanWriteSync => false;
        public override bool CanRead => false;
        public override bool CanSeek => false;
        public override bool CanWrite => !_closed;
        public override long Length => _length;
        public override long Position { get => _length; set => throw new NotSupportedException("a parted entry is written front to back"); }

        public override async Task WriteUint8ArrayAsync(Uint8Array data, CancellationToken cancellationToken = default)
        {
            if (_closed || _dir == null) throw new ObjectDisposedException(nameof(PartedWriteStream));
            long len = data.Length, off = 0;
            while (off < len)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (_part == null)
                {
                    _partHandle = await _dir.GetFileHandle(PartName(_index), create: true).ConfigureAwait(false);
                    _part = await FileSystemHandleWritableStream.Create(_partHandle, false, true).ConfigureAwait(false);
                    _partFill = 0;
                }
                long take = Math.Min(len - off, _partBytes - _partFill);
                if (off == 0 && take == len) await _part.WriteAsync(data, cancellationToken).ConfigureAwait(false);
                else
                {
                    using var view = data.SubArray(off, off + take);     // a view on the same storage, nothing is copied
                    await _part.WriteAsync(view, cancellationToken).ConfigureAwait(false);
                }
                off += take; _partFill += take; _length += take;
                if (_partFill >= _partBytes) await CommitPartAsync().ConfigureAwait(false);
            }
        }

        /// <summary>Close the part being written: from here its bytes are on disk under the part's name and count as stored.</summary>
        private async Task CommitPartAsync()
        {
            var part = _part; var handle = _partHandle;
            if (part == null) return;
            _part = null; _partHandle = null;
            try { await part.CloseAsync().ConfigureAwait(false); }
            finally { part.Dispose(); handle?.Dispose(); }
            _index++;
        }

        public override void WriteUint8Array(Uint8Array data) => throw new NotSupportedException("OPFS writes are asynchronous: use WriteUint8ArrayAsync");

        public override async ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken = default)
        {
            if (buffer.Length == 0) return;
            using var bytes = new Uint8Array(buffer.ToArray());       // a managed source (a stream of .NET bytes handed to PutAsync): the only way in
            await WriteUint8ArrayAsync(bytes, cancellationToken).ConfigureAwait(false);
        }

        public override Task WriteAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken)
            => WriteAsync(new ReadOnlyMemory<byte>(buffer, offset, count), cancellationToken).AsTask();

        /// <summary>Nothing to do: bytes become durable when their part is committed, and the store reports only committed parts as written.</summary>
        public override Task FlushAsync(CancellationToken cancellationToken) => Task.CompletedTask;
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException("a parted entry is written front to back");
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException("OPFS writes are asynchronous: use WriteAsync");

        public override async ValueTask DisposeAsync()
        {
            if (_closed) return;
            _closed = true;
            // commit what the last part holds: a cancelled or failed download keeps everything it wrote
            try { await CommitPartAsync().ConfigureAwait(false); } catch { /* the failure that ended the write is the one to report */ }
            _dir?.Dispose(); _dir = null;
            GC.SuppressFinalize(this);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing && !_closed) _ = DisposeAsync().AsTask();      // the commit cannot be awaited here: use await using
            base.Dispose(disposing);
        }
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
            if (_cacheDir != null) await SweepOrphansAsync(_cacheDir).ConfigureAwait(false);
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
