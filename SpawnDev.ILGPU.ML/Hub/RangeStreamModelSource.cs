using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// An <see cref="IModelSource"/> for model sets that may be larger than the room a browser origin has, where caching everything into OPFS
/// could fail part-way. Per file: a file already in the hub cache opens from OPFS; a file in <see cref="CacheFiles"/> downloads into OPFS
/// once (<see cref="HubModelSource.OpenAsync(string, string, CancellationToken)"/>); every other file streams from the hub with HTTP range
/// requests on every load and stores NOTHING. <see cref="PlanCacheAsync(IEnumerable{ValueTuple{string, string}}, BrowserStorageEstimate?, long, CancellationToken)"/>
/// picks the files to cache so that the fewest bytes stream. Readers should read in large chunks (each read is one request).
/// <para>
/// ⚠️ <c>navigator.storage.estimate()</c> is not a trustworthy limit (see <see cref="BrowserStorageEstimate"/>: Chrome reports usage + 10 GiB
/// and takes more). The plan therefore trusts only an exact headroom; when the headroom is open-ended every file is planned for the cache,
/// and a file whose write the browser really refuses (<c>QuotaExceededError</c>) drops its half-written entry and streams instead
/// (<see cref="CacheFallback"/>), so the load still succeeds.
/// </para>
/// </summary>
public class RangeStreamModelSource : IModelSource
{
    private readonly HubModelSource _hub;

    /// <param name="hub">A hub source constructed WITH an HttpClient (size probes and desktop range requests go through it).</param>
    public RangeStreamModelSource(HubModelSource hub) => _hub = hub ?? throw new ArgumentNullException(nameof(hub));

    /// <summary>(bytes read so far, file length) while a file streams, for a progress display. Cached downloads report through the source's OnProgress.</summary>
    public event Action<long, long>? OnProgress;

    /// <summary>"repo/path" of the files that are downloaded into the hub cache instead of streamed.</summary>
    public HashSet<string> CacheFiles { get; } = new(StringComparer.Ordinal);

    private static string Key(string repoId, string filePath) => repoId.Trim('/') + "/" + filePath.TrimStart('/');

    /// <summary>Raised with "repo/path" when the browser refused to store a file planned for the cache; that file streams from now on.</summary>
    public event Action<string>? CacheFallback;

    /// <summary>True when <paramref name="ex"/> (or anything it wraps) is the browser's <c>QuotaExceededError</c>.</summary>
    public static bool IsQuotaExceeded(Exception? ex)
    {
        for (; ex != null; ex = ex.InnerException)
            if (ex is JSException { Name: "QuotaExceededError" }) return true;
        return false;
    }

    /// <summary>Download a file planned for the cache into OPFS and open it. Virtual so a test can make the browser's refusal happen on demand (a full disk cannot be summoned).</summary>
    protected virtual Task<Stream> DownloadIntoCacheAsync(string repoId, string filePath, CancellationToken cancellationToken)
        => _hub.OpenAsync(repoId, filePath, cancellationToken);

    /// <summary>The cached copy's length, else the host's size for the file, without downloading it (<see cref="HubModelSource.GetSizeAsync"/>).</summary>
    public Task<long?> GetSizeAsync(string repoId, string filePath, CancellationToken cancellationToken = default) => _hub.GetSizeAsync(repoId, filePath, cancellationToken);

    public async Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
    {
        var cacheKey = _hub.CacheKey(repoId, filePath);
        if (await _hub.Store.ExistsAsync(cacheKey).ConfigureAwait(false))
            return await _hub.OpenAsync(repoId, filePath, cancellationToken).ConfigureAwait(false);
        if (CacheFiles.Contains(Key(repoId, filePath)))
        {
            try { return await DownloadIntoCacheAsync(repoId, filePath, cancellationToken).ConfigureAwait(false); }
            catch (Exception ex) when (IsQuotaExceeded(ex))
            {
                // The estimate promised room and the write was refused anyway. The half-written entry still holds quota: drop it, stream this file.
                CacheFiles.Remove(Key(repoId, filePath));
                try { await _hub.Store.RemoveAsync(cacheKey).ConfigureAwait(false); } catch { /* nothing left to remove */ }
                CacheFallback?.Invoke(Key(repoId, filePath));
            }
        }
        var probe = await _hub.OpenForInspectionAsync(repoId, filePath, cancellationToken).ConfigureAwait(false);
        if (!OperatingSystem.IsBrowser()) return new Counting(probe, this);
        // browser: fetch ranges that stay in JS (the loader uploads them to the GPU without touching the .NET heap)
        long length = probe.Length; probe.Dispose();
        var url = await ModelsHost.PreferredUrlAsync(HuggingFaceClient.GetDownloadUrl(repoId, filePath, _hub.Revision)).ConfigureAwait(false);   // not mirrored: Hugging Face directly
        return new FetchRangeStream(url, length) { Progress = (d, t) => OnProgress?.Invoke(d, t) };
    }

    /// <summary>
    /// Choose which of <paramref name="files"/> to cache: files already cached stay cached; of the rest, the subset with the most bytes that
    /// fits <paramref name="budgetBytes"/> (free quota minus a margin) goes into <see cref="CacheFiles"/> (exact search, a model set is a handful
    /// of files). Returns (bytes cached or to be cached, bytes that will stream on every load).
    /// </summary>
    public async Task<(long CachedBytes, long StreamedBytes)> PlanCacheAsync(IEnumerable<(string Repo, string Path)> files, long budgetBytes, CancellationToken cancellationToken = default)
    {
        return await PlanCoreAsync(files, budgetBytes, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// <see cref="PlanCacheAsync(IEnumerable{ValueTuple{string, string}}, long, CancellationToken)"/> with the budget taken from the browser's
    /// storage estimate: an exact headroom (minus <paramref name="marginBytes"/>) bounds the plan, an open-ended one
    /// (<see cref="BrowserStorageEstimate.HeadroomIsOpenEnded"/>) or an unknown one (null) plans every file for the cache.
    /// </summary>
    public Task<(long CachedBytes, long StreamedBytes)> PlanCacheAsync(IEnumerable<(string Repo, string Path)> files, BrowserStorageEstimate? estimate,
        long marginBytes = BrowserStorageEstimate.DefaultMarginBytes, CancellationToken cancellationToken = default)
        => PlanCoreAsync(files, estimate?.PlanningBudget(marginBytes) ?? long.MaxValue / 2, cancellationToken);

    /// <summary>
    /// The most model bytes the hub cache may hold, counted from its OWN files; null (the default) = no cap, the browser decides. When set,
    /// <see cref="PlanCacheAsync(IEnumerable{ValueTuple{string, string}}, long, CancellationToken)"/> plans only what fits under the cap next to
    /// the requested files already cached, and evicts least-recently-used entries OUTSIDE the requested set to make that room
    /// (<see cref="OpfsModelCache.TrimAsync"/>); files that do not fit stream.
    /// </summary>
    /// <remarks>
    /// Why an app-set cap and not the browser's numbers: Chrome's quota is a sliding window (see <see cref="BrowserStorageEstimate"/>) and its
    /// usage figure can miss most of what OPFS really holds - MEASURED 2026-11-24 (Lore, Chrome on Windows): 42.25 GB in the cache's files,
    /// <c>estimate()</c> usage 3.60 GB of a 14.3 GB quota. Without a cap the cache grows until the browser clears the WHOLE origin under disk
    /// pressure (it did, mid-session), which costs every model instead of the least recently used one.
    /// </remarks>
    public long? MaxCacheBytes { get; set; }

    private async Task<(long CachedBytes, long StreamedBytes)> PlanCoreAsync(IEnumerable<(string Repo, string Path)> files, long budgetBytes, CancellationToken cancellationToken)
    {
        CacheFiles.Clear();
        long cached = 0;
        var todo = new List<(string Key, long Bytes)>();
        var requestedKeys = new List<string>();
        foreach (var (repo, path) in files)
        {
            using var probe = await _hub.OpenForInspectionAsync(repo, path, cancellationToken).ConfigureAwait(false);
            var cacheKey = _hub.CacheKey(repo, path);
            requestedKeys.Add(cacheKey);
            if (await _hub.Store.ExistsAsync(cacheKey).ConfigureAwait(false)) cached += probe.Length;
            else todo.Add((Key(repo, path), probe.Length));
        }
        if (MaxCacheBytes is long cap)
        {
            if (cap < 0) throw new InvalidOperationException($"{nameof(MaxCacheBytes)} must not be negative");
            budgetBytes = Math.Min(budgetBytes, Math.Max(0, cap - cached));   // everything outside the requested set can be evicted for it
        }
        if (todo.Count > 20) throw new ArgumentException("PlanCacheAsync searches subsets exactly: at most 20 uncached files", nameof(files));
        int best = 0; long bestBytes = 0;
        for (int mask = 1; mask < 1 << todo.Count; mask++)
        {
            long sum = 0;
            for (int i = 0; i < todo.Count; i++) if ((mask >> i & 1) != 0) sum += todo[i].Bytes;
            if (sum <= budgetBytes && sum > bestBytes) { best = mask; bestBytes = sum; }
        }
        long streamed = 0;
        for (int i = 0; i < todo.Count; i++)
            if ((best >> i & 1) != 0) CacheFiles.Add(todo[i].Key); else streamed += todo[i].Bytes;
        // make the room now, before any planned download starts: the requested files (cached, partial or planned) are never evicted
        if (MaxCacheBytes is long max)
            await _hub.Cache.TrimAsync(Math.Max(0, max - bestBytes), requestedKeys, cancellationToken).ConfigureAwait(false);
        return (cached + bestBytes, streamed);
    }

    private sealed class Counting(HttpRangeStream inner, RangeStreamModelSource owner) : Stream
    {
        public override bool CanRead => true;
        public override bool CanSeek => true;
        public override bool CanWrite => false;
        public override long Length => inner.Length;
        public override long Position { get => inner.Position; set => inner.Position = value; }
        public override long Seek(long offset, SeekOrigin origin) => inner.Seek(offset, origin);
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count) => inner.Read(buffer, offset, count);
        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken ct = default)
        {
            int n = await inner.ReadAsync(buffer, ct).ConfigureAwait(false);
            owner.OnProgress?.Invoke(inner.Position, inner.Length);
            return n;
        }
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        protected override void Dispose(bool disposing) { if (disposing) inner.Dispose(); base.Dispose(disposing); }
    }

    public async Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
    {
        await using var s = await OpenAsync(repoId, filePath, cancellationToken).ConfigureAwait(false);
        var buf = new byte[s.Length];
        await s.ReadExactlyAsync(buf, cancellationToken).ConfigureAwait(false);
        return buf;
    }
}
