namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// An <see cref="IModelSource"/> for model sets larger than the origin's storage quota (Chrome grants an origin a share of the free disk:
/// 11 GB on a machine with 80 GB free), where caching everything into OPFS would fail part-way. Per file: a file already in the hub cache
/// opens from OPFS; a file in <see cref="CacheFiles"/> downloads into OPFS once (<see cref="HubModelSource.OpenAsync(string, string, CancellationToken)"/>);
/// every other file streams from the hub with HTTP range requests on every load and stores NOTHING. <see cref="PlanCacheAsync"/> picks the
/// files to cache so that the fewest bytes stream. Readers should read in large chunks (each read is one request).
/// </summary>
public sealed class RangeStreamModelSource : IModelSource
{
    private readonly HubModelSource _hub;

    /// <param name="hub">A hub source constructed WITH an HttpClient (size probes and desktop range requests go through it).</param>
    public RangeStreamModelSource(HubModelSource hub) => _hub = hub ?? throw new ArgumentNullException(nameof(hub));

    /// <summary>(bytes read so far, file length) while a file streams, for a progress display. Cached downloads report through the hub's OnProgress.</summary>
    public event Action<long, long>? OnProgress;

    /// <summary>"repo/path" of the files that are downloaded into the hub cache instead of streamed.</summary>
    public HashSet<string> CacheFiles { get; } = new(StringComparer.Ordinal);

    private static string Key(string repoId, string filePath) => repoId.Trim('/') + "/" + filePath.TrimStart('/');

    public async Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
    {
        if (CacheFiles.Contains(Key(repoId, filePath)) || await _hub.Store.ExistsAsync(_hub.CacheKey(repoId, filePath)).ConfigureAwait(false))
            return await _hub.OpenAsync(repoId, filePath, cancellationToken).ConfigureAwait(false);
        var probe = await _hub.OpenForInspectionAsync(repoId, filePath, cancellationToken).ConfigureAwait(false);
        if (!OperatingSystem.IsBrowser()) return new Counting(probe, this);
        // browser: fetch ranges that stay in JS (the loader uploads them to the GPU without touching the .NET heap)
        long length = probe.Length; probe.Dispose();
        return new FetchRangeStream(HuggingFaceClient.GetDownloadUrl(repoId, filePath, _hub.Revision), length) { Progress = (d, t) => OnProgress?.Invoke(d, t) };
    }

    /// <summary>
    /// Choose which of <paramref name="files"/> to cache: files already cached stay cached; of the rest, the subset with the most bytes that
    /// fits <paramref name="budgetBytes"/> (free quota minus a margin) goes into <see cref="CacheFiles"/> (exact search, a model set is a handful
    /// of files). Returns (bytes cached or to be cached, bytes that will stream on every load).
    /// </summary>
    public async Task<(long CachedBytes, long StreamedBytes)> PlanCacheAsync(IEnumerable<(string Repo, string Path)> files, long budgetBytes, CancellationToken cancellationToken = default)
    {
        CacheFiles.Clear();
        long cached = 0;
        var todo = new List<(string Key, long Bytes)>();
        foreach (var (repo, path) in files)
        {
            using var probe = await _hub.OpenForInspectionAsync(repo, path, cancellationToken).ConfigureAwait(false);
            if (await _hub.Store.ExistsAsync(_hub.CacheKey(repo, path)).ConfigureAwait(false)) cached += probe.Length;
            else todo.Add((Key(repo, path), probe.Length));
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
