namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// An <see cref="IModelSource"/> that reads model files straight from the hub with HTTP range requests (<see cref="HttpRangeStream"/>) and
/// keeps NOTHING in browser storage. For models larger than the origin's storage quota (Chrome grants an origin a share of the free disk:
/// 10 GB on a machine with 80 GB free), where caching into OPFS would fail part-way: every load streams again, the hub's own cache
/// serves it. Readers should read in large chunks (each read is one request).
/// </summary>
public sealed class RangeStreamModelSource : IModelSource
{
    private readonly HubModelSource _hub;

    /// <param name="hub">A hub source constructed WITH an HttpClient (range requests go through it).</param>
    public RangeStreamModelSource(HubModelSource hub) => _hub = hub ?? throw new ArgumentNullException(nameof(hub));

    /// <summary>(bytes read so far, file length) while a file streams, for a progress display.</summary>
    public event Action<long, long>? OnProgress;

    public async Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
    {
        var probe = await _hub.OpenForInspectionAsync(repoId, filePath, cancellationToken).ConfigureAwait(false);
        if (!OperatingSystem.IsBrowser()) return new Counting(probe, this);
        // browser: fetch ranges that stay in JS (the loader uploads them to the GPU without touching the .NET heap)
        long length = probe.Length; probe.Dispose();
        return new FetchRangeStream(HuggingFaceClient.GetDownloadUrl(repoId, filePath, _hub.Revision), length) { Progress = (d, t) => OnProgress?.Invoke(d, t) };
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
