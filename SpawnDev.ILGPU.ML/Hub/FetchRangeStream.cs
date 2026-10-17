using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.SpawnJS.Toolbox;

namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// A read-only stream over an HTTP resource that reads with browser <c>fetch</c> Range requests and keeps the bytes in JS
/// (<see cref="IJSReadStream"/>): a loader that understands it hands each chunk straight to the GPU (<c>IBrowserMemoryBuffer.CopyFromJS</c>)
/// and the data never enters the .NET heap. MEASURED in Chrome against the hub: a 16 MB range in ~1.2 s with fetch, where the
/// HttpClient-based <see cref="HttpRangeStream"/> managed ~1.2 MB/s overall (its byte[] round trip through the WASM heap is the cost).
/// Browser only. <see cref="ReadAsync(Memory{byte}, CancellationToken)"/> still works (copies into .NET) for readers that do not.
/// </summary>
public sealed class FetchRangeStream : Stream, IJSReadStream
{
    private readonly string _url;
    private readonly long _length;
    private long _position;

    public FetchRangeStream(string url, long length) { _url = url; _length = length; }

    /// <summary>Bytes fetched so far (for progress).</summary>
    public long BytesFetched { get; private set; }
    /// <summary>(position, length) after every fetch.</summary>
    public Action<long, long>? Progress { get; set; }

    public override bool CanRead => true;
    public override bool CanSeek => true;
    public override bool CanWrite => false;
    public override long Length => _length;
    public override long Position { get => _position; set => _position = value; }
    public bool CanReadSync => false;

    public override long Seek(long offset, SeekOrigin origin)
    {
        _position = origin switch { SeekOrigin.Begin => offset, SeekOrigin.Current => _position + offset, SeekOrigin.End => _length + offset, _ => _position };
        return _position;
    }

    /// <summary>Size of one range request. Reads are served from these chunks (one request per chunk, not per small tensor).</summary>
    public int ReadAhead { get; set; } = 16 << 20;
    /// <summary>
    /// Chunks kept in flight ahead of the read position. MEASURED in Chrome against the hub, 16 MB ranges: one at a time 34.9 MB/s,
    /// 4 in parallel 53.1 MB/s, 8 in parallel 57.2 MB/s.
    /// </summary>
    public int Depth { get; set; } = 4;

    // consecutive chunks [Start, End) from the first one still needed to _nextFetch; later ones may still be in flight
    private readonly List<(long Start, long End, Task<Uint8Array> Data)> _chunks = new();
    private long _nextFetch;

    public async Task<Uint8Array> ReadUint8ArrayAsync(int count, CancellationToken cancellationToken = default)
    {
        if (_position >= _length || count <= 0) return new Uint8Array(0);
        // a request must come back WHOLE (stream-copy loaders treat a short read as end of stream)
        long want = Math.Min(count, _length - _position);
        if (_chunks.Count == 0 || _position < _chunks[0].Start || _position > _nextFetch) Restart(_position);   // a seek outside what is fetched
        while (_chunks.Count > 0 && _chunks[0].End <= _position) { Release(_chunks[0].Data); _chunks.RemoveAt(0); }
        while (_nextFetch < _position + want) Schedule(cancellationToken);   // a read larger than the queue
        while (_nextFetch < _length && _chunks.Count < Depth) Schedule(cancellationToken);

        Uint8Array result;
        var first = _chunks[0];
        if (_position + want <= first.End)
            result = (await first.Data).SubArray(_position - first.Start, _position - first.Start + want);   // a view: no copy
        else
        {   // spans chunks: assembled on the JS side (still never enters the .NET heap)
            result = new Uint8Array(want);
            long pos = _position, filled = 0;
            for (int i = 0; filled < want; i++)
            {
                var c = _chunks[i];
                long from = pos - c.Start, to = Math.Min(c.End - c.Start, from + (want - filled));
                using var part = (await c.Data).SubArray(from, to);
                result.Set(part, filled);
                filled += to - from; pos += to - from;
            }
        }
        _position += want;
        Progress?.Invoke(_position, _length);
        return result;
    }

    private void Schedule(CancellationToken ct)
    {
        long s = _nextFetch, e = Math.Min(s + ReadAhead, _length);
        _chunks.Add((s, e, FetchRangeAsync(s, e - 1, ct)));
        _nextFetch = e;
        BytesFetched += e - s;
    }

    private void Restart(long position)
    {
        foreach (var c in _chunks) Release(c.Data);
        _chunks.Clear();
        _nextFetch = position;
    }

    // a chunk still in flight is released when it lands
    private static void Release(Task<Uint8Array> t) => _ = t.ContinueWith(d => { if (d.Status == TaskStatus.RanToCompletion) d.Result.Dispose(); }, TaskScheduler.Default);

    protected override void Dispose(bool disposing) { if (disposing) Restart(0); base.Dispose(disposing); }

    /// <summary>Per-request timeout (a stalled connection is aborted and retried) and attempts.</summary>
    public TimeSpan RequestTimeout { get; set; } = TimeSpan.FromSeconds(90);
    public int Attempts { get; set; } = 4;

    private async Task<Uint8Array> FetchRangeAsync(long start, long end, CancellationToken ct)
    {
        Exception? last = null;
        for (int attempt = 1; attempt <= Attempts; attempt++)
        {
            ct.ThrowIfCancellationRequested();
            try
            {
                using var window = SpawnJSRuntime.Instance!.Get<Window>("window");
                using var signal = AbortSignal.Timeout((float)RequestTimeout.TotalMilliseconds);   // aborts the request AND the body read
                using var resp = await window.Fetch(_url, new FetchOptions { Headers = new() { ["Range"] = $"bytes={start}-{end}" }, Signal = signal });
                if (!resp.Ok) throw new IOException($"GET {_url} bytes={start}-{end}: HTTP {resp.Status} {resp.StatusText}");
                using var ab = await resp.ArrayBuffer();
                var arr = new Uint8Array(ab);
                if (arr.Length != end - start + 1) { arr.Dispose(); throw new IOException($"GET {_url} bytes={start}-{end}: short read"); }
                return arr;
            }
            catch (Exception ex) when (ex is not OperationCanceledException || !ct.IsCancellationRequested)
            {
                last = ex;
                if (attempt < Attempts) await Task.Delay(1000 * attempt, ct);
            }
        }
        throw new IOException($"GET {_url} bytes={start}-{end} failed {Attempts} times", last);
    }

    public Uint8Array ReadUint8Array(int count) => throw new NotSupportedException("FetchRangeStream is async-only (browser fetch); use ReadUint8ArrayAsync.");

    public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken ct = default)
    {
        using var arr = await ReadUint8ArrayAsync(buffer.Length, ct);
        var bytes = arr.ReadBytes();
        bytes.AsSpan().CopyTo(buffer.Span);
        return bytes.Length;
    }

    public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken ct) => ReadAsync(buffer.AsMemory(offset, count), ct).AsTask();
    public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException("FetchRangeStream is async-only; use ReadAsync.");
    public override void Flush() { }
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
}
