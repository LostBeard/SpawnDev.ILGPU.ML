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

    /// <summary>Smallest fetch: reads are served from a read-ahead window of at least this many bytes (one request per window, not per small tensor).</summary>
    public int ReadAhead { get; set; } = 16 << 20;
    private Uint8Array? _window;
    private long _winStart, _winEnd;

    public async Task<Uint8Array> ReadUint8ArrayAsync(int count, CancellationToken cancellationToken = default)
    {
        if (_position >= _length || count <= 0) return new Uint8Array(0);
        // a request must come back WHOLE (stream-copy loaders treat a short read as end of stream), so a window that does not cover it is refetched
        long want = Math.Min(count, _length - _position);
        if (_window == null || _position < _winStart || _position + want > _winEnd)
        {
            _window?.Dispose(); _window = null;
            long start = _position, end = Math.Min(_position + Math.Max(count, ReadAhead), _length) - 1;
            _window = await FetchRangeAsync(start, end, cancellationToken);
            _winStart = start; _winEnd = start + _window.Length;
            BytesFetched += _window.Length;
            if (_window.Length == 0) return new Uint8Array(0);
        }
        long from = _position - _winStart, to = Math.Min(from + count, _winEnd - _winStart);
        var slice = _window.SubArray(from, to);          // a view on the same JS buffer: no copy
        _position += to - from;
        Progress?.Invoke(_position, _length);
        return slice;
    }

    protected override void Dispose(bool disposing) { if (disposing) { _window?.Dispose(); _window = null; } base.Dispose(disposing); }

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
