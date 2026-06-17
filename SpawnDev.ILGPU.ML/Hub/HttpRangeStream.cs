using System.Net.Http.Headers;

namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// A seekable, read-only <see cref="Stream"/> over an HTTP resource that supports byte-range requests — the
/// hub's <c>/hf</c> | <c>/ollama</c> web seed, which now ALWAYS serves (fetching + caching the missing chunks
/// from the origin on demand). Each read issues a Range GET for exactly the bytes at the current position, so a
/// model load — which seeks past weights and reads structure via <c>ReadAsync</c> — pulls only what it touches.
/// This is the raw-HTTP "cold load" path: usable the instant the hub is asked, before any <c>.torrent</c>/P2P
/// swarm exists (a warm load uses the torrent instead).
///
/// ASYNC-ONLY: in the browser (WASM) the single thread cannot block on async, so a synchronous <see cref="Read"/>
/// would deadlock — it throws. The ONNX stream reader (and the inspector) read via <c>ReadAsync</c>, which is
/// the supported path.
/// </summary>
internal sealed class HttpRangeStream : Stream
{
    private readonly HttpClient _http;
    private readonly string _url;
    private readonly long _length;
    private long _position;

    public HttpRangeStream(HttpClient http, string url, long length)
    {
        _http = http ?? throw new ArgumentNullException(nameof(http));
        _url = url ?? throw new ArgumentNullException(nameof(url));
        _length = length;
    }

    public override bool CanRead => true;
    public override bool CanSeek => true;
    public override bool CanWrite => false;
    public override long Length => _length;
    public override long Position { get => _position; set => _position = value; }

    public override long Seek(long offset, SeekOrigin origin)
    {
        _position = origin switch
        {
            SeekOrigin.Begin => offset,
            SeekOrigin.Current => _position + offset,
            SeekOrigin.End => _length + offset,
            _ => _position,
        };
        return _position;
    }

    public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken ct = default)
    {
        if (_position >= _length || buffer.Length == 0) return 0;
        long end = Math.Min(_position + buffer.Length, _length) - 1;   // clamp to EOF (RFC 7233)
        using var req = new HttpRequestMessage(HttpMethod.Get, _url);
        req.Headers.Range = new RangeHeaderValue(_position, end);
        using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
        resp.EnsureSuccessStatusCode();
        var bytes = await resp.Content.ReadAsByteArrayAsync(ct).ConfigureAwait(false);
        int n = Math.Min(bytes.Length, buffer.Length);
        bytes.AsSpan(0, n).CopyTo(buffer.Span);
        _position += n;
        return n;
    }

    public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken ct)
        => ReadAsync(buffer.AsMemory(offset, count), ct).AsTask();

    public override int Read(byte[] buffer, int offset, int count)
        => throw new NotSupportedException("HttpRangeStream is async-only (a sync Read deadlocks in WASM); use ReadAsync.");

    public override void Flush() { }
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
}
