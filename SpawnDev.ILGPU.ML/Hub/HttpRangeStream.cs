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
public sealed class HttpRangeStream : Stream
{
    private readonly HttpClient _http;
    private readonly string _url;
    private readonly long _length;
    private long _position;

    /// <summary>
    /// Total bytes actually pulled over the wire, as opposed to <see cref="Length"/>, which is the size of
    /// the resource. Their RATIO is what "structure only" means: an inspector seeks past every weight blob,
    /// so a correct structure-only pass over a 10 MB checkpoint fetches a few hundred KB. Exposed so that
    /// can be ASSERTED rather than assumed - a regression that quietly starts reading weights still returns
    /// the right structure, and would otherwise pass unnoticed.
    /// </summary>
    public long BytesFetched { get; private set; }

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

    /// <summary>
    /// How long a read may wait for the next bytes of a response before the connection counts as stalled and the range is requested again from the bytes
    /// already received. It restarts with every chunk that arrives, so a slow transfer that is moving never trips it. Without it a response whose body
    /// stopped mid-way blocked forever: <see cref="HttpClient.Timeout"/> stops applying once the headers arrive (2026-11-01: a test held an idle
    /// connection to the models host for 25 minutes).
    /// </summary>
    public TimeSpan StallTimeout { get; set; } = TimeSpan.FromSeconds(60);

    /// <summary>Failed attempts (a stall, a dropped connection, a short or refused response) a read survives before it throws.</summary>
    public int Attempts { get; set; } = 4;

    /// <summary>Times a read resumed after a stall, a dropped connection or a short response (diagnostics).</summary>
    public int Resumes { get; private set; }

    public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken ct = default)
    {
        if (_position >= _length || buffer.Length == 0) return 0;
        int want = (int)Math.Min(buffer.Length, _length - _position);
        int got = 0, failures = 0;
        Exception? last = null;
        while (got < want)
        {
            long start = _position + got, end = _position + want - 1;   // clamp to EOF (RFC 7233)
            using var stall = CancellationTokenSource.CreateLinkedTokenSource(ct);
            try
            {
                using var req = new HttpRequestMessage(HttpMethod.Get, _url);
                req.Headers.Range = new RangeHeaderValue(start, end);
                stall.CancelAfter(StallTimeout);
                using var resp = await ModelsHost.SendAsync(_http, req, HttpCompletionOption.ResponseHeadersRead, stall.Token).ConfigureAwait(false);
                resp.EnsureSuccessStatusCode();
                // a server that ignores Range answers 200 with the whole file: its first bytes are not the ones asked for
                bool whole = start == 0 && end == _length - 1;
                if (resp.StatusCode != System.Net.HttpStatusCode.PartialContent && !whole)
                    throw new RangeRefusedException($"GET {_url} bytes={start}-{end}: HTTP {(int)resp.StatusCode}, not 206 (the server ignored the range)");
                if (resp.Content.Headers.ContentRange is { From: long from } cr && from != start)
                    throw new RangeRefusedException($"GET {_url} bytes={start}-{end}: the server sent bytes from {from} ({cr})");
                using var body = await resp.Content.ReadAsStreamAsync(stall.Token).ConfigureAwait(false);
                int before = got;
                while (got < want)
                {
                    stall.CancelAfter(StallTimeout);                     // the stall clock restarts with every chunk
                    int n = await body.ReadAsync(buffer.Slice(got, want - got), stall.Token).ConfigureAwait(false);
                    if (n == 0) break;                                    // the body ended early (a dropped connection)
                    got += n;
                    BytesFetched += n;
                }
                if (got < want) throw new IOException($"GET {_url} bytes={start}-{end}: the body ended after {got - before} of {end - start + 1} bytes");
            }
            catch (RangeRefusedException) { throw; }
            catch (Exception ex) when (!ct.IsCancellationRequested && ex is IOException or HttpRequestException or OperationCanceledException)
            {
                last = ex is OperationCanceledException ? new TimeoutException($"GET {_url} bytes={start}-{end}: no bytes for {StallTimeout.TotalSeconds:F0} s", ex) : ex;
                if (++failures >= Attempts) throw new IOException($"GET {_url}: {failures} failed attempts at byte {_position + got}", last);
                Resumes++;
                await Task.Delay(TimeSpan.FromMilliseconds(500 * failures), ct).ConfigureAwait(false);
            }
        }
        _position += got;
        return got;
    }

    // a response that cannot be resumed from: retrying the same request would get the same wrong bytes
    private sealed class RangeRefusedException(string message) : IOException(message);

    public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken ct)
        => ReadAsync(buffer.AsMemory(offset, count), ct).AsTask();

    public override int Read(byte[] buffer, int offset, int count)
        => throw new NotSupportedException("HttpRangeStream is async-only (a sync Read deadlocks in WASM); use ReadAsync.");

    public override void Flush() { }
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
}
