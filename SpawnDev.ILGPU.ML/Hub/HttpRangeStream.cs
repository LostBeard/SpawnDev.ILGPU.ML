using System.Net.Http.Headers;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;

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

    // Where the ranges go once a response showed where the source lands (HttpClient followed the redirect: the response's RequestMessage carries the final
    // URI). Hugging Face answers /resolve/ with a 302 to a signed CDN URL and counts every /resolve/ request against its resolver limit (3,000 per 5 minutes
    // per client; 2026-11-19, Lore): a range per read to /resolve/ spent one each. Null until the first response, and again after a failure on it (a
    // signed URL expires). Not used when the client sends a default Authorization header: a signed URL refuses a second authentication.
    private string? _rangeUrl;   // null: not learned yet (or forgotten after a failure on it)
    private bool _resolved;      // _rangeUrl is where the source landed, not the source itself

    /// <summary>
    /// Requests sent to the source URL itself rather than to where it resolved: on Hugging Face one "resolver" call each. On a source that redirects: one
    /// per stream (two in the browser, which asks fetch where the redirect landed), plus one per expired signed URL. On one that does not: every request
    /// (there is nothing to resolve). Diagnostics; the tests assert it.
    /// </summary>
    public int ResolveRequests { get; private set; }

    /// <summary>Where range requests go now: the resolved URL (the source itself when it did not redirect), or null before the first response.</summary>
    public string? RangeUrl => _rangeUrl;

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
            string target = _resolved ? _rangeUrl! : _url;
            try
            {
                using var req = new HttpRequestMessage(HttpMethod.Get, target);
                req.Headers.Range = new RangeHeaderValue(start, end);
                stall.CancelAfter(StallTimeout);
                bool learn = _rangeUrl == null;
                if (!_resolved) ResolveRequests++;
                using var resp = await ModelsHost.SendAsync(_http, req, HttpCompletionOption.ResponseHeadersRead, stall.Token).ConfigureAwait(false);
                resp.EnsureSuccessStatusCode();
                if (learn) await LearnWhereItLandsAsync(resp).ConfigureAwait(false);
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
                // a failure on a resolved URL (a signed URL expires, or anything else): the next attempt resolves again
                if (_resolved && _rangeUrl == target) { _rangeUrl = null; _resolved = false; }
                last = ex is OperationCanceledException ? new TimeoutException($"GET {_url} bytes={start}-{end}: no bytes for {StallTimeout.TotalSeconds:F0} s", ex) : ex;
                if (++failures >= Attempts) throw new IOException($"GET {_url}: {failures} failed attempts at byte {_position + got}", last);
                Resumes++;
                await Task.Delay(TimeSpan.FromMilliseconds(500 * failures), ct).ConfigureAwait(false);
            }
        }
        _position += got;
        return got;
    }

    // Where the first response's request ended up: HttpClient follows a redirect and leaves the final URI on the response's RequestMessage (desktop). The
    // browser's HttpClient does not (its RequestUri stays the one requested), so there fetch is asked, whose Response.url is the final URL: one byte, one
    // more resolve per stream instead of one per read.
    private async Task LearnWhereItLandsAsync(HttpResponseMessage resp)
    {
        string? requested = Uri.TryCreate(_url, UriKind.Absolute, out var abs) ? abs.AbsoluteUri : _http.BaseAddress != null ? new Uri(_http.BaseAddress, _url).AbsoluteUri : null;
        string? landed = resp.RequestMessage?.RequestUri is { IsAbsoluteUri: true } final ? final.AbsoluteUri : null;
        if (OperatingSystem.IsBrowser() && landed != null)
        {
            ResolveRequests++;
            landed = await BrowserLandingAsync(landed).ConfigureAwait(false) ?? landed;
        }
        _resolved = landed != null && landed != requested && _http.DefaultRequestHeaders.Authorization == null;
        _rangeUrl = _resolved ? landed : _url;
    }

    private static async Task<string?> BrowserLandingAsync(string url)
    {
        try
        {
            if (SpawnJSRuntime.Instance is not { } js) return null;
            using var r = await js.Fetch(url, new FetchOptions { Headers = new() { ["Range"] = "bytes=0-0" } });
            return r.Ok && r.Redirected && !string.IsNullOrEmpty(r.Url) ? r.Url : null;
        }
        catch { return null; }   // no answer: the stream keeps the source URL
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
