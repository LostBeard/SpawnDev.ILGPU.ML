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
    private string _url;
    private bool _fellBack;
    private readonly long _length;
    private long _position;

    // Where the ranges go once a response showed where the source lands. Hugging Face answers /resolve/ with a 302 to a signed CDN URL, and every
    // request to /resolve/ is one of the Hub's "resolver" calls: 3,000 per 5 minutes per client (MEASURED 2026-11-19 by Lore, anonymous: ratelimit
    // "resolvers";r=2999;t=118). Sending each 16 MiB chunk there made a 27 GB checkpoint ~1,700 resolver calls a load. Null until the first answer, and
    // again after a failure on it (a signed URL expires): the next request goes to _url and resolves again.
    private string? _rangeUrl;
    private Task? _resolving;   // the request resolving now: chunks scheduled before its answer wait for it instead of each resolving

    public FetchRangeStream(string url, long length) { _url = url; _length = length; }

    /// <summary>
    /// Requests sent to the source URL itself rather than to where it resolved: on Hugging Face one "resolver" call each. On a source that redirects: one
    /// per stream, plus one per expired signed URL. On one that does not: every request (there is nothing to resolve). Diagnostics; the tests assert it.
    /// </summary>
    public int ResolveRequests { get; private set; }

    /// <summary>Where range requests go now: the resolved URL (the source itself when it did not redirect), or null before the first response.</summary>
    public string? RangeUrl => _rangeUrl;

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

    /// <summary>
    /// Ranges retried on the models host after a fast network failure (connection reset / refused) instead of falling back to Hugging Face at once.
    /// Diagnostics; the tests assert it.
    /// </summary>
    public int NetworkErrorRetries { get; private set; }

    private async Task<Uint8Array> FetchRangeAsync(long start, long end, CancellationToken ct)
    {
        Exception? last = null;
        bool retriedNetworkError = false;
        for (int attempt = 1; attempt <= Attempts; attempt++)
        {
            ct.ThrowIfCancellationRequested();
            bool reResolve = false;
            TimeSpan? retryAfter = null;
            TaskCompletionSource? gate = null;
            string target = _url;
            int? httpStatus = null;   // set when the server ANSWERED (with an error); null = no answer (network failure / timeout)
            var clock = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // one resolve at a time: the chunks scheduled before the first answer use its result
                while (_rangeUrl == null && _resolving is { IsCompleted: false } pending) await pending;
                target = _rangeUrl ?? _url;
                if (_rangeUrl == null)
                {
                    gate = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
                    _resolving = gate.Task;
                }
                if (target == _url) ResolveRequests++;
                using var window = SpawnJSRuntime.Instance!.Get<Window>("window");
                using var signal = AbortSignal.Timeout((float)RequestTimeout.TotalMilliseconds);   // aborts the request AND the body read
                using var resp = await window.Fetch(target, new FetchOptions { Headers = new() { ["Range"] = $"bytes={start}-{end}" }, Signal = signal });
                if (!resp.Ok)
                {
                    httpStatus = (int)resp.Status;
                    if ((int)resp.Status == 429) retryAfter = RetryAfter(resp);
                    throw new IOException($"GET {target} bytes={start}-{end}: HTTP {resp.Status} {resp.StatusText}");
                }
                // the first answer says where the source lands (fetch followed the redirect: Response.url is the final URL)
                if (gate != null) _rangeUrl = resp.Redirected && !string.IsNullOrEmpty(resp.Url) ? resp.Url : target;
                using var ab = await resp.ArrayBuffer();
                var arr = new Uint8Array(ab);
                if (arr.Length != end - start + 1) { arr.Dispose(); throw new IOException($"GET {target} bytes={start}-{end}: short read"); }
                return arr;
            }
            catch (Exception ex) when (ex is not OperationCanceledException || !ct.IsCancellationRequested)
            {
                last = ex;
                // a failure on a resolved URL (a signed URL expires, or anything else): resolve again, now, without the backoff
                if (target != _url && _rangeUrl == target) { _rangeUrl = null; reResolve = true; }
                // A FAST network failure with no answer (connection reset / refused) is usually transient: try the models host once more before giving
                // the whole rest of the stream to Hugging Face, which is several times slower to stream from (MEASURED 2026-11-23: one reset during a
                // 0.5 GB GGUF load turned 89 s into 302 s). An HTTP error answer (404 / 403 / 429 / 5xx) still falls back at once: the host has
                // answered. A timeout falls back at once too (a stalled host would otherwise cost RequestTimeout twice).
                bool fastNetworkFailure = httpStatus == null && clock.Elapsed < RequestTimeout * 0.9;
                if (!_fellBack && fastNetworkFailure && !retriedNetworkError && ModelsHost.GetFallbackUrl(_url) != null)
                {
                    retriedNetworkError = true;
                    NetworkErrorRetries++;
                    if (attempt < Attempts) await Task.Delay(TimeSpan.FromMilliseconds(500), ct);
                    continue;
                }
                // The models host could not give this range: switch to the Hugging Face twin for the rest of the stream. Safe mid-file because no
                // validator is sent here and both hosts serve identical bytes (every mirrored file was checked sha256 against Hugging Face).
                if (!_fellBack && ModelsHost.GetFallbackUrl(_url) is { } alt)
                {
                    _fellBack = true;
                    ModelsHost.ReportFallback(_url, alt, ex.Message);
                    _url = alt;
                    _rangeUrl = null;
                    continue;
                }
                if (reResolve && retryAfter == null) continue;
                if (attempt < Attempts) await Task.Delay(retryAfter ?? TimeSpan.FromMilliseconds(1000 * attempt), ct);
            }
            finally { gate?.TrySetResult(); }
        }
        throw new IOException($"GET {_url} bytes={start}-{end} failed {Attempts} times", last);
    }

    /// <summary>
    /// How long an HTTP 429 asks a client to wait: Retry-After (seconds), else the Hugging Face Hub's <c>ratelimit</c> header's t= (seconds to the end of
    /// its window), else 10 s; at most 5 minutes. Headers a CORS response does not expose read as missing.
    /// </summary>
    internal static TimeSpan RetryAfter(Response resp)
    {
        try
        {
            using var h = resp.Headers;
            if (int.TryParse(h.Get("retry-after"), out int s) && s >= 0) return TimeSpan.FromSeconds(Math.Min(s, 300));
            if (h.Get("ratelimit") is { } rl && System.Text.RegularExpressions.Regex.Match(rl, @"t=(\d+)") is { Success: true } m)
                return TimeSpan.FromSeconds(Math.Min(int.Parse(m.Groups[1].Value), 300));
        }
        catch { /* unreadable headers: the default */ }
        return TimeSpan.FromSeconds(10);
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
