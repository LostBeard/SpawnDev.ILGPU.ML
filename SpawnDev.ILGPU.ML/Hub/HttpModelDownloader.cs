using System.Collections.Concurrent;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>A download that is running right now.</summary>
/// <param name="Key">Store key being filled.</param>
/// <param name="Url">Where the bytes are coming from.</param>
/// <param name="BytesReceived">Bytes pulled so far, including any carried by a resume.</param>
/// <param name="TotalBytes">Expected total, or -1 when the origin did not say.</param>
/// <param name="Resumed">True when this continued an interrupted download.</param>
/// <param name="BytesPerSecond">Throughput over this attempt, 0 until enough time has passed to mean anything.</param>
public readonly record struct ActiveModelDownload(
    string Key, string Url, long BytesReceived, long TotalBytes, bool Resumed, double BytesPerSecond)
{
    /// <summary>Completion fraction in [0,1], or null when the total is unknown.</summary>
    public double? Fraction => TotalBytes > 0 ? Math.Clamp((double)BytesReceived / TotalBytes, 0d, 1d) : null;
}

/// <summary>
/// Downloads a file over HTTP into an <see cref="IResumableModelStore"/>, resuming an interrupted transfer
/// and never letting the payload touch the .NET/WASM managed heap.
/// </summary>
/// <remarks>
/// <para>
/// This is the TRANSPORT, and it is deliberately a separate object from the store. It knows about
/// <c>fetch</c>, ranges, ETags and retries; it knows nothing about OPFS. The store knows about durability;
/// it knows nothing about HTTP. Fusing the two is how model delivery ended up welded to one transport in
/// the first place - a torrent or file-system downloader drops in against the same store.
/// </para>
/// <para>
/// 🔴 The payload goes <c>fetch</c> -&gt; <see cref="Uint8Array"/> -&gt; store, and only a chunk's
/// <c>length</c> (a number) is ever read into .NET. Chunks are coalesced into one JS-side staging buffer
/// before each store write, because <c>fetch</c> delivers ~20 KB chunks while OPFS wants far larger ones
/// (MEASURED: 64 KiB reads ~85 MB/s, 16 MiB ~1560-1990 MB/s).
/// </para>
/// </remarks>
public class HttpModelDownloader
{
    /// <summary>Checkpoint the resume point every this many bytes, so an interrupted transfer resumes near where it stopped.</summary>
    private const long CheckpointInterval = 8L * 1024 * 1024;

    private readonly SpawnJSRuntime _js;
    private readonly IResumableModelStore _store;
    private readonly Dictionary<string, SemaphoreSlim> _keyGates = new();

    /// <summary>
    /// Coalesce incoming <c>fetch</c> chunks into a JS-side buffer of this size before writing to the
    /// store. 0 writes each chunk straight through. Default 16 MiB - see the class remarks.
    /// </summary>
    public int WriteBufferSize { get; set; } = 16 * 1024 * 1024;

    /// <summary>Minimum gap between progress reports. The final report is always sent.</summary>
    public int ProgressIntervalMs { get; set; } = 100;

    // ── FETCH vs OPFS-WRITE split ───────────────────────────────────────────────────────────────
    //
    // 🔴 "DOWNLOAD" IS TWO THINGS AND THIS PATH TIMED NEITHER. A download here is fetch -> coalesce
    // JS-side -> write to the store, and the reported MB/s covers all of it. MEASURED 2026-09-15:
    // Qwen3-1.7B (1.83 GB) downloaded at 41.6 MB/s, which I reported as "network bound". TJ: "not even
    // close. those files are downloading from a VM on this very same 1 GB/s lan." He is right - a
    // gigabit LAN should deliver 100+ MB/s, so 41.6 is about a third of the link and the bottleneck is
    // something we own, not the wire.
    //
    // Without this split there is no way to tell a slow SERVER/link from slow OPFS WRITES, and the
    // upload path had exactly this blind spot until it was instrumented an hour earlier (where it
    // turned out the GPU write was free and the read dominated 11.5:1). Same mistake, same fix:
    // measure the two halves instead of naming one.
    //
    // Cheap enough to leave on: two Stopwatch timestamps per fetch chunk.

    /// <summary>Cumulative ms inside <c>reader.Read()</c> - the fetch/network half. Reset per download.</summary>
    public static double LastFetchMs { get; private set; }
    /// <summary>Cumulative ms writing to the store (OPFS) including coalesced flushes. Reset per download.</summary>
    public static double LastStoreWriteMs { get; private set; }
    /// <summary>Bytes seen by the fetch half. Reset per download.</summary>
    public static long LastFetchBytes { get; private set; }

    private static void ResetTransferCensus()
    {
        LastFetchMs = 0; LastStoreWriteMs = 0; LastFetchBytes = 0; LastFetchChunks = 0;
        LastCheckpointMs = 0; LastCheckpoints = 0; LastCoalesceMs = 0;
    }
    /// <summary>Fetch chunks seen. Reset per download - the loop cost scales with THIS, not with bytes.</summary>
    public static long LastFetchChunks { get; private set; }

    // ⚠️ THE UNACCOUNTED TIME HAD (AT LEAST) TWO CANDIDATES AND DIVIDING BY ONE OF THEM PROVED NOTHING.
    // 12,211 ms of a 44 s download sat outside fetch and store-write. That is 132 us x 92,182 fetch
    // chunks (per-chunk interop) OR 56 ms x ~218 checkpoints (an OPFS flush + a resume-state write every
    // 8 MiB) - both divide neatly, and I asserted the first. TJ: "that does not seem correct."
    // These time the two suspects directly so the remainder is a real remainder.

    /// <summary>Cumulative ms in the every-8-MiB checkpoint (dest.FlushAsync + store.SetStateAsync).</summary>
    public static double LastCheckpointMs { get; private set; }
    /// <summary>Checkpoints taken. Reset per download.</summary>
    public static long LastCheckpoints { get; private set; }
    /// <summary>Cumulative ms in the JS-side coalescing copy (<c>writeBuffer.Set</c>) plus chunk length reads.</summary>
    public static double LastCoalesceMs { get; private set; }
    private static void AddFetch(double ms, long bytes) { LastFetchMs += ms; LastFetchBytes += bytes; LastFetchChunks++; }
    private static void AddStoreWrite(double ms) { LastStoreWriteMs += ms; }

    /// <summary>Chunks delivered by <c>fetch</c> during the most recent download (diagnostic).</summary>
    public long LastDownloadChunks { get; private set; }

    /// <summary>Store writes issued during the most recent download (diagnostic).</summary>
    public long LastDownloadWrites { get; private set; }

    /// <summary>Bytes transferred during the most recent download, excluding anything carried by a resume.</summary>
    public long LastDownloadBytes { get; private set; }

    private sealed class Tracker
    {
        public string Url = "";
        public long Received, Total = -1, StartedAt, StartReceived;
        public bool Resumed;
    }

    private readonly ConcurrentDictionary<string, Tracker> _active = new();

    /// <summary>
    /// Every download currently in flight through this downloader, for a cache/status UI.
    /// </summary>
    /// <remarks>
    /// The page showing progress is almost never the page doing the download, so per-call
    /// <see cref="IProgress{T}"/> cannot answer "what is downloading right now". This can, which is why the
    /// downloader should be a shared instance rather than one per caller.
    /// </remarks>
    public IReadOnlyList<ActiveModelDownload> ActiveDownloads =>
        _active.Select(kv =>
        {
            var t = kv.Value;
            var seconds = Math.Max(0.001, (Environment.TickCount64 - t.StartedAt) / 1000.0);
            // Rate over THIS attempt only - bytes carried in by a resume were not transferred now, and
            // counting them would show a wildly inflated speed for the first second of every resume.
            var rate = seconds < 0.25 ? 0 : (t.Received - t.StartReceived) / seconds;
            return new ActiveModelDownload(kv.Key, t.Url, t.Received, t.Total, t.Resumed, rate);
        }).ToList();

    /// <summary>Raised when a download starts, makes progress, or finishes - for a UI that wants to refresh
    /// without polling. Fired on the download's own execution context; marshal to the UI yourself.</summary>
    public event Action? ActiveDownloadsChanged;

    /// <summary>Create a downloader that fills <paramref name="store"/>.</summary>
    public HttpModelDownloader(SpawnJSRuntime js, IResumableModelStore store)
    {
        _js = js ?? throw new ArgumentNullException(nameof(js));
        _store = store ?? throw new ArgumentNullException(nameof(store));
    }

    /// <summary>
    /// Return a readable stream for <paramref name="key"/>, downloading from <paramref name="url"/> only if
    /// the store does not already hold it complete. Concurrent calls for the same key do not race: the
    /// second waits and is then served from the store rather than downloading again.
    /// </summary>
    public async Task<Stream> GetOrDownloadAsync(string url, string key,
        IProgress<ModelDownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        var gate = GetGate(key);
        await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var state = await _store.GetStateAsync(key, cancellationToken).ConfigureAwait(false);

            // Hit: complete, and either from this URL or from nowhere. An empty SourceRef means the entry
            // was handed to the store directly (a torrent stream, a picked file); re-downloading it over
            // HTTP because we cannot prove its origin would defeat the point of storing it.
            if (state.Complete && (state.SourceRef == url || string.IsNullOrEmpty(state.SourceRef)))
            {
                progress?.Report(new ModelDownloadProgress(state.BytesWritten, state.BytesWritten, false));
                return await OpenOrThrowAsync(key, cancellationToken).ConfigureAwait(false);
            }

            // Resume only a partial that was for THIS url.
            long resumeFrom = !state.Complete && state.SourceRef == url ? state.BytesWritten : 0;

            await DownloadAsync(url, key, resumeFrom, state.ETag, progress, cancellationToken).ConfigureAwait(false);
            return await OpenOrThrowAsync(key, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            // Here rather than inside DownloadAsync, so the entry cannot survive ANY exit path - success,
            // throw, or cancellation. A stuck entry would show as a download that never finishes in the UI.
            if (_active.TryRemove(key, out _)) ActiveDownloadsChanged?.Invoke();
            gate.Release();
        }
    }

    private async Task<Stream> OpenOrThrowAsync(string key, CancellationToken ct)
        => await _store.OpenReadAsync(key, ct).ConfigureAwait(false)
           ?? throw new IOException($"Store has no readable entry for '{key}' after a successful download.");

    /// <summary>
    /// Bytes pulled per ranged request in the segmented path. One <c>fetch</c> + one
    /// <c>response.bytes()</c> + one store write per segment.
    /// </summary>
    /// <remarks>
    /// 🔴 WHY SEGMENTS INSTEAD OF A CHUNK LOOP. Streaming the body pulls ~20 KB per
    /// <c>reader.Read()</c>, and each iteration crosses .NET-&gt;JS several times (read the result's
    /// value, read its length, copy it into the staging buffer, dispose the chunk, dispose the result).
    /// MEASURED 2026-09-15 on Qwen3-1.7B (1.83 GB) over the hub:
    /// <code>
    ///   DOWNLOAD 43.6s | fetch 29,356 ms | store write 2,211 ms
    ///   90,692 fetch chunks of 20 KiB
    ///   checkpoints  1,178 ms (109)   coalesce Set 1,164 ms   UNATTRIBUTED 9,690 ms
    /// </code>
    /// 9.7 s - 22% of the download - was per-iteration interop, and neither of the two things I first
    /// blamed (checkpointing, the staging copy) accounted for it: both were ~1.2 s. 9,690 ms over 90,692
    /// iterations is 107 us each, which is 4-5 SpawnJS crossings at the 4-19 us a bare crossing measured.
    /// <para>
    /// A 64 MiB segment fetched with <c>response.bytes()</c> is ONE Uint8Array in ONE crossing, so the
    /// whole file costs ~28 iterations instead of 90,692 - and the bytes still never touch the .NET heap.
    /// </para>
    /// <para>
    /// ⚠️ It also keeps everything the chunk loop gave us: progress per segment, a durable checkpoint per
    /// segment (resume granularity becomes 64 MiB rather than 8 MiB, still durable), and If-Range so a
    /// file that changes mid-download restarts instead of splicing two versions together.
    /// </para>
    /// <para>
    /// ⚠️ Peak JS memory is ONE segment. 64 MiB is the trade: large enough that per-request overhead is
    /// noise against a gigabit link, small enough to be nothing on any machine that can run a 1.8 GB model.
    /// </para>
    /// </remarks>
    public int SegmentBytes { get; set; } = 64 * 1024 * 1024;

    /// <summary>
    /// Fetch the file as bounded ranges instead of streaming one body. <b>Default FALSE - it MEASURED
    /// SLOWER.</b>
    /// </summary>
    /// <remarks>
    /// 🔴 A NEGATIVE RESULT, KEPT SO IT IS NOT RE-TRIED. Segmenting removes the per-chunk interop the
    /// streaming loop pays (see <see cref="SegmentBytes"/>) and it does remove it - but the download gets
    /// SLOWER overall. MEASURED 2026-09-15, Qwen3-1.7B (1.83 GB) over the hub, same machine, same file:
    /// <code>
    ///                     streaming          segmented 64 MiB
    ///   DOWNLOAD          43.6s 40.1 MB/s    46.6s 37.6 MB/s
    ///     fetch           29.4s 60 MB/s      43.4s 40 MB/s
    ///     store write      2.2s               1.9s
    ///     unattributed     9.7s               0.5s     &lt;-- the interop really did go away
    ///     iterations     90,692                 28
    /// </code>
    /// The 9.7 s of interop vanished exactly as intended, and it bought nothing: fetch throughput fell
    /// from 60 to 40 MB/s and the total rose by 3 s.
    /// <para>
    /// ⚠️ WHY, AND IT IS THE LESSON: one streaming body downloads WHILE we process it. Discrete range
    /// requests serialize - request, wait for the whole segment, write, request again - with no overlap
    /// and a fresh ramp per request. The streaming loop's "29.4 s of fetch" was overlapping with our own
    /// work, so treating it as pure cost and the 9.7 s as pure waste was reading a concurrency window as
    /// a bill. Overlapping the next fetch with the current write would recover the serialization but not
    /// the per-request throughput drop, so it still cannot beat 43.6 s.
    /// </para>
    /// <para>
    /// Kept because it is the correct path for an origin that cannot stream, and because the numbers
    /// above are worth more than the code.
    /// </para>
    /// </remarks>
    public bool UseSegmentedDownload { get; set; } = true;

    /// <summary>
    /// How many ranged requests to keep in flight. 1 = serial, which measured SLOWER than streaming.
    /// </summary>
    /// <remarks>
    /// 🔴 THE HUB LIMIT IS PER-CONNECTION, MEASURED WITH curl FROM THIS MACHINE 2026-09-15:
    /// <code>
    ///   single stream   38.9 MB/s
    ///   4 parallel      23.2 + 22.8 + 13.1 + 13.1 = 72.1 MB/s aggregate
    /// </code>
    /// A single connection leaves more than half the available throughput unused - and the browser's
    /// 40.1 MB/s wall-clock download matches curl's single-stream number exactly, so the transport was
    /// never our code.
    /// <para>
    /// ⚠️ THIS IS WHY PARALLEL HELPS WHERE SEGMENTING ALONE DID NOT. Serial 64 MiB ranges were SLOWER
    /// than streaming (46.6s vs 43.6s) because they removed the overlap a streaming body gets for free.
    /// Parallel ranges add capacity that is provably idle instead of removing work that was overlapping.
    /// </para>
    /// <para>
    /// ⚠️ FETCHED IN PARALLEL, WRITTEN IN ORDER: the store's write stream is sequential, so segments are
    /// requested concurrently and each written only when it is next in sequence. Peak JS memory is about
    /// <c>SegmentBytes x DownloadParallelism</c>.
    /// </para>
    /// </remarks>
    public int DownloadParallelism { get; set; } = 4;

    private async Task DownloadAsync(string url, string key, long resumeFrom, string? knownETag,
        IProgress<ModelDownloadProgress>? progress, CancellationToken ct)
    {
        var headers = new Dictionary<string, string>();
        // Segmented: ask for a BOUNDED range from the start, so the first response is a 206 carrying
        // Content-Range (which gives the true total) and a body small enough to take in one
        // response.bytes(). An origin that ignores Range answers 200 with the whole body and we fall
        // back to the streaming chunk loop below, exactly as before.
        if (UseSegmentedDownload && SegmentBytes > 0)
        {
            headers["Range"] = $"bytes={resumeFrom}-{resumeFrom + SegmentBytes - 1}";
            if (!string.IsNullOrEmpty(knownETag)) headers["If-Range"] = knownETag!;
        }
        else if (resumeFrom > 0)
        {
            headers["Range"] = $"bytes={resumeFrom}-";
            // If the origin's copy changed since our partial was written it answers 200 with the whole body
            // instead of 206, and we start over rather than splicing two different files together.
            if (!string.IsNullOrEmpty(knownETag)) headers["If-Range"] = knownETag!;
        }

        // 🔴 Cancellation must reach the NETWORK, not just this loop. Checking the token between chunks
        // leaves the HTTP request itself running: the browser keeps pulling the body, the connection and
        // bandwidth stay committed, and a cancel during a multi-GB model frees nothing. An AbortController
        // wired to the token aborts the fetch itself, and also unblocks a `reader.Read()` that is parked on
        // a stalled origin - which a token check between chunks can never do, because it never gets to run.
        using var abort = new AbortController();
        using var abortSignal = abort.Signal;
        using var abortReg = ct.Register(() => { try { abort.Abort(); } catch { /* already gone */ } });

        // Fetch from the RUNTIME, not from `window`: there is no `window` in a worker, and a model SHOULD be
        // loadable from a worker. SpawnJSRuntime.Fetch calls fetch() on whatever the global scope is.
        Response response;
        try
        {
            response = await _js.Fetch(url, new FetchOptions
            {
                Headers = headers.Count > 0 ? headers : null,
                Signal = abortSignal,
            }).ConfigureAwait(false);
        }
        catch (Exception ex) when (ct.IsCancellationRequested)
        {
            // An aborted fetch surfaces as a JS AbortError. Callers cancel with a token and expect the
            // token's exception, so translate rather than leaking the transport's error type.
            throw new OperationCanceledException($"Model download cancelled: {url}", ex, ct);
        }
        using var _response = response;

        // fetch() does NOT throw on 404/500 - it resolves with Ok=false and an ERROR BODY. Nothing is opened
        // or written before this check, so a failed response can never reach the store.
        if (!response.Ok)
            throw new HttpRequestException(
                $"Model download failed: {(int)response.Status} {response.StatusText} for {url}");

        var status = (int)response.Status;
        var resumed = resumeFrom > 0 && status == 206;
        if (resumeFrom > 0 && status != 206)
            resumeFrom = 0;   // origin ignored the Range, or If-Range failed: take the full body

        long total;
        string? etag;
        using (var respHeaders = response.Headers)
        {
            total = ParseTotalBytes(respHeaders, resumeFrom);
            etag = NullIfEmpty(respHeaders.Get("etag"));
        }

        await _store.SetStateAsync(key, url, total, resumeFrom, false, etag, ct).ConfigureAwait(false);

        var tracker = new Tracker
        {
            Url = url, Received = resumeFrom, Total = total, Resumed = resumed,
            StartedAt = Environment.TickCount64, StartReceived = resumeFrom,
        };
        _active[key] = tracker;
        ActiveDownloadsChanged?.Invoke();

        long received = resumeFrom;   // pulled off the network
        long written = resumeFrom;    // durable in the store (lags `received` while buffering)

        // Report ONCE up front, as soon as the total is known and before a single byte is read. Without
        // this a caller cannot render a progress bar at all until the first interval elapses - and for a
        // file that finishes faster than ProgressIntervalMs the ONLY report was the final 100%, which is
        // indistinguishable from no progress reporting at all. It also hands the UI the total immediately.
        // Report ONCE up front, as soon as the total is known and before a single byte is read. Without
        // this a caller cannot render a progress bar at all until the first interval elapses - and for a
        // file that finishes faster than ProgressIntervalMs the ONLY report was the final 100%, which is
        // indistinguishable from no progress reporting at all. It also hands the UI the total immediately.
        progress?.Report(new ModelDownloadProgress(received, total, resumed));

        // Deliberately NOT `TickCount64`: seeding with "now" suppressed the first in-loop report for a
        // whole interval. Seeding a full interval in the past means the next chunk reports immediately, so
        // a short download still produces real intermediate progress.
        var lastReport = Environment.TickCount64 - ProgressIntervalMs;
        var lastCheckpoint = written;
        ResetTransferCensus();
        long chunkCount = 0, writeCount = 0;
        Uint8Array? writeBuffer = null;
        long bufferFill = 0;

        var dest = await _store.OpenWriteAsync(key, resumeFrom, ct).ConfigureAwait(false);
        await using (dest.ConfigureAwait(false))
        {
            // ── SEGMENTED PATH: one fetch + one response.bytes() + one store write per 64 MiB ────────
            // Taken only when the origin honoured the bounded Range (206). Otherwise fall through to the
            // streaming loop, which is unchanged and remains the fallback for origins without ranges.
            if (UseSegmentedDownload && SegmentBytes > 0 && status == 206)
            {
                // Fetch segments CONCURRENTLY, write them IN ORDER. The store's write stream is
                // sequential, so a completed later segment waits its turn; the win is purely that the
                // network has several requests outstanding at once (see DownloadParallelism).
                async Task<Uint8Array> FetchSegmentAsync(long start)
                {
                    var h = new Dictionary<string, string>
                    {
                        ["Range"] = $"bytes={start}-{start + SegmentBytes - 1}",
                    };
                    // If-Range: a file that changed underneath us answers 200, and we fail loudly rather
                    // than splicing two different versions together.
                    if (!string.IsNullOrEmpty(etag)) h["If-Range"] = etag!;

                    var t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                    var r = await _js.Fetch(url, new FetchOptions { Headers = h, Signal = abortSignal })
                        .ConfigureAwait(false);
                    using (r)
                    {
                        if (!r.Ok)
                            throw new HttpRequestException(
                                $"Model download failed mid-file: {(int)r.Status} {r.StatusText} " +
                                $"for {url} at byte {start}");
                        if ((int)r.Status != 206)
                            throw new IOException(
                                $"Origin stopped honouring Range at byte {start} for {url} " +
                                $"(status {(int)r.Status}). The file may have changed mid-download; the " +
                                "partial is checkpointed, so a retry resumes from the last durable byte.");
                        var bytes = await r.Bytes().ConfigureAwait(false);
                        AddFetch((System.Diagnostics.Stopwatch.GetTimestamp() - t0)
                                 * (1000.0 / System.Diagnostics.Stopwatch.Frequency), bytes.Length);
                        return bytes;
                    }
                }

                var inFlight = new Queue<Task<Uint8Array>>();
                // The first segment's response is already in hand - consume it as segment 0.
                var firstT0 = System.Diagnostics.Stopwatch.GetTimestamp();
                var firstBytes = await response.Bytes().ConfigureAwait(false);
                AddFetch((System.Diagnostics.Stopwatch.GetTimestamp() - firstT0)
                         * (1000.0 / System.Diagnostics.Stopwatch.Frequency), firstBytes.Length);
                inFlight.Enqueue(Task.FromResult(firstBytes));

                long nextRequestAt = resumeFrom + firstBytes.Length;
                int parallel = Math.Max(1, DownloadParallelism);
                while (inFlight.Count < parallel && (total <= 0 || nextRequestAt < total))
                {
                    inFlight.Enqueue(FetchSegmentAsync(nextRequestAt));
                    nextRequestAt += SegmentBytes;
                }

                try
                {
                    while (inFlight.Count > 0)
                    {
                        ct.ThrowIfCancellationRequested();
                        using var segBytes = await inFlight.Dequeue().ConfigureAwait(false);
                        var segLen = segBytes.Length;
                        if (segLen == 0) break;

                        await WriteChunkAsync(dest, segBytes, ct).ConfigureAwait(false);
                        writeCount++;
                        written += segLen;
                        received += segLen;

                        // Durable checkpoint per segment - resume granularity is SegmentBytes.
                        var _cT0 = System.Diagnostics.Stopwatch.GetTimestamp();
                        await dest.FlushAsync(ct).ConfigureAwait(false);
                        await _store.SetStateAsync(key, url, total, written, false, etag, ct).ConfigureAwait(false);
                        LastCheckpointMs += (System.Diagnostics.Stopwatch.GetTimestamp() - _cT0)
                                            * (1000.0 / System.Diagnostics.Stopwatch.Frequency);
                        LastCheckpoints++;

                        tracker.Received = received;
                        progress?.Report(new ModelDownloadProgress(received, total, resumed));
                        ActiveDownloadsChanged?.Invoke();

                        if (total > 0 && received >= total) break;
                        // Top the pipeline back up.
                        if (total <= 0 || nextRequestAt < total)
                        {
                            inFlight.Enqueue(FetchSegmentAsync(nextRequestAt));
                            nextRequestAt += SegmentBytes;
                        }
                    }
                }
                finally
                {
                    // Drain anything still outstanding so a failure cannot leave fetches running.
                    while (inFlight.Count > 0)
                    {
                        try { (await inFlight.Dequeue().ConfigureAwait(false)).Dispose(); }
                        catch { /* already failing; nothing to add */ }
                    }
                }

                await dest.FlushAsync(ct).ConfigureAwait(false);
                await _store.SetStateAsync(key, url, total, written, true, etag, ct).ConfigureAwait(false);
                progress?.Report(new ModelDownloadProgress(received, total, resumed));
                LastDownloadChunks = LastFetchChunks;
                LastDownloadWrites = writeCount;
                LastDownloadBytes = written - resumeFrom;
                _active.TryRemove(key, out _);
                ActiveDownloadsChanged?.Invoke();
                return;
            }

            using var body = response.Body ?? throw new InvalidOperationException($"Response for {url} had no body.");
            using var reader = body.GetReader();
            try
            {
                while (true)
                {
                    if (ct.IsCancellationRequested)
                    {
                        await reader.Cancel().ConfigureAwait(false);
                        ct.ThrowIfCancellationRequested();
                    }

                    var _fetchT0 = System.Diagnostics.Stopwatch.GetTimestamp();
                    using var result = await reader.Read().ConfigureAwait(false);
                    AddFetch((System.Diagnostics.Stopwatch.GetTimestamp() - _fetchT0)
                             * (1000.0 / System.Diagnostics.Stopwatch.Frequency),
                             result.Done || result.Value == null ? 0 : result.Value.Length);
                    if (result.Done) break;

                    // 🔴 The chunk stays a JS Uint8Array from here to the store. Never ReadBytes() it.
                    using var chunk = result.Value;
                    if (chunk == null) continue;
                    var chunkLength = chunk.Length;
                    if (chunkLength == 0) continue;
                    chunkCount++;

                    if (WriteBufferSize <= 0 || chunkLength >= WriteBufferSize)
                    {
                        await FlushAsync().ConfigureAwait(false);   // keep byte order
                        await WriteChunkAsync(dest, chunk, ct).ConfigureAwait(false);
                        writeCount++;
                        written += chunkLength;
                    }
                    else
                    {
                        if (bufferFill + chunkLength > WriteBufferSize) await FlushAsync().ConfigureAwait(false);
                        // TypedArray.set: a JS-side copy into the staging buffer, no managed array involved.
                        var _coT0 = System.Diagnostics.Stopwatch.GetTimestamp();
                        writeBuffer ??= new Uint8Array(WriteBufferSize);
                        writeBuffer.Set(chunk, bufferFill);
                        LastCoalesceMs += (System.Diagnostics.Stopwatch.GetTimestamp() - _coT0)
                                          * (1000.0 / System.Diagnostics.Stopwatch.Frequency);
                        bufferFill += chunkLength;
                    }
                    received += chunkLength;

                    // Checkpoint only what is DURABLE - `written`, never `received`. A resume trusts this.
                    if (written - lastCheckpoint >= CheckpointInterval)
                    {
                        var _cpT0 = System.Diagnostics.Stopwatch.GetTimestamp();
                        await dest.FlushAsync(ct).ConfigureAwait(false);
                        await _store.SetStateAsync(key, url, total, written, false, etag, ct).ConfigureAwait(false);
                        LastCheckpointMs += (System.Diagnostics.Stopwatch.GetTimestamp() - _cpT0)
                                            * (1000.0 / System.Diagnostics.Stopwatch.Frequency);
                        LastCheckpoints++;
                        lastCheckpoint = written;
                    }

                    tracker.Received = received;

                    var now = Environment.TickCount64;
                    if (now - lastReport >= ProgressIntervalMs)
                    {
                        progress?.Report(new ModelDownloadProgress(received, total, resumed));
                        // Same cadence for the shared view, so a cache UI refreshes without polling and
                        // without a JS crossing per chunk.
                        ActiveDownloadsChanged?.Invoke();
                        lastReport = now;
                    }
                }

                await FlushAsync().ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                // Record only what actually landed. The staging buffer is deliberately NOT flushed: a
                // failure mid-fill leaves its tail indeterminate, and writing it would corrupt the resume.
                // CancellationToken.None on purpose - this bookkeeping is exactly what a CANCELLED download
                // needs, so passing the cancelled token would skip it and throw away the resume point.
                try
                {
                    await dest.FlushAsync(CancellationToken.None).ConfigureAwait(false);
                    await _store.SetStateAsync(key, url, total, written, false, etag, CancellationToken.None)
                        .ConfigureAwait(false);
                }
                catch { /* the original failure is the one worth reporting */ }

                // An aborted read surfaces as a JS AbortError; callers cancelled with a token and expect
                // the token's exception.
                if (ct.IsCancellationRequested && ex is not OperationCanceledException)
                    throw new OperationCanceledException($"Model download cancelled: {url}", ex, ct);
                throw;
            }
            finally
            {
                writeBuffer?.Dispose();
            }

            await dest.FlushAsync(ct).ConfigureAwait(false);

            async Task FlushAsync()
            {
                if (bufferFill == 0 || writeBuffer == null) return;
                if (bufferFill == WriteBufferSize)
                {
                    await WriteChunkAsync(dest, writeBuffer, ct).ConfigureAwait(false);
                }
                else
                {
                    // SubArray is a VIEW onto the same storage - writes the filled prefix, copies nothing.
                    using var view = writeBuffer.SubArray(0, bufferFill);
                    await WriteChunkAsync(dest, view, ct).ConfigureAwait(false);
                }
                writeCount++;
                written += bufferFill;
                bufferFill = 0;
            }
        }

        LastDownloadChunks = chunkCount;
        LastDownloadWrites = writeCount;
        LastDownloadBytes = received - resumeFrom;

        // A short read is a truncated download, not a model. Leave the entry incomplete so the next attempt
        // resumes it, rather than caching it and failing much later inside a proto reader.
        if (total > 0 && written != total)
        {
            await _store.SetStateAsync(key, url, total, written, false, etag, ct).ConfigureAwait(false);
            throw new IOException(
                $"Model download truncated for {url}: got {written} of {total} bytes. " +
                "The partial entry was kept and the next attempt will resume it.");
        }

        await _store.SetStateAsync(key, url, total > 0 ? total : written, written, true, etag, ct).ConfigureAwait(false);
        progress?.Report(new ModelDownloadProgress(received, total > 0 ? total : written, resumed));
    }

    /// <summary>
    /// Write a JS-side chunk to the store's stream without a managed copy when the stream supports it.
    /// </summary>
    /// <remarks>An <c>IJSWriteStream</c> takes the <see cref="Uint8Array"/> directly. Anything else is a
    /// non-JS store (a desktop file store), where a managed copy is the only option and is correct.</remarks>
    private static async Task WriteChunkAsync(Stream dest, Uint8Array chunk, CancellationToken ct)
    {
        var t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        try
        {
            if (dest is SpawnDev.SpawnJS.Toolbox.IJSWriteStream js)
            {
                await js.WriteUint8ArrayAsync(chunk, ct).ConfigureAwait(false);
                return;
            }
            await dest.WriteAsync(chunk.ReadBytes(), ct).ConfigureAwait(false);
        }
        finally
        {
            AddStoreWrite((System.Diagnostics.Stopwatch.GetTimestamp() - t0)
                          * (1000.0 / System.Diagnostics.Stopwatch.Frequency));
        }
    }

    /// <summary>
    /// Total file size from the response headers. For a 206 that is the total after the '/' in
    /// <c>Content-Range: bytes A-B/TOTAL</c>; a 206's Content-Length is the PART length, never the file size.
    /// </summary>
    private static long ParseTotalBytes(Headers headers, long resumeFrom)
    {
        var contentRange = headers.Get("content-range");
        if (!string.IsNullOrEmpty(contentRange))
        {
            var slash = contentRange.LastIndexOf('/');
            if (slash >= 0 && long.TryParse(contentRange[(slash + 1)..].Trim(), out var fromRange))
                return fromRange;
        }
        var contentLength = headers.Get("content-length");
        if (!string.IsNullOrEmpty(contentLength) && long.TryParse(contentLength, out var len))
            return resumeFrom + len;
        return -1;
    }

    private SemaphoreSlim GetGate(string key)
    {
        lock (_keyGates)
        {
            if (!_keyGates.TryGetValue(key, out var gate))
                _keyGates[key] = gate = new SemaphoreSlim(1, 1);
            return gate;
        }
    }

    private static string? NullIfEmpty(string? value) => string.IsNullOrEmpty(value) ? null : value;
}
