using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// <see cref="OpfsModelCache"/> - HTTP model delivery straight into OPFS, with no WebTorrent and no trip
/// through the .NET/WASM managed heap.
/// </summary>
/// <remarks>
/// These are browser-lane tests: OPFS does not exist on the desktop lanes, which skip via
/// <see cref="UnsupportedTestException"/>. The source is the SpawnDev hub, not huggingface.co - the hub is
/// what supplies CORS, caches the file, and keeps us out of HuggingFace's rate limiter.
/// </remarks>
public abstract partial class MLTestBase
{
    /// <summary>A real hub-served file, big enough to arrive in many chunks and to make a resume meaningful.</summary>
    private const string OpfsCacheTestUrl = "https://hub.spawndev.com:44365/hf/Xenova/distilgpt2/tokenizer.json";

    /// <summary>Browser-lane guard shared by every test here.</summary>
    private static SpawnJSRuntime RequireBrowserRuntime()
    {
        var js = SpawnJSRuntime.Instance;
        if (js == null || !js.IsBrowser)
            throw new UnsupportedTestException("OpfsModelCache is browser-only (no OPFS on this lane)");
        return js;
    }

    /// <summary>
    /// Download through the cache, then prove the SECOND open is served from OPFS and is byte-identical.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task OpfsModelCache_DownloadsCachesAndServes() => await RunTest(async accelerator =>
    {
        var js = RequireBrowserRuntime();
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");

        using var cache = new OpfsModelCache(js) { CacheDirectoryName = "ilgpu-ml-test-opfscache" };
        if (!await cache.IsAvailableAsync())
            throw new UnsupportedTestException("OPFS unavailable in this context");

        var key = OpfsModelCache.UrlToCacheKey(OpfsCacheTestUrl);
        await cache.RemoveAsync(key);   // cold start, every run

        // Independent expectation: ask the server directly, via HttpClient rather than via the cache, so
        // the assertion below is a cross-check and not the cache agreeing with itself.
        long expected = await ContentLengthAsync(http, OpfsCacheTestUrl, default);
        if (expected <= 0) throw new UnsupportedTestException("Hub did not report a content-length");

        if (await cache.IsCompleteAsync(key))
            throw new Exception("Entry reported complete before anything was downloaded.");

        var reports = new List<ModelDownloadProgress>();
        var progress = new Progress<ModelDownloadProgress>(p => reports.Add(p));

        long firstLength;
        byte[] firstHead;
        var stream = await cache.OpenOrDownloadAsync(OpfsCacheTestUrl, key, progress);
        await using (stream.ConfigureAwait(false))
        {
            firstLength = stream.Length;
            firstHead = new byte[64];
            stream.Position = 0;
            await stream.ReadExactlyAsync(firstHead);
        }

        if (firstLength != expected)
            throw new Exception($"Downloaded length {firstLength} != server content-length {expected}.");
        if (!await cache.IsCompleteAsync(key))
            throw new Exception("Entry is not marked complete after a successful download.");

        // Progress must have been reported and must finish at 100%. A Progress<T> post is async, so the
        // final report may still be in flight; assert on what did arrive plus the final state.
        Console.WriteLine($"[OpfsCache] progress reports: {reports.Count}");
        if (reports.Count == 0)
            throw new Exception("No progress was reported for a 2 MB download.");
        var last = reports[^1];
        Console.WriteLine($"[OpfsCache] last report: {last.BytesReceived}/{last.TotalBytes} " +
                          $"({last.Fraction:P1}) resumed={last.Resumed}");
        if (last.TotalBytes != expected)
            throw new Exception($"Progress total {last.TotalBytes} != server content-length {expected}.");

        // Second open: this must come straight off OPFS. Nothing re-downloads, and the bytes must match.
        var again = await cache.OpenCachedAsync(key)
            ?? throw new Exception("OpenCachedAsync returned null for an entry that reports complete.");
        await using (again.ConfigureAwait(false))
        {
            if (again.Length != firstLength)
                throw new Exception($"Cached length {again.Length} != first length {firstLength}.");
            var head = new byte[64];
            again.Position = 0;
            await again.ReadExactlyAsync(head);
            for (int i = 0; i < head.Length; i++)
                if (head[i] != firstHead[i])
                    throw new Exception($"Cached bytes differ from downloaded bytes at offset {i}.");
        }

        // It must be a JS-side stream, which is the whole point - that is what lets InferenceSession
        // upload weights JS->GPU without the model entering the managed heap.
        if (again is not SpawnDev.SpawnJS.Toolbox.IJSReadStream)
            throw new Exception($"Cached stream is {again.GetType().Name}, not an IJSReadStream.");

        Console.WriteLine($"[OpfsCache] {firstLength:N0} bytes cached and re-served from OPFS: PASS");
        await cache.RemoveAsync(key);
    });

    /// <summary>
    /// 🔴 The treason guard: downloading a 2 MB file must not allocate 2 MB of managed heap.
    /// </summary>
    /// <remarks>
    /// This is the test that would have caught <see cref="ModelCache"/>'s download path, which reads every
    /// chunk with <c>Uint8Array.ReadBytes()</c>, accumulates a <c>List&lt;byte[]&gt;</c>, and then
    /// concatenates it into one more <c>byte[]</c> - roughly 2x the file in managed allocations. Here the
    /// payload is only ever a JS <see cref="Uint8Array"/> handed from <c>fetch</c> to OPFS, so the managed
    /// allocation is interop bookkeeping and stays a small fraction of the file.
    /// <para>
    /// The threshold is 50% of the file rather than something tight: the point is to catch a whole-file
    /// copy (which lands at 100-200%), not to police ordinary interop overhead. RED-CHECKED by reinstating
    /// a per-chunk <c>ReadBytes()</c> and confirming this fails.
    /// </para>
    /// </remarks>
    [TestMethod(Timeout = 120000)]
    public async Task OpfsModelCache_DownloadDoesNotEnterManagedHeap() => await RunTest(async accelerator =>
    {
        var js = RequireBrowserRuntime();

        using var cache = new OpfsModelCache(js) { CacheDirectoryName = "ilgpu-ml-test-opfscache" };
        if (!await cache.IsAvailableAsync())
            throw new UnsupportedTestException("OPFS unavailable in this context");

        var key = OpfsModelCache.UrlToCacheKey(OpfsCacheTestUrl) + ".zerocopy";
        await cache.RemoveAsync(key);

        // Settle anything pending so the delta below is about the download and not about warm-up.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        var before = GC.GetTotalAllocatedBytes(precise: true);

        long length;
        var stream = await cache.OpenOrDownloadAsync(OpfsCacheTestUrl, key);
        await using (stream.ConfigureAwait(false))
        {
            length = stream.Length;   // NOTE: deliberately never read the payload into .NET here
        }

        var allocated = GC.GetTotalAllocatedBytes(precise: true) - before;
        var ratio = (double)allocated / length;
        Console.WriteLine($"[OpfsCache] file {length:N0} B, managed alloc during download {allocated:N0} B " +
                          $"({ratio:P1} of the file)");

        // What the source actually delivered, and what it cost in OPFS writes. This is the number that says
        // whether coalescing into WriteBufferSize is earning anything on this source.
        var avgChunk = cache.LastDownloadChunks > 0 ? cache.LastDownloadBytes / cache.LastDownloadChunks : 0;
        Console.WriteLine($"[OpfsCache] fetch chunks: {cache.LastDownloadChunks} (avg {avgChunk:N0} B) -> " +
                          $"OPFS writes: {cache.LastDownloadWrites} (buffer {cache.WriteBufferSize:N0} B)");

        if (length <= 0) throw new Exception("Downloaded nothing.");
        if (allocated > length / 2)
            throw new Exception(
                $"Download allocated {allocated:N0} bytes of managed heap for a {length:N0} byte file " +
                $"({ratio:P1}). Bulk bytes are crossing into .NET - the chunk must stay a Uint8Array from " +
                "fetch to OPFS.");

        Console.WriteLine("[OpfsCache] zero-copy download: PASS");
        await cache.RemoveAsync(key);
    });

    /// <summary>
    /// An interrupted download RESUMES from what it already has, and the resumed file is byte-correct.
    /// </summary>
    /// <remarks>
    /// The interruption is simulated the way it actually happens - a partial file plus a sidecar that says
    /// "incomplete, this many bytes confirmed" - so this exercises the real Range request, the real 206
    /// handling, and the real resume arithmetic. VERIFIED against the live hub: it answers
    /// <c>206</c> with <c>content-range: bytes N-…/TOTAL</c> and exposes Content-Range via CORS.
    /// </remarks>
    [TestMethod(Timeout = 120000)]
    public async Task OpfsModelCache_ResumesAPartialDownload() => await RunTest(async accelerator =>
    {
        var js = RequireBrowserRuntime();

        using var cache = new OpfsModelCache(js) { CacheDirectoryName = "ilgpu-ml-test-opfscache" };
        if (!await cache.IsAvailableAsync())
            throw new UnsupportedTestException("OPFS unavailable in this context");

        var key = OpfsModelCache.UrlToCacheKey(OpfsCacheTestUrl) + ".resume";
        await cache.RemoveAsync(key);

        // A complete reference copy to compare against.
        byte[] full;
        long total;
        var reference = await cache.OpenOrDownloadAsync(OpfsCacheTestUrl, key);
        await using (reference.ConfigureAwait(false))
        {
            total = reference.Length;
            full = new byte[total];
            reference.Position = 0;
            await reference.ReadExactlyAsync(full);
        }

        // Now cut it back to a partial and re-open: the cache must fetch only the tail.
        long cut = total / 3;
        await TruncateCacheEntryAsync(js, cache.CacheDirectoryName, key, OpfsCacheTestUrl, cut, total);

        if (await cache.IsCompleteAsync(key))
            throw new Exception("Entry still reports complete after being truncated - the sidecar is not being honoured.");

        var reports = new List<ModelDownloadProgress>();
        var progress = new Progress<ModelDownloadProgress>(p => reports.Add(p));

        var resumed = await cache.OpenOrDownloadAsync(OpfsCacheTestUrl, key, progress);
        await using (resumed.ConfigureAwait(false))
        {
            if (resumed.Length != total)
                throw new Exception($"Resumed length {resumed.Length} != original {total}.");
            var bytes = new byte[total];
            resumed.Position = 0;
            await resumed.ReadExactlyAsync(bytes);
            for (long i = 0; i < total; i++)
                if (bytes[i] != full[i])
                    throw new Exception($"Resumed file differs from the reference at offset {i}.");
        }

        var sawResume = reports.Exists(r => r.Resumed);
        Console.WriteLine($"[OpfsCache] resumed from {cut:N0}/{total:N0}; reports={reports.Count}; " +
                          $"flagged-resumed={sawResume}");
        if (!sawResume)
            throw new Exception(
                "No progress report was flagged Resumed - the hub answered 200 instead of 206, so the whole " +
                "file was re-downloaded rather than resumed.");

        Console.WriteLine("[OpfsCache] resume: PASS");
        await cache.RemoveAsync(key);
    });

    /// <summary>
    /// A failed response is NEVER cached. <c>fetch</c> resolves on 404 with an error BODY, and caching that
    /// body is how a 15-byte error page once became a permanently-cached "model" that failed identically on
    /// every later run until the cache was cleared by hand.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task OpfsModelCache_FailedDownloadIsNotCached() => await RunTest(async accelerator =>
    {
        var js = RequireBrowserRuntime();

        using var cache = new OpfsModelCache(js) { CacheDirectoryName = "ilgpu-ml-test-opfscache" };
        if (!await cache.IsAvailableAsync())
            throw new UnsupportedTestException("OPFS unavailable in this context");

        const string badUrl = "https://hub.spawndev.com:44365/hf/Xenova/distilgpt2/definitely-not-a-real-file.bin";
        var key = OpfsModelCache.UrlToCacheKey(badUrl);
        await cache.RemoveAsync(key);

        var threw = false;
        try
        {
            var s = await cache.OpenOrDownloadAsync(badUrl, key);
            await s.DisposeAsync();
        }
        catch (HttpRequestException ex)
        {
            threw = true;
            Console.WriteLine($"[OpfsCache] failed download threw as expected: {ex.Message}");
        }

        if (!threw)
            throw new Exception("A 404 download did not throw - an error body may have been cached as a model.");
        if (await cache.IsCompleteAsync(key))
            throw new Exception("A failed download left a COMPLETE cache entry behind.");

        Console.WriteLine("[OpfsCache] failed download not cached: PASS");
        await cache.RemoveAsync(key);
    });

    /// <summary>
    /// 🔴 IModelStore.PutAsync stores bytes from ANY Stream - and a JS-side source stays JS-side.
    /// </summary>
    /// <remarks>
    /// This is the transport-agnostic entry point: a torrent piece stream, a picked file, an HTTP response
    /// and a MemoryStream all arrive the same way. The zero-copy half is not wishful thinking - the copy
    /// uses Stream.CopyToAsync, and JSReadStreamBase overrides it to pump Uint8Array chunks JS-side when the
    /// destination is an IJSWriteStream. So this test puts an entry from ANOTHER OPFS entry (both JS-side)
    /// and asserts the managed allocation stays far below the payload, the same way the download guard does.
    /// </remarks>
    [TestMethod(Timeout = 120000)]
    public async Task OpfsModelCache_PutFromStreamStoresAndStaysJSSide() => await RunTest(async accelerator =>
    {
        var js = RequireBrowserRuntime();

        using var cache = new OpfsModelCache(js) { CacheDirectoryName = "ilgpu-ml-test-opfscache" };
        if (!await cache.IsAvailableAsync())
            throw new UnsupportedTestException("OPFS unavailable in this context");

        IModelStore store = cache;
        var sourceKey = OpfsModelCache.UrlToCacheKey(OpfsCacheTestUrl) + ".putsrc";
        var destKey = OpfsModelCache.UrlToCacheKey(OpfsCacheTestUrl) + ".putdst";
        await cache.RemoveAsync(sourceKey);
        await cache.RemoveAsync(destKey);

        // Get a real JS-side source stream (an OPFS entry), which is an IJSReadStream.
        long total;
        byte[] head = new byte[64];
        var src = await cache.OpenOrDownloadAsync(OpfsCacheTestUrl, sourceKey);
        await using (src.ConfigureAwait(false))
        {
            total = src.Length;
            src.Position = 0;
            await src.ReadExactlyAsync(head);
        }

        if (await store.ExistsAsync(destKey))
            throw new Exception("Destination reported present before anything was stored.");

        var reopened = await cache.OpenCachedAsync(sourceKey)
            ?? throw new Exception("Source entry vanished.");
        long allocated;
        await using (reopened.ConfigureAwait(false))
        {
            if (reopened is not SpawnDev.SpawnJS.Toolbox.IJSReadStream)
                throw new Exception($"Source is {reopened.GetType().Name}, not an IJSReadStream - the " +
                                    "zero-copy assertion below would be measuring the wrong thing.");
            GC.Collect();
            GC.WaitForPendingFinalizers();
            var before = GC.GetTotalAllocatedBytes(precise: true);
            await store.PutAsync(destKey, reopened);
            allocated = GC.GetTotalAllocatedBytes(precise: true) - before;
        }

        if (!await store.ExistsAsync(destKey))
            throw new Exception("Stored entry does not report as present and complete.");

        var dst = await store.OpenReadAsync(destKey)
            ?? throw new Exception("OpenReadAsync returned null for an entry that reports present.");
        await using (dst.ConfigureAwait(false))
        {
            if (dst.Length != total)
                throw new Exception($"Stored length {dst.Length} != source length {total}.");
            var got = new byte[64];
            dst.Position = 0;
            await dst.ReadExactlyAsync(got);
            for (int i = 0; i < got.Length; i++)
                if (got[i] != head[i])
                    throw new Exception($"Stored bytes differ from the source at offset {i}.");
        }

        var ratio = (double)allocated / total;
        Console.WriteLine($"[OpfsCache] PutAsync copied {total:N0} B JS->JS, managed alloc {allocated:N0} B ({ratio:P1})");
        if (allocated > total / 2)
            throw new Exception(
                $"PutAsync allocated {allocated:N0} bytes of managed heap for a {total:N0} byte JS-side " +
                $"source ({ratio:P1}). The CopyToAsync JS fast path is not being taken.");

        // And the stored entry must satisfy a plain OpenOrDownloadAsync without re-fetching it, even though
        // it has no origin URL recorded.
        var viaOpen = await cache.OpenOrDownloadAsync(OpfsCacheTestUrl, destKey);
        await using (viaOpen.ConfigureAwait(false))
        {
            if (viaOpen.Length != total)
                throw new Exception($"OpenOrDownloadAsync re-fetched a PutAsync entry: {viaOpen.Length} != {total}.");
        }

        Console.WriteLine("[OpfsCache] PutAsync from Stream: PASS");
        await cache.RemoveAsync(sourceKey);
        await cache.RemoveAsync(destKey);
    });

    /// <summary>Records progress SYNCHRONOUSLY, unlike <see cref="Progress{T}"/>.</summary>
    /// <remarks>
    /// 🔴 <c>Progress&lt;T&gt;.Report</c> POSTS to the captured SynchronizationContext, so its callbacks can
    /// still be in flight when the download returns. Counting reports through it measures the scheduler, not
    /// the downloader - which is exactly why the original "progress reports: 1" reading was not evidence of
    /// anything. <c>IProgress&lt;T&gt;</c> implemented directly is invoked inline.
    /// </remarks>
    private sealed class SyncProgress : IProgress<ModelDownloadProgress>
    {
        public readonly List<ModelDownloadProgress> Reports = new();
        public void Report(ModelDownloadProgress value) => Reports.Add(value);
    }

    /// <summary>
    /// 🔴 Progress must fire DURING a download, not only at the end.
    /// </summary>
    /// <remarks>
    /// The bug this guards: <c>lastReport</c> was seeded with <c>TickCount64</c>, suppressing the first
    /// in-loop report for a full <c>ProgressIntervalMs</c>, and nothing reported when the total became
    /// known. A file that finished faster than the interval therefore produced ONE report - the final 100% -
    /// which is indistinguishable from no progress at all, and useless for a progress bar.
    /// <para>
    /// Asserts on INTERMEDIATE reports (<c>BytesReceived &lt; TotalBytes</c>), because a test that only
    /// counts reports passes on a single final 100%.
    /// </para>
    /// </remarks>
    [TestMethod(Timeout = 120000)]
    public async Task OpfsModelCache_ProgressFiresDuringDownload() => await RunTest(async accelerator =>
    {
        var js = RequireBrowserRuntime();

        using var cache = new OpfsModelCache(js) { CacheDirectoryName = "ilgpu-ml-test-opfscache" };
        if (!await cache.IsAvailableAsync())
            throw new UnsupportedTestException("OPFS unavailable in this context");

        var key = OpfsModelCache.UrlToCacheKey(OpfsCacheTestUrl) + ".progress";
        await cache.RemoveAsync(key);

        // Report every chunk, so this measures the mechanism rather than the wall clock of one lane.
        cache.Downloader.ProgressIntervalMs = 0;

        var sink = new SyncProgress();
        var stream = await cache.OpenOrDownloadAsync(OpfsCacheTestUrl, key, sink);
        long total;
        await using (stream.ConfigureAwait(false)) total = stream.Length;

        var reports = sink.Reports;
        Console.WriteLine($"[OpfsCache] progress reports during download: {reports.Count}");
        if (reports.Count == 0)
            throw new Exception("No progress reported at all.");

        // 1) A report before any payload arrived, so a UI can render immediately with the total.
        var first = reports[0];
        if (first.BytesReceived != 0)
            throw new Exception($"First report was {first.BytesReceived} bytes, expected 0 - nothing " +
                                "reported before the transfer started, so a bar cannot render up front.");

        // 2) At least one report while genuinely mid-flight. This is the assertion that fails if only the
        //    final 100% is ever sent.
        int intermediate = reports.Count(r => r.BytesReceived > 0 && r.TotalBytes > 0 && r.BytesReceived < r.TotalBytes);
        if (intermediate == 0)
            throw new Exception(
                $"{reports.Count} report(s), but NONE were intermediate - progress only fired at the end, " +
                "which is useless as a progress bar.");

        // 3) Monotonic and terminating at 100%.
        long prev = -1;
        foreach (var r in reports)
        {
            if (r.BytesReceived < prev)
                throw new Exception($"Progress went backwards: {prev} -> {r.BytesReceived}.");
            prev = r.BytesReceived;
        }
        var last = reports[^1];
        if (last.BytesReceived != total || last.TotalBytes != total)
            throw new Exception($"Final report {last.BytesReceived}/{last.TotalBytes} != actual {total}.");

        var mid = reports.First(r => r.BytesReceived > 0 && r.BytesReceived < r.TotalBytes);
        Console.WriteLine($"[OpfsCache] {reports.Count} reports, {intermediate} intermediate; " +
                          $"first={first.BytesReceived}/{first.TotalBytes}, " +
                          $"a mid report={mid.BytesReceived}/{mid.TotalBytes} ({mid.Fraction:P1}), " +
                          $"last={last.BytesReceived}/{last.TotalBytes}");
        Console.WriteLine("[OpfsCache] progress fires during download: PASS");
        await cache.RemoveAsync(key);
    });

    /// <summary>
    /// 🔴 A download is cancellable, and a cancelled one leaves a RESUMABLE partial - never a usable entry.
    /// </summary>
    /// <remarks>
    /// Cancellation has to reach the network, not just the loop. Checking the token between chunks leaves
    /// the HTTP request running - bandwidth and connection stay committed, and on a multi-GB model a cancel
    /// would free nothing - so the fetch is wired to an AbortController. This asserts three things that each
    /// failed independently in earlier shapes: it throws OperationCanceledException (not a raw JS
    /// AbortError), it does NOT leave a complete entry, and what it kept is resumable to a byte-correct file.
    /// </remarks>
    [TestMethod(Timeout = 120000)]
    public async Task OpfsModelCache_DownloadIsCancellable() => await RunTest(async accelerator =>
    {
        var js = RequireBrowserRuntime();

        using var cache = new OpfsModelCache(js) { CacheDirectoryName = "ilgpu-ml-test-opfscache" };
        if (!await cache.IsAvailableAsync())
            throw new UnsupportedTestException("OPFS unavailable in this context");

        var key = OpfsModelCache.UrlToCacheKey(OpfsCacheTestUrl) + ".cancel";
        await cache.RemoveAsync(key);

        // Reference copy, so the resumed result can be compared byte for byte.
        byte[] full;
        long total;
        var reference = await cache.OpenOrDownloadAsync(OpfsCacheTestUrl, key + ".ref");
        await using (reference.ConfigureAwait(false))
        {
            total = reference.Length;
            full = new byte[total];
            reference.Position = 0;
            await reference.ReadExactlyAsync(full);
        }

        // Cancel as soon as the first bytes land - mid-transfer, not before it starts.
        using var cts = new CancellationTokenSource();
        cache.Downloader.ProgressIntervalMs = 0;
        var seen = 0L;
        var trigger = new CancelAfterFirstBytes(cts, p => seen = p.BytesReceived);

        var threw = false;
        try
        {
            var s = await cache.OpenOrDownloadAsync(OpfsCacheTestUrl, key, trigger, cts.Token);
            await s.DisposeAsync();
        }
        catch (OperationCanceledException)
        {
            threw = true;
        }

        if (!threw)
        {
            // A very fast lane can finish before the cancel is observed. That is not a failure of
            // cancellation, but it does mean this run proved nothing - say so rather than passing quietly.
            throw new UnsupportedTestException(
                $"Download completed before cancellation could be observed (saw {seen:N0} bytes); " +
                "this lane is too fast to exercise the cancel path with a 2 MB file.");
        }

        if (await cache.IsCompleteAsync(key))
            throw new Exception("A cancelled download left a COMPLETE cache entry behind.");

        // What survived must be a valid prefix AND must resume to a byte-correct file.
        var resumed = await cache.OpenOrDownloadAsync(OpfsCacheTestUrl, key);
        await using (resumed.ConfigureAwait(false))
        {
            if (resumed.Length != total)
                throw new Exception($"Resumed-after-cancel length {resumed.Length} != {total}.");
            var bytes = new byte[total];
            resumed.Position = 0;
            await resumed.ReadExactlyAsync(bytes);
            for (long i = 0; i < total; i++)
                if (bytes[i] != full[i])
                    throw new Exception($"Resumed-after-cancel file differs from the reference at offset {i}.");
        }

        Console.WriteLine($"[OpfsCache] cancelled after {seen:N0} bytes, resumed to a byte-correct {total:N0}: PASS");
        await cache.RemoveAsync(key);
        await cache.RemoveAsync(key + ".ref");
    });

    /// <summary>
    /// Cache management: list with sizes, remove one, clear all - through <see cref="IModelStore"/>.
    /// </summary>
    /// <remarks>
    /// A cache a user cannot inspect or reclaim is a disk leak. This drives the whole surface a management
    /// UI needs and checks the numbers are real: listed sizes must match the actual entries, removing one
    /// must leave the others, and clearing must take the sidecars with the payloads (an orphan sidecar
    /// reads back as a phantom entry).
    /// </remarks>
    [TestMethod(Timeout = 120000)]
    public async Task OpfsModelCache_ListsRemovesAndClearsWithSizes() => await RunTest(async accelerator =>
    {
        var js = RequireBrowserRuntime();

        using var cache = new OpfsModelCache(js) { CacheDirectoryName = "ilgpu-ml-test-mgmt" };
        if (!await cache.IsAvailableAsync())
            throw new UnsupportedTestException("OPFS unavailable in this context");

        IModelStore store = cache;
        await store.ClearAsync();
        if ((await store.ListAsync()).Count != 0)
            throw new Exception("Store is not empty after ClearAsync.");
        if (await store.GetTotalSizeAsync() != 0)
            throw new Exception("Total size is non-zero after ClearAsync.");

        // Two entries from one download, so both sizes are known exactly.
        var keyA = "mgmt-a";
        var keyB = "mgmt-b";
        long size;
        var src = await cache.OpenOrDownloadAsync(OpfsCacheTestUrl, keyA);
        await using (src.ConfigureAwait(false)) size = src.Length;

        var copy = await store.OpenReadAsync(keyA) ?? throw new Exception("Entry A missing.");
        await using (copy.ConfigureAwait(false)) await store.PutAsync(keyB, copy);

        var listed = await store.ListAsync();
        Console.WriteLine($"[OpfsCache] listed {listed.Count} entries: " +
                          string.Join(", ", listed.Select(e => $"{e.Key}={e.SizeBytes:N0}{(e.Complete ? "" : " (partial)")}")));
        if (listed.Count != 2)
            throw new Exception($"Expected 2 entries, listed {listed.Count}.");
        foreach (var e in listed)
        {
            if (e.SizeBytes != size)
                throw new Exception($"Listed size for '{e.Key}' is {e.SizeBytes}, actual is {size}.");
            if (!e.Complete)
                throw new Exception($"Entry '{e.Key}' listed as incomplete after a successful store.");
        }

        var totalAfterTwo = await store.GetTotalSizeAsync();
        if (totalAfterTwo < size * 2)
            throw new Exception($"Total size {totalAfterTwo:N0} is less than the two payloads ({size * 2:N0}).");

        // Individual removal leaves the other entry intact.
        await store.RemoveAsync(keyA);
        var afterRemove = await store.ListAsync();
        if (afterRemove.Count != 1 || afterRemove[0].Key != keyB)
            throw new Exception($"After removing '{keyA}', expected only '{keyB}', got " +
                                $"[{string.Join(",", afterRemove.Select(e => e.Key))}].");
        if (await store.ExistsAsync(keyA))
            throw new Exception($"'{keyA}' still reports present after removal.");
        if (!await store.ExistsAsync(keyB))
            throw new Exception($"Removing '{keyA}' also destroyed '{keyB}'.");

        var totalAfterRemove = await store.GetTotalSizeAsync();
        if (totalAfterRemove >= totalAfterTwo)
            throw new Exception($"Total size did not drop after removal: {totalAfterTwo:N0} -> {totalAfterRemove:N0}.");
        Console.WriteLine($"[OpfsCache] total {totalAfterTwo:N0} -> {totalAfterRemove:N0} after removing one");

        // Clear takes everything, sidecars included.
        await store.ClearAsync();
        if ((await store.ListAsync()).Count != 0)
            throw new Exception("Entries survived ClearAsync.");
        if (await store.GetTotalSizeAsync() != 0)
            throw new Exception("Bytes survived ClearAsync - sidecars were probably orphaned.");
        if (await store.ExistsAsync(keyB))
            throw new Exception($"'{keyB}' still reports present after ClearAsync.");

        Console.WriteLine("[OpfsCache] list / remove / clear with sizes: PASS");
    });

    /// <summary>Cancels the moment the first non-zero progress report arrives.</summary>
    private sealed class CancelAfterFirstBytes : IProgress<ModelDownloadProgress>
    {
        private readonly CancellationTokenSource _cts;
        private readonly Action<ModelDownloadProgress> _observe;
        public CancelAfterFirstBytes(CancellationTokenSource cts, Action<ModelDownloadProgress> observe)
        { _cts = cts; _observe = observe; }
        public void Report(ModelDownloadProgress value)
        {
            _observe(value);
            if (value.BytesReceived > 0) _cts.Cancel();
        }
    }

    /// <summary>
    /// Cut a cache entry down to <paramref name="keepBytes"/> and rewrite its sidecar to say "incomplete",
    /// which is exactly the state an interrupted download leaves behind.
    /// </summary>
    private static async Task TruncateCacheEntryAsync(SpawnJSRuntime js, string cacheDirName, string key,
        string url, long keepBytes, long total)
    {
        using var navigator = js.Get<Navigator>("navigator");
        using var storage = navigator.Storage;
        using var root = await storage.GetDirectory();
        using var dir = await root.GetDirectoryHandle(cacheDirName, create: true);

        // ⚠️ FileAccess.Write, not ReadWrite: OPFSStream supports Read OR Write and throws
        // NotSupportedException for ReadWrite (it is either a read blob stream or a writable stream).
        var data = await SpawnDev.SpawnJS.Toolbox.OPFSStream.OpenPath(dir, key, FileMode.Open, FileAccess.Write);
        await using (data.ConfigureAwait(false))
        {
            data.SetLength(keepBytes);
            await data.FlushAsync();
        }

        var meta = $"v=1\nurl={url}\ntotal={total}\nreceived={keepBytes}\ncomplete=0\netag=\n";
        using var metaHandle = await dir.GetFileHandle(key + ".meta", create: true);
        using var writable = await metaHandle.CreateWritable();
        using var bytes = new Uint8Array(System.Text.Encoding.UTF8.GetBytes(meta));
        await writable.Write(bytes);
        await writable.Close();
    }
}
