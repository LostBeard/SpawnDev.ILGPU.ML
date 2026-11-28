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
/// <see cref="UnsupportedTestException"/>. The source is the SpawnDev models host, not huggingface.co - it
/// supplies CORS, answers Range from the first byte, and falls back to Hugging Face by itself.
/// </remarks>
public abstract partial class MLTestBase
{
    /// <summary>A real models-host file, big enough to arrive in many chunks and to make a resume meaningful.</summary>
    private static readonly string OpfsCacheTestUrl = HuggingFaceClient.GetDownloadUrl("Xenova/distilgpt2", "tokenizer.json");

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

        using var cache = new OpfsModelCache(js) { CacheDirectoryName = "ilgpu-ml-test-opfscache", PartBytes = 256 * 1024 };
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
        long cut = await TruncateCacheEntryAsync(js, cache.CacheDirectoryName, key, OpfsCacheTestUrl, total / 3, total);
        if (cut <= 0 || cut >= total) throw new Exception($"setup: the entry was cut to {cut} of {total} bytes; the test needs a part boundary in between");

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

        if (cache.LastDownloadBytes != total - cut)
            throw new Exception($"The resume transferred {cache.LastDownloadBytes:N0} bytes; the missing tail is {total - cut:N0} (of {total:N0}).");
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

        var badUrl = HuggingFaceClient.GetDownloadUrl("Xenova/distilgpt2", "definitely-not-a-real-file.bin");
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
    /// EVERY cache operation works after the browser clears the origin's storage, not only a load (Lore 2026-11-24, 5.8.28-local.3: with a cache cap set the
    /// plan trims first, TrimAsync listed through the stale directory handle and threw before the load's own recovery could run). The cache directory is
    /// deleted behind the cache's back before each operation.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task OpfsModelCache_EveryOperationRecoversWhenTheBrowserClearsStorage() => await RunTest(async accelerator =>
    {
        var js = RequireBrowserRuntime();
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        const string dirName = "ilgpu-ml-test-cleared-ops", key = "fixture";
        using var cache = new OpfsModelCache(js) { CacheDirectoryName = dirName };
        // two first callers at once: the second must wait for the directory, not read "OPFS is not available"
        var available = await Task.WhenAll(cache.IsAvailableAsync(), cache.IsAvailableAsync());
        if (!available[0] && !available[1]) throw new UnsupportedTestException("OPFS unavailable in this context");
        if (available[0] != available[1]) throw new Exception($"two concurrent first calls disagree: {available[0]}, {available[1]}");
        using var hub = new HubModelSource(cache, http);
        using var navigator = js.Get<Navigator>("navigator");
        using var storage = navigator.Storage;
        using var root = await storage.GetDirectory();
        var payload = new byte[5000];
        for (int i = 0; i < payload.Length; i++) payload[i] = (byte)(i * 7 + 1);

        async Task AfterClear(string what, Func<Task> operation)
        {
            await cache.PutAsync(key, new MemoryStream(payload));
            if (!await cache.IsCompleteAsync(key)) throw new Exception($"{what}: setup did not store the fixture");
            await root.RemoveEntry(dirName, recursive: true);
            try { await operation(); }
            catch (Exception ex) { throw new Exception($"{what} after the storage was cleared: {ex.Message}", ex); }
            using var again = await root.GetDirectoryHandle(dirName);   // throws when the operation did not recreate it
        }
        try
        {
            await AfterClear("ListCachedAsync", async () => { var l = await cache.ListCachedAsync(); if (l.Count != 0) throw new Exception($"{l.Count} entries"); });
            await AfterClear("GetCacheSizeAsync", async () => { var n = await cache.GetCacheSizeAsync(); if (n != 0) throw new Exception($"{n} bytes"); });
            await AfterClear("TrimAsync", async () => { var t = await cache.TrimAsync(1); if (t != (0L, 0L)) throw new Exception($"{t}"); });
            await AfterClear("ClearAllAsync", () => cache.ClearAllAsync());
            await AfterClear("PutAsync", async () =>
            {
                await cache.PutAsync("other", new MemoryStream(payload));
                await using var s = await cache.OpenCachedAsync("other") ?? throw new Exception("the stored entry is not there");
                var got = new byte[payload.Length];
                await s.ReadExactlyAsync(got);
                if (!got.AsSpan().SequenceEqual(payload)) throw new Exception("stored bytes differ");
            });
            // Lore's case: plan with a cap (it trims), then load what was planned
            await AfterClear("PlanCacheAsync with a cap", async () =>
            {
                var ranged = new RangeStreamModelSource(hub) { MaxCacheBytes = 1L << 30 };
                var (cached, streamed) = await ranged.PlanCacheAsync(new[] { (Rs_Repo, "merges.txt") }, (BrowserStorageEstimate?)null);
                if (cached != Rs_MergesTxt || streamed != 0) throw new Exception($"cached {cached}, streamed {streamed}");
                await using var s = await ranged.OpenAsync(Rs_Repo, "merges.txt");
                if (s.Length != Rs_MergesTxt) throw new Exception($"length {s.Length}");
            });
            // several loads hit the stale handle at once
            await AfterClear("parallel loads", async () =>
            {
                var streams = await Task.WhenAll(hub.OpenAsync(Rs_Repo, "merges.txt"), hub.OpenAsync(Rs_Repo, "vocab.json"), hub.OpenAsync(Rs_Repo, "merges.txt"));
                var lengths = streams.Select(s => s.Length).ToArray();
                foreach (var s in streams) await s.DisposeAsync();
                if (!lengths.SequenceEqual(new[] { Rs_MergesTxt, Rs_VocabJson, Rs_MergesTxt })) throw new Exception($"lengths {string.Join(", ", lengths)}");
            });
            Console.WriteLine("[OpfsCache] every operation recovers after the storage was cleared: PASS");
        }
        finally { try { await root.RemoveEntry(dirName, recursive: true); } catch { } }
    });

    /// <summary>
    /// Continuing a download into a file that vanished must FAIL, not extend an empty file to the resume point: the gap would be zeros that count as
    /// downloaded bytes, and the finished file has the right length.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task OpfsModelCache_ContinuingAVanishedFileFailsInsteadOfZeroFilling() => await RunTest(async accelerator =>
    {
        var js = RequireBrowserRuntime();
        const string dirName = "ilgpu-ml-test-vanished", key = "partial";
        using var cache = new OpfsModelCache(js) { CacheDirectoryName = dirName };
        if (!await cache.IsAvailableAsync()) throw new UnsupportedTestException("OPFS unavailable in this context");
        using var navigator = js.Get<Navigator>("navigator");
        using var storage = navigator.Storage;
        using var root = await storage.GetDirectory();
        try
        {
            var w = await cache.OpenWriteAsync(key, 0);
            await using (w.ConfigureAwait(false)) { await w.WriteAsync(new byte[4000]); await w.FlushAsync(); }
            await cache.SetStateAsync(key, "http://example/x", 8000, 4000, false, null);
            if ((await cache.GetStateAsync(key)).BytesWritten != 4000) throw new Exception("setup: the partial entry is not resumable");
            using (var dir = await root.GetDirectoryHandle(dirName)) await dir.RemoveEntry(key + ".parts", recursive: true);
            bool failed = false;
            try { await using var s = await cache.OpenWriteAsync(key, 4000); }
            catch (IOException) { failed = true; }
            if (!failed) throw new Exception("continuing at 4000 in a vanished file succeeded: the first 4000 bytes are zeros");
            var state = await cache.GetStateAsync(key);
            if (state.BytesWritten != 0) throw new Exception($"after the failure {state.BytesWritten} bytes still count as downloaded");
            Console.WriteLine("[OpfsCache] continuing a vanished file fails, nothing counts as downloaded: PASS");
        }
        finally { try { await root.RemoveEntry(dirName, recursive: true); } catch { } }
    });

    /// <summary>
    /// Cut a cache entry back to the committed parts that fit in <paramref name="keepBytes"/> and rewrite its sidecar to say "incomplete": the state a
    /// download that was closed part-way leaves. Returns the bytes kept (a part boundary).
    /// </summary>
    private static async Task<long> TruncateCacheEntryAsync(SpawnJSRuntime js, string cacheDirName, string key,
        string url, long keepBytes, long total)
    {
        using var navigator = js.Get<Navigator>("navigator");
        using var storage = navigator.Storage;
        using var root = await storage.GetDirectory();
        using var dir = await root.GetDirectoryHandle(cacheDirName, create: true);
        using var parts = await dir.GetDirectoryHandle(key + ".parts");

        long kept = 0;
        bool cutting = false;
        for (int i = 0; ; i++)
        {
            string name = i.ToString("D6", System.Globalization.CultureInfo.InvariantCulture);
            long size;
            try { using var h = await parts.GetFileHandle(name); size = await h.GetSize(); }
            catch { break; }
            if (!cutting && kept + size <= keepBytes) { kept += size; continue; }
            cutting = true;
            await parts.RemoveEntry(name);
        }

        var meta = $"v=1\nurl={url}\ntotal={total}\nreceived={kept}\ncomplete=0\netag=\n";
        using var metaHandle = await dir.GetFileHandle(key + ".meta", create: true);
        using var writable = await metaHandle.CreateWritable();
        using var bytes = new Uint8Array(System.Text.Encoding.UTF8.GetBytes(meta));
        await writable.Write(bytes);
        await writable.Close();
        return kept;
    }

    /// <summary>
    /// 🔴 A download whose writer DIES (the browser is closed, the tab crashes: nothing is closed, nothing is disposed) keeps what it had committed, the
    /// next attempt continues from there, and the dead attempt leaves nothing on disk beyond one part.
    /// </summary>
    /// <remarks>
    /// The failure this guards (2026-11-27, the live /video demo): with one file per entry Chrome held a download's bytes in a swap file until close, so
    /// a browser closed mid-download left the real file at 0 bytes and the swap file (1.02 GB measured, 6 GB for the text encoder) on disk for good,
    /// and the retry downloaded the file again from byte 0. TJ's OS drive nearly filled. The older resume test cut a finished file back with a clean
    /// close, which is not how a download dies, and passed all along.
    /// Here the writer is ABANDONED: written to, never flushed, closed or disposed. RED-CHECKED against the one-file store: it reports 0 resumable
    /// bytes and the test fails on the first assertion.
    /// </remarks>
    [TestMethod(Timeout = 120000)]
    public async Task OpfsModelCache_ADownloadWhoseWriterDiesKeepsItsPartsAndContinues() => await RunTest(async accelerator =>
    {
        var js = RequireBrowserRuntime();
        const string dirName = "ilgpu-ml-test-dead-writer", key = "model";
        const int part = 256 * 1024;
        using var navigator = js.Get<Navigator>("navigator");
        using var storage = navigator.Storage;
        using var root = await storage.GetDirectory();
        try { await root.RemoveEntry(dirName, recursive: true); } catch { }
        object? abandoned = null;
        try
        {
            // the reference bytes, through another cache directory
            byte[] full;
            using (var refCache = new OpfsModelCache(js) { CacheDirectoryName = dirName + "-ref" })
            {
                if (!await refCache.IsAvailableAsync()) throw new UnsupportedTestException("OPFS unavailable in this context");
                var r = await refCache.OpenOrDownloadAsync(OpfsCacheTestUrl, "ref");
                await using (r.ConfigureAwait(false)) { full = new byte[r.Length]; await r.ReadExactlyAsync(full); }
            }
            long total = full.Length;
            if (total < 4L * part) throw new UnsupportedTestException($"the test file is {total} bytes; the test needs at least {4 * part}");

            // the dying attempt: two thirds of the file written and acknowledged in the sidecar, then nothing - no flush, no close, no dispose
            long sent = total * 2 / 3;
            using (var first = new OpfsModelCache(js) { CacheDirectoryName = dirName, PartBytes = part })
            {
                var w = await first.OpenWriteAsync(key, 0);
                abandoned = w;
                for (long o = 0; o < sent; o += 100_000) await w.WriteAsync(full.AsMemory((int)o, (int)Math.Min(100_000, sent - o)));
                await first.SetStateAsync(key, OpfsCacheTestUrl, total, sent, false, null);
            }

            // a new session
            using var cache = new OpfsModelCache(js) { CacheDirectoryName = dirName, PartBytes = part };
            var state = await cache.GetStateAsync(key);
            long committed = sent / part * part;
            Console.WriteLine($"[OpfsCache] dead writer: {sent:N0} of {total:N0} bytes were written, the store offers {state.BytesWritten:N0} to continue from (committed parts: {committed:N0})");
            if (state.Complete) throw new Exception("the interrupted entry reports complete");
            if (state.BytesWritten != committed)
                throw new Exception($"After a writer died with {sent:N0} bytes written the store offers {state.BytesWritten:N0} bytes to continue from; the committed parts hold {committed:N0}. " +
                                    "Zero means the attempt's bytes are lost (and, with a single file, still on disk in a swap file).");

            var reports = new List<ModelDownloadProgress>();
            var stream = await cache.OpenOrDownloadAsync(OpfsCacheTestUrl, key, new Progress<ModelDownloadProgress>(p => reports.Add(p)));
            await using (stream.ConfigureAwait(false))
            {
                if (stream.Length != total) throw new Exception($"the continued file is {stream.Length} bytes, expected {total}");
                var got = new byte[total];
                await stream.ReadExactlyAsync(got);
                for (long i = 0; i < total; i++) if (got[i] != full[i]) throw new Exception($"the continued file differs from the reference at offset {i}");
            }
            if (cache.LastDownloadBytes != total - committed)
                throw new Exception($"The second attempt transferred {cache.LastDownloadBytes:N0} bytes; only the {total - committed:N0} after the committed parts were missing.");

            // the disk: the model once, plus at most the one part the dead writer never committed (and the sidecar)
            long held = await cache.GetCacheSizeAsync();
            Console.WriteLine($"[OpfsCache] dead writer: second attempt transferred {cache.LastDownloadBytes:N0} bytes; the cache directory holds {held:N0} bytes for a {total:N0} byte model");
            if (held > total + part + 4096)
                throw new Exception($"The cache directory holds {held:N0} bytes for a {total:N0} byte model: the dead attempt left {held - total:N0} bytes behind (allowed: one part of {part:N0}).");
            Console.WriteLine("[OpfsCache] a download whose writer died continues from its committed parts and leaves no more than a part: PASS");
        }
        finally
        {
            GC.KeepAlive(abandoned);
            try { await root.RemoveEntry(dirName, recursive: true); } catch { }
            try { await root.RemoveEntry(dirName + "-ref", recursive: true); } catch { }
        }
    });

    /// <summary>
    /// <see cref="OpfsModelCache.SweepOrphansAsync()"/> never costs a write that is in progress: a swap file younger than
    /// <see cref="OpfsModelCache.OrphanSwapMinAge"/> is not touched, and when the age rule is switched off the writer still commits its bytes
    /// (MEASURED 2026-11-27: Chrome refuses to remove the swap file of a writer that is alive). Chrome only: other browsers keep no visible swap
    /// file, and the test skips there.
    /// The removal of a DEAD writer's swap file cannot be made to happen inside one page (the writer has to die with its browser):
    /// <c>tools/demo-smoke/video_demo_run.py --kill-after-gb</c> ends Chrome mid-download and checks that no swap file is left afterwards.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task OpfsModelCache_SweepNeverCostsAWriteInProgress() => await RunTest(async accelerator =>
    {
        var js = RequireBrowserRuntime();
        const string dirName = "ilgpu-ml-test-orphans";
        using var navigator = js.Get<Navigator>("navigator");
        using var storage = navigator.Storage;
        using var root = await storage.GetDirectory();
        try { await root.RemoveEntry(dirName, recursive: true); } catch { }
        object? abandoned = null;
        try
        {
            using var cache = new OpfsModelCache(js) { CacheDirectoryName = dirName };
            if (!await cache.IsAvailableAsync()) throw new UnsupportedTestException("OPFS unavailable in this context");
            using var dir = await root.GetDirectoryHandle(dirName);

            // a write in progress, as another tab's would be
            const int size = 300_000;
            using var handle = await dir.GetFileHandle("old-model", create: true);
            var writable = await handle.CreateWritable();
            abandoned = writable;
            using (var bytes = new Uint8Array(new byte[size])) await writable.Write(bytes);

            async Task<List<string>> SwapFilesAsync()
            {
                var names = new List<string>();
                foreach (var e in await dir.ValuesList()) { using var h = e; if (h.Name.EndsWith(".crswap", StringComparison.Ordinal)) names.Add(h.Name); }
                return names;
            }
            if ((await SwapFilesAsync()).Count == 0) throw new UnsupportedTestException("this browser keeps no visible swap file for an open writable");

            long freed = await cache.SweepOrphansAsync();
            if (freed != 0 || (await SwapFilesAsync()).Count == 0)
                throw new Exception($"a swap file a few milliseconds old was removed ({freed} bytes freed): it could be another tab's write in progress");

            cache.OrphanSwapMinAge = TimeSpan.Zero;
            freed = await cache.SweepOrphansAsync();
            int leftWhileOpen = (await SwapFilesAsync()).Count;
            await writable.Close();
            long committed = await handle.GetSize();
            Console.WriteLine($"[OpfsCache] sweep with the age rule off: freed {freed:N0} bytes, swap files left while the writer was open: {leftWhileOpen}; the writer then committed {committed:N0} bytes");
            if (committed != size) throw new Exception($"a write in progress lost its bytes to the sweep: the file holds {committed} of {size} bytes");
            if ((await SwapFilesAsync()).Count != 0) throw new Exception("a swap file is left after the writer closed");
            Console.WriteLine("[OpfsCache] a sweep never costs a write in progress: PASS");
        }
        finally
        {
            GC.KeepAlive(abandoned);
            try { await root.RemoveEntry(dirName, recursive: true); } catch { }
        }
    });

    /// <summary>
    /// Reading a big entry end to end the way a model upload does (IJSReadStream, 16 MiB at a time): stored as parts (every download since 5.8.36; the
    /// cache serves a Blob over the part files) against stored as one file (entries written before 5.8.36; served by OPFSStream). Same 384 MiB
    /// (3 parts of 128 MiB), three reads each, alternating; both reads must return the same bytes. Lore's SpawnGen visits (2026-11-27) read one-file
    /// entries only, so the parted path for big models had no measurement. MB/s go to the console.
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task Benchmark_OpfsModelCache_PartedVsOneFileRead() => await RunTest(async accelerator =>
    {
        var js = RequireBrowserRuntime();
        const string dirName = "ilgpu-ml-test-partread";
        const int chunk = 16 << 20, chunks = 24;                       // 384 MiB
        const long total = (long)chunk * chunks;
        using var navigator = js.Get<Navigator>("navigator");
        using var storage = navigator.Storage;
        using var root = await storage.GetDirectory();
        try { await root.RemoveEntry(dirName, recursive: true); } catch { }
        using var cache = new OpfsModelCache(js) { CacheDirectoryName = dirName };
        if (!await cache.IsAvailableAsync()) throw new UnsupportedTestException("OPFS unavailable in this context");
        try
        {
            // the same bytes for both: one 16 MiB pattern repeated (the byte at offset o is (o * 31 + o / 4093) & 255)
            var pattern = new byte[chunk];
            for (int i = 0; i < chunk; i++) pattern[i] = (byte)((i * 31 + i / 4093) & 255);
            using (var src = new RepeatStream(pattern, chunks)) await cache.PutAsync("parted", src);        // parts of PartBytes (128 MiB)
            using (var dir = await root.GetDirectoryHandle(dirName, create: true))
            {
                using (var fh = await dir.GetFileHandle("onefile", create: true))
                using (var w = await fh.CreateWritable())
                using (var u = new Uint8Array(pattern))
                {
                    for (int c = 0; c < chunks; c++) await w.Write(u);
                    await w.Close();
                }
                var meta = $"v=1\nurl=\ntotal={total}\nreceived={total}\ncomplete=1\netag=\n";
                using var mh = await dir.GetFileHandle("onefile.meta", create: true);
                using var mw = await mh.CreateWritable();
                using var mb = new Uint8Array(System.Text.Encoding.UTF8.GetBytes(meta));
                await mw.Write(mb); await mw.Close();
            }
            async Task<(double Ms, string Type, byte[] Probe)> ReadAll(string key)
            {
                Stream s;
                if (key == "onefile-blob")
                {
                    // the one-file entry through the same reader the parts use (a BlobStream over its File): splits "Blob reader" from "several parts"
                    using var d = await root.GetDirectoryHandle(dirName);
                    using var h = await d.GetFileHandle("onefile");
                    s = new SpawnDev.SpawnJS.Toolbox.BlobStream(await h.GetFile());
                }
                else s = await cache.OpenCachedAsync(key) ?? throw new Exception($"'{key}' did not open as a complete entry");
                await using (s.ConfigureAwait(false))
                {
                    if (s.Length != total) throw new Exception($"'{key}' is {s.Length} bytes, expected {total}");
                    var jr = s as SpawnDev.SpawnJS.Toolbox.IJSReadStream ?? throw new Exception($"'{key}' opened as {s.GetType().Name}, not an IJSReadStream");
                    var probe = new byte[chunks * 8];
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    long read = 0; int c = 0;
                    while (read < total)
                    {
                        using var a = await jr.ReadUint8ArrayAsync(chunk);
                        long n = a.Length; if (n <= 0) break;
                        if (c < chunks) { using var sub = a.SubArray(0, 8); var head = sub.ReadBytes(); System.Array.Copy(head, 0, probe, c * 8, Math.Min(8, head.Length)); }
                        read += n; c++;
                    }
                    double ms = sw.Elapsed.TotalMilliseconds;
                    if (read != total) throw new Exception($"'{key}' read {read} of {total} bytes");
                    return (ms, s.GetType().Name, probe);
                }
            }
            var parted = new List<double>(); var one = new List<double>(); var oneBlob = new List<double>(); string pType = "", oType = ""; byte[]? pProbe = null, oProbe = null, bProbe = null;
            for (int round = 0; round < 3; round++)
            {
                var p = await ReadAll("parted"); parted.Add(p.Ms); pType = p.Type; pProbe = p.Probe;
                var o = await ReadAll("onefile"); one.Add(o.Ms); oType = o.Type; oProbe = o.Probe;
                var b = await ReadAll("onefile-blob"); oneBlob.Add(b.Ms); bProbe = b.Probe;
            }
            if (!pProbe!.AsSpan().SequenceEqual(oProbe) || !bProbe!.AsSpan().SequenceEqual(oProbe)) throw new Exception("the three reads returned different bytes");
            parted.Sort(); one.Sort(); oneBlob.Sort();
            double mbps(double ms) => total / 1e6 / (ms / 1000);
            var report = $"384 MiB read in 16 MiB pieces, median of 3: parts ({pType}, 3 x 128 MiB) {parted[1]:F0} ms = {mbps(parted[1]):F0} MB/s; "
                + $"one file ({oType}) {one[1]:F0} ms = {mbps(one[1]):F0} MB/s; one file through BlobStream {oneBlob[1]:F0} ms = {mbps(oneBlob[1]):F0} MB/s";
            Console.WriteLine("[OpfsCache] " + report);
            return report;
        }
        finally { try { await root.RemoveEntry(dirName, recursive: true); } catch { } }
    });

    /// <summary>A read-only stream of <paramref name="block"/> repeated <paramref name="count"/> times (synthetic test content, nothing allocated per read).</summary>
    private sealed class RepeatStream(byte[] block, int count) : Stream
    {
        private long _pos;
        private readonly long _len = (long)block.Length * count;
        public override bool CanRead => true; public override bool CanSeek => false; public override bool CanWrite => false;
        public override long Length => _len;
        public override long Position { get => _pos; set => throw new NotSupportedException(); }
        public override int Read(byte[] buffer, int offset, int n)
        {
            int done = 0;
            while (done < n && _pos < _len)
            {
                int at = (int)(_pos % block.Length), take = (int)Math.Min(Math.Min(n - done, block.Length - at), _len - _pos);
                Buffer.BlockCopy(block, at, buffer, offset + done, take);
                done += take; _pos += take;
            }
            return done;
        }
        public override void Flush() { }
        public override long Seek(long o, SeekOrigin s) => throw new NotSupportedException();
        public override void SetLength(long v) => throw new NotSupportedException();
        public override void Write(byte[] b, int o, int n) => throw new NotSupportedException();
    }

    /// <summary>
    /// What reopening a complete multi-part entry costs: the Wan umT5 encoder opens its 6.4 GB file once per layer (24 layers + the final norm,
    /// UmT5Encoder.LoadLayerAsync), and the file is 48 parts of 128 MiB. Here the 2.1 MB test file is stored as 48 parts and reopened 25 times; each open
    /// reads 8 bytes near the end (the last part) and must match the first open's bytes. The per-open time goes to the console.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task Benchmark_OpfsModelCache_ReopenAPartedEntry() => await RunTest(async accelerator =>
    {
        var js = RequireBrowserRuntime();
        const string dirName = "ilgpu-ml-test-reopen";
        using var probe = new OpfsModelCache(js) { CacheDirectoryName = dirName };
        if (!await probe.IsAvailableAsync()) throw new UnsupportedTestException("OPFS unavailable in this context");
        var key = OpfsModelCache.UrlToCacheKey(OpfsCacheTestUrl) + ".reopen";
        await probe.RemoveAsync(key);
        long total;
        {
            var s = await probe.OpenOrDownloadAsync(OpfsCacheTestUrl, key + ".size");
            await using (s.ConfigureAwait(false)) total = s.Length;
            await probe.RemoveAsync(key + ".size");
        }
        const int parts = 48, opens = 25;
        using var cache = new OpfsModelCache(js) { CacheDirectoryName = dirName, PartBytes = (total + parts - 1) / parts };
        try
        {
            var tail = new byte[8]; long at = total - 16;
            var first = await cache.OpenOrDownloadAsync(OpfsCacheTestUrl, key);              // downloads and commits 48 parts
            await using (first.ConfigureAwait(false)) { first.Position = at; await first.ReadExactlyAsync(tail); }
            var times = new List<double>();
            var got = new byte[8];
            for (int i = 0; i < opens; i++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var s = await cache.OpenOrDownloadAsync(OpfsCacheTestUrl, key);              // the same URL, as the encoder's reopen passes it
                await using (s.ConfigureAwait(false)) { s.Position = at; await s.ReadExactlyAsync(got); }
                times.Add(sw.Elapsed.TotalMilliseconds);
                if (!got.AsSpan().SequenceEqual(tail)) throw new Exception($"open {i + 1}: the last part's bytes differ from the first open's");
            }
            times.Sort();
            double sum = 0; foreach (var t in times) sum += t;
            // the split: IsCompleteAsync reads the sidecar and lists the parts (sizes) like an open, without the Blob, the read or the sidecar write an open makes
            var check = new List<double>();
            for (int i = 0; i < opens; i++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                if (!await cache.IsCompleteAsync(key)) throw new Exception("the entry stopped being complete");
                check.Add(sw.Elapsed.TotalMilliseconds);
            }
            check.Sort();
            var report = $"reopen of a complete {parts}-part entry ({total:N0} bytes), {opens} opens with an 8-byte read in the last part: median {times[opens / 2]:F1} ms, max {times[^1]:F1} ms, total {sum:F0} ms; IsCompleteAsync (sidecar + part listing) median {check[opens / 2]:F1} ms";
            Console.WriteLine("[OpfsCache] " + report);
            return report;
        }
        finally { await cache.RemoveAsync(key); }
    });
}
