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
