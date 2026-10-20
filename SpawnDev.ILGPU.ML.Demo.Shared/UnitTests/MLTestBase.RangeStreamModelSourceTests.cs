using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// <see cref="BrowserStorageEstimate"/> and <see cref="RangeStreamModelSource"/>: how much of a big model set a browser origin caches.
/// </summary>
/// <remarks>
/// 2026-10-18: the TRELLIS.2 demo treated <c>quota - usage</c> from <c>navigator.storage.estimate()</c> as a budget and so cached only ~10 GB of
/// 11 GB of weights. Chrome 151 reports <c>usage + 10 GiB</c> whatever it would really accept (14 GiB written against a 10 GiB "quota", fresh profile,
/// different drive, no flags), so the 10 GB was never a limit. These tests pin the planner to that.
/// </remarks>
public abstract partial class MLTestBase
{
    // Real hub files with sizes probed independently (Content-Range of a 0-0 request): the planner's choices are checked against these numbers.
    private const string Rs_Repo = "Xenova/distilgpt2";
    private const long Rs_TokenizerJson = 2_107_653, Rs_VocabJson = 798_156, Rs_MergesTxt = 456_318;
    private static readonly (string Repo, string Path)[] Rs_Files = { (Rs_Repo, "tokenizer.json"), (Rs_Repo, "vocab.json"), (Rs_Repo, "merges.txt") };

    /// <summary>Chrome's shape: nothing used, 10 GiB reported; then used 14 GiB, 24 GiB reported. Headroom never shrinks, so it is a window and not a budget.</summary>
    [TestMethod]
    public Task RangeStreamModelSource_ChromeStorageWindowIsOpenEnded_ExactHeadroomIsABudget() => RunTest(async accelerator =>
    {
        const long GiB = 1L << 30;
        foreach (var usageGiB in new long[] { 0, 3, 14 })
        {
            var chrome = new BrowserStorageEstimate(Quota: (usageGiB + 10) * GiB, Usage: usageGiB * GiB);
            if (!chrome.HeadroomIsOpenEnded) throw new Exception($"usage {usageGiB} GiB / quota {usageGiB + 10} GiB must be open-ended");
            if (chrome.PlanningBudget() < 1000 * GiB) throw new Exception($"open-ended headroom must not bound a plan, got {chrome.PlanningBudget()}");
        }
        var exact = new BrowserStorageEstimate(Quota: 5 * GiB, Usage: 1 * GiB);   // 4 GiB left, and a browser that says so
        if (exact.HeadroomIsOpenEnded) throw new Exception("4 GiB of headroom is below the window: it is an exact figure");
        if (exact.PlanningBudget(marginBytes: 300_000_000) != 4 * GiB - 300_000_000) throw new Exception($"exact budget wrong: {exact.PlanningBudget(300_000_000)}");
        var over = new BrowserStorageEstimate(Quota: 2 * GiB, Usage: 3 * GiB);
        if (over.Headroom != 0 || over.PlanningBudget() != 0) throw new Exception("usage above quota must plan nothing, not a negative budget");
        Console.WriteLine("[RangeStream] BrowserStorageEstimate window/exact/over: PASS");
        await Task.CompletedTask;
    });

    /// <summary>A real DOMException named QuotaExceededError, rejected through a real JS Promise and awaited the way SpawnJS surfaces a failed browser call.</summary>
    private static async Task<JSException> Rs_QuotaErrorAsync()
    {
        using var reason = new DOMException("The quota has been exceeded.", "QuotaExceededError");
        using var promise = new Promise((Function resolve, Function reject) => reject.CallVoid(null, reason));
        try { await promise.ThenAsync<object?>(CancellationToken.None); }
        catch (JSException ex) { return ex; }
        throw new Exception("the rejected promise did not surface as a JSException");
    }

    private static (HubModelSource Hub, OpfsModelCache Cache) Rs_NewHub(SpawnJSRuntime js, HttpClient http, string dir)
    {
        var cache = new OpfsModelCache(js) { CacheDirectoryName = dir };
        return (new HubModelSource(cache, http), cache);
    }

    /// <summary>The plan with REAL file sizes from the hub: open-ended or unknown headroom caches every file, an exact one picks the best fitting subset.</summary>
    [TestMethod(Timeout = 120000)]
    public Task RangeStreamModelSource_PlanTrustsOnlyAnExactHeadroom() => RunTest(async accelerator =>
    {
        var js = RequireBrowserRuntime();
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var (hub, cache) = Rs_NewHub(js, http, "ilgpu-ml-test-rangeplan");
        using var _ = cache;
        if (!await cache.IsAvailableAsync()) throw new UnsupportedTestException("OPFS unavailable in this context");
        foreach (var (repo, path) in Rs_Files) await hub.Store.RemoveAsync(hub.CacheKey(repo, path));   // cold: cached files would change the arithmetic
        long total = Rs_TokenizerJson + Rs_VocabJson + Rs_MergesTxt;

        // independent cross-check of the sizes the numbers below rest on
        foreach (var (repo, path, expected) in new[] { (Rs_Repo, "tokenizer.json", Rs_TokenizerJson), (Rs_Repo, "vocab.json", Rs_VocabJson), (Rs_Repo, "merges.txt", Rs_MergesTxt) })
        {
            long served = await ContentLengthAsync(http, HuggingFaceClient.GetDownloadUrl(repo, path, hub.Revision), default);
            if (served != expected) throw new UnsupportedTestException($"hub serves {repo}/{path} as {served} bytes, expected {expected}");
        }

        // Chrome-shaped headroom (exactly the 10 GiB window): every file is planned for the cache. THE regression: the old plan bounded this by 10 GiB - 300 MB.
        var ranged = new RangeStreamModelSource(hub);
        var (cached, streamed) = await ranged.PlanCacheAsync(Rs_Files, new BrowserStorageEstimate(Quota: (3L << 30) + (10L << 30), Usage: 3L << 30));
        if (cached != total || streamed != 0 || ranged.CacheFiles.Count != 3) throw new Exception($"open-ended: cached {cached}, streamed {streamed}, files {ranged.CacheFiles.Count} (expected {total}, 0, 3)");

        // PRODUCTION SCALE: the real TRELLIS.2 weight list (~11 GB, sizes only - nothing downloads) against Chrome's window. The 3 MB files above fit
        // under the old 10 GiB - 300 MB budget too and could not tell the fix from the bug; this set does not fit under it (RED-CHECKED: the old budget streams ~1.3 GB).
        var trellis = new RangeStreamModelSource(hub);
        var (tCached, tStreamed) = await trellis.PlanCacheAsync(Trellis2Weights.HubWeightFiles(), new BrowserStorageEstimate(Quota: (3L << 30) + (10L << 30), Usage: 3L << 30));
        if (tCached + tStreamed <= (10L << 30)) throw new UnsupportedTestException($"TRELLIS.2 weights total {tCached + tStreamed} bytes: not above the 10 GiB window, the case proves nothing");
        if (tStreamed != 0 || trellis.CacheFiles.Count != Trellis2Weights.HubWeightFiles().Count)
            throw new Exception($"TRELLIS.2 on Chrome's window: cached {tCached}, streamed {tStreamed}, planned {trellis.CacheFiles.Count}/{Trellis2Weights.HubWeightFiles().Count} (expected everything cached)");

        // unknown estimate (no StorageManager): the same
        (cached, streamed) = await ranged.PlanCacheAsync(Rs_Files, (BrowserStorageEstimate?)null);
        if (cached != total || streamed != 0) throw new Exception($"unknown estimate: cached {cached}, streamed {streamed}");

        // exact 2,600,000 bytes of headroom, no margin: tokenizer+merges = 2,563,971 is the biggest subset that fits (tokenizer+vocab = 2,905,809 does not)
        (cached, streamed) = await ranged.PlanCacheAsync(Rs_Files, new BrowserStorageEstimate(Quota: 2_600_000, Usage: 0), marginBytes: 0);
        if (cached != Rs_TokenizerJson + Rs_MergesTxt || streamed != Rs_VocabJson
            || !ranged.CacheFiles.Contains($"{Rs_Repo}/tokenizer.json") || !ranged.CacheFiles.Contains($"{Rs_Repo}/merges.txt") || ranged.CacheFiles.Contains($"{Rs_Repo}/vocab.json"))
            throw new Exception($"exact: cached {cached}, streamed {streamed}, files [{string.Join(", ", ranged.CacheFiles)}]");

        // exact and full: nothing may be cached, all of it streams
        (cached, streamed) = await ranged.PlanCacheAsync(Rs_Files, new BrowserStorageEstimate(Quota: 1_000_000, Usage: 1_000_000), marginBytes: 0);
        if (cached != 0 || streamed != total || ranged.CacheFiles.Count != 0) throw new Exception($"full: cached {cached}, streamed {streamed}");
        Console.WriteLine("[RangeStream] plan open-ended / unknown / exact / full: PASS");
    });

    /// <summary>The browser's real error shape is recognised, however it is wrapped, and nothing else is.</summary>
    [TestMethod(Timeout = 60000)]
    public Task RangeStreamModelSource_RecognisesQuotaExceededError() => RunTest(async accelerator =>
    {
        RequireBrowserRuntime();
        var thrown = await Rs_QuotaErrorAsync();
        if (thrown.Name != "QuotaExceededError") throw new Exception($"JSException.Name was '{thrown.Name}', expected QuotaExceededError");
        if (!RangeStreamModelSource.IsQuotaExceeded(thrown)) throw new Exception("a real QuotaExceededError was not recognised");
        if (!RangeStreamModelSource.IsQuotaExceeded(new IOException("download failed", thrown))) throw new Exception("a wrapped QuotaExceededError was not recognised");
        if (RangeStreamModelSource.IsQuotaExceeded(new IOException("disk"))) throw new Exception("a plain IOException was taken for a quota error");
        if (RangeStreamModelSource.IsQuotaExceeded(new JSException("aborted", "AbortError"))) throw new Exception("an AbortError was taken for a quota error");
        if (RangeStreamModelSource.IsQuotaExceeded(null)) throw new Exception("null was taken for a quota error");
        Console.WriteLine("[RangeStream] QuotaExceededError recognised, wrapped and bare: PASS");
    });

    /// <summary>
    /// The browser refuses to store a file the plan promised room for: that file streams, its half-written entry is gone, the other files still
    /// cache, and the load succeeds. A full disk cannot be summoned, so the refusal itself is injected (a real QuotaExceededError, thrown in place of
    /// the OPFS download); everything after it - the fallback, the cleanup, the real ranged stream from the hub - is the production path.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public Task RangeStreamModelSource_RefusedWriteFallsBackToStreaming() => RunTest(async accelerator =>
    {
        var js = RequireBrowserRuntime();
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var (hub, cache) = Rs_NewHub(js, http, "ilgpu-ml-test-rangefallback");
        using var _ = cache;
        if (!await cache.IsAvailableAsync()) throw new UnsupportedTestException("OPFS unavailable in this context");
        foreach (var (repo, path) in Rs_Files) await hub.Store.RemoveAsync(hub.CacheKey(repo, path));

        var ranged = new QuotaRefusingSource(hub, js, refuse: $"{Rs_Repo}/tokenizer.json");
        var fallbacks = new List<string>();
        ranged.CacheFallback += fallbacks.Add;
        await ranged.PlanCacheAsync(Rs_Files, new BrowserStorageEstimate(Quota: (10L << 30), Usage: 0));   // Chrome-shaped: everything planned
        if (ranged.CacheFiles.Count != 3) throw new Exception($"setup: expected 3 planned files, got {ranged.CacheFiles.Count}");

        // the refused file: streams, correct length and bytes, nothing left in OPFS, planned set updated, event raised once
        byte[] head = new byte[64], expectedHead = new byte[64];
        await using (var s = await ranged.OpenAsync(Rs_Repo, "tokenizer.json"))
        {
            if (s.Length != Rs_TokenizerJson) throw new Exception($"streamed length {s.Length} != {Rs_TokenizerJson}");
            await s.ReadExactlyAsync(head);
        }
        using (var req = new HttpRequestMessage(HttpMethod.Get, HuggingFaceClient.GetDownloadUrl(Rs_Repo, "tokenizer.json", hub.Revision)))
        {
            req.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(0, 63);
            using var res = await http.SendAsync(req);
            expectedHead = await res.Content.ReadAsByteArrayAsync();
        }
        if (!head.AsSpan().SequenceEqual(expectedHead)) throw new Exception("streamed bytes differ from the first 64 bytes served by the models host");
        if (await hub.Store.ExistsAsync(hub.CacheKey(Rs_Repo, "tokenizer.json"))) throw new Exception("the refused file left an OPFS entry behind");
        if (ranged.CacheFiles.Contains($"{Rs_Repo}/tokenizer.json")) throw new Exception("the refused file is still planned for the cache");
        if (fallbacks.Count != 1 || fallbacks[0] != $"{Rs_Repo}/tokenizer.json") throw new Exception($"CacheFallback fired {fallbacks.Count}x: [{string.Join(", ", fallbacks)}]");

        // a file the browser accepts still caches (not an all-or-nothing fallback)
        await using (var s = await ranged.OpenAsync(Rs_Repo, "merges.txt")) { if (s.Length != Rs_MergesTxt) throw new Exception($"merges length {s.Length}"); }
        if (!await hub.Store.ExistsAsync(hub.CacheKey(Rs_Repo, "merges.txt"))) throw new Exception("an accepted file was not cached");
        if (fallbacks.Count != 1) throw new Exception("an accepted file raised CacheFallback");

        // any other failure is NOT swallowed as a quota fallback
        ranged.RefuseWith = new IOException("network down");
        bool surfaced = false;
        try { await using var s = await ranged.OpenAsync(Rs_Repo, "vocab.json"); }
        catch (IOException) { surfaced = true; }
        if (!surfaced) throw new Exception("a non-quota failure was swallowed");

        foreach (var (repo, path) in Rs_Files) await hub.Store.RemoveAsync(hub.CacheKey(repo, path));
        Console.WriteLine("[RangeStream] refused write -> streams, entry removed, other file cached, other errors surface: PASS");
    });

    /// <summary>Throws the browser's QuotaExceededError in place of the OPFS download for one file.</summary>
    private sealed class QuotaRefusingSource(HubModelSource hub, SpawnJSRuntime js, string refuse) : RangeStreamModelSource(hub)
    {
        public Exception? RefuseWith;
        protected override async Task<Stream> DownloadIntoCacheAsync(string repoId, string filePath, CancellationToken ct)
        {
            if (RefuseWith != null) throw RefuseWith;
            if (repoId.Trim('/') + "/" + filePath.TrimStart('/') != refuse) return await base.DownloadIntoCacheAsync(repoId, filePath, ct);
            throw await Rs_QuotaErrorAsync();
        }
    }
}
