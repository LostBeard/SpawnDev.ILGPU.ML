namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// The browser's fetch cache mode for an <see cref="HttpRequestMessage"/>, set by the raw option key (what Blazor's SetBrowserRequestCache sets), so this library takes no
/// dependency on the Blazor assembly. A <c>Cache-Control</c> request header is not an alternative cross-origin: it is not CORS-safelisted, so the browser sends a preflight first,
/// which static hosts refuse (GitHub Pages answers OPTIONS with 405) and the request fails with "Failed to fetch".
/// </summary>
internal static class BrowserCache
{
    private static readonly HttpRequestOptionsKey<IDictionary<string, object>> FetchOptionsKey = new("WebAssemblyFetchOptions");

    /// <summary><c>no-cache</c> (revalidate) or <c>no-store</c>: the fetch cache mode in the browser, a Cache-Control header elsewhere.</summary>
    public static void SetMode(HttpRequestMessage request, string mode)
    {
        if (OperatingSystem.IsBrowser())
            request.Options.Set(FetchOptionsKey, new Dictionary<string, object> { ["cache"] = mode });
        else
            request.Headers.CacheControl = new System.Net.Http.Headers.CacheControlHeaderValue { NoCache = true, NoStore = mode == "no-store" };
    }

    /// <summary>
    /// Every RANGE request in the browser goes around the HTTP cache. MEASURED in Chrome 2026-11-25 (a file GitHub Pages serves gzip-encoded): once a page has fetched it without a
    /// range, the cache holds the ENCODED entry and Chrome answers later ranges of it wrongly: "Failed to fetch" (net::ERR_CACHE_OPERATION_NOT_SUPPORTED), a 200 with the whole file,
    /// or a 206 with no body. With no-store the same ranges were byte-exact. Nothing is lost: a ranged model load is not served from the HTTP cache anyway.
    /// </summary>
    public static void RangeNoStore(HttpRequestMessage request)
    {
        if (OperatingSystem.IsBrowser())
            request.Options.Set(FetchOptionsKey, new Dictionary<string, object> { ["cache"] = "no-store" });
    }
}
