namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// Where models come from: <c>https://models.spawndev.com</c> (Cloudflare R2: free egress, plain static files, CORS open, Range works), with
/// Hugging Face as the fallback. The two use the SAME path layout (<c>{org}/{repo}/resolve/{revision}/{file}</c>), so a URL on the models host
/// maps to its Hugging Face twin by swapping the host. The SpawnDev hub (hub.spawndev.com) no longer serves models; it keeps the WebTorrent
/// tracker only, and redirects its old <c>/hf</c>, <c>/ollama</c> and <c>/src</c> endpoints here for clients already in the wild.
/// </summary>
/// <remarks>
/// <para>
/// 🔴 Every file served here must be reachable the same way a browser would fetch it: the host allows any request header and answers
/// Range with 206 on the FIRST request too, and Cloudflare compression and edge caching are OFF on it (see
/// <c>Projects\models.spawndev.com\README.md</c> for why each is off: each one broke browser downloads once).
/// </para>
/// <para>
/// Layout of the files that are not Hugging Face repos: Ollama layers at <c>ollama/{model}/{tag}/{model}-{tag}-{layer}.gguf</c>, members of a
/// release archive (an archive cannot be range-read, so its members are stored as plain files) at
/// <c>{host}{path of the archive without its extension}/{member without its top folder}</c>, for example
/// <c>github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-zh-en-emilia/vocos_24khz.onnx</c>.
/// </para>
/// </remarks>
public static class ModelsHost
{
    /// <summary>The SpawnDev models host.</summary>
    public const string DefaultBaseUrl = "https://models.spawndev.com";

    /// <summary>The Hugging Face origin: the fallback, with the same path layout.</summary>
    public const string HuggingFaceBaseUrl = "https://huggingface.co";

    /// <summary>The base URL downloads use. Point it at your own copy of the layout (or at <see cref="HuggingFaceBaseUrl"/>) to override.</summary>
    public static string BaseUrl { get; set; } = DefaultBaseUrl;

    /// <summary>
    /// Raised (url that failed, the Hugging Face url tried instead, why) each time a download falls back. For logs and tests; null by default so the
    /// library stays quiet.
    /// </summary>
    public static event Action<string, string, string>? Fallback;

    /// <summary>Where <paramref name="file"/> of <paramref name="repoId"/> lives on the models host: <c>{base}/{repo}/resolve/{revision}/{file}</c>.</summary>
    public static string GetUrl(string repoId, string file, string revision = "main")
        => $"{BaseUrl.TrimEnd('/')}/{repoId.Trim('/')}/resolve/{revision}/{file.TrimStart('/')}";

    /// <summary>Where an Ollama model layer lives: the name the hub cached it under.</summary>
    public static string GetOllamaUrl(string model, string tag, string layer)
        => $"{BaseUrl.TrimEnd('/')}/ollama/{model.Trim('/')}/{tag.Trim('/')}/{model.Trim('/')}-{tag.Trim('/')}-{layer.Trim('/')}.gguf";

    private static readonly string[] ArchiveExtensions = { ".tar.bz2", ".tar.gz", ".tgz", ".zip" };

    /// <summary>
    /// Where a member of a release archive lives on the models host (the archive itself is not served: a compressed archive cannot be range-read).
    /// <paramref name="member"/> is the path inside the archive INCLUDING its top folder, as the archive lists it
    /// (<c>sherpa-onnx-zipvoice-distill-zh-en-emilia/vocos_24khz.onnx</c>).
    /// </summary>
    public static string GetArchiveMemberUrl(string archiveUrl, string member)
    {
        if (!Uri.TryCreate(archiveUrl, UriKind.Absolute, out var u)) throw new ArgumentException("not an absolute URL", nameof(archiveUrl));
        var path = Uri.UnescapeDataString(u.AbsolutePath);
        var ext = ArchiveExtensions.FirstOrDefault(e => path.EndsWith(e, StringComparison.OrdinalIgnoreCase))
            ?? throw new ArgumentException($"'{archiveUrl}' is not a known archive type ({string.Join(", ", ArchiveExtensions)})", nameof(archiveUrl));
        int slash = member.IndexOf('/');
        if (slash < 0 || slash == member.Length - 1) throw new ArgumentException($"member '{member}' must sit under the archive's top folder", nameof(member));
        return $"{BaseUrl.TrimEnd('/')}/{u.Host}{path[..^ext.Length]}/{member[(slash + 1)..]}";
    }

    /// <summary>
    /// The Hugging Face twin of a URL on the models host, or null when there is none (the URL is not on the models host, or it is an Ollama layer or an
    /// archive member, which have no Hugging Face equivalent).
    /// </summary>
    public static string? GetFallbackUrl(string url)
    {
        var prefix = BaseUrl.TrimEnd('/') + "/";
        if (!url.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)) return null;
        var path = url[prefix.Length..];
        var seg = path.Split('?')[0].Split('/');                   // [datasets/]org/repo/resolve/revision/file...
        int d = seg.Length > 0 && seg[0] == "datasets" ? 1 : 0;
        if (seg.Length < 5 + d || seg[2 + d] != "resolve") return null;
        return $"{HuggingFaceBaseUrl}/{path}";
    }

    /// <summary>True for an answer that means "this host cannot give you the file right now" and a second source might: not found, forbidden, rate limited, server errors.</summary>
    public static bool ShouldFallBack(System.Net.HttpStatusCode status)
        => status is System.Net.HttpStatusCode.NotFound or System.Net.HttpStatusCode.Forbidden or System.Net.HttpStatusCode.TooManyRequests
            or System.Net.HttpStatusCode.RequestTimeout || (int)status >= 500;

    /// <summary>Total size of a URL: a 0-0 range GET reading <c>Content-Range</c>, with HEAD as a fallback; falls back to Hugging Face like every other request here.</summary>
    /// <remarks>
    /// Range probe FIRST, deliberately: a 0-0 range GET with <see cref="HttpCompletionOption.ResponseHeadersRead"/> is exactly one request and no body,
    /// same as HEAD, and it works against origins that refuse HEAD as well as those that do not. HEAD remains the fallback for an origin that refuses ranges.
    /// </remarks>
    public static async Task<long> ProbeSizeAsync(HttpClient http, string url, CancellationToken ct = default)
    {
        try
        {
            using var req = new HttpRequestMessage(HttpMethod.Get, url);
            req.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(0, 0);
            using var res = await SendAsync(http, req, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
            if (res.IsSuccessStatusCode)
            {
                // A 206's Content-Length is the PART (1 byte) - only Content-Range carries the file size.
                var total = res.Content.Headers.ContentRange?.Length;
                if (total is > 0) return total.Value;
                if (res.StatusCode == System.Net.HttpStatusCode.OK && res.Content.Headers.ContentLength is > 0 and var full)
                    return full;
            }
        }
        catch (HttpRequestException) { /* fall through to HEAD */ }

        using var head = new HttpRequestMessage(HttpMethod.Head, url);
        using var headRes = await SendAsync(http, head, HttpCompletionOption.ResponseContentRead, ct).ConfigureAwait(false);
        headRes.EnsureSuccessStatusCode();
        return headRes.Content.Headers.ContentLength ?? -1;
    }

    // ---------------------------------------------------------------------------------------------------
    // mirror index: which repos and files the models host has (mirrors.json, written by the mirror tool)
    // ---------------------------------------------------------------------------------------------------

    /// <summary>The index file at the root of the models host: <c>{"repos": {"org/repo": ["file", ...]}}</c>, every mirrored repo and its files.</summary>
    public const string MirrorIndexFile = "mirrors.json";

    /// <summary>
    /// Consult <see cref="MirrorIndexFile"/> before asking the models host for a file (default on): a repo or file that is not mirrored goes straight to
    /// Hugging Face, instead of a request that the host answers 404 (one red console line per file in a browser, and a wasted round trip). When the index
    /// cannot be read (a custom <see cref="BaseUrl"/> without one, no network) every file is tried on the host first, as before.
    /// </summary>
    public static bool UseMirrorIndex { get; set; } = true;

    private static readonly HttpClient IndexHttp = new() { Timeout = TimeSpan.FromSeconds(20) };
    private static readonly object IndexLock = new();
    private static Task<Dictionary<string, HashSet<string>>?>? _index;
    private static string? _indexBase;

    private static Task<Dictionary<string, HashSet<string>>?> MirrorIndexAsync()
    {
        lock (IndexLock)
        {
            var b = BaseUrl.TrimEnd('/');
            if (_index == null || _indexBase != b) { _indexBase = b; _index = LoadIndexAsync(b); }
            return _index;
        }
    }

    private static async Task<Dictionary<string, HashSet<string>>?> LoadIndexAsync(string baseUrl)
    {
        try
        {
            using var res = await IndexHttp.GetAsync($"{baseUrl}/{MirrorIndexFile}").ConfigureAwait(false);
            if (!res.IsSuccessStatusCode) return null;
            using var doc = System.Text.Json.JsonDocument.Parse(await res.Content.ReadAsStringAsync().ConfigureAwait(false));
            var repos = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
            foreach (var r in doc.RootElement.GetProperty("repos").EnumerateObject())
                repos[r.Name] = new HashSet<string>(r.Value.EnumerateArray().Select(f => f.GetString() ?? ""), StringComparer.Ordinal);
            return repos;
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or System.Text.Json.JsonException or KeyNotFoundException or InvalidOperationException)
        {
            return null;
        }
    }

    /// <summary>
    /// The URL to ask first for <paramref name="url"/>: itself, or its Hugging Face twin when the mirror index says the models host does not have that file
    /// (see <see cref="UseMirrorIndex"/>). A URL that is not a repo file on the models host comes back unchanged.
    /// </summary>
    public static async Task<string> PreferredUrlAsync(string url)
    {
        if (!UseMirrorIndex) return url;
        var alt = GetFallbackUrl(url);
        if (alt == null) return url;
        var idx = await MirrorIndexAsync().ConfigureAwait(false);
        if (idx == null) return url;
        var path = url[(BaseUrl.TrimEnd('/').Length + 1)..].Split('?')[0];
        var seg = path.Split('/');                                  // [datasets/]org/repo/resolve/revision/file...
        int d = seg[0] == "datasets" ? 1 : 0;
        string repo = string.Join('/', seg.Take(2 + d)), file = Uri.UnescapeDataString(string.Join('/', seg.Skip(4 + d)));
        return idx.TryGetValue(repo, out var files) && files.Contains(file) ? url : alt;
    }

    /// <summary>Forget the loaded mirror index (it is read again on the next request): for tests, or after a new repo was mirrored.</summary>
    public static void ResetMirrorIndex() { lock (IndexLock) { _index = null; _indexBase = null; } }

    /// <summary>Tell listeners a download fell back. Called by the transports.</summary>
    public static void ReportFallback(string failedUrl, string fallbackUrl, string reason) => Fallback?.Invoke(failedUrl, fallbackUrl, reason);

    /// <summary>
    /// <c>http.SendAsync</c> that falls back to the Hugging Face twin when the request is for the models host and it answers
    /// <see cref="ShouldFallBack"/> or cannot be reached. Any other request goes through untouched.
    /// </summary>
    /// <param name="fallbackAuthorization">Sent ONLY on the Hugging Face request (a token for a gated model belongs to Hugging Face, never to the models host).</param>
    public static async Task<HttpResponseMessage> SendAsync(HttpClient http, HttpRequestMessage request,
        HttpCompletionOption completion = HttpCompletionOption.ResponseContentRead, CancellationToken cancellationToken = default,
        System.Net.Http.Headers.AuthenticationHeaderValue? fallbackAuthorization = null)
    {
        // A relative request URI is resolved by HttpClient against its BaseAddress (the Blazor pattern, and the tests' "models/..." paths): resolve it the same way
        // to decide whether it is on the models host; one that stays relative has no Hugging Face twin and goes through untouched.
        var requestUri = request.RequestUri is { IsAbsoluteUri: false } rel && http.BaseAddress != null ? new Uri(http.BaseAddress, rel) : request.RequestUri;
        var alt = requestUri is { IsAbsoluteUri: true } uri ? GetFallbackUrl(uri.AbsoluteUri) : null;
        if (alt == null) return await http.SendAsync(request, completion, cancellationToken).ConfigureAwait(false);

        // not mirrored (the index says so): straight to Hugging Face, no 404 from the models host first
        if (await PreferredUrlAsync(requestUri!.AbsoluteUri).ConfigureAwait(false) == alt)
        {
            var direct = new HttpRequestMessage(request.Method, alt);   // not disposed here: the returned response may still reference it
            foreach (var h in request.Headers) direct.Headers.TryAddWithoutValidation(h.Key, h.Value);
            if (fallbackAuthorization != null) direct.Headers.Authorization = fallbackAuthorization;
            return await http.SendAsync(direct, completion, cancellationToken).ConfigureAwait(false);
        }

        string reason;
        try
        {
            var response = await http.SendAsync(request, completion, cancellationToken).ConfigureAwait(false);
            if (!ShouldFallBack(response.StatusCode)) return response;
            reason = $"HTTP {(int)response.StatusCode}";
            response.Dispose();
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException && !cancellationToken.IsCancellationRequested)
        {
            reason = ex.GetType().Name + ": " + ex.Message;
        }
        ReportFallback(requestUri!.AbsoluteUri, alt, reason);
        var second = new HttpRequestMessage(request.Method, alt);   // not disposed here: the returned response may still reference it
        foreach (var h in request.Headers) second.Headers.TryAddWithoutValidation(h.Key, h.Value);
        if (fallbackAuthorization != null) second.Headers.Authorization = fallbackAuthorization;
        return await http.SendAsync(second, completion, cancellationToken).ConfigureAwait(false);
    }
}
