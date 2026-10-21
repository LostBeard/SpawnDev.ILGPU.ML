using System.Net;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.SpawnJS;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// <see cref="ModelsHost"/>: where models come from (https://models.spawndev.com, Hugging Face as the fallback). Real requests to the real hosts,
/// no mocks: the URL shapes are checked against what the host actually serves, and the fallback is exercised with files that exist on Hugging Face
/// and are NOT on the models host.
/// </summary>
/// <remarks>
/// Fixtures: <c>Xenova/distilgpt2/generation_config.json</c> (124 bytes) is on Hugging Face and deliberately NOT mirrored, so a request for it on the
/// models host is a 404 that the fallback must turn into a 200. If it ever gets mirrored these tests fail loudly (the first one asserts it is
/// absent), and another unmirrored file must be chosen.
/// </remarks>
public abstract partial class MLTestBase
{
    private const string UnmirroredRepo = "Xenova/distilgpt2", UnmirroredFile = "generation_config.json";
    private const int UnmirroredBytes = 124;
    private const string SherpaVocoderArchive = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-zh-en-emilia.tar.bz2";
    private const string SherpaVocoderMember = "sherpa-onnx-zipvoice-distill-zh-en-emilia/vocos_24khz.onnx";

    /// <summary>Records the fallbacks one test provokes (only the ones for its own URLs, so tests running side by side cannot see each other's).</summary>
    private sealed class FallbackLog : IDisposable
    {
        private readonly Func<string, bool> _mine;
        public readonly List<(string Failed, string Tried, string Why)> Events = new();
        public FallbackLog(Func<string, bool> mine) { _mine = mine; ModelsHost.Fallback += OnFallback; }
        private void OnFallback(string failed, string tried, string why) { if (_mine(failed)) lock (Events) Events.Add((failed, tried, why)); }
        public void Dispose() => ModelsHost.Fallback -= OnFallback;
    }

    /// <summary>Every URL shape is built the way the host stores it, and the fallback maps by swapping the host. Pure: no network.</summary>
    [TestMethod(Timeout = 30000)]
    public async Task ModelsHost_BuildsEveryUrlShapeAndMapsToItsHuggingFaceTwin() => await RunTest(async accelerator =>
    {
        var b = ModelsHost.BaseUrl.TrimEnd('/');
        Ur_Expect(ModelsHost.GetUrl("Xenova/distilgpt2", "onnx/decoder_model.onnx") == $"{b}/Xenova/distilgpt2/resolve/main/onnx/decoder_model.onnx", "repo file url");
        Ur_Expect(HuggingFaceClient.GetDownloadUrl("Xenova/distilgpt2", "tokenizer.json") == $"{b}/Xenova/distilgpt2/resolve/main/tokenizer.json", "GetDownloadUrl builds the models-host url");
        Ur_Expect(HuggingFaceClient.GetDownloadUrl("Xenova/distilgpt2", "tokenizer.json", "abc123") == "https://huggingface.co/Xenova/distilgpt2/resolve/abc123/tokenizer.json", "only main is mirrored: another revision is Hugging Face's");
        Ur_Expect(ModelsHost.GetOllamaUrl("gemma4", "12b", "model") == $"{b}/ollama/gemma4/12b/gemma4-12b-model.gguf", "an Ollama layer is stored under the name the hub cached it by");
        Ur_Expect(ModelsHost.GetArchiveMemberUrl(SherpaVocoderArchive, SherpaVocoderMember)
            == $"{b}/github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-zh-en-emilia/vocos_24khz.onnx", "an archive member is a plain file under the archive's url minus the extension");

        Ur_Expect(ModelsHost.GetFallbackUrl($"{b}/Xenova/distilgpt2/resolve/main/tokenizer.json") == "https://huggingface.co/Xenova/distilgpt2/resolve/main/tokenizer.json", "repo file -> Hugging Face twin");
        Ur_Expect(ModelsHost.GetFallbackUrl($"{b}/datasets/LostBeard/spawnscene-samples/resolve/main/a/001.jpg") == "https://huggingface.co/datasets/LostBeard/spawnscene-samples/resolve/main/a/001.jpg", "dataset -> Hugging Face twin");
        Ur_Expect(ModelsHost.GetFallbackUrl($"{b}/Xenova/distilgpt2/resolve/main/tokenizer.json?_dl=1") == "https://huggingface.co/Xenova/distilgpt2/resolve/main/tokenizer.json?_dl=1", "the query string travels with it");
        Ur_Expect(ModelsHost.GetFallbackUrl($"{b}/ollama/gemma4/12b/gemma4-12b-model.gguf") == null, "an Ollama layer has no Hugging Face twin");
        Ur_Expect(ModelsHost.GetFallbackUrl(ModelsHost.GetArchiveMemberUrl(SherpaVocoderArchive, SherpaVocoderMember)) == null, "an archive member has no Hugging Face twin");
        Ur_Expect(ModelsHost.GetFallbackUrl("https://example.com/a/b/resolve/main/c") == null, "a foreign host is never rewritten");
        Ur_Expect(ModelsHost.GetFallbackUrl("https://huggingface.co/a/b/resolve/main/c") == null, "a Hugging Face url has nothing to fall back to");

        foreach (var s in new[] { HttpStatusCode.NotFound, HttpStatusCode.Forbidden, HttpStatusCode.TooManyRequests, HttpStatusCode.RequestTimeout, HttpStatusCode.BadGateway, HttpStatusCode.ServiceUnavailable })
            Ur_Expect(ModelsHost.ShouldFallBack(s), $"{(int)s} falls back");
        foreach (var s in new[] { HttpStatusCode.OK, HttpStatusCode.PartialContent, HttpStatusCode.NotModified, HttpStatusCode.RequestedRangeNotSatisfiable })
            Ur_Expect(!ModelsHost.ShouldFallBack(s), $"{(int)s} does not fall back");

        bool threw = false;
        try { ModelsHost.GetArchiveMemberUrl("https://example.com/model.onnx", "a/b"); } catch (ArgumentException) { threw = true; }
        Ur_Expect(threw, "a url that is not an archive is refused");
        threw = false;
        try { ModelsHost.GetArchiveMemberUrl(SherpaVocoderArchive, "no-top-folder.onnx"); } catch (ArgumentException) { threw = true; }
        Ur_Expect(threw, "a member without its top folder is refused");
        await Task.CompletedTask;
    });

    /// <summary>
    /// What the builders produce is what the host SERVES, for every shape (repo file, dataset file, Ollama layer, archive member): no fallback happens,
    /// because a fallback here would hide a file missing from the host - the exact mistake the fallback makes invisible.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task ModelsHost_ServesEveryUrlShapeItBuilds() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var urls = new (string What, string Url)[]
        {
            ("repo file", HuggingFaceClient.GetDownloadUrl("Xenova/distilgpt2", "tokenizer.json")),
            ("dataset file", ModelsHost.GetUrl("datasets/LostBeard/spawnscene-samples", "pinecone/001.jpg")),
            ("ollama layer", ModelsHost.GetOllamaUrl("gemma3", "4b", "license")),
            ("archive member", ModelsHost.GetArchiveMemberUrl(SherpaVocoderArchive, SherpaVocoderMember)),
        };
        using var log = new FallbackLog(u => urls.Any(x => x.Url == u));
        foreach (var (what, url) in urls)
        {
            long size;
            try { size = await ModelsHost.ProbeSizeAsync(http, url); }
            catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
            { throw new UnsupportedTestException($"models host unreachable ({ex.GetType().Name}) - needs internet"); }
            Ur_Expect(size > 0, $"{what}: the host serves {url} (size {size})");
        }
        Ur_Expect(log.Events.Count == 0, $"no request fell back to Hugging Face: {string.Join("; ", log.Events.Select(e => e.Failed + " (" + e.Why + ")"))}");
    });

    /// <summary>
    /// A file the host lacks (404) is fetched from Hugging Face's twin URL by itself, once, and a file nobody has is still a 404. Real network on both sides.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task ModelsHost_FallsBackToHuggingFaceForAFileTheHostLacks() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var url = HuggingFaceClient.GetDownloadUrl(UnmirroredRepo, UnmirroredFile);
        var nowhere = HuggingFaceClient.GetDownloadUrl(UnmirroredRepo, "definitely-not-a-real-file.bin");
        using var log = new FallbackLog(u => u == url || u == nowhere);

        HttpResponseMessage direct;
        try { direct = await http.GetAsync(url); }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        { throw new UnsupportedTestException($"models host unreachable ({ex.GetType().Name}) - needs internet"); }
        using (direct)
            Ur_Expect(direct.StatusCode == HttpStatusCode.NotFound, $"the fixture must NOT be mirrored on the models host (got {(int)direct.StatusCode}); pick another unmirrored file");

        using (var req = new HttpRequestMessage(HttpMethod.Get, url))
        using (var res = await ModelsHost.SendAsync(http, req))
        {
            Ur_Expect(res.StatusCode == HttpStatusCode.OK, $"the fallback answered {(int)res.StatusCode}");
            Ur_Expect((await res.Content.ReadAsByteArrayAsync()).Length == UnmirroredBytes, "the bytes are the file's");
        }
        Ur_Expect(log.Events.Count == 1 && log.Events[0].Failed == url && log.Events[0].Tried == $"https://huggingface.co/{UnmirroredRepo}/resolve/main/{UnmirroredFile}",
            $"exactly one fallback, to the Hugging Face twin: {string.Join("; ", log.Events.Select(e => e.Tried))}");

        var viaClient = await new HuggingFaceClient(http).DownloadFileAsync(UnmirroredRepo, UnmirroredFile);
        Ur_Expect(viaClient.Length == UnmirroredBytes, $"HuggingFaceClient.DownloadFileAsync falls back too ({viaClient.Length} bytes)");

        using (var req = new HttpRequestMessage(HttpMethod.Get, nowhere))
        using (var res = await ModelsHost.SendAsync(http, req))
            Ur_Expect(res.StatusCode == HttpStatusCode.NotFound, $"a file nobody has is still a 404, not a hang or a 200 (got {(int)res.StatusCode})");
    });

    /// <summary>
    /// 🔴 A Hugging Face token belongs to Hugging Face: the request to the models host must not carry it, the fallback request to huggingface.co must.
    /// A spy handler sits on the REAL network path and records what was sent (desktop: the browser's fetch cannot be spied on this way).
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task ModelsHost_TheHuggingFaceTokenNeverReachesTheModelsHost() => await RunTest(async accelerator =>
    {
        if (OperatingSystem.IsBrowser()) throw new UnsupportedTestException("a request spy needs HttpClientHandler, which the browser does not have");
        var spy = new RequestSpy(new HttpClientHandler());
        using var http = new HttpClient(spy) { Timeout = TimeSpan.FromMinutes(1) };
        try { await new HuggingFaceClient(http, "hf_test_token_not_a_real_one").DownloadFileAsync(UnmirroredRepo, UnmirroredFile); }
        catch (HttpRequestException) { /* Hugging Face rejects a fake token with 401: only what was SENT matters here */ }

        var toModels = spy.Requests.Where(r => r.Host.Contains("models.spawndev.com")).ToList();
        var toHf = spy.Requests.Where(r => r.Host.EndsWith("huggingface.co")).ToList();
        Ur_Expect(toModels.Count >= 1, "the models host was tried first");
        Ur_Expect(toModels.All(r => !r.HasAuthorization), "no request to the models host carries the token");
        Ur_Expect(toHf.Count >= 1 && toHf.All(r => r.HasAuthorization), $"the fallback request to Hugging Face carries it ({toHf.Count} request(s))");
        await Task.CompletedTask;
    });

    /// <summary>
    /// 🔴 A relative request URI (the Blazor pattern: an HttpClient with a BaseAddress and "models/x.onnx" paths, which the whole CreateFromFile test family uses) is resolved
    /// against BaseAddress, not rejected: a relative URI on a foreign host goes through untouched, one that resolves onto the models host still falls back to its
    /// Hugging Face twin. Offline and deterministic: a stub handler answers instead of the network (it is not an HttpClientHandler, so it runs in the browser too).
    /// This crashed with "This operation is not supported for a relative URI" in ModelsHost.SendAsync and took 22 tests down on every lane.
    /// </summary>
    [TestMethod(Timeout = 30000)]
    public async Task ModelsHost_RelativeRequestUriIsResolvedAgainstTheBaseAddress() => await RunTest(async accelerator =>
    {
        var seen = new List<string>();
        var stub = new StubHandler(r =>
        {
            seen.Add(r.RequestUri!.AbsoluteUri);
            return new HttpResponseMessage(r.RequestUri.Host.Contains("models") ? HttpStatusCode.NotFound : HttpStatusCode.OK) { Content = new ByteArrayContent(new byte[] { 1, 2, 3 }) };
        });

        // 1. foreign host: untouched, one request, answered as is
        using (var http = new HttpClient(stub) { BaseAddress = new Uri("https://app.example.test/") })
        using (var req = new HttpRequestMessage(HttpMethod.Get, "models/squeezenet/model.onnx"))
        using (var res = await ModelsHost.SendAsync(http, req))
        {
            Ur_Expect(res.StatusCode == HttpStatusCode.OK, $"a relative url on a foreign host goes through ({(int)res.StatusCode})");
            Ur_Expect(seen.Count == 1 && seen[0] == "https://app.example.test/models/squeezenet/model.onnx", $"one request, resolved against the base address: {string.Join(", ", seen)}");
        }

        // 2. resolves onto the models host: 404 there, falls back to the Hugging Face twin
        seen.Clear();
        var fallbacks = new List<(string Failed, string Tried)>();
        void OnFallback(string failed, string tried, string why) { lock (fallbacks) fallbacks.Add((failed, tried)); }
        var modelsBase = new Uri(ModelsHost.BaseUrl.TrimEnd('/') + "/");
        ModelsHost.Fallback += OnFallback;
        try
        {
            using var http = new HttpClient(stub) { BaseAddress = modelsBase };
            using var req = new HttpRequestMessage(HttpMethod.Get, "Xenova/distilgpt2/resolve/main/generation_config.json");
            using var res = await ModelsHost.SendAsync(http, req);
            Ur_Expect(res.StatusCode == HttpStatusCode.OK, $"the fallback answered ({(int)res.StatusCode})");
            Ur_Expect(seen.Count == 2 && seen[1] == "https://huggingface.co/Xenova/distilgpt2/resolve/main/generation_config.json", $"models host first, then the Hugging Face twin: {string.Join(", ", seen)}");
            Ur_Expect(fallbacks.Count == 1 && fallbacks[0].Failed == modelsBase + "Xenova/distilgpt2/resolve/main/generation_config.json", $"the fallback event carries the resolved url: {string.Join("; ", fallbacks.Select(f => f.Failed))}");
        }
        finally { ModelsHost.Fallback -= OnFallback; }
        await Task.CompletedTask;
    });

    private sealed class StubHandler(Func<HttpRequestMessage, HttpResponseMessage> answer) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) => Task.FromResult(answer(request));
    }

    private sealed class RequestSpy(HttpMessageHandler inner) : DelegatingHandler(inner)
    {
        public readonly List<(string Host, bool HasAuthorization)> Requests = new();
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            lock (Requests) Requests.Add((request.RequestUri!.Host, request.Headers.Authorization != null));
            return base.SendAsync(request, cancellationToken);
        }
    }

    /// <summary>The browser downloader (fetch into OPFS) falls back at the START of a download, and the stream reader falls back per range. Browser only.</summary>
    [TestMethod(Timeout = 120000)]
    public async Task ModelsHost_TheBrowserTransportsFallBackToHuggingFaceToo() => await RunTest(async accelerator =>
    {
        var js = RequireBrowserRuntime();
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var url = HuggingFaceClient.GetDownloadUrl(UnmirroredRepo, UnmirroredFile);
        using var log = new FallbackLog(u => u == url);

        // 1. the OPFS downloader (HttpModelDownloader): the first request 404s on the models host, the twin is fetched and cached.
        using var cache = new OpfsModelCache(js) { CacheDirectoryName = "ilgpu-ml-test-modelshost" };
        if (!await cache.IsAvailableAsync()) throw new UnsupportedTestException("OPFS unavailable in this context");
        using var source = new HubModelSource(cache, http);
        const string key = "modelshost-fallback-fixture";
        await cache.RemoveAsync(key);
        try
        {
            await using (var stream = await source.OpenUrlAsync(url, key, UnmirroredBytes))
                Ur_Expect(stream.Length == UnmirroredBytes, $"the downloader fetched the twin: {stream.Length} bytes");
            Ur_Expect(log.Events.Count == 1 && log.Events[0].Tried.StartsWith("https://huggingface.co/"), $"the downloader fell back once: {log.Events.Count}");
        }
        finally { await cache.RemoveAsync(key); }

        // 2. FetchRangeStream (the streaming path): a range the models host cannot give is read from Hugging Face for the rest of the stream.
        log.Events.Clear();
        await using (var ranged = new FetchRangeStream(url, UnmirroredBytes))
        {
            using var first = await ranged.ReadUint8ArrayAsync(50);
            Ur_Expect(first.Length == 50, $"the first range arrived ({first.Length} bytes)");
            using var rest = await ranged.ReadUint8ArrayAsync(100);
            Ur_Expect(rest.Length == UnmirroredBytes - 50, $"the rest arrived too ({rest.Length} bytes)");
        }
        Ur_Expect(log.Events.Count == 1, $"the range stream fell back exactly once ({log.Events.Count})");
    });
}
