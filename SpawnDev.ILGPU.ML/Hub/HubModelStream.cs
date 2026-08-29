using System.Net.Http.Json;
using SpawnDev.ILGPU.ML.Onnx;
using SpawnDev.WebTorrent;

namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// Opens a <b>seekable</b> <see cref="Stream"/> over a HuggingFace model file served by a SpawnDev
/// WebTorrent hub (the public <c>hub.spawndev.com</c> or your own <c>SpawnDev.WebTorrent.Server.HuggingFace</c>
/// deployment) — <b>without downloading the whole model</b>.
/// <para>
/// Flow: ask the hub for a magnet (<c>GET /magnet/{repoId}/{filePath}</c>, which blocks server-side
/// only until the file is fetched + torrented once), add it to the <see cref="WebTorrentClient"/>,
/// await metadata — resolved <b>peer-free</b> from the magnet's HTTP exact-source (<c>xs=</c>)
/// <c>.torrent</c> — and hand back a <see cref="Torrent"/> piece stream whose pieces download on demand.
/// </para>
/// <para>
/// Because <see cref="ModelInspectorHelper.InspectAsync(Stream, System.Threading.CancellationToken)"/>
/// seeks past every weight blob when the stream <c>CanSeek</c> (and the torrent stream can), inspecting
/// a multi-GB checkpoint downloads only the few hundred KB of graph structure — never the weights.
/// </para>
/// <code>
/// // DI: WebTorrentClient (singleton) + HttpClient are already registered.
/// var hub = new HubModelStream(webTorrentClient, httpClient);
/// var (inspection, compat) = await hub.InspectWithCompatibilityAsync(
///     "onnx-community/mobilenetv3_small_100.lamb_in1k", "onnx/model.onnx");
/// </code>
/// </summary>
public class HubModelStream
{
    /// <summary>The public SpawnDev hub running the HuggingFace proxy.</summary>
    public const string DefaultHubBaseUrl = "https://hub.spawndev.com:44365";

    private readonly WebTorrentClient _client;
    private readonly HttpClient _http;

    /// <summary>Base URL of the hub. Defaults to <see cref="DefaultHubBaseUrl"/>; point at your own deployment to override.</summary>
    public string HubBaseUrl { get; set; } = DefaultHubBaseUrl;

    /// <summary>
    /// Max time to wait on the <c>/magnet</c> request. A cold-cache first request makes the hub
    /// download + SHA-256 hash the entire file from HuggingFace before responding, so this is generous.
    /// </summary>
    public TimeSpan PrepareTimeout { get; set; } = TimeSpan.FromMinutes(5);

    /// <summary>Max time to wait for torrent metadata after adding the magnet (xs= fetch is normally sub-second).</summary>
    public TimeSpan MetadataTimeout { get; set; } = TimeSpan.FromSeconds(60);

    public HubModelStream(WebTorrentClient client, HttpClient http)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
        _http = http ?? throw new ArgumentNullException(nameof(http));
    }

    /// <summary>JSON shape returned by the hub's <c>/magnet</c> endpoint.</summary>
    public sealed record HubMagnetResult(string MagnetUri, string RepoId, string FilePath, string WebSeed);

    /// <summary>A model opened from the hub: a seekable read stream, plus the live torrent + file entry when it
    /// is served over P2P. For a COLD load — the hub is still preparing the torrent — the model streams raw over
    /// the always-serving /hf web seed and <see cref="Torrent"/> + <see cref="File"/> are null; use
    /// <see cref="Length"/> (which falls back to the stream) and <see cref="Stream"/>.</summary>
    public sealed record HubModel(Torrent? Torrent, TorrentFileInfo? File, Stream Stream)
    {
        /// <summary>Total file length — from the torrent file entry (P2P) or the stream (raw-HTTP cold load).</summary>
        public long Length => File?.Length ?? Stream.Length;
    }

    /// <summary>
    /// Ask the hub for the magnet URI for a model file. Blocks server-side until the file is fetched
    /// from HuggingFace and a <c>.torrent</c> is built (first request on a cold cache only).
    /// </summary>
    /// <param name="repoId">HuggingFace repo id, e.g. <c>onnx-community/mobilenetv3_small_100.lamb_in1k</c>.</param>
    /// <param name="filePath">File path within the repo, e.g. <c>onnx/model.onnx</c>.</param>
    public async Task<string> GetMagnetAsync(string repoId, string filePath, CancellationToken ct = default)
    {
        var url = $"{HubBaseUrl.TrimEnd('/')}/magnet/{repoId.Trim('/')}/{filePath.TrimStart('/')}";
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        cts.CancelAfter(PrepareTimeout);

        // The hub builds a .torrent on first request for a file (download from HF + SHA-256 hash). For a
        // large file (e.g. SD-Turbo's 1.7GB UNet) that server-side prep exceeds the hub's reverse-proxy
        // gateway timeout (~25s observed), so the first requests return 504 while prep continues in the
        // background. Retry transient gateway errors (502/503/504) with backoff until prep completes (200)
        // or PrepareTimeout elapses. A 404 (or any other 4xx) is a real "not found" and is NOT retried.
        var delay = TimeSpan.FromSeconds(2);
        try
        {
            while (true)
            {
                using var resp = await _http.GetAsync(url, cts.Token).ConfigureAwait(false);
                if (resp.IsSuccessStatusCode)
                {
                    var result = await resp.Content.ReadFromJsonAsync<HubMagnetResult>(cts.Token).ConfigureAwait(false);
                    if (result == null || string.IsNullOrEmpty(result.MagnetUri))
                        throw new InvalidOperationException($"Hub returned no magnet for '{repoId}/{filePath}' ({url}).");
                    return result.MagnetUri;
                }
                int status = (int)resp.StatusCode;
                if (status != 502 && status != 503 && status != 504)
                    throw new HttpRequestException($"Hub /magnet for '{repoId}/{filePath}' failed: HTTP {status} ({url}).");
                // Still preparing server-side — wait and retry (bounded by PrepareTimeout via cts).
                await Task.Delay(delay, cts.Token).ConfigureAwait(false);
                if (delay < TimeSpan.FromSeconds(15)) delay += TimeSpan.FromSeconds(2);
            }
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            throw new TimeoutException($"Hub /magnet did not finish preparing '{repoId}/{filePath}' within {PrepareTimeout.TotalSeconds:F0}s ({url}).");
        }
    }

    /// <summary>
    /// Open a seekable read stream over a hub-served model. The returned <see cref="HubModel.Stream"/>
    /// downloads torrent pieces on demand; dispose it when done. The torrent stays in the client
    /// (so a subsequent open / load reuses already-downloaded pieces).
    /// </summary>
    /// <param name="deselect">
    /// When <c>true</c>, the torrent is added DESELECTED: no piece is selected up front, so only the
    /// pieces a read actually touches are fetched (SpawnDev.WebTorrent 3.2.5 Fix B). This is what makes
    /// inspecting a multi-GB checkpoint pull only graph structure, never weights. When <c>false</c>
    /// (default) the whole file is selected so a plain read / model load downloads it in the background.
    /// </param>
    public async Task<HubModel> OpenAsync(string repoId, string filePath, bool deselect = false, CancellationToken ct = default)
    {
        // Add a LAZY-HASH torrent from the hub /hf web seed (SpawnDev.WebTorrent 3.2.11+). The model becomes a
        // PERSISTENT torrent from the first byte: it downloads on demand from the web seed, computes its infohash
        // as it goes, caches pieces to OPFS under a stable URL-derived key, and RESTORES on reload with ZERO
        // re-download — and it shows on the Model Cache page. This replaces the old non-persistent HttpRangeStream
        // fallback, which made NO torrent when the hub wasn't "ready", so every page refresh re-downloaded the whole
        // model and it never appeared on /cache. The /hf web seed serves immediately (the hub caches missing chunks
        // on demand), so the load still starts without waiting for the hub's first full server-side download.
        // (Phase 2 will adopt the hub's existing .torrent when ready to also seed P2P; the re-download fix is here.)
        var hfUrl = $"{HubBaseUrl.TrimEnd('/')}/hf/{repoId.Trim('/')}/{filePath.TrimStart('/')}";
        return await OpenTorrentAsync(hfUrl, repoId, filePath, deselect, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Open a SEEKABLE stream over the hub's <c>/hf</c> web seed using plain HTTP range requests - no
    /// WebTorrent, no magnet, no swarm.
    /// </summary>
    /// <remarks>
    /// <see cref="OpenAsync"/> is the right default: it makes the model a persistent torrent that caches to
    /// OPFS and restores across reloads. This is the SIMPLE path, and it exists because the two have very
    /// different failure surfaces. <see cref="OpenAsync"/> depends on torrent metadata, lazy-hash and peer
    /// discovery; this depends on one thing - our hub answering a range request. When a structure-only
    /// inspection is the whole job and the file is read once, that smaller surface is worth having, and it is
    /// what makes a HUB test a test of the HUB rather than of the swarm.
    ///
    /// The stream reports <see cref="HttpRangeStream.BytesFetched"/>, so a caller can verify it really did
    /// skip the weights.
    /// </remarks>
    public async Task<HttpRangeStream> OpenWebSeedAsync(string repoId, string filePath, CancellationToken ct = default)
    {
        var url = $"{HubBaseUrl.TrimEnd('/')}/hf/{repoId.Trim('/')}/{filePath.TrimStart('/')}";
        long size = await ProbeSizeAsync(url, ct).ConfigureAwait(false);
        return new HttpRangeStream(_http, url, size);
    }

    /// <summary>Fetch a SMALL hub file fully into a byte[] via a plain HTTP GET of the /hf web seed. Use for a
    /// KB-scale structure file (e.g. an external-data model's <c>model.onnx</c>, which holds only the graph — the
    /// weights live in <c>model.onnx_data</c>). Streaming that big weights file stays on the zero-copy torrent
    /// path; fetching the tiny structure over HTTP also sidesteps a WebTorrent lazy-hash collision when two
    /// web-seed files share a directory (<c>onnx/model.onnx</c> + <c>onnx/model.onnx_data</c>). Do NOT use for
    /// large single-file weights — that belongs on <see cref="OpenAsync"/>.</summary>
    public async Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken ct = default)
    {
        var url = $"{HubBaseUrl.TrimEnd('/')}/hf/{repoId.Trim('/')}/{filePath.TrimStart('/')}";
        using var resp = await _http.GetAsync(url, ct).ConfigureAwait(false);
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadAsByteArrayAsync(ct).ConfigureAwait(false);
    }

    /// <summary>Add a torrent (magnet URI OR an http(s) web-seed URL — the latter is a Lazy-Hash add) and open a
    /// seekable read stream over its (single) file. The torrent is persistent: pieces cache to OPFS and restore on
    /// reload, so a subsequent open reuses them with no re-download.</summary>
    private async Task<HubModel> OpenTorrentAsync(string magnetOrUrl, string repoId, string filePath, bool deselect, CancellationToken ct)
    {
        using var metaCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        metaCts.CancelAfter(MetadataTimeout);
        var opts = deselect ? new AddTorrentOptions { Deselect = true } : null;
        var torrent = await _client.AddAsync(magnetOrUrl, opts, metaCts.Token).ConfigureAwait(false);
        if (torrent.Files == null || torrent.Files.Length == 0)
            throw new InvalidOperationException($"Torrent for '{repoId}/{filePath}' resolved metadata but exposes no files.");
        var file = torrent.Files[0];
        return new HubModel(torrent, file, file.CreateReadStream());
    }

    /// <summary>Non-blocking hub status (one GET to /model | /ollama-model): "ready" (with a magnet) or
    /// "preparing". Best-effort — any failure returns null so the caller falls back to the cold web-seed stream.</summary>
    private async Task<HubModelResult?> GetModelStatusAsync(string url, CancellationToken ct)
    {
        try
        {
            using var resp = await _http.GetAsync(url, ct).ConfigureAwait(false);
            if (!resp.IsSuccessStatusCode) return null;
            return await resp.Content.ReadFromJsonAsync<HubModelResult>(ct).ConfigureAwait(false);
        }
        catch { return null; }
    }

    /// <summary>Total size of a hub web-seed resource via a 0-0 range probe (Content-Range '…/TOTAL').</summary>
    private async Task<long> ProbeSizeAsync(string url, CancellationToken ct)
    {
        // The web seed streams WHILE the hub is still caching the model (the "preparing" state) — proven: a range
        // request returns the real Content-Range total + serves bytes during preparing. But in the FIRST instants
        // of a cold model the hub hasn't resolved the total yet, so a 0-0 probe can transiently come back with
        // status 502/503 or a 206 whose Content-Range total isn't set (leaving only the 1-byte partial
        // Content-Length). That window closes within seconds (one origin HEAD), so RETRY through it rather than
        // failing — preserving the download-while-caching design (no waiting for the hub to finish acquiring).
        var deadline = TimeSpan.FromSeconds(45);
        var delay = TimeSpan.FromMilliseconds(500);
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        cts.CancelAfter(deadline);
        long lastSize = -1; int lastStatus = 0; object? lastCr = null, lastCl = null;
        try
        {
            while (true)
            {
                using var req = new HttpRequestMessage(HttpMethod.Get, url);
                req.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(0, 0);
                using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, cts.Token).ConfigureAwait(false);
                lastStatus = (int)resp.StatusCode;
                lastCr = resp.Content.Headers.ContentRange; lastCl = resp.Content.Headers.ContentLength;
                if (resp.IsSuccessStatusCode)
                {
                    // Trust ONLY the Content-Range TOTAL (`bytes 0-0/TOTAL`), or a full-200 Content-Length. A 206's
                    // Content-Length is the partial (1 byte) — never the file size.
                    long size = resp.Content.Headers.ContentRange?.Length
                                ?? (resp.StatusCode == System.Net.HttpStatusCode.OK ? resp.Content.Headers.ContentLength ?? -1 : -1);
                    if (size > 1) return size;
                    lastSize = size;
                }
                else if (lastStatus is not (502 or 503 or 504))
                    resp.EnsureSuccessStatusCode(); // a real error (404/403/…) — throw now, don't spin
                await Task.Delay(delay, cts.Token).ConfigureAwait(false);
                if (delay < TimeSpan.FromSeconds(3)) delay += TimeSpan.FromMilliseconds(500);
            }
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            throw new TimeoutException(
                $"Hub web seed did not report a usable file size for {url} within {deadline.TotalSeconds:F0}s " +
                $"(last status {lastStatus}, content-range '{lastCr}', content-length {lastCl}, size {lastSize}). " +
                "In a browser, also confirm the /hf endpoint exposes Content-Range via CORS.");
        }
    }

    /// <summary>JSON shape of the hub's non-blocking /model | /ollama-model status response.</summary>
    private sealed record HubModelResult(string? Status, string? MagnetUri, string? WebSeed);

    /// <summary>Ask the hub for the magnet URI of an OLLAMA model layer. The hub's OllamaProxy resolves the
    /// ollama registry manifest, fetches the layer blob, and seeds it as a torrent — same retry-on-prepare
    /// semantics as <see cref="GetMagnetAsync"/>. <paramref name="layer"/> ∈ <c>model</c> | <c>projector</c>
    /// (the GGUF weights / the mmproj), also <c>params</c>|<c>template</c>|<c>license</c>.</summary>
    public async Task<string> GetOllamaMagnetAsync(string model, string tag, string layer, CancellationToken ct = default)
    {
        // Poll the NON-BLOCKING /ollama-model endpoint (not blocking /ollama-magnet). On a cold cache a large
        // layer (gemma4's ~6.9 GB) takes minutes to fetch+seed server-side; the blocking endpoint would hold
        // the connection past the hub's reverse-proxy gateway timeout, and that gateway 504 carries NO CORS
        // header — so a browser fetch reports it as a CORS error, not a retriable 504. The non-blocking
        // endpoint instead returns a fast 200 ("preparing" until ready) every poll, always CORS-clean.
        var url = $"{HubBaseUrl.TrimEnd('/')}/ollama-model/{model.Trim('/')}/{tag.Trim('/')}/{layer.Trim('/')}";
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        cts.CancelAfter(PrepareTimeout);
        var delay = TimeSpan.FromSeconds(2);
        try
        {
            while (true)
            {
                using var resp = await _http.GetAsync(url, cts.Token).ConfigureAwait(false);
                if (resp.IsSuccessStatusCode)
                {
                    var result = await resp.Content.ReadFromJsonAsync<HubOllamaPrep>(cts.Token).ConfigureAwait(false);
                    if (string.Equals(result?.Status, "ready", StringComparison.OrdinalIgnoreCase) && !string.IsNullOrEmpty(result!.MagnetUri))
                        return result.MagnetUri;
                    // "preparing" — the hub is still fetching/seeding the blob; wait and poll again.
                }
                else
                {
                    int status = (int)resp.StatusCode;
                    if (status != 502 && status != 503 && status != 504)
                        throw new HttpRequestException($"Hub /ollama-model for '{model}:{tag}/{layer}' failed: HTTP {status} ({url}).");
                }
                await Task.Delay(delay, cts.Token).ConfigureAwait(false);
                if (delay < TimeSpan.FromSeconds(15)) delay += TimeSpan.FromSeconds(2);
            }
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            throw new TimeoutException($"Hub did not finish preparing ollama '{model}:{tag}/{layer}' within {PrepareTimeout.TotalSeconds:F0}s ({url}). Large cold-cache layers can take a while; retry.");
        }
    }

    /// <summary>Shape of the hub's non-blocking <c>/ollama-model</c> response (status + magnet when ready).</summary>
    private sealed record HubOllamaPrep(string? Status, string? MagnetUri);

    /// <summary>Open a seekable read stream over an OLLAMA model layer served by the hub (twin of
    /// <see cref="OpenAsync"/>). The returned file can be re-streamed with <c>HubModel.File.CreateReadStream()</c>
    /// for callers needing two concurrent readers (e.g. weight upload + token gather).</summary>
    public async Task<HubModel> OpenOllamaAsync(string model, string tag, string layer, bool deselect = false, CancellationToken ct = default)
    {
        // Add a LAZY-HASH torrent from the /ollama web seed (SpawnDev.WebTorrent 3.2.11+): the layer becomes a
        // PERSISTENT torrent — caches to OPFS, restores on reload with ZERO re-download, shows on /cache — replacing
        // the old non-persistent HttpRangeStream fallback that re-downloaded every refresh. The web seed serves
        // immediately while the hub caches missing chunks. (Hub-.torrent / P2P adoption when ready is Phase 2.)
        var seedUrl = $"{HubBaseUrl.TrimEnd('/')}/ollama/{model.Trim('/')}/{tag.Trim('/')}/{layer.Trim('/')}";
        return await OpenTorrentAsync(seedUrl, model, $"{tag}/{layer}", deselect, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Remove a previously-<see cref="OpenAsync"/>'d model's torrent from the client. Call this once the
    /// model is fully loaded (after disposing its <see cref="HubModel.Stream"/>) so the per-file torrents
    /// do NOT accumulate: an open torrent keeps its own web-seed connection running, and in the browser
    /// every torrent's piece-receive + SHA verify + OPFS write share the single WASM thread — so N open
    /// torrents = N concurrent downloads that starve the model currently being uploaded to the GPU (a
    /// multi-file model like SD-Turbo otherwise crawls even though the hub serves fast). Removing each
    /// torrent after its model loads keeps exactly one active at a time.
    /// </summary>
    public Task RemoveAsync(HubModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        // A cold raw-HTTP model has no torrent in the client — nothing to remove.
        return model.Torrent != null ? _client.RemoveAsync(model.Torrent) : Task.CompletedTask;
    }

    /// <summary>
    /// Inspect a hub-served model's structure WITHOUT downloading its weights — the inspector seeks
    /// past every weight blob, so only graph-structure pieces are fetched over the wire.
    /// </summary>
    public async Task<InspectionResult> InspectAsync(string repoId, string filePath, CancellationToken ct = default)
    {
        // deselect: true — inspection seeks past every weight blob, so only structure pieces are fetched.
        var model = await OpenAsync(repoId, filePath, deselect: true, ct).ConfigureAwait(false);
        try
        {
            return await ModelInspectorHelper.InspectAsync(model.Stream, ct).ConfigureAwait(false);
        }
        finally
        {
            await model.Stream.DisposeAsync().ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Inspect structure AND check operator compatibility from a single seekable pass over the
    /// hub-served model — weights are skipped, structure is read once.
    /// </summary>
    public async Task<(InspectionResult Inspection, CompatibilityResult Compatibility)> InspectWithCompatibilityAsync(
        string repoId, string filePath, Operators.OperatorRegistry? registry = null, CancellationToken ct = default)
    {
        // deselect: true — single seekable pass reads only structure pieces, never weights.
        var model = await OpenAsync(repoId, filePath, deselect: true, ct).ConfigureAwait(false);
        try
        {
            return await ModelInspectorHelper.InspectWithCompatibilityAsync(model.Stream, registry, ct).ConfigureAwait(false);
        }
        finally
        {
            await model.Stream.DisposeAsync().ConfigureAwait(false);
        }
    }
}
