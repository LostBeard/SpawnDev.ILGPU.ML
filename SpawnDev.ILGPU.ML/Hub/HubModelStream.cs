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

    /// <summary>A model opened from the hub: the live torrent, its file entry, and a seekable read stream.</summary>
    public sealed record HubModel(Torrent Torrent, TorrentFileInfo File, Stream Stream);

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
        var magnet = await GetMagnetAsync(repoId, filePath, ct).ConfigureAwait(false);

        using var metaCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        metaCts.CancelAfter(MetadataTimeout);
        var opts = deselect ? new AddTorrentOptions { Deselect = true } : null;
        var torrent = await _client.AddAsync(magnet, opts, metaCts.Token).ConfigureAwait(false);

        if (torrent.Files == null || torrent.Files.Length == 0)
            throw new InvalidOperationException(
                $"Torrent for '{repoId}/{filePath}' resolved metadata but exposes no files.");

        var file = torrent.Files[0];
        return new HubModel(torrent, file, file.CreateReadStream());
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
        return _client.RemoveAsync(model.Torrent);
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
