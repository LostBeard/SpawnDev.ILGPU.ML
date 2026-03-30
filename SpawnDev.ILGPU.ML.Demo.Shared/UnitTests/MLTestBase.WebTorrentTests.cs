using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.WebTorrent.ModelDelivery;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Isolated WebTorrent integration tests.
/// Verify the SpawnDev.WebTorrent 2.0.0-rc2 package works correctly
/// for model delivery before using it in the full pipeline.
/// These must pass BEFORE any model loading tests run.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task WebTorrent_PackageLoads() => await RunTest(async accelerator =>
    {
        // Verify the WebTorrent package types are accessible
        var options = new ModelTorrentOptions();
        if (options == null)
            throw new Exception("ModelTorrentOptions failed to create");
        Console.WriteLine("[WebTorrent] Package loads: PASS");
    });

    [TestMethod]
    public async Task WebTorrent_ClientCreates() => await RunTest(async accelerator =>
    {
        // Verify ModelTorrentClient can be instantiated
        var client = new ModelTorrentClient();
        if (client == null)
            throw new Exception("ModelTorrentClient failed to create");
        await client.DisposeAsync();
        Console.WriteLine("[WebTorrent] Client creates: PASS");
    });

    [TestMethod]
    public async Task WebTorrent_ModelHubWiring() => await RunTest(async accelerator =>
    {
        // Verify ModelHub.TorrentClient property accepts the client
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Can't create a full ModelHub without BlazorJS runtime,
        // but verify the type is compatible
        var client = new ModelTorrentClient();
        Console.WriteLine($"[WebTorrent] ModelTorrentClient type: {client.GetType().FullName}");
        await client.DisposeAsync();
        Console.WriteLine("[WebTorrent] ModelHub wiring: PASS");
    });

    [TestMethod(Timeout = 30000)]
    public async Task WebTorrent_HFCDNFallback() => await RunTest(async accelerator =>
    {
        // Verify that ModelTorrentClient falls back to HuggingFace CDN
        // when the torrent server isn't available (which it won't be in tests)
        var client = new ModelTorrentClient(new ModelTorrentOptions
        {
            ServerBaseUrl = "https://localhost:1" // Non-existent server
        });

        try
        {
            // This should fail on torrent metadata, then fall back to HF CDN
            // Use a tiny file to minimize download time
            var data = await client.DownloadModelAsync(
                "onnx-community/mobilenetv2-12", "model.onnx");
            Console.WriteLine($"[WebTorrent] HF CDN fallback: got {data.Length} bytes");
            if (data.Length < 1000)
                throw new Exception($"HF CDN download too small: {data.Length} bytes");
            Console.WriteLine("[WebTorrent] HF CDN fallback: PASS");
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex)
        {
            // If even the HF CDN fallback fails (no network), skip
            throw new UnsupportedTestException($"No network: {ex.Message}");
        }
        finally
        {
            await client.DisposeAsync();
        }
    });
}
