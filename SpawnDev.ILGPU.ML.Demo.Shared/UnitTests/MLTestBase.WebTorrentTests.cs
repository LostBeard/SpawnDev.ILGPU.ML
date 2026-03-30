using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.WebTorrent;
using SpawnDev.WebTorrent.ModelDelivery;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// WebTorrent HuggingFace model delivery tests.
/// These run FIRST — verify we can download models through WebTorrent
/// before any inference tests that depend on model files.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task WebTorrent_PackageLoads() => await RunTest(async accelerator =>
    {
        // Verify the WebTorrent types are accessible
        var clientType = typeof(WebTorrentClient);
        var torrentType = typeof(ModelTorrentClient);
        Console.WriteLine($"[WebTorrent] Client type: {clientType.FullName}");
        Console.WriteLine($"[WebTorrent] Torrent type: {torrentType.FullName}");
        Console.WriteLine("[WebTorrent] Package loads: PASS");
    });

    [TestMethod(Timeout = 60000)]
    public async Task WebTorrent_DownloadSmallModel() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Download a small model file through ModelTorrentClient
        // Uses HF CDN fallback since no torrent server in test environment
        var client = new ModelTorrentClient();
        try
        {
            var data = await client.DownloadModelAsync(
                "Xenova/distilgpt2", "tokenizer.json");
            Console.WriteLine($"[WebTorrent] Downloaded tokenizer.json: {data.Length} bytes");
            if (data.Length < 100)
                throw new Exception($"Download too small: {data.Length} bytes");
            // Verify it's valid JSON
            var text = System.Text.Encoding.UTF8.GetString(data);
            if (!text.Contains("model"))
                throw new Exception("Downloaded data doesn't look like tokenizer.json");
            Console.WriteLine("[WebTorrent] Download small model: PASS");
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network"))
        {
            throw new UnsupportedTestException($"No network: {ex.Message}");
        }
        finally
        {
            await client.DisposeAsync();
        }
    });

    [TestMethod(Timeout = 120000)]
    public async Task WebTorrent_DownloadOnnxModel() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Download an actual ONNX model file through WebTorrent
        var client = new ModelTorrentClient();
        try
        {
            var data = await client.DownloadModelAsync(
                "Xenova/distilgpt2", "onnx/decoder_model.onnx");
            Console.WriteLine($"[WebTorrent] Downloaded decoder_model.onnx: {data.Length / 1024 / 1024}MB");
            if (data.Length < 1_000_000)
                throw new Exception($"Model too small: {data.Length} bytes — expected ~330MB");

            // Verify it's a valid ONNX file (magic bytes)
            if (data[0] != 0x08) // protobuf field 1 varint
                Console.WriteLine("[WebTorrent] WARNING: unexpected first byte, might not be ONNX");

            // Actually load it to prove the download is usable
            using var session = InferenceSession.CreateFromOnnx(accelerator, data,
                inputShapes: new Dictionary<string, int[]> { ["input_ids"] = new[] { 1, 5 } },
                enableOptimization: false);
            Console.WriteLine($"[WebTorrent] Model loaded: {session.InputNames.Length} inputs, {session.OutputNames.Length} outputs");
            Console.WriteLine("[WebTorrent] Download ONNX model: PASS");
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network"))
        {
            throw new UnsupportedTestException($"No network: {ex.Message}");
        }
        finally
        {
            await client.DisposeAsync();
        }
    });
}
