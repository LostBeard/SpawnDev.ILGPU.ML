using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.WebTorrent;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// SpawnDev.WebTorrent package integration tests.
/// Verifies the ML demo's WebTorrent dependency loads and HuggingFace CDN downloads still work.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task WebTorrent_PackageLoads() => await RunTest(async accelerator =>
    {
        var clientType = typeof(WebTorrentClient);
        var torrentType = typeof(Torrent);
        Console.WriteLine($"[WebTorrent] Client type: {clientType.FullName}");
        Console.WriteLine($"[WebTorrent] Torrent type: {torrentType.FullName}");
        Console.WriteLine("[WebTorrent] Package loads: PASS");
    });

    [TestMethod(Timeout = 60000)]
    public async Task WebTorrent_DownloadSmallModel() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var hf = new HuggingFaceClient(http);
        try
        {
            var data = await hf.DownloadFileAsync("Xenova/distilgpt2", "tokenizer.json");
            Console.WriteLine($"[WebTorrent] Downloaded tokenizer.json: {data.Length} bytes");
            if (data.Length < 100)
                throw new Exception($"Download too small: {data.Length} bytes");
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
    });

    // HeavyModel: downloads a ~330MB GPT-2 decoder ONNX from the HuggingFace CDN AND compiles it
    // on the GPU — minutes of download + compile, same big-model class as the DA3/GPT-2 reference
    // tests. Gated out of the fast loop; run with PMT_EXCLUDE_CATEGORIES= when exercising it.
    [TestMethod(Timeout = 120000, Category = "HeavyModel")]
    public async Task WebTorrent_DownloadOnnxModel() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var hf = new HuggingFaceClient(http);
        try
        {
            var data = await hf.DownloadFileAsync("Xenova/distilgpt2", "onnx/decoder_model.onnx");
            Console.WriteLine($"[WebTorrent] Downloaded decoder_model.onnx: {data.Length / 1024 / 1024}MB");
            if (data.Length < 1_000_000)
                throw new Exception($"Model too small: {data.Length} bytes — expected ~330MB");

            if (data[0] != 0x08)
                Console.WriteLine("[WebTorrent] WARNING: unexpected first byte, might not be ONNX");

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
    });
}
