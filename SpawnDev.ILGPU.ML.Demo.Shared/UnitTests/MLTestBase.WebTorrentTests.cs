using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.WebTorrent;
using SpawnDev.UnitTesting;
using ILGPU.Runtime;

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

    // WEIGHT-LOAD MEASUREMENT (browser, real demo path): loads the 330MB text-gen model the way TextGenPage
    // does - HubModelStream over the DEMO's OPFS-backed WebTorrentClient -> CreateFromOnnxStreamAsync, which
    // streams each weight to the GPU. A/Bs the weight-UPLOAD path from the SAME OPFS-cached pieces (so download
    // is removed and only the upload differs): (A) zero-copy JS->GPU (Uint8Array -> IBrowserMemoryBuffer
    // .CopyFromJS, weights never enter the .NET heap) vs (B) the .NET byte[] chunked path. Confirms the
    // zero-copy path actually fired (ZeroCopyWeightBytes > 0) and times both.
    //
    // DIAGNOSTIC: this test THROWS its measurement as the result message on success - the PMT browser lane
    // surfaces only console error/warn counts, not Console.WriteLine, so throwing is the only way to read the
    // numbers. A "fail" with a [DLMEASURE] message is the expected, successful outcome. A real failure is the
    // "zero-copy did NOT fire" message.
    // HeavyModel + browser-only (zero-copy path needs OPFS + a browser GPU backend / crypto.subtle).
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task WebTorrent_Measure_DistilGpt2_Download() => await RunTest(async accelerator =>
    {
        var client = GetWebTorrentClient();
        if (client == null) throw new UnsupportedTestException("OPFS WebTorrentClient only wired in the browser demo lane");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Clear stale CROSS-RUN OPFS torrent state. The PMT browser profile is persistent, so a prior run's
        // persisted distilgpt2 pieces survive; re-accessing them via AddAsync throws OPFS NotReadableError
        // (a separate re-access bug, tracked). A clean cold run side-steps it; everything below is in-session.
        var fs = GetAsyncFS();
        if (fs != null && await fs.DirectoryExists("webtorrent")) await fs.Remove("webtorrent", true);

        var hub = new SpawnDev.ILGPU.ML.Hub.HubModelStream(client, http);
        const string repo = "Xenova/distilgpt2";
        const string file = "onnx/decoder_model.onnx";
        var inputShapes = new Dictionary<string, int[]> { ["input_ids"] = new[] { 1, 5 } };

        try
        {
            // ── Warm OPFS: fully download the model once so the A/B loads below read from cache (download
            //    removed -> the only difference between A and B is the weight-upload path). ──
            long total;
            var swDl = System.Diagnostics.Stopwatch.StartNew();
            {
                var warm = await hub.OpenAsync(repo, file, deselect: false);
                total = warm.File.Length;
                var buf = new byte[1 << 20];
                await using (warm.Stream)
                    while (await warm.Stream.ReadAsync(buf, 0, buf.Length) > 0) { }
            }
            swDl.Stop();

            // ── Load A: zero-copy JS->GPU weight upload (default). ──
            var swA = System.Diagnostics.Stopwatch.StartNew();
            long zcA;
            {
                var m = await hub.OpenAsync(repo, file, deselect: false);
                SpawnDev.ILGPU.ML.InferenceSession s;
                await using (m.Stream)
                    s = await SpawnDev.ILGPU.ML.InferenceSession.CreateFromOnnxStreamAsync(
                        accelerator, m.Stream, inputShapes: inputShapes, enableOptimization: false);
                using (s) zcA = s.ZeroCopyWeightBytes;
            }
            swA.Stop();

            // ── Load B: forced .NET byte[] upload path (same cached pieces). ──
            var swB = System.Diagnostics.Stopwatch.StartNew();
            long zcB;
            SpawnDev.ILGPU.ML.Tensors.BufferPool.DisableJsZeroCopyWeights = true;
            try
            {
                var m = await hub.OpenAsync(repo, file, deselect: false);
                SpawnDev.ILGPU.ML.InferenceSession s;
                await using (m.Stream)
                    s = await SpawnDev.ILGPU.ML.InferenceSession.CreateFromOnnxStreamAsync(
                        accelerator, m.Stream, inputShapes: inputShapes, enableOptimization: false);
                using (s) zcB = s.ZeroCopyWeightBytes;
            }
            finally { SpawnDev.ILGPU.ML.Tensors.BufferPool.DisableJsZeroCopyWeights = false; }
            swB.Stop();

            double tDl = swDl.Elapsed.TotalSeconds, tA = swA.Elapsed.TotalSeconds, tB = swB.Elapsed.TotalSeconds;
            int mb = (int)(total / 1024 / 1024);

            if (zcA == 0)
                throw new Exception($"[DLMEASURE] ZERO-COPY DID NOT FIRE: load A ZeroCopyWeightBytes=0 (fell back to .NET byte[]). " +
                    $"model={mb}MB warmDownload={tDl:F1}s loadA={tA:F1}s loadB(byte[])={tB:F1}s. Expected weights to upload JS->GPU.");

            // Success: report the measurement (thrown so PMT surfaces it).
            throw new Exception(
                $"[DLMEASURE] OK distilgpt2 decoder {mb}MB on {accelerator.AcceleratorType} | " +
                $"warmDownload={tDl:F1}s | loadA(zero-copy JS->GPU)={tA:F1}s zc={zcA / 1024 / 1024}MB | " +
                $"loadB(.NET byte[])={tB:F1}s zc={zcB} | upload-path delta(B-A)={tB - tA:F1}s | " +
                $"(raw-HTTP source ceiling ~37MB/s single-stream on the 1GB LAN)");
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network") || ex.Message.Contains("magnet"))
        {
            throw new UnsupportedTestException($"hub/network unavailable: {ex.Message}");
        }
    });

    // CORRECTNESS guard for the fp16/half zero-copy weight path (the SD-Turbo case): uploading raw fp16 bytes
    // straight into a GPU ILGPU.Half buffer via IBrowserMemoryBuffer.CopyFromJS must be byte-identical to the
    // .NET byte[] path (CopyFromCPU of the decoded Half[]). This proves the layout assumption - ILGPU.Half is
    // IEEE binary16, same 2-byte little-endian layout as the source - rather than assuming it. Fast, no model.
    // Gated to WebGPU/Wasm where CopyFromJS is immediate (WebGL defers the upload to the next dispatch, so an
    // immediate read-back wouldn't reflect it - the production weight path still works there via the kernel).
    [TestMethod]
    public async Task ZeroCopyHalf_CopyFromJS_MatchesByteArrayUpload() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.WebGPU or AcceleratorType.Wasm))
            throw new UnsupportedTestException("immediate Half read-back after CopyFromJS only on WebGPU/Wasm (WebGL defers upload to dispatch)");

        // fp16 values with varied bit patterns (sign, fraction, zero, fp16 max, small) to catch any layout bug.
        float[] vals = { 1.0f, -2.5f, 0.0f, 3.140625f, -0.0009765625f, 65504f, 0.5f, -1024f };
        int n = vals.Length; // even -> byte length n*2 is a multiple of 4
        var srcBytes = new byte[n * 2];
        var expected = new global::ILGPU.Half[n];
        for (int i = 0; i < n; i++)
        {
            var h = (System.Half)vals[i];
            var b = System.BitConverter.GetBytes(h); // IEEE binary16, little-endian
            srcBytes[i * 2] = b[0]; srcBytes[i * 2 + 1] = b[1];
            expected[i] = (global::ILGPU.Half)(float)h; // matches the byte[] path's decode
        }

        using var bufJs = accelerator.Allocate1D<global::ILGPU.Half>(n);
        if (bufJs.Buffer is not SpawnDev.ILGPU.IBrowserMemoryBuffer ibm)
            throw new UnsupportedTestException("buffer is not IBrowserMemoryBuffer (CopyFromJS unavailable)");

        // A: zero-copy JS -> GPU (the new fp16 path)
        using (var u8 = new SpawnDev.BlazorJS.JSObjects.Uint8Array(srcBytes))
            ibm.CopyFromJS(u8);
        var gotJs = await bufJs.CopyToHostAsync();

        // B: the .NET byte[] path equivalent (decoded Half[] via CopyFromCPU)
        using var bufCpu = accelerator.Allocate1D<global::ILGPU.Half>(n);
        bufCpu.View.CopyFromCPU(expected);
        var gotCpu = await bufCpu.CopyToHostAsync();

        for (int i = 0; i < n; i++)
        {
            if ((float)gotJs[i] != (float)expected[i])
                throw new Exception($"fp16 zero-copy mismatch at [{i}]: got {(float)gotJs[i]}, expected {(float)expected[i]} (src {vals[i]})");
            if ((float)gotJs[i] != (float)gotCpu[i])
                throw new Exception($"fp16 zero-copy != byte[] path at [{i}]: js {(float)gotJs[i]} vs cpu {(float)gotCpu[i]}");
        }
    });
}
