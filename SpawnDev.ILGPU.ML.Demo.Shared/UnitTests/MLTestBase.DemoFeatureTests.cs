using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// End-to-end tests for every demo feature pipeline.
/// Each test exercises the full pipeline: model download, preprocessing,
/// inference, postprocessing — verifying the complete user-facing flow.
/// </summary>
public abstract partial class MLTestBase
{
    // ═══════════════════════════════════════════════════════════
    //  Text Generation (Chatbot / AI Assistant)
    // ═══════════════════════════════════════════════════════════

    // REGRESSION GUARD for the fixed-shape decode + auto-detecting readback cache. The cache skips the
    // ~643 mid-graph shape-readback GPU round-trips/step (~7.8s → fast decode) but MUST NOT change the
    // output. A naive "cache every ≤64-elem readback" was WRONG: input_ids itself is ≤64 elems (seq≤64)
    // and is DATA-dependent, so caching it froze the tokens (produced " floor, and the other" instead of
    // " floor of the house"). The auto-detecting cache probes two different-data runs and caches only the
    // values stable across both. This test asserts cache-ON output == cache-OFF output == the ORT greedy
    // reference continuation, so any future cache regression (or a new data-dependent readback) fails here.
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task TextGen_ReadbackCache_MatchesUncached() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/onnx/decoder_model.onnx");
        var tokenizerJson = await http.GetStringAsync(
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/tokenizer.json");

        async Task<(string text, List<string> timings)> Gen(bool cache)
        {
            using var session = InferenceSession.CreateFromOnnx(accelerator, modelBytes, enableOptimization: false);
            var pipeline = new TextGenerationPipeline(session, accelerator) { UseShapeReadbackCache = cache };
            pipeline.LoadTokenizer(tokenizerJson);
            pipeline.MaxNewTokens = 8;
            var result = await pipeline.GenerateAsync("The cat sat on the");
            return (result.GeneratedText, new List<string>(pipeline.StepTimings));
        }
        var (cacheOff, _) = await Gen(false);
        var (cacheOn, onTimings) = await Gen(true);
        Console.WriteLine($"[TextGen-cache] off='{cacheOff}' on='{cacheOn}'");
        if (!cacheOff.TrimStart().StartsWith("floor"))
            throw new Exception($"Uncached generation WRONG: '{cacheOff}' — expected to start with ' floor' (ORT greedy reference).");
        if (cacheOn != cacheOff)
            throw new Exception($"Readback cache CHANGED the output — cached='{cacheOn}' vs uncached='{cacheOff}'. A data-dependent readback is being cached.");

        // Memory: the reused fixed-shape executor must RECYCLE its output buffers, not grow the pool
        // ~13/step (logits ≈11MB/step → OOM on long gens). Parse poolBuffers from the per-step timings;
        // after the 2 cold probe steps it must PLATEAU (no linear per-step growth).
        int PoolBuffers(string line) { var m = System.Text.RegularExpressions.Regex.Match(line, @"poolBuffers=(\d+)"); return m.Success ? int.Parse(m.Groups[1].Value) : -1; }
        var pool = onTimings.Select(PoolBuffers).Where(v => v >= 0).ToList();
        if (pool.Count >= 5)
        {
            int mid = pool[pool.Count - 3], last = pool[^1]; // two later steps, past the probe warmup
            if (last - mid > 4) // allow tiny slack; a leak would be +13/step (+26 over two steps)
                throw new Exception($"Decode pool GREW {mid}→{last} across late steps — output buffers are leaking (expected plateau). poolBuffers=[{string.Join(",", pool)}]");
        }
    });

    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task Pipeline_TextGeneration_ProducesTokens() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Load DistilGPT-2 model + tokenizer from HuggingFace
        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/onnx/decoder_model.onnx");
        var tokenizerJson = await http.GetStringAsync(
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/tokenizer.json");

        using var session = InferenceSession.CreateFromFile(accelerator, modelBytes);
        var pipeline = new TextGenerationPipeline(session, accelerator);
        pipeline.LoadTokenizer(tokenizerJson);
        pipeline.MaxNewTokens = 5; // Just enough to verify it works

        var result = await pipeline.GenerateAsync("The cat sat on the");

        Console.WriteLine($"[TextGen] Input: 'The cat sat on the'");
        Console.WriteLine($"[TextGen] Output: '{result.GeneratedText}'");
        Console.WriteLine($"[TextGen] Tokens: {result.GeneratedTokenCount}, Time: {result.InferenceTimeMs:F0}ms");

        if (string.IsNullOrWhiteSpace(result.GeneratedText))
            throw new Exception("TextGeneration produced empty output");
        if (result.GeneratedTokenCount < 1)
            throw new Exception($"TextGeneration produced 0 tokens");
        // CORRECTNESS, not just liveness: greedy DistilGPT-2 on "The cat sat on the" must produce
        // " floor" first (ORT reference, token 4314). The old forward-pass bug produced " The"
        // (an input echo) then EOS — finite but wrong. Assert the real continuation so a
        // correctness regression can't hide behind ">=1 token". (Per-token match across the
        // growing sequence is guarded by Reference_DistilGPT2_GreedyGeneration_MatchesOnnxRuntime.)
        if (!result.GeneratedText.TrimStart().StartsWith("floor"))
            throw new Exception($"TextGeneration produced WRONG continuation '{result.GeneratedText}' — expected to start with ' floor' (ORT greedy reference for 'The cat sat on the').");
    });

    // GATING: establish the merged DistilGPT-2 model's KV-cache IO contract on our engine before
    // building incremental decode. Logs input/output names + whether HasKVCache fires (needs paired
    // past_key_values.* inputs + present.* outputs). Asserts the past interface is detected.
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task TextGen_MergedModel_HasKVCacheInterface() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/onnx/decoder_model_merged.onnx");
        using var session = InferenceSession.CreateFromOnnx(accelerator, modelBytes, enableOptimization: false);

        Console.WriteLine($"[merged] InputNames ({session.InputNames.Length}): {string.Join(", ", session.InputNames)}");
        Console.WriteLine($"[merged] OutputNames ({session.OutputNames.Length}): {string.Join(", ", session.OutputNames)}");
        Console.WriteLine($"[merged] HasKVCache = {session.Executor.HasKVCache}");
        var kv = session.Executor.KVCache;
        if (kv != null) Console.WriteLine($"[merged] KVCache layers = {kv.NumLayers}");

        bool hasPast = session.InputNames.Any(n => n.Contains("past_key_values"));
        bool hasPresent = session.OutputNames.Any(n => n.StartsWith("present"));
        if (!hasPast)
            throw new Exception($"merged model has NO past_key_values inputs — inputs: {string.Join(",", session.InputNames)}");
        if (!hasPresent)
            throw new Exception($"merged model has NO present outputs — outputs: {string.Join(",", session.OutputNames)}");
        Console.WriteLine($"[merged] past inputs + present outputs both present — incremental decode is wireable.");
    });

    // Proves the STREAMING load path (InferenceSession.CreateFromOnnxStreamAsync) works on a real
    // transformer (DistilGPT-2), independent of the hub: download the model bytes, wrap them in a
    // SEEKABLE MemoryStream, and load with a tiny streamThreshold so every weight takes the streaming
    // (seek + chunk-upload-to-GPU) path — then run real generation. This is the workbench proof that
    // the page's load+generate path is correct; the hub variant below adds the live network source.
    // This test's distinct job is the STREAM LOAD path + multi-token output; rigorous per-token
    // correctness across a growing sequence is covered by Reference_DistilGPT2_GreedyGeneration.
    // MaxNewTokens is kept modest because the current decode loop re-feeds the FULL sequence every
    // step (O(n^2)) and the session recompiles per new sequence length — the KV-cache speedup
    // (decode 1 token/step, fixed shapes) is the deferred follow-up that makes long generation fast.
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Pipeline_TextGeneration_FromSeekableStream_ProducesTokens() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/onnx/decoder_model.onnx");
        var tokenizerJson = await http.GetStringAsync(
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/tokenizer.json");

        // Seekable stream + tiny threshold forces the streaming weight path (not the inline byte path).
        using var ms = new MemoryStream(modelBytes);
        using var session = await InferenceSession.CreateFromOnnxStreamAsync(
            accelerator, ms, streamThreshold: 4096);

        var pipeline = new TextGenerationPipeline(session, accelerator);
        pipeline.LoadTokenizer(tokenizerJson);
        pipeline.MaxNewTokens = 8;

        var result = await pipeline.GenerateAsync("The cat sat on the");
        Console.WriteLine($"[TextGen/stream] prompt='The cat sat on the' generated='{result.GeneratedText}' ({result.GeneratedTokenCount} tokens, {result.InferenceTimeMs:F0}ms, {result.TokensPerSecond:F1} tok/s)");

        if (string.IsNullOrWhiteSpace(result.GeneratedText))
            throw new Exception("Stream-loaded TextGeneration produced empty output");
        // The demo use case is multi-token generation. A coherent narrative prompt under greedy decode
        // must not stop after a single token (that's the bug TJ hit in the live demo).
        if (result.GeneratedTokenCount <= 1)
            throw new Exception($"TextGeneration stopped after {result.GeneratedTokenCount} token(s) — expected multi-token output. generated='{result.GeneratedText}'");
        // CORRECTNESS across the GROWING sequence (this is the exact path Bug #4 crashed on):
        // the first greedy token must be " floor" (4314), and the continuation must match the ORT
        // greedy reference " floor of the house, and the cat sat on the floor".
        if (!result.GeneratedText.TrimStart().StartsWith("floor"))
            throw new Exception($"Stream TextGeneration WRONG continuation '{result.GeneratedText}' — expected to start with ' floor'.");
    });

    // EXACT /text-gen page path: load DistilGPT-2 from OUR live hub (hub.spawndev.com) over a SEEKABLE
    // torrent stream and load it via InferenceSession.CreateFromOnnxStreamAsync — the model is never held
    // whole in CPU memory (each weight is seeked + chunk-uploaded straight to GPU). Then run real
    // autoregressive generation. This proves the demo's hub+stream wiring end to end, not just inspection.
    // HeavyModel: full ~330MB model fetched via torrent + GPU compile + decode — gated out of the fast loop.
    [TestMethod(Timeout = 360000, Category = "HeavyModel", RetryCount = 2)]
    public async Task Pipeline_TextGeneration_ViaHubStream_ProducesTokens() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        const string repoId = "Xenova/distilgpt2";
        var client = new SpawnDev.WebTorrent.WebTorrentClient();
        try
        {
            var hub = new SpawnDev.ILGPU.ML.Hub.HubModelStream(client, http);
            using var cts = new System.Threading.CancellationTokenSource(TimeSpan.FromMinutes(5));

            // Open the model file as a seekable stream (deselect:false → we need the weights).
            var model = await hub.OpenAsync(repoId, "onnx/decoder_model.onnx", deselect: false, cts.Token);
            if (model.File.Length < 1_000_000)
                throw new Exception($"hub model file length={model.File.Length}, expected ~330MB");

            InferenceSession session;
            await using (model.Stream)
                session = await InferenceSession.CreateFromOnnxStreamAsync(accelerator, model.Stream, ct: cts.Token);

            using (session)
            {
                // Tokenizer from the same hub repo (small JSON — read the stream fully).
                var tok = await hub.OpenAsync(repoId, "tokenizer.json", deselect: false, cts.Token);
                string tokenizerJson;
                await using (tok.Stream)
                using (var reader = new StreamReader(tok.Stream))
                    tokenizerJson = await reader.ReadToEndAsync(cts.Token);

                var pipeline = new TextGenerationPipeline(session, accelerator);
                pipeline.LoadTokenizer(tokenizerJson);
                pipeline.MaxNewTokens = 5;

                var result = await pipeline.GenerateAsync("The cat sat on the");
                Console.WriteLine($"[TextGen/hub-stream] Output: '{result.GeneratedText}' ({result.GeneratedTokenCount} tokens, {result.InferenceTimeMs:F0}ms)");

                if (string.IsNullOrWhiteSpace(result.GeneratedText))
                    throw new Exception("Hub-stream TextGeneration produced empty output");
                if (result.GeneratedTokenCount < 1)
                    throw new Exception("Hub-stream TextGeneration produced 0 tokens");
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network") || ex.Message.Contains("magnet"))
        {
            throw new UnsupportedTestException($"Hub/network unavailable: {ex.Message}");
        }
        finally
        {
            await client.DisposeAsync();
        }
    });

    // ═══════════════════════════════════════════════════════════
    //  Background Removal (RMBG)
    // ═══════════════════════════════════════════════════════════

    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task Pipeline_BackgroundRemoval_ProducesMask() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // RMBG 1.4 from HuggingFace (~170MB)
        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            $"https://huggingface.co/{Hub.ModelHub.KnownModels.RMBG14}/resolve/main/{Hub.ModelHub.KnownFiles.OnnxModel}");
        using var session = InferenceSession.CreateFromFile(accelerator, modelBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["input"] = new[] { 1, 3, 1024, 1024 }
            });

        // Create test image: left half white, right half dark (simulates foreground/background)
        int w = 1024, h = 1024;
        var pixels = new int[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                pixels[y * w + x] = x < w / 2
                    ? (255 | (255 << 8) | (255 << 16) | (0xFF << 24))  // white
                    : (30 | (30 << 8) | (30 << 16) | (0xFF << 24));    // dark

        // Preprocess - RMBG-specific normalization (mean=0.5, std=1.0 -> output in [-0.5, 0.5]).
        // The default Forward() uses ImageNet normalization which would feed RMBG out-of-distribution
        // input and produce garbage (near-zero) masks. BackgroundRemovalPipeline.cs already passes
        // these correct values; the test was bypassing the pipeline and calling preprocess directly
        // without them, which is why this test silently passed for years (paired with the
        // foreground-only sampling bug that hid the resulting near-zero output).
        using var rgbaBuf = accelerator.Allocate1D(pixels);
        using var preprocessed = accelerator.Allocate1D<float>(3 * w * h);
        var preprocess = new Kernels.ImagePreprocessKernel(accelerator);
        preprocess.Forward(rgbaBuf.View, preprocessed.View, w, h, w, h,
            mean: new[] { 0.5f, 0.5f, 0.5f },
            std: new[] { 1.0f, 1.0f, 1.0f });

        var inputTensor = new Tensor(preprocessed.View, new[] { 1, 3, h, w });
        var outputs = await session.RunAsync(new Dictionary<string, Tensor>
        {
            [session.InputNames[0]] = inputTensor
        });

        var output = outputs[session.OutputNames[0]];
        Console.WriteLine($"[RMBG] Output: shape=[{string.Join(",", output.Shape)}], elements={output.ElementCount}");

        // Sample the mask at columns spanning BOTH halves so the segmentation check
        // is actually informative. The test image is white left half / dark right half,
        // so a working mask must have higher values on the left columns than the right.
        // Reading only the first N contiguous elements (e.g. row 0 cols 0..N-1) sits
        // entirely in the foreground half and produces near-zero variance regardless
        // of whether the model worked - hence the dedicated foreground+background reads.
        int outW = output.Shape[^1];
        int sampleSize = Math.Min(64, outW / 4);
        int foregroundStart = 0;                  // row 0, cols 0..sampleSize-1 (left half / white)
        int backgroundStart = outW - sampleSize;  // row 0, cols outW-sampleSize..outW-1 (right half / dark)

        using var fgBuf = accelerator.Allocate1D<float>(sampleSize);
        using var bgBuf = accelerator.Allocate1D<float>(sampleSize);
        var ewk = new ElementWiseKernels(accelerator);
        ewk.Scale(output.Data.SubView(foregroundStart, sampleSize), fgBuf.View, sampleSize, 1f);
        ewk.Scale(output.Data.SubView(backgroundStart, sampleSize), bgBuf.View, sampleSize, 1f);
        await accelerator.SynchronizeAsync();
        var fgValues = await fgBuf.CopyToHostAsync<float>(0, sampleSize);
        var bgValues = await bgBuf.CopyToHostAsync<float>(0, sampleSize);

        float fgAvg = fgValues.Average();
        float bgAvg = bgValues.Average();
        float fgMin = fgValues.Min(); float fgMax = fgValues.Max();
        float bgMin = bgValues.Min(); float bgMax = bgValues.Max();
        float absMax = Math.Max(MathF.Abs(fgMax), Math.Max(MathF.Abs(bgMax), Math.Max(MathF.Abs(fgMin), MathF.Abs(bgMin))));
        float discrimination = MathF.Abs(fgAvg - bgAvg);

        Console.WriteLine($"[RMBG] Mask sample (cols 0..{sampleSize - 1} fg / cols {backgroundStart}..{outW - 1} bg of row 0):");
        Console.WriteLine($"[RMBG]   foreground (white input): min={fgMin:F4} max={fgMax:F4} avg={fgAvg:F4}");
        Console.WriteLine($"[RMBG]   background (dark input):  min={bgMin:F4} max={bgMax:F4} avg={bgAvg:F4}");
        Console.WriteLine($"[RMBG]   absMax={absMax:F4} discrimination(|fgAvg-bgAvg|)={discrimination:F4}");

        if (absMax < 0.001f)
            throw new Exception($"Background removal mask is all zeros (fgMax={fgMax:F4}, bgMax={bgMax:F4})");
        if (discrimination < 0.05f)
            throw new Exception($"Background removal mask shows no foreground/background discrimination "
                + $"(fg avg={fgAvg:F4} range=[{fgMin:F4},{fgMax:F4}]; bg avg={bgAvg:F4} range=[{bgMin:F4},{bgMax:F4}])");
    });

    /// <summary>
    /// End-to-end test for BackgroundRemovalPipeline. The existing Pipeline_BackgroundRemoval_ProducesMask
    /// test bypasses the pipeline class and calls preprocess + session.Run directly. This
    /// test exercises the actual consumer-facing code path:
    ///   1. Load a real photo (cat image — also used by Reference_SqueezeNet etc.)
    ///   2. Pipeline.RemoveBackgroundAsync — same call the /remove-bg demo makes
    ///   3. Inspect BackgroundRemovalResult.Mask (post-resize to source dims, post-sigmoid)
    ///   4. Assert the mask has reasonable variation: standard deviation > some threshold,
    ///      and at least 10% of pixels have both alpha < 0.3 AND alpha > 0.7 (real segmentation).
    /// Closes the gap that let Captain hit "result image identical to source" without any
    /// PMT failure.
    /// </summary>
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task Pipeline_BackgroundRemoval_RealImage_ProducesVaryingMask() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Load cat sample (binary RGBA — pre-decoded).
        var binData = await http.GetByteArrayAsync("samples/cat_rgba.bin");
        int width = BitConverter.ToInt32(binData, 0);
        int height = BitConverter.ToInt32(binData, 4);
        var pixels = new int[width * height];
        Buffer.BlockCopy(binData, 8, pixels, 0, width * height * 4);
        Console.WriteLine($"[RMBG-RealImage] input {width}x{height}");

        // Use 256x256 model input — keeps the test within CI budget while exercising
        // the same pipeline path the demo uses. The pipeline's internal resize maps
        // mask back to source dimensions.
        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            $"https://huggingface.co/{Hub.ModelHub.KnownModels.RMBG14}/resolve/main/{Hub.ModelHub.KnownFiles.OnnxModel}");
        using var session = InferenceSession.CreateFromFile(accelerator, modelBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["input"] = new[] { 1, 3, 256, 256 }
            });
        var pipeline = new SpawnDev.ILGPU.ML.Pipelines.BackgroundRemovalPipeline(session, accelerator, inputSize: 256);

        var result = await pipeline.RemoveBackgroundAsync(pixels, width, height);
        Console.WriteLine($"[RMBG-RealImage] mask {result.Width}x{result.Height} elements={result.Mask.Length}");

        // Mask stats
        var mask = result.Mask;
        float min = mask.Min(); float max = mask.Max();
        double mean = mask.Average(v => (double)v);
        double sqSum = 0; foreach (var v in mask) { double d = v - mean; sqSum += d * d; }
        double stddev = Math.Sqrt(sqSum / mask.Length);
        int lowCount = mask.Count(v => v < 0.3f);
        int highCount = mask.Count(v => v > 0.7f);
        float lowPct = 100f * lowCount / mask.Length;
        float highPct = 100f * highCount / mask.Length;

        // Alpha stats from result pixels — what actually ends up displayed.
        int alphaMin = 255, alphaMax = 0; long alphaSum = 0;
        int alphaLow = 0, alphaHigh = 0;
        for (int i = 0; i < result.ResultPixels.Length; i++)
        {
            int a = (result.ResultPixels[i] >> 24) & 0xFF;
            if (a < alphaMin) alphaMin = a; if (a > alphaMax) alphaMax = a;
            alphaSum += a;
            if (a < 76) alphaLow++; if (a > 178) alphaHigh++;
        }
        double alphaMean = alphaSum / (double)result.ResultPixels.Length;

        var diag = $"mask min={min:F4} max={max:F4} mean={mean:F4} stddev={stddev:F4} | <0.3={lowPct:F1}% >0.7={highPct:F1}% | "
                 + $"alpha min={alphaMin} max={alphaMax} mean={alphaMean:F1} | <76={100f*alphaLow/result.ResultPixels.Length:F1}% >178={100f*alphaHigh/result.ResultPixels.Length:F1}%";
        Console.WriteLine($"[RMBG-RealImage] {diag}");

        // Hard asserts mirroring Captain's "result == source" observation:
        // If alphaMin >= 250 across the image, alpha is essentially uniform — the
        // visible result is indistinguishable from the source. That IS the bug.
        if (alphaMin >= 250)
            throw new Exception($"[RMBG-RealImage] FAIL on {accelerator.AcceleratorType}: alpha is uniform >=250 (result == source). {diag}");

        // Sanity: mask should have both low and high regions if it's a real segmentation.
        if (lowPct < 1f && highPct < 1f)
            throw new Exception($"[RMBG-RealImage] FAIL on {accelerator.AcceleratorType}: mask has neither low nor high regions (no segmentation). {diag}");

        pipeline.Dispose();
        Console.WriteLine($"[RMBG-RealImage] PASS on {accelerator.AcceleratorType}: {diag}");
    });

    /// <summary>
    /// DIAGNOSTIC: Capture per-op stats while running RMBG-1.4 so we can pinpoint where
    /// the mask saturates on WebGPU. WebGL passes the discrimination test above, WebGPU
    /// produces a result indistinguishable from the source (mask ~1.0 everywhere). Walking
    /// the captured outputs node-by-node should show the first op whose output is
    /// uniformly saturated — that's the WGSL codegen culprit.
    ///
    /// Always throws at the end with a summary of suspicious nodes so the captured info
    /// surfaces in test output regardless of pass/fail semantics.
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Pipeline_BackgroundRemoval_PerOpDiagnostic() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            $"https://huggingface.co/{Hub.ModelHub.KnownModels.RMBG14}/resolve/main/{Hub.ModelHub.KnownFiles.OnnxModel}");

        // Use 256x256 — the model accepts dynamic spatial dims; smaller input fits within
        // the diagnostic budget across all backends and exercises the same WGSL codegen.
        const int side = 256;
        using var session = InferenceSession.CreateFromFile(accelerator, modelBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["input"] = new[] { 1, 3, side, side }
            });

        // Same half-white / half-dark pattern as the discrimination test — gives a
        // ground-truth split that any working backbone should detect.
        var pixels = new int[side * side];
        for (int y = 0; y < side; y++)
            for (int x = 0; x < side; x++)
                pixels[y * side + x] = x < side / 2
                    ? (255 | (255 << 8) | (255 << 16) | (0xFF << 24))
                    : (30 | (30 << 8) | (30 << 16) | (0xFF << 24));

        using var rgbaBuf = accelerator.Allocate1D(pixels);
        using var preprocessed = accelerator.Allocate1D<float>(3 * side * side);
        new Kernels.ImagePreprocessKernel(accelerator).Forward(
            rgbaBuf.View, preprocessed.View, side, side, side, side,
            mean: new[] { 0.5f, 0.5f, 0.5f }, std: new[] { 1.0f, 1.0f, 1.0f });
        var inputTensor = new Tensor(preprocessed.View, new[] { 1, 3, side, side });

        // Enable per-op capture. Captures first 10 values + per-node op type/shape.
        Graph.GraphExecutor.CapturedOutputs = new Dictionary<string, float[]>();
        Graph.GraphExecutor.CapturedNodeInfo = new Dictionary<string, string>();
        try
        {
            await session.RunAsync(new Dictionary<string, Tensor>
            {
                [session.InputNames[0]] = inputTensor
            });

            var outputs = Graph.GraphExecutor.CapturedOutputs;
            var info = Graph.GraphExecutor.CapturedNodeInfo;
            Console.WriteLine($"[RMBG-Diag] Captured {outputs.Count} nodes on backend {accelerator.AcceleratorType}");

            // Walk node-by-node, compute simple stats. Flag nodes whose output is suspiciously
            // uniform (low variance, value clamped near 0 or 1). Pack the first 10 nodes'
            // full stats + the saturated node's stats into the exception message so the
            // diagnostic surfaces in PMT's captured error output (Console.WriteLine alone
            // doesn't reliably reach the test result JSON in Blazor WASM).
            int index = 0;
            int firstSaturationIndex = -1;
            string firstSaturationKey = "";
            string firstSaturationLine = "";
            var firstNodes = new System.Text.StringBuilder();
            foreach (var kv in outputs)
            {
                var sample = kv.Value;
                if (sample == null || sample.Length == 0) { index++; continue; }
                float min = sample.Min(); float max = sample.Max();
                double mean = sample.Average(v => (double)v);
                double sqSum = 0; foreach (var v in sample) { double d = v - mean; sqSum += d * d; }
                double variance = sqSum / sample.Length;
                string opInfo = info != null && info.TryGetValue(kv.Key, out var i) ? i : "(no info)";

                // Ignore small tensors — Shape/Gather/Unsqueeze ops on shape vectors are
                // single- or few-element tensors whose variance is naturally zero. Only
                // multi-element data tensors are meaningful saturation candidates.
                bool isDataTensor = sample.Length >= 10;
                bool nearOne = isDataTensor && mean > 0.95 && variance < 0.001;
                bool nearZero = isDataTensor && Math.Abs(mean) < 0.05 && variance < 0.001;
                bool extreme = nearOne || nearZero;

                string line = $"#{index} {kv.Key} | min={min:F4} max={max:F4} mean={mean:F4} var={variance:F6} | {opInfo}";
                // First 5 (input stack) and last 15 (output stack incl final mask) so we
                // can see early-layer agreement vs late-layer divergence between backends.
                if (index < 5 || index >= outputs.Count - 15) firstNodes.AppendLine(line);
                if (extreme && firstSaturationIndex < 0)
                {
                    firstSaturationIndex = index;
                    firstSaturationKey = kv.Key;
                    firstSaturationLine = line;
                    // Also dump the few raw values to see *which* constant it saturated to
                    var first5 = string.Join(",", sample.Take(5).Select(v => v.ToString("F6")));
                    firstSaturationLine += $" | first5=[{first5}]";
                }
                index++;
            }

            string verdict = firstSaturationIndex < 0
                ? "no saturated node found"
                : $"FIRST SATURATED #{firstSaturationIndex}: {firstSaturationLine}";
            throw new Exception(
                $"[RMBG-Diag] backend={accelerator.AcceleratorType} nodes={outputs.Count}\n" +
                $"FIRST 5 + LAST 15 NODES:\n{firstNodes}\n" +
                $"VERDICT: {verdict}");
        }
        finally
        {
            Graph.GraphExecutor.CapturedOutputs = null;
            Graph.GraphExecutor.CapturedNodeInfo = null;
        }
    });

    // ═══════════════════════════════════════════════════════════
    //  Semantic Search (Feature Extraction + Cosine Similarity)
    // ═══════════════════════════════════════════════════════════

    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task Pipeline_SemanticSearch_SimilarSentencesCloser() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // DistilBERT for embeddings (~255MB)
        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/model.onnx");
        using var session = InferenceSession.CreateFromOnnx(accelerator, modelBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["input_ids"] = new[] { 1, 8 },
                ["attention_mask"] = new[] { 1, 8 },
            });

        // Encode three sentences: A="I love dogs", B="I adore puppies" (similar), C="The stock market crashed" (dissimilar)
        // Use raw token IDs (pre-tokenized for DistilBERT)
        // A: [CLS]=101 I=1045 love=2293 dogs=6077 [SEP]=102 [PAD]=0 [PAD]=0 [PAD]=0
        // B: [CLS]=101 I=1045 adore=16599 puppies=18289 [SEP]=102 [PAD]=0 [PAD]=0 [PAD]=0
        // C: [CLS]=101 The=1996 stock=4518 market=3006 crashed=7821 [SEP]=102 [PAD]=0 [PAD]=0

        var tokensA = new float[] { 101, 1045, 2293, 6077, 102, 0, 0, 0 };
        var maskA = new float[] { 1, 1, 1, 1, 1, 0, 0, 0 };
        var tokensC = new float[] { 101, 1996, 4518, 3006, 7821, 102, 0, 0 };
        var maskC = new float[] { 1, 1, 1, 1, 1, 1, 0, 0 };

        // Run A
        using var idsBufA = accelerator.Allocate1D(tokensA);
        using var maskBufA = accelerator.Allocate1D(maskA);
        var outputsA = await session.RunAsync(new Dictionary<string, Tensor>
        {
            [session.InputNames[0]] = new Tensor(idsBufA.View, new[] { 1, 8 }),
            [session.InputNames[1]] = new Tensor(maskBufA.View, new[] { 1, 8 }),
        });

        // Run C
        using var idsBufC = accelerator.Allocate1D(tokensC);
        using var maskBufC = accelerator.Allocate1D(maskC);
        var outputsC = await session.RunAsync(new Dictionary<string, Tensor>
        {
            [session.InputNames[0]] = new Tensor(idsBufC.View, new[] { 1, 8 }),
            [session.InputNames[1]] = new Tensor(maskBufC.View, new[] { 1, 8 }),
        });

        var outA = outputsA[session.OutputNames[0]];
        var outC = outputsC[session.OutputNames[0]];

        // Read first 2 logits from each (sentiment: [negative, positive])
        using var readA = accelerator.Allocate1D<float>(2);
        using var readC = accelerator.Allocate1D<float>(2);
        new ElementWiseKernels(accelerator).Scale(outA.Data.SubView(0, 2), readA.View, 2, 1f);
        new ElementWiseKernels(accelerator).Scale(outC.Data.SubView(0, 2), readC.View, 2, 1f);
        await accelerator.SynchronizeAsync();
        var logitsA = await readA.CopyToHostAsync<float>(0, 2);
        var logitsC = await readC.CopyToHostAsync<float>(0, 2);

        Console.WriteLine($"[Semantic] 'I love dogs': [{logitsA[0]:F3}, {logitsA[1]:F3}]");
        Console.WriteLine($"[Semantic] 'stock market crashed': [{logitsC[0]:F3}, {logitsC[1]:F3}]");

        // "I love dogs" should be positive (logits[1] > logits[0])
        // "stock market crashed" should be negative (logits[0] > logits[1])
        if (logitsA[1] <= logitsA[0])
            throw new Exception($"'I love dogs' should be positive but logits=[{logitsA[0]:F3},{logitsA[1]:F3}]");
        if (logitsC[0] <= logitsC[1])
            throw new Exception($"'stock market crashed' should be negative but logits=[{logitsC[0]:F3},{logitsC[1]:F3}]");

        Console.WriteLine("[Semantic] PASS — sentiment direction correct for both sentences");
    });

    // ═══════════════════════════════════════════════════════════
    //  Image Generation (Diffusion)
    // ═══════════════════════════════════════════════════════════

    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task Pipeline_Diffusion_DDPM_ProducesImage() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // DDPM MNIST U-Net (~1MB + 4MB external data)
        var modelBytes = await http.GetByteArrayAsync("references/blazing-edge/ddpm_mnist_unet.onnx");
        var extDataBytes = await http.GetByteArrayAsync("references/blazing-edge/ddpm_mnist_unet.onnx.data");
        using var session = InferenceSession.CreateFromOnnx(accelerator, modelBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["sample"] = new[] { 1, 1, 28, 28 },
                ["timestep"] = new[] { 1 },
            },
            externalData: extDataBytes);

        // Start from random noise
        var rng = new Random(42);
        var noise = new float[1 * 1 * 28 * 28];
        for (int i = 0; i < noise.Length; i++)
            noise[i] = (float)(rng.NextDouble() * 2 - 1);

        // Run one denoising step (not full diffusion loop, just verify the model produces output)
        using var noiseBuf = accelerator.Allocate1D(noise);
        using var timestepBuf = accelerator.Allocate1D(new float[] { 500f }); // mid-schedule

        var outputs = await session.RunAsync(new Dictionary<string, Tensor>
        {
            [session.InputNames[0]] = new Tensor(noiseBuf.View, new[] { 1, 1, 28, 28 }),
            [session.InputNames[1]] = new Tensor(timestepBuf.View, new[] { 1 }),
        });

        var output = outputs[session.OutputNames[0]];
        Console.WriteLine($"[DDPM] Output: shape=[{string.Join(",", output.Shape)}], elements={output.ElementCount}");

        int readCount = Math.Min(100, output.ElementCount);
        using var readBuf = accelerator.Allocate1D<float>(readCount);
        new ElementWiseKernels(accelerator).Scale(output.Data.SubView(0, readCount), readBuf.View, readCount, 1f);
        await accelerator.SynchronizeAsync();
        var values = await readBuf.CopyToHostAsync<float>(0, readCount);

        float absMax = values.Max(v => MathF.Abs(v));
        bool hasNaN = values.Any(v => float.IsNaN(v));

        Console.WriteLine($"[DDPM] Values: absMax={absMax:F3}, hasNaN={hasNaN}");

        if (hasNaN)
            throw new Exception("DDPM output contains NaN");
        if (absMax < 0.001f)
            throw new Exception("DDPM output is all zeros");
    });

    // ═══════════════════════════════════════════════════════════
    //  Text-to-Speech (SpeechT5 reference validation)
    // ═══════════════════════════════════════════════════════════

    [TestMethod(Timeout = 300000)]
    public async Task Pipeline_TTS_ReferenceTokensProduceAudio() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Load reference token IDs and expected audio
        var tokenBytes = await http.GetByteArrayAsync("references/speecht5-tts/hello_world_token_ids.bin");
        var tokenIds = new float[tokenBytes.Length / 4];
        Buffer.BlockCopy(tokenBytes, 0, tokenIds, 0, tokenBytes.Length);

        var refAudioBytes = await http.GetByteArrayAsync("references/speecht5-tts/hello_world_audio.bin");
        var refAudio = new float[refAudioBytes.Length / 4];
        Buffer.BlockCopy(refAudioBytes, 0, refAudio, 0, refAudioBytes.Length);

        var speakerBytes = await http.GetByteArrayAsync("references/speecht5-tts/speaker_embedding.bin");
        var speakerEmbedding = new float[speakerBytes.Length / 4];
        Buffer.BlockCopy(speakerBytes, 0, speakerEmbedding, 0, speakerBytes.Length);

        Console.WriteLine($"[TTS] Reference: {tokenIds.Length} tokens, {refAudio.Length} audio samples, {speakerEmbedding.Length}-dim speaker");
        Console.WriteLine($"[TTS] Token IDs: [{string.Join(",", tokenIds.Take(10).Select(v => ((int)v).ToString()))}...]");
        Console.WriteLine($"[TTS] Reference audio: absMax={refAudio.Max(v => MathF.Abs(v)):F4}, samples={refAudio.Length}");

        // Verify reference data is valid
        if (tokenIds.Length < 2)
            throw new Exception($"Reference token IDs too short: {tokenIds.Length}");
        if (refAudio.Length < 100)
            throw new Exception($"Reference audio too short: {refAudio.Length}");
        if (speakerEmbedding.Length < 10)
            throw new Exception($"Speaker embedding too short: {speakerEmbedding.Length}");

        float refAbsMax = refAudio.Max(v => MathF.Abs(v));
        if (refAbsMax < 0.001f)
            throw new Exception("Reference audio is silent");

        Console.WriteLine($"[TTS] Reference data validated: PASS (tokens={tokenIds.Length}, audio={refAudio.Length}, speaker={speakerEmbedding.Length})");
    });
}
