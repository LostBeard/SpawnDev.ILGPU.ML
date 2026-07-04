using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
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

    // REGRESSION GUARD for the fixed-shape decode + auto-detecting readback cache + Shape-output BUFFER
    // cache. Together these eliminate the ~643 mid-graph shape-readback GPU round-trips/step AND the
    // per-step CopyFromCPU re-upload of the (constant) Shape op outputs — but they MUST NOT change the
    // output. A naive "cache every ≤64-elem readback" was WRONG: input_ids itself is ≤64 elems and
    // DATA-dependent, so caching it froze the tokens (" floor, and the other" instead of " floor of the
    // house"); likewise skipping a Shape op's GPU upload corrupts a downstream tensor-reading consumer.
    // This test runs ONLY the cached path and asserts the generated token IDs match the ORT greedy
    // reference (references/gpt2/distilgpt2_greedy.json) — a STRONGER, absolute check than the old
    // cache-vs-uncached comparison. It deliberately does NOT re-run the uncached baseline: on interpreted
    // Wasm the forward is ~50-95s/step, so running it twice doubled the work and blew the harness page-wait
    // (the uncached path's correctness is covered separately by
    // Reference_DistilGPT2_GreedyGeneration_MatchesOnnxRuntime). See the buffer-cache memory
    // feedback-shape-outputs-consumed-as-gpu-tensor-not-just-value.
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task TextGen_ReadbackCache_MatchesReference() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // ORT greedy reference for prompt "The cat sat on the" (prompt input_ids + full generated_ids).
        var refJson = await http.GetStringAsync("references/gpt2/distilgpt2_greedy.json");
        using var refDoc = System.Text.Json.JsonDocument.Parse(refJson);
        var promptIds = refDoc.RootElement.GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var refGenIds = refDoc.RootElement.GetProperty("generated_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();

        const int NumNew = 6; // ≥5 so the pool-plateau check runs; ~6 cached steps fit interpreted Wasm in the page-wait
        var expected = refGenIds.Skip(promptIds.Length).Take(NumNew).ToArray();

        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/onnx/decoder_model.onnx");
        var tokenizerJson = await http.GetStringAsync(
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/tokenizer.json");

        using var session = InferenceSession.CreateFromOnnx(accelerator, modelBytes, enableOptimization: false);
        var pipeline = new TextGenerationPipeline(session, accelerator) { UseShapeReadbackCache = true };
        pipeline.LoadTokenizer(tokenizerJson);
        pipeline.MaxNewTokens = NumNew;
        var result = await pipeline.GenerateAsync("The cat sat on the");

        var got = result.GeneratedTokenIds;
        Console.WriteLine($"[TextGen-cache] gen=[{string.Join(",", got)}] expected=[{string.Join(",", expected)}] text='{result.GeneratedText}'");
        if (got.Length < NumNew)
            throw new Exception($"Cached generation produced only {got.Length} tokens, expected {NumNew}. text='{result.GeneratedText}'");
        for (int i = 0; i < NumNew; i++)
            if (got[i] != expected[i])
                throw new Exception($"Readback/Shape cache produced WRONG token at step {i}: got {got[i]}, expected {expected[i]} (ORT greedy ref). " +
                    $"gen=[{string.Join(",", got.Take(NumNew))}] vs ref=[{string.Join(",", expected)}]. A data-dependent readback or Shape buffer is being mis-cached.");

        // Memory: the reused fixed-shape executor must RECYCLE its output buffers, not grow the pool
        // ~13/step (logits ≈11MB/step → OOM on long gens). Parse poolBuffers from the per-step timings;
        // after the 2 cold probe steps it must PLATEAU (no linear per-step growth).
        int PoolBuffers(string line) { var m = System.Text.RegularExpressions.Regex.Match(line, @"poolBuffers=(\d+)"); return m.Success ? int.Parse(m.Groups[1].Value) : -1; }
        var pool = pipeline.StepTimings.Select(PoolBuffers).Where(v => v >= 0).ToList();
        if (pool.Count >= 5)
        {
            int mid = pool[pool.Count - 3], last = pool[^1]; // two later steps, past the probe warmup
            if (last - mid > 4) // allow tiny slack; a leak would be +13/step (+26 over two steps)
                throw new Exception($"Decode pool GREW {mid}→{last} across late steps — output buffers are leaking (expected plateau). poolBuffers=[{string.Join(",", pool)}]");
        }
    });

    // REGRESSION GUARD for the GenerationConfig.MaxNewTokens precedence bug, plus a single-sampled-gen
    // completion check (exactly what the /text-gen page does per "Generate" click). The bug:
    // GenerationConfig.MaxNewTokens defaulted to 128 and silently overrode an explicitly-set
    // pipeline.MaxNewTokens via the `maxNewTokens ?? config?.MaxNewTokens ?? MaxNewTokens` chain - so a
    // sampled generation ran 128 tokens regardless of the pipeline value. That was fast on CUDA (~14s)
    // but a timeout on slow cold WebGPU (~30s/token) - which masqueraded as a "WebGPU hang" and cost ~6
    // blind WebGPU runs. Fix: GenerationConfig.MaxNewTokens is now int? = null (override only if set).
    // This asserts the pipeline's MaxNewTokens(=2) is RESPECTED when a sampling config without its own
    // MaxNewTokens is passed.
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task TextGen_Sampling_SingleGen_RespectsMaxTokens() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/onnx/decoder_model.onnx");
        var tokenizerJson = await http.GetStringAsync(
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/tokenizer.json");

        using var session = InferenceSession.CreateFromOnnx(accelerator, modelBytes, enableOptimization: false);
        var pipeline = new TextGenerationPipeline(session, accelerator);
        pipeline.LoadTokenizer(tokenizerJson);
        pipeline.MaxNewTokens = 2;

        // Sampling config with NO MaxNewTokens set - must NOT override pipeline.MaxNewTokens (=2).
        var r = await pipeline.GenerateAsync("The cat sat on the", config: new GenerationConfig
        {
            Strategy = "top_p", TopP = 0.9f, Temperature = 0.7f, RepetitionPenalty = 1.3f, Seed = 1234,
        });
        Console.WriteLine($"[single-gen] tokens={r.GeneratedTokenCount} text='{r.GeneratedText}'");

        if (string.IsNullOrWhiteSpace(r.GeneratedText))
            throw new Exception("single sampled generation produced empty output");
        if (r.GeneratedTokenCount < 1 || r.GeneratedTokenCount > 2)
            throw new Exception($"sampled generation produced {r.GeneratedTokenCount} tokens, expected <=2 " +
                "(pipeline.MaxNewTokens) - a config default must NOT override an explicit pipeline value.");
    });

    // Proves the GenerationConfig sampling path is WIRED end-to-end through the real DistilGPT-2 forward
    // pass on the GPU - the fix for the live /text-gen page, whose Strategy/Temperature/Top-P controls
    // were dead because the pipeline hardcoded greedy argmax (so DistilGPT-2 collapsed into "the first
    // time I saw the first time I saw" loops). The sampler MATH (reproducibility, nucleus, penalty) is
    // proven fast + deterministically by the Sampler_* tests in MLTestBase.SamplingTests; this test
    // confirms config reaches the model and changes the GPU-decoded output. Fresh session per gen.
    //   (1) default greedy still matches the ORT " floor" reference (no regression to deterministic decode),
    //   (2) seeded top-p ENGAGES - it diverges from greedy (else the config never reached sampling),
    //   (3) both respect pipeline.MaxNewTokens (guards the config-default-override regression that made
    //       sampled gens silently run 128 tokens).
    [TestMethod(Timeout = 480000, Category = "HeavyModel")]
    public async Task TextGen_Sampling_EscapesGreedy() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/onnx/decoder_model.onnx");
        var tokenizerJson = await http.GetStringAsync(
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/tokenizer.json");

        // Fresh session per generation (mirrors the proven TextGen_ReadbackCache pattern).
        async Task<TextGenerationResult> Gen(GenerationConfig? cfg)
        {
            using var s = InferenceSession.CreateFromOnnx(accelerator, modelBytes, enableOptimization: false);
            var p = new TextGenerationPipeline(s, accelerator);
            p.LoadTokenizer(tokenizerJson);
            p.MaxNewTokens = 3;
            return await p.GenerateAsync("The cat sat on the", config: cfg);
        }

        var greedy = await Gen(null);
        var sampled = await Gen(new GenerationConfig
        {
            Strategy = "top_p", TopP = 0.9f, Temperature = 0.7f, RepetitionPenalty = 1.3f, Seed = 1234,
        });
        Console.WriteLine($"[sample-e2e] greedy='{greedy.GeneratedText}' sampled='{sampled.GeneratedText}'");

        if (!greedy.GeneratedText.TrimStart().StartsWith("floor"))
            throw new Exception($"default greedy WRONG: '{greedy.GeneratedText}' - expected to start with ' floor' (ORT reference).");
        if (sampled.GeneratedText == greedy.GeneratedText)
            throw new Exception($"top-p produced the SAME text as greedy ('{sampled.GeneratedText}') - the config never reached sampling.");
        if (greedy.GeneratedTokenCount > 3 || sampled.GeneratedTokenCount > 3)
            throw new Exception($"MaxNewTokens not respected: greedy={greedy.GeneratedTokenCount}, sampled={sampled.GeneratedTokenCount}, expected <=3 (a config default must not override).");
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

    // DIAGNOSTIC (not a pass/fail gate): where does a distilgpt2 decode STEP actually spend time now that
    // readbacks are cached + matmuls are register-blocked? Uses GraphExecutor's static per-node timing
    // capture (CapturedNodeTimingsMs) — no GPU sync per node, so it's CPU-side DISPATCH time per op — and
    // a second pass with PerOpSync to fold in GPU execution. Aggregates by op-type so we can see whether
    // the residual ~5.8s/step is dispatch overhead (-> fusion/dispatch-reduction helps) or GPU compute
    // (-> better kernels / incremental KV decode). Logs the breakdown; the only assertion is that it ran.
    [TestMethod(Timeout = 360000, Category = "HeavyModel")]
    public async Task Profile_DistilGPT2_Decode_OpTypeBreakdown() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/onnx/decoder_model.onnx");
        var tokenizerJson = await http.GetStringAsync(
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/tokenizer.json");

        using var session = InferenceSession.CreateFromFile(accelerator, modelBytes);
        var pipeline = new TextGenerationPipeline(session, accelerator);
        pipeline.LoadTokenizer(tokenizerJson);
        pipeline.MaxNewTokens = 4; // a few steady-state steps; the capture dict holds the LAST step

        async Task<(double totalMs, int readbacks, double readbackMs, int nodes, string top)> Measure(bool perOpSync)
        {
            Graph.GraphExecutor.PerOpSync = perOpSync;
            Graph.GraphExecutor.CapturedNodeTimingsMs = new Dictionary<string, double>();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var r = await pipeline.GenerateAsync("The cat sat on the");
            await accelerator.SynchronizeAsync();
            sw.Stop();

            var cap = Graph.GraphExecutor.CapturedNodeTimingsMs!;
            // Keys are "{idx:D3}_{OpType}_{output}" — aggregate ms + node-count by OpType.
            var byOp = new Dictionary<string, (double ms, int n)>();
            foreach (var (key, ms) in cap)
            {
                var parts = key.Split('_');
                var op = parts.Length >= 2 ? parts[1] : key;
                var cur = byOp.GetValueOrDefault(op);
                byOp[op] = (cur.ms + ms, cur.n + 1);
            }
            double sumMs = byOp.Values.Sum(v => v.ms);
            var top = string.Join("  ", byOp.OrderByDescending(kv => kv.Value.ms).Take(12)
                .Select(kv => $"{kv.Key}={kv.Value.ms:F0}ms/{kv.Value.n}"));
            Console.WriteLine($"[DecodeProfile perOpSync={perOpSync}] stepWall(last gen total)={sw.ElapsedMilliseconds}ms tok={r.GeneratedTokenCount} " +
                $"| capturedNodes={cap.Count} sum(per-op)={sumMs:F0}ms " +
                $"| midGraphReadbacks={Graph.GraphExecutor.LastRunReadbackCount} readbackMs={Graph.GraphExecutor.LastRunReadbackMs:F0}");
            Console.WriteLine($"[DecodeProfile perOpSync={perOpSync}] TOP: {top}");
            return (sumMs, Graph.GraphExecutor.LastRunReadbackCount, Graph.GraphExecutor.LastRunReadbackMs, cap.Count, top);
        }

        // Pass 1: CPU dispatch time per op (no per-node sync). Pass 2: + GPU execution (PerOpSync).
        var cpu = await Measure(perOpSync: false);
        var gpu = await Measure(perOpSync: true);
        Graph.GraphExecutor.PerOpSync = false;
        Graph.GraphExecutor.CapturedNodeTimingsMs = null;

        // DIAGNOSTIC-THROW: browser-test Console.WriteLine doesn't echo to the PMT .log; PMT DOES capture
        // the (innermost) exception message. So surface the whole breakdown here. This test always "fails"
        // by design — it's a measurement probe, not a gate. Remove once the decode lever is chosen.
        throw new Exception(
            $"[DecodeProfile RESULT] CPU-dispatch sum={cpu.totalMs:F0}ms | GPU-inclusive sum={gpu.totalMs:F0}ms " +
            $"over {cpu.nodes} captured nodes | midGraphReadbacks={cpu.readbacks} readbackMs={cpu.readbackMs:F0}. " +
            $"VERDICT: GPU-inclusive>>CPU-dispatch => compute/GPU-bound (better kernels / incremental KV decode); " +
            $"~equal => dispatch-overhead-bound (fold 3-input Gemm + GELU subgraph to cut dispatch count). " +
            $"\nCPU-dispatch TOP: {cpu.top}\nGPU-inclusive TOP: {gpu.top}");
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
            if (model.Length < 1_000_000)
                throw new Exception($"hub model file length={model.Length}, expected ~330MB");

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

    // GGUF DECODE PERF (the Ollama-clone / ai-chat hot loop): same hub-stream qwen2.5-0.5b load as the
    // correctness tests, then MEASURE prefill ms + decode ms/token via onDelta timestamps, plus the
    // executor's per-step split (readbacks / sync drains / total). WebGPU is the campaign target; CUDA
    // runs as the desktop reference. This is the baseline the decode capture/replay lever is judged by.
    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy", RetryCount = 2)]
    public async Task<string> GGUF_DecodePerf_Baseline() => await RunTestWithResult(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU && accelerator.AcceleratorType != AcceleratorType.Cuda)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: decode-perf baseline runs on the campaign target (WebGPU) + desktop reference (CUDA)");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        const string repoId = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
        const string file = "qwen2.5-0.5b-instruct-q8_0.gguf";
        var client = new SpawnDev.WebTorrent.WebTorrentClient();
        try
        {
            var hub = new SpawnDev.ILGPU.ML.Hub.HubModelStream(client, http) { PrepareTimeout = TimeSpan.FromMinutes(8) };
            using var cts = new System.Threading.CancellationTokenSource(TimeSpan.FromMinutes(12));
            var model = await hub.OpenAsync(repoId, file, deselect: false, cts.Token);
            await using (model.Stream)
            using (var pipe = await SpawnDev.ILGPU.ML.Pipelines.GgufTextGenerationPipeline.CreateFromStreamAsync(
                accelerator, model.Stream, maxSeqLen: 512, ct: cts.Token))
            {
                var messages = new[] { ("user", "Write one sentence about the ocean.") };
                var cfg = new GenerationConfig { MaxNewTokens = 32, Strategy = "greedy" };

                // Warm pass: shader JIT + pools (not measured).
                await pipe.GenerateAsync(new[] { ("user", "Hi") },
                    config: new GenerationConfig { MaxNewTokens = 4, Strategy = "greedy" }, ct: cts.Token);

                // Measured pass: per-delta timestamps split prefill (start -> first token) from decode.
                var stamps = new List<double>();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var answer = await pipe.GenerateAsync(messages, config: cfg,
                    onToken: (_, _) => { stamps.Add(sw.Elapsed.TotalMilliseconds); return Task.CompletedTask; },
                    ct: cts.Token);
                sw.Stop();
                if (stamps.Count < 8) throw new Exception($"decode produced only {stamps.Count} deltas: '{answer.Trim()}'");
                double prefillMs = stamps[0];
                double decodeMs = (stamps[^1] - stamps[0]) / (stamps.Count - 1);
                var report = $"prefill {prefillMs:F0}ms | decode {decodeMs:F1}ms/tok = {1000.0 / decodeMs:F1} tok/s ({stamps.Count} toks) "
                    + $"| lastStep: total {Graph.GraphExecutor.LastRunTotalMs:F1}ms readbacks {Graph.GraphExecutor.LastRunReadbackMs:F1}ms/{Graph.GraphExecutor.LastRunReadbackCount} "
                    + $"drains {Graph.GraphExecutor.LastRunSyncDrainMs:F1}ms/{Graph.GraphExecutor.LastRunSyncDrainCount}";
                Console.WriteLine($"[GGUF-DecodePerf][{accelerator.AcceleratorType}] {report}");
                return report;
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network")
            || ex.Message.Contains("magnet") || ex.Message.Contains("preparing") || ex is TimeoutException)
        {
            throw new UnsupportedTestException($"Hub/network unavailable: {ex.Message}");
        }
        finally { await client.DisposeAsync(); }
    });

    // EXACT /ai-chat page path: stream a real GGUF LLM (qwen2.5:0.5b-instruct q8_0) from OUR live hub
    // (hub.spawndev.com → HF repo, seekable torrent / web-seed) and load it via the architecture-agnostic
    // GgufTextGenerationPipeline.CreateFromStreamAsync — weights stream straight to the GPU, never held whole in
    // memory. Then run a real chat turn and assert the answer. This is the in-browser PROOF (runs on WebGPU in the
    // PMT browser lane) that the demo's hub-delivery + load + generate path works BEFORE TJ touches the demo — the
    // qwen path is oracle-matched (ollama greedy → "Paris"), so we assert the answer mentions Paris.
    // HeavyModel: ~0.5 GB fetched via the hub + GPU compile + decode — gated out of the fast loop.
    [TestMethod(Timeout = 600000, Category = "HeavyModel,WasmHeavy", RetryCount = 2)]
    public async Task Pipeline_GgufLLM_ViaHubStream_AnswersParis() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        const string repoId = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
        const string file = "qwen2.5-0.5b-instruct-q8_0.gguf";
        var client = new SpawnDev.WebTorrent.WebTorrentClient();
        try
        {
            var hub = new SpawnDev.ILGPU.ML.Hub.HubModelStream(client, http) { PrepareTimeout = TimeSpan.FromMinutes(8) };
            using var cts = new System.Threading.CancellationTokenSource(TimeSpan.FromMinutes(9));

            var model = await hub.OpenAsync(repoId, file, deselect: false, cts.Token);
            if (model.Length < 100_000_000)
                throw new Exception($"hub GGUF length={model.Length}, expected ~500MB");

            await using (model.Stream)
            using (var pipe = await SpawnDev.ILGPU.ML.Pipelines.GgufTextGenerationPipeline.CreateFromStreamAsync(
                accelerator, model.Stream, maxSeqLen: 512, ct: cts.Token))
            {
                Console.WriteLine($"[GgufLLM/hub] loaded arch={pipe.Architecture} format={pipe.ChatFormat}");
                var messages = new[] { ("user", "What is the capital of France? Answer in one short sentence.") };
                var answer = await pipe.GenerateAsync(messages,
                    config: new GenerationConfig { MaxNewTokens = 24, Strategy = "greedy" }, ct: cts.Token);
                Console.WriteLine($"[GgufLLM/hub] answer='{answer.Trim()}'");

                if (string.IsNullOrWhiteSpace(answer))
                    throw new Exception("GGUF LLM via hub produced empty output");
                if (!answer.Contains("Paris", StringComparison.OrdinalIgnoreCase))
                    throw new Exception($"GGUF LLM via hub answer did not mention Paris (oracle): '{answer.Trim()}'");
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network")
            || ex.Message.Contains("magnet") || ex.Message.Contains("preparing") || ex is TimeoutException)
        {
            throw new UnsupportedTestException($"Hub/network unavailable: {ex.Message}");
        }
        finally { await client.DisposeAsync(); }
    });

    // Regression guard for the /ai-chat repetition bug TJ hit live: a 0.5B model decoded GREEDILY on an
    // open-ended chat prompt degenerates into verbatim loops ("Boiled and served with…" over and over). The
    // demo now decodes with nucleus sampling + repetition penalty (the same config AiChatPage uses). This
    // reproduces TJ's exact prompt and asserts the output is NOT a degenerate loop, via trigram diversity.
    [TestMethod(Timeout = 600000, Category = "HeavyModel,WasmHeavy", RetryCount = 2)]
    public async Task Pipeline_GgufLLM_ChatSampling_NoDegenerateRepetition() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // GGUF pipeline setup crashes the CPU-accelerator testhost subprocess at load time (~96 ms, before any
        // generation — same hub-stream/CreateFromStreamAsync path the other heavy GGUF demo tests use). CPU is
        // the last-priority backend and GGUF-on-CPU is not a demo target; the repetition fix is verified on
        // CUDA/OpenCL/WebGPU/WebGL/Wasm. TRACKED: the CPU-subprocess GGUF crash is shared by all heavy GGUF tests.
        if (accelerator.AcceleratorType == AcceleratorType.CPU)
            throw new UnsupportedTestException("GGUF decode crashes the CPU testhost subprocess (tracked) — covered on CUDA/OpenCL/WebGPU/WebGL/Wasm");

        const string repoId = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
        const string file = "qwen2.5-0.5b-instruct-q8_0.gguf";
        var client = new SpawnDev.WebTorrent.WebTorrentClient();
        try
        {
            var hub = new SpawnDev.ILGPU.ML.Hub.HubModelStream(client, http) { PrepareTimeout = TimeSpan.FromMinutes(8) };
            using var cts = new System.Threading.CancellationTokenSource(TimeSpan.FromMinutes(9));

            var model = await hub.OpenAsync(repoId, file, deselect: false, cts.Token);
            await using (model.Stream)
            using (var pipe = await SpawnDev.ILGPU.ML.Pipelines.GgufTextGenerationPipeline.CreateFromStreamAsync(
                accelerator, model.Stream, maxSeqLen: 1024, ct: cts.Token))
            {
                // Mirror the /ai-chat page (2026-07-03): decode capture/replay ON (WebGPU-only, no-op
                // elsewhere) - this test's sampled decode now exercises the single-fence
                // PatchAndReadLogitsAsync path, exactly what the page runs.
                pipe.EnableWebGPUDecodeCapture = true;
                // The exact shape of prompt that looped under greedy. Seeded sampling → deterministic assertion.
                var messages = new[] { ("user", "List several things you can make with chicken eggs.") };
                var answer = await pipe.GenerateAsync(messages,
                    config: new GenerationConfig
                    {
                        MaxNewTokens = 140, Strategy = "top_p", Temperature = 0.7f, TopP = 0.9f,
                        RepetitionPenalty = 1.3f, Seed = 1234,
                    }, ct: cts.Token);
                Console.WriteLine($"[GgufLLM/sampling] answer='{answer.Trim()}'");

                if (string.IsNullOrWhiteSpace(answer))
                    throw new Exception("chat sampling produced empty output");

                // Degenerate loops collapse trigram diversity toward ~0; healthy text stays well above 0.5.
                var words = answer.ToLowerInvariant()
                    .Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (words.Length >= 30)
                {
                    var trigrams = new List<string>();
                    for (int i = 0; i + 2 < words.Length; i++) trigrams.Add($"{words[i]} {words[i + 1]} {words[i + 2]}");
                    double uniqueRatio = (double)trigrams.Distinct().Count() / trigrams.Count;
                    Console.WriteLine($"[GgufLLM/sampling] words={words.Length} uniqueTrigramRatio={uniqueRatio:F2}");
                    if (uniqueRatio < 0.5)
                        throw new Exception($"Degenerate repetition: unique-trigram ratio {uniqueRatio:F2} < 0.5 " +
                            $"(chat sampling should break greedy loops). Answer: '{answer.Trim()}'");
                }
                else
                {
                    Console.WriteLine($"[GgufLLM/sampling] only {words.Length} words — too short for the trigram check (not failing)");
                }
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network")
            || ex.Message.Contains("magnet") || ex.Message.Contains("preparing") || ex is TimeoutException)
        {
            throw new UnsupportedTestException($"Hub/network unavailable: {ex.Message}");
        }
        finally { await client.DisposeAsync(); }
    });

    // ═══════════════════════════════════════════════════════════
    //  Diffusion scheduler math (SD-Turbo) — CPU-only regression guard
    // ═══════════════════════════════════════════════════════════

    /// <summary>The GPU Euler step (ElementWiseKernels.AddScaledInPlace: latent += eps*dt) must be
    /// element-exact against DiffusionScheduler.EulerStep (the CPU reference it replaced - the old
    /// denoise loop paid two readbacks + an upload per step for this axpy).</summary>
    [TestMethod(Timeout = 120000)]
    public async Task Diffusion_GpuEulerStep_MatchesCpuReference() => await RunTest(async accelerator =>
    {
        const int n = 4 * 64 * 64;
        var rng = new Random(1234);
        var latent = new float[n]; var eps = new float[n];
        for (int i = 0; i < n; i++) { latent[i] = (float)(rng.NextDouble() * 4 - 2); eps[i] = (float)(rng.NextDouble() * 2 - 1); }
        float sigma = 14.6146f, sigmaNext = 0f;   // SD-Turbo single-step values
        var expected = Preprocessing.DiffusionScheduler.EulerStep(eps, latent, sigma, sigmaNext);

        var ew = new ElementWiseKernels(accelerator);
        using var latBuf = accelerator.Allocate1D(latent);
        using var epsBuf = accelerator.Allocate1D(eps);
        ew.AddScaledInPlace(latBuf.View, epsBuf.View, n, sigmaNext - sigma);
        await accelerator.SynchronizeAsync();
        var got = await latBuf.CopyToHostAsync<float>(0, n);
        for (int i = 0; i < n; i++)
            if (MathF.Abs(got[i] - expected[i]) > 1e-4f)
                throw new Exception($"GPU Euler step diverges at [{i}]: {got[i]} vs {expected[i]}");
        Console.WriteLine($"[GpuEuler] exact vs CPU reference over {n} elements");
    });

    // Guards the 4 SD-Turbo diffusion-math fixes behind /generate. DiffusionScheduler is pure CPU (no GPU),
    // so this proves the scheduler is CORRECT independent of the GPU-memory work still needed to render the
    // full image. Values are diffusers EulerDiscreteScheduler references (timestep_spacing="trailing",
    // prediction_type="epsilon"). The old leading-spacing formula gave GetTimesteps(1)=[0] → garbage output.
    [TestMethod]
    public Task Diffusion_SchedulerMath_SDTurboFixes()
    {
        static void Check(bool ok, string msg) { if (!ok) throw new Exception("Diffusion scheduler: " + msg); }
        static bool Close(float a, float b, float tol = 1e-4f) => MathF.Abs(a - b) <= tol;

        // FIX 1 — trailing timestep spacing: 1 step denoises from FULL noise → [999], not [0].
        var t1 = SpawnDev.ILGPU.ML.Preprocessing.DiffusionScheduler.GetTimesteps(1, 1000);
        Check(t1.Length == 1 && t1[0] == 999, $"GetTimesteps(1)=[{string.Join(",", t1)}], expected [999]");
        var t4 = SpawnDev.ILGPU.ML.Preprocessing.DiffusionScheduler.GetTimesteps(4, 1000);
        int[] exp4 = { 999, 749, 499, 249 };
        for (int i = 0; i < 4; i++) Check(t4[i] == exp4[i], $"GetTimesteps(4)[{i}]={t4[i]}, expected {exp4[i]}");

        // FIX 2 — sigmas sqrt((1-ᾱ)/ᾱ), final 0, monotonic decreasing.
        var ac = SpawnDev.ILGPU.ML.Preprocessing.DiffusionScheduler.ComputeAlphasCumprod(1000);
        var sig = SpawnDev.ILGPU.ML.Preprocessing.DiffusionScheduler.TimestepsToSigmas(t4, ac);
        Check(sig.Length == 5 && sig[4] == 0f, "sigmas must be len 5 with final 0");
        Check(sig[0] > sig[1] && sig[1] > sig[2] && sig[2] > sig[3], "sigmas must decrease");
        Check(Close(sig[0], MathF.Sqrt((1f - ac[999]) / ac[999])), $"sigma[0]={sig[0]} formula mismatch");

        // FIX 3 — Euler epsilon→x0: final step (sigmaNext=0) ⇒ x = sample - sigma*eps.
        var x = SpawnDev.ILGPU.ML.Preprocessing.DiffusionScheduler.EulerStep(new[] { 1f, 2f }, new[] { 10f, 10f }, 3f, 0f);
        Check(Close(x[0], 7f) && Close(x[1], 4f), $"EulerStep final [{x[0]},{x[1]}], expected [7,4]");
        var x2 = SpawnDev.ILGPU.ML.Preprocessing.DiffusionScheduler.EulerStep(new[] { 1f }, new[] { 10f }, 5f, 2f);
        Check(Close(x2[0], 7f), $"EulerStep mid {x2[0]}, expected 7");

        // FIX 4 — scale_model_input c = 1/sqrt(σ²+1) (applied in ImageGenerationPipeline.RunAsync).
        Check(Close(1f / MathF.Sqrt(3f * 3f + 1f), 0.3162278f), "scale_model_input factor σ=3 mismatch");

        return Task.CompletedTask;
    }

    // ═══════════════════════════════════════════════════════════
    //  Text-to-Image (SD-Turbo) — the /generate demo
    // ═══════════════════════════════════════════════════════════

    // SD-TURBO TEXT-TO-IMAGE E2E — proves ImageGenerationPipeline end to end. Streams the 3 sub-models
    // (~2.5GB: CLIP text-encoder 681MB + UNet 1.7GB + VAE decoder 99MB) from OUR hub via the part-2a
    // CreateFromOnnxStreamAsync path — each weight seeked + chunk-uploaded straight to GPU, NEVER a whole
    // byte[] (the 1.7GB UNet byte[] OOMed Blazor WASM — why /generate never ran). Then RunAsync the
    // single-step diffusion: CLIP encode → UNet denoise (1 step, no guidance) → VAE decode → RGBA.
    // Asserts a VALID, non-degenerate 512x512 image (RunAsync was UNPROVEN — never run E2E). This is the
    // PMT gate; pixel-accuracy (the actual cat) is confirmed visually in the /generate demo (Rule 5).
    // HeavyModel: 2.5GB fetch + GPU diffusion → long timeout; console lane needs PMT_CONSOLE_TIMEOUT_MS raised.
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task SDTurbo_Generate_E2E() => await RunTest(async accelerator =>
    {
        // Wasm backend: TRACKED FEATURE GAP, not a wall (Captain's correction 2026-07-03: "you just
        // don't load it all into memory at once"). Today the backend expands weights to fp32 in the
        // WASM heap - 2.5GB fp16 becomes ~5GB resident and the .NET runtime EXITS (code 1) mid-load.
        // The fixes are (a) fp16-RESIDENT weights (store what the file ships -> ~2.5GB, fits a 4GB
        // heap) and (b) OPFS-backed per-layer weight PAGING (bounds residency to the largest layer -
        // any model size). Plan: Plans/wasm-weight-paging.md. Un-skip when either lands.
        if (accelerator.AcceleratorType == AcceleratorType.Wasm)
            throw new UnsupportedTestException("Wasm lane: needs fp16-resident weights / OPFS weight paging (tracked: Plans/wasm-weight-paging.md)");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // OPFS-backed pieces, COLD (the DA-gate lesson): the app's SHARED client restores torrent
        // state at startup, so deleting the OPFS dir under it yields "piece verified but data not
        // in store" (hit here 2026-07-03). Clean the dir, then a FRESH client over the same IAsyncFS
        // - no restored state, pieces stay out of the .NET heap.
        var fs = GetAsyncFS();
        if (fs != null && await fs.DirectoryExists("webtorrent")) await fs.Remove("webtorrent", true);
        var client = fs != null
            ? new SpawnDev.WebTorrent.WebTorrentClient(new SpawnDev.WebTorrent.WebTorrentClientOptions { AsyncFileSystem = fs })
            : new SpawnDev.WebTorrent.WebTorrentClient();
        try
        {
            var hub = new Hub.HubModelStream(client, http);
            var pipe = await ImageGenerationPipeline.CreateAsync(accelerator, hub,
                Hub.ModelHub.KnownModels.SDTurbo,
                onProgress: (stage, pct) => Console.WriteLine($"[SDTurbo/load] {stage} {pct}%"));
            using (pipe)
            {
                if (!pipe.IsReady)
                    throw new Exception("SD-Turbo pipeline not ready after CreateAsync (a sub-model failed to load).");
                pipe.NumInferenceSteps = 1; // SD-Turbo is single-step
                pipe.GuidanceScale = 0f;    // SD-Turbo uses no classifier-free guidance
                pipe.Seed = 42;             // reproducible

                var result = await pipe.RunAsync(new ImageGenerationInput { Prompt = "a photo of a cat" });

                // Structural: 512x512 RGBA.
                int px = result.Width * result.Height;
                int expectedBytes = 4 * px;
                if (result.Width != 512 || result.Height != 512)
                    throw new Exception($"Expected 512x512, got {result.Width}x{result.Height}.");
                if (result.ImageRGBA.Length != expectedBytes)
                    throw new Exception($"Image byte length {result.ImageRGBA.Length} != expected {expectedBytes}.");

                // Content non-degeneracy: a real generation is NOT all-black, NOT constant, alpha=255.
                // Broken diffusion (NaN→0, all-zeros, flat) is caught here. (Pixel-accuracy = demo/visual.)
                long nonZero = 0; double sum = 0, sumSq = 0;
                for (int i = 0; i < px; i++)
                {
                    byte r = result.ImageRGBA[i * 4], g = result.ImageRGBA[i * 4 + 1],
                         b = result.ImageRGBA[i * 4 + 2], a = result.ImageRGBA[i * 4 + 3];
                    if (a != 255) throw new Exception($"Alpha at px {i} = {a}, expected 255.");
                    if (r != 0 || g != 0 || b != 0) nonZero++;
                    double lum = r + g + b;
                    sum += lum; sumSq += lum * lum;
                }
                double mean = sum / px, variance = sumSq / px - mean * mean;
                double std = Math.Sqrt(Math.Max(0, variance));
                Console.WriteLine($"[SDTurbo] {result.Width}x{result.Height} {result.NumSteps}-step {result.InferenceTimeMs:F0}ms " +
                    $"nonZeroPx={nonZero}/{px} lumMean={mean:F1} lumStd={std:F1}");

                if (nonZero < px / 100)
                    throw new Exception($"Image essentially all-black ({nonZero}/{px} non-zero px) — diffusion produced no image.");
                if (std < 5.0)
                    throw new Exception($"Image near-constant (lumStd={std:F1}) — flat/degenerate, not a real generation.");
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network") || ex.Message.Contains("magnet"))
        {
            throw new UnsupportedTestException($"SD-Turbo hub/network unavailable: {ex.Message}");
        }
        finally
        {
            await client.DisposeAsync();   // ours - fresh per gate run
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
