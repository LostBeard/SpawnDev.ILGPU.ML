using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public partial class MLTestBase
{
    /// <summary>
    /// LONG-CONTEXT identity + page-config perf gate (Captain's try-3 follow-up: quality concern at
    /// pastLen ~400 + 9 tok/s vs expected 25-30). (1) 200 GREEDY tokens at the page's maxSeqLen=4096,
    /// captured vs direct, char-exact - the affine patches must hold at DEPTH, not just the 48-token
    /// shallow gate (a non-affine cursor value would silently corrupt long generations exactly like a
    /// "dumb model"). (2) A 200-token SAMPLED (page config) generation timed per-token in the harness
    /// (no UI): the engine+sampler ms/tok that Captain's page wall-clock is compared against - the
    /// difference is page/UI/GC territory.
    /// </summary>
    [TestMethod(Timeout = 1200000, Category = "HeavyModel,WasmHeavy", RetryCount = 2)]
    public async Task<string> GGUF_WebGPU_DecodeCapture_LongContext()
        => await DecodeCaptureLongContextBody("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf", enableGemv: true);

    /// <summary>The same identity-at-depth gate on the LARGER arch dims (1.5B Q4_K_M): the affine
    /// patches were discovered on 0.5B; every model size the AI demo serves must be identity-proven
    /// before capture/replay is trusted there (Captain's 1.5B word-salad report - the rep-penalty
    /// window explained it, but "explains" isn't "proven" until the engine is formally exonerated).</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel,WasmHeavy", RetryCount = 2)]
    public async Task<string> GGUF_WebGPU_DecodeCapture_LongContext_Qwen15B()
        // enableGemv:false = what the AI demo actually runs today. The Q4_K GEMV is INVALID on WGSL
        // ("workgroupBarrier must only be called from uniform control flow" - the sub-block scale
        // cache barriers sit under a storage-dependent loop; Q8_0's GEMV is barrier-free in the hot
        // loop, which is why it compiles). Tracked: barrier-free Q4_K restructure or ILGPU
        // workgroupUniformLoad support, then flip this to true.
        => await DecodeCaptureLongContextBody("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf", enableGemv: false);

    private async Task<string> DecodeCaptureLongContextBody(string repoId, string file, bool enableGemv) => await RunTestWithResult(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: WebGPU long-context decode gate");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // OPFS-backed pieces, COLD: a bare in-memory client holds every fetched piece in the WASM
        // heap - a 0.5B (531MB) fits, the 1.5B (1.1GB) OOMs in WebConn's coalesced fetch. And the
        // app's SHARED client restores torrent state at startup, so deleting the OPFS dir under it
        // yields "piece verified but data not in store". The cold pattern (per WebTorrentTests):
        // clean the OPFS dir, then a FRESH client over the same IAsyncFS - no restored state, no
        // heap-resident pieces.
        var fs = GetAsyncFS();
        if (fs != null && await fs.DirectoryExists("webtorrent")) await fs.Remove("webtorrent", true);
        var client = fs != null
            ? new SpawnDev.WebTorrent.WebTorrentClient(new SpawnDev.WebTorrent.WebTorrentClientOptions { AsyncFileSystem = fs })
            : new SpawnDev.WebTorrent.WebTorrentClient();
        bool prevPrefix = GgufGenerator.EnablePrefixCache;
        bool prevGemv = FusedDequantMatMul.EnableWebGPUGemv;
        GgufGenerator.EnablePrefixCache = false;
        FusedDequantMatMul.EnableWebGPUGemv = enableGemv;
        try
        {
            var hub = new SpawnDev.ILGPU.ML.Hub.HubModelStream(client, http) { PrepareTimeout = TimeSpan.FromMinutes(8) };
            using var cts = new System.Threading.CancellationTokenSource(TimeSpan.FromMinutes(18));
            var model = await hub.OpenAsync(repoId, file, deselect: false, cts.Token);
            await using (model.Stream)
            using (var pipe = await SpawnDev.ILGPU.ML.Pipelines.GgufTextGenerationPipeline.CreateFromStreamAsync(
                accelerator, model.Stream, maxSeqLen: 4096, ct: cts.Token))   // the PAGE's maxSeqLen
            {
                var messages = new[] { ("user", "Tell a long story about a ship, its crew, and a storm. Keep going in detail.") };
                var greedyCfg = new GenerationConfig { MaxNewTokens = 200, Strategy = "greedy" };

                var direct = await pipe.GenerateAsync(messages, config: greedyCfg, ct: cts.Token);
                pipe.EnableWebGPUDecodeCapture = true;
                var captured = await pipe.GenerateAsync(messages, config: greedyCfg, ct: cts.Token);
                if (captured != direct)
                {
                    // Name the divergence point (first differing char) - depth tells which pastLen broke.
                    int at = 0; while (at < Math.Min(direct.Length, captured.Length) && direct[at] == captured[at]) at++;
                    throw new Exception($"LONG-CONTEXT DIVERGENCE at char {at}/{direct.Length}: "
                        + $"direct '...{direct.Substring(Math.Max(0, at - 40), Math.Min(80, direct.Length - Math.Max(0, at - 40)))}' vs "
                        + $"captured '...{captured.Substring(Math.Max(0, at - 40), Math.Min(80, captured.Length - Math.Max(0, at - 40)))}'");
                }

                // Page-config sampled perf, 200 tokens, no UI: the honest engine+sampler ms/tok.
                var stamps = new List<double>();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                await pipe.GenerateAsync(new[] { ("user", "Describe four seasons in a small village, at length.") },
                    config: new GenerationConfig
                    {
                        MaxNewTokens = 200, Strategy = "top_p", Temperature = 0.7f, TopP = 0.9f,
                        RepetitionPenalty = 1.3f, Seed = 42,
                    },
                    onToken: (_, _) => { stamps.Add(sw.Elapsed.TotalMilliseconds); return Task.CompletedTask; },
                    ct: cts.Token);
                double sampledMs = stamps.Count >= 20 ? (stamps[^1] - stamps[0]) / (stamps.Count - 1) : double.NaN;
                // Median of the LAST 50 inter-token gaps = the deep-context steady state.
                var gaps = new List<double>();
                for (int i = Math.Max(1, stamps.Count - 50); i < stamps.Count; i++) gaps.Add(stamps[i] - stamps[i - 1]);
                gaps.Sort();
                double deepMs = gaps.Count > 0 ? gaps[gaps.Count / 2] : double.NaN;

                var report = $"LONG-CONTEXT IDENTITY OK ({direct.Length} chars, 200 toks greedy @ maxSeq 4096) "
                    + $"| sampled(page cfg) {sampledMs:F1}ms/tok avg = {1000.0 / sampledMs:F1} tok/s; deep-context median {deepMs:F1}ms/tok "
                    + $"| step split: patch {pipe.LastDecodeCaptureStepMs?.Patch:F1} + plan {pipe.LastDecodeCaptureStepMs?.Replay:F1} + fence {pipe.LastDecodeCaptureStepMs?.Sync:F1}";
                Console.WriteLine($"[GGUF-LongContext] {report}");
                return report;
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network")
            || ex.Message.Contains("magnet") || ex.Message.Contains("preparing") || ex is TimeoutException)
        {
            throw new UnsupportedTestException($"Hub/network unavailable: {ex.Message}");
        }
        finally
        {
            GgufGenerator.EnablePrefixCache = prevPrefix;
            FusedDequantMatMul.EnableWebGPUGemv = prevGemv;
            await client.DisposeAsync();   // ours - fresh per gate run
        }
    });

    /// <summary>
    /// STAGE-2 GATE: token-identity of the full patched-replay decode loop. Same pipeline, same
    /// prompt, greedy: run A direct (capture OFF), run B with EnableWebGPUDecodeCapture (first decode
    /// step captures + probes, the rest are patched single-crossing replays). Prefix cache is
    /// disabled so run B re-prefills identically - the outputs must be TOKEN-IDENTICAL. Run C (a
    /// second captured turn, replay-only) is timed per-token: the production number vs the 686ms/tok
    /// direct baseline and the 20.8ms/tok same-state floor.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy", RetryCount = 2)]
    public async Task<string> GGUF_WebGPU_DecodeCapture_TokenIdentical() => await RunTestWithResult(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: WebGPU decode-capture gate");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        const string repoId = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
        const string file = "qwen2.5-0.5b-instruct-q8_0.gguf";
        var client = new SpawnDev.WebTorrent.WebTorrentClient();
        bool prevPrefix = GgufGenerator.EnablePrefixCache;
        bool prevGemv = FusedDequantMatMul.EnableWebGPUGemv;
        GgufGenerator.EnablePrefixCache = false;   // force identical full prefills for A and B
        // Hardware re-measurement of the cooperative GEMV (the 2026-06-13 "75x slower" exclusion was
        // SwiftShader-era, voided): both the direct and captured runs use it; token-identity still
        // gates correctness, and the replay-turn timing is the A/B vs the per-element 44.8ms/tok.
        FusedDequantMatMul.EnableWebGPUGemv = true;
        try
        {
            var hub = new SpawnDev.ILGPU.ML.Hub.HubModelStream(client, http) { PrepareTimeout = TimeSpan.FromMinutes(8) };
            using var cts = new System.Threading.CancellationTokenSource(TimeSpan.FromMinutes(12));
            var model = await hub.OpenAsync(repoId, file, deselect: false, cts.Token);
            await using (model.Stream)
            using (var pipe = await SpawnDev.ILGPU.ML.Pipelines.GgufTextGenerationPipeline.CreateFromStreamAsync(
                accelerator, model.Stream, maxSeqLen: 512, ct: cts.Token))
            {
                var messages = new[] { ("user", "Describe a lighthouse in two sentences.") };
                var cfg = new GenerationConfig { MaxNewTokens = 48, Strategy = "greedy" };

                var direct = await pipe.GenerateAsync(messages, config: cfg, ct: cts.Token);

                pipe.EnableWebGPUDecodeCapture = true;
                var captured = await pipe.GenerateAsync(messages, config: cfg, ct: cts.Token);
                var info = pipe.DecodeCaptureInfo;

                if (captured != direct)
                    throw new Exception($"TOKEN DIVERGENCE - direct: '{direct.Trim()}' vs captured: '{captured.Trim()}' "
                        + $"(capture: {info})");

                // Run C: replay-only turn (capture already built) - the production per-token number.
                var stamps = new List<double>();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                await pipe.GenerateAsync(new[] { ("user", "Name three colors and say why you like each.") },
                    config: new GenerationConfig { MaxNewTokens = 32, Strategy = "greedy" },
                    onToken: (_, _) => { stamps.Add(sw.Elapsed.TotalMilliseconds); return Task.CompletedTask; },
                    ct: cts.Token);
                double decodeMs = stamps.Count >= 8 ? (stamps[^1] - stamps[0]) / (stamps.Count - 1) : double.NaN;
                var split = pipe.LastDecodeCaptureStepMs;

                var report = $"TOKEN-IDENTICAL ({direct.Trim().Length} chars) | capture: ops={info?.Ops} patches: scalars={info?.Scalars} copies={info?.Copies} slots={info?.Slots} "
                    + $"| replay-turn decode {decodeMs:F1}ms/tok = {1000.0 / decodeMs:F1} tok/s (direct baseline 686ms/tok = 1.5 tok/s) "
                    + $"| lastStep split: patch {split?.Patch:F1}ms + planCall {split?.Replay:F1}ms + gpuWait {split?.Sync:F1}ms (residual = argmax/detok/loop) "
                    + $"| patch split: input {pipe.LastDecodeCapturePatchSplitMs?.Input:F1} scalars {pipe.LastDecodeCapturePatchSplitMs?.Scalars:F1} slots {pipe.LastDecodeCapturePatchSplitMs?.Slots:F1} copies {pipe.LastDecodeCapturePatchSplitMs?.Copies:F1}";
                Console.WriteLine($"[GGUF-DecodeCapture] {report}");
                return report;
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network")
            || ex.Message.Contains("magnet") || ex.Message.Contains("preparing") || ex is TimeoutException)
        {
            throw new UnsupportedTestException($"Hub/network unavailable: {ex.Message}");
        }
        finally { GgufGenerator.EnablePrefixCache = prevPrefix; FusedDequantMatMul.EnableWebGPUGemv = prevGemv; await client.DisposeAsync(); }
    });

    /// <summary>
    /// STAGE-1 WebGPU decode dispatch-plan capture/replay probe - the browser twin of Example 04's
    /// CUDA graph probe, against the measured 1.5 tok/s baseline (686ms/tok of which ~570ms is
    /// per-node dispatch orchestration; CUDA does 11.4ms/tok on the same card). SAME-STATE replay:
    /// freeze the KV cursor, capture ONE decode forward as a WebGPU dispatch plan, replay it -
    /// correctness gate = replay argmax equals a fresh non-captured forward at the exact state; the
    /// timed replays measure the true per-token GPU floor (what per-token param patching, stage 2,
    /// can reach). Also runs ReplayTimedAsync for the per-kernel attribution of a decode step.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy", RetryCount = 2)]
    public async Task<string> GGUF_WebGPU_DecodePlanReplay_Probe() => await RunTestWithResult(async accelerator =>
    {
        if (accelerator is not WebGPUAccelerator webGpu)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: WebGPU decode plan-replay probe (CUDA has its own graph probe in Example 04)");
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
                // Warm the decode state: a short greedy generation advances the KV cursor + JIT + pools.
                await pipe.GenerateAsync(new[] { ("user", "Write one sentence about the ocean.") },
                    config: new GenerationConfig { MaxNewTokens = 8, Strategy = "greedy" }, ct: cts.Token);

                var session = pipe.Session;
                int statePast = session.DecodePastLen;
                if (statePast < 8) throw new Exception($"expected a warm decode state, DecodePastLen={statePast}");
                const int nextTok = 785; // arbitrary valid token id - same-state probe, value is irrelevant

                // Stable single-token input buffer - the captured embedding gather bakes THIS buffer.
                using var capIn = accelerator.Allocate1D(new[] { (float)nextTok });
                var capInput = new Dictionary<string, Tensor>
                { ["input_ids"] = new Tensor(capIn.View, new[] { 1, 1 }, "input_ids") };

                async Task<(int arg, float[] logits)> RunDirectAsync()
                {
                    session.SetGGUFDecodePastLen(statePast);
                    var o = await session.RunDecodeStepAsync(capInput);
                    var t = o.TryGetValue("logits", out var l) ? l : o.Values.First();
                    int vocab = t.Shape[^1]; int so = t.ElementCount / vocab;
                    using var rd = accelerator.Allocate1D<float>(vocab);
                    await rd.View.CopyFromAsync(t.Data.SubView((long)(so - 1) * vocab, vocab));
                    await accelerator.SynchronizeAsync();
                    var v = await rd.CopyToHostAsync<float>(0, vocab);
                    int a = 0; for (int i = 1; i < vocab; i++) if (v[i] > v[a]) a = i;
                    return (a, v);
                }

                // Reference: fresh non-captured forward at the frozen state.
                var (refArg, refLogits) = await RunDirectAsync();

                // Capture regime (same as WebGPUGraphCapture / the CUDA probe): stable param slots,
                // fold ON / validate OFF, elide ON (post-fix), bind-group caching enforced off by
                // BeginDispatchCapture. Warm A (drains on) + warm B (drains suppressed) then capture.
                bool prevFold = GraphCompiler.ShapeSubgraphFoldEnabled;
                bool prevElide = GraphExecutor.ShapeInterpElideDispatch;
                bool prevValidate = GraphExecutor.ShapeInterpValidate;
                GraphCompiler.ShapeSubgraphFoldEnabled = true;
                GraphExecutor.ShapeInterpElideDispatch = true;
                GraphExecutor.ShapeInterpValidate = false;
                FusedAttentionKernel.UseStableCaptureSlots = true;
                GraphExecutor.UseCaptureParamSlots = true;
                try
                {
                    session.SetGGUFDecodePastLen(statePast);
                    await session.RunDecodeStepAsync(capInput);
                    await accelerator.SynchronizeAsync();

                    session.SetGGUFDecodePastLen(statePast);
                    GraphExecutor.SuppressDrains = true;
                    await session.RunDecodeStepAsync(capInput);
                    GraphExecutor.SuppressDrains = false;
                    await accelerator.SynchronizeAsync();

                    Dictionary<string, Tensor> capOut;
                    session.SetGGUFDecodePastLen(statePast);
                    GraphExecutor.SuppressDrains = true;
                    var plan = webGpu.BeginDispatchCapture();
                    try { capOut = await session.RunDecodeStepAsync(capInput); }
                    catch { webGpu.EndDispatchCapture().Dispose(); throw; }
                    finally { GraphExecutor.SuppressDrains = false; }
                    webGpu.EndDispatchCapture();
                    await accelerator.SynchronizeAsync();

                    using (plan)
                    {
                        var logitsT = capOut.TryGetValue("logits", out var cl) ? cl : capOut.Values.First();
                        int vocab = logitsT.Shape[^1]; int so = logitsT.ElementCount / vocab;

                        async Task<(int arg, float maxDiff)> ReadReplayAsync()
                        {
                            using var rd = accelerator.Allocate1D<float>(vocab);
                            await rd.View.CopyFromAsync(logitsT.Data.SubView((long)(so - 1) * vocab, vocab));
                            await accelerator.SynchronizeAsync();
                            var v = await rd.CopyToHostAsync<float>(0, vocab);
                            int a = 0; float md = 0f;
                            for (int i = 0; i < vocab; i++) { if (v[i] > v[a]) a = i; float d = MathF.Abs(v[i] - refLogits[i]); if (d > md) md = d; }
                            return (a, md);
                        }

                        // Correctness: one replay -> logits must match the non-captured reference.
                        await plan.ReplayAsync();
                        await accelerator.SynchronizeAsync();
                        var (repArg, maxDiff) = await ReadReplayAsync();
                        if (repArg != refArg)
                            throw new Exception($"replay argmax {repArg} != direct {refArg} (max|dLogit|={maxDiff:E3}, ops={plan.DispatchCount}) - same-state decode replay diverged");

                        // Timed replays: the per-token GPU floor (stage-2 target).
                        const int R = 20;
                        var rsw = System.Diagnostics.Stopwatch.StartNew();
                        for (int r = 0; r < R; r++) { await plan.ReplayAsync(); await accelerator.SynchronizeAsync(); }
                        rsw.Stop();
                        double replayMs = rsw.Elapsed.TotalMilliseconds / R;

                        // Per-kernel attribution of the decode step (chunked - long lines truncate).
                        var attribution = await plan.ReplayTimedAsync();
                        for (int ofs = 0; ofs < attribution.Length; ofs += 800)
                            Console.WriteLine($"[GGUF-PlanReplay][Attr {ofs / 800}] {attribution.Substring(ofs, Math.Min(800, attribution.Length - ofs))}");

                        string top = "";
                        try
                        {
                            using var doc = System.Text.Json.JsonDocument.Parse(attribution);
                            if (doc.RootElement.TryGetProperty("kernels", out var ks))
                                top = " | topGPU: " + string.Join(", ", ks.EnumerateArray().Take(5)
                                    .Select(k => $"{k.GetProperty("label").GetString()}={k.GetProperty("ms").GetDouble():F1}ms x{k.GetProperty("count").GetInt32()}"));
                        }
                        catch { }

                        var report = $"pastLen={statePast} ops={plan.DispatchCount} | direct(baseline test)=686ms/tok -> SAME-STATE replay {replayMs:F1}ms/tok "
                            + $"| argmax MATCH max|dLogit|={maxDiff:E2}{top}";
                        Console.WriteLine($"[GGUF-PlanReplay] {report}");
                        return report;
                    }
                }
                finally
                {
                    FusedAttentionKernel.UseStableCaptureSlots = false;
                    GraphExecutor.UseCaptureParamSlots = false;
                    GraphExecutor.SuppressDrains = false;
                    GraphCompiler.ShapeSubgraphFoldEnabled = prevFold;
                    GraphExecutor.ShapeInterpElideDispatch = prevElide;
                    GraphExecutor.ShapeInterpValidate = prevValidate;
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
}
