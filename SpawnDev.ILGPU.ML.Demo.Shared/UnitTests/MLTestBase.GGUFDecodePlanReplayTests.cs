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
