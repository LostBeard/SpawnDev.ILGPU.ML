using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Backend;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// SD-Turbo WebGPU PERF PROFILE (Tuvok 2026-07-09). One image gen with WebGPUBackend.EnableDispatchProfiling
/// on, reading the reliable per-surface split of a step's wall-time (the slices must SUM toward the total):
///   - ProfileSyncWaitMs   = time BLOCKED in queue.OnSubmittedWorkDone() drains  ≈ GPU EXECUTION of batched work
///   - ProfileReadbackWaitMs = time BLOCKED in MapAsync(Read) GPU->CPU readbacks
///   - ProfileCpu*Ms       = .NET-side per-dispatch prologue (shader-resolve / arg-build / bind-group / encode)
/// This settles where SD-Turbo's ~224s WebGPU gen goes: GPU kernel execution (sync-wait -> lever = faster
/// kernels / the f16 read-only-weight conv fix + fusion) vs readback (lever = the warm shape-readback cache)
/// vs CPU dispatch (measured ~1ms/dispatch = ~8s for 6428 nodes, already ruled out as dominant). NO PerOpSync,
/// so the gen runs at normal speed and the split reflects the real pipeline. WebGPU-only; returns the report.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> SDTurbo_WebGPU_Profile() => await RunTestWithResult(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: WebGPU-only perf profile (the dispatch-profiling counters only accumulate on the WebGPU path).");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var fs = GetAsyncFS();
        // Wipe the torrent dir for a CLEAN download (a stale/partial torrent gives "resolved metadata but
        // exposes no files"). The profiling window is RunAsync only (download/load lives in CreateAsync),
        // so a cold download doesn't pollute the gen split.
        if (fs != null && await fs.DirectoryExists("webtorrent")) await fs.Remove("webtorrent", true);
        var client = fs != null
            ? new SpawnDev.WebTorrent.WebTorrentClient(new SpawnDev.WebTorrent.WebTorrentClientOptions { AsyncFileSystem = fs })
            : new SpawnDev.WebTorrent.WebTorrentClient();
        try
        {
            var hub = new Hub.HubModelStream(client, http);
            var pipe = await ImageGenerationPipeline.CreateAsync(accelerator, hub, Hub.ModelHub.KnownModels.SDTurbo,
                onProgress: (stage, pct) => Console.WriteLine($"[SDTurbo/load] {stage} {pct}%"));
            using (pipe)
            {
                if (!pipe.IsReady) throw new Exception("SD-Turbo pipeline not ready after CreateAsync.");
                pipe.NumInferenceSteps = 1;
                pipe.GuidanceScale = 0f;
                pipe.Seed = 42;

                // NOTE: profiling the FIRST gen (no warm-up) to fit the harness window — so ShaderResolve
                // includes one-time shader COMPILE (a ~fixed cost, not per-gen). syncWait (GPU-exec) and
                // readbackWait are the split we care about and are compile-independent.
                WebGPUBackend.EnableDispatchProfiling = true;
                WebGPUBackend.ResetDispatchProfiling();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var result = await pipe.RunAsync(new ImageGenerationInput { Prompt = "a lighthouse in a storm" });
                sw.Stop();

                double sync = WebGPUBackend.ProfileSyncWaitMs; long syncN = WebGPUBackend.ProfileSyncWaitCount;
                double rb = WebGPUBackend.ProfileReadbackWaitMs; long rbN = WebGPUBackend.ProfileReadbackWaitCount;
                double cSR = WebGPUBackend.ProfileCpuShaderResolveMs, cAB = WebGPUBackend.ProfileCpuArgBuildMs,
                       cBG = WebGPUBackend.ProfileCpuBindGroupMs, cEN = WebGPUBackend.ProfileCpuEncodeMs;
                long cN = WebGPUBackend.ProfileCpuDispatchCount;
                double cpu = cSR + cAB + cBG + cEN;
                double accounted = sync + rb + cpu;
                double gen = sw.Elapsed.TotalMilliseconds;

                var report =
                    $"SDTurbo WebGPU profiled gen: wall={gen:F0}ms (InferenceTimeMs={result.InferenceTimeMs:F0}) | " +
                    $"syncWait(GPU-exec)={sync:F0}ms/{syncN} ({(gen > 0 ? 100 * sync / gen : 0):F0}%) | " +
                    $"readbackWait={rb:F0}ms/{rbN} ({(gen > 0 ? 100 * rb / gen : 0):F0}%) | " +
                    $"cpuPrologue={cpu:F0}ms/{cN} ({(gen > 0 ? 100 * cpu / gen : 0):F0}%) [SR{cSR:F0} AB{cAB:F0} BG{cBG:F0} EN{cEN:F0}] | " +
                    $"accounted={accounted:F0}ms ({(gen > 0 ? 100 * accounted / gen : 0):F0}% of wall; remainder = .NET compute / image encode / unmeasured)";
                // JS.LogError writes console.error (PMT captures the text) WITHOUT tripping #blazor-error-ui,
                // so the test stays green (Console.Error would redden it). Normal Console.WriteLine isn't
                // surfaced in PMT's summarized console output, hence the explicit LogError.
                SpawnDev.SpawnJS.SpawnJSRuntime.Instance?.LogError("[SDTURBO-PROFILE] " + report);
                return report;
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network") || ex.Message.Contains("magnet"))
        {
            throw new UnsupportedTestException($"SD-Turbo hub/network unavailable: {ex.Message}");
        }
        finally
        {
            WebGPUBackend.EnableDispatchProfiling = false;
            WebGPUBackend.ResetDispatchProfiling();
            await client.DisposeAsync();
        }
    });

    /// <summary>
    /// SD-Turbo WebGPU PER-OP cost (Tuvok 2026-07-10). TRUE per-op GPU attribution via PerOpSync (a drain
    /// after each node), aggregated by op-type (ms / count / avg), reported via JS.LogError so PMT captures
    /// it. Diff this against the CUDA per-op run (Examples/03 SDTURBO_PEROP=1, same aggregation) to find the
    /// op-types where WGSL codegen/exec is worst vs PTX - the "1000 lines of WGSL for 100 of PTX" suspects.
    /// SLOW (a sync per node x ~6428 nodes) - runs one gen. WebGPU-only.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> SDTurbo_WebGPU_PerOp() => await RunTestWithResult(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: WebGPU-only per-op profile");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var fs = GetAsyncFS();
        if (fs != null && await fs.DirectoryExists("webtorrent")) await fs.Remove("webtorrent", true);
        var client = fs != null
            ? new SpawnDev.WebTorrent.WebTorrentClient(new SpawnDev.WebTorrent.WebTorrentClientOptions { AsyncFileSystem = fs })
            : new SpawnDev.WebTorrent.WebTorrentClient();
        try
        {
            var hub = new Hub.HubModelStream(client, http);
            var pipe = await ImageGenerationPipeline.CreateAsync(accelerator, hub, Hub.ModelHub.KnownModels.SDTurbo,
                onProgress: (stage, pct) => Console.WriteLine($"[SDTurbo/load] {stage} {pct}%"));
            using (pipe)
            {
                if (!pipe.IsReady) throw new Exception("SD-Turbo pipeline not ready after CreateAsync.");
                pipe.NumInferenceSteps = 1;
                pipe.GuidanceScale = 0f;
                pipe.Seed = 42;

                SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs = new();
                SpawnDev.ILGPU.ML.Graph.GraphExecutor.PerOpSync = true; // true per-op GPU attribution (a drain/node)
                var result = await pipe.RunAsync(new ImageGenerationInput { Prompt = "a lighthouse in a storm" });
                var pt = SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs;
                SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs = null;
                SpawnDev.ILGPU.ML.Graph.GraphExecutor.PerOpSync = false;

                // Aggregate by TRUE op-type (no LINQ - WASM). "NNN_OpType_/path" -> "OpType".
                var byType = new Dictionary<string, double>();
                var cntType = new Dictionary<string, int>();
                double tot = 0;
                foreach (var kv in pt!)
                {
                    var p = kv.Key.Split('_');
                    var key = p.Length > 1 ? p[1] : kv.Key;
                    byType[key] = byType.TryGetValue(key, out var v) ? v + kv.Value : kv.Value;
                    cntType[key] = cntType.TryGetValue(key, out var c) ? c + 1 : 1;
                    tot += kv.Value;
                }
                var ordered = new List<KeyValuePair<string, double>>(byType);
                ordered.Sort((a, b) => b.Value.CompareTo(a.Value));
                var sb = new System.Text.StringBuilder();
                sb.Append($"[SDTURBO-PEROP-WEBGPU] total={tot:F0}ms/{pt.Count}nodes wall={result.InferenceTimeMs:F0}ms (PerOpSync). By op-type ms/count/avg: ");
                for (int i = 0; i < Math.Min(22, ordered.Count); i++)
                {
                    var k = ordered[i].Key;
                    sb.Append($"{k}={ordered[i].Value:F0}ms/{cntType[k]}n/{ordered[i].Value / cntType[k]:F2}avg({100 * ordered[i].Value / tot:F1}%); ");
                }
                var report = sb.ToString();
                SpawnDev.SpawnJS.SpawnJSRuntime.Instance.LogError(report);
                return report;
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network") || ex.Message.Contains("magnet"))
        {
            throw new UnsupportedTestException($"SD-Turbo hub/network unavailable: {ex.Message}");
        }
        finally
        {
            SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs = null;
            SpawnDev.ILGPU.ML.Graph.GraphExecutor.PerOpSync = false;
            await client.DisposeAsync();
        }
    });

    /// <summary>
    /// SD-Turbo WebGPU DISPATCH-ELIDE A/B (Tuvok 2026-07-10). The per-op diff showed ~5000 of 6428 nodes are
    /// trivial shape/constant ops (Unsqueeze/Gather/Concat/Add/Div/Cast) that cost ~free on CUDA but ~4ms+
    /// each on WebGPU (34-99x), and dominate the per-node orchestration. Dispatch-elide CPU-resolves those so
    /// they never dispatch - it's ON in the capture path (proven bit-identical) but OFF in the normal gen.
    /// This A/B measures the normal gen with elide OFF vs ON on the SAME resident model + seed: wall time,
    /// dispatch count, and an A-vs-B pixel diff (must be ~0 - elide is a pure orchestration no-op). WebGPU-only.
    /// </summary>
    [TestMethod(Timeout = 2400000, Category = "HeavyModel")]
    public async Task<string> SDTurbo_WebGPU_ElideAB() => await RunTestWithResult(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: WebGPU-only elide A/B");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var fs = GetAsyncFS();
        if (fs != null && await fs.DirectoryExists("webtorrent")) await fs.Remove("webtorrent", true);
        var client = fs != null
            ? new SpawnDev.WebTorrent.WebTorrentClient(new SpawnDev.WebTorrent.WebTorrentClientOptions { AsyncFileSystem = fs })
            : new SpawnDev.WebTorrent.WebTorrentClient();
        bool prevElide = SpawnDev.ILGPU.ML.Graph.GraphExecutor.ShapeInterpElideDispatch;
        bool prevFold = SpawnDev.ILGPU.ML.Graph.GraphCompiler.ShapeSubgraphFoldEnabled;
        try
        {
            var hub = new Hub.HubModelStream(client, http);
            var pipe = await ImageGenerationPipeline.CreateAsync(accelerator, hub, Hub.ModelHub.KnownModels.SDTurbo,
                onProgress: (stage, pct) => Console.WriteLine($"[SDTurbo/load] {stage} {pct}%"));
            using (pipe)
            {
                if (!pipe.IsReady) throw new Exception("SD-Turbo pipeline not ready after CreateAsync.");
                pipe.NumInferenceSteps = 1; pipe.GuidanceScale = 0f; pipe.Seed = 42;
                const string prompt = "a lighthouse in a storm";

                // A: elide OFF (current normal-gen default)
                SpawnDev.ILGPU.ML.Graph.GraphExecutor.ShapeInterpElideDispatch = false;
                WebGPUBackend.EnableDispatchProfiling = true; WebGPUBackend.ResetDispatchProfiling();
                var swA = System.Diagnostics.Stopwatch.StartNew();
                var rA = await pipe.RunAsync(new ImageGenerationInput { Prompt = prompt });
                swA.Stop(); long dispA = WebGPUBackend.ProfileCpuDispatchCount;

                // B: elide ON (CPU-resolve shape ops, don't dispatch them)
                SpawnDev.ILGPU.ML.Graph.GraphExecutor.ShapeInterpElideDispatch = true;
                SpawnDev.ILGPU.ML.Graph.GraphCompiler.ShapeSubgraphFoldEnabled = true;
                WebGPUBackend.ResetDispatchProfiling();
                var swB = System.Diagnostics.Stopwatch.StartNew();
                var rB = await pipe.RunAsync(new ImageGenerationInput { Prompt = prompt });
                swB.Stop(); long dispB = WebGPUBackend.ProfileCpuDispatchCount;

                // Validate B image + pixel-diff A vs B (same seed -> elide must be a pure no-op = ~0 diff).
                int px = rB.Width * rB.Height; long nz = 0; double sum = 0, sumSq = 0, diffPx = 0;
                for (int i = 0; i < px; i++)
                {
                    byte r = rB.ImageRGBA[i * 4], g = rB.ImageRGBA[i * 4 + 1], b = rB.ImageRGBA[i * 4 + 2];
                    if (r != 0 || g != 0 || b != 0) nz++;
                    double lum = r + g + b; sum += lum; sumSq += lum * lum;
                    if (i < rA.ImageRGBA.Length / 4 && (rA.ImageRGBA[i * 4] != r || rA.ImageRGBA[i * 4 + 1] != g || rA.ImageRGBA[i * 4 + 2] != b)) diffPx++;
                }
                double mean = sum / px, std = Math.Sqrt(Math.Max(0, sumSq / px - mean * mean));
                double aMs = swA.Elapsed.TotalMilliseconds, bMs = swB.Elapsed.TotalMilliseconds;
                var report = $"[SDTURBO-ELIDE-AB] elideOFF={aMs:F0}ms/{dispA}disp -> elideON={bMs:F0}ms/{dispB}disp = {(bMs > 0 ? aMs / bMs : 0):F2}x faster ({dispA - dispB} fewer dispatches); elideON image std={std:F1} nz={nz}/{px}; A-vs-B pixel-diff={diffPx:F0}/{px} ({(px > 0 ? 100.0 * diffPx / px : 0):F2}%)";
                SpawnDev.SpawnJS.SpawnJSRuntime.Instance?.LogError(report);
                if (std < 5.0) throw new Exception($"elide-ON image degenerate (std={std:F1}) - elide broke SD-Turbo. {report}");
                // Elide is a PURE orchestration no-op (CPU-resolve shape ops instead of dispatching them) - it must
                // NOT change any FEATURE math, so the elide-ON image must be bit-identical to elide-OFF. The CLIP
                // Range keystone (SatFloatToInt in TryComputeShapeOnCpu) was proven here at exactly 0.00% diff;
                // allow a tiny margin only against non-determinism. A regression that reintroduces the empty-shape
                // bug (or any elide divergence) shows up as a nonzero diff here.
                if (diffPx > px * 0.005)
                    throw new Exception($"elide-ON diverged from elide-OFF by {diffPx:F0}/{px} pixels ({100.0 * diffPx / px:F2}%) - elide must be bit-exact. {report}");
                return report;
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network") || ex.Message.Contains("magnet"))
        {
            throw new UnsupportedTestException($"SD-Turbo hub/network unavailable: {ex.Message}");
        }
        finally
        {
            SpawnDev.ILGPU.ML.Graph.GraphExecutor.ShapeInterpElideDispatch = prevElide;
            SpawnDev.ILGPU.ML.Graph.GraphCompiler.ShapeSubgraphFoldEnabled = prevFold;
            WebGPUBackend.EnableDispatchProfiling = false; WebGPUBackend.ResetDispatchProfiling();
            await client.DisposeAsync();
        }
    });

    // NOTE (Tuvok 2026-07-11): the one-shot SDTurbo_WebGPU_ElideEmptyDiag scaffold (used to capture the
    // empty-shape-interp log that located the keystone - node 25 Slice INT64_MAX-sentinel overflow) was
    // removed once SDTurbo_WebGPU_ElideAB proved elide bit-exact (0.00% diff). The gated, zero-cost
    // GraphExecutor.LogEmptyShapeInterp flag it drove REMAINS for future manual diagnosis.
}
