using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.WebGPU.Backend;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// MEASUREMENT (not a gate): where does a WARM, UNCAPTURED Depth Anything forward spend its time on WebGPU?
    /// Graph capture is OFF - the plain forward is what every new input shape pays, and TJ's bar (2026-09-30) is that
    /// it alone must beat Transformers.js (tools/dav3/dav3-tjs.mjs: DAv3 518 warm = 61 ms, system Chrome, RTX 4070).
    /// Prints one [DirectForwardProfile] line per warm run: wall, executor total, dispatches, and the per-dispatch CPU
    /// split from WebGPUBackend.EnableDispatchProfiling (shader resolve / arg build / bind group (raw create) / encode),
    /// plus GPU sync-wait and readback. Run scoped:
    ///   PMT_EXCLUDE_CATEGORIES=__none__ PMT_FILTER=DirectForwardProfile PMT_LANES=WebGPUTests PMT_CONSOLE_LOG=DirectForwardProfile
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task DirectForwardProfile_DAv3Small_518() => await DirectForwardProfile("onnx-community/depth-anything-v3-small",
        "onnx/model.onnx_data", new[] { 1, 1, 3, 518, 518 }, 518);

    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task DirectForwardProfile_DAv2Small_518() => await DirectForwardProfile("onnx-community/depth-anything-v2-small",
        "", new[] { 1, 3, 518, 518 }, 518);

    async Task DirectForwardProfile(string repo, string externalDataFile, int[] inputShape, int size) => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: the browser WebGPU direct forward is what is measured");
        var js = SpawnDev.SpawnJS.SpawnJSRuntime.Instance;
        var source = new HubModelSource(js, GetHttpClient());
        using var pipeline = await Pipelines.DepthEstimationPipeline.CreateFromHubAsync(accelerator, source, repo,
            externalDataFile: externalDataFile, inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = inputShape });
        pipeline.EnableGraphCapture = false;
        var rgba = new int[size * size];
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++) { int v = (x * 255 / (size - 1)) ^ (y & 0x3F); rgba[y * size + x] = (255 << 24) | (v << 16) | ((255 - v) << 8) | v; }

        // cold: shader compiles + first allocations (reported, not the target)
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using (var cold = (await pipeline.EstimateGpuRawAsync(rgba, size, size)).RawDepth) { }
        await accelerator.SynchronizeAsync();
        Console.WriteLine($"[DirectForwardProfile] {repo} {size} COLD fold=[{Graph.GraphExecutor.LastRunFoldState}]");
        Console.WriteLine($"[DirectForwardProfile] {repo} {size} COLD wall={sw.Elapsed.TotalMilliseconds:F0}ms nodes={pipeline.Session.NodeCount}");

        bool prevProf = WebGPUBackend.EnableDispatchProfiling;
        WebGPUBackend.EnableDispatchProfiling = true;
        try
        {
            for (int run = 1; run <= 3; run++)
            {
                WebGPUBackend.ResetDispatchProfiling();
                var wgAcc = (SpawnDev.ILGPU.WebGPU.WebGPUAccelerator)accelerator;
                long resolveHits0 = wgAcc.ShaderResolveCacheHits, resolveMisses0 = wgAcc.ShaderResolveCacheMisses;
                int gc0 = GC.CollectionCount(0), gc1 = GC.CollectionCount(1), gc2 = GC.CollectionCount(2);
                long alloc0 = GC.GetTotalAllocatedBytes(false);
                if (run == 3) Graph.GraphExecutor.OpProfile = new();
                sw.Restart();
                var r = await pipeline.EstimateGpuRawAsync(rgba, size, size);
                await accelerator.SynchronizeAsync();
                double wall = sw.Elapsed.TotalMilliseconds;
                r.RawDepth.Dispose();
                long n = WebGPUBackend.ProfileCpuDispatchCount;
                double cpu = WebGPUBackend.ProfileCpuShaderResolveMs + WebGPUBackend.ProfileCpuArgBuildMs + WebGPUBackend.ProfileCpuBindGroupMs + WebGPUBackend.ProfileCpuEncodeMs;
                Console.WriteLine($"[DirectForwardProfile] {repo} {size} WARM{run} wall={wall:F0}ms exec={Graph.GraphExecutor.LastRunTotalMs:F0}ms "
                    + $"dispatches={n} dispatchCpu={cpu:F0}ms ({(n > 0 ? cpu * 1000 / n : 0):F0}us/dispatch) = "
                    + $"shader {WebGPUBackend.ProfileCpuShaderResolveMs:F0} + args {WebGPUBackend.ProfileCpuArgBuildMs:F0} + "
                    + $"bindGroup {WebGPUBackend.ProfileCpuBindGroupMs:F0} (raw create {WebGPUBackend.ProfileCpuBindGroupCreateMs:F0}) + encode {WebGPUBackend.ProfileCpuEncodeMs:F0} | "
                    + $"syncWait={WebGPUBackend.ProfileSyncWaitMs:F0}ms x{WebGPUBackend.ProfileSyncWaitCount} readback={WebGPUBackend.ProfileReadbackWaitMs:F0}ms x{WebGPUBackend.ProfileReadbackWaitCount} | "
                    + $"outside dispatch = {wall - cpu - WebGPUBackend.ProfileSyncWaitMs - WebGPUBackend.ProfileReadbackWaitMs:F0}ms | TJS DAv3-518 warm ref = 61ms");
                Console.WriteLine($"[DirectForwardProfile] {repo} {size} WARM{run} shaderResolve hits={wgAcc.ShaderResolveCacheHits - resolveHits0} misses={wgAcc.ShaderResolveCacheMisses - resolveMisses0} cacheEnabled={WebGPUBackend.EnableShaderResolveCache} | batchSubmit={WebGPUBackend.ProfileBatchSubmitMs:F0}ms x{WebGPUBackend.ProfileBatchSubmitCount} | GC gen0/1/2={GC.CollectionCount(0) - gc0}/{GC.CollectionCount(1) - gc1}/{GC.CollectionCount(2) - gc2} alloc={(GC.GetTotalAllocatedBytes(false) - alloc0) / 1048576.0:F1}MiB");
                Console.WriteLine($"[DirectForwardProfile] {repo} {size} WARM{run} folded={Graph.GraphExecutor.LastRunFoldedNodes} [{Graph.GraphExecutor.LastRunFoldState}] drains={Graph.GraphExecutor.LastRunSyncDrainCount} ({Graph.GraphExecutor.LastRunSyncDrainMs:F0}ms, byte-cap {Graph.GraphExecutor.LastRunSyncDrainByBytesCount}, peak pending {Graph.GraphExecutor.LastRunPeakPendingReleaseBytes / 1048576.0:F0} MiB) readbacks={Graph.GraphExecutor.LastRunReadbackCount} ({Graph.GraphExecutor.LastRunReadbackMs:F0}ms)");
                if (Graph.GraphExecutor.OpProfile is { } prof)
                {
                    Graph.GraphExecutor.OpProfile = null;
                    foreach (var (op, e) in prof.OrderByDescending(kv => kv.Value.WallMs).Take(25))
                        Console.WriteLine($"[DirectForwardProfile] {repo} {size} OP {op,-24} n={e.Count,5} wall={e.WallMs,8:F1}ms ({e.WallMs * 1000 / e.Count,6:F0}us/node) dispatches={e.Dispatches,5} readbacks={e.Readbacks,4} drains={e.Drains,4} alloc={e.AllocBytes / 1024.0 / e.Count,7:F1}KB/node");
                    var asp = WebGPUBackend.ProfileCpuArgsSplitMs;
                    Console.WriteLine($"[DirectForwardProfile] {repo} {size} ARGS expand+manifest {asp[0]:F1} + views {asp[1]:F1} + scalars {asp[2]:F1} ms");
                    var ph = Graph.GraphExecutor.OpPhaseMs;
                    Console.WriteLine($"[DirectForwardProfile] {repo} {size} PHASES prelude {ph[0]:F1} + inputs {ph[1]:F1} + shapes {ph[2]:F1} + rent {ph[3]:F1} + execute {ph[4]:F1} + post {ph[5]:F1} ms");
                    foreach (var (who, cnt) in WebGPUBackend.ProfileBatchSubmitCallers.OrderByDescending(kv => kv.Value).Take(8))
                        Console.WriteLine($"[DirectForwardProfile] {repo} {size} SUBMIT x{cnt} {who}");
                    Console.WriteLine($"[DirectForwardProfile] {repo} {size} RunKernel alloc={WebGPUBackend.ProfileCpuAllocBytes / 1048576.0:F1}MiB over {WebGPUBackend.ProfileCpuDispatchCount} dispatches = shader {WebGPUBackend.ProfileCpuAllocByPhase[0] / 1048576.0:F1} + args {WebGPUBackend.ProfileCpuAllocByPhase[1] / 1048576.0:F1} + bindGroup {WebGPUBackend.ProfileCpuAllocByPhase[2] / 1048576.0:F1} + encode {WebGPUBackend.ProfileCpuAllocByPhase[3] / 1048576.0:F1} MiB");
                    // every readback of the warm run, with the executor's own reason (no-interp-case vs declined)
                    foreach (var name in Graph.GraphExecutor.LastRunReadbackNames)
                        Console.WriteLine($"[DirectForwardProfile] {repo} {size} READBACK {name}");
                }
            }
        }
        finally
        {
            WebGPUBackend.EnableDispatchProfiling = prevProf;
        }

        // GATE: GraphExecutor.QueueOrderedDrains (submit-only drain points) must not change a single value.
        bool prevQo = Graph.GraphExecutor.QueueOrderedDrains;
        float[] withWait, submitOnly, unfolded;
        try
        {
            Graph.GraphExecutor.QueueOrderedDrains = false;
            sw.Restart();
            var a1 = await pipeline.EstimateGpuRawAsync(rgba, size, size);
            await accelerator.SynchronizeAsync();
            double waitMs = sw.Elapsed.TotalMilliseconds;
            int waitDrains = Graph.GraphExecutor.LastRunSyncDrainCount;
            withWait = await a1.RawDepth.CopyToHostAsync<float>();
            a1.RawDepth.Dispose();
            Graph.GraphExecutor.QueueOrderedDrains = true;
            sw.Restart();
            var a2 = await pipeline.EstimateGpuRawAsync(rgba, size, size);
            await accelerator.SynchronizeAsync();
            double qoMs = sw.Elapsed.TotalMilliseconds;
            submitOnly = await a2.RawDepth.CopyToHostAsync<float>();
            a2.RawDepth.Dispose();
            // ...and input-independent folding must not change one either: the same forward with folding OFF.
            Graph.GraphExecutor.FoldInputIndependentNodes = false;
            sw.Restart();
            var a3 = await pipeline.EstimateGpuRawAsync(rgba, size, size);
            await accelerator.SynchronizeAsync();
            double unfoldedMs = sw.Elapsed.TotalMilliseconds;
            unfolded = await a3.RawDepth.CopyToHostAsync<float>();
            a3.RawDepth.Dispose();
            Graph.GraphExecutor.FoldInputIndependentNodes = true;
            Console.WriteLine($"[DirectForwardProfile] {repo} {size} FOLD off: {unfoldedMs:F0}ms (folded run above: {qoMs:F0}ms)");
            Console.WriteLine($"[DirectForwardProfile] {repo} {size} DRAINS awaited: {waitMs:F0}ms ({waitDrains} drains) vs submit-only: {qoMs:F0}ms");
        }
        finally
        {
            Graph.GraphExecutor.QueueOrderedDrains = prevQo;
            Graph.GraphExecutor.FoldInputIndependentNodes = true;
        }
        if (withWait.Length != submitOnly.Length) throw new Exception($"length {withWait.Length} vs {submitOnly.Length}");
        int diffs = 0;
        for (int i = 0; i < withWait.Length; i++) if (BitConverter.SingleToInt32Bits(withWait[i]) != BitConverter.SingleToInt32Bits(submitOnly[i])) diffs++;
        if (diffs != 0) throw new Exception($"submit-only drains changed {diffs} of {withWait.Length} depth values");
        int foldDiffs = 0;
        for (int i = 0; i < unfolded.Length; i++) if (BitConverter.SingleToInt32Bits(unfolded[i]) != BitConverter.SingleToInt32Bits(submitOnly[i])) foldDiffs++;
        if (foldDiffs != 0) throw new Exception($"input-independent folding changed {foldDiffs} of {unfolded.Length} depth values");
    });
}
