using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.ILGPU.Wasm;
using SpawnDev.ILGPU.Wasm.Algorithms;
using SpawnDev.ILGPU.WebGL;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Algorithms;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Where a model's cold first run goes in the browser (Bones' comparison, 2026-11-25: MobileNetV2 on WebGPU, our first run 7.4 s, ONNX Runtime Web's whole first visit 0.9 s).
/// A measurement, not a gate.
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// MobileNetV2 on WebGPU: create, first and second run; then the distinct WGSL shaders the first run compiled, compiled again in Chrome one at a time and all at once
    /// (<c>createComputePipelineAsync</c>, ILGPU's PrecompileComputeShaderAsync). Each copy gets its own entry-point name, so neither Chrome's in-process nor its disk shader cache
    /// (keyed on the generated code) can answer it: both numbers are real compiles. The difference is what compiling a model's shaders in parallel, before its first run, can save.
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task<string> ColdStart_MobileNetV2_ShaderCompile_SerialVsParallel() => await RunTest(accelerator => ColdStartMobileNetV2(accelerator));

    /// <summary>The same with <see cref="Conv2DKernel.Use1x1Kernel"/> off (1x1 convolutions on the general kernel): what the 1x1 kernel costs a first run. Run it alone (a cold page).</summary>
    [TestMethod(Timeout = 600000)]
    public async Task<string> ColdStart_MobileNetV2_No1x1Kernel() => await RunTest(async accelerator =>
    {
        bool was = Conv2DKernel.Use1x1Kernel;
        Conv2DKernel.Use1x1Kernel = false;
        try { return "1x1 kernel off: " + await ColdStartMobileNetV2(accelerator); }
        finally { Conv2DKernel.Use1x1Kernel = was; }
    });

    /// <summary>
    /// What ILGPU's session shader export / import (ShaderArtifactSerializer, Docs/precompiled-shaders.md) saves a returning visitor: MobileNetV2's first run on a FRESH
    /// WebGPU accelerator with the shader cache cleared (ILGPU generates every kernel's WGSL again) against one with the first session's export imported (early hits:
    /// no IR, no WGSL generation), alternated twice. Run it alone (a cold page). A measurement.
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyCpu")]   // fresh accelerators and sessions: kept off the shared light page (Phase C)
    public async Task<string> ColdStart_MobileNetV2_ShaderCacheImport() => await RunTest(async accelerator =>
    {
        if (accelerator is not WebGPUAccelerator) throw new UnsupportedTestException("ILGPU's shader cache is WebGPU");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var bytes = await http.GetByteArrayAsync("models/mobilenetv2/model.onnx");
        ShaderCacheStore.Enabled = false;   // this test drives ILGPU's export / import itself (the store's, between visits, is ColdStart_MobileNetV2_ReturnVisit)
        var input = new float[3 * 224 * 224];
        var rng = new Random(9);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        async Task<double> FirstRun(Accelerator acc)
        {
            using var session = await InferenceSession.CreateFromStreamAsync(acc, new MemoryStream(bytes, writable: false));
            using var t = OwnedTensor<float>.FromHost(acc, input, new[] { 1, 3, 224, 224 });
            var sw = Stopwatch.StartNew();
            using var o = await session.RunOwnedAsync(new Dictionary<string, Tensor<float>> { [session.InputNames[0]] = t });
            await o[session.OutputNames[0]].ToHostAsync();
            return sw.Elapsed.TotalMilliseconds;
        }
        double cold = await FirstRun(accelerator);
        var exportSw = Stopwatch.StartNew();
        string json = ShaderArtifactSerializer.ExportCache();
        double exportMs = exportSw.Elapsed.TotalMilliseconds;
        var lines = new List<string>();
        try
        {
            for (int round = 0; round < 2; round++)
                foreach (bool import in new[] { false, true })
                {
                    ShaderArtifactCache.Clear();
                    double importMs = 0; int registered = 0;
                    if (import) { var isw = Stopwatch.StartNew(); registered = ShaderArtifactSerializer.ImportCache(json); importMs = isw.Elapsed.TotalMilliseconds; }
                    long h0 = ShaderArtifactCache.Hits, ir0 = ShaderArtifactCache.IrSkippedHits, m0 = ShaderArtifactCache.Misses;
                    var b = MLContext.Create(); await b.WebGPU();
                    using var ctx = b.ToContext();
                    using var acc = await ctx.CreatePreferredAcceleratorAsync();
                    double first = await FirstRun(acc);
                    lines.Add($"{(import ? $"imported ({registered} kernels, {importMs:F0} ms)" : "cache cleared")}: first run {first:F0} ms " +
                              $"(hits {ShaderArtifactCache.Hits - h0}, of them IR skipped {ShaderArtifactCache.IrSkippedHits - ir0}, misses {ShaderArtifactCache.Misses - m0})");
                }
        }
        finally { ShaderArtifactCache.Clear(); ShaderArtifactSerializer.ImportCache(json); ShaderCacheStore.Enabled = OperatingSystem.IsBrowser(); }
        var report = $"MobileNetV2 WebGPU: cold first run {cold:F0} ms; export {json.Length / 1024} KB in {exportMs:F0} ms; fresh accelerator, " + string.Join("; ", lines);
        Console.WriteLine("[ColdStart] " + report);
        return report;
    });

    /// <summary>
    /// <see cref="ShaderCacheStore"/> end to end in one page: a first visit (MobileNetV2 with the store on, the default; its shaders generated, then stored in OPFS in
    /// the background), then a return visit simulated in the same page (ILGPU's in-memory cache cleared, the store's once-per-page import reset, a fresh WebGPU
    /// accelerator, a new session). The return visit must import the stored file, load every kernel from it (no misses), and give the first visit's outputs bit for
    /// bit. A cross-run return visit cannot be shown under PMT: every PMT run republishes the demo trimmed, which gives the assemblies new MVIDs (a new deployment).
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyCpu")]   // fresh accelerators and sessions: kept off the shared light page (Phase C)
    public async Task<string> ShaderCacheStore_RoundTrip_ReturnVisitImportsEveryKernel() => await RunTest(async accelerator =>
    {
        if (accelerator is not WebGPUAccelerator) throw new UnsupportedTestException("ILGPU's shader cache is WebGPU");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var bytes = await http.GetByteArrayAsync("models/mobilenetv2/model.onnx");
        var input = new float[3 * 224 * 224];
        var rng = new Random(7);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        async Task<(double First, float[] Out, long Hits, long Misses)> Visit(Accelerator acc)
        {
            long h0 = ShaderArtifactCache.Hits, m0 = ShaderArtifactCache.Misses;
            using var session = await InferenceSession.CreateFromStreamAsync(acc, new MemoryStream(bytes, writable: false));
            using var t = OwnedTensor<float>.FromHost(acc, input, new[] { 1, 3, 224, 224 });
            var sw = Stopwatch.StartNew();
            float[] o;
            using (var outs = await session.RunOwnedAsync(new Dictionary<string, Tensor<float>> { [session.InputNames[0]] = t }))
                o = await outs[session.OutputNames[0]].ToHostAsync();
            return (sw.Elapsed.TotalMilliseconds, o, ShaderArtifactCache.Hits - h0, ShaderArtifactCache.Misses - m0);
        }
        var contexts = new List<Context>();
        async Task<Accelerator> Fresh() { var b = MLContext.Create(); await b.WebGPU(); var ctx = b.ToContext(); contexts.Add(ctx); return await ctx.CreatePreferredAcceleratorAsync(); }
        bool storeWas = ShaderCacheStore.Enabled;
        ShaderCacheStore.Enabled = true;   // the test page keeps it off (MLTestBase); this test is the store
        try
        {
            // a first visit: nothing in memory (other tests in this page compiled other kernels), nothing imported, a fresh accelerator
            await ShaderCacheStore.ClearAsync();
            ShaderArtifactCache.Clear();
            ShaderCacheStore.ResetImportForTest();
            using var acc1 = await Fresh();
            var v1 = await Visit(acc1);
            await ShaderCacheStore.FlushAsync();
            if (ShaderCacheStore.LastExportBytes == 0) throw new Exception("The first visit generated shaders but nothing was stored.");

            // the return visit: nothing in memory, the stored file imported before any kernel loads
            ShaderArtifactCache.Clear();
            ShaderCacheStore.ResetImportForTest();
            using var acc2 = await Fresh();
            var v2 = await Visit(acc2);
            if (ShaderCacheStore.LastImported == 0) throw new Exception("The return visit imported nothing from the store.");
            if (v2.Misses != 0) throw new Exception($"The return visit generated {v2.Misses} shaders the store should have had ({v2.Hits} hits).");
            int differ = 0;
            for (int i = 0; i < v1.Out.Length; i++) if (BitConverter.SingleToInt32Bits(v1.Out[i]) != BitConverter.SingleToInt32Bits(v2.Out[i])) differ++;
            if (differ != 0) throw new Exception($"{differ} of {v1.Out.Length} outputs differ between the visits.");
            var report = $"MobileNetV2 WebGPU: first visit {v1.First:F0} ms ({v1.Misses} shaders generated, {v1.Hits} hits), stored " +
                         $"{ShaderCacheStore.LastExportBytes / 1024} KB as {ShaderCacheStore.Key}; return visit (fresh accelerator, same page) {v2.First:F0} ms, " +
                         $"{ShaderCacheStore.LastImported} kernels imported, {v2.Hits} hits, 0 misses; outputs identical";
            Console.WriteLine("[ColdStart] " + report);
            return report;
        }
        finally
        {
            ShaderCacheStore.Enabled = storeWas;
            await ShaderCacheStore.ClearAsync();
            foreach (var c in contexts) c.Dispose();
        }
    });

    /// <summary>The same with the 32x32-tile convolution kernel ON (<see cref="Conv2DKernel.SmallTileMaxGroups"/> 32; the browser default is off): what it costs a first run. Run it alone (a cold page).</summary>
    [TestMethod(Timeout = 600000)]
    public async Task<string> ColdStart_MobileNetV2_SmallTile() => await RunTest(async accelerator =>
    {
        int was = Conv2DKernel.SmallTileMaxGroups;
        Conv2DKernel.SmallTileMaxGroups = 32;
        try { return "small-tile kernel on: " + await ColdStartMobileNetV2(accelerator); }
        finally { Conv2DKernel.SmallTileMaxGroups = was; }
    });

    /// <summary>
    /// The browser's "device" stage of a cold start (Bones, 2026-11-26: ~400 ms of MobileNetV2's 1.5 s AOT cold visit), split: the builder, the WebGPU and WebGL probes, the context,
    /// the WebGPU device. Once the way most pages do it (AllAcceleratorsAsync, then the preferred accelerator) and once WebGPU alone. The page already holds a device, so the browser's
    /// own first-adapter cost is not in these numbers: the .NET side and the probes are. A measurement.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task<string> ColdStart_DeviceSetup_Stages() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU) throw new UnsupportedTestException("the browser's WebGPU device setup");
        string Ms(Stopwatch s) { var v = s.Elapsed.TotalMilliseconds; s.Restart(); return v.ToString("F0"); }
        var sw = Stopwatch.StartNew();
        var parts = new List<string>();
        // the way MobileNetV2's page did it
        {
            var b = MLContext.Create(); parts.Add("all: builder " + Ms(sw));
            b.Wasm(); b.EnableWasmAlgorithms(); parts.Add("wasm " + Ms(sw));
            await b.WebGPU(); b.EnableWebGPUAlgorithms(); parts.Add("webgpu probe " + Ms(sw));
            await b.WebGL(); parts.Add("webgl probe " + Ms(sw));
            using var ctx = b.ToContext(); parts.Add("context " + Ms(sw));
            using var acc = await ctx.CreatePreferredAcceleratorAsync(); parts.Add("device " + Ms(sw));
        }
        sw.Restart();
        {
            var b = MLContext.Create(); parts.Add("| webgpu only: builder " + Ms(sw));
            await b.WebGPU(); b.EnableWebGPUAlgorithms(); parts.Add("webgpu probe " + Ms(sw));
            using var ctx = b.ToContext(); parts.Add("context " + Ms(sw));
            using var acc = await ctx.CreatePreferredAcceleratorAsync(); parts.Add("device " + Ms(sw));
        }
        var report = "device setup (ms): " + string.Join(", ", parts);
        Console.WriteLine("[ColdStart] " + report);
        return report;
    });

    /// <summary>
    /// MobileNetV2's warm run (Bones, 2026-11-26: ours 13 ms AOT / 22 ms interpreted against ONNX Runtime Web's 4.9 ms on WebGPU): the plain session against its captured replay
    /// (<see cref="Graph.SessionGraphCapture"/>), input upload and output readback included, outputs compared bit for bit. How much of a warm run is per-node host work. A measurement.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task<string> Warm_MobileNetV2_DirectVsCapture() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.WebGPU or AcceleratorType.Cuda)) throw new UnsupportedTestException("capture is CUDA and WebGPU");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var bytes = await http.GetByteArrayAsync("models/mobilenetv2/model.onnx");
        using var session = await InferenceSession.CreateFromStreamAsync(accelerator, new MemoryStream(bytes, writable: false));
        string inName = session.InputNames[0], outName = session.OutputNames[0];
        var shape = new[] { 1, 3, 224, 224 };
        var rng = new Random(5);
        float[] Pic() { var p = new float[3 * 224 * 224]; for (int i = 0; i < p.Length; i++) p[i] = (float)(rng.NextDouble() * 2 - 1); return p; }
        float[] picA = Pic(), picB = Pic();
        using var inBuf = accelerator.Allocate1D<float>(picA.Length);
        async Task<float[]> Run(Func<Dictionary<string, Tensor>, Task<Dictionary<string, Tensor>>> run, float[] pic)
        {
            inBuf.CopyFromCPU(pic);
            var outs = await run(new Dictionary<string, Tensor> { [inName] = new Tensor(inBuf.View, shape) });
            return await outs[outName].Data.SubView(0, 1000).CopyToHostAsync();
        }
        const int Steps = 40;
        async Task<double> Median(Func<Dictionary<string, Tensor>, Task<Dictionary<string, Tensor>>> run)
        {
            var t = new List<double>();
            for (int i = 0; i < Steps; i++) { var sw = Stopwatch.StartNew(); await Run(run, (i & 1) == 0 ? picA : picB); t.Add(sw.Elapsed.TotalMilliseconds); }
            t.Sort(); return t[Steps / 2];
        }
        var direct = (Func<Dictionary<string, Tensor>, Task<Dictionary<string, Tensor>>>)session.RunAsync;
        for (int i = 0; i < 8; i++) await Run(direct, (i & 1) == 0 ? picA : picB);
        double directMs = await Median(direct);
        if (accelerator.AcceleratorType == AcceleratorType.WebGPU)
        {
            // the run ends with a submit, not a completion wait: the readback right after waits for the GPU anyway (one round trip instead of two)
            bool savedSkip = session.SkipCompletionWait;
            double skipMs, waitMs, waitAgain; float[] sA, wA;
            try
            {
                session.SkipCompletionWait = false; waitMs = await Median(direct); wA = await Run(direct, picA);
                session.SkipCompletionWait = true; skipMs = await Median(direct); sA = await Run(direct, picA);
                session.SkipCompletionWait = false; waitAgain = await Median(direct);
            }
            finally { session.SkipCompletionWait = savedSkip; }
            int sd = 0; for (int i = 0; i < 1000; i++) if (BitConverter.SingleToInt32Bits(sA[i]) != BitConverter.SingleToInt32Bits(wA[i])) sd++;
            Console.WriteLine($"[ColdStart] SkipCompletionWait (WebGPU, the default): warm median {skipMs:F1} ms against {waitMs:F1} / {waitAgain:F1} ms with the completion wait (before / after); outputs differ in {sd} of 1000");
        }
        if (accelerator.AcceleratorType == AcceleratorType.WebGPU)
        {
            // where a plain warm run's host time goes: ILGPU's per-dispatch phases and its GPU waits, averaged over 20 runs (profiling adds a little of its own)
            const int Prof = 20;
            var B = typeof(global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend);
            global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ResetDispatchProfiling();
            global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableDispatchProfiling = true;
            var swp = Stopwatch.StartNew();
            try { for (int i = 0; i < Prof; i++) await Run(direct, (i & 1) == 0 ? picA : picB); }
            finally { global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableDispatchProfiling = false; }
            double wall = swp.Elapsed.TotalMilliseconds / Prof;
            double P(string f) => Convert.ToDouble(B.GetField(f)!.GetValue(null)) / Prof;
            double resolve = P("ProfileCpuShaderResolveMs"), args = P("ProfileCpuArgBuildMs"), bind = P("ProfileCpuBindGroupMs"), enc = P("ProfileCpuEncodeMs"),
                   submit = P("ProfileBatchSubmitMs"), sync = P("ProfileSyncWaitMs"), readback = P("ProfileReadbackWaitMs");
            double dispatches = P("ProfileCpuDispatchCount");
            double rest = wall - resolve - args - bind - enc - submit - sync - readback;
            Console.WriteLine($"[ColdStart] host split per warm run (WebGPU, {Prof} runs, {dispatches:F0} dispatches each): wall {wall:F1} ms = shader resolve {resolve:F2}, args {args:F2}, " +
                              $"bind groups {bind:F2}, encode {enc:F2}, submit {submit:F2}, sync wait {sync:F2}, readback wait {readback:F2}, the rest (graph executor, pool, upload, .NET) {rest:F1}");

            // "the rest", split by the executor's own profiler (GraphExecutor.OpProfile / OpPhaseMs; its timestamps add a little): the input upload, RunAsync by
            // node-loop phase, the readback call, and host time + managed allocation by op type
            double upMs = 0, runMs = 0, readMs = 0; var phases = new double[Graph.GraphExecutor.OpPhaseMs.Length];
            var byOp = new Dictionary<string, (double Ms, int N, long Alloc)>();
            try
            {
                for (int i = 0; i < Prof; i++)
                {
                    var prof = new Dictionary<string, Graph.GraphExecutor.OpProfileEntry>();
                    Graph.GraphExecutor.OpProfile = prof;
                    var s1 = Stopwatch.StartNew();
                    inBuf.CopyFromCPU((i & 1) == 0 ? picA : picB);
                    upMs += s1.Elapsed.TotalMilliseconds; s1.Restart();
                    var outs = await session.RunAsync(new Dictionary<string, Tensor> { [inName] = new Tensor(inBuf.View, shape) });
                    runMs += s1.Elapsed.TotalMilliseconds;
                    Graph.GraphExecutor.OpProfile = null;
                    for (int p = 0; p < phases.Length; p++) phases[p] += Graph.GraphExecutor.OpPhaseMs[p];
                    s1.Restart();
                    await outs[outName].Data.SubView(0, 1000).CopyToHostAsync();
                    readMs += s1.Elapsed.TotalMilliseconds;
                    foreach (var (op, e) in prof)
                        byOp[op] = byOp.TryGetValue(op, out var a) ? (a.Ms + e.WallMs, a.N + e.Count, a.Alloc + e.AllocBytes) : (e.WallMs, e.Count, e.AllocBytes);
                }
            }
            finally { Graph.GraphExecutor.OpProfile = null; }
            double loop = phases.Sum();
            Console.WriteLine($"[ColdStart] RunAsync split per warm run ({Prof} runs, profiled): upload {upMs / Prof:F2} ms, RunAsync {runMs / Prof:F2} ms (node loop {loop / Prof:F2}: " +
                              string.Join(", ", Graph.GraphExecutor.OpPhaseNames.Select((n, p) => $"{n} {phases[p] / Prof:F2}")) +
                              $"; outside the loop {(runMs - loop) / Prof:F2}), readback incl. GPU wait {readMs / Prof:F2} ms");
            Console.WriteLine($"[ColdStart] node loop by op (per run): " + string.Join(", ", byOp.OrderByDescending(o => o.Value.Ms).Take(12)
                .Select(o => $"{o.Key} x{o.Value.N / Prof} {o.Value.Ms / Prof:F2} ms {o.Value.Alloc / Prof / 1024} KB")));

            // the fixed round trips, with the GPU idle: a sync with nothing queued, and the 1000-float readback alone (after a run and a sync)
            var syncT = new List<double>(); var readT = new List<double>(); var runSyncT = new List<double>();
            for (int i = 0; i < Prof; i++)
            {
                var s2 = Stopwatch.StartNew();
                await accelerator.SynchronizeAsync();
                syncT.Add(s2.Elapsed.TotalMilliseconds);
                inBuf.CopyFromCPU((i & 1) == 0 ? picA : picB);
                var outs = await session.RunAsync(new Dictionary<string, Tensor> { [inName] = new Tensor(inBuf.View, shape) });
                s2.Restart();
                await accelerator.SynchronizeAsync();
                runSyncT.Add(s2.Elapsed.TotalMilliseconds);
                s2.Restart();
                await outs[outName].Data.SubView(0, 1000).CopyToHostAsync();
                readT.Add(s2.Elapsed.TotalMilliseconds);
            }
            static double Med(List<double> l) { l.Sort(); return l[l.Count / 2]; }
            Console.WriteLine($"[ColdStart] round trips (median of {Prof}): empty SynchronizeAsync {Med(syncT):F2} ms; after RunAsync, SynchronizeAsync (GPU finishing the run) {Med(runSyncT):F2} ms, " +
                              $"then the 1000-float readback with the GPU idle {Med(readT):F2} ms");

            // the GPU starts only when the run's one command buffer is submitted at its end (MobileNetV2 is ~0.6 GFLOP, the submit budget 128): submitting during the run lets
            // it start while the host still encodes. Budgets alternated (A B C A B C), outputs compared bit for bit.
            double savedBudget = Graph.GraphExecutor.SubmitWorkBudgetFlops;
            var budgets = new[] { savedBudget, 0.15e9, 0.05e9 };
            var med = new double[budgets.Length][]; for (int b = 0; b < budgets.Length; b++) med[b] = new double[2];
            var outsBy = new float[budgets.Length][];
            try
            {
                for (int round = 0; round < 2; round++)
                    for (int b = 0; b < budgets.Length; b++)
                    {
                        Graph.GraphExecutor.SubmitWorkBudgetFlops = budgets[b];
                        for (int i = 0; i < 3; i++) await Run(direct, picA);
                        med[b][round] = await Median(direct);
                        outsBy[b] = await Run(direct, picA);
                    }
            }
            finally { Graph.GraphExecutor.SubmitWorkBudgetFlops = savedBudget; }
            int bd = 0; for (int b = 1; b < budgets.Length; b++) for (int i = 0; i < 1000; i++) if (BitConverter.SingleToInt32Bits(outsBy[b][i]) != BitConverter.SingleToInt32Bits(outsBy[0][i])) bd++;
            Console.WriteLine($"[ColdStart] submit budget A/B (warm median, two alternated rounds): " +
                              string.Join(", ", budgets.Select((v, b) => $"{v / 1e9:G3} GFLOP {med[b][0]:F1} / {med[b][1]:F1} ms")) + $"; outputs differ in {bd} values");
        }
        var dA = await Run(direct, picA); var dB = await Run(direct, picB);
        using var capture = new Graph.SessionGraphCapture(session, accelerator);
        var captured = (Func<Dictionary<string, Tensor>, Task<Dictionary<string, Tensor>>>)capture.RunAsync;
        int setup = 0;
        while (!capture.IsCaptured && setup < 4) { await Run(captured, picA); setup++; }
        if (!capture.IsCaptured) throw new Exception("capture never went live: " + capture.CaptureStatus);
        for (int i = 0; i < 4; i++) await Run(captured, (i & 1) == 0 ? picA : picB);
        double replayMs = await Median(captured);
        string split = capture.LastReplaySplit is var (inMs, planMs, syncMs) ? $" [last replay: input copy {inMs:F1}, plan call {planMs:F1}, sync {syncMs:F1} ms]" : "";
        var rA = await Run(captured, picA); var rB = await Run(captured, picB);
        int differ = 0;
        for (int i = 0; i < 1000; i++) { if (BitConverter.SingleToInt32Bits(rA[i]) != BitConverter.SingleToInt32Bits(dA[i])) differ++; if (BitConverter.SingleToInt32Bits(rB[i]) != BitConverter.SingleToInt32Bits(dB[i])) differ++; }
        if (capture.ProfileReplayAsync() is { } timed)
        {
            try
            {
                using var doc = System.Text.Json.JsonDocument.Parse(await timed);
                var root = doc.RootElement;
                if (root.TryGetProperty("kernels", out var ks))
                {
                    Console.WriteLine($"[ColdStart] GPU time (one timed replay): total {root.GetProperty("totalMs").GetDouble():F2} ms over {root.GetProperty("ops").GetInt32()} dispatches");
                    foreach (var k in ks.EnumerateArray().Take(12))
                        Console.WriteLine($"[ColdStart]   {k.GetProperty("label").GetString()}: {k.GetProperty("ms").GetDouble():F2} ms x{k.GetProperty("count").GetInt32()}");
                    // per dispatch (ILGPU 5.4.53+): the convolutions by their workgroup count - does a small grid (late 7x7 / 14x14 layers, few 64-wide tiles) cost
                    // more per FLOP than a large one? Plan order, so the model's layers top to bottom.
                    if (root.TryGetProperty("dispatches", out var ds))
                    {
                        var rows = ds.EnumerateArray().Select((d, i) => (i, label: d.GetProperty("label").GetString() ?? "",
                            ms: d.GetProperty("ms").ValueKind == System.Text.Json.JsonValueKind.Number ? d.GetProperty("ms").GetDouble() : double.NaN,
                            wg: string.Join("x", d.GetProperty("wg").EnumerateArray().Select(v => v.GetInt32())))).ToList();
                        Console.WriteLine($"[ColdStart] per dispatch (plan order, label ms wg): " + string.Join("; ", rows.Where(r => r.label.Contains("Conv"))
                            .Select(r => $"{r.i} {r.label.Replace("Kernel_", "").Replace("Impl", "")} {r.ms:F3} {r.wg}")));
                    }
                }
                else Console.WriteLine("[ColdStart] timed replay: " + root.GetRawText()[..Math.Min(200, root.GetRawText().Length)]);
            }
            catch (Exception ex) { Console.WriteLine($"[ColdStart] timed replay unavailable: {ex.GetType().Name}: {ex.Message}"); }
        }
        var report = $"MobileNetV2 {accelerator.AcceleratorType} warm, median of {Steps} (upload + run + readback): direct {directMs:F1} ms, captured replay {replayMs:F1} ms ({directMs / replayMs:F2}x){split}; " +
                     $"{capture.DispatchCount} dispatches; replay differs from direct in {differ} of 2000 values";
        Console.WriteLine("[ColdStart] " + report);
        if (differ != 0) throw new Exception("the replay is not bit-identical to the direct run: " + report);
        // where the GPU time goes, per op type: a sync after every node (each node's time includes one sync's latency: the smallest node time is that floor)
        var timings = new Dictionary<string, double>();
        bool perOp = Graph.GraphExecutor.PerOpSync;
        var prevTimings = Graph.GraphExecutor.CapturedNodeTimingsMs;
        try
        {
            Graph.GraphExecutor.PerOpSync = true;
            var runs = new List<Dictionary<string, double>>();
            for (int i = 0; i < 3; i++) { Graph.GraphExecutor.CapturedNodeTimingsMs = new(); await Run(direct, picA); runs.Add(Graph.GraphExecutor.CapturedNodeTimingsMs); }
            foreach (var key in runs[0].Keys) timings[key] = runs.Min(r => r.GetValueOrDefault(key, double.MaxValue));     // best of 3 per node
        }
        finally { Graph.GraphExecutor.PerOpSync = perOp; Graph.GraphExecutor.CapturedNodeTimingsMs = prevTimings; }
        double floor = timings.Values.Min();
        var byType = timings.GroupBy(kv => kv.Key.Split('_')[1]).Select(g => (Op: g.Key, N: g.Count(), Ms: g.Sum(kv => kv.Value), Net: g.Sum(kv => kv.Value - floor)))
            .OrderByDescending(x => x.Net).ToList();
        var opLine = $"per op type with a sync after each node ({timings.Count} nodes, sync floor {floor:F2} ms each): " +
                     string.Join(", ", byType.Select(x => $"{x.Op} x{x.N} {x.Ms:F1} ms (above floor {x.Net:F1})"));
        Console.WriteLine("[ColdStart] " + opLine);
        return report + "; " + opLine;
    });

    /// <summary>
    /// A/B of one graph-optimizer pass on MobileNetV2's WebGPU GPU time: two captured sessions in the same page, one with the pass skipped (ML_OPT_SKIP), their timed replays
    /// alternated (GPU timestamps, no host time), medians of 7 each. Env WARM_AB_PASS names the pass (default convact). A measurement.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task<string> Warm_MobileNetV2_PassAB_GpuTime() => await RunTest(accelerator => WarmPassAB(accelerator,
        Environment.GetEnvironmentVariable("WARM_AB_PASS") is { Length: > 0 } p ? p : "convact"));

    /// <summary>The same A/B for the 1x1 convolution kernel (<see cref="Conv2DKernel.Use1x1Kernel"/> off / on): what it saves per warm run, against what it costs a first run.</summary>
    [TestMethod(Timeout = 300000)]
    public async Task<string> Warm_MobileNetV2_1x1KernelAB_GpuTime() => await RunTest(accelerator => WarmPassAB(accelerator, "1x1"));

    /// <summary>The same A/B for the 32x32-tile convolution kernel (<see cref="Conv2DKernel.SmallTileMaxGroups"/> 0 / 32): the late small-grid layers.</summary>
    [TestMethod(Timeout = 300000, Category = "HeavyCpu")]   // fresh accelerators and sessions: kept off the shared light page (Phase C)
    public async Task<string> Warm_MobileNetV2_SmallTileAB_GpuTime() => await RunTest(accelerator => WarmPassAB(accelerator, "smalltile"));

    private async Task<string> WarmPassAB(Accelerator accelerator, string pass)
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU) throw new UnsupportedTestException("WebGPU timestamps");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var bytes = await http.GetByteArrayAsync("models/mobilenetv2/model.onnx");
        var rng = new Random(5);
        var pic = new float[3 * 224 * 224]; for (int i = 0; i < pic.Length; i++) pic[i] = (float)(rng.NextDouble() * 2 - 1);
        using var inBuf = accelerator.Allocate1D(pic);
        var shape = new[] { 1, 3, 224, 224 };
        async Task<(InferenceSession S, Graph.SessionGraphCapture C)> Make(bool skip)
        {
            var saved = Environment.GetEnvironmentVariable("ML_OPT_SKIP");
            bool saved1x1 = Conv2DKernel.Use1x1Kernel;
            int savedSmall = Conv2DKernel.SmallTileMaxGroups;
            // pass "1x1" / "smalltile": a convolution kernel off in the "off" session (a kernel switch, not an optimizer pass; the capture records whichever kernel ran)
            if (pass == "1x1") Conv2DKernel.Use1x1Kernel = !skip;
            else if (pass == "smalltile") Conv2DKernel.SmallTileMaxGroups = skip ? 0 : 32;
            else Environment.SetEnvironmentVariable("ML_OPT_SKIP", skip ? (string.IsNullOrEmpty(saved) ? pass : saved + "," + pass) : saved);
            try
            {
                var s = await InferenceSession.CreateFromStreamAsync(accelerator, new MemoryStream(bytes, writable: false));
                var c = new Graph.SessionGraphCapture(s, accelerator);
                for (int i = 0; i < 6 && !c.IsCaptured; i++) await c.RunAsync(new Dictionary<string, Tensor> { [s.InputNames[0]] = new Tensor(inBuf.View, shape) });
                if (!c.IsCaptured) throw new Exception("no capture: " + c.CaptureStatus);
                return (s, c);
            }
            finally { Environment.SetEnvironmentVariable("ML_OPT_SKIP", saved); Conv2DKernel.Use1x1Kernel = saved1x1; Conv2DKernel.SmallTileMaxGroups = savedSmall; }
        }
        var (sOff, cOff) = await Make(skip: true);
        var (sOn, cOn) = await Make(skip: false);
        try
        {
            static double Total(string json) { using var d = System.Text.Json.JsonDocument.Parse(json); return d.RootElement.GetProperty("totalMs").GetDouble(); }
            var off = new List<double>(); var on = new List<double>();
            for (int i = 0; i < 7; i++) { off.Add(Total(await cOff.ProfileReplayAsync()!)); on.Add(Total(await cOn.ProfileReplayAsync()!)); }
            off.Sort(); on.Sort();
            var report = $"MobileNetV2 WebGPU GPU time, pass '{pass}' off / on (median of 7 alternated timed replays): {off[3]:F2} / {on[3]:F2} ms (range {off[0]:F2}-{off[6]:F2} / {on[0]:F2}-{on[6]:F2}); " +
                         $"dispatches {cOff.DispatchCount} / {cOn.DispatchCount}";
            Console.WriteLine("[ColdStart] " + report);
            return report;
        }
        finally { cOff.Dispose(); cOn.Dispose(); sOff.Dispose(); sOn.Dispose(); }
    }

    private async Task<string> ColdStartMobileNetV2(Accelerator accelerator)
    {
        if (accelerator is not WebGPUAccelerator web) throw new UnsupportedTestException("the browser's WebGPU shader compile");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var shaders = new List<(string Name, string Wgsl)>();
        var prev = global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.OnShaderCompiled;
        global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.OnShaderCompiled = (n, s, e) =>
        {
            lock (shaders) shaders.Add((n, s));
            if (n is "Gemm" or "BicubicResizeImpl" or "RMSNormStatsImpl" or "GELUImpl")       // who loads a kernel MobileNetV2 has no op for
            {
                var frames = Environment.StackTrace.Split('\n').Select(f => f.Trim()).Where(f => f.Contains("SpawnDev.ILGPU.ML") && !f.Contains("UnitTests")).Take(4);
                Console.WriteLine($"[ColdStart] loaded {n} from: {string.Join(" <- ", frames)}");
            }
            prev?.Invoke(n, s, e);
        };
        double create, first, second;
        var input = new float[3 * 224 * 224];
        var rng = new Random(7);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        bool storeWas = ShaderCacheStore.Enabled;
        ShaderCacheStore.Enabled = false;   // this measures generating the shaders (a first visit), not importing a previous visit's
        try
        {
            var bytes = await http.GetByteArrayAsync("models/mobilenetv2/model.onnx");
            var sw = Stopwatch.StartNew();
            using var session = await InferenceSession.CreateFromStreamAsync(accelerator, new MemoryStream(bytes, writable: false));
            create = sw.Elapsed.TotalMilliseconds;
            async Task Run()
            {
                using var t = OwnedTensor<float>.FromHost(accelerator, input, new[] { 1, 3, 224, 224 });
                using var o = await session.RunOwnedAsync(new Dictionary<string, Tensor<float>> { [session.InputNames[0]] = t });
                await o[session.OutputNames[0]].ToHostAsync();
            }
            // the first run split by the executor's profiler: by node-loop phase, and per op (an op's first Execute loads its kernels: ILGPU's IL -> WGSL in .NET, then Chrome's
            // pipeline creation), with the shaders ILGPU generated meanwhile in order
            var firstProf = new Dictionary<string, Graph.GraphExecutor.OpProfileEntry>();
            Graph.GraphExecutor.OpProfile = firstProf;
            var firstNodes = new Dictionary<string, double>();
            Graph.GraphExecutor.CapturedNodeTimingsMs = firstNodes;
            int shadersBefore; lock (shaders) shadersBefore = shaders.Count;
            global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ResetDispatchProfiling();
            global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableDispatchProfiling = true;
            int gc0 = GC.CollectionCount(0), gc1 = GC.CollectionCount(1), gc2 = GC.CollectionCount(2);
            long alloc0 = GC.GetTotalAllocatedBytes(false);
            sw.Restart();
            try { await Run(); }
            finally { Graph.GraphExecutor.OpProfile = null; Graph.GraphExecutor.CapturedNodeTimingsMs = null; global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableDispatchProfiling = false; }
            first = sw.Elapsed.TotalMilliseconds;
            Console.WriteLine($"[ColdStart] first run, slowest node Executes (a node that loads a kernel pays its load): " +
                              string.Join(", ", firstNodes.OrderByDescending(kv => kv.Value).Take(10).Select(kv => $"{kv.Key} {kv.Value:F0}")) +
                              $"; the other {Math.Max(0, firstNodes.Count - 10)} nodes {firstNodes.OrderByDescending(kv => kv.Value).Skip(10).Sum(kv => kv.Value):F0} ms");
            var ph = Graph.GraphExecutor.OpPhaseMs;
            var WB = typeof(global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend);
            double Pf(string f) => Convert.ToDouble(WB.GetField(f)!.GetValue(null));
            Console.WriteLine($"[ColdStart] first run: shader resolve (ILGPU codegen + pipeline creation) {Pf("ProfileCpuShaderResolveMs"):F0} ms, args {Pf("ProfileCpuArgBuildMs"):F0}, " +
                              $"readback wait {Pf("ProfileReadbackWaitMs"):F0}; GC gen0/1/2 {GC.CollectionCount(0) - gc0}/{GC.CollectionCount(1) - gc1}/{GC.CollectionCount(2) - gc2}, " +
                              $"allocated {(GC.GetTotalAllocatedBytes(false) - alloc0) / 1048576.0:F0} MB");
            Console.WriteLine($"[ColdStart] first run by phase: " + string.Join(", ", Graph.GraphExecutor.OpPhaseNames.Select((n, p) => $"{n} {ph[p]:F0}")) +
                              $" ms (node loop {ph.Sum():F0} of {first:F0}); by op: " +
                              string.Join(", ", firstProf.OrderByDescending(kv => kv.Value.WallMs).Take(8).Select(kv => $"{kv.Key} x{kv.Value.Count} {kv.Value.WallMs:F0} ms")));
            sw.Restart(); await Run(); second = sw.Elapsed.TotalMilliseconds;
            // MEASURED 2026-11-25: TiledGemm.Gemm (308 KB, 3.9 s in Chrome on a quiet machine) came with every fp32 convolution through Conv2DGemm.RunF32 until that route was
            // removed (Conv_DirectVsTiled_Crossover: the direct implicit GEMM is as fast or faster); its single-mode variants are as large (239 and 203 KB), so the size is the
            // register tile, not the modes; keeping its inner loops rolled made it 285 KB and slower to compile. After: 7 shaders, 173 KB, 0.23 s, first run 849 ms (was 2388).
        }
        finally { global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.OnShaderCompiled = prev; ShaderCacheStore.Enabled = storeWas; }

        var distinct = shaders.GroupBy(s => s.Wgsl).Select(g => g.First()).Where(s => s.Wgsl.Contains("fn main(")).ToList();
        if (distinct.Count == 0)
            throw new UnsupportedTestException("earlier tests on this page compiled every kernel MobileNetV2 uses: a cold start needs this test alone (PMT_FILTER=ColdStart_)");
        long kb = distinct.Sum(s => (long)s.Wgsl.Length) / 1024;
        var native = web.NativeAccelerator;
        var perShader = new List<(string Name, int Kb, double Ms)>();
        async Task<(double Ms, int Ok)> Compile(string tag, bool parallel)
        {
            int before = native.CachedShaderCount;
            var copies = distinct.Select((s, i) => (Src: s.Wgsl.Replace("fn main(", $"fn {tag}_{i}("), Entry: $"{tag}_{i}", s.Name)).ToList();
            var sw = Stopwatch.StartNew();
            if (parallel) await Task.WhenAll(copies.Select(c => native.PrecompileComputeShaderAsync(c.Src, c.Entry, c.Name)));
            else
                foreach (var c in copies)
                {
                    var one = Stopwatch.StartNew();
                    await native.PrecompileComputeShaderAsync(c.Src, c.Entry, c.Name);
                    perShader.Add((c.Name, c.Src.Length / 1024, one.Elapsed.TotalMilliseconds));
                }
            return (sw.Elapsed.TotalMilliseconds, native.CachedShaderCount - before);
        }
        string salt = "s" + Guid.NewGuid().ToString("N")[..8];
        var serial = await Compile(salt + "a", parallel: false);
        var all = await Compile(salt + "b", parallel: true);
        var biggest = distinct.OrderByDescending(s => s.Wgsl.Length).Take(3).Select(s => $"{s.Name} {s.Wgsl.Length / 1024} KB");
        var report = $"MobileNetV2 WebGPU: create {create:F0} ms, first run {first:F0} ms, second {second:F0} ms; {distinct.Count} distinct shaders from {shaders.Count} compiles, {kb} KB WGSL " +
                     $"(largest: {string.Join(", ", biggest)}); compiled again in Chrome: one at a time {serial.Ms:F0} ms ({serial.Ok} ok), all at once {all.Ms:F0} ms ({all.Ok} ok)";
        Console.WriteLine("[ColdStart] " + report);
        foreach (var s in perShader.OrderByDescending(p => p.Ms))
            Console.WriteLine($"[ColdStart] shader {s.Name}: {s.Kb} KB, {s.Ms:F0} ms");
        if (serial.Ok == 0 || all.Ok == 0) throw new Exception("no shader compiled: the measurement measured nothing. " + report);
        return report;
    }
}
