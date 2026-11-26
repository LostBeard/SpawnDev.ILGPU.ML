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
    public async Task<string> Warm_MobileNetV2_PassAB_GpuTime() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU) throw new UnsupportedTestException("WebGPU timestamps");
        string pass = Environment.GetEnvironmentVariable("WARM_AB_PASS") is { Length: > 0 } p ? p : "convact";
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var bytes = await http.GetByteArrayAsync("models/mobilenetv2/model.onnx");
        var rng = new Random(5);
        var pic = new float[3 * 224 * 224]; for (int i = 0; i < pic.Length; i++) pic[i] = (float)(rng.NextDouble() * 2 - 1);
        using var inBuf = accelerator.Allocate1D(pic);
        var shape = new[] { 1, 3, 224, 224 };
        async Task<(InferenceSession S, Graph.SessionGraphCapture C)> Make(bool skip)
        {
            var saved = Environment.GetEnvironmentVariable("ML_OPT_SKIP");
            Environment.SetEnvironmentVariable("ML_OPT_SKIP", skip ? (string.IsNullOrEmpty(saved) ? pass : saved + "," + pass) : saved);
            try
            {
                var s = await InferenceSession.CreateFromStreamAsync(accelerator, new MemoryStream(bytes, writable: false));
                var c = new Graph.SessionGraphCapture(s, accelerator);
                for (int i = 0; i < 6 && !c.IsCaptured; i++) await c.RunAsync(new Dictionary<string, Tensor> { [s.InputNames[0]] = new Tensor(inBuf.View, shape) });
                if (!c.IsCaptured) throw new Exception("no capture: " + c.CaptureStatus);
                return (s, c);
            }
            finally { Environment.SetEnvironmentVariable("ML_OPT_SKIP", saved); }
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
    });

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
            sw.Restart(); await Run(); first = sw.Elapsed.TotalMilliseconds;
            sw.Restart(); await Run(); second = sw.Elapsed.TotalMilliseconds;
            // MEASURED 2026-11-25: TiledGemm.Gemm (308 KB, 3.9 s in Chrome on a quiet machine) came with every fp32 convolution through Conv2DGemm.RunF32 until that route was
            // removed (Conv_DirectVsTiled_Crossover: the direct implicit GEMM is as fast or faster); its single-mode variants are as large (239 and 203 KB), so the size is the
            // register tile, not the modes; keeping its inner loops rolled made it 285 KB and slower to compile. After: 7 shaders, 173 KB, 0.23 s, first run 849 ms (was 2388).
        }
        finally { global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.OnShaderCompiled = prev; }

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
