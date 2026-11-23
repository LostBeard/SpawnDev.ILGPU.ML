using System.Diagnostics;
using System.Runtime.CompilerServices;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Kernel lifetime under a demo-page model switch: load model A, use it, dispose it, load model B, use it, dispose it,
/// GC, load A again. ILGPU holds compiled kernels through weak references, so once A's owners are disposed and a GC
/// runs, A's kernels may be collected and the reload pays the full IL -> IR -> backend compile again. These tests
/// MEASURE that cost on real models (evidence for the ILGPU kernel-lifetime decision, 2026-11-22). The reload must also
/// be bit-exact with the first load, so a cache serving the wrong kernel fails the test.
/// <para>
/// A = depth-anything-v2-small (depth demo), B = yolov8n (detection demo). The GC runs from an UNWOUND stack
/// (<c>await Task.Delay</c> first): browser .NET scans the stack conservatively, and a GC from the continuation that
/// held the pipeline can keep it alive. Each load/run/dispose lives in its own non-inlined async method so no local of
/// the test body references a disposed pipeline; a WeakReference to each pipeline reports whether the GC really
/// collected it.
/// </para>
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>A -> B -> GC -> A, ShaderArtifactCache at its default (enabled).</summary>
    [TestMethod(Timeout = 1200000, Category = "HeavyModel")]
    public async Task<string> KernelLifetime_ModelSwitch_ABA() => await RunTest(accelerator => ModelSwitchABA(accelerator));

    /// <summary>The same with SpawnDev.ILGPU's process-wide ShaderArtifactCache off (restored afterwards).</summary>
    [TestMethod(Timeout = 1200000, Category = "HeavyModel")]
    public async Task<string> KernelLifetime_ModelSwitch_ABA_NoShaderArtifactCache() => await RunTest(async accelerator =>
    {
        bool was = ShaderArtifactCache.Enabled;
        ShaderArtifactCache.Enabled = false;
        try { return await ModelSwitchABA(accelerator); }
        finally { ShaderArtifactCache.Enabled = was; }
    });

    /// <summary>Run 1 with <c>ReleaseCompilerCaches()</c> after each model's warm first inference (cache on).</summary>
    [TestMethod(Timeout = 1200000, Category = "HeavyModel")]
    public async Task<string> KernelLifetime_ModelSwitch_ABA_Release() => await RunTest(accelerator => ModelSwitchABA(accelerator, release: true));

    /// <summary>Run 1 with the release and ShaderArtifactCache off.</summary>
    [TestMethod(Timeout = 1200000, Category = "HeavyModel")]
    public async Task<string> KernelLifetime_ModelSwitch_ABA_Release_NoShaderArtifactCache() => await RunTest(async accelerator =>
    {
        bool was = ShaderArtifactCache.Enabled;
        ShaderArtifactCache.Enabled = false;
        try { return await ModelSwitchABA(accelerator, release: true); }
        finally { ShaderArtifactCache.Enabled = was; }
    });

    sealed record ModelSwitchStep(string Name, double CreateMs, double FirstRunMs, double SecondRunMs,
        long Compiled, long Loaded, WeakReference Owner, float[]? Output);

    async Task<string> ModelSwitchABA(Accelerator accelerator, bool release = false)
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        // Downloads outside every timed step.
        var depthBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl("onnx-community/depth-anything-v2-small", "onnx/model.onnx"));
        var yoloBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl("salim4n/yolov8n-detect-onnx", "yolov8n-onnx-web/yolov8n.onnx"));

        var a1 = await DepthStep("A1 cold", accelerator, depthBytes, release);
        var b = await YoloStep("B", accelerator, yoloBytes, release);
        await CollectFromUnwoundStack();
        bool a1Alive = a1.Owner.IsAlive, bAlive = b.Owner.IsAlive;
        var a2 = await DepthStep("A2 after GC", accelerator, depthBytes, release);
        // Control: reload right after dispose, no GC (what retained kernels would give).
        var a3 = await DepthStep("A3 no GC", accelerator, depthBytes, release);

        for (int i = 0; i < a1.Output!.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(a2.Output![i]) != BitConverter.SingleToInt32Bits(a1.Output[i]))
                throw new Exception($"{BackendName}: reload A2 depth[{i}] = {a2.Output[i]}, first load = {a1.Output[i]}");
            if (BitConverter.SingleToInt32Bits(a3.Output![i]) != BitConverter.SingleToInt32Bits(a1.Output[i]))
                throw new Exception($"{BackendName}: reload A3 depth[{i}] = {a3.Output[i]}, first load = {a1.Output[i]}");
        }

        string Row(ModelSwitchStep s) =>
            $"{s.Name}: create {s.CreateMs:F0} ms, run1 {s.FirstRunMs:F0} ms, run2 {s.SecondRunMs:F0} ms, compiled +{s.Compiled}, loaded +{s.Loaded}";
        var report = $"{BackendName} artifactCache={ShaderArtifactCache.Enabled} release={release} | {Row(a1)} | {Row(b)} | " +
            $"GC: A1 pipeline alive={a1Alive}, B alive={bAlive} | {Row(a2)} | {Row(a3)} | A2/A3 bit-exact vs A1";
        Console.WriteLine($"[KernelLifetime] {report}");
        return report;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Shared cross-library protocol (Data, 2026-11-22, for ML / Codecs / ILGPU numbers that compare): ILGPU 5.4.44,
    // ReleaseCompilerCaches OFF. load A (+1 inference), dispose A, load B (+1 inference), dispose B, reload A
    // (+1 inference). After EVERY step: await Task.Delay(50), GC.Collect x2 + WaitForPendingFinalizers, then record
    // GC.GetTotalMemory(false), the step's wall ms, and the LoadedKernelCount / CompiledKernelCount deltas.
    // A = Qwen2.5-0.5B Instruct Q8_0 (the AI Chat page's model, decode capture on as the page sets it),
    // B = depth-anything-v2-small (the depth page's model). Greedy, so the reload's answer must equal the first one.
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>LLM A -> depth B -> LLM A under the shared protocol, ShaderArtifactCache at its default (enabled).</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> KernelLifetime_LlmDepthSwitch_ABA() => await RunTest(accelerator => LlmDepthSwitchABA(accelerator));

    /// <summary>The same with SpawnDev.ILGPU's process-wide ShaderArtifactCache off (restored afterwards).</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> KernelLifetime_LlmDepthSwitch_ABA_NoShaderArtifactCache() => await RunTest(async accelerator =>
    {
        bool was = ShaderArtifactCache.Enabled;
        ShaderArtifactCache.Enabled = false;
        try { return await LlmDepthSwitchABA(accelerator); }
        finally { ShaderArtifactCache.Enabled = was; }
    });

    /// <summary>Run 2 with <c>ReleaseCompilerCaches()</c> after each model's warm first inference (cache on).</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> KernelLifetime_LlmDepthSwitch_ABA_Release() => await RunTest(accelerator => LlmDepthSwitchABA(accelerator, release: true));

    /// <summary>Run 2 with the release and ShaderArtifactCache off.</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> KernelLifetime_LlmDepthSwitch_ABA_Release_NoShaderArtifactCache() => await RunTest(async accelerator =>
    {
        bool was = ShaderArtifactCache.Enabled;
        ShaderArtifactCache.Enabled = false;
        try { return await LlmDepthSwitchABA(accelerator, release: true); }
        finally { ShaderArtifactCache.Enabled = was; }
    });

    /// <summary>A loaded model kept between its load step and its dispose step (so the test body holds no typed local).</summary>
    sealed class SwitchModel : IAsyncDisposable
    {
        public IDisposable? Pipeline;
        public IDisposable? Session;
        public Stream? Stream;
        public WeakReference? Owner;
        public async ValueTask DisposeAsync()
        {
            Pipeline?.Dispose();
            Session?.Dispose();
            if (Stream != null) await Stream.DisposeAsync();
            Pipeline = null; Session = null; Stream = null;
        }
    }

    async Task<string> LlmDepthSwitchABA(Accelerator accelerator, bool release = false)
    {
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available for this backend");
        var hub = new ModelsHostStreamLoader(http);
        var depthBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl("onnx-community/depth-anything-v2-small", "onnx/model.onnx"));
        await CollectFromUnwoundStack();
        double heap0 = GC.GetTotalMemory(false) / 1048576.0;

        var rows = new List<string> { $"{BackendName} artifactCache={ShaderArtifactCache.Enabled} release={release} | start heap {heap0:F1} MiB" };
        async Task<T> Step<T>(string name, Func<Task<(T Result, string Detail)>> body)
        {
            long c0 = accelerator.CompiledKernelCount, l0 = accelerator.LoadedKernelCount;
            var sw = Stopwatch.StartNew();
            var (result, detail) = await body();
            double ms = sw.Elapsed.TotalMilliseconds;
            await CollectFromUnwoundStack();
            double heap = GC.GetTotalMemory(false) / 1048576.0;
            rows.Add($"{name}: {ms:F0} ms{detail}, compiled +{accelerator.CompiledKernelCount - c0}, loaded +{accelerator.LoadedKernelCount - l0}, heap {heap:F1} MiB{ShaderCacheNote()}");
            return result;
        }
        // WebGPU keeps one GPUShaderModule + GPUComputePipeline (+ its WGSL source as the cache key) per distinct kernel for the
        // accelerator's life, by design (pipeline creation is the expensive part). Reported per step so the heap a disposed model
        // leaves behind on WebGPU can be attributed (Data, 2026-11-22).
        string ShaderCacheNote() => accelerator is SpawnDev.ILGPU.WebGPU.WebGPUAccelerator wg ? $", cached shaders {wg.CachedShaderCount}" : "";

        var a1 = await Step("1 load+run A (LLM)", () => LlmLoadRun(accelerator, hub, release));
        string answer1 = a1.Answer;
        var weakA1 = a1.Model.Owner!;
        await Step("2 dispose A", async () => { await a1.Model.DisposeAsync(); return (0, ""); });
        a1 = default;
        var b = await Step("3 load+run B (depth)", () => DepthLoadRun(accelerator, depthBytes, release));
        var weakB = b.Owner!;
        await Step("4 dispose B", async () => { await b.DisposeAsync(); return (0, ""); });
        b = null;
        await CollectFromUnwoundStack();
        rows.Add($"collected: A alive={weakA1.IsAlive}, B alive={weakB.IsAlive}");
        var a2 = await Step("5 reload+run A (LLM)", () => LlmLoadRun(accelerator, hub, release));
        string answer2 = a2.Answer;
        await Step("6 dispose A", async () => { await a2.Model.DisposeAsync(); return (0, ""); });
        a2 = default;
        // Attribution (WebGPU only): how much of what the disposed models left behind is the accelerator's shader cache.
        if (accelerator is SpawnDev.ILGPU.WebGPU.WebGPUAccelerator webgpu)
            await Step("7 ClearShaderCache", () => { webgpu.ClearShaderCache(); return Task.FromResult((0, "")); });

        if (answer2 != answer1)
            throw new Exception($"{BackendName}: reloaded LLM answered differently. first: {answer1} | reload: {answer2}");
        rows.Add($"reload answer identical ({answer1.Length} chars)");
        var report = string.Join(" | ", rows);
        Console.WriteLine($"[KernelLifetime] {report}");
        return report;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    async Task<((SwitchModel Model, string Answer) Result, string Detail)> LlmLoadRun(Accelerator accelerator, ModelsHostStreamLoader hub, bool release)
    {
        var sw = Stopwatch.StartNew();
        var m = await hub.OpenAsync("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf", deselect: false);
        var pipe = await GgufTextGenerationPipeline.CreateFromStreamAsync(accelerator, m.Stream, maxSeqLen: 1024);
        pipe.EnableWebGPUDecodeCapture = true;   // as AiChatPage sets it (no-op off WebGPU)
        double loadMs = sw.Elapsed.TotalMilliseconds;
        var model = new SwitchModel { Pipeline = pipe, Stream = m.Stream, Owner = new WeakReference(pipe) };
        sw.Restart();
        string answer = await pipe.GenerateAsync(new List<(string, string)> { ("user", "What is the capital of France?") },
            new GenerationConfig { MaxNewTokens = 8, Strategy = "greedy" }, thinking: false);
        if (release) accelerator.ReleaseCompilerCaches();   // after the warm first inference, as an app would
        return ((model, answer), $" (load {loadMs:F0}, generate 8 tok {sw.Elapsed.TotalMilliseconds:F0})");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    async Task<(SwitchModel Result, string Detail)> DepthLoadRun(Accelerator accelerator, byte[] onnxBytes, bool release)
    {
        const int w = 64, h = 48;
        var px = new int[w * h];
        for (int i = 0; i < px.Length; i++) px[i] = (int)((uint)i * 2654435761u & 0xFFFFFFu) | unchecked((int)0xFF000000);
        using var frame = accelerator.Allocate1D(px);
        var sw = Stopwatch.StartNew();
        var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 3, 224, 224 } });
        var pipeline = new DepthEstimationPipeline(session, accelerator);
        double loadMs = sw.Elapsed.TotalMilliseconds;
        var model = new SwitchModel { Pipeline = pipeline, Session = session, Owner = new WeakReference(pipeline) };
        sw.Restart();
        var (raw, _, _, _, _) = await pipeline.EstimateGpuRawAsync(frame.View, w, h, w, h);   // reads min/max back: synced
        raw.Dispose();
        if (release) accelerator.ReleaseCompilerCaches();
        return (model, $" (load {loadMs:F0}, run {sw.Elapsed.TotalMilliseconds:F0})");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    static async Task CollectFromUnwoundStack()
    {
        await Task.Delay(50);
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    async Task<ModelSwitchStep> DepthStep(string name, Accelerator accelerator, byte[] onnxBytes, bool release)
    {
        const int w = 64, h = 48;
        var px = new int[w * h];
        var rng = new Random(1);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                px[y * w + x] = (int)(x * 255f / w) | ((int)(y * 255f / h) << 8) | (rng.Next(0, 256) << 16) | unchecked((int)0xFF000000);
        using var frame = accelerator.Allocate1D(px);

        long c0 = accelerator.CompiledKernelCount, l0 = accelerator.LoadedKernelCount;
        var sw = Stopwatch.StartNew();
        var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 3, 224, 224 } });
        var pipeline = new DepthEstimationPipeline(session, accelerator);
        double createMs = sw.Elapsed.TotalMilliseconds;
        float[] output;
        double run1, run2;
        try
        {
            sw.Restart();
            var (raw, _, _, ow, oh) = await pipeline.EstimateGpuRawAsync(frame.View, w, h, w, h);   // reads min/max back: synced
            run1 = sw.Elapsed.TotalMilliseconds;
            using (raw) output = await raw.CopyToHostAsync<float>(0, ow * oh);
            if (release) accelerator.ReleaseCompilerCaches();   // after the warm first inference, as an app would
            sw.Restart();
            var (raw2, _, _, _, _) = await pipeline.EstimateGpuRawAsync(frame.View, w, h, w, h);
            run2 = sw.Elapsed.TotalMilliseconds;
            raw2.Dispose();
        }
        finally
        {
            pipeline.Dispose();
            session.Dispose();
        }
        return new ModelSwitchStep(name, createMs, run1, run2,
            accelerator.CompiledKernelCount - c0, accelerator.LoadedKernelCount - l0, new WeakReference(pipeline), output);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    async Task<ModelSwitchStep> YoloStep(string name, Accelerator accelerator, byte[] onnxBytes, bool release)
    {
        var img = new int[640 * 480];
        for (int y = 0; y < 480; y++)
            for (int x = 0; x < 640; x++)
            {
                bool box = x > 200 && x < 400 && y > 100 && y < 350;
                img[y * 640 + x] = (box ? 200 : 50) | ((box ? 180 : 40) << 8) | ((box ? 160 : 30) << 16) | (0xFF << 24);
            }
        long c0 = accelerator.CompiledKernelCount, l0 = accelerator.LoadedKernelCount;
        var sw = Stopwatch.StartNew();
        var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes);
        var pipeline = new ObjectDetectionPipeline(session, accelerator);
        double createMs = sw.Elapsed.TotalMilliseconds;
        double run1, run2;
        try
        {
            sw.Restart();
            await pipeline.DetectAsync(img, 640, 480, confidenceThreshold: 0.1f);
            run1 = sw.Elapsed.TotalMilliseconds;
            if (release) accelerator.ReleaseCompilerCaches();
            sw.Restart();
            await pipeline.DetectAsync(img, 640, 480, confidenceThreshold: 0.1f);
            run2 = sw.Elapsed.TotalMilliseconds;
        }
        finally
        {
            pipeline.Dispose();
            session.Dispose();
        }
        return new ModelSwitchStep(name, createMs, run1, run2,
            accelerator.CompiledKernelCount - c0, accelerator.LoadedKernelCount - l0, new WeakReference(pipeline), null);
    }
}
