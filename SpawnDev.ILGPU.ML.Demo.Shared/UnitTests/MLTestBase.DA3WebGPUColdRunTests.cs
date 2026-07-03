using ILGPU;
using ILGPU.Runtime;
using SpawnDev.BlazorJS;
using SpawnDev.UnitTesting;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// WebGPU-only DA3-5D COLD-RUN measurement (Seven, DAv3 beat-ORT campaign) - the two numbers the
/// campaign is missing on the WebGPU lane:
///  1. Cold vs warm forward split (cold includes the shader-JIT wall previously measured at ~121s;
///     the unique-WGSL-module count comes from the _mldump/&lt;ts&gt;/wgsl dump this run produces -
///     it decides specialization-explosion vs per-module-cost for the JIT attack).
///  2. Depth range on WebGPU with Tuvok's interpreter in its validated config (readback-skip ON,
///     dispatch-elide OFF) - the desktop reference is 0.1365; one earlier interpreter-ON run showed
///     0.0434 (the unconfirmed anomaly Data is blocked on). The range is REPORTED, not asserted to a
///     specific value - flat/NaN still fail hard.
/// Mirrors the setup of Tuvok's DA3Small_Pipeline_5D_WebGPU_ProducesDepth (which gates WebGPU off to
/// protect its HeavyModel timeout); this variant is WebGPU-only with a 30-min timeout instead.
/// Run: PMT_EXCLUDE_CATEGORIES= PMT_FILTER=DA3_WebGPU_ColdRun dotnet test PlaywrightMultiTest/...
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// FIRST-DIVERGENCE HUNT for the WebGPU range deviation (0.161563 vs desktop 0.1365; the
    /// interp-OFF isolation exonerated the shape interpreter - the divergence is WGSL-lane).
    /// Runs the DA3-5D forward on WebGPU with GraphExecutor.CapturedOutputs enabled and compares
    /// per-node sampled outputs against a CUDA reference capture fetched from the test server
    /// (produced by the offline console script da3-cuda-node-capture.cs into
    /// Demo/wwwroot/test-refs/). Walks nodes in reference execution order and reports the FIRST
    /// node whose samples diverge beyond float noise, plus the next few, with node metadata -
    /// naming the kernel/op for the focused WGSL A/B.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> DA3_WebGPU_FirstDivergence_VsCudaCapture() => await RunTestWithResult(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: WebGPU-only divergence hunt");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        System.Text.Json.JsonElement reference;
        try
        {
            using var stream = await http.GetStreamAsync("test-refs/da3-cuda-node-capture.json");
            reference = (await System.Text.Json.JsonDocument.ParseAsync(stream)).RootElement;
        }
        catch (Exception ex)
        {
            throw new UnsupportedTestException($"CUDA reference capture not available (run da3-cuda-node-capture.cs first): {ex.Message}");
        }
        int refCaptureMax = reference.GetProperty("captureMaxElements").GetInt32();
        float refRange = reference.GetProperty("range").GetSingle();

        // Pure executor path (interpreter off) - compare the same path the isolation measured.
        Graph.GraphCompiler.ShapeSubgraphFoldEnabled = false;
        Graph.GraphExecutor.ShapeInterpValidate = false;
        Graph.GraphExecutor.ShapeInterpElideDispatch = false;

        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx");
        var extDataBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx_data");
        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 1, 3, 518, 518 } },
            externalData: extDataBytes);
        using var pipeline = new Pipelines.DepthEstimationPipeline(session, accelerator);

        const int W = 518, H = 518;
        var rgba = new int[W * H];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
            {
                int v = (int)(x / (float)(W - 1) * 255f);
                rgba[y * W + x] = (255 << 24) | (v << 16) | (v << 8) | v;
            }

        Graph.GraphExecutor.CapturedOutputs = new Dictionary<string, float[]>();
        Graph.GraphExecutor.CapturedNodeInfo = new Dictionary<string, string>();
        Graph.GraphExecutor.CaptureMaxElements = refCaptureMax;
        try
        {
            var (rawDepth, minD, maxD, _, _) = await pipeline.EstimateGpuRawAsync(rgba, W, H);
            rawDepth.Dispose();
            var mine = Graph.GraphExecutor.CapturedOutputs;
            var refNodes = reference.GetProperty("nodes");
            var refInfo = reference.GetProperty("info");

            static float MaxAbsErrOf(System.Text.Json.JsonElement refVals, float[] got, out float refMaxAbs)
            {
                int n = Math.Min(refVals.GetArrayLength(), got.Length);
                float maxAbsErr = 0; refMaxAbs = 0; int i = 0;
                foreach (var rv in refVals.EnumerateArray())
                {
                    if (i >= n) break;
                    float r = rv.GetSingle();
                    refMaxAbs = MathF.Max(refMaxAbs, MathF.Abs(r));
                    maxAbsErr = MathF.Max(maxAbsErr, MathF.Abs(r - got[i]));
                    i++;
                }
                return maxAbsErr;
            }

            int common = 0, missing = 0, divergentTotal = 0;
            var firstFew = new List<string>();
            string? firstKey = null, firstInfo = null;
            // Reference JSON preserves the CUDA capture's insertion order = execution order.
            foreach (var refNode in refNodes.EnumerateObject())
            {
                if (!mine.TryGetValue(refNode.Name, out var got)) { missing++; continue; }
                common++;
                float maxAbsErr = MaxAbsErrOf(refNode.Value, got, out float refMaxAbs);
                float relErr = maxAbsErr / (refMaxAbs + 1e-6f);
                if (maxAbsErr > 1e-4f && relErr > 1e-3f)
                {
                    divergentTotal++;
                    if (firstKey == null)
                    {
                        firstKey = refNode.Name;
                        firstInfo = refInfo.TryGetProperty(refNode.Name, out var inf) ? inf.GetString() : "";
                    }
                    if (firstFew.Count < 5)
                        firstFew.Add($"{refNode.Name} e={maxAbsErr:E1}");
                }
            }

            // Forensic on the FIRST divergent node: compare each of its INPUT tensors' captured
            // samples across backends (both captures already hold them) - if an input (e.g. the
            // Slice starts) differs, the bug is upstream shape-math; if inputs match, the kernel
            // that produced this node is guilty.
            string forensic = "";
            if (firstKey != null && firstInfo != null)
            {
                var inList = firstInfo.Split("in: [").ElementAtOrDefault(1)?.Split(']')[0]?.Split(',') ?? Array.Empty<string>();
                foreach (var inName in inList)
                {
                    // Find the producing node's key (keys are "idx_Op_outputName").
                    string? prodKey = null;
                    foreach (var p in refNodes.EnumerateObject())
                        if (p.Name.EndsWith("_" + inName.Trim())) { prodKey = p.Name; break; }
                    if (prodKey == null || !mine.TryGetValue(prodKey, out var gotIn)) { forensic += $" | in {inName.Trim()}: n/a"; continue; }
                    float e = MaxAbsErrOf(refNodes.GetProperty(prodKey), gotIn, out _);
                    var refPrev = string.Join(",", refNodes.GetProperty(prodKey).EnumerateArray().Take(4).Select(v => v.GetSingle().ToString("G4")));
                    var gotPrev = string.Join(",", gotIn.Take(4).Select(v => v.ToString("G4")));
                    forensic += $" | in {inName.Trim()}: e={e:E1} ref[{refPrev}] web[{gotPrev}]";
                }
            }

            // The first divergent node's own value pattern names the mechanism directly:
            // zeros -> bounds/length; the OTHER half's values -> params ignored; shifted -> offset bug.
            string firstVals = "";
            if (firstKey != null && mine.TryGetValue(firstKey, out var firstGot))
            {
                var refPrev = string.Join(",", refNodes.GetProperty(firstKey).EnumerateArray().Take(8).Select(v => v.GetSingle().ToString("G4")));
                var gotPrev = string.Join(",", firstGot.Take(8).Select(v => v.ToString("G4")));
                firstVals = $" OUTref[{refPrev}] OUTweb[{gotPrev}]";
            }
            var report = $"divergent={divergentTotal} FIRST={firstKey}{firstVals}{forensic}";
            Console.WriteLine($"[DA3-FirstDivergence] range web={maxD - minD:F6} ref={refRange:F6} common={common} miss={missing} first5: {string.Join("; ", firstFew)}");
            Console.WriteLine($"[DA3-FirstDivergence] {report}");
            return report;
        }
        finally
        {
            Graph.GraphExecutor.CapturedOutputs = null;
            Graph.GraphExecutor.CapturedNodeInfo = null;
            Graph.GraphExecutor.CaptureMaxElements = 1024;
        }
    });

    /// <summary>
    /// ADAPTER IDENTITY PROBE: every WebGPU perf number this campaign produced came from the PMT
    /// Playwright Chromium - if that browser fell back to the SwiftShader SOFTWARE adapter, the
    /// measured "~1000x slow WGSL kernels" (uniform 1-5 GFLOPS on kernels that hit 1500-6900 GFLOPS
    /// via CUDA/OpenCL on the same silicon) is an ENVIRONMENT artifact, not a codegen problem.
    /// Reports vendor/architecture/device/description + isFallbackAdapter.
    /// </summary>
    [TestMethod]
    public async Task<string> WebGPU_AdapterIdentity_Probe() => await RunTestWithResult(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: WebGPU-only adapter probe");
        var JS = SpawnDev.BlazorJS.BlazorJSRuntime.JS;
        using var gpu = JS.Get<SpawnDev.BlazorJS.JSObject>("navigator.gpu");
        if (gpu == null) throw new Exception("navigator.gpu missing");
        using var adapter = await gpu.JSRef!.CallAsync<SpawnDev.BlazorJS.JSObject>("requestAdapter");
        if (adapter == null) throw new Exception("requestAdapter returned null");
        string vendor = "?", arch = "?", device = "?", desc = "?";
        bool fallback = false;
        try
        {
            using var info = adapter.JSRef!.Get<SpawnDev.BlazorJS.JSObject?>("info");
            if (info != null)
            {
                vendor = info.JSRef!.Get<string?>("vendor") ?? "?";
                arch = info.JSRef!.Get<string?>("architecture") ?? "?";
                device = info.JSRef!.Get<string?>("device") ?? "?";
                desc = info.JSRef!.Get<string?>("description") ?? "?";
            }
        }
        catch { }
        try { fallback = adapter.JSRef!.Get<bool?>("isFallbackAdapter") ?? false; } catch { }
        var report = $"vendor={vendor} arch={arch} device={device} desc='{desc}' isFallbackAdapter={fallback}";
        Console.WriteLine($"[Benchmark][AdapterProbe] {report}");
        return report;
    });

    /// <summary>
    /// THE ORT-WEB FIGHT: WebGPU dispatch-plan capture/replay of the full DAv3-5D forward via
    /// <see cref="WebGPUGraphCapture"/> (the browser twin of the CUDA graph capture). Captures the
    /// ~1300-dispatch forward once under the stable regime, then replays it with a SINGLE interop
    /// crossing per frame. Gates: replay output matches the direct forward (the correctness oracle),
    /// and reports direct-vs-replay per-frame ms - the number that competes with ORT-Web's 73ms warm.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> DA3_WebGPU_PlanReplay_VsOrtWeb() => await RunTestWithResult(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: WebGPU plan-replay measurement (CUDA has its own graph capture)");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx");
        var extDataBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx_data");
        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 1, 3, 518, 518 } },
            externalData: extDataBytes);

        const int W = 518, H = 518;
        var rgba = new int[W * H];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) { int v = (int)(x / (float)(W - 1) * 255f); rgba[y * W + x] = (255 << 24) | (v << 16) | (v << 8) | v; }

        var preprocess = new Kernels.ImagePreprocessKernel(accelerator);
        using var rgbaBuf = accelerator.Allocate1D(rgba);
        using var inputBuf = accelerator.Allocate1D<float>(3 * W * H);
        preprocess.Forward(rgbaBuf.View, inputBuf.View, W, H, W, H);
        await accelerator.SynchronizeAsync();
        var inputTensor = new Tensors.Tensor(inputBuf.View, new[] { 1, 1, 3, W, H }, session.InputNames[0]);
        var inputs = new Dictionary<string, Tensors.Tensor> { [session.InputNames[0]] = inputTensor };

        // Direct reference forward (readback-skip regime; elide OFF - it has a WebGPU-specific gap,
        // tracked on the executor lane) - the correctness oracle AND the "before" per-frame time.
        Graph.GraphCompiler.ShapeSubgraphFoldEnabled = true;
        Graph.GraphExecutor.ShapeInterpValidate = false;
        Graph.GraphExecutor.ShapeInterpElideDispatch = false;
        try
        {
            var dsw = System.Diagnostics.Stopwatch.StartNew();
            var refOut = await session.RunAsync(inputs);
            await accelerator.SynchronizeAsync();
            dsw.Stop();
            double directMs = dsw.Elapsed.TotalMilliseconds;
            var refData = await refOut[session.OutputNames[0]].Data
                .SubView(0, refOut[session.OutputNames[0]].ElementCount).CopyToHostAsync();
            int outCount = refData.Length;

            Console.WriteLine($"[Benchmark][PlanReplay] direct forward {directMs:F0}ms - starting capture");
            // Capture once (warm A + warm B + recording pass happen inside).
            var csw = System.Diagnostics.Stopwatch.StartNew();
            using var cap = await WebGPUGraphCapture.TryCaptureAsync(session, inputs);
            csw.Stop();
            if (cap == null) throw new Exception("WebGPUGraphCapture.TryCaptureAsync returned null on WebGPU");
            Console.WriteLine($"[Benchmark][PlanReplay] capture done in {csw.Elapsed.TotalSeconds:F1}s, ops={cap.DispatchCount}");

            // Discriminator 1: was the CAPTURE PASS itself correct? (stable-slots/suppress-drains
            // regime correctness on WebGPU, independent of replay).
            var capT = cap.Outputs[session.OutputNames[0]];
            var cd = await capT.Data.SubView(0, outCount).CopyToHostAsync();
            float capDiff = 0f;
            for (int i = 0; i < outCount; i++) capDiff = MathF.Max(capDiff, MathF.Abs(cd[i] - refData[i]));

            // Discriminator 2: replay correctness - same input -> output must match the direct forward.
            var r1sw = System.Diagnostics.Stopwatch.StartNew();
            var replayOut = await cap.ReplayAsync(inputs);
            r1sw.Stop();
            Console.WriteLine($"[Benchmark][PlanReplay] first replay {r1sw.Elapsed.TotalMilliseconds:F0}ms");
            var outT = replayOut[session.OutputNames[0]];
            var rd = await outT.Data.SubView(0, outCount).CopyToHostAsync();
            float maxAbsDiff = 0f;
            for (int i = 0; i < outCount; i++) maxAbsDiff = MathF.Max(maxAbsDiff, MathF.Abs(rd[i] - refData[i]));
            if (maxAbsDiff > 1e-3f)
                throw new Exception($"plan replay DIVERGED: replayDiff={maxAbsDiff:E3} capturePassDiff={capDiff:E3} dispatches={cap.DispatchCount} (outCount={outCount}) - capturePassDiff>tol means the capture REGIME broke the forward; else the REPLAY itself diverges from a correct capture");

            Console.WriteLine($"[Benchmark][PlanReplay] correctness OK - timing loop");
            // Per-frame replay timing (the ORT-Web-73ms competitor) + the frame's internal split:
            // planCall = one interop crossing + JS encode loop + queue.submit (jsEncode/jsSubmit are
            // the JS-side sub-split of it); sync = the GPU-execution wait (onSubmittedWorkDone).
            // What the plan call costs is hideable by pipelining (encode frame N+1 during frame N's
            // GPU work); the sync term is the true GPU floor.
            const int R = 3;
            cap.CollectTimings = true;
            double sumPlan = 0, sumSync = 0, sumJsEncode = 0, sumJsSubmit = 0;
            var rsw = System.Diagnostics.Stopwatch.StartNew();
            for (int r = 0; r < R; r++)
            {
                await cap.ReplayAsync(inputs);
                sumPlan += cap.LastPlanCallMs; sumSync += cap.LastSyncMs;
                sumJsEncode += cap.LastJsEncodeMs; sumJsSubmit += cap.LastJsSubmitMs;
                Console.WriteLine($"[Benchmark][PlanReplay] frame {r}: planCall={cap.LastPlanCallMs:F1}ms (jsEncode={cap.LastJsEncodeMs:F1} jsSubmit={cap.LastJsSubmitMs:F1}) sync={cap.LastSyncMs:F1}ms");
            }
            rsw.Stop();
            cap.CollectTimings = false;
            double replayMs = rsw.Elapsed.TotalMilliseconds / R;
            string split = $"planCall {sumPlan / R:F1}ms (jsEncode {sumJsEncode / R:F1} + jsSubmit {sumJsSubmit / R:F1}) + gpuWait {sumSync / R:F1}ms";

            var report = $"dispatches={cap.DispatchCount} | direct {directMs:F0}ms -> replay {replayMs:F1}ms/frame = {directMs / replayMs:F1}x | split: {split} | maxAbsDiff={maxAbsDiff:E2} | ORT-Web warm ref 73ms";
            Console.WriteLine($"[DA3-PlanReplay] {report}");
            return report;
        }
        finally
        {
            Graph.GraphCompiler.ShapeSubgraphFoldEnabled = false;
            Graph.GraphExecutor.ShapeInterpElideDispatch = false;
        }
    });

    /// <summary>
    /// DISPATCH-ELIDE phantom-shape repro (Seven+Tuvok tag-team): under elide on CUDA, block-4
    /// FusedAttention receives q (Mul_output_0) as rank-5 [3,1,6,1370,64] though the Mul produced
    /// [1,6,1370,64]. FusedAttentionOperator's rank-throw now reports the identity evidence
    /// (elemCount/dataLen/objHash) that discriminates pool-tensor aliasing (elemCount = the wrong
    /// shape's 1,578,240) from shape-metadata mutation (elemCount = the real 526,080). CUDA-only;
    /// expected to FAIL with that diagnostic until the elide bug is fixed - run manually, not a
    /// green-suite member (Tuvok's rig test is the permanent gate once elide works).
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task<string> DA3_Cuda_ElidePhantomShape_IdentityEvidence() => await RunTestWithResult(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.Cuda)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: CUDA-only elide repro (the crash is backend-independent; CUDA is the fast lane)");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        Graph.GraphCompiler.ShapeSubgraphFoldEnabled = true;
        Graph.GraphExecutor.ShapeInterpValidate = false;
        Graph.GraphExecutor.ShapeInterpElideDispatch = true;
        try
        {
            var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
                "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx");
            var extDataBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
                "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx_data");
            using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
                inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 1, 3, 518, 518 } },
                externalData: extDataBytes);
            using var pipeline = new Pipelines.DepthEstimationPipeline(session, accelerator);

            const int W = 518, H = 518;
            var rgba = new int[W * H];
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                {
                    int v = (int)(x / (float)(W - 1) * 255f);
                    rgba[y * W + x] = (255 << 24) | (v << 16) | (v << 8) | v;
                }

            try
            {
                var swCold = System.Diagnostics.Stopwatch.StartNew();
                var (rawDepth, minD, maxD, _, _) = await pipeline.EstimateGpuRawAsync(rgba, W, H);
                swCold.Stop();
                rawDepth.Dispose();
                double coldExec = Graph.GraphExecutor.LastRunTotalMs;
                long resolved = Graph.GraphExecutor.LastRunShapeInterpResolved;
                // Warm forward = the orchestration-lever number (the clean CUDA baseline was ~1350ms,
                // ~1200ms of it per-node orchestration; elide's whole purpose is to cut that).
                var swWarm = System.Diagnostics.Stopwatch.StartNew();
                var (rawDepth2, minD2, maxD2, _, _) = await pipeline.EstimateGpuRawAsync(rgba, W, H);
                swWarm.Stop();
                rawDepth2.Dispose();
                var healed = $"ELIDE RAN CLEAN: range cold={maxD - minD:F6} warm={maxD2 - minD2:F6} (ref 0.136469) resolved={resolved} "
                    + $"| cold {swCold.Elapsed.TotalMilliseconds:F0}ms (exec {coldExec:F0}ms) | WARM {swWarm.Elapsed.TotalMilliseconds:F0}ms (exec {Graph.GraphExecutor.LastRunTotalMs:F0}ms) - phantom-shape bug not hit";
                Console.WriteLine($"[ElideRepro] {healed}");
                return healed;
            }
            catch (Exception ex) when (ex.Message.Contains("FusedAttention"))
            {
                // The evidence IS the deliverable - return it as resultText, green.
                var evidence = $"REPRODUCED: {ex.Message}";
                Console.WriteLine($"[ElideRepro] {evidence}");
                return evidence;
            }
        }
        finally
        {
            Graph.GraphCompiler.ShapeSubgraphFoldEnabled = false;
            Graph.GraphExecutor.ShapeInterpElideDispatch = false;
        }
    });

    /// <summary>
    /// RESOLUTION-PATH probe for the range deviation: the input forensics proved Slice_4's data and
    /// starts tensors are IDENTICAL across backends, so either SliceOperator resolved its params
    /// differently on WebGPU (missing ConstantValues entry -> silent full-copy fallback) or the
    /// kernel execution is wrong. This captures the RESOLVED starts/ends/steps + resolution path of
    /// every blocks.4 rope Slice on WebGPU (SliceOperator.CaptureResolvedParams); the offline
    /// console script prints the CUDA side. Params differ -> resolution bug; identical -> kernel.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> DA3_WebGPU_SliceResolution_Probe() => await RunTestWithResult(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: WebGPU-only probe");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        Graph.GraphCompiler.ShapeSubgraphFoldEnabled = false;
        Graph.GraphExecutor.ShapeInterpValidate = false;
        Graph.GraphExecutor.ShapeInterpElideDispatch = false;

        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx");
        var extDataBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx_data");
        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 1, 3, 518, 518 } },
            externalData: extDataBytes);
        using var pipeline = new Pipelines.DepthEstimationPipeline(session, accelerator);

        const int W = 518, H = 518;
        var rgba = new int[W * H];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
            {
                int v = (int)(x / (float)(W - 1) * 255f);
                rgba[y * W + x] = (255 << 24) | (v << 16) | (v << 8) | v;
            }

        Operators.SliceOperator.CaptureResolvedParams = new Dictionary<string, string>();
        try
        {
            var (rawDepth, minD, maxD, _, _) = await pipeline.EstimateGpuRawAsync(rgba, W, H);
            rawDepth.Dispose();
            var cap = Operators.SliceOperator.CaptureResolvedParams;
            int path3Total = cap.Values.Count(v => v.StartsWith("path=3"));
            // The whole blocks.4 attention Slice chain (q_norm split -> rope Slice_1/2 -> rotate-half),
            // compact: last-axis start:end + inShape/outShape - the executor-cascade forensics Tuvok needs
            // (inShape already [.,.,.,1] => upstream cascade; inShape [.,.,.,32] + out [.,.,.,1] => his
            // override not running/overwritten).
            // Rope-INTERNAL slices only (data input = a rope Slice output): the q_norm splits and Shape
            // slices already proved healthy; the corruption enters at this level.
            var chain = cap.Where(kv => kv.Key.Contains("blocks.4/attn/rope") && kv.Key.Contains("Slice_"))
                .Select(kv =>
                {
                    var v = kv.Value;
                    static string Grab(string s, string tag)
                    { int i = s.IndexOf(tag); if (i < 0) return "?"; int j = s.IndexOf(']', i); return s.Substring(i + tag.Length, j - i - tag.Length); }
                    var st = Grab(v, "starts=[").Split(','); var en = Grab(v, "ends=[").Split(',');
                    return $"{ShortKey(kv.Key)}[{st[^1]}:{en[^1]}] in[{Grab(v, "inShape=[")}] out[{Grab(v, "outShape=[")}]";
                }).ToList();
            var report = $"range={maxD - minD:F6} | slices={cap.Count} path3={path3Total} | b4: {string.Join(" ; ", chain)}";
            Console.WriteLine($"[DA3-SliceProbe] {report}");
            return report;
        }
        finally { Operators.SliceOperator.CaptureResolvedParams = null; }

        static string ShortKey(string k)
        {
            // First input (the data tensor) identifies the Slice; trim the shared prefix.
            var first = k.Split('|')[0];
            int i = first.IndexOf("rope");
            return i >= 0 ? first.Substring(i) : first;
        }
    });

    /// <summary>
    /// ISOLATION for the WebGPU range deviation (0.161563 vs desktop 0.1365, found by the cold run
    /// with the interpreter ON): the SAME forward with the interpreter fully OFF - the pure master
    /// executor path (every shape op readback happens, ~1416 x ~345ms, so ONE forward only).
    /// Range flips to 0.1365 → the interpreter/readback-skip is implicated (Tuvok's lane).
    /// Range stays 0.1616 → WGSL kernel divergence (Seven's lane, per-node capture next).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> DA3_WebGPU_InterpOff_RangeIsolation() => await RunTestWithResult(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: WebGPU-only range isolation");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        Graph.GraphCompiler.ShapeSubgraphFoldEnabled = false;
        Graph.GraphExecutor.ShapeInterpValidate = false;
        Graph.GraphExecutor.ShapeInterpElideDispatch = false;

        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx");
        var extDataBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx_data");
        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 1, 3, 518, 518 } },
            externalData: extDataBytes);
        using var pipeline = new Pipelines.DepthEstimationPipeline(session, accelerator);

        const int W = 518, H = 518;
        var rgba = new int[W * H];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
            {
                int v = (int)(x / (float)(W - 1) * 255f);
                rgba[y * W + x] = (255 << 24) | (v << 16) | (v << 8) | v;
            }

        var sw = Stopwatch.StartNew();
        var (rawDepth, minD, maxD, outW, outH) = await pipeline.EstimateGpuRawAsync(rgba, W, H);
        sw.Stop();
        rawDepth.Dispose();
        float range = maxD - minD;
        var report = $"INTERP-OFF forward {sw.Elapsed.TotalSeconds:F1}s (readbacks {Graph.GraphExecutor.LastRunReadbackCount}) | "
            + $"range={range:F6} (interp-ON was 0.161563, desktop ref 0.1365) | {outW}x{outH}";
        Console.WriteLine($"[DA3-InterpOff] {report}");
        if (range < 0.01f) throw new Exception($"flat depth (range={range:F6})");
        return report;
    });

    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> DA3_WebGPU_ColdRun_JitAndRange() => await RunTestWithResult(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: WebGPU-only cold-run measurement (desktop refs live in DA3Small_Pipeline_5D_WebGPU_ProducesDepth)");

        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Tuvok's validated interpreter config: runtime shape interpreter ON (browser-gated
        // readback-skip, 1416->109 readbacks), dispatch-elide OFF (fix in flight on his lane).
        Graph.GraphCompiler.ShapeSubgraphFoldEnabled = true;
        Graph.GraphExecutor.ShapeInterpValidate = false;
        Graph.GraphExecutor.ShapeInterpElideDispatch = false;
        try
        {
            var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
                "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx");
            var extDataBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
                "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx_data");

            var swCreate = Stopwatch.StartNew();
            using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
                inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 1, 3, 518, 518 } },
                externalData: extDataBytes);
            swCreate.Stop();

            using var pipeline = new Pipelines.DepthEstimationPipeline(session, accelerator);

            // Structured gradient input (left dark -> right bright) so depth must vary spatially.
            const int W = 518, H = 518;
            var rgba = new int[W * H];
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                {
                    int v = (int)(x / (float)(W - 1) * 255f);
                    rgba[y * W + x] = (255 << 24) | (v << 16) | (v << 8) | v;
                }

            // COLD forward: includes every shader JIT.
            var swCold = Stopwatch.StartNew();
            var (rawDepth, minD, maxD, outW, outH) = await pipeline.EstimateGpuRawAsync(rgba, W, H);
            swCold.Stop();
            long coldReadbacks = Graph.GraphExecutor.LastRunReadbackCount;
            long coldResolved = Graph.GraphExecutor.LastRunShapeInterpResolved;
            double coldExecMs = Graph.GraphExecutor.LastRunTotalMs;

            float range;
            using (rawDepth)
            {
                int outSize = outW * outH;
                var (nanCount, _, _) = await new ElementWiseKernels(accelerator)
                    .FiniteCheckOnGpuAsync(rawDepth.View.SubView(0, outSize), outSize);
                range = maxD - minD;
                if (nanCount > outSize / 10)
                    throw new Exception($"DA3 WebGPU cold run: {nanCount}/{outSize} NaN values");
                if (range < 0.01f)
                    throw new Exception($"DA3 WebGPU cold run: depth map flat (range={range:F6}) - forward pass wrong");
            }

            // WARM forward: JIT amortized - the number that competes with ORT-Web's 73ms.
            var swWarm = Stopwatch.StartNew();
            var (rawDepth2, minD2, maxD2, _, _) = await pipeline.EstimateGpuRawAsync(rgba, W, H);
            swWarm.Stop();
            rawDepth2.Dispose();
            float range2 = maxD2 - minD2;

            var report = $"create {swCreate.Elapsed.TotalSeconds:F1}s | COLD {swCold.Elapsed.TotalSeconds:F1}s (exec {coldExecMs:F0}ms, readbacks {coldReadbacks}, interpResolved {coldResolved}) | "
                + $"WARM {swWarm.Elapsed.TotalSeconds:F1}s (exec {Graph.GraphExecutor.LastRunTotalMs:F0}ms, readbacks {Graph.GraphExecutor.LastRunReadbackCount}) | "
                + $"range cold={range:F6} warm={range2:F6} (desktop ref 0.1365) | nodes={session.NodeCount} | {outW}x{outH}";
            Console.WriteLine($"[DA3-ColdRun] {report}");
            return report;
        }
        finally
        {
            // Leave no cross-test flag state (mirrors the rig's hygiene; RunTest zombie-eviction
            // covers captures, not these opt-in flags).
            Graph.GraphCompiler.ShapeSubgraphFoldEnabled = false;
            Graph.GraphExecutor.ShapeInterpValidate = false;
            Graph.GraphExecutor.ShapeInterpElideDispatch = false;
        }
    });
}
