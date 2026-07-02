using ILGPU;
using ILGPU.Runtime;
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
            // ONLY the rotate-half Slices (Slice_4/Slice_6 class): starts [0,0,0,16] - the divergent ones.
            var rope4 = cap.Where(kv => kv.Key.Contains("blocks.4/attn/rope") && kv.Value.Contains("starts=[0,0,0,16]"))
                .Select(kv => $"{ShortKey(kv.Key)} -> {kv.Value}").ToList();
            var report = $"range={maxD - minD:F6} | slices={cap.Count} path3Total={path3Total} | b4 rotate-half: {string.Join(" ; ", rope4)}";
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
