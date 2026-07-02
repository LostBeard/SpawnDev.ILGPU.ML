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
