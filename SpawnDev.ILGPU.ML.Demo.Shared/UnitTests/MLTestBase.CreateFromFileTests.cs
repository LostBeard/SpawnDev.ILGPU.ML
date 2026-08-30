using System.IO;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    // ──────────────────────────────────────────────────────────────
    // CreateFromFileAsync — universal model loading (auto-detects format)
    // These tests mirror existing pipeline tests but use the generic loader.
    // ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Load SqueezeNet via CreateFromFileAsync (.onnx auto-detected),
    /// classify a gradient image, verify non-uniform output.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task CreateFromFile_SqueezeNet_Classify() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        using var session = await InferenceSession.CreateFromFileAsync(
            accelerator, http, "models/squeezenet/model.onnx");
        Console.WriteLine($"[CreateFromFile_SN] {session}");

        // Gradient test image
        int w = 64, h = 64;
        var pixels = CreateGradientImage(w, h);

        var pipeline = new ClassificationPipeline(session, accelerator);
        var results = await pipeline.ClassifyAsync(pixels, w, h, 10);

        Console.WriteLine($"[CreateFromFile_SN] Top-5:");
        foreach (var r in results.Take(5))
            Console.WriteLine($"  {r.Label} ({r.Confidence:P2}, class {r.ClassIndex})");

        float ratio = results[0].Confidence / Math.Max(results[^1].Confidence, 1e-10f);
        if (ratio < 1.5f)
            throw new Exception($"Output uniform: ratio={ratio:F2}x");

        Console.WriteLine($"[CreateFromFile_SN] PASS — ratio={ratio:F1}x");
        pipeline.Dispose();
    });

    /// <summary>
    /// Load SqueezeNet via CreateFromFileAsync, classify real cat image.
    /// Verifies cat class (281-285) in top-10 predictions.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task CreateFromFile_SqueezeNet_CatClassification() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        var (pixels, width, height) = await LoadCatImage(http);
        Console.WriteLine($"[CreateFromFile_Cat] Cat image: {width}x{height}");

        using var session = await InferenceSession.CreateFromFileAsync(
            accelerator, http, "models/squeezenet/model.onnx");
        var pipeline = new ClassificationPipeline(session, accelerator);
        var results = await pipeline.ClassifyAsync(pixels, width, height, 10);

        Console.WriteLine($"[CreateFromFile_Cat] Top-5:");
        foreach (var r in results.Take(5))
            Console.WriteLine($"  {r.Label} ({r.Confidence:P2}, class {r.ClassIndex})");

        var catClasses = new HashSet<int> { 281, 282, 283, 284, 285 };
        bool foundCat = results.Any(r => catClasses.Contains(r.ClassIndex));
        if (!foundCat)
            throw new Exception($"No cat class in top-10. Got: [{string.Join(", ", results.Select(r => $"{r.ClassIndex}:{r.Label}"))}]");

        var catResult = results.First(r => catClasses.Contains(r.ClassIndex));
        Console.WriteLine($"[CreateFromFile_Cat] PASS — '{catResult.Label}' at {catResult.Confidence:P2}");
        pipeline.Dispose();
    });

    /// <summary>
    /// Load style-mosaic via CreateFromFileAsync, apply to gradient image.
    /// Verifies output differs from input.
    /// </summary>
    // 120s (not 60s): the same full 224x224 mosaic style inference as Reference_StyleMosaic,
    // which is legitimately ~57s on the slow CPU backend (HeavyCpu, serialized). 60s was never a
    // realistic budget for CPU; GPU backends finish in well under a second.
    [TestMethod(Timeout = 120000, Category = "HeavyCpu")]
    public async Task CreateFromFile_StyleTransfer_Mosaic() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        using var session = await InferenceSession.CreateFromFileAsync(
            accelerator, http, "models/style-mosaic/model.onnx");
        Console.WriteLine($"[CreateFromFile_Style] {session}");

        int w = 64, h = 64;
        var pixels = CreateGradientImage(w, h);

        var pipeline = new StyleTransferPipeline(session, accelerator);
        var result = await pipeline.TransferAsync(pixels, w, h);

        Console.WriteLine($"[CreateFromFile_Style] Output: {result.Width}x{result.Height}");

        var firstPixel = result.RgbaPixels[0];
        bool allSame = result.RgbaPixels.All(p => p == firstPixel);
        if (allSame)
            throw new Exception("Output is uniform — all pixels identical");

        int diffCount = 0;
        int checkPixels = Math.Min(pixels.Length, result.RgbaPixels.Length);
        for (int i = 0; i < checkPixels; i++)
            if (pixels[i] != result.RgbaPixels[i]) diffCount++;

        float diffPct = (float)diffCount / checkPixels;
        Console.WriteLine($"[CreateFromFile_Style] {diffPct:P1} pixels changed");

        if (diffPct < 0.5f)
            throw new Exception($"Style barely changed image: only {diffPct:P1} differ");

        Console.WriteLine("[CreateFromFile_Style] PASS");
        pipeline.Dispose();
    });

    /// <summary>
    /// Load ESPCN super-resolution via CreateFromFileAsync, upscale gradient image.
    /// Verifies output dimensions and non-uniform content.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task CreateFromFile_SuperResolution_ESPCN() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType == AcceleratorType.CPU)
            throw new UnsupportedTestException("ESPCN too slow on CPU — skipped");

        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        using var session = await InferenceSession.CreateFromFileAsync(
            accelerator, http, "models/super-resolution/model.onnx");
        Console.WriteLine($"[CreateFromFile_SR] {session}");

        int w = 32, h = 32;
        var pixels = CreateGradientImage(w, h);

        var pipeline = new SuperResolutionPipeline(session, accelerator, upscaleFactor: 3);
        var result = await pipeline.UpscaleAsync(pixels, w, h);

        Console.WriteLine($"[CreateFromFile_SR] Output: {result.Width}x{result.Height}");

        if (result.Width < 3 || result.Height < 3)
            throw new Exception($"Output too small: {result.Width}x{result.Height}");

        bool allBlack = result.RgbaPixels.All(p => (p & 0x00FFFFFF) == 0);
        if (allBlack)
            throw new Exception("Output is all black");

        var uniqueGrays = result.RgbaPixels.Select(p => p & 0xFF).Distinct().Count();
        Console.WriteLine($"[CreateFromFile_SR] Unique gray levels: {uniqueGrays}");

        if (uniqueGrays < 10)
            throw new Exception($"Output too flat: only {uniqueGrays} unique values");

        Console.WriteLine("[CreateFromFile_SR] PASS");
        pipeline.Dispose();
    });

    /// <summary>
    /// Load Depth Anything V2 Small via CreateFromFileAsync (95MB ONNX, 823 nodes).
    /// This is the model that was failing in the demo.
    /// Verifies: model loads, compiles, and can run inference.
    /// </summary>
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task CreateFromFile_DepthAnything_Load() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        Console.WriteLine("[CreateFromFile_Depth] Downloading Depth Anything V2 Small (95MB)...");
        using var session = await InferenceSession.CreateFromFileAsync(
            accelerator, http, "models/depth-anything-v2-small/model.onnx",
            inputShapes: new Dictionary<string, int[]>
            {
                ["pixel_values"] = new[] { 1, 3, 518, 518 }
            });
        Console.WriteLine($"[CreateFromFile_Depth] Loaded: {session}");
        Console.WriteLine($"[CreateFromFile_Depth] Nodes: {session.NodeCount}, Weights: {session.WeightCount}");
        Console.WriteLine($"[CreateFromFile_Depth] Ops: {string.Join(", ", session.OperatorTypes)}");

        // Verify model structure
        if (session.NodeCount < 100)
            throw new Exception($"Expected 800+ nodes, got {session.NodeCount}");
        if (session.WeightCount < 10)
            throw new Exception($"Expected many weights, got {session.WeightCount}");

        Console.WriteLine("[CreateFromFile_Depth] PASS — model loaded and compiled");
    });

    /// <summary>
    /// Load Depth Anything V2 Small and run actual depth estimation on a gradient image.
    /// Verifies depth map output has spatial variation (not flat).
    /// </summary>
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task CreateFromFile_DepthAnything_Inference() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        Console.WriteLine("[DepthInference] Loading model...");
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http, "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx");
        using var session = InferenceSession.CreateFromOnnx(
            accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["pixel_values"] = new[] { 1, 3, 224, 224 }
            });
        Console.WriteLine($"[DepthInference] {session}");

        var pipeline = new DepthEstimationPipeline(session, accelerator);

        // Gradient image — should produce varying depth
        int w = 64, h = 64;
        var pixels = CreateGradientImage(w, h);

        Console.WriteLine("[DepthInference] Running inference...");
        var result = await pipeline.EstimateAsync(pixels, w, h);
        Console.WriteLine($"[DepthInference] Output: {result.Width}x{result.Height}, depth range [{result.MinDepth:F4}, {result.MaxDepth:F4}]");

        // Verify output is not empty
        if (result.DepthMap == null || result.DepthMap.Length == 0)
            throw new Exception("Depth map is empty");

        // Verify depth map has variation (not all same value)
        float min = result.DepthMap.Min();
        float max = result.DepthMap.Max();
        float range = max - min;
        Console.WriteLine($"[DepthInference] Normalized range: {range:F4}");

        if (range < 0.01f)
            throw new Exception($"Depth map is flat: range={range:F6}");

        // Verify reasonable dimensions
        if (result.Width < 10 || result.Height < 10)
            throw new Exception($"Output too small: {result.Width}x{result.Height}");

        Console.WriteLine($"[DepthInference] PASS — {result.DepthMap.Length} depth values, range={range:F3}");
        pipeline.Dispose();
    });

    /// <summary>
    /// THE PAGE PATH vs THE HOST PATH: <see cref="DepthEstimationPipeline.EstimateGpuRawAsync"/> (what
    /// /depth actually renders from) must produce the SAME depth map as EstimateAsync (what every
    /// other gate tests). Captain's regression (2026-07-03): the page showed a featureless positional
    /// RAMP with plausible min/max while all six EstimateAsync gates stayed green - the two paths had
    /// no cross-check. Also guards that the raw map isn't a smooth ramp: real depth must correlate
    /// poorly with a pure x+y gradient.
    /// </summary>
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task CreateFromFile_DepthAnything_GpuRawMatchesHostPath()
        => await DepthGpuRawVsHostBody(inputSize: 224);

    /// <summary>The SAME gate at 518x518 - the /depth PAGE's actual compiled resolution (1369 patch
    /// tokens vs 224's 256). Captain's structureless-ramp report reproduces at 518 while every
    /// 224 gate stays green - resolution is the divergence axis, and the anti-ramp correlation
    /// guard is the quality oracle that catches a smooth-garbage forward.</summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task CreateFromFile_DepthAnything_GpuRawMatchesHostPath_518()
        => await DepthGpuRawVsHostBody(inputSize: 518);

    /// <summary>The CAPTURED path at the page's resolution: EnableGraphCapture replays the whole
    /// forward as one dispatch plan (the DAv3 lever - 66ms/frame vs multi-second direct). The
    /// captured raw map must still match the direct host path AND pass the content-free guard -
    /// this is the gate that lets /depth turn capture ON. Also times the replay estimate.</summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task CreateFromFile_DepthAnything_Captured_518_MatchesHost()
        => await DepthGpuRawVsHostBody(inputSize: 518, enableCapture: true);

    /// <summary>
    /// BYTE-IDENTITY of the two weight-delivery paths: ModelHub.LoadAsync (what the /depth page
    /// feeds CreateFromHuggingFaceAsync - JS fetch + browser cache) vs DownloadBytesChunkedAsync
    /// (what every depth gate uses). All compute stages are gate-green while the page ramps -
    /// corrupted hub bytes are the last suspect standing (this path has short-read history).
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task DepthAnything_HubBytes_MatchHttpBytes() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.WebGPU))
            throw new UnsupportedTestException("hub/browser path gate runs once, on the WebGPU lane");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var js = SpawnDev.SpawnJS.SpawnJSRuntime.Instance;
        if (!js.IsBrowser) throw new UnsupportedTestException("SpawnJSRuntime not available (not a browser lane)");

        var httpBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx");
        using var hub = new Hub.ModelHub(js);
        var hubBytes = await hub.LoadAsync(Hub.ModelHub.KnownModels.DepthAnythingV2Small, Hub.ModelHub.KnownFiles.OnnxModel);

        if (hubBytes.Length != httpBytes.Length)
            throw new Exception($"HUB BYTES WRONG LENGTH: hub={hubBytes.Length} http={httpBytes.Length} (delta {hubBytes.Length - (long)httpBytes.Length})");
        int firstDiff = -1;
        for (int i = 0; i < hubBytes.Length; i++)
            if (hubBytes[i] != httpBytes[i]) { firstDiff = i; break; }
        if (firstDiff >= 0)
        {
            int diffCount = 0;
            for (int i = firstDiff; i < hubBytes.Length; i++) if (hubBytes[i] != httpBytes[i]) diffCount++;
            throw new Exception($"HUB BYTES CORRUPT: first diff at {firstDiff}/{hubBytes.Length}, {diffCount} bytes differ");
        }
        Console.WriteLine($"[HubBytes] IDENTICAL: {hubBytes.Length} bytes");
    });

    private async Task DepthGpuRawVsHostBody(int inputSize, bool enableCapture = false) => await RunTest(async accelerator =>
    {
        // GPU lanes only: the gate runs the 95MB model TWICE (host + raw paths) - on the serialized
        // CPU lane that exceeds every timeout and starves the rest of the depth family (measured:
        // 4 downstream CPU timeouts). CPU EstimateAsync correctness is covered by the other gates.
        if (accelerator.AcceleratorType == AcceleratorType.CPU)
            throw new UnsupportedTestException("CPU too slow for the double-run cross-path gate");
        // Capture variant: CUDA graphs + WebGPU dispatch plans only (no-op elsewhere) - run it
        // where it exists so the gate is meaningful, skip the rest.
        if (enableCapture && accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("graph capture is CUDA/WebGPU only");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available for this backend");
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http, "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx");
        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 3, inputSize, inputSize } });
        var pipeline = new DepthEstimationPipeline(session, accelerator);

        // Real image (same fetch as CatImage) - a gradient input can't distinguish depth from a ramp.
        var (pixels, w, h) = await LoadCatImage(http);

        var host = await pipeline.EstimateAsync(pixels, w, h);
        pipeline.EnableGraphCapture = enableCapture;
        if (enableCapture)
        {
            // First raw call captures (pays a few warm forwards); the SECOND is the pure replay -
            // that's the one compared and timed (what every /depth estimate after the first costs).
            var (warmBuf, _, _, _, _) = await pipeline.EstimateGpuRawAsync(pixels, w, h);
            warmBuf.Dispose();
        }
        var swReplay = System.Diagnostics.Stopwatch.StartNew();
        var (rawBuf, minD, maxD, gw, gh) = await pipeline.EstimateGpuRawAsync(pixels, w, h);
        swReplay.Stop();
        if (enableCapture) Console.WriteLine($"[CaptureGate] replay estimate: {swReplay.Elapsed.TotalMilliseconds:F1}ms");
        try
        {
            if (gw != host.Width || gh != host.Height)
                throw new Exception($"dim mismatch: gpuRaw {gw}x{gh} vs host {host.Width}x{host.Height}");
            var raw = await rawBuf.CopyToHostAsync<float>(0, gw * gh);

            // 1) The two paths must agree. EstimateAsync returns a NORMALIZED map (0-1); the raw
            // path returns raw values - normalize identically before comparing (first gate cut
            // compared raw-vs-normalized and read its own mismatch as a 249% divergence - Rule 4c).
            double maxAbs = 0; float rawRange = Math.Max(1e-6f, maxD - minD);
            for (int i = 0; i < raw.Length; i++)
                maxAbs = Math.Max(maxAbs, Math.Abs((raw[i] - minD) / rawRange - host.DepthMap[i]));
            if (maxAbs > 0.02)
                throw new Exception($"PAGE PATH DIVERGES from host path: normalized maxAbs={maxAbs:F5}");

            // 2) Min/max reduction must match the buffer it reduced.
            float trueMin = raw.Min(), trueMax = raw.Max();
            if (Math.Abs(trueMin - minD) > 1e-3 || Math.Abs(trueMax - maxD) > 1e-3)
                throw new Exception($"MinMax reduction wrong: reported [{minD:F4},{maxD:F4}] vs actual [{trueMin:F4},{trueMax:F4}]");

            // 3) STRUCTURE guard (calibrated vs ORT ground truth + the 2026-07-03 bug): real DA2
            // depth of the cat has 1000+ strong normalized edges; the mis-fused-attention output
            // (double-scaled scores -> near-uniform softmax) had 6 and STILL passed range checks
            // and a lax rampCorr threshold. Edges are the honest oracle.
            double corr = CorrelationWithXYRamp(raw, gw, gh);
            int strongEdges = 0;
            for (int y = 0; y < gh; y++)
                for (int x = 1; x < gw; x++)
                    if (Math.Abs((raw[y * gw + x] - raw[y * gw + x - 1]) / rawRange) > 0.10f) strongEdges++;
            Console.WriteLine($"[GpuRawGate] maxAbs={maxAbs:F5} range=[{minD:F4},{maxD:F4}] rampCorr={corr:F3} strongEdges={strongEdges}");
            if (strongEdges < 100)
                throw new Exception($"Depth map is CONTENT-FREE: {strongEdges} strong edges (real output: 1000+; "
                    + $"the mis-fused-attention failure mode produced 6). rampCorr={corr:F3}");
        }
        finally { rawBuf.Dispose(); }
    });

    private static double CorrelationWithXYRamp(float[] map, int w, int h)
    {
        int n = map.Length;
        double meanM = 0, meanR = 0;
        for (int i = 0; i < n; i++) { meanM += map[i]; meanR += (i % w) + (i / w); }
        meanM /= n; meanR /= n;
        double cov = 0, varM = 0, varR = 0;
        for (int i = 0; i < n; i++)
        {
            double dm = map[i] - meanM, dr = (i % w) + (i / w) - meanR;
            cov += dm * dr; varM += dm * dm; varR += dr * dr;
        }
        return varM < 1e-9 ? 1.0 : cov / Math.Sqrt(varM * varR);   // a flat map counts as degenerate too
    }

    /// <summary>
    /// The COLORMAP seam isolated (no model): a checkerboard depth map through
    /// DepthToColormapPalette must come back as a two-color checkerboard. The /depth page renders a
    /// featureless positional RAMP while the raw depth is proven correct (GpuRawMatchesHostPath) -
    /// this splits the remaining suspects: a gradient here convicts the colormap kernel; a clean
    /// checkerboard convicts PresentAsync/the canvas renderer.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task DepthColormap_Checkerboard_NotARamp() => await RunTest(async accelerator =>
    {
        const int w = 64, h = 48, block = 8;
        var depth = new float[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                depth[y * w + x] = ((x / block + y / block) % 2 == 0) ? 0.2f : 0.8f;

        var post = new SpawnDev.ILGPU.ML.Kernels.ImagePostprocessKernel(accelerator);
        try
        {
            using var depthBuf = accelerator.Allocate1D(depth);
            using var rgba = accelerator.Allocate1D<int>(w * h);
            var depthView = new SpawnDev.ILGPU.ML.Tensors.TensorView<float>(depthBuf.View, new[] { h, w });
            var rgbaView = new SpawnDev.ILGPU.ML.Tensors.TensorView<int>(rgba.View, new[] { h, w });
            post.DepthToColormapPalette(depthView, rgbaView, 0f, 1f, 0 /* plasma */);
            await accelerator.SynchronizeAsync();
            var px = await rgba.CopyToHostAsync<int>(0, w * h);

            int distinct = px.Distinct().Count();
            if (distinct != 2)
                throw new Exception($"COLORMAP BROKEN: checkerboard depth produced {distinct} distinct colors (expected exactly 2) - "
                    + (distinct > w ? "a RAMP (position-dependent output)" : "wrong quantization"));
            // The two colors must follow the DEPTH pattern, not position.
            int cA = px[0];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    bool isA = (x / block + y / block) % 2 == 0;
                    if ((px[y * w + x] == cA) != isA)
                        throw new Exception($"COLORMAP pattern mismatch at ({x},{y}): color does not follow depth");
                }
            Console.WriteLine($"[ColormapOracle] PASS - 2 colors, pattern follows depth (cA=0x{cA:X8})");
        }
        finally { post.Dispose(); }
    });

    /// <summary>
    /// The SHARED-INSTANCE lifecycle (the /depth page's exact kernel sequence, 9cf63f3): ONE
    /// ImagePostprocessKernel runs ResizeBilinear then MinMaxAsync then DepthToColormapPalette.
    /// The standalone checkerboard oracle passes with a FRESH instance - this one reproduces the
    /// page's ordering, where the pipeline comment records a prior "scalar-slot drift -> flat-blue"
    /// battle on this exact seam.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task DepthColormap_SharedKernelLifecycle_NotARamp() => await RunTest(async accelerator =>
    {
        const int srcW = 32, srcH = 24, w = 64, h = 48, block = 8;
        var src = new float[srcW * srcH];
        for (int y = 0; y < srcH; y++)
            for (int x = 0; x < srcW; x++)
                src[y * srcW + x] = ((x / (block / 2) + y / (block / 2)) % 2 == 0) ? 0.2f : 0.8f;

        var post = new SpawnDev.ILGPU.ML.Kernels.ImagePostprocessKernel(accelerator);
        try
        {
            using var srcBuf = accelerator.Allocate1D(src);
            using var resized = accelerator.Allocate1D<float>(w * h);
            var srcView = new SpawnDev.ILGPU.ML.Tensors.TensorView<float>(srcBuf.View, new[] { srcH, srcW });
            var dstView = new SpawnDev.ILGPU.ML.Tensors.TensorView<float>(resized.View, new[] { h, w });
            post.ResizeBilinear(srcView, dstView);                       // page step 1
            var (minD, maxD) = await post.MinMaxAsync(resized.View, w * h); // page step 2 (the NEW reduction)
            if (Math.Abs(minD - 0.2f) > 1e-3 || Math.Abs(maxD - 0.8f) > 1e-3)
                throw new Exception($"MinMax after resize wrong: [{minD:F4},{maxD:F4}] expected [0.2,0.8]");

            using var rgba = accelerator.Allocate1D<int>(w * h);
            var depthView = new SpawnDev.ILGPU.ML.Tensors.TensorView<float>(resized.View, new[] { h, w });
            var rgbaView = new SpawnDev.ILGPU.ML.Tensors.TensorView<int>(rgba.View, new[] { h, w });
            post.DepthToColormapPalette(depthView, rgbaView, minD, maxD, 0); // page step 3
            await accelerator.SynchronizeAsync();
            var px = await rgba.CopyToHostAsync<int>(0, w * h);

            // Bilinear edges blend, so allow a few colors - but a positional RAMP has ~w*h distinct
            // values, and the dominant two must cover most of the image.
            int distinct = px.Distinct().Count();
            var top2 = px.GroupBy(v => v).OrderByDescending(g => g.Count()).Take(2).Sum(g => g.Count());
            double top2Frac = (double)top2 / px.Length;
            Console.WriteLine($"[SharedLifecycle] distinct={distinct} top2Frac={top2Frac:F3} minMax=[{minD:F3},{maxD:F3}]");
            if (top2Frac < 0.60)
                throw new Exception($"SHARED-INSTANCE COLORMAP BROKEN: top-2 colors cover only {top2Frac:P0} (ramp-like, {distinct} distinct)");
        }
        finally { post.Dispose(); }
    });

    /// <summary>
    /// Depth estimation with real cat image — more meaningful than gradient.
    /// </summary>
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task CreateFromFile_DepthAnything_CatImage() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        var (pixels, width, height) = await LoadCatImage(http);
        Console.WriteLine($"[DepthCat] Cat image: {width}x{height}");

        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http, "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx");
        using var session = InferenceSession.CreateFromOnnx(
            accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["pixel_values"] = new[] { 1, 3, 224, 224 }
            });
        var pipeline = new DepthEstimationPipeline(session, accelerator);

        Console.WriteLine("[DepthCat] Running inference...");
        var result = await pipeline.EstimateAsync(pixels, width, height);
        Console.WriteLine($"[DepthCat] Output: {result.Width}x{result.Height}, range [{result.MinDepth:F4}, {result.MaxDepth:F4}]");

        float range = result.DepthMap.Max() - result.DepthMap.Min();
        if (range < 0.01f)
            throw new Exception($"Depth map is flat: range={range:F6}");

        Console.WriteLine($"[DepthCat] PASS — {result.DepthMap.Length} values, range={range:F3}");
        pipeline.Dispose();
    });

    /// <summary>
    /// Depth Anything V2 Small: depth result must match source aspect ratio.
    /// Regression: the model input is square (518×518) and the depth tensor comes back square,
    /// so without GPU-side resize the result is squished against the original image in any
    /// side-by-side display. The pipeline must default to source dimensions, and the
    /// outputWidth/outputHeight params must drive both exact and aspect-preserving sizing.
    /// </summary>
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task CreateFromFile_DepthAnything_OutputAspectRatio() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http, "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx");
        using var session = InferenceSession.CreateFromOnnx(
            accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["pixel_values"] = new[] { 1, 3, 224, 224 }
            });
        var pipeline = new DepthEstimationPipeline(session, accelerator);

        // 96×32 source (3:1 landscape) — same ratio as the city image that exposed the bug.
        const int srcW = 96, srcH = 32;
        var pixels = CreateGradientImage(srcW, srcH);

        // Default: should match source dimensions exactly (preserves aspect 1:1 with input).
        var def = await pipeline.EstimateAsync(pixels, srcW, srcH);
        if (def.Width != srcW || def.Height != srcH)
            throw new Exception($"Default output must match source: expected {srcW}x{srcH}, got {def.Width}x{def.Height}");
        if (def.DepthMap.Length != srcW * srcH)
            throw new Exception($"DepthMap length mismatch: expected {srcW * srcH}, got {def.DepthMap.Length}");

        // Explicit width, derive height from source aspect: 200 × round(200*32/96) = 200 × 67.
        var widthOnly = await pipeline.EstimateAsync(pixels, srcW, srcH, outputWidth: 200);
        int expectedH = (int)MathF.Round(200f * srcH / srcW);
        if (widthOnly.Width != 200 || widthOnly.Height != expectedH)
            throw new Exception($"outputWidth=200 expected 200x{expectedH}, got {widthOnly.Width}x{widthOnly.Height}");

        // Explicit height, derive width from source aspect: round(50*96/32) × 50 = 150 × 50.
        var heightOnly = await pipeline.EstimateAsync(pixels, srcW, srcH, outputHeight: 50);
        int expectedW = (int)MathF.Round(50f * srcW / srcH);
        if (heightOnly.Width != expectedW || heightOnly.Height != 50)
            throw new Exception($"outputHeight=50 expected {expectedW}x50, got {heightOnly.Width}x{heightOnly.Height}");

        // Both explicit: exact size, no aspect preservation.
        var exact = await pipeline.EstimateAsync(pixels, srcW, srcH, outputWidth: 128, outputHeight: 128);
        if (exact.Width != 128 || exact.Height != 128)
            throw new Exception($"Exact output expected 128x128, got {exact.Width}x{exact.Height}");
        if (exact.DepthMap.Length != 128 * 128)
            throw new Exception($"Exact DepthMap length mismatch: expected {128 * 128}, got {exact.DepthMap.Length}");

        // Depth must still vary after resize (bilinear interpolation cannot create
        // variation from nothing, but a flat result would indicate the resize is broken).
        float range = def.DepthMap.Max() - def.DepthMap.Min();
        if (range < 0.01f)
            throw new Exception($"Resized depth map is flat: range={range:F6}");

        Console.WriteLine($"[DepthAspect] PASS — default={def.Width}x{def.Height}, w-only={widthOnly.Width}x{widthOnly.Height}, h-only={heightOnly.Width}x{heightOnly.Height}, exact={exact.Width}x{exact.Height}");
        pipeline.Dispose();
    });

    /// <summary>
    /// Streaming ONNX LOAD: <see cref="InferenceSession.CreateFromOnnxStreamAsync"/> loads SqueezeNet from a
    /// SEEKABLE stream WITHOUT holding the whole model in memory — each weight is seeked to and chunk-uploaded
    /// straight to the GPU. Proves FUNCTIONAL EQUIVALENCE to the byte[] load: the SAME cat image classified
    /// through BOTH sessions yields the same weight/node counts, the same top-5 class sequence, and a matching
    /// top-1 confidence. A tiny streamThreshold forces the weights through the streaming-upload path (not the
    /// small-tensor fallback), so this exercises AllocatePermanentFromStreamAsync. Foundation for loading a
    /// model directly from a TorrentReadStream / HTTP-Range / Blob source, and for sharded loading.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task CreateFromOnnxStream_SqueezeNet_MatchesByteLoad() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        var bytes = await http.GetByteArrayAsync("models/squeezenet/model.onnx");
        var (pixels, width, height) = await LoadCatImage(http);

        // Streamed load — tiny threshold forces real weights through the seek+chunk-upload path.
        using var ms = new MemoryStream(bytes, writable: false);
        using var streamSession = await InferenceSession.CreateFromOnnxStreamAsync(
            accelerator, ms, streamThreshold: 4096);
        // Reference: in-memory byte[] load.
        using var byteSession = InferenceSession.CreateFromOnnx(accelerator, bytes);

        Console.WriteLine($"[StreamLoad] stream: {streamSession}  byte: {byteSession}");

        if (streamSession.WeightCount != byteSession.WeightCount)
            throw new Exception($"weight count mismatch: stream={streamSession.WeightCount} byte={byteSession.WeightCount}");
        if (streamSession.NodeCount != byteSession.NodeCount)
            throw new Exception($"node count mismatch: stream={streamSession.NodeCount} byte={byteSession.NodeCount}");

        var streamPipe = new ClassificationPipeline(streamSession, accelerator);
        var bytePipe = new ClassificationPipeline(byteSession, accelerator);
        var streamRes = await streamPipe.ClassifyAsync(pixels, width, height, 10);
        var byteRes = await bytePipe.ClassifyAsync(pixels, width, height, 10);

        int n = Math.Min(5, Math.Min(streamRes.Count(), byteRes.Count()));
        if (n == 0) throw new Exception("classification returned no results");

        // Same load → same weights → identical top-5 class sequence.
        for (int i = 0; i < n; i++)
            if (streamRes[i].ClassIndex != byteRes[i].ClassIndex)
                throw new Exception(
                    $"top-{i} class differs: stream={streamRes[i].ClassIndex} byte={byteRes[i].ClassIndex} — " +
                    "streaming load produced different weights");

        float confDiff = Math.Abs(streamRes[0].Confidence - byteRes[0].Confidence);
        if (confDiff > 1e-3f)
            throw new Exception($"top-1 confidence mismatch: stream={streamRes[0].Confidence:F6} byte={byteRes[0].Confidence:F6} diff={confDiff:F6}");

        Console.WriteLine($"[StreamLoad] PASS — stream load == byte load (top-1 class {streamRes[0].ClassIndex} @ {streamRes[0].Confidence:P2}, {streamSession.WeightCount} weights, {streamSession.NodeCount} nodes)");
        streamPipe.Dispose();
        bytePipe.Dispose();
    });

    /// <summary>
    /// FP16 weight streaming: a fp16-source weight that needs an fp32 GPU buffer must upload the raw fp16
    /// bytes to the GPU and upcast Half→float ON THE GPU (browser: zero-copy, bytes never enter .NET) — NOT
    /// read into a managed byte[] + CPU BitConverter loop (the old path that pulled every SD-Turbo weight
    /// through .NET and made the browser load take ~10 min; Captain 2026-07-05). This proves the new GPU
    /// upcast in <see cref="BufferPool.AllocatePermanentFromStreamAsync"/> is BIT-EXACT to both the CPU
    /// convert path (forced via DisableJsZeroCopyWeights) and the reference (float)BitConverter.ToHalf.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task StreamLoadFp16_GpuUpcast_BitExact_VsCpu() => await RunTest(async accelerator =>
    {
        const int count = 8192;
        var rnd = new Random(20260705);
        var srcBytes = new byte[count * 2];
        var expected = new float[count];
        for (int i = 0; i < count; i++)
        {
            var h = (System.Half)((float)(rnd.NextDouble() * 40.0 - 20.0)); // finite fp16, no NaN/Inf edge cases
            ushort bits = BitConverter.HalfToUInt16Bits(h);
            srcBytes[i * 2] = (byte)(bits & 0xFF);
            srcBytes[i * 2 + 1] = (byte)(bits >> 8);
            expected[i] = (float)BitConverter.ToHalf(srcBytes, i * 2); // exactly what the code must produce
        }
        int[] shape = { count };

        // Read a loaded weight's GPU data back to host cross-backend: GPU→GPU CopyFrom into a temp buffer
        // (safe on all backends), then CopyToHostAsync (WebGPU-safe mapAsync path).
        async Task<float[]> ReadBack(SpawnDev.ILGPU.ML.Tensors.Tensor t)
        {
            using var dst = accelerator.Allocate1D<float>(count);
            dst.View.CopyFrom(t.Data);
            return await dst.CopyToHostAsync<float>(0, count);
        }

        bool priorFlag = SpawnDev.ILGPU.ML.Tensors.BufferPool.DisableJsZeroCopyWeights;
        var pool = new SpawnDev.ILGPU.ML.Tensors.BufferPool(accelerator);
        try
        {
            // NEW path: GPU Half→float upcast (bytes stream JS→GPU on browser; managed→Half→GPU-convert on desktop).
            SpawnDev.ILGPU.ML.Tensors.BufferPool.DisableJsZeroCopyWeights = false;
            using (var ms = new MemoryStream(srcBytes, writable: false))
            {
                var got = await ReadBack(await pool.AllocatePermanentFromStreamAsync(ms, 0, srcBytes.Length, 10, shape, "gpu"));
                int bad = 0;
                for (int i = 0; i < count; i++)
                    if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(expected[i])) bad++;
                if (bad > 0) throw new Exception($"GPU fp16 upcast NOT bit-exact: {bad}/{count} floats differ from (float)BitConverter.ToHalf");
            }

            // OLD path: CPU BitConverter loop (forced) — must also match (guards the fallback).
            SpawnDev.ILGPU.ML.Tensors.BufferPool.DisableJsZeroCopyWeights = true;
            using (var ms = new MemoryStream(srcBytes, writable: false))
            {
                var got = await ReadBack(await pool.AllocatePermanentFromStreamAsync(ms, 0, srcBytes.Length, 10, shape, "cpu"));
                int bad = 0;
                for (int i = 0; i < count; i++)
                    if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(expected[i])) bad++;
                if (bad > 0) throw new Exception($"CPU fp16 convert path regressed: {bad}/{count} differ");
            }
        }
        finally { SpawnDev.ILGPU.ML.Tensors.BufferPool.DisableJsZeroCopyWeights = priorFlag; pool.Dispose(); }
        Console.WriteLine($"[Fp16StreamLoad] PASS — GPU upcast == CPU convert == reference, {count} fp16 weights bit-exact");
    });

    // ──────────────────────────────────────────────────────────────
    // Helper: create a gradient test image
    // ──────────────────────────────────────────────────────────────
    private static int[] CreateGradientImage(int w, int h)
    {
        var pixels = new int[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                pixels[y * w + x] = (int)(x * 255f / w)
                    | ((int)(y * 255f / h) << 8)
                    | (128 << 16)
                    | (0xFF << 24);
        return pixels;
    }

    // ──────────────────────────────────────────────────────────────
    // Helper: load pre-decoded cat image from samples/cat_rgba.bin
    // ──────────────────────────────────────────────────────────────
    private static async Task<(int[] pixels, int width, int height)> LoadCatImage(System.Net.Http.HttpClient http)
    {
        var binData = await http.GetByteArrayAsync("samples/cat_rgba.bin");
        int width = BitConverter.ToInt32(binData, 0);
        int height = BitConverter.ToInt32(binData, 4);
        var pixels = new int[width * height];
        Buffer.BlockCopy(binData, 8, pixels, 0, width * height * 4);
        return (pixels, width, height);
    }

    // ──────────────────────────────────────────────────────────────
    // README model claims — compile/load verification
    // ──────────────────────────────────────────────────────────────

    /// <summary>MoveNet Lightning (ONNX) — README claims "Compiles (21 op types)".</summary>
    [TestMethod(Timeout = 60000)]
    public async Task CreateFromFile_MoveNet_Compiles() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        using var session = await InferenceSession.CreateFromFileAsync(
            accelerator, http, "models/movenet-lightning/model.onnx");
        Console.WriteLine($"[MoveNet] {session}");

        if (session.NodeCount < 10)
            throw new Exception($"Expected many nodes, got {session.NodeCount}");

        Console.WriteLine($"[MoveNet] PASS — {session.NodeCount} nodes, {session.WeightCount} weights");
    });

    /// <summary>
    /// MoveNet Lightning end-to-end inference test. The existing _Compiles test only
    /// verifies the model loads; this one runs full inference through the
    /// PoseEstimationPipeline against a synthetic image and validates the output is
    /// a valid pose tensor: [1, 1, 17, 3] = 51 floats, with x/y normalized to [0,1]
    /// and confidences in [0,1]. We don't expect a synthetic gradient image to
    /// produce a real person pose — we're verifying the inference pipeline runs
    /// to completion and produces structurally-valid output.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task CreateFromFile_MoveNet_Inference() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // CPU backend gate: the DemoConsole subprocess hosting CPU tests dies during
        // this test without emitting a TEST: json line — PMT surfaces it as a generic
        // "Test run failed" with no stack trace. The same model loads and runs cleanly
        // on WebGPU, WebGL, Wasm, CUDA, and OpenCL, so this is not a model-correctness
        // issue — it's an ILGPU CPU-backend interaction with one of MoveNet's
        // operators that needs deeper investigation. Tracked as follow-up. NOT marked
        // as "expected" — the skip exists because the underlying bug is unidentified,
        // not because CPU is incapable.
        if (accelerator.AcceleratorType == AcceleratorType.CPU)
            throw new UnsupportedTestException(
                "Tracked follow-up: CPU subprocess crashes during MoveNet inference without " +
                "emitting a TEST result. WebGPU/WebGL/Wasm/CUDA/OpenCL all pass. Investigate " +
                "ILGPU CPU backend with MoveNet's op set (NHWC layout, 21 op types).");

        using var session = await InferenceSession.CreateFromFileAsync(
            accelerator, http, "models/movenet-lightning/model.onnx");
        Console.WriteLine($"[MoveNet-Inference] {session}");

        // 256x256 gradient — gives the model some structure to anchor on without
        // requiring a pre-decoded RGBA bin file. The pipeline resizes to 192x192
        // internally so source dims aren't critical.
        const int W = 256, H = 256;
        var pixels = new int[W * H];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
            {
                int r = (int)(x * 255f / W);
                int g = (int)(y * 255f / H);
                int b = 128;
                pixels[y * W + x] = r | (g << 8) | (b << 16) | (0xFF << 24);
            }

        var pipeline = new PoseEstimationPipeline(session, accelerator);
        var result = await pipeline.EstimateAsync(pixels, W, H);

        Console.WriteLine($"[MoveNet-Inference] keypoints={result.Keypoints.Length}, infer={result.InferenceTimeMs:F0}ms");

        // Structural assertions - 17 standard COCO/MoveNet keypoints (nose, eyes, ears,
        // shoulders, elbows, wrists, hips, knees, ankles).
        if (result.Keypoints.Length != 17)
            throw new Exception($"Expected 17 keypoints, got {result.Keypoints.Length}");

        // Value-range checks. PoseSkeleton.DecodeMoveNetOutput scales y/x by image size,
        // so X should be in [0, W) and Y in [0, H). Confidence is normalized to [0,1].
        int validKeypoints = 0;
        foreach (var kp in result.Keypoints)
        {
            if (float.IsNaN(kp.X) || float.IsNaN(kp.Y) || float.IsNaN(kp.Confidence))
                throw new Exception($"NaN in keypoint '{kp.Name}': X={kp.X} Y={kp.Y} C={kp.Confidence}");
            if (kp.Confidence < 0f || kp.Confidence > 1f)
                throw new Exception($"Keypoint '{kp.Name}' confidence out of range: {kp.Confidence}");
            if (kp.Confidence > 0.05f) validKeypoints++;
        }

        // Sanity: a 17-element tensor that's all zeros would also pass the structural
        // checks. Verify SOME keypoint above the floor confidence so we know real values
        // came out of the model rather than uninitialized memory.
        if (validKeypoints == 0)
            throw new Exception("All keypoints have confidence < 0.05 — model output may be uninitialized");

        // Stronger structural assertions that catch the saturation bug we hit on
        // 2026-05-23: every Relu6 (Clip with opset 11+ tensor inputs) was silently
        // an identity op, so MobileNet activations exploded to 10^12 and the final
        // Sigmoid saturated everything to 0/1 with garbage coordinates. The old
        // "≥1 keypoint > 0.05" floor was satisfied by a single saturated 1.0 spike.
        int saturatedConf = result.Keypoints.Count(k => k.Confidence == 0f || k.Confidence == 1f);
        if (saturatedConf >= 15)
            throw new Exception($"Confidence saturated to 0/1 for {saturatedConf}/17 keypoints — " +
                                "inference is producing extreme outputs (Clip / activation explosion?).");

        // Coordinates must stay within image bounds. The OLD bug produced values
        // in the billions; any keypoint with X or Y outside [0, max(W, H)] means the
        // model is feeding non-normalized garbage into PoseSkeleton.DecodeMoveNetOutput.
        // Allow a small overshoot (1.5x) for keypoints predicted just off-edge.
        float coordLimit = Math.Max(W, H) * 1.5f;
        foreach (var kp in result.Keypoints)
        {
            if (Math.Abs(kp.X) > coordLimit || Math.Abs(kp.Y) > coordLimit)
                throw new Exception($"Keypoint '{kp.Name}' coordinates out of image range: " +
                                    $"X={kp.X} Y={kp.Y} (image {W}x{H}) — inference output is not in [0,1] normalized range.");
        }

        Console.WriteLine($"[MoveNet-Inference] PASS — {validKeypoints}/17 keypoints above 0.05 confidence, " +
                          $"{saturatedConf}/17 saturated (must stay < 15)");
        pipeline.Dispose();
    });

    /// <summary>
    /// DIAGNOSTIC: traces MoveNet's per-op outputs to identify where the keypoint
    /// decode produces values outside the [0, 1] normalized range. Captured per-op
    /// outputs are printed sorted by node index with their min/max so we can spot
    /// the first divergence into >1 magnitude. NOT a regression test - exists for
    /// the active pose-estimation root-cause investigation. Remove once fixed.
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task CreateFromFile_MoveNet_TraceIntermediates() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        // Browser backends already cover the same code path; only need one backend
        // for the trace. CUDA is fastest and most numerically reliable.
        if (accelerator.AcceleratorType != AcceleratorType.Cuda)
            throw new UnsupportedTestException("Trace only runs on CUDA - one backend is enough for op-by-op divergence hunting");

        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedOutputs = new Dictionary<string, float[]>();
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeInfo = new Dictionary<string, string>();

        try
        {
            using var session = await InferenceSession.CreateFromFileAsync(
                accelerator, http, "models/movenet-lightning/model.onnx");

            const int W = 192, H = 192;
            var pixels = new int[W * H];
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    pixels[y * W + x] = (int)(x * 255f / W) | ((int)(y * 255f / H) << 8) | (128 << 16) | (0xFF << 24);

            var pipeline = new PoseEstimationPipeline(session, accelerator);
            var result = await pipeline.EstimateAsync(pixels, W, H);
            pipeline.Dispose();

            var sb = new System.Text.StringBuilder();
            void Wl(string s) { Console.WriteLine(s); sb.AppendLine(s); }
            Wl($"=== MoveNet Final keypoints (raw decode) ===");
            foreach (var kp in result.Keypoints.Take(5))
                Wl($"  {kp.Name}: ({kp.X:F2}, {kp.Y:F2}) conf={kp.Confidence:F4}");

            Wl($"\n=== Per-op output ranges ({SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedOutputs.Count} captured) ===");
            var sorted = SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedOutputs
                .OrderBy(kvp =>
                {
                    var k = kvp.Key;
                    var us = k.IndexOf('_');
                    return us > 0 && int.TryParse(k.AsSpan(0, us), out var n) ? n : 99999;
                }).ToList();

            int firstHuge = -1;
            for (int i = 0; i < sorted.Count; i++)
            {
                var kvp = sorted[i];
                var vals = kvp.Value;
                if (vals == null || vals.Length == 0) continue;
                float mn = float.MaxValue, mx = float.MinValue;
                int nanCt = 0, infCt = 0;
                foreach (var v in vals)
                {
                    if (float.IsNaN(v)) nanCt++;
                    else if (float.IsInfinity(v)) infCt++;
                    else { if (v < mn) mn = v; if (v > mx) mx = v; }
                }
                bool huge = (Math.Abs(mn) > 1e5 || Math.Abs(mx) > 1e5);
                if (huge && firstHuge < 0) firstHuge = i;
                string flag = huge ? " <<HUGE>>" : "";
                string nflag = nanCt > 0 || infCt > 0 ? $" NaN={nanCt} Inf={infCt}" : "";
                // Print HUGE/NaN nodes always, plus nodes 100+ (output decode region).
                int nodeIdx = 99999;
                var us2 = kvp.Key.IndexOf('_');
                if (us2 > 0) int.TryParse(kvp.Key.AsSpan(0, us2), out nodeIdx);
                if (huge || nanCt > 0 || infCt > 0 || nodeIdx >= 100)
                    Wl($"  {kvp.Key,-100} min={mn,12:G5} max={mx,12:G5}{flag}{nflag}");
            }
            if (firstHuge >= 0)
                Wl($"\nFIRST_HUGE_NODE: {sorted[firstHuge].Key}");
            else
                Wl("\nNo huge values found - inference output range is normal");

        }
        finally
        {
            SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedOutputs = null;
            SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeInfo = null;
        }
    });

    /// <summary>EfficientNet-Lite0 (TFLite) — README claims it loads.</summary>
    [TestMethod(Timeout = 60000)]
    public async Task CreateFromFile_EfficientNetLite0_TFLite() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        using var session = await InferenceSession.CreateFromFileAsync(
            accelerator, http, "models/efficientnet-lite0/model.tflite");
        Console.WriteLine($"[EfficientNet] {session}");

        if (session.NodeCount < 10)
            throw new Exception($"Expected many nodes, got {session.NodeCount}");

        Console.WriteLine($"[EfficientNet] PASS — {session.NodeCount} nodes, {session.WeightCount} weights");
    });

    /// <summary>YOLOv8 Nano (ONNX) — README claims it loads.</summary>
    [TestMethod(Timeout = 60000)]
    public async Task CreateFromFile_YOLOv8Nano_ONNX() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        using var session = await InferenceSession.CreateFromFileAsync(
            accelerator, http, "models/yolov8n/model.onnx");
        Console.WriteLine($"[YOLOv8n] {session}");

        if (session.NodeCount < 10)
            throw new Exception($"Expected many nodes, got {session.NodeCount}");

        Console.WriteLine($"[YOLOv8n] PASS — {session.NodeCount} nodes, {session.WeightCount} weights");
    });

    /// <summary>
    /// The OPFS model cache handed back as a SEEKABLE JS-side stream (ModelHub.OpenStreamAsync), and a real
    /// session built from it — proving the browser load path never materialises the model on the .NET heap.
    /// </summary>
    /// <remarks>
    /// The gap this closes: <c>ModelCache</c> already reached the OPFS <c>File</c> (which IS a Blob) and then
    /// called <c>ArrayBuffer()</c> + <c>ReadBytes()</c>, copying the WHOLE model into the managed heap. The
    /// new path wraps that same File in a <c>BlobStream</c> instead.
    /// <para>
    /// The assertion that carries the claim is <c>stream is IJSReadStream</c>. That interface is not
    /// decoration: <c>BufferPool</c> tests for it to choose the zero-copy JS-&gt;GPU route, and
    /// <c>InferenceSession</c> arms <c>BrowserBufferPolicy.StrictHostCopyMaxBytes</c> ONLY for an
    /// IJSReadStream load — so a stream that merely seeks (HttpRangeStream, or a MemoryStream over a byte[])
    /// would satisfy a naive "it loaded" check while silently keeping every byte on the heap.
    /// </para>
    /// </remarks>
    [TestMethod(Timeout = 300000, Category = "HeavyModel", RetryCount = 2)]
    public async Task ModelHub_OpenStream_IsJSReadStream_AndLoadsASession() => await RunTest(async accelerator =>
    {
        var js = SpawnDev.SpawnJS.SpawnJSRuntime.Instance;
        if (js == null || !js.IsBrowser)
            throw new UnsupportedTestException("OPFS + BlobStream are browser-only (not a browser lane)");

        using var hub = new Hub.ModelHub(js);
        // ⚠️ NOT KnownModels.SqueezeNet + KnownFiles.OnnxModel — that pair 404s (verified: the repo exists,
            // the onnx/model.onnx path in it does not), and before the ModelCache status check landed a 404 body
            // was cached AS the model. This repo/file is the same one ModelInspector_Hub_InspectByUrl uses.
            var stream = await hub.OpenStreamAsync("onnx-community/mobilenetv3_small_100.lamb_in1k", "onnx/model.onnx");
        if (stream == null) throw new UnsupportedTestException("OPFS unavailable in this context");

        await using (stream)
        {
            // THE claim: JS-side, not merely seekable.
            if (stream is not SpawnDev.SpawnJS.Toolbox.IJSReadStream)
                throw new Exception($"OpenStreamAsync returned {stream.GetType().Name}, which is NOT an IJSReadStream "
                                  + "— BufferPool's zero-copy JS->GPU route cannot fire and the host-copy guard never arms");
            if (!stream.CanSeek) throw new Exception("stream is not seekable — the ONNX parser seeks back to every weight");
            if (stream.Length <= 0) throw new Exception($"stream length {stream.Length}");

            // And it must actually load. Byte-identity with the byte[] path is covered by
            // DepthAnything_HubBytes_MatchHttpBytes; this asserts the stream path produces a usable session.
            using var session = await InferenceSession.CreateFromOnnxStreamAsync(accelerator, stream);
            if (session.NodeCount <= 0) throw new Exception($"session NodeCount={session.NodeCount}");

            Console.WriteLine($"[ModelHub/stream] {stream.GetType().Name} ({stream.Length:N0} B, IJSReadStream) "
                            + $"-> session with {session.NodeCount} nodes, model never on the managed heap");
        }
    });
}
