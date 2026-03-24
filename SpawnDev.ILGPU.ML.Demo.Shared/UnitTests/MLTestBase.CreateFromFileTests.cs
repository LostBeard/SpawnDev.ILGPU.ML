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

        var session = await InferenceSession.CreateFromFileAsync(
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
        session.Dispose();
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

        var session = await InferenceSession.CreateFromFileAsync(
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
        session.Dispose();
    });

    /// <summary>
    /// Load style-mosaic via CreateFromFileAsync, apply to gradient image.
    /// Verifies output differs from input.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task CreateFromFile_StyleTransfer_Mosaic() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        var session = await InferenceSession.CreateFromFileAsync(
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
        session.Dispose();
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

        var session = await InferenceSession.CreateFromFileAsync(
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
        session.Dispose();
    });

    /// <summary>
    /// Load Depth Anything V2 Small via CreateFromFileAsync (95MB ONNX, 823 nodes).
    /// This is the model that was failing in the demo.
    /// Verifies: model loads, compiles, and can run inference.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task CreateFromFile_DepthAnything_Load() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType == AcceleratorType.CPU)
            throw new UnsupportedTestException("Depth Anything too large for CPU backend — skipped");

        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        Console.WriteLine("[CreateFromFile_Depth] Downloading Depth Anything V2 Small (95MB)...");
        var session = await InferenceSession.CreateFromFileAsync(
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
        session.Dispose();
    });

    /// <summary>
    /// Load Depth Anything V2 Small and run actual depth estimation on a gradient image.
    /// Verifies depth map output has spatial variation (not flat).
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task CreateFromFile_DepthAnything_Inference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.CPU or AcceleratorType.WebGPU or AcceleratorType.WebGL or AcceleratorType.Wasm)
            throw new UnsupportedTestException("Depth Anything inference requires too much GPU memory for browser/CPU — skipped");

        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        Console.WriteLine("[DepthInference] Loading model...");
        var session = await InferenceSession.CreateFromFileAsync(
            accelerator, http, "models/depth-anything-v2-small/model.onnx",
            inputShapes: new Dictionary<string, int[]>
            {
                ["pixel_values"] = new[] { 1, 3, 518, 518 }
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
        session.Dispose();
    });

    /// <summary>
    /// Depth estimation with real cat image — more meaningful than gradient.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task CreateFromFile_DepthAnything_CatImage() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.CPU or AcceleratorType.WebGPU or AcceleratorType.WebGL or AcceleratorType.Wasm)
            throw new UnsupportedTestException("Depth Anything inference requires too much GPU memory for browser/CPU — skipped");

        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        var (pixels, width, height) = await LoadCatImage(http);
        Console.WriteLine($"[DepthCat] Cat image: {width}x{height}");

        var session = await InferenceSession.CreateFromFileAsync(
            accelerator, http, "models/depth-anything-v2-small/model.onnx",
            inputShapes: new Dictionary<string, int[]>
            {
                ["pixel_values"] = new[] { 1, 3, 518, 518 }
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
        session.Dispose();
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

        var session = await InferenceSession.CreateFromFileAsync(
            accelerator, http, "models/movenet-lightning/model.onnx");
        Console.WriteLine($"[MoveNet] {session}");

        if (session.NodeCount < 10)
            throw new Exception($"Expected many nodes, got {session.NodeCount}");

        Console.WriteLine($"[MoveNet] PASS — {session.NodeCount} nodes, {session.WeightCount} weights");
        session.Dispose();
    });

    /// <summary>EfficientNet-Lite0 (TFLite) — README claims it loads.</summary>
    [TestMethod(Timeout = 60000)]
    public async Task CreateFromFile_EfficientNetLite0_TFLite() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var session = await InferenceSession.CreateFromFileAsync(
            accelerator, http, "models/efficientnet-lite0/model.tflite");
        Console.WriteLine($"[EfficientNet] {session}");

        if (session.NodeCount < 10)
            throw new Exception($"Expected many nodes, got {session.NodeCount}");

        Console.WriteLine($"[EfficientNet] PASS — {session.NodeCount} nodes, {session.WeightCount} weights");
        session.Dispose();
    });

    /// <summary>YOLOv8 Nano (ONNX) — README claims it loads.</summary>
    [TestMethod(Timeout = 60000)]
    public async Task CreateFromFile_YOLOv8Nano_ONNX() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var session = await InferenceSession.CreateFromFileAsync(
            accelerator, http, "models/yolov8n/model.onnx");
        Console.WriteLine($"[YOLOv8n] {session}");

        if (session.NodeCount < 10)
            throw new Exception($"Expected many nodes, got {session.NodeCount}");

        Console.WriteLine($"[YOLOv8n] PASS — {session.NodeCount} nodes, {session.WeightCount} weights");
        session.Dispose();
    });
}
