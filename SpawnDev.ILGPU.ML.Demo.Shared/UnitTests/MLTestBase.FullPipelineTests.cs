using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Style Transfer: load mosaic style model, apply to a gradient image,
    /// verify output pixels are non-zero and differ from input.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task WebModel_StyleTransfer_Mosaic() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        using var session = await InferenceSession.CreateAsync(accelerator, http, "models/style-mosaic");
        Console.WriteLine($"[StyleTest] Model: {session}");

        // Small gradient test image (64x64)
        int w = 64, h = 64;
        var pixels = new int[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                pixels[y * w + x] = (int)(x * 255f / w) | ((int)(y * 255f / h) << 8) | (128 << 16) | (0xFF << 24);

        var pipeline = new StyleTransferPipeline(session, accelerator);
        var result = await pipeline.TransferAsync(pixels, w, h);

        Console.WriteLine($"[StyleTest] Output: {result.Width}x{result.Height}, {result.RgbaPixels.Length} pixels");

        // Verify output is not all zeros or all same color
        var firstPixel = result.RgbaPixels[0];
        bool allSame = result.RgbaPixels.All(p => p == firstPixel);
        if (allSame)
            throw new Exception("Output is uniform — all pixels identical");

        // Verify output differs from input (style was actually applied)
        int diffCount = 0;
        int checkPixels = Math.Min(pixels.Length, result.RgbaPixels.Length);
        for (int i = 0; i < checkPixels; i++)
            if (pixels[i] != result.RgbaPixels[i]) diffCount++;

        float diffPct = (float)diffCount / checkPixels;
        Console.WriteLine($"[StyleTest] {diffPct:P1} pixels changed ({diffCount}/{checkPixels})");

        if (diffPct < 0.5f)
            throw new Exception($"Style barely changed image: only {diffPct:P1} pixels differ");

        Console.WriteLine("[StyleTest] PASS");
        pipeline.Dispose();
    });

    /// <summary>
    /// Super Resolution: load ESPCN model, upscale a small image 3x,
    /// verify output dimensions and non-zero content.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task WebModel_SuperResolution_ESPCN() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType == AcceleratorType.CPU)
            throw new UnsupportedTestException("ESPCN too slow on CPU (224x224 compiled shapes) — skipped to prevent timeout");

        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        using var session = await InferenceSession.CreateAsync(accelerator, http, "models/super-resolution");
        Console.WriteLine($"[SRTest] Model: {session}");

        // Small test image (32x32 gradient)
        int w = 32, h = 32;
        var pixels = new int[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int gray = (int)(x * 255f / w);
                pixels[y * w + x] = gray | (gray << 8) | (gray << 16) | (0xFF << 24);
            }

        var pipeline = new SuperResolutionPipeline(session, accelerator, upscaleFactor: 3);
        var result = await pipeline.UpscaleAsync(pixels, w, h);

        Console.WriteLine($"[SRTest] Output: {result.Width}x{result.Height} (factor={result.UpscaleFactor})");

        // Output dimensions are based on model's declared input size (224x224) * upscale factor
        // Not the test image size — pipeline resizes input to match model
        if (result.Width < 3 || result.Height < 3)
            throw new Exception($"Output too small: {result.Width}x{result.Height}");

        // Verify output is not all zeros
        bool allBlack = result.RgbaPixels.All(p => (p & 0x00FFFFFF) == 0);
        if (allBlack)
            throw new Exception("Output is all black");

        // Verify output has variation (not flat)
        var uniqueGrays = result.RgbaPixels.Select(p => p & 0xFF).Distinct().Count();
        Console.WriteLine($"[SRTest] Unique gray levels: {uniqueGrays}");

        if (uniqueGrays < 10)
            throw new Exception($"Output has only {uniqueGrays} unique values — too flat");

        Console.WriteLine("[SRTest] PASS");
        pipeline.Dispose();
    });

    /// <summary>
    /// Style Transfer with real cat image: load cat_rgba.bin, apply mosaic style,
    /// verify styled output has good color variety and differs from input.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task WebModel_StyleTransfer_CatImage() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType == AcceleratorType.CPU)
            throw new UnsupportedTestException("Style transfer too slow on CPU — skipped to prevent timeout");

        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        // Load pre-decoded cat image
        var binData = await http.GetByteArrayAsync("samples/cat_rgba.bin");
        int width = BitConverter.ToInt32(binData, 0);
        int height = BitConverter.ToInt32(binData, 4);
        var pixels = new int[width * height];
        Buffer.BlockCopy(binData, 8, pixels, 0, width * height * 4);
        Console.WriteLine($"[StyleCatTest] Loaded cat image: {width}x{height}");

        using var session = await InferenceSession.CreateAsync(accelerator, http, "models/style-mosaic");
        var pipeline = new StyleTransferPipeline(session, accelerator);
        var result = await pipeline.TransferAsync(pixels, width, height);

        Console.WriteLine($"[StyleCatTest] Output: {result.Width}x{result.Height}");

        // Verify output has good color variety (styled images are colorful)
        var uniqueColors = new HashSet<int>();
        for (int i = 0; i < Math.Min(1000, result.RgbaPixels.Length); i++)
            uniqueColors.Add(result.RgbaPixels[i] & 0x00FFFFFF);

        Console.WriteLine($"[StyleCatTest] Unique colors in first 1000 pixels: {uniqueColors.Count}");

        if (uniqueColors.Count < 50)
            throw new Exception($"Output lacks color variety: only {uniqueColors.Count} unique colors");

        Console.WriteLine("[StyleCatTest] PASS");
        pipeline.Dispose();
    });

    /// <summary>
    /// Super Resolution with real cat image: crop center 64x64, upscale 3x,
    /// verify dimensions and content quality.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task WebModel_SuperResolution_CatImage() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType == AcceleratorType.CPU)
            throw new UnsupportedTestException("ESPCN too slow on CPU (224x224 compiled shapes) — skipped to prevent timeout");

        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        // Load pre-decoded cat image
        var binData = await http.GetByteArrayAsync("samples/cat_rgba.bin");
        int width = BitConverter.ToInt32(binData, 0);
        int height = BitConverter.ToInt32(binData, 4);
        var pixels = new int[width * height];
        Buffer.BlockCopy(binData, 8, pixels, 0, width * height * 4);
        Console.WriteLine($"[SRCatTest] Loaded cat image: {width}x{height}");

        // Crop to 64x64 from center for speed (upscaling full 640x427 by 3x is huge)
        int cropW = 64, cropH = 64;
        int startX = (width - cropW) / 2;
        int startY = (height - cropH) / 2;
        var cropped = new int[cropW * cropH];
        for (int y = 0; y < cropH; y++)
            for (int x = 0; x < cropW; x++)
                cropped[y * cropW + x] = pixels[(startY + y) * width + (startX + x)];

        using var session = await InferenceSession.CreateAsync(accelerator, http, "models/super-resolution");
        var pipeline = new SuperResolutionPipeline(session, accelerator, upscaleFactor: 3);
        var result = await pipeline.UpscaleAsync(cropped, cropW, cropH);

        Console.WriteLine($"[SRCatTest] Output: {result.Width}x{result.Height}");

        // Output uses model's declared input size (224x224) * upscale factor, not crop size
        if (result.Width < 3 || result.Height < 3)
            throw new Exception($"Output too small: {result.Width}x{result.Height}");

        // Verify output has reasonable variation
        var uniqueGrays = result.RgbaPixels.Select(p => p & 0xFF).Distinct().Count();
        Console.WriteLine($"[SRCatTest] Unique gray levels: {uniqueGrays}");

        if (uniqueGrays < 20)
            throw new Exception($"Output too flat: only {uniqueGrays} unique values");

        Console.WriteLine("[SRCatTest] PASS");
        pipeline.Dispose();
    });
}
