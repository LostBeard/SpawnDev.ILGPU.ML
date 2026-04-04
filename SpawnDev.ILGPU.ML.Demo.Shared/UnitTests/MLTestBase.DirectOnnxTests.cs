using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Direct .onnx loading via HTTP: load SqueezeNet .onnx file, classify gradient image,
    /// verify non-uniform output. Tests the native ONNX parser path on all backends.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task DirectOnnx_SqueezeNet_ViaHttp() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        // Load model directly from .onnx via HTTP (no JSON extraction)
        using var session = await InferenceSession.CreateFromOnnxAsync(accelerator, http, "models/squeezenet/model.onnx");
        Console.WriteLine($"[DirectOnnxHttp] {session}");

        // Gradient test image
        int w = 64, h = 64;
        var pixels = new int[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                pixels[y * w + x] = (int)(x * 255f / w) | ((int)(y * 255f / h) << 8) | (128 << 16) | (0xFF << 24);

        var pipeline = new ClassificationPipeline(session, accelerator);
        var results = await pipeline.ClassifyAsync(pixels, w, h, 10);

        Console.WriteLine($"[DirectOnnxHttp] Top-5:");
        foreach (var r in results.Take(5))
            Console.WriteLine($"  {r.Label} ({r.Confidence:P2}, class {r.ClassIndex})");

        float ratio = results[0].Confidence / Math.Max(results[^1].Confidence, 1e-10f);
        if (ratio < 1.5f)
            throw new Exception($"Output uniform: ratio={ratio:F2}x");

        Console.WriteLine($"[DirectOnnxHttp] PASS — ratio={ratio:F1}x");
        pipeline.Dispose();
    });
}
