using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Full InferenceSession.CreateAsync test: load SqueezeNet via HTTP,
    /// run ClassificationPipeline, verify non-uniform output.
    /// Requires HttpClient from subclass (browser: DI, desktop: file server).
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task WebModel_SqueezeNet_FullPipeline() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        // Load model via InferenceSession.CreateAsync (the browser path)
        var session = await InferenceSession.CreateAsync(accelerator, http, "models/squeezenet");
        Console.WriteLine($"[WebModel] {session}");
        Console.WriteLine($"[WebModel] Input names: {string.Join(", ", session.InputNames)}");
        Console.WriteLine($"[WebModel] Output names: {string.Join(", ", session.OutputNames)}");

        // Create test image (gradient)
        int w = 64, h = 64;
        var pixels = new int[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                pixels[y * w + x] = (int)(x * 255f / w) | ((int)(y * 255f / h) << 8) | (128 << 16) | (0xFF << 24);

        // Run classification
        var pipeline = new SpawnDev.ILGPU.ML.Pipelines.ClassificationPipeline(session, accelerator);
        var results = await pipeline.ClassifyAsync(pixels, w, h, 10);

        Console.WriteLine($"[WebModel] Top-10:");
        foreach (var r in results)
            Console.WriteLine($"  {r.Label} ({r.Confidence:P2}, class {r.ClassIndex})");

        float topConf = results[0].Confidence;
        float botConf = results[^1].Confidence;
        float ratio = topConf / Math.Max(botConf, 1e-10f);
        float totalProb = results.Sum(r => r.Confidence);
        Console.WriteLine($"[WebModel] Ratio: {ratio:F2}x, top-10 sum: {totalProb:P2}");

        if (ratio < 1.5f)
            throw new Exception($"Output uniform: top={topConf:P4}, bot={botConf:P4}, ratio={ratio:F2}x");

        session.Dispose();
        pipeline.Dispose();
    });
}
