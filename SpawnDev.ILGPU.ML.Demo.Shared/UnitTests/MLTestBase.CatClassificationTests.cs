using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Loads the actual cat.jpg sample image (pre-decoded as cat_rgba.bin),
    /// runs it through SqueezeNet ClassificationPipeline, and verifies
    /// the top predictions include cat-related ImageNet classes.
    /// ImageNet cat classes: tabby=281, tiger_cat=282, Persian=283, Siamese=284, Egyptian=285
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task WebModel_SqueezeNet_CatClassification() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        // Load pre-decoded RGBA image (cat_rgba.bin = 8-byte header + raw RGBA pixels)
        var binData = await http.GetByteArrayAsync("samples/cat_rgba.bin");
        int width = BitConverter.ToInt32(binData, 0);
        int height = BitConverter.ToInt32(binData, 4);
        var pixels = new int[width * height];
        Buffer.BlockCopy(binData, 8, pixels, 0, width * height * 4);
        Console.WriteLine($"[CatTest] Loaded cat image: {width}x{height}, {pixels.Length} pixels");

        // Load model
        var session = await InferenceSession.CreateAsync(accelerator, http, "models/squeezenet");
        Console.WriteLine($"[CatTest] Model: {session}");

        // Run classification
        var pipeline = new ClassificationPipeline(session, accelerator);
        var results = await pipeline.ClassifyAsync(pixels, width, height, 10);

        Console.WriteLine($"[CatTest] Top-10:");
        foreach (var r in results)
            Console.WriteLine($"  {r.Label} ({r.Confidence:P2}, class {r.ClassIndex})");

        // Verify output is not uniform
        float topConf = results[0].Confidence;
        float botConf = results[^1].Confidence;
        float ratio = topConf / Math.Max(botConf, 1e-10f);
        Console.WriteLine($"[CatTest] Top/bottom ratio: {ratio:F1}x");

        if (ratio < 1.5f)
            throw new Exception($"Output uniform: top={topConf:P4}, bot={botConf:P4}, ratio={ratio:F1}x");

        // Verify top-10 contains at least one cat class (ImageNet 281-285)
        var catClasses = new HashSet<int> { 281, 282, 283, 284, 285 };
        bool foundCat = results.Any(r => catClasses.Contains(r.ClassIndex));

        if (!foundCat)
        {
            var topClasses = string.Join(", ", results.Select(r => $"{r.ClassIndex}:{r.Label}"));
            throw new Exception($"No cat class in top-10. Got: [{topClasses}]");
        }

        var catResult = results.First(r => catClasses.Contains(r.ClassIndex));
        Console.WriteLine($"[CatTest] PASS: Found '{catResult.Label}' (class {catResult.ClassIndex}) at {catResult.Confidence:P2}");

        pipeline.Dispose();
    });
}
