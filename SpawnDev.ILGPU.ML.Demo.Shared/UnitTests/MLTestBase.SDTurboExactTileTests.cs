using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// The exact tiled VAE decode (<see cref="ImageGenerationPipeline.VaeTileGrid"/> &gt; 0) at the sizes SD-Turbo draws: 256x256 (latent 32x32, which threw "mid-block output must be 512x64x64" because the
    /// tiled decoder hard-coded a 512 picture) and 512x512, each against the full single-pass decode of the same seed. The two pictures must agree to a few levels of 255 (the full decode runs
    /// the activations in the pipeline's precision, the tiled one in fp32). Desktop GPU; streams the 2.5 GB SD-Turbo set from the models host. Red check: a decoder that ignores the latent size throws at 256.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task SDTurbo_ExactTiledVae_MatchesFullDecode_At256And512() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.WebGPU or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("desktop GPU (the exact tiled decode is checked on CUDA / OpenCL)");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        using var pipe = await ImageGenerationPipeline.CreateAsync(accelerator, hub, Hub.ModelHub.KnownModels.SDTurbo);
        if (!pipe.IsReady) throw new Exception("SD-Turbo pipeline not ready after CreateAsync");
        pipe.NumInferenceSteps = 1; pipe.GuidanceScale = 0f; pipe.Seed = 42;
        var report = new List<string>();
        foreach (int size in new[] { 256, 512 })
        {
            pipe.Width = size; pipe.Height = size;
            pipe.VaeTileGrid = -1;
            var full = await pipe.RunAsync(new ImageGenerationInput { Prompt = "a photo of a cat" });
            pipe.VaeTileGrid = 2;
            var tiled = await pipe.RunAsync(new ImageGenerationInput { Prompt = "a photo of a cat" });
            if (full.Width != size || tiled.Width != size) throw new Exception($"expected {size} px, got {full.Width} / {tiled.Width}");
            double sum = 0; int worst = 0, n = size * size;
            for (int i = 0; i < n; i++)
                for (int c = 0; c < 3; c++) { int d = Math.Abs(full.ImageRGBA[i * 4 + c] - tiled.ImageRGBA[i * 4 + c]); sum += d; worst = Math.Max(worst, d); }
            double mean = sum / (3.0 * n);
            report.Add($"{size}: mean abs difference {mean:F2} levels, worst {worst}");
            Console.WriteLine($"[SDTurboTile] {size}x{size} exact 2x2 tiled vs full decode: mean abs difference {mean:F2} levels of 255, worst {worst}");
            if (!(mean < 3.0)) throw new Exception($"the tiled decode differs from the full decode at {size}: mean {mean:F2} levels, worst {worst}");
        }
        return string.Join("; ", report);
    });
}
