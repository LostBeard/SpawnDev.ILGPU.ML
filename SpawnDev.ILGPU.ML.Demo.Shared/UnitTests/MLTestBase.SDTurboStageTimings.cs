using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Where SD-Turbo's time goes on this device, from the pipeline's progress events: the text encoder (cold: includes the kernel compiles and weight upload of the first run), each UNet step and the
    /// VAE decode, for a cold 1-step run, a warm 1-step run with another seed and a warm 4-step run, at 512x512. Prints a table (<c>[SDTurboTimings]</c> lines); fails only when a stage is missing.
    /// A perf data point for comparing devices (an RX 590 on OpenCL measured UNet 7 s per step, VAE decode 13 s). Desktop GPU; streams the 2.5 GB set from the models host.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task SDTurbo_StageTimings() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("GPU lanes only");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var load = System.Diagnostics.Stopwatch.StartNew();
        using var pipe = await ImageGenerationPipeline.CreateAsync(accelerator, new ModelsHostStreamLoader(http), Hub.ModelHub.KnownModels.SDTurbo);
        double loadMs = load.Elapsed.TotalMilliseconds;
        pipe.GuidanceScale = 0f; pipe.Width = 512; pipe.Height = 512;
        var lines = new List<string>();
        foreach (var (label, steps, seed) in new[] { ("cold, 1 step", 1, 42), ("warm, 1 step, new seed", 1, 43), ("warm, 4 steps", 4, 44) })
        {
            pipe.NumInferenceSteps = steps; pipe.Seed = seed;
            var stamps = new List<(int Step, double Ms)>();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            void Handler(int step, int total) => stamps.Add((step, sw.Elapsed.TotalMilliseconds));
            pipe.OnProgress += Handler;
            try { await pipe.RunAsync(new ImageGenerationInput { Prompt = "a photo of a red apple on a wooden table" }); }
            finally { pipe.OnProgress -= Handler; }
            double total = sw.Elapsed.TotalMilliseconds;
            // events: 0 after the text encoder, 1..steps after each UNet step, steps + 1 after the UNet loop, steps + 2 after the VAE decode
            double Ms(int step) => stamps.FirstOrDefault(s => s.Step == step).Ms;
            if (stamps.Count < steps + 2) throw new Exception($"{label}: only {stamps.Count} progress events");
            double text = Ms(0), unet = (Ms(steps) - Ms(0)), vae = total - Ms(steps);
            lines.Add($"{label}: text encode {text:F0} ms, UNet {unet:F0} ms ({unet / steps:F0} ms per step), VAE decode and the rest {vae:F0} ms, total {total:F0} ms");
        }
        Console.WriteLine($"[SDTurboTimings] {accelerator.Name} ({accelerator.AcceleratorType}) 512x512: checkpoint load {loadMs:F0} ms");
        foreach (var l in lines) Console.WriteLine("[SDTurboTimings] " + l);
        return string.Join(" | ", lines);
    });
}
