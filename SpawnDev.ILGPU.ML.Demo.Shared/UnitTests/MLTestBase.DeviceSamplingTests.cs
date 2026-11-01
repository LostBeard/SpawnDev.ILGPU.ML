using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// The single-sequence engine's device draw (<see cref="GgufGenerator.DeviceSampling"/>: penalty, candidates and draw on the device, one int read per
    /// token) against the host draw (<see cref="TextGenerationSampler"/> on the logits read back): Qwen2.5-0.5B and Qwen3-0.6B (Q8_0), seeded top-p (the
    /// /ai-chat config), seeded top-k, and greedy with a repetition penalty; the same text on CUDA, OpenCL and WebGPU (decode capture on, the production
    /// chat path). Also prints each arm's decode rate (the second request of a pipeline, so the first-request kernel compiles are out of it).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task GGUF_DeviceSampling_MatchesHostSampler() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("the device-sampling gate runs on the fast lanes (CUDA, OpenCL, WebGPU)");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var warm = new List<(string, string)> { ("user", "Say hello.") };
        var messages = new List<(string, string)> { ("user", "Write a short paragraph about why the sky is blue.") };
        var configs = new (string Name, GenerationConfig Cfg)[]
        {
            ("top_p", new GenerationConfig { MaxNewTokens = 96, Strategy = "top_p", TopP = 0.9f, Temperature = 0.7f, RepetitionPenalty = 1.15f, Seed = 1234 }),
            ("top_k", new GenerationConfig { MaxNewTokens = 96, Strategy = "top_k", TopK = 40, Temperature = 0.8f, RepetitionPenalty = 1.1f, Seed = 7 }),
            ("greedy+penalty", new GenerationConfig { MaxNewTokens = 96, Strategy = "greedy", RepetitionPenalty = 1.3f }),
        };
        var notes = new List<string>();
        bool ok = true;
        foreach (var (repo, file) in new[] { ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf"), ("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf") })
        {
            (int Device, int Fallbacks) counts = default;
            double rate = 0, greedyRate = 0;
            async Task<List<(string Text, int Tokens, double Ms)>> Arm(bool device)
            {
                var m = await hub.OpenAsync(repo, file, deselect: false);
                await using (m.Stream)
                using (var pipe = await GgufTextGenerationPipeline.CreateFromStreamAsync(accelerator, m.Stream, maxSeqLen: 512))
                {
                    if (accelerator.AcceleratorType == AcceleratorType.WebGPU) pipe.EnableWebGPUDecodeCapture = true;
                    pipe.DeviceSampling = device;
                    var outs = new List<(string, int, double)>();
                    // warm: the decode graph, the capture and the sampler kernels compile here, not inside the timed requests
                    await pipe.GenerateAsync(warm, new GenerationConfig { MaxNewTokens = 8, Strategy = "greedy" }, thinking: false);
                    await pipe.GenerateAsync(warm, new GenerationConfig { MaxNewTokens = 8, Strategy = "top_p", TopP = 0.9f, Temperature = 0.7f, RepetitionPenalty = 1.1f, Seed = 1 }, thinking: false);
                    foreach (var (_, cfg) in configs)
                    {
                        int n = 0;
                        var sw = Stopwatch.StartNew();
                        var text = await pipe.GenerateAsync(messages, cfg, thinking: false, onToken: (_, _) => { n++; return Task.CompletedTask; });
                        sw.Stop();
                        outs.Add((text, n, sw.Elapsed.TotalMilliseconds));
                    }
                    // the decode rate: a long top-p + penalty request (the /ai-chat config), first token to last (no prefill), best of two
                    rate = 0;
                    var longCfg = new GenerationConfig { MaxNewTokens = 256, Strategy = "top_p", TopP = 0.9f, Temperature = 0.7f, RepetitionPenalty = 1.15f, Seed = 99 };
                    var story = new List<(string, string)> { ("user", "Write a long story about a lighthouse keeper and a storm.") };
                    for (int rep = 0; rep < 2; rep++)
                    {
                        int n = 0; var sw = new Stopwatch(); double firstMs = 0;
                        sw.Start();
                        await pipe.GenerateAsync(story, longCfg, thinking: false, onToken: (_, _) => { if (n++ == 0) firstMs = sw.Elapsed.TotalMilliseconds; return Task.CompletedTask; });
                        double decodeMs = sw.Elapsed.TotalMilliseconds - firstMs;
                        if (n > 1) rate = Math.Max(rate, (n - 1) * 1000.0 / decodeMs);
                    }
                    // the reference: plain greedy (the capture's folded argmax, one fence) on the same prompt
                    greedyRate = 0;
                    for (int rep = 0; rep < 2; rep++)
                    {
                        int n = 0; var sw = Stopwatch.StartNew(); double firstMs = 0;
                        await pipe.GenerateAsync(story, new GenerationConfig { MaxNewTokens = 256, Strategy = "greedy" }, thinking: false, onToken: (_, _) => { if (n++ == 0) firstMs = sw.Elapsed.TotalMilliseconds; return Task.CompletedTask; });
                        if (n > 1) greedyRate = Math.Max(greedyRate, (n - 1) * 1000.0 / (sw.Elapsed.TotalMilliseconds - firstMs));
                    }
                    counts = pipe.DeviceSamplerCounts;
                    return outs;
                }
            }
            var host = await Arm(false);
            double hostRate = rate;
            if (counts.Device != 0) { ok = false; notes.Add($"{file}: the host arm drew {counts.Device} tokens on the device"); }
            var dev = await Arm(true);
            double devRate = rate;
            notes.Add($"{file} decode rate (top_p + penalty, 256 tokens): host draw {hostRate:F1} tok/s, device draw {devRate:F1} tok/s (greedy, no penalty: {greedyRate:F1})");
            int devTokens = dev.Sum(d => d.Tokens);
            // the device arm must have drawn its tokens on the device (a silent host fallback would make the comparison trivial)
            if (counts.Device < devTokens - counts.Fallbacks) { ok = false; notes.Add($"{file}: only {counts.Device} of {devTokens} tokens drawn on the device ({counts.Fallbacks} fallbacks)"); }
            notes.Add($"{file}: device draws {counts.Device}, host fallbacks {counts.Fallbacks}");
            for (int c = 0; c < configs.Length; c++)
            {
                bool same = host[c].Text == dev[c].Text;
                ok &= same;
                int diverge = 0;
                while (diverge < Math.Min(host[c].Text.Length, dev[c].Text.Length) && host[c].Text[diverge] == dev[c].Text[diverge]) diverge++;
                notes.Add($"{file} {configs[c].Name}: {(same ? "identical" : $"DIFFERS at char {diverge}")} host {host[c].Ms:F0} ms / device {dev[c].Ms:F0} ms" +
                    (same ? $" | {Clip(dev[c].Text)}" : $" | host: {Clip(host[c].Text[Math.Max(0, diverge - 20)..])} | device: {Clip(dev[c].Text[Math.Max(0, diverge - 20)..])}"));
            }
        }
        var report = $"{accelerator.AcceleratorType}: " + string.Join(" || ", notes);
        Console.WriteLine("[DeviceSampling] " + report);
        if (!ok) throw new Exception(report);
        return report;

        static string Clip(string s) { s = s.Trim().Replace('\n', ' '); return s[..Math.Min(80, s.Length)]; }
    });
}
