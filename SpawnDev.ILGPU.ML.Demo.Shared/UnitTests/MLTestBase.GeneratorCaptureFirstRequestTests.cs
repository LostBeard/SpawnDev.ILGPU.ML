using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="GgufGenerator"/>'s first request after a load with WebGPU decode capture on (the request that builds the capture) against the same request
    /// with capture off, then a second request on each: SmolLM2-360M (llama), the model whose first captured answer came out "1111..." on WebGPU
    /// (2026-11-01, found by GgufTextGenerationPipeline_BatchEngine_MatchesGenerator), and Qwen2.5-0.5B (qwen2) beside it. Greedy, token for token.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task GgufGenerator_WebGPUCapture_FirstRequestMatchesDirect() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU) throw new UnsupportedTestException("WebGPU decode capture");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var q = new List<(string, string)> { ("user", "Name three primary colors and say why they are called primary.") };
        var q2 = new List<(string, string)> { ("user", "What is the capital of France?") };
        var cfg = new GenerationConfig { MaxNewTokens = 32, Strategy = "greedy" };
        var notes = new List<string>();
        bool ok = true;
        foreach (var (repo, file) in new[] { ("HuggingFaceTB/SmolLM2-360M-Instruct-GGUF", "smollm2-360m-instruct-q8_0.gguf"), ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf") })
        {
            async Task<(string First, string Second, string Info)> Run(bool capture)
            {
                var m = await hub.OpenAsync(repo, file, deselect: false);
                await using (m.Stream)
                using (var pipe = await GgufTextGenerationPipeline.CreateFromStreamAsync(accelerator, m.Stream, maxSeqLen: 1024, engine: GgufEngine.Generator))
                {
                    pipe.EnableWebGPUDecodeCapture = capture;
                    string a = await pipe.GenerateAsync(q, cfg, thinking: false);
                    var info = pipe.DecodeCaptureInfo;
                    string b = await pipe.GenerateAsync(q2, cfg, thinking: false);
                    return (a, b, info is { } i ? $"capture ops {i.Ops}, scalars {i.Scalars}, copies {i.Copies}, slots {i.Slots}" : "no capture");
                }
            }
            var direct = await Run(false);
            var cap = await Run(true);
            bool first = cap.First == direct.First, second = cap.Second == direct.Second;
            ok &= first && second;
            notes.Add($"{file}: first request {(first ? "same" : "DIFFERS")}, second {(second ? "same" : "DIFFERS")} ({cap.Info})" +
                (first ? "" : $" | direct: {Clip(direct.First)} | captured: {Clip(cap.First)}") + (second ? "" : $" | direct 2: {Clip(direct.Second)} | captured 2: {Clip(cap.Second)}"));
        }
        var report = string.Join(" || ", notes);
        Console.WriteLine("[CaptureFirst] " + report);
        if (!ok) throw new Exception(report);
        return report;

        static string Clip(string s) { s = s.Replace('\n', ' '); return s[..Math.Min(70, s.Length)]; }
    });
}
