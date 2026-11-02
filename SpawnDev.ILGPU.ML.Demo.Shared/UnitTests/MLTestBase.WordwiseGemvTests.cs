using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// The single-sequence engine's M = 1 decode GEMVs (word-wise <see cref="FusedDequantMatMul.ForwardRows"/> since 2026-10-31) against the per-type GEMVs
    /// before them (<see cref="FusedDequantMatMul.ForceGenericRows"/>): Qwen3-0.6B and Qwen2.5-0.5B (Q8_0), a fresh pipeline per arm (each answer is the first
    /// request after the model loads, as a chat app's is), greedy, WebGPU with decode capture on (the production chat path); the answers must be token
    /// for token the same. Added when SpawnDev.AI's qwen3 chat test answered its first request with an echo on ML 5.8.1-local.1.
    /// </summary>
    [TestMethod(Timeout = 1200000, Category = "HeavyModel")]
    public async Task GGUF_WordwiseGemv_TokenIdenticalToPerTypeGemv() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("the word-wise GEMVs run on CUDA, OpenCL and WebGPU");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var messages = new List<(string, string)> { ("user", "In one short sentence: what is a GPU?") };
        var notes = new List<string>();
        bool ok = true;
        bool saved = FusedDequantMatMul.ForceGenericRows;
        try
        {
            foreach (var (repo, file) in new[] { ("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"), ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf") })
            {
                async Task<string> Answer(bool generic)
                {
                    FusedDequantMatMul.ForceGenericRows = generic;
                    var m = await hub.OpenAsync(repo, file, deselect: false);
                    await using (m.Stream)
                    using (var pipe = await GgufTextGenerationPipeline.CreateFromStreamAsync(accelerator, m.Stream, maxSeqLen: 512, engine: SpawnDev.ILGPU.ML.Pipelines.GgufEngine.Generator))
                    {
                        if (accelerator.AcceleratorType == AcceleratorType.WebGPU) pipe.EnableWebGPUDecodeCapture = true;
                        return await pipe.GenerateAsync(messages, new GenerationConfig { MaxNewTokens = 48, Strategy = "greedy" }, thinking: false);
                    }
                }
                var perType = await Answer(true);
                var wordwise = await Answer(false);
                bool same = perType == wordwise;
                ok &= same;
                notes.Add($"{file}: {(same ? "identical" : "DIFFERS")} | per-type: {perType.Trim()[..Math.Min(90, perType.Trim().Length)]} | word-wise: {wordwise.Trim()[..Math.Min(90, wordwise.Trim().Length)]}");
            }
        }
        finally { FusedDequantMatMul.ForceGenericRows = saved; }
        // the actual cause of SpawnDev.AI's echo (2026-11-01): BuildChatPrompt(thinking: false) emits Qwen3's empty think block since 26d59426, and a
        // consumer that appends its own gets it twice. Shown here, not asserted as a gate: the doubled prompt's first answer, against the single one
        if (accelerator.AcceleratorType == AcceleratorType.Cuda)
        {
            var m = await hub.OpenAsync("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf", deselect: false);
            await using (m.Stream)
            {
                var gguf = await SpawnDev.ILGPU.ML.GGUF.GGUFParser.ParseHeaderAsync(m.Stream);
                var tok = SentencePieceTokenizer.FromGGUF(gguf)!;
                m.Stream.Seek(0, System.IO.SeekOrigin.Begin);
                using var session = await InferenceSession.CreateFromGGUFStreamAsync(accelerator, m.Stream);
                var gen = new GgufGenerator(session, accelerator, gguf, maxSeqLen: 512);
                var (single, stops) = ChatTemplates.BuildChatPrompt(gguf, tok, messages, thinking: false);
                tok.TryGetId("<think>", out var open); tok.TryGetId("</think>", out var close);
                var gap = tok.Encode("\n\n");
                var doubled = single.Concat(new[] { open }).Concat(gap).Concat(new[] { close }).Concat(gap).ToArray();
                var cfg = new GenerationConfig { MaxNewTokens = 32, Strategy = "greedy" };
                string Text(int[] ids) => tok.Decode(ids);
                var a = await gen.GenerateAsync(single, cfg, null, stops, null);
                var b = await gen.GenerateAsync(doubled, cfg, null, stops, null);
                notes.Add($"prompt with the think block once: '{a.Text.Trim()}' | twice (a consumer adding its own): '{b.Text.Trim()}'");
            }
        }
        var report = $"{accelerator.AcceleratorType}: " + string.Join(" || ", notes);
        Console.WriteLine("[WordwiseGemv] " + report);
        if (!ok) throw new Exception(report);
        return report;
    });
}
