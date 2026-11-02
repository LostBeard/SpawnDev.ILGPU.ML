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
    /// A measurement, not a gate: one chat answer's decode rate on the engine /ai-chat uses (<see cref="GgufTextGenerationPipeline"/>, decode capture on
    /// WebGPU, device draws) against <see cref="GgufBatchDecoder"/> with one sequence (device draws), in the same run, same lane, same model, prompt and
    /// sampler settings (top-p 0.9, T 0.7, penalty 1.15, 256 tokens), first token to last, best of two warm runs each. Decides whether the chat page should
    /// move to the batch decoder (it lacks the pipeline's multi-turn prefix cache, so it must be clearly faster per token to pay for that).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task GGUF_SingleSequence_PipelineVsBatchDecoder_Rate() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("a measurement for the fast lanes (CUDA, OpenCL, WebGPU)");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        const string repo = "Qwen/Qwen2.5-0.5B-Instruct-GGUF", file = "qwen2.5-0.5b-instruct-q8_0.gguf";
        var messages = new List<(string, string)> { ("user", "Write a long story about a lighthouse keeper and a storm.") };
        const float T = 0.7f, P = 0.9f, Pen = 1.15f;
        const int MaxNew = 256, Seed = 99;
        var notes = new List<string>();

        // (a) the pipeline, as the chat page drives it
        double pipeRate = 0; int pipeTokens = 0;
        {
            var m = await hub.OpenAsync(repo, file, deselect: false);
            await using (m.Stream)
            using (var pipe = await GgufTextGenerationPipeline.CreateFromStreamAsync(accelerator, m.Stream, maxSeqLen: 1024, engine: SpawnDev.ILGPU.ML.Pipelines.GgufEngine.Generator))
            {
                if (accelerator.AcceleratorType == AcceleratorType.WebGPU) pipe.EnableWebGPUDecodeCapture = true;
                var cfg = new GenerationConfig { MaxNewTokens = MaxNew, Strategy = "top_p", TopP = P, Temperature = T, RepetitionPenalty = Pen, Seed = Seed };
                await pipe.GenerateAsync(messages, new GenerationConfig { MaxNewTokens = 16, Strategy = "top_p", TopP = P, Temperature = T, RepetitionPenalty = Pen, Seed = 1 }, thinking: false);
                for (int rep = 0; rep < 2; rep++)
                {
                    int n = 0; double firstMs = 0; var sw = Stopwatch.StartNew();
                    await pipe.GenerateAsync(messages, cfg, thinking: false, onToken: (_, _) => { if (n++ == 0) firstMs = sw.Elapsed.TotalMilliseconds; return Task.CompletedTask; });
                    double ms = sw.Elapsed.TotalMilliseconds - firstMs;
                    if (n > 1 && (n - 1) * 1000.0 / ms > pipeRate) { pipeRate = (n - 1) * 1000.0 / ms; pipeTokens = n; }
                }
            }
        }
        notes.Add($"pipeline (the chat page's engine{(accelerator.AcceleratorType == AcceleratorType.WebGPU ? ", decode capture" : "")}): {pipeRate:F1} tok/s ({pipeTokens} tokens)");

        // (b) the batch decoder, one sequence
        double batchRate = 0; int batchTokens = 0;
        {
            var m = await hub.OpenAsync(repo, file, deselect: false);
            await using (m.Stream)
            {
                var gguf = await SpawnDev.ILGPU.ML.GGUF.GGUFParser.ParseHeaderAsync(m.Stream);
                var tok = SentencePieceTokenizer.FromGGUF(gguf)!;
                var (prompt, stops) = ChatTemplates.BuildChatPrompt(gguf, tok, messages, thinking: false);
                m.Stream.Seek(0, SeekOrigin.Begin);
                using var dec = await GgufBatchDecoder.CreateAsync(accelerator, m.Stream, maxSeq: 1024, maxBatch: 1);
                if (accelerator.AcceleratorType == AcceleratorType.WebGPU) dec.CaptureWebGPU = true;
                var stopSet = new HashSet<int>(stops);
                await dec.GenerateAsync(prompt, 1, 16, T, P, 1, stopSet, repetitionPenalty: Pen);
                for (int rep = 0; rep < 2; rep++)
                {
                    int n = 0; double firstMs = 0; var sw = Stopwatch.StartNew();
                    await dec.GenerateAsync(prompt, 1, MaxNew, T, P, Seed, stopSet, onToken: (_, _) => { if (n++ == 0) firstMs = sw.Elapsed.TotalMilliseconds; }, repetitionPenalty: Pen);
                    double ms = sw.Elapsed.TotalMilliseconds - firstMs;
                    if (n > 1 && (n - 1) * 1000.0 / ms > batchRate) { batchRate = (n - 1) * 1000.0 / ms; batchTokens = n; }
                }
            }
        }
        notes.Add($"batch decoder, one sequence{(accelerator.AcceleratorType == AcceleratorType.WebGPU ? ", captured steps" : "")}: {batchRate:F1} tok/s ({batchTokens} tokens)");
        // (c) the pipeline on its default engine (Auto: the batch decoder here), through the text layer the chat page uses
        double autoRate = 0; string autoEngine = "";
        {
            var m = await hub.OpenAsync(repo, file, deselect: false);
            await using (m.Stream)
            using (var pipe = await GgufTextGenerationPipeline.CreateFromStreamAsync(accelerator, m.Stream, maxSeqLen: 1024))
            {
                autoEngine = pipe.Engine.ToString();
                var cfg = new GenerationConfig { MaxNewTokens = MaxNew, Strategy = "top_p", TopP = P, Temperature = T, RepetitionPenalty = Pen, Seed = Seed };
                await pipe.GenerateAsync(messages, new GenerationConfig { MaxNewTokens = 16, Strategy = "top_p", TopP = P, Temperature = T, RepetitionPenalty = Pen, Seed = 1 }, thinking: false);
                for (int rep = 0; rep < 2; rep++)
                {
                    int n = 0; double firstMs = 0; var sw = Stopwatch.StartNew();
                    await pipe.GenerateAsync(messages, cfg, thinking: false, onToken: (_, _) => { if (n++ == 0) firstMs = sw.Elapsed.TotalMilliseconds; return Task.CompletedTask; });
                    double ms = sw.Elapsed.TotalMilliseconds - firstMs;
                    if (n > 1) autoRate = Math.Max(autoRate, (n - 1) * 1000.0 / ms);
                }
            }
        }
        notes.Add($"pipeline on Auto ({autoEngine}): {autoRate:F1} tok/s");
        notes.Add($"ratio {batchRate / Math.Max(pipeRate, 1e-9):F2}x");
        var report = $"{accelerator.AcceleratorType}: " + string.Join(" | ", notes);
        Console.WriteLine("[SingleSeqRate] " + report);
        // a measurement: fail only when an engine produced nothing (a broken run must not read as a number)
        if (pipeTokens < 2 || batchTokens < 2) throw new Exception(report);
        return report;
    });
}
