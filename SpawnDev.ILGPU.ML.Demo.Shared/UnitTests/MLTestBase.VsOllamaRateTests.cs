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
    /// A measurement, not a gate: SpawnDev.AI's chat model (Qwen3-1.7B Q8_0, the file its demo downloads) on the engine it uses
    /// (<see cref="GgufGenerator"/>, Auto = the batch decoder, decode capture on WebGPU), with the prompts and settings of <c>tools/OllamaCompareBench</c>
    /// (short and ~1,830-token prompts, greedy and top-p 0.9 / T 0.7 / penalty 1.1, 256 tokens, a run number first so no prefix is reused, median of 3
    /// warm runs), so a browser number sits next to Ollama's desktop one. Prints <c>[Benchmark] VsOllama ...</c> lines (browser: PMT_CONSOLE_LOG=VsOllama).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Bench_Qwen3_1_7B_VsOllama_Rate() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("a measurement for the fast lanes (CUDA, OpenCL, WebGPU)");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        const string repo = "Qwen/Qwen3-1.7B-GGUF", file = "Qwen3-1.7B-Q8_0.gguf";
        const int MaxNew = 256, Runs = 3, Ctx = 4096, Filler = 1800;
        string fillerText = string.Join(" ", Enumerable.Range(0, Filler).Select(i => new[] { "the", "keeper", "watched", "storm", "light", "over", "grey", "water" }[i % 8])) + "\n\n";
        string Content(bool longCtx, int run) => $"Run {run}. " + (longCtx ? fillerText : "") + "Write a long story about a lighthouse keeper and a storm.";
        double Med(List<double> v) => v.Count == 0 ? double.NaN : v.OrderBy(x => x).ElementAt(v.Count / 2);
        string lane = accelerator.AcceleratorType.ToString();

        var m = await hub.OpenAsync(repo, file, deselect: false);
        await using (m.Stream)
        {
            var gguf = await SpawnDev.ILGPU.ML.GGUF.GGUFParser.ParseHeaderAsync(m.Stream);
            var tok = SentencePieceTokenizer.FromGGUF(gguf)!;
            m.Stream.Seek(0, SeekOrigin.Begin);
            using var gen = await GgufGenerator.CreateAsync(accelerator, m.Stream, Ctx, gguf);
            bool prefixCache = GgufGenerator.EnablePrefixCache;
            GgufGenerator.EnablePrefixCache = false;
            bool q8VecSaved = Kernels.FusedDequantMatMul.Q8Vec; bool q8WebSaved = Kernels.FusedDequantMatMul.Q8VecWebGPU;
            try
            {
                // A/B arms (short prompt, greedy): the Q8_0 vector GEMV off
                // A/B arms after the four defaults (short prompt, greedy): the Q8_0 vector GEMV flipped, the attention kernel flipped
                var bd = gen.BatchDecoder;
                bool ilpDefault = bd?.AttentionIlp ?? false;
                bool q8Default = accelerator.AcceleratorType == AcceleratorType.WebGPU ? Kernels.FusedDequantMatMul.Q8VecWebGPU : Kernels.FusedDequantMatMul.Q8Vec;
                // (the first arm is a warm-up: clocks and caches settle; its line is printed but not compared)
                var arms = new List<(bool LongCtx, bool Greedy, bool Q8, bool Ilp)> { (false, true, q8Default, ilpDefault), (false, true, q8Default, ilpDefault), (false, false, q8Default, ilpDefault),
                    (true, true, q8Default, ilpDefault), (true, false, q8Default, ilpDefault), (false, true, !q8Default, ilpDefault), (false, true, q8Default, !ilpDefault),
                    (true, true, q8Default, !ilpDefault), (false, true, q8Default, ilpDefault) };
                foreach (var (longCtx, greedy, q8vec, ilp) in arms)
                    {
                        Kernels.FusedDequantMatMul.Q8Vec = q8vec; Kernels.FusedDequantMatMul.Q8VecWebGPU = q8vec;
                        if (bd != null) { bd.AttentionIlp = ilp; bd.InvalidateCapturedSteps(); }
                        var cfg = greedy ? new GenerationConfig { Strategy = "greedy", MaxNewTokens = MaxNew }
                            : new GenerationConfig { Strategy = "top_p", Temperature = 0.7f, TopP = 0.9f, RepetitionPenalty = 1.1f, RepeatLastN = 64, Seed = 1, MaxNewTokens = MaxNew };
                        var pre = new List<double>(); var dec = new List<double>(); int promptLen = 0, generated = 0;
                        for (int r = 0; r <= Runs; r++)
                        {
                            var (ids, stops) = ChatTemplates.BuildChatPrompt(gguf, tok, new List<(string, string)> { ("user", Content(longCtx, r)) }, thinking: false);
                            var sw = Stopwatch.StartNew(); double firstMs = -1;
                            var res = await gen.GenerateAsync(ids, cfg, stopTokenIds: stops, onDelta: _ => { if (firstMs < 0) firstMs = sw.Elapsed.TotalMilliseconds; return Task.CompletedTask; });
                            double ms = sw.Elapsed.TotalMilliseconds;
                            if (r == 0 || firstMs < 0 || res.GeneratedTokens < 2) continue;
                            promptLen = ids.Length; generated = res.GeneratedTokens;
                            pre.Add(ids.Length / (firstMs / 1000.0));
                            dec.Add((res.GeneratedTokens - 1) / ((ms - firstMs) / 1000.0));
                        }
                        Console.WriteLine($"[Benchmark] VsOllama {lane} {gen.Engine} {file} | {(longCtx ? "long" : "short")} prompt | {(greedy ? "greedy" : "sampled")} | q8vec {(q8vec ? "on" : "off")}, attn-ilp {(ilp ? "on" : "off")} | prompt {promptLen} tok, generated {generated} | prefill {Med(pre):F0} tok/s | decode {Med(dec):F1} tok/s");
                    }
            }
            finally { GgufGenerator.EnablePrefixCache = prefixCache; Kernels.FusedDequantMatMul.Q8Vec = q8VecSaved; Kernels.FusedDequantMatMul.Q8VecWebGPU = q8WebSaved; }
        }
    });
}
