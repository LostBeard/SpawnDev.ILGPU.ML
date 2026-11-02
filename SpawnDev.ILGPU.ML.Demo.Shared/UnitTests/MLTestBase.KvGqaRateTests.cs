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
    /// A measurement, not a gate (Data, 2026-11-02: SpawnDev.AI's in-gate WebGPU decode of qwen3:4b read 57.5 tok/s on local.11, 61.3 on local.10, with other load on
    /// the card): short-prompt greedy decode on <see cref="GgufBatchDecoder"/> (one sequence, captured steps on WebGPU) in three configurations, F32 caches with the
    /// per-head attention (local.10), F16 caches per-head, F16 caches with <see cref="GgufBatchDecoder.GqaAttention"/> (local.11's default). Each configuration gets
    /// its own decoder (the caches' type is fixed at creation, and a captured step keeps the kernels it recorded), the three interleaved over two rounds so load
    /// drift falls on all of them, best round each. Qwen3-0.6B Q8_0 (G = 2) and Qwen3-4B Q4_K_M (G = 4).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task GgufBatchDecoder_KvCacheAndGqa_ShortDecodeRate() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("a measurement for the fast lanes (CUDA, OpenCL, WebGPU)");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var messages = new List<(string, string)> { ("user", "Write a long story about a lighthouse keeper and a storm.") };
        const int MaxNew = 128;
        var configs = new (string Name, bool Half, bool Gqa)[] { ("F32 per-head", false, false), ("F16 per-head", true, false), ("F16 GQA", true, true) };
        var notes = new List<string>();
        bool halfSaved = GgufBatchDecoder.HalfKvCache;
        try
        {
            foreach (var (repo, file) in new[] { ("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"), ("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf") })
            {
                var best = new double[configs.Length];
                var tokens = new int[configs.Length][];
                for (int round = 0; round < 2; round++)
                {
                    for (int c = 0; c < configs.Length; c++)
                    {
                        GgufBatchDecoder.HalfKvCache = configs[c].Half;
                        var m = await hub.OpenAsync(repo, file, deselect: false);
                        await using (m.Stream)
                        {
                            var gguf = await SpawnDev.ILGPU.ML.GGUF.GGUFParser.ParseHeaderAsync(m.Stream);
                            var tok = SentencePieceTokenizer.FromGGUF(gguf)!;
                            var (prompt, stops) = ChatTemplates.BuildChatPrompt(gguf, tok, messages, thinking: false);
                            var stopSet = new HashSet<int>(stops);
                            m.Stream.Seek(0, SeekOrigin.Begin);
                            using var dec = await GgufBatchDecoder.CreateAsync(accelerator, m.Stream, maxSeq: 1024, maxBatch: 1);
                            if (accelerator.AcceleratorType == AcceleratorType.WebGPU) dec.CaptureWebGPU = true;
                            dec.GqaAttention = configs[c].Gqa;
                            await dec.GenerateAsync(prompt, 1, 16, 0f, 1f, 1, stopSet);
                            int n = 0; double firstMs = 0; var sw = Stopwatch.StartNew();
                            var got = await dec.GenerateAsync(prompt, 1, MaxNew, 0f, 1f, 1, stopSet, onToken: (_, _) => { if (n++ == 0) firstMs = sw.Elapsed.TotalMilliseconds; });
                            double rate = n > 1 ? (n - 1) * 1000.0 / (sw.Elapsed.TotalMilliseconds - firstMs) : 0;
                            best[c] = Math.Max(best[c], rate);
                            tokens[c] ??= got[0];
                        }
                    }
                }
                int same = 0;
                for (int i = 0; i < Math.Min(tokens[0].Length, tokens[2].Length) && tokens[0][i] == tokens[2][i]; i++) same++;
                notes.Add($"{file}: " + string.Join(", ", configs.Select((cf, c) => $"{cf.Name} {best[c]:F1} tok/s")) + $" (greedy F32 vs F16 GQA: first {same} of {tokens[0].Length} tokens equal)");
                if (best.Any(b => b <= 0)) throw new Exception($"{file}: a configuration produced no tokens");
            }
        }
        finally { GgufBatchDecoder.HalfKvCache = halfSaved; }
        var report = $"{accelerator.AcceleratorType}: " + string.Join(" || ", notes);
        Console.WriteLine("[KvGqaRate] " + report);
        return report;
    });
}
