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
    /// per-head attention (local.10), F16 caches per-head, F16 caches with <see cref="GgufBatchDecoder.GqaAttention"/> (local.11's default). Per round one decoder
    /// with F32 caches and one with F16 caches (the caches' type is fixed at creation; GqaAttention is switched on the F16 one, which captures again), two rounds,
    /// the GQA / per-head order alternating, best round each. Qwen3-0.6B Q8_0 (G = 2) and Qwen3-4B Q4_K_M (G = 4). Each decoder streams its model again (in the
    /// browser ~80 s for the 0.6B, ~260 s for the 4B), so progress lines say where a slow run is.
    /// MEASURED 2026-11-02, WebGPU, 0.6B: F32 per-head 251, F16 per-head 252, F16 GQA 216 tok/s (the GQA kernel's per-head reductions; fixed the same day).
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
        string[] names = { "F32 per-head", "F16 per-head", "F16 GQA" };
        var notes = new List<string>();
        bool halfSaved = GgufBatchDecoder.HalfKvCache;
        var swAll = Stopwatch.StartNew();
        void Progress(string what) => Console.WriteLine($"[KvGqaRate] progress {what} at {swAll.Elapsed.TotalSeconds:F1} s");
        try
        {
            foreach (var (repo, file) in new[] { ("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"), ("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf") })
            {
                var best = new double[names.Length];
                var tokens = new int[names.Length][];
                for (int round = 0; round < 2; round++)
                {
                    foreach (bool half in new[] { false, true })
                    {
                        GgufBatchDecoder.HalfKvCache = half;
                        Progress($"{file} round {round} {(half ? "F16" : "F32")} decoder: opening");
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
                            Progress($"{file} round {round} {(half ? "F16" : "F32")} decoder: created");
                            // F32: per-head only; F16: both attention kernels, the order alternating by round
                            var runs = !half ? new[] { (0, false) } : round == 0 ? new[] { (1, false), (2, true) } : new[] { (2, true), (1, false) };
                            foreach (var (c, gqa) in runs)
                            {
                                dec.GqaAttention = gqa;
                                await dec.GenerateAsync(prompt, 1, 16, 0f, 1f, 1, stopSet);
                                int n = 0; double firstMs = 0; var sw = Stopwatch.StartNew();
                                var got = await dec.GenerateAsync(prompt, 1, MaxNew, 0f, 1f, 1, stopSet, onToken: (_, _) => { if (n++ == 0) firstMs = sw.Elapsed.TotalMilliseconds; });
                                double rate = n > 1 ? (n - 1) * 1000.0 / (sw.Elapsed.TotalMilliseconds - firstMs) : 0;
                                best[c] = Math.Max(best[c], rate);
                                tokens[c] ??= got[0];
                                Progress($"{file} round {round} {names[c]}: {rate:F1} tok/s ({n} tokens)");
                            }
                        }
                    }
                }
                int same = 0;
                for (int i = 0; i < Math.Min(tokens[0].Length, tokens[2].Length) && tokens[0][i] == tokens[2][i]; i++) same++;
                notes.Add($"{file}: " + string.Join(", ", names.Select((nm, c) => $"{nm} {best[c]:F1} tok/s")) + $" (greedy F32 vs F16 GQA: first {same} of {tokens[0].Length} tokens equal)");
                if (best.Any(b => b <= 0)) throw new Exception($"{file}: a configuration produced no tokens");
            }
        }
        finally { GgufBatchDecoder.HalfKvCache = halfSaved; }
        var report = $"{accelerator.AcceleratorType}: " + string.Join(" || ", notes);
        Console.WriteLine("[KvGqaRate] " + report);
        return report;
    });

    /// <summary>
    /// A measurement, not a gate: the grouped-query decode attention's chunk size (<see cref="GgufBatchDecoder.GqaChunk"/>) against the per-head kernel at a short
    /// prompt, one F16-cache decoder a round (Qwen3-0.6B Q8_0, G = 2, captured steps on WebGPU), the routes switched on it (each switch captures again), two
    /// rounds in reversed order, best each. MEASURED 2026-11-02 on CUDA (llama3.1:8b): chunk size made no difference there; this is the WebGPU side.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task GgufBatchDecoder_GqaChunk_ShortDecodeRate() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("a measurement for the fast lanes (CUDA, OpenCL, WebGPU)");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var messages = new List<(string, string)> { ("user", "Write a long story about a lighthouse keeper and a storm.") };
        const int MaxNew = 128;
        var configs = new (string Name, bool Gqa, int Chunk)[] { ("per-head", false, 0), ("GQA 256", true, 256), ("GQA 128", true, 128), ("GQA 64", true, 64), ("GQA 32", true, 32) };
        var best = new double[configs.Length];
        var swAll = Stopwatch.StartNew();
        for (int round = 0; round < 2; round++)
        {
            Console.WriteLine($"[GqaChunkRate] progress round {round}: opening at {swAll.Elapsed.TotalSeconds:F1} s");
            var m = await hub.OpenAsync("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf", deselect: false);
            await using (m.Stream)
            {
                var gguf = await SpawnDev.ILGPU.ML.GGUF.GGUFParser.ParseHeaderAsync(m.Stream);
                var tok = SentencePieceTokenizer.FromGGUF(gguf)!;
                var (prompt, stops) = ChatTemplates.BuildChatPrompt(gguf, tok, messages, thinking: false);
                var stopSet = new HashSet<int>(stops);
                m.Stream.Seek(0, SeekOrigin.Begin);
                using var dec = await GgufBatchDecoder.CreateAsync(accelerator, m.Stream, maxSeq: 1024, maxBatch: 1);
                if (accelerator.AcceleratorType == AcceleratorType.WebGPU) dec.CaptureWebGPU = true;
                var order = Enumerable.Range(0, configs.Length).ToArray();
                if (round == 1) Array.Reverse(order);
                foreach (int c in order)
                {
                    dec.GqaAttention = configs[c].Gqa;
                    dec.GqaChunk = configs[c].Chunk;
                    await dec.GenerateAsync(prompt, 1, 16, 0f, 1f, 1, stopSet);
                    int n = 0; double firstMs = 0; var sw = Stopwatch.StartNew();
                    await dec.GenerateAsync(prompt, 1, MaxNew, 0f, 1f, 1, stopSet, onToken: (_, _) => { if (n++ == 0) firstMs = sw.Elapsed.TotalMilliseconds; });
                    double rate = n > 1 ? (n - 1) * 1000.0 / (sw.Elapsed.TotalMilliseconds - firstMs) : 0;
                    best[c] = Math.Max(best[c], rate);
                    Console.WriteLine($"[GqaChunkRate] progress round {round} {configs[c].Name}: {rate:F1} tok/s ({n} tokens) at {swAll.Elapsed.TotalSeconds:F1} s");
                }
            }
        }
        var report = $"{accelerator.AcceleratorType}: Qwen3-0.6B F16 caches, short prompt: " + string.Join(", ", configs.Select((cf, c) => $"{cf.Name} {best[c]:F1} tok/s"));
        Console.WriteLine("[GqaChunkRate] " + report);
        if (best.Any(b => b <= 0)) throw new Exception(report);
        return report;
    });
}
