using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;
using System.Diagnostics;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// A measurement for Data's report (2026-11-02): on local.7, llama3.1:8b decoded at 6.2 tok/s on <see cref="GgufBatchDecoder"/> and 11.8 on the session
    /// engine (CUDA, RTX 4070), while qwen3:8b went 20.9 -> 27.8. Per Ollama model from this machine's cache (env OLLAMA_BENCH_MODELS, default
    /// "llama3.1:8b,qwen3:8b"): the matrices' block types, greedy decode tok/s on each engine (128 tokens after a warm run, first token to last), and the batch
    /// decoder's per-kernel times over a profiled 32-step run (it synchronises after each kernel, so the shares matter, not the sum). Desktop lanes.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Bench_OllamaEngineRate_Profile() => await RunTest(async accelerator =>
    {
        if (OperatingSystem.IsBrowser() || accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL))
            throw new UnsupportedTestException("reads the local Ollama cache (CUDA / OpenCL lanes)");
        string root = Environment.GetEnvironmentVariable("OLLAMA_MODELS") is { Length: > 0 } env ? env
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".ollama", "models");
        string? Blob(string model, string tag)
        {
            var manifest = Path.Combine(root, "manifests", "registry.ollama.ai", "library", model, tag);
            if (!File.Exists(manifest)) return null;
            using var doc = JsonDocument.Parse(File.ReadAllText(manifest));
            foreach (var layer in doc.RootElement.GetProperty("layers").EnumerateArray())
                if (layer.GetProperty("mediaType").GetString()!.EndsWith(".model"))
                    return Path.Combine(root, "blobs", layer.GetProperty("digest").GetString()!.Replace(':', '-'));
            return null;
        }
        var models = (Environment.GetEnvironmentVariable("OLLAMA_BENCH_MODELS") is { Length: > 0 } list ? list : "llama3.1:8b,qwen3:8b").Split(',');
        // OLLAMA_BENCH_FILLER_WORDS: a long prompt (context cost: each decode step's attention reads the whole K / V cache), as SpawnDev.AI chat-bench's third argument
        int filler = int.TryParse(Environment.GetEnvironmentVariable("OLLAMA_BENCH_FILLER_WORDS"), out var fw) ? fw : 0;
        string fillerText = filler > 0 ? string.Join(" ", Enumerable.Range(0, filler).Select(i => new[] { "the", "keeper", "watched", "storm", "light", "over", "grey", "water" }[i % 8])) + "\n\n" : "";
        var q = new List<(string, string)> { ("user", fillerText + "Write a long story about a lighthouse keeper and a storm.") };
        const int MaxNew = 128;
        int MaxSeq = filler > 0 ? filler * 2 + 512 : 1024;
        var notes = new List<string>();
        foreach (var spec in models)
        {
            var parts = spec.Split(':');
            var path = Blob(parts[0], parts[1]);
            if (path == null || !File.Exists(path)) { notes.Add($"{spec}: not cached"); continue; }
            SpawnDev.ILGPU.ML.GGUF.GGUFModel gguf;
            using (var fs = File.OpenRead(path)) gguf = await SpawnDev.ILGPU.ML.GGUF.GGUFParser.ParseHeaderAsync(fs);
            var tok = SentencePieceTokenizer.FromGGUF(gguf)!;
            var (prompt, stopIds) = ChatTemplates.BuildChatPrompt(gguf, tok, q, thinking: false);
            var stops = new HashSet<int>(stopIds) { tok.EosId };
            // the matrices' block types (layer 0 and the head): the GEMV route each takes
            string Ty(string n) => gguf.Tensors.FirstOrDefault(t => t.Name == n)?.Type.ToString() ?? "-";
            string census = $"q {Ty("blk.0.attn_q.weight")} k {Ty("blk.0.attn_k.weight")} v {Ty("blk.0.attn_v.weight")} o {Ty("blk.0.attn_output.weight")} gate {Ty("blk.0.ffn_gate.weight")} " +
                $"up {Ty("blk.0.ffn_up.weight")} down {Ty("blk.0.ffn_down.weight")} head {Ty("output.weight")} embd {Ty("token_embd.weight")}; layers {gguf.BlockCount}, vocab {tok.VocabSize}";
            var mixed = gguf.Tensors.Where(t => t.Name.StartsWith("blk.")).GroupBy(t => t.Name[(t.Name.IndexOf('.', 4) + 1)..]).Where(g => g.Select(t => t.Type).Distinct().Count() > 1)
                .Select(g => $"{g.Key}: {string.Join("/", g.GroupBy(t => t.Type).Select(x => $"{x.Key} x{x.Count()}"))}");
            census += mixed.Any() ? " | mixed per layer: " + string.Join("; ", mixed) : "";

            // the session engine
            double sessRate = 0;
            using (var gen = await GgufTextGenerationPipeline.CreateFromFileAsync(accelerator, path, maxSeqLen: MaxSeq, engine: GgufEngine.Generator))
            {
                var cfg = new GenerationConfig { MaxNewTokens = MaxNew, Strategy = "greedy" };
                await gen.GenerateAsync(q, new GenerationConfig { MaxNewTokens = 8, Strategy = "greedy" }, thinking: false);
                int n = 0; double firstMs = 0; var sw = Stopwatch.StartNew();
                await gen.GenerateAsync(q, cfg, thinking: false, onToken: (_, _) => { if (n++ == 0) firstMs = sw.Elapsed.TotalMilliseconds; return Task.CompletedTask; });
                if (n > 1) sessRate = (n - 1) * 1000.0 / (sw.Elapsed.TotalMilliseconds - firstMs);
            }
            // the batch decoder: rate, then a profiled run
            double batchRate = 0; string prof;
            using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true))
            using (var dec = await GgufBatchDecoder.CreateAsync(accelerator, fs, maxSeq: MaxSeq, maxBatch: 1))
            {
                await dec.GenerateAsync(prompt, 1, 8, 0f, 1f, 1, stops);
                int n = 0; double firstMs = 0; var sw = Stopwatch.StartNew();
                await dec.GenerateAsync(prompt, 1, MaxNew, 0f, 1f, 1, stops, onToken: (_, _) => { if (n++ == 0) firstMs = sw.Elapsed.TotalMilliseconds; });
                if (n > 1) batchRate = (n - 1) * 1000.0 / (sw.Elapsed.TotalMilliseconds - firstMs);
                var t = dec.DecodeTimings;
                string timings = t.Steps > 0 ? $"per step: embed {t.Embed / t.Steps:F2} device {t.Device / t.Steps:F2} readback {t.Readback / t.Steps:F2} ms ({t.Steps} steps)" : "no step timings";
                await dec.PrefillAsync(prompt);
                dec.ProfileKernels = true;
                dec.KernelTimes.Clear();
                var next = new[] { prompt[^1] };
                for (int s = 0; s < 32; s++) await dec.StepAsync(next);
                dec.ProfileKernels = false;
                double sum = dec.KernelTimes.Values.Sum();
                prof = timings + " | profiled kernels (32 steps, ms/step, share): " + string.Join(", ", dec.KernelTimes.OrderByDescending(kv => kv.Value).Take(10)
                    .Select(kv => $"{kv.Key} {kv.Value / 32:F2} ({kv.Value * 100 / sum:F0}%)")) + $"; sum {sum / 32:F1} ms/step";
                // the prefix-reuse prefill (Vertex's case, Data 2026-11-02: 104-346 new tokens over a reused 3-5k prefix took 2.2-5.6 s): a generation records the
                // cache, then the same prompt with a different ~200-token tail just before the question is prefilled with reuse on (timed alone, synchronised), and
                // the same prompt prefilled from row 0 as the reference
                if (filler > 0)
                {
                    var q2 = new List<(string, string)> { ("user", fillerText + string.Join(" ", Enumerable.Range(0, 150).Select(i => new[] { "a", "gull", "cried", "near", "the", "rocks" }[i % 6])) + "\n\nWrite a long story about a lighthouse keeper and a storm.") };
                    var (prompt2, _) = ChatTemplates.BuildChatPrompt(gguf, tok, q2, thinking: false);
                    await dec.GenerateAsync(prompt, 1, 8, 0f, 1f, 1, stops);                 // records the cache rows
                    var swR = Stopwatch.StartNew();
                    await dec.PrefillAsync(prompt2, 0f, reusePrefix: true);
                    await accelerator.SynchronizeAsync();
                    double reuseMs = swR.Elapsed.TotalMilliseconds;
                    int reused = dec.LastReusedPrefix;
                    var swF = Stopwatch.StartNew();
                    await dec.PrefillAsync(prompt2, 0f, reusePrefix: false);
                    await accelerator.SynchronizeAsync();
                    double fullMs = swF.Elapsed.TotalMilliseconds;
                    int fresh = prompt2.Length - reused;
                    prof += $" | prefix reuse: {prompt2.Length}-token prompt, {reused} rows reused, {fresh} new rows prefilled in {reuseMs:F0} ms ({fresh * 1000.0 / reuseMs:F0} new tok/s); the whole prompt from row 0 in {fullMs:F0} ms ({prompt2.Length * 1000.0 / fullMs:F0} tok/s)";
                    // where each prefill's time goes (synchronised per kernel, so the shares matter): the reuse prefill, then the whole prompt
                    string Shares(string what)
                    {
                        double total = dec.KernelTimes.Values.Sum();
                        return $" | {what} profiled ({total:F0} ms): " + string.Join(", ", dec.KernelTimes.OrderByDescending(kv => kv.Value).Take(12).Select(kv => $"{kv.Key} {kv.Value:F0} ({kv.Value * 100 / total:F0}%)"));
                    }
                    await dec.GenerateAsync(prompt, 1, 8, 0f, 1f, 1, stops);
                    dec.ProfileKernels = true;
                    await dec.PrefillAsync(prompt2, 0f, reusePrefix: true);
                    prof += Shares($"reuse prefill ({dec.LastReusedPrefix} reused)");
                    await dec.PrefillAsync(prompt2, 0f, reusePrefix: false);
                    prof += Shares("whole prefill");
                    dec.ProfileKernels = false;
                }
            }
            notes.Add($"{spec}: prompt {prompt.Length} tokens, KV {(GgufBatchDecoder.HalfKvCache ? "F16" : "F32")}: session {sessRate:F1} tok/s, batch {batchRate:F1} tok/s | {census} | {prof}");
        }
        var report = $"{accelerator.AcceleratorType}: " + string.Join(" || ", notes);
        Console.WriteLine("[OllamaEngineRate] " + report);
        return report;
    });
}
