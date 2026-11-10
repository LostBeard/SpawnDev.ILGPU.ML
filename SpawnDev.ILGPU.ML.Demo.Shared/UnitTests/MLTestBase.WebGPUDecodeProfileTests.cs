using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;
using System.Diagnostics;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// A measurement, not a gate: where a WebGPU decode token's GPU time goes, for SpawnDev.AI's chat model (Qwen3-1.7B Q8_0) on the batch decoder (captured
    /// steps). After 64 greedy tokens from a short prompt and again after an ~1,830-token prompt: the recorded step's per-token GPU floor (plain replays) and its
    /// per-kernel GPU ms (timestamped replay; PMT's Chrome has --enable-webgpu-developer-features, so timestamps are not quantized). Prints
    /// <c>[Benchmark] WebGPUDecodeProfile ...</c> (PMT_CONSOLE_LOG=WebGPUDecodeProfile). Added 2026-11-06: the CUDA decode wins did not carry over to WebGPU.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Bench_Qwen3_1_7B_WebGPU_DecodeStepProfile() => await RunTest(accelerator => DecodeStepProfile(accelerator, "Qwen/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q8_0.gguf"));

    /// <summary>
    /// <see cref="Bench_Qwen3_1_7B_WebGPU_DecodeStepProfile"/> on Qwen3-0.6B Q8_0: the same vocabulary (151,936: the candidate-selection kernels do the
    /// same work), a 640 MB file. Added 2026-11-09: models.spawndev.com served the 1.8 GB 1.7B file at 0.3-1.1 MB/s that night (no edge caching), so
    /// the 1.7B bench spent PMT's whole 600 s page wait loading the model.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Bench_Qwen3_0_6B_WebGPU_DecodeStepProfile() => await RunTest(accelerator => DecodeStepProfile(accelerator, "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"));

    private async Task<string> DecodeStepProfile(Accelerator accelerator, string repo, string file)
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU) throw new UnsupportedTestException("WebGPU dispatch-plan timestamps");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var m = await hub.OpenAsync(repo, file, deselect: false);
        await using (m.Stream)
        {
            var gguf = await SpawnDev.ILGPU.ML.GGUF.GGUFParser.ParseHeaderAsync(m.Stream);
            var tok = SentencePieceTokenizer.FromGGUF(gguf)!;
            m.Stream.Seek(0, SeekOrigin.Begin);
            using var gen = await GgufGenerator.CreateAsync(accelerator, m.Stream, 4096, gguf);
            var dec = gen.BatchDecoder ?? throw new Exception($"expected the batch decoder, got {gen.Engine}");
            bool prefixSaved = GgufGenerator.EnablePrefixCache, vecSaved = dec.AttentionVec, ilpSaved = dec.AttentionIlp, shufSaved = dec.AttentionShuffle, combSaved = dec.AttentionCombineGroup, topkSaved = dec.TopKGroup;
            GgufGenerator.EnablePrefixCache = false;
            var report = new List<string>();
            try
            {
                string filler = string.Join(" ", Enumerable.Range(0, 1800).Select(i => new[] { "the", "keeper", "watched", "storm", "light", "over", "grey", "water" }[i % 8])) + "\n\n";

                const string shortP = "Write a long story about a lighthouse keeper and a storm.";
                // arms: the WebGPU default decode attention (vector kernel, four-ahead P.V, 64-key chunks, shuffle reductions; 2026-11-09) with the candidate
                // selection on the serial chunk kernels or the group ones (TopKGroup: the CUDA default since 2026-11-09, +3% decode there) and the chunk merge
                // per value or per group (AttentionCombineGroup, the CUDA default; first WebGPU A/B inconclusive), at both lengths, interleaved twice (tok/s
                // run-to-run noise is ~5%; the per-kernel GPU ms are the steadier number). Earlier arms: AttentionVec off lost on 2026-11-08; vec without ILP /
                // shuffle lost on 2026-11-09 (attention GPU 0.509 / 0.807 -> 0.401 / ~0.60 ms a token).
                var arms = new List<(string, string, bool, bool)>();
                for (int round = 0; round < 2; round++)
                    foreach (var (topk, combine) in new[] { (false, false), (true, false), (true, true) })
                    {
                        string name = (topk ? "topk-group" : "topk-serial") + (combine ? "+group-merge" : "");
                        arms.Add(($"short {name}", shortP, topk, combine));
                        arms.Add(($"long {name}", filler + shortP, topk, combine));
                    }
                foreach (var (label, text, topk, combine) in arms)
                {
                    dec.AttentionVec = true; dec.AttentionIlp = true; dec.AttentionShuffle = true; dec.AttentionCombineGroup = combine; dec.TopKGroup = topk;
                    dec.InvalidateCapturedSteps();
                    var (ids, stops) = ChatTemplates.BuildChatPrompt(gguf, tok, new List<(string, string)> { ("user", text) }, thinking: false);
                    var sw = Stopwatch.StartNew(); double firstMs = -1;
                    var res = await gen.GenerateAsync(ids, new GenerationConfig { Strategy = "greedy", MaxNewTokens = 64 }, stopTokenIds: stops,
                        onDelta: _ => { if (firstMs < 0) firstMs = sw.Elapsed.TotalMilliseconds; return Task.CompletedTask; });
                    double rate = (res.GeneratedTokens - 1) / ((sw.Elapsed.TotalMilliseconds - firstMs) / 1000.0);
                    var prof = await dec.ProfileRecordedStepAsync(20) ?? throw new Exception("no recorded decode step replayed");
                    using var doc = JsonDocument.Parse(prof.Attribution);
                    var root = doc.RootElement;
                    string head = $"{file} {label} [{dec.LastDecodeAttentionKernel}, {dec.LastAttentionMergeKernel} merge] prompt ({ids.Length} tokens): decode {rate:F1} tok/s ({1000.0 / rate:F2} ms a token); recorded step GPU floor {prof.FloorMs:F2} ms";
                    if (root.TryGetProperty("totalMs", out var tot)) head += $", timestamped total {tot.GetDouble():F2} ms over {root.GetProperty("ops").GetInt32()} dispatches";
                    Console.WriteLine($"[Benchmark] WebGPUDecodeProfile {head}");
                    report.Add(head);
                    if (root.TryGetProperty("kernels", out var ks))
                        foreach (var k in ks.EnumerateArray().Take(12))
                            Console.WriteLine($"[Benchmark] WebGPUDecodeProfile   {label}: {k.GetProperty("ms").GetDouble(),7:F3} ms x{k.GetProperty("count").GetInt32(),3} (max {k.GetProperty("maxMs").GetDouble():F3})  {k.GetProperty("label").GetString()}");
                    else Console.WriteLine($"[Benchmark] WebGPUDecodeProfile   {label}: {prof.Attribution}");
                }
            }
            finally { GgufGenerator.EnablePrefixCache = prefixSaved; dec.AttentionVec = vecSaved; dec.AttentionIlp = ilpSaved; dec.AttentionShuffle = shufSaved; dec.AttentionCombineGroup = combSaved; dec.TopKGroup = topkSaved; }
            return string.Join("; ", report);
        }
    }
}
