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
    public async Task Bench_Qwen3_1_7B_WebGPU_DecodeStepProfile() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU) throw new UnsupportedTestException("WebGPU dispatch-plan timestamps");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var m = await hub.OpenAsync("Qwen/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q8_0.gguf", deselect: false);
        await using (m.Stream)
        {
            var gguf = await SpawnDev.ILGPU.ML.GGUF.GGUFParser.ParseHeaderAsync(m.Stream);
            var tok = SentencePieceTokenizer.FromGGUF(gguf)!;
            m.Stream.Seek(0, SeekOrigin.Begin);
            using var gen = await GgufGenerator.CreateAsync(accelerator, m.Stream, 4096, gguf);
            var dec = gen.BatchDecoder ?? throw new Exception($"expected the batch decoder, got {gen.Engine}");
            bool prefixSaved = GgufGenerator.EnablePrefixCache, vecSaved = dec.AttentionVec, ilpSaved = dec.AttentionIlp, shufSaved = dec.AttentionShuffle;
            GgufGenerator.EnablePrefixCache = false;
            var report = new List<string>();
            try
            {
                string filler = string.Join(" ", Enumerable.Range(0, 1800).Select(i => new[] { "the", "keeper", "watched", "storm", "light", "over", "grey", "water" }[i % 8])) + "\n\n";

                const string shortP = "Write a long story about a lighthouse keeper and a storm.";
                // arms: the vector decode attention (the WebGPU default since 2026-11-08) as it is, with AttentionIlp (64-key chunks, four V rows loaded ahead in
                // P.V), and with AttentionShuffle as well (both the CUDA default since 2026-11-09: long-prompt decode +8.3% there), at both lengths, interleaved
                // twice (tok/s run-to-run noise is ~5%; the per-kernel GPU ms are the steadier number). AttentionVec off (the plain kernels) lost on 2026-11-08.
                var arms = new List<(string, string, bool, bool)>();
                for (int round = 0; round < 2; round++)
                    foreach (var (ilp, shuffle) in new[] { (false, false), (true, false), (true, true) })
                    {
                        string name = "vec" + (ilp ? "-ilp" : "") + (shuffle ? "-shuffle" : "");
                        arms.Add(($"short {name}", shortP, ilp, shuffle));
                        arms.Add(($"long {name}", filler + shortP, ilp, shuffle));
                    }
                foreach (var (label, text, ilp, shuffle) in arms)
                {
                    dec.AttentionVec = true; dec.AttentionIlp = ilp; dec.AttentionShuffle = shuffle;
                    dec.InvalidateCapturedSteps();
                    var (ids, stops) = ChatTemplates.BuildChatPrompt(gguf, tok, new List<(string, string)> { ("user", text) }, thinking: false);
                    var sw = Stopwatch.StartNew(); double firstMs = -1;
                    var res = await gen.GenerateAsync(ids, new GenerationConfig { Strategy = "greedy", MaxNewTokens = 64 }, stopTokenIds: stops,
                        onDelta: _ => { if (firstMs < 0) firstMs = sw.Elapsed.TotalMilliseconds; return Task.CompletedTask; });
                    double rate = (res.GeneratedTokens - 1) / ((sw.Elapsed.TotalMilliseconds - firstMs) / 1000.0);
                    var prof = await dec.ProfileRecordedStepAsync(20) ?? throw new Exception("no recorded decode step replayed");
                    using var doc = JsonDocument.Parse(prof.Attribution);
                    var root = doc.RootElement;
                    string head = $"{label} [{dec.LastDecodeAttentionKernel}] prompt ({ids.Length} tokens): decode {rate:F1} tok/s ({1000.0 / rate:F2} ms a token); recorded step GPU floor {prof.FloorMs:F2} ms";
                    if (root.TryGetProperty("totalMs", out var tot)) head += $", timestamped total {tot.GetDouble():F2} ms over {root.GetProperty("ops").GetInt32()} dispatches";
                    Console.WriteLine($"[Benchmark] WebGPUDecodeProfile {head}");
                    report.Add(head);
                    if (root.TryGetProperty("kernels", out var ks))
                        foreach (var k in ks.EnumerateArray().Take(4))
                            Console.WriteLine($"[Benchmark] WebGPUDecodeProfile   {label}: {k.GetProperty("ms").GetDouble(),7:F3} ms x{k.GetProperty("count").GetInt32(),3} (max {k.GetProperty("maxMs").GetDouble():F3})  {k.GetProperty("label").GetString()}");
                    else Console.WriteLine($"[Benchmark] WebGPUDecodeProfile   {label}: {prof.Attribution}");
                }
            }
            finally { GgufGenerator.EnablePrefixCache = prefixSaved; dec.AttentionVec = vecSaved; dec.AttentionIlp = ilpSaved; dec.AttentionShuffle = shufSaved; }
            return string.Join("; ", report);
        }
    });
}
