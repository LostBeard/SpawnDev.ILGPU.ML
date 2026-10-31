using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="GgufBatchDecoder"/> against the engine it batches (Qwen2.5-0.5B-Instruct Q8_0, the models host): its greedy answer must be the existing generator's
    /// (<see cref="GgufTextGenerationPipeline"/>, itself checked against Ollama) word for word; four greedy sequences decoded together must all equal it (the batched step, the
    /// forked caches and the multi-row GEMV change nothing); four sampled sequences (different seeds) must not all agree. Every draw from the device's candidates
    /// (<see cref="GgufBatchDecoder.SampleAsync"/>) over 24 sampled steps of four sequences must equal <see cref="GgufBatchDecoder.SampleLogits"/> on the full logits row
    /// with the same generator state, and at most a tenth of them may need the full row (the fallback). Times of one and four sequences are reported.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task GgufBatchDecoder_GreedyMatchesGenerator() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("a GPU decode test");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var messages = new List<(string, string)> { ("user", "Name three primary colors and say why they are called primary.") };
        const int maxNew = 48;
        string want;
        {
            var m = await hub.OpenAsync("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf", deselect: false);
            await using (m.Stream)
            using (var pipe = await GgufTextGenerationPipeline.CreateFromStreamAsync(accelerator, m.Stream, maxSeqLen: 512))
                want = await pipe.GenerateAsync(messages, new GenerationConfig { MaxNewTokens = maxNew, Strategy = "greedy" }, thinking: false);
        }
        var src = await hub.OpenAsync("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf", deselect: false);
        using var dec = await GgufBatchDecoder.CreateAsync(accelerator, src.Stream, maxSeq: 512, maxBatch: 4);
        var tok = SentencePieceTokenizer.FromGGUF(dec.Model)!;
        var (ids, stops) = ChatTemplates.BuildChatPrompt(dec.Model, tok, messages, thinking: false);
        var stopSet = stops.ToHashSet(); if (tok.EosId >= 0) stopSet.Add(tok.EosId);
        string Text(int[] t) { var d = tok.CreateStreamingDecoder(); var sb = new System.Text.StringBuilder(); foreach (var x in t) sb.Append(d.Push(x)); sb.Append(d.Finish()); return sb.ToString(); }
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var one = await dec.GenerateAsync(ids, 1, maxNew, 0f, 1f, 1, stopSet);
        double t1 = sw.Elapsed.TotalSeconds;
        sw.Restart();
        var four = await dec.GenerateAsync(ids, 4, maxNew, 0f, 1f, 1, stopSet);
        double t4 = sw.Elapsed.TotalSeconds;
        var dt = dec.DecodeTimings;
        var sampled = await dec.GenerateAsync(ids, 4, maxNew, 0.8f, 0.95f, 11, stopSet);
        // the device candidates against the full-row sampler, draw for draw
        int draws = 0, mismatches = 0;
        await dec.PrefillAsync(ids, 0.8f);
        dec.Fork(4);
        var nextTok = new int[4];
        for (int step = 0; step < 24; step++)
        {
            var full = await dec.ReadLogitsAsync();
            for (int s = 0; s < 4; s++)
            {
                int row = step == 0 ? 0 : s, seed = 1000 * step + s;
                int a = await dec.SampleAsync(row, 0.95f, new Random(seed));
                int b = GgufBatchDecoder.SampleLogits(full, row, dec.Vocab, 0.8f, 0.95f, new Random(seed));
                draws++; if (a != b) mismatches++;
                nextTok[s] = b;
            }
            await dec.StepAsync(nextTok, 0.8f);
        }
        int wide = dec.RowReads;
        string got = Text(one[0]);
        bool same = got == want, rowsEqual = four.All(r => r.SequenceEqual(one[0]));
        int distinct = sampled.Select(Text).Distinct().Count();
        var report = $"{accelerator.AcceleratorType}: greedy {(same ? "= the generator" : "DIFFERS from the generator")} ({one[0].Length} tokens); four greedy rows {(rowsEqual ? "identical" : "DIFFER")}; four samples {distinct} distinct; "
            + $"one sequence {one[0].Length / t1:F1} tok/s, four {four.Sum(r => r.Length) / t4:F1} tok/s; a four-sequence step: embedding {dt.Embed / dt.Steps:F1} ms, device {dt.Device / dt.Steps:F1} ms, candidates read {dt.Readback / dt.Steps:F1} ms; device draws {draws - mismatches}/{draws} = the full-row sampler ({wide} read the row)";
        Console.WriteLine("[BatchDecode] " + report);
        if (!same) throw new Exception(report + $"\nwant: {want}\ngot:  {got}");
        if (!rowsEqual || distinct < 2 || mismatches > 0 || wide * 10 > draws) throw new Exception(report);      // the row read is a safety net: a device path that keeps falling back is broken
        return report;
    });
}
