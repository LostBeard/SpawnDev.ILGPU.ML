using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="GgufBatchDecoder.GqaAttention"/> (2026-11-02): the split-K decode attention grouped by kv head (each cached K / V row read once for the query
    /// heads that share it) against the per-head kernel. Qwen2.5-0.5B (14 query heads on 2 kv heads: G = 7), a ~700-token prompt (several 256-key chunks: the
    /// partials and their merge), four sequences forked and four batched decode steps: the logits within 1e-5 relative RMS, and 32 greedy tokens the same.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task GgufBatchDecoder_GqaAttention_MatchesPerHead() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("the fast lanes (CUDA, OpenCL, WebGPU)");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var src = await hub.OpenAsync("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf", deselect: false);
        using var dec = await GgufBatchDecoder.CreateAsync(accelerator, src.Stream, maxSeq: 1536, maxBatch: 4);
        if (!dec.FusedAttention) throw new UnsupportedTestException("no split-K attention on this accelerator");
        var tok = SentencePieceTokenizer.FromGGUF(dec.Model)!;
        var text = string.Concat(Enumerable.Repeat("The keeper climbed the stairs of the lighthouse every evening to light the lamp. ", 34));
        var (ids, stops) = ChatTemplates.BuildChatPrompt(dec.Model, tok, new List<(string, string)> { ("user", text + "What does the keeper do each evening?") }, thinking: false);
        var stopSet = stops.ToHashSet(); if (tok.EosId >= 0) stopSet.Add(tok.EosId);
        async Task<(float[] Steps, int[] Greedy)> Run(bool gqa)
        {
            dec.GqaAttention = gqa;
            await dec.PrefillAsync(ids, 0f);
            dec.Fork(4);
            var nx = new[] { 11, 22, 33, 44 };
            for (int s = 0; s < 4; s++) await dec.StepAsync(nx.Select(x => x + s).ToArray(), 0f);
            var steps = await dec.ReadLogitsAsync();
            var greedy = (await dec.GenerateAsync(ids, 1, 32, 0f, 1f, 1, stopSet))[0];
            return (steps, greedy);
        }
        var perHead = await Run(false);
        var grouped = await Run(true);
        double se = 0, sr = 0;
        for (int i = 0; i < perHead.Steps.Length; i++) { double d = grouped.Steps[i] - perHead.Steps[i]; se += d * d; sr += (double)perHead.Steps[i] * perHead.Steps[i]; }
        float rel = (float)Math.Sqrt(se / Math.Max(1e-30, sr));
        bool same = grouped.Greedy.SequenceEqual(perHead.Greedy);
        var report = $"{accelerator.AcceleratorType}: {ids.Length}-token prompt, G = {dec.Model.AttentionHeadCount / dec.Model.AttentionHeadCountKV}: grouped vs per-head attention, 4 batched steps' logits rel RMS {rel:E2}; 32 greedy tokens {(same ? "same" : "DIFFER")}";
        Console.WriteLine("[GqaAttn] " + report);
        if (rel > 1e-5f || !same) throw new Exception(report);
        return report;
    });

    /// <summary>
    /// <see cref="GgufBatchDecoder.HalfKvCache"/> (default on since 2026-11-02): the K / V caches in 16-bit float. Against a decoder loaded with F32 caches,
    /// Qwen2.5-0.5B, a ~300-token prompt: <see cref="GgufBatchDecoder.KvCacheBytes"/> is half; the prefill's last-row logits and four batched decode steps'
    /// logits (four sequences forked from the prompt) stay within 2e-3 relative RMS; 48 greedy tokens are the same. The kernels that read or write the caches
    /// (bias + rotary + store, the store, flash prefill, split-K and three-kernel attention, fork) all run in each decoder.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task GgufBatchDecoder_HalfKvCache_MatchesF32() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("the fast lanes (CUDA, OpenCL, WebGPU)");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var text = string.Concat(Enumerable.Repeat("The keeper climbed the stairs of the lighthouse every evening to light the lamp. ", 14));
        bool saved = GgufBatchDecoder.HalfKvCache;
        async Task<(long Bytes, float[] Pre, float[] Steps, int[] Greedy)> Load(bool half)
        {
            GgufBatchDecoder.HalfKvCache = half;
            try
            {
                var src = await hub.OpenAsync("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf", deselect: false);
                using var dec = await GgufBatchDecoder.CreateAsync(accelerator, src.Stream, maxSeq: 1024, maxBatch: 4);
                var tok = SentencePieceTokenizer.FromGGUF(dec.Model)!;
                var (ids, stops) = ChatTemplates.BuildChatPrompt(dec.Model, tok, new List<(string, string)> { ("user", text + "What does the keeper do each evening?") }, thinking: false);
                var stopSet = stops.ToHashSet(); if (tok.EosId >= 0) stopSet.Add(tok.EosId);
                await dec.PrefillAsync(ids, 0f);
                var pre = await dec.ReadLogitsAsync();
                dec.Fork(4);
                var nx = new[] { 11, 22, 33, 44 };
                for (int s = 0; s < 4; s++) await dec.StepAsync(nx.Select(x => x + s).ToArray(), 0f);
                var steps = await dec.ReadLogitsAsync();
                var greedy = (await dec.GenerateAsync(ids, 1, 48, 0f, 1f, 1, stopSet))[0];
                return (dec.KvCacheBytes, pre, steps, greedy);
            }
            finally { GgufBatchDecoder.HalfKvCache = saved; }
        }
        var f32 = await Load(false);
        var f16 = await Load(true);
        static float RelRms(float[] a, float[] b) { double se = 0, sr = 0; for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; se += d * d; sr += (double)b[i] * b[i]; } return (float)Math.Sqrt(se / Math.Max(1e-30, sr)); }
        float dPre = RelRms(f16.Pre, f32.Pre), dSteps = RelRms(f16.Steps, f32.Steps);
        bool bytesOk = f16.Bytes * 2 == f32.Bytes, logitsOk = dPre < 2e-3f && dSteps < 2e-3f, greedyOk = f16.Greedy.SequenceEqual(f32.Greedy);
        var report = $"{accelerator.AcceleratorType}: KV cache {f32.Bytes / 1048576.0:F0} MiB F32 -> {f16.Bytes / 1048576.0:F0} MiB F16 ({(bytesOk ? "half" : "NOT half")}); logits rel RMS vs F32: prefill {dPre:E2}, 4 batched steps {dSteps:E2}; 48 greedy tokens {(greedyOk ? "same" : "DIFFER")}";
        Console.WriteLine("[HalfKv] " + report);
        if (!bytesOk || !logitsOk || !greedyOk) throw new Exception(report);
        return report;
    });
}
