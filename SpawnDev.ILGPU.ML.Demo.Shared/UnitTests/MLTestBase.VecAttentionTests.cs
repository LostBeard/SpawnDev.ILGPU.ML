using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="GgufBatchDecoder.AttentionVec"/> (2026-11-08): the grouped-query decode attention with dead-chunk exit and scores from 16-byte K loads reduced
    /// across lanes (Warp.ShuffleXor), against the same decoder's plain grouped kernel, F16 caches (the route's requirement). Two head shapes: Qwen2.5-0.5B
    /// (G = 7, head width 64: 8 lanes a key row) and Qwen3-0.6B (G = 2, width 128: 16 lanes). A ~700-token prompt, four sequences forked, four batched steps:
    /// the first step's logits within 3e-4 and the fourth within 5e-4 relative RMS, 32 greedy tokens the same, and switching the route captures again on WebGPU.
    /// The bound is float order, measured: the dot products sum in another order (8-term partials reduced across lanes), and Qwen2.5's large attention logits
    /// carry that through softmax - 7.9e-5 / 1.3e-4 at step 1 on CUDA / WebGPU, 1.6e-4 at step 4, the size an order-only change gave on 2026-11-02 with F16
    /// caches (1.7-2.2e-4, GgufBatchDecoder_GqaAttention_MatchesPerHead's notes). A dropped or wrong key row moves the logits by 1e-1 and the greedy tokens.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task GgufBatchDecoder_VecAttention_MatchesGrouped() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("the vector route needs 32-wide warp shuffles (CUDA, WebGPU with subgroups)");
        if (accelerator.WarpSize != 32) throw new UnsupportedTestException($"warp size {accelerator.WarpSize}");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var reports = new List<string>();
        foreach (var (repo, file) in new[] { ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf"), ("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf") })
        {
            var src = await hub.OpenAsync(repo, file, deselect: false);
            using var dec = await GgufBatchDecoder.CreateAsync(accelerator, src.Stream, maxSeq: 1536, maxBatch: 4);
            if (!dec.FusedAttention || !dec.GqaAttention) throw new UnsupportedTestException("no grouped split-K attention on this accelerator");
            var tok = SentencePieceTokenizer.FromGGUF(dec.Model)!;
            var text = string.Concat(Enumerable.Repeat("The keeper climbed the stairs of the lighthouse every evening to light the lamp. ", 34));
            var (ids, stops) = ChatTemplates.BuildChatPrompt(dec.Model, tok, new List<(string, string)> { ("user", text + "What does the keeper do each evening?") }, thinking: false);
            var stopSet = stops.ToHashSet(); if (tok.EosId >= 0) stopSet.Add(tok.EosId);
            async Task<(float[] Step1, float[] Step4, int[] Greedy)> Run(bool vec)
            {
                dec.AttentionVec = vec;
                await dec.PrefillAsync(ids, 0f);
                dec.Fork(4);
                var nx = new[] { 11, 22, 33, 44 };
                await dec.StepAsync(nx, 0f);
                var s1 = await dec.ReadLogitsAsync();
                for (int s = 1; s < 4; s++) await dec.StepAsync(nx.Select(x => x + s).ToArray(), 0f);
                var s4 = await dec.ReadLogitsAsync();
                var greedy = (await dec.GenerateAsync(ids, 1, 32, 0f, 1f, 1, stopSet))[0];
                return (s1, s4, greedy);
            }
            static float Rel(float[] a, float[] b) { double se = 0, sr = 0; for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; se += d * d; sr += (double)b[i] * b[i]; } return (float)Math.Sqrt(se / Math.Max(1e-30, sr)); }
            var plain = await Run(false);
            int capturesBefore = dec.CaptureStats.Captures;
            var vec = await Run(true);
            bool recaptured = !dec.CaptureWebGPU || capturesBefore == 0 || dec.CaptureStats.Captures > capturesBefore;
            float r1 = Rel(vec.Step1, plain.Step1), r4 = Rel(vec.Step4, plain.Step4);
            bool same = vec.Greedy.SequenceEqual(plain.Greedy);
            var report = $"{file}: G = {dec.Model.AttentionHeadCount / dec.Model.AttentionHeadCountKV}, D = {dec.HeadDim}, {ids.Length}-token prompt: "
                + $"step 1 rel RMS {r1:E2}, step 4 {r4:E2}; 32 greedy tokens {(same ? "same" : "DIFFER")}; captures {capturesBefore} -> {dec.CaptureStats.Captures}";
            Console.WriteLine($"[VecAttn] {report} on {accelerator.AcceleratorType}");
            reports.Add(report);
            if (r1 > 3e-4f || r4 > 5e-4f || !same || !recaptured) throw new Exception(report);
        }
        return string.Join("; ", reports);
    });
}
