using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="GgufBatchDecoder.OverlapV"/> (CUDA: v's projection on a second stream beside q|k) must not change a single bit: the overlap only reorders
    /// independent work. Found 2026-11-10: both streams' GEMMs ran on ONE FusedDequantMatMul, whose prefill paths stage the weights in shared scratch
    /// (the f16 scratch GEMM from 512 rows, the opt-in int8 GEMM always), so v's unpack could overwrite q|k's weights mid-GEMM. A model whose v block
    /// type differs from q|k (qwen2.5 1.5B Q4_K_M: v Q6_K in some layers) with a prompt over 512 tokens, overlap off vs on: the prefill's logits, 4 decode
    /// steps' logits and 32 greedy tokens BIT-IDENTICAL, with the default fp32 activations and with the opt-in int8 activations.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task<string> GgufBatchDecoder_OverlapV_LongPrefill_BitIdentical() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.Cuda) throw new UnsupportedTestException("OverlapV is CUDA-only");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var src = await hub.OpenAsync("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf", deselect: false);
        using var dec = await GgufBatchDecoder.CreateAsync(accelerator, src.Stream, maxSeq: 2048, maxBatch: 1);
        var tok = SentencePieceTokenizer.FromGGUF(dec.Model)!;
        var text = string.Concat(Enumerable.Repeat("The keeper climbed the stairs of the lighthouse every evening to light the lamp, and counted the ships. ", 40));
        var (ids, stops) = ChatTemplates.BuildChatPrompt(dec.Model, tok, new List<(string, string)> { ("user", text + "How many stairs were there?") }, thinking: false);
        if (ids.Length < 600) throw new Exception($"prompt is only {ids.Length} tokens; the scratch GEMMs need over 512 rows");
        var stopSet = stops.ToHashSet(); if (tok.EosId >= 0) stopSet.Add(tok.EosId);
        static double MaxAbs(float[] a, float[] b) { double m = 0; for (int i = 0; i < Math.Min(a.Length, b.Length); i++) m = Math.Max(m, Math.Abs((double)a[i] - b[i])); return m; }
        static double Range(float[] a) => a.Max() - a.Min();
        static int Diff(float[] a, float[] b) { if (a.Length != b.Length) return -1; int n = 0; for (int i = 0; i < a.Length; i++) if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) n++; return n; }
        async Task<(float[] P, float[] S4, int[] G)> Run(bool overlap)
        {
            dec.OverlapV = overlap;
            dec.InvalidateCapturedSteps();
            await dec.PrefillAsync(ids, 0f);
            var p = await dec.ReadLogitsAsync();
            for (int s = 0; s < 4; s++) await dec.StepAsync(new[] { 101 + s }, 0f);
            var s4 = await dec.ReadLogitsAsync();
            var g = (await dec.GenerateAsync(ids, 1, 32, 0f, 1f, 1, stopSet))[0];
            return (p, s4, g);
        }
        bool savedOverlap = dec.OverlapV, savedI8 = SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv;
        var notes = new List<string>();
        try
        {
            foreach (bool int8 in new[] { false, true })
            {
                SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv = int8;
                var off = await Run(false);
                var on = await Run(true);
                int dp = Diff(off.P, on.P), ds = Diff(off.S4, on.S4);
                bool same = off.G.SequenceEqual(on.G);
                string note = $"{(int8 ? "int8" : "fp32")} activations, {ids.Length}-token prompt: overlap off vs on - prefill {dp}, step 4 {ds} logits differ in bits (max |diff| {MaxAbs(off.P, on.P):E2} / {MaxAbs(off.S4, on.S4):E2}, logit range {Range(off.P):F1}), 32 greedy tokens {(same ? "same" : "DIFFER")}";
                notes.Add(note);
                Console.WriteLine("[OverlapV] " + note);
            }
        }
        finally { dec.OverlapV = savedOverlap; SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv = savedI8; dec.InvalidateCapturedSteps(); }
        if (notes.Any(n => !n.Contains("prefill 0, step 4 0") || n.Contains("DIFFER"))) throw new Exception(string.Join("; ", notes));
        return string.Join("; ", notes);
    });
}
