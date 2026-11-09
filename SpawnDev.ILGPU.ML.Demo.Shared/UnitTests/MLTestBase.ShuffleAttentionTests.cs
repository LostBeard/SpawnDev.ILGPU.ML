using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="GgufBatchDecoder.AttentionShuffle"/> (2026-11-09): the grouped-query decode attention with each head's chunk max and exp-sum reduced by
    /// Warp.ShuffleXor instead of the shared-memory tree, on the ILP kernel and on the vector kernel (whose P.V can also load four V rows ahead,
    /// <see cref="GgufBatchDecoder.AttentionIlp"/>). The shuffle path keeps the tree's pairing for the sum, a max takes no rounding, and the four-ahead P.V sums
    /// in the same order, so every variant's logits must be BIT-IDENTICAL to its base kernel's (ILP with the tree; the vector kernel with the tree and one V row at
    /// a time), not merely close. The vector variants need an fp16 cache. Two head shapes: Qwen2.5-0.5B (G = 7, head width 64: 2 warps a group) and Qwen3-0.6B (G = 2, width
    /// 128: 4 warps). A ~700-token prompt (11 chunks of 64 keys live, the last one partial), four sequences forked, four batched steps, 32 greedy tokens; the
    /// route is checked each run (LastDecodeAttentionKernel), and switching it must capture again where steps are captured.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task GgufBatchDecoder_ShuffleAttention_BitIdentical() => await RunTest(async accelerator =>
    {
        if (!global::SpawnDev.ILGPU.AcceleratorRequirementsExtensions.SupportsWarpShuffle(accelerator, 32)) throw new UnsupportedTestException($"no 32-lane warp shuffle (warp size {accelerator.WarpSize})");
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
            async Task<(float[] Step1, float[] Step4, int[] Greedy, string Route)> Run(bool vec, bool ilp, bool shuffle)
            {
                dec.AttentionVec = vec; dec.AttentionIlp = ilp; dec.AttentionShuffle = shuffle;
                await dec.PrefillAsync(ids, 0f);
                dec.Fork(4);
                var nx = new[] { 11, 22, 33, 44 };
                await dec.StepAsync(nx, 0f);
                string route = dec.LastDecodeAttentionKernel ?? "(none)";
                var s1 = await dec.ReadLogitsAsync();
                for (int s = 1; s < 4; s++) await dec.StepAsync(nx.Select(x => x + s).ToArray(), 0f);
                var s4 = await dec.ReadLogitsAsync();
                var greedy = (await dec.GenerateAsync(ids, 1, 32, 0f, 1f, 1, stopSet))[0];
                return (s1, s4, greedy, route);
            }
            static int Diff(float[] a, float[] b)
            {
                if (a.Length != b.Length) return -1;
                int n = 0;
                for (int i = 0; i < a.Length; i++) if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) n++;
                return n;
            }
            // one chunk size for every arm: GqaChunkKeys gives 64 keys with AttentionIlp and AttChunk / G without, so at G = 2 toggling ILP alone changed the
            // chunking (and the float sums) - seen 2026-11-09 as 607,493 differing logits on Qwen3 that were not the kernel's
            dec.GqaChunk = 64;
            // (vec, ilp, shuffle, expected route); the first of each family is its base
            var arms = new List<(bool Vec, bool Ilp, bool Shuffle, string Route)> { (false, true, false, "gqa-ilp"), (false, true, true, "gqa-ilp-shuffle") };
            if (dec.KvCacheIsHalf)
                arms.AddRange(new[] { (true, false, false, "gqa-vec"), (true, false, true, "gqa-vec-shuffle"), (true, true, false, "gqa-vec-ilp"), (true, true, true, "gqa-vec-ilp-shuffle") });
            (float[] Step1, float[] Step4, int[] Greedy, string Route) baseRun = default;
            var parts = new List<string>();
            foreach (var arm in arms)
            {
                int capturesBefore = dec.CaptureStats.Captures;
                var run = await Run(arm.Vec, arm.Ilp, arm.Shuffle);
                if (run.Route != arm.Route) throw new Exception($"{file}: arm {arm.Route} ran on '{run.Route}'");
                bool isBase = arm.Route is "gqa-ilp" or "gqa-vec";
                if (isBase) { baseRun = run; parts.Add($"{arm.Route} (base)"); continue; }
                bool recaptured = !dec.CaptureWebGPU || capturesBefore == 0 || dec.CaptureStats.Captures > capturesBefore;
                int d1 = Diff(run.Step1, baseRun.Step1), d4 = Diff(run.Step4, baseRun.Step4);
                bool same = run.Greedy.SequenceEqual(baseRun.Greedy);
                string part = $"{arm.Route}: step 1 {d1} of {run.Step1.Length} logits differ in bits, step 4 {d4}, 32 greedy tokens {(same ? "same" : "DIFFER")}, captures {capturesBefore} -> {dec.CaptureStats.Captures}";
                parts.Add(part);
                if (d1 != 0 || d4 != 0 || !same || !recaptured) throw new Exception($"{file}: {part}");
            }
            var report = $"{file}: G = {dec.Model.AttentionHeadCount / dec.Model.AttentionHeadCountKV}, D = {dec.HeadDim}, {ids.Length}-token prompt: " + string.Join("; ", parts);
            Console.WriteLine($"[ShuffleAttn] {report} on {accelerator.AcceleratorType}");
            reports.Add(report);
        }
        return string.Join("; ", reports);
    });
}
