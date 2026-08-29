using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// GGUF incremental KV-cache decode regression suite. Locks the equivalence the cache MUST hold:
/// feeding a sequence one token at a time through <see cref="InferenceSession.RunDecodeStepAsync"/>
/// (each step computes only the new token's K/V and attends it against the cached history with
/// kv_offset = pastLen, RoPE re-offset to the token's absolute position) produces the SAME logits,
/// per position, as a single full-recompute <see cref="InferenceSession.RunAsync"/> over the whole
/// sequence. If the cache write/read layout, the RoPE offset, or the FusedAttention kv_offset is
/// wrong, the incremental logits diverge from the full-recompute reference — caught here on every
/// backend the harness runs. The cache write/read is one async CopyFromAsync path (orders the copy
/// against the producing kernel on the Wasm worker pool), so this also proves browser parity.
/// </summary>
public partial class MLTestBase
{
    /// <summary>
    /// The KV PACK path's missing correctness gate: a sequence long enough to CROSS the pack's grow-only
    /// capacity boundary, checked against full-recompute.
    ///
    /// Why this was missing (2026-07-16): `GGUFDecodeKVCache.EnsurePackCapacity`/`EnsureBf16Scratch` seed at
    /// `Math.Min(64, maxSeqLen)` and double. Every existing decode gate uses ctx=64 with a 4-6 token sequence,
    /// so the capacity NEVER grows and the post-growth path was untested. It matters because the pack is the
    /// **WebGL+BF16-only** route (GraphExecutor.cs:2597 `stridedOk`) - every other backend feeds FusedAttention
    /// the strided store and never runs this code. Measured on the real LFM2: two identical generations agree
    /// while the pack capacity matches (both 64, both 256, both 4096) and DISAGREE when they differ (64 vs 128,
    /// i.e. after a growth) - so the buffer's CAPACITY is leaking into the result, which it must not (the live
    /// region is contiguous at offset 0 and PackedAsync returns SubView(0, live)).
    ///
    /// That evidence came from an A==B probe, which proves CONSISTENCY, not correctness - both runs can agree
    /// and both be wrong. THIS test is the correctness oracle: full-recompute is the reference, so if the pack
    /// path is wrong after a growth it fails here regardless of self-consistency. seqLen=70 > 64 forces exactly
    /// one growth (64 -> 128) mid-sequence.
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task GGUFDecodeKVCache_BF16_AcrossPackGrowth_MatchesFullRecompute() => await RunTest(async accelerator =>
    {
        const int embd = 256, vocab = 32, ffn = 320, ctx = 256, seqLen = 70;   // 70 > 64 => one pack growth
        var bytes = BuildTinyQuantizedLlamaGGUF(embd, vocab, ffn, ctx, new Random(9));
        var model = GGUFParser.Parse(bytes);

        int nLayers = (int)model.BlockCount, nHeads = (int)model.AttentionHeadCount;
        int defNKV = (int)model.AttentionHeadCountKV; if (defNKV == 0) defNKV = nHeads;
        int defHd = embd / nHeads;
        var kvHeadsArr = new int[nLayers]; var hdArr = new int[nLayers];
        for (int L = 0; L < nLayers; L++)
        { var cfg = GGUFGraphBuilder.GetLayerAttnConfig(model, L, nHeads, defNKV, defHd); kvHeadsArr[L] = cfg.NKVHeads; hdArr[L] = cfg.HeadDim; }

        var rng = new Random(11);
        var seq = new float[seqLen];
        for (int i = 0; i < seqLen; i++) seq[i] = rng.Next(0, vocab);

        // Reference: ONE full-recompute forward over the whole sequence.
        using var session = InferenceSession.CreateFromGGUF(accelerator, bytes);
        using var inFull = accelerator.Allocate1D(seq);
        var outFull = await session.RunAsync(new Dictionary<string, Tensor>
        { ["input_ids"] = new Tensor(inFull.View, new[] { 1, seqLen }, "input_ids") });
        var logitsFullT = outFull.TryGetValue("logits", out var lf) ? lf : outFull.Values.First();
        using var readFull = accelerator.Allocate1D<float>(seqLen * vocab);
        await readFull.View.CopyFromAsync(logitsFullT.Data.SubView(0, seqLen * vocab));
        await accelerator.SynchronizeAsync();
        var logitsFull = await readFull.CopyToHostAsync<float>(0, seqLen * vocab);

        // BF16 (the production store precision, and the only one that takes the WebGL pack path).
        using var kv = new GGUFDecodeKVCache(accelerator, kvHeadsArr, hdArr, maxSeqLen: ctx, precision: KVCachePrecision.BF16);
        session.EnableGGUFDecode(kv);
        try
        {
            for (int pos = 0; pos < seqLen; pos++)
            {
                using var inTok = accelerator.Allocate1D(new[] { seq[pos] });
                var outStep = await session.RunDecodeStepAsync(new Dictionary<string, Tensor>
                { ["input_ids"] = new Tensor(inTok.View, new[] { 1, 1 }, "input_ids") });
                var stepT = outStep.TryGetValue("logits", out var ls) ? ls : outStep.Values.First();
                using var readStep = accelerator.Allocate1D<float>(vocab);
                await readStep.View.CopyFromAsync(stepT.Data.SubView(0, vocab));
                await accelerator.SynchronizeAsync();
                var stepLogits = await readStep.CopyToHostAsync<float>(0, vocab);

                int argFull = 0, argKV = 0;
                for (int v = 1; v < vocab; v++)
                {
                    if (logitsFull[pos * vocab + v] > logitsFull[pos * vocab + argFull]) argFull = v;
                    if (stepLogits[v] > stepLogits[argKV]) argKV = v;
                }
                if (argFull != argKV)
                    throw new Exception($"[pack-growth] step {pos} (KV len {pos + 1}{(pos + 1 == 65 ? " - FIRST STEP PAST THE 64-TOKEN PACK SEED" : "")}): " +
                        $"decode argmax {argKV} != full-recompute argmax {argFull}. The pack path is WRONG after its " +
                        "grow-only capacity doubled (WebGL+BF16 is the only backend that runs it).");
                // ARGMAX-ONLY is the gate for BF16, deliberately. My first cut also checked logits at 6% and
                // failed at step 1 (KV len 2) on EVERY backend incl. CUDA - which cannot be the pack path, since
                // CUDA never runs it. That was bf16 STORE rounding on a random-weight model whose logits are
                // ~127: 0.4%/value compounds through attention past a 6% band. I had set the tolerance from an
                // assumption instead of from the data. The existing suite says the same thing: for BF16 "argmax
                // must still match EXACTLY (the real gate)". A capacity leak / layout bug causes GROSS
                // divergence (the real LFM2 symptom is attention returning exactly 0), which flips argmax far
                // outside any rounding - so argmax cannot hide one, and a tight logit band only produces noise.
            }
            Console.WriteLine($"[pack-growth] decode == full-recompute for all {seqLen} positions across the 64->128 pack growth.");
        }
        finally { session.DisableGGUFDecode(); }
    });

    [TestMethod(Timeout = 120000, Category = "HeavyCpu")]
    public async Task GGUFDecodeKVCache_IncrementalMatchesFullRecompute() => await RunTest(async accelerator =>
    {
        const int embd = 256, vocab = 32, ffn = 320, ctx = 64;
        // Reuse the tiny quantized llama GGUF from the recompile suite (same partial class). Its
        // FusedAttention nodes carry the "layer" tag the decode intercept needs; the cache is
        // arch-agnostic (works for any GGUF decoder, gemma4 included).
        var bytes = BuildTinyQuantizedLlamaGGUF(embd, vocab, ffn, ctx, new Random(7));
        var model = GGUFParser.Parse(bytes);

        int nLayers = (int)model.BlockCount, nHeads = (int)model.AttentionHeadCount;
        int defNKV = (int)model.AttentionHeadCountKV; if (defNKV == 0) defNKV = nHeads;
        int defHd = embd / nHeads;
        var kvHeadsArr = new int[nLayers]; var hdArr = new int[nLayers];
        for (int L = 0; L < nLayers; L++)
        { var cfg = GGUFGraphBuilder.GetLayerAttnConfig(model, L, nHeads, defNKV, defHd); kvHeadsArr[L] = cfg.NKVHeads; hdArr[L] = cfg.HeadDim; }

        var seq = new float[] { 3, 9, 21, 5 };

        // ── Reference: ONE full-recompute forward over the whole sequence ──
        using var session = InferenceSession.CreateFromGGUF(accelerator, bytes);
        using var inFull = accelerator.Allocate1D(seq);
        var outFull = await session.RunAsync(new Dictionary<string, Tensor>
        { ["input_ids"] = new Tensor(inFull.View, new[] { 1, seq.Length }, "input_ids") });
        var logitsFullT = outFull.TryGetValue("logits", out var lf) ? lf : outFull.Values.First();
        if (logitsFullT.Shape[^1] != vocab) throw new Exception($"full logits last dim {logitsFullT.Shape[^1]}, want {vocab}");
        using var readFull = accelerator.Allocate1D<float>(seq.Length * vocab);
        await readFull.View.CopyFromAsync(logitsFullT.Data.SubView(0, seq.Length * vocab));
        await accelerator.SynchronizeAsync();
        var logitsFull = await readFull.CopyToHostAsync<float>(0, seq.Length * vocab);

        // ── KV-cache: feed the SAME tokens one at a time; each step's logits must match the
        //    corresponding position of the full-recompute run. Run BOTH storage precisions:
        //    • F32   — exact store, tight tolerance: the layout / kv_offset / RoPE-offset gate. bf16
        //              rounding would mask a subtle indexing bug here, so this mode keeps it sharp.
        //    • BF16  — production VRAM-halving store. Argmax must still match EXACTLY (the real gate);
        //              logits within bf16's ~0.4%/value rounding (a layout bug causes gross divergence
        //              + argmax flips, far outside this, so the loose tol still can't hide one). ──
        async Task RunPrecision(KVCachePrecision precision, float relTol, float absFloor)
        {
            using var kv = new GGUFDecodeKVCache(accelerator, kvHeadsArr, hdArr, maxSeqLen: ctx, precision: precision);
            session.EnableGGUFDecode(kv);
            try
            {
                for (int pos = 0; pos < seq.Length; pos++)
                {
                    using var inTok = accelerator.Allocate1D(new[] { seq[pos] });
                    var outStep = await session.RunDecodeStepAsync(new Dictionary<string, Tensor>
                    { ["input_ids"] = new Tensor(inTok.View, new[] { 1, 1 }, "input_ids") });
                    var stepT = outStep.TryGetValue("logits", out var ls) ? ls : outStep.Values.First();
                    int stepSeq = stepT.ElementCount / vocab;
                    if (stepSeq != 1) throw new Exception($"[{precision}] decode step {pos}: expected 1 logit position, got {stepSeq}");
                    using var readStep = accelerator.Allocate1D<float>(vocab);
                    await readStep.View.CopyFromAsync(stepT.Data.SubView(0, vocab));
                    await accelerator.SynchronizeAsync();
                    var stepLogits = await readStep.CopyToHostAsync<float>(0, vocab);

                    // argmax: STRICT in both modes — the top token must match the full forward exactly.
                    int argFull = 0, argKV = 0;
                    for (int v = 1; v < vocab; v++)
                    {
                        if (logitsFull[pos * vocab + v] > logitsFull[pos * vocab + argFull]) argFull = v;
                        if (stepLogits[v] > stepLogits[argKV]) argKV = v;
                    }
                    if (argFull != argKV)
                        throw new Exception($"[{precision}] decode step {pos}: KV argmax {argKV} != full-recompute argmax {argFull} " +
                            "— the incremental cache diverges from the full forward (layout / kv_offset / RoPE-offset bug).");
                    for (int v = 0; v < vocab; v++)
                    {
                        float fv = logitsFull[pos * vocab + v];
                        float tol = MathF.Max(absFloor, MathF.Abs(fv) * relTol);
                        if (MathF.Abs(stepLogits[v] - fv) > tol)
                            throw new Exception($"[{precision}] decode step {pos} vocab {v}: KV logit {stepLogits[v]} vs full {fv} (tol {tol}) " +
                                "— incremental decode is not numerically equivalent to full-recompute.");
                    }
                }
                Console.WriteLine($"[GGUFDecodeKVCache] {precision}: incremental decode == full-recompute for all {seq.Length} positions " +
                    $"({nLayers} layer(s), kvHeads={kvHeadsArr[0]}, hd={hdArr[0]}) — argmax + logits match.");
            }
            finally { session.DisableGGUFDecode(); }
        }

        await RunPrecision(KVCachePrecision.F32, relTol: 2e-3f, absFloor: 2e-3f);   // exact store: tight layout gate
        await RunPrecision(KVCachePrecision.BF16, relTol: 6e-2f, absFloor: 6e-2f);  // bf16 store: argmax-strict, bf16 tol
    });

    /// <summary>
    /// Last-position-only logits (<see cref="GGUFGraphBuilder.EnableLastPositionLogits"/>): the graph slices the
    /// final hidden state to its LAST sequence position before output_norm + the LM head, so the head runs at M=1.
    /// For generation only the last token's logits are sampled, so the sliced graph's single logit row MUST equal
    /// the full-recompute graph's LAST position. argmax must match EXACTLY (the token actually sampled); the logit
    /// values agree within a loose tol because the M=1 head is a coalesced GEMV (different K-reduction order than
    /// the M=seq GEMM/per-element head — same numerics-order caveat as the tiled-GEMM A/B). All 6 backends; the
    /// Slice node also exercises the GGUF graph's last-position path on every backend.
    /// </summary>
    [TestMethod]
    public async Task GGUFLastPositionLogits_MatchesFullLastPosition() => await RunTest(async accelerator =>
    {
        const int embd = 256, vocab = 32, ffn = 320, ctx = 64;
        var bytes = BuildTinyQuantizedLlamaGGUF(embd, vocab, ffn, ctx, new Random(7));
        var seq = new float[] { 3, 9, 21, 5 };

        async Task<float[]> LastLogits(bool lastPosOnly)
        {
            bool saved = GGUFGraphBuilder.EnableLastPositionLogits;
            try
            {
                GGUFGraphBuilder.EnableLastPositionLogits = lastPosOnly;
                using var session = InferenceSession.CreateFromGGUF(accelerator, bytes);
                using var inBuf = accelerator.Allocate1D(seq);
                var outputs = await session.RunAsync(new Dictionary<string, Tensor>
                { ["input_ids"] = new Tensor(inBuf.View, new[] { 1, seq.Length }, "input_ids") });
                var lt = outputs.TryGetValue("logits", out var l) ? l : outputs.Values.First();
                int outSeq = lt.ElementCount / vocab;
                if (lastPosOnly && outSeq != 1)
                    throw new Exception($"last-position graph produced {outSeq} logit rows, expected 1 ([1,1,vocab])");
                using var read = accelerator.Allocate1D<float>(vocab);
                await read.View.CopyFromAsync(lt.Data.SubView((long)(outSeq - 1) * vocab, vocab));
                await accelerator.SynchronizeAsync();
                return await read.CopyToHostAsync<float>(0, vocab);
            }
            finally { GGUFGraphBuilder.EnableLastPositionLogits = saved; }
        }

        var full = await LastLogits(lastPosOnly: false); // all positions, take the last
        var sliced = await LastLogits(lastPosOnly: true); // [1,1,vocab]

        int argFull = 0, argSliced = 0;
        for (int v = 1; v < vocab; v++)
        {
            if (full[v] > full[argFull]) argFull = v;
            if (sliced[v] > sliced[argSliced]) argSliced = v;
        }
        if (argFull != argSliced)
            throw new Exception($"GGUFLastPositionLogits argmax {argSliced} != full-recompute last-position argmax {argFull}");

        float maxErr = 0;
        for (int v = 0; v < vocab; v++) maxErr = MathF.Max(maxErr, MathF.Abs(sliced[v] - full[v]));
        Console.WriteLine($"[GGUFLastPos] sliced last-position logits vs full[last]: argmax={argSliced}, maxErr={maxErr:E3}");
        if (maxErr > 1e-2f)
            throw new Exception($"GGUFLastPositionLogits maxErr={maxErr:E3} > 1e-2 — sliced last position differs from full-recompute last position");
    });
}
