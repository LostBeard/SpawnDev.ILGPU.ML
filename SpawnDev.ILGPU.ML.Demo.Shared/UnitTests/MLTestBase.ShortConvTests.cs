using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// KERNEL-level CPU-oracle gate for LFM2's ShortConv (the LIV double-gated causal depthwise conv).
///
/// Why this file exists (2026-07-16): ShortConv shipped "WebGPU-verified" on nothing but a full-model
/// E2E whose oracle was Contains("Paris") - which passes on token soup, because LFM2 emits "Paris"
/// before it degenerates. The demo produced garbage on WebGPU while CUDA was coherent, and no kernel
/// test existed to localize it. A CPU reference comparison is the minimum bar (global Rule 1); these
/// tests run on every PMT backend, so a backend-specific WGSL/GLSL defect fails HERE - in seconds, on
/// random data with a real oracle - instead of hiding behind a 730MB model's plausible-looking text.
///
/// Real LFM2-1.2B dims are used (H=2048, L=3, from lfm2.embedding_length / lfm2.shortconv.l_cache).
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>CPU reference for the fused shortconv, mirroring ShortConvKernel's contract exactly:
    ///   Bx[t,c]   = B[t,c] * x[t,c]                              (B = chunk 0, x = chunk 2)
    ///   conv[t,c] = sum_k W[c,k] * Bx[t-(L-1)+k, c]              (causal; taps before `state` are 0)
    ///   y[t,c]    = C[t,c] * conv[t,c]                           (C = chunk 1, current token)
    /// bcx is [seq,3H]; state (optional) is the previous [stateRows,3H] bcx rows.</summary>
    protected static float[] CpuShortConv(float[] bcx, float[] weight, int seq, int H, int L,
        float[]? state = null, int stateRows = 0)
    {
        var y = new float[seq * H];
        for (int t = 0; t < seq; t++)
            for (int c = 0; c < H; c++)
            {
                double acc = 0.0;
                for (int k = 0; k < L; k++)
                {
                    int tt = t - (L - 1) + k;
                    float b, x;
                    if (tt >= 0)
                    {
                        int row = tt * 3 * H;
                        b = bcx[row + c];
                        x = bcx[row + 2 * H + c];
                    }
                    else
                    {
                        int st = tt + stateRows;
                        if (state != null && st >= 0)
                        {
                            int row = st * 3 * H;
                            b = state[row + c];
                            x = state[row + 2 * H + c];
                        }
                        else { b = 0f; x = 0f; }
                    }
                    acc += (double)weight[c * L + k] * ((double)b * x);
                }
                y[t * H + c] = bcx[t * 3 * H + H + c] * (float)acc;
            }
        return y;
    }

    // The core gate: real LFM2 shape (H=2048, L=3), multi-token prefill, random data, CPU oracle.
    // Runs on every PMT backend - CUDA/OpenCL/CPU pass while WebGPU/WebGL fail => a codegen defect.
    [TestMethod(Timeout = 300000)]
    public async Task ShortConv_Lfm2Dims_MatchesCpu() => await RunTest(async accelerator =>
    {
        const int seq = 12, H = 2048, L = 3;
        var bcx = RandomFloats(seq * 3 * H, seed: 401, scale: 1f);
        var weight = RandomFloats(H * L, seed: 402, scale: 1f);
        var expected = CpuShortConv(bcx, weight, seq, H, L);

        using var bcxBuf = accelerator.Allocate1D((float[])bcx.Clone());
        using var wBuf = accelerator.Allocate1D((float[])weight.Clone());
        using var yBuf = accelerator.Allocate1D<float>(seq * H);
        using var sc = new ShortConvKernel(accelerator);
        sc.Forward(bcxBuf.View, wBuf.View, yBuf.View, seq, H, L);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, yBuf.View, expected, 1e-4f, "ShortConv[LFM2 2048x3]: ");
    });

    // Single-token (seq=1) is the DECODE shape: every tap except the last comes from state/zero-pad.
    // A loop/bounds defect that a 12-token prefill averages over shows up sharply here.
    [TestMethod(Timeout = 300000)]
    public async Task ShortConv_SingleToken_MatchesCpu() => await RunTest(async accelerator =>
    {
        const int seq = 1, H = 2048, L = 3;
        var bcx = RandomFloats(seq * 3 * H, seed: 403, scale: 1f);
        var weight = RandomFloats(H * L, seed: 404, scale: 1f);
        var expected = CpuShortConv(bcx, weight, seq, H, L);

        using var bcxBuf = accelerator.Allocate1D((float[])bcx.Clone());
        using var wBuf = accelerator.Allocate1D((float[])weight.Clone());
        using var yBuf = accelerator.Allocate1D<float>(seq * H);
        using var sc = new ShortConvKernel(accelerator);
        sc.Forward(bcxBuf.View, wBuf.View, yBuf.View, seq, H, L);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, yBuf.View, expected, 1e-4f, "ShortConv[seq=1]: ");
    });

    // The state path (KV-decode): taps before the chunk must read `state`, not zero-pad.
    [TestMethod(Timeout = 300000)]
    public async Task ShortConvWithState_MatchesCpu() => await RunTest(async accelerator =>
    {
        const int seq = 4, H = 512, L = 3, stateRows = L - 1;
        var bcx = RandomFloats(seq * 3 * H, seed: 405, scale: 1f);
        var weight = RandomFloats(H * L, seed: 406, scale: 1f);
        var state = RandomFloats(stateRows * 3 * H, seed: 407, scale: 1f);
        var expected = CpuShortConv(bcx, weight, seq, H, L, state, stateRows);

        using var bcxBuf = accelerator.Allocate1D((float[])bcx.Clone());
        using var wBuf = accelerator.Allocate1D((float[])weight.Clone());
        using var stBuf = accelerator.Allocate1D((float[])state.Clone());
        using var yBuf = accelerator.Allocate1D<float>(seq * H);
        using var sc = new ShortConvKernel(accelerator);
        sc.ForwardWithState(bcxBuf.View, wBuf.View, yBuf.View, stBuf.View, seq, H, L, stateRows);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, yBuf.View, expected, 1e-4f, "ShortConvWithState: ");
    });

    /// <summary>Tiny synthetic LFM2 GGUF: layer 0 = shortconv mixer (blk.0.shortconv.*), layer 1 = attention -
    /// the real arch's hybrid interleave, at a size every backend runs in milliseconds. The graph builder selects
    /// the shortconv mixer purely on tensor presence, and L is derived from the conv weight (ne=[L,H]), so no
    /// shortconv metadata is needed.</summary>
    private static byte[] BuildTinyLfm2GGUF(int embd, int vocab, int ffn, int ctx, int L, Random rng)
    {
        byte[] F32Bytes(int count, float center = 0f, float spread = 0.5f)
        {
            var b = new byte[count * 4];
            for (int i = 0; i < count; i++)
                BitConverter.GetBytes(center + (float)(rng.NextDouble() - 0.5) * spread).CopyTo(b, i * 4);
            return b;
        }
        byte[] QRows(SpawnDev.ILGPU.ML.GGUF.GGMLType t, int k, int n)
        {
            int rb = RowBytes(t, k);
            var all = new byte[n * rb];
            for (int r = 0; r < n; r++)
                Buffer.BlockCopy(MakeBlocks(t, k, rng), 0, all, r * rb, rb);
            return all;
        }
        var GG = SpawnDev.ILGPU.ML.GGUF.GGMLType.Q4_K;
        var tensors = new List<(string Name, long[] Ne, SpawnDev.ILGPU.ML.GGUF.GGMLType Type, byte[] Data)>
        {
            ("token_embd.weight", new long[] { embd, vocab }, SpawnDev.ILGPU.ML.GGUF.GGMLType.Q6_K,
                QRows(SpawnDev.ILGPU.ML.GGUF.GGMLType.Q6_K, embd, vocab)),

            // ── layer 0: shortconv mixer ──
            ("blk.0.attn_norm.weight", new long[] { embd }, SpawnDev.ILGPU.ML.GGUF.GGMLType.F32, F32Bytes(embd, center: 1f, spread: 0.2f)),
            ("blk.0.shortconv.in_proj.weight", new long[] { embd, 3 * embd }, GG, QRows(GG, embd, 3 * embd)),
            // conv weight ne = [L, H] (fastest-first) -> our [H, L]; F32 like the real GGUF's conv kernel.
            ("blk.0.shortconv.conv.weight", new long[] { L, embd }, SpawnDev.ILGPU.ML.GGUF.GGMLType.F32, F32Bytes(embd * L, spread: 1.0f)),
            ("blk.0.shortconv.out_proj.weight", new long[] { embd, embd }, GG, QRows(GG, embd, embd)),
            ("blk.0.ffn_norm.weight", new long[] { embd }, SpawnDev.ILGPU.ML.GGUF.GGMLType.F32, F32Bytes(embd, center: 1f, spread: 0.2f)),
            ("blk.0.ffn_gate.weight", new long[] { embd, ffn }, GG, QRows(GG, embd, ffn)),
            ("blk.0.ffn_up.weight", new long[] { embd, ffn }, GG, QRows(GG, embd, ffn)),
            ("blk.0.ffn_down.weight", new long[] { ffn, embd }, SpawnDev.ILGPU.ML.GGUF.GGMLType.F32, F32Bytes(ffn * embd, spread: 0.1f)),

            // ── layer 1: attention (the hybrid's other half) ──
            ("blk.1.attn_norm.weight", new long[] { embd }, SpawnDev.ILGPU.ML.GGUF.GGMLType.F32, F32Bytes(embd, center: 1f, spread: 0.2f)),
            ("blk.1.attn_q.weight", new long[] { embd, embd }, GG, QRows(GG, embd, embd)),
            ("blk.1.attn_k.weight", new long[] { embd, embd }, GG, QRows(GG, embd, embd)),
            ("blk.1.attn_v.weight", new long[] { embd, embd }, GG, QRows(GG, embd, embd)),
            ("blk.1.attn_output.weight", new long[] { embd, embd }, GG, QRows(GG, embd, embd)),
            ("blk.1.ffn_norm.weight", new long[] { embd }, SpawnDev.ILGPU.ML.GGUF.GGMLType.F32, F32Bytes(embd, center: 1f, spread: 0.2f)),
            ("blk.1.ffn_gate.weight", new long[] { embd, ffn }, GG, QRows(GG, embd, ffn)),
            ("blk.1.ffn_up.weight", new long[] { embd, ffn }, GG, QRows(GG, embd, ffn)),
            ("blk.1.ffn_down.weight", new long[] { ffn, embd }, SpawnDev.ILGPU.ML.GGUF.GGMLType.F32, F32Bytes(ffn * embd, spread: 0.1f)),

            ("output_norm.weight", new long[] { embd }, SpawnDev.ILGPU.ML.GGUF.GGMLType.F32, F32Bytes(embd, center: 1f, spread: 0.2f)),
            // NO output.weight -> tied-embedding LM head.
        };
        var metadata = new (string Key, object Value)[]
        {
            ("general.architecture", "lfm2"),
            ("general.name", "tiny-lfm2-test"),
            ("lfm2.embedding_length", (uint)embd),
            ("lfm2.block_count", 2u),
            ("lfm2.attention.head_count", 4u),
            ("lfm2.attention.head_count_kv", 4u),
            ("lfm2.vocab_size", (uint)vocab),
            ("lfm2.feed_forward_length", (uint)ffn),
            ("lfm2.context_length", (uint)ctx),
        };
        return SerializeGGUF(tensors, metadata);
    }

    /// <summary>
    /// THE gate the demo actually depends on: LFM2 incremental KV-decode must equal full-recompute, on every
    /// backend. Prefill+decode is how the demo generates; the shortconv layers carry their history in
    /// ShortConvStateCache (auto-created by EnableGGUFDecode when the graph has ShortConv nodes) rather than the
    /// KV cache. A conv-state defect makes the model coherent for the prefill then progressively garbage - which
    /// is exactly what shipped (2026-07-16), because decode equivalence was only ever verified on CUDA/OpenCL.
    /// Tiny synthetic hybrid model (shortconv layer + attention layer) so this runs in ms on all 6 backends.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task Lfm2Decode_IncrementalMatchesFullRecompute() => await RunTest(async accelerator =>
    {
        // embd must be a multiple of the Q6_K block size (256) - token_embd is Q6_K and the gather
        // dequantizes whole blocks per row.
        const int embd = 256, vocab = 32, ffn = 320, ctx = 64, L = 3;
        var bytes = BuildTinyLfm2GGUF(embd, vocab, ffn, ctx, L, new Random(11));
        var model = SpawnDev.ILGPU.ML.GGUF.GGUFParser.Parse(bytes);

        int nLayers = (int)model.BlockCount, nHeads = (int)model.AttentionHeadCount;
        int defNKV = (int)model.AttentionHeadCountKV; if (defNKV == 0) defNKV = nHeads;
        int defHd = embd / nHeads;
        var kvHeadsArr = new int[nLayers]; var hdArr = new int[nLayers];
        for (int i = 0; i < nLayers; i++)
        {
            var cfg = SpawnDev.ILGPU.ML.GGUF.GGUFGraphBuilder.GetLayerAttnConfig(model, i, nHeads, defNKV, defHd);
            kvHeadsArr[i] = cfg.NKVHeads; hdArr[i] = cfg.HeadDim;
        }

        // Longer than L so the conv history actually spans decode-step boundaries (the whole point).
        var seq = new float[] { 3, 9, 21, 5, 17, 2 };

        using var session = InferenceSession.CreateFromGGUF(accelerator, bytes);
        using var inFull = accelerator.Allocate1D(seq);
        var outFull = await session.RunAsync(new Dictionary<string, Tensor>
        { ["input_ids"] = new Tensor(inFull.View, new[] { 1, seq.Length }, "input_ids") });
        var logitsFullT = outFull.TryGetValue("logits", out var lf) ? lf : outFull.Values.First();
        using var readFull = accelerator.Allocate1D<float>(seq.Length * vocab);
        await readFull.View.CopyFromAsync(logitsFullT.Data.SubView(0, seq.Length * vocab));
        await accelerator.SynchronizeAsync();
        var logitsFull = await readFull.CopyToHostAsync<float>(0, seq.Length * vocab);

        using var kv = new ShortConvKernelDecodeCacheScope(accelerator, kvHeadsArr, hdArr, ctx);
        session.EnableGGUFDecode(kv.Cache);
        try
        {
            for (int pos = 0; pos < seq.Length; pos++)
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
                    throw new Exception($"[LFM2 decode] step {pos}: decode argmax {argKV} != full-recompute argmax {argFull} " +
                        "- incremental decode diverges (conv-state cache / KV / RoPE-offset). This is the defect that " +
                        "makes the demo coherent at first, then degenerate.");
                for (int v = 0; v < vocab; v++)
                {
                    float fv = logitsFull[pos * vocab + v];
                    float tol = MathF.Max(2e-3f, MathF.Abs(fv) * 2e-3f);
                    if (MathF.Abs(stepLogits[v] - fv) > tol)
                        throw new Exception($"[LFM2 decode] step {pos} vocab {v}: decode logit {stepLogits[v]} vs full {fv} " +
                            $"(tol {tol}) - incremental decode is not numerically equivalent to full-recompute.");
                }
            }
            Console.WriteLine($"[LFM2 decode] incremental == full-recompute for all {seq.Length} positions (shortconv + attention layers).");
        }
        finally { session.DisableGGUFDecode(); }
    });

    /// <summary>Owns the decode KV cache for the LFM2 decode test (the conv-state cache is auto-created by
    /// EnableGGUFDecode).</summary>
    private sealed class ShortConvKernelDecodeCacheScope : IDisposable
    {
        public GGUFDecodeKVCache Cache { get; }
        public ShortConvKernelDecodeCacheScope(Accelerator acc, int[] kvHeads, int[] hd, int maxSeqLen)
            => Cache = new GGUFDecodeKVCache(acc, kvHeads, hd, maxSeqLen: maxSeqLen);
        public void Dispose() => Cache.Dispose();
    }

    // Decode equivalence (the property the demo actually depends on): prefill N tokens, then feed the
    // remaining tokens ONE AT A TIME through ShortConvStateCache. The concatenated result must equal the
    // full-sequence conv. This is what "coherent for a few tokens then garbage" looks like when broken.
    [TestMethod(Timeout = 300000)]
    public async Task ShortConvStateCache_TokenByToken_MatchesFullSequence() => await RunTest(async accelerator =>
    {
        const int total = 8, prefill = 5, H = 256, L = 3;
        var bcx = RandomFloats(total * 3 * H, seed: 408, scale: 1f);
        var weight = RandomFloats(H * L, seed: 409, scale: 1f);
        var expected = CpuShortConv(bcx, weight, total, H, L);

        using var bcxBuf = accelerator.Allocate1D((float[])bcx.Clone());
        using var wBuf = accelerator.Allocate1D((float[])weight.Clone());
        using var yBuf = accelerator.Allocate1D<float>(total * H);
        using var sc = new ShortConvKernel(accelerator);
        using var cache = new ShortConvStateCache(accelerator, sc);

        // Prefill (pastLen=0 => zero-pad + snapshot state).
        cache.Forward(layer: 0, bcxBuf.View.SubView(0, (long)prefill * 3 * H), wBuf.View,
            yBuf.View.SubView(0, (long)prefill * H), prefill, H, L, pastLen: 0);
        // Decode the rest one token at a time (pastLen>0 => prepend state).
        for (int t = prefill; t < total; t++)
            cache.Forward(layer: 0, bcxBuf.View.SubView((long)t * 3 * H, 3 * H), wBuf.View,
                yBuf.View.SubView((long)t * H, H), 1, H, L, pastLen: t);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, yBuf.View, expected, 1e-4f, "ShortConv decode-vs-full: ");
    });
}
