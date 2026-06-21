using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Tests for fused attention kernel: correctness vs CPU reference.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task FusedAttention_MatchesCPU() => await RunTest(async accelerator =>
    {
        int BH = 1, SQ = 2, SKV = 3, D = 4;
        var rng = new Random(42);

        var Q = new float[BH * SQ * D];
        var K = new float[BH * SKV * D];
        var V = new float[BH * SKV * D];
        for (int i = 0; i < Q.Length; i++) Q[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < K.Length; i++) K[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < V.Length; i++) V[i] = (float)(rng.NextDouble() * 2 - 1);

        float scale = 1f / MathF.Sqrt(D);

        // CPU reference
        var expected = new float[BH * SQ * D];
        for (int bh = 0; bh < BH; bh++)
            for (int sq = 0; sq < SQ; sq++)
            {
                // Compute scores
                var scores = new float[SKV];
                float maxS = float.MinValue;
                for (int kv = 0; kv < SKV; kv++)
                {
                    float dot = 0;
                    for (int d = 0; d < D; d++)
                        dot += Q[(bh * SQ + sq) * D + d] * K[(bh * SKV + kv) * D + d];
                    scores[kv] = dot * scale;
                    if (scores[kv] > maxS) maxS = scores[kv];
                }
                // Softmax
                float sumE = 0;
                for (int kv = 0; kv < SKV; kv++) { scores[kv] = MathF.Exp(scores[kv] - maxS); sumE += scores[kv]; }
                for (int kv = 0; kv < SKV; kv++) scores[kv] /= sumE;
                // Weighted V
                for (int d = 0; d < D; d++)
                {
                    float val = 0;
                    for (int kv = 0; kv < SKV; kv++)
                        val += scores[kv] * V[(bh * SKV + kv) * D + d];
                    expected[(bh * SQ + sq) * D + d] = val;
                }
            }

        // GPU fused attention
        using var qBuf = accelerator.Allocate1D(Q);
        using var kBuf = accelerator.Allocate1D(K);
        using var vBuf = accelerator.Allocate1D(V);
        using var outBuf = accelerator.Allocate1D<float>(BH * SQ * D);

        var fused = new FusedAttentionKernel(accelerator);
        fused.Forward(qBuf.View, kBuf.View, vBuf.View, outBuf.View, BH, SQ, SKV, D);
        await accelerator.SynchronizeAsync();
        var gpuOut = await outBuf.CopyToHostAsync<float>(0, BH * SQ * D);

        float maxErr = 0;
        for (int i = 0; i < expected.Length; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(gpuOut[i] - expected[i]));

        Console.WriteLine($"[FusedAttention] maxErr={maxErr:E3} vs CPU reference");
        if (maxErr > 0.01f)
            throw new Exception($"FusedAttention maxErr={maxErr:E3} exceeds tolerance 0.01");
    });

    /// <summary>
    /// ForwardStrided (the KV-cache decode path): K/V read in a native type with an explicit per-head stride.
    /// Covers GQA + causal + sliding-window + kvOffset (the gemma4 attention features) in three configs:
    /// (A) f32, contiguous stride = SKV*D — the correctness ANCHOR (byte-identical math to the existing kernel);
    /// (B) bf16 K/V vs a bf16-rounded reference; (C) a maxSeq-strided store (store laid out [kvHeads, maxSeq, D]
    /// with maxSeq &gt; SKV, stride = maxSeq*D) — proving the kernel reads the live SKV tokens out of the padded
    /// store DIRECTLY (the whole point: no per-token repack). All 6 backends.
    /// </summary>
    [TestMethod]
    public async Task FusedAttention_Strided_MatchesReference() => await RunTest(async accelerator =>
    {
        int nHeads = 4, kvHeads = 2, SQ = 3, SKV = 6, D = 8, kvOffset = 2, window = 4, maxSeq = 10;
        bool causal = true;
        var rng = new Random(7);
        float scale = 1f / MathF.Sqrt(D);
        int group = nHeads / kvHeads;

        var Q = new float[nHeads * SQ * D];
        var K = new float[kvHeads * SKV * D];
        var V = new float[kvHeads * SKV * D];
        for (int i = 0; i < Q.Length; i++) Q[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < K.Length; i++) K[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < V.Length; i++) V[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] Reference(Func<int, int, int, float> getK, Func<int, int, int, float> getV)
        {
            var exp = new float[nHeads * SQ * D];
            for (int bh = 0; bh < nHeads; bh++)
            {
                int kvh = bh / group;
                for (int sq = 0; sq < SQ; sq++)
                {
                    int qPos = kvOffset + sq;
                    var sc = new float[SKV]; float mx = -1e30f;
                    for (int kv = 0; kv < SKV; kv++)
                    {
                        bool ok = (!causal || kv <= qPos) && (kv > qPos - window);
                        float dot = 0; for (int d = 0; d < D; d++) dot += Q[(bh * SQ + sq) * D + d] * getK(kvh, kv, d);
                        sc[kv] = ok ? dot * scale : -1e30f;
                        if (sc[kv] > mx) mx = sc[kv];
                    }
                    float sum = 0; for (int kv = 0; kv < SKV; kv++) { sc[kv] = MathF.Exp(sc[kv] - mx); sum += sc[kv]; }
                    for (int d = 0; d < D; d++)
                    {
                        float v = 0; for (int kv = 0; kv < SKV; kv++) v += sc[kv] * getV(kvh, kv, d);
                        exp[(bh * SQ + sq) * D + d] = v / (sum + 1e-10f);
                    }
                }
            }
            return exp;
        }

        var fused = new FusedAttentionKernel(accelerator);
        using var qBuf = accelerator.Allocate1D(Q);
        using var outBuf = accelerator.Allocate1D<float>(nHeads * SQ * D);
        async Task Assert(float[] reference, float tol, string name)
        {
            await accelerator.SynchronizeAsync();
            var gpu = await outBuf.CopyToHostAsync<float>(0, nHeads * SQ * D);
            float e = 0; for (int i = 0; i < gpu.Length; i++) e = MathF.Max(e, MathF.Abs(gpu[i] - reference[i]));
            if (e > tol) throw new Exception($"FusedAttention_Strided {name}: maxErr={e:E3} > {tol}");
        }

        // (A) f32 contiguous — the anchor.
        var refF32 = Reference((h, kv, d) => K[(h * SKV + kv) * D + d], (h, kv, d) => V[(h * SKV + kv) * D + d]);
        using (var kBuf = accelerator.Allocate1D(K))
        using (var vBuf = accelerator.Allocate1D(V))
        {
            fused.ForwardStrided<float>(qBuf.View, kBuf.View, vBuf.View, outBuf.View, nHeads, kvHeads, SQ, SKV, D, causal, window, kvOffset, scale, SKV * D);
            await Assert(refF32, 1e-3f, "f32 contiguous (anchor)");
        }

        // (B) bf16 K/V vs a bf16-rounded reference.
        var Kb = new global::ILGPU.BFloat16[K.Length]; for (int i = 0; i < K.Length; i++) Kb[i] = (global::ILGPU.BFloat16)K[i];
        var Vb = new global::ILGPU.BFloat16[V.Length]; for (int i = 0; i < V.Length; i++) Vb[i] = (global::ILGPU.BFloat16)V[i];
        var refBf = Reference((h, kv, d) => (float)Kb[(h * SKV + kv) * D + d], (h, kv, d) => (float)Vb[(h * SKV + kv) * D + d]);
        using (var kBuf = accelerator.Allocate1D(Kb))
        using (var vBuf = accelerator.Allocate1D(Vb))
        {
            fused.ForwardStrided<global::ILGPU.BFloat16>(qBuf.View, kBuf.View, vBuf.View, outBuf.View, nHeads, kvHeads, SQ, SKV, D, causal, window, kvOffset, scale, SKV * D);
            await Assert(refBf, 2e-2f, "bf16 contiguous");
        }

        // (C) maxSeq-strided store (the real cache layout): live SKV tokens out of a [kvHeads, maxSeq, D] store.
        var Ks = new float[kvHeads * maxSeq * D]; var Vs = new float[kvHeads * maxSeq * D];
        for (int h = 0; h < kvHeads; h++) for (int kv = 0; kv < SKV; kv++) for (int d = 0; d < D; d++)
        { Ks[(h * maxSeq + kv) * D + d] = K[(h * SKV + kv) * D + d]; Vs[(h * maxSeq + kv) * D + d] = V[(h * SKV + kv) * D + d]; }
        using (var kBuf = accelerator.Allocate1D(Ks))
        using (var vBuf = accelerator.Allocate1D(Vs))
        {
            fused.ForwardStrided<float>(qBuf.View, kBuf.View, vBuf.View, outBuf.View, nHeads, kvHeads, SQ, SKV, D, causal, window, kvOffset, scale, maxSeq * D);
            await Assert(refF32, 1e-3f, "f32 maxSeq-strided store");
        }
    });

    /// <summary>
    /// The grouped-per-query kernel (env GGUF_ATTN_GROUP / <see cref="FusedAttentionKernel.EnableGroupedAttention"/>)
    /// computes each Q·K score ONCE in shared memory then replays the EXACT per-element online-softmax recurrence
    /// per output dim — so it must be BIT-IDENTICAL to the per-element kernel. This asserts grouped output ==
    /// per-element output (the anchor), NOT a fresh CPU reference, across every attention feature: GQA, causal,
    /// sliding-window, kvOffset, custom scale, attention sinks, D=256 (gemma4's head dim, &gt; the group size),
    /// the bf16 strided KV path, a maxSeq-strided store, AND the SKV-overflow fall-back (grouped on but SKV beyond
    /// the shared cap must transparently run the per-element kernel). On WebGL/WebGPU the dispatch falls back to
    /// per-element (no workgroup shared memory / slow workgroup reduction), so the equality holds trivially there.
    /// All 6 backends.
    /// </summary>
    [TestMethod]
    public async Task FusedAttention_Grouped_MatchesPerElement() => await RunTest(async accelerator =>
    {
        var rng = new Random(13);
        var fused = new FusedAttentionKernel(accelerator);
        bool savedFlag = FusedAttentionKernel.EnableGroupedAttention;

        // Run `body` once with the grouped path forced OFF (per-element anchor) and once forced ON, returning
        // the max abs difference between the two output buffers. The grouped kernel only actually engages on a
        // non-browser-GPU backend within the shared-memory gate; otherwise it falls back to per-element and the
        // difference is exactly zero — either way the kernels must agree. NOTE: `body` MUST await
        // SynchronizeAsync before its own input buffers leave scope — on WebGPU/Wasm disposing a buffer that a
        // still-pending dispatch references corrupts the submit (CLAUDE.md "Never Dispose Buffers Before Flush").
        async Task<float> Compare(int outLen, Func<FusedAttentionKernel, MemoryBuffer1D<float, Stride1D.Dense>, Task> body)
        {
            using var outA = accelerator.Allocate1D<float>(outLen);
            using var outB = accelerator.Allocate1D<float>(outLen);
            try
            {
                FusedAttentionKernel.EnableGroupedAttention = false;
                await body(fused, outA);
                FusedAttentionKernel.EnableGroupedAttention = true;
                await body(fused, outB);
            }
            finally { FusedAttentionKernel.EnableGroupedAttention = savedFlag; }
            var a = await outA.CopyToHostAsync<float>(0, outLen);
            var b = await outB.CopyToHostAsync<float>(0, outLen);
            float e = 0; for (int i = 0; i < outLen; i++) e = MathF.Max(e, MathF.Abs(a[i] - b[i]));
            return e;
        }

        float[] Rand(int n) { var x = new float[n]; for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 2 - 1); return x; }
        void Check(float e, string name) { Console.WriteLine($"[Grouped] {name}: maxDiff={e:E3}"); if (e > 1e-4f) throw new Exception($"FusedAttention grouped {name}: maxDiff={e:E3} > 1e-4 (must match per-element)"); }

        // (1) Contiguous Forward, GQA + causal + sliding-window + custom scale. SKV (40) > group size (64? no) —
        // pick SKV spanning multiple phase-1 strides and SQ > 1 so the kvOffset/causal interplay is exercised.
        {
            int nHeads = 6, kvHeads = 3, SQ = 5, SKV = 40, D = 128;
            var Q = Rand(nHeads * SQ * D); var K = Rand(kvHeads * SKV * D); var V = Rand(kvHeads * SKV * D);
            float scale = 0.123f;
            float e = await Compare(nHeads * SQ * D, async (f, o) =>
            {
                using var qb = accelerator.Allocate1D(Q); using var kb = accelerator.Allocate1D(K); using var vb = accelerator.Allocate1D(V);
                f.Forward(qb.View, kb.View, vb.View, o.View, nHeads, kvHeads, SQ, SKV, D, causal: true, window: 24, kvOffset: 7, scale: scale);
                await accelerator.SynchronizeAsync();
            });
            Check(e, "contiguous GQA+causal+window D=128");
        }

        // (2) Contiguous Forward with attention SINKS (gpt-oss): per-head sink logit folded into the denominator.
        {
            int nHeads = 4, kvHeads = 2, SQ = 3, SKV = 30, D = 64;
            var Q = Rand(nHeads * SQ * D); var K = Rand(kvHeads * SKV * D); var V = Rand(kvHeads * SKV * D);
            var sinks = Rand(nHeads);
            float e = await Compare(nHeads * SQ * D, async (f, o) =>
            {
                using var qb = accelerator.Allocate1D(Q); using var kb = accelerator.Allocate1D(K); using var vb = accelerator.Allocate1D(V);
                using var sb = accelerator.Allocate1D(sinks);
                f.Forward(qb.View, kb.View, vb.View, o.View, nHeads, kvHeads, SQ, SKV, D, causal: true, window: int.MaxValue, kvOffset: 0, scale: 0f, sinks: sb.View, sinkCount: nHeads);
                await accelerator.SynchronizeAsync();
            });
            Check(e, "contiguous attention sinks D=64");
        }

        // (3) D=256 (gemma4 head dim) — exercises the phase-2 d-slice loop with D > the group size.
        {
            int nHeads = 2, kvHeads = 1, SQ = 4, SKV = 20, D = 256;
            var Q = Rand(nHeads * SQ * D); var K = Rand(kvHeads * SKV * D); var V = Rand(kvHeads * SKV * D);
            float e = await Compare(nHeads * SQ * D, async (f, o) =>
            {
                using var qb = accelerator.Allocate1D(Q); using var kb = accelerator.Allocate1D(K); using var vb = accelerator.Allocate1D(V);
                f.Forward(qb.View, kb.View, vb.View, o.View, nHeads, kvHeads, SQ, SKV, D, causal: true, window: int.MaxValue, kvOffset: 0, scale: 0f);
                await accelerator.SynchronizeAsync();
            });
            Check(e, "contiguous D=256 (gemma head dim)");
        }

        // (4) Strided bf16 KV (the prefill hotpath) on a maxSeq-strided store + GQA + causal + window + kvOffset.
        {
            int nHeads = 4, kvHeads = 2, SQ = 3, SKV = 18, D = 128, kvOffset = 5, maxSeq = 32;
            var Q = Rand(nHeads * SQ * D);
            var Kf = Rand(kvHeads * SKV * D); var Vf = Rand(kvHeads * SKV * D);
            var Ks = new global::ILGPU.BFloat16[kvHeads * maxSeq * D]; var Vs = new global::ILGPU.BFloat16[kvHeads * maxSeq * D];
            for (int h = 0; h < kvHeads; h++) for (int kv = 0; kv < SKV; kv++) for (int d = 0; d < D; d++)
            { Ks[(h * maxSeq + kv) * D + d] = (global::ILGPU.BFloat16)Kf[(h * SKV + kv) * D + d]; Vs[(h * maxSeq + kv) * D + d] = (global::ILGPU.BFloat16)Vf[(h * SKV + kv) * D + d]; }
            float e = await Compare(nHeads * SQ * D, async (f, o) =>
            {
                using var qb = accelerator.Allocate1D(Q); using var kb = accelerator.Allocate1D(Ks); using var vb = accelerator.Allocate1D(Vs);
                f.ForwardStrided<global::ILGPU.BFloat16>(qb.View, kb.View, vb.View, o.View, nHeads, kvHeads, SQ, SKV, D, causal: true, window: 12, kvOffset: kvOffset, scale: 0f, kvRowStride: maxSeq * D);
                await accelerator.SynchronizeAsync();
            });
            Check(e, "strided bf16 maxSeq-store GQA D=128");
        }

        // (5) Large SKV ABOVE the single-pass cap (4096) → the KV-TILED kernel, spanning many 512-blocks (the
        // multi-block online-softmax carry + a partial last block). Must equal the per-element kernel exactly.
        {
            int nHeads = 2, kvHeads = 1, SQ = 2, SKV = 5000, D = 64, kvOffset = 100;
            var Q = Rand(nHeads * SQ * D); var K = Rand(kvHeads * SKV * D); var V = Rand(kvHeads * SKV * D);
            float e = await Compare(nHeads * SQ * D, async (f, o) =>
            {
                using var qb = accelerator.Allocate1D(Q); using var kb = accelerator.Allocate1D(K); using var vb = accelerator.Allocate1D(V);
                f.Forward(qb.View, kb.View, vb.View, o.View, nHeads, kvHeads, SQ, SKV, D, causal: true, window: 700, kvOffset: kvOffset, scale: 0f);
                await accelerator.SynchronizeAsync();
            });
            Check(e, "large SKV=5000 (>cap) KV-tiled + causal + window");
        }
    });
}
