using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// FusedAttentionKernel masking correctness - causal, sliding-window (gemma4's 5:1
/// SWA/global interleave passes a per-layer window), KV-cache decode offset, and the
/// no-mask regression (original API behavior unchanged). Oracle = plain CPU softmax
/// attention with an explicit mask predicate, computed independently of the kernel's
/// online-softmax/arithmetic-mask formulation.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task AttnMask_Causal_MatchesCPU() => await RunTest(async accelerator =>
        await AttnMaskOracle(accelerator, batchHeads: 2, seqQ: 8, seqKV: 8, headDim: 16,
            causal: true, window: int.MaxValue, kvOffset: 0));

    [TestMethod]
    public async Task AttnMask_SlidingWindow_MatchesCPU() => await RunTest(async accelerator =>
        await AttnMaskOracle(accelerator, batchHeads: 2, seqQ: 12, seqKV: 12, headDim: 16,
            causal: true, window: 4, kvOffset: 0));

    [TestMethod]
    public async Task AttnMask_DecodeWithKvOffset_MatchesCPU() => await RunTest(async accelerator =>
        // Single-token decode against a 16-entry KV cache, query at position 15,
        // sliding window 8 - the gemma4 decode shape.
        await AttnMaskOracle(accelerator, batchHeads: 4, seqQ: 1, seqKV: 16, headDim: 32,
            causal: true, window: 8, kvOffset: 15));

    [TestMethod]
    public async Task AttnMask_NoMask_RegressionMatchesCPU() => await RunTest(async accelerator =>
        // Original bidirectional API (defaults) - behavior must be unchanged.
        await AttnMaskOracle(accelerator, batchHeads: 2, seqQ: 6, seqKV: 6, headDim: 16,
            causal: false, window: int.MaxValue, kvOffset: 0));

    private static async Task AttnMaskOracle(Accelerator accelerator,
        int batchHeads, int seqQ, int seqKV, int headDim,
        bool causal, int window, int kvOffset)
    {
        var rng = new Random(17 + seqQ * 31 + window);
        var q = new float[batchHeads * seqQ * headDim];
        var k = new float[batchHeads * seqKV * headDim];
        var v = new float[batchHeads * seqKV * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < k.Length; i++) k[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);

        // CPU oracle: explicit mask predicate + standard two-pass softmax.
        float scale = 1f / MathF.Sqrt(headDim);
        var expected = new float[batchHeads * seqQ * headDim];
        for (int bh = 0; bh < batchHeads; bh++)
            for (int sq = 0; sq < seqQ; sq++)
            {
                long qPos = kvOffset + sq;
                var scores = new double[seqKV];
                var valid = new bool[seqKV];
                double max = double.NegativeInfinity;
                for (int kv = 0; kv < seqKV; kv++)
                {
                    valid[kv] = (!causal || kv <= qPos) && kv > qPos - (long)window;
                    if (!valid[kv]) continue;
                    double dot = 0;
                    for (int dd = 0; dd < headDim; dd++)
                        dot += q[(bh * seqQ + sq) * headDim + dd] * k[(bh * seqKV + kv) * headDim + dd];
                    scores[kv] = dot * scale;
                    if (scores[kv] > max) max = scores[kv];
                }
                double sum = 0;
                for (int kv = 0; kv < seqKV; kv++)
                    if (valid[kv]) sum += Math.Exp(scores[kv] - max);
                for (int dd = 0; dd < headDim; dd++)
                {
                    double acc = 0;
                    for (int kv = 0; kv < seqKV; kv++)
                        if (valid[kv])
                            acc += Math.Exp(scores[kv] - max) / sum * v[(bh * seqKV + kv) * headDim + dd];
                    expected[(bh * seqQ + sq) * headDim + dd] = (float)acc;
                }
            }

        using var qBuf = accelerator.Allocate1D(q);
        using var kBuf = accelerator.Allocate1D(k);
        using var vBuf = accelerator.Allocate1D(v);
        using var outBuf = accelerator.Allocate1D<float>(expected.Length);

        using var attn = new FusedAttentionKernel(accelerator);
        attn.Forward(qBuf.View, kBuf.View, vBuf.View, outBuf.View,
            batchHeads, seqQ, seqKV, headDim, causal, window, kvOffset);
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, expected.Length);

        AssertCloseQuant(got, expected, 2e-3f,
            $"AttnMask causal={causal} window={window} kvOffset={kvOffset}");
        Console.WriteLine($"[AttnMask] causal={causal} window={window} kvOffset={kvOffset} " +
            $"BH={batchHeads} SQ={seqQ} SKV={seqKV} D={headDim}: matches CPU oracle");
    }
}
