using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// RoPE + FusedAttention OPERATOR tests - the production node path (Resolve -> Execute
/// with attributes), per the gemma4 contract: GQA head grouping, explicit scale
/// (query_pre_attn_scalar), multi-head rows_per_position RoPE, per-layer window/base.
/// Oracles are independent CPU implementations.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task AttnOp_RoPE_MultiHeadRowsPerPosition_MatchesCPU() => await RunTest(async accelerator =>
    {
        // Pre-transpose layout [seq, heads, headDim]: every head of a position rotates
        // by THAT position's angle (rows_per_position = heads).
        const int seq = 6, heads = 3, headDim = 16;
        const float ropeBase = 1000000f; // gemma4 global-layer base
        const int kvOffset = 4;
        var rng = new Random(61);
        var x = new float[seq * heads * headDim];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        var expected = (float[])x.Clone();
        int half = headDim / 2;
        for (int s = 0; s < seq; s++)
            for (int h = 0; h < heads; h++)
                for (int i = 0; i < half; i++)
                {
                    long pos = kvOffset + s;
                    double theta = pos / Math.Pow(ropeBase, 2.0 * i / headDim);
                    double c = Math.Cos(theta), sn = Math.Sin(theta);
                    int row = (s * heads + h) * headDim;
                    double x0 = x[row + i], x1 = x[row + i + half];
                    expected[row + i] = (float)(x0 * c - x1 * sn);
                    expected[row + i + half] = (float)(x0 * sn + x1 * c);
                }

        using var inBuf = accelerator.Allocate1D(x);
        using var outBuf = accelerator.Allocate1D<float>(x.Length);
        using var registry = new OperatorRegistry(accelerator);
        var op = registry.Resolve("RoPE");
        var pool = new BufferPool(accelerator);
        op.Execute(new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { seq, heads, headDim }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { seq, heads, headDim }) },
            Attributes = new Dictionary<string, object>
            {
                ["rope_base"] = ropeBase,
                ["kv_offset"] = kvOffset,
                ["rows_per_position"] = heads,
            },
            Pool = pool,
            InputNames = new[] { "x" },
        });
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, x.Length);

        AssertCloseQuant(got, expected, 2e-3f, "RoPE op multi-head");
        Console.WriteLine("[AttnOp] RoPE rows_per_position=heads matches CPU oracle");
    });

    [TestMethod]
    public async Task AttnOp_FusedAttention_GQA_SWA_MatchesCPU() => await RunTest(async accelerator =>
        // gemma4-shaped: 4 query heads on 2 kv heads, causal + window 4.
        await FusedAttnOpOracle(accelerator, nHeads: 4, kvHeads: 2, seqQ: 10, seqKV: 10,
            headDim: 16, causal: true, window: 4, scale: 0f, kvOffset: 0));

    [TestMethod]
    public async Task AttnOp_FusedAttention_ExplicitScale_MatchesCPU() => await RunTest(async accelerator =>
        // gemma's query_pre_attn_scalar path: scale != 1/sqrt(headDim).
        await FusedAttnOpOracle(accelerator, nHeads: 2, kvHeads: 2, seqQ: 6, seqKV: 6,
            headDim: 16, causal: true, window: 0, scale: 0.0883883476f, kvOffset: 0));

    [TestMethod]
    public async Task AttnOp_FusedAttention_GQA_DecodeShape_MatchesCPU() => await RunTest(async accelerator =>
        // Single-token decode: 1 kv head (gemma4 global layer), query at cache position 11.
        await FusedAttnOpOracle(accelerator, nHeads: 4, kvHeads: 1, seqQ: 1, seqKV: 12,
            headDim: 32, causal: true, window: 0, scale: 0f, kvOffset: 11));

    private static async Task FusedAttnOpOracle(Accelerator accelerator,
        int nHeads, int kvHeads, int seqQ, int seqKV, int headDim,
        bool causal, int window, float scale, int kvOffset)
    {
        var rng = new Random(71 + nHeads + window);
        var q = new float[nHeads * seqQ * headDim];
        var k = new float[kvHeads * seqKV * headDim];
        var v = new float[kvHeads * seqKV * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < k.Length; i++) k[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);

        float effScale = scale > 0f ? scale : 1f / MathF.Sqrt(headDim);
        long effWindow = window <= 0 ? long.MaxValue : window;
        int group = nHeads / kvHeads;

        var expected = new float[nHeads * seqQ * headDim];
        for (int h = 0; h < nHeads; h++)
        {
            int kvh = h / group;
            for (int sq = 0; sq < seqQ; sq++)
            {
                long qPos = kvOffset + sq;
                var scores = new double[seqKV];
                var valid = new bool[seqKV];
                double max = double.NegativeInfinity;
                for (int kv = 0; kv < seqKV; kv++)
                {
                    valid[kv] = (!causal || kv <= qPos) && kv > qPos - effWindow;
                    if (!valid[kv]) continue;
                    double dot = 0;
                    for (int dd = 0; dd < headDim; dd++)
                        dot += q[(h * seqQ + sq) * headDim + dd] * k[(kvh * seqKV + kv) * headDim + dd];
                    scores[kv] = dot * effScale;
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
                            acc += Math.Exp(scores[kv] - max) / sum * v[(kvh * seqKV + kv) * headDim + dd];
                    expected[(h * seqQ + sq) * headDim + dd] = (float)acc;
                }
            }
        }

        using var qBuf = accelerator.Allocate1D(q);
        using var kBuf = accelerator.Allocate1D(k);
        using var vBuf = accelerator.Allocate1D(v);
        using var outBuf = accelerator.Allocate1D<float>(expected.Length);
        using var registry = new OperatorRegistry(accelerator);
        var op = registry.Resolve("FusedAttention");
        var pool = new BufferPool(accelerator);
        op.Execute(new OnnxOpContext
        {
            Inputs = new[]
            {
                new Tensor(qBuf.View, new[] { nHeads, seqQ, headDim }),
                new Tensor(kBuf.View, new[] { kvHeads, seqKV, headDim }),
                new Tensor(vBuf.View, new[] { kvHeads, seqKV, headDim }),
            },
            Outputs = new[] { new Tensor(outBuf.View, new[] { nHeads, seqQ, headDim }) },
            Attributes = new Dictionary<string, object>
            {
                ["n_heads"] = nHeads,
                ["n_kv_heads"] = kvHeads,
                ["head_dim"] = headDim,
                ["causal"] = causal ? 1 : 0,
                ["window"] = window,
                ["scale"] = scale,
                ["kv_offset"] = kvOffset,
            },
            Pool = pool,
            InputNames = new[] { "q", "k", "v" },
        });
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, expected.Length);

        AssertCloseQuant(got, expected, 2e-3f,
            $"FusedAttention op GQA {nHeads}/{kvHeads} window={window} scale={scale}");
        Console.WriteLine($"[AttnOp] FusedAttention nH={nHeads} kvH={kvHeads} window={window} " +
            $"scale={(scale > 0 ? scale.ToString("F4") : "default")} kvOffset={kvOffset}: matches CPU oracle");
    }
}
