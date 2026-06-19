using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Mixture-of-Experts (gpt-oss / OpenAI-MoE) operator tests. The reference is a self-contained CPU port of
/// llama.cpp build_moe_ffn for the gpt-oss config (SOFTMAX_WEIGHT gating, top-k, per-expert gate/up/down +
/// biases, SwiGLU-OAI alpha=1.702/limit=7), matched against MoEOperator on the GPU (fp32 experts isolate the
/// routing + activation logic from quantized weight decode, which is covered by the FusedDequant MXFP4 tests).
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// Attention sinks (gpt-oss): a per-head learned logit joins the softmax denominator (its value is 0).
    /// FusedAttentionKernel with sinks vs a CPU softmax-with-sink reference. Also implicitly proves the
    /// no-sinks path is unchanged (sinkCount=0 leaves the kernel identical).
    /// </summary>
    [TestMethod]
    public Task Attention_Sinks_MatchesCpuReference() => RunTest(async accelerator =>
    {
        int nHeads = 2, seqQ = 2, seqKV = 3, D = 4;
        float scale = 1f / MathF.Sqrt(D);
        var rng = new Random(83);
        float R() => (float)(rng.NextDouble() * 2 - 1);
        var Q = new float[nHeads * seqQ * D];
        var K = new float[nHeads * seqKV * D];
        var V = new float[nHeads * seqKV * D];
        var sinks = new float[nHeads];
        foreach (var a in new[] { Q, K, V, sinks }) for (int i = 0; i < a.Length; i++) a[i] = R();

        // CPU reference: bidirectional (no mask), sink in the softmax max + denominator, 0 value contribution.
        var expected = new float[nHeads * seqQ * D];
        for (int h = 0; h < nHeads; h++)
            for (int q = 0; q < seqQ; q++)
            {
                var sc = new float[seqKV];
                float m = sinks[h];
                for (int kv = 0; kv < seqKV; kv++)
                {
                    float dot = 0f;
                    for (int dd = 0; dd < D; dd++) dot += Q[(h * seqQ + q) * D + dd] * K[(h * seqKV + kv) * D + dd];
                    sc[kv] = dot * scale; m = MathF.Max(m, sc[kv]);
                }
                float denom = MathF.Exp(sinks[h] - m);
                for (int kv = 0; kv < seqKV; kv++) denom += MathF.Exp(sc[kv] - m);
                for (int dd = 0; dd < D; dd++)
                {
                    float acc = 0f;
                    for (int kv = 0; kv < seqKV; kv++) acc += MathF.Exp(sc[kv] - m) * V[(h * seqKV + kv) * D + dd];
                    expected[(h * seqQ + q) * D + dd] = acc / denom;
                }
            }

        using var qb = accelerator.Allocate1D(Q);
        using var kb = accelerator.Allocate1D(K);
        using var vb = accelerator.Allocate1D(V);
        using var sb = accelerator.Allocate1D(sinks);
        using var ob = accelerator.Allocate1D<float>(nHeads * seqQ * D);
        using var attn = new Kernels.FusedAttentionKernel(accelerator);
        int bigWindow = seqKV + seqQ + 8;
        attn.Forward(qb.View, kb.View, vb.View, ob.View, nHeads, nHeads, seqQ, seqKV, D,
            causal: false, window: bigWindow, kvOffset: 0, scale: scale, sinks: sb.View, sinkCount: nHeads);
        await accelerator.SynchronizeAsync();
        var gpu = await ob.CopyToHostAsync<float>(0, nHeads * seqQ * D);

        float maxErr = 0f;
        for (int i = 0; i < expected.Length; i++) maxErr = MathF.Max(maxErr, MathF.Abs(gpu[i] - expected[i]));
        if (maxErr > 1e-4f) throw new Exception($"Attention sinks vs CPU maxErr={maxErr:E3} (expected < 1e-4)");

        // Also exercise the OPERATOR path: a 4th input (sinks) must route to the kernel's sink fold.
        var registry = new OperatorRegistry(accelerator);
        using var ob2 = accelerator.Allocate1D<float>(nHeads * seqQ * D);
        var pool = new BufferPool(accelerator);
        try
        {
            var ctx = new OnnxOpContext
            {
                Inputs = new[]
                {
                    new Tensor(qb.View, new[] { nHeads, seqQ, D }, "q"),
                    new Tensor(kb.View, new[] { nHeads, seqKV, D }, "k"),
                    new Tensor(vb.View, new[] { nHeads, seqKV, D }, "v"),
                    new Tensor(sb.View, new[] { nHeads }, "sinks"),
                },
                Outputs = new[] { new Tensor(ob2.View, new[] { nHeads, seqQ, D }, "out") },
                Attributes = new Dictionary<string, object>
                {
                    ["n_heads"] = (long)nHeads, ["n_kv_heads"] = (long)nHeads, ["head_dim"] = (long)D,
                    ["causal"] = 0L, ["window"] = (long)bigWindow, ["scale"] = scale,
                },
                Pool = pool,
                InputNames = new[] { "q", "k", "v", "sinks" },
                Registry = registry,
            };
            new FusedAttentionOperator(registry).Execute(ctx);
            await accelerator.SynchronizeAsync();
            var gpuOp = await ob2.CopyToHostAsync<float>(0, nHeads * seqQ * D);
            float opErr = 0f;
            for (int i = 0; i < expected.Length; i++) opErr = MathF.Max(opErr, MathF.Abs(gpuOp[i] - expected[i]));
            if (opErr > 1e-4f) throw new Exception($"FusedAttentionOperator sinks (4th input) vs CPU maxErr={opErr:E3}");
        }
        finally { pool.Dispose(); }
    });

    [TestMethod]
    public Task MoE_GptOss_MatchesCpuReference() => RunTest(async accelerator =>
    {
        int nEmbd = 8, nFf = 16, nExpert = 4, topK = 2, T = 3;
        const float alpha = 1.702f, limit = 7.0f;
        var rng = new Random(71);
        float R() => (float)(rng.NextDouble() * 2 - 1);

        var x = new float[T * nEmbd];
        var gateInp = new float[nExpert * nEmbd];          // [n_expert][n_embd]
        var gateInpB = new float[nExpert];
        var gateExps = new float[nExpert * nFf * nEmbd];   // [e][n_ff][n_embd]
        var gateExpsB = new float[nFf * nExpert];          // [n_ff][n_expert] -> e slice at e*nFf
        var upExps = new float[nExpert * nFf * nEmbd];
        var upExpsB = new float[nFf * nExpert];
        var downExps = new float[nExpert * nEmbd * nFf];   // [e][n_embd][n_ff]
        var downExpsB = new float[nEmbd * nExpert];        // [n_embd][n_expert] -> e slice at e*nEmbd
        foreach (var a in new[] { x, gateInp, gateInpB, gateExps, gateExpsB, upExps, upExpsB, downExps, downExpsB })
            for (int i = 0; i < a.Length; i++) a[i] = R();

        // ── CPU reference ──
        static float SwiGluOai(float g, float u, float al, float lim)
        {
            float xg = MathF.Min(g, lim);
            float yu = MathF.Max(-lim, MathF.Min(u, lim));
            return (xg / (1f + MathF.Exp(al * (-xg)))) * (yu + 1f);
        }
        var expected = new float[T * nEmbd];
        for (int t = 0; t < T; t++)
        {
            var logit = new float[nExpert];
            for (int e = 0; e < nExpert; e++)
            {
                float s = gateInpB[e];
                for (int k = 0; k < nEmbd; k++) s += x[t * nEmbd + k] * gateInp[e * nEmbd + k];
                logit[e] = s;
            }
            // top-k by logit
            var used = new bool[nExpert]; var sel = new int[topK]; var w = new float[topK];
            for (int i = 0; i < topK; i++)
            {
                int best = -1; float bv = float.NegativeInfinity;
                for (int e = 0; e < nExpert; e++) if (!used[e] && logit[e] > bv) { bv = logit[e]; best = e; }
                used[best] = true; sel[i] = best; w[i] = bv;
            }
            float m = float.NegativeInfinity; for (int i = 0; i < topK; i++) m = MathF.Max(m, w[i]);
            float sum = 0f; for (int i = 0; i < topK; i++) { w[i] = MathF.Exp(w[i] - m); sum += w[i]; }
            for (int i = 0; i < topK; i++) w[i] /= sum;

            for (int i = 0; i < topK; i++)
            {
                int e = sel[i];
                var act = new float[nFf];
                for (int ff = 0; ff < nFf; ff++)
                {
                    // expert e's gate/up bias slice is contiguous at e*nFf (matches MoEOperator.AddExpertBias).
                    float gg = gateExpsB[e * nFf + ff];
                    float uu = upExpsB[e * nFf + ff];
                    for (int k = 0; k < nEmbd; k++)
                    {
                        gg += x[t * nEmbd + k] * gateExps[(e * nFf + ff) * nEmbd + k];
                        uu += x[t * nEmbd + k] * upExps[(e * nFf + ff) * nEmbd + k];
                    }
                    act[ff] = SwiGluOai(gg, uu, alpha, limit);
                }
                for (int mm = 0; mm < nEmbd; mm++)
                {
                    float dn = downExpsB[e * nEmbd + mm];
                    for (int k = 0; k < nFf; k++) dn += act[k] * downExps[(e * nEmbd + mm) * nFf + k];
                    expected[t * nEmbd + mm] += w[i] * dn;
                }
            }
        }

        // ── GPU via MoEOperator ──
        var registry = new OperatorRegistry(accelerator);
        using var xb = accelerator.Allocate1D(x);
        using var giB = accelerator.Allocate1D(gateInp);
        using var giBb = accelerator.Allocate1D(gateInpB);
        using var geB = accelerator.Allocate1D(gateExps);
        using var geBb = accelerator.Allocate1D(gateExpsB);
        using var ueB = accelerator.Allocate1D(upExps);
        using var ueBb = accelerator.Allocate1D(upExpsB);
        using var deB = accelerator.Allocate1D(downExps);
        using var deBb = accelerator.Allocate1D(downExpsB);
        using var outb = accelerator.Allocate1D<float>(T * nEmbd);
        var pool = new BufferPool(accelerator);
        try
        {
            var inputs = new[]
            {
                new Tensor(xb.View, new[] { T, nEmbd }, "x"),
                new Tensor(giB.View, new[] { nExpert, nEmbd }, "gate_inp"),
                new Tensor(giBb.View, new[] { nExpert }, "gate_inp_b"),
                new Tensor(geB.View, new[] { nExpert, nFf, nEmbd }, "gate_exps"),
                new Tensor(geBb.View, new[] { nFf, nExpert }, "gate_exps_b"),
                new Tensor(ueB.View, new[] { nExpert, nFf, nEmbd }, "up_exps"),
                new Tensor(ueBb.View, new[] { nFf, nExpert }, "up_exps_b"),
                new Tensor(deB.View, new[] { nExpert, nEmbd, nFf }, "down_exps"),
                new Tensor(deBb.View, new[] { nEmbd, nExpert }, "down_exps_b"),
            };
            var ctx = new OnnxOpContext
            {
                Inputs = inputs,
                Outputs = new[] { new Tensor(outb.View, new[] { T, nEmbd }, "out") },
                Attributes = new Dictionary<string, object>
                {
                    ["n_expert"] = (long)nExpert, ["n_expert_used"] = (long)topK, ["n_ff"] = (long)nFf,
                    ["alpha"] = alpha, ["limit"] = limit,
                },
                Pool = pool,
                InputNames = inputs.Select(i => i.Name!).ToArray(),
                Registry = registry,
            };
            // ExecuteAsync = the production (GraphExecutor.RunAsync) path: browser-safe router readback via
            // CopyToHostAsync (sync Execute's GetAsArray1D throws on WebGPU/WebGL/Wasm).
            await new MoEOperator(registry).ExecuteAsync(ctx);
            await accelerator.SynchronizeAsync();
            var gpu = await outb.CopyToHostAsync<float>(0, T * nEmbd);

            float maxErr = 0f;
            for (int i = 0; i < expected.Length; i++) maxErr = MathF.Max(maxErr, MathF.Abs(gpu[i] - expected[i]));
            if (maxErr > 2e-4f)
                throw new Exception($"MoE gpt-oss vs CPU reference maxErr={maxErr:E3} (expected < 2e-4)");
        }
        finally { pool.Dispose(); }
    });
}
