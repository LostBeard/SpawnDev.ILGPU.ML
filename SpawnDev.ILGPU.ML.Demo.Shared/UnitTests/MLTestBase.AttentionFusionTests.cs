using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Verifies the GraphOptimizer self-attention fusion pass (Plans/...): the standard diffusers/torch decomposed
/// self-attention subgraph (K→Transpose, Q·Kᵀ MatMul → Mul(scale) → Add(zero-bias) → Softmax → Cast → probs·V
/// MatMul) is rewritten into ONE flash-style FusedAttention node — and the FUSED graph produces the same result
/// as a CPU reference attention on every backend. This is the SD-UNet memory win (the [B·H,S,S] scores are
/// never materialized). Mirrors the exact SD-Turbo down_block.0/attn1 pattern (mask is identically zero).
/// </summary>
public abstract partial class MLTestBase
{
    private static GraphNode N(string op, string[] ins, string[] outs, Dictionary<string, JsonElement>? attrs = null)
        => new() { OpType = op, Inputs = ins.ToList(), Outputs = outs.ToList(), Attributes = attrs };

    // CPU reference: out[h,i,:] = sum_j softmax_j(scale * Q[h,i,·]·K[h,j,·]) * V[h,j,:]
    private static float[] AttentionCpu(float[] q, float[] k, float[] v, int heads, int seq, int hd, float scale)
    {
        var o = new float[heads * seq * hd];
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < seq; i++)
            {
                var sc = new double[seq];
                double mx = double.NegativeInfinity;
                for (int j = 0; j < seq; j++)
                {
                    double dot = 0;
                    for (int d = 0; d < hd; d++) dot += q[(h * seq + i) * hd + d] * (double)k[(h * seq + j) * hd + d];
                    sc[j] = dot * scale; if (sc[j] > mx) mx = sc[j];
                }
                double sum = 0;
                for (int j = 0; j < seq; j++) { sc[j] = Math.Exp(sc[j] - mx); sum += sc[j]; }
                for (int d = 0; d < hd; d++)
                {
                    double acc = 0;
                    for (int j = 0; j < seq; j++) acc += sc[j] / sum * v[(h * seq + j) * hd + d];
                    o[(h * seq + i) * hd + d] = (float)acc;
                }
            }
        return o;
    }

    [TestMethod]
    public async Task AttentionFusion_FusedMatchesCpu_AllBackends() => await RunTest(async accelerator =>
    {
        const int heads = 2, seq = 6, hd = 4;
        const float scale = 0.5f;
        var je0 = JsonSerializer.SerializeToElement(0);
        var perm = JsonSerializer.SerializeToElement(new[] { 0, 2, 1 });   // transpose last two dims
        var axisNeg1 = JsonSerializer.SerializeToElement(-1);

        // Decomposed self-attention matching SD-Turbo down_block.0/attn1 (scale Mul + zero-bias Add + Softmax).
        var graph = new ModelGraph
        {
            Name = "attn_fuse_test",
            Inputs = new()
            {
                new() { Name = "Q", Shape = new[] { heads, seq, hd } },
                new() { Name = "K", Shape = new[] { heads, seq, hd } },
                new() { Name = "V", Shape = new[] { heads, seq, hd } },
            },
            Outputs = new() { new() { Name = "attn_out", Shape = new[] { heads, seq, hd } } },
            Initializers = new() { ["scale_c"] = new[] { 1 }, ["zero_bias"] = new[] { 1 } },
            Nodes = new()
            {
                N("Transpose", new[] { "K" }, new[] { "Kt" }, new() { ["perm"] = perm }),
                N("MatMul", new[] { "Q", "Kt" }, new[] { "scores" }),
                N("Mul", new[] { "scores", "scale_c" }, new[] { "scaled" }),
                N("Add", new[] { "scaled", "zero_bias" }, new[] { "biased" }),
                N("Softmax", new[] { "biased" }, new[] { "probs" }, new() { ["axis"] = axisNeg1 }),
                N("Cast", new[] { "probs" }, new[] { "probs_c" }, new() { ["to"] = JsonSerializer.SerializeToElement(1) }),
                N("MatMul", new[] { "probs_c", "V" }, new[] { "attn_out" }),
            },
        };
        // The scale constant must be resolvable at fusion time (FloatConstantData), like a real Constant-as-init.
        graph.FloatConstantData = new() { ["scale_c"] = new[] { scale }, ["zero_bias"] = new[] { 0f } };
        graph.ConstantData = new() { ["scale_c"] = new[] { 0 }, ["zero_bias"] = new[] { 0 } };

        // ── Rewrite assertions: the pass must fire and collapse the core into one FusedAttention node. ──
        var optimized = GraphOptimizer.Optimize(graph);
        int fused = optimized.Nodes.Count(n => n.OpType == "FusedAttention");
        if (fused != 1) throw new Exception($"expected 1 FusedAttention node, got {fused} on {BackendName}");
        if (optimized.Nodes.Any(n => n.OpType is "Softmax" or "Transpose"))
            throw new Exception($"fusion left Softmax/Transpose behind on {BackendName}");
        var fa = optimized.Nodes.First(n => n.OpType == "FusedAttention");
        if (fa.Inputs[0] != "Q" || fa.Inputs[1] != "K" || fa.Inputs[2] != "V" || fa.Outputs[0] != "attn_out")
            throw new Exception($"FusedAttention wiring wrong: in=[{string.Join(",", fa.Inputs)}] out={fa.Outputs[0]} on {BackendName}");
        float faScale = fa.Attributes!["scale"].GetSingle();
        if (MathF.Abs(faScale - scale) > 1e-6f) throw new Exception($"fused scale {faScale} != {scale} on {BackendName}");

        // ── Execution equivalence: run the FUSED graph, compare to the CPU reference. ──
        var rng = new Random(909);
        var q = RandFloats(heads * seq * hd, rng);
        var k = RandFloats(heads * seq * hd, rng);
        var v = RandFloats(heads * seq * hd, rng);
        var expected = AttentionCpu(q, k, v, heads, seq, hd, scale);

        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);   // Compile re-runs Optimize internally
        using var pool = new BufferPool(accelerator);
        using var ex = new GraphExecutor(accelerator, compiled, new Dictionary<string, Tensor>());
        using var qB = accelerator.Allocate1D(q);
        using var kB = accelerator.Allocate1D(k);
        using var vB = accelerator.Allocate1D(v);
        int n = heads * seq * hd;
        using var host = accelerator.Allocate1D<float>(n);
        var outs = await ex.RunAsync(new Dictionary<string, Tensor>
        {
            ["Q"] = new Tensor(qB.View, new[] { heads, seq, hd }),
            ["K"] = new Tensor(kB.View, new[] { heads, seq, hd }),
            ["V"] = new Tensor(vB.View, new[] { heads, seq, hd }),
        });
        await host.View.CopyFromAsync(outs["attn_out"].Data.SubView(0, n));
        await accelerator.SynchronizeAsync();
        var got = await host.CopyToHostAsync<float>(0, n);

        float worst = 0;
        for (int i = 0; i < n; i++) worst = MathF.Max(worst, MathF.Abs(got[i] - expected[i]));
        if (worst > 2e-3f)
            throw new Exception($"fused attention diverged from CPU ref (worst |Δ|={worst:E3}) on {BackendName}");
        Console.WriteLine($"[AttentionFusion] fused graph matches CPU ref (worst |Δ|={worst:E3}) on {BackendName}");
    });

    private static float[] RandFloats(int n, Random rng)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }
}
