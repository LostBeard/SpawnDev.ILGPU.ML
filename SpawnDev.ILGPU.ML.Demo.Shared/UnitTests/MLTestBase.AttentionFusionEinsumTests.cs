using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Verifies the EINSUM-form attention fusion pass (GraphOptimizer.FuseAttentionEinsum). DINOv2-lineage ONNX
/// exports (Depth Anything V3) emit attention as Einsum(Q·Kᵀ) → scale → Softmax → Einsum(probs·V) instead of
/// the MatMul form, so the original FuseAttention pass never fires on them and attention fell to
/// EinsumOperator's per-batch MatMul loop (the dominant per-op cost on DAv3-518, all backends). Covers both
/// K layouts (natural "P.id,P.jd->P.ij" and pre-transposed "P.id,P.dj->P.ij" with an explicit Transpose) and
/// both scale cases (explicit Mul node folded into the scale attr; NO scale node => attr must be 1.0, because
/// the unfused graph applied none - the kernel's 1/sqrt(hd) default would diverge). Numeric gate: the FUSED
/// graph must match the CPU reference attention on every backend.
/// </summary>
public abstract partial class MLTestBase
{
    private static JsonElement JEq(string equation) => JsonSerializer.SerializeToElement(equation);

    [TestMethod]
    public async Task AttentionFusionEinsum_NaturalK_FusedMatchesCpu_AllBackends() => await RunTest(async accelerator =>
    {
        // DAv3 shape class: rank-4 [1, heads, seq, hd], K fed un-transposed, explicit Mul scale, Cast after Softmax.
        const int heads = 2, seq = 6, hd = 4;
        const float scale = 0.5f;
        var graph = new ModelGraph
        {
            Name = "attn_fuse_einsum_natural",
            Inputs = new()
            {
                new() { Name = "Q", Shape = new[] { 1, heads, seq, hd } },
                new() { Name = "K", Shape = new[] { 1, heads, seq, hd } },
                new() { Name = "V", Shape = new[] { 1, heads, seq, hd } },
            },
            Outputs = new() { new() { Name = "attn_out", Shape = new[] { 1, heads, seq, hd } } },
            Initializers = new() { ["scale_c"] = new[] { 1 } },
            Nodes = new()
            {
                N("Einsum", new[] { "Q", "K" }, new[] { "scores" }, new() { ["equation"] = JEq("bhid,bhjd->bhij") }),
                N("Mul", new[] { "scores", "scale_c" }, new[] { "scaled" }),
                N("Softmax", new[] { "scaled" }, new[] { "probs" }, new() { ["axis"] = JsonSerializer.SerializeToElement(-1) }),
                N("Cast", new[] { "probs" }, new[] { "probs_c" }, new() { ["to"] = JsonSerializer.SerializeToElement(1) }),
                N("Einsum", new[] { "probs_c", "V" }, new[] { "attn_out" }, new() { ["equation"] = JEq("bhij,bhjd->bhid") }),
            },
            FloatConstantData = new() { ["scale_c"] = new[] { scale } },
            ConstantData = new() { ["scale_c"] = new[] { 0 } },
        };

        var optimized = GraphOptimizer.Optimize(graph);
        int fused = optimized.Nodes.Count(n => n.OpType == "FusedAttention");
        if (fused != 1) throw new Exception($"expected 1 FusedAttention node, got {fused} on {BackendName}");
        if (optimized.Nodes.Any(n => n.OpType is "Softmax" or "Einsum"))
            throw new Exception($"fusion left Softmax/Einsum behind on {BackendName}");
        var fa = optimized.Nodes.First(n => n.OpType == "FusedAttention");
        if (fa.Inputs[0] != "Q" || fa.Inputs[1] != "K" || fa.Inputs[2] != "V" || fa.Outputs[0] != "attn_out")
            throw new Exception($"FusedAttention wiring wrong: in=[{string.Join(",", fa.Inputs)}] out={fa.Outputs[0]} on {BackendName}");
        float faScale = fa.Attributes!["scale"].GetSingle();
        if (MathF.Abs(faScale - scale) > 1e-6f) throw new Exception($"fused scale {faScale} != {scale} on {BackendName}");

        await AssertFusedEinsumGraphMatchesCpu(accelerator, graph, heads, seq, hd, scale, seed: 909);
        Console.WriteLine($"[AttentionFusionEinsum] natural-K fused graph matches CPU ref on {BackendName}");
    });

    [TestMethod]
    public async Task AttentionFusionEinsum_TransposedK_NoScale_FusedMatchesCpu_AllBackends() => await RunTest(async accelerator =>
    {
        // K pre-transposed via explicit Transpose[0,1,3,2] and NO scale node anywhere between Q·Kᵀ and Softmax:
        // the fused node must carry scale=1.0 (bit-consistent with the unfused graph), never the kernel default.
        const int heads = 2, seq = 5, hd = 4;
        var graph = new ModelGraph
        {
            Name = "attn_fuse_einsum_transposed_noscale",
            Inputs = new()
            {
                new() { Name = "Q", Shape = new[] { 1, heads, seq, hd } },
                new() { Name = "K", Shape = new[] { 1, heads, seq, hd } },
                new() { Name = "V", Shape = new[] { 1, heads, seq, hd } },
            },
            Outputs = new() { new() { Name = "attn_out", Shape = new[] { 1, heads, seq, hd } } },
            Initializers = new(),
            Nodes = new()
            {
                N("Transpose", new[] { "K" }, new[] { "Kt" }, new() { ["perm"] = JsonSerializer.SerializeToElement(new[] { 0, 1, 3, 2 }) }),
                N("Einsum", new[] { "Q", "Kt" }, new[] { "scores" }, new() { ["equation"] = JEq("bhid,bhdj->bhij") }),
                N("Softmax", new[] { "scores" }, new[] { "probs" }, new() { ["axis"] = JsonSerializer.SerializeToElement(-1) }),
                N("Einsum", new[] { "probs", "V" }, new[] { "attn_out" }, new() { ["equation"] = JEq("bhij,bhjd->bhid") }),
            },
        };

        var optimized = GraphOptimizer.Optimize(graph);
        int fused = optimized.Nodes.Count(n => n.OpType == "FusedAttention");
        if (fused != 1) throw new Exception($"expected 1 FusedAttention node, got {fused} on {BackendName}");
        if (optimized.Nodes.Any(n => n.OpType is "Softmax" or "Einsum" or "Transpose"))
            throw new Exception($"fusion left Softmax/Einsum/Transpose behind on {BackendName}");
        var fa = optimized.Nodes.First(n => n.OpType == "FusedAttention");
        if (fa.Inputs[0] != "Q" || fa.Inputs[1] != "K" || fa.Inputs[2] != "V")
            throw new Exception($"FusedAttention wiring wrong (K must be the UN-transposed tensor): in=[{string.Join(",", fa.Inputs)}] on {BackendName}");
        float faScale = fa.Attributes!["scale"].GetSingle();
        if (MathF.Abs(faScale - 1f) > 1e-6f)
            throw new Exception($"no-scale-node graph must fuse with scale=1.0, got {faScale} on {BackendName}");

        await AssertFusedEinsumGraphMatchesCpu(accelerator, graph, heads, seq, hd, scale: 1f, seed: 4242);
        Console.WriteLine($"[AttentionFusionEinsum] transposed-K no-scale fused graph matches CPU ref on {BackendName}");
    });

    /// <summary>Runs the graph through the real Compile→Execute path (Compile re-runs Optimize internally) and
    /// gates the fused output against the CPU reference attention. Rank-4 [1,H,S,D] is flat-identical to the
    /// reference's [H,S,D] layout.</summary>
    private async Task AssertFusedEinsumGraphMatchesCpu(
        Accelerator accelerator, ModelGraph graph, int heads, int seq, int hd, float scale, int seed)
    {
        var rng = new Random(seed);
        var q = RandFloats(heads * seq * hd, rng);
        var k = RandFloats(heads * seq * hd, rng);
        var v = RandFloats(heads * seq * hd, rng);
        var expected = AttentionCpu(q, k, v, heads, seq, hd, scale);

        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);
        using var pool = new BufferPool(accelerator);
        using var ex = new GraphExecutor(accelerator, compiled, new Dictionary<string, Tensor>());
        using var qB = accelerator.Allocate1D(q);
        using var kB = accelerator.Allocate1D(k);
        using var vB = accelerator.Allocate1D(v);
        int n = heads * seq * hd;
        using var host = accelerator.Allocate1D<float>(n);
        var outs = await ex.RunAsync(new Dictionary<string, Tensor>
        {
            ["Q"] = new Tensor(qB.View, new[] { 1, heads, seq, hd }),
            ["K"] = new Tensor(kB.View, new[] { 1, heads, seq, hd }),
            ["V"] = new Tensor(vB.View, new[] { 1, heads, seq, hd }),
        });
        await host.View.CopyFromAsync(outs["attn_out"].Data.SubView(0, n));
        await accelerator.SynchronizeAsync();
        var got = await host.CopyToHostAsync<float>(0, n);

        float worst = 0;
        for (int i = 0; i < n; i++) worst = MathF.Max(worst, MathF.Abs(got[i] - expected[i]));
        if (worst > 2e-3f)
            throw new Exception($"einsum-fused attention diverged from CPU ref (worst |Δ|={worst:E3}) on {BackendName}");
    }
}
