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

    /// <summary>
    /// DINOv2/DAv3 export form: Q and K are PRE-SCALED before the QK^T MatMul by the SAME runtime-computed
    /// scalar (s = sqrt(1/sqrt(head_dim)) via a Shape → Slice(-1,MAX) → Cast → Sqrt → Div → Sqrt chain), the
    /// K side as Mul(Transpose(K), s), and there is NO scale node between the MatMul and the Softmax. The
    /// fusion must (a) prove s is scalar (the Slice-of-Shape rule), (b) commute the K-side Mul with the
    /// Transpose (retarget the Mul to un-transposed K, drop the Transpose), and (c) emit scale=1.0 - the
    /// kernel's 1/sqrt(hd) default would double-scale. Numeric gate: fused graph vs CPU reference at the
    /// chain's true total scale s² = 1/sqrt(hd). This is the exact structure of all 12 DAv3-Small blocks.
    /// </summary>
    [TestMethod]
    public async Task AttentionFusion_PreScaledQK_DAv3Form_FusedMatchesCpu_AllBackends() => await RunTest(async accelerator =>
    {
        const int heads = 2, seq = 6, hd = 4;
        float expectedTotalScale = 1f / MathF.Sqrt(hd); // s² where s = sqrt(1/sqrt(hd))
        var axisNeg1 = JsonSerializer.SerializeToElement(-1);
        // The scale chain is RUNTIME-computed from Shape(K) like the real export (a fully const-derived
        // chain gets constant-folded and its intermediates then have no backing tensors), but uses
        // Gather(idx=3) instead of the export's Slice(-1,MAX): positive-index Gather-of-Shape is the
        // bog-standard runtime form, deterministic on every backend, and hits IsProvablyScalar's Gather
        // rule. s = sqrt(1/sqrt(shape[3]=4)) -> total s² = 0.5. The export's exact Slice(-1,MAX) chain is
        // covered structurally by AttentionFusion_PreScaledQK_SliceShapeScaleChain_FusesStatically below
        // and end-to-end by the DAv3 rig (its indices are int64-typed initializers there).
        var graph = new ModelGraph
        {
            Name = "attn_fuse_dav3_prescaled",
            Inputs = new()
            {
                new() { Name = "Q", Shape = new[] { 1, heads, seq, hd } },
                new() { Name = "K", Shape = new[] { 1, heads, seq, hd } },
                new() { Name = "V", Shape = new[] { 1, heads, seq, hd } },
            },
            Outputs = new() { new() { Name = "attn_out", Shape = new[] { 1, heads, seq, hd } } },
            Initializers = new() { ["c_idx3"] = new[] { 1 }, ["c_one"] = new[] { 1 } },
            Nodes = new()
            {
                // s = sqrt(1 / sqrt(head_dim)) - computed at RUNTIME from Shape(K), like the real export.
                N("Shape", new[] { "K" }, new[] { "kshape" }),
                N("Gather", new[] { "kshape", "c_idx3" }, new[] { "hd_i" }, new() { ["axis"] = JsonSerializer.SerializeToElement(0) }),
                N("Cast", new[] { "hd_i" }, new[] { "hd_f" }, new() { ["to"] = JsonSerializer.SerializeToElement(1) }),
                N("Sqrt", new[] { "hd_f" }, new[] { "sqrt_hd" }),
                N("Div", new[] { "c_one", "sqrt_hd" }, new[] { "inv_sqrt_hd" }),
                N("Sqrt", new[] { "inv_sqrt_hd" }, new[] { "s" }),
                // Pre-scaled Q and K; K transposed BEFORE its scale Mul (the guard-killing order).
                N("Mul", new[] { "Q", "s" }, new[] { "qs" }),
                N("Transpose", new[] { "K" }, new[] { "kt" }, new() { ["perm"] = JsonSerializer.SerializeToElement(new[] { 0, 1, 3, 2 }) }),
                N("Mul", new[] { "kt", "s" }, new[] { "kts" }),
                N("MatMul", new[] { "qs", "kts" }, new[] { "scores" }),
                N("Softmax", new[] { "scores" }, new[] { "probs" }, new() { ["axis"] = axisNeg1 }),
                N("MatMul", new[] { "probs", "V" }, new[] { "attn_out" }),
            },
            ConstantData = new() { ["c_idx3"] = new[] { 3 }, ["c_one"] = new[] { 1 } },
            FloatConstantData = new() { ["c_idx3"] = new[] { 3f }, ["c_one"] = new[] { 1f } },
        };

        var optimized = GraphOptimizer.Optimize(graph);
        int fused = optimized.Nodes.Count(n => n.OpType == "FusedAttention");
        if (fused != 1) throw new Exception($"expected 1 FusedAttention node, got {fused} on {BackendName}");
        if (optimized.Nodes.Any(n => n.OpType is "Softmax" or "Transpose"))
            throw new Exception($"fusion left Softmax/Transpose behind on {BackendName}");
        var fa = optimized.Nodes.First(n => n.OpType == "FusedAttention");
        if (fa.Inputs[0] != "qs" || fa.Inputs[1] != "kts" || fa.Inputs[2] != "V")
            throw new Exception($"FusedAttention wiring wrong: in=[{string.Join(",", fa.Inputs)}] on {BackendName}");
        float faScale = fa.Attributes!["scale"].GetSingle();
        if (MathF.Abs(faScale - 1f) > 1e-6f)
            throw new Exception($"pre-scaled form must fuse with scale=1.0 (graph scaling lives in the Muls), got {faScale} on {BackendName}");
        // The K-side Mul must be RETARGETED to the un-transposed K (the commute rewrite).
        var kMul = optimized.Nodes.First(n => n.OpType == "Mul" && n.Outputs[0] == "kts");
        if (kMul.Inputs[0] != "K")
            throw new Exception($"K-side Mul not retargeted to un-transposed K: in=[{string.Join(",", kMul.Inputs)}] on {BackendName}");

        // Numeric gate through the real Compile→Execute path (the runtime s-chain executes too).
        var rng = new Random(31337);
        var q = RandFloats(heads * seq * hd, rng);
        var k = RandFloats(heads * seq * hd, rng);
        var v = RandFloats(heads * seq * hd, rng);
        var expected = AttentionCpu(q, k, v, heads, seq, hd, expectedTotalScale);

        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);
        using var pool = new BufferPool(accelerator);
        // The runtime s-chain (Shape -> Gather -> ... -> Sqrt) STAYS in the fused graph and executes, so its
        // small constant initializers must exist as real tensors (production uploads every initializer).
        using var cIdx3 = accelerator.Allocate1D(new[] { 3f });
        using var cOne = accelerator.Allocate1D(new[] { 1f });
        var weights = new Dictionary<string, Tensor>
        {
            ["c_idx3"] = new Tensor(cIdx3.View, new[] { 1 }),
            ["c_one"] = new Tensor(cOne.View, new[] { 1 }),
        };
        using var ex = new GraphExecutor(accelerator, compiled, weights);
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
            throw new Exception($"DAv3-form fused attention diverged from CPU ref (worst |Δ|={worst:E3}) on {BackendName}");
        Console.WriteLine($"[AttentionFusion] DAv3 pre-scaled form fused graph matches CPU ref (worst |Δ|={worst:E3}) on {BackendName}");
    });

    /// <summary>
    /// OPTIMIZER-ONLY gate for the EXACT DAv3 scale-chain structure: s comes from
    /// Shape(K) → Slice(starts=-1, ends=INT64MAX, step=1) → Cast → Sqrt → Div → Sqrt (the `IsProvablyScalar`
    /// Slice-of-Shape rule with int64-MAX sentinel ends). Asserts the fusion FIRES and rewires correctly.
    /// No execution here: the end-to-end runtime semantics of this chain are gated on the real DAv3 rig,
    /// where the indices are int64-typed initializers (a float-wired synthetic Slice executes differently).
    /// Verified offline against the real DAv3-Small graph: 12/12 blocks fuse with this rule.
    /// </summary>
    [TestMethod]
    public async Task AttentionFusion_PreScaledQK_SliceShapeScaleChain_FusesStatically() => await RunTest(accelerator =>
    {
        const int heads = 2, seq = 6, hd = 4;
        var graph = new ModelGraph
        {
            Name = "attn_fuse_dav3_slice_chain",
            Inputs = new()
            {
                new() { Name = "Q", Shape = new[] { 1, heads, seq, hd } },
                new() { Name = "K", Shape = new[] { 1, heads, seq, hd } },
                new() { Name = "V", Shape = new[] { 1, heads, seq, hd } },
            },
            Outputs = new() { new() { Name = "attn_out", Shape = new[] { 1, heads, seq, hd } } },
            Initializers = new()
            {
                ["c_m1"] = new[] { 1 }, ["c_max"] = new[] { 1 }, ["c_ax0"] = new[] { 1 },
                ["c_st1"] = new[] { 1 }, ["c_one"] = new[] { 1 },
            },
            Nodes = new()
            {
                N("Shape", new[] { "K" }, new[] { "kshape" }),
                N("Slice", new[] { "kshape", "c_m1", "c_max", "c_ax0", "c_st1" }, new[] { "hd_last" }),
                N("Cast", new[] { "hd_last" }, new[] { "hd_f" }, new() { ["to"] = JsonSerializer.SerializeToElement(1) }),
                N("Sqrt", new[] { "hd_f" }, new[] { "sqrt_hd" }),
                N("Div", new[] { "c_one", "sqrt_hd" }, new[] { "inv_sqrt_hd" }),
                N("Sqrt", new[] { "inv_sqrt_hd" }, new[] { "s" }),
                N("Mul", new[] { "Q", "s" }, new[] { "qs" }),
                N("Transpose", new[] { "K" }, new[] { "kt" }, new() { ["perm"] = JsonSerializer.SerializeToElement(new[] { 0, 1, 3, 2 }) }),
                N("Mul", new[] { "kt", "s" }, new[] { "kts" }),
                N("MatMul", new[] { "qs", "kts" }, new[] { "scores" }),
                N("Softmax", new[] { "scores" }, new[] { "probs" }, new() { ["axis"] = JsonSerializer.SerializeToElement(-1) }),
                N("MatMul", new[] { "probs", "V" }, new[] { "attn_out" }),
            },
            ConstantData = new()
            {
                ["c_m1"] = new[] { -1 }, ["c_max"] = new[] { int.MaxValue }, ["c_ax0"] = new[] { 0 },
                ["c_st1"] = new[] { 1 }, ["c_one"] = new[] { 1 },
            },
            FloatConstantData = new()
            {
                ["c_m1"] = new[] { -1f }, ["c_max"] = new[] { 9.2e18f }, ["c_ax0"] = new[] { 0f },
                ["c_st1"] = new[] { 1f }, ["c_one"] = new[] { 1f },
            },
        };

        var optimized = GraphOptimizer.Optimize(graph);
        int fused = optimized.Nodes.Count(n => n.OpType == "FusedAttention");
        if (fused != 1) throw new Exception($"expected 1 FusedAttention node, got {fused} on {BackendName}");
        if (optimized.Nodes.Any(n => n.OpType is "Softmax" or "Transpose"))
            throw new Exception($"fusion left Softmax/Transpose behind on {BackendName}");
        var fa = optimized.Nodes.First(n => n.OpType == "FusedAttention");
        if (fa.Inputs[0] != "qs" || fa.Inputs[1] != "kts" || fa.Inputs[2] != "V")
            throw new Exception($"FusedAttention wiring wrong: in=[{string.Join(",", fa.Inputs)}] on {BackendName}");
        if (MathF.Abs(fa.Attributes!["scale"].GetSingle() - 1f) > 1e-6f)
            throw new Exception($"pre-scaled form must fuse with scale=1.0 on {BackendName}");
        var kMul = optimized.Nodes.First(n => n.OpType == "Mul" && n.Outputs[0] == "kts");
        if (kMul.Inputs[0] != "K")
            throw new Exception($"K-side Mul not retargeted to un-transposed K: in=[{string.Join(",", kMul.Inputs)}] on {BackendName}");
        Console.WriteLine($"[AttentionFusion] Slice-of-Shape scale chain fuses statically (scale=1.0, K retargeted) on {BackendName}");
        return Task.CompletedTask;
    });
}
