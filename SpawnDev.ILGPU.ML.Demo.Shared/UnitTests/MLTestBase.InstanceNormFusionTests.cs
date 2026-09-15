using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// <c>GraphOptimizer.FuseInstanceNorm</c>: the nine-node ADAPTIVE normalisation chain collapsed into one
/// <c>FusedInstanceNorm</c> node, and the conservative rule that keeps it off a LayerNorm.
/// </summary>
/// <remarks>
/// <para>
/// ⭐ WHY THE PASS EXISTS. MEASURED on Kokoro-82M: 65 of these chains, 585 nodes, 24% of a 2,463-node
/// graph, none of which the LayerNorm pass can claim - its square is <c>Mul(d, d)</c> not <c>Pow(d, 2)</c>,
/// its axes arrive as an INPUT, and its scale and bias are computed at runtime from a style vector. Fusing
/// 62 of them took the graph to 1,885 nodes and WebGPU from 2,088 ms to 1,837 ms for the same utterance,
/// WebGL 7,521 -> 6,013 ms, CPU 312 -> 247 s. The answer got very slightly BETTER (0.9917 -> 0.9930
/// against onnxruntime), because one kernel rounds less than nine nodes.
/// </para>
/// <para>
/// 🔴 THE SECOND TEST IS THE IMPORTANT ONE. The optimizer runs BEFORE shape inference, so nothing in the
/// pass can see whether scale is shaped like the normalised axis (LayerNorm) or the channel axis
/// (InstanceNorm) - and the two are the SAME NINE NODES. The pass therefore claims a chain only when the
/// parameters are not initializers. Remove that rule and the first test still passes; only
/// <see cref="InstanceNormFusion_LeavesAConstantParameterChainAlone"/> notices.
/// </para>
/// </remarks>
public abstract partial class MLTestBase
{
    private const int InC = 4, InL = 7;
    private const float InEps = 1e-5f;

    /// <summary>Per-channel instance norm, written the long way round from the definition.</summary>
    private static float[] CpuInstanceNormAffine(float[] x, float[] scale, float[] bias,
        int channels, int spatial, float eps)
    {
        var y = new float[x.Length];
        for (var c = 0; c < channels; c++)
        {
            double mean = 0;
            for (var t = 0; t < spatial; t++) mean += x[c * spatial + t];
            mean /= spatial;
            double varsum = 0;
            for (var t = 0; t < spatial; t++)
            {
                var d = x[c * spatial + t] - mean;
                varsum += d * d;
            }
            var std = Math.Sqrt(varsum / spatial + eps);
            for (var t = 0; t < spatial; t++)
                y[c * spatial + t] = (float)(scale[c] * (x[c * spatial + t] - mean) / std + bias[c]);
        }
        return y;
    }

    /// <summary>The chain an adaptive-norm exporter emits, with the parameters as runtime tensors.</summary>
    private static ModelGraph AdaptiveNormGraph(bool parametersAreInitializers)
    {
        var axes = JsonSerializer.SerializeToElement(new[] { 2 });
        var keep = JsonSerializer.SerializeToElement(1);

        var inputs = new List<GraphValueInfo> { new() { Name = "X", Shape = new[] { 1, InC, InL } } };
        var initializers = new Dictionary<string, int[]> { ["eps_c"] = new[] { 1 } };
        if (parametersAreInitializers)
        {
            initializers["S"] = new[] { 1, InC, 1 };
            initializers["B"] = new[] { 1, InC, 1 };
        }
        else
        {
            inputs.Add(new() { Name = "S", Shape = new[] { 1, InC, 1 } });
            inputs.Add(new() { Name = "B", Shape = new[] { 1, InC, 1 } });
        }

        var graph = new ModelGraph
        {
            Name = "adain_fuse_test",
            Inputs = inputs,
            Outputs = new() { new() { Name = "Y", Shape = new[] { 1, InC, InL } } },
            Initializers = initializers,
            Nodes = new()
            {
                // ⚠️ `axes` as an ATTRIBUTE here. The input form is covered by the real model; what this
                // fixture is for is the rest of the shape, and an attribute keeps it readable.
                N("ReduceMean", new[] { "X" }, new[] { "mean" }, new() { ["axes"] = axes, ["keepdims"] = keep }),
                N("Sub", new[] { "X", "mean" }, new[] { "d" }),
                N("Mul", new[] { "d", "d" }, new[] { "d2" }),
                N("ReduceMean", new[] { "d2" }, new[] { "var" }, new() { ["axes"] = axes, ["keepdims"] = keep }),
                N("Add", new[] { "var", "eps_c" }, new[] { "ve" }),
                N("Sqrt", new[] { "ve" }, new[] { "std" }),
                N("Div", new[] { "d", "std" }, new[] { "n" }),
                // Scale FIRST, which is the order the real export uses - the pass must accept either.
                N("Mul", new[] { "S", "n" }, new[] { "y0" }),
                N("Add", new[] { "y0", "B" }, new[] { "Y" }),
            },
        };
        graph.FloatConstantData = new() { ["eps_c"] = new[] { InEps } };
        graph.ConstantData = new() { ["eps_c"] = new[] { 0 } };
        return graph;
    }

    /// <summary>The pass fires, collapses the chain, and computes the same thing.</summary>
    [TestMethod]
    public async Task InstanceNormFusion_CollapsesChainAndMatchesCpu() => await RunTest(async accelerator =>
    {
        var graph = AdaptiveNormGraph(parametersAreInitializers: false);

        var optimized = GraphOptimizer.Optimize(graph);
        var fusedCount = optimized.Nodes.Count(n => n.OpType == "FusedInstanceNorm");
        if (fusedCount != 1)
            throw new Exception($"expected 1 FusedInstanceNorm, got {fusedCount} on {BackendName} "
                + $"({string.Join(",", optimized.Nodes.Select(n => n.OpType))})");
        if (optimized.Nodes.Any(n => n.OpType is "ReduceMean" or "Sqrt" or "Div" or "Sub"))
            throw new Exception($"fusion left chain nodes behind on {BackendName}: "
                + string.Join(",", optimized.Nodes.Select(n => n.OpType)));
        var fn = optimized.Nodes.First(n => n.OpType == "FusedInstanceNorm");
        if (fn.Inputs[0] != "X" || fn.Inputs[1] != "S" || fn.Inputs[2] != "B" || fn.Outputs[0] != "Y")
            throw new Exception($"wiring wrong: in=[{string.Join(",", fn.Inputs)}] out={fn.Outputs[0]}");
        if (fn.Attributes!["axis"].GetInt32() != 2)
            throw new Exception($"axis {fn.Attributes["axis"].GetInt32()} != 2 on {BackendName}");
        if (MathF.Abs(fn.Attributes["epsilon"].GetSingle() - InEps) > 1e-12f)
            throw new Exception($"epsilon {fn.Attributes["epsilon"].GetSingle()} != {InEps}");

        var rng = new Random(7717);
        var x = RandFloats(InC * InL, rng);
        var scale = RandFloats(InC, rng);
        var bias = RandFloats(InC, rng);
        var expected = CpuInstanceNormAffine(x, scale, bias, InC, InL, InEps);

        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);   // Compile re-runs Optimize internally
        using var ex = new GraphExecutor(accelerator, compiled, new Dictionary<string, Tensor>());
        using var xB = accelerator.Allocate1D(x);
        using var sB = accelerator.Allocate1D(scale);
        using var bB = accelerator.Allocate1D(bias);
        using var host = accelerator.Allocate1D<float>(x.Length);
        var outs = await ex.RunAsync(new Dictionary<string, Tensor>
        {
            ["X"] = new Tensor(xB.View, new[] { 1, InC, InL }),
            ["S"] = new Tensor(sB.View, new[] { 1, InC, 1 }),
            ["B"] = new Tensor(bB.View, new[] { 1, InC, 1 }),
        });
        await host.View.CopyFromAsync(outs["Y"].Data.SubView(0, x.Length));
        await accelerator.SynchronizeAsync();
        var got = await host.CopyToHostAsync<float>(0, x.Length);

        var worst = 0f;
        for (var i = 0; i < x.Length; i++) worst = MathF.Max(worst, MathF.Abs(got[i] - expected[i]));
        if (worst > 2e-4f)
            throw new Exception($"fused adaptive norm diverged from the CPU definition "
                + $"(worst |delta|={worst:E3}) on {BackendName}");
        Console.WriteLine($"[InstanceNormFusion] 9 nodes -> 1, matches the definition "
            + $"(worst |delta|={worst:E3}) on {BackendName}");
    });

    /// <summary>
    /// The identical chain with CONSTANT parameters must be left alone by this pass.
    /// </summary>
    /// <remarks>
    /// 🔴 This is the guard, not a detail. A LayerNorm is these same nine nodes with scale and bias shaped
    /// like the NORMALISED axis instead of the channel axis, and the optimizer cannot see shapes - so the
    /// only thing separating "fuse it" from "compute something else entirely" is that a LayerNorm's
    /// parameters are always weights and an adaptive norm's never are. Delete that rule and this test is
    /// the one that fails.
    /// </remarks>
    [TestMethod]
    public async Task InstanceNormFusion_LeavesAConstantParameterChainAlone() => await RunPureTest(() =>
    {
        var graph = AdaptiveNormGraph(parametersAreInitializers: true);
        var optimized = GraphOptimizer.Optimize(graph);
        if (optimized.Nodes.Any(n => n.OpType == "FusedInstanceNorm"))
            throw new Exception("FuseInstanceNorm claimed a chain whose scale and bias are initializers - "
                + "it cannot tell that apart from a LayerNorm without shapes, so it must decline");
        if (!GraphOptimizer.LastInstanceNormRejects.ContainsKey(
                "scale/bias are initializers - possibly a LayerNorm"))
            throw new Exception("the pass declined for some other reason than the initializer rule, so "
                + "this test is not exercising the guard it claims to: "
                + string.Join(", ", GraphOptimizer.LastInstanceNormRejects.Select(kv => $"{kv.Key}={kv.Value}")));
        return Task.CompletedTask;
    });
}
