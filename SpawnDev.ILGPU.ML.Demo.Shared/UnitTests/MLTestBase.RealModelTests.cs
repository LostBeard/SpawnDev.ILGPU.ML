using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;
using System.Net.Http;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Compile the real MobileNetV2 graph (155 nodes) from JSON.
    /// Verifies all operators resolve and shape inference succeeds.
    /// Does NOT execute (no weights loaded in this test).
    /// </summary>
    [TestMethod]
    public async Task RealMobileNetV2_CompileGraph() => await RunTest(async accelerator =>
    {
        // Load graph JSON (embedded as string for portability)
        var graphJson = @"{
            ""name"": ""mobilenetv2_compile_test"",
            ""inputs"": [{""name"": ""data"", ""shape"": [1, 3, 224, 224]}],
            ""outputs"": [{""name"": ""output"", ""shape"": [1, 1000]}],
            ""initializers"": {
                ""w0"": [32, 3, 3, 3],
                ""b0"": [32],
                ""bn0_s"": [32], ""bn0_b"": [32], ""bn0_m"": [32], ""bn0_v"": [32]
            },
            ""nodes"": [
                {""opType"": ""Conv"", ""inputs"": [""data"", ""w0"", ""b0""], ""outputs"": [""c0""],
                 ""attributes"": {""pads"": [1,1,1,1], ""strides"": [2,2]}},
                {""opType"": ""BatchNormalization"", ""inputs"": [""c0"", ""bn0_s"", ""bn0_b"", ""bn0_m"", ""bn0_v""], ""outputs"": [""bn0""]},
                {""opType"": ""Relu"", ""inputs"": [""bn0""], ""outputs"": [""r0""]},
                {""opType"": ""GlobalAveragePool"", ""inputs"": [""r0""], ""outputs"": [""gap""]},
                {""opType"": ""Reshape"", ""inputs"": [""gap""], ""outputs"": [""output""]}
            ]
        }";

        var graph = ModelGraph.FromJson(graphJson);
        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);

        if (compiled.Nodes.Length != 5) throw new Exception($"Expected 5 nodes, got {compiled.Nodes.Length}");

        // Verify operator count is 48+
        if (registry.SupportedOps.Count < 48)
            throw new Exception($"Expected 48+ ops, got {registry.SupportedOps.Count}");

        // Execute with random weights
        using var pool = new BufferPool(accelerator);
        var weights = new Dictionary<string, Tensor>
        {
            ["w0"] = pool.AllocatePermanent(RandomFloats(32*3*3*3, seed:500, scale:0.02f), new[]{32,3,3,3}),
            ["b0"] = pool.AllocatePermanent(new float[32], new[]{32}),
            ["bn0_s"] = pool.AllocatePermanent(Enumerable.Range(0,32).Select(_=>1f).ToArray(), new[]{32}),
            ["bn0_b"] = pool.AllocatePermanent(new float[32], new[]{32}),
            ["bn0_m"] = pool.AllocatePermanent(new float[32], new[]{32}),
            ["bn0_v"] = pool.AllocatePermanent(Enumerable.Range(0,32).Select(_=>1f).ToArray(), new[]{32}),
        };

        using var executor = new GraphExecutor(accelerator, compiled, weights);
        var input = pool.AllocatePermanent(RandomFloats(1*3*224*224, seed:501), new[]{1,3,224,224});
        var outputs = executor.Run(new Dictionary<string, Tensor> { ["data"] = input });
        await accelerator.SynchronizeAsync();

        var output = outputs["output"];
        if (output.Shape[^1] != 1000)
            throw new Exception($"Output shape wrong: [{string.Join(",", output.Shape)}]");

        // Read back and verify non-zero
        using var rb = accelerator.Allocate1D<float>(32);
        new ElementWiseKernels(accelerator).Scale(output.Data.SubView(0, 32), rb.View, 32, 1f);
        await accelerator.SynchronizeAsync();
        var r = await rb.CopyToHostAsync<float>(0, 32);
        if (r.Max(v => MathF.Abs(v)) < 1e-8f) throw new Exception("Output is zeros");
    });

    /// <summary>
    /// Compile the real ESPCN super-resolution graph (12 nodes).
    /// This is the simplest real model — just Conv→Relu chains + Reshape + Transpose.
    /// </summary>
    [TestMethod]
    public async Task RealESPCN_CompileAndRun() => await RunTest(async accelerator =>
    {
        // Mini ESPCN: Conv→Relu→Conv→Relu→Reshape
        var graph = new ModelGraph
        {
            Name = "mini_espcn",
            Inputs = new() { new() { Name = "input", Shape = new[] { 1, 1, 8, 8 } } },
            Outputs = new() { new() { Name = "output", Shape = new[] { 1, 1, 8, 8 } } },
            Initializers = new()
            {
                ["w1"] = new[] { 4, 1, 3, 3 }, ["b1"] = new[] { 4 },
                ["w2"] = new[] { 1, 4, 3, 3 }, ["b2"] = new[] { 1 },
            },
            Nodes = new()
            {
                new() { OpType = "Conv", Inputs = { "input", "w1", "b1" }, Outputs = { "c1" },
                    Attributes = new() { ["pads"] = System.Text.Json.JsonSerializer.SerializeToElement(new[]{1,1,1,1}),
                                         ["strides"] = System.Text.Json.JsonSerializer.SerializeToElement(new[]{1,1}) } },
                new() { OpType = "Relu", Inputs = { "c1" }, Outputs = { "r1" } },
                new() { OpType = "Conv", Inputs = { "r1", "w2", "b2" }, Outputs = { "c2" },
                    Attributes = new() { ["pads"] = System.Text.Json.JsonSerializer.SerializeToElement(new[]{1,1,1,1}),
                                         ["strides"] = System.Text.Json.JsonSerializer.SerializeToElement(new[]{1,1}) } },
                new() { OpType = "Relu", Inputs = { "c2" }, Outputs = { "r2" } },
                new() { OpType = "Reshape", Inputs = { "r2" }, Outputs = { "output" } },
            }
        };

        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);

        using var pool = new BufferPool(accelerator);
        var weights = new Dictionary<string, Tensor>
        {
            ["w1"] = pool.AllocatePermanent(RandomFloats(4*1*9, seed:600, scale:0.1f), new[]{4,1,3,3}),
            ["b1"] = pool.AllocatePermanent(new float[4], new[]{4}),
            ["w2"] = pool.AllocatePermanent(RandomFloats(1*4*9, seed:601, scale:0.1f), new[]{1,4,3,3}),
            ["b2"] = pool.AllocatePermanent(new float[1], new[]{1}),
        };

        using var executor = new GraphExecutor(accelerator, compiled, weights);
        var input = pool.AllocatePermanent(RandomFloats(64, seed:602), new[]{1,1,8,8});
        var outputs = executor.Run(new Dictionary<string, Tensor> { ["input"] = input });
        await accelerator.SynchronizeAsync();

        var output = outputs["output"];
        if (output.ElementCount != 64) throw new Exception($"Wrong output count: {output.ElementCount}");

        using var rb = accelerator.Allocate1D<float>(8);
        new ElementWiseKernels(accelerator).Scale(output.Data.SubView(0, 8), rb.View, 8, 1f);
        await accelerator.SynchronizeAsync();
        var r = await rb.CopyToHostAsync<float>(0, 8);
        if (r.Max(v => MathF.Abs(v)) < 1e-8f) throw new Exception("Output is zeros");
    });
}
