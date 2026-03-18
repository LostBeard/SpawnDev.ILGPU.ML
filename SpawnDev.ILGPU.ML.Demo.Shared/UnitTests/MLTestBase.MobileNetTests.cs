using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task MobileNetV2_GraphCompiles() => await RunTest(async accelerator =>
    {
        // Load the real MobileNetV2 graph JSON (embedded in test assembly or loaded from file)
        // For this test, build a mini version: Conv→BN→Relu→GlobalAvgPool→Conv→Reshape
        var graph = new ModelGraph
        {
            Name = "mini_mobilenet",
            Inputs = new() { new() { Name = "input", Shape = new[] { 1, 3, 8, 8 } } },
            Outputs = new() { new() { Name = "output", Shape = new[] { 1, 10 } } },
            Initializers = new()
            {
                ["conv1_w"] = new[] { 16, 3, 3, 3 },
                ["conv1_b"] = new[] { 16 },
                ["bn1_scale"] = new[] { 16 },
                ["bn1_bias"] = new[] { 16 },
                ["bn1_mean"] = new[] { 16 },
                ["bn1_var"] = new[] { 16 },
                ["conv2_w"] = new[] { 10, 16, 1, 1 },
                ["conv2_b"] = new[] { 10 },
            },
            Nodes = new()
            {
                new() { OpType = "Conv", Inputs = { "input", "conv1_w", "conv1_b" }, Outputs = { "conv1_out" },
                    Attributes = new() {
                        ["pads"] = System.Text.Json.JsonSerializer.SerializeToElement(new[] { 1, 1, 1, 1 }),
                        ["strides"] = System.Text.Json.JsonSerializer.SerializeToElement(new[] { 1, 1 })
                    } },
                new() { OpType = "BatchNormalization",
                    Inputs = { "conv1_out", "bn1_scale", "bn1_bias", "bn1_mean", "bn1_var" },
                    Outputs = { "bn1_out" } },
                new() { OpType = "Relu", Inputs = { "bn1_out" }, Outputs = { "relu1_out" } },
                new() { OpType = "GlobalAveragePool", Inputs = { "relu1_out" }, Outputs = { "gap_out" } },
                new() { OpType = "Conv", Inputs = { "gap_out", "conv2_w", "conv2_b" }, Outputs = { "conv2_out" } },
                new() { OpType = "Reshape", Inputs = { "conv2_out" }, Outputs = { "output" } },
            }
        };

        var registry = new OperatorRegistry(accelerator);
        var compiler = new GraphCompiler(registry);
        var compiled = compiler.Compile(graph);

        if (compiled.Nodes.Length != 6) throw new Exception($"Expected 6 nodes, got {compiled.Nodes.Length}");

        // Create random weights
        using var pool = new BufferPool(accelerator);
        var weights = new Dictionary<string, Tensor>
        {
            ["conv1_w"] = pool.AllocatePermanent(RandomFloats(16 * 3 * 3 * 3, seed: 300, scale: 0.1f), new[] { 16, 3, 3, 3 }),
            ["conv1_b"] = pool.AllocatePermanent(RandomFloats(16, seed: 301, scale: 0.01f), new[] { 16 }),
            ["bn1_scale"] = pool.AllocatePermanent(Enumerable.Range(0, 16).Select(_ => 1f).ToArray(), new[] { 16 }),
            ["bn1_bias"] = pool.AllocatePermanent(new float[16], new[] { 16 }),
            ["bn1_mean"] = pool.AllocatePermanent(new float[16], new[] { 16 }),
            ["bn1_var"] = pool.AllocatePermanent(Enumerable.Range(0, 16).Select(_ => 1f).ToArray(), new[] { 16 }),
            ["conv2_w"] = pool.AllocatePermanent(RandomFloats(10 * 16 * 1 * 1, seed: 302, scale: 0.1f), new[] { 10, 16, 1, 1 }),
            ["conv2_b"] = pool.AllocatePermanent(RandomFloats(10, seed: 303, scale: 0.01f), new[] { 10 }),
        };

        // Run
        using var executor = new GraphExecutor(accelerator, compiled, weights);
        var inputTensor = pool.AllocatePermanent(RandomFloats(1 * 3 * 8 * 8, seed: 304), new[] { 1, 3, 8, 8 });
        var outputs = executor.Run(new Dictionary<string, Tensor> { ["input"] = inputTensor });

        await accelerator.SynchronizeAsync();

        // Check output exists and has correct shape
        if (!outputs.ContainsKey("output")) throw new Exception("Missing output tensor");
        var output = outputs["output"];
        if (output.ElementCount != 10) throw new Exception($"Expected 10 outputs, got {output.ElementCount}");

        // Read output — should be non-zero (random weights produce non-zero output)
        using var readBuf = accelerator.Allocate1D<float>(10);
        var ew = new ElementWiseKernels(accelerator);
        ew.Scale(output.Data.SubView(0, 10), readBuf.View, 10, 1f);
        await accelerator.SynchronizeAsync();
        var result = await readBuf.CopyToHostAsync<float>(0, 10);

        float maxAbs = result.Max(v => MathF.Abs(v));
        if (maxAbs < 1e-6f) throw new Exception("Output is all zeros — graph didn't compute");
    });

    [TestMethod]
    public async Task SqueezeNet_MiniGraphExecution() => await RunTest(async accelerator =>
    {
        // Mini SqueezeNet: Conv→Relu→Conv→Relu→Conv→Relu→Concat→Conv→GlobalAvgPool→Reshape
        var graph = new ModelGraph
        {
            Name = "mini_squeezenet",
            Inputs = new() { new() { Name = "data", Shape = new[] { 1, 3, 8, 8 } } },
            Outputs = new() { new() { Name = "output", Shape = new[] { 1, 8 } } },
            Initializers = new()
            {
                ["c1w"] = new[] { 4, 3, 3, 3 }, ["c1b"] = new[] { 4 },
                ["c2aw"] = new[] { 4, 4, 1, 1 }, ["c2ab"] = new[] { 4 },
                ["c2bw"] = new[] { 4, 4, 1, 1 }, ["c2bb"] = new[] { 4 },
                ["c3w"] = new[] { 8, 8, 1, 1 }, ["c3b"] = new[] { 8 },
            },
            Nodes = new()
            {
                new() { OpType = "Conv", Inputs = { "data", "c1w", "c1b" }, Outputs = { "c1" },
                    Attributes = new() { ["pads"] = System.Text.Json.JsonSerializer.SerializeToElement(new[] { 1,1,1,1 }),
                                         ["strides"] = System.Text.Json.JsonSerializer.SerializeToElement(new[] { 1,1 }) } },
                new() { OpType = "Relu", Inputs = { "c1" }, Outputs = { "r1" } },
                new() { OpType = "Conv", Inputs = { "r1", "c2aw", "c2ab" }, Outputs = { "c2a" } },
                new() { OpType = "Relu", Inputs = { "c2a" }, Outputs = { "r2a" } },
                new() { OpType = "Conv", Inputs = { "r1", "c2bw", "c2bb" }, Outputs = { "c2b" } },
                new() { OpType = "Relu", Inputs = { "c2b" }, Outputs = { "r2b" } },
                new() { OpType = "Concat", Inputs = { "r2a", "r2b" }, Outputs = { "cat" },
                    Attributes = new() { ["axis"] = System.Text.Json.JsonSerializer.SerializeToElement(1) } },
                new() { OpType = "Conv", Inputs = { "cat", "c3w", "c3b" }, Outputs = { "c3" } },
                new() { OpType = "GlobalAveragePool", Inputs = { "c3" }, Outputs = { "gap" } },
                new() { OpType = "Reshape", Inputs = { "gap" }, Outputs = { "output" } },
            }
        };

        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);

        using var pool = new BufferPool(accelerator);
        var weights = new Dictionary<string, Tensor>
        {
            ["c1w"] = pool.AllocatePermanent(RandomFloats(4*3*9, seed:400, scale:0.1f), new[]{4,3,3,3}),
            ["c1b"] = pool.AllocatePermanent(new float[4], new[]{4}),
            ["c2aw"] = pool.AllocatePermanent(RandomFloats(16, seed:401, scale:0.1f), new[]{4,4,1,1}),
            ["c2ab"] = pool.AllocatePermanent(new float[4], new[]{4}),
            ["c2bw"] = pool.AllocatePermanent(RandomFloats(16, seed:402, scale:0.1f), new[]{4,4,1,1}),
            ["c2bb"] = pool.AllocatePermanent(new float[4], new[]{4}),
            ["c3w"] = pool.AllocatePermanent(RandomFloats(64, seed:403, scale:0.1f), new[]{8,8,1,1}),
            ["c3b"] = pool.AllocatePermanent(new float[8], new[]{8}),
        };

        using var executor = new GraphExecutor(accelerator, compiled, weights);
        var input = pool.AllocatePermanent(RandomFloats(192, seed:404), new[]{1,3,8,8});
        var outputs = executor.Run(new Dictionary<string, Tensor> { ["data"] = input });
        await accelerator.SynchronizeAsync();

        var output = outputs["output"];
        if (output.ElementCount != 8) throw new Exception($"Wrong count: {output.ElementCount}");

        using var rb = accelerator.Allocate1D<float>(8);
        new ElementWiseKernels(accelerator).Scale(output.Data.SubView(0, 8), rb.View, 8, 1f);
        await accelerator.SynchronizeAsync();
        var r = await rb.CopyToHostAsync<float>(0, 8);
        if (r.Max(v => MathF.Abs(v)) < 1e-6f) throw new Exception("Output is zeros");
    });
}
