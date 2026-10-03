using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task Graph_CompileSimpleModel() => await RunTest(async accelerator =>
    {
        var registry = new OperatorRegistry(accelerator);

        // Build a simple graph: output = Relu(MatMul(input, weight) + bias)
        var graph = new ModelGraph
        {
            Name = "simple_mlp",
            Inputs = new() { new() { Name = "input", Shape = new[] { 4, 8 } } },
            Outputs = new() { new() { Name = "relu_out", Shape = new[] { 4, 3 } } },
            Initializers = new()
            {
                ["weight"] = new[] { 8, 3 },
                ["bias"] = new[] { 3 },
            },
            Nodes = new()
            {
                new() { OpType = "MatMul", Inputs = { "input", "weight" }, Outputs = { "mm_out" } },
                new() { OpType = "Add", Inputs = { "mm_out", "bias" }, Outputs = { "add_out" } },
                new() { OpType = "Relu", Inputs = { "add_out" }, Outputs = { "relu_out" } },
            }
        };

        var compiler = new GraphCompiler(registry) { EnableOptimization = false };
        var compiled = compiler.Compile(graph);

        if (compiled.Nodes.Length != 3) throw new Exception($"Expected 3 nodes, got {compiled.Nodes.Length}");
        if (compiled.InputNames.Length != 1) throw new Exception("Wrong input count");
        if (compiled.OutputNames[0] != "relu_out") throw new Exception("Wrong output name");

        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Graph_FlattenConcatWithEmptyShapeSlice() => await RunTest(async accelerator =>
    {
        // REGRESSION (Video Depth Anything): torch's x.flatten(0, 1) exports as
        //   Reshape(x, Concat(Shape(x)[0:0], [-1], Shape(x)[2:5]))
        // and Shape(x)[0:0] is a KNOWN EMPTY constant. The compile-time Concat fold refused any empty input, so
        // the target shape went unresolved and the Reshape inferred [1,1,3,4,6] -> [72,1,1,1] - silently, with
        // every later shape built on it. The output must be [1,3,4,6], element for element the input.
        int[] inShape = { 1, 1, 3, 4, 6 };
        int count = 72;
        var graph = new ModelGraph
        {
            Name = "flatten01",
            Inputs = new() { new() { Name = "x", Shape = inShape } },
            Outputs = new() { new() { Name = "y", Shape = new[] { 1, 3, 4, 6 } } },
            Initializers = new()
            {
                ["s0"] = new[] { 1 }, ["e0"] = new[] { 1 }, ["s2"] = new[] { 1 }, ["e5"] = new[] { 1 },
                ["ax0"] = new[] { 1 }, ["m1"] = new[] { 1 },
            },
            Nodes = new()
            {
                new() { OpType = "Shape", Inputs = { "x" }, Outputs = { "shp" } },
                new() { OpType = "Slice", Inputs = { "shp", "s0", "e0", "ax0" }, Outputs = { "lead" } },   // [0:0] = []
                new() { OpType = "Slice", Inputs = { "shp", "s2", "e5", "ax0" }, Outputs = { "tail" } },   // [2:5]
                new() { OpType = "Concat", Inputs = { "lead", "m1", "tail" }, Outputs = { "target" },
                    Attributes = new() { ["axis"] = System.Text.Json.JsonSerializer.SerializeToElement(0L) } },
                new() { OpType = "Reshape", Inputs = { "x", "target" }, Outputs = { "flat" } },
                // An intermediate, so the graph's declared output shape cannot stand in for the Reshape's inference.
                new() { OpType = "Relu", Inputs = { "flat" }, Outputs = { "y" } },
            }
        };
        graph.ConstantData = new()
        {
            ["s0"] = new[] { 0 }, ["e0"] = new[] { 0 }, ["s2"] = new[] { 2 }, ["e5"] = new[] { 5 },
            ["ax0"] = new[] { 0 }, ["m1"] = new[] { -1 },
        };
        graph.FloatConstantData = graph.ConstantData.ToDictionary(kv => kv.Key, kv => kv.Value.Select(v => (float)v).ToArray());

        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);
        // The COMPILE-TIME shape is what later shape inference builds on (the runtime path recomputes this
        // Reshape on its own and would hide the bug): it must already be [1,3,4,6].
        var reshapeShape = compiled.Nodes.Single(nd => nd.OpType == "Reshape").OutputShapes[0];
        if (!reshapeShape.SequenceEqual(new[] { 1, 3, 4, 6 }))
            throw new Exception($"compile-time flatten(0,1) shape [{string.Join(",", reshapeShape)}], expected [1,3,4,6]");
        using var pool = new BufferPool(accelerator);
        var weights = graph.ConstantData.ToDictionary(kv => kv.Key,
            kv => pool.AllocatePermanent(kv.Value.Select(v => (float)v).ToArray(), new[] { kv.Value.Length }));
        using var ex = new GraphExecutor(accelerator, compiled, weights, graph.FloatConstantData);
        var data = Enumerable.Range(0, count).Select(i => (float)i).ToArray();
        var outs = await ex.RunAsync(new Dictionary<string, Tensor> { ["x"] = pool.AllocatePermanent(data, inShape) });
        var y = outs["y"];
        if (!y.Shape.SequenceEqual(new[] { 1, 3, 4, 6 }))
            throw new Exception($"flatten(0,1) gave [{string.Join(",", y.Shape)}], expected [1,3,4,6] on {BackendName}");
        using var host = accelerator.Allocate1D<float>(count);
        await host.View.CopyFromAsync(y.Data.SubView(0, count));
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, host.View, data, 0f, "flatten(0,1): ");
    });

    [TestMethod]
    public async Task Graph_ExecuteSimpleMLP() => await RunTest(async accelerator =>
    {
        var registry = new OperatorRegistry(accelerator);

        int M = 4, K = 8, N = 3;
        var inputData = RandomFloats(M * K, seed: 180);
        var weightData = RandomFloats(K * N, seed: 181, scale: 0.5f);
        var biasData = RandomFloats(N, seed: 182, scale: 0.1f);

        // CPU reference: Relu(MatMul(input, weight) + bias)
        var mmOut = CpuMatMul(inputData, weightData, M, K, N);
        var expected = new float[M * N];
        for (int i = 0; i < M * N; i++)
        {
            float v = mmOut[i] + biasData[i % N]; // Add bias (broadcast)
            expected[i] = MathF.Max(0, v); // Relu
        }

        // Build graph
        var graph = new ModelGraph
        {
            Name = "test_mlp",
            Inputs = new() { new() { Name = "input", Shape = new[] { M, K } } },
            Outputs = new() { new() { Name = "relu_out", Shape = new[] { M, N } } },
            Initializers = new()
            {
                ["weight"] = new[] { K, N },
                ["bias"] = new[] { N },
            },
            Nodes = new()
            {
                new() { OpType = "MatMul", Inputs = { "input", "weight" }, Outputs = { "mm_out" } },
                new() { OpType = "Add", Inputs = { "mm_out", "bias" }, Outputs = { "add_out" } },
                new() { OpType = "Relu", Inputs = { "add_out" }, Outputs = { "relu_out" } },
            }
        };

        // Compile
        var compiler = new GraphCompiler(registry);
        var compiled = compiler.Compile(graph);

        // Create weight tensors
        using var pool = new BufferPool(accelerator);
        var weights = new Dictionary<string, Tensor>
        {
            ["weight"] = pool.AllocatePermanent(weightData, new[] { K, N }),
            ["bias"] = pool.AllocatePermanent(biasData, new[] { N }),
        };

        // Execute
        using var executor = new GraphExecutor(accelerator, compiled, weights);
        var inputTensor = pool.AllocatePermanent(inputData, new[] { M, K });
        var outputs = await executor.RunAsync(new Dictionary<string, Tensor> { ["input"] = inputTensor });

        await accelerator.SynchronizeAsync();

        // Read output
        var outputTensor = outputs["relu_out"];
        using var readBuf = accelerator.Allocate1D<float>(M * N);
        var ew = new ElementWiseKernels(accelerator);
        ew.Scale(outputTensor.Data.SubView(0, M * N), readBuf.View, M * N, 1f);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, readBuf.View.SubView(0, M * N), expected, K * 2e-6f, "Graph MLP: ");
    });

    [TestMethod]
    public async Task Graph_SerializeRoundTrip() => await RunTest(async accelerator =>
    {
        var graph = new ModelGraph
        {
            Name = "test",
            Inputs = new() { new() { Name = "x", Shape = new[] { 1, 3, 224, 224 } } },
            Outputs = new() { new() { Name = "y", Shape = new[] { 1, 1000 } } },
            Initializers = new() { ["w"] = new[] { 3, 64, 7, 7 } },
            Nodes = new()
            {
                new() { OpType = "Conv", Inputs = { "x", "w" }, Outputs = { "conv_out" },
                    Attributes = new() { ["pads"] = System.Text.Json.JsonSerializer.SerializeToElement(new[] { 3, 3, 3, 3 }),
                                         ["strides"] = System.Text.Json.JsonSerializer.SerializeToElement(new[] { 2, 2 }) } },
            }
        };

        var json = graph.ToJson();
        var restored = ModelGraph.FromJson(json);

        if (restored.Name != "test") throw new Exception("Name mismatch");
        if (restored.Nodes.Count != 1) throw new Exception("Node count mismatch");
        if (restored.Nodes[0].OpType != "Conv") throw new Exception("OpType mismatch");
        if (restored.Inputs[0].Shape[2] != 224) throw new Exception("Shape mismatch");

        await Task.CompletedTask;
    });
}
