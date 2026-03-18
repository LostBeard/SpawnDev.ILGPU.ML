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
    /// <summary>
    /// Run a mini classification graph (Conv→Relu→GlobalAvgPool→Reshape) with known
    /// non-random weights and verify the output is discriminative (not uniform).
    /// Tests the full pipeline on the current backend.
    /// </summary>
    [TestMethod]
    public async Task Pipeline_ConvReluPoolReshape_NonUniformOutput() => await RunTest(async accelerator =>
    {
        // Build a minimal classification graph
        var graph = new ModelGraph
        {
            Name = "pipeline_test",
            Inputs = new() { new() { Name = "input", Shape = new[] { 1, 3, 8, 8 } } },
            Outputs = new() { new() { Name = "output", Shape = new[] { 1, 4 } } },
            Initializers = new()
            {
                ["conv_w"] = new[] { 4, 3, 3, 3 },
                ["conv_b"] = new[] { 4 },
            },
            Nodes = new()
            {
                new() { OpType = "Conv", Inputs = { "input", "conv_w", "conv_b" }, Outputs = { "c1" },
                    Attributes = new() { ["pads"] = System.Text.Json.JsonSerializer.SerializeToElement(new[]{1,1,1,1}),
                                         ["strides"] = System.Text.Json.JsonSerializer.SerializeToElement(new[]{1,1}) } },
                new() { OpType = "Relu", Inputs = { "c1" }, Outputs = { "r1" } },
                new() { OpType = "GlobalAveragePool", Inputs = { "r1" }, Outputs = { "gap" } },
                new() { OpType = "Reshape", Inputs = { "gap" }, Outputs = { "output" } },
            }
        };

        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);

        // Create KNOWN weights — each filter has a different pattern so outputs differ
        using var pool = new BufferPool(accelerator);
        var convW = new float[4 * 3 * 3 * 3]; // 4 filters
        // Filter 0: all positive → large activation
        for (int i = 0; i < 3 * 9; i++) convW[0 * 27 + i] = 0.1f;
        // Filter 1: all negative → zero after ReLU
        for (int i = 0; i < 3 * 9; i++) convW[1 * 27 + i] = -0.1f;
        // Filter 2: mixed
        for (int i = 0; i < 3 * 9; i++) convW[2 * 27 + i] = (i % 2 == 0) ? 0.2f : -0.05f;
        // Filter 3: strong single channel
        for (int i = 0; i < 9; i++) convW[3 * 27 + i] = 0.3f; // Only R channel

        var weights = new Dictionary<string, Tensor>
        {
            ["conv_w"] = pool.AllocatePermanent(convW, new[] { 4, 3, 3, 3 }),
            ["conv_b"] = pool.AllocatePermanent(new float[] { 0.01f, 0.02f, 0.03f, 0.04f }, new[] { 4 }),
        };

        // Create a non-uniform input image (gradient)
        var inputData = new float[1 * 3 * 8 * 8];
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < 8; y++)
                for (int x = 0; x < 8; x++)
                    inputData[c * 64 + y * 8 + x] = (c * 64 + y * 8 + x) / 192f; // [0, 1] range

        var input = pool.AllocatePermanent(inputData, new[] { 1, 3, 8, 8 });

        // Execute
        using var executor = new GraphExecutor(accelerator, compiled, weights);
        var outputs = executor.Run(new Dictionary<string, Tensor> { ["input"] = input });
        await accelerator.SynchronizeAsync();

        // Read output
        var output = outputs["output"];
        if (output.ElementCount != 4) throw new Exception($"Wrong output count: {output.ElementCount}");

        using var readBuf = accelerator.Allocate1D<float>(4);
        new ElementWiseKernels(accelerator).Scale(output.Data.SubView(0, 4), readBuf.View, 4, 1f);
        await accelerator.SynchronizeAsync();
        var logits = await readBuf.CopyToHostAsync<float>(0, 4);

        // Verify output is non-uniform
        float min = logits.Min();
        float max = logits.Max();
        float range = max - min;

        var logitStr = string.Join(", ", logits.Select(v => v.ToString("F4")));
        Console.WriteLine($"[Pipeline] Logits: [{logitStr}], range={range:F4}");

        if (range < 0.01f)
            throw new Exception($"Output is near-uniform (range={range:F6}). Logits: [{logitStr}]");

        // Filter 1 (all negative weights) should produce 0 after ReLU
        if (logits[1] > 0.03f)
            throw new Exception($"Filter 1 should be ~0 after ReLU, got {logits[1]:F4}");

        // Filter 0 (all positive) should produce the largest value
        if (logits[0] < logits[1])
            throw new Exception($"Filter 0 (positive) should be > filter 1 (negative→relu→0)");
    });
}
