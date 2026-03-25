using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Edge case tests for InferenceSession and graph compilation.
/// Verify graceful error handling for invalid inputs.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task EdgeCase_UnknownOperator_ThrowsClearError() => await RunTest(async accelerator =>
    {
        var graph = new ModelGraph
        {
            Name = "test",
            Inputs = new() { new GraphValueInfo { Name = "input", Shape = new[] { 1, 4 } } },
            Outputs = new() { new GraphValueInfo { Name = "output", Shape = new[] { 1, 4 } } },
            Initializers = new(),
            Nodes = new()
            {
                new GraphNode
                {
                    OpType = "CompletelyFakeOperator_DoesNotExist",
                    Inputs = new() { "input" },
                    Outputs = new() { "output" },
                }
            }
        };

        bool threwClearError = false;
        try
        {
            var registry = new Operators.OperatorRegistry(accelerator);
            var compiled = new GraphCompiler(registry).Compile(graph);
        }
        catch (Exception ex)
        {
            threwClearError = ex.Message.Contains("CompletelyFakeOperator") ||
                              ex.Message.Contains("not supported") ||
                              ex.Message.Contains("not registered");
        }

        if (!threwClearError)
            throw new Exception("Expected clear error for unknown operator, got no error or unclear error");

        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task EdgeCase_EmptyGraph_Compiles() => await RunTest(async accelerator =>
    {
        var graph = new ModelGraph
        {
            Name = "empty",
            Inputs = new() { new GraphValueInfo { Name = "input", Shape = new[] { 1, 4 } } },
            Outputs = new() { new GraphValueInfo { Name = "input", Shape = new[] { 1, 4 } } },
            Initializers = new(),
            Nodes = new()
        };

        var registry = new Operators.OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);

        if (compiled == null)
            throw new Exception("Empty graph should compile (pass-through)");
        if (compiled.Nodes.Length != 0)
            throw new Exception($"Empty graph should have 0 nodes, got {compiled.Nodes.Length}");

        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task EdgeCase_SoftmaxZeroDim_ThrowsClearError() => await RunTest(async accelerator =>
    {
        // Verify our Softmax validation catches zero-dimension tensors
        var registry = new Operators.OperatorRegistry(accelerator);
        using var pool = new Tensors.BufferPool(accelerator);

        var input = pool.AllocatePermanent(new float[] { 1, 2, 3, 4 }, new[] { 1, 4 });
        // Create output with intentionally wrong shape (has 0 dim)
        // This tests the validation we added for DepthAnything
        var output = pool.Rent(new[] { 1, 0, 4 }, "_test");

        bool threwClearError = false;
        try
        {
            var softmax = registry.Resolve("Softmax");
            var ctx = new Operators.OnnxOpContext
            {
                Inputs = new[] { new Tensors.Tensor(input.Data, new[] { 1, 0, 4 }) },
                Outputs = new[] { output },
                Attributes = new() { ["axis"] = (long)1 },
                Pool = pool,
                InputNames = new[] { "input" },
            };
            softmax.Execute(ctx);
        }
        catch (InvalidOperationException ex)
        {
            threwClearError = ex.Message.Contains("dimension") && ex.Message.Contains("0");
        }

        if (!threwClearError)
            throw new Exception("Softmax should throw clear error for zero-dimension tensor");

        await Task.CompletedTask;
    });
}
