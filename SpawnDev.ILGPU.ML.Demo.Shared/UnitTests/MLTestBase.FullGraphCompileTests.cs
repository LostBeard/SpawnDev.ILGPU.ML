using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Verify OperatorRegistry has 48+ operators and covers all demo model needs.
    /// </summary>
    [TestMethod]
    public async Task OperatorCoverage_AllDemoModels() => await RunTest(async accelerator =>
    {
        var registry = new OperatorRegistry(accelerator);

        // All operators needed by the 8 demo models
        string[] allNeeded = {
            "Conv", "BatchNormalization", "Relu", "Add", "GlobalAveragePool", "Reshape",
            "MaxPool", "Concat", "Dropout", "AveragePool",
            "MatMul", "LayerNormalization", "Transpose", "Softmax", "Resize",
            "Sigmoid", "Mul",
            "Pad", "InstanceNormalization", "Constant", "Shape", "Gather", "Cast",
            "Floor", "Unsqueeze", "Slice", "Div", "Upsample",
        };

        var missing = new List<string>();
        foreach (var op in allNeeded)
        {
            if (!registry.IsSupported(op))
                missing.Add(op);
        }

        if (missing.Count > 0)
            throw new Exception($"Missing operators for demo models: {string.Join(", ", missing)}");

        if (registry.SupportedOps.Count < 48)
            throw new Exception($"Expected 48+ operators, got {registry.SupportedOps.Count}");

        await Task.CompletedTask;
    });
}
