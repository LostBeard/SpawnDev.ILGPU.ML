using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// Fused Linear operator: Output = Activation(MatMul(Input, Weights) + Bias)
/// Executes matrix multiplication, bias addition, and activation in a single kernel dispatch.
/// Eliminates 2 out of 3 global memory write cycles.
///
/// Inputs: [input, weights, bias]
/// Outputs: [result]
/// Attributes: activation ("none", "Relu", "Gelu", "Sigmoid", "Tanh", "Clip")
/// </summary>
public class FusedLinearOperator : IOnnxOperator
{
    private readonly OperatorRegistry _registry;
    private FusedLinearKernel? _kernel;

    public FusedLinearOperator(OperatorRegistry registry) => _registry = registry;

    public string OpType => "FusedLinear";

    public int[][] InferOutputShapes(int[][] inputShapes, Dictionary<string, object> attributes)
    {
        // Input: [M, K], Weights: [K, N] → Output: [M, N]
        if (inputShapes.Length < 2) return new[] { inputShapes[0] };
        int M = inputShapes[0][^2];
        int N = inputShapes[1][^1];
        var outShape = inputShapes[0].ToArray();
        outShape[^1] = N;
        return new[] { outShape };
    }

    public void Execute(OnnxOpContext ctx)
    {
        var input = ctx.Inputs[0];  // [M, K]
        var weights = ctx.Inputs[1]; // [K, N]
        var bias = ctx.Inputs[2];    // [N]
        var output = ctx.Outputs[0]; // [M, N]

        int M = input.Shape[^2];
        int K = input.Shape[^1];
        int N = weights.Shape[^1];

        // Determine activation type
        var actStr = ctx.GetString("activation", "none").ToLowerInvariant();
        var activation = actStr switch
        {
            "relu" => FusedActivation.ReLU,
            "gelu" => FusedActivation.GELU,
            "sigmoid" => FusedActivation.Sigmoid,
            "tanh" => FusedActivation.Tanh,
            "silu" => FusedActivation.SiLU,
            "clip" => FusedActivation.ReLU, // RELU6 approximation
            _ => FusedActivation.None
        };

        _kernel ??= new FusedLinearKernel(_registry.Accelerator);
        _kernel.Forward(
            input.Data.SubView(0, M * K),
            weights.Data.SubView(0, K * N),
            bias.Data.SubView(0, N),
            output.Data.SubView(0, M * N),
            M, K, N, activation);
    }
}
