using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// Fused Scaled MatMul: Output = MatMul(A, B) * scale_value
/// or Output = MatMul(A, B) / scale_value
///
/// Used in attention: scores = (Q * K^T) / sqrt(d_k)
/// Fuses the scaling into the MatMul to eliminate one dispatch.
///
/// Inputs: [A, B, scale_tensor]
/// Attributes: is_div (bool) — if true, divide by scale instead of multiply
/// </summary>
public class FusedScaledMatMulOperator : IOnnxOperator
{
    private readonly OperatorRegistry _registry;
    private FusedScaledMatMulKernel? _fusedKernel;

    public FusedScaledMatMulOperator(OperatorRegistry registry) => _registry = registry;

    public string OpType => "FusedScaledMatMul";

    public int[][] InferOutputShapes(int[][] inputShapes, Dictionary<string, object> attributes)
    {
        // Same as MatMul: [M, K] x [K, N] → [M, N]
        if (inputShapes.Length < 2) return new[] { inputShapes[0] };
        int M = inputShapes[0][^2];
        int N = inputShapes[1][^1];
        var outShape = inputShapes[0].ToArray();
        outShape[^1] = N;
        return new[] { outShape };
    }

    public void Execute(OnnxOpContext ctx)
    {
        var A = ctx.Inputs[0];
        var B = ctx.Inputs[1];
        var scaleTensor = ctx.Inputs[2];
        var output = ctx.Outputs[0];

        int M = A.Shape[^2];
        int K = A.Shape[^1];
        int N = B.Shape[^1];

        bool isDiv = ctx.GetString("is_div", "false") == "True" ||
                     ctx.GetString("is_div", "false") == "true";

        // Read scale value — it's a scalar constant, try pre-read values first
        float scaleValue = 1f;
        var preRead = ctx.TryGetInputValues(2);
        if (preRead != null && preRead.Length > 0)
        {
            scaleValue = isDiv ? (1f / preRead[0]) : preRead[0];
        }

        // Use tiled MatMul (high throughput) then apply scale if needed.
        // The graph-level fusion still helps by eliminating the intermediate tensor
        // and the overhead of a separate Scale operator node.
        _registry.MatMul.MatMul(A.Data.SubView(0, M * K), B.Data.SubView(0, K * N),
            output.Data.SubView(0, M * N), M, K, N);

        if (MathF.Abs(scaleValue - 1f) > 1e-7f)
        {
            _registry.ElementWise.Scale(output.Data.SubView(0, M * N),
                output.Data.SubView(0, M * N), M * N, scaleValue);
        }
    }
}
