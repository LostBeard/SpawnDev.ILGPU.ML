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
    private MatMulOperator? _matMul;

    public FusedScaledMatMulOperator(OperatorRegistry registry) => _registry = registry;

    public string OpType => "FusedScaledMatMul";

    public int[][] InferOutputShapes(int[][] inputShapes, Dictionary<string, object> attributes)
    {
        // Exactly MatMul's shape (batch broadcast, rank-1 operands): it IS a MatMul with a scale.
        if (inputShapes.Length < 2) return new[] { inputShapes[0] };
        return new[] { MatMulOperator.OutputShape(inputShapes[0], inputShapes[1]) };
    }

    public void Execute(OnnxOpContext ctx)
    {
        var A = ctx.Inputs[0];
        var B = ctx.Inputs[1];
        var scaleTensor = ctx.Inputs[2];
        var output = ctx.Outputs[0];

        bool isDiv = ctx.GetString("is_div", "false") == "True" ||
                     ctx.GetString("is_div", "false") == "true";

        // The scale: the value the optimizer resolved (already inverted for Div), else the executor's pre-read
        // constant. Never neither - that used to fall through to 1.0 and return UNSCALED scores without a word.
        float scaleValue;
        var preRead = ctx.TryGetInputValues(2);
        if (ctx.Attributes.ContainsKey("scale"))
            scaleValue = ctx.GetFloat("scale", 1f);
        else if (preRead != null && preRead.Length > 0)
            scaleValue = isDiv ? (1f / preRead[0]) : preRead[0];
        else
            throw new InvalidOperationException(
                $"FusedScaledMatMul: the scale input '{(ctx.InputNames.Length > 2 ? ctx.InputNames[2] : "?")}' is neither resolved at optimize time "
              + "nor a pre-read runtime constant, so its value is unknown here. Refusing to compute unscaled scores.");

        // The MatMul itself is MatMulOperator's, so every operand form it handles (a shared 2-D B under a batched A,
        // broadcast batch dims, rank-1 operands) is handled here too. This used to call BatchedMatMul whenever either
        // operand was not rank 2, which strides B per batch entry: wrong for a shared B and for [1,H,K,N] x [B,H,M,K].
        (_matMul ??= new MatMulOperator(_registry)).ExecuteDense(A, B, output);

        // Apply scale in-place — do NOT use Scale(output, output) which violates WebGPU aliasing
        int totalElements = output.ElementCount;
        if (MathF.Abs(scaleValue - 1f) > 1e-7f)
        {
            _registry.ElementWise.ScaleInPlace(output.Data.SubView(0, totalElements),
                totalElements, scaleValue);
        }
    }
}
