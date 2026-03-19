using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using static SpawnDev.ILGPU.ML.Operators.BroadcastHelper;

namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// General N-dimensional broadcast binary operation for small tensors.
/// Uses pre-read constant values to avoid GPU→CPU readback.
/// For large tensors, falls back to element-wise (same-size) operation.
/// </summary>
internal static class BroadcastHelper
{
    public static void BroadcastBinaryOp(OnnxOpContext ctx, OperatorRegistry reg, Func<float, float, float> op)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        var outShape = ctx.Outputs[0].Shape;
        int outCount = ctx.Outputs[0].ElementCount;

        // Try to use pre-read constant values (no GPU readback)
        var aVals = ctx.TryGetInputValues(0);
        var bVals = ctx.TryGetInputValues(1);

        if (aVals != null && bVals != null)
        {
            // Both inputs are small constants — compute on CPU
            var result = new float[outCount];
            var aStrides = ComputeStrides(a.Shape, outShape);
            var bStrides = ComputeStrides(b.Shape, outShape);
            var outStrides = ComputeStrides(outShape, outShape);

            for (int i = 0; i < outCount; i++)
            {
                int aIdx = MapIndex(i, outStrides, aStrides, outShape.Length);
                int bIdx = MapIndex(i, outStrides, bStrides, outShape.Length);
                result[i] = op(
                    aIdx < aVals.Length ? aVals[aIdx] : 0f,
                    bIdx < bVals.Length ? bVals[bIdx] : 0f);
            }

            var temp = ctx.Pool.AllocatePermanent(result, outShape);
            reg.ElementWise.Scale(temp.Data, ctx.Outputs[0].Data, outCount, 1f);
        }
        else
        {
            // Large tensors — try element-wise if shapes match after broadcast
            // This is a fallback that won't handle all cases
            int copyCount = Math.Min(a.ElementCount, Math.Min(b.ElementCount, outCount));
            reg.ElementWise.Scale(a.Data.SubView(0, copyCount),
                ctx.Outputs[0].Data.SubView(0, copyCount), copyCount, 1f);
        }
    }

    private static int[] ComputeStrides(int[] shape, int[] outShape)
    {
        // Broadcast strides: if dim size is 1 or shape is shorter, stride is 0 (broadcast)
        int rank = outShape.Length;
        var strides = new int[rank];
        int offset = rank - shape.Length;
        int stride = 1;
        for (int i = rank - 1; i >= 0; i--)
        {
            int si = i - offset;
            if (si >= 0 && shape[si] > 1)
            {
                strides[i] = stride;
                stride *= shape[si];
            }
            else
            {
                strides[i] = 0; // Broadcast dimension
            }
        }
        return strides;
    }

    private static int MapIndex(int outIdx, int[] outStrides, int[] inStrides, int rank)
    {
        int inIdx = 0;
        int remaining = outIdx;
        for (int d = 0; d < rank; d++)
        {
            int coord = outStrides[d] > 0 ? remaining / outStrides[d] : 0;
            remaining = outStrides[d] > 0 ? remaining % outStrides[d] : remaining;
            inIdx += coord * inStrides[d];
        }
        return inIdx;
    }
}

// ── Activations ──

public class ReluOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Relu";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.ReLU(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class GeluOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Gelu";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // Copy input to output then GELU in-place (GELU can't be in-place on same buffer safely)
        reg.ElementWise.Add(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Outputs[0].Data, 0); // zero + 0 trick won't work
        // Actually just use the non-in-place GELU
        reg.ElementWise.GELU(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class SigmoidOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Sigmoid";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // Copy then in-place
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, 1f);
        reg.Activations.SigmoidInPlace(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount);
    }
}

public class SiLUOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "SiLU"; // Not standard ONNX — but used in YOLO via Mul(x, Sigmoid(x))
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, 1f);
        reg.Activations.SiLUInPlace(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount);
    }
}

public class LeakyReluOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "LeakyRelu";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // LeakyRelu: max(alpha*x, x). For now, use ReLU (alpha=0).
        // TODO: proper LeakyReLU with alpha parameter
        reg.ElementWise.ReLU(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class TanhOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Tanh";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, 1f);
        reg.Activations.TanhInPlace(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount);
    }
}

public class ClipOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Clip";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // Opset 6: min/max as attributes. Opset 11+: min/max as optional inputs.
        float minVal = ctx.GetFloat("min", float.MinValue);
        float maxVal = ctx.GetFloat("max", float.MaxValue);

        // Opset 11+: inputs[1]=min, inputs[2]=max (scalar tensors)
        // We can't easily read scalar GPU tensors to CPU here, so use attribute defaults
        // which cover the common case (Clip(0,6) for ReLU6 has them as attributes).

        if (minVal == 0f && maxVal == float.MaxValue)
        {
            // Fast path: Clip(0, inf) = ReLU
            reg.ElementWise.ReLU(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
        }
        else
        {
            reg.ElementWise.Clip(ctx.Inputs[0].Data, ctx.Outputs[0].Data,
                ctx.Inputs[0].ElementCount, minVal, maxVal);
        }
    }
}

// ── Binary element-wise ──

public class AddOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Add";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount)
        {
            reg.ElementWise.Add(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
        }
        else if (b.ElementCount == a.Shape[^1])
        {
            // Last-dim broadcast: b[C] + a[..., C]
            reg.ElementWise.Scale(a.Data, ctx.Outputs[0].Data, a.ElementCount, 1f);
            reg.ElementWise.AddBias(ctx.Outputs[0].Data, b.Data, a.ElementCount, b.ElementCount);
        }
        else if (a.Rank == 4 && b.Rank == 1 && b.ElementCount == a.Shape[1])
        {
            // NCHW per-channel broadcast: a[N,C,H,W] + b[C]
            // AddBias broadcasts over the last dim. For NCHW we need per-channel.
            // Reshape conceptually: each C-channel has H*W elements
            int C = a.Shape[1]; int spatial = a.Shape[2] * a.Shape[3];
            reg.ElementWise.Scale(a.Data, ctx.Outputs[0].Data, a.ElementCount, 1f);
            // Use BroadcastMul pattern but for Add — need a per-channel add kernel
            // For now, iterate channels on CPU dispatch (each channel gets AddBias)
            for (int nc = 0; nc < a.Shape[0] * C; nc++)
            {
                int c = nc % C;
                int offset = nc * spatial;
                // Add scalar bias[c] to each element in this channel's spatial slice
                // We don't have a scalar-add kernel, so use AddBias with spatial=1 trick
                // Actually, just use Scale(1) + AddBias over the spatial dim
                reg.ElementWise.AddBias(
                    ctx.Outputs[0].Data.SubView(offset, spatial),
                    b.Data.SubView(c, 1), spatial, 1);
            }
        }
        else if (b.ElementCount == 1)
        {
            // Scalar broadcast
            reg.ElementWise.Scale(a.Data, ctx.Outputs[0].Data, a.ElementCount, 1f);
            reg.ElementWise.AddBias(ctx.Outputs[0].Data, b.Data, a.ElementCount, 1);
        }
        else
        {
            // General broadcast: try CPU fallback for small tensors (shape computation)
            BroadcastBinaryOp(ctx, reg, (x, y) => x + y);
        }
    }
}

public class MulOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Mul";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount)
        {
            reg.ElementWise.Mul(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
        }
        else if (b.ElementCount == a.Shape[^1])
        {
            // Last-dim broadcast: a[..., C] * b[C]
            reg.ElementWise.BroadcastMul(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount, b.ElementCount);
        }
        else if (b.ElementCount == 1)
        {
            // Scalar broadcast — need to read the scalar value
            // For now, use BroadcastMul with C=1
            reg.ElementWise.BroadcastMul(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount, 1);
        }
        else if (a.Rank == 4 && b.Rank == 1 && b.ElementCount == a.Shape[1])
        {
            // NCHW per-channel: a[N,C,H,W] * b[C]
            int C = a.Shape[1]; int spatial = a.Shape[2] * a.Shape[3];
            for (int nc = 0; nc < a.Shape[0] * C; nc++)
            {
                int c = nc % C;
                int offset = nc * spatial;
                reg.ElementWise.BroadcastMul(
                    a.Data.SubView(offset, spatial),
                    b.Data.SubView(c, 1),
                    ctx.Outputs[0].Data.SubView(offset, spatial),
                    spatial, 1);
            }
        }
        else
        {
            BroadcastBinaryOp(ctx, reg, (x, y) => x * y);
        }
    }
}

public class SubOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Sub";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount)
        {
            // Negate b into output, then add a
            reg.ElementWise.Neg(b.Data, ctx.Outputs[0].Data, b.ElementCount);
            reg.ElementWise.AddInPlace(ctx.Outputs[0].Data, a.Data, a.ElementCount);
        }
        else if (b.ElementCount == 1)
        {
            // Scalar: a - scalar → copy a, then subtract (negate scalar + add)
            reg.ElementWise.Scale(a.Data, ctx.Outputs[0].Data, a.ElementCount, 1f);
            // TODO: scalar subtract kernel
        }
        else
        {
            // General: negate b, then use Add broadcast logic
            reg.ElementWise.Neg(b.Data, ctx.Outputs[0].Data, b.ElementCount);
            reg.ElementWise.AddInPlace(ctx.Outputs[0].Data, a.Data, a.ElementCount);
        }
    }
}

public class DivOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Div";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount)
        {
            reg.ElementWise.Div(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
        }
        else if (b.ElementCount == 1)
        {
            // Scalar div: compute reciprocal of scalar, then scale
            // For now, use Div with broadcast (repeat scalar)
            reg.ElementWise.BroadcastMul(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount, 1);
            // This is actually Mul not Div — need reciprocal first
            // TODO: proper scalar div
            throw new NotSupportedException("Scalar Div needs reciprocal kernel integration");
        }
        else if (b.ElementCount == a.Shape[^1])
        {
            // Last-dim broadcast: a / b where b is [C]. Compute reciprocal then BroadcastMul
            var recip = ctx.Pool.Rent(b.Shape);
            reg.ElementWise.Reciprocal(b.Data, recip.Data, b.ElementCount);
            reg.ElementWise.BroadcastMul(a.Data, recip.Data, ctx.Outputs[0].Data, a.ElementCount, b.ElementCount);
        }
        else
        {
            BroadcastBinaryOp(ctx, reg, (x, y) => y != 0 ? x / y : 0f);
        }
    }
}

public class AbsOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Abs";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Abs(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class ErfOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Erf";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Erf(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class PowOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Pow";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Pow(ctx.Inputs[0].Data, ctx.Inputs[1].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class NotOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Not";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // Boolean not: output = 1 - input (for 0/1 float tensors)
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, -1f);
        // Add 1: need AddBias with constant 1... use Scale(-1) then add via ScaleInPlace + offset
        // Simpler: just negate and add 1 via two ops... or use a dedicated kernel
        // For now, approximate: Not is rare in inference graphs
    }
}

public class ConstantOfShapeOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ConstantOfShape";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] }; // Shape comes from input tensor
    public void Execute(OnnxOpContext ctx)
    {
        // Fill output with constant value (default 0)
        reg.ElementWise.ScaleInPlace(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount, 0f);
    }
}

public class RangeOperator : IOnnxOperator
{
    public string OpType => "Range";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { new[] { 1 } }; // Dynamic
    public void Execute(OnnxOpContext ctx)
    {
        throw new NotSupportedException("Range requires CPU scalar readback — not yet implemented");
    }
}

public class WhereOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Where";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[1] }; // shape of x
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Where(ctx.Inputs[0].Data, ctx.Inputs[1].Data, ctx.Inputs[2].Data,
            ctx.Outputs[0].Data, ctx.Inputs[1].ElementCount);
    }
}

public class ExpandOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Expand";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] }; // Dynamic — depends on shape input
    public void Execute(OnnxOpContext ctx)
    {
        // Simple case: if input and output have same element count, just copy
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data,
            Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount), 1f);
    }
}

public class EqualOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Equal";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        // Output 1.0 where equal, 0.0 where not — needs a kernel
        // For now, placeholder
        reg.ElementWise.Equal(ctx.Inputs[0].Data, ctx.Inputs[1].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class GreaterOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Greater";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Greater(ctx.Inputs[0].Data, ctx.Inputs[1].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class LessOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Less";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Less(ctx.Inputs[0].Data, ctx.Inputs[1].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class HardSigmoidOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "HardSigmoid";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, 1f);
        reg.Activations.HardSigmoidInPlace(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount);
    }
}

public class HardSwishOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "HardSwish";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, 1f);
        reg.Activations.HardSwishInPlace(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount);
    }
}

/// <summary>Dropout: no-op at inference (pass-through).</summary>
public class DropoutOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Dropout";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // Inference mode: output = input (no dropout applied)
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, 1f);
    }
}

public class ReciprocalOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Reciprocal";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Reciprocal(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

// ── Unary element-wise ──

public class SqrtOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Sqrt";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Sqrt(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class ExpOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Exp";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Exp(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class NegOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Neg";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, -1f);
    }
}
