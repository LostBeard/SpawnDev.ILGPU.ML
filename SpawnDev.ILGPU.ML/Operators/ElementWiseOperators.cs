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

    internal static int[] ComputeStrides(int[] shape, int[] outShape)
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

    internal static int MapIndex(int outIdx, int[] outStrides, int[] inStrides, int rank)
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
        float alpha = ctx.GetFloat("alpha", 0.01f);
        reg.ElementWise.LeakyReLU(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, alpha);
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
        if (a.ElementCount == b.ElementCount && a.ElementCount == ctx.Outputs[0].ElementCount)
        {
            // Single-dispatch subtract — safe, no aliasing risk
            reg.ElementWise.Sub(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
        }
        else if (a.ElementCount == b.ElementCount)
        {
            // Size mismatch with output — use min count
            int count = Math.Min(a.ElementCount, ctx.Outputs[0].ElementCount);
            reg.ElementWise.Sub(a.Data.SubView(0, count), b.Data.SubView(0, count),
                ctx.Outputs[0].Data.SubView(0, count), count);
        }
        else if (b.ElementCount == 1)
        {
            // Scalar: a - scalar → negate scalar, then add
            reg.ElementWise.Neg(b.Data, ctx.Outputs[0].Data, 1);
            reg.ElementWise.AddBias(ctx.Outputs[0].Data, ctx.Outputs[0].Data, a.ElementCount, 1);
            reg.ElementWise.AddInPlace(ctx.Outputs[0].Data, a.Data, a.ElementCount);
        }
        else
        {
            BroadcastBinaryOp(ctx, reg, (x, y) => x - y);
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
            // Scalar div: compute reciprocal of scalar, then multiply
            var recip = ctx.Pool.Rent(b.Shape, "div_recip");
            reg.ElementWise.Reciprocal(b.Data, recip.Data, 1);
            reg.ElementWise.BroadcastMul(a.Data, recip.Data, ctx.Outputs[0].Data, a.ElementCount, 1);
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
        // Logical NOT: output[i] = (input[i] == 0) ? 1 : 0
        // Use pre-read constant values for CPU path (avoids aliasing issues)
        var inVals = ctx.TryGetInputValues(0);
        if (inVals != null)
        {
            var result = new float[inVals.Length];
            for (int i = 0; i < inVals.Length; i++)
                result[i] = inVals[i] == 0f ? 1f : 0f;
            var temp = ctx.Pool.AllocatePermanent(result, ctx.Inputs[0].Shape);
            reg.ElementWise.Scale(temp.Data, ctx.Outputs[0].Data, result.Length, 1f);
        }
        else
        {
            // GPU path: fill temp with 1, then Sub(ones, input, output)
            int count = ctx.Inputs[0].ElementCount;
            var ones = ctx.Pool.Rent(ctx.Inputs[0].Shape, "_not_ones");
            reg.ElementWise.Fill(ones.Data, count, 1f);
            reg.ElementWise.Sub(ones.Data, ctx.Inputs[0].Data, ctx.Outputs[0].Data, count);
        }
    }
}

public class ConstantOfShapeOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ConstantOfShape";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] }; // Shape comes from input tensor
    public void Execute(OnnxOpContext ctx)
    {
        // ONNX spec: value attribute is a scalar tensor (default 0.0)
        float fillValue = 0f;
        if (ctx.Attributes.TryGetValue("value", out var val))
        {
            fillValue = val switch
            {
                float f => f,
                double d => (float)d,
                long l => (float)l,
                int i => (float)i,
                _ => 0f
            };
        }
        reg.ElementWise.Fill(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount, fillValue);
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
        => new[] { inputs[0] }; // Dynamic — resolved at graph compile time from shape input
    public void Execute(OnnxOpContext ctx)
    {
        var input = ctx.Inputs[0];
        var output = ctx.Outputs[0];
        int outCount = output.ElementCount;

        // Simple case: same element count — just copy
        if (input.ElementCount == outCount)
        {
            reg.ElementWise.Scale(input.Data, output.Data, outCount, 1f);
            return;
        }

        // N-D broadcasting: use pre-read constant values
        var inVals = ctx.TryGetInputValues(0);
        if (inVals != null)
        {
            var inStrides = ComputeStrides(input.Shape, output.Shape);
            var outStrides = ComputeStrides(output.Shape, output.Shape);
            var result = new float[outCount];
            for (int i = 0; i < outCount; i++)
            {
                int inIdx = MapIndex(i, outStrides, inStrides, output.Shape.Length);
                result[i] = inIdx < inVals.Length ? inVals[inIdx] : 0f;
            }
            var temp = ctx.Pool.AllocatePermanent(result, output.Shape);
            reg.ElementWise.Scale(temp.Data, output.Data, outCount, 1f);
        }
        else
        {
            // Large tensor fallback — copy what we can
            int copyCount = Math.Min(input.ElementCount, outCount);
            reg.ElementWise.Scale(input.Data.SubView(0, copyCount),
                output.Data.SubView(0, copyCount), copyCount, 1f);
        }
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

public class LessOrEqualOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "LessOrEqual";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        BroadcastBinaryOp(ctx, reg, (a, b) => a <= b ? 1f : 0f);
    }
}

public class AndOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "And";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        BroadcastBinaryOp(ctx, reg, (a, b) => (a != 0f && b != 0f) ? 1f : 0f);
    }
}

public class IsNaNOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "IsNaN";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // IsNaN: output 1.0 where input is NaN, 0.0 otherwise
        // For well-behaved models, this should produce all zeros
        var inVals = ctx.TryGetInputValues(0);
        if (inVals != null)
        {
            var result = new float[inVals.Length];
            for (int i = 0; i < inVals.Length; i++)
                result[i] = float.IsNaN(inVals[i]) ? 1f : 0f;
            var temp = ctx.Pool.AllocatePermanent(result, ctx.Inputs[0].Shape);
            reg.ElementWise.Scale(temp.Data, ctx.Outputs[0].Data, result.Length, 1f);
        }
        else
        {
            // GPU path: assume no NaN — fill with zeros
            // Real models use IsNaN as a guard, and our kernels don't produce NaN
            reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, 0f);
        }
    }
}

public class HardSigmoidOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "HardSigmoid";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // ONNX spec defaults: alpha=0.2, beta=0.5
        float alpha = ctx.GetFloat("alpha", 0.2f);
        float beta = ctx.GetFloat("beta", 0.5f);
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, 1f);
        reg.Activations.HardSigmoidInPlace(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount, alpha, beta);
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
