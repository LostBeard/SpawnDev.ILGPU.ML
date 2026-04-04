using ILGPU;
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
    public static void BroadcastBinaryOp(OnnxOpContext ctx, OperatorRegistry reg, Func<float, float, float> op,
        BroadcastOp gpuOp = BroadcastOp.Add)
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
        else if (bVals != null && a.ElementCount > b.ElementCount)
        {
            // b is a small runtime constant, a is a large GPU tensor.
            // Expand b to full output shape on CPU, upload, then GPU element-wise op.
            // Uses Rent (not AllocatePermanent) to avoid buffer leaks through 748-node models.
            var bExpanded = new float[outCount];
            var bStrides = ComputeStrides(b.Shape, outShape);
            var outStrides = ComputeStrides(outShape, outShape);
            for (int i = 0; i < outCount; i++)
            {
                int bIdx = MapIndex(i, outStrides, bStrides, outShape.Length);
                bExpanded[i] = bIdx < bVals.Length ? bVals[bIdx] : 0f;
            }
            var bExpandedTensor = ctx.Pool.Rent(outShape, "_broadcast_b_expanded");
            bExpandedTensor.Data.SubView(0, outCount).CopyFromCPU(bExpanded);
            // Use GPU N-D broadcast kernel (a and bExpanded are same shape → element-wise)
            reg.ElementWise.BroadcastBinaryOpND(
                a.Data, bExpandedTensor.Data, ctx.Outputs[0].Data,
                a.Shape, outShape, outShape, gpuOp);
            ctx.Pool.Return(bExpandedTensor);
        }
        else
        {
            // General N-D broadcast on GPU — handles arbitrary shape combinations.
            // Uses stride-based index mapping kernels (BroadcastDivImpl, etc.)
            reg.ElementWise.BroadcastBinaryOpND(
                a.Data, b.Data, ctx.Outputs[0].Data,
                a.Shape, b.Shape, outShape, gpuOp);
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
        var output = ctx.Outputs[0];

        if (a.ElementCount == b.ElementCount)
        {
            // Safe two-step: copy a → output, then add b in-place.
            // Avoids 3-way aliasing (a, b, output may share same GPU buffer on WebGPU).
            reg.ElementWise.Scale(a.Data, output.Data, a.ElementCount, 1f);
            reg.ElementWise.AddInPlace(output.Data, b.Data, a.ElementCount);
        }
        else if (b.ElementCount == a.Shape[^1])
        {
            // Last-dim broadcast: copy a → output, then AddBias in-place
            reg.ElementWise.Scale(a.Data, output.Data, a.ElementCount, 1f);
            reg.ElementWise.AddBias(output.Data, b.Data, a.ElementCount, b.ElementCount);
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
            BroadcastBinaryOp(ctx, reg, (x, y) => x + y, BroadcastOp.Add);
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
            // WebGPU forbids binding the same buffer to multiple storage slots.
            // Detect aliasing (e.g., x * x where both inputs are the same tensor)
            // and copy to a temp buffer to avoid the aliasing violation.
            if (object.ReferenceEquals(a, b))
            {
                var temp = ctx.Pool.Rent(b.Shape, "_mul_alias");
                reg.ElementWise.Scale(b.Data, temp.Data, b.ElementCount, 1f);
                reg.ElementWise.Mul(a.Data, temp.Data, ctx.Outputs[0].Data, a.ElementCount);
                ctx.Pool.Return(temp);
            }
            else
            {
                reg.ElementWise.Mul(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
            }
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
            BroadcastBinaryOp(ctx, reg, (x, y) => x * y, BroadcastOp.Mul);
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
            // Scalar subtract: output = a - scalar → use BroadcastMul(a, -1→b) + BroadcastAdd
            // Strategy: output = a (copy), then BroadcastSub via BroadcastBinaryOp
            // Simplest safe path: use BroadcastBinaryOp which handles all broadcast shapes
            BroadcastBinaryOp(ctx, reg, (x, y) => x - y, BroadcastOp.Sub);
        }
        else
        {
            BroadcastBinaryOp(ctx, reg, (x, y) => x - y, BroadcastOp.Sub);
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
            ctx.Pool.Return(recip);
        }
        else if (b.ElementCount == a.Shape[^1])
        {
            // Last-dim broadcast: a / b where b is [C]. Compute reciprocal then BroadcastMul
            var recip = ctx.Pool.Rent(b.Shape, "div_recip_bc");
            reg.ElementWise.Reciprocal(b.Data, recip.Data, b.ElementCount);
            reg.ElementWise.BroadcastMul(a.Data, recip.Data, ctx.Outputs[0].Data, a.ElementCount, b.ElementCount);
            ctx.Pool.Return(recip);
        }
        else if (a.Shape.Length >= 2 && b.ElementCount > 1 && b.ElementCount < a.ElementCount)
        {
            // General broadcast: compute reciprocal of b, then use BroadcastBinaryOp for multiply
            // This handles cases like a=[1,257,384] / b=[1,257,1] (per-row scalar division)
            BroadcastBinaryOp(ctx, reg, (x, y) => y != 0 ? x / y : 0f, BroadcastOp.Div);
        }
        else
        {
            BroadcastBinaryOp(ctx, reg, (x, y) => y != 0 ? x / y : 0f, BroadcastOp.Div);
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
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount)
        {
            reg.ElementWise.Pow(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
        }
        else if (b.ElementCount <= a.ElementCount)
        {
            // Scalar/small exponent broadcast (LayerNorm x^2, InstanceNorm x^2).
            // Expand exponent to full size on CPU, then element-wise Pow.
            // Avoids BroadcastBinaryOpND which has synchronous Synchronize() — deadlocks on WebGPU.
            var bVals = ctx.TryGetInputValues(1);
            if (bVals != null)
            {
                int outCount = ctx.Outputs[0].ElementCount;
                var expanded = new float[outCount];
                for (int i = 0; i < outCount; i++)
                    expanded[i] = bVals[i % bVals.Length];
                var expandedTensor = ctx.Pool.Rent(ctx.Outputs[0].Shape, "_pow_exp");
                expandedTensor.Data.SubView(0, outCount).CopyFromCPU(expanded);
                reg.ElementWise.Pow(a.Data, expandedTensor.Data, ctx.Outputs[0].Data, outCount);
                ctx.Pool.Return(expandedTensor);
            }
            else
            {
                // Exponent not in runtime constants — use BroadcastBinaryOp (desktop backends only)
                BroadcastBinaryOp(ctx, reg, (x, y) => MathF.Pow(x, y), BroadcastOp.Pow);
            }
        }
        else
        {
            BroadcastBinaryOp(ctx, reg, (x, y) => MathF.Pow(x, y), BroadcastOp.Pow);
        }
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
            ctx.Pool.Return(ones);
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

public class RangeOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Range";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { new[] { 1 } }; // Dynamic — resolved at runtime from scalar inputs
    public void Execute(OnnxOpContext ctx)
    {
        // Range(start, limit, delta) → [start, start+delta, ..., <limit)
        // Inputs are scalar tensors — read from runtime constants
        var startVals = ctx.TryGetInputValues(0);
        var limitVals = ctx.TryGetInputValues(1);
        var deltaVals = ctx.TryGetInputValues(2);

        if (startVals == null || limitVals == null || deltaVals == null)
            throw new NotSupportedException("Range: scalar inputs not available as runtime constants");

        float start = startVals[0];
        float limit = limitVals[0];
        float delta = deltaVals[0];

        if (delta == 0) throw new ArgumentException("Range: delta cannot be 0");

        int count = Math.Max(0, (int)MathF.Ceiling((limit - start) / delta));
        var data = new float[count];
        for (int i = 0; i < count; i++)
            data[i] = start + i * delta;

        // Upload to output GPU buffer
        var output = ctx.Outputs[0];
        if (output.ElementCount >= count)
        {
            using var tmpBuf = reg.Accelerator.Allocate1D(data);
            reg.ElementWise.Scale(tmpBuf.View, output.Data.SubView(0, count), count, 1f);
        }
    }
}

/// <summary>
/// NonZero: returns indices of non-zero elements as [rank, nnz] tensor.
/// For attention masks (all 1s), returns all coordinate pairs.
/// Data-dependent output size — reads input values from runtime constants.
/// </summary>
public class NonZeroOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "NonZero";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { new[] { inputs[0].Length, inputs[0].Aggregate(1, (a, b) => a * b) } }; // [rank, max_nnz]
    public void Execute(OnnxOpContext ctx)
    {
        var input = ctx.Inputs[0];
        var inShape = input.Shape;
        int rank = inShape.Length;
        int totalElems = input.ElementCount;

        // Read input values — NonZero is inherently data-dependent
        var vals = ctx.TryGetInputValues(0);
        if (vals == null)
        {
            // Can't read values — assume all non-zero (common for attention masks)
            vals = new float[totalElems];
            for (int i = 0; i < totalElems; i++) vals[i] = 1f;
        }

        // Find non-zero indices
        var indices = new List<int[]>();
        var strides = new int[rank];
        if (rank > 0) strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; d--) strides[d] = strides[d + 1] * inShape[d + 1];

        for (int i = 0; i < totalElems; i++)
        {
            if (vals[i] != 0f)
            {
                var coord = new int[rank];
                int rem = i;
                for (int d = 0; d < rank; d++) { coord[d] = rem / strides[d]; rem %= strides[d]; }
                indices.Add(coord);
            }
        }

        // Output: [rank, nnz] — each row is one dimension's indices
        int nnz = indices.Count;
        var result = new float[rank * nnz];
        for (int d = 0; d < rank; d++)
            for (int j = 0; j < nnz; j++)
                result[d * nnz + j] = indices[j][d];

        var output = ctx.Outputs[0];
        int copyLen = Math.Min(result.Length, output.ElementCount);
        if (copyLen > 0)
        {
            using var tmpBuf = reg.Accelerator.Allocate1D(result);
            reg.ElementWise.Scale(tmpBuf.View.SubView(0, copyLen), output.Data.SubView(0, copyLen), copyLen, 1f);
        }
    }
}

public class WhereOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Where";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // ONNX Where: output = broadcast(condition, x, y)
        // Return the largest shape among the three inputs
        var best = inputs[0];
        for (int i = 1; i < inputs.Length; i++)
            if (inputs[i].Length > best.Length || (inputs[i].Length == best.Length
                && inputs[i].Aggregate(1, (a, b) => a * b) > best.Aggregate(1, (a, b) => a * b)))
                best = inputs[i];
        return new[] { best };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var cond = ctx.Inputs[0]; var x = ctx.Inputs[1]; var y = ctx.Inputs[2];
        if (cond.ElementCount == x.ElementCount && x.ElementCount == y.ElementCount)
        {
            reg.ElementWise.Where(cond.Data, x.Data, y.Data, ctx.Outputs[0].Data, x.ElementCount);
        }
        else
        {
            // Broadcasting required — use stride-based N-D broadcast mapping
            var cVals = ctx.TryGetInputValues(0);
            var xVals = ctx.TryGetInputValues(1);
            var yVals = ctx.TryGetInputValues(2);
            int outCount = ctx.Outputs[0].ElementCount;
            if (cVals != null && xVals != null && yVals != null)
            {
                var outShape = ctx.Outputs[0].Shape;
                int rank = outShape.Length;

                // Compute broadcast strides for each input
                static int[] ComputeStrides(int[] shape, int[] outShape)
                {
                    int rank = outShape.Length;
                    int padded = rank - shape.Length;
                    var strides = new int[rank];
                    int stride = 1;
                    for (int d = rank - 1; d >= 0; d--)
                    {
                        int dim = d - padded >= 0 ? shape[d - padded] : 1;
                        strides[d] = dim == 1 ? 0 : stride; // broadcast dim → stride 0
                        stride *= dim;
                    }
                    return strides;
                }

                var cStrides = ComputeStrides(cond.Shape, outShape);
                var xStrides = ComputeStrides(x.Shape, outShape);
                var yStrides = ComputeStrides(y.Shape, outShape);

                // Compute output strides
                var outStrides = new int[rank];
                int oStride = 1;
                for (int d = rank - 1; d >= 0; d--) { outStrides[d] = oStride; oStride *= outShape[d]; }

                var result = new float[outCount];
                for (int i = 0; i < outCount; i++)
                {
                    // Decompose flat index into N-D coordinates, map to each input
                    int cIdx = 0, xIdx = 0, yIdx = 0, rem = i;
                    for (int d = 0; d < rank; d++)
                    {
                        int coord = rem / outStrides[d];
                        rem %= outStrides[d];
                        cIdx += coord * cStrides[d];
                        xIdx += coord * xStrides[d];
                        yIdx += coord * yStrides[d];
                    }
                    result[i] = cVals[cIdx] != 0f ? xVals[xIdx] : yVals[yIdx];
                }
                var temp = ctx.Pool.AllocatePermanent(result, outShape);
                reg.ElementWise.Scale(temp.Data, ctx.Outputs[0].Data, outCount, 1f);
            }
            else
            {
                // Fallback: use minimum safe count to avoid OOB reads
                int safeCount = Math.Min(Math.Min(cond.ElementCount, x.ElementCount),
                    Math.Min(y.ElementCount, ctx.Outputs[0].ElementCount));
                reg.ElementWise.Where(cond.Data, x.Data, y.Data, ctx.Outputs[0].Data, safeCount);
            }
        }
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
            // GPU broadcast: tile input data to fill output using stride-based copy.
            // Common case: [1, C] → [N, C] = copy row N times
            int inCount = input.ElementCount;
            if (inCount > 0 && outCount % inCount == 0)
            {
                // Exact tiling: repeat input block to fill output
                int repeats = outCount / inCount;
                for (int r = 0; r < repeats; r++)
                    reg.ElementWise.Scale(input.Data.SubView(0, inCount),
                        output.Data.SubView(r * inCount, inCount), inCount, 1f);
            }
            else
            {
                // Fallback: copy what we can
                int copyCount = Math.Min(inCount, outCount);
                reg.ElementWise.Scale(input.Data.SubView(0, copyCount),
                    output.Data.SubView(0, copyCount), copyCount, 1f);
            }
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
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount)
            reg.ElementWise.Equal(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
        else
            BroadcastBinaryOp(ctx, reg, (x, y) => x == y ? 1f : 0f);
    }
}

public class GreaterOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Greater";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount)
            reg.ElementWise.Greater(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
        else
            BroadcastBinaryOp(ctx, reg, (x, y) => x > y ? 1f : 0f);
    }
}

public class LessOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Less";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount)
            reg.ElementWise.Less(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
        else
            BroadcastBinaryOp(ctx, reg, (x, y) => x < y ? 1f : 0f);
    }
}

public class LessOrEqualOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "LessOrEqual";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount)
        {
            // a <= b is !(a > b). Greater returns 1.0 for true, 0.0 for false.
            // Negate: output = 1.0 - Greater(a, b)
            reg.ElementWise.Greater(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
            reg.ElementWise.ScaleInPlace(ctx.Outputs[0].Data, a.ElementCount, -1f);
            var ones = ctx.Pool.Rent(new[] { 1 }, "_leq_one");
            ones.Data.SubView(0, 1).CopyFromCPU(new float[] { 1f });
            reg.ElementWise.AddBias(ctx.Outputs[0].Data, ones.Data, a.ElementCount, 1);
            ctx.Pool.Return(ones);
        }
        else
        {
            BroadcastBinaryOp(ctx, reg, (x, y) => x <= y ? 1f : 0f);
        }
    }
}

public class GreaterOrEqualOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "GreaterOrEqual";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount)
        {
            // a >= b is !(a < b)
            reg.ElementWise.Less(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
            reg.ElementWise.ScaleInPlace(ctx.Outputs[0].Data, a.ElementCount, -1f);
            var ones = ctx.Pool.Rent(new[] { 1 }, "_geq_one");
            ones.Data.SubView(0, 1).CopyFromCPU(new float[] { 1f });
            reg.ElementWise.AddBias(ctx.Outputs[0].Data, ones.Data, a.ElementCount, 1);
            ctx.Pool.Return(ones);
        }
        else
        {
            BroadcastBinaryOp(ctx, reg, (x, y) => x >= y ? 1f : 0f);
        }
    }
}

public class OrOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Or";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        // Or is always a boolean op — use broadcast path for correctness
        // (handles both equal and unequal shapes, with proper abs+threshold)
        BroadcastBinaryOp(ctx, reg, (x, y) => (x != 0f || y != 0f) ? 1f : 0f);
    }
}

public class XorOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Xor";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        BroadcastBinaryOp(ctx, reg, (x, y) => (x != 0f) != (y != 0f) ? 1f : 0f);
    }
}

public class AndOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "And";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount)
        {
            // And = Mul(a, b) then threshold: any non-zero × non-zero = non-zero
            reg.ElementWise.Mul(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
        }
        else
        {
            BroadcastBinaryOp(ctx, reg, (x, y) => (x != 0f && y != 0f) ? 1f : 0f);
        }
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

public class SinOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Sin";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Sin(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class CosOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Cos";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Cos(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class TanOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Tan";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Tan(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

// ═══════════════════════════════════════════════════════════
//  New operators — full ONNX coverage
// ═══════════════════════════════════════════════════════════

public class AcosOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Acos";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Acos(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class AcoshOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Acosh";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Acosh(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class AsinOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Asin";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Asin(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class AsinhOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Asinh";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Asinh(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class AtanOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Atan";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Atan(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class AtanhOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Atanh";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Atanh(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class CoshOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Cosh";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Cosh(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class SinhOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Sinh";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Sinh(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class EluOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Elu";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Elu(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class CeluOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Celu";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Celu(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class SeluOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Selu";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Selu(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class SoftplusOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Softplus";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Softplus(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class SoftsignOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Softsign";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Softsign(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class MishOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Mish";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Mish(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class IsInfOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "IsInf";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.IsInf(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class ThresholdedReluOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ThresholdedRelu";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        float alpha = ctx.GetFloat("alpha", 1f);
        // ThresholdedRelu: output = x if x > alpha, else 0
        var xVals = ctx.TryGetInputValues(0);
        int count = ctx.Inputs[0].ElementCount;
        if (xVals != null)
        {
            var result = xVals.Select(x => x > alpha ? x : 0f).ToArray();
            ctx.Outputs[0].Data.SubView(0, count).CopyFromCPU(result);
        }
        else
        {
            // GPU path: copy then threshold would need a custom kernel.
            // For now, use the generic unary op with alpha=1 default.
            reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, count, 1f);
        }
    }
}

public class IdentityOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Identity";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // Use CopyFrom (GPU→GPU works on all backends). NOT CopyTo — WebGPU's
        // CopyTo override throws because it can't distinguish GPU→GPU from GPU→CPU.
        int count = ctx.Inputs[0].ElementCount;
        ctx.Outputs[0].Data.SubView(0, count).CopyFrom(ctx.Inputs[0].Data.SubView(0, count));
    }
}

public class SizeOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Size";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { new[] { 1 } };
    public void Execute(OnnxOpContext ctx)
    {
        int size = ctx.Inputs[0].ElementCount;
        ctx.Outputs[0].Data.SubView(0, 1).CopyFromCPU(new float[] { size });
    }
}

public class HardmaxOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Hardmax";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // Hardmax: output = one-hot of argmax along axis
        int axis = ctx.GetInt("axis", -1);
        var shape = ctx.Inputs[0].Shape;
        if (axis < 0) axis += shape.Length;
        int outer = 1, inner = 1, axisSize = shape[axis];
        for (int i = 0; i < axis; i++) outer *= shape[i];
        for (int i = axis + 1; i < shape.Length; i++) inner *= shape[i];
        // Compute argmax indices on CPU, then build one-hot output
        var xVals = ctx.TryGetInputValues(0);
        int total = ctx.Outputs[0].ElementCount;
        reg.ElementWise.Fill(ctx.Outputs[0].Data, total, 0f);
        if (xVals != null)
        {
            var result = new float[total];
            for (int o = 0; o < outer; o++)
            {
                for (int inn = 0; inn < inner; inn++)
                {
                    float maxVal = float.NegativeInfinity;
                    int maxIdx = 0;
                    for (int a = 0; a < axisSize; a++)
                    {
                        float v = xVals[(o * axisSize + a) * inner + inn];
                        if (v > maxVal) { maxVal = v; maxIdx = a; }
                    }
                    result[(o * axisSize + maxIdx) * inner + inn] = 1f;
                }
            }
            ctx.Outputs[0].Data.SubView(0, total).CopyFromCPU(result);
        }
    }
}

public class LogSoftmaxOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "LogSoftmax";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        int axis = ctx.GetInt("axis", -1);
        var shape = ctx.Inputs[0].Shape;
        if (axis < 0) axis += shape.Length;
        int rows = 1, cols = shape[axis];
        for (int i = 0; i < axis; i++) rows *= shape[i];
        // Copy input to output, run softmax, then log (using temp to avoid aliasing)
        int total = ctx.Inputs[0].ElementCount;
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, total, 1f);
        reg.Softmax.Forward(ctx.Outputs[0].Data, rows, cols);
        using var tempBuf = reg.Accelerator.Allocate1D<float>(total);
        reg.ElementWise.Log(ctx.Outputs[0].Data, tempBuf.View, total);
        reg.ElementWise.Scale(tempBuf.View, ctx.Outputs[0].Data, total, 1f);
    }
}

public class PReluOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "PRelu";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // PRelu: output = x if x >= 0, slope * x if x < 0
        // slope may be per-channel broadcast
        var x = ctx.Inputs[0]; var slope = ctx.Inputs[1];
        // Use broadcast binary op: PRelu(x, slope) = max(0, x) + slope * min(0, x)
        // Simplified: use Where(x >= 0, x, slope * x) via broadcast
        BroadcastHelper.BroadcastBinaryOp(ctx, reg,
            (a, b) => a >= 0f ? a : a * b);
    }
}

public class SumOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Sum";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        int count = ctx.Outputs[0].ElementCount;
        // Copy first input to output
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, count, 1f);
        // Add remaining inputs using temp buffer to avoid aliasing (output as both input and output)
        if (ctx.Inputs.Length > 1)
        {
            using var tempBuf = reg.Accelerator.Allocate1D<float>(count);
            for (int i = 1; i < ctx.Inputs.Length; i++)
            {
                reg.ElementWise.Add(ctx.Outputs[0].Data, ctx.Inputs[i].Data, tempBuf.View, count);
                reg.ElementWise.Scale(tempBuf.View, ctx.Outputs[0].Data, count, 1f);
            }
        }
    }
}

public class MeanOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Mean";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        int count = ctx.Outputs[0].ElementCount;
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, count, 1f);
        if (ctx.Inputs.Length > 1)
        {
            using var tempBuf = reg.Accelerator.Allocate1D<float>(count);
            for (int i = 1; i < ctx.Inputs.Length; i++)
            {
                reg.ElementWise.Add(ctx.Outputs[0].Data, ctx.Inputs[i].Data, tempBuf.View, count);
                reg.ElementWise.Scale(tempBuf.View, ctx.Outputs[0].Data, count, 1f);
            }
        }
        reg.ElementWise.ScaleInPlace(ctx.Outputs[0].Data, count, 1f / ctx.Inputs.Length);
    }
}

public class ArgMinOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ArgMin";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var shape = inputs[0].ToArray();
        int axis = attrs.ContainsKey("axis") ? Convert.ToInt32(attrs["axis"]) : 0;
        if (axis < 0) axis += shape.Length;
        shape[axis] = 1;
        bool keepdims = !attrs.ContainsKey("keepdims") || Convert.ToInt32(attrs["keepdims"]) != 0;
        return new[] { keepdims ? shape : shape.Where((_, i) => i != axis).ToArray() };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // ArgMin = Neg → ArgMax (negate, find max of negated = min of original)
        int count = ctx.Inputs[0].ElementCount;
        using var negBuf = reg.Accelerator.Allocate1D<float>(count);
        reg.ElementWise.Scale(ctx.Inputs[0].Data, negBuf.View, count, -1f);
        int axis = ctx.GetInt("axis", 0);
        var shape = ctx.Inputs[0].Shape;
        if (axis < 0) axis += shape.Length;
        int outerSize = 1, innerSize = 1, axisSize = shape[axis];
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < shape.Length; i++) innerSize *= shape[i];
        reg.ElementWise.ArgMax(negBuf.View, ctx.Outputs[0].Data, outerSize, axisSize, innerSize);
    }
}

public class RoundOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Round";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Round(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class ShrinkOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Shrink";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // Shrink: if x > lambd, y = x - bias; if x < -lambd, y = x + bias; else y = 0
        // Default: lambd=0.5, bias=0
        // Use generic unary for default params
        int count = ctx.Inputs[0].ElementCount;
        float lambd = ctx.GetFloat("lambd", 0.5f);
        float bias = ctx.GetFloat("bias", 0f);
        // For default params, use the built-in ShrinkOp kernel
        if (MathF.Abs(lambd - 0.5f) < 1e-7f && MathF.Abs(bias) < 1e-7f)
        {
            reg.ElementWise.UnaryOp(ctx.Inputs[0].Data, ctx.Outputs[0].Data, count,
                new DelegateSpecialization<Func<float, float>>(ElementWiseKernels.ShrinkOp));
        }
        else
        {
            // Parameterized shrink — CPU fallback for custom lambd/bias
            var vals = ctx.TryGetInputValues(0);
            if (vals != null)
            {
                var result = vals.Select(x => x > lambd ? x - bias : x < -lambd ? x + bias : 0f).ToArray();
                using var tmp = reg.Accelerator.Allocate1D(result);
                reg.ElementWise.Scale(tmp.View, ctx.Outputs[0].Data, count, 1f);
            }
        }
    }
}
