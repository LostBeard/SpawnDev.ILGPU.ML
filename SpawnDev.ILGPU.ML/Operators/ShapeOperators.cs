using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using static SpawnDev.ILGPU.ML.Operators.BroadcastHelper;

namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// Zero-copy shape manipulation operators.
/// When input and output share the same buffer, no GPU work is needed.
/// When they don't (graph executor allocated a separate output), we do a trivial copy.
/// </summary>

public class ReshapeOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Reshape";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] }; // Resolved from shape tensor at graph compile time
    public void Execute(OnnxOpContext ctx)
    {
        // Copy input to output
        int count = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount);
        if (count > 0)
            reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, count),
                ctx.Outputs[0].Data.SubView(0, count), count, 1f);

        // If target shape is available in runtime constants, apply it to the output tensor.
        // This fixes the case where the pre-allocated output buffer has the wrong compiled shape
        // but the runtime Reshape target (from Concat of Shape values) is correct.
        var targetShape = ctx.TryGetInputValues(1);
        if (targetShape != null && targetShape.Length > 0)
        {
            var resolved = targetShape.Select(v => (int)v).ToArray();
            int inputElems = ctx.Inputs[0].ElementCount;
            // Handle -1 (infer) and 0 (copy from input)
            for (int j = 0; j < resolved.Length; j++)
                if (resolved[j] == 0 && j < ctx.Inputs[0].Shape.Length) resolved[j] = ctx.Inputs[0].Shape[j];
            int negIdx = Array.IndexOf(resolved, -1);
            if (negIdx >= 0)
            {
                int known = 1;
                for (int j = 0; j < resolved.Length; j++) if (j != negIdx && resolved[j] > 0) known *= resolved[j];
                resolved[negIdx] = known > 0 ? inputElems / known : 1;
            }
            // Validate element count matches
            if (resolved.All(d => d > 0) && resolved.Aggregate(1L, (a, b) => a * b) == inputElems)
                ctx.Outputs[0].Shape = resolved;
        }
    }
}

public class UnsqueezeOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Unsqueeze";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Unsqueeze inserts size-1 dimensions at the specified axes
        if (attrs.TryGetValue("axes", out var axesObj) && axesObj is long[] axes)
        {
            var inShape = inputs[0];
            int outRank = inShape.Length + axes.Length;
            var outShape = new int[outRank];

            // Normalize negative axes
            var normalizedAxes = new HashSet<int>();
            foreach (var ax in axes)
            {
                int a = (int)ax;
                if (a < 0) a += outRank;
                normalizedAxes.Add(a);
            }

            int inIdx = 0;
            for (int i = 0; i < outRank; i++)
            {
                if (normalizedAxes.Contains(i))
                    outShape[i] = 1;
                else
                    outShape[i] = inIdx < inShape.Length ? inShape[inIdx++] : 1;
            }
            return new[] { outShape };
        }
        // If axes come from input[1] (opset >= 13), shape resolved at runtime
        return new[] { inputs[0] };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // Unsqueeze changes shape but data is identical — copy input to output
        int count = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount);
        if (count > 0)
            reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, count),
                ctx.Outputs[0].Data.SubView(0, count), count, 1f);
    }
}

public class SqueezeOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Squeeze";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Squeeze removes size-1 dimensions at the specified axes
        if (attrs.TryGetValue("axes", out var axesObj) && axesObj is long[] axes)
        {
            var inShape = inputs[0];
            var normalizedAxes = new HashSet<int>();
            foreach (var ax in axes)
            {
                int a = (int)ax;
                if (a < 0) a += inShape.Length;
                normalizedAxes.Add(a);
            }
            var outShape = new List<int>();
            for (int i = 0; i < inShape.Length; i++)
            {
                if (!normalizedAxes.Contains(i))
                    outShape.Add(inShape[i]);
            }
            return new[] { outShape.ToArray() };
        }
        // No axes specified: remove all size-1 dims
        return new[] { inputs[0].Where(d => d != 1).ToArray() };
    }
    public void Execute(OnnxOpContext ctx)
    {
        int count = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount);
        if (count > 0)
            reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, count),
                ctx.Outputs[0].Data.SubView(0, count), count, 1f);
    }
}

/// <summary>Constant: output is a constant value (stored as initializer). No-op at runtime.</summary>
public class ConstantOperator : IOnnxOperator
{
    public string OpType => "Constant";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { new[] { 1 } };
    public void Execute(OnnxOpContext ctx) { }
}

/// <summary>Cast: type conversion. For float→float, this is a copy.
/// For float→int, truncates toward zero (matching numpy/ONNX behavior).</summary>
public class CastOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Cast";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // ONNX 'to' attribute: 1=float, 6=int32, 7=int64, 9=bool, 11=double
        int targetType = ctx.GetInt("to", 1);
        int count = ctx.Inputs[0].ElementCount;

        // For small tensors (shape vectors, scalars), use pre-read constant values.
        // Shape tensors may only exist in ConstantData/runtime constants — their GPU buffers
        // can be empty if they were produced by folded/eliminated nodes.
        var inVals = ctx.TryGetInputValues(0);
        if (inVals != null && count <= 64)
        {
            float[] result;
            if (targetType == 6 || targetType == 7 || targetType == 12 || targetType == 5 || targetType == 3 || targetType == 2)
                result = inVals.Select(v => (float)Math.Truncate(v)).ToArray();
            else
                result = inVals.ToArray();
            var temp = ctx.Pool.AllocatePermanent(result, ctx.Outputs[0].Shape);
            reg.ElementWise.Scale(temp.Data, ctx.Outputs[0].Data, count, 1f);
            return;
        }

        if (targetType == 6 || targetType == 7 || targetType == 12 || targetType == 5 || targetType == 3 || targetType == 2)
        {
            // Cast to integer type: truncate toward zero (like C-style cast)
            reg.ElementWise.Truncate(ctx.Inputs[0].Data, ctx.Outputs[0].Data, count);
        }
        else
        {
            // Float-to-float or other: just copy
            reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, count, 1f);
        }
    }
}

public class FloorOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Floor";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Floor(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class CeilOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Ceil";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Ceil(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class LogOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Log";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Log(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class MinOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Min";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount)
            reg.ElementWise.Min(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
        else
            BroadcastBinaryOp(ctx, reg, (x, y) => MathF.Min(x, y));
    }
}

public class MaxOnnxOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Max";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount)
            reg.ElementWise.Max(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
        else
            BroadcastBinaryOp(ctx, reg, (x, y) => MathF.Max(x, y));
    }
}

/// <summary>Upsample: resize using nearest or linear mode. Scales from second input.</summary>
public class UpsampleOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Upsample";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] }; // Actual shape depends on scales tensor — resolved at runtime
    public void Execute(OnnxOpContext ctx)
    {
        var input = ctx.Inputs[0];
        var output = ctx.Outputs[0];
        string mode = ctx.GetString("mode", "nearest");

        // For nearest mode: each output element copies from the nearest input element
        // Output shape should already be allocated by the executor based on compiled shapes
        if (mode == "nearest")
        {
            // Use the Resize operator's nearest-neighbor implementation
            reg.ElementWise.NearestUpsample(input.Data, output.Data,
                input.Shape, output.Shape);
        }
        else
        {
            // Linear mode — use bilinear upsample
            int inH = input.Shape.Length >= 4 ? input.Shape[2] : input.Shape[0];
            int inW = input.Shape.Length >= 4 ? input.Shape[3] : input.Shape[1];
            int outH = output.Shape.Length >= 4 ? output.Shape[2] : output.Shape[0];
            int outW = output.Shape.Length >= 4 ? output.Shape[3] : output.Shape[1];
            int channels = input.ElementCount / (inH * inW);
            reg.ElementWise.BilinearUpsample(input.Data, output.Data,
                channels, inH, inW, outH, outW);
        }
    }
}

/// <summary>Shape: outputs the shape of the input as a 1D tensor of dimension values (stored as float).</summary>
public class ShapeOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Shape";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { new[] { inputs[0].Length } };
    public void Execute(OnnxOpContext ctx)
    {
        // Output the input's shape dimensions as float values via a temporary GPU buffer
        var shape = ctx.Inputs[0].Shape;
        var shapeData = new float[shape.Length];
        for (int i = 0; i < shape.Length; i++)
            shapeData[i] = shape[i];
        // Upload via pool
        var temp = ctx.Pool.AllocatePermanent(shapeData, new[] { shape.Length });
        reg.ElementWise.Scale(temp.Data, ctx.Outputs[0].Data, shape.Length, 1f);
    }
}

public class FlattenOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Flatten";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        int axis = attrs.ContainsKey("axis") ? Convert.ToInt32(attrs["axis"]) : 1;
        if (axis < 0) axis += inputs[0].Length;
        int dim0 = 1; for (int i = 0; i < axis; i++) dim0 *= inputs[0][i];
        int dim1 = 1; for (int i = axis; i < inputs[0].Length; i++) dim1 *= inputs[0][i];
        return new[] { new[] { dim0, dim1 } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // Flatten is just a reshape — data layout doesn't change, just copy input to output
        int count = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount);
        if (count > 0)
            reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, count),
                ctx.Outputs[0].Data.SubView(0, count), count, 1f);
    }
}

/// <summary>Tile: repeat a tensor along each dimension by specified counts.</summary>
public class TileOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Tile";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] }; // Dynamic — depends on repeats tensor
    public void Execute(OnnxOpContext ctx)
    {
        var input = ctx.Inputs[0];
        var output = ctx.Outputs[0];
        int inCount = input.ElementCount;
        int outCount = output.ElementCount;

        // Get repeat counts from runtime constants
        var repeats = ctx.TryGetInputValues(1);

        if (inCount == outCount)
        {
            // No tiling needed — just copy
            reg.ElementWise.Scale(input.Data.SubView(0, inCount), output.Data.SubView(0, outCount), inCount, 1f);
            return;
        }

        if (repeats != null)
        {
            // CPU-side tiling for small tensors or when repeats are known
            var inShape = input.Shape;
            var outShape = new int[inShape.Length];
            for (int i = 0; i < inShape.Length; i++)
                outShape[i] = inShape[i] * (i < repeats.Length ? (int)repeats[i] : 1);

            // Simple tiling: if output is an exact multiple of input, tile
            if (outCount > 0 && outCount % inCount == 0)
            {
                int tiles = outCount / inCount;
                for (int t = 0; t < tiles; t++)
                    reg.ElementWise.Scale(input.Data.SubView(0, inCount),
                        output.Data.SubView(t * inCount, inCount), inCount, 1f);
            }
            else
            {
                // General N-D tiling via index mapping
                var inVals = ctx.TryGetInputValues(0);
                if (inVals != null)
                {
                    var result = new float[outCount];
                    var inStrides = BroadcastHelper.ComputeStrides(inShape, inShape);
                    for (int i = 0; i < outCount; i++)
                    {
                        // Map output index to input index via modulo per dimension
                        int remaining = i;
                        int inIdx = 0;
                        for (int d = inShape.Length - 1; d >= 0; d--)
                        {
                            int outDim = outShape[d];
                            int coord = remaining % outDim;
                            remaining /= outDim;
                            int inCoord = coord % inShape[d];
                            inIdx += inCoord * inStrides[d];
                        }
                        result[i] = inIdx < inVals.Length ? inVals[inIdx] : 0f;
                    }
                    var temp = ctx.Pool.AllocatePermanent(result, outShape);
                    reg.ElementWise.Scale(temp.Data, output.Data, outCount, 1f);
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
        else
        {
            // No repeat info — copy input to output
            int copyCount = Math.Min(inCount, outCount);
            reg.ElementWise.Scale(input.Data.SubView(0, copyCount),
                output.Data.SubView(0, copyCount), copyCount, 1f);
        }
    }
}

/// <summary>CumSum: cumulative sum along an axis.</summary>
public class CumSumOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "CumSum";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        var input = ctx.Inputs[0];
        var output = ctx.Outputs[0];
        int count = input.ElementCount;

        // Get axis from second input (constant tensor)
        var axisVals = ctx.TryGetInputValues(1);
        int axis = axisVals != null && axisVals.Length > 0 ? (int)axisVals[0] : 0;
        if (axis < 0) axis += input.Shape.Length;
        int exclusive = ctx.GetInt("exclusive", 0);
        int reverse = ctx.GetInt("reverse", 0);

        var inVals = ctx.TryGetInputValues(0);
        if (inVals != null)
        {
            var result = new float[count];
            var shape = input.Shape;

            int outer = 1; for (int i = 0; i < axis; i++) outer *= shape[i];
            int axisSize = shape[axis];
            int inner = 1; for (int i = axis + 1; i < shape.Length; i++) inner *= shape[i];

            for (int o = 0; o < outer; o++)
                for (int inn = 0; inn < inner; inn++)
                {
                    float sum = 0;
                    for (int a = 0; a < axisSize; a++)
                    {
                        int ai = reverse != 0 ? axisSize - 1 - a : a;
                        int idx = (o * axisSize + ai) * inner + inn;
                        if (exclusive != 0)
                        {
                            result[idx] = sum;
                            sum += inVals[idx];
                        }
                        else
                        {
                            sum += inVals[idx];
                            result[idx] = sum;
                        }
                    }
                }

            output.Data.SubView(0, count).CopyFromCPU(result);
        }
        else
        {
            output.Data.SubView(0, count).CopyFrom(input.Data.SubView(0, count));
        }
    }
}

/// <summary>OneHot: generate one-hot encoded tensor from indices.</summary>
public class OneHotOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "OneHot";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] }; // Dynamic — depends on depth
    public void Execute(OnnxOpContext ctx)
    {
        var indices = ctx.Inputs[0];
        var output = ctx.Outputs[0];
        int axis = ctx.GetInt("axis", -1);

        var idxVals = ctx.TryGetInputValues(0);
        var depthVals = ctx.TryGetInputValues(1);
        var valueVals = ctx.TryGetInputValues(2);

        if (idxVals != null && depthVals != null && valueVals != null)
        {
            int depth = (int)depthVals[0];
            float offValue = valueVals.Length > 0 ? valueVals[0] : 0f;
            float onValue = valueVals.Length > 1 ? valueVals[1] : 1f;

            int outCount = output.ElementCount;
            var result = new float[outCount];
            Array.Fill(result, offValue);

            for (int i = 0; i < idxVals.Length; i++)
            {
                int idx = (int)idxVals[i];
                if (idx < 0) idx += depth;
                if (idx >= 0 && idx < depth && i * depth + idx < outCount)
                    result[i * depth + idx] = onValue;
            }

            output.Data.SubView(0, outCount).CopyFromCPU(result);
        }
    }
}

/// <summary>GatherElements: index into data along an axis using indices tensor.</summary>
public class GatherElementsOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "GatherElements";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[1] }; // output shape = indices shape
    public void Execute(OnnxOpContext ctx)
    {
        var data = ctx.Inputs[0];
        var indices = ctx.Inputs[1];
        var output = ctx.Outputs[0];
        int axis = ctx.GetInt("axis", 0);
        if (axis < 0) axis += data.Shape.Length;

        int outCount = output.ElementCount;
        var idxVals = ctx.TryGetInputValues(1);
        var dataVals = ctx.TryGetInputValues(0);

        if (idxVals != null && dataVals != null)
        {
            var result = new float[outCount];
            var shape = data.Shape;
            var idxShape = indices.Shape;
            var strides = new int[shape.Length];
            strides[^1] = 1;
            for (int i = shape.Length - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];

            for (int i = 0; i < outCount; i++)
            {
                int remaining = i;
                int srcIdx = 0;
                for (int d = idxShape.Length - 1; d >= 0; d--)
                {
                    int coord = remaining % idxShape[d];
                    remaining /= idxShape[d];
                    if (d == axis)
                    {
                        int gatherIdx = (int)idxVals[i];
                        if (gatherIdx < 0) gatherIdx += shape[d];
                        srcIdx += gatherIdx * strides[d];
                    }
                    else
                        srcIdx += coord * strides[d];
                }
                result[i] = srcIdx >= 0 && srcIdx < dataVals.Length ? dataVals[srcIdx] : 0f;
            }
            output.Data.SubView(0, outCount).CopyFromCPU(result);
        }
        else
        {
            int copyCount = Math.Min(data.ElementCount, outCount);
            output.Data.SubView(0, copyCount).CopyFrom(data.Data.SubView(0, copyCount));
        }
    }
}

/// <summary>Mod: element-wise modulo operation.</summary>
public class ModOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Mod";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        var aVals = ctx.TryGetInputValues(0);
        var bVals = ctx.TryGetInputValues(1);
        if (aVals != null && bVals != null)
        {
            int fmod = ctx.GetInt("fmod", 0);
            int outCount = ctx.Outputs[0].ElementCount;
            var result = new float[outCount];
            for (int i = 0; i < outCount; i++)
            {
                float av = i < aVals.Length ? aVals[i % aVals.Length] : 0f;
                float bv = i < bVals.Length ? bVals[i % bVals.Length] : 1f;
                result[i] = fmod != 0 ? av % bv : (bv != 0 ? (int)av % (int)bv : 0f);
            }
            ctx.Outputs[0].Data.SubView(0, outCount).CopyFromCPU(result);
        }
        else
        {
            BroadcastHelper.BroadcastBinaryOp(ctx, reg, (x, y) => y != 0 ? x % y : 0f);
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  New operators — full ONNX coverage (batch 3)
// ═══════════════════════════════════════════════════════════

// ── Bitwise operators (operate on float-encoded integers) ──

public class BitwiseAndOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "BitwiseAnd";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx) =>
        BroadcastHelper.BroadcastBinaryOp(ctx, reg, (a, b) => (float)((int)a & (int)b));
}

public class BitwiseOrOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "BitwiseOr";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx) =>
        BroadcastHelper.BroadcastBinaryOp(ctx, reg, (a, b) => (float)((int)a | (int)b));
}

public class BitwiseXorOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "BitwiseXor";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx) =>
        BroadcastHelper.BroadcastBinaryOp(ctx, reg, (a, b) => (float)((int)a ^ (int)b));
}

public class BitwiseNotOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "BitwiseNot";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        int count = ctx.Inputs[0].ElementCount;
        reg.ElementWise.UnaryOp(ctx.Inputs[0].Data, ctx.Outputs[0].Data, count,
            new DelegateSpecialization<Func<float, float>>(x => (float)(~(int)x)));
    }
}

public class BitShiftOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "BitShift";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        string direction = ctx.GetString("direction", "LEFT");
        if (direction == "LEFT")
            BroadcastHelper.BroadcastBinaryOp(ctx, reg, (a, b) => (float)((int)a << (int)b));
        else
            BroadcastHelper.BroadcastBinaryOp(ctx, reg, (a, b) => (float)((int)a >> (int)b));
    }
}

// ── Quantization operators ──

public class DequantizeLinearOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "DequantizeLinear";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // y = (x - zero_point) * scale
        var x = ctx.Inputs[0]; var scale = ctx.Inputs[1];
        var zp = ctx.Inputs.Length > 2 ? ctx.Inputs[2] : null;
        int count = x.ElementCount;
        if (zp != null)
        {
            // x - zero_point → temp, then temp * scale → output
            using var temp = reg.Accelerator.Allocate1D<float>(count);
            reg.ElementWise.Sub(x.Data, zp.Data, temp.View, count);
            reg.ElementWise.Mul(temp.View, scale.Data, ctx.Outputs[0].Data, count);
        }
        else
        {
            reg.ElementWise.Mul(x.Data, scale.Data, ctx.Outputs[0].Data, count);
        }
    }
}

public class QuantizeLinearOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "QuantizeLinear";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // y = clamp(round(x / scale) + zero_point, qmin, qmax)
        var x = ctx.Inputs[0]; var scale = ctx.Inputs[1];
        int count = x.ElementCount;
        reg.ElementWise.Div(x.Data, scale.Data, ctx.Outputs[0].Data, count);
        reg.ElementWise.Round(ctx.Outputs[0].Data, ctx.Outputs[0].Data, count);
        if (ctx.Inputs.Length > 2 && ctx.Inputs[2] != null)
            reg.ElementWise.Add(ctx.Outputs[0].Data, ctx.Inputs[2].Data, ctx.Outputs[0].Data, count);
        reg.ElementWise.Clip(ctx.Outputs[0].Data, ctx.Outputs[0].Data, count, 0f, 255f);
    }
}

public class DynamicQuantizeLinearOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "DynamicQuantizeLinear";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0], new[] { 1 }, new[] { 1 } }; // y, y_scale, y_zero_point
    public void Execute(OnnxOpContext ctx)
    {
        // ONNX DynamicQuantizeLinear: compute uint8 quantization from input range
        // y_scale = (max(0, max(x)) - min(0, min(x))) / 255
        // y_zero_point = saturate(round(-min(0, min(x)) / y_scale))
        // y = saturate(round(x / y_scale) + y_zero_point, 0, 255)
        int count = ctx.Inputs[0].ElementCount;
        var input = ctx.Inputs[0].Data;

        // Try constant path first (avoids GPU→CPU readback, works on all backends)
        var xVals = ctx.TryGetInputValues(0);
        if (xVals != null)
        {
            float xMax = Math.Max(0f, xVals.Max());
            float xMin = Math.Min(0f, xVals.Min());
            float yScale = (xMax - xMin) / 255f;
            if (yScale == 0f) yScale = 1f;
            float yZeroPoint = MathF.Max(0f, MathF.Min(255f, MathF.Round(-xMin / yScale)));
            var result = xVals.Select(x => MathF.Max(0f, MathF.Min(255f, MathF.Round(x / yScale) + yZeroPoint))).ToArray();
            ctx.Outputs[0].Data.SubView(0, count).CopyFromCPU(result);
            ctx.Outputs[1].Data.SubView(0, 1).CopyFromCPU(new[] { yScale });
            ctx.Outputs[2].Data.SubView(0, 1).CopyFromCPU(new[] { yZeroPoint });
            return;
        }

        // GPU path: reductions for min/max, then fused quantize kernel
        using var maxBuf = reg.Accelerator.Allocate1D<float>(1);
        using var minBuf = reg.Accelerator.Allocate1D<float>(1);
        reg.Reductions.ReduceMax(input, maxBuf.View, 1, count, 1);
        reg.Reductions.ReduceMin(input, minBuf.View, 1, count, 1);
        // Fused kernel: reads max/min scalars on GPU, computes scale/zp/quantized output
        reg.ElementWise.DynamicQuantize(input, ctx.Outputs[0].Data, ctx.Outputs[1].Data,
            ctx.Outputs[2].Data, maxBuf.View, minBuf.View, count);
    }
}

// ── CastLike ──

public class CastLikeOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "CastLike";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // CastLike: cast input to same type as target_type input. Our engine is all float32.
        int count = ctx.Inputs[0].ElementCount;
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, count, 1f);
    }
}

// ── Compress ──

public class CompressOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Compress";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Output size depends on runtime condition values — estimate as input size
        return new[] { new[] { inputs[0].Aggregate(1, (a, b) => a * b) } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // Compress: select elements where condition is non-zero
        // CPU fallback — Compress is rare in inference
        var condVals = ctx.TryGetInputValues(1);
        if (condVals != null)
        {
            var inputVals = ctx.TryGetInputValues(0);
            if (inputVals != null)
            {
                var result = new List<float>();
                for (int i = 0; i < Math.Min(inputVals.Length, condVals.Length); i++)
                    if (condVals[i] != 0f) result.Add(inputVals[i]);
                if (result.Count > 0)
                {
                    ctx.Outputs[0].Data.SubView(0, result.Count).CopyFromCPU(result.ToArray());
                }
            }
        }
    }
}

// ── EyeLike ──

public class EyeLikeOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "EyeLike";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        var shape = ctx.Inputs[0].Shape;
        int rows = shape[^2], cols = shape[^1];
        int k = ctx.GetInt("k", 0);
        var data = new float[rows * cols];
        for (int i = 0; i < rows; i++)
        {
            int j = i + k;
            if (j >= 0 && j < cols) data[i * cols + j] = 1f;
        }
        ctx.Outputs[0].Data.SubView(0, rows * cols).CopyFromCPU(data);
    }
}

// ── LRN (Local Response Normalization) ──

public class LRNOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "LRN";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // LRN: y[n,c,h,w] = x[n,c,h,w] / (bias + alpha/size * sum(x[n,c',h,w]^2))^beta
        int size = ctx.GetInt("size", 5);
        float alpha = ctx.GetFloat("alpha", 0.0001f);
        float beta = ctx.GetFloat("beta", 0.75f);
        float bias = ctx.GetFloat("bias", 1f);

        var shape = ctx.Inputs[0].Shape;
        int C = shape[1];
        int spatial = 1;
        for (int i = 2; i < shape.Length; i++) spatial *= shape[i];
        int halfSize = size / 2;
        int total = ctx.Inputs[0].ElementCount;

        // GPU path: params buffer [C, spatial, halfSize, size], fparams [alpha, beta, bias]
        var paramsBuf = reg.Accelerator.Allocate1D(new int[] { C, spatial, halfSize, size });
        var fparamsBuf = reg.Accelerator.Allocate1D(new float[] { alpha, beta, bias });
        reg.ElementWise.LRN(ctx.Inputs[0].Data, ctx.Outputs[0].Data, paramsBuf.View, fparamsBuf.View, total);
        reg.Accelerator.Synchronize();
        paramsBuf.Dispose();
        fparamsBuf.Dispose();
    }
}

// ── MeanVarianceNormalization ──

public class MeanVarianceNormalizationOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "MeanVarianceNormalization";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // MVN: normalize over specified axes (default: 0,2,3 = batch+spatial, keep channel)
        // y = (x - mean(x, axes)) / sqrt(variance(x, axes) + eps)
        var xVals = ctx.TryGetInputValues(0);
        if (xVals == null)
        {
            reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, 1f);
            return;
        }

        var shape = ctx.Inputs[0].Shape;
        int rank = shape.Length;

        // Parse axes — default is {0, 2, 3} for NCHW (normalize over batch+spatial, keep channel)
        int[] axes;
        if (ctx.Attributes.TryGetValue("axes", out var axObj) && axObj is long[] axLong)
            axes = axLong.Select(a => (int)(a < 0 ? a + rank : a)).ToArray();
        else
            axes = rank == 4 ? new[] { 0, 2, 3 } : Enumerable.Range(0, rank).Where(i => i != 1).ToArray();

        var axisSet = new HashSet<int>(axes);
        float eps = 1e-9f;

        // Compute strides
        var strides = new int[rank];
        strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; d--) strides[d] = strides[d + 1] * shape[d + 1];

        // Determine reduction group: dimensions NOT in axes are the "keep" dims
        int totalElements = xVals.Length;
        int reductionSize = 1;
        foreach (int a in axes) reductionSize *= shape[a];
        int groupCount = totalElements / reductionSize;

        var result = new float[totalElements];

        // For each unique combination of kept dimensions, compute mean+var over reduction dims
        var keepDims = Enumerable.Range(0, rank).Where(d => !axisSet.Contains(d)).ToArray();
        var keepShape = keepDims.Select(d => shape[d]).ToArray();
        int keepTotal = keepShape.Length > 0 ? keepShape.Aggregate(1, (a, b) => a * b) : 1;

        for (int g = 0; g < keepTotal; g++)
        {
            // Decode group index into kept-dimension coordinates
            var keepCoords = new int[keepDims.Length];
            int tmp = g;
            for (int i = keepDims.Length - 1; i >= 0; i--)
            {
                keepCoords[i] = tmp % keepShape[i];
                tmp /= keepShape[i];
            }

            // Collect all indices belonging to this group
            var indices = new List<int>();
            CollectIndices(shape, axes, axisSet, keepDims, keepCoords, strides, 0, 0, indices);

            // Compute mean
            float mean = 0f;
            foreach (int idx in indices) mean += xVals[idx];
            mean /= indices.Count;

            // Compute variance
            float variance = 0f;
            foreach (int idx in indices)
            {
                float diff = xVals[idx] - mean;
                variance += diff * diff;
            }
            variance /= indices.Count;
            float invStd = 1f / MathF.Sqrt(variance + eps);

            // Normalize
            foreach (int idx in indices)
                result[idx] = (xVals[idx] - mean) * invStd;
        }

        int total = result.Length;
        ctx.Outputs[0].Data.SubView(0, total).CopyFromCPU(result);
    }

    private static void CollectIndices(int[] shape, int[] axes, HashSet<int> axisSet, int[] keepDims, int[] keepCoords, int[] strides, int dim, int baseIdx, List<int> indices)
    {
        if (dim == shape.Length) { indices.Add(baseIdx); return; }
        if (axisSet.Contains(dim))
        {
            for (int i = 0; i < shape[dim]; i++)
                CollectIndices(shape, axes, axisSet, keepDims, keepCoords, strides, dim + 1, baseIdx + i * strides[dim], indices);
        }
        else
        {
            int keepIdx = Array.IndexOf(keepDims, dim);
            CollectIndices(shape, axes, axisSet, keepDims, keepCoords, strides, dim + 1, baseIdx + keepCoords[keepIdx] * strides[dim], indices);
        }
    }
}

// ── Random operators ──

public class RandomNormalOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "RandomNormal";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var shape = attrs.ContainsKey("shape") ? ((long[])attrs["shape"]).Select(x => (int)x).ToArray() : new[] { 1 };
        return new[] { shape };
    }
    public void Execute(OnnxOpContext ctx)
    {
        float mean = ctx.GetFloat("mean", 0f);
        float scale = ctx.GetFloat("scale", 1f);
        int seed = ctx.GetInt("seed", 0);
        var rng = seed != 0 ? new Random(seed) : new Random();
        int count = ctx.Outputs[0].ElementCount;
        var data = new float[count];
        for (int i = 0; i < count; i++)
        {
            // Box-Muller transform
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            data[i] = (float)(mean + scale * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        ctx.Outputs[0].Data.SubView(0, count).CopyFromCPU(data);
    }
}

public class RandomNormalLikeOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "RandomNormalLike";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => new RandomNormalOperator(reg).Execute(ctx);
}

public class RandomUniformOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "RandomUniform";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var shape = attrs.ContainsKey("shape") ? ((long[])attrs["shape"]).Select(x => (int)x).ToArray() : new[] { 1 };
        return new[] { shape };
    }
    public void Execute(OnnxOpContext ctx)
    {
        float low = ctx.GetFloat("low", 0f);
        float high = ctx.GetFloat("high", 1f);
        int seed = ctx.GetInt("seed", 0);
        var rng = seed != 0 ? new Random(seed) : new Random();
        int count = ctx.Outputs[0].ElementCount;
        var data = new float[count];
        for (int i = 0; i < count; i++)
            data[i] = low + (float)rng.NextDouble() * (high - low);
        ctx.Outputs[0].Data.SubView(0, count).CopyFromCPU(data);
    }
}

public class RandomUniformLikeOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "RandomUniformLike";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => new RandomUniformOperator(reg).Execute(ctx);
}

// ── ReverseSequence ──

public class ReverseSequenceOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ReverseSequence";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        int batchAxis = ctx.GetInt("batch_axis", 1);
        int timeAxis = ctx.GetInt("time_axis", 0);
        var shape = ctx.Inputs[0].Shape;
        int batchSize = shape[batchAxis];
        int timeSize = shape[timeAxis];
        int innerSize = 1;
        for (int d = Math.Max(batchAxis, timeAxis) + 1; d < shape.Length; d++) innerSize *= shape[d];
        int total = ctx.Inputs[0].ElementCount;

        // GPU path: one thread per element, reads seqLens from GPU
        var paramsBuf = reg.Accelerator.Allocate1D(new int[] { batchAxis, timeAxis, batchSize, timeSize, innerSize });
        reg.ElementWise.ReverseSequence(ctx.Inputs[0].Data, ctx.Outputs[0].Data,
            ctx.Inputs[1].Data, paramsBuf.View, total);
        reg.Accelerator.Synchronize();
        paramsBuf.Dispose();
    }
}

// ── Scatter (deprecated, use ScatterElements) ──

public class ScatterOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Scatter";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => new ScatterElementsOperator(reg).Execute(ctx);
}

// ── Unique ──

public class UniqueOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Unique";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0], inputs[0], inputs[0], new[] { 1 } }; // Y, indices, inverse_indices, counts
    public void Execute(OnnxOpContext ctx)
    {
        // Unique: return sorted unique elements with indices, inverse_indices, and counts
        var xVals = ctx.TryGetInputValues(0);
        if (xVals == null)
        {
            int count = ctx.Inputs[0].ElementCount;
            reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, count, 1f);
            return;
        }

        int sorted = ctx.GetInt("sorted", 1);

        // Find unique values preserving first-occurrence order
        var uniqueValues = new List<float>();
        var firstIdx = new List<int>(); // index of first occurrence
        var inverseMap = new int[xVals.Length]; // maps each input to its unique index
        var valueToUniqueIdx = new Dictionary<float, int>();

        for (int i = 0; i < xVals.Length; i++)
        {
            if (!valueToUniqueIdx.TryGetValue(xVals[i], out int uid))
            {
                uid = uniqueValues.Count;
                valueToUniqueIdx[xVals[i]] = uid;
                uniqueValues.Add(xVals[i]);
                firstIdx.Add(i);
            }
            inverseMap[i] = uid;
        }

        // Sort if requested
        if (sorted != 0)
        {
            var sortOrder = Enumerable.Range(0, uniqueValues.Count)
                .OrderBy(i => uniqueValues[i]).ToArray();
            var remapping = new int[uniqueValues.Count];
            for (int i = 0; i < sortOrder.Length; i++) remapping[sortOrder[i]] = i;

            var sortedValues = sortOrder.Select(i => uniqueValues[i]).ToList();
            var sortedFirstIdx = sortOrder.Select(i => firstIdx[i]).ToList();
            uniqueValues = sortedValues;
            firstIdx = sortedFirstIdx;
            for (int i = 0; i < inverseMap.Length; i++) inverseMap[i] = remapping[inverseMap[i]];
        }

        // Compute counts
        var counts = new float[uniqueValues.Count];
        foreach (int uid in inverseMap) counts[uid]++;

        // Output 0: Y (unique values)
        int yLen = Math.Min(uniqueValues.Count, ctx.Outputs[0].ElementCount);
        if (yLen > 0)
            ctx.Outputs[0].Data.SubView(0, yLen).CopyFromCPU(uniqueValues.ToArray().AsSpan(0, yLen).ToArray());

        // Output 1: indices (first occurrence of each unique value)
        if (ctx.Outputs.Length > 1 && ctx.Outputs[1] != null)
        {
            var idxArr = firstIdx.Select(i => (float)i).ToArray();
            int idxLen = Math.Min(idxArr.Length, ctx.Outputs[1].ElementCount);
            if (idxLen > 0)
                ctx.Outputs[1].Data.SubView(0, idxLen).CopyFromCPU(idxArr.AsSpan(0, idxLen).ToArray());
        }

        // Output 2: inverse_indices (maps each input element to its unique index)
        if (ctx.Outputs.Length > 2 && ctx.Outputs[2] != null)
        {
            var invArr = inverseMap.Select(i => (float)i).ToArray();
            int invLen = Math.Min(invArr.Length, ctx.Outputs[2].ElementCount);
            if (invLen > 0)
                ctx.Outputs[2].Data.SubView(0, invLen).CopyFromCPU(invArr.AsSpan(0, invLen).ToArray());
        }

        // Output 3: counts (how many times each unique value appears)
        if (ctx.Outputs.Length > 3 && ctx.Outputs[3] != null)
        {
            int cntLen = Math.Min(counts.Length, ctx.Outputs[3].ElementCount);
            if (cntLen > 0)
                ctx.Outputs[3].Data.SubView(0, cntLen).CopyFromCPU(counts.AsSpan(0, cntLen).ToArray());
        }
    }
}

// ── Window functions ──

public class HannWindowOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "HannWindow";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var sizeVals = inputs.Length > 0 ? inputs[0] : new[] { 1 };
        return new[] { sizeVals };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var sizeVals = ctx.TryGetInputValues(0);
        int size = sizeVals != null && sizeVals.Length > 0 ? (int)sizeVals[0] : 1;
        var data = new float[size];
        for (int i = 0; i < size; i++)
            data[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / (size - 1)));
        ctx.Outputs[0].Data.SubView(0, size).CopyFromCPU(data);
    }
}

public class HammingWindowOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "HammingWindow";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new HannWindowOperator(reg).InferOutputShapes(inputs, attrs);
    public void Execute(OnnxOpContext ctx)
    {
        var sizeVals = ctx.TryGetInputValues(0);
        int size = sizeVals != null && sizeVals.Length > 0 ? (int)sizeVals[0] : 1;
        var data = new float[size];
        for (int i = 0; i < size; i++)
            data[i] = 0.54f - 0.46f * MathF.Cos(2f * MathF.PI * i / (size - 1));
        ctx.Outputs[0].Data.SubView(0, size).CopyFromCPU(data);
    }
}

public class BlackmanWindowOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "BlackmanWindow";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new HannWindowOperator(reg).InferOutputShapes(inputs, attrs);
    public void Execute(OnnxOpContext ctx)
    {
        var sizeVals = ctx.TryGetInputValues(0);
        int size = sizeVals != null && sizeVals.Length > 0 ? (int)sizeVals[0] : 1;
        var data = new float[size];
        for (int i = 0; i < size; i++)
        {
            float t = 2f * MathF.PI * i / (size - 1);
            data[i] = 0.42f - 0.5f * MathF.Cos(t) + 0.08f * MathF.Cos(2f * t);
        }
        ctx.Outputs[0].Data.SubView(0, size).CopyFromCPU(data);
    }
}

// ── Multinomial ──

public class MultinomialOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Multinomial";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        int sampleSize = attrs.ContainsKey("sample_size") ? Convert.ToInt32(attrs["sample_size"]) : 1;
        return new[] { new[] { inputs[0][0], sampleSize } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // Multinomial: sample indices from log-probability distribution
        // Input: [batch, num_classes] log-probabilities
        // Output: [batch, sample_size] sampled class indices
        var logProbs = ctx.TryGetInputValues(0);
        int sampleSize = ctx.GetInt("sample_size", 1);
        int seed = ctx.GetInt("seed", 0);
        var rng = seed != 0 ? new Random(seed) : new Random();

        var shape = ctx.Inputs[0].Shape;
        int batch = shape[0], numClasses = shape[1];
        var result = new float[batch * sampleSize];

        for (int b = 0; b < batch; b++)
        {
            // Convert log-probs to cumulative probabilities
            var probs = new float[numClasses];
            float maxLogProb = float.NegativeInfinity;
            for (int c = 0; c < numClasses; c++)
            {
                float lp = logProbs != null ? logProbs[b * numClasses + c] : 0f;
                if (lp > maxLogProb) maxLogProb = lp;
            }
            float sumExp = 0f;
            for (int c = 0; c < numClasses; c++)
            {
                float lp = logProbs != null ? logProbs[b * numClasses + c] : 0f;
                probs[c] = MathF.Exp(lp - maxLogProb);
                sumExp += probs[c];
            }
            // Normalize and build CDF
            float cumSum = 0f;
            for (int c = 0; c < numClasses; c++)
            {
                probs[c] /= sumExp;
                cumSum += probs[c];
                probs[c] = cumSum;
            }
            // Sample
            for (int s = 0; s < sampleSize; s++)
            {
                float u = (float)rng.NextDouble();
                int idx = 0;
                for (int c = 0; c < numClasses; c++)
                    if (u <= probs[c]) { idx = c; break; }
                result[b * sampleSize + s] = idx;
            }
        }

        int copyLen = Math.Min(result.Length, ctx.Outputs[0].ElementCount);
        if (copyLen < result.Length) { var t = new float[copyLen]; Array.Copy(result, t, copyLen); ctx.Outputs[0].Data.SubView(0, copyLen).CopyFromCPU(t); }
        else ctx.Outputs[0].Data.SubView(0, copyLen).CopyFromCPU(result);
    }
}

// ── Loss functions (training) ──

public class NegativeLogLikelihoodLossOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "NegativeLogLikelihoodLoss";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        string reduction = attrs.ContainsKey("reduction") ? attrs["reduction"].ToString()! : "mean";
        if (reduction == "none") return new[] { inputs.Length > 1 ? inputs[1] : new[] { inputs[0][0] } };
        return new[] { new[] { 1 } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // NLLLoss: -sum(target * log_prob) / N
        // Input[0] = log-probabilities [N, C], Input[1] = target [N] (class indices)
        var logProbs = ctx.TryGetInputValues(0);
        var targets = ctx.TryGetInputValues(1);
        if (logProbs == null || targets == null)
        {
            reg.ElementWise.Fill(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount, 0f);
            return;
        }
        var shape = ctx.Inputs[0].Shape;
        int N = shape[0], C = shape.Length > 1 ? shape[1] : 1;
        float loss = 0f;
        for (int n = 0; n < N; n++)
        {
            int target = (int)targets[n];
            if (target >= 0 && target < C)
                loss -= logProbs[n * C + target];
        }
        string reduction = ctx.GetString("reduction", "mean");
        if (reduction == "mean" && N > 0) loss /= N;
        ctx.Outputs[0].Data.SubView(0, 1).CopyFromCPU(new[] { loss });
    }
}

public class SoftmaxCrossEntropyLossOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "SoftmaxCrossEntropyLoss";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        string reduction = attrs.ContainsKey("reduction") ? attrs["reduction"].ToString()! : "mean";
        if (reduction == "none") return new[] { new[] { inputs[0][0] }, inputs[0] };
        return new[] { new[] { 1 }, inputs[0] };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // SoftmaxCE: softmax(logits) → -log(p[target]) → reduce
        // Input[0] = logits [N, C], Input[1] = target [N]
        var logits = ctx.TryGetInputValues(0);
        var targets = ctx.TryGetInputValues(1);
        if (logits == null || targets == null)
        {
            reg.ElementWise.Fill(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount, 0f);
            if (ctx.Outputs.Length > 1) reg.ElementWise.Fill(ctx.Outputs[1].Data, ctx.Outputs[1].ElementCount, 0f);
            return;
        }
        var shape = ctx.Inputs[0].Shape;
        int N = shape[0], C = shape.Length > 1 ? shape[1] : 1;
        var logProbs = new float[N * C];
        float totalLoss = 0f;

        for (int n = 0; n < N; n++)
        {
            // Softmax
            float maxVal = float.NegativeInfinity;
            for (int c = 0; c < C; c++) maxVal = Math.Max(maxVal, logits[n * C + c]);
            float sumExp = 0f;
            for (int c = 0; c < C; c++) { logProbs[n * C + c] = MathF.Exp(logits[n * C + c] - maxVal); sumExp += logProbs[n * C + c]; }
            for (int c = 0; c < C; c++) logProbs[n * C + c] = MathF.Log(logProbs[n * C + c] / sumExp);
            // Loss
            int target = (int)targets[n];
            if (target >= 0 && target < C) totalLoss -= logProbs[n * C + target];
        }
        string reduction = ctx.GetString("reduction", "mean");
        if (reduction == "mean" && N > 0) totalLoss /= N;
        ctx.Outputs[0].Data.SubView(0, 1).CopyFromCPU(new[] { totalLoss });
        // Output[1] = log probabilities
        if (ctx.Outputs.Length > 1)
        {
            int lpCount = Math.Min(logProbs.Length, ctx.Outputs[1].ElementCount);
            ctx.Outputs[1].Data.SubView(0, lpCount).CopyFromCPU(logProbs.AsSpan(0, lpCount).ToArray());
        }
    }
}
