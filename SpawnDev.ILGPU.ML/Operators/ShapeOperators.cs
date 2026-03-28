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
