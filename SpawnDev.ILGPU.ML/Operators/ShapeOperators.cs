using SpawnDev.ILGPU.ML.Tensors;

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
        // Copy input to output (different buffers allocated by executor).
        // Zero-copy would be ideal but requires buffer aliasing in the executor.
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data,
            ctx.Inputs[0].ElementCount, 1f);
    }
}

public class UnsqueezeOperator : IOnnxOperator
{
    public string OpType => "Unsqueeze";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) { }
}

public class SqueezeOperator : IOnnxOperator
{
    public string OpType => "Squeeze";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) { }
}

/// <summary>Constant: output is a constant value (stored as initializer). No-op at runtime.</summary>
public class ConstantOperator : IOnnxOperator
{
    public string OpType => "Constant";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { new[] { 1 } };
    public void Execute(OnnxOpContext ctx) { }
}

/// <summary>Cast: type conversion. For float→float, this is a copy.</summary>
public class CastOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Cast";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, 1f);
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
        reg.ElementWise.Min(ctx.Inputs[0].Data, ctx.Inputs[1].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class MaxOnnxOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Max";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Max(ctx.Inputs[0].Data, ctx.Inputs[1].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
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

public class FlattenOperator : IOnnxOperator
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
    public void Execute(OnnxOpContext ctx) { }
}
