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

/// <summary>Upsample: resize using nearest or linear mode.</summary>
public class UpsampleOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Upsample";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        throw new NotSupportedException("Upsample not yet implemented — need scales tensor readback");
    }
}

/// <summary>Shape: outputs the shape of the input as a 1D int64 tensor. TODO: needs int64 support.</summary>
public class ShapeOperator : IOnnxOperator
{
    public string OpType => "Shape";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { new[] { inputs[0].Length } };
    public void Execute(OnnxOpContext ctx)
    {
        throw new NotSupportedException("Shape operator requires int64 tensor output — not yet implemented");
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
