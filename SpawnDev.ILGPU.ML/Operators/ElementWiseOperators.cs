using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Operators;

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
        // Clip: inputs[0]=data, inputs[1]=min (optional), inputs[2]=max (optional)
        // For now, copy and clamp using scale as identity
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, 1f);
        reg.ElementWise.ReLUInPlace(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount); // simplified: Clip(0, inf) = ReLU
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
        else if (b.ElementCount == a.Shape[^1]) // per-channel broadcast
        {
            // Copy a to output, then AddBias
            reg.ElementWise.Scale(a.Data, ctx.Outputs[0].Data, a.ElementCount, 1f);
            reg.ElementWise.AddBias(ctx.Outputs[0].Data, b.Data, a.ElementCount, b.ElementCount);
        }
        else
        {
            throw new NotSupportedException($"Add broadcast not yet implemented for shapes [{string.Join(",", a.Shape)}] + [{string.Join(",", b.Shape)}]");
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
            reg.ElementWise.BroadcastMul(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount, b.ElementCount);
        }
        else
        {
            throw new NotSupportedException($"Mul broadcast not yet implemented for shapes [{string.Join(",", a.Shape)}] * [{string.Join(",", b.Shape)}]");
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
        // Sub(a, b) = a + (-b): negate b into output, then add a
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        reg.ElementWise.Scale(b.Data, ctx.Outputs[0].Data, b.ElementCount, -1f);
        reg.ElementWise.AddInPlace(ctx.Outputs[0].Data, a.Data, a.ElementCount);
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
        else
        {
            // Broadcast div: compute reciprocal of b, then broadcast multiply
            // For now, only support per-channel broadcast (b is last dim of a)
            throw new NotSupportedException($"Div broadcast not yet implemented for shapes [{string.Join(",", a.Shape)}] / [{string.Join(",", b.Shape)}]");
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
