using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Operators;

// ── MatMul ──

public class MatMulOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "MatMul";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var a = inputs[0]; var b = inputs[1];
        int M = a[^2]; int N = b[^1];
        // Handle batched: broadcast leading dims
        var outShape = new List<int>();
        int maxLeading = Math.Max(a.Length - 2, b.Length - 2);
        for (int i = 0; i < maxLeading; i++)
        {
            int da = i < a.Length - 2 ? a[i] : 1;
            int db = i < b.Length - 2 ? b[i] : 1;
            outShape.Add(Math.Max(da, db));
        }
        outShape.Add(M);
        outShape.Add(N);
        return new[] { outShape.ToArray() };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        int M = a.Shape[^2]; int K = a.Shape[^1]; int N = b.Shape[^1];
        if (a.Rank == 2 && b.Rank == 2)
        {
            reg.MatMul.MatMul(a.Data, b.Data, ctx.Outputs[0].Data, M, K, N);
        }
        else
        {
            int batch = a.ElementCount / (M * K);
            reg.MatMul.BatchedMatMul(a.Data, b.Data, ctx.Outputs[0].Data, batch, M, K, N);
        }
    }
}

// ── Softmax ──

public class SoftmaxOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Softmax";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        int axis = ctx.GetInt("axis", -1);
        var shape = ctx.Inputs[0].Shape;
        if (axis < 0) axis += shape.Length;
        int rows = 1; for (int i = 0; i < axis; i++) rows *= shape[i];
        int cols = 1; for (int i = axis; i < shape.Length; i++) cols *= shape[i];
        // Copy to output then in-place softmax
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, 1f);
        reg.Softmax.Forward(ctx.Outputs[0].Data, rows, cols);
    }
}

// ── LayerNormalization ──

public class LayerNormOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "LayerNormalization";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] }; // Same shape as input
    public void Execute(OnnxOpContext ctx)
    {
        int axis = ctx.GetInt("axis", -1);
        float eps = ctx.GetFloat("epsilon", 1e-5f);
        var shape = ctx.Inputs[0].Shape;
        if (axis < 0) axis += shape.Length;
        int rows = 1; for (int i = 0; i < axis; i++) rows *= shape[i];
        int C = 1; for (int i = axis; i < shape.Length; i++) C *= shape[i];
        reg.LayerNorm.Forward(ctx.Inputs[0].Data, ctx.Outputs[0].Data,
            ctx.Inputs[1].Data, ctx.Inputs[2].Data, rows, C, eps);
    }
}

// ── BatchNormalization ──

public class BatchNormOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "BatchNormalization";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // inputs: X, scale, B, input_mean, input_var
        var shape = ctx.Inputs[0].Shape;
        int N = shape[0]; int C = shape[1];
        int spatial = 1; for (int i = 2; i < shape.Length; i++) spatial *= shape[i];
        reg.Normalization.BatchNorm(ctx.Inputs[0].Data, ctx.Outputs[0].Data,
            ctx.Inputs[1].Data, ctx.Inputs[2].Data,
            ctx.Inputs[3].Data, ctx.Inputs[4].Data, N, C, spatial);
    }
}

// ── Conv ──

public class ConvOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Conv";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Simplified: 2D conv only
        var x = inputs[0]; var w = inputs[1];
        var pads = attrs.ContainsKey("pads") ? ((long[])attrs["pads"]).Select(p => (int)p).ToArray() : new int[4];
        var strides = attrs.ContainsKey("strides") ? ((long[])attrs["strides"]).Select(s => (int)s).ToArray() : new[] { 1, 1 };
        int outC = w[0]; int kH = w[2]; int kW = w[3];
        int outH = (x[2] + pads[0] + pads[2] - kH) / strides[0] + 1;
        int outW = (x[3] + pads[1] + pads[3] - kW) / strides[1] + 1;
        return new[] { new[] { x[0], outC, outH, outW } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var x = ctx.Inputs[0]; var w = ctx.Inputs[1];
        var pads = ctx.GetInts("pads"); int pad = pads.Length > 0 ? pads[0] : 0;
        var strides = ctx.GetInts("strides"); int stride = strides.Length > 0 ? strides[0] : 1;
        int inC = x.Shape[1]; int inH = x.Shape[2]; int inW = x.Shape[3];
        int outC = w.Shape[0]; int kH = w.Shape[2]; int kW = w.Shape[3];
        var bias = ctx.Inputs.Length > 2 ? ctx.Inputs[2].Data : default;
        // For batch > 1, need to loop (current Conv2D is single-image)
        reg.Conv2D.Forward(x.Data, w.Data, bias, ctx.Outputs[0].Data,
            inC, inH, inW, outC, kH, kW, stride, pad);
    }
}

// ── Pooling ──

public class GlobalAvgPoolOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "GlobalAveragePool";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var s = inputs[0];
        return new[] { new[] { s[0], s[1], 1, 1 } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var s = ctx.Inputs[0].Shape;
        int N = s[0]; int C = s[1];
        int spatial = 1; for (int i = 2; i < s.Length; i++) spatial *= s[i];
        reg.Pooling.GlobalAvgPool(ctx.Inputs[0].Data, ctx.Outputs[0].Data, N, C, spatial);
    }
}

// ── Reductions ──

public class ReduceMeanOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ReduceMean";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var shape = inputs[0];
        var axes = attrs.ContainsKey("axes") ? ((long[])attrs["axes"]).Select(a => (int)(a < 0 ? a + shape.Length : a)).ToArray() : new[] { shape.Length - 1 };
        bool keepdims = !attrs.ContainsKey("keepdims") || Convert.ToInt32(attrs["keepdims"]) != 0;
        var outShape = new List<int>();
        for (int i = 0; i < shape.Length; i++)
        {
            if (axes.Contains(i))
            { if (keepdims) outShape.Add(1); }
            else outShape.Add(shape[i]);
        }
        return new[] { outShape.ToArray() };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var shape = ctx.Inputs[0].Shape;
        var axes = ctx.GetLongs("axes");
        int axis = axes.Length > 0 ? (int)(axes[0] < 0 ? axes[0] + shape.Length : axes[0]) : shape.Length - 1;
        int outer = 1; for (int i = 0; i < axis; i++) outer *= shape[i];
        int reduce = shape[axis];
        int inner = 1; for (int i = axis + 1; i < shape.Length; i++) inner *= shape[i];
        reg.Reductions.ReduceMean(ctx.Inputs[0].Data, ctx.Outputs[0].Data, outer, reduce, inner);
    }
}

public class ReduceSumOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ReduceSum";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Simplified: same as ReduceMean
        return new ReduceMeanOperator(reg).InferOutputShapes(inputs, attrs);
    }
    public void Execute(OnnxOpContext ctx)
    {
        var shape = ctx.Inputs[0].Shape;
        var axes = ctx.GetLongs("axes");
        int axis = axes.Length > 0 ? (int)(axes[0] < 0 ? axes[0] + shape.Length : axes[0]) : shape.Length - 1;
        int outer = 1; for (int i = 0; i < axis; i++) outer *= shape[i];
        int reduce = shape[axis];
        int inner = 1; for (int i = axis + 1; i < shape.Length; i++) inner *= shape[i];
        reg.Reductions.ReduceSum(ctx.Inputs[0].Data, ctx.Outputs[0].Data, outer, reduce, inner);
    }
}

// ── Gather ──

public class GatherOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Gather";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Simplified: axis=0, indices is 1D
        var dataShape = inputs[0];
        var idxShape = inputs[1];
        int innerSize = 1; for (int i = 1; i < dataShape.Length; i++) innerSize *= dataShape[i];
        int numIdx = TensorHelpers.ElementCount(idxShape);
        return new[] { new[] { numIdx, innerSize } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var data = ctx.Inputs[0]; var indices = ctx.Inputs[1];
        int innerSize = 1; for (int i = 1; i < data.Shape.Length; i++) innerSize *= data.Shape[i];
        // indices tensor is float but contains ints — need int view
        // For now, this requires the indices to already be in an int buffer
        // TODO: support float→int index conversion
        throw new NotSupportedException("Gather requires int indices buffer — not yet wired for float tensors");
    }
}

// ── Concat ──

public class ConcatOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Concat";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        int axis = attrs.ContainsKey("axis") ? Convert.ToInt32(attrs["axis"]) : 0;
        if (axis < 0) axis += inputs[0].Length;
        var outShape = (int[])inputs[0].Clone();
        for (int i = 1; i < inputs.Length; i++)
            outShape[axis] += inputs[i][axis];
        return new[] { outShape };
    }
    public void Execute(OnnxOpContext ctx)
    {
        int axis = ctx.GetInt("axis", 0);
        if (axis < 0) axis += ctx.Inputs[0].Shape.Length;
        // For last-dim concat of 2 inputs, use existing ConcatLastDim
        if (axis == ctx.Inputs[0].Rank - 1 && ctx.Inputs.Length == 2)
        {
            int T = 1; for (int i = 0; i < axis; i++) T *= ctx.Inputs[0].Shape[i];
            int C = ctx.Inputs[0].Shape[axis];
            reg.ElementWise.ConcatLastDim(ctx.Inputs[0].Data, ctx.Inputs[1].Data,
                ctx.Outputs[0].Data, T, C);
        }
        else
        {
            throw new NotSupportedException($"Concat on axis {axis} with {ctx.Inputs.Length} inputs not yet implemented");
        }
    }
}

// ── Transpose ──

public class TransposeOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Transpose";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var perm = attrs.ContainsKey("perm") ? ((long[])attrs["perm"]).Select(p => (int)p).ToArray()
                 : Enumerable.Range(0, inputs[0].Length).Reverse().ToArray();
        var outShape = new int[inputs[0].Length];
        for (int i = 0; i < perm.Length; i++) outShape[i] = inputs[0][perm[i]];
        return new[] { outShape };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var perm = ctx.GetInts("perm");
        if (perm.Length == 0)
            perm = Enumerable.Range(0, ctx.Inputs[0].Rank).Reverse().ToArray();
        reg.Transpose.Transpose(ctx.Inputs[0].Data, ctx.Outputs[0].Data,
            ctx.Inputs[0].Shape, perm);
    }
}
