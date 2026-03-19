using ILGPU.Runtime;
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
        int group = ctx.GetInt("group", 1);
        int inC = x.Shape[1]; int inH = x.Shape[2]; int inW = x.Shape[3];
        int outC = w.Shape[0]; int kH = w.Shape[2]; int kW = w.Shape[3];
        var bias = ctx.Inputs.Length > 2 && ctx.Inputs[2] != null ? ctx.Inputs[2].Data : default;

        if (group == inC && group == outC)
        {
            // Depthwise convolution: each channel convolved independently
            reg.Conv2D.ForwardDepthwise(x.Data, w.Data, bias, ctx.Outputs[0].Data,
                inC, inH, inW, kH, kW, stride, pad);
        }
        else if (group == 1)
        {
            // Standard convolution
            reg.Conv2D.Forward(x.Data, w.Data, bias, ctx.Outputs[0].Data,
                inC, inH, inW, outC, kH, kW, stride, pad);
        }
        else
        {
            throw new NotSupportedException($"Conv with group={group} (inC={inC}, outC={outC}) not yet implemented — only group=1 and depthwise (group=inC=outC) supported");
        }
    }
}

// ── ArgMax ──

public class ArgMaxOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ArgMax";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        int axis = attrs.ContainsKey("axis") ? (int)(long)attrs["axis"] : 0;
        bool keepdims = !attrs.ContainsKey("keepdims") || (long)attrs["keepdims"] != 0;
        var shape = inputs[0].ToList();
        if (axis < 0) axis += shape.Count;
        if (keepdims) { shape[axis] = 1; }
        else { shape.RemoveAt(axis); }
        return new[] { shape.ToArray() };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // CPU-side ArgMax (small output tensor)
        var input = ctx.Inputs[0];
        int axis = ctx.GetInt("axis", 0);
        if (axis < 0) axis += input.Shape.Length;

        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= input.Shape[i];
        int axisSize = input.Shape[axis];
        int innerSize = 1;
        for (int i = axis + 1; i < input.Shape.Length; i++) innerSize *= input.Shape[i];

        // Read input to CPU — use pre-read values if available
        int total = input.ElementCount;
        var data = ctx.TryGetInputValues(0);
        if (data == null || data.Length != total)
        {
            // For large tensors that weren't pre-read, we need sync copy (CPU-only)
            data = new float[total];
            input.Data.SubView(0, total).CopyToCPU(data);
        }

        // Compute argmax
        var result = new float[outerSize * innerSize];
        for (int o = 0; o < outerSize; o++)
        {
            for (int inn = 0; inn < innerSize; inn++)
            {
                float maxVal = float.NegativeInfinity;
                int maxIdx = 0;
                for (int a = 0; a < axisSize; a++)
                {
                    float v = data[o * axisSize * innerSize + a * innerSize + inn];
                    if (v > maxVal) { maxVal = v; maxIdx = a; }
                }
                result[o * innerSize + inn] = maxIdx;
            }
        }

        // Upload result
        var temp = ctx.Pool.AllocatePermanent(result, ctx.Outputs[0].Shape);
        reg.ElementWise.Scale(temp.Data, ctx.Outputs[0].Data, result.Length, 1f);
    }
}

// ── GatherND ──

public class GatherNDOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "GatherND";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[1] }; // Simplified — output shape depends on indices
    public void Execute(OnnxOpContext ctx)
    {
        // CPU-side GatherND (indices are typically small)
        var data = ctx.Inputs[0];
        var indices = ctx.Inputs[1];
        int batchDims = ctx.GetInt("batch_dims", 0);

        int dataTotal = data.ElementCount;
        int idxTotal = indices.ElementCount;
        // Use pre-read values if available (avoids GPU→CPU readback on browser backends)
        var dataArr = ctx.TryGetInputValues(0);
        if (dataArr == null || dataArr.Length != dataTotal)
        {
            dataArr = new float[dataTotal];
            data.Data.SubView(0, dataTotal).CopyToCPU(dataArr);
        }
        var idxArr = ctx.TryGetInputValues(1);
        if (idxArr == null || idxArr.Length != idxTotal)
        {
            idxArr = new float[idxTotal];
            indices.Data.SubView(0, idxTotal).CopyToCPU(idxArr);
        }

        // Simple 1D gather for common case
        int outputSize = ctx.Outputs[0].ElementCount;
        var result = new float[outputSize];

        // Compute strides for data tensor
        var dataShape = data.Shape;
        var strides = new int[dataShape.Length];
        strides[^1] = 1;
        for (int i = dataShape.Length - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * dataShape[i + 1];

        int lastIdxDim = indices.Shape[^1];
        int numSlices = idxTotal / lastIdxDim;
        int sliceSize = 1;
        for (int i = lastIdxDim; i < dataShape.Length; i++)
            sliceSize *= dataShape[i];

        for (int s = 0; s < numSlices && s * sliceSize < outputSize; s++)
        {
            int offset = 0;
            for (int d = 0; d < lastIdxDim; d++)
                offset += (int)idxArr[s * lastIdxDim + d] * strides[d];

            for (int j = 0; j < sliceSize && s * sliceSize + j < outputSize; j++)
                result[s * sliceSize + j] = (offset + j < dataTotal) ? dataArr[offset + j] : 0f;
        }

        var temp = ctx.Pool.AllocatePermanent(result, ctx.Outputs[0].Shape);
        reg.ElementWise.Scale(temp.Data, ctx.Outputs[0].Data, outputSize, 1f);
    }
}

// ── ConvTranspose ──

public class ConvTransposeOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ConvTranspose";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var x = inputs[0]; var w = inputs[1];
        var strides = attrs.ContainsKey("strides") ? ((long[])attrs["strides"]).Select(s => (int)s).ToArray() : new[] { 1, 1 };
        var pads = attrs.ContainsKey("pads") ? ((long[])attrs["pads"]).Select(p => (int)p).ToArray() : new int[4];
        int outC = w[1]; int kH = w[2]; int kW = w[3];
        int outH = (x[2] - 1) * strides[0] - pads[0] - pads[2] + kH;
        int outW = (x[3] - 1) * strides[1] - pads[1] - pads[3] + kW;
        return new[] { new[] { x[0], outC, outH, outW } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var x = ctx.Inputs[0]; var w = ctx.Inputs[1];
        var strides = ctx.GetInts("strides"); int stride = strides.Length > 0 ? strides[0] : 1;
        var pads = ctx.GetInts("pads"); int pad = pads.Length > 0 ? pads[0] : 0;
        int inC = x.Shape[1]; int inH = x.Shape[2]; int inW = x.Shape[3];
        int outC = w.Shape[1]; int kH = w.Shape[2]; int kW = w.Shape[3];
        var bias = ctx.Inputs.Length > 2 && ctx.Inputs[2] != null ? ctx.Inputs[2].Data : default;
        reg.ConvTranspose.Forward(x.Data, w.Data, bias, ctx.Outputs[0].Data,
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

// ── ReduceMax / ReduceMin ──

public class ReduceMaxOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ReduceMax";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new ReduceMeanOperator(reg).InferOutputShapes(inputs, attrs);
    public void Execute(OnnxOpContext ctx)
    {
        var shape = ctx.Inputs[0].Shape;
        var axes = ctx.GetLongs("axes");
        int axis = axes.Length > 0 ? (int)(axes[0] < 0 ? axes[0] + shape.Length : axes[0]) : shape.Length - 1;
        int outer = 1; for (int i = 0; i < axis; i++) outer *= shape[i];
        int reduce = shape[axis];
        int inner = 1; for (int i = axis + 1; i < shape.Length; i++) inner *= shape[i];
        reg.Reductions.ReduceMax(ctx.Inputs[0].Data, ctx.Outputs[0].Data, outer, reduce, inner);
    }
}

public class ReduceMinOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ReduceMin";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new ReduceMeanOperator(reg).InferOutputShapes(inputs, attrs);
    public void Execute(OnnxOpContext ctx)
    {
        var shape = ctx.Inputs[0].Shape;
        var axes = ctx.GetLongs("axes");
        int axis = axes.Length > 0 ? (int)(axes[0] < 0 ? axes[0] + shape.Length : axes[0]) : shape.Length - 1;
        int outer = 1; for (int i = 0; i < axis; i++) outer *= shape[i];
        int reduce = shape[axis];
        int inner = 1; for (int i = axis + 1; i < shape.Length; i++) inner *= shape[i];
        reg.Reductions.ReduceMin(ctx.Inputs[0].Data, ctx.Outputs[0].Data, outer, reduce, inner);
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
        int axis = ctx.GetInt("axis", 0);
        if (axis < 0) axis += data.Shape.Length;

        // Get index values from pre-read constants (avoids GPU→CPU readback)
        var idxFloats = ctx.TryGetInputValues(1);
        if (idxFloats == null)
        {
            // For small index tensors, fallback to treating as identity gather
            // (copy input to output)
            int copyCount = Math.Min(data.ElementCount, ctx.Outputs[0].ElementCount);
            reg.ElementWise.Scale(data.Data.SubView(0, copyCount),
                ctx.Outputs[0].Data.SubView(0, copyCount), copyCount, 1f);
            return;
        }

        int numIdx = idxFloats.Length;
        int innerSize = 1;
        for (int i = axis + 1; i < data.Shape.Length; i++) innerSize *= data.Shape[i];
        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= data.Shape[i];
        int axisSize = data.Shape[axis];

        // For each index, copy the corresponding slice
        for (int o = 0; o < outerSize; o++)
        {
            for (int idx = 0; idx < numIdx; idx++)
            {
                int srcIdx = (int)idxFloats[idx];
                if (srcIdx < 0) srcIdx += axisSize; // Negative indexing
                if (srcIdx < 0 || srcIdx >= axisSize) srcIdx = 0;

                int srcOffset = (o * axisSize + srcIdx) * innerSize;
                int dstOffset = (o * numIdx + idx) * innerSize;
                reg.ElementWise.Scale(
                    data.Data.SubView(srcOffset, innerSize),
                    ctx.Outputs[0].Data.SubView(dstOffset, innerSize),
                    innerSize, 1f);
            }
        }
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

        // General concat: copy each input's blocks to the output at the correct offset.
        // For axis=1 (NCHW channel concat): outer=N, concat dim=C, inner=H*W
        var shape0 = ctx.Inputs[0].Shape;
        int outer = 1; for (int i = 0; i < axis; i++) outer *= shape0[i];
        int inner = 1; for (int i = axis + 1; i < shape0.Length; i++) inner *= shape0[i];

        int outOffset = 0;
        int totalConcatDim = 0;
        for (int n = 0; n < ctx.Inputs.Length; n++)
            totalConcatDim += ctx.Inputs[n].Shape[axis];

        // For each outer block, copy each input's slice
        for (int n = 0; n < ctx.Inputs.Length; n++)
        {
            var inp = ctx.Inputs[n];
            int concatDim = inp.Shape[axis];
            int blockSize = concatDim * inner;

            for (int o = 0; o < outer; o++)
            {
                int srcOffset = o * blockSize;
                int dstOffset = o * totalConcatDim * inner + outOffset;

                // Bounds-safe copy — clamp to actual tensor size
                int actualSrcLen = Math.Min(blockSize, (int)inp.Data.Length - srcOffset);
                int actualDstLen = Math.Min(blockSize, (int)ctx.Outputs[0].Data.Length - dstOffset);
                int copyLen = Math.Min(actualSrcLen, actualDstLen);
                if (copyLen <= 0 || srcOffset < 0 || dstOffset < 0) continue;

                reg.ElementWise.Scale(
                    inp.Data.SubView(srcOffset, copyLen),
                    ctx.Outputs[0].Data.SubView(dstOffset, copyLen),
                    copyLen, 1f);
            }
            outOffset += concatDim * inner;
        }
    }
}

// ── InstanceNormalization ──

public class InstanceNormOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "InstanceNormalization";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        var shape = ctx.Inputs[0].Shape;
        int N = shape[0]; int C = shape[1];
        int spatial = 1; for (int i = 2; i < shape.Length; i++) spatial *= shape[i];
        reg.Normalization.InstanceNorm(ctx.Inputs[0].Data, ctx.Outputs[0].Data,
            ctx.Inputs[1].Data, ctx.Inputs[2].Data, N, C, spatial);
    }
}

// ── Gemm (General Matrix Multiply) ──

public class GemmOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Gemm";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        int transA = attrs.ContainsKey("transA") ? Convert.ToInt32(attrs["transA"]) : 0;
        int transB = attrs.ContainsKey("transB") ? Convert.ToInt32(attrs["transB"]) : 0;
        int M = transA != 0 ? inputs[0][1] : inputs[0][0];
        int N = transB != 0 ? inputs[1][0] : inputs[1][1];
        return new[] { new[] { M, N } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        float alpha = ctx.GetFloat("alpha", 1f);
        float beta = ctx.GetFloat("beta", 1f);
        int transA = ctx.GetInt("transA", 0);
        int transB = ctx.GetInt("transB", 0);
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];

        if (transA != 0)
            throw new NotSupportedException("Gemm with transA=1 not yet implemented");

        int M = a.Shape[0]; int K = a.Shape[1];
        int N = transB != 0 ? b.Shape[0] : b.Shape[1];

        if (transB != 0)
        {
            // B is [N, K], need [K, N] for MatMul. Transpose it.
            var bT = ctx.Pool.Rent(new[] { K, N });
            reg.Transpose.Transpose(b.Data, bT.Data, b.Shape, new[] { 1, 0 });
            reg.MatMul.MatMul(a.Data, bT.Data, ctx.Outputs[0].Data, M, K, N);
        }
        else
        {
            reg.MatMul.MatMul(a.Data, b.Data, ctx.Outputs[0].Data, M, K, N);
        }

        if (ctx.Inputs.Length > 2 && ctx.Inputs[2] != null && beta != 0f)
            reg.ElementWise.AddBias(ctx.Outputs[0].Data, ctx.Inputs[2].Data, M * N, N);

        if (alpha != 1f)
            reg.ElementWise.ScaleInPlace(ctx.Outputs[0].Data, M * N, alpha);
    }
}

// ── MaxPool ──

public class MaxPoolOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "MaxPool";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var x = inputs[0];
        var kernelShape = attrs.ContainsKey("kernel_shape") ? ((long[])attrs["kernel_shape"]).Select(k => (int)k).ToArray() : new[] { 2, 2 };
        var strides = attrs.ContainsKey("strides") ? ((long[])attrs["strides"]).Select(s => (int)s).ToArray() : new[] { 1, 1 };
        var pads = attrs.ContainsKey("pads") ? ((long[])attrs["pads"]).Select(p => (int)p).ToArray() : new int[4];
        int outH = (x[2] + pads[0] + pads[2] - kernelShape[0]) / strides[0] + 1;
        int outW = (x[3] + pads[1] + pads[3] - kernelShape[1]) / strides[1] + 1;
        return new[] { new[] { x[0], x[1], outH, outW } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var x = ctx.Inputs[0];
        var ks = ctx.GetInts("kernel_shape"); int kH = ks.Length > 0 ? ks[0] : 2; int kW = ks.Length > 1 ? ks[1] : kH;
        var st = ctx.GetInts("strides"); int sH = st.Length > 0 ? st[0] : 1; int sW = st.Length > 1 ? st[1] : sH;
        var pa = ctx.GetInts("pads"); int pH = pa.Length > 0 ? pa[0] : 0; int pW = pa.Length > 1 ? pa[1] : 0;
        reg.Pooling.MaxPool2D(x.Data, ctx.Outputs[0].Data, x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3], kH, kW, sH, sW, pH, pW);
    }
}

// ── AveragePool ──

public class AveragePoolOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "AveragePool";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new MaxPoolOperator(reg).InferOutputShapes(inputs, attrs);
    public void Execute(OnnxOpContext ctx)
    {
        var x = ctx.Inputs[0];
        var ks = ctx.GetInts("kernel_shape"); int kH = ks[0]; int kW = ks.Length > 1 ? ks[1] : kH;
        var st = ctx.GetInts("strides"); int sH = st.Length > 0 ? st[0] : 1; int sW = st.Length > 1 ? st[1] : sH;
        var pa = ctx.GetInts("pads"); int pH = pa.Length > 0 ? pa[0] : 0; int pW = pa.Length > 1 ? pa[1] : 0;
        reg.Pooling.AvgPool2D(x.Data, ctx.Outputs[0].Data, x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3], kH, kW, sH, sW, pH, pW);
    }
}

// ── Resize ──

public class ResizeOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Resize";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] }; // Requires sizes input — resolved at runtime
    public void Execute(OnnxOpContext ctx)
    {
        // Simplified: NCHW bilinear resize using sizes from output shape
        var inShape = ctx.Inputs[0].Shape;
        var outShape = ctx.Outputs[0].Shape;
        int C = inShape[0] * inShape[1]; // N*C for batch
        int inH = inShape[2]; int inW = inShape[3];
        int outH = outShape[2]; int outW = outShape[3];
        // Use align_corners based on coordinate_transform_mode attribute
        var mode = ctx.GetString("coordinate_transformation_mode", "half_pixel");
        if (mode == "align_corners")
            reg.ElementWise.BilinearUpsampleAlignCorners(ctx.Inputs[0].Data, ctx.Outputs[0].Data, C, inH, inW, outH, outW);
        else
            reg.ElementWise.BilinearUpsample(ctx.Inputs[0].Data, ctx.Outputs[0].Data, C, inH, inW, outH, outW);
    }
}

// ── Pad ──

public class PadOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Pad";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Try to get pads from attributes (opset < 11)
        if (attrs.TryGetValue("pads", out var padsObj) && padsObj is long[] padsLong)
        {
            var shape = (int[])inputs[0].Clone();
            int rank = shape.Length;
            for (int i = 0; i < rank; i++)
                shape[i] += (int)padsLong[i] + (int)padsLong[rank + i];
            return new[] { shape };
        }
        // For opset >= 11, pads come from input[1] — resolved at runtime
        return new[] { inputs[0] };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var input = ctx.Inputs[0];
        int rank = input.Shape.Length;

        // Get pads: opset < 11 uses attribute, opset >= 11 uses tensor input[1]
        int[] pads;
        var attrPads = ctx.GetInts("pads");
        if (attrPads.Length > 0)
        {
            pads = attrPads;
        }
        else if (ctx.Inputs.Length > 1 && ctx.Inputs[1] != null)
        {
            // Read pads from pre-extracted constant values (no GPU→CPU readback)
            var preRead = ctx.TryGetInputValues(1);
            if (preRead != null)
            {
                pads = preRead.Select(v => (int)v).ToArray();
            }
            else
            {
                // Fallback for non-constant pads (shouldn't happen for typical models)
                pads = new int[ctx.Inputs[1].ElementCount];
            }
        }
        else
        {
            // No padding — just copy
            reg.ElementWise.Scale(input.Data, ctx.Outputs[0].Data, input.ElementCount, 1f);
            return;
        }

        // Get constant value (opset >= 11: input[2], else attribute)
        float constVal = 0f;
        if (ctx.Inputs.Length > 2 && ctx.Inputs[2] != null && ctx.Inputs[2].ElementCount > 0)
        {
            var preRead = ctx.TryGetInputValues(2);
            if (preRead != null && preRead.Length > 0)
                constVal = preRead[0];
        }

        // Get mode
        string modeStr = ctx.GetString("mode", "constant");
        int mode = modeStr switch
        {
            "constant" => 0,
            "edge" => 1,
            "reflect" => 2,
            _ => 0
        };

        reg.Pad.Forward(input.Data, ctx.Outputs[0].Data, input.Shape, pads, mode, constVal);
    }
}

// ── Split ──

public class SplitOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Split";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] }; // Multiple outputs — simplified
    public void Execute(OnnxOpContext ctx)
    {
        // Simplified: copy input to first output
        int copyCount = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount);
        reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, copyCount),
            ctx.Outputs[0].Data.SubView(0, copyCount), copyCount, 1f);
    }
}

// ── Slice ──

public class SliceOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Slice";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] }; // Dynamic — depends on starts/ends inputs
    public void Execute(OnnxOpContext ctx)
    {
        // Simplified: contiguous slice — copy the matching portion
        int copyCount = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount);
        reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, copyCount),
            ctx.Outputs[0].Data.SubView(0, copyCount), copyCount, 1f);
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
