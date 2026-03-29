using ILGPU;
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
        // Handle rank-1 vectors: treat as [1, K] or [K, 1]
        if (a.ElementCount == 0 || b.ElementCount == 0) return; // empty tensor
        int M = a.Rank >= 2 ? a.Shape[^2] : 1;
        int K = a.Shape[^1];
        int N = b.Rank >= 2 ? b.Shape[^1] : 1;
        if (M == 0 || K == 0 || N == 0) return; // degenerate dimensions

        // Check if weight B is quantized (Q4_0) — use fused dequant kernel
        string? bName = ctx.InputNames.Length > 1 ? ctx.InputNames[1] : null;
        if (bName != null && ctx.QuantizedWeights != null
            && ctx.QuantizedWeights.TryGetValue(bName, out var q4Data))
        {
            reg.FusedDequant.Forward(a.Data, q4Data, ctx.Outputs[0].Data, M, K, N);
            return;
        }

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
        // Clamp axis to valid range when it exceeds rank (e.g., ONNX compiled axis=-1
        // stored as positive value from higher-rank context, but runtime tensor is lower rank).
        // Still throw for genuinely invalid shapes (zero-rank, zero-dim).
        if (shape.Length == 0 || shape.Any(d => d <= 0))
            throw new InvalidOperationException(
                $"Softmax: invalid shape [{string.Join(",", shape)}] (rank={shape.Length})");
        if (axis >= shape.Length) axis = shape.Length - 1;
        if (axis < 0) axis = 0;

        // Copy input to output first
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, 1f);

        // ONNX opset 13+: Softmax operates on a SINGLE axis.
        // For shape [A, B, C, D] with axis=2: softmax over C for each (A*B) × D combination.
        // We reshape to [outer, axisDim, inner] and run softmax on rows of size axisDim.
        int outer = 1; for (int i = 0; i < axis; i++) outer *= shape[i];
        int axisDim = shape[axis];
        int inner = 1; for (int i = axis + 1; i < shape.Length; i++) inner *= shape[i];

        if (axisDim <= 0)
            throw new InvalidOperationException($"Softmax axis {axis} has dimension {axisDim} in shape [{string.Join(",", shape)}]. " +
                $"This indicates a shape inference bug upstream — a 0-dimension tensor should not reach Softmax.");

        if (inner == 1)
        {
            // Simple case: softmax over the last dim — standard row softmax
            reg.Softmax.Forward(ctx.Outputs[0].Data, outer, axisDim);
        }
        else
        {
            // General case: axis is not the last dim.
            // Transpose so axis becomes last: [outer, axisDim, inner] → [outer, inner, axisDim]
            // Then softmax over rows of axisDim, then transpose back.
            int totalElems = ctx.Inputs[0].ElementCount;
            var transposed = ctx.Pool.Rent(new[] { totalElems });

            // Transpose [outer, axisDim, inner] → [outer, inner, axisDim]
            // Input layout:  [o][a][i] at offset o*(axisDim*inner) + a*inner + i
            // Output layout: [o][i][a] at offset o*(inner*axisDim) + i*axisDim + a
            reg.Transpose.Transpose(ctx.Outputs[0].Data, transposed.Data,
                new[] { outer, axisDim, inner }, new[] { 0, 2, 1 });

            // Softmax over rows of axisDim (now contiguous)
            reg.Softmax.Forward(transposed.Data, outer * inner, axisDim);

            // Transpose back: [outer, inner, axisDim] → [outer, axisDim, inner]
            reg.Transpose.Transpose(transposed.Data, ctx.Outputs[0].Data,
                new[] { outer, inner, axisDim }, new[] { 0, 2, 1 });
        }
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
        var x = inputs[0]; var w = inputs[1];
        var strides = attrs.ContainsKey("strides") ? ((long[])attrs["strides"]).Select(s => (int)s).ToArray() : new[] { 1, 1 };
        int outC = w[0];

        // Handle auto_pad (SAME_UPPER/SAME_LOWER from TFLite models)
        string autoPad = attrs.ContainsKey("auto_pad") ? attrs["auto_pad"].ToString()! : "NOTSET";
        int[] pads;
        if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER")
        {
            if (x.Length == 3)
            {
                int outL = (int)Math.Ceiling((double)x[2] / strides[0]);
                int padTotal = Math.Max(0, (outL - 1) * strides[0] + w[2] - x[2]);
                pads = new[] { padTotal / 2, padTotal - padTotal / 2 };
            }
            else
            {
                int sH = strides[0], sW = strides.Length > 1 ? strides[1] : 1;
                int outH = (int)Math.Ceiling((double)x[2] / sH);
                int outW = (int)Math.Ceiling((double)x[3] / sW);
                int padH = Math.Max(0, (outH - 1) * sH + w[2] - x[2]);
                int padW = Math.Max(0, (outW - 1) * sW + w[3] - x[3]);
                pads = autoPad == "SAME_UPPER"
                    ? new[] { padH / 2, padW / 2, padH - padH / 2, padW - padW / 2 }
                    : new[] { padH - padH / 2, padW - padW / 2, padH / 2, padW / 2 };
            }
        }
        else
        {
            pads = attrs.ContainsKey("pads") ? ((long[])attrs["pads"]).Select(p => (int)p).ToArray() : new int[x.Length == 3 ? 2 : 4];
        }

        if (x.Length == 3)
        {
            int kL = w[2];
            var dilations = attrs.ContainsKey("dilations") ? ((long[])attrs["dilations"]).Select(d => (int)d).ToArray() : new[] { 1 };
            int dilation = dilations.Length > 0 ? dilations[0] : 1;
            int outL = (x[2] + (pads.Length >= 2 ? pads[0] + pads[1] : 0) - dilation * (kL - 1) - 1) / strides[0] + 1;
            return new[] { new[] { x[0], outC, outL } };
        }
        else
        {
            int kH = w[2]; int kW = w[3];
            int outH = (x[2] + pads[0] + (pads.Length > 2 ? pads[2] : 0) - kH) / strides[0] + 1;
            int outW = (x[3] + (pads.Length > 1 ? pads[1] : 0) + (pads.Length > 3 ? pads[3] : 0) - kW) / (strides.Length > 1 ? strides[1] : 1) + 1;
            return new[] { new[] { x[0], outC, outH, outW } };
        }
    }
    public void Execute(OnnxOpContext ctx)
    {
        var x = ctx.Inputs[0]; var w = ctx.Inputs[1];
        var fmt = ctx.Format;
        var strides = ctx.GetInts("strides"); int stride = strides.Length > 0 ? strides[0] : 1;

        // Handle auto_pad (SAME_UPPER/SAME_LOWER from TFLite models)
        var autoPad = ctx.Attributes.TryGetValue("auto_pad", out var ap) ? ap.ToString()! : "NOTSET";
        int pad;
        if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER")
        {
            int inH = x.Shape.Length >= 4 ? x.Shape[LayoutHelper.HeightAxis(fmt)] : (x.Shape.Length >= 3 ? x.Shape[2] : 1);
            int kH = w.Shape.Length >= 4 ? w.Shape[LayoutHelper.HeightAxis(fmt)] : (w.Shape.Length >= 3 ? w.Shape[2] : 1);
            int padH = Math.Max(0, ((int)Math.Ceiling((double)inH / stride) - 1) * stride + kH - inH);
            pad = autoPad == "SAME_UPPER" ? padH / 2 : padH - padH / 2;
        }
        else
        {
            var pads = ctx.GetInts("pads");
            pad = pads.Length > 0 ? pads[0] : 0;
        }
        int group = ctx.GetInt("group", 1);
        var (_, inC_from_x, _, _) = x.Shape.Length >= 4 ? LayoutHelper.GetDims(x.Shape, fmt) : (1, x.Shape.Length > 1 ? x.Shape[1] : 1, 1, 1);
        // group = -1 is the TFLite depthwise sentinel — resolve to inC
        if (group == -1) group = inC_from_x;
        int outC = w.Shape[0];

        // Always provide a valid bias buffer (zero-filled if no bias input)
        ArrayView1D<float, Stride1D.Dense> bias;
        if (ctx.Inputs.Length > 2 && ctx.Inputs[2] != null)
        {
            bias = ctx.Inputs[2].Data;
        }
        else
        {
            // Upload fresh zeros — Pool.Rent reuses buffers with stale data
            bias = ctx.Pool.AllocatePermanent(new float[outC], new[] { outC }, "_conv_zero_bias").Data;
        }

        // (Debug diagnostics removed)

        if (x.Shape.Length == 3)
        {
            // Conv1D: [N, C, L]
            int inC = x.Shape[1]; int inL = x.Shape[2];
            int kL = w.Shape[2];
            var dilations = ctx.GetInts("dilations"); int dilation = dilations.Length > 0 ? dilations[0] : 1;
            reg.Conv1D.Forward(x.Data, w.Data, bias, ctx.Outputs[0].Data,
                inC, inL, outC, kL, stride, pad, dilation, group);
        }
        else
        {
            // Conv2D: layout-aware dim extraction
            var (_, inC, inH, inW) = LayoutHelper.GetDims(x.Shape, fmt);
            var (_, _, kH, kW) = LayoutHelper.GetWeightDims(w.Shape, fmt);

            if (group == inC && (group == outC || outC == 1))
            {
                // Depthwise conv: group=inC, each channel convolved independently.
                // outC may be 1 for TFLite depthwise (weight shape [1,kH,kW,C] transposed).
                reg.Conv2D.ForwardDepthwise(x.Data, w.Data, bias, ctx.Outputs[0].Data,
                    inC, inH, inW, kH, kW, stride, pad);
            }
            else if (group == 1)
            {
                reg.Conv2D.Forward(x.Data, w.Data, bias, ctx.Outputs[0].Data,
                    inC, inH, inW, outC, kH, kW, stride, pad);
            }
            else if (group > 1 && inC % group == 0 && outC % group == 0)
            {
                // General grouped convolution: split into groups, conv each, concat
                int inCPerGroup = inC / group;
                int outCPerGroup = outC / group;
                for (int g = 0; g < group; g++)
                {
                    int inOffset = g * inCPerGroup * inH * inW;
                    int wOffset = g * outCPerGroup * inCPerGroup * kH * kW;
                    int outOffset = g * outCPerGroup * ctx.Outputs[0].Shape[2] * ctx.Outputs[0].Shape[3];
                    // Use standard conv for each group slice
                    reg.Conv2D.Forward(
                        x.Data.SubView(inOffset, inCPerGroup * inH * inW),
                        w.Data.SubView(wOffset, outCPerGroup * inCPerGroup * kH * kW),
                        bias.SubView(g * outCPerGroup, outCPerGroup),
                        ctx.Outputs[0].Data.SubView(outOffset, outCPerGroup * ctx.Outputs[0].Shape[2] * ctx.Outputs[0].Shape[3]),
                        inCPerGroup, inH, inW, outCPerGroup, kH, kW, stride, pad);
                }
            }
            else
            {
                // Group doesn't evenly divide channels — likely shape inference error
                throw new NotSupportedException($"Conv with group={group} (inC={inC}, outC={outC}) not supported — group must divide both inC and outC");
            }
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
        var input = ctx.Inputs[0];
        int axis = ctx.GetInt("axis", 0);
        if (axis < 0) axis += input.Shape.Length;
        if (axis < 0 || axis >= input.Shape.Length)
            throw new InvalidOperationException(
                $"ArgMax axis {axis} out of range for shape [{string.Join(",", input.Shape)}] (rank={input.Shape.Length})");

        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= input.Shape[i];
        int axisSize = input.Shape[axis];
        int innerSize = 1;
        for (int i = axis + 1; i < input.Shape.Length; i++) innerSize *= input.Shape[i];

        // GPU ArgMax kernel — works on all backends including WebGPU/Wasm
        reg.ElementWise.ArgMax(input.Data, ctx.Outputs[0].Data, outerSize, axisSize, innerSize);
        return;

        // Legacy CPU fallback below (unreachable, kept for reference)
        #pragma warning disable CS0162
        int total = input.ElementCount;
        var data = ctx.TryGetInputValues(0);
        if (data == null || data.Length != total)
        {
            try
            {
                data = new float[total];
                input.Data.SubView(0, total).CopyToCPU(data);
            }
            catch (NotSupportedException)
            {
                throw new NotSupportedException(
                    $"ArgMax requires CPU readback but this backend doesn't support synchronous copies.");
            }
        }

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

    // GPU GatherND kernel: each thread copies one element of the output.
    // params: [lastIdxDim, sliceSize, dataTotal, strides[0], strides[1], ...]
    private MemoryBuffer1D<int, Stride1D.Dense>? _lastParamsBuf;
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // data
        ArrayView1D<float, Stride1D.Dense>,  // indices
        ArrayView1D<float, Stride1D.Dense>,  // output
        ArrayView1D<int, Stride1D.Dense>>?   // params
        _gatherNDKernel;

    private static void GatherNDImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int lastIdxDim = p[0];
        int sliceSize = p[1];
        int dataTotal = p[2];

        // Which slice and element within slice
        int sliceIdx = idx / sliceSize;
        int elemInSlice = idx % sliceSize;

        // Compute flat offset from multi-dimensional index
        int flatOffset = 0;
        for (int d = 0; d < lastIdxDim; d++)
        {
            int dimIdx = (int)indices[sliceIdx * lastIdxDim + d];
            flatOffset += dimIdx * p[3 + d]; // strides[d]
        }

        int srcIdx = flatOffset + elemInSlice;
        output[idx] = srcIdx >= 0 && srcIdx < dataTotal ? data[srcIdx] : 0f;
    }
    public void Execute(OnnxOpContext ctx)
    {
        var data = ctx.Inputs[0];
        var indices = ctx.Inputs[1];
        int batchDims = ctx.GetInt("batch_dims", 0);

        int dataTotal = data.ElementCount;
        int idxTotal = indices.ElementCount;
        int outputSize = ctx.Outputs[0].ElementCount;

        // Try to get indices from runtime constants (avoids GPU→CPU sync on WebGPU)
        var idxArr = ctx.TryGetInputValues(1);

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

        if (idxArr != null)
        {
            // GPU path: indices on CPU, data stays on GPU. Copy slices via SubView.
            for (int s = 0; s < numSlices && s * sliceSize < outputSize; s++)
            {
                int offset = 0;
                for (int d = 0; d < lastIdxDim; d++)
                    offset += (int)idxArr[s * lastIdxDim + d] * strides[d];

                int copyLen = Math.Min(sliceSize, outputSize - s * sliceSize);
                int dstOffset = s * sliceSize;
                if (offset >= 0 && copyLen > 0
                    && offset + copyLen <= (int)data.Data.Length
                    && dstOffset + copyLen <= (int)ctx.Outputs[0].Data.Length)
                {
                    reg.ElementWise.Scale(
                        data.Data.SubView(offset, copyLen),
                        ctx.Outputs[0].Data.SubView(dstOffset, copyLen),
                        copyLen, 1f);
                }
            }
        }
        else
        {
            // GPU-only path: both data and indices stay on GPU.
            // Upload strides as a params buffer, dispatch one thread per output element.
            // Each thread reads its index from the indices tensor.
            // params: [lastIdxDim, sliceSize, dataTotal, strides[0], strides[1], ...]
            var paramsArr = new int[3 + strides.Length];
            paramsArr[0] = lastIdxDim;
            paramsArr[1] = sliceSize;
            paramsArr[2] = dataTotal;
            for (int i = 0; i < strides.Length; i++) paramsArr[3 + i] = strides[i];
            _lastParamsBuf?.Dispose();
            _lastParamsBuf = reg.Accelerator.Allocate1D(paramsArr);

            _gatherNDKernel ??= reg.Accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(GatherNDImpl);
            _gatherNDKernel(outputSize, data.Data, indices.Data, ctx.Outputs[0].Data, _lastParamsBuf.View);

            // Legacy CPU fallback below (unreachable on WebGPU, kept for reference)
            #pragma warning disable CS0162
            return;
            var dataArr = ctx.TryGetInputValues(0);
            if (dataArr == null || dataArr.Length != dataTotal)
            {
                dataArr = new float[dataTotal];
                data.Data.SubView(0, dataTotal).CopyToCPU(dataArr);
            }
            var idxArrFallback = new float[idxTotal];
            indices.Data.SubView(0, idxTotal).CopyToCPU(idxArrFallback);

            var result = new float[outputSize];
            for (int s = 0; s < numSlices && s * sliceSize < outputSize; s++)
            {
                int offset = 0;
                for (int d = 0; d < lastIdxDim; d++)
                    offset += (int)idxArrFallback[s * lastIdxDim + d] * strides[d];

                for (int j = 0; j < sliceSize && s * sliceSize + j < outputSize; j++)
                    result[s * sliceSize + j] = (offset + j < dataTotal) ? dataArr[offset + j] : 0f;
            }

            var temp = ctx.Pool.AllocatePermanent(result, ctx.Outputs[0].Shape);
            reg.ElementWise.Scale(temp.Data, ctx.Outputs[0].Data, outputSize, 1f);
        }
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
        if (x.Shape.Length < 4)
            throw new InvalidOperationException(
                $"ConvTranspose expects 4D input [N,C,H,W], got shape [{string.Join(",", x.Shape)}] (rank={x.Shape.Length}). " +
                $"This may be caused by an upstream Resize/Expand with unresolved dynamic shapes.");
        var strides = ctx.GetInts("strides"); int stride = strides.Length > 0 ? strides[0] : 1;
        var pads = ctx.GetInts("pads"); int pad = pads.Length > 0 ? pads[0] : 0;
        int inC = x.Shape[1]; int inH = x.Shape[2]; int inW = x.Shape[3];
        int outC = w.Shape[1]; int kH = w.Shape[2]; int kW = w.Shape[3];
        // Always provide a valid bias buffer — no conditional branch in kernel.
        // ANGLE's HLSL optimizer changes FP evaluation when a branch precedes
        // the accumulation loop, causing 0.009 error on WebGL.
        var bias = ctx.Inputs.Length > 2 && ctx.Inputs[2] != null
            ? ctx.Inputs[2].Data
            : ctx.Pool.Rent(new[] { outC }, "_conv_zero_bias").Data;
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
        var normalizedAxes = axes.Length > 0
            ? axes.Select(a => (int)(a < 0 ? a + shape.Length : a)).OrderBy(a => a).ToArray()
            : new[] { shape.Length - 1 };

        // Compute outer (dims before first axis), reduce (product of all axes dims), inner (dims after last axis)
        // This works correctly when axes are contiguous (e.g., [2,3] for spatial reduction)
        int firstAxis = normalizedAxes[0];
        int lastAxis = normalizedAxes[^1];
        int outer = 1; for (int i = 0; i < firstAxis; i++) outer *= shape[i];
        int reduce = 1; for (int i = firstAxis; i <= lastAxis; i++) reduce *= shape[i];
        int inner = 1; for (int i = lastAxis + 1; i < shape.Length; i++) inner *= shape[i];
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
        var normalizedAxes = axes.Length > 0
            ? axes.Select(a => (int)(a < 0 ? a + shape.Length : a)).OrderBy(a => a).ToArray()
            : new[] { shape.Length - 1 };

        int firstAxis = normalizedAxes[0];
        int lastAxis = normalizedAxes[^1];
        int outer = 1; for (int i = 0; i < firstAxis; i++) outer *= shape[i];
        int reduce = 1; for (int i = firstAxis; i <= lastAxis; i++) reduce *= shape[i];
        int inner = 1; for (int i = lastAxis + 1; i < shape.Length; i++) inner *= shape[i];
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
        var normalizedAxes = axes.Length > 0
            ? axes.Select(a => (int)(a < 0 ? a + shape.Length : a)).OrderBy(a => a).ToArray()
            : new[] { shape.Length - 1 };

        int firstAxis = normalizedAxes[0];
        int lastAxis = normalizedAxes[^1];
        int outer = 1; for (int i = 0; i < firstAxis; i++) outer *= shape[i];
        int reduce = 1; for (int i = firstAxis; i <= lastAxis; i++) reduce *= shape[i];
        int inner = 1; for (int i = lastAxis + 1; i < shape.Length; i++) inner *= shape[i];
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
        var normalizedAxes = axes.Length > 0
            ? axes.Select(a => (int)(a < 0 ? a + shape.Length : a)).OrderBy(a => a).ToArray()
            : new[] { shape.Length - 1 };

        int firstAxis = normalizedAxes[0];
        int lastAxis = normalizedAxes[^1];
        int outer = 1; for (int i = 0; i < firstAxis; i++) outer *= shape[i];
        int reduce = 1; for (int i = firstAxis; i <= lastAxis; i++) reduce *= shape[i];
        int inner = 1; for (int i = lastAxis + 1; i < shape.Length; i++) inner *= shape[i];
        reg.Reductions.ReduceMin(ctx.Inputs[0].Data, ctx.Outputs[0].Data, outer, reduce, inner);
    }
}

// ── Gather ──

public class GatherOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Gather";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // ONNX Gather spec: output shape = data.shape[:axis] + indices.shape + data.shape[axis+1:]
        var dataShape = inputs[0];
        var idxShape = inputs[1];
        int axis = attrs.ContainsKey("axis") ? Convert.ToInt32(attrs["axis"]) : 0;
        if (axis < 0) axis += dataShape.Length;
        // Clamp axis for rank-1 tensors (constant-folded Shape→Gather chains)
        if (axis >= dataShape.Length) axis = Math.Max(0, dataShape.Length - 1);

        // For multi-dimensional data with single-element [1] indices, treat the index
        // as scalar [] to avoid adding an extra dimension. This is critical for:
        // - Token extraction: Gather(data=[1,seq,hidden], idx=[0], axis=1) → [1,hidden] not [1,1,hidden]
        // - Attention reshaping: prevents 5D shapes like [1,1,257,257,0] from cascading
        // Shape extraction on 1D vectors (axis=0 on [N]) keeps [1] for Concat compatibility.
        var effectiveIdxShape = (dataShape.Length > 1 && idxShape.Length == 1 && idxShape[0] == 1)
            ? Array.Empty<int>() : idxShape;

        var outShape = new List<int>();
        // Dims before axis
        for (int i = 0; i < axis; i++) outShape.Add(dataShape[i]);
        // Index shape dims (replaces the gathered axis)
        foreach (var d in effectiveIdxShape) outShape.Add(d);
        // Dims after axis
        for (int i = axis + 1; i < dataShape.Length; i++) outShape.Add(dataShape[i]);

        return new[] { outShape.ToArray() };
    }
    public void Execute(OnnxOpContext ctx)
    {
        if (ctx.Inputs.Length < 2)
            throw new InvalidOperationException($"Gather requires 2 inputs, got {ctx.Inputs.Length}");

        var data = ctx.Inputs[0]; var indices = ctx.Inputs[1];
        int rawAxis = ctx.GetInt("axis", 0);
        int axis = rawAxis < 0 ? rawAxis + data.Shape.Length : rawAxis;
        // Clamp axis for constant-folded tensors: Shape→Gather chains produce rank-1
        // vectors where the original axis referenced a higher-rank tensor's dimension.
        // The constant-folded result is flat, so axis must be 0.
        if (axis >= data.Shape.Length)
            axis = 0;

        // Get index values from pre-read constants (avoids GPU→CPU readback)
        var idxFloats = ctx.TryGetInputValues(1);
        if (idxFloats == null && axis == 0)
        {
            // GPU-side Gather: indices are runtime tensors on GPU (e.g., NLP token IDs).
            // Use float-index kernel that casts to int inside the GPU kernel.
            int numIdx = indices.ElementCount;
            int innerSize = 1;
            for (int i = 1; i < data.Shape.Length; i++) innerSize *= data.Shape[i];
            int dataRows = data.Shape[0];
            if (numIdx <= 0 || innerSize <= 0)
                throw new InvalidOperationException($"Gather axis=0 invalid dims: numIdx={numIdx} innerSize={innerSize} dataRows={dataRows} data=[{string.Join(",", data.Shape)}] indices=[{string.Join(",", indices.Shape)}] output=[{string.Join(",", ctx.Outputs[0].Shape)}]");
            reg.Gather.GatherAxis0Float(data.Data, indices.Data, ctx.Outputs[0].Data,
                numIdx, innerSize, dataRows);
            return;
        }
        else if (idxFloats == null)
        {
            // Non-axis-0: use GPU kernel with runtime indices
            int numIdx = indices.ElementCount;
            int axisSize = data.Shape[axis];
            int innerSize = 1;
            for (int i = axis + 1; i < data.Shape.Length; i++) innerSize *= data.Shape[i];
            int outerSize = 1;
            for (int i = 0; i < axis; i++) outerSize *= data.Shape[i];
            reg.Gather.GatherGenericFloat(data.Data, indices.Data, ctx.Outputs[0].Data,
                numIdx, innerSize, outerSize, axisSize);
            return;
        }

        int numIdx2 = idxFloats.Length;
        int innerSize2 = 1;
        for (int i = axis + 1; i < data.Shape.Length; i++) innerSize2 *= data.Shape[i];
        int outerSize2 = 1;
        for (int i = 0; i < axis; i++) outerSize2 *= data.Shape[i];
        int axisSize2 = data.Shape[axis];

        // CPU-side Gather with pre-read indices (for constant/small index tensors)
        for (int o = 0; o < outerSize2; o++)
        {
            for (int idx = 0; idx < numIdx2; idx++)
            {
                int srcIdx = (int)idxFloats[idx];
                if (srcIdx < 0) srcIdx += axisSize2;
                if (srcIdx < 0 || srcIdx >= axisSize2) srcIdx = 0;

                int srcOffset = (o * axisSize2 + srcIdx) * innerSize2;
                int dstOffset = (o * numIdx2 + idx) * innerSize2;
                // Bounds check: skip if source or dest would exceed buffer
                if (srcOffset + innerSize2 > data.ElementCount ||
                    dstOffset + innerSize2 > ctx.Outputs[0].ElementCount)
                    continue;
                reg.ElementWise.Scale(
                    data.Data.SubView(srcOffset, innerSize2),
                    ctx.Outputs[0].Data.SubView(dstOffset, innerSize2),
                    innerSize2, 1f);
            }
        }
    }
}

// ── ScatterND ──

public class ScatterNDOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ScatterND";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Output shape = data shape (scatter updates into a copy of data)
        return new[] { inputs[0] };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var data = ctx.Inputs[0]; var indices = ctx.Inputs[1]; var updates = ctx.Inputs[2];
        var output = ctx.Outputs[0];

        // Read reduction mode (ONNX opset 16+: none, add, mul)
        string reduction = "none";
        if (ctx.Attributes.TryGetValue("reduction", out var redObj) && redObj is string redStr)
            reduction = redStr.ToLowerInvariant();
        if (reduction != "none")
            throw new NotSupportedException($"ScatterND: reduction='{reduction}' not yet implemented");

        // Copy data to output first
        reg.ElementWise.Scale(data.Data, output.Data, data.ElementCount, 1f);

        // Read indices from GPU (small tensor, constant in most models)
        var idxFloats = ctx.TryGetInputValues(1);
        if (idxFloats == null)
            throw new NotSupportedException("ScatterND: runtime indices not pre-read — add to ConstantData");

        // ScatterND: indices is [num_updates, index_depth] where index_depth indexes into data dims
        var idxShape = indices.Shape;
        int numUpdates = 1;
        for (int i = 0; i < idxShape.Length - 1; i++) numUpdates *= idxShape[i];
        int indexDepth = idxShape[^1];

        if (indexDepth > data.Shape.Length)
        {
            // Shape mismatch from compile-time inference — output already has data copy
            return;
        }

        // Compute element size for the slice that each update covers
        int sliceSize = 1;
        for (int i = indexDepth; i < data.Shape.Length; i++) sliceSize *= data.Shape[i];

        // For each update, compute flat offset into data and copy update slice
        for (int u = 0; u < numUpdates; u++)
        {
            // Compute flat offset from multi-dimensional index
            int flatOffset = 0;
            int stride = data.ElementCount;
            for (int d = 0; d < indexDepth; d++)
            {
                stride /= data.Shape[d];
                int idx = (int)idxFloats[u * indexDepth + d];
                if (idx < 0) idx += data.Shape[d];
                if (idx < 0 || idx >= data.Shape[d])
                {
                    // OOB index — skip this update rather than crashing.
                    // Can happen when compiled shapes don't match runtime shapes (e.g., DA3 subgraph).
                    flatOffset = -1;
                    break;
                }
                flatOffset += idx * stride;
            }

            if (flatOffset < 0 || flatOffset + sliceSize > output.ElementCount)
                continue; // Skip OOB scatter

            // Copy update slice to output at computed offset
            reg.ElementWise.Scale(
                updates.Data.SubView(u * sliceSize, sliceSize),
                output.Data.SubView(flatOffset, sliceSize),
                sliceSize, 1f);
        }
    }
}

// ── Concat ──

public class ConcatOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Concat";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        if (inputs.Length == 0 || inputs[0].Length == 0)
            return new[] { inputs.Length > 0 ? inputs[0] : Array.Empty<int>() };
        int axis = attrs.ContainsKey("axis") ? Convert.ToInt32(attrs["axis"]) : 0;
        if (axis < 0) axis += inputs[0].Length;
        if (axis < 0 || axis >= inputs[0].Length) return new[] { inputs[0] };
        var outShape = (int[])inputs[0].Clone();
        for (int i = 1; i < inputs.Length; i++)
            if (inputs[i].Length > axis) outShape[axis] += inputs[i][axis];
        return new[] { outShape };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // Normalize scalar inputs: treat rank-0 tensors as [1] for concat purposes.
        // Common in Shape→Gather→Unsqueeze→Concat chains where Gather outputs a scalar.
        for (int n = 0; n < ctx.Inputs.Length; n++)
        {
            if (ctx.Inputs[n].Shape.Length == 0 && ctx.Inputs[n].ElementCount > 0)
                ctx.Inputs[n] = new Tensors.Tensor(ctx.Inputs[n].Data, new[] { ctx.Inputs[n].ElementCount }, ctx.Inputs[n].Name);
        }

        int axis = ctx.GetInt("axis", 0);
        if (axis < 0) axis += ctx.Inputs[0].Shape.Length;

        // General concat: copy each input's blocks to the output at the correct offset.
        // For axis=1 (NCHW channel concat): outer=N, concat dim=C, inner=H*W
        var shape0 = ctx.Inputs[0].Shape;
        // Handle rank mismatch: if axis >= some input's rank, treat as flat concat (axis=0)
        // This handles CLIP's pattern where [768] and [1,4,...] are concatenated on axis=1
        bool rankMismatch = ctx.Inputs.Any(t => axis >= t.Shape.Length);
        if (rankMismatch && axis > 0)
            axis = 0; // Fall back to flat concat

        int outer = 1; for (int i = 0; i < axis; i++) outer *= shape0[i];
        int inner = 1; for (int i = axis + 1; i < shape0.Length; i++) inner *= shape0[i];

        int outOffset = 0;
        int totalConcatDim = 0;
        for (int n = 0; n < ctx.Inputs.Length; n++)
        {
            if (axis >= ctx.Inputs[n].Shape.Length)
                throw new InvalidOperationException(
                    $"Concat axis={axis} out of range for input[{n}] shape=[{string.Join(",", ctx.Inputs[n].Shape)}] (rank={ctx.Inputs[n].Shape.Length}). " +
                    $"All inputs: [{string.Join("; ", ctx.Inputs.Select(t => $"[{string.Join(",", t.Shape)}]"))}]");
            totalConcatDim += ctx.Inputs[n].Shape[axis];
        }

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

// ── GroupNormalization ──

public class GroupNormOperator : IOnnxOperator
{
    private readonly Kernels.GroupNormKernel _kernel;
    public GroupNormOperator(Accelerator accelerator) => _kernel = new Kernels.GroupNormKernel(accelerator);
    public string OpType => "GroupNormalization";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        var shape = ctx.Inputs[0].Shape;
        int N = shape[0]; int C = shape[1];
        int spatial = 1; for (int i = 2; i < shape.Length; i++) spatial *= shape[i];
        int numGroups = ctx.GetInt("num_groups", 32);
        float eps = ctx.GetFloat("epsilon", 1e-5f);
        _kernel.Forward(ctx.Inputs[0].Data, ctx.Outputs[0].Data,
            ctx.Inputs[1].Data, ctx.Inputs[2].Data,
            N, C, spatial, numGroups, eps);
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
        var a = inputs[0]; var b = inputs[1];
        int N = transB != 0 ? b[0] : b[^1];
        // Gemm: A[M,K] @ B[K,N] + C → [M,N]
        // For 3D+ inputs, preserve leading dims (batch/seq) instead of flattening.
        // The Execute method flattens internally but the output shape should match the model's expectations.
        if (a.Length > 2)
        {
            // Output: [...leading_dims, N]
            var outShape = new int[a.Length];
            for (int i = 0; i < a.Length - 1; i++) outShape[i] = a[i];
            outShape[a.Length - 1] = N;
            return new[] { outShape };
        }
        int M = transA != 0 ? a[^1] : a[0];
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

        // Handle higher-rank inputs (e.g., [1,1,768] from Gather with axis > 0).
        // Flatten to 2D by treating all dims except last as batch/M, last dim as K.
        int M, K;
        if (a.Shape.Length > 2)
        {
            K = a.Shape[^1];
            M = a.ElementCount / K;
        }
        else
        {
            M = a.Shape[0]; K = a.Shape[1];
        }

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
    {
        int axis = attrs.ContainsKey("axis") ? Convert.ToInt32(attrs["axis"]) : 0;
        if (axis < 0) axis += inputs[0].Length;
        var inShape = inputs[0];

        // Guard: if axis is out of bounds, throw with full context (don't let fallback hide the bug)
        if (axis >= inShape.Length)
            throw new InvalidOperationException(
                $"Split.InferOutputShapes: axis={axis} >= rank={inShape.Length} for input shape=[{string.Join(",", inShape)}]. " +
                $"Attrs: [{string.Join(",", attrs.Select(kv => $"{kv.Key}={kv.Value}"))}]");

        // Get split sizes from attribute (opset < 13) or default to equal splits
        int[] splitSizes;
        if (attrs.TryGetValue("split", out var splitObj) && splitObj is long[] splitLongs)
        {
            splitSizes = splitLongs.Select(s => (int)s).ToArray();
        }
        else
        {
            // Default: split into equal parts. Use num_outputs attr or input[1] length.
            int numOutputs = attrs.ContainsKey("num_outputs") ? Convert.ToInt32(attrs["num_outputs"]) : 2;
            int dimSize = inShape[axis];
            int partSize = dimSize / numOutputs;
            splitSizes = Enumerable.Repeat(partSize, numOutputs).ToArray();
            // Handle remainder
            if (dimSize % numOutputs != 0)
                splitSizes[numOutputs - 1] = dimSize - partSize * (numOutputs - 1);
        }

        var result = new int[splitSizes.Length][];
        for (int i = 0; i < splitSizes.Length; i++)
        {
            result[i] = (int[])inShape.Clone();
            result[i][axis] = splitSizes[i];
        }
        return result;
    }
    public void Execute(OnnxOpContext ctx)
    {
        int axis = ctx.GetInt("axis", 0);
        if (axis < 0) axis += ctx.Inputs[0].Shape.Length;
        var inShape = ctx.Inputs[0].Shape;

        // Guard: validate axis is in bounds
        if (axis >= inShape.Length)
            throw new InvalidOperationException(
                $"Split: axis={axis} but input shape=[{string.Join(",", inShape)}] (rank={inShape.Length}). " +
                $"Inputs={ctx.Inputs.Length}, Outputs={ctx.Outputs.Length}. " +
                $"Input shapes: {string.Join("; ", ctx.Inputs.Select(t => $"[{string.Join(",", t.Shape)}]"))}. " +
                $"Output shapes: {string.Join("; ", ctx.Outputs.Select(t => $"[{string.Join(",", t.Shape)}]"))}. " +
                $"InputNames: {string.Join(",", ctx.InputNames)}");

        // Compute strides for the split
        int outer = 1; for (int i = 0; i < axis; i++) outer *= inShape[i];
        int inner = 1; for (int i = axis + 1; i < inShape.Length; i++) inner *= inShape[i];
        int axisDim = inShape[axis];

        // Split into each output
        int axisOffset = 0;
        for (int outIdx = 0; outIdx < ctx.Outputs.Length; outIdx++)
        {
            if (ctx.Outputs[outIdx] == null) continue;
            var outShape = ctx.Outputs[outIdx].Shape;
            if (axis >= outShape.Length)
                throw new InvalidOperationException(
                    $"Split axis={axis} but output[{outIdx}] shape=[{string.Join(",", outShape)}] (len={outShape.Length}). " +
                    $"Input shape=[{string.Join(",", inShape)}], {ctx.Outputs.Length} outputs.");
            int splitSize = outShape[axis];
            int blockSize = splitSize * inner;

            for (int o = 0; o < outer; o++)
            {
                int srcOffset = o * axisDim * inner + axisOffset * inner;
                int dstOffset = o * blockSize;
                int copyLen = Math.Min(blockSize, ctx.Outputs[outIdx].ElementCount - dstOffset);
                if (copyLen <= 0) continue;

                reg.ElementWise.Scale(
                    ctx.Inputs[0].Data.SubView(srcOffset, copyLen),
                    ctx.Outputs[outIdx].Data.SubView(dstOffset, copyLen),
                    copyLen, 1f);
            }
            axisOffset += splitSize;
        }
    }
}

// ── Slice ──

public class SliceOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Slice";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Try to compute output shape from attributes (opset < 10)
        if (attrs.TryGetValue("starts", out var startsObj) && startsObj is long[] starts
            && attrs.TryGetValue("ends", out var endsObj) && endsObj is long[] ends)
        {
            var axes = attrs.TryGetValue("axes", out var axObj) && axObj is long[] ax
                ? ax.Select(a => (int)a).ToArray()
                : Enumerable.Range(0, starts.Length).ToArray();
            var steps = attrs.TryGetValue("steps", out var stObj) && stObj is long[] st
                ? st.Select(s => (int)s).ToArray()
                : Enumerable.Repeat(1, starts.Length).ToArray();

            var outShape = (int[])inputs[0].Clone();
            for (int idx = 0; idx < axes.Length; idx++)
            {
                int dim = axes[idx] < 0 ? axes[idx] + outShape.Length : axes[idx];
                int s2 = (int)starts[idx]; int e2 = (int)ends[idx]; int st2 = steps[idx];
                if (s2 < 0) s2 += outShape[dim];
                if (e2 < 0) e2 += outShape[dim];
                s2 = Math.Clamp(s2, 0, outShape[dim]);
                e2 = Math.Clamp(e2, 0, outShape[dim]);
                outShape[dim] = (e2 - s2 + st2 - 1) / st2;
            }
            return new[] { outShape };
        }
        return new[] { inputs[0] }; // Dynamic — resolved at runtime
    }
    public void Execute(OnnxOpContext ctx)
    {
        var input = ctx.Inputs[0];
        var inShape = input.Shape;
        int rank = inShape.Length;

        // Resolve starts, ends, axes, steps — priority order:
        // 1. Compiler-resolved attributes (_resolved_starts etc.) — most reliable
        // 2. Pre-read constant values from tensor inputs (opset >= 10)
        // 3. Attributes (opset < 10)
        // 4. Full copy fallback
        int[] starts, ends, axes, steps;

        var resolvedStarts = ctx.GetInts("_resolved_starts");
        var resolvedEnds = ctx.GetInts("_resolved_ends");

        if (resolvedStarts.Length > 0 && resolvedEnds.Length > 0)
        {
            // Path 1: compiler resolved at compile time — handles opset >= 10 with constant params
            starts = new int[rank]; ends = new int[rank]; steps = new int[rank];
            for (int d = 0; d < rank; d++) { starts[d] = 0; ends[d] = inShape[d]; steps[d] = 1; }
            var rAxes = ctx.GetInts("_resolved_axes");
            var rSteps = ctx.GetInts("_resolved_steps");
            for (int ri = 0; ri < rAxes.Length; ri++)
            {
                int rax = rAxes[ri] < 0 ? rAxes[ri] + rank : rAxes[ri];
                if (rax < 0 || rax >= rank) continue; // Skip out-of-range axes
                starts[rax] = resolvedStarts[ri];
                ends[rax] = resolvedEnds[ri];
                if (ri < rSteps.Length) steps[rax] = rSteps[ri];
            }
            axes = Enumerable.Range(0, rank).ToArray();
        }
        else if (ctx.Inputs.Length >= 3 && ctx.Inputs[1] != null
            && ctx.TryGetInputValues(1) is float[] startsF && ctx.TryGetInputValues(2) is float[] endsF)
        {
            // Path 2: runtime constant values from tensor inputs
            // Clamp to int range — ONNX uses INT64_MAX (9.2e18) as "to end" sentinel
            starts = startsF.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();
            ends = endsF.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();
            axes = ctx.Inputs.Length > 3 && ctx.TryGetInputValues(3) is float[] axF
                ? axF.Select(v => (int)v).ToArray() : Enumerable.Range(0, starts.Length).ToArray();
            steps = ctx.Inputs.Length > 4 && ctx.TryGetInputValues(4) is float[] stF
                ? stF.Select(v => (int)v).ToArray() : Enumerable.Repeat(1, starts.Length).ToArray();
        }
        else
        {
            // Path 3: attributes (opset < 10)
            var attrStarts = ctx.GetInts("starts");
            var attrEnds = ctx.GetInts("ends");
            var attrAxes = ctx.GetInts("axes");
            var attrSteps = ctx.GetInts("steps");
            starts = attrStarts.Length > 0 ? attrStarts : new int[rank];
            ends = attrEnds.Length > 0 ? attrEnds : inShape.ToArray();
            axes = attrAxes.Length > 0 ? attrAxes : Enumerable.Range(0, starts.Length).ToArray();
            steps = attrSteps.Length > 0 ? attrSteps : Enumerable.Repeat(1, starts.Length).ToArray();
        }

        // Normalize negative indices and clamp
        var sliceStarts = new int[rank];
        var sliceEnds = new int[rank];
        var sliceSteps = new int[rank];
        for (int i = 0; i < rank; i++) { sliceStarts[i] = 0; sliceEnds[i] = inShape[i]; sliceSteps[i] = 1; }
        for (int i = 0; i < axes.Length; i++)
        {
            int ax = axes[i] < 0 ? axes[i] + rank : axes[i];
            if (ax < 0 || ax >= rank) continue; // Skip out-of-range axes
            int s = i < starts.Length ? starts[i] : 0;
            int e = i < ends.Length ? ends[i] : inShape[ax];
            int st = i < steps.Length ? steps[i] : 1;
            if (s < 0) s += inShape[ax];
            if (e < 0) e += inShape[ax];
            s = Math.Clamp(s, 0, inShape[ax]);
            e = Math.Clamp(e, 0, inShape[ax]);
            sliceStarts[ax] = s;
            sliceEnds[ax] = e;
            sliceSteps[ax] = st;
        }

        // Compute output shape and strides
        var outShape = ctx.Outputs[0].Shape;
        var inStrides = new int[rank];
        inStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) inStrides[i] = inStrides[i + 1] * inShape[i + 1];

        // CPU-side index computation, GPU copy per contiguous block
        // For simplicity with small tensors, compute full index mapping on CPU
        int outCount = ctx.Outputs[0].ElementCount;
        if (outCount <= 65536)
        {
            // Small tensor: compute on CPU via pre-read values
            var inVals = ctx.TryGetInputValues(0);
            if (inVals != null)
            {
                var result = new float[outCount];
                int outIdx = 0;
                SliceCPU(inVals, result, inShape, sliceStarts, sliceEnds, sliceSteps, inStrides, rank, 0, 0, ref outIdx);
                var temp = ctx.Pool.AllocatePermanent(result, outShape);
                reg.ElementWise.Scale(temp.Data, ctx.Outputs[0].Data, outCount, 1f);
                return;
            }
        }

        // Fallback for large tensors: copy contiguous slices along last axis
        int outIdx2 = 0;
        SliceGPU(input.Data, ctx.Outputs[0].Data, inShape, sliceStarts, sliceEnds, sliceSteps, inStrides, rank, 0, 0, ref outIdx2, reg);
    }

    private static void SliceCPU(float[] input, float[] output, int[] shape,
        int[] starts, int[] ends, int[] steps, int[] strides, int rank, int dim, int inOffset, ref int outIdx)
    {
        if (dim == rank)
        {
            if (outIdx < output.Length && inOffset < input.Length)
                output[outIdx++] = input[inOffset];
            return;
        }
        for (int i = starts[dim]; i < ends[dim]; i += steps[dim])
            SliceCPU(input, output, shape, starts, ends, steps, strides, rank, dim + 1, inOffset + i * strides[dim], ref outIdx);
    }

    private void SliceGPU(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output, int[] shape,
        int[] starts, int[] ends, int[] steps, int[] strides, int rank, int dim, int inOffset, ref int outIdx,
        OperatorRegistry reg2)
    {
        if (dim == rank - 1)
        {
            // Copy contiguous run along last axis
            int start = starts[dim]; int end = ends[dim]; int step = steps[dim];
            if (step == 1)
            {
                int len = end - start;
                if (len > 0 && outIdx + len <= (int)output.Length)
                {
                    reg2.ElementWise.Scale(input.SubView(inOffset + start, len), output.SubView(outIdx, len), len, 1f);
                    outIdx += len;
                }
            }
            else
            {
                for (int i = start; i < end; i += step)
                {
                    if (outIdx < (int)output.Length)
                        reg2.ElementWise.Scale(input.SubView(inOffset + i, 1), output.SubView(outIdx, 1), 1, 1f);
                    outIdx++;
                }
            }
            return;
        }
        for (int i = starts[dim]; i < ends[dim]; i += steps[dim])
            SliceGPU(input, output, shape, starts, ends, steps, strides, rank, dim + 1, inOffset + i * strides[dim], ref outIdx, reg2);
    }
}

// ── Transpose ──

public class TransposeOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Transpose";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        if (inputs.Length == 0 || inputs[0].Length == 0) return new[] { inputs.Length > 0 ? inputs[0] : Array.Empty<int>() };
        var perm = attrs.ContainsKey("perm") ? ((long[])attrs["perm"]).Select(p => (int)p).ToArray()
                 : Enumerable.Range(0, inputs[0].Length).Reverse().ToArray();
        // Guard: perm must match input rank
        if (perm.Length != inputs[0].Length || perm.Any(p => p >= inputs[0].Length))
            return new[] { inputs[0] }; // Fallback
        var outShape = new int[inputs[0].Length];
        for (int i = 0; i < perm.Length; i++) outShape[i] = inputs[0][perm[i]];
        return new[] { outShape };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var perm = ctx.GetInts("perm");
        if (perm.Length == 0)
            perm = Enumerable.Range(0, ctx.Inputs[0].Rank).Reverse().ToArray();
        // Guard: perm length must match input rank — if not, fall back to reverse
        if (perm.Length != ctx.Inputs[0].Rank)
        {
            if (InferenceSession.VerboseLogging)
                Console.WriteLine($"[Transpose] WARN: perm[{perm.Length}] != rank[{ctx.Inputs[0].Rank}], shape=[{string.Join(",", ctx.Inputs[0].Shape)}], attrs={string.Join(",", ctx.Attributes.Select(kv => $"{kv.Key}={kv.Value}"))}");
            perm = Enumerable.Range(0, ctx.Inputs[0].Rank).Reverse().ToArray();
        }
        reg.Transpose.Transpose(ctx.Inputs[0].Data, ctx.Outputs[0].Data,
            ctx.Inputs[0].Shape, perm);
    }
}
