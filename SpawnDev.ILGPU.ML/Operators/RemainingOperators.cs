using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Operators;

// Full ONNX operator coverage — remaining 42 operators (batch 4).
// Stubs for operators that need full implementations. Having the operator
// registered means the model loads without "unknown operator" errors.
// Missing functionality can be added incrementally.

public class LpNormalizationOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "LpNormalization";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        int p = ctx.GetInt("p", 2);
        int axis = ctx.GetInt("axis", -1);
        var shape = ctx.Inputs[0].Shape;
        if (axis < 0) axis += shape.Length;

        int outer = 1; for (int i = 0; i < axis; i++) outer *= shape[i];
        int axisSize = shape[axis];
        int inner = 1; for (int i = axis + 1; i < shape.Length; i++) inner *= shape[i];

        // GPU path: one thread per output element (gather, not scatter — WebGL TF compatible)
        var paramsData = new float[] { axisSize, inner, p };
        var paramsBuf = ctx.Pool.Rent(new[] { paramsData.Length });
        paramsBuf.Data.SubView(0, paramsData.Length).CopyFromCPU(paramsData);
        reg.ElementWise.LpNorm(ctx.Inputs[0].Data, ctx.Outputs[0].Data, paramsBuf.Data, outer * axisSize * inner);
    }
}
public class GlobalLpPoolOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "GlobalLpPool";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var s = inputs[0].ToArray(); for (int j = 2; j < s.Length; j++) s[j] = 1; return new[] { s };
    }
    public void Execute(OnnxOpContext ctx)
    {
        int p = ctx.GetInt("p", 2);
        var shape = ctx.Inputs[0].Shape;
        int N = shape[0], C = shape[1];
        int spatial = 1; for (int i = 2; i < shape.Length; i++) spatial *= shape[i];

        // GPU path: one thread per (N, C) pair, iterates spatial
        var paramsData = new float[] { C, spatial, p };
        var paramsBuf = ctx.Pool.Rent(new[] { paramsData.Length });
        paramsBuf.Data.SubView(0, paramsData.Length).CopyFromCPU(paramsData);
        reg.ElementWise.GlobalLpPool(ctx.Inputs[0].Data, ctx.Outputs[0].Data, paramsBuf.Data, N * C);
    }
}
public class LpPoolOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "LpPool";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Same output shape calculation as AveragePool
        var x = inputs[0];
        var kernelShape = attrs.ContainsKey("kernel_shape") ? ((long[])attrs["kernel_shape"]).Select(v => (int)v).ToArray() : new[] { 1, 1 };
        var strides = attrs.ContainsKey("strides") ? ((long[])attrs["strides"]).Select(v => (int)v).ToArray() : new[] { 1, 1 };
        var pads = attrs.ContainsKey("pads") ? ((long[])attrs["pads"]).Select(v => (int)v).ToArray() : new int[4];
        int kH = kernelShape[0], kW = kernelShape.Length > 1 ? kernelShape[1] : 1;
        int sH = strides[0], sW = strides.Length > 1 ? strides[1] : 1;
        int pH = pads.Length > 0 ? pads[0] : 0, pW = pads.Length > 1 ? pads[1] : 0;
        int outH = (x[2] + 2 * pH - kH) / sH + 1;
        int outW = (x[3] + 2 * pW - kW) / sW + 1;
        return new[] { new[] { x[0], x[1], outH, outW } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        int p = ctx.GetInt("p", 2);
        var inShape = ctx.Inputs[0].Shape;
        var outShape = ctx.Outputs[0].Shape;
        int N = inShape[0], C = inShape[1], H = inShape[2], W = inShape[3];
        int outH = outShape[2], outW = outShape[3];
        var kernelShape = ctx.GetInts("kernel_shape", new[] { 1, 1 });
        var strides = ctx.GetInts("strides", new[] { 1, 1 });
        var pads = ctx.GetInts("pads", new int[4]);
        int kH = kernelShape[0], kW = kernelShape.Length > 1 ? kernelShape[1] : 1;
        int sH = strides[0], sW = strides.Length > 1 ? strides[1] : 1;
        int pH = pads.Length > 0 ? pads[0] : 0, pW = pads.Length > 1 ? pads[1] : 0;
        int totalOutput = N * C * outH * outW;

        // GPU path: one thread per output element, iterates kernel window
        var paramsData = new float[] { C, H, W, outH, outW, kH, kW, sH, sW, pH, pW, p };
        var paramsBuf = ctx.Pool.Rent(new[] { paramsData.Length });
        paramsBuf.Data.SubView(0, paramsData.Length).CopyFromCPU(paramsData);
        reg.ElementWise.LpPool(ctx.Inputs[0].Data, ctx.Outputs[0].Data, paramsBuf.Data, totalOutput);
    }
}
public class DetOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Det";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var shape = inputs[0];
        return new[] { shape.Length > 2 ? shape[..^2] : new[] { 1 } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // Determinant via LU decomposition (Gaussian elimination)
        var xVals = ctx.TryGetInputValues(0);
        if (xVals == null) { reg.ElementWise.Fill(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount, 0f); return; }
        var shape = ctx.Inputs[0].Shape;
        int M = shape[^1]; // square matrix dimension
        int batch = ctx.Inputs[0].ElementCount / (M * M);
        var result = new float[batch];
        for (int b = 0; b < batch; b++)
        {
            // Copy matrix for in-place elimination
            var mat = new float[M * M];
            Array.Copy(xVals, b * M * M, mat, 0, M * M);
            float det = 1f;
            for (int i = 0; i < M; i++)
            {
                // Partial pivoting
                int maxRow = i;
                float maxVal = MathF.Abs(mat[i * M + i]);
                for (int k = i + 1; k < M; k++)
                {
                    float v = MathF.Abs(mat[k * M + i]);
                    if (v > maxVal) { maxVal = v; maxRow = k; }
                }
                if (maxRow != i)
                {
                    for (int j = 0; j < M; j++)
                        (mat[i * M + j], mat[maxRow * M + j]) = (mat[maxRow * M + j], mat[i * M + j]);
                    det = -det; // row swap flips sign
                }
                float pivot = mat[i * M + i];
                if (MathF.Abs(pivot) < 1e-12f) { det = 0f; break; }
                det *= pivot;
                for (int k = i + 1; k < M; k++)
                {
                    float factor = mat[k * M + i] / pivot;
                    for (int j = i + 1; j < M; j++)
                        mat[k * M + j] -= factor * mat[i * M + j];
                }
            }
            result[b] = det;
        }
        int copyLen = Math.Min(result.Length, ctx.Outputs[0].ElementCount);
        if (copyLen < result.Length) { var t = new float[copyLen]; Array.Copy(result, t, copyLen); ctx.Outputs[0].Data.SubView(0, copyLen).CopyFromCPU(t); }
        else ctx.Outputs[0].Data.SubView(0, copyLen).CopyFromCPU(result);
    }
}
public class BernoulliOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Bernoulli";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        int count = ctx.Outputs[0].ElementCount;
        int seed = ctx.GetInt("seed", 0);
        if (seed == 0) seed = Environment.TickCount;
        // GPU path: per-thread xorshift PRNG, reads probability from input
        reg.ElementWise.Bernoulli(ctx.Inputs[0].Data, ctx.Outputs[0].Data, count, seed);
    }
}
public class CenterCropPadOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "CenterCropPad";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Output shape comes from the shape input (input[1])
        if (inputs.Length > 1) return new[] { inputs[1] };
        return new[] { inputs[0] };
    }
    public void Execute(OnnxOpContext ctx)
    {
        int outCount = ctx.Outputs[0].ElementCount;
        var inShape = ctx.Inputs[0].Shape;
        var outShape = ctx.Outputs[0].Shape;
        int rank = inShape.Length;

        // Build params: [rank, inShape..., outShape..., inStrides...]
        var paramsData = new int[1 + 3 * rank];
        paramsData[0] = rank;
        for (int d = 0; d < rank; d++) paramsData[1 + d] = inShape[d];
        for (int d = 0; d < rank; d++) paramsData[1 + rank + d] = outShape[d];
        // Compute input strides
        paramsData[1 + 3 * rank - 1] = 1;
        for (int d = rank - 2; d >= 0; d--)
            paramsData[1 + 2 * rank + d] = paramsData[1 + 2 * rank + d + 1] * inShape[d + 1];

        // GPU path: one thread per output element, reads centered input
        var paramsFloatData = paramsData.Select(v => (float)v).ToArray();
        var paramsBuf = ctx.Pool.Rent(new[] { paramsFloatData.Length });
        paramsBuf.Data.SubView(0, paramsFloatData.Length).CopyFromCPU(paramsFloatData);
        reg.ElementWise.CenterCropPad(ctx.Inputs[0].Data, ctx.Outputs[0].Data, paramsBuf.Data, outCount);
    }
}
public class MaxRoiPoolOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "MaxRoiPool";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        int numRois = inputs.Length > 1 ? inputs[1][0] : 1;
        int C = inputs[0][1];
        var pooledShape = attrs.ContainsKey("pooled_shape") ? ((long[])attrs["pooled_shape"]).Select(x => (int)x).ToArray() : new[] { 1, 1 };
        return new[] { new[] { numRois, C, pooledShape[0], pooledShape[1] } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // MaxRoiPool: max pooling over ROI regions (older R-CNN models)
        var xVals = ctx.TryGetInputValues(0);
        var roiVals = ctx.TryGetInputValues(1);
        if (xVals == null || roiVals == null) { reg.ElementWise.Fill(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount, 0f); return; }

        var xShape = ctx.Inputs[0].Shape;
        int C = xShape[1], H = xShape[2], W = xShape[3];
        int numRois = ctx.Inputs[1].Shape[0];
        var pooledShape = ctx.GetInts("pooled_shape", new[] { 1, 1 });
        float spatialScale = ctx.GetFloat("spatial_scale", 1f);
        int pH = pooledShape[0], pW = pooledShape[1];

        var result = new float[numRois * C * pH * pW];
        for (int r = 0; r < numRois; r++)
        {
            int batchIdx = (int)roiVals[r * 5];
            float x1 = roiVals[r * 5 + 1] * spatialScale;
            float y1 = roiVals[r * 5 + 2] * spatialScale;
            float x2 = roiVals[r * 5 + 3] * spatialScale;
            float y2 = roiVals[r * 5 + 4] * spatialScale;

            float roiH = Math.Max(y2 - y1 + 1f, 1f), roiW = Math.Max(x2 - x1 + 1f, 1f);
            float binH = roiH / pH, binW = roiW / pW;

            for (int c = 0; c < C; c++)
            {
                int chOff = (batchIdx * C + c) * H;
                for (int oh = 0; oh < pH; oh++)
                {
                    int hStart = (int)MathF.Floor(y1 + oh * binH);
                    int hEnd = (int)MathF.Ceiling(y1 + (oh + 1) * binH);
                    for (int ow = 0; ow < pW; ow++)
                    {
                        int wStart = (int)MathF.Floor(x1 + ow * binW);
                        int wEnd = (int)MathF.Ceiling(x1 + (ow + 1) * binW);
                        float maxVal = float.NegativeInfinity;
                        for (int ih = Math.Max(0, hStart); ih < Math.Min(H, hEnd); ih++)
                            for (int iw = Math.Max(0, wStart); iw < Math.Min(W, wEnd); iw++)
                                maxVal = Math.Max(maxVal, xVals[(chOff + ih) * W + iw]);
                        result[((r * C + c) * pH + oh) * pW + ow] = float.IsNegativeInfinity(maxVal) ? 0f : maxVal;
                    }
                }
            }
        }
        int copyLen = Math.Min(result.Length, ctx.Outputs[0].ElementCount);
        if (copyLen < result.Length)
        {
            var trimmed = new float[copyLen];
            Array.Copy(result, trimmed, copyLen);
            ctx.Outputs[0].Data.SubView(0, copyLen).CopyFromCPU(trimmed);
        }
        else
            ctx.Outputs[0].Data.SubView(0, copyLen).CopyFromCPU(result);
    }
}
public class MaxUnpoolOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "MaxUnpool";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // If output_shape is provided (input[2]), use it. Otherwise infer from kernel/stride.
        if (inputs.Length > 2) return new[] { inputs[2] };
        return new[] { inputs[0] }; // fallback
    }
    public void Execute(OnnxOpContext ctx)
    {
        int outCount = ctx.Outputs[0].ElementCount;
        int inCount = ctx.Inputs[0].ElementCount;
        reg.ElementWise.Fill(ctx.Outputs[0].Data, outCount, 0f);
        if (inCount > 0 && ctx.Inputs.Length > 1)
        {
            // GPU scatter: each thread writes one value to its index position
            reg.ElementWise.MaxUnpool(ctx.Inputs[0].Data, ctx.Inputs[1].Data,
                ctx.Outputs[0].Data, inCount, outCount);
        }
    }
}
public class ImageDecoderOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ImageDecoder";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Output shape depends on the image — cannot determine at compile time
        // Default to a placeholder; the actual shape is set at runtime
        return new[] { new[] { 1, 3, 224, 224 } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // ImageDecoder: decode PNG/JPEG/BMP bytes to [H, W, C] tensor.
        // Our engine works with pre-decoded float tensors. If the model includes
        // an ImageDecoder node, the input bytes should have been preprocessed
        // before reaching the graph executor.
        // Pass through any float data that exists, otherwise fill zeros.
        if (ctx.Inputs.Length > 0 && ctx.Inputs[0].ElementCount > 0)
        {
            int count = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount);
            reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, count), ctx.Outputs[0].Data.SubView(0, count), count, 1f);
        }
        else
        {
            reg.ElementWise.Fill(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount, 0f);
        }
    }
}
public class AffineGridOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "AffineGrid";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // theta: [N, 2, 3], size: [N, C, H, W] → output: [N, H, W, 2]
        var sizeVals = inputs.Length > 1 ? inputs[1] : inputs[0];
        int N = sizeVals[0], H = sizeVals.Length > 2 ? sizeVals[2] : 1, W = sizeVals.Length > 3 ? sizeVals[3] : 1;
        return new[] { new[] { N, H, W, 2 } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var sizeVals = ctx.TryGetInputValues(1); // [N, C, H, W]
        if (sizeVals == null)
        {
            reg.ElementWise.Fill(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount, 0f);
            return;
        }
        int N = (int)sizeVals[0], H = (int)sizeVals[2], W = (int)sizeVals[3];
        int alignCorners = ctx.GetInt("align_corners", 0);

        // GPU path: theta is on GPU, one thread per pixel
        var paramsData = new float[] { H, W, alignCorners };
        var paramsBuf = ctx.Pool.Rent(new[] { paramsData.Length });
        paramsBuf.Data.SubView(0, paramsData.Length).CopyFromCPU(paramsData);
        // One thread per scalar output (x + y interleaved) — gather, WebGL TF compatible
        reg.ElementWise.AffineGrid(ctx.Inputs[0].Data, ctx.Outputs[0].Data, paramsBuf.Data, N * H * W * 2);
    }
}
public class GridSampleOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "GridSample";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Input: [N, C, Hin, Win], Grid: [N, Hout, Wout, 2]
        // Output: [N, C, Hout, Wout]
        var x = inputs[0]; var grid = inputs[1];
        return new[] { new[] { x[0], x[1], grid[1], grid[2] } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var xShape = ctx.Inputs[0].Shape; // [N, C, Hin, Win]
        var gridShape = ctx.Inputs[1].Shape; // [N, Hout, Wout, 2]
        int N = xShape[0], C = xShape[1], Hin = xShape[2], Win = xShape[3];
        int Hout = gridShape[1], Wout = gridShape[2];
        int alignCorners = ctx.GetInt("align_corners", 0);
        int totalOutput = N * C * Hout * Wout;

        // GPU path: bilinear interpolation, one thread per output element
        var paramsData = new float[] { N, C, Hin, Win, Hout, Wout, alignCorners };
        var paramsBuf = ctx.Pool.Rent(new[] { paramsData.Length });
        paramsBuf.Data.SubView(0, paramsData.Length).CopyFromCPU(paramsData);
        reg.ElementWise.GridSample(ctx.Inputs[0].Data, ctx.Inputs[1].Data, ctx.Outputs[0].Data, paramsBuf.Data, totalOutput);
    }
}
public class Col2ImOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Col2Im";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Input: [N, C*kH*kW, L], image_shape: [H, W]
        // Output: [N, C, H, W]
        if (inputs.Length > 1 && inputs[1].Length >= 2)
        {
            int N = inputs[0][0];
            var blockShape = attrs.ContainsKey("block_shape") ? ((long[])attrs["block_shape"]).Select(x => (int)x).ToArray() : new[] { 1, 1 };
            int C = inputs[0][1] / (blockShape[0] * blockShape[1]);
            return new[] { new[] { N, C, inputs[1][0], inputs[1][1] } };
        }
        return new[] { inputs[0] };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var imageShapeVals = ctx.TryGetInputValues(1);
        var blockShape = ctx.GetInts("block_shape", new[] { 1, 1 });
        var strides = ctx.GetInts("strides", new[] { 1, 1 });
        var pads = ctx.GetInts("pads", new int[4]);

        var xShape = ctx.Inputs[0].Shape;
        int N = xShape[0]; int colDim = xShape[1]; int L = xShape[2];
        int kH = blockShape.Length > 0 ? blockShape[0] : 1;
        int kW = blockShape.Length > 1 ? blockShape[1] : 1;
        int C = colDim / (kH * kW);
        int outH = imageShapeVals != null && imageShapeVals.Length > 0 ? (int)imageShapeVals[0] : 1;
        int outW = imageShapeVals != null && imageShapeVals.Length > 1 ? (int)imageShapeVals[1] : 1;
        int sH = strides.Length > 0 ? strides[0] : 1;
        int sW = strides.Length > 1 ? strides[1] : 1;
        int pH = pads.Length > 0 ? pads[0] : 0;
        int pW = pads.Length > 1 ? pads[1] : 0;
        int paddedW = outW + 2 * pW;
        int blocksW = (paddedW - kW) / sW + 1;

        // One thread per output position — gather kernel writes every output unconditionally,
        // so no Fill pre-pass needed. Also correct for overlapping kernels (the prior scatter-add
        // had a race condition for stride < kernel).
        int outCount = ctx.Outputs[0].ElementCount;
        var paramsData = new float[] { C, L, kH, kW, outH, outW, sH, sW, pH, pW, blocksW, colDim };
        var paramsBuf = ctx.Pool.Rent(new[] { paramsData.Length });
        paramsBuf.Data.SubView(0, paramsData.Length).CopyFromCPU(paramsData);
        reg.ElementWise.Col2Im(ctx.Inputs[0].Data, ctx.Outputs[0].Data, paramsBuf.Data, outCount);
    }
}
public class DeformConvOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "DeformConv";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Same output shape logic as regular Conv
        return new ConvOperator(reg).InferOutputShapes(inputs, attrs);
    }
    public void Execute(OnnxOpContext ctx)
    {
        var x = ctx.Inputs[0]; var w = ctx.Inputs[1];
        if (x.Shape.Length < 4 || w.Shape.Length < 4)
        {
            reg.ElementWise.Fill(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount, 0f);
            return;
        }

        var xShape = x.Shape; var wShape = w.Shape;
        int N = xShape[0], inC = xShape[1], H = xShape[2], W = xShape[3];
        int outC = wShape[0], kH = wShape[2], kW = wShape[3];
        int group = ctx.GetInt("group", 1);
        int offsetGroup = ctx.GetInt("offset_group", 1);
        var strides = ctx.GetInts("strides", new[] { 1, 1 });
        var pads = ctx.GetInts("pads", new int[4]);
        var dilations = ctx.GetInts("dilations", new[] { 1, 1 });
        int sH = strides[0], sW = strides.Length > 1 ? strides[1] : sH;
        int pH = pads[0], pW = pads.Length > 1 ? pads[1] : pH;
        int dH = dilations[0], dW = dilations.Length > 1 ? dilations[1] : dH;

        int outH = (H + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
        int outW = (W + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
        int totalOutput = N * outC * outH * outW;

        // GPU path: offsets tensor on GPU, one thread per output element
        if (ctx.Inputs.Length > 2 && ctx.Inputs[2] != null)
        {
            var paramsData = new float[] { inC, H, W, outC, outH, outW, kH, kW, sH, sW, pH, pW, group, offsetGroup };
            var paramsBuf = ctx.Pool.Rent(new[] { paramsData.Length });
            paramsBuf.Data.SubView(0, paramsData.Length).CopyFromCPU(paramsData);
            reg.ElementWise.DeformConv(x.Data, w.Data, ctx.Inputs[2].Data,
                ctx.Outputs[0].Data, paramsBuf.Data, totalOutput);
            // Add bias if provided
            if (ctx.Inputs.Length > 3 && ctx.Inputs[3] != null)
                reg.ElementWise.AddBias(ctx.Outputs[0].Data, ctx.Inputs[3].Data, totalOutput, outC);
        }
        else
        {
            // Fallback: regular conv without offsets
            var bias = ctx.Inputs.Length > 3 && ctx.Inputs[3] != null ? ctx.Inputs[3].Data : default;
            reg.Conv2D.Forward(x.Data, w.Data, bias, ctx.Outputs[0].Data,
                inC, H, W, outC, kH, kW, sH, pH);
        }
    }
}
public class RoiAlignOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "RoiAlign";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Input: X [N, C, H, W], rois [num_rois, 4], batch_indices [num_rois]
        // Output: [num_rois, C, output_height, output_width]
        int C = inputs[0][1];
        int numRois = inputs.Length > 1 ? inputs[1][0] : 1;
        int outH = attrs.ContainsKey("output_height") ? Convert.ToInt32(attrs["output_height"]) : 1;
        int outW = attrs.ContainsKey("output_width") ? Convert.ToInt32(attrs["output_width"]) : 1;
        return new[] { new[] { numRois, C, outH, outW } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // RoiAlign: bilinear interpolation over regions of interest
        var xVals = ctx.TryGetInputValues(0);
        var roiVals = ctx.TryGetInputValues(1);
        var batchIdxVals = ctx.Inputs.Length > 2 ? ctx.TryGetInputValues(2) : null;
        if (xVals == null || roiVals == null)
        {
            reg.ElementWise.Fill(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount, 0f);
            return;
        }

        var xShape = ctx.Inputs[0].Shape;
        int N = xShape[0], C = xShape[1], H = xShape[2], W = xShape[3];
        int numRois = ctx.Inputs[1].Shape[0];
        int outH = ctx.GetInt("output_height", 1);
        int outW = ctx.GetInt("output_width", 1);
        float spatialScale = ctx.GetFloat("spatial_scale", 1f);
        int samplingRatio = ctx.GetInt("sampling_ratio", 0);

        var result = new float[numRois * C * outH * outW];

        for (int r = 0; r < numRois; r++)
        {
            int batchIdx = batchIdxVals != null ? (int)batchIdxVals[r] : 0;
            float x1 = roiVals[r * 4] * spatialScale;
            float y1 = roiVals[r * 4 + 1] * spatialScale;
            float x2 = roiVals[r * 4 + 2] * spatialScale;
            float y2 = roiVals[r * 4 + 3] * spatialScale;

            float roiW = x2 - x1, roiH = y2 - y1;
            float binH = roiH / outH, binW = roiW / outW;
            int sampleH = samplingRatio > 0 ? samplingRatio : Math.Max(1, (int)MathF.Ceiling(binH));
            int sampleW = samplingRatio > 0 ? samplingRatio : Math.Max(1, (int)MathF.Ceiling(binW));

            for (int c = 0; c < C; c++)
            {
                int chOff = (batchIdx * C + c) * H;
                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        float sum = 0f;
                        int count = 0;
                        for (int sy = 0; sy < sampleH; sy++)
                        {
                            float fy = y1 + (oh + (sy + 0.5f) / sampleH) * binH;
                            for (int sx = 0; sx < sampleW; sx++)
                            {
                                float fx = x1 + (ow + (sx + 0.5f) / sampleW) * binW;
                                // Bilinear interpolation
                                int ix0 = (int)MathF.Floor(fx), iy0 = (int)MathF.Floor(fy);
                                int ix1 = ix0 + 1, iy1 = iy0 + 1;
                                float tx = fx - ix0, ty = fy - iy0;
                                float v00 = 0, v01 = 0, v10 = 0, v11 = 0;
                                if (ix0 >= 0 && ix0 < W && iy0 >= 0 && iy0 < H) v00 = xVals[(chOff + iy0) * W + ix0];
                                if (ix1 >= 0 && ix1 < W && iy0 >= 0 && iy0 < H) v01 = xVals[(chOff + iy0) * W + ix1];
                                if (ix0 >= 0 && ix0 < W && iy1 >= 0 && iy1 < H) v10 = xVals[(chOff + iy1) * W + ix0];
                                if (ix1 >= 0 && ix1 < W && iy1 >= 0 && iy1 < H) v11 = xVals[(chOff + iy1) * W + ix1];
                                sum += v00 * (1 - tx) * (1 - ty) + v01 * tx * (1 - ty) + v10 * (1 - tx) * ty + v11 * tx * ty;
                                count++;
                            }
                        }
                        result[((r * C + c) * outH + oh) * outW + ow] = count > 0 ? sum / count : 0f;
                    }
                }
            }
        }

        int copyLen = Math.Min(result.Length, ctx.Outputs[0].ElementCount);
        if (copyLen < result.Length) { var t = new float[copyLen]; Array.Copy(result, t, copyLen); ctx.Outputs[0].Data.SubView(0, copyLen).CopyFromCPU(t); }
        else ctx.Outputs[0].Data.SubView(0, copyLen).CopyFromCPU(result);
    }
}
public class ConvIntegerOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ConvInteger";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Same shape as regular Conv
        return new ConvOperator(reg).InferOutputShapes(inputs, attrs);
    }
    public void Execute(OnnxOpContext ctx)
    {
        // ConvInteger: y = conv(x - x_zero_point, w - w_zero_point)
        // Inputs: x, w, [x_zero_point], [w_zero_point]
        var x = ctx.Inputs[0]; var w = ctx.Inputs[1];
        var xShape = x.Shape; var wShape = w.Shape;
        if (xShape.Length < 4 || wShape.Length < 4)
        {
            reg.ElementWise.Fill(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount, 0f);
            return;
        }

        // CPU-side zero-point subtraction for correctness
        var xVals = ctx.TryGetInputValues(0);
        var wVals = ctx.TryGetInputValues(1);
        if (xVals == null || wVals == null)
        {
            // Can't read values — fall back to direct conv (ignoring zero points)
            int st = ctx.GetInts("strides", new[] { 1, 1 })[0];
            int pd = ctx.GetInts("pads", new int[4])[0];
            reg.Conv2D.Forward(x.Data, w.Data, default, ctx.Outputs[0].Data,
                xShape[1], xShape[2], xShape[3], wShape[0], wShape[2], wShape[3], st, pd);
            return;
        }

        var xAdj = (float[])xVals.Clone();
        var wAdj = (float[])wVals.Clone();

        // Subtract zero points
        float xZp = 0f, wZp = 0f;
        if (ctx.Inputs.Length > 2 && ctx.Inputs[2] != null)
        {
            var zpVals = ctx.TryGetInputValues(2);
            if (zpVals != null && zpVals.Length > 0) xZp = zpVals[0];
        }
        if (ctx.Inputs.Length > 3 && ctx.Inputs[3] != null)
        {
            var zpVals = ctx.TryGetInputValues(3);
            if (zpVals != null && zpVals.Length > 0) wZp = zpVals[0];
        }
        for (int i = 0; i < xAdj.Length; i++) xAdj[i] -= xZp;
        for (int i = 0; i < wAdj.Length; i++) wAdj[i] -= wZp;

        // Upload adjusted data via Pool.Rent — buffers stay alive past the WebGPU command encoder flush
        var xBufMem = ctx.Pool.Rent(new[] { xAdj.Length });
        xBufMem.Data.SubView(0, xAdj.Length).CopyFromCPU(xAdj);
        var wBufMem = ctx.Pool.Rent(new[] { wAdj.Length });
        wBufMem.Data.SubView(0, wAdj.Length).CopyFromCPU(wAdj);
        var zeroBias = ctx.Pool.Rent(new[] { wShape[0] }); // Conv2D always reads bias — must provide zero-filled buffer
        zeroBias.Data.SubView(0, wShape[0]).CopyFromCPU(new float[wShape[0]]);
        int stride = ctx.GetInts("strides", new[] { 1, 1 })[0];
        int pad = ctx.GetInts("pads", new int[4])[0];
        int oH = (xShape[2] + 2 * pad - wShape[2]) / stride + 1;
        int oW = (xShape[3] + 2 * pad - wShape[3]) / stride + 1;
        var outBufMem = ctx.Pool.Rent(new[] { xShape[0] * wShape[0] * oH * oW });
        reg.Conv2D.Forward(xBufMem.Data, wBufMem.Data, zeroBias.Data, outBufMem.Data,
            xShape[1], xShape[2], xShape[3], wShape[0], wShape[2], wShape[3], stride, pad);
        int copyLen = Math.Min(outBufMem.ElementCount, ctx.Outputs[0].ElementCount);
        reg.ElementWise.Scale(outBufMem.Data.SubView(0, copyLen), ctx.Outputs[0].Data.SubView(0, copyLen), copyLen, 1f);
    }
}

public class MatMulIntegerOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "MatMulInteger";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // ⚠️ MatMulInteger is N-D in ONNX exactly as MatMul is, and this used to return a rank-2
        // shape unconditionally: `[a[0], N]`. For a batched activation [S, 1, K] that yields [S, N] - the
        // size-1 batch axis is DELETED, and the old code additionally read K from a[1] (= 1), so it
        // contracted over a single element and produced wrong VALUES on top of the wrong shape.
        //
        // Nothing downstream notices a missing axis directly. What it does is poison the runtime shape
        // arithmetic: a later Shape/Gather reads dim 1 and gets the feature count instead of the batch,
        // so a Reshape target and a Slice bound both come out wrong, and the graph finally dies ~180 nodes
        // later on a broadcast that has nothing to do with the cause. Measured on ZipVoice's int8 text
        // encoder: node 28 produced [106,192] where onnxruntime gives [13,1,192], and node 222 failed with
        // "Shapes [106,432] and [106,432,1] are not broadcastable".
        //
        // Rank-2 behaviour is byte-for-byte unchanged: for a rank-2 A, a[^2] == a[0] and a[^1] == a[1].
        var a = inputs[0]; var b = inputs[1];
        int M = a.Length >= 2 ? a[^2] : 1;
        int N = b.Length >= 2 ? b[^1] : b[0];
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
        // MatMulInteger: y = matmul(A - a_zero_point, B - b_zero_point)
        // Inputs: A, B, [a_zero_point], [b_zero_point]
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        var aShape = a.Shape; var bShape = b.Shape;
        // Read the contraction dim from the LAST axis, not axis 1 - they coincide only at rank 2.
        int K = aShape[^1];
        int M = aShape.Length >= 2 ? aShape[^2] : 1;
        int N = bShape.Length > 1 ? bShape[^1] : bShape[0];
        // Every leading axis is just more rows sharing the same weight, which is how MatMulOperator treats
        // a batched activation against a 2-D weight. batch=1 collapses to rows == M.
        int rows = K > 0 ? (int)(a.ElementCount / K) : M;

        // Subtract zero points if provided
        var aAdj = ctx.Pool.Rent(aShape, "_mmi_a");
        reg.ElementWise.Scale(a.Data, aAdj.Data, a.ElementCount, 1f);
        if (ctx.Inputs.Length > 2 && ctx.Inputs[2] != null)
        {
            var azp = ctx.TryGetInputValues(2);
            if (azp != null && azp.Length > 0)
            {
                // a_zero_point is scalar, or PER-ROW of A [M, K] (one value per row of the M axis).
                // ⚠️ The per-row branch used to call the raw `Add`, which indexes BOTH operands by the
                // dispatch index and therefore read a.ElementCount (= M*K) values out of an M-element
                // zero-point buffer. And `AddBias` is NOT the fix: it indexes bias[i % C], which is
                // per-COLUMN. A row-major element i belongs to row i / K, so the row form is required -
                // the column form runs happily and computes the wrong number.
                var zpBuf = ctx.Pool.Rent(ctx.Inputs[2].Shape, "_mmi_azp");
                reg.ElementWise.Scale(ctx.Inputs[2].Data, zpBuf.Data, ctx.Inputs[2].ElementCount, -1f);
                if (azp.Length == 1)
                    reg.ElementWise.AddBias(aAdj.Data, zpBuf.Data, a.ElementCount, 1);
                else if (azp.Length == rows && K > 0)
                    // rows, not M: with a batched A every leading axis contributes rows, and a per-row
                    // zero point has one value per row of the FLATTENED matrix.
                    reg.ElementWise.AddRowBias(aAdj.Data, zpBuf.Data, a.ElementCount, K);
                else
                    throw new NotSupportedException(
                        $"MatMulInteger a_zero_point has {azp.Length} values; expected 1 (per-tensor) or " +
                        $"{rows} (per-row of A [{string.Join(",", aShape)}] flattened to [{rows},{K}]).");
                ctx.Pool.Return(zpBuf);
            }
        }

        var bAdj = ctx.Pool.Rent(bShape, "_mmi_b");
        reg.ElementWise.Scale(b.Data, bAdj.Data, b.ElementCount, 1f);
        if (ctx.Inputs.Length > 3 && ctx.Inputs[3] != null)
        {
            var bzp = ctx.TryGetInputValues(3);
            if (bzp != null && bzp.Length > 0)
            {
                // b_zero_point is scalar, or PER-COLUMN of B [K, N]. Row-major element i sits in column
                // i % N, so this one genuinely IS AddBias's bias[i % C] shape - unlike a_zero_point above.
                // The raw `Add` it used to call read b.ElementCount (= K*N) values out of an N-element
                // buffer.
                var zpBuf = ctx.Pool.Rent(ctx.Inputs[3].Shape, "_mmi_bzp");
                reg.ElementWise.Scale(ctx.Inputs[3].Data, zpBuf.Data, ctx.Inputs[3].ElementCount, -1f);
                if (bzp.Length == 1)
                    reg.ElementWise.AddBias(bAdj.Data, zpBuf.Data, b.ElementCount, 1);
                else if (bzp.Length == N)
                    reg.ElementWise.AddBias(bAdj.Data, zpBuf.Data, b.ElementCount, N);
                else
                    throw new NotSupportedException(
                        $"MatMulInteger b_zero_point has {bzp.Length} values; expected 1 (per-tensor) or " +
                        $"{N} (per-column of B [{K},{N}]).");
                ctx.Pool.Return(zpBuf);
            }
        }

        // A 2-D B is a shared weight: flatten all of A's rows into one [rows, K] @ [K, N]. Passing M here
        // instead of rows computed only the FIRST batch slice and left the rest of the output untouched.
        if (bShape.Length <= 2)
        {
            reg.MatMul.MatMul(aAdj.Data, bAdj.Data, ctx.Outputs[0].Data, rows, K, N);
        }
        else
        {
            int batch = (M > 0 && K > 0) ? (int)(a.ElementCount / ((long)M * K)) : 1;
            reg.MatMul.BatchedMatMul(aAdj.Data, bAdj.Data, ctx.Outputs[0].Data, batch, M, K, N);
        }
        ctx.Pool.Return(aAdj);
        ctx.Pool.Return(bAdj);
    }
}

public class QLinearConvOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "QLinearConv";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Inputs: x, x_scale, x_zero, w, w_scale, w_zero, y_scale, y_zero, [B]
        // Use x shape[0] and w shape for output dims
        if (inputs.Length >= 4) return new ConvOperator(reg).InferOutputShapes(new[] { inputs[0], inputs[3] }, attrs);
        return new[] { inputs[0] };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // QLinearConv: y = quantize(conv(dequantize(x), dequantize(w)) + B, y_scale, y_zero)
        // Inputs: x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, [B]
        if (ctx.Inputs.Length < 8)
        {
            reg.ElementWise.Fill(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount, 0f);
            return;
        }

        var x = ctx.Inputs[0]; var w = ctx.Inputs[3];
        var xShape = x.Shape; var wShape = w.Shape;
        if (xShape.Length < 4 || wShape.Length < 4)
        {
            reg.ElementWise.Fill(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount, 0f);
            return;
        }

        // CPU-side dequantization for correctness — Conv2D kernel needs contiguous
        // offset-0 buffers, not pool SubViews which have arbitrary offsets
        var xVals = ctx.TryGetInputValues(0);
        var wVals = ctx.TryGetInputValues(3);
        if (xVals == null || wVals == null)
        {
            // Can't read — fall back to direct conv without dequant
            int st = ctx.GetInts("strides", new[] { 1, 1 })[0];
            int pd = ctx.GetInts("pads", new int[4])[0];
            reg.Conv2D.Forward(x.Data, w.Data, default, ctx.Outputs[0].Data,
                xShape[1], xShape[2], xShape[3], wShape[0], wShape[2], wShape[3], st, pd);
            return;
        }

        // Dequantize: float_val = (int_val - zero_point) * scale
        var xScale = ctx.TryGetInputValues(1);
        var xZero = ctx.TryGetInputValues(2);
        var wScale = ctx.TryGetInputValues(4);
        var wZero = ctx.TryGetInputValues(5);

        var xDequant = (float[])xVals.Clone();
        var wDequant = (float[])wVals.Clone();

        float xZp = xZero != null && xZero.Length > 0 ? xZero[0] : 0f;
        float xSc = xScale != null && xScale.Length > 0 ? xScale[0] : 1f;
        float wZp = wZero != null && wZero.Length > 0 ? wZero[0] : 0f;
        float wSc = wScale != null && wScale.Length > 0 ? wScale[0] : 1f;

        for (int i = 0; i < xDequant.Length; i++) xDequant[i] = (xDequant[i] - xZp) * xSc;
        for (int i = 0; i < wDequant.Length; i++) wDequant[i] = (wDequant[i] - wZp) * wSc;

        // Upload via Pool.Rent — buffers stay alive past the WebGPU command encoder flush
        var xBufMem = ctx.Pool.Rent(new[] { xDequant.Length });
        xBufMem.Data.SubView(0, xDequant.Length).CopyFromCPU(xDequant);
        var wBufMem = ctx.Pool.Rent(new[] { wDequant.Length });
        wBufMem.Data.SubView(0, wDequant.Length).CopyFromCPU(wDequant);
        int stride = ctx.GetInts("strides", new[] { 1, 1 })[0];
        int pad = ctx.GetInts("pads", new int[4])[0];
        int oH = (xShape[2] + 2 * pad - wShape[2]) / stride + 1;
        int oW = (xShape[3] + 2 * pad - wShape[3]) / stride + 1;
        var outBufMem = ctx.Pool.Rent(new[] { xShape[0] * wShape[0] * oH * oW });
        // Conv2D always reads bias — must provide valid buffer. Zero-fill if no bias provided.
        var hasBias = ctx.Inputs.Length > 8 && ctx.Inputs[8] != null;
        ArrayView1D<float, Stride1D.Dense> biasView;
        if (hasBias)
        {
            biasView = ctx.Inputs[8].Data;
        }
        else
        {
            var zeroBias = ctx.Pool.Rent(new[] { wShape[0] });
            zeroBias.Data.SubView(0, wShape[0]).CopyFromCPU(new float[wShape[0]]);
            biasView = zeroBias.Data;
        }
        reg.Conv2D.Forward(xBufMem.Data, wBufMem.Data, biasView, outBufMem.Data,
            xShape[1], xShape[2], xShape[3], wShape[0], wShape[2], wShape[3], stride, pad);
        int copyLen = Math.Min(outBufMem.ElementCount, ctx.Outputs[0].ElementCount);
        reg.ElementWise.Scale(outBufMem.Data.SubView(0, copyLen), ctx.Outputs[0].Data.SubView(0, copyLen), copyLen, 1f);

        // Requantize output: y_quant = (y_float / y_scale) + y_zero
        var yScale = ctx.TryGetInputValues(6);
        var yZero = ctx.TryGetInputValues(7);
        float ySc = yScale != null && yScale.Length > 0 ? yScale[0] : 1f;
        float yZp = yZero != null && yZero.Length > 0 ? yZero[0] : 0f;
        if (ySc != 1f || yZp != 0f)
        {
            int outCount = ctx.Outputs[0].ElementCount;
            if (ySc != 1f && ySc != 0f)
                reg.ElementWise.ScaleInPlace(ctx.Outputs[0].Data, outCount, 1f / ySc);
            if (yZp != 0f)
            {
                var zpBuf = ctx.Pool.Rent(new[] { 1 });
                zpBuf.Data.SubView(0, 1).CopyFromCPU(new[] { yZp });
                reg.ElementWise.AddBias(ctx.Outputs[0].Data, zpBuf.Data, outCount, 1);
            }
        }
    }
}

public class QLinearMatMulOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "QLinearMatMul";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Inputs: a, a_scale, a_zero, b, b_scale, b_zero, y_scale, y_zero
        //
        // ⚠️ N-D, like MatMul and MatMulInteger. This returned a rank-2 shape unconditionally, which
        // DELETES a batch axis for the [seq, batch, features] activations every quantised transformer uses -
        // the identical defect found and fixed in MatMulIntegerOperator (ZipVoice int8 text encoder, node
        // 28: ours [106,192] vs onnxruntime [13,1,192]). Rank-2 behaviour is unchanged, since a[^2] == a[0]
        // and b[^1] == b[1] at rank 2.
        if (inputs.Length >= 4)
        {
            var a = inputs[0]; var b = inputs[3];
            int M = a.Length >= 2 ? a[^2] : 1;
            int N = b.Length >= 2 ? b[^1] : b[0];
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
        return new[] { inputs[0] };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // QLinearMatMul: y = quantize(matmul(dequantize(a), dequantize(b)), y_scale, y_zero)
        // Inputs: a, a_scale, a_zero, b, b_scale, b_zero, y_scale, y_zero
        if (ctx.Inputs.Length < 8)
        {
            reg.ElementWise.Fill(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount, 0f);
            return;
        }

        var a = ctx.Inputs[0]; var b = ctx.Inputs[3];
        var aShape = a.Shape; var bShape = b.Shape;
        // Contraction dim is the LAST axis; it coincides with axis 1 only at rank 2.
        int K = aShape[^1];
        int M = aShape.Length >= 2 ? aShape[^2] : 1;
        int N = bShape.Length > 1 ? bShape[^1] : bShape[0];
        // Leading axes are just more rows sharing the same weight (see MatMulOperator).
        int rows = K > 0 ? (int)(a.ElementCount / K) : M;

        // Dequantize a: (a - a_zero) * a_scale
        var aScale = ctx.TryGetInputValues(1);
        var aZero = ctx.TryGetInputValues(2);
        var aDequant = ctx.Pool.Rent(aShape, "_qlm_a");
        reg.ElementWise.Scale(a.Data, aDequant.Data, a.ElementCount, 1f);
        if (aZero != null && aZero.Length > 0)
        {
            var zpBuf = ctx.Pool.Rent(new[] { 1 }, "_qlm_azp");
            zpBuf.Data.SubView(0, 1).CopyFromCPU(new[] { -aZero[0] });
            reg.ElementWise.AddBias(aDequant.Data, zpBuf.Data, a.ElementCount, 1);
            ctx.Pool.Return(zpBuf);
        }
        if (aScale != null && aScale.Length > 0)
            reg.ElementWise.ScaleInPlace(aDequant.Data, a.ElementCount, aScale[0]);

        // Dequantize b: (b - b_zero) * b_scale
        var bScale = ctx.TryGetInputValues(4);
        var bZero = ctx.TryGetInputValues(5);
        var bDequant = ctx.Pool.Rent(bShape, "_qlm_b");
        reg.ElementWise.Scale(b.Data, bDequant.Data, b.ElementCount, 1f);
        if (bZero != null && bZero.Length > 0)
        {
            var zpBuf = ctx.Pool.Rent(new[] { 1 }, "_qlm_bzp");
            zpBuf.Data.SubView(0, 1).CopyFromCPU(new[] { -bZero[0] });
            reg.ElementWise.AddBias(bDequant.Data, zpBuf.Data, b.ElementCount, 1);
            ctx.Pool.Return(zpBuf);
        }
        if (bScale != null && bScale.Length > 0)
            reg.ElementWise.ScaleInPlace(bDequant.Data, b.ElementCount, bScale[0]);

        // MatMul. A 2-D b is a shared weight: flatten every leading axis of a into rows. Passing M here
        // computed only the first batch slice and left the remainder of the output untouched.
        if (bShape.Length <= 2)
        {
            reg.MatMul.MatMul(aDequant.Data, bDequant.Data, ctx.Outputs[0].Data, rows, K, N);
        }
        else
        {
            int batch = (M > 0 && K > 0) ? (int)(a.ElementCount / ((long)M * K)) : 1;
            reg.MatMul.BatchedMatMul(aDequant.Data, bDequant.Data, ctx.Outputs[0].Data, batch, M, K, N);
        }

        // Requantize: round(y / y_scale) + y_zero
        var yScale = ctx.TryGetInputValues(6);
        var yZero = ctx.TryGetInputValues(7);
        int outCount = ctx.Outputs[0].ElementCount;
        if (yScale != null && yScale.Length > 0 && yScale[0] != 0f)
            reg.ElementWise.ScaleInPlace(ctx.Outputs[0].Data, outCount, 1f / yScale[0]);
        if (yZero != null && yZero.Length > 0)
        {
            var zpBuf = ctx.Pool.Rent(new[] { 1 }, "_qlm_yzp");
            zpBuf.Data.SubView(0, 1).CopyFromCPU(new[] { yZero[0] });
            reg.ElementWise.AddBias(ctx.Outputs[0].Data, zpBuf.Data, outCount, 1);
            ctx.Pool.Return(zpBuf);
        }

        ctx.Pool.Return(aDequant);
        ctx.Pool.Return(bDequant);
    }
}
// DFT, STFT, MelWeightMatrix moved to SignalOperators.cs with full implementations
public class SequenceConstructOperator(OperatorRegistry reg) : IOnnxOperator { public string OpType => "SequenceConstruct"; public int[][] InferOutputShapes(int[][] i, Dictionary<string, object> a) => new[] { i[0] }; public void Execute(OnnxOpContext ctx) { if (ctx.Inputs.Length > 0) { int c = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount); reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, c), ctx.Outputs[0].Data.SubView(0, c), c, 1f); } } }
public class SequenceEmptyOperator(OperatorRegistry reg) : IOnnxOperator { public string OpType => "SequenceEmpty"; public int[][] InferOutputShapes(int[][] i, Dictionary<string, object> a) => new[] { new[] { 0 } }; public void Execute(OnnxOpContext ctx) { } }
public class SequenceAtOperator(OperatorRegistry reg) : IOnnxOperator { public string OpType => "SequenceAt"; public int[][] InferOutputShapes(int[][] i, Dictionary<string, object> a) => new[] { i[0] }; public void Execute(OnnxOpContext ctx) { int c = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount); if (c > 0) reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, c), ctx.Outputs[0].Data.SubView(0, c), c, 1f); } }
public class SequenceInsertOperator(OperatorRegistry reg) : IOnnxOperator { public string OpType => "SequenceInsert"; public int[][] InferOutputShapes(int[][] i, Dictionary<string, object> a) => new[] { i[0] }; public void Execute(OnnxOpContext ctx) { int c = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount); if (c > 0) reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, c), ctx.Outputs[0].Data.SubView(0, c), c, 1f); } }
public class SequenceEraseOperator(OperatorRegistry reg) : IOnnxOperator { public string OpType => "SequenceErase"; public int[][] InferOutputShapes(int[][] i, Dictionary<string, object> a) => new[] { i[0] }; public void Execute(OnnxOpContext ctx) { int c = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount); if (c > 0) reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, c), ctx.Outputs[0].Data.SubView(0, c), c, 1f); } }
public class SequenceLengthOperator(OperatorRegistry reg) : IOnnxOperator { public string OpType => "SequenceLength"; public int[][] InferOutputShapes(int[][] i, Dictionary<string, object> a) => new[] { new[] { 1 } }; public void Execute(OnnxOpContext ctx) => reg.ElementWise.Fill(ctx.Outputs[0].Data, 1, (float)ctx.Inputs.Length); }
public class SequenceMapOperator(OperatorRegistry reg) : IOnnxOperator { public string OpType => "SequenceMap"; public int[][] InferOutputShapes(int[][] i, Dictionary<string, object> a) => new[] { i[0] }; public void Execute(OnnxOpContext ctx) { int c = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount); if (c > 0) reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, c), ctx.Outputs[0].Data.SubView(0, c), c, 1f); } }
public class ConcatFromSequenceOperator(OperatorRegistry reg) : IOnnxOperator { public string OpType => "ConcatFromSequence"; public int[][] InferOutputShapes(int[][] i, Dictionary<string, object> a) => new[] { i[0] }; public void Execute(OnnxOpContext ctx) { int c = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount); if (c > 0) reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, c), ctx.Outputs[0].Data.SubView(0, c), c, 1f); } }
public class SplitToSequenceOperator(OperatorRegistry reg) : IOnnxOperator { public string OpType => "SplitToSequence"; public int[][] InferOutputShapes(int[][] i, Dictionary<string, object> a) => new[] { i[0] }; public void Execute(OnnxOpContext ctx) { int c = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount); if (c > 0) reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, c), ctx.Outputs[0].Data.SubView(0, c), c, 1f); } }
public class OptionalOperator(OperatorRegistry reg) : IOnnxOperator { public string OpType => "Optional"; public int[][] InferOutputShapes(int[][] i, Dictionary<string, object> a) => new[] { i.Length > 0 ? i[0] : new[] { 1 } }; public void Execute(OnnxOpContext ctx) { if (ctx.Inputs.Length > 0) { int c = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount); if (c > 0) reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, c), ctx.Outputs[0].Data.SubView(0, c), c, 1f); } } }
public class OptionalGetElementOperator(OperatorRegistry reg) : IOnnxOperator { public string OpType => "OptionalGetElement"; public int[][] InferOutputShapes(int[][] i, Dictionary<string, object> a) => new[] { i[0] }; public void Execute(OnnxOpContext ctx) { int c = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount); if (c > 0) reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, c), ctx.Outputs[0].Data.SubView(0, c), c, 1f); } }
public class OptionalHasElementOperator(OperatorRegistry reg) : IOnnxOperator { public string OpType => "OptionalHasElement"; public int[][] InferOutputShapes(int[][] i, Dictionary<string, object> a) => new[] { new[] { 1 } }; public void Execute(OnnxOpContext ctx) => reg.ElementWise.Fill(ctx.Outputs[0].Data, 1, ctx.Inputs.Length > 0 ? 1f : 0f); }
// String operators: ONNX string type is not representable as GPU float tensors.
// These operators pass through input data as-is. Models using string ops typically
// have a preprocessing graph that converts strings to token IDs before the main
// inference graph — by that point, data is float and string ops are not in the path.
public class StringConcatOperator(OperatorRegistry reg) : IOnnxOperator { public string OpType => "StringConcat"; public int[][] InferOutputShapes(int[][] i, Dictionary<string, object> a) => new[] { i[0] }; public void Execute(OnnxOpContext ctx) { int c = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount); if (c > 0) reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, c), ctx.Outputs[0].Data.SubView(0, c), c, 1f); } }
public class StringNormalizerOperator(OperatorRegistry reg) : IOnnxOperator { public string OpType => "StringNormalizer"; public int[][] InferOutputShapes(int[][] i, Dictionary<string, object> a) => new[] { i[0] }; public void Execute(OnnxOpContext ctx) { int c = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount); if (c > 0) reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, c), ctx.Outputs[0].Data.SubView(0, c), c, 1f); } }
public class StringSplitOperator(OperatorRegistry reg) : IOnnxOperator { public string OpType => "StringSplit"; public int[][] InferOutputShapes(int[][] i, Dictionary<string, object> a) => new[] { i[0], i[0], new[] { 1 } }; public void Execute(OnnxOpContext ctx) { int c = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount); if (c > 0) reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, c), ctx.Outputs[0].Data.SubView(0, c), c, 1f); } }
// ── Control flow operators with real subgraph execution ──
// If/Loop/Scan compile embedded ONNX subgraphs and execute them via GraphCompiler+GraphExecutor.
// Subgraphs are stored as OnnxGraphProto in operator attributes (then_branch, else_branch, body).

/// <summary>Helper: compile and execute a subgraph with given inputs.</summary>
/// <summary>Resolves the values a control-flow body reads from the enclosing graph.</summary>
internal static class OuterScope
{
    /// <summary>
    /// Add every tensor <paramref name="subgraph"/> references but does not itself produce.
    /// </summary>
    /// <remarks>
    /// ⚠️ ONNX subgraphs capture from the enclosing scope implicitly - a branch body can name any value
    /// visible where the node sits, without declaring it. An If node carries only its condition, so
    /// without this the body receives nothing and fails to compile ("shapes=(?; ...)") or reads a tensor
    /// that was never bound.
    ///
    /// "References but does not produce" is the whole rule: anything a node inside the body outputs, or
    /// that is an initializer or a declared input, belongs to the body. Everything else must come from
    /// outside.
    /// </remarks>
    /// <summary>ML_TRACE_OUTER_SCOPE=1 reports what a branch captured, and what it could not.</summary>
    internal static readonly bool TraceOuterScope =
        Environment.GetEnvironmentVariable("ML_TRACE_OUTER_SCOPE") == "1";

    public static void Add(OnnxOpContext ctx, Onnx.OnnxGraphProto subgraph, Dictionary<string, Tensor> into)
    {
        var scope = ctx.ScopeTensors;
        if (scope == null) return;

        var produced = new HashSet<string>(StringComparer.Ordinal);
        foreach (var n in subgraph.Nodes)
            foreach (var o in n.Outputs)
                if (!string.IsNullOrEmpty(o)) produced.Add(o);
        foreach (var init in subgraph.Initializers) produced.Add(init.Name);
        foreach (var inp in subgraph.Inputs) produced.Add(inp.Name);

        foreach (var n in subgraph.Nodes)
            foreach (var inName in n.Inputs)
            {
                if (string.IsNullOrEmpty(inName) || produced.Contains(inName)) continue;
                if (into.ContainsKey(inName)) continue;
                if (scope.TryGetValue(inName, out var t)) { into[inName] = t; continue; }

                // ⚠️ Not every outer value is a TENSOR. Small shape/index values are resolved on the CPU and
                // their dispatches ELIDED, so they live in runtimeConstants and never appear in the tensor
                // map - which is exactly what a branch reading `Gather_output_0` needs. Materialise them
                // into a small buffer so the body is self-contained: the compiler gets a shape (without one
                // it infers nothing and crashes on the first node) and the executor gets a value.
                //
                // Rented under a stable name, so this reuses one buffer per value rather than allocating
                // per execution - a per-call device allocation is what makes a graph uncapturable.
                if (ctx.ConstantValues != null && ctx.ConstantValues.TryGetValue(inName, out var vals)
                    && vals != null && vals.Length > 0)
                {
                    var buf = ctx.Pool.Rent(new[] { vals.Length }, "_outerscope_" + inName);
                    buf.Data.SubView(0, vals.Length).CopyFromCPU(vals);
                    into[inName] = new Tensor(buf.Data.SubView(0, vals.Length), new[] { vals.Length }, inName);
                    if (TraceOuterScope)
                        Console.WriteLine($"[OuterScope] '{inName}' materialised from runtimeConstants "
                                        + $"({vals.Length} value(s)) - it was elided, not a live tensor");
                    continue;
                }

                if (TraceOuterScope)
                    // A referenced value the enclosing scope does not have. Naming it matters: the compile
                    // failure downstream reports only "shapes=(?)", which cannot distinguish "never
                    // captured" from "captured with an unknown shape".
                    Console.WriteLine($"[OuterScope] '{inName}' referenced by the branch but NOT present in "
                                    + $"the enclosing scope ({scope.Count} tensors live)");
            }
        if (TraceOuterScope)
            Console.WriteLine($"[OuterScope] captured {into.Count} tensor(s) for the branch: "
                            + string.Join(", ", into.Keys.Take(8)));
    }
}

internal static class SubgraphRunner
{
    /// <summary>Control-flow BODY executions - every If/Loop/Scan body, not just If.</summary>
    /// <remarks>
    /// ⚠️ Counted HERE rather than in IfOperator on purpose. The hazard capture guards against is a device
    /// allocation inside the recording window, and it comes from executing a body - any body. Counting only
    /// If would report zero for a graph whose Loop ran every call, and a capture decision made on that
    /// number would be exactly wrong.
    /// </remarks>
    public static int ExecutionCount;

    /// <summary>Zero the body-execution counter, so a capture decision covers a known window.</summary>
    public static void ResetExecutionCount() => ExecutionCount = 0;

    /// <summary>
    /// Execute a subgraph (OnnxGraphProto) with the given input tensors.
    /// Returns output tensors. The caller is responsible for copying results to their output buffers.
    /// </summary>
    public static Dictionary<string, Tensor>? Execute(
        OnnxOpContext ctx, Onnx.OnnxGraphProto subgraph,
        Dictionary<string, Tensor> subgraphInputs)
    {
        System.Threading.Interlocked.Increment(ref ExecutionCount);
        var plan = GetOrBuildPlan(ctx, subgraph, subgraphInputs);
        if (plan == null) return null;
        return MergeDeclaredOutputs(subgraph, plan.Executor.Run(subgraphInputs), plan.Constants);
    }

    /// <summary>
    /// Adds any declared subgraph output that no NODE produces, taking it from the weights.
    /// </summary>
    /// <remarks>
    /// ⚠️ A branch is very often a single <c>Constant</c> holding a table. Those are folded into
    /// initializers so the Constant operator cannot overwrite the table with the scalar
    /// <c>ExtractTensorScalar</c> hands it - which leaves the subgraph with ZERO nodes and its declared
    /// output produced by nothing. The executor then returns an empty result and the caller copies nothing,
    /// so the branch silently yields zeros. Declared outputs are part of the contract whether a node
    /// computed them or not.
    /// </remarks>
    private static Dictionary<string, Tensor>? MergeDeclaredOutputs(
        Onnx.OnnxGraphProto subgraph, Dictionary<string, Tensor>? result,
        Dictionary<string, Tensor> weights)
    {
        result ??= new Dictionary<string, Tensor>();
        foreach (var declared in subgraph.Outputs)
        {
            if (string.IsNullOrEmpty(declared.Name) || result.ContainsKey(declared.Name)) continue;
            if (weights.TryGetValue(declared.Name, out var t)) result[declared.Name] = t;
        }
        return result;
    }

    /// <summary>
    /// Browser-safe async subgraph execution (used by the control-flow operators' ExecuteAsync).
    /// Identical compile/weight setup as <see cref="Execute"/> but drives the subgraph through
    /// <c>GraphExecutor.RunAsync</c>, so any GPU-&gt;CPU readback inside the subgraph (or in the
    /// operators it contains) uses the async path that works on WebGPU/WebGL/Wasm.
    /// </summary>
    public static async Task<Dictionary<string, Tensor>?> ExecuteAsync(
        OnnxOpContext ctx, Onnx.OnnxGraphProto subgraph,
        Dictionary<string, Tensor> subgraphInputs)
    {
        // ⚠️ The SAME cached plan as the sync path. This is the entry point the control-flow operators
        // actually use (IfOperator.ExecuteAsync -> here), so caching only the sync path left every real
        // execution rebuilding - and the crash it was meant to prevent still fired, from this line.
        System.Threading.Interlocked.Increment(ref ExecutionCount);
        var plan = GetOrBuildPlan(ctx, subgraph, subgraphInputs);
        if (plan == null) return null;
        return MergeDeclaredOutputs(subgraph, await plan.Executor.RunAsync(subgraphInputs), plan.Constants);
    }

    /// <summary>
    /// Shared subgraph setup: converts the OnnxGraphProto to ModelGraph IR, compiles it, builds
    /// the weight map (subgraph initializers + outer-scope tensors), and returns a ready
    /// GraphExecutor. Returns null when there is no registry. Pure setup — no execution.
    /// </summary>
    /// <summary>
    /// The compiled plan for this subgraph at these input shapes - built once, reused after.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠️ <c>BuildExecutor</c> used to run on EVERY execution, and it is not cheap: it converts the graph
    /// to IR, runs the full <see cref="Graph.GraphCompiler"/>, and calls <c>Pool.AllocatePermanent</c> for
    /// every initializer and Constant table. ZipVoice's If branch is a single Constant holding a
    /// <c>[1999, 48]</c> positional table - 384 KB allocated PERMANENTLY, per call, never freed.
    /// </para>
    /// <para>
    /// ⚠️ It also made control flow UNCAPTURABLE, which is the expensive part. A device allocation inside a
    /// capture window is unrecoverable - an uncatchable 0xC0000005 on CUDA, a hung device on WebGPU - so
    /// <see cref="Graph.SessionGraphCapture"/> has to refuse any graph containing If/Loop/Scan. In the
    /// browser that refusal costs about 20x: ZipVoice's decoder spends ~4.5 ms per node on interop
    /// crossings that a replayed plan does in microseconds (measured: 0 readbacks, 575 ms of syncs, and
    /// 38.7 s across ~8,520 node executions).
    /// </para>
    /// <para>
    /// ⚠️ The executor's <c>_weights</c> OVERRIDE the tensors passed to <c>Run</c> - weights are registered
    /// after inputs. The old per-call build merged this call's outer-scope tensors into weights, which was
    /// correct only because the executor was thrown away afterwards. A cached executor carrying them would
    /// let the FIRST call's outer-scope tensors shadow every later call's inputs forever, silently. So the
    /// plan keeps ONLY constants, and outer-scope tensors reach the body through <c>Run</c> alone - where
    /// they are registered, ref-counted as external, and never released.
    /// </para>
    /// </remarks>
    private static OperatorRegistry.SubgraphPlan? GetOrBuildPlan(
        OnnxOpContext ctx, Onnx.OnnxGraphProto subgraph, Dictionary<string, Tensor> subgraphInputs)
    {
        if (ctx.Registry == null) return null;

        // A GraphExecutor is shape-specialised, so the shapes are part of the identity.
        var sig = string.Join("|", subgraphInputs.OrderBy(kv => kv.Key, StringComparer.Ordinal)
            .Select(kv => $"{kv.Key}:{string.Join(",", kv.Value.Shape)}"));

        if (ctx.Registry.SubgraphPlans.TryGetValue(sig, out var bucket))
            foreach (var candidate in bucket)
                if (ReferenceEquals(candidate.Subgraph, subgraph))
                    return candidate;

        var executor = BuildExecutor(ctx, subgraph, subgraphInputs, out var constants);
        if (executor == null) return null;

        var plan = new OperatorRegistry.SubgraphPlan
        {
            Subgraph = subgraph, Executor = executor, Constants = constants,
        };
        if (bucket == null) ctx.Registry.SubgraphPlans[sig] = bucket = new List<OperatorRegistry.SubgraphPlan>();
        bucket.Add(plan);
        return plan;
    }

    private static Graph.GraphExecutor? BuildExecutor(
        OnnxOpContext ctx, Onnx.OnnxGraphProto subgraph,
        Dictionary<string, Tensor> subgraphInputs,
        out Dictionary<string, Tensor> weightsOut)
    {
        weightsOut = new Dictionary<string, Tensor>();
        if (ctx.Registry == null) return null;

        // Convert OnnxGraphProto to ModelGraph IR
        var modelGraph = ConvertToModelGraph(subgraph, subgraphInputs);

        // Compile
        var compiler = new Graph.GraphCompiler(ctx.Registry);
        var compiled = compiler.Compile(modelGraph);

        // Build weights from subgraph initializers
        var weights = new Dictionary<string, Tensor>();
        // Runtime constants visible to the body: the enclosing graph's, plus the body's own Constant nodes.
        var subgraphConstants = ctx.ConstantValues != null
            ? new Dictionary<string, float[]>(ctx.ConstantValues)
            : new Dictionary<string, float[]>();
        foreach (var init in subgraph.Initializers)
        {
            var floats = init.ToFloatArray();
            if (floats.Length > 0)
            {
                var shape = init.Dims.Select(d => (int)d).ToArray();
                if (shape.Length == 0) shape = new[] { floats.Length };
                weights[init.Name] = ctx.Pool.AllocatePermanent(floats, shape, init.Name);
            }
        }

        // A subgraph usually carries a lookup table as a Constant NODE rather than an initializer, and
        // initializers alone miss it. ZipVoice's If then_branch is exactly one Constant holding the
        // [1999, 48] relative positional-encoding table.
        // ⚠️ It cannot be left to the Constant operator either: a TENSOR attribute reaches an operator
        // through ConvertAttribute -> ExtractTensorScalar, which returns ONE number. Before this, the
        // branch produced a correctly shaped tensor of ZEROS.
        foreach (var cn in subgraph.Nodes)
        {
            if (cn.OpType != "Constant" || cn.Outputs.Count == 0 || string.IsNullOrEmpty(cn.Outputs[0])) continue;
            var valueAttr = cn.Attributes.FirstOrDefault(a => a.Name == "value");
            if (valueAttr?.T == null) continue;
            var cfloats = valueAttr.T.ToFloatArray();
            if (cfloats.Length == 0) continue;
            var cshape = valueAttr.T.Dims.Select(d => (int)d).ToArray();
            if (cshape.Length == 0) cshape = new[] { cfloats.Length };
            weights[cn.Outputs[0]] = ctx.Pool.AllocatePermanent(cfloats, cshape, cn.Outputs[0]);
            // ⚠️ ALSO as a runtime constant, not only as a GPU tensor. Ops that need a value on the CPU -
            // Range's start/limit/delta, Slice's bounds, Reshape's dims - read runtimeConstants, and a
            // Constant node declared INSIDE a subgraph never reached that map: the branch compiled and then
            // died with "Range: scalar inputs not available as runtime constants ... delta=null" while the
            // very same value sat in weights as a buffer.
            //
            // Bounded deliberately: a lookup table belongs on the GPU and nothing reads it as a scalar. The
            // 4096 cap keeps the [1999, 48] positional table (95,952 values) out of this map while letting
            // through the scalars and short vectors that shape arithmetic actually consumes.
            if (cfloats.Length <= 4096)
                subgraphConstants[cn.Outputs[0]] = cfloats;
        }

        // ⚠️ Outer-scope tensors are deliberately NOT merged in here any more. They reach the body through
        // Run(subgraphInputs), which registers them and ref-counts them as external ("never release") -
        // so nothing is lost - and keeping them OUT of weights is what makes this executor reusable.
        // Weights override Run's inputs (registered after them), so a cached executor holding call 1's
        // outer-scope tensors would shadow every later call's inputs, forever, without an error.

        weightsOut = weights;
        return new Graph.GraphExecutor(
            ctx.Registry.Accelerator, compiled, weights,
            subgraphConstants, registry: ctx.Registry);
    }

    /// <param name="outerScope">
    /// Tensors the subgraph reads from the ENCLOSING graph, with their runtime shapes.
    /// </param>
    /// <remarks>
    /// ⚠️ ONNX lets a subgraph reference values from the enclosing scope without declaring them as inputs,
    /// and they were never declared here - so the compiler had no shape for them and inferred nothing
    /// downstream. That is invisible while every branch a model actually takes happens to be simple.
    ///
    /// ZipVoice's decoder made it visible: an utterance longer than its precomputed [1999, 48] positional
    /// table takes a DIFFERENT If branch, 156 nodes that read the parent's Gather output. Compiling it
    /// crashed with "Node 0/156 'Sub' ... shapes=(?; [1])" - the `?` being the outer-scope value. Short
    /// utterances take the other branch (a single Constant), which is why every test until now passed and
    /// the first realistic chat reply did not.
    ///
    /// The shapes are known at runtime, so they are declared here.
    /// </remarks>
    private static Graph.ModelGraph ConvertToModelGraph(
        Onnx.OnnxGraphProto onnxGraph, Dictionary<string, Tensor>? outerScope = null)
    {
        var graph = new Graph.ModelGraph { Name = onnxGraph.Name };

        foreach (var input in onnxGraph.Inputs)
        {
            if (onnxGraph.Initializers.Any(i => i.Name == input.Name)) continue;
            graph.Inputs.Add(new Graph.GraphValueInfo
            {
                Name = input.Name,
                Shape = input.Shape?.Select(d => (int)(d.DimValue ?? 1)).ToArray() ?? new[] { 1 }
            });
        }

        if (outerScope != null)
        {
            var declared = new HashSet<string>(graph.Inputs.Select(i => i.Name), StringComparer.Ordinal);
            foreach (var (name, tensor) in outerScope)
            {
                if (declared.Contains(name)) continue;
                if (onnxGraph.Initializers.Any(i => i.Name == name)) continue;
                graph.Inputs.Add(new Graph.GraphValueInfo
                {
                    Name = name,
                    Shape = (int[])tensor.Shape.Clone(),
                });
            }
        }

        foreach (var output in onnxGraph.Outputs)
        {
            graph.Outputs.Add(new Graph.GraphValueInfo
            {
                Name = output.Name,
                Shape = output.Shape?.Select(d => (int)(d.DimValue ?? 1)).ToArray() ?? new[] { 1 }
            });
        }

        // ONNX rank-0 is recorded, never flattened away - see ModelGraph.ScalarTensorNames. Storage stays
        // 1-element rank-1; only shape inference needs to tell a scalar from a [1] vector.
        graph.ScalarTensorNames = new HashSet<string>(StringComparer.Ordinal);

        foreach (var init in onnxGraph.Initializers)
        {
            graph.Initializers[init.Name] = init.Dims.Select(d => (int)d).ToArray();
            if (!init.Dims.Any()) graph.ScalarTensorNames.Add(init.Name);
        }

        // Fold Constant NODES holding a real tensor into initializers: BuildExecutor supplies their data,
        // so the node itself must not run - it would overwrite the table with the scalar that
        // ExtractTensorScalar hands the Constant operator.
        var foldedConstants = new HashSet<string>(StringComparer.Ordinal);
        foreach (var n in onnxGraph.Nodes)
        {
            if (n.OpType != "Constant" || n.Outputs.Count == 0 || string.IsNullOrEmpty(n.Outputs[0])) continue;
            var valueAttr = n.Attributes.FirstOrDefault(a => a.Name == "value");
            if (valueAttr?.T == null) continue;
            var dims = valueAttr.T.Dims.Select(d => (int)d).ToArray();
            // A rank-0 Constant is STORED as [1] (a pool buffer cannot be rank-0), but its true rank is
            // remembered so Gather can apply the ONNX rule instead of guessing from the value's length.
            if (dims.Length == 0)
            {
                dims = new[] { Math.Max(1, (int)valueAttr.T.ElementCount) };
                if (valueAttr.T.ElementCount <= 1) graph.ScalarTensorNames.Add(n.Outputs[0]);
            }
            graph.Initializers[n.Outputs[0]] = dims;
            foldedConstants.Add(n.Outputs[0]);
        }

        if (Environment.GetEnvironmentVariable("ML_TRACE_SCALARS") == "1")
            Console.WriteLine($"[scalars] subgraph '{onnxGraph.Name}': {graph.ScalarTensorNames.Count} rank-0 of "
                + $"{onnxGraph.Nodes.Count} nodes -> {string.Join(", ", graph.ScalarTensorNames.Where(n => n.Contains("Constant_13") || n.Contains("Constant_1_")))}");

        // Convert node attributes to the typed dictionary format expected by GraphNode
        graph.Nodes = onnxGraph.Nodes.Where(n => !(n.OpType == "Constant" && n.Outputs.Count > 0
                                                  && foldedConstants.Contains(n.Outputs[0]))).Select(n =>
        {
            var typedAttrs = n.Attributes.ToDictionary(
                a => a.Name,
                a => Onnx.OnnxLoader.ConvertAttributePublic(a));

            // Serialize typed attrs to JsonElement for GraphNode (which uses JsonElement)
            Dictionary<string, System.Text.Json.JsonElement>? jsonAttrs = null;
            if (typedAttrs.Count > 0)
            {
                jsonAttrs = new Dictionary<string, System.Text.Json.JsonElement>();
                foreach (var (key, value) in typedAttrs)
                {
                    // A nested subgraph cannot be serialised to JSON; it travels out of band, the same
                    // way the outer graph carries one (see GraphNode.RawAttributes). Dropping it here
                    // silently disabled nested control flow.
                    if (value is Onnx.OnnxGraphProto) continue;   // handled below via RawAttributes
                    try
                    {
                        var json = System.Text.Json.JsonSerializer.Serialize(value);
                        jsonAttrs[key] = System.Text.Json.JsonDocument.Parse(json).RootElement.Clone();
                    }
                    catch { /* Skip non-serializable attributes */ }
                }
            }

            Dictionary<string, object>? rawAttrs = null;
            foreach (var (key, value) in typedAttrs)
                if (value is Onnx.OnnxGraphProto)
                    (rawAttrs ??= new Dictionary<string, object>())[key] = value;

            return new Graph.GraphNode
            {
                OpType = n.OpType,
                Inputs = n.Inputs.ToList(),
                Outputs = n.Outputs.ToList(),
                Attributes = jsonAttrs,
                RawAttributes = rawAttrs,
            };
        }).ToList();

        return graph;
    }
}

public class IfOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "If";

    /// <summary>How many times each branch has been taken since <see cref="ResetBranchCensus"/>.</summary>
    /// <remarks>
    /// ⚠️ WHY THESE EXIST. Which branch an <c>If</c> takes is not a detail - it can be most of a model's
    /// cost. MEASURED 2026-09-03: ZipVoice's <c>fm_decoder</c> is 8,621 nodes with FIVE <c>If</c> nodes,
    /// each the standard "reuse the cached positional table if it is long enough, else rebuild it". The
    /// then branch is a SINGLE <c>Constant</c>; the else branch is <b>254 nodes</b>. So if the else branch
    /// is being taken, every Euler step runs 1,270 extra nodes to rebuild a table that depends only on the
    /// sequence length - which does not change across the steps of one utterance - and at the ~4.3 ms per
    /// node this engine costs in a browser that is seconds per step, on the stage that is 82% of a
    /// synthesis.
    /// <para>
    /// Two interlocked ints, incremented per execution: no allocation, no reflection, no logging, so this
    /// is not the kind of always-on diagnostic that taxes a sweep. A boolean "does it have control flow"
    /// cannot answer the question that decides the work.
    /// </para>
    /// </remarks>
    public static int ThenBranchCount;
    /// <summary>Times the else branch has been taken. See <see cref="ThenBranchCount"/>.</summary>
    public static int ElseBranchCount;

    /// <summary>Zero the branch census, so a measurement covers a known window.</summary>
    public static void ResetBranchCensus() { ThenBranchCount = 0; ElseBranchCount = 0; }

    /// <summary>
    /// 🔴 DIAGNOSTIC ONLY - skip the branch subgraph entirely, leaving the output UNWRITTEN.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠️ THIS PRODUCES WRONG RESULTS BY DESIGN. It exists to answer one question that cannot be answered
    /// by reasoning: <b>is executing a branch subgraph inside a capture window what hangs the device?</b>
    /// </para>
    /// <para>
    /// MEASURED 2026-09-03: <c>SessionGraphCapture</c> refuses any graph containing control flow, so
    /// ZipVoice's decoder - 80% of a synthesis, ~8,621 nodes at roughly 1 ms per node in a browser - is
    /// never captured. Lifting the refusal hung the GPU outright (DXGI_ERROR_DEVICE_HUNG) even though
    /// <c>SubgraphRunner</c> caches its compiled plans and <c>WebGPUGraphCapture</c> proves its pool is
    /// primed. Census in the same run: <c>then=21, else=0</c> - every If takes the single-<c>Constant</c>
    /// branch, so the 254-node branch is not involved at all.
    /// </para>
    /// <para>
    /// With this set, an If does no subgraph work whatsoever. If capture then goes LIVE and the device
    /// survives, the subgraph execution is the cause and folding these Ifs away is worth building. If it
    /// still hangs, folding would not have helped and the cause is elsewhere - which is exactly the thing
    /// worth knowing BEFORE writing a constant-folding pass, a weight-hoisting path and a compiler stage.
    /// </para>
    /// </remarks>
    public static bool BypassSubgraphForCaptureProbe { get; set; }

    /// <summary>
    /// How many times a control-flow body has actually been run through <see cref="SubgraphRunner"/>.
    /// </summary>
    /// <remarks>
    /// ⚠️ This is what makes capture decidable by OBSERVATION rather than by a blanket refusal. The hazard
    /// capture guards against is a device allocation inside the recording window, and that comes from
    /// executing a subgraph there - MEASURED 2026-09-03: with the subgraph execution removed, the same
    /// decoder captured 8,197 dispatches with no device hang, and its steady-state Euler step went
    /// 8,578 ms -> 147 ms. The mere PRESENCE of an If is not the problem; running a body is.
    /// <para>
    /// So <see cref="Graph.SessionGraphCapture"/> watches this counter across a warm forward: zero means no
    /// body ran and recording is safe, non-zero means refuse exactly as before. A branch that is a single
    /// <c>Constant</c> never touches SubgraphRunner (see <see cref="TryWriteConstantBranch"/>), which is
    /// what takes ZipVoice's five Ifs out of the window.
    /// </para>
    /// </remarks>
    /// <summary>Zero the subgraph-execution counter, so a capture decision covers a known window.</summary>
    // Materialised single-Constant branches. Keyed by the branch proto, which is long-lived and
    // reference-stable for the life of the session.
    private static readonly Dictionary<Onnx.OnnxGraphProto, (float[] Values, int[] Shape)> _constBranchValues = new();
    private static readonly object _constBranchLock = new();

    /// <summary>
    /// Write a branch that is exactly ONE <c>Constant</c> node straight into the output, without
    /// <see cref="SubgraphRunner"/>. Returns false for anything else, leaving the normal path to run.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠️ WHY IT MATTERS FAR MORE THAN IT LOOKS. This is the entire reason ZipVoice's decoder can be
    /// captured. Its five Ifs each choose between a cached positional table (ONE Constant) and a 254-node
    /// recomputation, and a census over a real utterance says <c>then=21, else=0</c> - the big branch never
    /// runs. Sending that trivial case through a subgraph executor is what put an allocation inside the
    /// capture window and hung the device; writing it directly removes the body entirely.
    /// </para>
    /// <para>
    /// ⚠️ The buffer is rented under a STABLE name and written ONCE. That is deliberate on two counts. A
    /// per-call rent is an allocation, which is precisely what makes a graph uncapturable. And the upload
    /// is a <c>queue.writeBuffer</c>, not a dispatch, so a recorded plan would not replay it - writing once
    /// into a buffer nothing else touches means the value is already correct on every replay, rather than
    /// depending on a write that the plan cannot contain.
    /// </para>
    /// <para>
    /// ⚠️ Conservative on size: it declines unless the output buffer's element count matches the constant
    /// exactly. An If cannot be sized statically in general - ONNX requires branches to agree on rank, not
    /// dims - so a mismatch means the compile-time shape came from the OTHER branch, and
    /// <see cref="SubgraphOutputCopy"/>'s adoption path must handle it. Declining is always safe.
    /// </para>
    /// </remarks>
    private static bool TryWriteConstantBranch(OnnxOpContext ctx, Onnx.OnnxGraphProto sub)
    {
        if (sub.Nodes.Count != 1 || sub.Outputs.Count != 1 || ctx.Outputs.Length < 1) return false;
        var only = sub.Nodes[0];
        if (only.OpType != "Constant" || only.Outputs.Count < 1) return false;
        if (!string.Equals(only.Outputs[0], sub.Outputs[0].Name, StringComparison.Ordinal)) return false;

        float[] values; int[] shape;
        lock (_constBranchLock)
        {
            if (!_constBranchValues.TryGetValue(sub, out var cached))
            {
                var valueAttr = only.Attributes.FirstOrDefault(a => a.Name == "value");
                if (valueAttr?.T == null) return false;
                cached = (valueAttr.T.ToFloatArray(), valueAttr.T.Dims.Select(d => (int)d).ToArray());
                _constBranchValues[sub] = cached;
            }
            (values, shape) = cached;
        }
        if (values.Length == 0) return false;
        if (shape.Length == 0 || TensorHelpers.ElementCount(shape) != values.Length)
            shape = new[] { values.Length };

        // 🔴 A STABLE, NAMED buffer - not ctx.Outputs[0]. This is the difference between correct and
        // confidently wrong, and it was MEASURED the wrong way round first.
        //
        // Writing into the node's own output buffer works right up until the graph is CAPTURED. A replayed
        // plan re-executes recorded DISPATCHES only, so this operator never runs again - while the buffer it
        // wrote into is an ordinary pooled intermediate that the pool recycles to other tensors between
        // runs. The plan then reads whatever landed there. MEASURED 2026-09-03 on the first attempt:
        // replaying the captured decoder changed ALL 73,216 samples, worst 0.452549 - a completely
        // different utterance, still fluent enough to pass every amplitude and duration check. Only the
        // sample-level A/B against a direct forward caught it.
        //
        // A NAMED rent is retained by the pool under that name (BufferPool._namedBuffers) and handed back
        // for the same name every time, so nothing else is ever given this memory and the value survives
        // for the life of the plan. Same mechanism SubgraphOutputCopy uses, for the same reason.
        var stable = ctx.Pool.Rent(shape, "_ifconst_" + sub.Outputs[0].Name);

        // Uploaded on every real execution rather than once. A queue write is cheap, a stale table is not,
        // and "write once" would need to prove the pool never rebound the name - which this does not need
        // to know. Replays skip this entirely and read the value the last real execution left.
        stable.Data.SubView(0, values.Length).CopyFromCPU(values);
        ctx.Outputs[0] = stable;   // aliases the executor's nodeOutputs, exactly as SubgraphOutputCopy does
        return true;
    }

    private static void CountBranch(bool condition)
    {
        if (condition) System.Threading.Interlocked.Increment(ref ThenBranchCount);
        else System.Threading.Interlocked.Increment(ref ElseBranchCount);
    }

    /// <summary>
    /// Output shapes come from the BRANCH SUBGRAPHS, which declare them, and never from the inputs.
    /// </summary>
    /// <remarks>
    /// ⚠️ This used to return <c>inputs[0]</c> - and <c>inputs[0]</c> of an If is the CONDITION, a bool
    /// scalar. So the output buffer was allocated with ONE element for whatever the branch produced, and
    /// <see cref="Execute"/>'s <c>Math.Min</c> clamp then silently truncated the branch's real result to
    /// that single value. No error, no wrong shape reported upward - just a scalar where a tensor belongs.
    /// <para>
    /// MEASURED on ZipVoice's text encoder, which reaches its relative positional-encoding table through
    /// exactly one If (the standard "reuse the cached table if it is long enough, else extend it" shape):
    /// onnxruntime returns <c>[1999, 48]</c> of sin/cos values in [-1, 1]; we returned <c>[1]</c> holding
    /// 1.0. Every relative-position bias in all four encoder layers was therefore computed from a scalar,
    /// which is why the encoder diverged from ORT by 18.6% of peak while still producing correct SHAPES
    /// downstream - <c>linear_pos</c> collapsed [1, 25, 16] to [1, 16] and nothing complained.
    /// </para>
    /// <para>
    /// ONNX requires both branches to agree on output types and ranks, so either declaration is a valid
    /// source. A branch whose dims are fully static is preferred, since a symbolic one cannot size a
    /// buffer; then_branch is tried first only to be deterministic.
    /// </para>
    /// </remarks>
    public int[][] InferOutputShapes(int[][] i, Dictionary<string, object> a)
    {
        foreach (var key in new[] { "then_branch", "else_branch" })
        {

            if (a.TryGetValue(key, out var obj) && obj is Onnx.OnnxGraphProto sub
                && TryDeclaredOutputShapes(sub, out var shapes))
                return shapes;
        }

        // Neither branch declares usable dims. Returning the condition's shape would be the old silent
        // truncation, so fall back to a single element and let the runtime resize path handle it.
        return new[] { i.Length > 1 ? i[1] : new[] { 1 } };
    }

    private static bool TryDeclaredOutputShapes(Onnx.OnnxGraphProto sub, out int[][] shapes)
        => SubgraphShapes.TryDeclaredOutputShapes(sub, out shapes);
    public void Execute(OnnxOpContext ctx)
    {
        // If: evaluate condition scalar, execute then_branch or else_branch subgraph
        // Input[0] = condition (bool scalar), Attrs: then_branch (GraphProto), else_branch (GraphProto)
        // ⚠️ ctx.TryGetInputValues returns COMPILE-TIME CONSTANTS ONLY, so a condition computed at runtime
        // read as null and this defaulted to FALSE - the else branch, always, regardless of the condition.
        bool condition = false;
        var condVals = OperatorInputReader.Read(reg, ctx, 0);
        if (condVals != null && condVals.Length > 0)
            condition = condVals[0] != 0f;
        else if (ctx.TryGetInputValues(0) == null)
            throw new NotSupportedException(
                "If could not read its condition. On a browser backend this operator needs the async path "
                + "(GraphExecutor.RunAsync); defaulting to the else branch silently picks the wrong one.");

        // Select branch subgraph
        CountBranch(condition);
        if (BypassSubgraphForCaptureProbe) return;   // DIAGNOSTIC: leaves the output unwritten
        string branchKey = condition ? "then_branch" : "else_branch";
        if (ctx.Attributes.TryGetValue(branchKey, out var branchObj) && branchObj is Onnx.OnnxGraphProto subgraph)
        {
            // A branch that is one Constant is written directly - no SubgraphRunner, so nothing
            // allocates inside a capture window. See TryWriteConstantBranch.
            if (TryWriteConstantBranch(ctx, subgraph)) return;
            // Subgraph inputs reference outer graph tensors — pass all available tensors
            var subInputs = new Dictionary<string, Tensor>();
            OuterScope.Add(ctx, subgraph, subInputs);
            for (int i = 0; i < ctx.InputNames.Length; i++)
            {
                if (!string.IsNullOrEmpty(ctx.InputNames[i]) && i < ctx.Inputs.Length)
                    subInputs[ctx.InputNames[i]] = ctx.Inputs[i];
            }

            var result = SubgraphRunner.Execute(ctx, subgraph, subInputs);
            if (result != null)
            {
                // ⚠️ Was a foreach over the result DICTIONARY, assigning outputs by enumeration order. The
                // contract is the branch's DECLARED output order, which a dictionary does not preserve, so a
                // multi-output If could map results to the wrong slots. Shares the async path's copy, which
                // walks subgraph.Outputs and adopts the executed branch's shape.
                SubgraphOutputCopy.Apply(reg, ctx, subgraph, result);
                return;
            }
        }

        // Fallback: pass through first input
        if (ctx.Inputs.Length > 0 && ctx.Outputs.Length > 0)
        {
            int c = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount);
            if (c > 0) reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, c), ctx.Outputs[0].Data.SubView(0, c), c, 1f);
        }
    }

    /// <summary>Browser-safe async If: runs the selected branch through the async subgraph runner.</summary>
    /// <remarks>
    /// ⚠️ This used to read the condition from <c>ctx.TryGetInputValues</c> - "pre-read constants", as its
    /// old comment put it - which returns null for a condition computed at runtime. It then defaulted to
    /// FALSE and took the else branch every time, regardless of the actual condition. Because
    /// <c>GraphExecutor.RunAsync</c> calls THIS method and not <c>Execute</c>, fixing only the sync path
    /// changed nothing at all: the same logic lived twice.
    /// </remarks>
    public async Task ExecuteAsync(OnnxOpContext ctx)
    {
        var condVals = await OperatorInputReader.ReadAsync(reg, ctx, 0);
        if (condVals == null || condVals.Length == 0)
            throw new NotSupportedException(
                "If could not read its condition. Defaulting to a branch would silently pick the wrong one.");
        bool condition = condVals[0] != 0f;

        CountBranch(condition);
        if (BypassSubgraphForCaptureProbe) return;   // DIAGNOSTIC: leaves the output unwritten
        string branchKey = condition ? "then_branch" : "else_branch";
        if (ctx.Attributes.TryGetValue(branchKey, out var branchObj) && branchObj is Onnx.OnnxGraphProto subgraph)
        {
            // A branch that is one Constant is written directly - no SubgraphRunner, so nothing
            // allocates inside a capture window. See TryWriteConstantBranch.
            if (TryWriteConstantBranch(ctx, subgraph)) return;
            var subInputs = new Dictionary<string, Tensor>();
            OuterScope.Add(ctx, subgraph, subInputs);
            for (int i = 0; i < ctx.InputNames.Length; i++)
            {
                if (!string.IsNullOrEmpty(ctx.InputNames[i]) && i < ctx.Inputs.Length)
                    subInputs[ctx.InputNames[i]] = ctx.Inputs[i];
            }

            var result = await SubgraphRunner.ExecuteAsync(ctx, subgraph, subInputs);
            if (result != null)
            {
                SubgraphOutputCopy.Apply(reg, ctx, subgraph, result);
                return;
            }
        }

        if (ctx.Inputs.Length > 0 && ctx.Outputs.Length > 0)
        {
            int c = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount);
            if (c > 0) reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, c), ctx.Outputs[0].Data.SubView(0, c), c, 1f);
        }
    }
}

public class LoopOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Loop";

    /// <summary>
    /// Output shapes come from the BODY subgraph, never from the inputs.
    /// </summary>
    /// <remarks>
    /// ⚠️ This used to return <c>inputs[0]</c>, and <c>inputs[0]</c> of a Loop is the SCALAR max trip
    /// count - so every output was allocated with one element, exactly the bug just fixed in
    /// <c>If</c> (there it was the condition). Body outputs are
    /// <c>[condition, carried..., scan_outputs...]</c> while the node's outputs are
    /// <c>[carried..., scan_outputs...]</c>, so the carried shapes are the body's outputs offset by one.
    /// <para>
    /// A SCAN output gains a leading iteration dimension whose length is the trip count, which is not
    /// knowable from shapes alone (it is a runtime value, and the loop may also stop early on its
    /// condition). Rather than guess - guessing is what produced a scalar here before - a Loop with scan
    /// outputs throws and says so. Loop-carried state alone is the common case and is handled exactly.
    /// </para>
    /// </remarks>
    public int[][] InferOutputShapes(int[][] i, Dictionary<string, object> a)
    {
        if (!a.TryGetValue("body", out var bodyObj) || bodyObj is not Onnx.OnnxGraphProto body
            || !SubgraphShapes.TryDeclaredOutputShapes(body, out var bodyShapes) || bodyShapes.Length < 1)
            return new[] { i.Length > 2 ? i[2] : new[] { 1 } };

        // Node inputs: [max_trip_count, condition, carried...]
        int carried = Math.Max(0, i.Length - 2);
        int bodyScanOutputs = bodyShapes.Length - 1 - carried;
        if (bodyScanOutputs > 0)
            throw new NotSupportedException(
                $"Loop with {bodyScanOutputs} scan output(s) is not supported: the iteration dimension is a "
                + "runtime trip count, so its shape cannot be inferred. Loop-carried state is supported.");

        var result = new int[carried][];
        for (int k = 0; k < carried; k++) result[k] = bodyShapes[k + 1];   // skip the body's condition
        return result.Length > 0 ? result : new[] { new[] { 1 } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // Loop: Input[0]=max_trip_count, Input[1]=condition, Input[2+]=initial carried state
        // Attr: body (GraphProto) — body inputs: [iteration, condition, carried...], outputs: [condition, carried..., scan_outputs...]
        // Both of these are runtime scalars in a real model, so TryGetInputValues returned null for them
        // and the loop silently fell back to its 100-iteration safety limit.
        int maxTrips = 100; // safety limit
        var tripVals = OperatorInputReader.Read(reg, ctx, 0);
        if (tripVals != null && tripVals.Length > 0 && tripVals[0] > 0)
            maxTrips = Math.Min((int)tripVals[0], 10000);

        bool keepGoing = true;
        var condVals = OperatorInputReader.Read(reg, ctx, 1);
        if (condVals != null && condVals.Length > 0)
            keepGoing = condVals[0] != 0f;

        if (ctx.Attributes.TryGetValue("body", out var bodyObj) && bodyObj is Onnx.OnnxGraphProto bodyGraph)
        {
            // Initialize carried state from Input[2+]
            int numCarried = ctx.Inputs.Length - 2;
            var carriedState = new Tensor[numCarried];
            for (int i = 0; i < numCarried && i + 2 < ctx.Inputs.Length; i++)
                carriedState[i] = ctx.Inputs[i + 2];

            // Iterate
            for (int iter = 0; iter < maxTrips && keepGoing; iter++)
            {
                // Build body inputs: [iteration_num, condition, carried_state...]
                var subInputs = new Dictionary<string, Tensor>();

                // Body graph expects specific input names from its input list
                var bodyInputNames = bodyGraph.Inputs.Select(i => i.Name).ToList();

                // Input 0: iteration number (scalar)
                if (bodyInputNames.Count > 0)
                {
                    var iterTensor = ctx.Pool.Rent(new[] { 1 }, "_loop_iter");
                    iterTensor.Data.SubView(0, 1).CopyFromCPU(new[] { (float)iter });
                    subInputs[bodyInputNames[0]] = iterTensor;
                }

                // Input 1: condition (scalar bool)
                if (bodyInputNames.Count > 1)
                {
                    var condTensor = ctx.Pool.Rent(new[] { 1 }, "_loop_cond");
                    condTensor.Data.SubView(0, 1).CopyFromCPU(new[] { keepGoing ? 1f : 0f });
                    subInputs[bodyInputNames[1]] = condTensor;
                }

                // Input 2+: carried state
                for (int i = 0; i < numCarried && i + 2 < bodyInputNames.Count; i++)
                    subInputs[bodyInputNames[i + 2]] = carriedState[i];

                // Also pass outer scope tensors
                for (int i = 0; i < ctx.InputNames.Length; i++)
                {
                    if (!string.IsNullOrEmpty(ctx.InputNames[i]) && i < ctx.Inputs.Length && !subInputs.ContainsKey(ctx.InputNames[i]))
                        subInputs[ctx.InputNames[i]] = ctx.Inputs[i];
                }

                var result = SubgraphRunner.Execute(ctx, bodyGraph, subInputs);
                if (result == null) break;

                // Body outputs: [0]=condition, [1+]=carried state, [numCarried+1+]=scan outputs
                var bodyOutputNames = bodyGraph.Outputs.Select(o => o.Name).ToList();

                // Output 0: updated condition
                if (bodyOutputNames.Count > 0 && result.TryGetValue(bodyOutputNames[0], out var newCond))
                {
                    var cv = new float[1];
                    newCond.Data.SubView(0, 1).CopyToCPU(cv);
                    reg.Accelerator.Synchronize();
                    keepGoing = cv[0] != 0f;
                }

                // Output 1+: updated carried state
                for (int i = 0; i < numCarried && i + 1 < bodyOutputNames.Count; i++)
                {
                    if (result.TryGetValue(bodyOutputNames[i + 1], out var newState))
                        carriedState[i] = newState;
                }
            }

            // Copy final carried state to outputs
            for (int i = 0; i < numCarried && i < ctx.Outputs.Length; i++)
            {
                int c = Math.Min(carriedState[i].ElementCount, ctx.Outputs[i].ElementCount);
                if (c > 0) reg.ElementWise.Scale(carriedState[i].Data.SubView(0, c), ctx.Outputs[i].Data.SubView(0, c), c, 1f);
            }
            return;
        }

        // Fallback: pass through carried state
        if (ctx.Inputs.Length > 2 && ctx.Outputs.Length > 0)
        {
            int c = Math.Min(ctx.Inputs[2].ElementCount, ctx.Outputs[0].ElementCount);
            if (c > 0) reg.ElementWise.Scale(ctx.Inputs[2].Data.SubView(0, c), ctx.Outputs[0].Data.SubView(0, c), c, 1f);
        }
    }

    /// <summary>Browser-safe async Loop: runs the body via the async subgraph runner and reads the
    /// per-iteration loop condition back with the async <c>CopyToHostAsync</c> (the sync Execute
    /// uses <c>CopyToCPU</c>+<c>Synchronize</c>, which throws on WebGPU/WebGL/Wasm).</summary>
    public async Task ExecuteAsync(OnnxOpContext ctx)
    {
        int maxTrips = 100;
        // ⚠️ Same logic lives in Execute and here, and GraphExecutor.RunAsync calls THIS one. Reading the
        // trip count from compile-time constants returned null for a runtime scalar, so the loop silently
        // fell back to its 100-iteration safety limit - measured as carried_final 100x the correct value.
        var tripVals = await OperatorInputReader.ReadAsync(reg, ctx, 0);
        if (tripVals != null && tripVals.Length > 0 && tripVals[0] > 0)
            maxTrips = Math.Min((int)tripVals[0], 10000);

        bool keepGoing = true;
        var condVals = await OperatorInputReader.ReadAsync(reg, ctx, 1);
        if (condVals != null && condVals.Length > 0)
            keepGoing = condVals[0] != 0f;

        if (ctx.Attributes.TryGetValue("body", out var bodyObj) && bodyObj is Onnx.OnnxGraphProto bodyGraph)
        {
            int numCarried = ctx.Inputs.Length - 2;
            var carriedState = new Tensor[numCarried];
            for (int i = 0; i < numCarried && i + 2 < ctx.Inputs.Length; i++)
                carriedState[i] = ctx.Inputs[i + 2];

            for (int iter = 0; iter < maxTrips && keepGoing; iter++)
            {
                var subInputs = new Dictionary<string, Tensor>();
                var bodyInputNames = bodyGraph.Inputs.Select(i => i.Name).ToList();

                if (bodyInputNames.Count > 0)
                {
                    var iterTensor = ctx.Pool.Rent(new[] { 1 }, "_loop_iter");
                    iterTensor.Data.SubView(0, 1).CopyFromCPU(new[] { (float)iter });
                    subInputs[bodyInputNames[0]] = iterTensor;
                }

                if (bodyInputNames.Count > 1)
                {
                    var condTensor = ctx.Pool.Rent(new[] { 1 }, "_loop_cond");
                    condTensor.Data.SubView(0, 1).CopyFromCPU(new[] { keepGoing ? 1f : 0f });
                    subInputs[bodyInputNames[1]] = condTensor;
                }

                for (int i = 0; i < numCarried && i + 2 < bodyInputNames.Count; i++)
                    subInputs[bodyInputNames[i + 2]] = carriedState[i];

                for (int i = 0; i < ctx.InputNames.Length; i++)
                {
                    if (!string.IsNullOrEmpty(ctx.InputNames[i]) && i < ctx.Inputs.Length && !subInputs.ContainsKey(ctx.InputNames[i]))
                        subInputs[ctx.InputNames[i]] = ctx.Inputs[i];
                }

                var result = await SubgraphRunner.ExecuteAsync(ctx, bodyGraph, subInputs);
                if (result == null) break;

                var bodyOutputNames = bodyGraph.Outputs.Select(o => o.Name).ToList();

                // Output 0: updated condition — async GPU->CPU readback (browser-safe).
                if (bodyOutputNames.Count > 0 && result.TryGetValue(bodyOutputNames[0], out var newCond))
                {
                    using var condBuf = reg.Accelerator.Allocate1D<float>(1);
                    condBuf.View.SubView(0, 1).CopyFrom(newCond.Data.SubView(0, 1));
                    var cv = await condBuf.CopyToHostAsync<float>(0, 1);
                    keepGoing = cv[0] != 0f;
                }

                for (int i = 0; i < numCarried && i + 1 < bodyOutputNames.Count; i++)
                {
                    if (result.TryGetValue(bodyOutputNames[i + 1], out var newState))
                        carriedState[i] = newState;
                }
            }

            for (int i = 0; i < numCarried && i < ctx.Outputs.Length; i++)
            {
                int c = Math.Min(carriedState[i].ElementCount, ctx.Outputs[i].ElementCount);
                if (c > 0) reg.ElementWise.Scale(carriedState[i].Data.SubView(0, c), ctx.Outputs[i].Data.SubView(0, c), c, 1f);
            }
            return;
        }

        if (ctx.Inputs.Length > 2 && ctx.Outputs.Length > 0)
        {
            int c = Math.Min(ctx.Inputs[2].ElementCount, ctx.Outputs[0].ElementCount);
            if (c > 0) reg.ElementWise.Scale(ctx.Inputs[2].Data.SubView(0, c), ctx.Outputs[0].Data.SubView(0, c), c, 1f);
        }
    }
}

public class ScanOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Scan";

    /// <summary>
    /// Output shapes come from the BODY subgraph, with the sequence length taken from the scanned input.
    /// </summary>
    /// <remarks>
    /// ⚠️ This used to return <c>inputs[0]</c> - the first STATE input - so every output was sized like it,
    /// the same class of bug as <c>If</c> and <c>Loop</c>.
    /// <para>
    /// Unlike Loop, Scan is fully inferable: its scan outputs gain a leading dimension equal to the
    /// SEQUENCE LENGTH, and that is the leading dimension of the scanned input, which is a shape and
    /// therefore known at compile time. Node inputs are <c>[state..., scan_input...]</c>, body outputs are
    /// <c>[state..., scan_output...]</c> (no condition, unlike Loop).
    /// </para>
    /// </remarks>
    public int[][] InferOutputShapes(int[][] i, Dictionary<string, object> a)
    {
        if (!a.TryGetValue("body", out var bodyObj) || bodyObj is not Onnx.OnnxGraphProto body
            || !SubgraphShapes.TryDeclaredOutputShapes(body, out var bodyShapes) || bodyShapes.Length == 0)
            return new[] { i.Length > 0 ? i[0] : new[] { 1 } };

        int numScanInputs = a.TryGetValue("num_scan_inputs", out var nsi) ? Convert.ToInt32(nsi) : 1;
        int numState = Math.Max(0, i.Length - numScanInputs);
        int numScanOutputs = Math.Max(0, bodyShapes.Length - numState);

        // The scanned input's leading dimension IS the sequence length.
        int seqLen = 1;
        if (numState < i.Length && i[numState] is { Length: > 0 } firstScanned) seqLen = firstScanned[0];

        var result = new int[bodyShapes.Length][];
        for (int k = 0; k < numState && k < bodyShapes.Length; k++)
            result[k] = bodyShapes[k];                            // final state: same shape as body state
        for (int j = 0; j < numScanOutputs; j++)
        {
            var perStep = bodyShapes[numState + j];
            var stacked = new int[perStep.Length + 1];
            stacked[0] = seqLen;                                  // stacked over the sequence
            Array.Copy(perStep, 0, stacked, 1, perStep.Length);
            result[numState + j] = stacked;
        }
        return result;
    }
    public void Execute(OnnxOpContext ctx)
    {
        // Scan: sequential scan over input sequence, applying body subgraph at each step.
        // Inputs: [state_0..state_N, scan_input_0..scan_input_M]
        // Outputs: [final_state_0..state_N, scan_output_0..scan_output_M]
        // Attr: body (GraphProto), num_scan_inputs (int)
        int numScanInputs = ctx.GetInt("num_scan_inputs", 1);

        if (ctx.Attributes.TryGetValue("body", out var bodyObj) && bodyObj is Onnx.OnnxGraphProto bodyGraph)
        {
            int numStateInputs = ctx.Inputs.Length - numScanInputs;
            if (numStateInputs < 0) numStateInputs = 0;

            // Initialize state from initial state inputs
            var state = new Tensor[numStateInputs];
            for (int i = 0; i < numStateInputs; i++)
                state[i] = ctx.Inputs[i];

            // Determine sequence length from first scan input
            int seqLen = 1;
            if (numScanInputs > 0 && numStateInputs < ctx.Inputs.Length)
            {
                var scanInput = ctx.Inputs[numStateInputs];
                seqLen = scanInput.Shape[0]; // scan along first dimension
            }

            var bodyInputNames = bodyGraph.Inputs.Select(i => i.Name).ToList();
            var bodyOutputNames = bodyGraph.Outputs.Select(o => o.Name).ToList();

            // Process each sequence element
            for (int step = 0; step < seqLen; step++)
            {
                var subInputs = new Dictionary<string, Tensor>();

                // State inputs
                for (int i = 0; i < numStateInputs && i < bodyInputNames.Count; i++)
                    subInputs[bodyInputNames[i]] = state[i];

                // Scan inputs: slice along sequence dimension
                for (int si = 0; si < numScanInputs; si++)
                {
                    int inputIdx = numStateInputs + si;
                    int bodyIdx = numStateInputs + si;
                    if (inputIdx < ctx.Inputs.Length && bodyIdx < bodyInputNames.Count)
                    {
                        var fullInput = ctx.Inputs[inputIdx];
                        int sliceSize = fullInput.ElementCount / seqLen;
                        var slice = ctx.Pool.Rent(fullInput.Shape[1..], "_scan_slice");
                        reg.ElementWise.Scale(fullInput.Data.SubView(step * sliceSize, sliceSize), slice.Data.SubView(0, sliceSize), sliceSize, 1f);
                        subInputs[bodyInputNames[bodyIdx]] = slice;
                    }
                }

                // Also pass outer scope tensors
                for (int i = 0; i < ctx.InputNames.Length; i++)
                {
                    if (!string.IsNullOrEmpty(ctx.InputNames[i]) && i < ctx.Inputs.Length && !subInputs.ContainsKey(ctx.InputNames[i]))
                        subInputs[ctx.InputNames[i]] = ctx.Inputs[i];
                }

                var result = SubgraphRunner.Execute(ctx, bodyGraph, subInputs);
                if (result == null) break;

                // Body outputs: [state_0..state_N, scan_output_0..scan_output_M]
                for (int i = 0; i < numStateInputs && i < bodyOutputNames.Count; i++)
                {
                    if (result.TryGetValue(bodyOutputNames[i], out var newState))
                        state[i] = newState;
                }

                // ⚠️ SCAN OUTPUTS were never written - the loop tracked state and ignored everything after
                // it, so `stacked` came back all zeros while the final state was correct. Each step's body
                // output is one slice of the stacked result, written at this step's offset.
                for (int j = 0; numStateInputs + j < bodyOutputNames.Count; j++)
                {
                    int outIdx = numStateInputs + j;
                    if (outIdx >= ctx.Outputs.Length) break;
                    if (!result.TryGetValue(bodyOutputNames[outIdx], out var stepOut)) continue;
                    int per = stepOut.ElementCount;
                    int offset = step * per;
                    if (per > 0 && offset + per <= ctx.Outputs[outIdx].ElementCount)
                        reg.ElementWise.Scale(stepOut.Data.SubView(0, per),
                            ctx.Outputs[outIdx].Data.SubView(offset, per), per, 1f);
                }
            }

            // Copy final state to state outputs
            for (int i = 0; i < numStateInputs && i < ctx.Outputs.Length; i++)
            {
                int c = Math.Min(state[i].ElementCount, ctx.Outputs[i].ElementCount);
                if (c > 0) reg.ElementWise.Scale(state[i].Data.SubView(0, c), ctx.Outputs[i].Data.SubView(0, c), c, 1f);
            }
            return;
        }

        // Fallback: pass through input
        if (ctx.Inputs.Length > 0 && ctx.Outputs.Length > 0)
        {
            int c = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount);
            if (c > 0) reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, c), ctx.Outputs[0].Data.SubView(0, c), c, 1f);
        }
    }

    /// <summary>Browser-safe async Scan: identical sequential scan but runs the body subgraph via
    /// the async subgraph runner (so any GPU-&gt;CPU readback inside the body uses the async path).</summary>
    public async Task ExecuteAsync(OnnxOpContext ctx)
    {
        int numScanInputs = ctx.GetInt("num_scan_inputs", 1);

        if (ctx.Attributes.TryGetValue("body", out var bodyObj) && bodyObj is Onnx.OnnxGraphProto bodyGraph)
        {
            int numStateInputs = ctx.Inputs.Length - numScanInputs;
            if (numStateInputs < 0) numStateInputs = 0;

            var state = new Tensor[numStateInputs];
            for (int i = 0; i < numStateInputs; i++)
                state[i] = ctx.Inputs[i];

            int seqLen = 1;
            if (numScanInputs > 0 && numStateInputs < ctx.Inputs.Length)
            {
                var scanInput = ctx.Inputs[numStateInputs];
                seqLen = scanInput.Shape[0];
            }

            var bodyInputNames = bodyGraph.Inputs.Select(i => i.Name).ToList();
            var bodyOutputNames = bodyGraph.Outputs.Select(o => o.Name).ToList();

            for (int step = 0; step < seqLen; step++)
            {
                var subInputs = new Dictionary<string, Tensor>();

                for (int i = 0; i < numStateInputs && i < bodyInputNames.Count; i++)
                    subInputs[bodyInputNames[i]] = state[i];

                for (int si = 0; si < numScanInputs; si++)
                {
                    int inputIdx = numStateInputs + si;
                    int bodyIdx = numStateInputs + si;
                    if (inputIdx < ctx.Inputs.Length && bodyIdx < bodyInputNames.Count)
                    {
                        var fullInput = ctx.Inputs[inputIdx];
                        int sliceSize = fullInput.ElementCount / seqLen;
                        var slice = ctx.Pool.Rent(fullInput.Shape[1..], "_scan_slice");
                        reg.ElementWise.Scale(fullInput.Data.SubView(step * sliceSize, sliceSize), slice.Data.SubView(0, sliceSize), sliceSize, 1f);
                        subInputs[bodyInputNames[bodyIdx]] = slice;
                    }
                }

                for (int i = 0; i < ctx.InputNames.Length; i++)
                {
                    if (!string.IsNullOrEmpty(ctx.InputNames[i]) && i < ctx.Inputs.Length && !subInputs.ContainsKey(ctx.InputNames[i]))
                        subInputs[ctx.InputNames[i]] = ctx.Inputs[i];
                }

                var result = await SubgraphRunner.ExecuteAsync(ctx, bodyGraph, subInputs);
                if (result == null) break;

                for (int i = 0; i < numStateInputs && i < bodyOutputNames.Count; i++)
                {
                    if (result.TryGetValue(bodyOutputNames[i], out var newState))
                        state[i] = newState;
                }

                // ⚠️ SCAN OUTPUTS were never written - the loop tracked state and ignored everything after
                // it, so `stacked` came back all zeros while the final state was correct. Each step's body
                // output is one slice of the stacked result, written at this step's offset.
                for (int j = 0; numStateInputs + j < bodyOutputNames.Count; j++)
                {
                    int outIdx = numStateInputs + j;
                    if (outIdx >= ctx.Outputs.Length) break;
                    if (!result.TryGetValue(bodyOutputNames[outIdx], out var stepOut)) continue;
                    int per = stepOut.ElementCount;
                    int offset = step * per;
                    if (per > 0 && offset + per <= ctx.Outputs[outIdx].ElementCount)
                        reg.ElementWise.Scale(stepOut.Data.SubView(0, per),
                            ctx.Outputs[outIdx].Data.SubView(offset, per), per, 1f);
                }
            }

            for (int i = 0; i < numStateInputs && i < ctx.Outputs.Length; i++)
            {
                int c = Math.Min(state[i].ElementCount, ctx.Outputs[i].ElementCount);
                if (c > 0) reg.ElementWise.Scale(state[i].Data.SubView(0, c), ctx.Outputs[i].Data.SubView(0, c), c, 1f);
            }
            return;
        }

        if (ctx.Inputs.Length > 0 && ctx.Outputs.Length > 0)
        {
            int c = Math.Min(ctx.Inputs[0].ElementCount, ctx.Outputs[0].ElementCount);
            if (c > 0) reg.ElementWise.Scale(ctx.Inputs[0].Data.SubView(0, c), ctx.Outputs[0].Data.SubView(0, c), c, 1f);
        }
    }
}
// RNN, LSTM, GRU moved to RecurrentOperators.cs with full implementations

/// <summary>
/// Shapes declared by a control-flow subgraph. Used by <c>If</c>, <c>Loop</c> and <c>Scan</c>.
/// </summary>
/// <remarks>
/// ⚠️ All three of those operators previously inferred their output shapes from <c>inputs[0]</c>, which is
/// the condition for If, the trip count for Loop and the first state for Scan - never the output. Every one
/// therefore allocated a buffer of the wrong size and the branch/body result was silently truncated into
/// it. The subgraph declares what it produces; that is the only correct source.
/// </remarks>
internal static class SubgraphShapes
{
    /// <summary>Declared output shapes of a subgraph, or false if any dimension is symbolic.</summary>
    public static bool TryDeclaredOutputShapes(Onnx.OnnxGraphProto sub, out int[][] shapes)
    {
        shapes = Array.Empty<int[]>();
        if (sub.Outputs.Count == 0) return false;

        var result = new int[sub.Outputs.Count][];
        for (int o = 0; o < sub.Outputs.Count; o++)
        {
            var dims = sub.Outputs[o].Shape;
            if (dims.Count == 0) return false;
            var shape = new int[dims.Count];
            for (int d = 0; d < dims.Count; d++)
            {
                // A symbolic dim cannot size a buffer, and guessing 1 is how a scalar got here before.
                if (dims[d].DimValue is not { } v || v <= 0) return false;
                shape[d] = (int)v;
            }
            result[o] = shape;
        }
        shapes = result;
        return true;
    }
}

/// <summary>
/// Copies a subgraph's results into an operator's outputs, in the subgraph's DECLARED output order.
/// </summary>
/// <remarks>
/// ⚠️ The control-flow operators used to walk the result DICTIONARY and assign to outputs by position.
/// Dictionary order is not the graph's output order, and it stopped being even incidentally right once
/// declared-but-unproduced outputs (a branch that is a single folded Constant) were merged in. For a
/// single-output If it was harmless; for Loop and Scan it silently permutes results.
/// </remarks>
internal static class SubgraphOutputCopy
{
    public static void Apply(OperatorRegistry reg, OnnxOpContext ctx,
        Onnx.OnnxGraphProto subgraph, Dictionary<string, Tensor> result, int skipLeading = 0)
    {
        int outIdx = 0;
        for (int d = skipLeading; d < subgraph.Outputs.Count && outIdx < ctx.Outputs.Length; d++)
        {
            var name = subgraph.Outputs[d].Name;
            if (string.IsNullOrEmpty(name) || !result.TryGetValue(name, out var tensor)) { outIdx++; continue; }
            CopyOrAdopt(reg, ctx, outIdx, tensor, name);
            outIdx++;
        }
    }

    /// <summary>
    /// Write one branch/body output into <c>ctx.Outputs[outIdx]</c>, ADOPTING the branch's shape when it
    /// differs from the buffer that was preallocated at compile time.
    /// </summary>
    /// <remarks>
    /// ⚠️ This used to be <c>Math.Min(tensor.ElementCount, ctx.Outputs[outIdx].ElementCount)</c>, which
    /// silently TRUNCATES a branch that produced more than the compile-time shape - and leaves the stale
    /// compile-time shape visible to every downstream consumer, so nothing reports a problem.
    ///
    /// An If cannot be sized statically in general: ONNX requires the branches to agree on rank and dtype,
    /// NOT on dims. <c>InferOutputShapes</c> therefore has to pick one branch's declaration (the only
    /// statically usable one), and whenever the OTHER branch runs and its dims differ, the buffer is the
    /// wrong size. Only the executed branch knows the answer, so the shape has to be adopted here.
    ///
    /// MEASURED on ZipVoice's fm_decoder relative positional encoding. then_branch is a constant
    /// <c>[1999,48]</c> table; else_branch RECOMPUTES it as <c>[2*T-1, 48]</c> and runs for any utterance
    /// past T=1000. At T=1197 the branch produced <c>[2393,48]</c> = 114,864 values into a 95,952-element
    /// buffer: 18,912 dropped, the table reported as <c>[1999,48]</c>, and the decoder diverged from
    /// onnxruntime by 104% of peak (max |d| 5.675) while every shape downstream still looked plausible.
    /// Long replies were audibly wrong rather than broken, which is the worst way for this to fail.
    ///
    /// Rented under a stable per-output name, so this is one buffer reused per If output rather than an
    /// allocation per call - a per-call device allocation is what makes a graph uncapturable.
    /// </remarks>
    private static void CopyOrAdopt(OperatorRegistry reg, OnnxOpContext ctx, int outIdx,
        Tensor tensor, string name)
    {
        var dst = ctx.Outputs[outIdx];
        if (tensor.ElementCount <= 0) return;

        if (dst == null || !dst.Shape.AsSpan().SequenceEqual(tensor.Shape))
        {
            var adopted = ctx.Pool.Rent(tensor.Shape, "_branchout_" + name);
            reg.ElementWise.Scale(tensor.Data.SubView(0, tensor.ElementCount),
                                  adopted.Data.SubView(0, tensor.ElementCount), tensor.ElementCount, 1f);
            ctx.Outputs[outIdx] = adopted;   // aliases the executor's nodeOutputs - see OnnxOpContext.Outputs
            return;
        }

        reg.ElementWise.Scale(tensor.Data.SubView(0, tensor.ElementCount),
                              dst.Data.SubView(0, tensor.ElementCount), tensor.ElementCount, 1f);
    }
}
