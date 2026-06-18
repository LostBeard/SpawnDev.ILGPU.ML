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

        // FAIL LOUDLY on an undersized output view: every path below writes one float per
        // (input row x N) - (a.ElementCount/K) rows in total (= M*N rank-2, batch*M*N batched).
        // M/N come from the RUNTIME input shapes, but the output buffer was allocated from the
        // COMPILE-TIME shape; if a compiler bug pins the output smaller (e.g. the declared-
        // output -1 -> 1 override, 2026-06-12) the kernel silently corrupts pool memory past
        // the buffer - or traps, backend-dependent. Name the cause instead.
        long requiredOut = (long)(a.ElementCount / K) * N;
        if (ctx.Outputs[0].Data.Length < requiredOut)
            throw new InvalidOperationException(
                $"MatMul: output '{ctx.Outputs[0].Name ?? "?"}' view holds {ctx.Outputs[0].Data.Length} " +
                $"elements but the kernel writes {requiredOut} (a=[{string.Join(",", a.Shape)}], " +
                $"b=[{string.Join(",", b.Shape)}], M={M},K={K},N={N}). The output buffer was sized " +
                "from a stale/wrong compile-time shape - shape inference and runtime disagree.");

        // Quantized weight B — use the fused dequant kernel for its GGML type. The type
        // is MANDATORY: every GGML layout decodes differently, and decoding one as
        // another (the old hardcoded-Q4_0 behavior) produces silent garbage for every
        // K-quant model. The GGUF loader sets reg.QuantizedWeightTypes alongside the
        // byte views and only admits FusedDequantMatMul.Supports types.
        string? bName = ctx.InputNames.Length > 1 ? ctx.InputNames[1] : null;
        if (bName != null && ctx.QuantizedWeights != null
            && ctx.QuantizedWeights.TryGetValue(bName, out var qData))
        {
            if (reg.QuantizedWeightTypes == null
                || !reg.QuantizedWeightTypes.TryGetValue(bName, out var qType))
                throw new InvalidOperationException(
                    $"MatMul: quantized weight '{bName}' has no GGML type registered " +
                    "(OperatorRegistry.QuantizedWeightTypes). Refusing to guess a block " +
                    "layout - the loader must record the type with the bytes.");
            // DIAGNOSTIC (env GGUF_MM_CHECK=1): catch an undersized activation/output buffer (a recompile
            // that didn't resize) vs an in-kernel index bug — the M*K input / M*N output must fit.
            // Checks the PHYSICAL view lengths (Data.Length), not just shape-derived ElementCount:
            // Tensor.Shape is settable post-construction, so ElementCount can claim more elements
            // than the view actually backs — the exact blind spot that hid the 2026-06-12 recompile
            // fault (ElementCount said 20480, and the real problem was elsewhere entirely).
            if (System.Environment.GetEnvironmentVariable("GGUF_MM_CHECK") == "1"
                && (a.Data.Length < (long)M * K || ctx.Outputs[0].Data.Length < (long)M * N))
                Console.WriteLine($"[MatMul-Q] BUFFER MISMATCH '{bName}': a.Data={a.Data.Length} (need M*K={(long)M * K}), " +
                    $"out.Data={ctx.Outputs[0].Data.Length} (need M*N={(long)M * N}), M={M},K={K},N={N}, " +
                    $"aShape=[{string.Join(",", a.Shape)}], outShape=[{string.Join(",", ctx.Outputs[0].Shape)}]");
            reg.FusedDequant.Forward(a.Data, qData, ctx.Outputs[0].Data, M, K, N, qType);
            return;
        }

        // FAIL LOUDLY before the F32 kernels: a ShapeOnly B (quantized weight whose floats never
        // exist) reaching this point means the quantized byte-view map was not wired into this
        // executor (e.g. a shape-recompiled executor constructed without quantizedWeights — the
        // gemma4 seq>1 CUDA illegal-access, 2026-06-12). Dispatching the empty view is a GPU
        // null-pointer read; this exception names the actual cause instead.
        if (!LowPWeightDispatch.IsLowP(b) && b.Data.Length < (long)K * N)
            throw new InvalidOperationException(
                $"MatMul: B '{bName ?? b.Name ?? "?"}' has no usable F32 data (view length {b.Data.Length}, " +
                $"needs K*N={(long)K * N}). " +
                (ctx.QuantizedWeights == null
                    ? "ctx.QuantizedWeights is NULL — this executor was constructed without the session's " +
                      "quantized byte-view map (shape-recompiled executor missing quantizedWeights?)."
                    : $"ctx.QuantizedWeights has {ctx.QuantizedWeights.Count} entries but not '{bName}'."));

        // Native low-p weights: if B is a low-p-backed weight (Half/bf16/FP8, no float buffer), route to the
        // generic low-p-weight kernel (reads the native type, converts in-register, fp32 accumulate, no f32
        // temp). Activations (A) stay fp32. fp32 weights take the all-fp32 path.
        if (a.Rank == 2 && b.Rank == 2)
        {
            if (LowPWeightDispatch.IsLowP(b)) LowPWeightDispatch.MatMul(reg.MatMul, a.Data, b, ctx.Outputs[0].Data, M, K, N);
            else reg.MatMul.MatMul(a.Data, b.Data, ctx.Outputs[0].Data, M, K, N);
        }
        else
        {
            int batch = a.ElementCount / (M * K);
            if (LowPWeightDispatch.IsLowP(b))
                LowPWeightDispatch.BatchedMatMul(reg.MatMul, a.Data, b, ctx.Outputs[0].Data, batch, M, K, N);
            else
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
            ctx.Pool.Return(transposed);
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

// ── RMSNormalization ──
//
// True RMSNorm (NO mean-centering): output = x / sqrt(mean(x^2) + eps) * weight.
// This is what every RMS decoder (llama/mistral/qwen/gemma via GGUF) needs. The 2-input
// "LayerNormalization" the GGUF builder used to emit routed to the MEAN-CENTERED LayerNorm
// kernel (wrong math) AND crashed reading the absent bias Inputs[2]; structural-only tests
// never executed it so it went unnoticed. The kernel (NormalizationKernels.RMSNorm) already
// existed unwrapped — this operator wires it.
//
//   inputs: x [..., C]    (required)
//           weight [C]    (optional — absent = WEIGHTLESS unit-gain, e.g. gemma4's V-norm)
//   attrs:  axis:i (default -1), epsilon:f (default 1e-6 — the RMS-decoder convention; gemma = 1e-6)
//   out:    same shape as x
public class RMSNormOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "RMSNormalization";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        int axis = ctx.GetInt("axis", -1);
        float eps = ctx.GetFloat("epsilon", 1e-6f);
        var shape = ctx.Inputs[0].Shape;
        if (axis < 0) axis += shape.Length;
        int rows = 1; for (int i = 0; i < axis; i++) rows *= shape[i];
        int C = 1; for (int i = axis; i < shape.Length; i++) C *= shape[i];

        bool hasWeight = ctx.Inputs.Length > 1 && ctx.Inputs[1] != null;
        if (hasWeight)
            reg.Normalization.RMSNorm(ctx.Inputs[0].Data, ctx.Outputs[0].Data,
                ctx.Inputs[1].Data, rows, C, eps);
        else
            reg.Normalization.RMSNorm(ctx.Inputs[0].Data, ctx.Outputs[0].Data, rows, C, eps);
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
        var (N, C, _, _) = shape.Length >= 4 ? LayoutHelper.GetDims(shape, ctx.Format)
            : (shape[0], shape.Length > 1 ? shape[1] : 1, 1, 1);
        int spatial = ctx.Inputs[0].ElementCount / (N * C);
        float epsilon = ctx.GetFloat("epsilon", 1e-5f); // TF/Keras BN exports 1e-3; ONNX default 1e-5
        reg.Normalization.BatchNorm(ctx.Inputs[0].Data, ctx.Outputs[0].Data,
            ctx.Inputs[1].Data, ctx.Inputs[2].Data,
            ctx.Inputs[3].Data, ctx.Inputs[4].Data, N, C, spatial, epsilon);
    }
}

// ── Conv ──

public class ConvOperator(OperatorRegistry reg) : IOnnxOperator, IPrecisionAwareOperator
{
    public string OpType => "Conv";

    /// <summary>Resolve the 2D conv spatial params (stride, asymmetric pads [top,left,bottom,right], dilations)
    /// from ctx attributes + input/weight shapes. Shared by <see cref="Execute"/> and the precision-aware path so
    /// the SAME_UPPER/SAME_LOWER and asymmetric-pad logic has a single source of truth.</summary>
    internal static (int stride, int padTop, int padLeft, int padBottom, int padRight, int dilationH, int dilationW)
        ResolveConv2DSpatialParams(OnnxOpContext ctx, int[] xShape, int[] wShape, DataFormat fmt)
    {
        var strides = ctx.GetInts("strides"); int stride = strides.Length > 0 ? strides[0] : 1;
        // ONNX `pads` = [x1_begin, x2_begin, x1_end, x2_end] = [top, left, bottom, right] for 2D.
        // Stride-2 SAME convs export asymmetric pads like [0,0,1,1] — keep all four (collapsing shears the grid).
        var autoPad = ctx.Attributes.TryGetValue("auto_pad", out var ap) ? ap.ToString()! : "NOTSET";
        int padTop, padLeft, padBottom, padRight;
        if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER")
        {
            int inH = xShape.Length >= 4 ? xShape[LayoutHelper.HeightAxis(fmt)] : (xShape.Length >= 3 ? xShape[2] : 1);
            int inW = xShape.Length >= 4 ? xShape[LayoutHelper.WidthAxis(fmt)] : 1;
            int kHa = wShape.Length >= 4 ? wShape[LayoutHelper.HeightAxis(fmt)] : (wShape.Length >= 3 ? wShape[2] : 1);
            int kWa = wShape.Length >= 4 ? wShape[LayoutHelper.WidthAxis(fmt)] : (wShape.Length >= 3 ? (wShape.Length > 3 ? wShape[3] : 1) : 1);
            int strideW = strides.Length > 1 ? strides[1] : stride;
            int padH = Math.Max(0, ((int)Math.Ceiling((double)inH / stride) - 1) * stride + kHa - inH);
            int padW = Math.Max(0, ((int)Math.Ceiling((double)inW / strideW) - 1) * strideW + kWa - inW);
            if (autoPad == "SAME_UPPER") { padTop = padH / 2; padBottom = padH - padH / 2; padLeft = padW / 2; padRight = padW - padW / 2; }
            else { padTop = padH - padH / 2; padBottom = padH / 2; padLeft = padW - padW / 2; padRight = padW / 2; }
        }
        else
        {
            var pads = ctx.GetInts("pads");
            padTop    = pads.Length > 0 ? pads[0] : 0;
            padLeft   = pads.Length > 1 ? pads[1] : 0;
            padBottom = pads.Length > 2 ? pads[2] : padTop;
            padRight  = pads.Length > 3 ? pads[3] : padLeft;
        }
        var dilationsAttr = ctx.GetInts("dilations");
        int dilationH = dilationsAttr.Length > 0 ? dilationsAttr[0] : 1;
        int dilationW = dilationsAttr.Length > 1 ? dilationsAttr[1] : dilationH;
        return (stride, padTop, padLeft, padBottom, padRight, dilationH, dilationW);
    }

    /// <summary>Precision-aware (F16) path: standard group-1 NCHW 2D Conv with a low-p activation input and an
    /// fp32 weight/bias → low-p output, double-accumulate, NO fp32 temp. Returns false (→ fp32 fallback) for any
    /// other case (Conv1D, depthwise, grouped, NHWC, fp16 weight, fp32 input).</summary>
    public bool TryExecuteHalf(OnnxOpContext ctx, PrecisionAwareInput[] inputs, HalfTensor output, Kernels.PrecisionAwareKernels pak)
    {
        if (ctx.Format != DataFormat.NCHW) return false;
        if (inputs.Length < 2) return false;
        var xIn = inputs[0]; var wIn = inputs[1];
        // Activation input must be low-p; weight must be a Tensor (fp32 .Data OR fp16 .HalfData — both handled).
        if (!xIn.IsHalf || wIn.Float == null) return false;
        // This precision-aware path only handles an fp32 OR fp16 weight (pak.Conv2DHalfWeight is Half-typed).
        // A non-fp16 low-p weight (bf16/FP8) falls back to the main ConvOperator.Execute path, which routes it
        // natively via LowPWeightDispatch — so it stays native, just not on this F16-activation fast path.
        if (LowPWeightDispatch.IsLowP(wIn.Float) && !wIn.Float.IsHalf) return false;
        var xShape = xIn.Half!.Shape; var wShape = wIn.Float.Shape;
        if (xShape.Length != 4 || wShape.Length != 4) return false;

        int group = ctx.GetInt("group", 1);
        var (_, inC, inH, inW) = LayoutHelper.GetDims(xShape, ctx.Format);
        if (group == -1) group = inC;
        int outC = wShape[0];
        if (group != 1) return false;               // only standard (non-grouped, non-depthwise) here
        var (_, _, kH, kW) = LayoutHelper.GetWeightDims(wShape, ctx.Format);
        var (stride, padTop, padLeft, padBottom, padRight, dilationH, dilationW) =
            ResolveConv2DSpatialParams(ctx, xShape, wShape, ctx.Format);

        // Bias: fp32 input[2] if present (and fp32 — fall back if it's a fp16-stored bias, which is rare/tiny),
        // else the shared zero-bias buffer.
        ArrayView1D<float, Stride1D.Dense> bias;
        if (inputs.Length > 2 && inputs[2].Float != null)
        {
            if (LowPWeightDispatch.IsLowP(inputs[2].Float!)) return false;   // any low-p bias has no fp32 Data — fall back to fp32 conv
            bias = inputs[2].Float!.Data;
        }
        else bias = reg.GetOrCreateZeroBias(outC);

        // The VAE loads most conv weights as fp16: a Tensor with IsHalf=true has an EMPTY .Data (fp32) and a
        // populated .HalfData. Route to the half-weight kernel; only genuinely fp32 weights use Conv2D<T>.
        if (wIn.Float.IsHalf)
            pak.Conv2DHalfWeight<global::ILGPU.Half>(xIn.Half!.Data, wIn.Float.HalfData, bias, output.Data,
                inC, inH, inW, outC, kH, kW, stride, padTop, padLeft, padBottom, padRight, dilationH, dilationW);
        else
            pak.Conv2D<global::ILGPU.Half>(xIn.Half!.Data, wIn.Float.Data, bias, output.Data,
                inC, inH, inW, outC, kH, kW, stride, padTop, padLeft, padBottom, padRight, dilationH, dilationW);
        return true;
    }
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var x = inputs[0]; var w = inputs[1];
        var fmt = attrs.ContainsKey("_data_format") && attrs["_data_format"].ToString() == "NHWC"
            ? DataFormat.NHWC : DataFormat.NCHW;
        var strides = attrs.ContainsKey("strides") ? ((long[])attrs["strides"]).Select(s => (int)s).ToArray() : new[] { 1, 1 };
        var (_, xC_dim, xH_dim, xW_dim) = x.Length >= 4 ? LayoutHelper.GetDims(x, fmt) : (1, 1, 1, 1);
        var (wOutC, _, wKH, wKW) = w.Length >= 4 ? LayoutHelper.GetWeightDims(w, fmt) : (w[0], 1, w.Length > 2 ? w[2] : 1, w.Length > 3 ? w[3] : 1);
        int outC = wOutC;
        // TFLite-style depthwise (NHWC weight [1, kH, kW, inC]) lands here as wOutC=1.
        // Execute() detects depthwise via group==inC AND (outC==1 || group==outC) and
        // dispatches inC*outH*outW threads writing to the output buffer. Without this
        // override InferOutputShapes returns [N, outH, outW, 1] = inC-times-too-small
        // and the dispatch OOBs (Wasm trap; CUDA/CPU silent overrun). Fix: when group
        // attribute matches input channel count, treat as depthwise and report the
        // actual output channel count (= inC) so the buffer pool allocates the right
        // size.
        int groupAttr = attrs.ContainsKey("group")
            ? (int)((long)attrs["group"])
            : 1;
        // Resolve TFLite depthwise sentinel: group=-1 means group=inC (per ConvOperator.Execute).
        if (groupAttr == -1) groupAttr = xC_dim;
        if (groupAttr > 1 && groupAttr == xC_dim && (wOutC == 1 || wOutC == groupAttr))
            outC = xC_dim;

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
                // Use layout-aware dims — x[2]/x[3] are H/W only in NCHW, not NHWC
                var sameDilations = attrs.ContainsKey("dilations") ? ((long[])attrs["dilations"]).Select(d => (int)d).ToArray() : new[] { 1, 1 };
                int sdH = sameDilations.Length > 0 ? sameDilations[0] : 1;
                int sdW = sameDilations.Length > 1 ? sameDilations[1] : sdH;
                int sameEffKH = sdH * (wKH - 1) + 1;
                int sameEffKW = sdW * (wKW - 1) + 1;
                int sH = strides[0], sW = strides.Length > 1 ? strides[1] : 1;
                int targetOutH = (int)Math.Ceiling((double)xH_dim / sH);
                int targetOutW = (int)Math.Ceiling((double)xW_dim / sW);
                int padH = Math.Max(0, (targetOutH - 1) * sH + sameEffKH - xH_dim);
                int padW = Math.Max(0, (targetOutW - 1) * sW + sameEffKW - xW_dim);
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
            var (_, _, xH, xW) = LayoutHelper.GetDims(x, fmt);
            var dilations2d = attrs.ContainsKey("dilations") ? ((long[])attrs["dilations"]).Select(d => (int)d).ToArray() : new[] { 1, 1 };
            int dH = dilations2d.Length > 0 ? dilations2d[0] : 1;
            int dW = dilations2d.Length > 1 ? dilations2d[1] : dH;
            int effectiveKH = dH * (wKH - 1) + 1;
            int effectiveKW = dW * (wKW - 1) + 1;
            int outH = (xH + pads[0] + (pads.Length > 2 ? pads[2] : 0) - effectiveKH) / strides[0] + 1;
            int outW = (xW + (pads.Length > 1 ? pads[1] : 0) + (pads.Length > 3 ? pads[3] : 0) - effectiveKW) / (strides.Length > 1 ? strides[1] : 1) + 1;
            return fmt == DataFormat.NHWC
                ? new[] { new[] { x[0], outH, outW, outC } }
                : new[] { new[] { x[0], outC, outH, outW } };
        }
    }
    public void Execute(OnnxOpContext ctx)
    {
        var x = ctx.Inputs[0]; var w = ctx.Inputs[1];
        var fmt = ctx.Format;
        var (stride, padTop, padLeft, padBottom, padRight, dilationH, dilationW) =
            ResolveConv2DSpatialParams(ctx, x.Shape, w.Shape, fmt);
        int pad = padTop; // Conv1D below uses the (symmetric) begin pad
        int group = ctx.GetInt("group", 1);
        var (_, inC_from_x, _, _) = x.Shape.Length >= 4 ? LayoutHelper.GetDims(x.Shape, fmt) : (1, x.Shape.Length > 1 ? x.Shape[1] : 1, 1, 1);
        // group = -1 is the TFLite depthwise sentinel — resolve to inC
        if (group == -1) group = inC_from_x;
        int outC = w.Shape[0];

        // f16 weights are wired only for the standard NCHW group-1 2D Conv path so far. The loader gates
        // which fp16 weights load as half (only those whose consumer is half-capable), so this should never
        // fire — it's a hard guard so a half weight can't silently reach a path with no half kernel (its
        // float Data is empty). Add depthwise/NHWC/grouped/Conv1D half kernels to widen this.
        if (LowPWeightDispatch.IsLowP(w) && (group != 1 || fmt != DataFormat.NCHW || x.Shape.Length != 4))
            throw new NotSupportedException(
                $"low-p ({w.DType}) Conv weight reached an unsupported path (group={group}, fmt={fmt}, rank={x.Shape.Length}); " +
                "only standard NCHW group-1 2D Conv has a low-p-weight kernel so far.");

        // Always provide a valid bias buffer (zero-filled if no bias input)
        ArrayView1D<float, Stride1D.Dense> bias;
        if (ctx.Inputs.Length > 2 && ctx.Inputs[2] != null)
        {
            bias = ctx.Inputs[2].Data;
        }
        else
        {
            // Cached zero-bias buffer per outC, owned by the OperatorRegistry.
            // Was: AllocatePermanent on every call — leaked one buffer per biasless-Conv invocation.
            bias = reg.GetOrCreateZeroBias(outC);
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
                // Depthwise conv
                if (fmt == DataFormat.NHWC)
                    reg.Conv2D.ForwardDepthwiseNHWCPadded(x.Data, w.Data, bias, ctx.Outputs[0].Data,
                        inC, inH, inW, kH, kW, stride, padTop, padLeft, padBottom, padRight, dilationH, dilationW);
                else
                    reg.Conv2D.ForwardDepthwisePadded(x.Data, w.Data, bias, ctx.Outputs[0].Data,
                        inC, inH, inW, kH, kW, stride, padTop, padLeft, padBottom, padRight, dilationH, dilationW);
            }
            else if (group == 1)
            {
                if (fmt == DataFormat.NHWC)
                    reg.Conv2D.ForwardNHWCPadded(x.Data, w.Data, bias, ctx.Outputs[0].Data,
                        inC, inH, inW, outC, kH, kW, stride, padTop, padLeft, padBottom, padRight, dilationH, dilationW);
                else if (LowPWeightDispatch.IsLowP(w)) // native low-p weight (NCHW group-1) -> generic low-p kernel, fp32 accumulate
                    LowPWeightDispatch.Conv2DPadded(reg.Conv2D, x.Data, w, bias, ctx.Outputs[0].Data,
                        inC, inH, inW, outC, kH, kW, stride, padTop, padLeft, padBottom, padRight, dilationH, dilationW);
                else
                    reg.Conv2D.ForwardPadded(x.Data, w.Data, bias, ctx.Outputs[0].Data,
                        inC, inH, inW, outC, kH, kW, stride, padTop, padLeft, padBottom, padRight, dilationH, dilationW);
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
                    reg.Conv2D.ForwardPadded(
                        x.Data.SubView(inOffset, inCPerGroup * inH * inW),
                        w.Data.SubView(wOffset, outCPerGroup * inCPerGroup * kH * kW),
                        bias.SubView(g * outCPerGroup, outCPerGroup),
                        ctx.Outputs[0].Data.SubView(outOffset, outCPerGroup * ctx.Outputs[0].Shape[2] * ctx.Outputs[0].Shape[3]),
                        inCPerGroup, inH, inW, outCPerGroup, kH, kW, stride, padTop, padLeft, padBottom, padRight, dilationH, dilationW);
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
    }
}

// ── GatherND ──

public class GatherNDOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "GatherND";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // ONNX GatherND output shape per spec:
        //   batch_dims  = attr (default 0)
        //   data_shape  = [b_0..b_{bd-1}, d_0..d_{r-1}]      (b = batch dims, d = data dims)
        //   index_shape = [b_0..b_{bd-1}, i_0..i_{q-1}, k]   (k = index last dim)
        //   out_shape   = [b_0..b_{bd-1}, i_0..i_{q-1}, d_k..d_{r-1}]
        // The previous impl returned inputs[1] (indices shape) which produced
        // wrongly-sized output buffers — downstream code wrote only the leading
        // portion and left the rest as zeros, breaking MoveNet keypoint decode.
        int batchDims = attrs.TryGetValue("batch_dims", out var bdObj)
            ? Convert.ToInt32(bdObj) : 0;
        var data = inputs[0];
        var idx  = inputs[1];
        int lastIdxDim = idx[^1];

        var outShape = new List<int>();
        // 1) batch dims (shared between data + indices)
        for (int i = 0; i < batchDims; i++) outShape.Add(idx[i]);
        // 2) indices "middle" dims (everything between batch dims and the trailing dim)
        for (int i = batchDims; i < idx.Length - 1; i++) outShape.Add(idx[i]);
        // 3) remaining data dims after the indexed dimensions
        for (int i = batchDims + lastIdxDim; i < data.Length; i++) outShape.Add(data[i]);

        // Defensive: if for any reason the spec walk produces zero dims (e.g.
        // unresolved dynamic shape upstream), fall back to a 1-element scalar
        // rather than crashing the compiler — downstream Reshape can still
        // recover real shape at runtime.
        if (outShape.Count == 0) outShape.Add(1);
        return new[] { outShape.ToArray() };
    }

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
        Tensor? zeroBias = null;
        var bias = ctx.Inputs.Length > 2 && ctx.Inputs[2] != null
            ? ctx.Inputs[2].Data
            : (zeroBias = ctx.Pool.Rent(new[] { outC }, "_conv_zero_bias")).Data;
        reg.ConvTranspose.Forward(x.Data, w.Data, bias, ctx.Outputs[0].Data,
            inC, inH, inW, outC, kH, kW, stride, pad);
        if (zeroBias != null) ctx.Pool.Return(zeroBias);
    }
}

// ── Pooling ──

public class GlobalAvgPoolOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "GlobalAveragePool";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var s = inputs[0];
        var fmt = attrs.ContainsKey("_data_format") && attrs["_data_format"].ToString() == "NHWC"
            ? DataFormat.NHWC : DataFormat.NCHW;
        var (n, c, _, _) = s.Length >= 4 ? LayoutHelper.GetDims(s, fmt) : (s[0], s.Length > 1 ? s[1] : 1, 1, 1);
        return fmt == DataFormat.NHWC
            ? new[] { new[] { n, 1, 1, c } }
            : new[] { new[] { n, c, 1, 1 } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var s = ctx.Inputs[0].Shape;
        var (N, C, _, _) = s.Length >= 4 ? LayoutHelper.GetDims(s, ctx.Format)
            : (s[0], s.Length > 1 ? s[1] : 1, 1, 1);
        int spatial = ctx.Inputs[0].ElementCount / (N * C);
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

        // Quantized table (GGUF embedding lookup) — fused dequant-Gather keeps the
        // table COMPRESSED in GPU memory and decodes only the gathered rows in-register.
        // Never expand a quantized embedding table to F32 (VRAM blow-up) and never
        // CPU-dequantize it (interpreted Blazor WASM cannot afford the pass).
        string? dataName = ctx.InputNames.Length > 0 ? ctx.InputNames[0] : null;
        if (dataName != null && ctx.QuantizedWeights != null
            && ctx.QuantizedWeights.TryGetValue(dataName, out var qTable))
        {
            if (reg.QuantizedWeightTypes == null
                || !reg.QuantizedWeightTypes.TryGetValue(dataName, out var qType))
                throw new InvalidOperationException(
                    $"Gather: quantized table '{dataName}' has no GGML type registered " +
                    "(OperatorRegistry.QuantizedWeightTypes).");
            if (axis != 0 || data.Shape.Length != 2)
                throw new NotSupportedException(
                    $"Gather on a quantized table supports axis=0 over a 2-D [rows, rowLength] " +
                    $"table only (embedding lookup); got axis={axis}, rank={data.Shape.Length}.");

            int rows = data.Shape[0];
            int rowLength = data.Shape[1];
            int numIdx = indices.ElementCount;

            // Indices may be a runtime GPU tensor (token IDs) or pre-read constants;
            // constants go through a small GPU upload (CopyFromCPU = immediate
            // writeBuffer, safe) so ONE kernel serves both.
            var idxConst = ctx.TryGetInputValues(1);
            var idxView = indices.Data;
            if (idxConst != null)
            {
                var idxTensor = ctx.Pool.Rent(new[] { numIdx }, "gather_idx_const");
                idxTensor.Data.CopyFromCPU(idxConst);
                idxView = idxTensor.Data;
            }
            reg.FusedDequantGather.GatherAxis0(qTable, idxView, ctx.Outputs[0].Data,
                numIdx, rowLength, rows, qType);
            return;
        }

        // FAIL LOUDLY before the F32 paths: a ShapeOnly data tensor (quantized table whose floats
        // never exist) here means the quantized byte-view map was not wired into this executor —
        // same failure class as the MatMul guard above. Every path below reads data.Data.
        if (data.Data.Length < data.ElementCount)
            throw new InvalidOperationException(
                $"Gather: data '{dataName ?? data.Name ?? "?"}' has no usable F32 data (view length " +
                $"{data.Data.Length}, shape [{string.Join(",", data.Shape)}] = {data.ElementCount} elements). " +
                (ctx.QuantizedWeights == null
                    ? "ctx.QuantizedWeights is NULL — this executor was constructed without the session's " +
                      "quantized byte-view map (shape-recompiled executor missing quantizedWeights?)."
                    : $"ctx.QuantizedWeights has {ctx.QuantizedWeights.Count} entries but not '{dataName}'."));

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

        // Constant-indices path: use the fused GPU Gather kernel rather than a
        // CPU-side loop dispatching one Scale per (outer, idx) pair. The
        // existing per-pair loop pays Wasm worker-pool round-trip overhead per
        // call and accumulates to >440s for a single Gather node in DA3-Small's
        // depth head (node 2541 /head/Gather_57, identified via the
        // BREAK_AT-based bisection 2026-05-05). The indices tensor is already
        // GPU-resident on .Data; dispatch one parallel kernel covering all
        // outer*numIdx*inner output elements.
        if (axis == 0)
        {
            reg.Gather.GatherAxis0Float(data.Data, indices.Data, ctx.Outputs[0].Data,
                numIdx2, innerSize2, axisSize2);
        }
        else
        {
            reg.Gather.GatherGenericFloat(data.Data, indices.Data, ctx.Outputs[0].Data,
                numIdx2, innerSize2, outerSize2, axisSize2);
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

        // Read reduction mode (ONNX opset 16+: none, add, mul, min, max)
        string reduction = "none";
        if (ctx.Attributes.TryGetValue("reduction", out var redObj) && redObj is string redStr)
            reduction = redStr.ToLowerInvariant();

        // Copy data to output first
        reg.ElementWise.Scale(data.Data, output.Data, data.ElementCount, 1f);

        // Read indices from GPU (small tensor, constant in most models)
        var idxFloats = ctx.TryGetInputValues(1);
        if (idxFloats == null)
        {
            // Runtime indices not available as constants — fall back to identity (output = data copy)
            return;
        }

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

        // For reduction modes that need current output values, read them
        bool needsReduction = reduction != "none";
        float[]? outputVals = null;
        float[]? updateVals = null;
        if (needsReduction)
        {
            outputVals = ctx.TryGetInputValues(0); // data was already copied to output
            updateVals = ctx.TryGetInputValues(2);
        }

        // For each update, compute flat offset into data and apply
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
                    flatOffset = -1;
                    break;
                }
                flatOffset += idx * stride;
            }

            if (flatOffset < 0 || flatOffset + sliceSize > output.ElementCount)
                continue; // Skip OOB scatter

            if (!needsReduction)
            {
                // reduction="none": overwrite output slice with update
                reg.ElementWise.Scale(
                    updates.Data.SubView(u * sliceSize, sliceSize),
                    output.Data.SubView(flatOffset, sliceSize),
                    sliceSize, 1f);
            }
            else if (outputVals != null && updateVals != null)
            {
                // Apply reduction on CPU, then upload the result slice
                var resultSlice = new float[sliceSize];
                for (int s = 0; s < sliceSize; s++)
                {
                    float existing = outputVals[flatOffset + s];
                    float upd = updateVals[u * sliceSize + s];
                    resultSlice[s] = reduction switch
                    {
                        "add" => existing + upd,
                        "mul" => existing * upd,
                        "min" => Math.Min(existing, upd),
                        "max" => Math.Max(existing, upd),
                        _ => upd
                    };
                }
                output.Data.SubView(flatOffset, sliceSize).CopyFromCPU(resultSlice.AsSpan(0, sliceSize).ToArray());
            }
            else
            {
                // Fallback: treat as overwrite if values aren't available
                reg.ElementWise.Scale(
                    updates.Data.SubView(u * sliceSize, sliceSize),
                    output.Data.SubView(flatOffset, sliceSize),
                    sliceSize, 1f);
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

        int totalConcatDim = 0;
        for (int n = 0; n < ctx.Inputs.Length; n++)
        {
            if (axis >= ctx.Inputs[n].Shape.Length)
                throw new InvalidOperationException(
                    $"Concat axis={axis} out of range for input[{n}] shape=[{string.Join(",", ctx.Inputs[n].Shape)}] (rank={ctx.Inputs[n].Shape.Length}). " +
                    $"All inputs: [{string.Join("; ", ctx.Inputs.Select(t => $"[{string.Join(",", t.Shape)}]"))}]");
            totalConcatDim += ctx.Inputs[n].Shape[axis];
        }

        // Fused-kernel fast path for 2/3/4-input concats: ONE GPU dispatch
        // instead of N*outer separate Scale calls. 2026-05-05 diagnosis showed
        // RoPE Concat nodes 1500-2271ms each on Wasm dominated by per-dispatch
        // overhead; this path collapses to a single dispatch + zero param upload.
        // Falls through to the per-pair Scale loop for >4 inputs (rare).
        if (Kernels.ConcatKernel.CanHandle(ctx.Inputs.Length))
        {
            var output = ctx.Outputs[0].Data;

            // WebGPU forbids binding the same GPU buffer to multiple read_write storage
            // slots in one dispatch. ONNX models can list the same tensor twice in a
            // Concat input (e.g. DA3-Small node ~196). Detect aliasing by comparing
            // Tensor object references; copy any duplicate to a fresh pool buffer.
            // Pool.Rent without a name: buffer lives in pool._allBuffers until session
            // end — never disposed early, so WebGPU command-encoder references stay valid.
            var views = new ArrayView1D<float, Stride1D.Dense>[ctx.Inputs.Length];
            for (int n = 0; n < ctx.Inputs.Length; n++)
                views[n] = ctx.Inputs[n].Data;
            for (int i = 1; i < ctx.Inputs.Length; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    if (object.ReferenceEquals(ctx.Inputs[i], ctx.Inputs[j]))
                    {
                        var src = ctx.Inputs[i];
                        var copy = ctx.Pool.Rent(src.Shape);
                        copy.Data.SubView(0, src.ElementCount).CopyFrom(src.Data.SubView(0, src.ElementCount));
                        views[i] = copy.Data;
                        break;
                    }
                }
            }

            // The fused kernel writes output[idx] for idx in [0, launchTotal); launchTotal must fit the
            // output buffer. It can overrun when an upstream shape op (e.g. a Reshape that failed to
            // resolve its target) leaves the concat inputs with mismatched non-axis dims — outer/inner
            // are taken from input0 alone, so a differing input1 makes launchTotal != the true output
            // size. The CPU backend caught this as a hard "X index out of bounds" abort; the GPU
            // backends wrote out of bounds SILENTLY. Fail loud with a precise diagnostic instead.
            long launchTotal = (long)outer * totalConcatDim * inner;
            if (launchTotal > output.Length)
                throw new InvalidOperationException(
                    $"Concat fused launch extent {launchTotal} (outer={outer} * totalConcatDim={totalConcatDim} * inner={inner}) " +
                    $"exceeds output buffer {output.Length}. Inputs [{string.Join("; ", ctx.Inputs.Select(t => "[" + string.Join(",", t.Shape) + "]"))}] " +
                    $"axis={axis} -> output [{string.Join(",", ctx.Outputs[0].Shape)}]. An upstream shape op did not resolve " +
                    "its target shape (mismatched concat inputs). Failing loud instead of writing out of bounds.");

            switch (ctx.Inputs.Length)
            {
                case 2:
                    reg.Concat.Concat2(output,
                        views[0], views[1],
                        outer, inner,
                        ctx.Inputs[0].Shape[axis], ctx.Inputs[1].Shape[axis]);
                    return;
                case 3:
                    reg.Concat.Concat3(output,
                        views[0], views[1], views[2],
                        outer, inner,
                        ctx.Inputs[0].Shape[axis], ctx.Inputs[1].Shape[axis], ctx.Inputs[2].Shape[axis]);
                    return;
                case 4:
                    reg.Concat.Concat4(output,
                        views[0], views[1], views[2], views[3],
                        outer, inner,
                        ctx.Inputs[0].Shape[axis], ctx.Inputs[1].Shape[axis],
                        ctx.Inputs[2].Shape[axis], ctx.Inputs[3].Shape[axis]);
                    return;
            }
        }

        // Fallback: per-pair Scale dispatches (correct, but pays per-dispatch overhead)
        int outOffset = 0;
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

public class InstanceNormOperator(OperatorRegistry reg) : IOnnxOperator, IPrecisionAwareOperator
{
    public string OpType => "InstanceNormalization";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        var shape = ctx.Inputs[0].Shape;
        var (N, C, _, _) = shape.Length >= 4 ? LayoutHelper.GetDims(shape, ctx.Format)
            : (shape[0], shape.Length > 1 ? shape[1] : 1, 1, 1);
        int spatial = ctx.Inputs[0].ElementCount / (N * C);
        reg.Normalization.InstanceNorm(ctx.Inputs[0].Data, ctx.Outputs[0].Data,
            ctx.Inputs[1].Data, ctx.Inputs[2].Data, N, C, spatial);
    }
    /// <summary>Precision-aware (F16) path: InstanceNorm == GroupNorm with one group per channel (G=C). Reads the
    /// low-p activation, accumulates mean/var in fp32, writes low-p; scale/bias stay fp32. eps=1e-5f matches the
    /// fp32 InstanceNorm kernel. Returns false unless input[0] is low-p and scale/bias are fp32.</summary>
    public bool TryExecuteHalf(OnnxOpContext ctx, PrecisionAwareInput[] inputs, HalfTensor output, Kernels.PrecisionAwareKernels pak)
    {
        if (inputs.Length < 3 || !inputs[0].IsHalf) return false;
        // scale/bias must be fp32 Tensors (any low-p Tensor has empty .Data → fall back; they're tiny anyway).
        if (inputs[1].Float is not { } s || LowPWeightDispatch.IsLowP(s)
            || inputs[2].Float is not { } b || LowPWeightDispatch.IsLowP(b)) return false;
        var shape = inputs[0].Half!.Shape;
        var (N, C, _, _) = shape.Length >= 4 ? LayoutHelper.GetDims(shape, ctx.Format)
            : (shape[0], shape.Length > 1 ? shape[1] : 1, 1, 1);
        int spatial = inputs[0].ElementCount / (N * C);
        pak.GroupNorm<global::ILGPU.Half>(inputs[0].Half!.Data, output.Data,
            inputs[1].Float!.Data, inputs[2].Float!.Data, N, C, spatial, numGroups: C, epsilon: 1e-5f);
        return true;
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

        // Handle higher-rank inputs (e.g., [1,1,768] from Gather with axis > 0).
        // Flatten to 2D by treating all dims except last as batch/M, last dim as K.
        int M, K;
        if (transA != 0)
        {
            // A is [K, M], transposed to [M, K]
            if (a.Shape.Length > 2)
            {
                M = a.Shape[^1];
                K = a.ElementCount / M;
            }
            else
            {
                K = a.Shape[0]; M = a.Shape[1];
            }
        }
        else
        {
            if (a.Shape.Length > 2)
            {
                K = a.Shape[^1];
                M = a.ElementCount / K;
            }
            else
            {
                M = a.Shape[0]; K = a.Shape[1];
            }
        }

        int N = transB != 0 ? b.Shape[0] : b.Shape[1];

        // Resolve actual data views, transposing as needed
        Tensor? aTransposed = null, bTransposed = null;
        var aData = a.Data;
        var bData = b.Data;

        if (transA != 0)
        {
            // A is [K, M], need [M, K] for MatMul
            aTransposed = ctx.Pool.Rent(new[] { M, K }, "_gemm_aT");
            reg.Transpose.Transpose(a.Data, aTransposed.Data, a.Shape.Length == 2 ? a.Shape : new[] { K, M }, new[] { 1, 0 });
            aData = aTransposed.Data;
        }

        // NATIVE low-p weight (Half/bf16/FP8, no f32 .Data): keep it native - the kernel converts each weight
        // to float in-register (PrecisionConvert), fp32 accumulate, no f32 transpose temp (Rule 4 zero-copy).
        // transB=1 (B stored [N,K] - the Linear/Dense export) uses the transposed-weight kernel directly;
        // transB=0 (B is [K,N]) uses the standard low-p matmul. A is the fp32 activation (transA handled above).
        if (LowPWeightDispatch.IsLowP(b))
        {
            if (transB != 0)
                LowPWeightDispatch.MatMulTransB(reg.MatMul, aData, b, ctx.Outputs[0].Data, M, K, N);
            else
                LowPWeightDispatch.MatMul(reg.MatMul, aData, b, ctx.Outputs[0].Data, M, K, N);
        }
        else
        {
            if (transB != 0)
            {
                // B is [N, K], need [K, N] for MatMul
                bTransposed = ctx.Pool.Rent(new[] { K, N }, "_gemm_bT");
                reg.Transpose.Transpose(b.Data, bTransposed.Data, b.Shape.Length == 2 ? b.Shape : new[] { N, K }, new[] { 1, 0 });
                bData = bTransposed.Data;
            }
            reg.MatMul.MatMul(aData, bData, ctx.Outputs[0].Data, M, K, N);
        }

        if (aTransposed != null) ctx.Pool.Return(aTransposed);
        if (bTransposed != null) ctx.Pool.Return(bTransposed);

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
        var fmt = attrs.ContainsKey("_data_format") && attrs["_data_format"].ToString() == "NHWC"
            ? DataFormat.NHWC : DataFormat.NCHW;
        var (n, c, xH, xW) = x.Length >= 4 ? LayoutHelper.GetDims(x, fmt) : (x[0], x.Length > 1 ? x[1] : 1, 1, 1);
        var kernelShape = attrs.ContainsKey("kernel_shape") ? ((long[])attrs["kernel_shape"]).Select(k => (int)k).ToArray() : new[] { 2, 2 };
        var strides = attrs.ContainsKey("strides") ? ((long[])attrs["strides"]).Select(s => (int)s).ToArray() : new[] { 1, 1 };
        var pads = attrs.ContainsKey("pads") ? ((long[])attrs["pads"]).Select(p => (int)p).ToArray() : new int[4];
        // Handle auto_pad for TFLite
        string autoPad = attrs.ContainsKey("auto_pad") ? attrs["auto_pad"].ToString()! : "NOTSET";
        if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER")
        {
            int padH = Math.Max(0, ((int)Math.Ceiling((double)xH / strides[0]) - 1) * strides[0] + kernelShape[0] - xH);
            int padW = Math.Max(0, ((int)Math.Ceiling((double)xW / strides[1]) - 1) * strides[1] + kernelShape[1] - xW);
            pads = new[] { padH / 2, padW / 2, padH - padH / 2, padW - padW / 2 };
        }
        int outH = (xH + pads[0] + pads[2] - kernelShape[0]) / strides[0] + 1;
        int outW = (xW + pads[1] + pads[3] - kernelShape[1]) / strides[1] + 1;
        return fmt == DataFormat.NHWC
            ? new[] { new[] { n, outH, outW, c } }
            : new[] { new[] { n, c, outH, outW } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var x = ctx.Inputs[0];
        var (N, C, H, W) = x.Shape.Length >= 4 ? LayoutHelper.GetDims(x.Shape, ctx.Format) : (x.Shape[0], 1, 1, 1);
        var ks = ctx.GetInts("kernel_shape"); int kH = ks.Length > 0 ? ks[0] : 2; int kW = ks.Length > 1 ? ks[1] : kH;
        var st = ctx.GetInts("strides"); int sH = st.Length > 0 ? st[0] : 1; int sW = st.Length > 1 ? st[1] : sH;
        var pa = ctx.GetInts("pads"); int pH = pa.Length > 0 ? pa[0] : 0; int pW = pa.Length > 1 ? pa[1] : 0;
        // Handle auto_pad
        var autoPad = ctx.Attributes.TryGetValue("auto_pad", out var ap2) ? ap2.ToString()! : "NOTSET";
        if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER")
        {
            int padH = Math.Max(0, ((int)Math.Ceiling((double)H / sH) - 1) * sH + kH - H);
            int padW = Math.Max(0, ((int)Math.Ceiling((double)W / sW) - 1) * sW + kW - W);
            pH = autoPad == "SAME_UPPER" ? padH / 2 : padH - padH / 2;
            pW = autoPad == "SAME_UPPER" ? padW / 2 : padW - padW / 2;
        }
        reg.Pooling.MaxPool2D(x.Data, ctx.Outputs[0].Data, N, C, H, W, kH, kW, sH, sW, pH, pW);
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
        var (N, C, H, W) = x.Shape.Length >= 4 ? LayoutHelper.GetDims(x.Shape, ctx.Format) : (x.Shape[0], 1, 1, 1);
        var ks = ctx.GetInts("kernel_shape"); int kH = ks[0]; int kW = ks.Length > 1 ? ks[1] : kH;
        var st = ctx.GetInts("strides"); int sH = st.Length > 0 ? st[0] : 1; int sW = st.Length > 1 ? st[1] : sH;
        var pa = ctx.GetInts("pads"); int pH = pa.Length > 0 ? pa[0] : 0; int pW = pa.Length > 1 ? pa[1] : 0;
        var autoPad = ctx.Attributes.TryGetValue("auto_pad", out var ap2) ? ap2.ToString()! : "NOTSET";
        if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER")
        {
            int padH = Math.Max(0, ((int)Math.Ceiling((double)H / sH) - 1) * sH + kH - H);
            int padW = Math.Max(0, ((int)Math.Ceiling((double)W / sW) - 1) * sW + kW - W);
            pH = autoPad == "SAME_UPPER" ? padH / 2 : padH - padH / 2;
            pW = autoPad == "SAME_UPPER" ? padW / 2 : padW - padW / 2;
        }
        reg.Pooling.AvgPool2D(x.Data, ctx.Outputs[0].Data, N, C, H, W, kH, kW, sH, sW, pH, pW);
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
        var inShape = ctx.Inputs[0].Shape;
        var outShape = ctx.Outputs[0].Shape;
        int C = inShape[0] * inShape[1]; // N*C for batch
        int inH = inShape[2]; int inW = inShape[3];
        int outH = outShape[2]; int outW = outShape[3];

        // ONNX Resize `mode`: "nearest" (the op's DEFAULT) | "linear" | "cubic". We previously IGNORED it and
        // always bilinear-resized — which low-passes a nearest Resize, blurring the whole image. The SD-Turbo
        // VAE decoder upsamples with mode="nearest" (diffusers Upsample2D F.interpolate nearest), so bilinear
        // smoothing made every generated image soft. Honor `mode` (matches UpsampleOperator). Verified: with
        // nearest, our VAE decode matches the ONNX Runtime oracle (sharp) instead of the soft bilinear output.
        var mode = ctx.GetString("mode", "nearest");
        if (mode == "nearest")
        {
            reg.ElementWise.NearestUpsample(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].Shape, ctx.Outputs[0].Shape);
            return;
        }
        // linear/cubic → bilinear (cubic approximated). align_corners per the coordinate-transform mode.
        var ctMode = ctx.GetString("coordinate_transformation_mode", "half_pixel");
        if (ctMode == "align_corners")
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
        else if (ctx.TryGetInputValues(1) is float[] startsF && ctx.TryGetInputValues(2) is float[] endsF)
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
                // Direct CPU->GPU upload (was AllocatePermanent + Scale leak).
                ctx.Outputs[0].Data.SubView(0, outCount).CopyFromCPU(result);
                return;
            }
        }

        // Fused-kernel fast path: ONE GPU dispatch instead of N recursive Scale
        // calls per contiguous run. Engaged when all steps are positive and rank
        // is within SliceKernel.MAX_RANK (covers the vast majority of ONNX
        // slice patterns; transformer attention RoPE blocks specifically).
        // 2026-05-05 diagnosis (commit `54d3eae`) showed Slice nodes 700-1100ms
        // each on Wasm in DA3-Small RoPE blocks driven by per-dispatch overhead;
        // this path collapses that to one dispatch + small param upload.
        bool allPositiveSteps = true;
        for (int d = 0; d < rank; d++) { if (sliceSteps[d] <= 0) { allPositiveSteps = false; break; } }
        if (allPositiveSteps && rank <= Kernels.SliceKernel.MAX_RANK)
        {
            reg.Slice.Slice(input.Data, ctx.Outputs[0].Data,
                sliceStarts, sliceSteps, outShape, inStrides, rank, outCount);
            return;
        }

        // Fallback for negative steps or extreme rank: copy contiguous slices along last axis
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

// ═══════════════════════════════════════════════════════════
//  New operators — full ONNX coverage (batch 2)
// ═══════════════════════════════════════════════════════════

// ── Reduce variants (use same axis decomposition as ReduceSum) ──

public class ReduceProdOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ReduceProd";
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
        reg.Reductions.ReduceProd(ctx.Inputs[0].Data, ctx.Outputs[0].Data, outer, reduce, inner);
    }
}

public class ReduceL1Operator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ReduceL1";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new ReduceMeanOperator(reg).InferOutputShapes(inputs, attrs);
    public void Execute(OnnxOpContext ctx)
    {
        // ReduceL1 = ReduceSum(Abs(x))
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
        // Abs input into temp, then ReduceSum
        int count = ctx.Inputs[0].ElementCount;
        var absBuf = ctx.Pool.Rent(new[] { count });
        reg.ElementWise.Abs(ctx.Inputs[0].Data, absBuf.Data, count);
        reg.Reductions.ReduceSum(absBuf.Data, ctx.Outputs[0].Data, outer, reduce, inner);
    }
}

public class ReduceL2Operator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ReduceL2";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new ReduceMeanOperator(reg).InferOutputShapes(inputs, attrs);
    public void Execute(OnnxOpContext ctx)
    {
        // ReduceL2 = Sqrt(ReduceSum(x^2))
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
        // Square input, ReduceSum, then Sqrt
        int count = ctx.Inputs[0].ElementCount;
        var sqBuf = ctx.Pool.Rent(new[] { count });
        reg.ElementWise.Mul(ctx.Inputs[0].Data, ctx.Inputs[0].Data, sqBuf.Data, count);
        reg.Reductions.ReduceSum(sqBuf.Data, ctx.Outputs[0].Data, outer, reduce, inner);
        int outCount = ctx.Outputs[0].ElementCount;
        reg.ElementWise.Sqrt(ctx.Outputs[0].Data, ctx.Outputs[0].Data, outCount);
    }
}

public class ReduceSumSquareOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ReduceSumSquare";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new ReduceMeanOperator(reg).InferOutputShapes(inputs, attrs);
    public void Execute(OnnxOpContext ctx)
    {
        // ReduceSumSquare = ReduceSum(x^2)
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
        int count = ctx.Inputs[0].ElementCount;
        var sqBuf = ctx.Pool.Rent(new[] { count });
        reg.ElementWise.Mul(ctx.Inputs[0].Data, ctx.Inputs[0].Data, sqBuf.Data, count);
        reg.Reductions.ReduceSum(sqBuf.Data, ctx.Outputs[0].Data, outer, reduce, inner);
    }
}

public class ReduceLogSumOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ReduceLogSum";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new ReduceMeanOperator(reg).InferOutputShapes(inputs, attrs);
    public void Execute(OnnxOpContext ctx)
    {
        // ReduceLogSum = Log(ReduceSum(x))
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
        int outCount = ctx.Outputs[0].ElementCount;
        reg.ElementWise.Log(ctx.Outputs[0].Data, ctx.Outputs[0].Data, outCount);
    }
}

public class ReduceLogSumExpOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ReduceLogSumExp";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new ReduceMeanOperator(reg).InferOutputShapes(inputs, attrs);
    public void Execute(OnnxOpContext ctx)
    {
        // ReduceLogSumExp = Log(ReduceSum(Exp(x)))
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
        int count = ctx.Inputs[0].ElementCount;
        var expBuf = ctx.Pool.Rent(new[] { count });
        reg.ElementWise.Exp(ctx.Inputs[0].Data, expBuf.Data, count);
        reg.Reductions.ReduceSum(expBuf.Data, ctx.Outputs[0].Data, outer, reduce, inner);
        int outCount = ctx.Outputs[0].ElementCount;
        reg.ElementWise.Log(ctx.Outputs[0].Data, ctx.Outputs[0].Data, outCount);
    }
}

// ── GlobalMaxPool ──

public class GlobalMaxPoolOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "GlobalMaxPool";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // [N, C, H, W] → [N, C, 1, 1]
        var shape = inputs[0].ToArray();
        for (int i = 2; i < shape.Length; i++) shape[i] = 1;
        return new[] { shape };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // GlobalMaxPool = ReduceMax over spatial dims
        var shape = ctx.Inputs[0].Shape;
        int N = shape[0], C = shape[1];
        int spatial = 1;
        for (int i = 2; i < shape.Length; i++) spatial *= shape[i];
        reg.Reductions.ReduceMax(ctx.Inputs[0].Data, ctx.Outputs[0].Data, N * C, spatial, 1);
    }
}

// ── SpaceToDepth ──

public class SpaceToDepthOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "SpaceToDepth";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        int blocksize = attrs.ContainsKey("blocksize") ? Convert.ToInt32(attrs["blocksize"]) : 2;
        var s = inputs[0]; // [N, C, H, W]
        return new[] { new[] { s[0], s[1] * blocksize * blocksize, s[2] / blocksize, s[3] / blocksize } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // SpaceToDepth: [N, C, H, W] → [N, C*bs*bs, H/bs, W/bs]
        // Inverse of DepthToSpace. Rearrange spatial blocks into channels.
        int blocksize = ctx.GetInt("blocksize", 2);
        var shape = ctx.Inputs[0].Shape;
        int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
        int outC = C * blocksize * blocksize, outH = H / blocksize, outW = W / blocksize;
        // Reshape → Transpose → Reshape via intermediate steps
        // [N,C,H/bs,bs,W/bs,bs] → [N,C,bs,bs,H/bs,W/bs] → [N,C*bs*bs,H/bs,W/bs]
        // Use transpose kernel for the rearrangement
        reg.Transpose.Transpose(ctx.Inputs[0].Data, ctx.Outputs[0].Data,
            new[] { N, C, outH, blocksize, outW, blocksize },
            new[] { 0, 1, 3, 5, 2, 4 });
    }
}

// ── Trilu (triangular part of a matrix) ──

public class TriluOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Trilu";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        int upper = ctx.GetInt("upper", 1);
        int k = 0;
        if (ctx.Inputs.Length > 1)
        {
            var kVals = ctx.TryGetInputValues(1);
            if (kVals != null && kVals.Length > 0) k = (int)kVals[0];
        }

        var shape = ctx.Inputs[0].Shape;
        int rows = shape[^2], cols = shape[^1];
        int total = ctx.Inputs[0].ElementCount;
        int batchStride = rows * cols;

        // GPU path: params buffer [rows, cols, k, upper, batchStride]
        var paramsData = new float[] { rows, cols, k, upper, batchStride };
        var paramsBuf = ctx.Pool.Rent(new[] { paramsData.Length });
        paramsBuf.Data.SubView(0, paramsData.Length).CopyFromCPU(paramsData);
        reg.ElementWise.Trilu(ctx.Inputs[0].Data, ctx.Outputs[0].Data, paramsBuf.Data, total);
    }
}

// ── ScatterElements ──

public class ScatterElementsOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ScatterElements";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // ScatterElements: output = copy of data, then output[indices[i]] = updates[i] along axis
        var dataVals = ctx.TryGetInputValues(0);
        var idxVals = ctx.TryGetInputValues(1);
        var updateVals = ctx.TryGetInputValues(2);

        int total = ctx.Inputs[0].ElementCount;
        // Copy data to output first
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, total, 1f);

        if (dataVals == null || idxVals == null || updateVals == null) return;

        int axis = ctx.GetInt("axis", 0);
        var shape = ctx.Inputs[0].Shape;
        if (axis < 0) axis += shape.Length;

        int rank = shape.Length;
        int axisSize = shape[axis];

        // Compute strides
        var strides = new int[rank];
        int stride = 1;
        for (int d = rank - 1; d >= 0; d--) { strides[d] = stride; stride *= shape[d]; }

        // Apply scatter: for each element in indices, replace the corresponding output element
        var result = (float[])dataVals.Clone();
        var idxShape = ctx.Inputs[1].Shape;
        var idxStrides = new int[rank];
        stride = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            idxStrides[d] = stride;
            stride *= d < idxShape.Length ? idxShape[d] : 1;
        }

        for (int i = 0; i < idxVals.Length; i++)
        {
            // Decompose flat index into N-D coordinates using idx tensor shape
            int rem = i;
            var coords = new int[rank];
            for (int d = 0; d < rank && d < idxShape.Length; d++)
            {
                coords[d] = rem / idxStrides[d];
                rem %= idxStrides[d];
            }
            // Replace the axis coordinate with the index value
            int scatterIdx = (int)idxVals[i];
            if (scatterIdx < 0) scatterIdx += axisSize;
            if (scatterIdx < 0 || scatterIdx >= axisSize) continue;
            coords[axis] = scatterIdx;

            // Compute flat output index
            int outIdx = 0;
            for (int d = 0; d < rank; d++) outIdx += coords[d] * strides[d];
            if (outIdx >= 0 && outIdx < result.Length)
                result[outIdx] = updateVals[i];
        }

        ctx.Outputs[0].Data.SubView(0, total).CopyFromCPU(result);
    }
}

// ── NonMaxSuppression ──

public class NonMaxSuppressionOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "NonMaxSuppression";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // Output: [num_selected_indices, 3] — (batch_index, class_index, box_index)
        // Max possible = maxOutputBoxesPerClass * numClasses * batchSize
        int maxOut = 200; // Conservative estimate — actual size determined at runtime
        return new[] { new[] { maxOut, 3 } };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // NMS: greedy selection — inherently sequential, runs on CPU.
        // Inputs: boxes [N, num_boxes, 4], scores [N, num_classes, num_boxes]
        // Optional: max_output_boxes_per_class, iou_threshold, score_threshold
        int maxBoxes = 200;
        float iouThreshold = 0.5f;
        float scoreThreshold = 0f;

        if (ctx.Inputs.Length > 2) { var v = ctx.TryGetInputValues(2); if (v != null && v.Length > 0) maxBoxes = (int)v[0]; }
        if (ctx.Inputs.Length > 3) { var v = ctx.TryGetInputValues(3); if (v != null && v.Length > 0) iouThreshold = v[0]; }
        if (ctx.Inputs.Length > 4) { var v = ctx.TryGetInputValues(4); if (v != null && v.Length > 0) scoreThreshold = v[0]; }

        // Read boxes and scores to CPU (NMS is small data, CPU is fine)
        var boxVals = ctx.TryGetInputValues(0);
        var scoreVals = ctx.TryGetInputValues(1);
        if (boxVals == null || scoreVals == null)
        {
            reg.ElementWise.Fill(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount, 0f);
            return;
        }

        var boxShape = ctx.Inputs[0].Shape; // [N, num_boxes, 4]
        var scoreShape = ctx.Inputs[1].Shape; // [N, num_classes, num_boxes]
        int N = boxShape[0], numBoxes = boxShape[1];
        int numClasses = scoreShape.Length >= 3 ? scoreShape[1] : 1;

        var selected = new List<float>();
        for (int n = 0; n < N && selected.Count < maxBoxes * 3; n++)
        {
            for (int c = 0; c < numClasses && selected.Count < maxBoxes * 3; c++)
            {
                // Get scores for this batch+class, sort by score descending
                var indices = Enumerable.Range(0, numBoxes)
                    .Select(b => (idx: b, score: scoreVals[n * numClasses * numBoxes + c * numBoxes + b]))
                    .Where(x => x.score > scoreThreshold)
                    .OrderByDescending(x => x.score)
                    .Select(x => x.idx).ToList();

                var keep = new List<int>();
                var suppressed = new HashSet<int>();
                foreach (var idx in indices)
                {
                    if (suppressed.Contains(idx)) continue;
                    keep.Add(idx);
                    if (keep.Count >= maxBoxes) break;
                    // Suppress overlapping boxes
                    int bOff = n * numBoxes * 4 + idx * 4;
                    float y1a = boxVals[bOff], x1a = boxVals[bOff + 1], y2a = boxVals[bOff + 2], x2a = boxVals[bOff + 3];
                    float areaA = Math.Max(0, y2a - y1a) * Math.Max(0, x2a - x1a);
                    foreach (var other in indices)
                    {
                        if (suppressed.Contains(other) || other == idx) continue;
                        int oOff = n * numBoxes * 4 + other * 4;
                        float y1b = boxVals[oOff], x1b = boxVals[oOff + 1], y2b = boxVals[oOff + 2], x2b = boxVals[oOff + 3];
                        float interY = Math.Max(0, Math.Min(y2a, y2b) - Math.Max(y1a, y1b));
                        float interX = Math.Max(0, Math.Min(x2a, x2b) - Math.Max(x1a, x1b));
                        float inter = interY * interX;
                        float areaB = Math.Max(0, y2b - y1b) * Math.Max(0, x2b - x1b);
                        float iou = inter / (areaA + areaB - inter + 1e-10f);
                        if (iou > iouThreshold) suppressed.Add(other);
                    }
                }
                foreach (var k in keep) { selected.Add(n); selected.Add(c); selected.Add(k); }
            }
        }

        // Upload result
        int resultCount = Math.Min(selected.Count, ctx.Outputs[0].ElementCount);
        if (resultCount > 0)
        {
            var selArr = selected.ToArray();
            if (resultCount < selArr.Length) { var t = new float[resultCount]; Array.Copy(selArr, t, resultCount); ctx.Outputs[0].Data.SubView(0, resultCount).CopyFromCPU(t); }
            else ctx.Outputs[0].Data.SubView(0, resultCount).CopyFromCPU(selArr);
        }
        else
        {
            reg.ElementWise.Fill(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount, 0f);
        }
    }
}
