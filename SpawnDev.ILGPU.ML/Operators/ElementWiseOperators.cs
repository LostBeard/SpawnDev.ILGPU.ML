using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using static SpawnDev.ILGPU.ML.Operators.BroadcastHelper;

namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// General N-dimensional broadcast binary operation for small tensors.
/// Uses pre-read constant values to avoid GPU→CPU readback.
/// For large tensors, falls back to element-wise (same-size) operation.
/// </summary>
internal static class BroadcastHelper
{
    // NO DEFAULT on gpuOp, deliberately. It used to default to BroadcastOp.Add, and every call site that
    // omitted it silently performed ADDITION on the two GPU branches below while the CPU-constant branch
    // used the correct `op` lambda - so the defect only appeared once the operands stopped being
    // CPU-resolvable. It cost a day through And/Or/Xor and was still live in Min, Max and PRelu. A required
    // parameter turns "I forgot the GPU op" from a wrong answer into a compile error.
    public static void BroadcastBinaryOp(OnnxOpContext ctx, OperatorRegistry reg, Func<float, float, float> op,
        BroadcastOp gpuOp)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];

        // The output was sized by shape inference at COMPILE time. In a graph whose shapes are only known
        // at runtime (whisper's decoder builds its causal mask from the live sequence length), the inputs
        // were placeholders then, so the output can be far too small - [1] where the real broadcast of
        // [1,1,4,1] against [1,1,1,4] is [1,1,4,4]. Writing into it anyway produces a tensor of the wrong
        // shape holding a few correct values, which is how whisper's causal mask came out as "row 0 only"
        // and every decoder position could see the future. Recompute from the REAL input shapes.
        var trueShape = Tensors.TensorHelpers.BroadcastShape(a.Shape, b.Shape);
        int trueCount = Tensors.TensorHelpers.ElementCount(trueShape);
        if (trueCount != ctx.Outputs[0].ElementCount)
        {
            if (ctx.Outputs[0].Data.Length >= trueCount)
                ctx.Outputs[0].Shape = trueShape;      // buffer is big enough; correct the metadata
            else
                ctx.Outputs[0] = ctx.Pool.Rent(trueShape, "_broadcast_out");   // executor reads ctx.Outputs after Execute
        }

        var outShape = ctx.Outputs[0].Shape;
        int outCount = ctx.Outputs[0].ElementCount;

        // Try to use pre-read constant values (no GPU readback)
        var aVals = ctx.TryGetInputValues(0);
        var bVals = ctx.TryGetInputValues(1);

        if (aVals != null && bVals != null)
        {
            // Both inputs are small constants — compute on CPU
            var result = new float[outCount];
            var aStrides = ComputeStrides(a.Shape, outShape);
            var bStrides = ComputeStrides(b.Shape, outShape);
            var outStrides = ComputeStrides(outShape, outShape);

            for (int i = 0; i < outCount; i++)
            {
                int aIdx = MapIndex(i, outStrides, aStrides, outShape.Length);
                int bIdx = MapIndex(i, outStrides, bStrides, outShape.Length);
                result[i] = op(
                    aIdx < aVals.Length ? aVals[aIdx] : 0f,
                    bIdx < bVals.Length ? bVals[bIdx] : 0f);
            }

            // Direct CPU->GPU upload to output (was AllocatePermanent + Scale, leaked a permanent buffer per
            // call). Both inputs constant → output constant. Under CUDA-graph capture: arena stable-slot +
            // captured GPU CopyFrom so replay reproduces the write (skipping leaves the pooled buffer stale).
            if (SpawnDev.ILGPU.ML.Graph.GraphExecutor.UseCaptureParamSlots)
                Kernels.CaptureParamArena.CaptureConstWrite(reg.Accelerator, ctx.Outputs[0].Data.SubView(0, outCount), result);
            else
                ctx.Outputs[0].Data.SubView(0, outCount).CopyFromCPU(result);
        }
        else if (bVals != null && bVals.Length == 1 && a.ElementCount > 1)
        {
            // ⚠️ SCALAR fast path. The general branch below expands the constant to the FULL output
            // shape in a host array, then uploads it - so subtracting a single zero-point from a
            // 460k-element activation allocated a 460k float[], ran a 460k-iteration MapIndex loop on the
            // CPU, and pushed 1.8 MB across the bus. Per call. On ZipVoice's decoder that made `Sub` the
            // single largest GPU-attributed cost at 27.8% (406 calls, 1.26 ms each) against `Mul` at
            // 0.034 ms for the same kind of elementwise work.
            //
            // Nothing about it was necessary: BroadcastBinaryOpND already broadcasts on the GPU from a
            // stride map, so the operand only has to BE there - it does not have to be expanded first. One
            // element is uploaded instead of `outCount`.
            var scalarShape = new[] { 1 };
            var scalarTensor = ctx.Pool.Rent(scalarShape, "_broadcast_scalar");
            if (SpawnDev.ILGPU.ML.Graph.GraphExecutor.UseCaptureParamSlots)
            {
                // Same reasoning as the expanded path: a pooled transient's H2D is skipped during capture
                // replay, so a deterministic constant needs a stable arena slot instead.
                var sView = Kernels.CaptureParamArena.Shared(reg.Accelerator).RentStableSlotFloat(bVals);
                scalarTensor = new Tensor(sView, scalarShape, "_broadcast_scalar");
            }
            else
            {
                scalarTensor.Data.SubView(0, 1).CopyFromCPU(bVals);
            }
            reg.ElementWise.BroadcastBinaryOpND(
                a.Data, scalarTensor.Data, ctx.Outputs[0].Data,
                a.Shape, scalarShape, outShape, gpuOp);
            ctx.Pool.Return(scalarTensor);
        }
        else if (bVals != null && a.ElementCount > b.ElementCount)
        {
            // b is a small runtime constant, a is a large GPU tensor.
            // Expand b to full output shape on CPU, upload, then GPU element-wise op.
            // Uses Rent (not AllocatePermanent) to avoid buffer leaks through 748-node models.
            var bExpanded = new float[outCount];
            var bStrides = ComputeStrides(b.Shape, outShape);
            var outStrides = ComputeStrides(outShape, outShape);
            for (int i = 0; i < outCount; i++)
            {
                int bIdx = MapIndex(i, outStrides, bStrides, outShape.Length);
                bExpanded[i] = bIdx < bVals.Length ? bVals[bIdx] : 0f;
            }
            Tensor bExpandedTensor;
            if (SpawnDev.ILGPU.ML.Graph.GraphExecutor.UseCaptureParamSlots)
            {
                // CUDA-graph capture: "_broadcast_b_expanded" is a transient reused pool buffer, so skipping its
                // H2D would leave stale data. The expanded constant is deterministic — use a stable arena slot
                // (written in warm, skip-write in capture). Not pool-registered → Pool.Return is a safe no-op.
                var bView = Kernels.CaptureParamArena.Shared(reg.Accelerator).RentStableSlotFloat(bExpanded);
                bExpandedTensor = new Tensor(bView, outShape, "_broadcast_b_expanded");
            }
            else
            {
                bExpandedTensor = ctx.Pool.Rent(outShape, "_broadcast_b_expanded");
                bExpandedTensor.Data.SubView(0, outCount).CopyFromCPU(bExpanded);
            }
            // Use GPU N-D broadcast kernel (a and bExpanded are same shape → element-wise)
            reg.ElementWise.BroadcastBinaryOpND(
                a.Data, bExpandedTensor.Data, ctx.Outputs[0].Data,
                a.Shape, outShape, outShape, gpuOp);
            ctx.Pool.Return(bExpandedTensor);
        }
        else
        {
            // General N-D broadcast on GPU — handles arbitrary shape combinations.
            // Uses stride-based index mapping kernels (BroadcastDivImpl, etc.)
            reg.ElementWise.BroadcastBinaryOpND(
                a.Data, b.Data, ctx.Outputs[0].Data,
                a.Shape, b.Shape, outShape, gpuOp);
        }
    }

    /// <summary>
    /// WebGPU and WebGL forbid binding the same GPU buffer to two storage slots in one dispatch. When an
    /// ONNX graph feeds the SAME tensor to BOTH operands of a binary op (e.g. <c>x - x</c>, <c>x / x</c> —
    /// the executor hands back the same <see cref="Tensor"/> instance for a repeated input name), the op
    /// would bind one buffer to two slots and the backend throws an aliasing violation. This de-aliases by
    /// copying the second operand into a pooled temp. Returns <paramref name="b"/> unchanged when there is
    /// no aliasing; otherwise the pooled copy (the caller must <c>ctx.Pool.Return(rented)</c> after the
    /// dispatch). <c>CopyFrom</c> is a native GPU→GPU command (no kernel dispatch) valid on every backend.
    /// </summary>
    internal static Tensor DeAliasSecondOperand(OnnxOpContext ctx, Tensor a, Tensor b, out Tensor? rented)
    {
        rented = null;
        if (!object.ReferenceEquals(a, b)) return b;
        rented = ctx.Pool.Rent(b.Shape, "_dealias");
        rented.Data.SubView(0, b.ElementCount).CopyFrom(b.Data.SubView(0, b.ElementCount));
        return rented;
    }

    internal static int[] ComputeStrides(int[] shape, int[] outShape)
    {
        // Broadcast strides: if dim size is 1 or shape is shorter, stride is 0 (broadcast)
        int rank = outShape.Length;
        var strides = new int[rank];
        int offset = rank - shape.Length;
        int stride = 1;
        for (int i = rank - 1; i >= 0; i--)
        {
            int si = i - offset;
            if (si >= 0 && shape[si] > 1)
            {
                strides[i] = stride;
                stride *= shape[si];
            }
            else
            {
                strides[i] = 0; // Broadcast dimension
            }
        }
        return strides;
    }

    internal static int MapIndex(int outIdx, int[] outStrides, int[] inStrides, int rank)
    {
        int inIdx = 0;
        int remaining = outIdx;
        for (int d = 0; d < rank; d++)
        {
            int coord = outStrides[d] > 0 ? remaining / outStrides[d] : 0;
            remaining = outStrides[d] > 0 ? remaining % outStrides[d] : remaining;
            inIdx += coord * inStrides[d];
        }
        return inIdx;
    }
}

// ── Activations ──

public class ReluOperator(OperatorRegistry reg) : IOnnxOperator, IPrecisionAwareOperator
{
    public string OpType => "Relu";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.ReLU(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
    public bool TryExecuteHalf(OnnxOpContext ctx, PrecisionAwareInput[] inputs, Tensors.HalfTensor output, Kernels.PrecisionAwareKernels pak)
    {
        if (inputs.Length < 1 || !inputs[0].IsHalf) return false;
        pak.Relu<global::ILGPU.Half>(inputs[0].Half!.Data, output.Data, output.ElementCount);
        return true;
    }
}

public class GeluOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Gelu";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.GELU(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class SigmoidOperator(OperatorRegistry reg) : IOnnxOperator, IPrecisionAwareOperator
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
    public bool TryExecuteHalf(OnnxOpContext ctx, PrecisionAwareInput[] inputs, Tensors.HalfTensor output, Kernels.PrecisionAwareKernels pak)
    {
        if (inputs.Length < 1 || !inputs[0].IsHalf) return false;
        pak.Sigmoid<global::ILGPU.Half>(inputs[0].Half!.Data, output.Data, output.ElementCount);
        return true;
    }
}

public class SiLUOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "SiLU"; // Not standard ONNX — but used in YOLO via Mul(x, Sigmoid(x))
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, 1f);
        reg.Activations.SiLUInPlace(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount);
    }
}

/// <summary>Fused SwiGLU (gate, up) → (gate·sigmoid(gate))·up in ONE kernel — replaces the SiLU MLP's
/// Sigmoid + Mul + Mul (3 dispatches → 1; biggest on WebGPU dispatch overhead). Bit-identical to that chain.</summary>
public class SwiGLUOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "SwiGLU";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.Activations.SwiGLU(ctx.Inputs[0].Data, ctx.Inputs[1].Data, ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount);
    }
}

public class LeakyReluOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "LeakyRelu";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        float alpha = ctx.GetFloat("alpha", 0.01f);
        reg.ElementWise.LeakyReLU(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, alpha);
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
        // Opset 6: min/max as attributes. Opset 11+ (the default tf2onnx, PyTorch,
        // and onnxruntime-tools emit today): min/max as optional INPUT tensors
        // (1-element initializers), NOT attributes.
        //
        // The previous impl only read attributes. Modern opset 11+ Clip(0,6) / Relu6
        // silently became identity (minVal=float.MinValue, maxVal=float.MaxValue, the
        // fallback branch never matched). MobileNetV2-FPN-based MoveNet exported by
        // tf2onnx 1.16 has Relu6 between every conv block; without the [0, 6] clamp,
        // activations doubled per block and saturated to 10^12 by the output decode,
        // producing all-zero / all-saturated keypoint confidences.
        float minVal = ctx.GetFloat("min", float.MinValue);
        float maxVal = ctx.GetFloat("max", float.MaxValue);

        // Opset 11+: pull scalar min from inputs[1], scalar max from inputs[2].
        // TryGetInputValues returns the pre-extracted initializer / Constant-node float[]
        // (already in CPU memory at session-init time for small constants).
        if (ctx.Inputs.Length > 1 && ctx.InputNames.Length > 1 && !string.IsNullOrEmpty(ctx.InputNames[1]))
        {
            var minVals = ctx.TryGetInputValues(1);
            if (minVals != null && minVals.Length > 0) minVal = minVals[0];
        }
        if (ctx.Inputs.Length > 2 && ctx.InputNames.Length > 2 && !string.IsNullOrEmpty(ctx.InputNames[2]))
        {
            var maxVals = ctx.TryGetInputValues(2);
            if (maxVals != null && maxVals.Length > 0) maxVal = maxVals[0];
        }

        if (minVal == 0f && maxVal == float.MaxValue)
        {
            // Fast path: Clip(0, inf) = ReLU
            reg.ElementWise.ReLU(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
        }
        else
        {
            reg.ElementWise.Clip(ctx.Inputs[0].Data, ctx.Outputs[0].Data,
                ctx.Inputs[0].ElementCount, minVal, maxVal);
        }
    }
}

// ── Binary element-wise ──

/// <summary>
/// Last dimension of a tensor, or 0 for a rank-0 (scalar) one.
/// </summary>
/// <remarks>
/// The broadcast branches below compare an operand's element count against the other's last dimension.
/// Written as <c>Shape[^1]</c> that throws on a scalar, whose shape array is empty - and a scalar is a
/// perfectly legal operand, so the guard has to answer "no last dimension" instead of failing. Returning
/// 0 makes those comparisons false, which routes scalars to their own branches.
/// </remarks>
internal static class BroadcastShapeHelpers
{
    public static int LastDim(Tensor t) => t.Shape.Length > 0 ? t.Shape[^1] : 0;
}

public class AddOperator(OperatorRegistry reg) : IOnnxOperator, IPrecisionAwareOperator
{
    public string OpType => "Add";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public bool TryExecuteHalf(OnnxOpContext ctx, PrecisionAwareInput[] inputs, Tensors.HalfTensor output, Kernels.PrecisionAwareKernels pak)
    {
        // Only the elementwise, no-broadcast, both-low-p residual add (the large case). Broadcast / fp32 → fallback.
        if (inputs.Length < 2 || !inputs[0].IsHalf || !inputs[1].IsHalf) return false;
        if (inputs[0].ElementCount != output.ElementCount || inputs[1].ElementCount != output.ElementCount) return false;
        pak.Add<global::ILGPU.Half>(inputs[0].Half!.Data, inputs[1].Half!.Data, output.Data, output.ElementCount);
        return true;
    }
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        var output = ctx.Outputs[0];

        if (a.ElementCount == b.ElementCount && a.ElementCount == ctx.Outputs[0].ElementCount)
        {
            // Safe two-step: copy a → output, then add b in-place.
            // Avoids 3-way aliasing (a, b, output may share same GPU buffer on WebGPU).
            reg.ElementWise.Scale(a.Data, output.Data, a.ElementCount, 1f);
            reg.ElementWise.AddInPlace(output.Data, b.Data, a.ElementCount);
        }
        else if (b.ElementCount == 1 && a.ElementCount == output.ElementCount)
        {
            // Scalar broadcast. This has to come BEFORE any branch that reads a last dimension: a rank-0
            // tensor has no last dimension, and Shape[^1] on an empty shape throws rather than returning
            // anything. Not an edge case here - the flow-matching decoder feeds its timestep in as a true
            // 0-d tensor and adds it to a [1,384] embedding, which crashed the whole graph.
            reg.ElementWise.Scale(a.Data, output.Data, a.ElementCount, 1f);
            reg.ElementWise.AddBias(output.Data, b.Data, a.ElementCount, 1);
        }
        else if (a.ElementCount == 1 && b.ElementCount == output.ElementCount)
        {
            // Same case with the operands the other way round; addition commutes, so the scalar is
            // applied to a copy of b.
            reg.ElementWise.Scale(b.Data, output.Data, b.ElementCount, 1f);
            reg.ElementWise.AddBias(output.Data, a.Data, b.ElementCount, 1);
        }
        else if (b.ElementCount == BroadcastShapeHelpers.LastDim(a) && b.Shape.Length > 0 && b.Shape[^1] == b.ElementCount)
        {
            // Last-dim broadcast: copy a → output, then AddBias in-place. The second guard (all of b's
            // elements in its LAST dim) keeps a per-channel bias shaped [C,1,1] — which also satisfies
            // ElementCount==a.Shape[^1] when C==W (SD-VAE GroupNorm β [256,1,1]) — from being applied per-W;
            // it falls through to the general N-D broadcast (correct per-channel). See MulOperator for detail.
            reg.ElementWise.Scale(a.Data, output.Data, a.ElementCount, 1f);
            reg.ElementWise.AddBias(output.Data, b.Data, a.ElementCount, b.ElementCount);
        }
        else if (a.Rank == 4 && b.Rank == 1 && b.ElementCount == a.Shape[1])
        {
            // NCHW per-channel broadcast: a[N,C,H,W] + b[C]
            // AddBias broadcasts over the last dim. For NCHW we need per-channel.
            // Reshape conceptually: each C-channel has H*W elements
            int C = a.Shape[1]; int spatial = a.Shape[2] * a.Shape[3];
            reg.ElementWise.Scale(a.Data, ctx.Outputs[0].Data, a.ElementCount, 1f);
            // Use BroadcastMul pattern but for Add — need a per-channel add kernel
            // For now, iterate channels on CPU dispatch (each channel gets AddBias)
            for (int nc = 0; nc < a.Shape[0] * C; nc++)
            {
                int c = nc % C;
                int offset = nc * spatial;
                // Add scalar bias[c] to each element in this channel's spatial slice
                // We don't have a scalar-add kernel, so use AddBias with spatial=1 trick
                // Actually, just use Scale(1) + AddBias over the spatial dim
                reg.ElementWise.AddBias(
                    ctx.Outputs[0].Data.SubView(offset, spatial),
                    b.Data.SubView(c, 1), spatial, 1);
            }
        }
        else if (b.ElementCount == 1)
        {
            // Scalar broadcast
            reg.ElementWise.Scale(a.Data, ctx.Outputs[0].Data, a.ElementCount, 1f);
            reg.ElementWise.AddBias(ctx.Outputs[0].Data, b.Data, a.ElementCount, 1);
        }
        else
        {
            // General broadcast: try CPU fallback for small tensors (shape computation)
            BroadcastBinaryOp(ctx, reg, (x, y) => x + y, BroadcastOp.Add);
        }
    }
}

public class MulOperator(OperatorRegistry reg) : IOnnxOperator, IPrecisionAwareOperator
{
    public string OpType => "Mul";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public bool TryExecuteHalf(OnnxOpContext ctx, PrecisionAwareInput[] inputs, Tensors.HalfTensor output, Kernels.PrecisionAwareKernels pak)
    {
        // Only the elementwise, no-broadcast, both-low-p gate (e.g. SiLU = x·sigmoid(x)). Broadcast / fp32 → fallback.
        if (inputs.Length < 2 || !inputs[0].IsHalf || !inputs[1].IsHalf) return false;
        if (inputs[0].ElementCount != output.ElementCount || inputs[1].ElementCount != output.ElementCount) return false;
        pak.Mul<global::ILGPU.Half>(inputs[0].Half!.Data, inputs[1].Half!.Data, output.Data, output.ElementCount);
        return true;
    }
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount && a.ElementCount == ctx.Outputs[0].ElementCount)
        {
            // De-alias if both operands are the same tensor (x * x): WebGPU/WebGL forbid binding one
            // buffer to two storage slots. (Shared helper — same guard Sub/Div use.)
            var bb = DeAliasSecondOperand(ctx, a, b, out var rented);
            reg.ElementWise.Mul(a.Data, bb.Data, ctx.Outputs[0].Data, a.ElementCount);
            if (rented != null) ctx.Pool.Return(rented);
        }
        else if (b.ElementCount == 1 && a.ElementCount == ctx.Outputs[0].ElementCount)
        {
            // Scalar broadcast, before any last-dim branch - see AddOperator for why a rank-0 operand
            // cannot be allowed to reach Shape[^1].
            reg.ElementWise.BroadcastMul(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount, 1);
        }
        else if (a.ElementCount == 1 && b.ElementCount == ctx.Outputs[0].ElementCount)
        {
            // Multiplication commutes, so the scalar operand can be the broadcast one.
            reg.ElementWise.BroadcastMul(b.Data, a.Data, ctx.Outputs[0].Data, b.ElementCount, 1);
        }
        else if (b.ElementCount == BroadcastShapeHelpers.LastDim(a) && b.Shape.Length > 0 && b.Shape[^1] == b.ElementCount)
        {
            // Last-dim broadcast: a[..., C] * b[C]. The second guard (all of b's elements in its LAST dim)
            // is essential: a per-channel weight shaped [C,1,1] also has ElementCount==a.Shape[^1] when C==W
            // (e.g. SD-VAE GroupNorm γ [256,1,1] on a [1,256,256,256] map), but it must broadcast over the
            // CHANNEL axis, not the last (W) axis. Such [C,1,1] tensors fall through to the general N-D
            // broadcast below (which maps strides per-axis correctly). Without this, γ/β were applied per-W.
            reg.ElementWise.BroadcastMul(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount, b.ElementCount);
        }
        else if (b.ElementCount == 1)
        {
            // Scalar broadcast — need to read the scalar value
            // For now, use BroadcastMul with C=1
            reg.ElementWise.BroadcastMul(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount, 1);
        }
        else if (a.Rank == 4 && b.Rank == 1 && b.ElementCount == a.Shape[1])
        {
            // NCHW per-channel: a[N,C,H,W] * b[C]
            int C = a.Shape[1]; int spatial = a.Shape[2] * a.Shape[3];
            for (int nc = 0; nc < a.Shape[0] * C; nc++)
            {
                int c = nc % C;
                int offset = nc * spatial;
                reg.ElementWise.BroadcastMul(
                    a.Data.SubView(offset, spatial),
                    b.Data.SubView(c, 1),
                    ctx.Outputs[0].Data.SubView(offset, spatial),
                    spatial, 1);
            }
        }
        else
        {
            BroadcastBinaryOp(ctx, reg, (x, y) => x * y, BroadcastOp.Mul);
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
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount && a.ElementCount == ctx.Outputs[0].ElementCount)
        {
            // Single-dispatch subtract. De-alias if both operands are the same tensor (e.g. an ONNX
            // graph Sub fed x to both inputs → x - x): WebGPU/WebGL reject binding one buffer to two
            // storage slots. (This is the SD-Turbo UNet aliasing crash — node 'Sub' with identical inputs.)
            var bb = DeAliasSecondOperand(ctx, a, b, out var rented);
            reg.ElementWise.Sub(a.Data, bb.Data, ctx.Outputs[0].Data, a.ElementCount);
            if (rented != null) ctx.Pool.Return(rented);
        }
        else if (a.ElementCount == b.ElementCount)
        {
            // Size mismatch with output — use min count
            int count = Math.Min(a.ElementCount, ctx.Outputs[0].ElementCount);
            reg.ElementWise.Sub(a.Data.SubView(0, count), b.Data.SubView(0, count),
                ctx.Outputs[0].Data.SubView(0, count), count);
        }
        else if (b.ElementCount == 1)
        {
            // Scalar subtract: output = a - scalar → use BroadcastMul(a, -1→b) + BroadcastAdd
            // Strategy: output = a (copy), then BroadcastSub via BroadcastBinaryOp
            // Simplest safe path: use BroadcastBinaryOp which handles all broadcast shapes
            BroadcastBinaryOp(ctx, reg, (x, y) => x - y, BroadcastOp.Sub);
        }
        else
        {
            BroadcastBinaryOp(ctx, reg, (x, y) => x - y, BroadcastOp.Sub);
        }
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
        if (a.ElementCount == b.ElementCount && a.ElementCount == ctx.Outputs[0].ElementCount)
        {
            // De-alias if both operands are the same tensor (x / x): WebGPU/WebGL reject binding one
            // buffer to two storage slots (same class of crash as the SD-Turbo Sub aliasing).
            var bb = DeAliasSecondOperand(ctx, a, b, out var rented);
            reg.ElementWise.Div(a.Data, bb.Data, ctx.Outputs[0].Data, a.ElementCount);
            if (rented != null) ctx.Pool.Return(rented);
        }
        else if (b.ElementCount == 1)
        {
            // Scalar div: compute reciprocal of scalar, then multiply
            var recip = ctx.Pool.Rent(b.Shape, "div_recip");
            reg.ElementWise.Reciprocal(b.Data, recip.Data, 1);
            reg.ElementWise.BroadcastMul(a.Data, recip.Data, ctx.Outputs[0].Data, a.ElementCount, 1);
            ctx.Pool.Return(recip);
        }
        else if (a.ElementCount == 1 && b.ElementCount > 1)
        {
            // Scalar NUMERATOR. Division does not commute, so this cannot reuse the reciprocal trick
            // above; it goes through the general broadcast. Placed before the last-dim branch because a
            // rank-0 numerator has no last dimension to read.
            BroadcastBinaryOp(ctx, reg, (x, y) => y != 0 ? x / y : 0f, BroadcastOp.Div);
        }
        else if (b.ElementCount == BroadcastShapeHelpers.LastDim(a))
        {
            // Last-dim broadcast: a / b where b is [C]. Compute reciprocal then BroadcastMul
            var recip = ctx.Pool.Rent(b.Shape, "div_recip_bc");
            reg.ElementWise.Reciprocal(b.Data, recip.Data, b.ElementCount);
            reg.ElementWise.BroadcastMul(a.Data, recip.Data, ctx.Outputs[0].Data, a.ElementCount, b.ElementCount);
            ctx.Pool.Return(recip);
        }
        else if (a.Shape.Length >= 2 && b.ElementCount > 1 && b.ElementCount < a.ElementCount)
        {
            // General broadcast: compute reciprocal of b, then use BroadcastBinaryOp for multiply
            // This handles cases like a=[1,257,384] / b=[1,257,1] (per-row scalar division)
            BroadcastBinaryOp(ctx, reg, (x, y) => y != 0 ? x / y : 0f, BroadcastOp.Div);
        }
        else
        {
            BroadcastBinaryOp(ctx, reg, (x, y) => y != 0 ? x / y : 0f, BroadcastOp.Div);
        }

        // ONNX spec: Div on integer dtypes truncates toward zero (C-style).
        // TF's tf.floordiv exports as Cast(int)+Div(int,int); without this trunc step
        // 887/48 reads as 18.479 instead of 18, breaking downstream (argmax mod 48)
        // keypoint X-coord decode in MoveNet by a uniform ~0.4 per keypoint.
        // In-place trunc keeps it to one extra single-binding dispatch (WebGPU-safe)
        // and only when the dtype actually demands it.
        if (ctx.AllInputsAreInteger())
        {
            int outCount = ctx.Outputs[0].ElementCount;
            reg.ElementWise.TruncateInPlace(ctx.Outputs[0].Data, outCount);
            System.Threading.Interlocked.Increment(ref Graph.GraphExecutor.LastRunIntegerDivCount);
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
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount && a.ElementCount == ctx.Outputs[0].ElementCount)
        {
            reg.ElementWise.Pow(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
        }
        else if (b.ElementCount <= a.ElementCount)
        {
            // Scalar/small exponent broadcast (LayerNorm x^2, InstanceNorm x^2).
            // Expand exponent to full size on CPU, then element-wise Pow.
            // Avoids BroadcastBinaryOpND's synchronous Synchronize()+readback path, which is unsafe on
            // browser: a sync Synchronize() only flushes (dispatches) without awaiting (NOT a deadlock),
            // so a following synchronous readback reads stale/not-yet-computed data.
            var bVals = ctx.TryGetInputValues(1);
            if (bVals != null)
            {
                int outCount = ctx.Outputs[0].ElementCount;
                var expanded = new float[outCount];
                for (int i = 0; i < outCount; i++)
                    expanded[i] = bVals[i % bVals.Length];
                Tensor expandedTensor;
                if (SpawnDev.ILGPU.ML.Graph.GraphExecutor.UseCaptureParamSlots)
                {
                    // CUDA-graph capture: the exponent is constant for a fixed input shape, but "_pow_exp" is a
                    // transient reused pool buffer — skipping its H2D would leave STALE data. Use a stable arena
                    // slot (deterministic k-th float rent → this Pow's exponent), written in warm, skip-write in
                    // capture. Not pool-registered, so the Pool.Return below is a safe no-op.
                    var expView = Kernels.CaptureParamArena.Shared(reg.Accelerator).RentStableSlotFloat(expanded);
                    expandedTensor = new Tensor(expView, ctx.Outputs[0].Shape, "_pow_exp");
                }
                else
                {
                    expandedTensor = ctx.Pool.Rent(ctx.Outputs[0].Shape, "_pow_exp");
                    expandedTensor.Data.SubView(0, outCount).CopyFromCPU(expanded);
                }
                reg.ElementWise.Pow(a.Data, expandedTensor.Data, ctx.Outputs[0].Data, outCount);
                ctx.Pool.Return(expandedTensor);
            }
            else
            {
                // Exponent not in runtime constants — use BroadcastBinaryOp (desktop backends only)
                BroadcastBinaryOp(ctx, reg, (x, y) => MathF.Pow(x, y), BroadcastOp.Pow);
            }
        }
        else
        {
            BroadcastBinaryOp(ctx, reg, (x, y) => MathF.Pow(x, y), BroadcastOp.Pow);
        }
    }
}

public class NotOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Not";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // Logical NOT: output[i] = (input[i] == 0) ? 1 : 0
        // Use pre-read constant values for CPU path (avoids aliasing issues)
        var inVals = ctx.TryGetInputValues(0);
        if (inVals != null)
        {
            var result = new float[inVals.Length];
            for (int i = 0; i < inVals.Length; i++)
                result[i] = inVals[i] == 0f ? 1f : 0f;
            // Direct CPU->GPU upload (was AllocatePermanent + Scale leak).
            ctx.Outputs[0].Data.SubView(0, result.Length).CopyFromCPU(result);
        }
        else
        {
            // GPU path: fill temp with 1, then Sub(ones, input, output)
            int count = ctx.Inputs[0].ElementCount;
            var ones = ctx.Pool.Rent(ctx.Inputs[0].Shape, "_not_ones");
            reg.ElementWise.Fill(ones.Data, count, 1f);
            reg.ElementWise.Sub(ones.Data, ctx.Inputs[0].Data, ctx.Outputs[0].Data, count);
            ctx.Pool.Return(ones);
        }
    }
}

public class ConstantOfShapeOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ConstantOfShape";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // The output shape is the input tensor's VALUES. The compiler injects them as _resolved_shape
        // whenever it can fold them; without that, the best available guess is the input's own shape,
        // which is right only by coincidence.
        if (attrs.TryGetValue("_resolved_shape", out var resolved) && resolved is long[] dims && dims.Length > 0)
        {
            var shape = new int[dims.Length];
            for (int k = 0; k < dims.Length; k++) shape[k] = (int)dims[k];
            return new[] { shape };
        }
        return new[] { inputs[0] };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // ONNX spec: value attribute is a scalar tensor (default 0.0)
        float fillValue = 0f;
        if (ctx.Attributes.TryGetValue("value", out var val))
        {
            fillValue = val switch
            {
                float f => f,
                double d => (float)d,
                long l => (float)l,
                int i => (float)i,
                _ => 0f
            };
        }
        reg.ElementWise.Fill(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount, fillValue);
    }
}

public class RangeOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Range";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { new[] { 1 } }; // Dynamic — resolved at runtime from scalar inputs
    public void Execute(OnnxOpContext ctx)
    {
        // Range(start, limit, delta) → [start, start+delta, ..., <limit)
        // Inputs are scalar tensors — read from runtime constants
        var startVals = ctx.TryGetInputValues(0);
        var limitVals = ctx.TryGetInputValues(1);
        var deltaVals = ctx.TryGetInputValues(2);

        if (startVals == null || limitVals == null || deltaVals == null
            || startVals.Length == 0 || limitVals.Length == 0 || deltaVals.Length == 0)
        {
            // Mid-capture the scalar runtime constants can be UNAVAILABLE/EMPTY (readbacks
            // suppressed) - startVals[0] threw IndexOutOfRange under WebGPU capture (SD-Turbo CLIP,
            // 2026-07-03). Range is deterministic per shape and the output buffer already holds the
            // warm-pass value; the H2D upload below is capture-skipped anyway - nothing to do.
            if (SpawnDev.ILGPU.ML.Graph.GraphExecutor.SuppressDrains) return;
            static string D(float[]? v) => v == null ? "null" : v.Length == 0 ? "EMPTY" : $"[{v[0]}]";
            throw new NotSupportedException(
                $"Range: scalar inputs not available as runtime constants: "
                + $"start('{(ctx.InputNames.Length > 0 ? ctx.InputNames[0] : "?")}')={D(startVals)} "
                + $"limit('{(ctx.InputNames.Length > 1 ? ctx.InputNames[1] : "?")}')={D(limitVals)} "
                + $"delta('{(ctx.InputNames.Length > 2 ? ctx.InputNames[2] : "?")}')={D(deltaVals)} "
                + $"elide={SpawnDev.ILGPU.ML.Graph.GraphExecutor.ShapeInterpElideDispatch}");
        }

        float start = startVals[0];
        float limit = limitVals[0];
        float delta = deltaVals[0];

        if (delta == 0) throw new ArgumentException("Range: delta cannot be 0");

        int count = Math.Max(0, (int)MathF.Ceiling((limit - start) / delta));
        var data = new float[count];
        for (int i = 0; i < count; i++)
            data[i] = start + i * delta;

        // Upload to output GPU buffer. During CUDA-graph capture a synchronous CopyFromCPU (H2D) is illegal;
        // Range is deterministic for a fixed input shape and the pool is deterministic, so the buffer already
        // holds this value from the warm pass — skip the re-upload during the capture pass.
        var output = ctx.Outputs[0];
        if (output.ElementCount >= count && !SpawnDev.ILGPU.ML.Graph.GraphExecutor.SuppressDrains)
        {
            output.Data.SubView(0, count).CopyFromCPU(data);
        }
    }
}

/// <summary>
/// NonZero: returns indices of non-zero elements as [rank, nnz] tensor.
/// For attention masks (all 1s), returns all coordinate pairs.
/// Data-dependent output size — reads input values from runtime constants.
/// </summary>
public class NonZeroOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "NonZero";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { new[] { inputs[0].Length, inputs[0].Aggregate(1, (a, b) => a * b) } }; // [rank, max_nnz]
    public void Execute(OnnxOpContext ctx)
    {
        var input = ctx.Inputs[0];
        var inShape = input.Shape;
        int rank = inShape.Length;
        int totalElems = input.ElementCount;

        // Read input values — NonZero is inherently data-dependent
        var vals = ctx.TryGetInputValues(0);
        if (vals == null)
        {
            // Can't read values — assume all non-zero (common for attention masks)
            vals = new float[totalElems];
            for (int i = 0; i < totalElems; i++) vals[i] = 1f;
        }

        // Find non-zero indices
        var indices = new List<int[]>();
        var strides = new int[rank];
        if (rank > 0) strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; d--) strides[d] = strides[d + 1] * inShape[d + 1];

        for (int i = 0; i < totalElems; i++)
        {
            if (vals[i] != 0f)
            {
                var coord = new int[rank];
                int rem = i;
                for (int d = 0; d < rank; d++) { coord[d] = rem / strides[d]; rem %= strides[d]; }
                indices.Add(coord);
            }
        }

        // Output: [rank, nnz] — each row is one dimension's indices
        int nnz = indices.Count;
        var result = new float[rank * nnz];
        for (int d = 0; d < rank; d++)
            for (int j = 0; j < nnz; j++)
                result[d * nnz + j] = indices[j][d];

        var output = ctx.Outputs[0];
        int copyLen = Math.Min(result.Length, output.ElementCount);
        if (copyLen > 0)
        {
            output.Data.SubView(0, copyLen).CopyFromCPU(result.AsSpan(0, copyLen).ToArray());
        }
    }
}

public class WhereOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Where";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // ONNX Where: output = broadcast(condition, x, y)
        // Return the largest shape among the three inputs
        var best = inputs[0];
        for (int i = 1; i < inputs.Length; i++)
            if (inputs[i].Length > best.Length || (inputs[i].Length == best.Length
                && inputs[i].Aggregate(1, (a, b) => a * b) > best.Aggregate(1, (a, b) => a * b)))
                best = inputs[i];
        return new[] { best };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var cond = ctx.Inputs[0]; var x = ctx.Inputs[1]; var y = ctx.Inputs[2];
        if (cond.ElementCount == x.ElementCount && x.ElementCount == y.ElementCount)
        {
            reg.ElementWise.Where(cond.Data, x.Data, y.Data, ctx.Outputs[0].Data, x.ElementCount);
        }
        else
        {
            // Broadcasting required — use stride-based N-D broadcast mapping
            var cVals = ctx.TryGetInputValues(0);
            var xVals = ctx.TryGetInputValues(1);
            var yVals = ctx.TryGetInputValues(2);
            int outCount = ctx.Outputs[0].ElementCount;
            if (cVals != null && xVals != null && yVals != null)
            {
                var outShape = ctx.Outputs[0].Shape;
                int rank = outShape.Length;

                // Compute broadcast strides for each input
                static int[] ComputeStrides(int[] shape, int[] outShape)
                {
                    int rank = outShape.Length;
                    int padded = rank - shape.Length;
                    var strides = new int[rank];
                    int stride = 1;
                    for (int d = rank - 1; d >= 0; d--)
                    {
                        int dim = d - padded >= 0 ? shape[d - padded] : 1;
                        strides[d] = dim == 1 ? 0 : stride; // broadcast dim → stride 0
                        stride *= dim;
                    }
                    return strides;
                }

                var cStrides = ComputeStrides(cond.Shape, outShape);
                var xStrides = ComputeStrides(x.Shape, outShape);
                var yStrides = ComputeStrides(y.Shape, outShape);

                // Compute output strides
                var outStrides = new int[rank];
                int oStride = 1;
                for (int d = rank - 1; d >= 0; d--) { outStrides[d] = oStride; oStride *= outShape[d]; }

                var result = new float[outCount];
                for (int i = 0; i < outCount; i++)
                {
                    // Decompose flat index into N-D coordinates, map to each input
                    int cIdx = 0, xIdx = 0, yIdx = 0, rem = i;
                    for (int d = 0; d < rank; d++)
                    {
                        int coord = rem / outStrides[d];
                        rem %= outStrides[d];
                        cIdx += coord * cStrides[d];
                        xIdx += coord * xStrides[d];
                        yIdx += coord * yStrides[d];
                    }
                    result[i] = cVals[cIdx] != 0f ? xVals[xIdx] : yVals[yIdx];
                }
                // Direct CPU->GPU upload (was AllocatePermanent + Scale leak).
                ctx.Outputs[0].Data.SubView(0, outCount).CopyFromCPU(result);
            }
            else
            {
                // Not all three inputs are small CPU constants — the common case is the
                // causal-mask Where, where x = attention scores [1, heads, seq, seq] is a
                // runtime GPU tensor (so xVals == null), cond = mask [1, 1, seq, seq], and
                // y = a scalar mask_value. Do a real stride-based N-D broadcast on the GPU.
                //
                // The previous fallback called ElementWise.Where over
                // min(elementCounts) elements — which is 1 when y is a scalar — writing only
                // output[0] and leaving the rest as stale pooled-buffer garbage. That silently
                // destroyed the causal mask in EVERY decoder layer (GPT-2/DistilGPT-2: flat,
                // input-echoing logits), and corrupted any broadcasting Where on a runtime tensor.
                reg.ElementWise.WhereBroadcastND(
                    cond.Data, x.Data, y.Data, ctx.Outputs[0].Data,
                    cond.Shape, x.Shape, y.Shape, ctx.Outputs[0].Shape);
            }
        }
    }
}

public class ExpandOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Expand";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] }; // Dynamic — resolved at graph compile time from shape input
    public void Execute(OnnxOpContext ctx)
    {
        var input = ctx.Inputs[0];
        var output = ctx.Outputs[0];
        int outCount = output.ElementCount;

        // Simple case: same element count — just copy
        if (input.ElementCount == outCount)
        {
            reg.ElementWise.Scale(input.Data, output.Data, outCount, 1f);
            return;
        }

        // N-D broadcasting: use pre-read constant values
        var inVals = ctx.TryGetInputValues(0);
        if (inVals != null)
        {
            var inStrides = ComputeStrides(input.Shape, output.Shape);
            var outStrides = ComputeStrides(output.Shape, output.Shape);
            var result = new float[outCount];
            for (int i = 0; i < outCount; i++)
            {
                int inIdx = MapIndex(i, outStrides, inStrides, output.Shape.Length);
                result[i] = inIdx < inVals.Length ? inVals[inIdx] : 0f;
            }
            // Direct CPU->GPU upload to output. Was: AllocatePermanent + Scale, which
            // leaked a fresh GPU buffer per call (permanent buffers live until session
            // dispose). DA3-Small inference exhausted Wasm 4GB memory on its 5+ Expand
            // ops with this pattern. CopyFromCPU is the same GPU-resident write path
            // (queue.writeBuffer on WebGPU, equivalent on other backends), no temp.
            // Under CUDA-graph capture: arena stable-slot + captured GPU CopyFrom so replay reproduces the write.
            if (SpawnDev.ILGPU.ML.Graph.GraphExecutor.UseCaptureParamSlots)
                Kernels.CaptureParamArena.CaptureConstWrite(reg.Accelerator, output.Data.SubView(0, outCount), result);
            else
                output.Data.SubView(0, outCount).CopyFromCPU(result);
        }
        else
        {
            // GPU broadcast: use the dedicated Expand kernel - one parallel
            // dispatch with packed broadcast strides instead of repeated
            // per-row Scale calls. The prior per-row loop dispatched
            // Scale `repeats` times (1 per output row), and on Wasm each
            // dispatch pays worker-pool round-trip overhead. DA3-Small DPT
            // head Expand nodes (1432_Expand_5, 1013_Expand_2) accumulated
            // 59s + 44s respectively in the diagnostic 2026-05-05 -
            // single-dispatch fix is ~hundreds-of-x speedup.
            int inCount = input.ElementCount;
            if (inCount > 0)
            {
                reg.MissingElementWise.Expand(input.Data, output.Data, input.Shape, output.Shape);
            }
            else
            {
                // Empty input fallback: leave output zero-filled (matches existing semantics for
                // the previous outCount=0/inCount=0 corner cases).
            }
        }
    }
}

public class EqualOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Equal";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount && a.ElementCount == ctx.Outputs[0].ElementCount)
            reg.ElementWise.Equal(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
        else
            BroadcastBinaryOp(ctx, reg, (x, y) => x == y ? 1f : 0f, BroadcastOp.Equal);
    }
}

public class GreaterOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Greater";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount && a.ElementCount == ctx.Outputs[0].ElementCount)
            reg.ElementWise.Greater(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
        else
            BroadcastBinaryOp(ctx, reg, (x, y) => x > y ? 1f : 0f, BroadcastOp.Greater);
    }
}

public class LessOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Less";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount && a.ElementCount == ctx.Outputs[0].ElementCount)
            reg.ElementWise.Less(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
        else
            BroadcastBinaryOp(ctx, reg, (x, y) => x < y ? 1f : 0f, BroadcastOp.Less);
    }
}

public class LessOrEqualOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "LessOrEqual";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount && a.ElementCount == ctx.Outputs[0].ElementCount)
        {
            // a <= b is !(a > b). Greater returns 1.0 for true, 0.0 for false.
            // Negate: output = 1.0 - Greater(a, b)
            reg.ElementWise.Greater(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
            reg.ElementWise.ScaleInPlace(ctx.Outputs[0].Data, a.ElementCount, -1f);
            // +1 as a kernel ARGUMENT, not via a rented buffer and a per-call host->device copy. The old
            // form issued one synchronous H2D transfer to deliver a single float, which on CUDA implicitly
            // synchronises the stream and is ILLEGAL mid-graph-capture (it aborted the ZipVoice decoder
            // capture with "operation not permitted when stream is capturing").
            reg.ElementWise.AddScalarInPlace(ctx.Outputs[0].Data, a.ElementCount, 1f);
        }
        else
        {
            BroadcastBinaryOp(ctx, reg, (x, y) => x <= y ? 1f : 0f, BroadcastOp.LessOrEqual);
        }
    }
}

public class GreaterOrEqualOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "GreaterOrEqual";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount && a.ElementCount == ctx.Outputs[0].ElementCount)
        {
            // a >= b is !(a < b)
            reg.ElementWise.Less(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
            reg.ElementWise.ScaleInPlace(ctx.Outputs[0].Data, a.ElementCount, -1f);
            // +1 as a kernel ARGUMENT, not via a rented buffer and a per-call host->device copy. The old
            // form issued one synchronous H2D transfer to deliver a single float, which on CUDA implicitly
            // synchronises the stream and is ILLEGAL mid-graph-capture (it aborted the ZipVoice decoder
            // capture with "operation not permitted when stream is capturing").
            reg.ElementWise.AddScalarInPlace(ctx.Outputs[0].Data, a.ElementCount, 1f);
        }
        else
        {
            BroadcastBinaryOp(ctx, reg, (x, y) => x >= y ? 1f : 0f, BroadcastOp.GreaterOrEqual);
        }
    }
}

public class OrOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Or";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        // Or is always a boolean op — use broadcast path for correctness
        // (handles both equal and unequal shapes, with proper abs+threshold)
        BroadcastBinaryOp(ctx, reg, (x, y) => (x != 0f || y != 0f) ? 1f : 0f, BroadcastOp.Or);
    }
}

public class XorOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Xor";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        BroadcastBinaryOp(ctx, reg, (x, y) => (x != 0f) != (y != 0f) ? 1f : 0f, BroadcastOp.Xor);
    }
}

public class AndOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "And";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { Tensors.TensorHelpers.BroadcastShape(inputs[0], inputs[1]) };
    public void Execute(OnnxOpContext ctx)
    {
        var a = ctx.Inputs[0]; var b = ctx.Inputs[1];
        if (a.ElementCount == b.ElementCount && a.ElementCount == ctx.Outputs[0].ElementCount)
        {
            // And = Mul(a, b) then threshold: any non-zero × non-zero = non-zero
            reg.ElementWise.Mul(a.Data, b.Data, ctx.Outputs[0].Data, a.ElementCount);
        }
        else
        {
            BroadcastBinaryOp(ctx, reg, (x, y) => (x != 0f && y != 0f) ? 1f : 0f, BroadcastOp.And);
        }
    }
}

public class IsNaNOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "IsNaN";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        int count = ctx.Inputs[0].ElementCount;
        reg.ElementWise.IsNaN(ctx.Inputs[0].Data, ctx.Outputs[0].Data, count);
    }
}

public class HardSigmoidOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "HardSigmoid";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // ONNX spec defaults: alpha=0.2, beta=0.5
        float alpha = ctx.GetFloat("alpha", 0.2f);
        float beta = ctx.GetFloat("beta", 0.5f);
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, 1f);
        reg.Activations.HardSigmoidInPlace(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount, alpha, beta);
    }
}

public class HardSwishOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "HardSwish";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount, 1f);
        reg.Activations.HardSwishInPlace(ctx.Outputs[0].Data, ctx.Outputs[0].ElementCount);
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

public class SinOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Sin";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Sin(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class CosOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Cos";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Cos(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

public class TanOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Tan";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        reg.ElementWise.Tan(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
    }
}

// ═══════════════════════════════════════════════════════════
//  New operators — full ONNX coverage
// ═══════════════════════════════════════════════════════════

public class AcosOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Acos";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Acos(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class AcoshOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Acosh";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Acosh(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class AsinOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Asin";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Asin(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class AsinhOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Asinh";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Asinh(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class AtanOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Atan";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Atan(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class AtanhOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Atanh";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Atanh(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class CoshOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Cosh";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Cosh(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class SinhOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Sinh";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Sinh(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class EluOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Elu";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Elu(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class CeluOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Celu";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Celu(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class SeluOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Selu";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Selu(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class SoftplusOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Softplus";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Softplus(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class SoftsignOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Softsign";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Softsign(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class MishOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Mish";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Mish(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class IsInfOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "IsInf";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.IsInf(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class ThresholdedReluOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ThresholdedRelu";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        float alpha = ctx.GetFloat("alpha", 1f);
        int count = ctx.Inputs[0].ElementCount;
        reg.ElementWise.ThresholdedRelu(ctx.Inputs[0].Data, ctx.Outputs[0].Data, count, alpha);
    }
}

public class IdentityOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Identity";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // Use CopyFrom (GPU→GPU works on all backends). NOT CopyTo — WebGPU's
        // CopyTo override throws because it can't distinguish GPU→GPU from GPU→CPU.
        int count = ctx.Inputs[0].ElementCount;
        ctx.Outputs[0].Data.SubView(0, count).CopyFrom(ctx.Inputs[0].Data.SubView(0, count));
    }
}

public class SizeOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Size";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { new[] { 1 } };
    public void Execute(OnnxOpContext ctx)
    {
        int size = ctx.Inputs[0].ElementCount;
        ctx.Outputs[0].Data.SubView(0, 1).CopyFromCPU(new float[] { size });
    }
}

public class HardmaxOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Hardmax";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // Hardmax: output = one-hot of argmax along axis
        int axis = ctx.GetInt("axis", -1);
        var shape = ctx.Inputs[0].Shape;
        if (axis < 0) axis += shape.Length;
        int outer = 1, inner = 1, axisSize = shape[axis];
        for (int i = 0; i < axis; i++) outer *= shape[i];
        for (int i = axis + 1; i < shape.Length; i++) inner *= shape[i];
        int total = ctx.Outputs[0].ElementCount;

        if (inner == 1)
        {
            // GPU path: one thread per output element — gather, WebGL TF compatible
            reg.ElementWise.Hardmax(ctx.Inputs[0].Data, ctx.Outputs[0].Data, outer * axisSize, axisSize);
        }
        else
        {
            // General case with inner dims — CPU fallback
            reg.ElementWise.Fill(ctx.Outputs[0].Data, total, 0f);
            var xVals = ctx.TryGetInputValues(0);
            if (xVals != null)
            {
                var result = new float[total];
                for (int o = 0; o < outer; o++)
                    for (int inn = 0; inn < inner; inn++)
                    {
                        float maxVal = float.NegativeInfinity;
                        int maxIdx = 0;
                        for (int a = 0; a < axisSize; a++)
                        {
                            float v = xVals[(o * axisSize + a) * inner + inn];
                            if (v > maxVal) { maxVal = v; maxIdx = a; }
                        }
                        result[(o * axisSize + maxIdx) * inner + inn] = 1f;
                    }
                ctx.Outputs[0].Data.SubView(0, total).CopyFromCPU(result);
            }
        }
    }
}

public class LogSoftmaxOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "LogSoftmax";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        int axis = ctx.GetInt("axis", -1);
        var shape = ctx.Inputs[0].Shape;
        if (axis < 0) axis += shape.Length;
        int rows = 1, cols = shape[axis];
        for (int i = 0; i < axis; i++) rows *= shape[i];
        // Copy input to output, run softmax, then log (using temp to avoid aliasing)
        int total = ctx.Inputs[0].ElementCount;
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, total, 1f);
        reg.Softmax.Forward(ctx.Outputs[0].Data, rows, cols);
        var tempBuf = ctx.Pool.Rent(new[] { total });
        reg.ElementWise.Log(ctx.Outputs[0].Data, tempBuf.Data, total);
        reg.ElementWise.Scale(tempBuf.Data, ctx.Outputs[0].Data, total, 1f);
    }
}

public class PReluOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "PRelu";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // PRelu: output = x if x >= 0, slope * x if x < 0
        // slope may be per-channel broadcast
        var x = ctx.Inputs[0]; var slope = ctx.Inputs[1];
        // Use broadcast binary op: PRelu(x, slope) = max(0, x) + slope * min(0, x)
        // Simplified: use Where(x >= 0, x, slope * x) via broadcast
        BroadcastHelper.BroadcastBinaryOp(ctx, reg,
            (a, b) => a >= 0f ? a : a * b, BroadcastOp.PRelu);
    }
}

public class SumOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Sum";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        int count = ctx.Outputs[0].ElementCount;
        // Copy first input to output
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, count, 1f);
        // Add remaining inputs using temp buffer to avoid aliasing (output as both input and output)
        if (ctx.Inputs.Length > 1)
        {
            var tempBuf = ctx.Pool.Rent(new[] { count });
            for (int i = 1; i < ctx.Inputs.Length; i++)
            {
                // ⚠️ `Add` indexes BOTH operands by the dispatch index, so an input SMALLER than the output
                // is read out of bounds - and ONNX Sum/Mean are variadic with multidirectional
                // broadcasting, so a smaller input is legal. A scalar is the case that actually occurs and
                // AddBias broadcasts it correctly; anything else is refused rather than silently read past
                // the end of the buffer (on a GPU backend that is a silent wrong answer, not a crash).
                int inCount = ctx.Inputs[i].ElementCount;
                if (inCount == count)
                {
                    reg.ElementWise.Add(ctx.Outputs[0].Data, ctx.Inputs[i].Data, tempBuf.Data, count);
                    reg.ElementWise.Scale(tempBuf.Data, ctx.Outputs[0].Data, count, 1f);
                }
                else if (inCount == 1)
                {
                    reg.ElementWise.AddBias(ctx.Outputs[0].Data, ctx.Inputs[i].Data, count, 1);
                }
                else
                {
                    ctx.Pool.Return(tempBuf);
                    throw new NotSupportedException(
                        $"{OpType}: input {i} has {inCount} elements against an output of {count}. Only " +
                        "equal-size or scalar inputs are supported; general multidirectional broadcasting " +
                        "across variadic inputs is not implemented.");
                }
            }
        }
    }
}

public class MeanOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Mean";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        int count = ctx.Outputs[0].ElementCount;
        reg.ElementWise.Scale(ctx.Inputs[0].Data, ctx.Outputs[0].Data, count, 1f);
        if (ctx.Inputs.Length > 1)
        {
            var tempBuf = ctx.Pool.Rent(new[] { count });
            for (int i = 1; i < ctx.Inputs.Length; i++)
            {
                // ⚠️ `Add` indexes BOTH operands by the dispatch index, so an input SMALLER than the output
                // is read out of bounds - and ONNX Sum/Mean are variadic with multidirectional
                // broadcasting, so a smaller input is legal. A scalar is the case that actually occurs and
                // AddBias broadcasts it correctly; anything else is refused rather than silently read past
                // the end of the buffer (on a GPU backend that is a silent wrong answer, not a crash).
                int inCount = ctx.Inputs[i].ElementCount;
                if (inCount == count)
                {
                    reg.ElementWise.Add(ctx.Outputs[0].Data, ctx.Inputs[i].Data, tempBuf.Data, count);
                    reg.ElementWise.Scale(tempBuf.Data, ctx.Outputs[0].Data, count, 1f);
                }
                else if (inCount == 1)
                {
                    reg.ElementWise.AddBias(ctx.Outputs[0].Data, ctx.Inputs[i].Data, count, 1);
                }
                else
                {
                    ctx.Pool.Return(tempBuf);
                    throw new NotSupportedException(
                        $"{OpType}: input {i} has {inCount} elements against an output of {count}. Only " +
                        "equal-size or scalar inputs are supported; general multidirectional broadcasting " +
                        "across variadic inputs is not implemented.");
                }
            }
        }
        reg.ElementWise.ScaleInPlace(ctx.Outputs[0].Data, count, 1f / ctx.Inputs.Length);
    }
}

public class ArgMinOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ArgMin";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var shape = inputs[0].ToArray();
        int axis = attrs.ContainsKey("axis") ? Convert.ToInt32(attrs["axis"]) : 0;
        if (axis < 0) axis += shape.Length;
        shape[axis] = 1;
        bool keepdims = !attrs.ContainsKey("keepdims") || Convert.ToInt32(attrs["keepdims"]) != 0;
        return new[] { keepdims ? shape : shape.Where((_, i) => i != axis).ToArray() };
    }
    public void Execute(OnnxOpContext ctx)
    {
        // ArgMin = Neg → ArgMax (negate, find max of negated = min of original)
        int count = ctx.Inputs[0].ElementCount;
        var negBuf = ctx.Pool.Rent(new[] { count });
        reg.ElementWise.Scale(ctx.Inputs[0].Data, negBuf.Data, count, -1f);
        int axis = ctx.GetInt("axis", 0);
        var shape = ctx.Inputs[0].Shape;
        if (axis < 0) axis += shape.Length;
        int outerSize = 1, innerSize = 1, axisSize = shape[axis];
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < shape.Length; i++) innerSize *= shape[i];
        reg.ElementWise.ArgMax(negBuf.Data, ctx.Outputs[0].Data, outerSize, axisSize, innerSize);
    }
}

public class RoundOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Round";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx) => reg.ElementWise.Round(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}

public class ShrinkOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Shrink";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        // Shrink: if x > lambd, y = x - bias; if x < -lambd, y = x + bias; else y = 0
        // Default: lambd=0.5, bias=0
        // Use generic unary for default params
        int count = ctx.Inputs[0].ElementCount;
        float lambd = ctx.GetFloat("lambd", 0.5f);
        float bias = ctx.GetFloat("bias", 0f);
        // For default params, use the built-in ShrinkOp kernel
        if (MathF.Abs(lambd - 0.5f) < 1e-7f && MathF.Abs(bias) < 1e-7f)
        {
            reg.ElementWise.UnaryOp(ctx.Inputs[0].Data, ctx.Outputs[0].Data, count,
                new DelegateSpecialization<Func<float, float>>(ElementWiseKernels.ShrinkOp));
        }
        else
        {
            // Parameterized shrink — GPU kernel with custom lambd/bias
            var fparamsData = new float[] { lambd, bias };
            var fparamsBuf = ctx.Pool.Rent(new[] { fparamsData.Length });
            fparamsBuf.Data.SubView(0, fparamsData.Length).CopyFromCPU(fparamsData);
            reg.ElementWise.ShrinkParam(ctx.Inputs[0].Data, ctx.Outputs[0].Data, fparamsBuf.Data, count);
        }
    }
}
