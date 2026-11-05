using SpawnDev.ILGPU.ML.Training;

namespace SpawnDev.ILGPU.ML.Operators;

// Gradient operators emitted by the graph autograd (Plans/training-autograd-lora-2026-11-04.md, M1.3). Each takes the forward
// op's inputs (and output where the math needs it) plus the incoming gradient, and carries the forward op's attributes.

/// <summary>RMSNormGrad(x, w, dy) -> dx. Attributes as RMSNormalization (epsilon, axis).</summary>
public class RMSNormGradOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "RMSNormGrad";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        int axis = ctx.GetInt("axis", -1);
        float eps = ctx.GetFloat("epsilon", 1e-6f);
        var shape = ctx.Inputs[0].Shape;
        if (axis < 0) axis += shape.Length;
        int rows = 1; for (int i = 0; i < axis; i++) rows *= shape[i];
        int C = 1; for (int i = axis; i < shape.Length; i++) C *= shape[i];
        var scratch = ctx.Pool.Rent(new[] { 2 * rows }, "_rmsnorm_grad_stats");
        reg.Backward.RMSNorm(ctx.Inputs[0].Data, ctx.Inputs[1].Data, ctx.Inputs[2].Data, ctx.Outputs[0].Data, scratch.Data, rows, C, eps);
        ctx.Pool.Return(scratch);
    }
}

/// <summary>
/// MatMulGradInput(dy [..., N], W [K, N] [, accumulate [..., K]]) -> dx [..., K] = dy W^T (+ accumulate: the autograd folds a gradient sum here): the input gradient of <c>MatMul(x, W)</c> through a FROZEN weight in
/// whatever form the loader kept it - GGUF-quantized bytes (decoded in-register, never expanded), native low precision (F16/BF16/FP8), or f32.
/// </summary>
public class MatMulGradInputOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "MatMulGradInput";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var dy = inputs[0]; var w = inputs[1];
        var o = (int[])dy.Clone();
        o[^1] = w[0];
        return new[] { o };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var dy = ctx.Inputs[0]; var w = ctx.Inputs[1];
        int K = w.Shape[0], N = w.Shape[1];
        int M = dy.ElementCount / N;
        var dx = ctx.Outputs[0].Data.SubView(0, M * K);
        if (w.Name != null && ctx.QuantizedWeights != null && ctx.QuantizedWeights.TryGetValue(w.Name, out var q))
        {
            if (reg.QuantizedWeightTypes == null || !reg.QuantizedWeightTypes.TryGetValue(w.Name, out var type))
                throw new InvalidOperationException($"MatMulGradInput: quantized weight '{w.Name}' has no GGML type registered");
            reg.Backward.MatMulTransposedQuant(dy.Data.SubView(0, M * N), q, dx, M, K, N, type, Accumulate(ctx, M * K));
            return;
        }
        // The forward's B is [K, N]; dx = dy [M, N] x B^T [N, K] is the "transB" product of B stored [K, N] viewed as [N', K'] = [K, N].
        if (LowPWeightDispatch.IsLowP(w))
        {
            LowPWeightDispatch.MatMulTransB(reg.MatMul, dy.Data, w, dx, M, N, K);
            if (ctx.Inputs.Length > 2) reg.Backward.AddInto(dx, Accumulate(ctx, M * K), (long)M * K);
            return;
        }
        var wT = ctx.Pool.Rent(new[] { N, K }, "_matmul_grad_wT");
        reg.Transpose.Transpose(w.Data, wT.Data, new[] { K, N }, new[] { 1, 0 });
        reg.MatMul.MatMul(dy.Data, wT.Data, dx, M, N, K);
        ctx.Pool.Return(wT);
        if (ctx.Inputs.Length > 2) reg.Backward.AddInto(dx, Accumulate(ctx, M * K), (long)M * K);
    }

    private static global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> Accumulate(OnnxOpContext ctx, int count)
        => ctx.Inputs.Length > 2 ? ctx.Inputs[2].Data.SubView(0, count) : default;
}

/// <summary>
/// LoraLinear(x [..., F], A_T [F, R], B_T [R, N], yBase [..., N]) -> (y = yBase + scale (x A_T) B_T, xa = x A_T [..., R]): a LoRA adapter on a
/// projection whose base output is yBase (<see cref="GraphAutograd.AddLora"/>). Attribute <c>scale</c> (alpha / rank). xa is kept for the backward.
/// </summary>
public class LoraLinearOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "LoraLinear";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var xa = (int[])inputs[0].Clone();
        xa[^1] = inputs[1][1];
        return new[] { inputs[3], xa };
    }
    public void Execute(OnnxOpContext ctx)
    {
        var x = ctx.Inputs[0]; var aT = ctx.Inputs[1]; var bT = ctx.Inputs[2]; var yBase = ctx.Inputs[3];
        int F = aT.Shape[0], R = aT.Shape[1], N = bT.Shape[1];
        int M = x.ElementCount / F;
        reg.Backward.LoraForward(x.Data, aT.Data, bT.Data, yBase.Data, ctx.Outputs[0].Data, ctx.Outputs[1].Data, M, F, N, R, ctx.GetFloat("scale", 1f));
    }
}

/// <summary>
/// LoraLinearGrad(x, A_T, B_T, dy, xa [, accumulate]) -> (dx [+ accumulate], dA_T, dB_T) for <see cref="LoraLinearOperator"/> (dy is the gradient of y; yBase's gradient is dy
/// itself). Attributes <c>scale</c>, <c>need_dx</c> (0: dx is not computed - the adapter's input carries no gradient).
/// </summary>
public class LoraLinearGradOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "LoraLinearGrad";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0], inputs[1], inputs[2] };
    public void Execute(OnnxOpContext ctx)
    {
        var x = ctx.Inputs[0]; var aT = ctx.Inputs[1]; var bT = ctx.Inputs[2]; var dy = ctx.Inputs[3]; var xa = ctx.Inputs[4];
        int F = aT.Shape[0], R = aT.Shape[1], N = bT.Shape[1];
        int M = x.ElementCount / F;
        reg.Backward.LoraBackward(x.Data, aT.Data, bT.Data, dy.Data, xa.Data, ctx.Outputs[0].Data, ctx.Outputs[1].Data, ctx.Outputs[2].Data,
            M, F, N, R, ctx.GetFloat("scale", 1f), ctx.GetInt("need_dx", 1) == 1, ctx.Inputs.Length > 5 ? ctx.Inputs[5].Data : default);
    }
}

/// <summary>
/// MatMulGroup(x, W_1, ..., W_n) -> (x W_1, ..., x W_n), n = 2 or 3: sibling projections of one input (q | k | v, gate | up) that
/// GraphAutograd.GroupSiblingMatMuls joined. Q4_K / Q6_K weights on CUDA sm_80+ run as ONE scratch GEMM (FusedDequantMatMul.ForwardGroup:
/// the input packed once, the weights dequantized side by side, each output's columns written to its own tensor; bit-identical to separate
/// MatMuls); anything else runs the MatMul operator once per weight, exactly as the separate nodes did.
/// </summary>
public class MatMulGroupOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "MatMulGroup";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var outs = new int[inputs.Length - 1][];
        for (int i = 1; i < inputs.Length; i++)
        {
            var o = (int[])inputs[0].Clone();
            o[^1] = inputs[i][^1];
            outs[i - 1] = o;
        }
        return outs;
    }
    public void Execute(OnnxOpContext ctx)
    {
        var x = ctx.Inputs[0];
        int n = ctx.Inputs.Length - 1;
        int K = ctx.Inputs[1].Shape[0];
        int M = x.ElementCount / K;
        if (ctx.QuantizedWeights != null && reg.QuantizedWeightTypes != null && ctx.InputNames.Length == n + 1)
        {
            var ws = new List<global::ILGPU.Runtime.ArrayView1D<byte, global::ILGPU.Stride1D.Dense>>();
            var types = new List<GGUF.GGMLType>();
            var ns = new List<int>();
            var outs = new List<global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>>();
            for (int i = 0; i < n; i++)
            {
                string w = ctx.InputNames[1 + i];
                if (!ctx.QuantizedWeights.TryGetValue(w, out var q) || !reg.QuantizedWeightTypes.TryGetValue(w, out var t) || ctx.Inputs[1 + i].Shape[0] != K) break;
                ws.Add(q); types.Add(t); ns.Add(ctx.Inputs[1 + i].Shape[1]); outs.Add(ctx.Outputs[i].Data);
            }
            if (ws.Count == n && reg.FusedDequant.ForwardGroup(x.Data, ws, types, ns, outs, M, K)) return;
        }
        var mm = reg.Resolve("MatMul");
        for (int i = 0; i < n; i++)
            mm.Execute(new OnnxOpContext
            {
                Inputs = new[] { x, ctx.Inputs[1 + i] },
                Outputs = new[] { ctx.Outputs[i] },
                Attributes = new Dictionary<string, object>(),
                Pool = ctx.Pool,
                Format = ctx.Format,
                InputNames = ctx.InputNames.Length == n + 1 ? new[] { ctx.InputNames[0], ctx.InputNames[1 + i] } : Array.Empty<string>(),
                ConstantValues = ctx.ConstantValues,
                IsInputTainted = ctx.IsInputTainted,
                HostValues = ctx.HostValues,
                OutputNames = ctx.OutputNames.Length == n ? new[] { ctx.OutputNames[i] } : Array.Empty<string>(),
                QuantizedWeights = ctx.QuantizedWeights,
                Registry = ctx.Registry,
                ScopeTensors = ctx.ScopeTensors,
                IntegerTensorNames = ctx.IntegerTensorNames,
            });
    }
}

/// <summary>
/// GatherRowsGrad(dy [1, n, C], positions [n], like [1, T, C]) -> dx shaped like <c>like</c>: the gradient of Gather(like, positions, axis=1)
/// for DISTINCT positions (rows not gathered get zero).
/// </summary>
public class GatherRowsGradOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "GatherRowsGrad";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[2] };
    public void Execute(OnnxOpContext ctx)
    {
        var dy = ctx.Inputs[0]; var pos = ctx.Inputs[1]; var like = ctx.Inputs[2];
        int C = like.Shape[^1];
        int rows = like.ElementCount / C, n = pos.ElementCount;
        reg.Backward.GatherRowsGrad(dy.Data, pos.Data, ctx.Outputs[0].Data, rows, n, C);
    }
}

/// <summary>SwiGLUGrad(g, u, dy) -> (dg, du) for SwiGLU(g, u) = silu(g) * u.</summary>
public class SwiGLUGradOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "SwiGLUGrad";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0], inputs[1] };
    public void Execute(OnnxOpContext ctx)
        => reg.Backward.SwiGLU(ctx.Inputs[0].Data, ctx.Inputs[1].Data, ctx.Inputs[2].Data, ctx.Outputs[0].Data, ctx.Outputs[1].Data,
            ctx.Outputs[0].ElementCount);
}

/// <summary>RoPEGrad(dy, [freq_factors]) -> dx: the adjoint rotation of RoPE with the same attributes and inputs.</summary>
public class RoPEGradOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "RoPEGrad";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        var dy = ctx.Inputs[0];
        int headDim = dy.Shape[^1];
        int rows = dy.ElementCount / headDim;
        int rotaryDim = ctx.GetInt("rotary_dim", 0);
        if (rotaryDim <= 0) rotaryDim = headDim;
        // Optional 2nd input = the forward's per-pair frequency factors (llama 3.x long-context scaling), as RoPE takes them.
        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>? ff = null;
        if (ctx.Inputs.Length > 1 && ctx.Inputs[1] != null && ctx.Inputs[1].ElementCount > 0)
        {
            if (ctx.Inputs[1].ElementCount < rotaryDim / 2)
                throw new InvalidOperationException($"RoPEGrad freq_factors input must have rotary_dim/2 = {rotaryDim / 2} entries; got {ctx.Inputs[1].ElementCount}.");
            ff = ctx.Inputs[1].Data;
        }
        reg.Backward.RoPE(dy.Data, ctx.Outputs[0].Data, ff, rows, headDim, ctx.GetInt("kv_offset", 0), ctx.GetFloat("rope_base", 10000f),
            rotaryDim, ctx.GetInt("interleaved", 0) == 1, ctx.GetInt("rows_per_position", 1));
    }
}

/// <summary>
/// SoftmaxCrossEntropyLossGrad(scores [N..., C], labels [N...]) -> dscores, for ONNX SoftmaxCrossEntropyLoss with reduction=mean
/// and ignore_index (default -100): rows whose label is ignore_index get zero gradient and do not count in the mean.
/// </summary>
public class SoftmaxCrossEntropyLossGradOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "SoftmaxCrossEntropyLossGrad";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0] };
    public void Execute(OnnxOpContext ctx)
    {
        var scores = ctx.Inputs[0];
        if (ctx.GetString("reduction", "mean") != "mean")
            throw new NotSupportedException("SoftmaxCrossEntropyLossGrad: only reduction=mean");
        int C = scores.Shape[^1];
        int rows = scores.ElementCount / C;
        var scratch = ctx.Pool.Rent(new[] { 1 + rows }, "_ce_grad_stats");
        int ignore = ctx.Attributes.ContainsKey("ignore_index") ? ctx.GetInt("ignore_index", int.MinValue) : int.MinValue;   // ONNX: absent = none
        reg.Backward.SoftmaxCrossEntropy(scores.Data, ctx.Inputs[1].Data, ctx.Outputs[0].Data, scratch.Data, rows, C, ignore);
        ctx.Pool.Return(scratch);
    }
}

/// <summary>
/// FusedAttentionGrad(q, k, v, out, dout) -> (dq, dk, dv) for FusedAttention with the same attributes. Seq-major q/k/v/out only
/// (the layout GGUFGraphBuilder emits); no attention sinks.
/// </summary>
public class FusedAttentionGradOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "FusedAttentionGrad";
    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs) => new[] { inputs[0], inputs[1], inputs[2] };
    public void Execute(OnnxOpContext ctx)
    {
        if (ctx.GetInt("seq_major_q", 0) != 1 || ctx.GetInt("seq_major_kv", 0) != 1 || ctx.GetInt("seq_major_out", 0) != 1)
            throw new NotSupportedException("FusedAttentionGrad: only the seq-major layout (seq_major_q/kv/out = 1)");
        var q = ctx.Inputs[0]; var k = ctx.Inputs[1];
        int heads = ctx.GetInt("n_heads", 0), hd = ctx.GetInt("head_dim", 0);
        if (heads <= 0 || hd <= 0) throw new NotSupportedException("FusedAttentionGrad: n_heads and head_dim attributes are required");
        int kvHeads = ctx.GetInt("n_kv_heads", heads);
        float scale = ctx.GetFloat("scale", 0f);
        var d = new BackwardKernels.AttnDims
        {
            Heads = heads, KvHeads = kvHeads, HeadDim = hd,
            SeqQ = q.ElementCount / (heads * hd), SeqKV = k.ElementCount / (kvHeads * hd),
            Causal = ctx.GetInt("causal", 1), Window = Math.Max(0, ctx.GetInt("window", 0)), KvOffset = ctx.GetInt("kv_offset", 0),
            Scale = scale > 0f ? scale : 1f / MathF.Sqrt(hd),
        };
        var stats = ctx.Pool.Rent(new[] { 2 * heads * d.SeqQ }, "_attn_grad_stats");
        reg.Backward.Attention(q.Data, k.Data, ctx.Inputs[2].Data, ctx.Inputs[3].Data, ctx.Inputs[4].Data,
            ctx.Outputs[0].Data, ctx.Outputs[1].Data, ctx.Outputs[2].Data, stats.Data, d);
        ctx.Pool.Return(stats);
    }
}
