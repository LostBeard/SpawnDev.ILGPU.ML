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
        reg.Backward.SoftmaxCrossEntropy(scores.Data, ctx.Inputs[1].Data, ctx.Outputs[0].Data, scratch.Data, rows, C, ctx.GetInt("ignore_index", -100));
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
