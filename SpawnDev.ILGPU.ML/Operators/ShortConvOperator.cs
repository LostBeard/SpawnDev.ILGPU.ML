namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// LFM2 short-convolution mixer (custom op "ShortConv"). Consumes the in_proj output BCx and the depthwise
/// conv kernel; emits the gated causal-conv result y, ready for out_proj. The split/gate/conv/gate math lives
/// in <see cref="Kernels.ShortConvKernel"/> (verified against llama.cpp src/models/lfm2.cpp).
///
///   inputs:  BCx [1, seq, 3H]   (in_proj output)
///            weight [H, L]       (shortconv.conv.weight; L = lfm2.shortconv.l_cache)
///   output:  y   [1, seq, H]
/// </summary>
public sealed class ShortConvOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "ShortConv";

    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var bcx = inputs[0];                 // [1, seq, 3H]  (or [seq, 3H])
        var outShape = (int[])bcx.Clone();
        outShape[^1] = bcx[^1] / 3;          // 3H -> H
        return new[] { outShape };
    }

    public void Execute(OnnxOpContext ctx)
    {
        var bcxShape = ctx.Inputs[0].Shape;
        int seq = bcxShape.Length >= 2 ? bcxShape[^2] : 1;
        int H = bcxShape[^1] / 3;
        // weight [H, L] -> L is the per-channel tap count.
        int L = H > 0 ? (int)(ctx.Inputs[1].ElementCount / H) : 0;
        if (L <= 0)
            throw new InvalidOperationException($"ShortConv: bad conv weight (elems={ctx.Inputs[1].ElementCount}, H={H}).");

        reg.ShortConv.Forward(ctx.Inputs[0].Data, ctx.Inputs[1].Data, ctx.Outputs[0].Data, seq, H, L);
    }
}
