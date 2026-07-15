using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// Qwen3-Next "Gated DeltaNet" linear-attention mixer (custom op "GatedDeltaNet"), GGUF arch qwen35's 24
/// linear layers. Orchestrates: causal conv1d(k=4)+SiLU on the fused [q|k|v] (splitting into contiguous
/// q/k/v), per-head L2-norm of q/k, the delta-rule recurrence scan (<see cref="Kernels.GatedDeltaNetKernel"/>,
/// with the per-head decay/beta gates folded in), and a gated RMSNorm. The projections around it
/// (attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_out) are separate graph MatMul nodes.
///
///   inputs:  qkv [1,seq,qDim+kDim+vDim] (attn_qkv out), z [1,seq,vDim] (attn_gate out),
///            a [1,seq,numVHeads] (ssm_alpha out), b [1,seq,numVHeads] (ssm_beta out),
///            convW (ssm_conv1d [C,L]), ssmA (ssm_a [numVHeads]), ssmDt (ssm_dt [numVHeads]),
///            ssmNorm (ssm_norm [head_v_dim])
///   output:  o [1,seq,vDim]  (feeds the ssm_out projection)
///   attrs:   num_k_heads, head_k_dim, num_v_heads, head_v_dim
///
/// This first version is FULL-RECOMPUTE (prefill semantics): the recurrent state is zeroed each call and
/// discarded. A GatedDeltaNetStateCache (the recurrent analogue of ShortConvStateCache) for O(1) KV-decode
/// is the follow-up, wired via a GraphExecutor intercept like ShortConv.
/// </summary>
public sealed class GatedDeltaNetOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "GatedDeltaNet";

    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { (int[])inputs[1].Clone() };   // output shape == z shape [1, seq, vDim]

    public void Execute(OnnxOpContext ctx)
    {
        var qkv = ctx.Inputs[0];
        var z = ctx.Inputs[1];
        var a = ctx.Inputs[2];
        var b = ctx.Inputs[3];
        var convW = ctx.Inputs[4];
        var ssmA = ctx.Inputs[5];
        var ssmDt = ctx.Inputs[6];
        var ssmNorm = ctx.Inputs[7];

        int seq = qkv.Shape.Length >= 2 ? qkv.Shape[^2] : 1;
        int C = qkv.Shape[^1];
        int numKHeads = ctx.GetInt("num_k_heads");
        int headKDim = ctx.GetInt("head_k_dim");
        int numVHeads = ctx.GetInt("num_v_heads");
        int headVDim = ctx.GetInt("head_v_dim");
        int qDim = numKHeads * headKDim, kDim = qDim, vDim = numVHeads * headVDim;
        int L = C > 0 ? (int)(convW.ElementCount / C) : 0;
        if (L <= 0 || qDim + kDim + vDim != C)
            throw new InvalidOperationException($"GatedDeltaNet: bad dims C={C} qDim={qDim} kDim={kDim} vDim={vDim} L={L}.");

        var scan = reg.GatedDeltaNetScan;
        var ops = reg.GatedDeltaNetOps;

        // Scratch (returned to the pool after use).
        var q = ctx.Pool.Rent(new[] { 1, seq, qDim }, "gdn_q");
        var k = ctx.Pool.Rent(new[] { 1, seq, kDim }, "gdn_k");
        var v = ctx.Pool.Rent(new[] { 1, seq, vDim }, "gdn_v");
        var scanOut = ctx.Pool.Rent(new[] { 1, seq, vDim }, "gdn_scan");
        var state = ctx.Pool.Rent(new[] { numVHeads, headVDim, headKDim }, "gdn_state");
        try
        {
            // conv(k=4)+SiLU on [q|k|v], split to contiguous q/k/v.
            ops.CausalConvSiluSplit(qkv.Data, convW.Data, q.Data, k.Data, v.Data, seq, qDim, kDim, vDim, L);
            // per-head L2 norm of q and k (over head_k_dim).
            ops.L2NormHeads(q.Data, seq * numKHeads, headKDim);
            ops.L2NormHeads(k.Data, seq * numKHeads, headKDim);
            // fresh recurrent state (prefill/full-recompute).
            ops.Zero(state.Data, numVHeads * headVDim * headKDim);
            // delta-rule scan (gates g/beta folded in from a,b,ssm_a,ssm_dt).
            scan.Forward(q.Data, k.Data, v.Data, a.Data, b.Data, ssmA.Data, ssmDt.Data,
                scanOut.Data, state.Data, seq, numKHeads, headKDim, numVHeads, headVDim);
            // gated RMSNorm: rmsnorm(scan_head, ssm_norm) * silu(z_head).
            ops.GatedRmsNorm(scanOut.Data, ssmNorm.Data, z.Data, ctx.Outputs[0].Data, seq * numVHeads, headVDim);
        }
        finally
        {
            ctx.Pool.Return(q); ctx.Pool.Return(k); ctx.Pool.Return(v);
            ctx.Pool.Return(scanOut); ctx.Pool.Return(state);
        }
    }
}
