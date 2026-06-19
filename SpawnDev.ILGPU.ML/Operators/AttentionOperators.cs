using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Operators;

// gemma4 fused-attention operator layer (contract agreed with the graph wiring:
// seven-to-tuvok-CONTRACT-AGREED 2026-06-11). The GGUF graph emits per-layer nodes;
// every attribute (rope_base, window, n_kv_heads, ...) is selected per layer by the
// graph builder from GGUF metadata - these operators just honor the node.

/// <summary>
/// RoPE node - rotary position embedding via <see cref="Kernels.RoPEKernel"/>.
/// in: [x] (last dim = head_dim; rows = everything before it)
///     [freq_factors] OPTIONAL second input, [rotary_dim/2] floats (gemma4 global
///     layers pass rope_freqs.weight; absent = scalar-base behavior). ggml semantics:
///     the per-pair theta is DIVIDED by its factor.
/// attrs: rope_base:f (default 10000), rotary_dim:i (default = head_dim),
///        interleaved:i 0/1 (default 0 = NeoX split-half),
///        kv_offset:i (default 0; sequence position of row 0),
///        rows_per_position:i (default 1; pass heads for the pre-transpose
///        [seq, heads, head_dim] layout so all of a position's heads share its angle)
/// out: [x_roped] (same shape)
/// </summary>
public class RoPEOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "RoPE";

    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };

    public void Execute(OnnxOpContext ctx)
    {
        var x = ctx.Inputs[0];
        int headDim = x.Shape[^1];
        int rows = x.ElementCount / headDim;

        float ropeBase = ctx.GetFloat("rope_base", 10000f);
        int rotaryDim = ctx.GetInt("rotary_dim", 0);
        if (rotaryDim <= 0) rotaryDim = headDim;
        bool interleaved = ctx.GetInt("interleaved", 0) == 1;
        int kvOffset = ctx.GetInt("kv_offset", 0);
        int rowsPerPosition = ctx.GetInt("rows_per_position", 1);

        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>? freqFactors = null;
        if (ctx.Inputs.Length > 1 && ctx.Inputs[1].ElementCount > 0)
        {
            var ff = ctx.Inputs[1];
            if (ff.ElementCount < rotaryDim / 2)
                throw new InvalidOperationException(
                    $"RoPE freq_factors input must have rotary_dim/2 = {rotaryDim / 2} entries; got {ff.ElementCount}.");
            freqFactors = ff.Data;
        }

        reg.RoPE.Apply(x.Data, ctx.Outputs[0].Data, rows, headDim, kvOffset,
            ropeBase, rotaryDim, interleaved, rowsPerPosition, freqFactors);
    }
}

/// <summary>
/// FusedAttention node - the whole masked, online-softmax softmax(QK^T·scale)V in one
/// dispatch via <see cref="Kernels.FusedAttentionKernel"/>, with grouped-query heads.
/// in: [q, k, v] - q [n_heads, seqQ, head_dim] flat (the post-transpose [1,H,S,D]
///     collapses to exactly this); k, v [n_kv_heads, seqKV, head_dim] flat.
///     q and k are already RoPE'd (separate RoPE nodes); v is raw.
/// attrs: n_heads:i (required), n_kv_heads:i (default = n_heads), head_dim:i (required),
///        causal:i (default 1), window:i (0 or >= seqKV = global/no window),
///        scale:f (default/<= 0 = 1/sqrt(head_dim); gemma passes its
///        query_pre_attn_scalar-derived value), kv_offset:i (default 0)
/// out: [attn] [n_heads, seqQ, head_dim] (same layout as q).
/// </summary>
public class FusedAttentionOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "FusedAttention";

    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
        => new[] { inputs[0] };

    public void Execute(OnnxOpContext ctx)
    {
        var q = ctx.Inputs[0];
        var k = ctx.Inputs[1];
        var v = ctx.Inputs[2];

        int nHeads = ctx.GetInt("n_heads", 0);
        int headDim = ctx.GetInt("head_dim", 0);
        // When the attrs are absent (the graph-fusion path — see GraphOptimizer.FuseAttention), derive the
        // layout from q's runtime shape: rank-3 [batch·heads, seq, head_dim] or rank-4 [batch, heads, seq,
        // head_dim]. This makes one FusedAttention node work for any heads/head_dim/resolution without the
        // fusion pass having to know shapes at compile time (it doesn't). The gemma4 path keeps passing attrs.
        if (nHeads <= 0 || headDim <= 0)
        {
            var qs = q.Shape;
            if (qs.Length == 3) { nHeads = qs[0]; headDim = qs[2]; }
            else if (qs.Length == 4) { nHeads = qs[0] * qs[1]; headDim = qs[3]; }
            else
                throw new InvalidOperationException(
                    $"FusedAttention needs n_heads+head_dim attrs, or a rank-3/4 q to derive them; got q rank {qs.Length}.");
        }
        int kvHeads = ctx.GetInt("n_kv_heads", nHeads);
        bool causal = ctx.GetInt("causal", 1) == 1;
        int window = ctx.GetInt("window", 0);
        float scale = ctx.GetFloat("scale", 0f);
        int kvOffset = ctx.GetInt("kv_offset", 0);

        int seqQ = q.ElementCount / (nHeads * headDim);
        int seqKV = k.ElementCount / (kvHeads * headDim);
        if (seqQ * nHeads * headDim != q.ElementCount || seqKV * kvHeads * headDim != k.ElementCount)
            throw new InvalidOperationException(
                $"FusedAttention shape mismatch: q={q.ElementCount} elems vs n_heads={nHeads}*head_dim={headDim}; " +
                $"k={k.ElementCount} elems vs n_kv_heads={kvHeads}*head_dim={headDim}.");

        // Contract: window 0 OR >= seqKV both mean global (no window constraint).
        int effWindow = window <= 0 ? int.MaxValue : window;

        // Optional 4th input = per-head attention sinks (gpt-oss attn_sinks, [n_head]): a learned logit
        // folded into the softmax denominator (0 value contribution). Absent => sinkCount 0 = plain attention.
        if (ctx.Inputs.Length > 3 && ctx.Inputs[3] != null && ctx.Inputs[3].ElementCount > 0)
        {
            var sinks = ctx.Inputs[3];
            reg.FusedAttention.Forward(q.Data, k.Data, v.Data, ctx.Outputs[0].Data,
                nHeads, kvHeads, seqQ, seqKV, headDim, causal, effWindow, kvOffset, scale,
                sinks: sinks.Data, sinkCount: sinks.ElementCount);
        }
        else
        {
            reg.FusedAttention.Forward(q.Data, k.Data, v.Data, ctx.Outputs[0].Data,
                nHeads, kvHeads, seqQ, seqKV, headDim, causal, effWindow, kvOffset, scale);
        }
    }
}
