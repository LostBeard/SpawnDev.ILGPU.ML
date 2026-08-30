using ILGPU;
using ILGPU.Runtime;
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
///        scale:f (default/&lt;= 0 = 1/sqrt(head_dim); gemma passes its
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
                // Include tensor-identity evidence: a wrong-rank q here has meant the executor handed us a
                // stale/aliased tensor object, not a mis-computed shape. elemCount matching the WRONG
                // shape's volume = the Tensor object was re-rented to another node's output while our dict
                // entry still pointed at it (pool aliasing); elemCount matching the real producer's volume
                // = shape metadata mutated on a live tensor. objHash correlates identity across the
                // executor's Rent/Return logs.
                throw new InvalidOperationException(
                    $"FusedAttention needs n_heads+head_dim attrs, or a rank-3/4 q to derive them; got q rank {qs.Length} "
                    + $"(q shape=[{string.Join(",", qs)}] elemCount={q.ElementCount} dataLen={q.Data.Length} objHash={System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(q):x8}; "
                    + $"k shape=[{string.Join(",", k.Shape)}] elemCount={k.ElementCount} objHash={System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(k):x8}; "
                    + $"v shape=[{string.Join(",", v.Shape)}] elemCount={v.ElementCount} objHash={System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(v):x8}).");
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
        var sinksT = ctx.Inputs.Length > 3 && ctx.Inputs[3] != null && ctx.Inputs[3].ElementCount > 0 ? ctx.Inputs[3] : null;
        ArrayView1D<float, Stride1D.Dense>? sinksView = sinksT != null ? sinksT.Data : (ArrayView1D<float, Stride1D.Dense>?)null;
        int sinkCount = sinksT?.ElementCount ?? 0;

        // seq_major_out: the graph builder sets this when it has DROPPED the post-attention Transpose[0,2,1,3],
        // so FusedAttention must write its output directly in seq-major [1,seq,heads,hd] layout (p[11]).
        bool seqMajor = ctx.GetInt("seq_major_out", 0) == 1;
        // seq_major_q: the graph dropped the Q PRE-attention transpose, so Q is fed seq-major [1,seq,heads,hd]
        // and FusedAttention reads it seq-major (p[12]). Independent of seq_major_out (the output side, step 1).
        bool seqMajorQ = ctx.GetInt("seq_major_q", 0) == 1;
        // seq_major_kv: the graph dropped the K/V PRE-attention transposes — K/V are seq-major [.,seq,kvHeads,hd]
        // (and for decode the KV-cache store is seq-major), so FusedAttention reads K/V seq-major (p[13]).
        bool seqMajorKV = ctx.GetInt("seq_major_kv", 0) == 1;

        // DECODE KV-cache path: K/V are the cache's [kvHeads, maxSeq, hd] store, read DIRECTLY in their native
        // type (bf16 / f32) with the store's per-head row pitch as the stride — NO per-token repack + bf16→f32
        // widen. Signalled by kv_seq_len (the LIVE history length, since the store buffer is maxSeq-padded so
        // seqKV-from-ElementCount would over-read into padding). Dispatch on the store's DType.
        int kvSeqLen = ctx.GetInt("kv_seq_len", 0);
        if (kvSeqLen > 0)
        {
            int kvRowStride = k.Shape[^2] * k.Shape[^1]; // maxSeq * hd — the store's per-head element pitch
            if (LowPWeightDispatch.IsLowP(k))
                reg.FusedAttention.ForwardStrided(q.Data, k.AsView<BFloat16>(), v.AsView<BFloat16>(), ctx.Outputs[0].Data,
                    nHeads, kvHeads, seqQ, kvSeqLen, headDim, causal, effWindow, kvOffset, scale, kvRowStride, sinksView, sinkCount, seqMajor, seqMajorQ, seqMajorKV);
            else
                reg.FusedAttention.ForwardStrided(q.Data, k.Data, v.Data, ctx.Outputs[0].Data,
                    nHeads, kvHeads, seqQ, kvSeqLen, headDim, causal, effWindow, kvOffset, scale, kvRowStride, sinksView, sinkCount, seqMajor, seqMajorQ, seqMajorKV);
            return;
        }

        reg.FusedAttention.Forward(q.Data, k.Data, v.Data, ctx.Outputs[0].Data,
            nHeads, kvHeads, seqQ, seqKV, headDim, causal, effWindow, kvOffset, scale,
            sinks: sinksView, sinkCount: sinkCount, seqMajorOut: seqMajor, seqMajorQ: seqMajorQ, seqMajorKV: seqMajorKV);
    }
}
