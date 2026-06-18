using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Fused multi-head attention kernel — entire attention block in a single GPU dispatch.
/// Computes: softmax(mask(Q @ K^T / sqrt(d))) @ V in one kernel, eliminating
/// 3+ dispatch boundaries (command buffer submissions on WebGPU).
///
/// Standard (unfused) attention requires 5 dispatches:
///   1. Q @ K^T (MatMul)
///   2. Scale by 1/sqrt(d) (ElementWise)
///   3. Softmax (two-pass: max + exp+sum)
///   4. Scores @ V (MatMul)
///   5. Output projection
///
/// This kernel does steps 1-4 in a single dispatch per head, with the attention matrix
/// never materialized in global memory (online softmax, single pass over KV).
///
/// MASKING (decoder LLMs - gemma4 etc.): causal + sliding-window, computed from indices
/// in-kernel (no mask tensor). Query position = kvOffset + sq, so KV-cache decode
/// (seqQ = 1 at an arbitrary position) works. Sliding window is PER CALL: gemma-style
/// 5:1 SWA/global interleaves pass their per-layer window; global layers pass
/// window &gt;= seqKV (no window constraint). The caller wires per-layer values
/// (GGUF graph wiring owns that); this kernel just honors the parameters.
///
/// SHAPE RULE: the kernel body is BRANCH-FREE - no if/else, ternaries, or early returns
/// inside the KV loop. Masking is sign-bit arithmetic driving scores to -1e10 (exp -> 0),
/// and the online-softmax max-update is MathF.Max/Exp algebra (the correction factor is
/// exactly 1 when the max does not change). Any branch construct inlined into a loop body
/// multiplies the WebGL GLSL emitter's block duplication catastrophically (31.5MB shader
/// OOM - measured 2026-06-11, repro filed to the ILGPU lane; see FusedDequantMatMul).
/// </summary>
public class FusedAttentionKernel : IDisposable
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _kernel;

    // Dummy 1-element sinks buffer for the no-sinks case (the kernel always takes a sinks view but only
    // reads it when sinkCount > 0). gpt-oss/OpenAI-MoE attention passes real per-head sink logits.
    private MemoryBuffer1D<float, Stride1D.Dense>? _dummySinks;

    // Params-buffer ring: each call gets its own buffer (per-layer window/kvOffset/seqKV
    // differ within one batched command encoder, so a single mutated buffer would feed
    // pending dispatches the WRONG params), but the ring bounds memory - a buffer is
    // reused (disposed-by-overwrite) only after RingSize subsequent calls, far past any
    // realistic unflushed batch depth. Replaces the old "dispose previous on next call"
    // pattern, which freed a buffer a still-pending dispatch could be reading.
    private const int RingSize = 64;
    private readonly MemoryBuffer1D<int, Stride1D.Dense>?[] _paramsRing
        = new MemoryBuffer1D<int, Stride1D.Dense>?[RingSize];
    private int _ringNext;

    public FusedAttentionKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Fused bidirectional attention (no mask) - original API, behavior unchanged.
    /// Q [B*H, seqQ, D], K [B*H, seqKV, D], V [B*H, seqKV, D] → output [B*H, seqQ, D].
    /// </summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<float, Stride1D.Dense> K,
        ArrayView1D<float, Stride1D.Dense> V,
        ArrayView1D<float, Stride1D.Dense> output,
        int batchHeads, int seqQ, int seqKV, int headDim) =>
        Forward(Q, K, V, output, batchHeads, seqQ, seqKV, headDim,
            causal: false, window: int.MaxValue, kvOffset: 0);

    /// <summary>
    /// Fused attention with index-computed masking (Q and K/V share head count).
    /// Query position = <paramref name="kvOffset"/> + sq (pass the KV-cache length for
    /// single-token decode). <paramref name="causal"/> hides kv &gt; qPos.
    /// <paramref name="window"/> additionally hides kv &lt;= qPos - window (sliding-window
    /// attention; pass int.MaxValue or anything &gt;= seqKV for no window).
    /// </summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<float, Stride1D.Dense> K,
        ArrayView1D<float, Stride1D.Dense> V,
        ArrayView1D<float, Stride1D.Dense> output,
        int batchHeads, int seqQ, int seqKV, int headDim,
        bool causal, int window, int kvOffset) =>
        Forward(Q, K, V, output, batchHeads, batchHeads, seqQ, seqKV, headDim,
            causal, window, kvOffset, scale: 0f);

    /// <summary>
    /// Fused attention with masking, GROUPED-QUERY heads, and an explicit scale.
    /// Q [nHeads, seqQ, D]; K, V [kvHeads, seqKV, D]; output [nHeads, seqQ, D] - query
    /// head h attends kv head h / (nHeads / kvHeads) (GQA; gemma4 runs 8 kv heads on
    /// sliding layers and 1 on global layers). <paramref name="scale"/> &lt;= 0 means the
    /// default 1/sqrt(headDim); gemma-family models pass their query_pre_attn_scalar-
    /// derived value instead.
    /// </summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<float, Stride1D.Dense> K,
        ArrayView1D<float, Stride1D.Dense> V,
        ArrayView1D<float, Stride1D.Dense> output,
        int nHeads, int kvHeads, int seqQ, int seqKV, int headDim,
        bool causal, int window, int kvOffset, float scale,
        ArrayView1D<float, Stride1D.Dense>? sinks = null, int sinkCount = 0)
    {
        if (window <= 0) throw new ArgumentOutOfRangeException(nameof(window), "window must be positive");
        if (kvHeads <= 0 || nHeads % kvHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(kvHeads),
                $"kvHeads ({kvHeads}) must evenly divide nHeads ({nHeads}) for grouped-query attention.");
        // Clamp so the in-kernel sign-bit arithmetic cannot underflow int.MinValue:
        // any window >= seqKV + seqQ + kvOffset constrains nothing.
        long noConstraint = (long)seqKV + seqQ + Math.Max(kvOffset, 0) + 1;
        int effWindow = (int)Math.Min(window, noConstraint);

        // Exact float scale passed as raw bits (the old (int)(scale*10000) quantized the
        // scale to 1e-4 - a real precision loss for large headDim).
        float effScale = scale > 0f ? scale : 1f / MathF.Sqrt(headDim);
        var paramsData = new int[]
        {
            nHeads, seqQ, seqKV, headDim,
            BitConverter.SingleToInt32Bits(effScale),
            causal ? 1 : 0, effWindow, kvOffset,
            nHeads / kvHeads, // GQA group size: query head h reads kv head h / group
            sinkCount,        // p[9]: >0 => fold per-head sink logit into the softmax denominator
        };

        var slot = _ringNext;
        _ringNext = (_ringNext + 1) % RingSize;
        _paramsRing[slot]?.Dispose();
        _paramsRing[slot] = _accelerator.Allocate1D(paramsData);

        _kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(FusedAttentionImpl);

        _dummySinks ??= _accelerator.Allocate1D(new float[1]);
        var sinksView = sinks ?? _dummySinks.View;

        // One thread per output element: nHeads * seqQ * headDim
        _kernel(nHeads * seqQ * headDim, Q, K, V, output, sinksView, _paramsRing[slot]!.View);
    }

    /// <summary>
    /// Per-element fused attention with Online Softmax (single pass) and arithmetic
    /// masking. Each thread computes one output value by iterating all KV positions.
    /// BRANCH-FREE BODY (see class doc): masked positions get score -1e10 via sign-bit
    /// masks (their exp underflows to 0 and they never win the running max); the
    /// max-update is pure Max/Exp algebra (correction factor is exactly 1 when the
    /// running max is unchanged, so unconditional application is identical math).
    /// </summary>
    private static void FusedAttentionImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<float, Stride1D.Dense> K,
        ArrayView1D<float, Stride1D.Dense> V,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> sinks,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int BH = p[0], SQ = p[1], SKV = p[2], D = p[3];
        float scale = Interop.IntAsFloat((uint)p[4]);
        int causal = p[5];
        int window = p[6];
        int kvOffset = p[7];
        int gqaGroup = p[8]; // nHeads / kvHeads; query head bh reads kv head bh / gqaGroup
        int sinkCount = p[9]; // 0 => no sinks; else per-head sink logit count (gpt-oss attention sinks)

        // Decompose index: [bh, sq, d]
        int d = idx % D;
        int sq = (idx / D) % SQ;
        int bh = idx / (SQ * D);

        if (bh >= BH) return;

        int kvHead = bh / gqaGroup;
        int qBase = (bh * SQ + sq) * D;
        int qPos = kvOffset + sq;

        float runningMax = -1e10f;
        float runningSum = 0f;
        float weightedV = 0f;

        for (int kv = 0; kv < SKV; kv++)
        {
            int kBase = (kvHead * SKV + kv) * D;
            float dot = 0f;
            for (int dd = 0; dd < D; dd++)
                dot += Q[qBase + dd] * K[kBase + dd];
            float score = dot * scale;

            // valid = (!causal || kv <= qPos) && (kv > qPos - window), as 0/1 ints:
            // causalOk: causal=0 -> always 1; else 1 when qPos - kv >= 0.
            // windowOk: 1 when qPos - window - kv < 0 (sign bit).
            int causalOk = 1 - (causal & ((qPos - kv) >> 31) & 1);
            int windowOk = ((qPos - window - kv) >> 31) & 1;
            int valid = causalOk & windowOk;
            // Masked score must sit STRICTLY below the runningMax init (-1e10): at
            // exactly -1e10 a masked position preceding any valid one would hit
            // exp(score - newMax) = exp(0) = 1 and contribute full weight. -1e30
            // keeps the init as the max and its exp underflows cleanly to 0.
            score = score * valid + -1e30f * (1 - valid);

            // Branch-free online softmax: when score <= runningMax the correction is
            // exp(0) = 1 and nothing shifts; otherwise the running terms rescale.
            float newMax = MathF.Max(runningMax, score);
            float correction = MathF.Exp(runningMax - newMax);
            float weight = MathF.Exp(score - newMax);
            runningSum = runningSum * correction + weight;
            weightedV = weightedV * correction + weight * V[kBase + d];
            runningMax = newMax;
        }

        // Attention sinks (gpt-oss): a per-head learned logit joins the softmax as if it were one more score
        // whose value vector is 0 - it participates in the max + denominator but adds nothing to the numerator.
        // Equivalent to concatenating the sink to the scores before softmax. sinkCount is uniform across all
        // threads (no warp divergence). sink index = head within the batch (bh % sinkCount).
        if (sinkCount > 0)
        {
            float sink = sinks[bh % sinkCount];
            float newMax = MathF.Max(runningMax, sink);
            float correction = MathF.Exp(runningMax - newMax);
            runningSum = runningSum * correction + MathF.Exp(sink - newMax);
            weightedV = weightedV * correction; // sink contributes 0 to the value sum
            runningMax = newMax;
        }

        output[idx] = weightedV / (runningSum + 1e-10f);
    }

    public void Dispose()
    {
        foreach (var buf in _paramsRing) buf?.Dispose();
        Array.Clear(_paramsRing);
        _dummySinks?.Dispose();
        _dummySinks = null;
    }
}
