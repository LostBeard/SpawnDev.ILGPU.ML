using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Fused multi-head attention kernel — entire attention block in a single GPU dispatch.
/// Computes: softmax(Q @ K^T / sqrt(d)) @ V in one kernel, eliminating
/// 3+ dispatch boundaries (command buffer submissions on WebGPU).
///
/// Standard (unfused) attention requires 5 dispatches:
///   1. Q @ K^T (MatMul)
///   2. Scale by 1/sqrt(d) (ElementWise)
///   3. Softmax (two-pass: max + exp+sum)
///   4. Scores @ V (MatMul)
///   5. Output projection
///
/// This kernel does steps 1-4 in a single dispatch per head.
/// Each thread computes one output element by iterating over all KV positions.
///
/// For small sequence lengths (≤512, typical for browser LLMs), this is
/// faster than tiled MatMul because the attention matrix is computed and
/// consumed immediately — never materialized in global memory.
/// </summary>
public class FusedAttentionKernel
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _kernel;

    public FusedAttentionKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Fused attention: output = softmax(Q @ K^T / sqrt(d)) @ V
    /// Q [B*H, seqQ, D], K [B*H, seqKV, D], V [B*H, seqKV, D] → output [B*H, seqQ, D]
    /// All heads processed in parallel. Params buffer holds dimensions.
    /// </summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<float, Stride1D.Dense> K,
        ArrayView1D<float, Stride1D.Dense> V,
        ArrayView1D<float, Stride1D.Dense> output,
        int batchHeads, int seqQ, int seqKV, int headDim)
    {
        // params: [batchHeads, seqQ, seqKV, headDim, scale*10000]
        float scale = 1f / MathF.Sqrt(headDim);
        var paramsData = new int[] { batchHeads, seqQ, seqKV, headDim, (int)(scale * 10000f) };
        using var paramsBuf = _accelerator.Allocate1D(paramsData);

        _kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(FusedAttentionImpl);

        // One thread per output element: batchHeads * seqQ * headDim
        _kernel(batchHeads * seqQ * headDim, Q, K, V, output, paramsBuf.View);
    }

    /// <summary>
    /// Per-element fused attention. Each thread computes one value of the output
    /// by iterating over all KV positions (the attention "row").
    /// </summary>
    private static void FusedAttentionImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<float, Stride1D.Dense> K,
        ArrayView1D<float, Stride1D.Dense> V,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int BH = p[0], SQ = p[1], SKV = p[2], D = p[3];
        float scale = p[4] / 10000f;

        // Decompose index: [bh, sq, d]
        int d = idx % D;
        int sq = (idx / D) % SQ;
        int bh = idx / (SQ * D);

        if (bh >= BH) return;

        // Q vector for this position: Q[bh, sq, :]
        int qBase = (bh * SQ + sq) * D;

        // Step 1+2: Compute QK^T scores and find max (for numerical stability)
        float maxScore = -1e10f;
        for (int kv = 0; kv < SKV; kv++)
        {
            int kBase = (bh * SKV + kv) * D;
            float dot = 0f;
            for (int dd = 0; dd < D; dd++)
                dot += Q[qBase + dd] * K[kBase + dd];
            dot *= scale;
            if (dot > maxScore) maxScore = dot;
        }

        // Step 3: Softmax — compute exp(score - max) and sum
        // Step 4: Weighted sum of V at dimension d
        float sumExp = 0f;
        float weightedV = 0f;
        for (int kv = 0; kv < SKV; kv++)
        {
            int kBase = (bh * SKV + kv) * D;
            float dot = 0f;
            for (int dd = 0; dd < D; dd++)
                dot += Q[qBase + dd] * K[kBase + dd];
            dot *= scale;

            float weight = MathF.Exp(dot - maxScore);
            sumExp += weight;
            weightedV += weight * V[(bh * SKV + kv) * D + d];
        }

        output[idx] = weightedV / (sumExp + 1e-10f);
    }
}
