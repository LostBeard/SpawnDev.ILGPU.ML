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

    // Deferred disposal: keep params buffer alive until the NEXT call so the GPU
    // has time to read it. Same pattern as BroadcastBinaryOpND — WebGPU dispatch
    // is async, disposing immediately causes the kernel to read garbage params.
    private MemoryBuffer1D<int, Stride1D.Dense>? _lastParamsBuf;

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
        _lastParamsBuf?.Dispose();
        _lastParamsBuf = _accelerator.Allocate1D(paramsData);

        _kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(FusedAttentionImpl);

        // One thread per output element: batchHeads * seqQ * headDim
        _kernel(batchHeads * seqQ * headDim, Q, K, V, output, _lastParamsBuf.View);
    }

    /// <summary>
    /// Per-element fused attention with Online Softmax (single pass).
    /// Each thread computes one output value by iterating over all KV positions once.
    /// No dot product recomputation — eliminates WebGPU precision divergence
    /// between two passes that caused maxErr 0.44.
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

        int qBase = (bh * SQ + sq) * D;

        // Online Softmax: single pass over KV positions
        // Maintains running max, running sum, and running weighted V
        float runningMax = -1e10f;
        float runningSum = 0f;
        float weightedV = 0f;

        for (int kv = 0; kv < SKV; kv++)
        {
            int kBase = (bh * SKV + kv) * D;
            float dot = 0f;
            for (int dd = 0; dd < D; dd++)
                dot += Q[qBase + dd] * K[kBase + dd];
            float score = dot * scale;

            // Online Softmax update
            if (score > runningMax)
            {
                float correction = MathF.Exp(runningMax - score);
                runningSum *= correction;
                weightedV *= correction;
                runningMax = score;
            }

            float weight = MathF.Exp(score - runningMax);
            runningSum += weight;
            weightedV += weight * V[(bh * SKV + kv) * D + d];
        }

        output[idx] = weightedV / (runningSum + 1e-10f);
    }
}
