using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// TurboQuant GPU kernels for data-oblivious KV cache quantization.
/// Pipeline: normalize → sign-flip → FWHT → nearest centroid → bit-pack.
/// Reverse: unpack → centroid lookup → inverse FWHT → sign-flip → denormalize.
///
/// 4-bit quantization gives 7.9x KV cache compression with zero accuracy loss
/// and no calibration data needed.
/// </summary>
public class TurboQuantKernels
{
    private readonly Accelerator _accelerator;
    private readonly FWHTKernel _fwht;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int>? _normalizeKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int>? _denormalizeKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _signFlipKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, int>? _quantizeKernel;
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int>? _dequantizeKernel;

    public TurboQuantKernels(Accelerator accelerator)
    {
        _accelerator = accelerator;
        _fwht = new FWHTKernel(accelerator);
    }

    // ═══════════════════════════════════════════════════════════
    //  Step 1: Normalize (store norm separately)
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Normalize vectors to unit length. Stores norms separately for reconstruction.
    /// input [numVecs, d] → output [numVecs, d] (unit vectors) + norms [numVecs]
    /// </summary>
    public void Normalize(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> norms,
        int numVecs, int d)
    {
        _normalizeKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int>(NormalizeImpl);
        _normalizeKernel(numVecs, input, output, norms, d);
    }

    private static void NormalizeImpl(Index1D vecIdx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> norms,
        int d)
    {
        int offset = vecIdx * d;
        float sumSq = 0f;
        for (int i = 0; i < d; i++)
            sumSq += input[offset + i] * input[offset + i];
        float norm = MathF.Sqrt(sumSq);
        norms[vecIdx] = norm;
        float invNorm = norm > 1e-12f ? 1f / norm : 0f;
        for (int i = 0; i < d; i++)
            output[offset + i] = input[offset + i] * invNorm;
    }

    // ═══════════════════════════════════════════════════════════
    //  Step 2: Sign flip (deterministic random signs)
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Apply element-wise random sign flips. Deterministic given the sign vector.
    /// </summary>
    public void SignFlip(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> signs,
        int count)
    {
        _signFlipKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(SignFlipImpl);
        _signFlipKernel(count, input, output, signs);
    }

    private static void SignFlipImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> signs)
    {
        output[idx] = signs[idx % signs.IntLength] != 0 ? -input[idx] : input[idx];
    }

    // ═══════════════════════════════════════════════════════════
    //  Step 3: FWHT (delegates to FWHTKernel)
    // ═══════════════════════════════════════════════════════════

    /// <summary>Apply FWHT to each vector in the batch.</summary>
    public FWHTKernel FWHT => _fwht;

    // ═══════════════════════════════════════════════════════════
    //  Step 4: Quantize (nearest centroid lookup)
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Quantize each element to its nearest codebook centroid.
    /// input [count], codebook [numCentroids] → indices [count]
    /// </summary>
    public void Quantize(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> codebook,
        ArrayView1D<int, Stride1D.Dense> indices,
        int count, int numCentroids)
    {
        _quantizeKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, int>(QuantizeImpl);
        _quantizeKernel(count, input, codebook, indices, numCentroids);
    }

    private static void QuantizeImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> codebook,
        ArrayView1D<int, Stride1D.Dense> indices,
        int K)
    {
        float val = input[idx];
        int bestIdx = 0;
        float bestDist = MathF.Abs(val - codebook[0]);
        for (int k = 1; k < K; k++)
        {
            float dist = MathF.Abs(val - codebook[k]);
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = k;
            }
        }
        indices[idx] = bestIdx;
    }

    // ═══════════════════════════════════════════════════════════
    //  Step 5: Dequantize (centroid lookup)
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Dequantize: replace each index with its codebook centroid value.
    /// indices [count], codebook [numCentroids] → output [count]
    /// </summary>
    public void Dequantize(
        ArrayView1D<int, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> codebook,
        ArrayView1D<float, Stride1D.Dense> output,
        int count, int numCentroids)
    {
        _dequantizeKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int>(DequantizeImpl);
        _dequantizeKernel(count, indices, codebook, output, numCentroids);
    }

    private static void DequantizeImpl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> codebook,
        ArrayView1D<float, Stride1D.Dense> output,
        int K)
    {
        output[idx] = codebook[indices[idx]];
    }

    // ═══════════════════════════════════════════════════════════
    //  Step 6: Bit-pack (compact storage)
    // ═══════════════════════════════════════════════════════════

    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, int>? _bitPack4Kernel;
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, int>? _bitUnpack4Kernel;

    /// <summary>
    /// Pack 4-bit indices into 32-bit ints (8 indices per int).
    /// indices [count] (each 0-15) → packed [count/8]
    /// </summary>
    public void BitPack4(
        ArrayView1D<int, Stride1D.Dense> indices,
        ArrayView1D<int, Stride1D.Dense> packed,
        int count)
    {
        int packedCount = (count + 7) / 8;
        _bitPack4Kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            int>(BitPack4Impl);
        _bitPack4Kernel(packedCount, indices, packed, count);
    }

    private static void BitPack4Impl(Index1D packIdx,
        ArrayView1D<int, Stride1D.Dense> indices,
        ArrayView1D<int, Stride1D.Dense> packed,
        int totalCount)
    {
        int result = 0;
        int baseIdx = packIdx * 8;
        for (int i = 0; i < 8 && baseIdx + i < totalCount; i++)
        {
            int val = indices[baseIdx + i] & 0xF; // mask to 4 bits
            result |= val << (i * 4);
        }
        packed[packIdx] = result;
    }

    /// <summary>
    /// Unpack 32-bit ints back to 4-bit indices.
    /// packed [count/8] → indices [count] (each 0-15)
    /// </summary>
    public void BitUnpack4(
        ArrayView1D<int, Stride1D.Dense> packed,
        ArrayView1D<int, Stride1D.Dense> indices,
        int count)
    {
        _bitUnpack4Kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            int>(BitUnpack4Impl);
        _bitUnpack4Kernel(count, packed, indices, count);
    }

    private static void BitUnpack4Impl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> packed,
        ArrayView1D<int, Stride1D.Dense> indices,
        int totalCount)
    {
        int packIdx = idx / 8;
        int bitOffset = (idx % 8) * 4;
        indices[idx] = (packed[packIdx] >> bitOffset) & 0xF;
    }

    // ═══════════════════════════════════════════════════════════
    //  Fused Quantized Attention
    // ═══════════════════════════════════════════════════════════

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int, int, float>? _fusedAttentionKernel;

    /// <summary>
    /// Fused attention with quantized KV cache.
    /// Dequantizes K and V on-the-fly during attention computation.
    /// Q [numQueries, headDim] (float)
    /// K_packed [numKV, headDim/8] (4-bit packed int), K_codebook [16] (float)
    /// V_packed [numKV, headDim/8] (4-bit packed int), V_codebook [16] (float)
    /// K_norms [numKV], V_norms [numKV] (float)
    /// → output [numQueries, headDim] (float)
    /// </summary>
    public void FusedQuantizedAttention(
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<int, Stride1D.Dense> K_packed,
        ArrayView1D<float, Stride1D.Dense> K_codebook,
        ArrayView1D<int, Stride1D.Dense> V_packed,
        ArrayView1D<float, Stride1D.Dense> V_codebook,
        ArrayView1D<float, Stride1D.Dense> K_norms,
        ArrayView1D<float, Stride1D.Dense> V_norms,
        ArrayView1D<float, Stride1D.Dense> output,
        int numQueries, int numKV, int headDim, float scale)
    {
        _fusedAttentionKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int, float>(FusedAttentionImpl);
        _fusedAttentionKernel(numQueries, Q, K_packed, K_codebook, V_packed, V_codebook,
            K_norms, V_norms, output, numQueries, numKV, headDim, scale);
    }

    /// <summary>
    /// Per-query fused attention: compute QK^T scores, softmax, then weighted sum of V.
    /// K and V are dequantized on-the-fly from 4-bit packed representation.
    /// </summary>
    private static void FusedAttentionImpl(Index1D queryIdx,
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<int, Stride1D.Dense> K_packed,
        ArrayView1D<float, Stride1D.Dense> K_codebook,
        ArrayView1D<int, Stride1D.Dense> V_packed,
        ArrayView1D<float, Stride1D.Dense> V_codebook,
        ArrayView1D<float, Stride1D.Dense> K_norms,
        ArrayView1D<float, Stride1D.Dense> V_norms,
        ArrayView1D<float, Stride1D.Dense> output,
        int numQ, int numKV, int D, float scale)
    {
        int packedDim = D / 8; // 4-bit: 8 values per int

        // Step 1: Compute attention scores QK^T (dequantize K on-the-fly)
        // Use a fixed-size local array for scores (max 1024 KV positions)
        float maxScore = -1e10f;

        // First pass: compute scores and find max
        // We'll do two passes to avoid needing a large local array
        // Pass 1: find max score
        for (int kv = 0; kv < numKV; kv++)
        {
            float dot = 0f;
            float kNorm = K_norms[kv];
            for (int p = 0; p < packedDim; p++)
            {
                int packed = K_packed[kv * packedDim + p];
                for (int b = 0; b < 8 && p * 8 + b < D; b++)
                {
                    int idx = (packed >> (b * 4)) & 0xF;
                    float kVal = K_codebook[idx] * kNorm;
                    dot += Q[queryIdx * D + p * 8 + b] * kVal;
                }
            }
            float score = dot * scale;
            if (score > maxScore) maxScore = score;
        }

        // Pass 2: compute exp(score - max) and sum for softmax, and accumulate weighted V
        float sumExp = 0f;
        // Zero the output
        for (int d = 0; d < D; d++)
            output[queryIdx * D + d] = 0f;

        for (int kv = 0; kv < numKV; kv++)
        {
            // Recompute score (trading compute for memory — no local array needed)
            float dot = 0f;
            float kNorm = K_norms[kv];
            for (int p = 0; p < packedDim; p++)
            {
                int packed = K_packed[kv * packedDim + p];
                for (int b = 0; b < 8 && p * 8 + b < D; b++)
                {
                    int idx = (packed >> (b * 4)) & 0xF;
                    float kVal = K_codebook[idx] * kNorm;
                    dot += Q[queryIdx * D + p * 8 + b] * kVal;
                }
            }
            float weight = MathF.Exp(dot * scale - maxScore);
            sumExp += weight;

            // Accumulate weighted V (dequantize on-the-fly)
            float vNorm = V_norms[kv];
            for (int p = 0; p < packedDim; p++)
            {
                int packed = V_packed[kv * packedDim + p];
                for (int b = 0; b < 8 && p * 8 + b < D; b++)
                {
                    int idx = (packed >> (b * 4)) & 0xF;
                    float vVal = V_codebook[idx] * vNorm;
                    output[queryIdx * D + p * 8 + b] += weight * vVal;
                }
            }
        }

        // Normalize by softmax sum
        float invSum = 1f / (sumExp + 1e-10f);
        for (int d = 0; d < D; d++)
            output[queryIdx * D + d] *= invSum;
    }

    // ═══════════════════════════════════════════════════════════
    //  Denormalize (restore original scale)
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Denormalize: multiply unit vectors by stored norms.
    /// input [numVecs, d] + norms [numVecs] → output [numVecs, d]
    /// </summary>
    public void Denormalize(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> norms,
        int numVecs, int d)
    {
        _denormalizeKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int>(DenormalizeImpl);
        _denormalizeKernel(numVecs, input, output, norms, d);
    }

    private static void DenormalizeImpl(Index1D vecIdx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> norms,
        int d)
    {
        int offset = vecIdx * d;
        float norm = norms[vecIdx];
        for (int i = 0; i < d; i++)
            output[offset + i] = input[offset + i] * norm;
    }
}
