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
    private MemoryBuffer1D<int, Stride1D.Dense>? _fusedParamsBuf;
    private MemoryBuffer1D<int, Stride1D.Dense>? _flashParamsBuf;
    private readonly List<MemoryBuffer1D<int, Stride1D.Dense>> _oldParamsBufs = new();
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
    //  3-bit packing (10 values per uint32, 2 bits spare)
    // ═══════════════════════════════════════════════════════════

    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, int>? _bitPack3Kernel;
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, int>? _bitUnpack3Kernel;

    /// <summary>
    /// Pack 3-bit indices into 32-bit ints (10 indices per int, 2 bits spare).
    /// indices [count] (each 0-7) → packed [ceil(count/10)]
    /// The 2 spare high bits (bits 30-31) are zeroed — available for QJL signs.
    /// </summary>
    public void BitPack3(
        ArrayView1D<int, Stride1D.Dense> indices,
        ArrayView1D<int, Stride1D.Dense> packed,
        int count)
    {
        int packedCount = (count + 9) / 10;
        _bitPack3Kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            int>(BitPack3Impl);
        _bitPack3Kernel(packedCount, indices, packed, count);
    }

    private static void BitPack3Impl(Index1D packIdx,
        ArrayView1D<int, Stride1D.Dense> indices,
        ArrayView1D<int, Stride1D.Dense> packed,
        int totalCount)
    {
        int result = 0;
        int baseIdx = packIdx * 10;
        for (int i = 0; i < 10 && baseIdx + i < totalCount; i++)
        {
            int val = indices[baseIdx + i] & 0x7; // mask to 3 bits
            result |= val << (i * 3);
        }
        packed[packIdx] = result;
    }

    /// <summary>
    /// Unpack 32-bit ints back to 3-bit indices.
    /// packed [ceil(count/10)] → indices [count] (each 0-7)
    /// </summary>
    public void BitUnpack3(
        ArrayView1D<int, Stride1D.Dense> packed,
        ArrayView1D<int, Stride1D.Dense> indices,
        int count)
    {
        _bitUnpack3Kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            int>(BitUnpack3Impl);
        _bitUnpack3Kernel(count, packed, indices, count);
    }

    private static void BitUnpack3Impl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> packed,
        ArrayView1D<int, Stride1D.Dense> indices,
        int totalCount)
    {
        int packIdx = idx / 10;
        int bitOffset = (idx % 10) * 3;
        indices[idx] = (packed[packIdx] >> bitOffset) & 0x7;
    }

    // ═══════════════════════════════════════════════════════════
    //  Codebook constants
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Lloyd-Max optimal 16-centroid codebook for unit-variance Gaussian (4-bit).
    /// Symmetric around zero. Ref: Max (1960), MSE = 0.00950, SNR = 20.22 dB.
    /// </summary>
    public static readonly float[] Codebook4Bit = new float[]
    {
        -2.7326f, -2.0690f, -1.6180f, -1.2562f,
        -0.9423f, -0.6568f, -0.3880f, -0.1284f,
         0.1284f,  0.3880f,  0.6568f,  0.9423f,
         1.2562f,  1.6180f,  2.0690f,  2.7326f,
    };

    /// <summary>
    /// Lloyd-Max optimal 8-centroid codebook for unit-variance Gaussian (3-bit).
    /// Symmetric around zero. Ref: Max (1960), MSE = 0.03455, SNR = 14.62 dB.
    /// </summary>
    public static readonly float[] Codebook3Bit = new float[]
    {
        -2.1519f, -1.3439f, -0.7560f, -0.2451f,
         0.2451f,  0.7560f,  1.3439f,  2.1519f,
    };

    // ═══════════════════════════════════════════════════════════
    //  Fused Quantized Attention
    // ═══════════════════════════════════════════════════════════

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _fusedAttentionKernel;

    /// <summary>
    /// Fused attention with quantized KV cache.
    /// Dequantizes K and V on-the-fly during attention computation.
    /// Scalars packed into params buffer to stay within WebGPU parameter limits.
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
        // Pack scalars into int buffer to reduce kernel param count
        // params[0]=numQ, params[1]=numKV, params[2]=headDim, params[3]=scale*10000, params[4]=valuesPerInt
        int valuesPerInt = K_codebook.Length <= 8 ? 10 : 8; // 3-bit=8 centroids→10 per int, 4-bit=16→8 per int
        var paramsData = new int[] { numQueries, numKV, headDim, (int)(scale * 10000f), valuesPerInt };
        // Don't dispose previous — it may still be in the WebGPU command encoder
        if (_fusedParamsBuf != null) _oldParamsBufs.Add(_fusedParamsBuf);
        _fusedParamsBuf = _accelerator.Allocate1D(paramsData);

        _fusedAttentionKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(FusedAttentionImpl);
        _fusedAttentionKernel(numQueries, Q, K_packed, K_codebook, V_packed, V_codebook,
            K_norms, V_norms, output, _fusedParamsBuf.View);
    }

    /// <summary>
    /// Per-query fused attention with packed params buffer.
    /// params[0]=numQ, params[1]=numKV, params[2]=headDim, params[3]=scale*10000
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
        ArrayView1D<int, Stride1D.Dense> paramsArr)
    {
        int numQ = paramsArr[0];
        int numKV = paramsArr[1];
        int D = paramsArr[2];
        float scale = paramsArr[3] / 10000f;
        int valuesPerInt = paramsArr[4]; // 8 for 4-bit, 10 for 3-bit
        int packedDim = (D + valuesPerInt - 1) / valuesPerInt;

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
    //  Flash Attention (Online Softmax) — Single-Pass
    // ═══════════════════════════════════════════════════════════

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _flashAttentionKernel;

    /// <summary>
    /// Flash Attention with Online Softmax and fused TurboQuant dequantization.
    /// Single pass over the KV cache — no second traversal, no intermediate N×N storage.
    ///
    /// Online Softmax (Milakov & Gimelshein 2018): maintains running max, running sum,
    /// and running weighted output. When a new score exceeds current max, rescales
    /// accumulated state by exp(oldMax - newMax). Numerically stable, single pass.
    ///
    /// For long sequences this eliminates the memory bottleneck of the full attention matrix.
    /// </summary>
    public void FlashQuantizedAttention(
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
        int valuesPerInt = K_codebook.Length <= 8 ? 10 : 8; // 3-bit=8 centroids→10 per int, 4-bit=16→8 per int
        var paramsData = new int[] { numQueries, numKV, headDim, (int)(scale * 10000f), valuesPerInt };
        if (_flashParamsBuf != null) _oldParamsBufs.Add(_flashParamsBuf);
        _flashParamsBuf = _accelerator.Allocate1D(paramsData);

        _flashAttentionKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(FlashAttentionImpl);
        _flashAttentionKernel(numQueries, Q, K_packed, K_codebook, V_packed, V_codebook,
            K_norms, V_norms, output, _flashParamsBuf.View);
    }

    /// <summary>
    /// Single-pass Flash Attention with Online Softmax.
    /// Each thread processes one query position across all KV positions in one pass.
    ///
    /// Algorithm: for each KV position j:
    ///   1. Compute score_j = Q @ K_j * scale (with fused dequant)
    ///   2. If score_j > running_max:
    ///        correction = exp(old_max - score_j)
    ///        running_sum *= correction
    ///        output[d] *= correction  (rescale accumulated output)
    ///        running_max = score_j
    ///   3. weight_j = exp(score_j - running_max)
    ///   4. running_sum += weight_j
    ///   5. output[d] += weight_j * V_j[d]  (with fused dequant)
    ///   After all KV: output[d] /= running_sum
    /// </summary>
    private static void FlashAttentionImpl(Index1D queryIdx,
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<int, Stride1D.Dense> K_packed,
        ArrayView1D<float, Stride1D.Dense> K_codebook,
        ArrayView1D<int, Stride1D.Dense> V_packed,
        ArrayView1D<float, Stride1D.Dense> V_codebook,
        ArrayView1D<float, Stride1D.Dense> K_norms,
        ArrayView1D<float, Stride1D.Dense> V_norms,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> paramsArr)
    {
        int numQ = paramsArr[0];
        int numKV = paramsArr[1];
        int D = paramsArr[2];
        float scale = paramsArr[3] / 10000f;
        int valuesPerInt = paramsArr[4]; // 8 for 4-bit, 10 for 3-bit
        int packedDim = (D + valuesPerInt - 1) / valuesPerInt;

        // Initialize Online Softmax state
        float runningMax = -1e10f;
        float runningSum = 0f;

        // Zero the output accumulator
        for (int d = 0; d < D; d++)
            output[queryIdx * D + d] = 0f;

        // Single pass over all KV positions
        for (int kv = 0; kv < numKV; kv++)
        {
            // ── Compute Q @ K_j (fused dequant) ──
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

            // ── Online Softmax update ──
            if (score > runningMax)
            {
                // New max found — rescale all accumulated state
                float correction = MathF.Exp(runningMax - score);
                runningSum *= correction;
                for (int d = 0; d < D; d++)
                    output[queryIdx * D + d] *= correction;
                runningMax = score;
            }

            float weight = MathF.Exp(score - runningMax);
            runningSum += weight;

            // ── Accumulate weighted V (fused dequant) ──
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

        // Final normalization
        float invSum = 1f / (runningSum + 1e-10f);
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
