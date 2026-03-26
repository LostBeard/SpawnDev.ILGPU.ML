using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Compressed KV cache using TurboQuant (4-bit quantization via FWHT).
/// Stores K and V tensors in ~8x less GPU memory than FP32, enabling
/// longer sequence generation within browser memory limits.
///
/// Pipeline integration:
///   1. After each inference step: Append(present.N.key, present.N.value)
///   2. Before next step: GetDequantizedK/V(layer) → feed as past_key_values
///   3. (Future) Fused attention: pass packed data directly to FusedQuantizedAttention
///
/// Memory savings for GPT-2 (12 layers, 12 heads, 64 head_dim):
///   FP32: 12 * 2 * seq * 768 * 4 bytes = 73 KB/token
///   TurboQuant 4-bit: 12 * 2 * seq * (768/8 * 4 + 768*4 + 16*4) = ~10 KB/token (7.3x compression)
/// </summary>
public class QuantizedKVCache : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly TurboQuantKernels _quant;
    private readonly int _numLayers;
    private readonly int _headDim;
    private readonly int _numHeads;
    private readonly int _maxSeqLen;

    // Per-layer compressed storage
    private readonly LayerCache[] _layers;

    // Shared codebook (Lloyd-Max optimal for Gaussian, symmetric around 0)
    private MemoryBuffer1D<float, Stride1D.Dense>? _codebook;

    // Shared sign vector for sign-flip step
    private MemoryBuffer1D<int, Stride1D.Dense>? _signs;

    // Temp buffers for quantization pipeline (reused across calls)
    private MemoryBuffer1D<float, Stride1D.Dense>? _tempNormalized;
    private MemoryBuffer1D<float, Stride1D.Dense>? _tempFlipped;
    private MemoryBuffer1D<float, Stride1D.Dense>? _tempTransformed;
    private MemoryBuffer1D<int, Stride1D.Dense>? _tempIndices;

    /// <summary>Current sequence position (number of tokens cached).</summary>
    public int CurrentSeqLen { get; private set; }

    /// <summary>Maximum sequence length this cache can hold.</summary>
    public int MaxSeqLen => _maxSeqLen;

    /// <summary>Number of transformer layers.</summary>
    public int NumLayers => _numLayers;

    /// <summary>Whether the cache has any tokens stored.</summary>
    public bool HasCache => CurrentSeqLen > 0;

    /// <summary>
    /// Create a quantized KV cache for a model with explicit KV cache pattern.
    /// </summary>
    /// <param name="accelerator">GPU accelerator</param>
    /// <param name="cacheInfo">KV cache analysis from KVCacheAnalyzer</param>
    /// <param name="maxSeqLen">Maximum sequence length to support (default: 2048)</param>
    public QuantizedKVCache(Accelerator accelerator, KVCacheAnalyzer.KVCacheInfo cacheInfo, int maxSeqLen = 2048)
    {
        _accelerator = accelerator;
        _quant = new TurboQuantKernels(accelerator);
        _numLayers = cacheInfo.NumLayers;
        _maxSeqLen = maxSeqLen;

        // Extract dimensions from first layer's shape [batch, heads, seq, head_dim]
        var firstShape = cacheInfo.Layers[0].Shape;
        if (firstShape != null && firstShape.Length >= 4)
        {
            _numHeads = firstShape[1];
            _headDim = firstShape[3];
        }
        else
        {
            // Fallback: common defaults
            _numHeads = 12;
            _headDim = 64;
        }

        int vecDim = _numHeads * _headDim; // total elements per token per K or V
        int packedDim = vecDim / 8; // 4-bit: 8 values per int32

        // Allocate per-layer storage
        _layers = new LayerCache[_numLayers];
        for (int i = 0; i < _numLayers; i++)
        {
            _layers[i] = new LayerCache
            {
                // Packed 4-bit indices: [maxSeq, vecDim/8]
                K_packed = accelerator.Allocate1D<int>(maxSeqLen * packedDim),
                V_packed = accelerator.Allocate1D<int>(maxSeqLen * packedDim),
                // Per-token norms: [maxSeq]
                K_norms = accelerator.Allocate1D<float>(maxSeqLen),
                V_norms = accelerator.Allocate1D<float>(maxSeqLen),
            };
        }

        // Lloyd-Max codebook for 4-bit (16 centroids), optimal for standard Gaussian
        // These are symmetric: codebook[i] = -codebook[15-i]
        var codebookData = new float[]
        {
            -1.7476f, -1.2562f, -0.9423f, -0.6849f,
            -0.4587f, -0.2505f, -0.0527f,  0.1429f,
             0.3423f,  0.5526f,  0.7841f,  1.0500f,
             1.3709f,  1.7853f,  2.3822f,  3.3604f,
        };
        _codebook = accelerator.Allocate1D(codebookData);

        // Deterministic sign vector (seeded PRNG for reproducibility)
        var rng = new Random(42);
        var signData = new int[vecDim];
        for (int i = 0; i < vecDim; i++)
            signData[i] = rng.Next(2); // 0 or 1
        _signs = accelerator.Allocate1D(signData);

        // Temp buffers for quantization pipeline
        _tempNormalized = accelerator.Allocate1D<float>(vecDim);
        _tempFlipped = accelerator.Allocate1D<float>(vecDim);
        _tempTransformed = accelerator.Allocate1D<float>(vecDim);
        _tempIndices = accelerator.Allocate1D<int>(vecDim);
    }

    /// <summary>
    /// Append one token's KV data for a specific layer.
    /// Quantizes the input tensor and stores it in the compressed cache.
    /// </summary>
    /// <param name="layer">Layer index</param>
    /// <param name="keyData">Key tensor data for this token [numHeads * headDim] floats</param>
    /// <param name="valueData">Value tensor data for this token [numHeads * headDim] floats</param>
    public void Append(int layer, ArrayView1D<float, Stride1D.Dense> keyData,
        ArrayView1D<float, Stride1D.Dense> valueData)
    {
        if (CurrentSeqLen >= _maxSeqLen)
            throw new InvalidOperationException($"KV cache full ({_maxSeqLen} tokens). Increase maxSeqLen or implement sliding window.");

        int vecDim = _numHeads * _headDim;
        int packedDim = vecDim / 8;
        int seqPos = CurrentSeqLen;
        var lc = _layers[layer];

        // Quantize key
        QuantizeVector(keyData, lc.K_packed!.View.SubView(seqPos * packedDim, packedDim),
            lc.K_norms!.View.SubView(seqPos, 1), vecDim);

        // Quantize value
        QuantizeVector(valueData, lc.V_packed!.View.SubView(seqPos * packedDim, packedDim),
            lc.V_norms!.View.SubView(seqPos, 1), vecDim);
    }

    /// <summary>
    /// Call after all layers have been appended for the current token.
    /// Advances the sequence position counter.
    /// </summary>
    public void AdvanceToken()
    {
        CurrentSeqLen++;
    }

    /// <summary>
    /// Get dequantized key tensor for a layer as FP32.
    /// Returns [CurrentSeqLen, numHeads * headDim] on GPU.
    /// </summary>
    public Tensors.Tensor GetDequantizedK(int layer, int[] shape)
    {
        return GetDequantized(layer, isKey: true, shape);
    }

    /// <summary>
    /// Get dequantized value tensor for a layer as FP32.
    /// Returns [CurrentSeqLen, numHeads * headDim] on GPU.
    /// </summary>
    public Tensors.Tensor GetDequantizedV(int layer, int[] shape)
    {
        return GetDequantized(layer, isKey: false, shape);
    }

    /// <summary>
    /// Get packed K data and metadata for fused attention (no dequantization needed).
    /// </summary>
    public (ArrayView1D<int, Stride1D.Dense> packed, ArrayView1D<float, Stride1D.Dense> codebook,
        ArrayView1D<float, Stride1D.Dense> norms, int seqLen, int headDim)
        GetPackedK(int layer)
    {
        var lc = _layers[layer];
        int packedDim = _numHeads * _headDim / 8;
        return (lc.K_packed!.View.SubView(0, CurrentSeqLen * packedDim),
            _codebook!.View, lc.K_norms!.View.SubView(0, CurrentSeqLen),
            CurrentSeqLen, _numHeads * _headDim);
    }

    /// <summary>
    /// Get packed V data and metadata for fused attention (no dequantization needed).
    /// </summary>
    public (ArrayView1D<int, Stride1D.Dense> packed, ArrayView1D<float, Stride1D.Dense> codebook,
        ArrayView1D<float, Stride1D.Dense> norms, int seqLen, int headDim)
        GetPackedV(int layer)
    {
        var lc = _layers[layer];
        int packedDim = _numHeads * _headDim / 8;
        return (lc.V_packed!.View.SubView(0, CurrentSeqLen * packedDim),
            _codebook!.View, lc.V_norms!.View.SubView(0, CurrentSeqLen),
            CurrentSeqLen, _numHeads * _headDim);
    }

    /// <summary>Reset the cache (clear all stored tokens).</summary>
    public void Reset()
    {
        CurrentSeqLen = 0;
    }

    // ═══════════════════════════════════════════════════════════
    //  Internal: quantization pipeline
    // ═══════════════════════════════════════════════════════════

    private void QuantizeVector(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> packedOut,
        ArrayView1D<float, Stride1D.Dense> normOut, int vecDim)
    {
        // Step 1: Normalize → unit vector + store norm
        _quant.Normalize(input, _tempNormalized!.View.SubView(0, vecDim), normOut, 1, vecDim);

        // Step 2: Sign flip (randomize distribution)
        _quant.SignFlip(_tempNormalized!.View.SubView(0, vecDim),
            _tempFlipped!.View.SubView(0, vecDim), _signs!.View, vecDim);

        // Step 3: FWHT (Walsh-Hadamard transform — decorrelate)
        _quant.FWHT.ForwardBatch(_tempFlipped!.View.SubView(0, vecDim),
            _tempTransformed!.View.SubView(0, vecDim), 1, vecDim);

        // Step 4: Quantize to 4-bit codebook indices
        _quant.Quantize(_tempTransformed!.View.SubView(0, vecDim),
            _codebook!.View, _tempIndices!.View.SubView(0, vecDim), vecDim, 16);

        // Step 5: Bit-pack 4-bit indices into int32s
        _quant.BitPack4(_tempIndices!.View.SubView(0, vecDim), packedOut, vecDim);
    }

    private Tensors.Tensor GetDequantized(int layer, bool isKey, int[] shape)
    {
        var lc = _layers[layer];
        int vecDim = _numHeads * _headDim;
        int packedDim = vecDim / 8;
        int totalElems = CurrentSeqLen * vecDim;

        // Allocate output buffer
        var outBuf = _accelerator.Allocate1D<float>(totalElems);

        // Temp buffers for dequant pipeline (per-token)
        using var tempUnpacked = _accelerator.Allocate1D<int>(vecDim);
        using var tempDequant = _accelerator.Allocate1D<float>(vecDim);
        using var tempInvFWHT = _accelerator.Allocate1D<float>(vecDim);
        using var tempInvFlip = _accelerator.Allocate1D<float>(vecDim);

        var packed = isKey ? lc.K_packed! : lc.V_packed!;
        var norms = isKey ? lc.K_norms! : lc.V_norms!;

        for (int t = 0; t < CurrentSeqLen; t++)
        {
            // Step 1: Unpack 4-bit
            _quant.BitUnpack4(packed.View.SubView(t * packedDim, packedDim),
                tempUnpacked.View.SubView(0, vecDim), vecDim);

            // Step 2: Dequantize (codebook lookup)
            _quant.Dequantize(tempUnpacked.View.SubView(0, vecDim), _codebook!.View,
                tempDequant.View.SubView(0, vecDim), vecDim, 16);

            // Step 3: Inverse FWHT
            _quant.FWHT.ForwardBatch(tempDequant.View.SubView(0, vecDim),
                tempInvFWHT.View.SubView(0, vecDim), 1, vecDim);

            // Step 4: Inverse sign flip
            _quant.SignFlip(tempInvFWHT.View.SubView(0, vecDim),
                tempInvFlip.View.SubView(0, vecDim), _signs!.View, vecDim);

            // Step 5: Denormalize (restore magnitude)
            _quant.Denormalize(tempInvFlip.View.SubView(0, vecDim),
                outBuf.View.SubView(t * vecDim, vecDim),
                norms.View.SubView(t, 1), 1, vecDim);
        }

        return new Tensors.Tensor(outBuf.View, shape);
    }

    // ═══════════════════════════════════════════════════════════
    //  Internal types
    // ═══════════════════════════════════════════════════════════

    private class LayerCache
    {
        public MemoryBuffer1D<int, Stride1D.Dense>? K_packed;
        public MemoryBuffer1D<int, Stride1D.Dense>? V_packed;
        public MemoryBuffer1D<float, Stride1D.Dense>? K_norms;
        public MemoryBuffer1D<float, Stride1D.Dense>? V_norms;

        public void Dispose()
        {
            K_packed?.Dispose();
            V_packed?.Dispose();
            K_norms?.Dispose();
            V_norms?.Dispose();
        }
    }

    public void Dispose()
    {
        foreach (var lc in _layers) lc.Dispose();
        _codebook?.Dispose();
        _signs?.Dispose();
        _tempNormalized?.Dispose();
        _tempFlipped?.Dispose();
        _tempTransformed?.Dispose();
        _tempIndices?.Dispose();
    }
}
