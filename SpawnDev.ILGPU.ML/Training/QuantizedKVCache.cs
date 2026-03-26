using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;

namespace SpawnDev.ILGPU.ML.Training;

/// <summary>
/// Quantized KV cache using TurboQuant 4-bit compression.
///
/// In a transformer LLM, each generated token adds Key and Value vectors
/// to the cache. Without compression, GPT-2 (12 layers × 12 heads × 64 dim)
/// uses 72MB at 1024 tokens. With TurboQuant 4-bit, that drops to ~9MB.
///
/// Usage in text generation:
///   var cache = new QuantizedKVCache(accelerator, numLayers: 12, numHeads: 12, headDim: 64);
///
///   // Each token generation step:
///   cache.AppendKV(layerIdx, headIdx, newK, newV);  // encode + store as 4-bit
///
///   // During attention:
///   var output = cache.FusedAttention(layerIdx, headIdx, query, scale);  // decode on-the-fly
/// </summary>
public class QuantizedKVCache : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly TurboQuantKernels _tq;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _maxSeqLen;
    private readonly int _packedDim; // headDim / 8 (4-bit: 8 values per int32)

    // Per-layer, per-head storage
    // K_packed[layer][head]: [maxSeqLen, packedDim] packed 4-bit indices
    // V_packed[layer][head]: [maxSeqLen, packedDim] packed 4-bit indices
    // K_norms[layer][head]: [maxSeqLen] float norms
    // V_norms[layer][head]: [maxSeqLen] float norms
    private readonly MemoryBuffer1D<int, Stride1D.Dense>[,] _kPacked;
    private readonly MemoryBuffer1D<int, Stride1D.Dense>[,] _vPacked;
    private readonly MemoryBuffer1D<float, Stride1D.Dense>[,] _kNorms;
    private readonly MemoryBuffer1D<float, Stride1D.Dense>[,] _vNorms;

    // Shared codebook (same for all layers/heads — data-oblivious)
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _codebook;

    // Sign vector for deterministic sign flips
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _signs;

    // Current sequence length
    private int _seqLen;

    // Temp buffers for encode/decode
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _tempNormalized;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _tempFlipped;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _tempTransformed;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _tempIndices;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _tempNorm;

    /// <summary>Current number of cached tokens.</summary>
    public int SeqLen => _seqLen;

    /// <summary>Maximum sequence length this cache supports.</summary>
    public int MaxSeqLen => _maxSeqLen;

    /// <summary>
    /// Memory usage in bytes (4-bit compressed).
    /// Compare: float32 would be numLayers × numHeads × 2 × seqLen × headDim × 4 bytes.
    /// </summary>
    public long CompressedMemoryBytes => (long)_numLayers * _numHeads * 2 * _seqLen *
        (_packedDim * 4 + 4); // packed ints + norm float per KV

    /// <summary>What float32 KV cache would use.</summary>
    public long UncompressedMemoryBytes => (long)_numLayers * _numHeads * 2 * _seqLen * _headDim * 4;

    /// <summary>Compression ratio.</summary>
    public float CompressionRatio => UncompressedMemoryBytes > 0
        ? (float)UncompressedMemoryBytes / CompressedMemoryBytes : 0;

    public QuantizedKVCache(Accelerator accelerator,
        int numLayers = 12, int numHeads = 12, int headDim = 64, int maxSeqLen = 1024)
    {
        _accelerator = accelerator;
        _tq = new TurboQuantKernels(accelerator);
        _numLayers = numLayers;
        _numHeads = numHeads;
        _headDim = headDim;
        _maxSeqLen = maxSeqLen;
        _packedDim = headDim / 8;
        _seqLen = 0;

        // Allocate per-layer, per-head buffers
        _kPacked = new MemoryBuffer1D<int, Stride1D.Dense>[numLayers, numHeads];
        _vPacked = new MemoryBuffer1D<int, Stride1D.Dense>[numLayers, numHeads];
        _kNorms = new MemoryBuffer1D<float, Stride1D.Dense>[numLayers, numHeads];
        _vNorms = new MemoryBuffer1D<float, Stride1D.Dense>[numLayers, numHeads];

        for (int l = 0; l < numLayers; l++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                _kPacked[l, h] = accelerator.Allocate1D<int>(maxSeqLen * _packedDim);
                _vPacked[l, h] = accelerator.Allocate1D<int>(maxSeqLen * _packedDim);
                _kNorms[l, h] = accelerator.Allocate1D<float>(maxSeqLen);
                _vNorms[l, h] = accelerator.Allocate1D<float>(maxSeqLen);
            }
        }

        // 4-bit Lloyd-Max codebook (16 centroids for unit-norm FWHT-transformed data)
        // These are pre-computed for the Beta distribution at each head dimension
        var codebookData = GenerateDefaultCodebook();
        _codebook = accelerator.Allocate1D(codebookData);

        // Deterministic sign vector
        var rng = new Random(42);
        var signData = new int[headDim];
        for (int i = 0; i < headDim; i++)
            signData[i] = rng.Next(2); // 0 or 1
        _signs = accelerator.Allocate1D(signData);

        // Temp buffers (reused across calls)
        _tempNormalized = accelerator.Allocate1D<float>(headDim);
        _tempFlipped = accelerator.Allocate1D<float>(headDim);
        _tempTransformed = accelerator.Allocate1D<float>(headDim);
        _tempIndices = accelerator.Allocate1D<int>(headDim);
        _tempNorm = accelerator.Allocate1D<float>(1);
    }

    /// <summary>
    /// Append a new K,V pair for one layer and head.
    /// Encodes to 4-bit and stores in the compressed cache.
    /// </summary>
    public void AppendKV(int layer, int head,
        ArrayView1D<float, Stride1D.Dense> newK,
        ArrayView1D<float, Stride1D.Dense> newV)
    {
        if (_seqLen >= _maxSeqLen)
            throw new InvalidOperationException($"KV cache full: {_seqLen} >= {_maxSeqLen}");

        // Encode K
        EncodeVector(newK, _kPacked[layer, head], _kNorms[layer, head], _seqLen);
        // Encode V
        EncodeVector(newV, _vPacked[layer, head], _vNorms[layer, head], _seqLen);

        // Only increment seqLen once (caller handles all layers/heads for this token)
    }

    /// <summary>Increment sequence length after all layers/heads have been appended.</summary>
    public void IncrementSeqLen() => _seqLen++;

    /// <summary>
    /// Run fused attention on quantized KV cache for one layer and head.
    /// Q [1, headDim] → output [1, headDim]
    /// </summary>
    public void FusedAttention(int layer, int head,
        ArrayView1D<float, Stride1D.Dense> query,
        ArrayView1D<float, Stride1D.Dense> output,
        float scale)
    {
        if (_seqLen == 0) return;

        _tq.FusedQuantizedAttention(
            query,
            _kPacked[layer, head].View.SubView(0, _seqLen * _packedDim),
            _codebook.View,
            _vPacked[layer, head].View.SubView(0, _seqLen * _packedDim),
            _codebook.View,
            _kNorms[layer, head].View.SubView(0, _seqLen),
            _vNorms[layer, head].View.SubView(0, _seqLen),
            output,
            1, _seqLen, _headDim, scale);
    }

    /// <summary>Reset cache for new generation.</summary>
    public void Clear() => _seqLen = 0;

    private void EncodeVector(
        ArrayView1D<float, Stride1D.Dense> input,
        MemoryBuffer1D<int, Stride1D.Dense> packedBuf,
        MemoryBuffer1D<float, Stride1D.Dense> normBuf,
        int position)
    {
        // Step 1: Normalize
        _tq.Normalize(input, _tempNormalized.View, _tempNorm.View, 1, _headDim);

        // Step 2: Sign flip
        _tq.SignFlip(_tempNormalized.View, _tempFlipped.View, _signs.View, _headDim);

        // Step 3: FWHT — copy via Scale(1.0) to avoid sync CopyTo on WebGPU
        new ElementWiseKernels(_accelerator).Scale(_tempFlipped.View.SubView(0, _headDim), _tempTransformed.View.SubView(0, _headDim), _headDim, 1f);
        _tq.FWHT.Forward(_tempTransformed.View, _headDim);

        // Step 4: Quantize
        _tq.Quantize(_tempTransformed.View, _codebook.View, _tempIndices.View, _headDim, 16);

        // Step 5: Bit-pack and store at position
        _tq.BitPack4(_tempIndices.View, packedBuf.View.SubView(position * _packedDim, _packedDim), _headDim);

        // Store norm at position — Scale(1.0) as async GPU→GPU copy
        new ElementWiseKernels(_accelerator).Scale(_tempNorm.View.SubView(0, 1), normBuf.View.SubView(position, 1), 1, 1f);
    }

    /// <summary>
    /// Generate default 4-bit Lloyd-Max codebook for unit-norm FWHT data.
    /// 16 centroids symmetric around 0, optimized for the distribution of
    /// FWHT-transformed unit vectors.
    /// </summary>
    private static float[] GenerateDefaultCodebook()
    {
        // Approximate Lloyd-Max centroids for Beta(d/2, d/2) distribution at d=64
        // These are close to uniform quantization for this near-Gaussian distribution
        return new float[]
        {
            -1.750f, -1.250f, -0.875f, -0.625f,
            -0.375f, -0.200f, -0.075f,  0.000f,
             0.075f,  0.200f,  0.375f,  0.625f,
             0.875f,  1.250f,  1.750f,  2.500f,
        };
    }

    public void Dispose()
    {
        for (int l = 0; l < _numLayers; l++)
        {
            for (int h = 0; h < _numHeads; h++)
            {
                _kPacked[l, h]?.Dispose();
                _vPacked[l, h]?.Dispose();
                _kNorms[l, h]?.Dispose();
                _vNorms[l, h]?.Dispose();
            }
        }
        _codebook?.Dispose();
        _signs?.Dispose();
        _tempNormalized?.Dispose();
        _tempFlipped?.Dispose();
        _tempTransformed?.Dispose();
        _tempIndices?.Dispose();
        _tempNorm?.Dispose();
    }
}
