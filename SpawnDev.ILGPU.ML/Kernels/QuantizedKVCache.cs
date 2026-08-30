using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Quantization mode for KV cache compression.
/// </summary>
public enum KVQuantMode
{
    /// <summary>
    /// Auto-select the best mode. Maps to TurboQuant3BitQJL — the paper's recommended
    /// default with unbiased attention inner products at same storage as 4-bit.
    /// Benchmark validated: 0.9944 cosine similarity on CUDA (March 2026).
    /// </summary>
    Auto,

    /// <summary>
    /// 4-bit quantization: 16 Lloyd-Max centroids, 8 values per uint32.
    /// ~4x compression. Safe default with good accuracy.
    /// </summary>
    TurboQuant4Bit,

    /// <summary>
    /// 3-bit quantization: 8 Lloyd-Max centroids, 10 values per uint32.
    /// ~5.3x compression. Maximum memory savings for browser environments.
    /// </summary>
    TurboQuant3Bit,

    /// <summary>
    /// 3-bit + 1-bit QJL error correction: 8 centroids + residual sign bit, 8 values per uint32.
    /// ~4x compression (same as 4-bit) but with unbiased attention inner products.
    /// Paper's "zero-loss" mode — best accuracy for long-context generation.
    /// </summary>
    TurboQuant3BitQJL,
}

/// <summary>
/// Compressed KV cache using TurboQuant quantization via FWHT.
/// Supports 4-bit, 3-bit, and 3-bit+QJL modes via <see cref="KVQuantMode"/>.
///
/// Pipeline integration:
///   1. After each inference step: Append(present.N.key, present.N.value)
///   2. Before next step: GetDequantizedK/V(layer) → feed as past_key_values
///   3. (Future) Fused attention: pass packed data directly to FusedQuantizedAttention
///
/// Memory savings for GPT-2 (12 layers, 12 heads, 64 head_dim) at 1024 tokens:
///   FP32:            73 KB/token
///   TurboQuant4Bit:  ~10 KB/token (7.3x compression)
///   TurboQuant3Bit:  ~7 KB/token  (10.4x compression)
///   TurboQuant3BitQJL: ~10 KB/token (7.3x, same storage as 4-bit, better accuracy)
/// </summary>
public class QuantizedKVCache : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly TurboQuantKernels _quant;
    private readonly int _numLayers;
    private readonly int _headDim;
    private readonly int _numHeads;
    private readonly int _maxSeqLen;
    private readonly KVQuantMode _mode;
    private readonly int _bitsPerValue; // 3 or 4
    private readonly int _valuesPerInt; // 8 (4-bit/3+1) or 10 (pure 3-bit)

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

    // Temp buffers for the DEQUANT pipeline (GetDequantized). Member buffers for the same
    // reason as the FlashAttention temps below: GetDequantized is synchronous, so a
    // method-local `using var` is disposed while its dispatches are still queued on the
    // browser backends (WebGPU encoder not yet submitted; Wasm worker queue not yet
    // drained -> freed-SharedArrayBuffer read). Allocated lazily; vecDim is fixed.
    private MemoryBuffer1D<int, Stride1D.Dense>? _dqUnpacked;
    private MemoryBuffer1D<float, Stride1D.Dense>? _dqDequant;
    private MemoryBuffer1D<float, Stride1D.Dense>? _dqInvFWHT;
    private MemoryBuffer1D<float, Stride1D.Dense>? _dqInvFlip;
    // Outgrown dequant output buffers awaiting safe disposal: a smaller buffer may still
    // be referenced by queued dispatches on the browser backends, so it retires at
    // cache Dispose() instead of at grow time (geometric growth keeps this list tiny).
    private readonly List<IDisposable> _retiredBuffers = new();

    // Pooled temp buffers for FlashAttention. These MUST be member buffers (not method-local
    // `using var`): on WebGPU the kernels dispatched by FlashAttention are batched in the
    // command encoder and not submitted until the CALLER calls SynchronizeAsync. A method-local
    // `using var` is disposed when FlashAttention returns — before that submit — so the GPU
    // buffer is destroyed while still referenced by a pending dispatch ("Buffer used in submit
    // while destroyed"). Pooling them on the instance keeps them alive past the flush and avoids
    // an internal Synchronize that would defeat WebGPU command batching. Reused across calls
    // (single-token decode reuses the same scratch); grown if a larger numQueries arrives.
    private MemoryBuffer1D<float, Stride1D.Dense>? _faQNormalized;
    private MemoryBuffer1D<float, Stride1D.Dense>? _faQNorms;
    private MemoryBuffer1D<float, Stride1D.Dense>? _faQFlipped;
    private MemoryBuffer1D<float, Stride1D.Dense>? _faQTransformed;
    private MemoryBuffer1D<float, Stride1D.Dense>? _faAttnOut;
    private MemoryBuffer1D<float, Stride1D.Dense>? _faInvFWHT;
    private int _faElemCapacity;   // capacity of the numQueries*vecDim buffers
    private int _faQueryCapacity;  // capacity of the per-query qNorms buffer

    /// <summary>Current sequence position (number of tokens cached).</summary>
    public int CurrentSeqLen { get; private set; }

    /// <summary>Maximum sequence length this cache can hold.</summary>
    public int MaxSeqLen => _maxSeqLen;

    /// <summary>Number of transformer layers.</summary>
    public int NumLayers => _numLayers;

    /// <summary>Whether the cache has any tokens stored.</summary>
    public bool HasCache => CurrentSeqLen > 0;

    /// <summary>The quantization mode this cache is using.</summary>
    public KVQuantMode Mode => _mode;

    /// <summary>
    /// Create a quantized KV cache for a model with explicit KV cache pattern.
    /// </summary>
    /// <param name="accelerator">GPU accelerator</param>
    /// <param name="cacheInfo">KV cache analysis from KVCacheAnalyzer</param>
    /// <param name="maxSeqLen">Maximum sequence length to support (default: 2048)</param>
    /// <param name="quantMode">Quantization mode (default: Auto)</param>
    public QuantizedKVCache(Accelerator accelerator, KVCacheAnalyzer.KVCacheInfo cacheInfo,
        int maxSeqLen = 2048, KVQuantMode quantMode = KVQuantMode.Auto)
    {
        _accelerator = accelerator;
        _quant = new TurboQuantKernels(accelerator);
        _numLayers = cacheInfo.NumLayers;
        _maxSeqLen = maxSeqLen;

        // Resolve Auto to concrete mode — 3+1 QJL is the paper's recommended default:
        // same storage as 4-bit but with unbiased attention inner products (Google Research, March 2026)
        _mode = quantMode == KVQuantMode.Auto ? KVQuantMode.TurboQuant3BitQJL : quantMode;
        _bitsPerValue = _mode == KVQuantMode.TurboQuant3Bit ? 3 : 4;
        _valuesPerInt = _mode == KVQuantMode.TurboQuant3Bit ? 10 : 8;

        // Extract dimensions from first layer's shape [batch, heads, seq, head_dim].
        // ⚠️ NO FALLBACK. This used to default to 12 heads x 64 when the shape was unknown, which sizes
        // every packed buffer in this cache. The default is exactly right for the GPT-2 family and wrong
        // for everything else, and nothing downstream re-checks it: GraphExecutor computes the REAL vector
        // width from the present tensor and hands over a view of that width, so a cache built on the guess
        // reads past the end of it (MEASURED 2026-08-30 on whisper-tiny, 6 heads: cache assumed 768,
        // executor supplied 384, TurboQuant's ComputeNorms ran off the buffer). Guessing the geometry of a
        // model's KV cache is not a recoverable position - throw, and let the caller disable the cache
        // rather than quantize into a wrongly-sized buffer.
        var firstShape = cacheInfo.Layers[0].Shape;
        if (firstShape == null || firstShape.Length < 4)
            throw new InvalidOperationException(
                $"KV cache geometry unknown: layer 0's past input '{cacheInfo.Layers[0].PastKeyInput}' has " +
                $"shape [{(firstShape == null ? "null" : string.Join(",", firstShape))}], expected rank 4 " +
                "[batch, heads, seq, head_dim]. TurboQuant cannot size its packed buffers without it.");
        _numHeads = firstShape[1];
        _headDim = firstShape[3];
        if (_numHeads <= 0 || _headDim <= 0)
            throw new InvalidOperationException(
                $"KV cache geometry invalid: heads={_numHeads}, headDim={_headDim} from " +
                $"'{cacheInfo.Layers[0].PastKeyInput}' shape [{string.Join(",", firstShape)}].");

        int vecDim = _numHeads * _headDim; // total elements per token per K or V
        int packedDim = (vecDim + _valuesPerInt - 1) / _valuesPerInt;

        // Allocate per-layer storage
        _layers = new LayerCache[_numLayers];
        for (int i = 0; i < _numLayers; i++)
        {
            _layers[i] = new LayerCache
            {
                K_packed = accelerator.Allocate1D<int>(maxSeqLen * packedDim),
                V_packed = accelerator.Allocate1D<int>(maxSeqLen * packedDim),
                K_norms = accelerator.Allocate1D<float>(maxSeqLen),
                V_norms = accelerator.Allocate1D<float>(maxSeqLen),
            };
        }

        // Select codebook based on mode
        var codebookData = _mode == KVQuantMode.TurboQuant4Bit
            ? TurboQuantKernels.Codebook4Bit
            : TurboQuantKernels.Codebook3Bit;
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
        int packedDim = (vecDim + _valuesPerInt - 1) / _valuesPerInt;
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
        int packedDim = (_numHeads * _headDim + _valuesPerInt - 1) / _valuesPerInt;
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
        int packedDim = (_numHeads * _headDim + _valuesPerInt - 1) / _valuesPerInt;
        return (lc.V_packed!.View.SubView(0, CurrentSeqLen * packedDim),
            _codebook!.View, lc.V_norms!.View.SubView(0, CurrentSeqLen),
            CurrentSeqLen, _numHeads * _headDim);
    }

    /// <summary>
    /// Run Flash Attention (Online Softmax) directly on the quantized KV cache for a layer.
    /// Single-pass attention with fused dequantization — no intermediate buffers needed.
    ///
    /// The KV cache stores data in the Hadamard domain (normalized → sign-flipped → FWHT → scaled).
    /// Since FWHT preserves inner products (Q·K = HQ·HK), we transform Q into the same domain,
    /// run attention there, then inverse-transform the output back to the original domain.
    /// </summary>
    /// <param name="layer">Transformer layer index</param>
    /// <param name="query">Query tensor [numQueries * vecDim] in original domain</param>
    /// <param name="output">Output tensor [numQueries * vecDim] in original domain</param>
    /// <param name="numQueries">Number of query positions</param>
    /// <param name="scale">Attention scale factor (typically 1/√headDim)</param>
    public void FlashAttention(int layer, ArrayView1D<float, Stride1D.Dense> query,
        ArrayView1D<float, Stride1D.Dense> output, int numQueries, float scale)
    {
        int vecDim = _numHeads * _headDim;
        var kPacked = GetPackedK(layer);
        var vPacked = GetPackedV(layer);

        _vCodebookCopy ??= _accelerator.Allocate1D(
            _mode == KVQuantMode.TurboQuant4Bit
                ? TurboQuantKernels.Codebook4Bit
                : TurboQuantKernels.Codebook3Bit);

        // Ensure the pooled temp buffers are large enough (allocated once, reused, grown only
        // if a larger numQueries arrives). Kept alive on the instance so they survive until the
        // CALLER flushes the command encoder — see the _faQNormalized field comment.
        int elemCount = numQueries * vecDim;
        EnsureFlashAttentionScratch(elemCount, numQueries);
        var qNormalized = _faQNormalized!.View.SubView(0, elemCount);
        var qNorms = _faQNorms!.View.SubView(0, numQueries);
        var qFlipped = _faQFlipped!.View.SubView(0, elemCount);
        var qTransformed = _faQTransformed!.View.SubView(0, elemCount);
        var attnOut = _faAttnOut!.View.SubView(0, elemCount);
        var invFWHT = _faInvFWHT!.View.SubView(0, elemCount);

        // Forward-transform Q into the Hadamard domain (same transform as KV encoding,
        // but WITHOUT the √d scaling — K has √d baked in for codebook matching,
        // so Q stays at 1/√d variance to keep the dot product correctly scaled)
        _quant.Normalize(query, qNormalized, qNorms, numQueries, vecDim);
        _quant.SignFlip(qNormalized, qFlipped, _signs!.View, elemCount);
        _quant.FWHT.ForwardBatch(qFlipped, qTransformed, numQueries, vecDim);

        // Run Flash Attention in the quantized Hadamard domain
        _quant.FlashQuantizedAttention(
            qTransformed, kPacked.packed, _codebook!.View,
            vPacked.packed, _vCodebookCopy.View,
            kPacked.norms, vPacked.norms, attnOut,
            numQueries, CurrentSeqLen, vecDim, scale,
            bitsPerValue: _bitsPerValue, valuesPerInt: _valuesPerInt);

        // Inverse-transform the output back to original domain.
        // The attention output is a weighted sum of V values in the scaled Hadamard domain.
        // V had √vecDim baked in (from pre-quantization scaling), so scale by 1/√vecDim.
        // Then inverse FWHT + inverse sign flip recovers the original V domain.
        // Do NOT apply Q_norm — the output is V-domain, not Q-domain.
        // The kernel already multiplied by V_norms inside the attention loop.
        float invSqrtD = 1f / MathF.Sqrt(vecDim);
        new ElementWiseKernels(_accelerator).ScaleInPlace(attnOut, elemCount, invSqrtD);
        _quant.FWHT.ForwardBatch(attnOut, invFWHT, numQueries, vecDim);
        _quant.SignFlip(invFWHT, output, _signs!.View, elemCount);
    }

    /// <summary>
    /// Lazily (re)allocate the pooled FlashAttention scratch buffers so they cover
    /// <paramref name="elemCount"/> (= numQueries * vecDim) floats and <paramref name="numQueries"/>
    /// norms. Reused across calls; only reallocated when a larger request arrives.
    /// </summary>
    private void EnsureFlashAttentionScratch(int elemCount, int numQueries)
    {
        if (_faQNormalized == null || _faElemCapacity < elemCount)
        {
            _faQNormalized?.Dispose();
            _faQFlipped?.Dispose();
            _faQTransformed?.Dispose();
            _faAttnOut?.Dispose();
            _faInvFWHT?.Dispose();
            _faQNormalized = _accelerator.Allocate1D<float>(elemCount);
            _faQFlipped = _accelerator.Allocate1D<float>(elemCount);
            _faQTransformed = _accelerator.Allocate1D<float>(elemCount);
            _faAttnOut = _accelerator.Allocate1D<float>(elemCount);
            _faInvFWHT = _accelerator.Allocate1D<float>(elemCount);
            _faElemCapacity = elemCount;
        }
        if (_faQNorms == null || _faQueryCapacity < numQueries)
        {
            _faQNorms?.Dispose();
            _faQNorms = _accelerator.Allocate1D<float>(numQueries);
            _faQueryCapacity = numQueries;
        }
    }

    // Separate codebook copy for V (WebGPU aliasing prevention)
    private MemoryBuffer1D<float, Stride1D.Dense>? _vCodebookCopy;

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

        // Step 3b: Scale by √d to match N(0,1) codebook.
        // FWHT normalizes by 1/√d, so output of unit vector has variance ~1/d.
        // The codebook expects N(0,1) variance. Multiply by √d to restore.
        float sqrtD = MathF.Sqrt(vecDim);
        new ElementWiseKernels(_accelerator).ScaleInPlace(
            _tempTransformed!.View.SubView(0, vecDim), vecDim, sqrtD);

        // Step 4: Quantize to nearest codebook centroid
        int numCentroids = _mode == KVQuantMode.TurboQuant4Bit ? 16 : 8;
        _quant.Quantize(_tempTransformed!.View.SubView(0, vecDim),
            _codebook!.View, _tempIndices!.View.SubView(0, vecDim), vecDim, numCentroids);

        // Step 5: Bit-pack indices
        if (_mode == KVQuantMode.TurboQuant3Bit)
            _quant.BitPack3(_tempIndices!.View.SubView(0, vecDim), packedOut, vecDim);
        else
            _quant.BitPack4(_tempIndices!.View.SubView(0, vecDim), packedOut, vecDim);
        // Note: TurboQuant3BitQJL uses BitPack4 (3 bits value + 1 bit QJL sign = 4 bits)
        // QJL sign computation will be added when QJL kernel is implemented
    }

    private Tensors.Tensor GetDequantized(int layer, bool isKey, int[] shape)
    {
        var lc = _layers[layer];
        int vecDim = _numHeads * _headDim;
        int packedDim = (vecDim + _valuesPerInt - 1) / _valuesPerInt;
        int totalElems = CurrentSeqLen * vecDim;

        // Output buffer: per-(layer, K/V) instance-owned, grow-only. A fresh Allocate1D
        // here LEAKED one buffer per layer per decode step (the returned Tensor doesn't
        // own its view; nothing disposed it), and disposing it eagerly would hit the same
        // queued-dispatch hazard as the temps. Grown geometrically as the sequence grows;
        // the previous step's reads are complete before the next step's overwrite (steps
        // are sequential with a drain between). Old buffers retire at cache Dispose().
        ref var outSlot = ref isKey ? ref lc.K_dequantOut : ref lc.V_dequantOut;
        if (outSlot == null || outSlot.Length < totalElems)
        {
            long grown = Math.Max(totalElems, (outSlot?.Length ?? 0) * 2);
            if (outSlot != null) _retiredBuffers.Add(outSlot);
            outSlot = _accelerator.Allocate1D<float>(grown);
        }
        var outBuf = outSlot;

        // Temp buffers for dequant pipeline (per-token). Instance-owned, NOT `using var`:
        // this method is synchronous, so locals would be disposed while the dispatches
        // below are still queued on the browser backends (see the field comment).
        var tempUnpacked = _dqUnpacked ??= _accelerator.Allocate1D<int>(vecDim);
        var tempDequant = _dqDequant ??= _accelerator.Allocate1D<float>(vecDim);
        var tempInvFWHT = _dqInvFWHT ??= _accelerator.Allocate1D<float>(vecDim);
        var tempInvFlip = _dqInvFlip ??= _accelerator.Allocate1D<float>(vecDim);

        var packed = isKey ? lc.K_packed! : lc.V_packed!;
        var norms = isKey ? lc.K_norms! : lc.V_norms!;

        int numCentroids = _mode == KVQuantMode.TurboQuant4Bit ? 16 : 8;

        for (int t = 0; t < CurrentSeqLen; t++)
        {
            // Step 1: Unpack
            if (_mode == KVQuantMode.TurboQuant3Bit)
                _quant.BitUnpack3(packed.View.SubView(t * packedDim, packedDim),
                    tempUnpacked.View.SubView(0, vecDim), vecDim);
            else
                _quant.BitUnpack4(packed.View.SubView(t * packedDim, packedDim),
                    tempUnpacked.View.SubView(0, vecDim), vecDim);

            // For 3+1 QJL mode, mask to lower 3 bits for centroid lookup
            // (upper bit is QJL sign, handled separately when QJL kernel is implemented)

            // Step 2: Dequantize (codebook lookup)
            _quant.Dequantize(tempUnpacked.View.SubView(0, vecDim), _codebook!.View,
                tempDequant.View.SubView(0, vecDim), vecDim, numCentroids);

            // Step 2b: Scale by 1/√d to undo the pre-quantization √d scaling
            float invSqrtD = 1f / MathF.Sqrt(vecDim);
            new ElementWiseKernels(_accelerator).ScaleInPlace(
                tempDequant.View.SubView(0, vecDim), vecDim, invSqrtD);

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
        // Grow-only dequant output buffers (GetDequantized) — reused across decode steps.
        public MemoryBuffer1D<float, Stride1D.Dense>? K_dequantOut;
        public MemoryBuffer1D<float, Stride1D.Dense>? V_dequantOut;

        public void Dispose()
        {
            K_packed?.Dispose();
            V_packed?.Dispose();
            K_norms?.Dispose();
            V_norms?.Dispose();
            K_dequantOut?.Dispose();
            V_dequantOut?.Dispose();
        }
    }

    public void Dispose()
    {
        foreach (var lc in _layers) lc.Dispose();
        _codebook?.Dispose();
        _vCodebookCopy?.Dispose();
        _signs?.Dispose();
        _tempNormalized?.Dispose();
        _tempFlipped?.Dispose();
        _tempTransformed?.Dispose();
        _tempIndices?.Dispose();
        _dqUnpacked?.Dispose();
        _dqDequant?.Dispose();
        _dqInvFWHT?.Dispose();
        _dqInvFlip?.Dispose();
        foreach (var b in _retiredBuffers) b.Dispose();
        _retiredBuffers.Clear();
        _faQNormalized?.Dispose();
        _faQNorms?.Dispose();
        _faQFlipped?.Dispose();
        _faQTransformed?.Dispose();
        _faAttnOut?.Dispose();
        _faInvFWHT?.Dispose();
    }
}
