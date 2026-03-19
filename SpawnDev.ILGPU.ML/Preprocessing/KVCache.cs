using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Key-Value cache for autoregressive transformer generation.
/// Stores computed K and V tensors from previous tokens so they don't need
/// to be recomputed at each generation step.
///
/// Used by: text-generation (GPT-2, LLaMA, Qwen), speech recognition (Whisper decoder),
/// translation, summarization, image captioning.
///
/// Architecture: each transformer layer has its own K and V cache.
/// Shape per layer: K=[batch, heads, seqLen, headDim], V=[batch, heads, seqLen, headDim]
/// On each new token, we append one position to seqLen.
/// </summary>
public class KVCache : IDisposable
{
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _maxSeqLen;

    // Cache storage: [numLayers][2] where [layer][0]=K, [layer][1]=V
    // Each is a flat float array: [numHeads * currentSeqLen * headDim]
    private readonly float[][][] _cache;
    private int _currentSeqLen;

    /// <summary>Current sequence length (number of tokens cached).</summary>
    public int CurrentSeqLen => _currentSeqLen;

    /// <summary>Maximum sequence length.</summary>
    public int MaxSeqLen => _maxSeqLen;

    /// <summary>Whether the cache has any content.</summary>
    public bool IsEmpty => _currentSeqLen == 0;

    /// <summary>
    /// Create a KV cache for a transformer model.
    /// </summary>
    /// <param name="numLayers">Number of transformer layers</param>
    /// <param name="numHeads">Number of attention heads</param>
    /// <param name="headDim">Dimension per head</param>
    /// <param name="maxSeqLen">Maximum sequence length to support</param>
    public KVCache(int numLayers, int numHeads, int headDim, int maxSeqLen = 2048)
    {
        _numLayers = numLayers;
        _numHeads = numHeads;
        _headDim = headDim;
        _maxSeqLen = maxSeqLen;

        _cache = new float[numLayers][][];
        for (int l = 0; l < numLayers; l++)
        {
            _cache[l] = new float[2][];
            _cache[l][0] = new float[numHeads * maxSeqLen * headDim]; // K
            _cache[l][1] = new float[numHeads * maxSeqLen * headDim]; // V
        }

        _currentSeqLen = 0;
    }

    /// <summary>
    /// Append new K and V values for one token across all layers.
    /// Called once per generated token.
    /// </summary>
    /// <param name="layer">Layer index</param>
    /// <param name="newK">New K values [numHeads * headDim] for one token</param>
    /// <param name="newV">New V values [numHeads * headDim] for one token</param>
    public void Append(int layer, float[] newK, float[] newV)
    {
        if (_currentSeqLen >= _maxSeqLen)
            throw new InvalidOperationException($"KV cache full: {_currentSeqLen} >= {_maxSeqLen}");

        int tokenSize = _numHeads * _headDim;

        // Copy new K/V into the cache at the current position
        // Layout: [head, seqPos, headDim]
        for (int h = 0; h < _numHeads; h++)
        {
            int srcOffset = h * _headDim;
            int dstOffset = h * _maxSeqLen * _headDim + _currentSeqLen * _headDim;
            Array.Copy(newK, srcOffset, _cache[layer][0], dstOffset, _headDim);
            Array.Copy(newV, srcOffset, _cache[layer][1], dstOffset, _headDim);
        }
    }

    /// <summary>
    /// Advance the sequence position after appending to all layers.
    /// Call once after appending K/V to every layer for a token.
    /// </summary>
    public void AdvancePosition()
    {
        _currentSeqLen++;
    }

    /// <summary>
    /// Get the full K cache for a layer up to the current sequence length.
    /// Returns [numHeads, currentSeqLen, headDim] as a flat array.
    /// </summary>
    public float[] GetK(int layer)
    {
        return ExtractCache(_cache[layer][0]);
    }

    /// <summary>
    /// Get the full V cache for a layer up to the current sequence length.
    /// Returns [numHeads, currentSeqLen, headDim] as a flat array.
    /// </summary>
    public float[] GetV(int layer)
    {
        return ExtractCache(_cache[layer][1]);
    }

    /// <summary>
    /// Get K and V caches for a layer.
    /// Returns (K, V) each as [numHeads, currentSeqLen, headDim].
    /// </summary>
    public (float[] K, float[] V) GetKV(int layer)
    {
        return (GetK(layer), GetV(layer));
    }

    /// <summary>
    /// Reset the cache (start a new generation).
    /// </summary>
    public void Clear()
    {
        _currentSeqLen = 0;
        // No need to zero the arrays — we track currentSeqLen
    }

    /// <summary>
    /// Trim the cache to keep only the last N tokens (sliding window).
    /// Useful for very long sequences.
    /// </summary>
    public void TrimToLast(int keepTokens)
    {
        if (keepTokens >= _currentSeqLen) return;

        int dropTokens = _currentSeqLen - keepTokens;

        for (int l = 0; l < _numLayers; l++)
        {
            for (int kv = 0; kv < 2; kv++)
            {
                for (int h = 0; h < _numHeads; h++)
                {
                    int srcOffset = h * _maxSeqLen * _headDim + dropTokens * _headDim;
                    int dstOffset = h * _maxSeqLen * _headDim;
                    Array.Copy(_cache[l][kv], srcOffset, _cache[l][kv], dstOffset, keepTokens * _headDim);
                }
            }
        }

        _currentSeqLen = keepTokens;
    }

    private float[] ExtractCache(float[] fullCache)
    {
        int outputSize = _numHeads * _currentSeqLen * _headDim;
        var result = new float[outputSize];

        for (int h = 0; h < _numHeads; h++)
        {
            int srcOffset = h * _maxSeqLen * _headDim;
            int dstOffset = h * _currentSeqLen * _headDim;
            Array.Copy(fullCache, srcOffset, result, dstOffset, _currentSeqLen * _headDim);
        }

        return result;
    }

    public void Dispose()
    {
        // CPU arrays are GC'd — nothing to explicitly dispose
    }
}

/// <summary>
/// Configuration for autoregressive generation with KV cache.
/// Describes a model's architecture for cache allocation.
/// </summary>
public class KVCacheConfig
{
    public int NumLayers { get; init; }
    public int NumHeads { get; init; }
    public int HeadDim { get; init; }
    public int MaxSeqLen { get; init; } = 2048;

    /// <summary>GPT-2 Small (12 layers, 12 heads, 64 dim).</summary>
    public static KVCacheConfig GPT2Small => new() { NumLayers = 12, NumHeads = 12, HeadDim = 64, MaxSeqLen = 1024 };

    /// <summary>GPT-2 Medium (24 layers, 16 heads, 64 dim).</summary>
    public static KVCacheConfig GPT2Medium => new() { NumLayers = 24, NumHeads = 16, HeadDim = 64, MaxSeqLen = 1024 };

    /// <summary>Whisper Tiny decoder (4 layers, 6 heads, 64 dim).</summary>
    public static KVCacheConfig WhisperTiny => new() { NumLayers = 4, NumHeads = 6, HeadDim = 64, MaxSeqLen = 448 };

    /// <summary>Whisper Base decoder (6 layers, 8 heads, 64 dim).</summary>
    public static KVCacheConfig WhisperBase => new() { NumLayers = 6, NumHeads = 8, HeadDim = 64, MaxSeqLen = 448 };

    /// <summary>Qwen 0.6B (28 layers, 16 heads, 64 dim).</summary>
    public static KVCacheConfig Qwen06B => new() { NumLayers = 28, NumHeads = 16, HeadDim = 64, MaxSeqLen = 2048 };

    /// <summary>Estimate memory usage in bytes for this config.</summary>
    public long EstimateMemoryBytes => (long)NumLayers * 2 * NumHeads * MaxSeqLen * HeadDim * sizeof(float);

    /// <summary>Estimate memory usage in MB.</summary>
    public double EstimateMemoryMB => EstimateMemoryBytes / (1024.0 * 1024.0);
}
