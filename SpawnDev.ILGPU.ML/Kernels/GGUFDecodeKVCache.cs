using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// FULL-PRECISION per-layer K/V cache for GGUF autoregressive decode (gemma4 etc.).
///
/// Distinct from <see cref="QuantizedKVCache"/> (the ONNX-export TurboQuant path): this is a
/// GGUF-native, F32, no-compression cache used by the incremental decode loop. Correctness-first;
/// TurboQuant compression can layer on later as opt-in.
///
/// LAYOUT: head-major per layer — store[layer] holds K and V as flat [kvHeads, maxSeq, hd]
/// (head h's tokens are contiguous at h*maxSeq*hd, token t at +t*hd). This matches what
/// <c>FusedAttention</c> consumes (K/V flat [kvHeads, seqKV, hd]); the only mismatch is the per-head
/// STRIDE (maxSeq vs the live seqKV=len), so <see cref="PackedK"/>/<see cref="PackedV"/> repack to a
/// contiguous [kvHeads, len, hd] buffer for the kernel. Per-layer dims VARY in gemma4 (kvHeads 8
/// sliding / 1 global, hd 256 / 512), so each layer carries its own (kvHeads, headDim).
///
/// POSITION OWNERSHIP: the cache is a dumb store — the CALLER (the executor decode-intercept) owns
/// the running token position. Per FusedAttention node it calls <see cref="Write"/> to place the
/// step's new K/V at [atToken, atToken+nTokens), then <see cref="PackedK"/>/<see cref="PackedV"/> with
/// the TOTAL length (atToken+nTokens) to read the full history+new for the kernel, with
/// kv_offset = atToken. Prefill is just the first Write with nTokens = prompt length at atToken 0.
///
/// Browser-buffer discipline (see CLAUDE.md Wasm note + QuantizedKVCache): all GPU ops are
/// <c>CopyFrom</c> (portable on every backend, no sync wait); buffers are instance-owned; outgrown
/// repack buffers retire at <see cref="Dispose"/> (a smaller buffer may still be referenced by a
/// queued dispatch on the browser backends).
/// </summary>
public sealed class GGUFDecodeKVCache : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly int _maxSeqLen;

    private sealed class LayerCache
    {
        public int KvHeads;
        public int HeadDim;
        public MemoryBuffer1D<float, Stride1D.Dense>? K;     // [kvHeads, maxSeq, hd]
        public MemoryBuffer1D<float, Stride1D.Dense>? V;     // [kvHeads, maxSeq, hd]
        public MemoryBuffer1D<float, Stride1D.Dense>? PackK; // repack scratch [kvHeads, len, hd], grow-only
        public MemoryBuffer1D<float, Stride1D.Dense>? PackV;
        public int PackCapacityTokens;                       // PackK/PackV sized for this many tokens
    }

    private readonly LayerCache[] _layers;
    private readonly List<IDisposable> _retired = new();

    /// <summary>Max tokens the cache can hold.</summary>
    public int MaxSeqLen => _maxSeqLen;

    /// <summary>Number of attention layers.</summary>
    public int NumLayers => _layers.Length;

    /// <summary>
    /// Allocate the cache. <paramref name="kvHeadsPerLayer"/> / <paramref name="headDimPerLayer"/> are
    /// the per-layer attention geometry (gemma4 interleaves sliding/global with different values).
    /// </summary>
    public GGUFDecodeKVCache(Accelerator accelerator, int[] kvHeadsPerLayer, int[] headDimPerLayer, int maxSeqLen = 4096)
    {
        if (kvHeadsPerLayer.Length != headDimPerLayer.Length)
            throw new ArgumentException("kvHeadsPerLayer and headDimPerLayer must have the same length (one per layer).");
        _accelerator = accelerator;
        _maxSeqLen = maxSeqLen;
        _layers = new LayerCache[kvHeadsPerLayer.Length];
        for (int i = 0; i < _layers.Length; i++)
        {
            int kvHeads = kvHeadsPerLayer[i], hd = headDimPerLayer[i];
            long elems = (long)kvHeads * maxSeqLen * hd;
            _layers[i] = new LayerCache
            {
                KvHeads = kvHeads,
                HeadDim = hd,
                K = accelerator.Allocate1D<float>(elems),
                V = accelerator.Allocate1D<float>(elems),
            };
        }
    }

    /// <summary>
    /// Write this step's K and V for one layer into the store at [atToken, atToken+nTokens).
    /// <paramref name="k"/>/<paramref name="v"/> are the post-transpose head-major slices the graph
    /// produced for the step's tokens: flat [kvHeads, nTokens, hd] (head h's nTokens at h*nTokens*hd).
    /// Decode step: nTokens=1. Prefill: nTokens = prompt length, atToken = 0.
    /// </summary>
    public void Write(int layer, ArrayView1D<float, Stride1D.Dense> k, ArrayView1D<float, Stride1D.Dense> v, int atToken, int nTokens)
    {
        var lc = _layers[layer];
        if (atToken + nTokens > _maxSeqLen)
            throw new InvalidOperationException($"GGUFDecodeKVCache overflow: {atToken}+{nTokens} > {_maxSeqLen}. Increase maxSeqLen or add a sliding window.");
        int hd = lc.HeadDim, kvHeads = lc.KvHeads;
        for (int h = 0; h < kvHeads; h++)
        {
            // store[h, atToken : atToken+nTokens, :] = src[h, 0:nTokens, :]
            lc.K!.View.SubView((long)h * _maxSeqLen * hd + (long)atToken * hd, (long)nTokens * hd)
                .CopyFrom(k.SubView((long)h * nTokens * hd, (long)nTokens * hd));
            lc.V!.View.SubView((long)h * _maxSeqLen * hd + (long)atToken * hd, (long)nTokens * hd)
                .CopyFrom(v.SubView((long)h * nTokens * hd, (long)nTokens * hd));
        }
    }

    /// <summary>Repack the first <paramref name="totalLen"/> tokens of one layer into a CONTIGUOUS
    /// [kvHeads, totalLen, hd] buffer (the store's per-head stride is maxSeq*hd; FusedAttention needs
    /// totalLen*hd). Grow-only scratch, retired at Dispose.</summary>
    public ArrayView1D<float, Stride1D.Dense> PackedK(int layer, int totalLen) => Packed(layer, totalLen, isKey: true);
    /// <summary>Repack the first <paramref name="totalLen"/> tokens' V into contiguous [kvHeads, totalLen, hd].</summary>
    public ArrayView1D<float, Stride1D.Dense> PackedV(int layer, int totalLen) => Packed(layer, totalLen, isKey: false);

    private ArrayView1D<float, Stride1D.Dense> Packed(int layer, int totalLen, bool isKey)
    {
        var lc = _layers[layer];
        int hd = lc.HeadDim, kvHeads = lc.KvHeads;
        EnsurePackCapacity(lc, totalLen);
        var store = (isKey ? lc.K : lc.V)!.View;
        var pack = (isKey ? lc.PackK : lc.PackV)!.View;
        for (int h = 0; h < kvHeads; h++)
            pack.SubView((long)h * totalLen * hd, (long)totalLen * hd)
                .CopyFrom(store.SubView((long)h * _maxSeqLen * hd, (long)totalLen * hd));
        return pack.SubView(0, (long)kvHeads * totalLen * hd);
    }

    private void EnsurePackCapacity(LayerCache lc, int tokens)
    {
        if (lc.PackK != null && tokens <= lc.PackCapacityTokens) return;
        // Geometric growth; retire the old (smaller) buffers at Dispose (browser dispatch safety).
        int cap = Math.Max(tokens, lc.PackCapacityTokens == 0 ? Math.Min(64, _maxSeqLen) : lc.PackCapacityTokens * 2);
        cap = Math.Min(cap, _maxSeqLen);
        long elems = (long)lc.KvHeads * cap * lc.HeadDim;
        if (lc.PackK != null) { _retired.Add(lc.PackK); _retired.Add(lc.PackV!); }
        lc.PackK = _accelerator.Allocate1D<float>(elems);
        lc.PackV = _accelerator.Allocate1D<float>(elems);
        lc.PackCapacityTokens = cap;
    }

    /// <summary>Per-layer kvHeads (for the FusedAttention attrs at read time).</summary>
    public int KvHeads(int layer) => _layers[layer].KvHeads;
    /// <summary>Per-layer head_dim.</summary>
    public int HeadDim(int layer) => _layers[layer].HeadDim;

    /// <summary>Release all GPU buffers.</summary>
    public void Dispose()
    {
        foreach (var lc in _layers)
        {
            lc.K?.Dispose(); lc.V?.Dispose();
            lc.PackK?.Dispose(); lc.PackV?.Dispose();
        }
        foreach (var r in _retired) r.Dispose();
        _retired.Clear();
    }
}
