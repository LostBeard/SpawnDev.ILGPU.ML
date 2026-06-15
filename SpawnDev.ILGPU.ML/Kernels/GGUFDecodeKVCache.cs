using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// BF16-STORAGE per-layer K/V cache for GGUF autoregressive decode (gemma4 etc.).
///
/// Distinct from <see cref="QuantizedKVCache"/> (the ONNX-export TurboQuant path): this is a GGUF-native
/// incremental-decode cache used by the decode loop. The big permanent store (sized to <c>maxSeqLen</c>) is
/// kept in **bf16 (16-bit)** — HALF the VRAM of the old f32 store — and is the dominant KV-cache consumer
/// (the per-step repack scratch is sized to the live length, typically a small fraction of maxSeqLen). bf16
/// storage is pure bit-conversion (f32→bf16 = round + drop low 16 bits; bf16→f32 = zero-extend &lt;&lt; 16),
/// so it needs NO bf16 arithmetic and works on every backend including CUDA. Attention still computes in f32:
/// the repack converts bf16→f32, so <c>FusedAttention</c> and the executor are UNCHANGED (still consume f32
/// contiguous K/V). VRAM is priceless on a 12 GB card next to a 7 GB model — this buys back ~half the cache.
///
/// LAYOUT: head-major per layer — store[layer] holds K and V as flat bf16 [kvHeads, maxSeq, hd] (head h's
/// tokens contiguous at h*maxSeq*hd, token t at +t*hd). The repack produces a contiguous f32 [kvHeads, len,
/// hd] (the store's per-head stride is maxSeq*hd; the kernel wants len*hd). Per-layer dims VARY in gemma4
/// (kvHeads 8 sliding / 1 global, hd 256 / 512), so each layer carries its own (kvHeads, headDim).
///
/// POSITION OWNERSHIP: the cache is a dumb store — the CALLER (the executor decode-intercept) owns the running
/// token position. Per FusedAttention node it calls <see cref="Write"/> to place the step's new K/V at
/// [atToken, atToken+nTokens), then <see cref="PackedK"/>/<see cref="PackedV"/> with the TOTAL length to read
/// the full history+new (f32) for the kernel. Prefill is the first Write with nTokens = prompt length at 0.
///
/// Browser-buffer discipline: outgrown repack buffers retire at <see cref="Dispose"/>.
/// </summary>
public sealed class GGUFDecodeKVCache : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly int _maxSeqLen;

    private sealed class LayerCache
    {
        public int KvHeads;
        public int HeadDim;
        public MemoryBuffer1D<ushort, Stride1D.Dense>? K;   // bf16 bits, [kvHeads, maxSeq, hd]
        public MemoryBuffer1D<ushort, Stride1D.Dense>? V;   // bf16 bits, [kvHeads, maxSeq, hd]
        public MemoryBuffer1D<float, Stride1D.Dense>? PackK; // f32 repack scratch [kvHeads, len, hd], grow-only
        public MemoryBuffer1D<float, Stride1D.Dense>? PackV;
        public int PackCapacityTokens;
    }

    private readonly LayerCache[] _layers;
    private readonly List<IDisposable> _retired = new();

    // f32 src [kvHeads,nTokens,hd] → bf16 store [kvHeads,maxSeq,hd] at token offset; and bf16 store → f32 pack.
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<ushort, Stride1D.Dense>, int, int, int, int, int>? _writeKernel;
    private Action<Index1D, ArrayView1D<ushort, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int>? _packKernel;

    /// <summary>Max tokens the cache can hold.</summary>
    public int MaxSeqLen => _maxSeqLen;

    /// <summary>Number of attention layers.</summary>
    public int NumLayers => _layers.Length;

    /// <summary>
    /// Allocate the cache. <paramref name="kvHeadsPerLayer"/> / <paramref name="headDimPerLayer"/> are the
    /// per-layer attention geometry (gemma4 interleaves sliding/global with different values).
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
                K = accelerator.Allocate1D<ushort>(elems),   // bf16 = half the f32 footprint
                V = accelerator.Allocate1D<ushort>(elems),
            };
        }
    }

    /// <summary>
    /// Write this step's K and V for one layer into the bf16 store at [atToken, atToken+nTokens).
    /// <paramref name="k"/>/<paramref name="v"/> are the post-transpose head-major f32 slices the graph
    /// produced: flat [kvHeads, nTokens, hd]. Decode step: nTokens=1. Prefill: nTokens=prompt length, atToken=0.
    /// </summary>
    public void Write(int layer, ArrayView1D<float, Stride1D.Dense> k, ArrayView1D<float, Stride1D.Dense> v, int atToken, int nTokens)
    {
        var lc = _layers[layer];
        if (atToken + nTokens > _maxSeqLen)
            throw new InvalidOperationException($"GGUFDecodeKVCache overflow: {atToken}+{nTokens} > {_maxSeqLen}. Increase maxSeqLen or add a sliding window.");
        int hd = lc.HeadDim, kvHeads = lc.KvHeads;
        _writeKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<ushort, Stride1D.Dense>, int, int, int, int, int>(WriteConvertImpl);
        int total = kvHeads * nTokens * hd;
        _writeKernel(total, k, lc.K!.View, kvHeads, nTokens, hd, atToken, _maxSeqLen);
        _writeKernel(total, v, lc.V!.View, kvHeads, nTokens, hd, atToken, _maxSeqLen);
    }

    /// <summary>Repack the first <paramref name="totalLen"/> tokens of one layer into a CONTIGUOUS f32
    /// [kvHeads, totalLen, hd] buffer (bf16 store → f32), for FusedAttention. Grow-only scratch.</summary>
    public ArrayView1D<float, Stride1D.Dense> PackedK(int layer, int totalLen) => Packed(layer, totalLen, isKey: true);
    /// <summary>Repack the first <paramref name="totalLen"/> tokens' V into contiguous f32 [kvHeads, totalLen, hd].</summary>
    public ArrayView1D<float, Stride1D.Dense> PackedV(int layer, int totalLen) => Packed(layer, totalLen, isKey: false);

    private ArrayView1D<float, Stride1D.Dense> Packed(int layer, int totalLen, bool isKey)
    {
        var lc = _layers[layer];
        int hd = lc.HeadDim, kvHeads = lc.KvHeads;
        EnsurePackCapacity(lc, totalLen);
        var store = (isKey ? lc.K : lc.V)!.View;
        var pack = (isKey ? lc.PackK : lc.PackV)!.View;
        _packKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<ushort, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int>(PackConvertImpl);
        _packKernel(kvHeads * totalLen * hd, store, pack, kvHeads, totalLen, hd, _maxSeqLen);
        return pack.SubView(0, (long)kvHeads * totalLen * hd);
    }

    private void EnsurePackCapacity(LayerCache lc, int tokens)
    {
        if (lc.PackK != null && tokens <= lc.PackCapacityTokens) return;
        int cap = Math.Max(tokens, lc.PackCapacityTokens == 0 ? Math.Min(64, _maxSeqLen) : lc.PackCapacityTokens * 2);
        cap = Math.Min(cap, _maxSeqLen);
        long elems = (long)lc.KvHeads * cap * lc.HeadDim;
        if (lc.PackK != null) { _retired.Add(lc.PackK); _retired.Add(lc.PackV!); }
        lc.PackK = _accelerator.Allocate1D<float>(elems);
        lc.PackV = _accelerator.Allocate1D<float>(elems);
        lc.PackCapacityTokens = cap;
    }

    // f32 src [kvHeads,nTokens,hd] → bf16 store [kvHeads,maxSeq,hd] at [atToken, atToken+nTokens).
    // bf16 = round-to-nearest-even then drop the low 16 bits (matches the standard bf16 narrowing).
    private static void WriteConvertImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> src, ArrayView1D<ushort, Stride1D.Dense> store,
        int kvHeads, int nTokens, int hd, int atToken, int maxSeq)
    {
        int total = kvHeads * nTokens * hd;
        if (idx >= total) return;
        int d = idx % hd;
        int t = (idx / hd) % nTokens;
        int h = idx / (nTokens * hd);
        uint bits = Interop.FloatAsInt(src[(long)h * nTokens * hd + (long)t * hd + d]);
        uint rne = bits + 0x7FFFu + ((bits >> 16) & 1u);   // round-to-nearest-even
        store[(long)h * maxSeq * hd + (long)(atToken + t) * hd + d] = (ushort)(rne >> 16);
    }

    // bf16 store [kvHeads,maxSeq,hd] first totalLen tokens → f32 pack [kvHeads,totalLen,hd] (zero-extend << 16).
    private static void PackConvertImpl(Index1D idx,
        ArrayView1D<ushort, Stride1D.Dense> store, ArrayView1D<float, Stride1D.Dense> pack,
        int kvHeads, int totalLen, int hd, int maxSeq)
    {
        int total = kvHeads * totalLen * hd;
        if (idx >= total) return;
        int d = idx % hd;
        int t = (idx / hd) % totalLen;
        int h = idx / (totalLen * hd);
        ushort bf = store[(long)h * maxSeq * hd + (long)t * hd + d];
        pack[(long)h * totalLen * hd + (long)t * hd + d] = Interop.IntAsFloat((uint)bf << 16);
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
