using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>Storage precision for the GGUF decode K/V cache.</summary>
public enum KVCachePrecision
{
    /// <summary>16-bit <see cref="BFloat16"/> store — HALF the VRAM, ~0.4% per-value relative error.
    /// The intended production default (on a 12 GB card next to a 7 GB model the KV cache is the binding
    /// constraint). BLOCKED as of 2026-06-15: ILGPU's BFloat16 CUDA codegen mis-compiles an
    /// ArrayView&lt;BFloat16&gt; store/load (returns zeros once a launch exceeds ~128 elements, and under
    /// repeated launches) — a library bug handed to Geordi (DevComms
    /// tuvok-to-geordi-ILGPU-BFloat16-cuda-store-zeros-2026-06-15). Flip <see cref="GGUFDecodeKVCache"/>'s
    /// default back to BF16 + re-enable the test's bf16 arm once that lands.</summary>
    BF16,
    /// <summary>32-bit float store — exact (no storage rounding). The SAFE default while bf16 is blocked;
    /// also what the regression test uses for its tight layout/RoPE/kv_offset gate (bf16 rounding would
    /// mask a subtle indexing bug there).</summary>
    F32,
}

/// <summary>
/// Per-layer K/V cache for GGUF autoregressive decode (gemma4 etc.).
///
/// Distinct from <see cref="QuantizedKVCache"/> (the ONNX-export TurboQuant path): this is a GGUF-native
/// incremental-decode cache. The big permanent store (sized to <c>maxSeqLen</c>) is the dominant KV-cache
/// consumer (the per-step repack scratch is sized to the live length, a small fraction of maxSeqLen). In the
/// default <see cref="KVCachePrecision.BF16"/> mode it is kept in bf16 (<see cref="BFloat16"/>, 16-bit) —
/// HALF the VRAM of an f32 store — using ILGPU's first-class <see cref="BFloat16"/> type, whose f32↔bf16
/// conversion is verified byte-identical across all 6 backends (this REPLACES an earlier manual
/// ushort+bit-shift version that the WebGL GLSL emitter mis-compiled). Attention always computes in f32: the
/// repack widens the store to f32, so <c>FusedAttention</c> and the executor are UNCHANGED regardless of
/// precision. <see cref="KVCachePrecision.F32"/> stores exactly (no rounding) — the regression suite uses it
/// to assert the layout/offset/RoPE equivalence at f32 tolerance, then asserts bf16 mode stays argmax-stable
/// within bf16 tolerance.
///
/// LAYOUT: head-major per layer — store[layer] holds K and V flat [kvHeads, maxSeq, hd]. The repack produces
/// contiguous f32 [kvHeads, len, hd] (store per-head stride is maxSeq*hd; the kernel wants len*hd). Per-layer
/// dims VARY in gemma4 (kvHeads 8 sliding / 1 global, hd 256 / 512).
///
/// POSITION OWNERSHIP: the cache is a dumb store — the CALLER (executor decode-intercept) owns the running
/// token position; per FusedAttention node it <see cref="Write"/>s the new K/V at [atToken, atToken+nTokens)
/// then reads <see cref="PackedK"/>/<see cref="PackedV"/> (f32) for the kernel. Prefill = first Write at 0.
/// </summary>
public sealed class GGUFDecodeKVCache : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly int _maxSeqLen;
    private readonly KVCachePrecision _precision;

    private sealed class LayerCache
    {
        public int KvHeads;
        public int HeadDim;
        // Exactly one of the f32 / bf16 pair is non-null, selected by _precision. [kvHeads, maxSeq, hd].
        public MemoryBuffer1D<float, Stride1D.Dense>? Kf, Vf;
        public MemoryBuffer1D<BFloat16, Stride1D.Dense>? Kb, Vb;
        public MemoryBuffer1D<float, Stride1D.Dense>? PackK;  // f32 repack scratch [kvHeads, len, hd], grow-only
        public MemoryBuffer1D<float, Stride1D.Dense>? PackV;
        public int PackCapacityTokens;
    }

    private readonly LayerCache[] _layers;
    private readonly List<IDisposable> _retired = new();

    // F32 path uses per-head CopyFrom (no kernel — WebGL-safe). bf16 path: f32↔bf16 conversion kernels.
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>, int, int, int, int, int>? _writeB;
    private Action<Index1D, ArrayView1D<BFloat16, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int>? _packB;

    /// <summary>Max tokens the cache can hold.</summary>
    public int MaxSeqLen => _maxSeqLen;

    /// <summary>Number of attention layers.</summary>
    public int NumLayers => _layers.Length;

    /// <summary>The storage precision this cache was allocated with.</summary>
    public KVCachePrecision Precision => _precision;

    /// <summary>
    /// Allocate the cache. <paramref name="kvHeadsPerLayer"/> / <paramref name="headDimPerLayer"/> are the
    /// per-layer attention geometry (gemma4 interleaves sliding/global with different values).
    /// <paramref name="precision"/> defaults to F32 (the SAFE default while <see cref="KVCachePrecision.BF16"/>
    /// is blocked on an ILGPU BFloat16 CUDA codegen bug — see that enum member).
    /// </summary>
    public GGUFDecodeKVCache(Accelerator accelerator, int[] kvHeadsPerLayer, int[] headDimPerLayer,
        int maxSeqLen = 4096, KVCachePrecision precision = KVCachePrecision.F32)
    {
        if (kvHeadsPerLayer.Length != headDimPerLayer.Length)
            throw new ArgumentException("kvHeadsPerLayer and headDimPerLayer must have the same length (one per layer).");
        _accelerator = accelerator;
        _maxSeqLen = maxSeqLen;
        _precision = precision;
        _layers = new LayerCache[kvHeadsPerLayer.Length];
        for (int i = 0; i < _layers.Length; i++)
        {
            int kvHeads = kvHeadsPerLayer[i], hd = headDimPerLayer[i];
            long elems = (long)kvHeads * maxSeqLen * hd;
            var lc = new LayerCache { KvHeads = kvHeads, HeadDim = hd };
            if (precision == KVCachePrecision.BF16)
            {
                lc.Kb = accelerator.Allocate1D<BFloat16>(elems);   // bf16 = half the f32 footprint
                lc.Vb = accelerator.Allocate1D<BFloat16>(elems);
            }
            else
            {
                lc.Kf = accelerator.Allocate1D<float>(elems);
                lc.Vf = accelerator.Allocate1D<float>(elems);
            }
            _layers[i] = lc;
        }
    }

    /// <summary>
    /// Write this step's K and V for one layer into the store at [atToken, atToken+nTokens).
    /// <paramref name="k"/>/<paramref name="v"/> are the post-transpose head-major f32 slices the graph
    /// produced: flat [kvHeads, nTokens, hd]. Decode step: nTokens=1. Prefill: nTokens=prompt length, atToken=0.
    /// </summary>
    public void Write(int layer, ArrayView1D<float, Stride1D.Dense> k, ArrayView1D<float, Stride1D.Dense> v, int atToken, int nTokens)
    {
        var lc = _layers[layer];
        if (atToken + nTokens > _maxSeqLen)
            throw new InvalidOperationException($"GGUFDecodeKVCache overflow: {atToken}+{nTokens} > {_maxSeqLen}. Increase maxSeqLen or add a sliding window.");
        int hd = lc.HeadDim, kvHeads = lc.KvHeads;
        if (_precision == KVCachePrecision.BF16)
        {
            // bf16 store needs an f32→bf16 conversion, so a kernel (not CopyFrom). NOTE: this scatters into the
            // maxSeq-strided store (write-index ≠ thread-index), which the WebGL Transform-Feedback path
            // corrupts — bf16 is currently blocked on the ILGPU BFloat16 CUDA bug anyway; a WebGL-safe bf16
            // write (one store per thread at its own index) is part of un-blocking it later.
            int total = kvHeads * nTokens * hd;
            _writeB ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>, int, int, int, int, int>(WriteBF16Impl);
            _writeB(total, k, lc.Kb!.View, kvHeads, nTokens, hd, atToken, _maxSeqLen);
            _writeB(total, v, lc.Vb!.View, kvHeads, nTokens, hd, atToken, _maxSeqLen);
        }
        else
        {
            // F32: per-head native CopyFrom (GPU→GPU). Works on ALL backends — WebGL included — because it's a
            // buffer copy, not a Transform-Feedback scatter (a write-index≠thread-index kernel silently
            // corrupts on WebGL). store[h, atToken : atToken+nTokens, :] = src[h, 0:nTokens, :].
            for (int h = 0; h < kvHeads; h++)
            {
                lc.Kf!.View.SubView((long)h * _maxSeqLen * hd + (long)atToken * hd, (long)nTokens * hd)
                    .CopyFrom(k.SubView((long)h * nTokens * hd, (long)nTokens * hd));
                lc.Vf!.View.SubView((long)h * _maxSeqLen * hd + (long)atToken * hd, (long)nTokens * hd)
                    .CopyFrom(v.SubView((long)h * nTokens * hd, (long)nTokens * hd));
            }
        }
    }

    /// <summary>Repack the first <paramref name="totalLen"/> tokens of one layer into a CONTIGUOUS f32
    /// [kvHeads, totalLen, hd] buffer (store → f32), for FusedAttention. Grow-only scratch.</summary>
    public ArrayView1D<float, Stride1D.Dense> PackedK(int layer, int totalLen) => Packed(layer, totalLen, isKey: true);
    /// <summary>Repack the first <paramref name="totalLen"/> tokens' V into contiguous f32 [kvHeads, totalLen, hd].</summary>
    public ArrayView1D<float, Stride1D.Dense> PackedV(int layer, int totalLen) => Packed(layer, totalLen, isKey: false);

    private ArrayView1D<float, Stride1D.Dense> Packed(int layer, int totalLen, bool isKey)
    {
        var lc = _layers[layer];
        int hd = lc.HeadDim, kvHeads = lc.KvHeads;
        EnsurePackCapacity(lc, totalLen);
        var pack = (isKey ? lc.PackK : lc.PackV)!.View;
        if (_precision == KVCachePrecision.BF16)
        {
            var store = (isKey ? lc.Kb : lc.Vb)!.View;
            _packB ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<BFloat16, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int>(PackBF16Impl);
            _packB(kvHeads * totalLen * hd, store, pack, kvHeads, totalLen, hd, _maxSeqLen);
        }
        else
        {
            // F32: per-head native CopyFrom (store maxSeq-stride → contiguous pack). WebGL-safe (buffer copy).
            var store = (isKey ? lc.Kf : lc.Vf)!.View;
            for (int h = 0; h < kvHeads; h++)
                pack.SubView((long)h * totalLen * hd, (long)totalLen * hd)
                    .CopyFrom(store.SubView((long)h * _maxSeqLen * hd, (long)totalLen * hd));
        }
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

    // ── kernels: index math is identical across precisions; only the element type/conversion differs ──

    // The (BFloat16)float cast is ILGPU's verified round-to-nearest-even narrowing (correct on every backend).
    private static void WriteBF16Impl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> src, ArrayView1D<BFloat16, Stride1D.Dense> store,
        int kvHeads, int nTokens, int hd, int atToken, int maxSeq)
    {
        int total = kvHeads * nTokens * hd;
        if (idx >= total) return;
        int d = idx % hd, t = (idx / hd) % nTokens, h = idx / (nTokens * hd);
        store[(long)h * maxSeq * hd + (long)(atToken + t) * hd + d] = (BFloat16)src[(long)h * nTokens * hd + (long)t * hd + d];
    }

    private static void PackBF16Impl(Index1D idx,
        ArrayView1D<BFloat16, Stride1D.Dense> store, ArrayView1D<float, Stride1D.Dense> pack,
        int kvHeads, int totalLen, int hd, int maxSeq)
    {
        int total = kvHeads * totalLen * hd;
        if (idx >= total) return;
        int d = idx % hd, t = (idx / hd) % totalLen, h = idx / (totalLen * hd);
        pack[(long)h * totalLen * hd + (long)t * hd + d] = (float)store[(long)h * maxSeq * hd + (long)t * hd + d];
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
            lc.Kf?.Dispose(); lc.Vf?.Dispose();
            lc.Kb?.Dispose(); lc.Vb?.Dispose();
            lc.PackK?.Dispose(); lc.PackV?.Dispose();
        }
        foreach (var r in _retired) r.Dispose();
        _retired.Clear();
    }
}
