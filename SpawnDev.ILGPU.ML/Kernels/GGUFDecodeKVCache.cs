using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>Storage precision for the GGUF decode K/V cache.</summary>
public enum KVCachePrecision
{
    /// <summary>16-bit <see cref="BFloat16"/> store — HALF the VRAM, ~0.4% per-value relative error. The
    /// production default: on a 12 GB card next to a 7 GB model the KV cache is the binding constraint.
    /// (Was briefly blocked on an ILGPU BFloat16 CUDA store bug — fixed in SpawnDev.ILGPU 4.13.0-local.4,
    /// Geordi; re-enabled 2026-06-16.)</summary>
    BF16,
    /// <summary>32-bit float store — exact (no storage rounding). What the regression test uses for its
    /// tight layout/RoPE/kv_offset gate (bf16 rounding would mask a subtle indexing bug there).</summary>
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
        // bf16 path: a CONTIGUOUS bf16 conversion scratch so the f32↔bf16 convert kernel writes at its own
        // index (WebGL-safe), and the strided store↔contiguous moves are CopyFrom (also WebGL-safe). Grow-only.
        public MemoryBuffer1D<BFloat16, Stride1D.Dense>? Bf16Scratch;
        public int Bf16ScratchTokens;
    }

    private readonly LayerCache[] _layers;
    private readonly List<IDisposable> _retired = new();

    // bf16 path (ONE path, all backends): a CONTIGUOUS element-wise convert kernel (write-index == thread-index
    // — WebGL-safe, no Transform-Feedback scatter), then CopyFromAsync between the contiguous scratch and the
    // maxSeq-strided store. CopyFromAsync (not sync CopyFrom) is what orders the copy against the convert kernel
    // on the Wasm worker pool — exactly what it was added for; sync CopyFrom silently races there.
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>>? _f32ToBf16;
    private Action<Index1D, ArrayView1D<BFloat16, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _bf16ToF32;

    /// <summary>Max tokens the cache can hold.</summary>
    public int MaxSeqLen => _maxSeqLen;

    /// <summary>Number of attention layers.</summary>
    public int NumLayers => _layers.Length;

    /// <summary>The storage precision this cache was allocated with.</summary>
    public KVCachePrecision Precision => _precision;

    /// <summary>
    /// Allocate the cache. <paramref name="kvHeadsPerLayer"/> / <paramref name="headDimPerLayer"/> are the
    /// per-layer attention geometry (gemma4 interleaves sliding/global with different values).
    /// <paramref name="precision"/> defaults to bf16 (the production VRAM win — half the KV footprint).
    /// </summary>
    public GGUFDecodeKVCache(Accelerator accelerator, int[] kvHeadsPerLayer, int[] headDimPerLayer,
        int maxSeqLen = 4096, KVCachePrecision precision = KVCachePrecision.BF16)
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
    public async Task WriteAsync(int layer, ArrayView1D<float, Stride1D.Dense> k, ArrayView1D<float, Stride1D.Dense> v, int atToken, int nTokens)
    {
        var lc = _layers[layer];
        if (atToken + nTokens > _maxSeqLen)
            throw new InvalidOperationException($"GGUFDecodeKVCache overflow: {atToken}+{nTokens} > {_maxSeqLen}. Increase maxSeqLen or add a sliding window.");
        int hd = lc.HeadDim, kvHeads = lc.KvHeads;
        if (_precision == KVCachePrecision.BF16)
        {
            // Convert f32→bf16 into a CONTIGUOUS scratch (kernel writes its OWN index — WebGL-safe, no scatter),
            // then CopyFromAsync scratch → the maxSeq-strided store. CopyFromAsync orders the copy against the
            // convert kernel on the Wasm worker pool (sync CopyFrom silently races there). Batched per K/V.
            int total = kvHeads * nTokens * hd;
            EnsureBf16Scratch(lc, nTokens);
            var scratch = lc.Bf16Scratch!.View;
            _f32ToBf16 ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>>(F32ToBf16Impl);
            _f32ToBf16(total, k, scratch);
            var tasks = new List<Task>(kvHeads);
            for (int h = 0; h < kvHeads; h++)
                tasks.Add(lc.Kb!.View.SubView((long)h * _maxSeqLen * hd + (long)atToken * hd, (long)nTokens * hd)
                    .CopyFromAsync(scratch.SubView((long)h * nTokens * hd, (long)nTokens * hd)));
            await Task.WhenAll(tasks).ConfigureAwait(false);
            _f32ToBf16(total, v, scratch);
            tasks.Clear();
            for (int h = 0; h < kvHeads; h++)
                tasks.Add(lc.Vb!.View.SubView((long)h * _maxSeqLen * hd + (long)atToken * hd, (long)nTokens * hd)
                    .CopyFromAsync(scratch.SubView((long)h * nTokens * hd, (long)nTokens * hd)));
            await Task.WhenAll(tasks).ConfigureAwait(false);
        }
        else
        {
            // F32: per-head CopyFromAsync (k/v are graph node OUTPUTS; the async copy orders against their
            // producing kernel on Wasm — a sync CopyFrom of a node output silently races there).
            var tasks = new List<Task>(kvHeads * 2);
            for (int h = 0; h < kvHeads; h++)
            {
                tasks.Add(lc.Kf!.View.SubView((long)h * _maxSeqLen * hd + (long)atToken * hd, (long)nTokens * hd)
                    .CopyFromAsync(k.SubView((long)h * nTokens * hd, (long)nTokens * hd)));
                tasks.Add(lc.Vf!.View.SubView((long)h * _maxSeqLen * hd + (long)atToken * hd, (long)nTokens * hd)
                    .CopyFromAsync(v.SubView((long)h * nTokens * hd, (long)nTokens * hd)));
            }
            await Task.WhenAll(tasks).ConfigureAwait(false);
        }
    }

    /// <summary>Repack the first <paramref name="totalLen"/> tokens of one layer into a CONTIGUOUS f32
    /// [kvHeads, totalLen, hd] buffer (store → f32), for FusedAttention. Grow-only scratch.</summary>
    public Task<ArrayView1D<float, Stride1D.Dense>> PackedKAsync(int layer, int totalLen) => PackedAsync(layer, totalLen, isKey: true);
    /// <summary>Repack the first <paramref name="totalLen"/> tokens' V into contiguous f32 [kvHeads, totalLen, hd].</summary>
    public Task<ArrayView1D<float, Stride1D.Dense>> PackedVAsync(int layer, int totalLen) => PackedAsync(layer, totalLen, isKey: false);

    private async Task<ArrayView1D<float, Stride1D.Dense>> PackedAsync(int layer, int totalLen, bool isKey)
    {
        var lc = _layers[layer];
        int hd = lc.HeadDim, kvHeads = lc.KvHeads;
        EnsurePackCapacity(lc, totalLen);
        var pack = (isKey ? lc.PackK : lc.PackV)!.View;
        if (_precision == KVCachePrecision.BF16)
        {
            // CopyFromAsync the strided bf16 store → a CONTIGUOUS scratch (ordered on Wasm), then convert scratch
            // → f32 pack (the convert KERNEL reads scratch — a kernel-input read IS ordered after the copy).
            var store = (isKey ? lc.Kb : lc.Vb)!.View;
            EnsureBf16Scratch(lc, totalLen);
            var scratch = lc.Bf16Scratch!.View;
            var tasks = new List<Task>(kvHeads);
            for (int h = 0; h < kvHeads; h++)
                tasks.Add(scratch.SubView((long)h * totalLen * hd, (long)totalLen * hd)
                    .CopyFromAsync(store.SubView((long)h * _maxSeqLen * hd, (long)totalLen * hd)));
            await Task.WhenAll(tasks).ConfigureAwait(false);
            _bf16ToF32 ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<BFloat16, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(Bf16ToF32Impl);
            _bf16ToF32(kvHeads * totalLen * hd, scratch.SubView(0, (long)kvHeads * totalLen * hd), pack);
        }
        else
        {
            // F32: per-head CopyFromAsync (store maxSeq-stride → contiguous pack).
            var store = (isKey ? lc.Kf : lc.Vf)!.View;
            var tasks = new List<Task>(kvHeads);
            for (int h = 0; h < kvHeads; h++)
                tasks.Add(pack.SubView((long)h * totalLen * hd, (long)totalLen * hd)
                    .CopyFromAsync(store.SubView((long)h * _maxSeqLen * hd, (long)totalLen * hd)));
            await Task.WhenAll(tasks).ConfigureAwait(false);
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

    /// <summary>Grow-only contiguous bf16 conversion scratch, sized to hold <paramref name="tokens"/> tokens.</summary>
    private void EnsureBf16Scratch(LayerCache lc, int tokens)
    {
        if (lc.Bf16Scratch != null && tokens <= lc.Bf16ScratchTokens) return;
        int cap = Math.Max(tokens, lc.Bf16ScratchTokens == 0 ? Math.Min(64, _maxSeqLen) : lc.Bf16ScratchTokens * 2);
        cap = Math.Min(cap, _maxSeqLen);
        if (lc.Bf16Scratch != null) _retired.Add(lc.Bf16Scratch);
        lc.Bf16Scratch = _accelerator.Allocate1D<BFloat16>((long)lc.KvHeads * cap * lc.HeadDim);
        lc.Bf16ScratchTokens = cap;
    }

    // ── contiguous element-wise converts: one store per thread at its OWN index (WebGL-safe). The strided
    //    store↔scratch moves are CopyFrom (also WebGL-safe). The (BFloat16)float cast is ILGPU's verified RNE. ──
    private static void F32ToBf16Impl(Index1D idx, ArrayView1D<float, Stride1D.Dense> src, ArrayView1D<BFloat16, Stride1D.Dense> dst)
    {
        if (idx >= dst.Length) return;
        dst[idx] = (BFloat16)src[idx];
    }

    private static void Bf16ToF32Impl(Index1D idx, ArrayView1D<BFloat16, Stride1D.Dense> src, ArrayView1D<float, Stride1D.Dense> dst)
    {
        if (idx >= dst.Length) return;
        dst[idx] = (float)src[idx];
    }

    /// <summary>Per-layer kvHeads (for the FusedAttention attrs at read time).</summary>
    public int KvHeads(int layer) => _layers[layer].KvHeads;
    /// <summary>Per-layer head_dim.</summary>
    public int HeadDim(int layer) => _layers[layer].HeadDim;

    /// <summary>Wrap one layer's K (or V) store as a Tensor for the strided FusedAttention decode path: the FULL
    /// <c>[1, kvHeads, maxSeq, hd]</c> store (bf16 via <see cref="Tensor.FromLowP{T}"/>, or f32), read DIRECTLY by
    /// <c>FusedAttention.ForwardStrided</c> with a per-head element stride of <c>maxSeq*hd</c> (derived from this
    /// shape) — NO per-token repack/bf16→f32-widen. The caller passes the LIVE history length via the node's
    /// <c>kv_seq_len</c> attr (the store is maxSeq-padded; only the first <c>kv_seq_len</c> tokens are attended).</summary>
    public Tensor StoreK(int layer, string name) => StoreTensor(_layers[layer], isKey: true, name);
    /// <summary>V counterpart of <see cref="StoreK"/>.</summary>
    public Tensor StoreV(int layer, string name) => StoreTensor(_layers[layer], isKey: false, name);

    private Tensor StoreTensor(LayerCache lc, bool isKey, string name)
    {
        var shape = new[] { 1, lc.KvHeads, _maxSeqLen, lc.HeadDim };
        return _precision == KVCachePrecision.BF16
            ? Tensor.FromLowP((isKey ? lc.Kb : lc.Vb)!.View, TensorDataType.BFloat16, shape, name)
            : new Tensor((isKey ? lc.Kf : lc.Vf)!.View, shape, name);
    }

    /// <summary>Release all GPU buffers.</summary>
    public void Dispose()
    {
        foreach (var lc in _layers)
        {
            lc.Kf?.Dispose(); lc.Vf?.Dispose();
            lc.Kb?.Dispose(); lc.Vb?.Dispose();
            lc.PackK?.Dispose(); lc.PackV?.Dispose();
            lc.Bf16Scratch?.Dispose();
        }
        foreach (var r in _retired) r.Dispose();
        _retired.Clear();
    }
}
