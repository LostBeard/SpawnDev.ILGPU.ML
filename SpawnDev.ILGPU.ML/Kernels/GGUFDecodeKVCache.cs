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
    private readonly bool _isCuda;

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
    }

    private readonly LayerCache[] _layers;
    private readonly List<IDisposable> _retired = new();

    // ONE bf16 conversion scratch for the whole cache - allocated once, never resized, shared by every
    // layer. See Bf16Scratch() for why resizing it produced wrong tokens on WebGL.
    private MemoryBuffer1D<BFloat16, Stride1D.Dense>? _bf16Scratch;

    // bf16 path (ONE path, all backends): a CONTIGUOUS element-wise convert kernel (write-index == thread-index
    // — WebGL-safe, no Transform-Feedback scatter), then a queue/work-stream-ordered sync CopyFrom (via
    // CaptureSafeCopy) between the contiguous scratch and the maxSeq-strided store. (CopyFrom is now reliably
    // ordered against the convert kernel on every backend incl the Wasm worker pool — the race that once forced
    // CopyFromAsync here is fixed in SpawnDev.ILGPU.)
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
        _isCuda = accelerator.AcceleratorType == AcceleratorType.Cuda;
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

    // CUDA: a stream-ordered device-to-device enqueue (cuMemcpyAsync on DefaultStream, NO host
    // SynchronizeStream). This is (a) CUDA-graph-CAPTURE-safe — a synchronize is illegal during
    // cuStreamBeginCapture, and this records as a graph memcpy node instead — and (b) faster than
    // CopyFromAsync in normal decode: on CUDA the whole forward runs on ONE stream, so stream ordering
    // already guarantees the consumer kernel reads after this copy, making the per-copy sync that
    // CopyFromAsync adds pure overhead. Browser/other backends KEEP CopyFromAsync — their worker/queue
    // model needs its explicit ordering (a sync CopyFrom silently races on the Wasm worker pool; see the
    // bf16 note above). DefaultStream is the capture stream during capture (via Accelerator.WithDefaultStream).
    private Task CaptureSafeCopy<T>(ArrayView1D<T, Stride1D.Dense> dst, ArrayView1D<T, Stride1D.Dense> src)
        where T : unmanaged
    {
        if (_isCuda) { dst.CopyFrom(_accelerator.DefaultStream, src); return Task.CompletedTask; }
        // Every other backend: a sync CopyFrom is queue/work-stream-ordered — the consumer kernel reads it AFTER
        // the copy, no race. (The Wasm worker-pool race that once forced an awaited CopyFromAsync here is fixed in
        // SpawnDev.ILGPU; CopyFrom is now reliably ordered on all 6 backends, native CopyBufferToBuffer on WebGPU /
        // TF on WebGL.) So we drop the awaited per-copy GPU round-trip the browser pays dearly for — 2×nLayers per
        // decode token on WebGPU AND Wasm. CUDA keeps the explicit DefaultStream form for graph-capture safety.
        dst.CopyFrom(src);
        return Task.CompletedTask;
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
        // SEQ-MAJOR store [maxSeq, kvHeads, hd]: the per-step K/V arrive seq-major [nTokens, kvHeads, hd] (the graph
        // dropped the K/V pre-attention transpose, step 3), so each writes as ONE CONTIGUOUS run at row atToken — no
        // per-head strided copy. (CaptureSafeCopy = CopyFromAsync orders the copy against the producing kernel on the
        // Wasm worker pool — a sync CopyFrom of a node output silently races there.)
        long dstOff = (long)atToken * kvHeads * hd;
        long count = (long)nTokens * kvHeads * hd;
        if (_precision == KVCachePrecision.BF16)
        {
            // Convert f32→bf16 into a CONTIGUOUS scratch (kernel writes its OWN index — WebGL-safe), then one
            // CopyFromAsync scratch → store per K/V. K and V share the scratch, so V waits for K's copy.
            var scratch = Bf16Scratch();
            _f32ToBf16 ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>>(F32ToBf16Impl);
            _f32ToBf16((int)count, k, scratch);
            await CaptureSafeCopy(lc.Kb!.View.SubView(dstOff, count), scratch.SubView(0, count)).ConfigureAwait(false);
            _f32ToBf16((int)count, v, scratch);
            await CaptureSafeCopy(lc.Vb!.View.SubView(dstOff, count), scratch.SubView(0, count)).ConfigureAwait(false);
        }
        else
        {
            // F32: one contiguous CopyFromAsync each (k/v are graph node OUTPUTS; the async copy orders against
            // their producing kernel on Wasm — a sync CopyFrom of a node output silently races there).
            await Task.WhenAll(
                CaptureSafeCopy(lc.Kf!.View.SubView(dstOff, count), k.SubView(0, count)),
                CaptureSafeCopy(lc.Vf!.View.SubView(dstOff, count), v.SubView(0, count))).ConfigureAwait(false);
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
        // SEQ-MAJOR store [maxSeq, kvHeads, hd]: the live first-totalLen region is CONTIGUOUS [totalLen, kvHeads, hd]
        // at offset 0 — no per-head strided gather. The pack/Tensor the caller wraps is therefore seq-major too
        // (the WebGL+bf16 fallback passes seq_major_kv so FusedAttention reads it seq-major).
        long live = (long)kvHeads * totalLen * hd;
        if (_precision == KVCachePrecision.BF16)
        {
            // CopyFromAsync the live bf16 store region → a CONTIGUOUS scratch (a BUFFER copy — WebGL-safe), then
            // convert scratch → f32 pack. The convert KERNEL must NOT read the bf16 store directly: WebGL's bf16
            // sub-word kernel read of the store mis-addresses (the ILGPU limitation this whole fallback exists for).
            // Seq-major makes the live region contiguous, so it's ONE copy (was a per-head strided gather).
            var store = (isKey ? lc.Kb : lc.Vb)!.View;
            var scratch = Bf16Scratch();
            await CaptureSafeCopy(scratch.SubView(0, live), store.SubView(0, live)).ConfigureAwait(false);
            _bf16ToF32 ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<BFloat16, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(Bf16ToF32Impl);
            // BOTH views sliced to the LIVE region. Passing the full-capacity `pack` here was the
            // across-pack-growth bug: the pack is grow-only, so after its capacity doubles (64 -> 128)
            // dst.Length is the CAPACITY while src is only `live` long, and the kernel's bounds guard -
            // which tests dst.Length - stops excluding the padded threads an auto-grouped launch adds
            // beyond the extent. Those threads then read src OUT OF BOUNDS. While capacity happened to
            // equal live (every step up to 64) the guard was accidentally right, which is exactly why
            // this only broke after the first doubling.
            _bf16ToF32(kvHeads * totalLen * hd, scratch.SubView(0, live), pack.SubView(0, live));
        }
        else
        {
            // F32: one contiguous CopyFromAsync of the live region (this branch only fires if a non-WebGL caller
            // asks for a packed copy; WebGL+f32 reads the seq-major store strided directly).
            var store = (isKey ? lc.Kf : lc.Vf)!.View;
            await CaptureSafeCopy(pack.SubView(0, live), store.SubView(0, live)).ConfigureAwait(false);
        }
        return pack.SubView(0, live);
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

    /// <summary>
    /// The contiguous bf16 conversion scratch: allocated ONCE, at full size, and SHARED by every layer.
    /// </summary>
    /// <remarks>
    /// It used to be per-layer and grow-only (64 tokens, doubling). That produced wrong tokens on WebGL,
    /// and the growth was the trigger - ISOLATED by disabling each growth independently:
    ///
    ///   pack growth OFF, scratch growth ON  -> still fails, same step
    ///   pack growth ON,  scratch growth OFF -> PASSES
    ///
    /// So this buffer, not the pack, is what `GGUFDecodeKVCache_BF16_AcrossPackGrowth_MatchesFullRecompute`
    /// was catching - the test name and the earlier "the pack CAPACITY is leaking into the result" reading
    /// both blamed the wrong buffer. (The pack's capacity genuinely does differ from its live length, and
    /// pinning it changed nothing.)
    ///
    /// WHY resizing breaks it: this is the one BFloat16 buffer a kernel reads directly, and sub-word access
    /// is exactly where the WebGL backend is weak - the whole reason this scratch exists is that the kernel
    /// cannot read the bf16 STORE (see PackedAsync). Reallocating it at a different length changes the
    /// backing texture's layout, and the copy into it then mis-addresses. Never resizing it removes the
    /// class of bug rather than dodging one instance.
    ///
    /// Allocating once at maxSeqLen also uses LESS memory than the grow-only version it replaces, not more:
    /// the old one was PER LAYER, so at long context it converged on nLayers full-size buffers. This is one.
    /// It is safe to share because the scratch is transient within a single PackedAsync call - layers run
    /// sequentially, and K then V within a layer, which is the same ordering the per-layer buffer already
    /// relied on when K and V shared it.
    /// </remarks>
    private ArrayView1D<BFloat16, Stride1D.Dense> Bf16Scratch()
    {
        if (_bf16Scratch == null)
        {
            long elems = 0;
            foreach (var l in _layers) elems = Math.Max(elems, (long)l.KvHeads * _maxSeqLen * l.HeadDim);
            _bf16Scratch = _accelerator.Allocate1D<BFloat16>(elems);
        }
        return _bf16Scratch.View;
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
        // Guard BOTH views. An auto-grouped launch rounds the thread count up to the group size, so idx runs
        // past the extent; guarding only the destination silently permits an out-of-bounds READ whenever the
        // destination is longer than the source - which is the normal state of a GROW-ONLY buffer, and was the
        // across-pack-growth corruption. A guard that is only correct while two lengths happen to be equal is
        // not a guard.
        if (idx >= src.Length || idx >= dst.Length) return;
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
        // SEQ-MAJOR store [1, maxSeq, kvHeads, hd] (step 3): FusedAttention reads it with seqMajorKV (kvHead offset
        // hd, per-token stride kvHeads*hd) and IGNORES the p[10] kvRowStride. Shape kept truthful for the operator.
        var shape = new[] { 1, _maxSeqLen, lc.KvHeads, lc.HeadDim };
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
        }
        _bf16Scratch?.Dispose();
        foreach (var r in _retired) r.Dispose();
        _retired.Clear();
    }
}
