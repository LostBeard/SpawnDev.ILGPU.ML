using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Fused multi-head attention kernel — entire attention block in a single GPU dispatch.
/// Computes: softmax(mask(Q @ K^T / sqrt(d))) @ V in one kernel, eliminating
/// 3+ dispatch boundaries (command buffer submissions on WebGPU).
///
/// Standard (unfused) attention requires 5 dispatches:
///   1. Q @ K^T (MatMul)
///   2. Scale by 1/sqrt(d) (ElementWise)
///   3. Softmax (two-pass: max + exp+sum)
///   4. Scores @ V (MatMul)
///   5. Output projection
///
/// This kernel does steps 1-4 in a single dispatch per head, with the attention matrix
/// never materialized in global memory (online softmax, single pass over KV).
///
/// MASKING (decoder LLMs - gemma4 etc.): causal + sliding-window, computed from indices
/// in-kernel (no mask tensor). Query position = kvOffset + sq, so KV-cache decode
/// (seqQ = 1 at an arbitrary position) works. Sliding window is PER CALL: gemma-style
/// 5:1 SWA/global interleaves pass their per-layer window; global layers pass
/// window &gt;= seqKV (no window constraint). The caller wires per-layer values
/// (GGUF graph wiring owns that); this kernel just honors the parameters.
///
/// SHAPE RULE: the kernel body is BRANCH-FREE - no if/else, ternaries, or early returns
/// inside the KV loop. Masking is sign-bit arithmetic driving scores to -1e10 (exp -> 0),
/// and the online-softmax max-update is MathF.Max/Exp algebra (the correction factor is
/// exactly 1 when the max does not change). Any branch construct inlined into a loop body
/// multiplies the WebGL GLSL emitter's block duplication catastrophically (31.5MB shader
/// OOM - measured 2026-06-11, repro filed to the ILGPU lane; see FusedDequantMatMul).
/// </summary>
public class FusedAttentionKernel : IDisposable
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _kernel;

    // Strided + native-low-p K/V variant (the KV-cache decode path): K/V read in their native type
    // (BFloat16 / float) and converted to float in-register, with a per-head element STRIDE (p[10]) that
    // decouples the store's row pitch from the logical seqKV — so the cache reads its maxSeq-strided store
    // DIRECTLY (no per-token repack/widen). One compiled kernel per concrete T, cached.
    private readonly Dictionary<Type, object> _stridedKernels = new();
    // Barrier-free per-query kernel cache (the universal non-grouped path; replaces the per-element strided kernel).
    private readonly Dictionary<Type, object> _perQueryStridedKernels = new();
    private readonly Dictionary<Type, object> _perQueryRegisterKernels = new();

    // Dummy 1-element sinks buffer for the no-sinks case (the kernel always takes a sinks view but only
    // reads it when sinkCount > 0). gpt-oss/OpenAI-MoE attention passes real per-head sink logits.
    private MemoryBuffer1D<float, Stride1D.Dense>? _dummySinks;

    // Params-buffer ring: each call gets its own slot (per-layer window/kvOffset/seqKV differ within one batched
    // command encoder, so a single mutated buffer would feed pending dispatches the WRONG params), but the ring
    // bounds memory - a slot is reused only after RingSize subsequent calls, far past any realistic unflushed
    // batch depth. Each slot is allocated ONCE (at ParamSize) and reused via CopyFromCPU; an earlier version did
    // a fresh Allocate1D per call (~48 attention nodes/token => ~48 tiny GPU allocs/token).
    private const int RingSize = 64;
    private const int ParamSize = 16; // >= the largest params array (ForwardStrided = 11 ints); kernel reads only the prefix
    private readonly MemoryBuffer1D<int, Stride1D.Dense>?[] _paramsRing
        = new MemoryBuffer1D<int, Stride1D.Dense>?[RingSize];
    private int _ringNext;

    // CUDA-GRAPH CAPTURE: stable per-forward params slots. The ring (below) hands out a DIFFERENT device
    // pointer each call, so a captured kernel node would bake the capture-step's slot and replay (no host
    // code runs) would never see refreshed params. With stable slots, the k-th attention call of every
    // forward gets the SAME device buffer (counter auto-resets when GraphExecutor.ForwardGeneration ticks),
    // so the captured node reads a stable buffer the host can refresh between replays. During the actual
    // capture pass (GraphExecutor.SuppressDrains) the H2D is SKIPPED — a CopyFromCPU synchronizes, which is
    // illegal mid-capture; the slot already holds the immediately-preceding warm pass's params (same state).
    public static bool UseStableCaptureSlots;
    private const int CaptureSlotMax = 512;   // >= attention nodes per forward (28-layer qwen = 28)
    private readonly MemoryBuffer1D<int, Stride1D.Dense>?[] _captureSlots
        = new MemoryBuffer1D<int, Stride1D.Dense>?[CaptureSlotMax];
    private int _captureSlotNext;
    private long _captureSlotGen = -1;

    // Lazily allocate ring slot `_ringNext` and upload paramsData into it, returning the exact-length view.
    private ArrayView1D<int, Stride1D.Dense> RentParamsSlot(int[] paramsData)
    {
        if (UseStableCaptureSlots)
        {
            long gen = SpawnDev.ILGPU.ML.Graph.GraphExecutor.ForwardGeneration;
            if (gen != _captureSlotGen) { _captureSlotGen = gen; _captureSlotNext = 0; }
            int slot = _captureSlotNext++;
            var sbuf = _captureSlots[slot] ??= _accelerator.Allocate1D<int>(ParamSize);
            var sview = sbuf.View.SubView(0, paramsData.Length);
            // Skip the synchronizing H2D during capture; the warm pass already populated this stable slot.
            if (!SpawnDev.ILGPU.ML.Graph.GraphExecutor.SuppressDrains)
                sview.CopyFromCPU(paramsData);
            return sview;
        }
        var slotIdx = _ringNext;
        _ringNext = (_ringNext + 1) % RingSize;
        var buf = _paramsRing[slotIdx] ??= _accelerator.Allocate1D<int>(ParamSize);
        var view = buf.View.SubView(0, paramsData.Length);
        view.CopyFromCPU(paramsData);
        return view;
    }

    // ── Grouped-per-query attention (the prefill win) ──
    // The per-element kernels above launch ONE thread per (bh, sq, d) output element, and each thread
    // recomputes the FULL D-length Q·K dot once per output dim d — the score depends only on (bh, sq, kv),
    // so the dot is computed D times redundantly (D=128 → 128×). At long prompts this made FusedAttention
    // ~70% of prefill. The grouped kernel runs ONE thread GROUP per (bh, sq) query: phase 1 computes each
    // Q·K score EXACTLY ONCE into shared scores[SKV] (cooperatively across the group), then phase 2 has each
    // thread own a slice of the D output dims and replay the IDENTICAL online-softmax recurrence per dim
    // reading the shared score instead of recomputing the dot. Because the per-output-element math is
    // reproduced operation-for-operation (same dd-order dot, same Max/Exp recurrence, same sink epilogue),
    // the output is BIT-IDENTICAL to the per-element kernel — only the redundant dot work is removed.
    //
    // Shape: one group per (bh, sq) = nHeads*seqQ groups, AttnGroupSize threads each. The dot loop and the
    // V-accumulation are each O(SKV·D) per query (vs the per-element kernel's O(SKV·D²) dot), so ~D/2 less work.
    //
    // GATING (mirrors the GEMV / tiled-GEMM): shared scores[] is a fixed-size static allocation, so SKV must
    // be ≤ AttnSharedSkvMax and headDim ≤ AttnHeadDimMax — anything larger falls back to the per-element
    // kernel (huge-context prefill needs kv-tiled flash attention, the documented follow-up). Browser-GPU
    // backends keep the per-element kernel too: WebGL has no workgroup shared memory, and WebGPU's
    // workgroup reduction maps ~75× slow onto Tint/Dawn (same reason the cooperative GEMV is desktop-only).
    // Opt-in (env GGUF_ATTN_GROUP=1 / EnableGroupedAttention) until the full 6-backend sweep promotes it.
    private const int AttnGroupSize = 64;       // power of two; ≤ CPU's 64-thread group cap
    private const int AttnSharedSkvMax = 4096;  // the single-pass grouped kernel holds all scores in shared (16KB);
                                                // used when SKV ≤ this (the fast common case — fewer barriers, no
                                                // per-kv owned-dim branch than the tiled kernel below)
    private const int AttnKvBlock = 512;        // the KV-TILED kernel (SKV > AttnSharedSkvMax) processes KV one
                                                // 512-block at a time → scoresBlk[] is 2KB, so SKV is UNBOUNDED
    private const int AttnHeadDimMax = 256;     // qShared[]; covers gemma4's 256-dim heads. The tiled kernel has each
                                                // thread own up to AttnHeadDimMax/AttnGroupSize = 4 dims (wV0..wV3)
    public static bool EnableGroupedAttention =
        Environment.GetEnvironmentVariable("GGUF_ATTN_GROUP") == "1";

    // BENCHMARK/diagnostic only: force the legacy per-element attention (skip the barrier-free per-query path) so
    // a benchmark can A/B the two on the same backend. Leave false in production.
    public static bool DisablePerQuery;

    // Opt-in (GGUF_ATTN_REG=1) warp-cooperative REGISTER per-query attention: T=D/16 lanes cooperate per query,
    // each holding a const-16 REGISTER accumulator tile (Geordi's scalar-replace recipe) — no shared-mem slice,
    // no barrier; the Q·K dot is split across the T lanes + butterfly-reduced via Warp.ShuffleXor. CUDA-first
    // (warp==32 + Warp.Shuffle); other backends keep the shared-slice per-query. Requires D % 16 == 0.
    public static bool EnableRegisterAttention =
        Environment.GetEnvironmentVariable("GGUF_ATTN_REG") == "1";
    private const int RegTileD = 16; // per-lane register tile width (≤16 scalar-replaces; divides 64/128/256)

    private Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _groupedKernel, _tiledKernel;
    private readonly Dictionary<Type, object> _groupedStridedKernels = new();
    private readonly Dictionary<Type, object> _tiledStridedKernels = new();

    public FusedAttentionKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Fused bidirectional attention (no mask) - original API, behavior unchanged.
    /// Q [B*H, seqQ, D], K [B*H, seqKV, D], V [B*H, seqKV, D] → output [B*H, seqQ, D].
    /// </summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<float, Stride1D.Dense> K,
        ArrayView1D<float, Stride1D.Dense> V,
        ArrayView1D<float, Stride1D.Dense> output,
        int batchHeads, int seqQ, int seqKV, int headDim) =>
        Forward(Q, K, V, output, batchHeads, seqQ, seqKV, headDim,
            causal: false, window: int.MaxValue, kvOffset: 0);

    /// <summary>
    /// Fused attention with index-computed masking (Q and K/V share head count).
    /// Query position = <paramref name="kvOffset"/> + sq (pass the KV-cache length for
    /// single-token decode). <paramref name="causal"/> hides kv &gt; qPos.
    /// <paramref name="window"/> additionally hides kv &lt;= qPos - window (sliding-window
    /// attention; pass int.MaxValue or anything &gt;= seqKV for no window).
    /// </summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<float, Stride1D.Dense> K,
        ArrayView1D<float, Stride1D.Dense> V,
        ArrayView1D<float, Stride1D.Dense> output,
        int batchHeads, int seqQ, int seqKV, int headDim,
        bool causal, int window, int kvOffset) =>
        Forward(Q, K, V, output, batchHeads, batchHeads, seqQ, seqKV, headDim,
            causal, window, kvOffset, scale: 0f);

    /// <summary>
    /// Fused attention with masking, GROUPED-QUERY heads, and an explicit scale.
    /// Q [nHeads, seqQ, D]; K, V [kvHeads, seqKV, D]; output [nHeads, seqQ, D] - query
    /// head h attends kv head h / (nHeads / kvHeads) (GQA; gemma4 runs 8 kv heads on
    /// sliding layers and 1 on global layers). <paramref name="scale"/> &lt;= 0 means the
    /// default 1/sqrt(headDim); gemma-family models pass their query_pre_attn_scalar-
    /// derived value instead.
    /// </summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<float, Stride1D.Dense> K,
        ArrayView1D<float, Stride1D.Dense> V,
        ArrayView1D<float, Stride1D.Dense> output,
        int nHeads, int kvHeads, int seqQ, int seqKV, int headDim,
        bool causal, int window, int kvOffset, float scale,
        ArrayView1D<float, Stride1D.Dense>? sinks = null, int sinkCount = 0,
        bool seqMajorOut = false, bool seqMajorQ = false, bool seqMajorKV = false)
    {
        if (window <= 0) throw new ArgumentOutOfRangeException(nameof(window), "window must be positive");
        if (kvHeads <= 0 || nHeads % kvHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(kvHeads),
                $"kvHeads ({kvHeads}) must evenly divide nHeads ({nHeads}) for grouped-query attention.");
        // Clamp so the in-kernel sign-bit arithmetic cannot underflow int.MinValue:
        // any window >= seqKV + seqQ + kvOffset constrains nothing.
        long noConstraint = (long)seqKV + seqQ + Math.Max(kvOffset, 0) + 1;
        int effWindow = (int)Math.Min(window, noConstraint);

        // Exact float scale passed as raw bits (the old (int)(scale*10000) quantized the
        // scale to 1e-4 - a real precision loss for large headDim).
        float effScale = scale > 0f ? scale : 1f / MathF.Sqrt(headDim);
        var paramsData = new int[]
        {
            nHeads, seqQ, seqKV, headDim,
            BitConverter.SingleToInt32Bits(effScale),
            causal ? 1 : 0, effWindow, kvOffset,
            nHeads / kvHeads, // GQA group size: query head h reads kv head h / group
            sinkCount,        // p[9]: >0 => fold per-head sink logit into the softmax denominator
            seqKV * headDim,  // p[10]: kvRowStride (contiguous K/V = seqKV*headDim; read only by the per-query kernel)
            seqMajorOut ? 1 : 0, // p[11]: write output SEQ-major [1,seq,heads,hd] (oBase=(sq*BH+bh)*D) instead of
                                 // heads-major — lets the graph drop the post-attention Transpose[0,2,1,3] (universal).
            seqMajorQ ? 1 : 0,   // p[12]: read Q SEQ-major (qBase=(sq*BH+bh)*D) — lets the graph drop the Q
                                 // PRE-attention Transpose[0,2,1,3] (step 2; K/V keep theirs until the KV-cache goes seq-major).
            seqMajorKV ? 1 : 0,  // p[13]: read K/V SEQ-major ([kv,kvHeads,hd] → kBase=(kv*kvHeads+kvHead)*D, headStride=D,
                                 // tokenStride=kvHeads*D) — drops the K/V PRE-attention transposes (step 3, KV-cache seq-major).
        };

        var paramsView = RentParamsSlot(paramsData);

        _dummySinks ??= _accelerator.Allocate1D(new float[1]);
        var sinksView = sinks ?? _dummySinks.View;

        // Grouped-per-query path (the dot computed ONCE; bit-identical, opt-in, non-browser-GPU). SKV ≤ cap uses
        // the fast single-pass kernel (all scores in shared); larger SKV uses the KV-tiled kernel (unbounded).
        if (UseGrouped(seqKV, headDim))
        {
            var cfg = new KernelConfig(nHeads * seqQ, AttnGroupSize);
            if (seqKV <= AttnSharedSkvMax)
            {
                _groupedKernel ??= _accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<int, Stride1D.Dense>>(FusedAttentionGroupedImpl);
                _groupedKernel(cfg, Q, K, V, output, sinksView, paramsView);
            }
            else
            {
                _tiledKernel ??= _accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<int, Stride1D.Dense>>(FusedAttentionTiledImpl);
                _tiledKernel(cfg, Q, K, V, output, sinksView, paramsView);
            }
            return;
        }

        // Barrier-free per-query path (same as ForwardStrided): the universal non-grouped attention. T=float here
        // (contiguous K/V); with kvRowStride=seqKV*headDim (p[10] above) it's byte-identical to the per-element
        // kernel. Excludes WebGL (no workgroup shared memory) and headDim > MaxAttnHeadDimPQ → per-element below.
        if (!DisablePerQuery && headDim <= MaxAttnHeadDimPQ && _accelerator.AcceleratorType != AcceleratorType.WebGL)
        {
            if (!_perQueryStridedKernels.TryGetValue(typeof(float), out var pq))
                _perQueryStridedKernels[typeof(float)] = pq = _accelerator.LoadStreamKernel<
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(FusedAttentionPerQueryStridedImpl<float>);
            int pqBlocks = (nHeads * seqQ + PQGroup - 1) / PQGroup;
            ((Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>)pq)(
                new KernelConfig(pqBlocks, PQGroup), Q, K, V, output, sinksView, paramsView);
            return;
        }

        _kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(FusedAttentionImpl);

        // One thread per output element: nHeads * seqQ * headDim
        _kernel(nHeads * seqQ * headDim, Q, K, V, output, sinksView, paramsView);
    }

    /// <summary>
    /// Whether the grouped-per-query kernel applies: opted in, a non-browser-GPU backend (WebGL has no
    /// workgroup shared memory; WebGPU's workgroup reduction is ~75× slow on Tint/Dawn), and headDim fits the
    /// per-thread owned-dim unroll (≤ AttnHeadDimMax). SKV is unbounded (the kernel KV-tiles into shared blocks).
    /// Otherwise the per-element kernel runs.
    /// </summary>
    private bool UseGrouped(int seqKV, int headDim) =>
        EnableGroupedAttention
        && _accelerator.AcceleratorType != AcceleratorType.WebGL
        && _accelerator.AcceleratorType != AcceleratorType.WebGPU
        && headDim <= AttnHeadDimMax;

    /// <summary>
    /// Fused attention reading K/V in a NATIVE type <typeparamref name="T"/> (<c>BFloat16</c> or <c>float</c>),
    /// converted to float in-register, with an explicit per-head element stride <paramref name="kvRowStride"/>
    /// for K/V. This lets a KV-cache read its <c>[kvHeads, maxSeq, hd]</c> store DIRECTLY (pass
    /// <c>kvRowStride = maxSeq*hd</c>) instead of repacking + bf16→f32-widening the whole history into a
    /// contiguous f32 buffer every token. Q stays f32; output is f32. With <c>T=float</c> and
    /// <c>kvRowStride = seqKV*headDim</c> this is byte-identical to <see cref="Forward(ArrayView1D{float,Stride1D.Dense},ArrayView1D{float,Stride1D.Dense},ArrayView1D{float,Stride1D.Dense},ArrayView1D{float,Stride1D.Dense},int,int,int,int,int,bool,int,int,float,ArrayView1D{float,Stride1D.Dense}?,int)"/>.
    /// </summary>
    public void ForwardStrided<T>(
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<T, Stride1D.Dense> K,
        ArrayView1D<T, Stride1D.Dense> V,
        ArrayView1D<float, Stride1D.Dense> output,
        int nHeads, int kvHeads, int seqQ, int seqKV, int headDim,
        bool causal, int window, int kvOffset, float scale, int kvRowStride,
        ArrayView1D<float, Stride1D.Dense>? sinks = null, int sinkCount = 0,
        bool seqMajorOut = false, bool seqMajorQ = false, bool seqMajorKV = false)
        where T : unmanaged, INumber<T>
    {
        if (window <= 0) throw new ArgumentOutOfRangeException(nameof(window), "window must be positive");
        if (kvHeads <= 0 || nHeads % kvHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(kvHeads),
                $"kvHeads ({kvHeads}) must evenly divide nHeads ({nHeads}) for grouped-query attention.");
        long noConstraint = (long)seqKV + seqQ + Math.Max(kvOffset, 0) + 1;
        int effWindow = (int)Math.Min(window, noConstraint);
        float effScale = scale > 0f ? scale : 1f / MathF.Sqrt(headDim);
        var paramsData = new int[]
        {
            nHeads, seqQ, seqKV, headDim,
            BitConverter.SingleToInt32Bits(effScale),
            causal ? 1 : 0, effWindow, kvOffset,
            nHeads / kvHeads,
            sinkCount,
            kvRowStride, // p[10]: per-head element stride of K/V (maxSeq*hd for the strided store; seqKV*hd contiguous)
            seqMajorOut ? 1 : 0, // p[11]: write output SEQ-major (oBase=(sq*BH+bh)*D) — lets the graph drop the
                                 // post-attention Transpose[0,2,1,3] (universal data-movement elimination).
            seqMajorQ ? 1 : 0,   // p[12]: read Q SEQ-major (qBase=(sq*BH+bh)*D) — lets the graph drop the Q
                                 // PRE-attention Transpose[0,2,1,3] (step 2).
            seqMajorKV ? 1 : 0,  // p[13]: read K/V SEQ-major (kBase=(kv*kvHeads+kvHead)*D) — drops the K/V PRE-attention
                                 // transposes (step 3). For the strided decode store, the store itself is seq-major.
        };

        var paramsView = RentParamsSlot(paramsData);

        _dummySinks ??= _accelerator.Allocate1D(new float[1]);
        var sinksView = sinks ?? _dummySinks.View;

        // Grouped-per-query path (bit-identical, opt-in, non-browser-GPU) — this is the prefill hotpath (the
        // KV-cache strided bf16 read). SKV ≤ cap = fast single-pass kernel; larger SKV = KV-tiled (unbounded).
        if (UseGrouped(seqKV, headDim))
        {
            var dict = seqKV <= AttnSharedSkvMax ? _groupedStridedKernels : _tiledStridedKernels;
            if (!dict.TryGetValue(typeof(T), out var gk))
                dict[typeof(T)] = gk = _accelerator.LoadStreamKernel<
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                    ArrayView1D<T, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(
                    seqKV <= AttnSharedSkvMax ? FusedAttentionGroupedStridedImpl<T> : FusedAttentionTiledStridedImpl<T>);
            ((Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                ArrayView1D<T, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>)gk)(
                new KernelConfig(nHeads * seqQ, AttnGroupSize), Q, K, V, output, sinksView, paramsView);
            return;
        }

        // Non-grouped path (WebGPU always; desktop when grouped not opted in): the BARRIER-FREE per-query kernel
        // — one thread per (bh, sq), each accumulating all D outputs in its OWN shared-memory slice (no barrier,
        // no reduction), O(SKV·D) (kills the per-element kernel's D× redundant dot). Byte-identical output.
        // EXCLUDES WebGL: it has NO workgroup shared memory (Transform-Feedback model), so the shared slice is
        // invalid there — WebGL keeps the shared-mem-free per-element kernel. headDim ≤ MaxAttnHeadDimPQ (the
        // per-thread slice width); larger heads fall back to per-element.
        // Opt-in warp-cooperative REGISTER per-query (CUDA-first: warp==32 + Warp.Shuffle; D%16==0). T=D/16 lanes
        // share a query, each holding a 16-wide register acc tile; block = one warp holding 32/T queries.
        if (EnableRegisterAttention && _accelerator.AcceleratorType == AcceleratorType.Cuda
            && headDim % RegTileD == 0 && _accelerator.WarpSize == 32)
        {
            if (!_perQueryRegisterKernels.TryGetValue(typeof(T), out var rg))
                _perQueryRegisterKernels[typeof(T)] = rg = _accelerator.LoadStreamKernel<
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                    ArrayView1D<T, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(FusedAttentionPerQueryRegisterImpl<T>);
            int regT = headDim / RegTileD, regQPerWarp = 32 / regT;
            int regWarps = (nHeads * seqQ + regQPerWarp - 1) / regQPerWarp;
            ((Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                ArrayView1D<T, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>)rg)(
                new KernelConfig(regWarps, 32), Q, K, V, output, sinksView, paramsView);
            return;
        }

        if (!DisablePerQuery && headDim <= MaxAttnHeadDimPQ && _accelerator.AcceleratorType != AcceleratorType.WebGL)
        {
            if (!_perQueryStridedKernels.TryGetValue(typeof(T), out var pq))
                _perQueryStridedKernels[typeof(T)] = pq = _accelerator.LoadStreamKernel<
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                    ArrayView1D<T, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(FusedAttentionPerQueryStridedImpl<T>);
            int pqQueries = nHeads * seqQ;
            int pqBlocks = (pqQueries + PQGroup - 1) / PQGroup;
            ((Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                ArrayView1D<T, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>)pq)(
                new KernelConfig(pqBlocks, PQGroup), Q, K, V, output, sinksView, paramsView);
            return;
        }

        if (!_stridedKernels.TryGetValue(typeof(T), out var k))
            _stridedKernels[typeof(T)] = k = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                ArrayView1D<T, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(FusedAttentionStridedImpl<T>);

        ((Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
            ArrayView1D<T, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>)k)(
            nHeads * seqQ * headDim, Q, K, V, output, sinksView, paramsView);
    }

    /// <summary>
    /// Per-element fused attention with Online Softmax (single pass) and arithmetic
    /// masking. Each thread computes one output value by iterating all KV positions.
    /// BRANCH-FREE BODY (see class doc): masked positions get score -1e10 via sign-bit
    /// masks (their exp underflows to 0 and they never win the running max); the
    /// max-update is pure Max/Exp algebra (correction factor is exactly 1 when the
    /// running max is unchanged, so unconditional application is identical math).
    /// </summary>
    private static void FusedAttentionImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<float, Stride1D.Dense> K,
        ArrayView1D<float, Stride1D.Dense> V,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> sinks,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int BH = p[0], SQ = p[1], SKV = p[2], D = p[3];
        float scale = Interop.IntAsFloat((uint)p[4]);
        int causal = p[5];
        int window = p[6];
        int kvOffset = p[7];
        int gqaGroup = p[8]; // nHeads / kvHeads; query head bh reads kv head bh / gqaGroup
        int sinkCount = p[9]; // 0 => no sinks; else per-head sink logit count (gpt-oss attention sinks)

        // Decompose index. p[11]=1 → idx enumerates the SEQ-MAJOR output [1,seq,heads,hd] (head varies faster
        // than seq), so each thread's OWN slot output[idx] already IS its seq-major position — one store per
        // thread, NO scatter, so it's WebGL-safe (and lets the graph drop the post-attention transpose). The Q
        // read still uses the heads-major qBase. p[11]=0 → heads-major [1,heads,seq,hd] (the original layout).
        int d = idx % D;
        int sq, bh;
        if (p[11] == 1) { bh = (idx / D) % BH; sq = idx / (BH * D); }
        else { sq = (idx / D) % SQ; bh = idx / (SQ * D); }

        if (bh >= BH) return;

        int kvHead = bh / gqaGroup;
        // Q read base: seq-major (sq*BH+bh)*D when p[12]=1 (graph dropped the Q pre-transpose), else heads-major.
        int qBase = p[12] == 1 ? (sq * BH + bh) * D : (bh * SQ + sq) * D;
        int qPos = kvOffset + sq;
        // K/V layout (p[13]=1 seq-major [kv,kvHeads,hd]: head offset D, token stride kvHeads*D; else head-major
        // contiguous [kvHeads,SKV,hd]: head offset SKV*D, token D). Drops the K/V pre-attention transposes.
        int kvHeadStride = p[13] == 1 ? D : SKV * D;
        int kvTokenStride = p[13] == 1 ? (BH / gqaGroup) * D : D;

        float runningMax = -1e10f;
        float runningSum = 0f;
        float weightedV = 0f;

        for (int kv = 0; kv < SKV; kv++)
        {
            int kBase = kvHead * kvHeadStride + kv * kvTokenStride;
            float dot = 0f;
            for (int dd = 0; dd < D; dd++)
                dot += Q[qBase + dd] * K[kBase + dd];
            float score = dot * scale;

            // valid = (!causal || kv <= qPos) && (kv > qPos - window), as 0/1 ints:
            // causalOk: causal=0 -> always 1; else 1 when qPos - kv >= 0.
            // windowOk: 1 when qPos - window - kv < 0 (sign bit).
            int causalOk = 1 - (causal & ((qPos - kv) >> 31) & 1);
            int windowOk = ((qPos - window - kv) >> 31) & 1;
            int valid = causalOk & windowOk;
            // Masked score must sit STRICTLY below the runningMax init (-1e10): at
            // exactly -1e10 a masked position preceding any valid one would hit
            // exp(score - newMax) = exp(0) = 1 and contribute full weight. -1e30
            // keeps the init as the max and its exp underflows cleanly to 0.
            score = score * valid + -1e30f * (1 - valid);

            // Branch-free online softmax: when score <= runningMax the correction is
            // exp(0) = 1 and nothing shifts; otherwise the running terms rescale.
            float newMax = MathF.Max(runningMax, score);
            float correction = MathF.Exp(runningMax - newMax);
            float weight = MathF.Exp(score - newMax);
            runningSum = runningSum * correction + weight;
            weightedV = weightedV * correction + weight * V[kBase + d];
            runningMax = newMax;
        }

        // Attention sinks (gpt-oss): a per-head learned logit joins the softmax as if it were one more score
        // whose value vector is 0 - it participates in the max + denominator but adds nothing to the numerator.
        // Equivalent to concatenating the sink to the scores before softmax. sinkCount is uniform across all
        // threads (no warp divergence). sink index = head within the batch (bh % sinkCount).
        if (sinkCount > 0)
        {
            float sink = sinks[bh % sinkCount];
            float newMax = MathF.Max(runningMax, sink);
            float correction = MathF.Exp(runningMax - newMax);
            runningSum = runningSum * correction + MathF.Exp(sink - newMax);
            weightedV = weightedV * correction; // sink contributes 0 to the value sum
            runningMax = newMax;
        }

        output[idx] = weightedV / (runningSum + 1e-10f);
    }

    /// <summary>
    /// Strided + native-low-p K/V variant of <see cref="FusedAttentionImpl"/>: identical online-softmax math
    /// and branch-free masking, but K/V are read in type <typeparamref name="T"/> and converted to float
    /// in-register (<c>PrecisionConvert.ConvertToSingle</c> — branchless; <c>T=float</c> lowers to nothing), and
    /// the per-head base uses the explicit stride <c>p[10]</c> instead of <c>SKV</c>. With <c>T=float</c> and
    /// <c>p[10]=SKV*D</c> it is byte-identical to <see cref="FusedAttentionImpl"/> (the correctness anchor).
    /// </summary>
    private static void FusedAttentionStridedImpl<T>(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<T, Stride1D.Dense> K,
        ArrayView1D<T, Stride1D.Dense> V,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> sinks,
        ArrayView1D<int, Stride1D.Dense> p)
        where T : unmanaged, INumber<T>
    {
        int BH = p[0], SQ = p[1], SKV = p[2], D = p[3];
        float scale = Interop.IntAsFloat((uint)p[4]);
        int causal = p[5];
        int window = p[6];
        int kvOffset = p[7];
        int gqaGroup = p[8];
        int sinkCount = p[9];
        int kvStride = p[10]; // per-head element stride of K/V (decouples the store row pitch from SKV)

        // p[11]=1 → idx enumerates the SEQ-MAJOR output (head faster than seq) so output[idx] is the thread's own
        // seq-major slot (no scatter, WebGL-safe; graph drops the post-attn transpose). Q read stays heads-major.
        int d = idx % D;
        int sq, bh;
        if (p[11] == 1) { bh = (idx / D) % BH; sq = idx / (BH * D); }
        else { sq = (idx / D) % SQ; bh = idx / (SQ * D); }
        if (bh >= BH) return;

        int kvHead = bh / gqaGroup;
        // Q read base: seq-major (sq*BH+bh)*D when p[12]=1 (graph dropped the Q pre-transpose), else heads-major.
        int qBase = p[12] == 1 ? (sq * BH + bh) * D : (bh * SQ + sq) * D;
        int qPos = kvOffset + sq;
        // K/V layout (p[13]=1 seq-major store [kv,kvHeads,hd]: head offset D, token stride kvHeads*D; else
        // head-major strided: head offset kvStride, token D). Drops the K/V pre-attention transposes (step 3).
        int kvHeadStride = p[13] == 1 ? D : kvStride;
        int kvTokenStride = p[13] == 1 ? (BH / gqaGroup) * D : D;

        float runningMax = -1e10f;
        float runningSum = 0f;
        float weightedV = 0f;

        for (int kv = 0; kv < SKV; kv++)
        {
            int kBase = kvHead * kvHeadStride + kv * kvTokenStride;
            float dot = 0f;
            for (int dd = 0; dd < D; dd++)
                dot += Q[qBase + dd] * PrecisionConvert.ConvertToSingle(K[kBase + dd]);
            float score = dot * scale;

            int causalOk = 1 - (causal & ((qPos - kv) >> 31) & 1);
            int windowOk = ((qPos - window - kv) >> 31) & 1;
            int valid = causalOk & windowOk;
            score = score * valid + -1e30f * (1 - valid);

            float newMax = MathF.Max(runningMax, score);
            float correction = MathF.Exp(runningMax - newMax);
            float weight = MathF.Exp(score - newMax);
            runningSum = runningSum * correction + weight;
            weightedV = weightedV * correction + weight * PrecisionConvert.ConvertToSingle(V[kBase + d]);
            runningMax = newMax;
        }

        if (sinkCount > 0)
        {
            float sink = sinks[bh % sinkCount];
            float newMax = MathF.Max(runningMax, sink);
            float correction = MathF.Exp(runningMax - newMax);
            runningSum = runningSum * correction + MathF.Exp(sink - newMax);
            weightedV = weightedV * correction;
            runningMax = newMax;
        }

        output[idx] = weightedV / (runningSum + 1e-10f);
    }

    // Per-query kernel: threads/block and the max headDim each thread's shared-memory accumulator slice covers
    // (gemma4 = 256, qwen = 128). Shared size = PQGroup * MaxAttnHeadDimPQ floats = 8*256*4 = 8 KB (within
    // WebGPU's 16 KB workgroup-storage limit). PQGroup is small because each thread owns a FULL D-wide slice.
    private const int MaxAttnHeadDimPQ = 256;
    private const int PQGroup = 8;

    /// <summary>
    /// BARRIER-FREE per-query attention — ONE thread per (bh, sq). Identical online-softmax + branch-free masking
    /// to <see cref="FusedAttentionStridedImpl{T}"/>, but each thread computes the Q·K dot ONCE per kv and
    /// accumulates ALL D outputs — so the per-element kernel's D× redundant dot is gone (O(SKV·D) per query vs
    /// O(SKV·D²)). The D accumulators live in the thread's OWN slice of workgroup shared memory (no cross-thread
    /// access → NO <c>Group.Barrier</c>, NO workgroup reduction). Because the running max/sum/correction sequence
    /// is independent of the output dim, the per-dim result is BYTE-IDENTICAL to the per-element kernel (the
    /// correctness anchor). The point: no barrier/reduction means it runs FAST on WebGPU/WebGL (where the grouped
    /// kernel's reduction is ~75× slow on Tint/Dawn and is therefore disabled) AND on every other backend — the
    /// universal non-grouped attention path. (Shared mem is used as per-thread scratch only because ILGPU cannot
    /// yet lower a dynamically-indexed device-LOCAL array in this kernel — flagged to Geordi for the cleaner form.)
    /// </summary>
    private static void FusedAttentionPerQueryStridedImpl<T>(
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<T, Stride1D.Dense> K,
        ArrayView1D<T, Stride1D.Dense> V,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> sinks,
        ArrayView1D<int, Stride1D.Dense> p)
        where T : unmanaged, INumber<T>
    {
        int BH = p[0], SQ = p[1], SKV = p[2], D = p[3];
        float scale = Interop.IntAsFloat((uint)p[4]);
        int causal = p[5];
        int window = p[6];
        int kvOffset = p[7];
        int gqaGroup = p[8];
        int sinkCount = p[9];
        int kvStride = p[10];

        int query = Grid.IdxX * PQGroup + Group.IdxX;   // global query index = (bh, sq)
        if (query >= BH * SQ) return;
        int sq = query % SQ;
        int bh = query / SQ;

        int kvHead = bh / gqaGroup;
        // Per-thread bases: heads-major [1,heads,seq,hd] vs seq-major [1,seq,heads,hd]. p[12]=1 → READ Q seq-major
        // (graph dropped the Q pre-transpose, step 2); p[11]=1 → WRITE output seq-major (dropped the post-attention
        // transpose, step 1). Independent flags (step 1 shipped with Q still heads-major / p[12]=0).
        int hBase = (bh * SQ + sq) * D, sBase = (sq * BH + bh) * D;
        int qBase = p[12] == 1 ? sBase : hBase;
        int oBase = p[11] == 1 ? sBase : hBase;
        int qPos = kvOffset + sq;
        // K/V layout (p[13]=1 seq-major store [kv,kvHeads,hd]: head offset D, token stride kvHeads*D; else head-major
        // strided: head offset kvStride, token D). Drops the K/V pre-attention transposes (step 3).
        int kvHeadStride = p[13] == 1 ? D : kvStride;
        int kvTokenStride = p[13] == 1 ? (BH / gqaGroup) * D : D;

        // This thread's private D-wide accumulator slice in shared memory (no other thread touches it).
        var sh = SharedMemory.Allocate<float>(PQGroup * MaxAttnHeadDimPQ);
        int accBase = Group.IdxX * MaxAttnHeadDimPQ;
        for (int dd = 0; dd < D; dd++) sh[accBase + dd] = 0f;

        float runningMax = -1e10f;
        float runningSum = 0f;

        for (int kv = 0; kv < SKV; kv++)
        {
            int kBase = kvHead * kvHeadStride + kv * kvTokenStride;
            float dot = 0f;
            for (int dd = 0; dd < D; dd++)
                dot += Q[qBase + dd] * PrecisionConvert.ConvertToSingle(K[kBase + dd]);
            float score = dot * scale;

            int causalOk = 1 - (causal & ((qPos - kv) >> 31) & 1);
            int windowOk = ((qPos - window - kv) >> 31) & 1;
            int valid = causalOk & windowOk;
            score = score * valid + -1e30f * (1 - valid);

            float newMax = MathF.Max(runningMax, score);
            float correction = MathF.Exp(runningMax - newMax);
            float weight = MathF.Exp(score - newMax);
            runningSum = runningSum * correction + weight;
            for (int dd = 0; dd < D; dd++)
                sh[accBase + dd] = sh[accBase + dd] * correction + weight * PrecisionConvert.ConvertToSingle(V[kBase + dd]);
            runningMax = newMax;
        }

        if (sinkCount > 0)
        {
            float sink = sinks[bh % sinkCount];
            float newMax = MathF.Max(runningMax, sink);
            float correction = MathF.Exp(runningMax - newMax);
            runningSum = runningSum * correction + MathF.Exp(sink - newMax);
            for (int dd = 0; dd < D; dd++) sh[accBase + dd] = sh[accBase + dd] * correction;
            runningMax = newMax;
        }

        float inv = 1f / (runningSum + 1e-10f);
        for (int dd = 0; dd < D; dd++)
            output[oBase + dd] = sh[accBase + dd] * inv;
    }

    /// <summary>Warp-cooperative REGISTER per-query attention (opt-in GGUF_ATTN_REG, CUDA-first). T = D/RegTileD
    /// lanes cooperate on ONE query; each lane owns dims [t·16,(t+1)·16) for BOTH the Q·K dot-partial AND the output,
    /// holding its 16 online-softmax accumulators in REGISTERS (the const-16 array scalar-replaces — Geordi's recipe;
    /// NO shared-mem slice, NO barrier). Per kv: each lane computes its partial dot; the T lanes butterfly-reduce it
    /// via Warp.ShuffleXor (aligned power-of-2 group → every lane gets the full dot); then each lane runs the SAME
    /// scalar online-softmax recurrence and updates its 16 register accs. Same masking/recurrence as
    /// FusedAttentionPerQueryStridedImpl; the dot is summed per-tile+shuffle (vs sequential) so it matches to GEMV
    /// float-reduction tolerance (argmax-identical). Requires warp==32 + D%16==0; block = one warp holding 32/T queries.</summary>
    private static void FusedAttentionPerQueryRegisterImpl<T>(
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<T, Stride1D.Dense> K,
        ArrayView1D<T, Stride1D.Dense> V,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> sinks,
        ArrayView1D<int, Stride1D.Dense> p)
        where T : unmanaged, INumber<T>
    {
        int BH = p[0], SQ = p[1], SKV = p[2], D = p[3];
        float scale = Interop.IntAsFloat((uint)p[4]);
        int causal = p[5];
        int window = p[6];
        int kvOffset = p[7];
        int gqaGroup = p[8];
        int sinkCount = p[9];
        int kvStride = p[10];

        int lane = Group.IdxX;          // 0..31 (block = one 32-lane warp)
        int nLanes = D / RegTileD;      // lanes cooperating per query (4/8/16)
        int qPerWarp = 32 / nLanes;     // queries per warp
        int query = Grid.IdxX * qPerWarp + lane / nLanes;
        int t = lane % nLanes;          // this lane's dim tile
        if (query >= BH * SQ) return;

        int sq = query % SQ;
        int bh = query / SQ;
        int kvHead = bh / gqaGroup;
        int hBase = (bh * SQ + sq) * D, sBase = (sq * BH + bh) * D;
        int qBase = p[12] == 1 ? sBase : hBase;
        int oBase = p[11] == 1 ? sBase : hBase;
        int qPos = kvOffset + sq;
        int kvHeadStride = p[13] == 1 ? D : kvStride;
        int kvTokenStride = p[13] == 1 ? (BH / gqaGroup) * D : D;
        int myDim = t * RegTileD;       // first head-dim this lane owns

        var acc = new float[RegTileD];  // REGISTERS (const 16 → scalar-replaced; no shared mem)
        for (int d = 0; d < RegTileD; d++) acc[d] = 0f;
        float runningMax = -1e10f;
        float runningSum = 0f;

        for (int kv = 0; kv < SKV; kv++)
        {
            int kBase = kvHead * kvHeadStride + kv * kvTokenStride;
            float pd = 0f;
            for (int d = 0; d < RegTileD; d++)
                pd += Q[qBase + myDim + d] * PrecisionConvert.ConvertToSingle(K[kBase + myDim + d]);
            // butterfly-reduce the partial dot across this query's nLanes (aligned power-of-2 → all lanes get it).
            for (int off = nLanes >> 1; off > 0; off >>= 1)
                pd += Warp.ShuffleXor(pd, off);
            float score = pd * scale;
            int causalOk = 1 - (causal & ((qPos - kv) >> 31) & 1);
            int windowOk = ((qPos - window - kv) >> 31) & 1;
            int valid = causalOk & windowOk;
            score = score * valid + -1e30f * (1 - valid);

            float newMax = MathF.Max(runningMax, score);
            float correction = MathF.Exp(runningMax - newMax);
            float weight = MathF.Exp(score - newMax);
            runningSum = runningSum * correction + weight;
            for (int d = 0; d < RegTileD; d++)
                acc[d] = acc[d] * correction + weight * PrecisionConvert.ConvertToSingle(V[kBase + myDim + d]);
            runningMax = newMax;
        }

        if (sinkCount > 0)
        {
            float sink = sinks[bh % sinkCount];
            float newMax = MathF.Max(runningMax, sink);
            float correction = MathF.Exp(runningMax - newMax);
            runningSum = runningSum * correction + MathF.Exp(sink - newMax);
            for (int d = 0; d < RegTileD; d++) acc[d] = acc[d] * correction;
            runningMax = newMax;
        }

        float inv = 1f / (runningSum + 1e-10f);
        for (int d = 0; d < RegTileD; d++)
            output[oBase + myDim + d] = acc[d] * inv;
    }

    /// <summary>
    /// Single-pass grouped-per-query attention (contiguous f32 K/V) — the fast path for SKV ≤ AttnSharedSkvMax.
    /// One thread GROUP per (bh, sq): phase 1 computes EVERY Q·K score once into shared <c>scores[]</c>; phase 2
    /// has each thread own a slice of the D output dims and replay the per-element online softmax over the resident
    /// scores. Fewer barriers + a simpler inner loop than the KV-tiled kernel, so it's faster when all scores fit
    /// shared. BIT-IDENTICAL to <see cref="FusedAttentionImpl"/> (same dd-order dot, same per-kv recurrence order).
    /// </summary>
    private static void FusedAttentionGroupedImpl(
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<float, Stride1D.Dense> K,
        ArrayView1D<float, Stride1D.Dense> V,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> sinks,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int BH = p[0], SQ = p[1], SKV = p[2], D = p[3];
        float scale = Interop.IntAsFloat((uint)p[4]);
        int causal = p[5];
        int window = p[6];
        int kvOffset = p[7];
        int gqaGroup = p[8];
        int sinkCount = p[9];

        int g = Grid.IdxX;
        int tid = Group.IdxX;
        int bh = g / SQ;
        int sq = g % SQ;

        int kvHead = bh / gqaGroup;
        // Per-thread bases: heads-major [1,heads,seq,hd] vs seq-major [1,seq,heads,hd]. p[12]=1 → READ Q seq-major
        // (graph dropped the Q pre-transpose, step 2); p[11]=1 → WRITE output seq-major (dropped the post-attention
        // transpose, step 1). Independent flags (step 1 shipped with Q still heads-major / p[12]=0).
        int hBase = (bh * SQ + sq) * D, sBase = (sq * BH + bh) * D;
        int qBase = p[12] == 1 ? sBase : hBase;
        int oBase = p[11] == 1 ? sBase : hBase;
        int qPos = kvOffset + sq;
        // K/V layout (p[13]=1 seq-major [kv,kvHeads,hd]: head offset D, token stride kvHeads*D; else head-major
        // contiguous: head offset SKV*D, token D). Drops the K/V pre-attention transposes (step 3).
        int kvHeadStride = p[13] == 1 ? D : SKV * D;
        int kvTokenStride = p[13] == 1 ? (BH / gqaGroup) * D : D;

        var qSh = SharedMemory.Allocate<float>(AttnHeadDimMax);
        var scores = SharedMemory.Allocate<float>(AttnSharedSkvMax);

        for (int dd = tid; dd < D; dd += AttnGroupSize)
            qSh[dd] = Q[qBase + dd];
        Group.Barrier();

        for (int kv = tid; kv < SKV; kv += AttnGroupSize)
        {
            int kBase = kvHead * kvHeadStride + kv * kvTokenStride;
            float dot = 0f;
            for (int dd = 0; dd < D; dd++)
                dot += qSh[dd] * K[kBase + dd];
            float score = dot * scale;
            int causalOk = 1 - (causal & ((qPos - kv) >> 31) & 1);
            int windowOk = ((qPos - window - kv) >> 31) & 1;
            int valid = causalOk & windowOk;
            scores[kv] = score * valid + -1e30f * (1 - valid);
        }
        Group.Barrier();

        for (int d = tid; d < D; d += AttnGroupSize)
        {
            float runningMax = -1e10f;
            float runningSum = 0f;
            float weightedV = 0f;
            for (int kv = 0; kv < SKV; kv++)
            {
                float score = scores[kv];
                float newMax = MathF.Max(runningMax, score);
                float correction = MathF.Exp(runningMax - newMax);
                float weight = MathF.Exp(score - newMax);
                runningSum = runningSum * correction + weight;
                weightedV = weightedV * correction + weight * V[kvHead * kvHeadStride + kv * kvTokenStride + d];
                runningMax = newMax;
            }
            if (sinkCount > 0)
            {
                float sink = sinks[bh % sinkCount];
                float newMax = MathF.Max(runningMax, sink);
                float correction = MathF.Exp(runningMax - newMax);
                runningSum = runningSum * correction + MathF.Exp(sink - newMax);
                weightedV = weightedV * correction;
                runningMax = newMax;
            }
            output[oBase + d] = weightedV / (runningSum + 1e-10f);
        }
    }

    /// <summary>Strided + native-low-p <typeparamref name="T"/> variant of <see cref="FusedAttentionGroupedImpl"/>
    /// (single-pass, SKV ≤ AttnSharedSkvMax). Byte-identical at T=float, p[10]=SKV*D.</summary>
    private static void FusedAttentionGroupedStridedImpl<T>(
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<T, Stride1D.Dense> K,
        ArrayView1D<T, Stride1D.Dense> V,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> sinks,
        ArrayView1D<int, Stride1D.Dense> p)
        where T : unmanaged, INumber<T>
    {
        int BH = p[0], SQ = p[1], SKV = p[2], D = p[3];
        float scale = Interop.IntAsFloat((uint)p[4]);
        int causal = p[5];
        int window = p[6];
        int kvOffset = p[7];
        int gqaGroup = p[8];
        int sinkCount = p[9];
        int kvStride = p[10];

        int g = Grid.IdxX;
        int tid = Group.IdxX;
        int bh = g / SQ;
        int sq = g % SQ;

        int kvHead = bh / gqaGroup;
        // Per-thread bases: heads-major [1,heads,seq,hd] vs seq-major [1,seq,heads,hd]. p[12]=1 → READ Q seq-major
        // (graph dropped the Q pre-transpose, step 2); p[11]=1 → WRITE output seq-major (dropped the post-attention
        // transpose, step 1). Independent flags (step 1 shipped with Q still heads-major / p[12]=0).
        int hBase = (bh * SQ + sq) * D, sBase = (sq * BH + bh) * D;
        int qBase = p[12] == 1 ? sBase : hBase;
        int oBase = p[11] == 1 ? sBase : hBase;
        int qPos = kvOffset + sq;
        // K/V layout (p[13]=1 seq-major store [kv,kvHeads,hd]: head offset D, token stride kvHeads*D; else head-major
        // strided: head offset kvStride, token D). Drops the K/V pre-attention transposes (step 3).
        int kvHeadStride = p[13] == 1 ? D : kvStride;
        int kvTokenStride = p[13] == 1 ? (BH / gqaGroup) * D : D;

        var qSh = SharedMemory.Allocate<float>(AttnHeadDimMax);
        var scores = SharedMemory.Allocate<float>(AttnSharedSkvMax);

        for (int dd = tid; dd < D; dd += AttnGroupSize)
            qSh[dd] = Q[qBase + dd];
        Group.Barrier();

        for (int kv = tid; kv < SKV; kv += AttnGroupSize)
        {
            int kBase = kvHead * kvHeadStride + kv * kvTokenStride;
            float dot = 0f;
            for (int dd = 0; dd < D; dd++)
                dot += qSh[dd] * PrecisionConvert.ConvertToSingle(K[kBase + dd]);
            float score = dot * scale;
            int causalOk = 1 - (causal & ((qPos - kv) >> 31) & 1);
            int windowOk = ((qPos - window - kv) >> 31) & 1;
            int valid = causalOk & windowOk;
            scores[kv] = score * valid + -1e30f * (1 - valid);
        }
        Group.Barrier();

        for (int d = tid; d < D; d += AttnGroupSize)
        {
            float runningMax = -1e10f;
            float runningSum = 0f;
            float weightedV = 0f;
            for (int kv = 0; kv < SKV; kv++)
            {
                float score = scores[kv];
                float newMax = MathF.Max(runningMax, score);
                float correction = MathF.Exp(runningMax - newMax);
                float weight = MathF.Exp(score - newMax);
                runningSum = runningSum * correction + weight;
                weightedV = weightedV * correction + weight * PrecisionConvert.ConvertToSingle(V[kvHead * kvHeadStride + kv * kvTokenStride + d]);
                runningMax = newMax;
            }
            if (sinkCount > 0)
            {
                float sink = sinks[bh % sinkCount];
                float newMax = MathF.Max(runningMax, sink);
                float correction = MathF.Exp(runningMax - newMax);
                runningSum = runningSum * correction + MathF.Exp(sink - newMax);
                weightedV = weightedV * correction;
                runningMax = newMax;
            }
            output[oBase + d] = weightedV / (runningSum + 1e-10f);
        }
    }

    /// <summary>
    /// KV-TILED grouped attention (contiguous f32 K/V) — the unbounded-SKV path (SKV &gt; AttnSharedSkvMax).
    /// One thread GROUP per (bh, sq); KV is processed one <see cref="AttnKvBlock"/> at a time so only a single
    /// block of scores is resident, making SKV unbounded. Each thread keeps ONE online softmax shared across its
    /// up-to-4 owned output dims + a weightedV per dim, and the per-kv recurrence runs IN ORDER, so the result is
    /// BIT-IDENTICAL to <see cref="FusedAttentionGroupedImpl"/> and the per-element <see cref="FusedAttentionImpl"/>.
    /// </summary>
    private static void FusedAttentionTiledImpl(
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<float, Stride1D.Dense> K,
        ArrayView1D<float, Stride1D.Dense> V,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> sinks,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int BH = p[0], SQ = p[1], SKV = p[2], D = p[3];
        float scale = Interop.IntAsFloat((uint)p[4]);
        int causal = p[5];
        int window = p[6];
        int kvOffset = p[7];
        int gqaGroup = p[8];
        int sinkCount = p[9];

        int g = Grid.IdxX;        // one group per (bh, sq)
        int tid = Group.IdxX;     // 0..AttnGroupSize-1
        int bh = g / SQ;
        int sq = g % SQ;

        int kvHead = bh / gqaGroup;
        // Per-thread bases: heads-major [1,heads,seq,hd] vs seq-major [1,seq,heads,hd]. p[12]=1 → READ Q seq-major
        // (graph dropped the Q pre-transpose, step 2); p[11]=1 → WRITE output seq-major (dropped the post-attention
        // transpose, step 1). Independent flags (step 1 shipped with Q still heads-major / p[12]=0).
        int hBase = (bh * SQ + sq) * D, sBase = (sq * BH + bh) * D;
        int qBase = p[12] == 1 ? sBase : hBase;
        int oBase = p[11] == 1 ? sBase : hBase;
        int qPos = kvOffset + sq;
        // K/V layout (p[13]=1 seq-major [kv,kvHeads,hd]: head offset D, token stride kvHeads*D; else head-major
        // contiguous: head offset SKV*D, token D). Drops the K/V pre-attention transposes (step 3).
        int kvHeadStride = p[13] == 1 ? D : SKV * D;
        int kvTokenStride = p[13] == 1 ? (BH / gqaGroup) * D : D;

        var qSh = SharedMemory.Allocate<float>(AttnHeadDimMax);
        var scoresBlk = SharedMemory.Allocate<float>(AttnKvBlock);

        // Load this query's Q row into shared once (an exact copy → the dot stays bit-identical).
        for (int dd = tid; dd < D; dd += AttnGroupSize)
            qSh[dd] = Q[qBase + dd];
        Group.Barrier();

        // Each thread owns up to 4 output dims d0..d3 = tid + {0,1,2,3}·G (< D), keeping ONE online softmax
        // (runningMax/runningSum, identical across the dims since they depend only on the scores) and a weightedV
        // per owned dim. KV is processed in blocks: each block's scores are computed ONCE into shared scoresBlk[],
        // then the per-kv online recurrence runs over the block IN ORDER — the SAME per-kv order as
        // FusedAttentionImpl, so the result is BIT-IDENTICAL while the shared footprint is one block (unbounded SKV).
        int d0 = tid, d1 = tid + AttnGroupSize, d2 = tid + 2 * AttnGroupSize, d3 = tid + 3 * AttnGroupSize;
        float runningMax = -1e10f, runningSum = 0f;
        float wV0 = 0f, wV1 = 0f, wV2 = 0f, wV3 = 0f;

        for (int blockStart = 0; blockStart < SKV; blockStart += AttnKvBlock)
        {
            int curBK = Math.Min(AttnKvBlock, SKV - blockStart);
            for (int j = tid; j < curBK; j += AttnGroupSize)
            {
                int kv = blockStart + j;
                int kBase = kvHead * kvHeadStride + kv * kvTokenStride;
                float dot = 0f;
                for (int dd = 0; dd < D; dd++)
                    dot += qSh[dd] * K[kBase + dd];
                float score = dot * scale;
                int causalOk = 1 - (causal & ((qPos - kv) >> 31) & 1);
                int windowOk = ((qPos - window - kv) >> 31) & 1;
                int valid = causalOk & windowOk;
                scoresBlk[j] = score * valid + -1e30f * (1 - valid);
            }
            Group.Barrier();
            for (int j = 0; j < curBK; j++)
            {
                int vBase = kvHead * kvHeadStride + (blockStart + j) * kvTokenStride;
                float score = scoresBlk[j];
                float newMax = MathF.Max(runningMax, score);
                float correction = MathF.Exp(runningMax - newMax);
                float weight = MathF.Exp(score - newMax);
                runningSum = runningSum * correction + weight;
                if (d0 < D) wV0 = wV0 * correction + weight * V[vBase + d0];
                if (d1 < D) wV1 = wV1 * correction + weight * V[vBase + d1];
                if (d2 < D) wV2 = wV2 * correction + weight * V[vBase + d2];
                if (d3 < D) wV3 = wV3 * correction + weight * V[vBase + d3];
                runningMax = newMax;
            }
            Group.Barrier();
        }

        if (sinkCount > 0)
        {
            float sink = sinks[bh % sinkCount];
            float newMax = MathF.Max(runningMax, sink);
            float correction = MathF.Exp(runningMax - newMax);
            runningSum = runningSum * correction + MathF.Exp(sink - newMax);
            wV0 *= correction; wV1 *= correction; wV2 *= correction; wV3 *= correction;
            runningMax = newMax;
        }

        // Division (not reciprocal-multiply) to match FusedAttentionImpl bit-for-bit.
        if (d0 < D) output[oBase + d0] = wV0 / (runningSum + 1e-10f);
        if (d1 < D) output[oBase + d1] = wV1 / (runningSum + 1e-10f);
        if (d2 < D) output[oBase + d2] = wV2 / (runningSum + 1e-10f);
        if (d3 < D) output[oBase + d3] = wV3 / (runningSum + 1e-10f);
    }

    /// <summary>
    /// Strided + native-low-p <typeparamref name="T"/> K/V variant of <see cref="FusedAttentionTiledImpl"/>
    /// (the unbounded-SKV KV-cache prefill path). K/V read in type T and converted in-register
    /// (<c>PrecisionConvert.ConvertToSingle</c>) with the per-head element stride <c>p[10]</c>. With T=float and
    /// <c>p[10]=SKV*D</c> it is byte-identical to <see cref="FusedAttentionTiledImpl"/> and, in turn, to the
    /// per-element <see cref="FusedAttentionStridedImpl{T}"/> (the correctness anchor).
    /// </summary>
    private static void FusedAttentionTiledStridedImpl<T>(
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<T, Stride1D.Dense> K,
        ArrayView1D<T, Stride1D.Dense> V,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> sinks,
        ArrayView1D<int, Stride1D.Dense> p)
        where T : unmanaged, INumber<T>
    {
        int BH = p[0], SQ = p[1], SKV = p[2], D = p[3];
        float scale = Interop.IntAsFloat((uint)p[4]);
        int causal = p[5];
        int window = p[6];
        int kvOffset = p[7];
        int gqaGroup = p[8];
        int sinkCount = p[9];
        int kvStride = p[10];

        int g = Grid.IdxX;
        int tid = Group.IdxX;
        int bh = g / SQ;
        int sq = g % SQ;

        int kvHead = bh / gqaGroup;
        // Per-thread bases: heads-major [1,heads,seq,hd] vs seq-major [1,seq,heads,hd]. p[12]=1 → READ Q seq-major
        // (graph dropped the Q pre-transpose, step 2); p[11]=1 → WRITE output seq-major (dropped the post-attention
        // transpose, step 1). Independent flags (step 1 shipped with Q still heads-major / p[12]=0).
        int hBase = (bh * SQ + sq) * D, sBase = (sq * BH + bh) * D;
        int qBase = p[12] == 1 ? sBase : hBase;
        int oBase = p[11] == 1 ? sBase : hBase;
        int qPos = kvOffset + sq;
        // K/V layout (p[13]=1 seq-major store [kv,kvHeads,hd]: head offset D, token stride kvHeads*D; else head-major
        // strided: head offset kvStride, token D). Drops the K/V pre-attention transposes (step 3).
        int kvHeadStride = p[13] == 1 ? D : kvStride;
        int kvTokenStride = p[13] == 1 ? (BH / gqaGroup) * D : D;

        var qSh = SharedMemory.Allocate<float>(AttnHeadDimMax);
        var scoresBlk = SharedMemory.Allocate<float>(AttnKvBlock);

        for (int dd = tid; dd < D; dd += AttnGroupSize)
            qSh[dd] = Q[qBase + dd];
        Group.Barrier();

        // KV-tiled grouped attention (see FusedAttentionTiledImpl) — strided low-p K/V read in type T and
        // converted in-register; per-head base uses the explicit stride p[10]. Bit-identical to the contiguous
        // grouped kernel (and the per-element strided kernel) at T=float, p[10]=SKV*D.
        int d0 = tid, d1 = tid + AttnGroupSize, d2 = tid + 2 * AttnGroupSize, d3 = tid + 3 * AttnGroupSize;
        float runningMax = -1e10f, runningSum = 0f;
        float wV0 = 0f, wV1 = 0f, wV2 = 0f, wV3 = 0f;

        for (int blockStart = 0; blockStart < SKV; blockStart += AttnKvBlock)
        {
            int curBK = Math.Min(AttnKvBlock, SKV - blockStart);
            for (int j = tid; j < curBK; j += AttnGroupSize)
            {
                int kv = blockStart + j;
                int kBase = kvHead * kvHeadStride + kv * kvTokenStride;
                float dot = 0f;
                for (int dd = 0; dd < D; dd++)
                    dot += qSh[dd] * PrecisionConvert.ConvertToSingle(K[kBase + dd]);
                float score = dot * scale;
                int causalOk = 1 - (causal & ((qPos - kv) >> 31) & 1);
                int windowOk = ((qPos - window - kv) >> 31) & 1;
                int valid = causalOk & windowOk;
                scoresBlk[j] = score * valid + -1e30f * (1 - valid);
            }
            Group.Barrier();
            for (int j = 0; j < curBK; j++)
            {
                int vBase = kvHead * kvHeadStride + (blockStart + j) * kvTokenStride;
                float score = scoresBlk[j];
                float newMax = MathF.Max(runningMax, score);
                float correction = MathF.Exp(runningMax - newMax);
                float weight = MathF.Exp(score - newMax);
                runningSum = runningSum * correction + weight;
                if (d0 < D) wV0 = wV0 * correction + weight * PrecisionConvert.ConvertToSingle(V[vBase + d0]);
                if (d1 < D) wV1 = wV1 * correction + weight * PrecisionConvert.ConvertToSingle(V[vBase + d1]);
                if (d2 < D) wV2 = wV2 * correction + weight * PrecisionConvert.ConvertToSingle(V[vBase + d2]);
                if (d3 < D) wV3 = wV3 * correction + weight * PrecisionConvert.ConvertToSingle(V[vBase + d3]);
                runningMax = newMax;
            }
            Group.Barrier();
        }

        if (sinkCount > 0)
        {
            float sink = sinks[bh % sinkCount];
            float newMax = MathF.Max(runningMax, sink);
            float correction = MathF.Exp(runningMax - newMax);
            runningSum = runningSum * correction + MathF.Exp(sink - newMax);
            wV0 *= correction; wV1 *= correction; wV2 *= correction; wV3 *= correction;
            runningMax = newMax;
        }

        if (d0 < D) output[oBase + d0] = wV0 / (runningSum + 1e-10f);
        if (d1 < D) output[oBase + d1] = wV1 / (runningSum + 1e-10f);
        if (d2 < D) output[oBase + d2] = wV2 / (runningSum + 1e-10f);
        if (d3 < D) output[oBase + d3] = wV3 / (runningSum + 1e-10f);
    }

    public void Dispose()
    {
        foreach (var buf in _paramsRing) buf?.Dispose();
        Array.Clear(_paramsRing);
        _dummySinks?.Dispose();
        _dummySinks = null;
    }
}
