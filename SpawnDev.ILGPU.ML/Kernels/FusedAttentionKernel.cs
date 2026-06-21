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

    // Lazily allocate ring slot `_ringNext` and upload paramsData into it, returning the exact-length view.
    private ArrayView1D<int, Stride1D.Dense> RentParamsSlot(int[] paramsData)
    {
        var slot = _ringNext;
        _ringNext = (_ringNext + 1) % RingSize;
        var buf = _paramsRing[slot] ??= _accelerator.Allocate1D<int>(ParamSize);
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
    private const int AttnSharedSkvMax = 4096;  // scores[] fits CUDA's 48KB shared (16KB) with occupancy headroom
    private const int AttnHeadDimMax = 256;     // qShared[]; covers gemma4's 256-dim heads
    public static bool EnableGroupedAttention =
        Environment.GetEnvironmentVariable("GGUF_ATTN_GROUP") == "1";

    private Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _groupedKernel;
    private readonly Dictionary<Type, object> _groupedStridedKernels = new();

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
        ArrayView1D<float, Stride1D.Dense>? sinks = null, int sinkCount = 0)
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
        };

        var paramsView = RentParamsSlot(paramsData);

        _dummySinks ??= _accelerator.Allocate1D(new float[1]);
        var sinksView = sinks ?? _dummySinks.View;

        // Grouped-per-query path (the dot computed ONCE; bit-identical, opt-in, non-browser-GPU, SKV-gated).
        if (UseGrouped(seqKV, headDim))
        {
            _groupedKernel ??= _accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(FusedAttentionGroupedImpl);
            _groupedKernel(new KernelConfig(nHeads * seqQ, AttnGroupSize), Q, K, V, output, sinksView, paramsView);
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
    /// workgroup shared memory; WebGPU's workgroup reduction is ~75× slow on Tint/Dawn), and the per-query
    /// scores[]+qShared[] fit the fixed shared allocations. Otherwise the per-element kernel runs.
    /// </summary>
    private bool UseGrouped(int seqKV, int headDim) =>
        EnableGroupedAttention
        && _accelerator.AcceleratorType != AcceleratorType.WebGL
        && _accelerator.AcceleratorType != AcceleratorType.WebGPU
        && seqKV <= AttnSharedSkvMax
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
        ArrayView1D<float, Stride1D.Dense>? sinks = null, int sinkCount = 0)
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
        };

        var paramsView = RentParamsSlot(paramsData);

        _dummySinks ??= _accelerator.Allocate1D(new float[1]);
        var sinksView = sinks ?? _dummySinks.View;

        // Grouped-per-query path (bit-identical, opt-in, non-browser-GPU, SKV-gated) — this is the prefill
        // hotpath (the KV-cache strided bf16 read), so the win lands HERE as well as the contiguous Forward.
        if (UseGrouped(seqKV, headDim))
        {
            if (!_groupedStridedKernels.TryGetValue(typeof(T), out var gk))
                _groupedStridedKernels[typeof(T)] = gk = _accelerator.LoadStreamKernel<
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                    ArrayView1D<T, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(
                    FusedAttentionGroupedStridedImpl<T>);
            ((Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                ArrayView1D<T, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>)gk)(
                new KernelConfig(nHeads * seqQ, AttnGroupSize), Q, K, V, output, sinksView, paramsView);
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

        // Decompose index: [bh, sq, d]
        int d = idx % D;
        int sq = (idx / D) % SQ;
        int bh = idx / (SQ * D);

        if (bh >= BH) return;

        int kvHead = bh / gqaGroup;
        int qBase = (bh * SQ + sq) * D;
        int qPos = kvOffset + sq;

        float runningMax = -1e10f;
        float runningSum = 0f;
        float weightedV = 0f;

        for (int kv = 0; kv < SKV; kv++)
        {
            int kBase = (kvHead * SKV + kv) * D;
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

        int d = idx % D;
        int sq = (idx / D) % SQ;
        int bh = idx / (SQ * D);
        if (bh >= BH) return;

        int kvHead = bh / gqaGroup;
        int qBase = (bh * SQ + sq) * D;
        int qPos = kvOffset + sq;

        float runningMax = -1e10f;
        float runningSum = 0f;
        float weightedV = 0f;

        for (int kv = 0; kv < SKV; kv++)
        {
            int kBase = kvHead * kvStride + kv * D;
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

    /// <summary>
    /// Grouped-per-query fused attention (contiguous f32 K/V). One thread GROUP per (bh, sq):
    /// phase 1 computes each Q·K score EXACTLY ONCE into shared <c>scores[]</c> (cooperatively, kills the
    /// per-element kernel's D-fold redundant dot); phase 2 has each thread own a slice of the D output dims
    /// and replay the IDENTICAL online-softmax recurrence of <see cref="FusedAttentionImpl"/> per dim,
    /// reading the shared score instead of recomputing the dot — so the result is BIT-IDENTICAL.
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

        int g = Grid.IdxX;        // one group per (bh, sq)
        int tid = Group.IdxX;     // 0..AttnGroupSize-1
        int bh = g / SQ;
        int sq = g % SQ;

        int kvHead = bh / gqaGroup;
        int qBase = (bh * SQ + sq) * D;
        int qPos = kvOffset + sq;

        var qSh = SharedMemory.Allocate<float>(AttnHeadDimMax);
        var scores = SharedMemory.Allocate<float>(AttnSharedSkvMax);

        // Load this query's Q row into shared once (an exact copy → the dot stays bit-identical).
        for (int dd = tid; dd < D; dd += AttnGroupSize)
            qSh[dd] = Q[qBase + dd];
        Group.Barrier();

        // Phase 1: each kv score computed ONCE. Masking is the SAME branch-free sign-bit arithmetic as the
        // per-element kernel, so the stored value (incl. the -1e30 sentinel for masked positions) matches.
        for (int kv = tid; kv < SKV; kv += AttnGroupSize)
        {
            int kBase = (kvHead * SKV + kv) * D;
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

        // Phase 2: each thread owns output dims d = tid, tid+G, … < D and replays the per-element online
        // softmax over the resident scores — operation-for-operation identical to FusedAttentionImpl.
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
                weightedV = weightedV * correction + weight * V[(kvHead * SKV + kv) * D + d];
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

            output[qBase + d] = weightedV / (runningSum + 1e-10f);
        }
    }

    /// <summary>
    /// Strided + native-low-p <typeparamref name="T"/> K/V variant of <see cref="FusedAttentionGroupedImpl"/>
    /// (the KV-cache prefill path). Identical grouped structure; K/V read in type T and converted in-register
    /// (<c>PrecisionConvert.ConvertToSingle</c>) with the per-head element stride <c>p[10]</c>. With T=float and
    /// <c>p[10]=SKV*D</c> it is byte-identical to <see cref="FusedAttentionGroupedImpl"/> and, in turn, to the
    /// per-element <see cref="FusedAttentionStridedImpl{T}"/> (the correctness anchor).
    /// </summary>
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
        int qBase = (bh * SQ + sq) * D;
        int qPos = kvOffset + sq;

        var qSh = SharedMemory.Allocate<float>(AttnHeadDimMax);
        var scores = SharedMemory.Allocate<float>(AttnSharedSkvMax);

        for (int dd = tid; dd < D; dd += AttnGroupSize)
            qSh[dd] = Q[qBase + dd];
        Group.Barrier();

        for (int kv = tid; kv < SKV; kv += AttnGroupSize)
        {
            int kBase = kvHead * kvStride + kv * D;
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
                weightedV = weightedV * correction + weight * PrecisionConvert.ConvertToSingle(V[kvHead * kvStride + kv * D + d]);
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

            output[qBase + d] = weightedV / (runningSum + 1e-10f);
        }
    }

    public void Dispose()
    {
        foreach (var buf in _paramsRing) buf?.Dispose();
        Array.Clear(_paramsRing);
        _dummySinks?.Dispose();
        _dummySinks = null;
    }
}
