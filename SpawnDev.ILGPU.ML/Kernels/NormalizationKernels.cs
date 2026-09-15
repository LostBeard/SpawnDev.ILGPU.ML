using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU normalization kernels beyond LayerNorm.
/// BatchNorm (inference mode), GroupNorm, InstanceNorm, RMSNorm.
/// All use auto-grouped 1D dispatch.
/// </summary>
public class NormalizationKernels : IDisposable
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int, float>? _batchNormKernel;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, float>? _rmsNormStatsKernel;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int>? _rmsNormApplyKernel;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int>? _rmsNormApplyNoWeightKernel;

    // Single-pass (group-per-row) RMSNorm — loaded lazily on the first non-WebGL call. _dummyRmsWeight is the
    // 1-element placeholder for the weightless path (hasWeight=0, the kernel never reads it); _rmsFusedGroup
    // caches the chosen group size.
    private Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, float, int>? _rmsNormFusedKernel;
    private Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, float, int>? _addRmsNormFusedKernel;
    private MemoryBuffer1D<float, Stride1D.Dense>? _dummyRmsWeight;
    private int _rmsFusedGroup;
    // Upper bound on the single-pass RMSNorm group size — also the compile-time size of the kernel's
    // per-thread partial-sums shared array (RMSNormFusedImpl). The runtime group T is capped to this.
    private const int MaxRmsGroup = 256;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, float>? _instanceNormMeanVarKernel;

    // Cooperative (group-per-slice) InstanceNorm Pass 1 — the same treatment RMSNorm already has above, in the
    // sibling that never got it. Loaded lazily on the first non-WebGL call; _iNormCoopGroup caches the group size.
    private Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, float>? _instanceNormMeanVarCoopKernel;
    private Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int>? _instanceNormPartialStatsCoopKernel;
    private Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int>? _instanceNormPartialSqDevCoopKernel;
    private int _iNormCoopGroup;
    // Upper bound on the cooperative Pass-1 group size — also the compile-time size of the kernel's per-thread
    // partial-sums shared array (InstanceNormMeanVarCoopImpl). The runtime group T is capped to this.
    private const int MaxINormGroup = 256;
    // Below this many spatial elements per slice the group would spend more time at its two barriers than in the
    // loop, so the serial kernel stays. Both sides are covered by the committed tests: InstanceNorm_MatchesCpu
    // (spatial=64) takes the serial path, InstanceNorm_StyleTransferDims/StyleMosaicShape (spatial=50176) the
    // cooperative one — so neither branch is a gate that never fires.
    private const int MinCoopSpatialPerThread = 4;

    // A/B ESCAPE HATCH: ILGPU_ML_INORM_COOP=0 forces the old one-thread-per-slice Pass 1 back on, so the
    // cooperative path can be priced against it in one run (DemoConsole INORMBENCH) instead of across two
    // builds. Same opt-out convention as FusedDequantMatMul's GGUF_GEMV_V2. Read ONCE into a static — a
    // getenv on a per-call kernel path is itself a cost, and this must not move the number it measures.
    // ⚠️ This switch gates the DISPATCH, i.e. the thing it names. Prove that before trusting any A/B from it:
    // with COOP=0 the StyleMosaic row must move, and if it does not, the switch is not reaching the code.
    private static readonly bool CoopInstanceNormEnabled =
        Environment.GetEnvironmentVariable("ILGPU_ML_INORM_COOP") != "0";

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int>? _instanceNormApplyKernel;

    // In-place apply: ONE feature buffer (data, read+write) instead of separate input+output. A SINGLE
    // read_write binding, so WebGPU's "no buffer bound to two storage slots" rule is satisfied (unlike calling
    // the two-param apply with input==output). Pass-2 reads data[idx] then writes the same [idx] AFTER pass-1
    // computed the per-slice stats → correct in-place. The executor uses this on a single-consumer input to
    // drop the 256 MiB VAE GroupNorm output buffer.
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int>? _instanceNormApplyInPlaceKernel;

    // Per-slice partial sum + sumSq (exact tiled decode global-stat combine).
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>? _instanceNormPartialStatsKernel;

    // Per-slice partial Σ(x-mean)² given an external mean (stable two-pass variance for tiled GroupNorm).
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>? _instanceNormPartialSqDevKernel;

    public NormalizationKernels(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// BatchNorm inference: output = scale * (input - mean) / sqrt(var + eps) + bias.
    /// One thread per element. NCHW layout.
    /// params: [N, C, spatial]
    /// </summary>
    private static void BatchNormImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> scale,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> mean,
        ArrayView1D<float, Stride1D.Dense> variance,
        int N, int C, int spatial, float eps)
    {
        // Determine which channel this element belongs to
        int c = (idx / spatial) % C;

        float x = input[idx];
        float invStd = 1f / MathF.Sqrt(variance[c] + eps);
        output[idx] = scale[c] * (x - mean[c]) * invStd + bias[c];
    }

    /// <summary>
    /// RMSNorm Pass 1: compute invRms per row. One thread per row.
    /// Writes exactly 1 value per thread (invRms[row]) — TF compatible.
    /// </summary>
    private static void RMSNormStatsImpl(Index1D row,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> invRms,
        int C, float epsilon)
    {
        int offset = row * C;
        double sumSq = 0.0;
        for (int i = 0; i < C; i++)
        {
            double v = (double)input[offset + i];
            sumSq += v * v;
        }
        invRms[row] = 1f / MathF.Sqrt((float)(sumSq / C) + epsilon);
    }

    /// <summary>
    /// RMSNorm Pass 2: apply normalization per element. One thread per element.
    /// Writes exactly 1 value per thread — TF compatible.
    /// </summary>
    private static void RMSNormApplyImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> invRms,
        int C)
    {
        int row = idx / C;
        int col = idx % C;
        output[idx] = input[idx] * invRms[row] * weight[col];
    }

    /// <summary>
    /// RMSNorm Pass 2, WEIGHTLESS: apply normalization per element with unit gain (no learned scale).
    /// gemma4's V-norm is a plain <c>ggml_rms_norm</c> with no weight (output = input * invRms).
    /// </summary>
    private static void RMSNormApplyNoWeightImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> invRms,
        int C)
    {
        int row = idx / C;
        output[idx] = input[idx] * invRms[row];
    }

    /// <summary>
    /// Single-pass RMSNorm — one GROUP per row. The WHOLE group cooperatively computes the row's
    /// sum-of-squares (each thread reduces a strided slice in f64 in-register, then thread 0 combines the T
    /// partials), then the whole group applies the normalization. Fuses the two-pass stats + apply into ONE
    /// dispatch — no second dispatch, no invRms global round-trip, no scratch buffer. Needs group shared memory
    /// + a barrier, so it is gated to backends with a group (WebGL's TF path keeps the two-pass).
    ///
    /// PERF (2026-06-22): the prior version had THREAD 0 ALONE sum all C elements in f64 while T-1 threads
    /// idled at the barrier — a single core doing C f64 mul-adds at the 4070's 1/64 f64 rate ≈ 140 µs/call =
    /// 22.6% of decode (qwen 7B, the #2 op after MatMul). Parallelizing the reduction across the group
    /// cuts it ~6x. The f64 accumulation per thread is kept (precision); partials cross threads in f32 shared
    /// (sum of ~C/T squares each → negligible vs the 2e-4 RMSNorm test tolerance). NOT byte-identical to the
    /// serial two-pass anymore (the tree-order f64 sum differs ~1e-7), which the CPU-reference tests allow.
    /// <paramref name="hasWeight"/> 0 = weightless unit gain (gemma4's V-norm); else multiply by weight[col].
    /// </summary>
    private static void RMSNormFusedImpl(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> weight,
        int C, float epsilon, int hasWeight)
    {
        int row = Grid.IdxX;     // one group per row
        int tid = Group.IdxX;
        int T = Group.DimX;
        int offset = row * C;

        var part = SharedMemory.Allocate<float>(MaxRmsGroup);   // per-thread partial sum-of-squares (T <= MaxRmsGroup)
        var inv = SharedMemory.Allocate<float>(1);

        // Each thread reduces its strided slice in f64 in-register (T-way parallel over C), then publishes a
        // partial. Thread 0 combines the T partials in f64.
        double local = 0.0;
        for (int i = tid; i < C; i += T) { double v = (double)input[offset + i]; local += v * v; }
        part[tid] = (float)local;
        Group.Barrier();

        if (tid == 0)
        {
            double sumSq = 0.0;
            for (int t = 0; t < T; t++) sumSq += part[t];
            inv[0] = 1f / MathF.Sqrt((float)(sumSq / C) + epsilon);
        }
        Group.Barrier();

        float invRms = inv[0];
        for (int i = tid; i < C; i += T)
            output[offset + i] = input[offset + i] * invRms * (hasWeight != 0 ? weight[i] : 1f);
    }

    /// <summary>Fused (residual-Add + RMSNorm) in ONE cooperative pass: reads x = a + b per element, writes the
    /// residual sum to <paramref name="residualOut"/> (the residual stream the NEXT add consumes) AND the normalized
    /// result to <paramref name="normedOut"/> — replacing a separate Add kernel + RMSNorm. Same single-pass
    /// reduction shape as <see cref="RMSNormFusedImpl"/> (sum of squares of x, f64 partials → tree combine), so it
    /// matches the Add→RMSNorm chain to the RMSNorm test tolerance. Non-WebGL (shared mem + barrier); WebGL falls
    /// back to ElementWise.Add + the two-pass norm (same op, two kernels).</summary>
    private static void AddRMSNormFusedImpl(
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> residualOut,
        ArrayView1D<float, Stride1D.Dense> normedOut,
        ArrayView1D<float, Stride1D.Dense> weight,
        int C, float epsilon, int hasWeight)
    {
        int row = Grid.IdxX;
        int tid = Group.IdxX;
        int T = Group.DimX;
        int offset = row * C;

        var part = SharedMemory.Allocate<float>(MaxRmsGroup);
        var inv = SharedMemory.Allocate<float>(1);

        double local = 0.0;
        for (int i = tid; i < C; i += T)
        {
            float x = a[offset + i] + b[offset + i];
            residualOut[offset + i] = x;        // the residual stream (read by the next residual Add)
            double v = (double)x; local += v * v;
        }
        part[tid] = (float)local;
        Group.Barrier();

        if (tid == 0)
        {
            double sumSq = 0.0;
            for (int t = 0; t < T; t++) sumSq += part[t];
            inv[0] = 1f / MathF.Sqrt((float)(sumSq / C) + epsilon);
        }
        Group.Barrier();

        float invRms = inv[0];
        for (int i = tid; i < C; i += T)
            normedOut[offset + i] = residualOut[offset + i] * invRms * (hasWeight != 0 ? weight[i] : 1f);
    }

    /// <summary>
    /// InstanceNorm Pass 1: compute mean and invStd per (N,C) slice.
    /// One thread per slice. Each thread loops over spatial once for mean, once for variance.
    /// Output: means[N*C] and invStds[N*C].
    /// </summary>
    private static void InstanceNormMeanVarImpl(Index1D sliceIdx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> means,
        ArrayView1D<float, Stride1D.Dense> invStds,
        int spatial, float eps)
    {
        int ncBase = sliceIdx * spatial;
        // FLOAT accumulate (NOT double): f64 in this kernel triggers the WebGPU/WebGL f64-emulation path which
        // produces NaN/Inf here (PMT InstanceNorm tests went red on both browser backends). The tiled VAE decode
        // matches this by ALSO using float partial-stat kernels (same order at grid=1) — see InstanceNormPartialStats.
        float sum = 0f;
        for (int i = 0; i < spatial; i++)
            sum += input[ncBase + i];
        float mean = sum / spatial;
        means[sliceIdx] = mean;

        float varSum = 0f;
        for (int i = 0; i < spatial; i++)
        {
            float d = input[ncBase + i] - mean;
            varSum += d * d;
        }
        invStds[sliceIdx] = 1f / MathF.Sqrt(varSum / spatial + eps);
    }

    /// <summary>
    /// InstanceNorm Pass 1, COOPERATIVE: one GROUP per (N,C) slice instead of one THREAD, with the T threads
    /// splitting the spatial loop between them and combining through shared memory. Same two-pass math and same
    /// float accumulation as <see cref="InstanceNormMeanVarImpl"/>, so the results agree to the CPU-reference
    /// tolerance; the summation ORDER differs (strided partials, not one serial sweep), which if anything is the
    /// more accurate of the two — the partials are ~spatial/T terms long instead of spatial.
    ///
    /// 🔴 WHY THIS EXISTS. <see cref="InstanceNormMeanVarImpl"/> launches exactly N*C threads, each looping
    /// <paramref name="spatial"/> TWICE. For Kokoro's adaptive-norm blocks that is small; for a feature map it is
    /// not: at N=1, C=32, spatial=50176 it is 32 threads doing 100,352 serial strided loads each, so the dispatch
    /// duration grows with spatial and is bounded by nothing. On 2026-09-15 that dispatch ran past the Windows TDR
    /// budget on WebGPU and the OS removed the device — DXGI_ERROR_DEVICE_HUNG in
    /// InstanceNorm_StyleMosaicShape_MatchesCpu, then 345 downstream DEVICE_REMOVED failures as every later WebGPU
    /// test failed to create an accelerator on the dead device. A scoped re-run of the same test PASSED, which is
    /// what a dispatch sitting ON the TDR boundary looks like; a pass/fail re-run can never settle it, only the
    /// measured duration can (DemoConsole INORMBENCH prices it).
    ///
    /// Filling the group removes the exposure by construction rather than by widening a timeout.
    /// </summary>
    private static void InstanceNormMeanVarCoopImpl(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> means,
        ArrayView1D<float, Stride1D.Dense> invStds,
        int spatial, float eps)
    {
        int slice = Grid.IdxX;      // one group per (N,C) slice
        int tid = Group.IdxX;
        int T = Group.DimX;
        int ncBase = slice * spatial;

        var part = SharedMemory.Allocate<float>(MaxINormGroup);   // per-thread partials (T <= MaxINormGroup)
        var bcast = SharedMemory.Allocate<float>(1);              // the slice mean, published to the group

        // Phase A — mean. FLOAT accumulate (NOT double): f64 here takes the WebGPU/WebGL f64-emulation path,
        // which produces NaN/Inf in this kernel (the constraint InstanceNormMeanVarImpl already documents).
        float local = 0f;
        for (int i = tid; i < spatial; i += T) local += input[ncBase + i];
        part[tid] = local;
        Group.Barrier();

        if (tid == 0)
        {
            float sum = 0f;
            for (int t = 0; t < T; t++) sum += part[t];
            bcast[0] = sum / spatial;
        }
        Group.Barrier();
        float mean = bcast[0];

        // Phase B — Σ(x-mean)² around that mean: the SAME numerically-stable two-pass form the serial kernel
        // uses, not Σx²−(Σx)² (see InstanceNormPartialSqDevImpl on why the one-pass form cancels catastrophically
        // on conv-biased feature maps: large mean, small variance).
        //
        // ⚠️ part[] is REUSED here, and that is safe without another barrier: the only reader of the phase-A
        // partials is thread 0, between the first and second barrier. Every thread reaches the write below only
        // after passing the SECOND barrier, which thread 0 reaches only after finishing those reads.
        float localVar = 0f;
        for (int i = tid; i < spatial; i += T)
        {
            float d = input[ncBase + i] - mean;
            localVar += d * d;
        }
        part[tid] = localVar;
        Group.Barrier();

        if (tid == 0)
        {
            float varSum = 0f;
            for (int t = 0; t < T; t++) varSum += part[t];
            means[slice] = mean;
            invStds[slice] = 1f / MathF.Sqrt(varSum / spatial + eps);
        }
    }

    /// <summary>
    /// InstanceNorm Pass 2: apply normalization using pre-computed mean/invStd.
    /// One thread per element. No loops — O(1) per thread.
    /// </summary>
    private static void InstanceNormApplyImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> scale,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> means,
        ArrayView1D<float, Stride1D.Dense> invStds,
        int N, int C, int spatial)
    {
        int c = (idx / spatial) % C;
        int sliceIdx = idx / spatial;
        output[idx] = scale[c] * (input[idx] - means[sliceIdx]) * invStds[sliceIdx] + bias[c];
    }

    /// <summary>InstanceNorm Pass 2, IN PLACE: one read_write buffer (<paramref name="data"/>). Identical math
    /// to <see cref="InstanceNormApplyImpl"/> but reads and writes the same element of one buffer — a single
    /// binding (WebGPU-legal). Each thread reads data[idx] then writes it; stats were already computed in pass 1.</summary>
    private static void InstanceNormApplyInPlaceImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> scale,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> means,
        ArrayView1D<float, Stride1D.Dense> invStds,
        int N, int C, int spatial)
    {
        int c = (idx / spatial) % C;
        int sliceIdx = idx / spatial;
        data[idx] = scale[c] * (data[idx] - means[sliceIdx]) * invStds[sliceIdx] + bias[c];
    }

    /// <summary>Partial InstanceNorm stats per (N,C) slice: sum and sum-of-squares over <paramref name="spatial"/>
    /// elements (one thread per slice). For exact tiled decode — each tile contributes its partial sum/sumSq/count
    /// over its NON-overlap core; the caller combines across tiles into global mean/var, then applies with
    /// <see cref="InstanceNormApplyWithStats"/>. Double-precision accumulate keeps the combine exact.</summary>
    private static void InstanceNormPartialStatsImpl(Index1D sliceIdx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> sums,
        ArrayView1D<float, Stride1D.Dense> sumSqs,
        int spatial)
    {
        int ncBase = sliceIdx * spatial;
        // FLOAT accumulate (NOT double): keeps the tiled GroupNorm browser-safe (f64 in-kernel NaNs on WebGPU/WebGL).
        // The host combines these float partials in double (BufferPool-side, not a kernel).
        //
        // ⚠️ THE "order-matched to the full decode's InstanceNorm, so at grid=1 the per-tile partial == the full
        // single-pass sum exactly" CLAIM THAT USED TO BE HERE IS NO LONGER TRUE, and nothing depended on it:
        // InstanceNorm() Pass 1 now reduces COOPERATIVELY on a long slice (InstanceNormMeanVarCoopImpl), a
        // different summation order. Nothing cross-checks the two - TiledStatSync_GlobalStatsMatchFullInstanceNorm
        // and the TiledGroupNorm tests both score against a CPU double reference at their own tolerance, and
        // TiledVaeOps only ever combines these partials with each other. Left as a note rather than deleted,
        // because a reader who remembers the old invariant needs to know it was retired deliberately.
        //
        // ⚠️ THIS KERNEL IS NOW THE FALLBACK, not the default: InstanceNormPartialStats dispatches
        // InstanceNormPartialStatsCoopImpl whenever CoopStatsApplies says so, and reaches this body only on
        // WebGL, a tiny group, or a slice too short to divide. It kept the one-thread-per-slice shape that cost
        // the WebGPU lane on 2026-09-15 - and on the LARGEST slices in the engine, since this is the tiled VAE
        // decode's stat pass. MEASURED (RTX 4070, [1,32,50176]): 2.01 ms serial -> 0.04 ms cooperative.
        float sum = 0f, sumSq = 0f;
        for (int i = 0; i < spatial; i++)
        {
            float v = input[ncBase + i];
            sum += v; sumSq += v * v;
        }
        sums[sliceIdx] = sum;
        sumSqs[sliceIdx] = sumSq;
    }

    /// <summary>Partial sum of squared deviations from an EXTERNALLY-supplied per-slice mean: Σ(x - means[slice])²
    /// over <paramref name="spatial"/> (one thread per slice, double accumulate). The numerically-stable second
    /// pass for the tiled GroupNorm — pairs with a global-mean first pass to avoid the catastrophic cancellation
    /// of Σx² − (Σx)² when a group has a large mean and small variance (conv-biased feature maps).</summary>
    private static void InstanceNormPartialSqDevImpl(Index1D sliceIdx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> sqDevs,
        ArrayView1D<float, Stride1D.Dense> means,
        int spatial)
    {
        int ncBase = sliceIdx * spatial;
        // FLOAT accumulate (browser-safe; see InstanceNormPartialStats). Stable two-pass Σ(x-mean)² from a known
        // global mean — avoids the Σx²−(Σx)² cancellation when a group has a large mean + small variance.
        float mean = means[sliceIdx], sumSq = 0f;
        for (int i = 0; i < spatial; i++)
        {
            float d = input[ncBase + i] - mean;
            sumSq += d * d;
        }
        sqDevs[sliceIdx] = sumSq;
    }

    // ── Public API ──

    /// <summary>
    /// BatchNorm inference mode. Input/output: [N, C, H, W] flat.
    /// scale, bias, mean, variance: [C] each.
    /// </summary>
    public void BatchNorm(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> scale,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> mean,
        ArrayView1D<float, Stride1D.Dense> variance,
        int N, int C, int spatial, float epsilon = 1e-5f)
    {
        EnsureLoaded();
        _batchNormKernel!(N * C * spatial, input, output, scale, bias, mean, variance, N, C, spatial, epsilon);
    }

    // Single-pass fused RMSNorm on any backend with a group (everything but WebGL's TF path); fuses the
    // two-pass stats + apply into one dispatch (no second dispatch, no invRms round-trip). Returns false to
    // fall through to the two-pass path. hasWeight 0 = weightless. The whole group reduces the sum-of-squares
    // cooperatively (see RMSNormFusedImpl); matches the CPU reference within the 2e-4 RMSNorm test tolerance.
    // The weight view is unused when hasWeight==0 (pass the dummy).
    private bool TryFusedRMSNorm(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output, ArrayView1D<float, Stride1D.Dense> weight,
        int rows, int C, float epsilon, int hasWeight)
    {
        if (rows <= 0 || _accelerator.AcceleratorType == AcceleratorType.WebGL) return false;
        int T = _rmsFusedGroup != 0 ? _rmsFusedGroup
            : (_rmsFusedGroup = Math.Min(MaxRmsGroup, (int)_accelerator.MaxNumThreadsPerGroup));
        if (T < 32) return false; // group too small to be worth it — keep the two-pass
        _rmsNormFusedKernel ??= _accelerator.LoadStreamKernel<
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, float, int>(RMSNormFusedImpl);
        _rmsNormFusedKernel(new KernelConfig(new Index1D(rows), new Index1D(T)),
            input, output, weight, C, epsilon, hasWeight);
        return true;
    }

    /// <summary>Fused residual-Add + RMSNorm in one cooperative pass: residualOut = a+b, normedOut = rmsnorm(a+b)·w.
    /// Returns false on WebGL / tiny groups (the caller does the Add + two-pass norm fallback). hasWeight 0 = unit gain.</summary>
    public bool AddRMSNormFused(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> residualOut, ArrayView1D<float, Stride1D.Dense> normedOut,
        ArrayView1D<float, Stride1D.Dense> weight, int rows, int C, float epsilon, int hasWeight)
    {
        EnsureLoaded();
        if (rows <= 0 || _accelerator.AcceleratorType == AcceleratorType.WebGL) return false;
        int T = _rmsFusedGroup != 0 ? _rmsFusedGroup
            : (_rmsFusedGroup = Math.Min(MaxRmsGroup, (int)_accelerator.MaxNumThreadsPerGroup));
        if (T < 32) return false;
        if (hasWeight == 0) { _dummyRmsWeight ??= _accelerator.Allocate1D(new float[1]); weight = _dummyRmsWeight.View; }
        _addRmsNormFusedKernel ??= _accelerator.LoadStreamKernel<
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, float, int>(AddRMSNormFusedImpl);
        _addRmsNormFusedKernel(new KernelConfig(new Index1D(rows), new Index1D(T)),
            a, b, residualOut, normedOut, weight, C, epsilon, hasWeight);
        return true;
    }

    /// <summary>
    /// RMSNorm: input [rows, C] → output [rows, C]. weight: [C]. Single-pass (one group/row) where a group is
    /// available; two-pass on WebGL (TF).
    /// </summary>
    public void RMSNorm(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> weight,
        int rows, int C, float epsilon = 1e-6f)
    {
        EnsureLoaded();
        if (TryFusedRMSNorm(input, output, weight, rows, C, epsilon, hasWeight: 1)) return;

        // WebGL two-pass. invRms scratch from the reusable ring (no per-call alloc / no _allTempBufs growth);
        // the ring depth keeps a slot out of reach of the still-pending Pass 2 of an earlier call (RentInvRms).
        var rmsInvRms = RentInvRms(rows);
        _rmsNormStatsKernel!(rows, input, rmsInvRms, C, epsilon);
        _rmsNormApplyKernel!(rows * C, input, output, weight, rmsInvRms, C);
    }

    /// <summary>
    /// Weightless RMSNorm: input [rows, C] → output [rows, C], unit gain (no learned scale).
    /// gemma4 applies this to V (a plain <c>ggml_rms_norm</c> with no weight) before attention.
    /// </summary>
    public void RMSNorm(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int rows, int C, float epsilon = 1e-6f)
    {
        EnsureLoaded();
        _dummyRmsWeight ??= _accelerator.Allocate1D(new float[1]);
        if (TryFusedRMSNorm(input, output, _dummyRmsWeight.View, rows, C, epsilon, hasWeight: 0)) return;

        var rmsInvRms = RentInvRms(rows);
        _rmsNormStatsKernel!(rows, input, rmsInvRms, C, epsilon);
        _rmsNormApplyNoWeightKernel!(rows * C, input, output, rmsInvRms, C);
    }

    /// <summary>
    /// InstanceNorm: normalize each (N, C) slice over spatial dims.
    /// Input: [N, C, H, W]. scale, bias: [C].
    /// </summary>
    /// <summary>
    /// InstanceNorm: two-pass approach (O(N) instead of O(N²)).
    /// Pass 1: compute mean + invStd per (N,C) slice (N*C threads, each loops spatial).
    /// Pass 2: normalize each element (N*C*spatial threads, no loops).
    /// </summary>
    /// <summary>
    /// DIAGNOSTIC: when set, InstanceNorm Pass 1 GPU buffer pairs (means, invStds) for
    /// the first few calls are appended here. Caller is responsible for async readback
    /// (CopyToHostAsync) since this happens in a sync InstanceNorm path.
    /// Only captures the first <see cref="CaptureInstanceNormMaxCalls"/> calls.
    /// Off by default; opt-in for codegen bug investigation (e.g. WebGL Style transfer
    /// mean error 36-56 vs WebGPU pass).
    /// </summary>
    /// <summary>Runs InstanceNorm Pass 1 as one GROUP per slice (see <see cref="InstanceNormMeanVarCoopImpl"/>),
    /// returning false when the caller must fall through to the one-thread-per-slice serial kernel: WebGL (no
    /// shared memory / no group barrier in the TF path — the same exclusion <see cref="TryFusedRMSNorm"/> makes),
    /// a group too small to be worth the barriers, or a slice too short to divide.</summary>
    /// <summary>The ONE predicate all three cooperative per-slice stat paths share (mean/var, partial stats,
    /// partial sq-dev), so they cannot drift into disagreeing about when a group is worth using. Returns false -
    /// caller falls through to its one-thread-per-slice kernel - on WebGL (no shared memory / no group barrier in
    /// the TF path, the same exclusion <see cref="TryFusedRMSNorm"/> makes), a group too small to pay for its
    /// barriers, or a slice too short to divide.</summary>
    /// <summary>DIAGNOSTIC: which Pass-1 path the LAST per-slice stat dispatch took, and the group size it
    /// used. Exists because on 2026-09-15 a CUDA measurement of the cooperative path was allowed to stand in
    /// for the WebGPU behaviour, and WebGPU is the backend whose device was being lost - "it is 39x faster"
    /// was true and irrelevant. A path that is merely BELIEVED to be taken is not evidence.</summary>
    public static bool LastStatsPathWasCooperative { get; private set; }
    /// <summary>The group size the last cooperative stat dispatch used; 0 when it took the serial path.</summary>
    public static int LastStatsGroupSize { get; private set; }

    private bool CoopStatsApplies(int numSlices, int spatial, out int groupSize)
    {
        groupSize = 0;
        LastStatsPathWasCooperative = false;
        LastStatsGroupSize = 0;
        if (!CoopInstanceNormEnabled) return false;
        if (numSlices <= 0 || _accelerator.AcceleratorType == AcceleratorType.WebGL) return false;
        int T = _iNormCoopGroup != 0 ? _iNormCoopGroup
            : (_iNormCoopGroup = Math.Min(MaxINormGroup, (int)_accelerator.MaxNumThreadsPerGroup));
        if (T < 32) return false;                                 // group too small — keep the serial kernel
        if (spatial < T * MinCoopSpatialPerThread) return false;  // slice too short to divide profitably
        groupSize = T;
        LastStatsPathWasCooperative = true;
        LastStatsGroupSize = T;
        return true;
    }

    private bool TryCoopInstanceNormStats(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> means, ArrayView1D<float, Stride1D.Dense> invStds,
        int numSlices, int spatial, float epsilon)
    {
        if (!CoopStatsApplies(numSlices, spatial, out int T)) return false;
        _instanceNormMeanVarCoopKernel ??= _accelerator.LoadStreamKernel<
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, float>(InstanceNormMeanVarCoopImpl);
        _instanceNormMeanVarCoopKernel(new KernelConfig(new Index1D(numSlices), new Index1D(T)),
            input, means, invStds, spatial, epsilon);
        return true;
    }

    public static List<(int callIdx, int N, int C, int spatial, MemoryBuffer1D<float, Stride1D.Dense> means, MemoryBuffer1D<float, Stride1D.Dense> invStds)>? CapturedInstanceNormPass1Outputs { get; set; }
    public static int CaptureInstanceNormMaxCalls { get; set; } = 4;
    private static int _instanceNormCallIdx;

    public void InstanceNorm(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> scale,
        ArrayView1D<float, Stride1D.Dense> bias,
        int N, int C, int spatial, float epsilon = 1e-5f)
    {
        EnsureLoaded();
        int numSlices = N * C;

        // Per-call temp buffers — eliminates the shared-state race under async dispatch
        // on Wasm where Pass 1 of a subsequent call would overwrite means/invStds before
        // the previous call's Pass 2 had finished reading them. Buffers stay alive in
        // _allTempBufs until Dispose() (typical InferenceSession lifetime).
        var (inMeans, inInvStds) = GetStatsScratch(numSlices);

        // Pass 1: compute mean + invStd per slice
        // ⚠️ The CALLER's epsilon, not a constant. ONNX InstanceNormalization declares `epsilon`
        // and this ignored it, so a model asking for anything but the 1e-5 default was quietly
        // computed wrong. The default here keeps every existing caller on the number it already had.
        if (!TryCoopInstanceNormStats(input, inMeans.View, inInvStds.View, numSlices, spatial, epsilon))
            _instanceNormMeanVarKernel!(numSlices, input, inMeans.View, inInvStds.View, spatial, epsilon);

        // DIAGNOSTIC capture (opt-in): record buffer refs so caller can async-read.
        // The temp buffers are held alive by _allTempBufs until Dispose, so it's
        // safe for the caller to read them after the inference completes.
        // Capture-cap is by list size so caller can reset between tests by setting
        // CapturedInstanceNormPass1Outputs = new().
        var capList = CapturedInstanceNormPass1Outputs;
        if (capList != null)
        {
            lock (capList)
            {
                if (capList.Count < CaptureInstanceNormMaxCalls)
                    capList.Add((capList.Count, N, C, spatial, inMeans, inInvStds));
            }
        }

        // Pass 2: apply normalization
        _instanceNormApplyKernel!(N * C * spatial, input, output, scale, bias, inMeans.View, inInvStds.View, N, C, spatial);
    }

    /// <summary>InstanceNorm IN PLACE: normalize <paramref name="data"/> over each (N,C) slice, writing back into
    /// the SAME buffer (no separate output). Saves the output buffer (a 256 MiB feature map in the SD VAE). Pass-1
    /// reads data for the per-slice mean/invStd; pass-2 reads+writes data in place via a single read_write binding
    /// (WebGPU-legal). Numerically identical to <see cref="InstanceNorm"/> with output==input.</summary>
    public void InstanceNormInPlace(ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> scale,
        ArrayView1D<float, Stride1D.Dense> bias,
        int N, int C, int spatial, float epsilon = 1e-5f)
    {
        EnsureLoaded();
        int numSlices = N * C;
        var (inMeans, inInvStds) = GetStatsScratch(numSlices);
        if (!TryCoopInstanceNormStats(data, inMeans.View, inInvStds.View, numSlices, spatial, epsilon))
            _instanceNormMeanVarKernel!(numSlices, data, inMeans.View, inInvStds.View, spatial, epsilon);
        _instanceNormApplyInPlaceKernel!(N * C * spatial, data, scale, bias, inMeans.View, inInvStds.View, N, C, spatial);
    }

    /// <summary>Compute per-(N,C)-slice partial sum and sumSq over <paramref name="spatial"/> elements of
    /// <paramref name="input"/>. For exact tiled decode: combine these across tiles (× per-tile core counts) into
    /// global mean/invStd. <paramref name="sums"/>/<paramref name="sumSqs"/> are length N*C.</summary>
    public void InstanceNormPartialStats(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> sums, ArrayView1D<float, Stride1D.Dense> sumSqs,
        int N, int C, int spatial)
    {
        EnsureLoaded();
        if (CoopStatsApplies(N * C, spatial, out int T))
        {
            _instanceNormPartialStatsCoopKernel ??= _accelerator.LoadStreamKernel<
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, int>(InstanceNormPartialStatsCoopImpl);
            _instanceNormPartialStatsCoopKernel(new KernelConfig(new Index1D(N * C), new Index1D(T)),
                input, sums, sumSqs, spatial);
            return;
        }
        _instanceNormPartialStatsKernel!(N * C, input, sums, sumSqs, spatial);
    }

    /// <summary>Cooperative <see cref="InstanceNormPartialStatsImpl"/>: one GROUP per slice. Same float
    /// accumulation, same two outputs; the T threads split the spatial sweep and combine through shared memory.
    /// This is the tiled VAE decode's stat pass, and its slices are the LARGEST in the engine (a 256 MiB feature
    /// map), so it carried the same unbounded-dispatch exposure that cost the WebGPU lane on 2026-09-15.</summary>
    private static void InstanceNormPartialStatsCoopImpl(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> sums,
        ArrayView1D<float, Stride1D.Dense> sumSqs,
        int spatial)
    {
        int slice = Grid.IdxX;
        int tid = Group.IdxX;
        int T = Group.DimX;
        int ncBase = slice * spatial;

        var partSum = SharedMemory.Allocate<float>(MaxINormGroup);
        var partSq = SharedMemory.Allocate<float>(MaxINormGroup);

        float sum = 0f, sumSq = 0f;
        for (int i = tid; i < spatial; i += T)
        {
            float v = input[ncBase + i];
            sum += v; sumSq += v * v;
        }
        partSum[tid] = sum;
        partSq[tid] = sumSq;
        Group.Barrier();

        if (tid == 0)
        {
            float s = 0f, q = 0f;
            for (int t = 0; t < T; t++) { s += partSum[t]; q += partSq[t]; }
            sums[slice] = s;
            sumSqs[slice] = q;
        }
    }

    /// <summary>Cooperative <see cref="InstanceNormPartialSqDevImpl"/>: one GROUP per slice, same stable
    /// Σ(x-mean)² around an externally supplied mean.</summary>
    private static void InstanceNormPartialSqDevCoopImpl(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> sqDevs,
        ArrayView1D<float, Stride1D.Dense> means,
        int spatial)
    {
        int slice = Grid.IdxX;
        int tid = Group.IdxX;
        int T = Group.DimX;
        int ncBase = slice * spatial;
        float mean = means[slice];

        var part = SharedMemory.Allocate<float>(MaxINormGroup);

        float local = 0f;
        for (int i = tid; i < spatial; i += T)
        {
            float d = input[ncBase + i] - mean;
            local += d * d;
        }
        part[tid] = local;
        Group.Barrier();

        if (tid == 0)
        {
            float q = 0f;
            for (int t = 0; t < T; t++) q += part[t];
            sqDevs[slice] = q;
        }
    }

    /// <summary>Partial Σ(x-mean)² per slice given an external per-slice <paramref name="means"/> (length N*C).
    /// The stable second pass for the tiled GroupNorm variance combine. See <see cref="InstanceNormPartialSqDevImpl"/>.</summary>
    public void InstanceNormPartialSqDev(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> sqDevs, ArrayView1D<float, Stride1D.Dense> means,
        int N, int C, int spatial)
    {
        EnsureLoaded();
        if (CoopStatsApplies(N * C, spatial, out int T))
        {
            _instanceNormPartialSqDevCoopKernel ??= _accelerator.LoadStreamKernel<
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, int>(InstanceNormPartialSqDevCoopImpl);
            _instanceNormPartialSqDevCoopKernel(new KernelConfig(new Index1D(N * C), new Index1D(T)),
                input, sqDevs, means, spatial);
            return;
        }
        _instanceNormPartialSqDevKernel!(N * C, input, sqDevs, means, spatial);
    }

    /// <summary>Apply InstanceNorm IN PLACE using EXTERNALLY-provided per-slice means/invStds (skips the local
    /// stat pass). For exact tiled decode: each tile applies the GLOBAL stats so there are no per-tile brightness
    /// seams. <paramref name="means"/>/<paramref name="invStds"/> are length N*C; single read_write binding.</summary>
    public void InstanceNormApplyWithStats(ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> scale, ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> means, ArrayView1D<float, Stride1D.Dense> invStds,
        int N, int C, int spatial)
    {
        EnsureLoaded();
        _instanceNormApplyInPlaceKernel!(N * C * spatial, data, scale, bias, means, invStds, N, C, spatial);
    }

    // Per-call temp buffers for InstanceNorm and RMSNorm two-pass kernels.
    // Sharing across calls would race: Pass 1 of call N+1 overwrites mean/invStd/invRms
    // before Pass 2 of call N has finished reading. Buffers stay alive in this list
    // until Dispose() (typical InferenceSession lifetime).
    private readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _allTempBufs = new();
    private MemoryBuffer1D<float, Stride1D.Dense>? _capMeans, _capInvStds;

    // 🔴 RETIRE, DO NOT DISPOSE. A captured plan (WebGPU bind groups / a CUDA graph's baked pointers) binds
    // whichever scratch buffer was live when it was recorded, and reads it at every replay. Freeing it
    // because a LATER capture needed a bigger one destroys memory an earlier plan still uses - WebGPU
    // reports that at the next submit as "[Buffer ...] used in submit while destroyed", naming whichever
    // innocent caller synchronizes next. Same defect the CaptureParamArena's _retired list documents.
    private readonly List<IDisposable> _capRetired = new();

    /// <summary>Per-call InstanceNorm mean/invStd scratch. Normal mode: fresh buffers (Wasm
    /// async-safety - a reused pair races pending dispatches), held in _allTempBufs until Dispose.
    /// Capture mode (UseCaptureParamSlots): ONE reused pair sized during the warm passes - a
    /// per-call cuMemAlloc mid-capture is a native 0xC0000005 (SD-Turbo UNet GroupNorm under
    /// CudaGraphCapture, 2026-07-03); sequential stream order makes reuse safe there.</summary>
    private (MemoryBuffer1D<float, Stride1D.Dense> Means, MemoryBuffer1D<float, Stride1D.Dense> InvStds) GetStatsScratch(int numSlices)
    {
        if (Graph.GraphExecutor.UseCaptureParamSlots)
        {
            if (_capMeans == null || _capMeans.Length < numSlices)
            {
                if (Graph.GraphExecutor.SuppressDrains)
                    throw new InvalidOperationException(
                        $"InstanceNorm scratch would grow to {numSlices} mid-capture - warm passes must cover the largest shape first.");
                if (_capMeans != null) _capRetired.Add(_capMeans);
                if (_capInvStds != null) _capRetired.Add(_capInvStds);
                _capMeans = _accelerator.Allocate1D<float>(numSlices);
                _capInvStds = _accelerator.Allocate1D<float>(numSlices);
            }
            return (_capMeans, _capInvStds!);
        }
        // 🔴 THE SAME LEAK THE RMSNorm RING ALREADY FIXED, in the sibling that never got it. A per-call
        // Allocate1D appended to _allTempBufs is freed only at Dispose, so every InstanceNorm call cost
        // two accelerator buffers for the life of the session. That was tolerable when InstanceNorm was
        // rare; it is not now that GraphOptimizer.FuseInstanceNorm emits one per adaptive-norm block -
        // MEASURED on Kokoro-82M: 62 blocks, so ~124 buffers leaked PER UTTERANCE, growing without bound
        // across a conversation.
        //
        // The ring keeps the property the per-call alloc was protecting: a slot is reused only after
        // StatsRingSize calls, far past the two-dispatch lifetime of one call's mean/invStd pair, so the
        // Pass1/Pass2 race under async dispatch cannot reappear.
        //
        // ⚠️ EXCEPT when the diagnostic capture is armed. CapturedInstanceNormPass1Outputs hands these
        // buffers to the caller to read AFTER the forward, which a ring would overwrite. That path keeps
        // the per-call allocation deliberately - it is opt-in, bounded by CaptureInstanceNormMaxCalls,
        // and correctness of a diagnostic beats its memory.
        if (CapturedInstanceNormPass1Outputs != null)
        {
            var dmeans = _accelerator.Allocate1D<float>(numSlices);
            var dinvStds = _accelerator.Allocate1D<float>(numSlices);
            _allTempBufs.Add(dmeans);
            _allTempBufs.Add(dinvStds);
            return (dmeans, dinvStds);
        }

        var slot = _statsNext;
        _statsNext = (_statsNext + 1) % StatsRingSize;
        var m = _statsMeansRing[slot];
        if (m == null || m.Length < numSlices)
        {
            m?.Dispose();
            _statsMeansRing[slot] = m = _accelerator.Allocate1D<float>(numSlices);
        }
        var v = _statsInvStdsRing[slot];
        if (v == null || v.Length < numSlices)
        {
            v?.Dispose();
            _statsInvStdsRing[slot] = v = _accelerator.Allocate1D<float>(numSlices);
        }
        return (m, v);
    }

    // InstanceNorm/GroupNorm mean+invStd ring - see GetStatsScratch. Sized like the RMSNorm ring and for
    // the same reason; 64 comfortably exceeds the 62 adaptive-norm blocks one Kokoro forward runs, so a
    // slot is not revisited within a forward at all.
    private const int StatsRingSize = 64;
    private readonly MemoryBuffer1D<float, Stride1D.Dense>?[] _statsMeansRing
        = new MemoryBuffer1D<float, Stride1D.Dense>?[StatsRingSize];
    private readonly MemoryBuffer1D<float, Stride1D.Dense>?[] _statsInvStdsRing
        = new MemoryBuffer1D<float, Stride1D.Dense>?[StatsRingSize];
    private int _statsNext;

    // RMSNorm two-pass invRms ring: the invRms buffer (one float per row) is written by Pass 1 and read by the
    // immediately-following Pass 2 (apply). A FIXED ring of reusable buffers (each grown to the max rows it has
    // seen) replaces the per-call Allocate1D that was appended to _allTempBufs and never freed until Dispose —
    // for a 48-layer decode that was ~288 tiny buffers/token accumulating for the WHOLE generation. A slot is
    // reused only after InvRmsRingSize calls, far past the two-dispatch lifetime of any one call's invRms, so
    // the Pass1/Pass2 race the per-call alloc avoided cannot reappear.
    private const int InvRmsRingSize = 64;
    private readonly MemoryBuffer1D<float, Stride1D.Dense>?[] _invRmsRing = new MemoryBuffer1D<float, Stride1D.Dense>?[InvRmsRingSize];
    private int _invRmsNext;

    private ArrayView1D<float, Stride1D.Dense> RentInvRms(int rows)
    {
        var slot = _invRmsNext;
        _invRmsNext = (_invRmsNext + 1) % InvRmsRingSize;
        var buf = _invRmsRing[slot];
        if (buf == null || buf.Length < rows)
        {
            buf?.Dispose();
            _invRmsRing[slot] = buf = _accelerator.Allocate1D<float>(rows);
        }
        return buf.View.SubView(0, rows);
    }

    /// <summary>Free the per-call mean/invStd temp buffers (held alive across calls to avoid the two-pass race)
    /// and the RMSNorm invRms ring. Previously these leaked until the accelerator was torn down; now released
    /// with the kernel owner.</summary>
    public void Dispose()
    {
        foreach (var b in _allTempBufs) try { b.Dispose(); } catch { }
        _allTempBufs.Clear();
        for (var i = 0; i < StatsRingSize; i++)
        {
            try { _statsMeansRing[i]?.Dispose(); } catch { }
            try { _statsInvStdsRing[i]?.Dispose(); } catch { }
            _statsMeansRing[i] = null;
            _statsInvStdsRing[i] = null;
        }
        try { _capMeans?.Dispose(); } catch { }
        try { _capInvStds?.Dispose(); } catch { }
        _capMeans = null; _capInvStds = null;
        foreach (var b in _capRetired) { try { b.Dispose(); } catch { } }
        _capRetired.Clear();
        foreach (var b in _invRmsRing) try { b?.Dispose(); } catch { }
        try { _dummyRmsWeight?.Dispose(); } catch { }
        _dummyRmsWeight = null;
    }

    private void EnsureLoaded()
    {
        var a = _accelerator;
        _batchNormKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int, float>(BatchNormImpl);
        _instanceNormMeanVarKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, float>(InstanceNormMeanVarImpl);
        _instanceNormApplyKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int>(InstanceNormApplyImpl);
        _instanceNormApplyInPlaceKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int>(InstanceNormApplyInPlaceImpl);
        _instanceNormPartialStatsKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(InstanceNormPartialStatsImpl);
        _instanceNormPartialSqDevKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(InstanceNormPartialSqDevImpl);
        _rmsNormStatsKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, float>(RMSNormStatsImpl);
        _rmsNormApplyKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int>(RMSNormApplyImpl);
        _rmsNormApplyNoWeightKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int>(RMSNormApplyNoWeightImpl);
    }
}
