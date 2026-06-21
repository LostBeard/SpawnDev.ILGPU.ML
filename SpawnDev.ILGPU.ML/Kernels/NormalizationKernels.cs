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

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, float>? _instanceNormMeanVarKernel;

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
        // FLOAT accumulate (NOT double): keeps the tiled GroupNorm browser-safe (f64 in-kernel NaNs on WebGPU/WebGL)
        // AND order-matched to the full decode's float InstanceNorm (so at grid=1 the per-tile partial == the full
        // single-pass sum, exactly). The host combines these float partials in double (BufferPool-side, not a kernel).
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

    /// <summary>
    /// RMSNorm: input [rows, C] → output [rows, C]. weight: [C].
    /// Two-pass for WebGL TF compatibility.
    /// </summary>
    public void RMSNorm(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> weight,
        int rows, int C, float epsilon = 1e-6f)
    {
        EnsureLoaded();

        // invRms scratch from the reusable ring (no per-call alloc / no unbounded _allTempBufs growth). The ring
        // depth keeps a slot out of reach of the still-pending Pass 2 of an earlier call (see RentInvRms).
        var rmsInvRms = RentInvRms(rows);

        // Pass 1: compute invRms per row
        _rmsNormStatsKernel!(rows, input, rmsInvRms, C, epsilon);

        // Pass 2: apply normalization per element
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
    public static List<(int callIdx, int N, int C, int spatial, MemoryBuffer1D<float, Stride1D.Dense> means, MemoryBuffer1D<float, Stride1D.Dense> invStds)>? CapturedInstanceNormPass1Outputs { get; set; }
    public static int CaptureInstanceNormMaxCalls { get; set; } = 4;
    private static int _instanceNormCallIdx;

    public void InstanceNorm(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> scale,
        ArrayView1D<float, Stride1D.Dense> bias,
        int N, int C, int spatial)
    {
        EnsureLoaded();
        int numSlices = N * C;

        // Per-call temp buffers — eliminates the shared-state race under async dispatch
        // on Wasm where Pass 1 of a subsequent call would overwrite means/invStds before
        // the previous call's Pass 2 had finished reading them. Buffers stay alive in
        // _allTempBufs until Dispose() (typical InferenceSession lifetime).
        var inMeans = _accelerator.Allocate1D<float>(numSlices);
        var inInvStds = _accelerator.Allocate1D<float>(numSlices);
        _allTempBufs.Add(inMeans);
        _allTempBufs.Add(inInvStds);

        // Pass 1: compute mean + invStd per slice
        _instanceNormMeanVarKernel!(numSlices, input, inMeans.View, inInvStds.View, spatial, 1e-5f);

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
        int N, int C, int spatial)
    {
        EnsureLoaded();
        int numSlices = N * C;
        var inMeans = _accelerator.Allocate1D<float>(numSlices);
        var inInvStds = _accelerator.Allocate1D<float>(numSlices);
        _allTempBufs.Add(inMeans);
        _allTempBufs.Add(inInvStds);
        _instanceNormMeanVarKernel!(numSlices, data, inMeans.View, inInvStds.View, spatial, 1e-5f);
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
        _instanceNormPartialStatsKernel!(N * C, input, sums, sumSqs, spatial);
    }

    /// <summary>Partial Σ(x-mean)² per slice given an external per-slice <paramref name="means"/> (length N*C).
    /// The stable second pass for the tiled GroupNorm variance combine. See <see cref="InstanceNormPartialSqDevImpl"/>.</summary>
    public void InstanceNormPartialSqDev(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> sqDevs, ArrayView1D<float, Stride1D.Dense> means,
        int N, int C, int spatial)
    {
        EnsureLoaded();
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
        foreach (var b in _invRmsRing) try { b?.Dispose(); } catch { }
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
