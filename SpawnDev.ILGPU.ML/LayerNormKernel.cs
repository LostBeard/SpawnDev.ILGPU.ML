using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// Layer Normalization: normalize over the last dimension (C), apply learned gamma + beta.
/// Uses auto-grouped kernels (no shared memory) to avoid WGSL variable redeclaration bug
/// with multiple LoadStreamKernel calls on the same accelerator.
///
/// Two-pass approach (WebGL TF compatible — each thread writes at most one output element):
///   Pass 1: One thread per row — compute mean and invStd via Welford. Write to temp buffers.
///   Pass 2: One thread per element — apply normalization using pre-computed stats.
/// </summary>
public class LayerNormKernel : IDisposable
{
    private readonly Accelerator _accelerator;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // input [rows*C]
        ArrayView1D<float, Stride1D.Dense>,  // means [rows]
        ArrayView1D<float, Stride1D.Dense>,  // invStds [rows]
        int, float>?                          // C, epsilon
        _meanVarKernel;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // input [rows*C]
        ArrayView1D<float, Stride1D.Dense>,  // output [rows*C]
        ArrayView1D<float, Stride1D.Dense>,  // gamma [C]
        ArrayView1D<float, Stride1D.Dense>,  // beta [C]
        ArrayView1D<float, Stride1D.Dense>,  // means [rows]
        ArrayView1D<float, Stride1D.Dense>,  // invStds [rows]
        int>?                                 // C
        _applyKernel;

    private MemoryBuffer1D<float, Stride1D.Dense>? _means;
    private MemoryBuffer1D<float, Stride1D.Dense>? _invStds;

    // Single-pass (group-per-row) fused LayerNorm - loaded lazily on the first non-WebGL call. Mirrors the
    // proven RMSNormFusedImpl pattern (NormalizationKernels.cs): the two-pass path's Pass 1 had ONE THREAD
    // do a serial C-length f64 Welford per row - at DAv3-518 that is 1,370 threads on the whole card
    // (~1/8 occupancy) each running C=384 f64 steps, and on WebGPU every f64 op is Dekker-EMULATED.
    // The fused kernel gives the row a whole group and needs no f64-emulated Welford, no mean/invStd
    // global round-trip, and one dispatch instead of two.
    private Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, float>? _fusedKernel;
    private int _fusedGroup;
    // Upper bound on the fused group size - also the compile-time size of the kernel's per-thread
    // partial-sums shared array. The runtime group T is capped to this.
    private const int MaxLnGroup = 256;

    public LayerNormKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Single-pass LayerNorm - one GROUP per row, two barrier-separated cooperative reductions.
    /// Phase 1: strided per-thread f64 partial SUMS → thread 0 combines → mean. Phase 2: strided per-thread
    /// f64 partial sums of (x-mean)² → thread 0 combines → invStd. Subtracting the mean BEFORE squaring is
    /// Welford-grade stable (no E[x²]-E[x]² cancellation), so the only divergence from the serial two-pass
    /// Welford is the f32 partial exchange + tree order (~1e-7 relative - within the CPU-reference test
    /// tolerance, same accepted trade as RMSNormFusedImpl). The row is re-read for phase 2 and the apply,
    /// but it is L2/L1-resident by then. Shared partials array is reused across both reductions
    /// (barrier-separated). Gated to backends with a group (WebGL's TF path keeps the two-pass).
    /// </summary>
    private static void LayerNormFusedImpl(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> gamma,
        ArrayView1D<float, Stride1D.Dense> beta,
        int C, float epsilon)
    {
        int row = Grid.IdxX;   // one group per row
        int tid = Group.IdxX;
        int T = Group.DimX;
        int offset = row * C;

        var part = SharedMemory.Allocate<float>(MaxLnGroup);
        var stats = SharedMemory.Allocate<float>(2); // [0]=mean, [1]=invStd

        // KAHAN-compensated f32 accumulation instead of f64 (2026-07-03). On WebGPU every f64 op is
        // Dekker-EMULATED (~15-20 f32 ops); the f64 partial sums made this kernel ~6x its bandwidth
        // floor (attribution: 5.7ms/48 dispatches on the DAv3 frame, 0.119ms vs ~21us of traffic).
        // Kahan is ~4 f32 ops per add with near-f64 accuracy at these lengths (C=384 partials +
        // T<=256 combine; error ~2eps vs the oracle's 1e-4 tolerance), and it is the SAME C# on all
        // 6 backends. WGSL/PTX/.NET do not reassociate float ops, so the compensation term survives.
        float localSum = 0f, cSum = 0f;
        for (int i = tid; i < C; i += T)
        {
            float y = input[offset + i] - cSum;
            float s = localSum + y;
            cSum = (s - localSum) - y;
            localSum = s;
        }
        part[tid] = localSum;
        Group.Barrier();

        if (tid == 0)
        {
            float sum = 0f, c0 = 0f;
            for (int t = 0; t < T; t++)
            {
                float y = part[t] - c0;
                float s = sum + y;
                c0 = (s - sum) - y;
                sum = s;
            }
            stats[0] = sum / C;
        }
        Group.Barrier();

        float mean = stats[0];
        float localVar = 0f, cVar = 0f;
        for (int i = tid; i < C; i += T)
        {
            float d = input[offset + i] - mean;
            float y = d * d - cVar;
            float s = localVar + y;
            cVar = (s - localVar) - y;
            localVar = s;
        }
        part[tid] = localVar;
        Group.Barrier();

        if (tid == 0)
        {
            float m2 = 0f, c1 = 0f;
            for (int t = 0; t < T; t++)
            {
                float y = part[t] - c1;
                float s = m2 + y;
                c1 = (s - m2) - y;
                m2 = s;
            }
            stats[1] = 1f / MathF.Sqrt(m2 / C + epsilon);
        }
        Group.Barrier();

        float invStd = stats[1];
        for (int i = tid; i < C; i += T)
            output[offset + i] = gamma[i] * ((input[offset + i] - mean) * invStd) + beta[i];
    }

    // Fused single-pass dispatch - same gate shape as NormalizationKernels.TryFusedRMSNorm. Returns false to
    // fall through to the two-pass path (WebGL, tiny groups).
    private bool TryFusedLayerNorm(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> gamma,
        ArrayView1D<float, Stride1D.Dense> beta,
        int rows, int C, float epsilon)
    {
        if (rows <= 0 || _accelerator.AcceleratorType == AcceleratorType.WebGL) return false;
        int T = _fusedGroup != 0 ? _fusedGroup
            : (_fusedGroup = Math.Min(MaxLnGroup, (int)_accelerator.MaxNumThreadsPerGroup));
        if (T < 32) return false; // group too small to be worth it - keep the two-pass
        _fusedKernel ??= _accelerator.LoadStreamKernel<
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, float>(LayerNormFusedImpl);
        _fusedKernel(new KernelConfig(new Index1D(rows), new Index1D(T)),
            input, output, gamma, beta, C, epsilon);
        return true;
    }

    /// <summary>
    /// Pass 1: One thread per row. Compute mean and invStd via double-precision Welford.
    /// Writes exactly 2 values per thread (means[row], invStds[row]) — TF compatible.
    /// </summary>
    private static void LayerNormMeanVarImpl(
        Index1D row,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> means,
        ArrayView1D<float, Stride1D.Dense> invStds,
        int C, float epsilon)
    {
        int offset = row * C;

        // Double-precision Welford: numerically stable single-pass mean + variance.
        double mean = 0.0;
        double m2 = 0.0;
        for (int i = 0; i < C; i++)
        {
            double x = (double)input[offset + i];
            double delta = x - mean;
            mean += delta / (i + 1);
            double delta2 = x - mean;
            m2 += delta * delta2;
        }
        means[row] = (float)mean;
        invStds[row] = 1f / MathF.Sqrt((float)(m2 / C) + epsilon);
    }

    /// <summary>
    /// Pass 2: One thread per element. Apply normalization using pre-computed stats.
    /// Writes exactly 1 value per thread — TF compatible.
    /// </summary>
    private static void LayerNormApplyImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> gamma,
        ArrayView1D<float, Stride1D.Dense> beta,
        ArrayView1D<float, Stride1D.Dense> means,
        ArrayView1D<float, Stride1D.Dense> invStds,
        int C)
    {
        int row = idx / C;
        int col = idx % C;
        output[idx] = gamma[col] * ((input[idx] - means[row]) * invStds[row]) + beta[col];
    }

    public void Forward(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> gamma,
        ArrayView1D<float, Stride1D.Dense> beta,
        int rows, int C, float epsilon = 1e-6f)
    {
        var accelerator = _accelerator;

        // Single-pass group-per-row path (all backends with a group; WebGL falls through to the two-pass).
        if (TryFusedLayerNorm(input, output, gamma, beta, rows, C, epsilon)) return;

        EnsureLoaded(accelerator);

        // Allocate/resize persistent temp buffers
        if (_means == null || _means.Length < rows)
        {
            _means?.Dispose();
            _means = accelerator.Allocate1D<float>(rows);
        }
        if (_invStds == null || _invStds.Length < rows)
        {
            _invStds?.Dispose();
            _invStds = accelerator.Allocate1D<float>(rows);
        }

        // Pass 1: compute mean + invStd per row
        _meanVarKernel!(rows, input, _means.View, _invStds.View, C, epsilon);

        // Pass 2: apply normalization per element
        _applyKernel!(rows * C, input, output, gamma, beta, _means.View, _invStds.View, C);
    }

    private void EnsureLoaded(Accelerator accelerator)
    {
        _meanVarKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, float>(LayerNormMeanVarImpl);
        _applyKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int>(LayerNormApplyImpl);
    }

    public async Task DiagnosticAsync()
    {
        
        var accelerator = _accelerator;
        EnsureLoaded(accelerator);

        var input = new float[] { 1f, 2f, 3f, 4f };
        var gamma = new float[] { 1f, 1f, 1f, 1f };
        var beta = new float[] { 0f, 0f, 0f, 0f };

        using var inputBuf = accelerator.Allocate1D(input);
        using var outputBuf = accelerator.Allocate1D<float>(4);
        using var gammaBuf = accelerator.Allocate1D(gamma);
        using var betaBuf = accelerator.Allocate1D(beta);

        Forward(inputBuf.View, outputBuf.View, gammaBuf.View, betaBuf.View, 1, 4);
        await accelerator.SynchronizeAsync();

        var gpuOut = await outputBuf.CopyToHostAsync<float>(0, 4);
        if (InferenceSession.VerboseLogging) Console.WriteLine($"[LayerNorm] Diagnostic [1,2,3,4]: [{string.Join(", ", gpuOut.Select(v => v.ToString("F4")))}] (expect [-1.3416, -0.4472, 0.4472, 1.3416])");
    }

    public async Task<(float maxError, float avgError)> ValidateAsync(int rows = 1369, int C = 384)
    {
        
        var accelerator = _accelerator;
        EnsureLoaded(accelerator);

        var rng = new Random(42);
        var inputData = new float[rows * C];
        var gammaData = new float[C];
        var betaData = new float[C];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < C; i++) gammaData[i] = (float)(rng.NextDouble() * 0.5 + 0.75);
        for (int i = 0; i < C; i++) betaData[i] = (float)(rng.NextDouble() * 0.2 - 0.1);

        var cpuOut = new float[rows * C];
        for (int r = 0; r < rows; r++)
        {
            float mean = 0;
            for (int i = 0; i < C; i++) mean += inputData[r * C + i];
            mean /= C;
            float var_ = 0;
            for (int i = 0; i < C; i++) { float d = inputData[r * C + i] - mean; var_ += d * d; }
            float invStd = 1f / MathF.Sqrt(var_ / C + 1e-6f);
            for (int i = 0; i < C; i++)
                cpuOut[r * C + i] = gammaData[i] * ((inputData[r * C + i] - mean) * invStd) + betaData[i];
        }

        using var inputBuf = accelerator.Allocate1D(inputData);
        using var outputBuf = accelerator.Allocate1D<float>(rows * C);
        using var gammaBuf = accelerator.Allocate1D(gammaData);
        using var betaBuf = accelerator.Allocate1D(betaData);

        Forward(inputBuf.View, outputBuf.View, gammaBuf.View, betaBuf.View, rows, C);
        await accelerator.SynchronizeAsync();
        var gpuOut = await outputBuf.CopyToHostAsync<float>(0, rows * C);

        float maxErr = 0f, sumErr = 0f;
        for (int i = 0; i < cpuOut.Length; i++)
        {
            float err = MathF.Abs(cpuOut[i] - gpuOut[i]);
            if (err > maxErr) maxErr = err;
            sumErr += err;
        }
        if (InferenceSession.VerboseLogging) Console.WriteLine($"[LayerNorm] Validate {rows}x{C}: maxErr={maxErr:E3}, avgErr={sumErr / cpuOut.Length:E3}");
        return (maxErr, sumErr / cpuOut.Length);
    }

    public void Dispose()
    {
        _means?.Dispose();
        _invStds?.Dispose();
    }
}
