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

    public LayerNormKernel(Accelerator accelerator) => _accelerator = accelerator;

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
