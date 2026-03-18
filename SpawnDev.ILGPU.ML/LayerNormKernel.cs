using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// Layer Normalization: normalize over the last dimension (C), apply learned gamma + beta.
/// Uses auto-grouped kernels (no shared memory) to avoid WGSL variable redeclaration bug
/// with multiple LoadStreamKernel calls on the same accelerator.
///
/// Two-pass approach:
///   Pass 1: Each thread handles one row — computes mean, variance, and normalizes in one pass.
///   (C=384 is small enough for sequential processing per row.)
///
/// Future home: SpawnDev.ILGPU.ML
/// </summary>
public class LayerNormKernel
{
    private readonly Accelerator _accelerator;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // input [rows*C]
        ArrayView1D<float, Stride1D.Dense>,  // output [rows*C]
        ArrayView1D<float, Stride1D.Dense>,  // gamma [C]
        ArrayView1D<float, Stride1D.Dense>,  // beta [C]
        int, float>?                          // C, epsilon
        _kernel;

    public LayerNormKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// One thread per row. Sequential mean/var/normalize over C elements.
    /// For C=384, this is ~1200 FLOPs per thread — trivial.
    /// </summary>
    private static void LayerNormRowImpl(
        Index1D row,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> gamma,
        ArrayView1D<float, Stride1D.Dense> beta,
        int C, float epsilon)
    {
        int offset = row * C;

        // Mean
        float sum = 0f;
        for (int i = 0; i < C; i++)
            sum += input[offset + i];
        float mean = sum / C;

        // Variance
        float varSum = 0f;
        for (int i = 0; i < C; i++)
        {
            float diff = input[offset + i] - mean;
            varSum += diff * diff;
        }
        float invStd = 1f / MathF.Sqrt(varSum / C + epsilon);

        // Normalize + scale + bias
        for (int i = 0; i < C; i++)
            output[offset + i] = gamma[i] * ((input[offset + i] - mean) * invStd) + beta[i];
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
        _kernel!(rows, input, output, gamma, beta, C, epsilon);
    }

    private void EnsureLoaded(Accelerator accelerator)
    {
        _kernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, float>(LayerNormRowImpl);
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
        Console.WriteLine($"[LayerNorm] Diagnostic [1,2,3,4]: [{string.Join(", ", gpuOut.Select(v => v.ToString("F4")))}] (expect [-1.3416, -0.4472, 0.4472, 1.3416])");
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
        Console.WriteLine($"[LayerNorm] Validate {rows}x{C}: maxErr={maxErr:E3}, avgErr={sumErr / cpuOut.Length:E3}");
        return (maxErr, sumErr / cpuOut.Length);
    }
}
