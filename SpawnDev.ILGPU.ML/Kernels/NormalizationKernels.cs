using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU normalization kernels beyond LayerNorm.
/// BatchNorm (inference mode), GroupNorm, InstanceNorm, RMSNorm.
/// All use auto-grouped 1D dispatch.
/// </summary>
public class NormalizationKernels
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _batchNormKernel;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, float>? _rmsNormKernel;

    private MemoryBuffer1D<int, Stride1D.Dense>? _bnParams;

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
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int C = p[1]; int spatial = p[2];
        float eps = 1e-5f;

        // Determine which channel this element belongs to
        int c = (idx / spatial) % C;

        float x = input[idx];
        float invStd = 1f / MathF.Sqrt(variance[c] + eps);
        output[idx] = scale[c] * (x - mean[c]) * invStd + bias[c];
    }

    /// <summary>
    /// RMSNorm: output = input / RMS(input) * weight, where RMS = sqrt(mean(x^2) + eps).
    /// One thread per row. Sequential over C elements per row.
    /// Used by LLaMA, Mistral, etc.
    /// </summary>
    private static void RMSNormImpl(Index1D row,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> weight,
        int C, float epsilon)
    {
        int offset = row * C;
        float sumSq = 0f;
        for (int i = 0; i < C; i++)
        {
            float v = input[offset + i];
            sumSq += v * v;
        }
        float rms = MathF.Sqrt(sumSq / C + epsilon);
        float invRms = 1f / rms;
        for (int i = 0; i < C; i++)
            output[offset + i] = input[offset + i] * invRms * weight[i];
    }

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _instanceNormKernel;

    /// <summary>
    /// InstanceNorm: normalize each (N, C) slice independently over spatial dims.
    /// One thread per element. params: [N, C, spatial]
    /// </summary>
    private static void InstanceNormImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> scale,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int C = p[1]; int spatial = p[2];
        float eps = 1e-5f;
        int c = (idx / spatial) % C;
        int ncBase = (idx / spatial / C) * C * spatial + c * spatial;

        // Compute mean and variance for this (n, c) slice
        float sum = 0f;
        for (int i = 0; i < spatial; i++)
            sum += input[ncBase + i];
        float mean = sum / spatial;

        float varSum = 0f;
        for (int i = 0; i < spatial; i++)
        {
            float d = input[ncBase + i] - mean;
            varSum += d * d;
        }
        float invStd = 1f / MathF.Sqrt(varSum / spatial + eps);

        output[idx] = scale[c] * (input[idx] - mean) * invStd + bias[c];
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
        int N, int C, int spatial)
    {
        EnsureLoaded();
        _bnParams ??= _accelerator.Allocate1D<int>(3);
        _bnParams.CopyFromCPU(new[] { N, C, spatial });
        _batchNormKernel!(N * C * spatial, input, output, scale, bias, mean, variance, _bnParams.View);
    }

    /// <summary>
    /// RMSNorm: input [rows, C] → output [rows, C]. weight: [C].
    /// </summary>
    public void RMSNorm(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> weight,
        int rows, int C, float epsilon = 1e-6f)
    {
        EnsureLoaded();
        _rmsNormKernel!(rows, input, output, weight, C, epsilon);
    }

    /// <summary>
    /// InstanceNorm: normalize each (N, C) slice over spatial dims.
    /// Input: [N, C, H, W]. scale, bias: [C].
    /// </summary>
    public void InstanceNorm(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> scale,
        ArrayView1D<float, Stride1D.Dense> bias,
        int N, int C, int spatial)
    {
        EnsureLoaded();
        _bnParams ??= _accelerator.Allocate1D<int>(3);
        _bnParams.CopyFromCPU(new[] { N, C, spatial });
        _instanceNormKernel!(N * C * spatial, input, output, scale, bias, _bnParams.View);
    }

    private void EnsureLoaded()
    {
        var a = _accelerator;
        _batchNormKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(BatchNormImpl);
        _instanceNormKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(InstanceNormImpl);
        _rmsNormKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, float>(RMSNormImpl);
    }
}
