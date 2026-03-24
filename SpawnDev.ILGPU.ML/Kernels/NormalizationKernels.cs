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

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, float>? _rmsNormStatsKernel;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int>? _rmsNormApplyKernel;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, float>? _instanceNormMeanVarKernel;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _instanceNormApplyKernel;

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
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int C = p[1]; int spatial = p[2];
        int c = (idx / spatial) % C;
        int sliceIdx = idx / spatial;
        output[idx] = scale[c] * (input[idx] - means[sliceIdx]) * invStds[sliceIdx] + bias[c];
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
        _bnParams.CopyFromCPU(new int[] { N, C, spatial });
        _batchNormKernel!(N * C * spatial, input, output, scale, bias, mean, variance, _bnParams.View);
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

        // Allocate/resize persistent temp buffer
        if (_rmsInvRms == null || _rmsInvRms.Length < rows)
        {
            _rmsInvRms?.Dispose();
            _rmsInvRms = _accelerator.Allocate1D<float>(rows);
        }

        // Pass 1: compute invRms per row
        _rmsNormStatsKernel!(rows, input, _rmsInvRms.View, C, epsilon);

        // Pass 2: apply normalization per element
        _rmsNormApplyKernel!(rows * C, input, output, weight, _rmsInvRms.View, C);
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
    public void InstanceNorm(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> scale,
        ArrayView1D<float, Stride1D.Dense> bias,
        int N, int C, int spatial)
    {
        EnsureLoaded();
        int numSlices = N * C;

        // Allocate temp buffers for means and invStds
        if (_inMeans == null || _inMeans.Length < numSlices)
        {
            _inMeans?.Dispose();
            _inMeans = _accelerator.Allocate1D<float>(numSlices);
        }
        if (_inInvStds == null || _inInvStds.Length < numSlices)
        {
            _inInvStds?.Dispose();
            _inInvStds = _accelerator.Allocate1D<float>(numSlices);
        }

        // Pass 1: compute mean + invStd per slice
        _instanceNormMeanVarKernel!(numSlices, input, _inMeans.View, _inInvStds.View, spatial, 1e-5f);

        // Pass 2: apply normalization
        _bnParams ??= _accelerator.Allocate1D<int>(3);
        _bnParams.CopyFromCPU(new int[] { N, C, spatial });
        _instanceNormApplyKernel!(N * C * spatial, input, output, scale, bias, _inMeans.View, _inInvStds.View, _bnParams.View);
    }

    private MemoryBuffer1D<float, Stride1D.Dense>? _inMeans;
    private MemoryBuffer1D<float, Stride1D.Dense>? _inInvStds;
    private MemoryBuffer1D<float, Stride1D.Dense>? _rmsInvRms;

    private void EnsureLoaded()
    {
        var a = _accelerator;
        _batchNormKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(BatchNormImpl);
        _instanceNormMeanVarKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, float>(InstanceNormMeanVarImpl);
        _instanceNormApplyKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(InstanceNormApplyImpl);
        _rmsNormStatsKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, float>(RMSNormStatsImpl);
        _rmsNormApplyKernel ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int>(RMSNormApplyImpl);
    }
}
