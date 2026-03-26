using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Group Normalization GPU kernel.
/// Divides channels into groups, normalizes each group independently.
/// Used by U-Net architectures (LGM, Stable Diffusion, etc.).
///
/// GroupNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
/// where mean and var are computed per group (not per channel or per batch).
/// </summary>
public class GroupNormKernel
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int, float>? _groupNormKernel;

    public GroupNormKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Apply GroupNorm to a 4D tensor [B, C, H, W].
    /// weight [C] and bias [C] are per-channel affine parameters.
    /// </summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        int batchSize, int channels, int spatial, int numGroups,
        float epsilon = 1e-5f)
    {
        // One thread per (batch, group) pair
        _groupNormKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int, float>(GroupNormImpl);
        _groupNormKernel(batchSize * numGroups, input, output, weight, bias,
            batchSize, channels, spatial, numGroups, epsilon);
    }

    private static void GroupNormImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        int B, int C, int S, int G, float eps)
    {
        int batch = idx / G;
        int group = idx % G;
        int channelsPerGroup = C / G;
        int groupSize = channelsPerGroup * S; // elements per group

        // Compute mean over group
        float sum = 0f;
        for (int c = 0; c < channelsPerGroup; c++)
        {
            int channelIdx = group * channelsPerGroup + c;
            int offset = batch * C * S + channelIdx * S;
            for (int s = 0; s < S; s++)
                sum += input[offset + s];
        }
        float mean = sum / groupSize;

        // Compute variance over group
        float varSum = 0f;
        for (int c = 0; c < channelsPerGroup; c++)
        {
            int channelIdx = group * channelsPerGroup + c;
            int offset = batch * C * S + channelIdx * S;
            for (int s = 0; s < S; s++)
            {
                float diff = input[offset + s] - mean;
                varSum += diff * diff;
            }
        }
        float invStd = 1f / MathF.Sqrt(varSum / groupSize + eps);

        // Normalize + affine transform (per-channel weight and bias)
        for (int c = 0; c < channelsPerGroup; c++)
        {
            int channelIdx = group * channelsPerGroup + c;
            int offset = batch * C * S + channelIdx * S;
            float w = weight[channelIdx];
            float b = bias[channelIdx];
            for (int s = 0; s < S; s++)
            {
                output[offset + s] = w * (input[offset + s] - mean) * invStd + b;
            }
        }
    }
}
