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
        // One thread per output element — gather-only, WebGL TF compatible.
        _groupNormKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int, float>(GroupNormImpl);
        _groupNormKernel(batchSize * channels * spatial, input, output, weight, bias,
            batchSize, channels, spatial, numGroups, epsilon);
    }

    /// <summary>
    /// One thread per output position. Each thread recomputes its group's mean+variance
    /// (groupSize reads) and produces its single scalar output. Replaces a scatter pattern
    /// where one (batch, group) thread wrote (channelsPerGroup * spatial) outputs — that
    /// pattern is silently dropped by WebGL Transform Feedback. Total work is the same
    /// arithmetic intensity; just redistributed across many more threads.
    /// </summary>
    private static void GroupNormImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        int B, int C, int S, int G, float eps)
    {
        // Decode idx → (batch, channel, s)
        int s = idx % S;
        int bc = idx / S;
        int channel = bc % C;
        int batch = bc / C;
        int channelsPerGroup = C / G;
        int group = channel / channelsPerGroup;
        int groupSize = channelsPerGroup * S;
        int groupBase = batch * C * S + group * channelsPerGroup * S;

        // Compute mean over the whole (batch, group)
        float sum = 0f;
        for (int gi = 0; gi < groupSize; gi++)
            sum += input[groupBase + gi];
        float mean = sum / groupSize;

        // Compute variance
        float varSum = 0f;
        for (int gi = 0; gi < groupSize; gi++)
        {
            float diff = input[groupBase + gi] - mean;
            varSum += diff * diff;
        }
        float invStd = 1f / MathF.Sqrt(varSum / groupSize + eps);

        // Normalize + affine for this single element
        output[idx] = weight[channel] * (input[idx] - mean) * invStd + bias[channel];
    }
}
