using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// 2D Convolution kernel for neural network inference.
/// Supports:
///   - Arbitrary kernel sizes (1×1, 3×3, 14×14)
///   - Stride and padding
///   - Group=1 only (standard convolution)
///
/// Used for:
///   - Patch embedding (14×14 stride 14)
///   - DPT head projections (1×1)
///   - DPT refinement (3×3 pad 1)
///   - ConvTranspose via output padding (future)
///
/// Layout: NCHW (batch × channels × height × width)
/// Weights: [outChannels, inChannels, kH, kW]
/// </summary>
public class Conv2DKernel
{
    private readonly Accelerator _accelerator;

    // General Conv2D: one thread per output element
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // input [C_in * H * W]
        ArrayView1D<float, Stride1D.Dense>,  // weight [C_out * C_in * kH * kW]
        ArrayView1D<float, Stride1D.Dense>,  // bias [C_out] (or empty)
        ArrayView1D<float, Stride1D.Dense>,  // output [C_out * outH * outW]
        int, int, int,   // inC, inH, inW
        int, int, int,   // outC, kH, kW
        int, int>?       // stride, padding
        _conv2dKernel;

    public Conv2DKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Conv2D: one thread per output element.
    /// output[oc, oy, ox] = bias[oc] + Σ_ic Σ_ky Σ_kx input[ic, oy*stride+ky-pad, ox*stride+kx-pad] * weight[oc, ic, ky, kx]
    /// </summary>
    private static void Conv2DImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW,
        int outC, int kH, int kW,
        int stride, int padding)
    {
        int outH = (inH + 2 * padding - kH) / stride + 1;
        int outW = (inW + 2 * padding - kW) / stride + 1;

        // Decompose linear index → (oc, oy, ox)
        int ox = idx % outW;
        int rem = idx / outW;
        int oy = rem % outH;
        int oc = rem / outH;

        float sum = (bias.Length > 0) ? bias[oc] : 0f;

        // Convolve over input channels and kernel spatial dims
        for (int ic = 0; ic < inC; ic++)
        {
            for (int ky = 0; ky < kH; ky++)
            {
                int iy = oy * stride + ky - padding;
                if (iy < 0 || iy >= inH) continue;

                for (int kx = 0; kx < kW; kx++)
                {
                    int ix = ox * stride + kx - padding;
                    if (ix < 0 || ix >= inW) continue;

                    float inputVal = input[ic * inH * inW + iy * inW + ix];
                    float weightVal = weight[oc * inC * kH * kW + ic * kH * kW + ky * kW + kx];
                    sum += inputVal * weightVal;
                }
            }
        }

        output[idx] = sum;
    }

    /// <summary>
    /// Run Conv2D. Input: [inC, inH, inW]. Output: [outC, outH, outW].
    /// Weight: [outC, inC, kH, kW]. Bias: [outC] or empty.
    /// </summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW,
        int outC, int kH, int kW,
        int stride = 1, int padding = 0)
    {
        EnsureLoaded();

        int outH = (inH + 2 * padding - kH) / stride + 1;
        int outW = (inW + 2 * padding - kW) / stride + 1;
        int totalOutputElements = outC * outH * outW;

        _conv2dKernel!(totalOutputElements, input, weight, bias, output,
            inC, inH, inW, outC, kH, kW, stride, padding);
    }

    private void EnsureLoaded()
    {
        _conv2dKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int,
            int, int, int,
            int, int>(Conv2DImpl);
    }
}
