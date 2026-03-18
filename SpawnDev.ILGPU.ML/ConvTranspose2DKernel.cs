using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// Transposed 2D Convolution (deconvolution / fractionally-strided convolution).
/// Used for the DPT head resize_layers that upsample spatial resolution.
///
/// Weight layout (PyTorch ConvTranspose2d): [inC, outC, kH, kW]
///
/// Output size (no output_padding): outH = (inH - 1) * stride - 2 * padding + kH
///
/// Implemented in "gather" direction — one thread per output element, no atomics.
/// For stride == kernel_size (no overlap): each output element receives exactly one contribution,
/// making this very efficient (k=4,s=4 or k=2,s=2).
/// </summary>
public class ConvTranspose2DKernel
{
    private readonly Accelerator _accelerator;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // input
        ArrayView1D<float, Stride1D.Dense>,  // weight [inC, outC, kH, kW]
        ArrayView1D<float, Stride1D.Dense>,  // bias [outC] or empty
        ArrayView1D<float, Stride1D.Dense>,  // output
        int, int, int,    // inC, inH, inW
        int, int, int,    // outC, kH, kW
        int, int>?        // stride, padding
        _kernel;

    public ConvTranspose2DKernel(Accelerator accelerator) => _accelerator = accelerator;

    private static void ConvTranspose2DImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW,
        int outC, int kH, int kW,
        int stride, int padding)
    {
        // Output dimensions (no output_padding)
        int outH = (inH - 1) * stride - 2 * padding + kH;
        int outW = (inW - 1) * stride - 2 * padding + kW;

        // Decompose linear index → (oc, oy, ox)
        int ox = idx % outW;
        int rem = idx / outW;
        int oy = rem % outH;
        int oc = rem / outH;

        float sum = (bias.Length > 0) ? bias[oc] : 0f;

        for (int ic = 0; ic < inC; ic++)
        {
            for (int ky = 0; ky < kH; ky++)
            {
                // For ConvTranspose: input[iy] contributes to output[iy*stride + ky].
                // Gather direction: for output[oy], need iy = (oy + padding - ky) / stride
                int diffY = oy + padding - ky;
                if (diffY < 0 || diffY % stride != 0) continue;
                int iy = diffY / stride;
                if (iy >= inH) continue;

                for (int kx = 0; kx < kW; kx++)
                {
                    int diffX = ox + padding - kx;
                    if (diffX < 0 || diffX % stride != 0) continue;
                    int ix = diffX / stride;
                    if (ix >= inW) continue;

                    // weight layout: [inC, outC, kH, kW]
                    float w = weight[ic * outC * kH * kW + oc * kH * kW + ky * kW + kx];
                    sum += input[ic * inH * inW + iy * inW + ix] * w;
                }
            }
        }

        output[idx] = sum;
    }

    /// <summary>
    /// Run ConvTranspose2D. Input: [inC, inH, inW]. Output: [outC, outH, outW].
    /// Weight: [inC, outC, kH, kW] (PyTorch ConvTranspose2d convention).
    /// Bias: [outC] or empty.
    /// Output size: outH = (inH-1)*stride - 2*padding + kH
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
        int outH = (inH - 1) * stride - 2 * padding + kH;
        int outW = (inW - 1) * stride - 2 * padding + kW;
        _kernel!(outC * outH * outW, input, weight, bias, output,
            inC, inH, inW, outC, kH, kW, stride, padding);
    }

    /// <summary>Compute output spatial size for given parameters.</summary>
    public static int OutputSize(int inputSize, int kernelSize, int stride, int padding)
        => (inputSize - 1) * stride - 2 * padding + kernelSize;

    private void EnsureLoaded()
    {
        _kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int,
            int, int, int,
            int, int>(ConvTranspose2DImpl);
    }
}
