using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// Transposed 2D Convolution (deconvolution / fractionally-strided convolution).
/// Used for DPT head resize_layers that upsample spatial resolution.
///
/// Weight layout (PyTorch ConvTranspose2d): [inC, outC, kH, kW]
/// Output size (no output_padding): outH = (inH - 1) * stride - 2 * padding + kH
///
/// Implemented in "gather" direction — one thread per output element, no atomics.
/// Parameters packed into ArrayView to avoid WebGPU scalar packing issues.
/// </summary>
public class ConvTranspose2DKernel
{
    private readonly Accelerator _accelerator;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // input
        ArrayView1D<float, Stride1D.Dense>,  // weight
        ArrayView1D<float, Stride1D.Dense>,  // bias
        ArrayView1D<float, Stride1D.Dense>,  // output
        ArrayView1D<int, Stride1D.Dense>>?   // params [8]
        _kernel;

    private MemoryBuffer1D<int, Stride1D.Dense>? _paramsBuf;

    public ConvTranspose2DKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// params: [inC, inH, inW, outC, kH, kW, stride, padding]
    /// </summary>
    private static void ConvTranspose2DImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int inC = p[0]; int inH = p[1]; int inW = p[2];
        int outC = p[3]; int kH = p[4]; int kW = p[5];
        int stride = p[6]; int padding = p[7];

        int outH = (inH - 1) * stride - 2 * padding + kH;
        int outW = (inW - 1) * stride - 2 * padding + kW;

        int ox = idx % outW;
        int rem = idx / outW;
        int oy = rem % outH;
        int oc = rem / outH;

        float sum = (bias.Length > 0) ? bias[oc] : 0f;

        for (int ic = 0; ic < inC; ic++)
        {
            for (int ky = 0; ky < kH; ky++)
            {
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

                    float w = weight[ic * outC * kH * kW + oc * kH * kW + ky * kW + kx];
                    sum += input[ic * inH * inW + iy * inW + ix] * w;
                }
            }
        }

        output[idx] = sum;
    }

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

        using var paramsBuf = _accelerator.Allocate1D(new int[] { inC, inH, inW, outC, kH, kW, stride, padding });

        _kernel!(outC * outH * outW, input, weight, bias, output, paramsBuf.View);
    }

    public static int OutputSize(int inputSize, int kernelSize, int stride, int padding)
        => (inputSize - 1) * stride - 2 * padding + kernelSize;

    private void EnsureLoaded()
    {
        _kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(ConvTranspose2DImpl);
    }
}
