using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// 2D Convolution kernel for neural network inference.
/// Supports arbitrary kernel sizes (1×1, 3×3, 14×14), stride, and padding.
/// Group=1 only (standard convolution).
///
/// Layout: NCHW (batch × channels × height × width)
/// Weights: [outChannels, inChannels, kH, kW]
///
/// Parameters are packed into an ArrayView to avoid WebGPU uniform buffer
/// packing issues with high scalar parameter counts.
/// </summary>
public class Conv2DKernel
{
    private readonly Accelerator _accelerator;

    // Conv2D with params packed into a buffer: [inC, inH, inW, outC, kH, kW, stride, padding]
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // input
        ArrayView1D<float, Stride1D.Dense>,  // weight
        ArrayView1D<float, Stride1D.Dense>,  // bias
        ArrayView1D<float, Stride1D.Dense>,  // output
        ArrayView1D<int, Stride1D.Dense>>?   // params [8]
        _conv2dKernel;

    // Persistent params buffer (reused across calls)
    private MemoryBuffer1D<int, Stride1D.Dense>? _paramsBuf;

    public Conv2DKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Conv2D: one thread per output element. Parameters read from params buffer.
    /// params[0]=inC, [1]=inH, [2]=inW, [3]=outC, [4]=kH, [5]=kW, [6]=stride, [7]=padding
    /// </summary>
    private static void Conv2DImpl(
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

        int outH = (inH + 2 * padding - kH) / stride + 1;
        int outW = (inW + 2 * padding - kW) / stride + 1;

        int ox = idx % outW;
        int rem = idx / outW;
        int oy = rem % outH;
        int oc = rem / outH;

        float sum = (bias.Length > 0) ? bias[oc] : 0f;

        // Single flat loop to avoid WebGPU WGSL codegen issues with triple-nested loops.
        int kHkW = kH * kW;
        int totalIter = inC * kHkW;
        for (int i = 0; i < totalIter; i++)
        {
            int ic = i / kHkW;
            int kRem = i % kHkW;
            int ky = kRem / kW;
            int kx = kRem % kW;

            int iy = oy * stride + ky - padding;
            int ix = ox * stride + kx - padding;

            if (iy >= 0 && iy < inH && ix >= 0 && ix < inW)
            {
                float inputVal = input[ic * inH * inW + iy * inW + ix];
                float weightVal = weight[oc * inC * kHkW + ic * kHkW + ky * kW + kx];
                sum += inputVal * weightVal;
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

        // Pack params into buffer
        if (_paramsBuf == null)
            _paramsBuf = _accelerator.Allocate1D<int>(8);
        _paramsBuf.CopyFromCPU(new int[] { inC, inH, inW, outC, kH, kW, stride, padding });

        _conv2dKernel!(totalOutputElements, input, weight, bias, output, _paramsBuf.View);
    }

    private void EnsureLoaded()
    {
        _conv2dKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(Conv2DImpl);
    }
}
