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
public class ConvTranspose2DKernel : IDisposable
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

        // BATCH-aware decode: idx spans (batch * outC * outH * outW). batch=1 → b==0, inBatchBase==0 =
        // byte-identical to the old kernel; only batch>1 (DAv3 multi-view, N views through the DPT resize_layers
        // ConvTranspose) changes. Without this it computed only view 0 and left every later view stale.
        int perBatchOut = outC * outH * outW;
        int b = idx / perBatchOut;
        int r = idx - b * perBatchOut;
        int ox = r % outW;
        int rem = r / outW;
        int oy = rem % outH;
        int oc = rem / outH;
        int inBatchBase = b * inC * inH * inW;

        float sum = bias[oc]; // f32 accumulation (ML-standard; see Conv2DKernel). Always read — no branch (ANGLE workaround)

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

                    sum += input[inBatchBase + ic * inH * inW + iy * inW + ix] * weight[ic * outC * kH * kW + oc * kH * kW + ky * kW + kx];
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
        int stride = 1, int padding = 0, int batch = 1)
    {
        EnsureLoaded();
        int outH = (inH - 1) * stride - 2 * padding + kH;
        int outW = (inW - 1) * stride - 2 * padding + kW;
        if (batch < 1) batch = 1;

        var packed = new int[] { inC, inH, inW, outC, kH, kW, stride, padding };
        ArrayView1D<int, Stride1D.Dense> paramsView;
        if (Graph.GraphExecutor.UseCaptureParamSlots)
        {
            // CUDA-graph capture: stable per-forward slot (the cached _paramsBuf's per-call CopyFromCPU is a
            // synchronous H2D, illegal mid-capture, and would alias across distinct ConvTranspose configs).
            paramsView = Kernels.CaptureParamArena.Shared(_accelerator).RentStableSlot(packed);
        }
        else
        {
            _paramsBuf ??= _accelerator.Allocate1D<int>(8);
            _paramsBuf.CopyFromCPU(packed);
            paramsView = _paramsBuf.View;
        }

        // Extent spans ALL batches; the kernel decodes the batch index (DAv3 multi-view = N views).
        _kernel!(batch * outC * outH * outW, input, weight, bias, output, paramsView);
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

    public void Dispose() => _paramsBuf?.Dispose();
}
