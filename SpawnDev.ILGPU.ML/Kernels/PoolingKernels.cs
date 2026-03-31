using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU pooling kernels: MaxPool2D, AvgPool2D, GlobalAvgPool.
/// Layout: NCHW. One thread per output element.
/// </summary>
public class PoolingKernels
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _maxPool2d;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _avgPool2d;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int>? _globalAvgPool;

    public PoolingKernels(Accelerator accelerator) => _accelerator = accelerator;

    // params: [N, C, inH, inW, kH, kW, strideH, strideW, padH, padW]
    private static void MaxPool2DImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int N = p[0]; int C = p[1]; int inH = p[2]; int inW = p[3];
        int kH = p[4]; int kW = p[5]; int sH = p[6]; int sW = p[7]; int pH = p[8]; int pW = p[9];
        int outH = (inH + 2 * pH - kH) / sH + 1;
        int outW = (inW + 2 * pW - kW) / sW + 1;

        int ow = idx % outW; int rem = idx / outW;
        int oh = rem % outH; rem /= outH;
        int c = rem % C; int n = rem / C;

        float max = -1e38f;
        int inBase = (n * C + c) * inH * inW;
        int totalK = kH * kW;
        for (int k = 0; k < totalK; k++)
        {
            int ky = k / kW; int kx = k % kW;
            int iy = oh * sH + ky - pH;
            int ix = ow * sW + kx - pW;
            if (iy >= 0 && iy < inH && ix >= 0 && ix < inW)
            {
                float v = input[inBase + iy * inW + ix];
                if (v > max) max = v;
            }
        }
        output[idx] = max;
    }

    private static void AvgPool2DImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int N = p[0]; int C = p[1]; int inH = p[2]; int inW = p[3];
        int kH = p[4]; int kW = p[5]; int sH = p[6]; int sW = p[7]; int pH = p[8]; int pW = p[9];
        int outH = (inH + 2 * pH - kH) / sH + 1;
        int outW = (inW + 2 * pW - kW) / sW + 1;

        int ow = idx % outW; int rem = idx / outW;
        int oh = rem % outH; rem /= outH;
        int c = rem % C; int n = rem / C;

        float sum = 0f; int count = 0;
        int inBase = (n * C + c) * inH * inW;
        int totalK = kH * kW;
        for (int k = 0; k < totalK; k++)
        {
            int ky = k / kW; int kx = k % kW;
            int iy = oh * sH + ky - pH;
            int ix = ow * sW + kx - pW;
            if (iy >= 0 && iy < inH && ix >= 0 && ix < inW)
            {
                sum += input[inBase + iy * inW + ix];
                count++;
            }
        }
        output[idx] = count > 0 ? sum / count : 0f;
    }

    /// <summary>GlobalAvgPool: [N, C, H, W] → [N, C, 1, 1]. One thread per (n, c).</summary>
    private static void GlobalAvgPoolImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int spatial, int dummy)
    {
        float sum = 0f;
        int baseIdx = idx * spatial;
        for (int i = 0; i < spatial; i++)
            sum += input[baseIdx + i];
        output[idx] = sum / spatial;
    }

    // ── Public API ──

    private MemoryBuffer1D<int, Stride1D.Dense>? _poolParams;

    public void MaxPool2D(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int N, int C, int inH, int inW, int kH, int kW, int strideH, int strideW, int padH, int padW)
    {
        EnsureLoaded();
        int outH = (inH + 2 * padH - kH) / strideH + 1;
        int outW = (inW + 2 * padW - kW) / strideW + 1;
        if (outH <= 0 || outW <= 0)
            throw new InvalidOperationException(
                $"MaxPool2D output dimensions are invalid: outH={outH}, outW={outW} " +
                $"(N={N}, C={C}, inH={inH}, inW={inW}, kH={kH}, kW={kW}, sH={strideH}, sW={strideW}, padH={padH}, padW={padW})");
        _poolParams ??= _accelerator.Allocate1D<int>(10);
        _poolParams.CopyFromCPU(new int[] { N, C, inH, inW, kH, kW, strideH, strideW, padH, padW });
        _maxPool2d!(N * C * outH * outW, input, output, _poolParams.View);
    }

    public void AvgPool2D(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int N, int C, int inH, int inW, int kH, int kW, int strideH, int strideW, int padH, int padW)
    {
        EnsureLoaded();
        int outH = (inH + 2 * padH - kH) / strideH + 1;
        int outW = (inW + 2 * padW - kW) / strideW + 1;
        if (outH <= 0 || outW <= 0)
            throw new InvalidOperationException(
                $"AvgPool2D output dimensions are invalid: outH={outH}, outW={outW} " +
                $"(N={N}, C={C}, inH={inH}, inW={inW}, kH={kH}, kW={kW}, sH={strideH}, sW={strideW}, padH={padH}, padW={padW})");
        _poolParams ??= _accelerator.Allocate1D<int>(10);
        _poolParams.CopyFromCPU(new int[] { N, C, inH, inW, kH, kW, strideH, strideW, padH, padW });
        _avgPool2d!(N * C * outH * outW, input, output, _poolParams.View);
    }

    public void GlobalAvgPool(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int N, int C, int spatial)
    {
        EnsureLoaded();
        _globalAvgPool!(N * C, input, output, spatial, 0);
    }

    private void EnsureLoaded()
    {
        var a = _accelerator;
        _maxPool2d ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(MaxPool2DImpl);
        _avgPool2d ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(AvgPool2DImpl);
        _globalAvgPool ??= a.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int>(GlobalAvgPoolImpl);
    }
}
