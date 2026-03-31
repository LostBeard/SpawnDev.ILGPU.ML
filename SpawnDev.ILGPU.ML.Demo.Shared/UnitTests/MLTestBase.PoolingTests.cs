using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task MaxPool2D_3x3Stride2() => await RunTest(async accelerator =>
    {
        int N = 1, C = 3, inH = 8, inW = 8, kH = 3, kW = 3, sH = 2, sW = 2, pH = 1, pW = 1;
        int outH = (inH + 2 * pH - kH) / sH + 1; // 4
        int outW = (inW + 2 * pW - kW) / sW + 1; // 4
        var input = RandomFloats(N * C * inH * inW, seed: 130, scale: 5f);

        // CPU reference
        var expected = new float[N * C * outH * outW];
        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++)
                    {
                        float max = float.MinValue;
                        for (int ky = 0; ky < kH; ky++)
                            for (int kx = 0; kx < kW; kx++)
                            {
                                int iy = oh * sH + ky - pH;
                                int ix = ow * sW + kx - pW;
                                if (iy >= 0 && iy < inH && ix >= 0 && ix < inW)
                                    max = MathF.Max(max, input[(n * C + c) * inH * inW + iy * inW + ix]);
                            }
                        expected[(n * C + c) * outH * outW + oh * outW + ow] = max;
                    }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(N * C * outH * outW);
        var pool = new PoolingKernels(accelerator);
        pool.MaxPool2D(inBuf.View, outBuf.View, N, C, inH, inW, kH, kW, sH, sW, pH, pW);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, N * C * outH * outW), expected, 1e-5f, "MaxPool2D: ");
    });

    [TestMethod]
    public async Task GlobalAvgPool_MatchesCpu() => await RunTest(async accelerator =>
    {
        int N = 2, C = 64, H = 7, W = 7;
        int spatial = H * W;
        var input = RandomFloats(N * C * spatial, seed: 131);

        var expected = new float[N * C];
        for (int nc = 0; nc < N * C; nc++)
        {
            float sum = 0;
            for (int i = 0; i < spatial; i++) sum += input[nc * spatial + i];
            expected[nc] = sum / spatial;
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(N * C);
        var pool = new PoolingKernels(accelerator);
        pool.GlobalAvgPool(inBuf.View, outBuf.View, N, C, spatial);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, N * C), expected, spatial * 1e-6f, "GlobalAvgPool: ");
    });
}
