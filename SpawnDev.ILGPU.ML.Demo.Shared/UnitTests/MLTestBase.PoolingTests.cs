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

    /// <summary>
    /// GlobalAvgPool over a FULL-RESOLUTION feature map, not a classifier head.
    /// </summary>
    /// <remarks>
    /// ⚠️ THE COVERAGE GAP THIS CLOSES. <see cref="GlobalAvgPool_MatchesCpu"/> is the only other test of this
    /// kernel and it uses spatial=49 (a 7x7 head). `GlobalAvgPoolImpl` is "one thread per (n, c)" with a SERIAL
    /// loop over spatial inside, so its per-thread work - and therefore its dispatch duration - is entirely a
    /// function of the one dimension that test holds at 49. That is the same gap
    /// [[ref-one-thread-per-output]] records for the reductions: *"Every reduction test in the suite used ~6
    /// elements... the gap was in the test shape, not the test count."* A 49-element test cannot fail for any
    /// reason that only appears at 50,176, which is where the 2026-09-15 InstanceNorm device hang lived.
    ///
    /// The CPU reference accumulates in DOUBLE while the kernel accumulates in float over 50,176 terms, so this
    /// asserts the mean is right, not that two float summation orders agree bit for bit.
    ///
    /// 🔴 THE TOLERANCE IS ABSOLUTE AND DELIBERATELY NOT `spatial * 1e-6f`, WHICH IS WHAT THE 7x7 TEST USES.
    /// That formula scales with the very dimension being enlarged: at spatial=50,176 it is **0.05**, while the
    /// expected values here are means of 50,176 samples drawn from +/-5 and so land around **+/-0.022**. The
    /// tolerance would have been bigger than the signal, and this test could not have failed for any reason at
    /// all - a green that asserts nothing. A float sum of N terms carries roughly sqrt(N)*eps*mean|x| of error,
    /// about 7e-5 here, so 1e-3 is generous for the arithmetic while staying ~20x under the signal it is
    /// checking. Red-check any tolerance that is a FUNCTION of the size you just increased.
    /// </remarks>
    [TestMethod]
    public async Task GlobalAvgPool_FullResolutionSpatial_MatchesCpu() => await RunTest(async accelerator =>
    {
        int N = 1, C = 16, H = 224, W = 224;
        int spatial = H * W;                       // 50,176 - the StyleMosaic/VAE-scale feature map
        var input = RandomFloats(N * C * spatial, seed: 9152, scale: 5f);

        var expected = new float[N * C];
        for (int nc = 0; nc < N * C; nc++)
        {
            double sum = 0;
            for (int i = 0; i < spatial; i++) sum += input[nc * spatial + i];
            expected[nc] = (float)(sum / spatial);
        }
        // ✅ RED-CHECKED 2026-09-15: perturbing `expected` by +2e-3 (just over the tolerance) failed this test
        // on ALL SIX backends - WebGPU, Wasm, WebGL, CPU, CUDA, OpenCL - while GlobalAvgPool_MatchesCpu kept
        // passing. The assertion is live and discriminates at the intended magnitude on every lane.

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(N * C);
        var pool = new PoolingKernels(accelerator);
        pool.GlobalAvgPool(inBuf.View, outBuf.View, N, C, spatial);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, N * C), expected, 1e-3f,
            "GlobalAvgPool 1x16x224x224: ");
    });
}
