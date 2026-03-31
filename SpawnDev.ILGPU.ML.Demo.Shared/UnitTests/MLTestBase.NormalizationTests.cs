using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task BatchNorm_MatchesCpu() => await RunTest(async accelerator =>
    {
        int N = 2, C = 64, H = 7, W = 7;
        int spatial = H * W;
        var input = RandomFloats(N * C * spatial, seed: 140);
        var scale = RandomFloats(C, seed: 141, scale: 0.5f);
        var bias = RandomFloats(C, seed: 142, scale: 0.1f);
        var mean = RandomFloats(C, seed: 143, scale: 2f);
        var variance = RandomFloats(C, seed: 144, scale: 1f);
        for (int i = 0; i < C; i++) variance[i] = MathF.Abs(variance[i]) + 0.1f; // positive

        // CPU reference
        float eps = 1e-5f;
        var expected = new float[N * C * spatial];
        for (int i = 0; i < expected.Length; i++)
        {
            int c = (i / spatial) % C;
            float invStd = 1f / MathF.Sqrt(variance[c] + eps);
            expected[i] = scale[c] * (input[i] - mean[c]) * invStd + bias[c];
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(N * C * spatial);
        using var sBuf = accelerator.Allocate1D(scale);
        using var bBuf = accelerator.Allocate1D(bias);
        using var mBuf = accelerator.Allocate1D(mean);
        using var vBuf = accelerator.Allocate1D(variance);

        var norm = new NormalizationKernels(accelerator);
        norm.BatchNorm(inBuf.View, outBuf.View, sBuf.View, bBuf.View, mBuf.View, vBuf.View, N, C, spatial);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, N * C * spatial), expected, 1e-4f, "BatchNorm: ");
    });

    [TestMethod]
    public async Task InstanceNorm_MatchesCpu() => await RunTest(async accelerator =>
    {
        int N = 1, C = 3, H = 8, W = 8;
        int spatial = H * W;
        int total = N * C * spatial;
        var input = RandomFloats(total, seed: 150, scale: 5f);
        var scale = RandomFloats(C, seed: 151, scale: 1f);
        var bias = RandomFloats(C, seed: 152, scale: 0.5f);
        for (int i = 0; i < C; i++) scale[i] = MathF.Abs(scale[i]) + 0.5f;

        // CPU reference: normalize each (n,c) slice independently
        float eps = 1e-5f;
        var expected = new float[total];
        for (int n = 0; n < N; n++)
        {
            for (int c = 0; c < C; c++)
            {
                int sliceBase = (n * C + c) * spatial;
                float sum = 0;
                for (int i = 0; i < spatial; i++) sum += input[sliceBase + i];
                float mean = sum / spatial;
                float varSum = 0;
                for (int i = 0; i < spatial; i++)
                {
                    float d = input[sliceBase + i] - mean;
                    varSum += d * d;
                }
                float invStd = 1f / MathF.Sqrt(varSum / spatial + eps);
                for (int i = 0; i < spatial; i++)
                    expected[sliceBase + i] = scale[c] * (input[sliceBase + i] - mean) * invStd + bias[c];
            }
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(total);
        using var sBuf = accelerator.Allocate1D(scale);
        using var bBuf = accelerator.Allocate1D(bias);

        var norm = new NormalizationKernels(accelerator);
        norm.InstanceNorm(inBuf.View, outBuf.View, sBuf.View, bBuf.View, N, C, spatial);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, total), expected, 1e-4f, "InstanceNorm small: ");
    });

    [TestMethod]
    public async Task InstanceNorm_StyleTransferDims_MatchesCpu() => await RunTest(async accelerator =>
    {
        // Style transfer dimensions: [1, 3, 224, 224]
        int N = 1, C = 3, H = 224, W = 224;
        int spatial = H * W;
        int total = N * C * spatial;
        var input = RandomFloats(total, seed: 160, scale: 255f);
        var scale = new float[] { 1f, 1f, 1f };
        var bias = new float[] { 0f, 0f, 0f };

        // CPU reference
        float eps = 1e-5f;
        var expected = new float[total];
        for (int n = 0; n < N; n++)
        {
            for (int c = 0; c < C; c++)
            {
                int sliceBase = (n * C + c) * spatial;
                float sum = 0;
                for (int i = 0; i < spatial; i++) sum += input[sliceBase + i];
                float mean = sum / spatial;
                float varSum = 0;
                for (int i = 0; i < spatial; i++)
                {
                    float d = input[sliceBase + i] - mean;
                    varSum += d * d;
                }
                float invStd = 1f / MathF.Sqrt(varSum / spatial + eps);
                for (int i = 0; i < spatial; i++)
                    expected[sliceBase + i] = scale[c] * (input[sliceBase + i] - mean) * invStd + bias[c];
            }
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(total);
        using var sBuf = accelerator.Allocate1D(scale);
        using var bBuf = accelerator.Allocate1D(bias);

        var norm = new NormalizationKernels(accelerator);
        norm.InstanceNorm(inBuf.View, outBuf.View, sBuf.View, bBuf.View, N, C, spatial);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, total), expected, 1e-3f, "InstanceNorm 224x224: ");
    });

    [TestMethod]
    public async Task RMSNorm_MatchesCpu() => await RunTest(async accelerator =>
    {
        int rows = 100, C = 384;
        var input = RandomFloats(rows * C, seed: 145);
        var weight = RandomFloats(C, seed: 146, scale: 0.5f);
        for (int i = 0; i < C; i++) weight[i] = MathF.Abs(weight[i]) + 0.5f;

        // CPU reference
        float eps = 1e-6f;
        var expected = new float[rows * C];
        for (int r = 0; r < rows; r++)
        {
            float sumSq = 0;
            for (int i = 0; i < C; i++)
            {
                float v = input[r * C + i];
                sumSq += v * v;
            }
            float rms = MathF.Sqrt(sumSq / C + eps);
            float invRms = 1f / rms;
            for (int i = 0; i < C; i++)
                expected[r * C + i] = input[r * C + i] * invRms * weight[i];
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(rows * C);
        using var wBuf = accelerator.Allocate1D(weight);

        var norm = new NormalizationKernels(accelerator);
        norm.RMSNorm(inBuf.View, outBuf.View, wBuf.View, rows, C);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, rows * C), expected, 1e-4f, "RMSNorm: ");
    });
}
