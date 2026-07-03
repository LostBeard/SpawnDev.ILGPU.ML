using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task LayerNorm_Dav3Dimensions() => await RunTest(async accelerator =>
    {
        int rows = 1370, C = 384;
        var input = RandomFloats(rows * C, seed: 42);
        var gamma = RandomFloats(C, seed: 43, scale: 2f);
        var beta = RandomFloats(C, seed: 44, scale: 0.1f);
        for (int i = 0; i < C; i++) gamma[i] = MathF.Abs(gamma[i]) + 0.1f;

        var expected = CpuLayerNorm(input, gamma, beta, rows, C);

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(rows * C);
        using var gBuf = accelerator.Allocate1D(gamma);
        using var bBuf = accelerator.Allocate1D(beta);

        var layerNorm = new LayerNormKernel(accelerator);
        layerNorm.Forward(inBuf.View, outBuf.View, gBuf.View, bBuf.View, rows, C);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "LayerNorm [1370x384]: ");
    });

    /// <summary>
    /// GPU min/max reduction (ImagePostprocessKernel.MinMaxAsync - the 8-byte readback replacing the
    /// video path's full-map host readback): random data with negatives, count NOT a multiple of the
    /// 1024-thread partial stage, exact match vs host min/max (float comparisons, no arithmetic -
    /// result must be EXACT). Runs on all 6 backends (one store per thread per array + a positional
    /// 2-slot final store - WebGL-safe).
    /// </summary>
    [TestMethod]
    public async Task MinMax_GpuReduction_MatchesHost() => await RunTest(async accelerator =>
    {
        int count = 518 * 518 + 37;   // video-path size, deliberately not 1024-aligned
        var data = RandomFloats(count, seed: 777, scale: 5f);
        data[123] = -42.5f; data[count - 1] = 99.25f;   // known extremes at awkward positions

        float expMin = data[0], expMax = data[0];
        foreach (var v in data) { if (v < expMin) expMin = v; if (v > expMax) expMax = v; }

        using var buf = accelerator.Allocate1D(data);
        using var post = new Kernels.ImagePostprocessKernel(accelerator);
        var (mn, mx) = await post.MinMaxAsync(buf.View, count);
        if (mn != expMin || mx != expMax)
            throw new Exception($"GPU min/max mismatch: got ({mn}, {mx}) expected ({expMin}, {expMax})");
    });

    [TestMethod]
    public async Task GELU_MatchesCpuErf() => await RunTest(async accelerator =>
    {
        int count = 1000;
        var input = RandomFloats(count, seed: 1, scale: 3f);

        var expected = new float[count];
        for (int i = 0; i < count; i++)
        {
            float x = input[i];
            double z = x / Math.Sqrt(2.0);
            double az = Math.Abs(z);
            double t = 1.0 / (1.0 + 0.3275911 * az);
            double erfAbs = 1.0 - (0.254829592 * t - 0.284496736 * t * t + 1.421413741 * t * t * t
                - 1.453152027 * t * t * t * t + 1.061405429 * t * t * t * t * t) * Math.Exp(-az * az);
            double erf = z < 0 ? -erfAbs : erfAbs;
            expected[i] = (float)(0.5 * x * (1.0 + erf));
        }

        using var buf = accelerator.Allocate1D((float[])input.Clone());
        var ew = new ElementWiseKernels(accelerator);
        ew.GELUInPlace(buf.View, count);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, buf.View, expected, 1e-5f, "GELU erf: ");
    });

    [TestMethod]
    public async Task BroadcastMul_LayerScale() => await RunTest(async accelerator =>
    {
        int T = 1370, C = 384;
        var input = RandomFloats(T * C, seed: 10);
        var gamma = RandomFloats(C, seed: 11, scale: 0.1f);

        var expected = new float[T * C];
        for (int i = 0; i < T * C; i++)
            expected[i] = input[i] * gamma[i % C];

        using var inBuf = accelerator.Allocate1D(input);
        using var gBuf = accelerator.Allocate1D(gamma);
        using var outBuf = accelerator.Allocate1D<float>(T * C);

        var ew = new ElementWiseKernels(accelerator);
        ew.BroadcastMul(inBuf.View, gBuf.View, outBuf.View, T * C, C);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-5f, "BroadcastMul: ");
    });

    [TestMethod]
    public async Task Softmax_AttentionDimensions() => await RunTest(async accelerator =>
    {
        int rows = 96, cols = 1370;
        var data = RandomFloats(rows * cols, seed: 50, scale: 2f);

        var expected = (float[])data.Clone();
        for (int r = 0; r < rows; r++)
        {
            float max = float.MinValue;
            for (int c = 0; c < cols; c++) max = MathF.Max(max, expected[r * cols + c]);
            float sum = 0;
            for (int c = 0; c < cols; c++) { expected[r * cols + c] = MathF.Exp(expected[r * cols + c] - max); sum += expected[r * cols + c]; }
            for (int c = 0; c < cols; c++) expected[r * cols + c] /= sum;
        }

        using var buf = accelerator.Allocate1D((float[])data.Clone());
        var softmax = new SoftmaxKernel(accelerator);
        softmax.Forward(buf.View, rows, cols);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, buf.View, expected, 1e-5f, "Softmax: ");
    });

    [TestMethod]
    public async Task TransposeLastTwo_RoundTrip() => await RunTest(async accelerator =>
    {
        int batch = 6, rows = 1370, cols = 64;
        var input = RandomFloats(batch * rows * cols, seed: 30);

        using var inBuf = accelerator.Allocate1D(input);
        using var transBuf = accelerator.Allocate1D<float>(batch * rows * cols);
        using var roundBuf = accelerator.Allocate1D<float>(batch * rows * cols);

        var ew = new ElementWiseKernels(accelerator);
        ew.TransposeLastTwo(inBuf.View, transBuf.View, batch, rows, cols);
        ew.TransposeLastTwo(transBuf.View, roundBuf.View, batch, cols, rows);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, roundBuf.View, input, 0f, "Transpose round-trip: ");
    });
}
