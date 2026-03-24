using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Verify fused linear (MatMul + Bias + ReLU) matches separate operations.
    /// </summary>
    [TestMethod]
    public async Task FusedLinear_Relu_MatchesSeparate() => await RunTest(async accelerator =>
    {
        int M = 32, K = 64, N = 48;
        var input = RandomFloats(M * K, seed: 300);
        var weights = RandomFloats(K * N, seed: 301, scale: 0.1f);
        var bias = RandomFloats(N, seed: 302, scale: 0.5f);

        // CPU reference: MatMul + Bias + ReLU
        var expected = new float[M * N];
        for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
        {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += input[r * K + k] * weights[k * N + c];
            float val = sum + bias[c];
            expected[r * N + c] = val > 0 ? val : 0; // ReLU
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weights);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(M * N);

        var fused = new FusedLinearKernel(accelerator);
        fused.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, M, K, N, FusedActivation.ReLU);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, M * N);
        AssertClose(expected, actual, 1e-3f, "FusedLinear+ReLU: ");
    });

    /// <summary>
    /// Verify fused linear with GELU matches separate operations.
    /// </summary>
    [TestMethod]
    public async Task FusedLinear_Gelu_MatchesSeparate() => await RunTest(async accelerator =>
    {
        int M = 16, K = 32, N = 24;
        var input = RandomFloats(M * K, seed: 310);
        var weights = RandomFloats(K * N, seed: 311, scale: 0.1f);
        var bias = RandomFloats(N, seed: 312, scale: 0.3f);

        // CPU reference: MatMul + Bias + GELU
        var expected = new float[M * N];
        for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
        {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += input[r * K + k] * weights[k * N + c];
            float x = sum + bias[c];
            // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            if (x > 10f) expected[r * N + c] = x;
            else if (x < -10f) expected[r * N + c] = 0f;
            else expected[r * N + c] = 0.5f * x * (1f + ErfApprox(x * 0.7071067811865475f));
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weights);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(M * N);

        var fused = new FusedLinearKernel(accelerator);
        fused.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, M, K, N, FusedActivation.GELU);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, M * N);
        AssertClose(expected, actual, 1e-2f, "FusedLinear+GELU: ");
    });

    /// <summary>
    /// Verify fused linear with no activation matches plain MatMul + Bias.
    /// </summary>
    [TestMethod]
    public async Task FusedLinear_None_MatchesMatMulPlusBias() => await RunTest(async accelerator =>
    {
        int M = 8, K = 16, N = 12;
        var input = RandomFloats(M * K, seed: 320);
        var weights = RandomFloats(K * N, seed: 321, scale: 0.2f);
        var bias = RandomFloats(N, seed: 322);

        // CPU reference: MatMul + Bias
        var expected = CpuMatMul(input, weights, M, K, N);
        for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
            expected[r * N + c] += bias[c];

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weights);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(M * N);

        var fused = new FusedLinearKernel(accelerator);
        fused.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, M, K, N, FusedActivation.None);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, M * N);
        AssertClose(expected, actual, 1e-3f, "FusedLinear+None: ");
    });
}
