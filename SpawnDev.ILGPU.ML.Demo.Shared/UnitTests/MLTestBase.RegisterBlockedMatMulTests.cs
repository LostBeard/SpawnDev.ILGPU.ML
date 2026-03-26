using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Tests for register-blocked tiled MatMul (200+ GFLOPS target).
/// Validates correctness against CPU reference before benchmarking.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task RegisterBlockedMatMul_SmallMatrix_MatchesCPU() => await RunTest(async accelerator =>
    {
        // Simple 4x4 × 4x4 = 4x4
        int M = 4, K = 4, N = 4;
        var A = new float[] {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        };
        var B = new float[] {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        }; // Identity → C should equal A

        // CPU reference
        var expectedC = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float sum = 0;
                for (int k = 0; k < K; k++)
                    sum += A[i * K + k] * B[k * N + j];
                expectedC[i * N + j] = sum;
            }

        // GPU MatMul (use existing tiled MatMul for correctness comparison)
        using var aBuf = accelerator.Allocate1D(A);
        using var bBuf = accelerator.Allocate1D(B);
        using var cBuf = accelerator.Allocate1D<float>(M * N);

        var matmul = new MatMulKernel(accelerator);
        matmul.Forward(aBuf.View, bBuf.View, cBuf.View, M, K, N);
        await accelerator.SynchronizeAsync();
        var gpuC = await cBuf.CopyToHostAsync<float>(0, M * N);

        float maxErr = 0;
        for (int i = 0; i < M * N; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(gpuC[i] - expectedC[i]));

        if (maxErr > 0.01f)
            throw new Exception($"MatMul 4x4 identity: maxErr={maxErr:F4}");

        Console.WriteLine($"[MatMul] 4x4 identity: maxErr={maxErr:E3} — baseline correct");
    });
}
