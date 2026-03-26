using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Tests for register-blocked tiled MatMul (200+ GFLOPS target).
/// Validates correctness against CPU reference, then benchmarks.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task RegisterBlockedMatMul_SmallMatrix_MatchesCPU() => await RunTest(async accelerator =>
    {
        // Small matrix — hits the simple fallback path (< 64×64)
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

        var expectedC = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float sum = 0;
                for (int k = 0; k < K; k++)
                    sum += A[i * K + k] * B[k * N + j];
                expectedC[i * N + j] = sum;
            }

        using var aBuf = accelerator.Allocate1D(A);
        using var bBuf = accelerator.Allocate1D(B);
        using var cBuf = accelerator.Allocate1D<float>(M * N);

        var matmul = new RegisterBlockedMatMul(accelerator);
        matmul.MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
        await accelerator.SynchronizeAsync();
        var gpuC = await cBuf.CopyToHostAsync<float>(0, M * N);

        float maxErr = 0;
        for (int i = 0; i < M * N; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(gpuC[i] - expectedC[i]));

        if (maxErr > 0.01f)
            throw new Exception($"RegisterBlockedMatMul 4x4 identity: maxErr={maxErr:F4}");

        Console.WriteLine($"[RegisterBlockedMatMul] 4x4 fallback: maxErr={maxErr:E3} — correct");
    });

    [TestMethod]
    public async Task RegisterBlockedMatMul_LargeMatrix_MatchesCPU() => await RunTest(async accelerator =>
    {
        // 128×128 — large enough to use the register-blocked tiled path (TILE=64)
        int M = 128, K = 128, N = 128;
        var rng = new Random(42);
        var A = new float[M * K];
        var B = new float[K * N];
        for (int i = 0; i < A.Length; i++) A[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < B.Length; i++) B[i] = (float)(rng.NextDouble() * 2 - 1);

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

        using var aBuf = accelerator.Allocate1D(A);
        using var bBuf = accelerator.Allocate1D(B);
        using var cBuf = accelerator.Allocate1D<float>(M * N);

        var matmul = new RegisterBlockedMatMul(accelerator);
        matmul.MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
        await accelerator.SynchronizeAsync();
        var gpuC = await cBuf.CopyToHostAsync<float>(0, M * N);

        float maxErr = 0;
        for (int i = 0; i < M * N; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(gpuC[i] - expectedC[i]));

        // Float accumulation over 128 elements — allow slightly wider tolerance
        if (maxErr > 0.05f)
            throw new Exception($"RegisterBlockedMatMul 128x128: maxErr={maxErr:F4}");

        Console.WriteLine($"[RegisterBlockedMatMul] 128x128 tiled: maxErr={maxErr:E3} — correct");
    });

    [TestMethod]
    public async Task RegisterBlockedMatMul_NonSquare_MatchesCPU() => await RunTest(async accelerator =>
    {
        // Non-square matrix: 96×256 × 256×64 — tests edge cases where tiles don't divide evenly
        int M = 96, K = 256, N = 64;
        var rng = new Random(123);
        var A = new float[M * K];
        var B = new float[K * N];
        for (int i = 0; i < A.Length; i++) A[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < B.Length; i++) B[i] = (float)(rng.NextDouble() * 2 - 1);

        var expectedC = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float sum = 0;
                for (int k = 0; k < K; k++)
                    sum += A[i * K + k] * B[k * N + j];
                expectedC[i * N + j] = sum;
            }

        using var aBuf = accelerator.Allocate1D(A);
        using var bBuf = accelerator.Allocate1D(B);
        using var cBuf = accelerator.Allocate1D<float>(M * N);

        var matmul = new RegisterBlockedMatMul(accelerator);
        matmul.MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
        await accelerator.SynchronizeAsync();
        var gpuC = await cBuf.CopyToHostAsync<float>(0, M * N);

        float maxErr = 0;
        for (int i = 0; i < M * N; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(gpuC[i] - expectedC[i]));

        if (maxErr > 0.1f)
            throw new Exception($"RegisterBlockedMatMul 96x64 (non-square): maxErr={maxErr:F4}");

        Console.WriteLine($"[RegisterBlockedMatMul] 96×256×64 non-square: maxErr={maxErr:E3} — correct");
    });
}
