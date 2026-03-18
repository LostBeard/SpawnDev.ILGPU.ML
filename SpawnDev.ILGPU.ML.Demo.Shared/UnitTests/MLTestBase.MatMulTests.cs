using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task MatMul_QkvDimensions() => await RunTest(async accelerator =>
    {
        int M = 1370, K = 384, N = 1152;
        var A = RandomFloats(M * K, seed: 100);
        var B = RandomFloats(K * N, seed: 200);
        var expected = CpuMatMul(A, B, M, K, N);

        using var aBuf = accelerator.Allocate1D(A);
        using var bBuf = accelerator.Allocate1D(B);
        using var cBuf = accelerator.Allocate1D<float>(M * N);

        var matMul = new MatMulKernel(accelerator);
        matMul.MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
        await accelerator.SynchronizeAsync();

        var actual = await cBuf.CopyToHostAsync<float>(0, M * N);
        AssertClose(expected, actual, K * 2e-6f, $"QKV MatMul [{M}x{K}]x[{K}x{N}]: ");
    });

    [TestMethod]
    public async Task MatMul_MlpFc2Dimensions() => await RunTest(async accelerator =>
    {
        int M = 1370, K = 1536, N = 384;
        var A = RandomFloats(M * K, seed: 300);
        var B = RandomFloats(K * N, seed: 400);
        var expected = CpuMatMul(A, B, M, K, N);

        using var aBuf = accelerator.Allocate1D(A);
        using var bBuf = accelerator.Allocate1D(B);
        using var cBuf = accelerator.Allocate1D<float>(M * N);

        var matMul = new MatMulKernel(accelerator);
        matMul.MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
        await accelerator.SynchronizeAsync();

        var actual = await cBuf.CopyToHostAsync<float>(0, M * N);
        AssertClose(expected, actual, K * 2e-6f, $"MLP fc2 [{M}x{K}]x[{K}x{N}]: ");
    });

    [TestMethod]
    public async Task MatMul_BatchedAttentionScores() => await RunTest(async accelerator =>
    {
        int batch = 6, M = 1370, K = 64, N = 1370;
        var A = RandomFloats(batch * M * K, seed: 500);
        var B = RandomFloats(batch * K * N, seed: 600);

        var expected = new float[batch * M * N];
        for (int b = 0; b < batch; b++)
        {
            var aSlice = new float[M * K];
            var bSlice = new float[K * N];
            Array.Copy(A, b * M * K, aSlice, 0, M * K);
            Array.Copy(B, b * K * N, bSlice, 0, K * N);
            var cSlice = CpuMatMul(aSlice, bSlice, M, K, N);
            Array.Copy(cSlice, 0, expected, b * M * N, M * N);
        }

        using var aBuf = accelerator.Allocate1D(A);
        using var bBuf = accelerator.Allocate1D(B);
        using var cBuf = accelerator.Allocate1D<float>(batch * M * N);

        var matMul = new MatMulKernel(accelerator);
        matMul.BatchedMatMul(aBuf.View, bBuf.View, cBuf.View, batch, M, K, N);
        await accelerator.SynchronizeAsync();

        var actual = await cBuf.CopyToHostAsync<float>(0, batch * M * N);
        AssertClose(expected, actual, K * 2e-6f, "Attention scores: ");
    });

    [TestMethod]
    public async Task MatMul_SmallNonAligned() => await RunTest(async accelerator =>
    {
        int M = 17, K = 33, N = 15;
        var A = RandomFloats(M * K, seed: 700);
        var B = RandomFloats(K * N, seed: 800);
        var expected = CpuMatMul(A, B, M, K, N);

        using var aBuf = accelerator.Allocate1D(A);
        using var bBuf = accelerator.Allocate1D(B);
        using var cBuf = accelerator.Allocate1D<float>(M * N);

        var matMul = new MatMulKernel(accelerator);
        matMul.MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
        await accelerator.SynchronizeAsync();

        var actual = await cBuf.CopyToHostAsync<float>(0, M * N);
        AssertClose(expected, actual, K * 2e-6f, "Non-aligned MatMul: ");
    });
}
