namespace SpawnDev.ILGPU.ML.Tests;

/// <summary>
/// MatMul kernel tests — verifies GPU output matches CPU reference.
/// Covers the exact dimensions used by DAv3 transformer blocks.
/// </summary>
public class MatMulKernelTests : KernelTestBase
{
    private readonly MatMulKernel _matMul;

    public MatMulKernelTests(AcceleratorFixture fixture) : base(fixture)
    {
        _matMul = new MatMulKernel(Accelerator);
    }

    [Fact]
    public void MatMul_SmallIdentity_ReturnsInput()
    {
        int M = 4, K = 4, N = 4;
        var A = RandomFloats(M * K, seed: 1);
        var B = new float[K * N];
        for (int i = 0; i < K; i++) B[i * N + i] = 1f; // Identity

        var expected = (float[])A.Clone();

        using var aBuf = Accelerator.Allocate1D(A);
        using var bBuf = Accelerator.Allocate1D(B);
        using var cBuf = Accelerator.Allocate1D<float>(M * N);

        _matMul.MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
        Accelerator.Synchronize();

        var actual = cBuf.GetAsArray1D();
        AssertClose(expected, actual, 1e-5f, "Identity MatMul: ");
    }

    [Fact]
    public void MatMul_SmallRandom_MatchesCpu()
    {
        int M = 16, K = 16, N = 16;
        var A = RandomFloats(M * K, seed: 10);
        var B = RandomFloats(K * N, seed: 20);
        var expected = CpuMatMul(A, B, M, K, N);

        using var aBuf = Accelerator.Allocate1D(A);
        using var bBuf = Accelerator.Allocate1D(B);
        using var cBuf = Accelerator.Allocate1D<float>(M * N);

        _matMul.MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
        Accelerator.Synchronize();

        var actual = cBuf.GetAsArray1D();
        AssertClose(expected, actual, 1e-3f, "Small random MatMul: ");
    }

    [Theory]
    [InlineData(1370, 384, 1152, "QKV projection")]
    [InlineData(1370, 384, 384, "Attention projection")]
    [InlineData(1370, 384, 1536, "MLP fc1")]
    [InlineData(1370, 1536, 384, "MLP fc2")]
    public void MatMul_Dav3Dimensions_MatchesCpu(int M, int K, int N, string label)
    {
        var A = RandomFloats(M * K, seed: 100);
        var B = RandomFloats(K * N, seed: 200);
        var expected = CpuMatMul(A, B, M, K, N);

        using var aBuf = Accelerator.Allocate1D(A);
        using var bBuf = Accelerator.Allocate1D(B);
        using var cBuf = Accelerator.Allocate1D<float>(M * N);

        _matMul.MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
        Accelerator.Synchronize();

        var actual = cBuf.GetAsArray1D();
        // MatMul accumulates K products — larger K means more FP32 rounding
        float tolerance = K * 2e-6f; // scale tolerance with K
        AssertClose(expected, actual, tolerance, $"{label} [{M}×{K}]×[{K}×{N}]: ");
    }

    [Fact]
    public void BatchedMatMul_AttentionScores_MatchesCpu()
    {
        // Q×K^T: [6, 1370, 64] × [6, 64, 1370] → [6, 1370, 1370]
        int batch = 6, M = 1370, K = 64, N = 1370;
        var A = RandomFloats(batch * M * K, seed: 300);
        var B = RandomFloats(batch * K * N, seed: 400);

        // CPU reference: per-batch matmul
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

        using var aBuf = Accelerator.Allocate1D(A);
        using var bBuf = Accelerator.Allocate1D(B);
        using var cBuf = Accelerator.Allocate1D<float>(batch * M * N);

        _matMul.BatchedMatMul(aBuf.View, bBuf.View, cBuf.View, batch, M, K, N);
        Accelerator.Synchronize();

        var actual = cBuf.GetAsArray1D();
        AssertClose(expected, actual, K * 2e-6f, "Attention scores: ");
    }

    [Fact]
    public void MatMul_NonTileAligned_MatchesCpu()
    {
        // Test with dimensions that don't align to tile size (16)
        int M = 17, K = 33, N = 15;
        var A = RandomFloats(M * K, seed: 500);
        var B = RandomFloats(K * N, seed: 600);
        var expected = CpuMatMul(A, B, M, K, N);

        using var aBuf = Accelerator.Allocate1D(A);
        using var bBuf = Accelerator.Allocate1D(B);
        using var cBuf = Accelerator.Allocate1D<float>(M * N);

        _matMul.MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
        Accelerator.Synchronize();

        var actual = cBuf.GetAsArray1D();
        AssertClose(expected, actual, K * 2e-6f, "Non-aligned MatMul: ");
    }
}
