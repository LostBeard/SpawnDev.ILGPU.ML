namespace SpawnDev.ILGPU.ML.Tests;

public class LayerNormKernelTests : KernelTestBase
{
    private readonly LayerNormKernel _layerNorm;

    public LayerNormKernelTests(AcceleratorFixture fixture) : base(fixture)
    {
        _layerNorm = new LayerNormKernel(Accelerator);
    }

    [Fact]
    public void LayerNorm_UnitGammZeroBeta_Normalizes()
    {
        int rows = 1, C = 4;
        var input = new float[] { 1f, 2f, 3f, 4f };
        var gamma = new float[] { 1f, 1f, 1f, 1f };
        var beta = new float[] { 0f, 0f, 0f, 0f };
        var expected = CpuLayerNorm(input, gamma, beta, rows, C);

        using var inBuf = Accelerator.Allocate1D(input);
        using var outBuf = Accelerator.Allocate1D<float>(rows * C);
        using var gBuf = Accelerator.Allocate1D(gamma);
        using var bBuf = Accelerator.Allocate1D(beta);

        _layerNorm.Forward(inBuf.View, outBuf.View, gBuf.View, bBuf.View, rows, C);
        Accelerator.Synchronize();

        var actual = outBuf.GetAsArray1D();
        AssertClose(expected, actual, 1e-5f, "Unit LN: ");

        // Verify mean ≈ 0 and std ≈ 1
        float mean = actual.Average();
        Assert.True(MathF.Abs(mean) < 1e-5f, $"Mean should be ~0, got {mean}");
    }

    [Theory]
    [InlineData(1, 384, "Single row DAv3")]
    [InlineData(1370, 384, "Full DAv3 T_FULL")]
    [InlineData(1369, 384, "DAv3 patches only")]
    [InlineData(1370, 768, "DPT head concat")]
    public void LayerNorm_Dav3Dimensions_MatchesCpu(int rows, int C, string label)
    {
        var input = RandomFloats(rows * C, seed: 42);
        var gamma = RandomFloats(C, seed: 43, scale: 2f);
        var beta = RandomFloats(C, seed: 44, scale: 0.1f);
        // Make gamma positive (typical for trained models)
        for (int i = 0; i < C; i++) gamma[i] = MathF.Abs(gamma[i]) + 0.1f;

        var expected = CpuLayerNorm(input, gamma, beta, rows, C);

        using var inBuf = Accelerator.Allocate1D(input);
        using var outBuf = Accelerator.Allocate1D<float>(rows * C);
        using var gBuf = Accelerator.Allocate1D(gamma);
        using var bBuf = Accelerator.Allocate1D(beta);

        _layerNorm.Forward(inBuf.View, outBuf.View, gBuf.View, bBuf.View, rows, C);
        Accelerator.Synchronize();

        var actual = outBuf.GetAsArray1D();
        AssertClose(expected, actual, 1e-4f, $"{label}: ");
    }
}
