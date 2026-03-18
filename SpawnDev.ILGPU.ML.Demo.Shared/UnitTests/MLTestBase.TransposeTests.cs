using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task Transpose_2DSwap() => await RunTest(async accelerator =>
    {
        // [3, 4] → [4, 3] with perm=[1, 0]
        int[] shape = { 3, 4 };
        int[] perm = { 1, 0 };
        var input = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        var expected = new float[] { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(12);
        var t = new TransposeKernel(accelerator);
        t.Transpose(inBuf.View, outBuf.View, shape, perm);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, 12);
        AssertClose(expected, actual, 0f, "Transpose 2D: ");
    });

    [TestMethod]
    public async Task Transpose_3DNCHW_to_NHWC() => await RunTest(async accelerator =>
    {
        // [2, 3, 4] perm=[0, 2, 1] → [2, 4, 3] (like NCHW→NHWC for C=3, HW=4)
        int[] shape = { 2, 3, 4 };
        int[] perm = { 0, 2, 1 };
        var input = RandomFloats(24, seed: 150);

        // CPU reference
        int[] outShape = { 2, 4, 3 };
        var expected = new float[24];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 4; k++)
                    expected[i * 12 + k * 3 + j] = input[i * 12 + j * 4 + k];

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(24);
        var t = new TransposeKernel(accelerator);
        t.Transpose(inBuf.View, outBuf.View, shape, perm);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, 24);
        AssertClose(expected, actual, 0f, "Transpose 3D: ");
    });

    [TestMethod]
    public async Task Transpose_4D_NCHW_to_NHWC() => await RunTest(async accelerator =>
    {
        // [1, 3, 2, 2] perm=[0, 2, 3, 1] → [1, 2, 2, 3]
        int[] shape = { 1, 3, 2, 2 };
        int[] perm = { 0, 2, 3, 1 };
        var input = RandomFloats(12, seed: 151);

        // CPU reference
        var expected = new float[12];
        for (int n = 0; n < 1; n++)
            for (int c = 0; c < 3; c++)
                for (int h = 0; h < 2; h++)
                    for (int w = 0; w < 2; w++)
                    {
                        int srcIdx = n * 12 + c * 4 + h * 2 + w;
                        int dstIdx = n * 12 + h * 6 + w * 3 + c;
                        expected[dstIdx] = input[srcIdx];
                    }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(12);
        var t = new TransposeKernel(accelerator);
        t.Transpose(inBuf.View, outBuf.View, shape, perm);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, 12);
        AssertClose(expected, actual, 0f, "Transpose 4D NCHW→NHWC: ");
    });
}
