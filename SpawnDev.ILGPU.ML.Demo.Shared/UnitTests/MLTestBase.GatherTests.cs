using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task GatherAxis0_EmbeddingLookup() => await RunTest(async accelerator =>
    {
        // Simulate embedding lookup: vocab [100, 64], indices [5] → output [5, 64]
        int vocabSize = 100, embedDim = 64, numIndices = 5;
        var data = RandomFloats(vocabSize * embedDim, seed: 160);
        var indices = new int[] { 3, 42, 0, 99, 17 };

        var expected = new float[numIndices * embedDim];
        for (int i = 0; i < numIndices; i++)
            Array.Copy(data, indices[i] * embedDim, expected, i * embedDim, embedDim);

        using var dataBuf = accelerator.Allocate1D(data);
        using var idxBuf = accelerator.Allocate1D(indices);
        using var outBuf = accelerator.Allocate1D<float>(numIndices * embedDim);

        var gather = new GatherKernel(accelerator);
        gather.GatherAxis0(dataBuf.View, idxBuf.View, outBuf.View, numIndices, embedDim);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, numIndices * embedDim), expected, 0f, "Gather embedding: ");
    });
}
