using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Concat axis=1 (channel concat): merge two NCHW tensors along C dimension.
    /// This is the pattern SqueezeNet's fire modules use.
    /// </summary>
    [TestMethod]
    public async Task Concat_Axis1_ChannelMerge() => await RunTest(async accelerator =>
    {
        int N = 1, C1 = 4, C2 = 8, H = 2, W = 2;
        int spatial = H * W;

        var data1 = new float[N * C1 * spatial];
        var data2 = new float[N * C2 * spatial];
        for (int i = 0; i < data1.Length; i++) data1[i] = i + 1;
        for (int i = 0; i < data2.Length; i++) data2[i] = 100 + i + 1;

        int outC = C1 + C2;
        var expected = new float[N * outC * spatial];
        for (int n = 0; n < N; n++)
        {
            for (int c = 0; c < C1; c++)
                for (int s = 0; s < spatial; s++)
                    expected[n * outC * spatial + c * spatial + s] = data1[n * C1 * spatial + c * spatial + s];
            for (int c = 0; c < C2; c++)
                for (int s = 0; s < spatial; s++)
                    expected[n * outC * spatial + (C1 + c) * spatial + s] = data2[n * C2 * spatial + c * spatial + s];
        }

        using var buf1 = accelerator.Allocate1D(data1);
        using var buf2 = accelerator.Allocate1D(data2);
        using var outBuf = accelerator.Allocate1D<float>(N * outC * spatial);

        var pool = new BufferPool(accelerator);
        var t1 = new Tensor(buf1.View, new[] { N, C1, H, W });
        var t2 = new Tensor(buf2.View, new[] { N, C2, H, W });
        var tOut = new Tensor(outBuf.View, new[] { N, outC, H, W });

        var reg = new OperatorRegistry(accelerator);
        var concat = new ConcatOperator(reg);
        concat.Execute(new OnnxOpContext
        {
            Inputs = new[] { t1, t2 },
            Outputs = new[] { tOut },
            Attributes = new Dictionary<string, object> { ["axis"] = 1L },
            Pool = pool,
            InputNames = new[] { "a", "b" },
            ConstantValues = new Dictionary<string, float[]>(),
        });
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, N * outC * spatial), expected, 0f, "Concat axis=1: ");
    });
}
