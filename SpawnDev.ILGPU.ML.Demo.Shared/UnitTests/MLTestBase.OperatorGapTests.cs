using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Tests for operators that lacked dedicated test coverage.
/// Each test verifies operator correctness against CPU reference.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task ArgMax_Axis1_MatchesCpu() => await RunTest(async accelerator =>
    {
        // [2, 4] → argmax along axis 1 → [2]
        var data = new float[] { 1, 5, 3, 2, 4, 2, 6, 1 };
        var expected = new float[] { 1, 2 }; // indices of max: 5 at idx 1, 6 at idx 2

        using var inBuf = accelerator.Allocate1D(data);
        using var outBuf = accelerator.Allocate1D<float>(2);
        new ElementWiseKernels(accelerator).ArgMax(inBuf.View, outBuf.View, 2, 4, 1);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 2);

        for (int i = 0; i < 2; i++)
            if (MathF.Abs(result[i] - expected[i]) > 0.01f)
                throw new Exception($"ArgMax[{i}]: got {result[i]}, expected {expected[i]}");
        Console.WriteLine("[ArgMax] PASS");
    });

    [TestMethod]
    public async Task LeakyRelu_NegativeSlope_MatchesCpu() => await RunTest(async accelerator =>
    {
        var data = new float[] { -2, -1, 0, 1, 2 };
        float alpha = 0.01f;
        var expected = new float[] { -0.02f, -0.01f, 0, 1, 2 };

        using var inBuf = accelerator.Allocate1D(data);
        using var outBuf = accelerator.Allocate1D<float>(5);
        var reg = new OperatorRegistry(accelerator);
        var op = reg.Resolve("LeakyRelu");
        op.Execute(new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 5 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 5 }) },
            Attributes = new Dictionary<string, object> { ["alpha"] = (double)alpha },
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "x" },
        });
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 5);

        float maxErr = 0;
        for (int i = 0; i < 5; i++) maxErr = MathF.Max(maxErr, MathF.Abs(result[i] - expected[i]));
        if (maxErr > 0.01f) throw new Exception($"LeakyRelu maxErr={maxErr}");
        Console.WriteLine("[LeakyRelu] PASS");
    });

    [TestMethod]
    public async Task Ceil_MatchesCpu() => await RunTest(async accelerator =>
    {
        var data = new float[] { 1.1f, -1.1f, 0.5f, -0.5f, 2.0f };
        var expected = new float[] { 2, -1, 1, 0, 2 };

        using var inBuf = accelerator.Allocate1D(data);
        using var outBuf = accelerator.Allocate1D<float>(5);
        new ElementWiseKernels(accelerator).Ceil(inBuf.View, outBuf.View, 5);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 5);

        for (int i = 0; i < 5; i++)
            if (result[i] != expected[i])
                throw new Exception($"Ceil[{i}]: got {result[i]}, expected {expected[i]}");
        Console.WriteLine("[Ceil] PASS");
    });

    [TestMethod]
    public async Task Reciprocal_MatchesCpu() => await RunTest(async accelerator =>
    {
        var data = new float[] { 2, 4, 0.5f, -1, 10 };
        var expected = new float[] { 0.5f, 0.25f, 2f, -1f, 0.1f };

        using var inBuf = accelerator.Allocate1D(data);
        using var outBuf = accelerator.Allocate1D<float>(5);
        new ElementWiseKernels(accelerator).Reciprocal(inBuf.View, outBuf.View, 5);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 5);

        float maxErr = 0;
        for (int i = 0; i < 5; i++) maxErr = MathF.Max(maxErr, MathF.Abs(result[i] - expected[i]));
        if (maxErr > 0.001f) throw new Exception($"Reciprocal maxErr={maxErr}");
        Console.WriteLine("[Reciprocal] PASS");
    });

    [TestMethod]
    public async Task ReduceMin_MatchesCpu() => await RunTest(async accelerator =>
    {
        // [2, 3] → reducemin axis=1 → [2]
        var data = new float[] { 3, 1, 2, 6, 4, 5 };

        using var inBuf = accelerator.Allocate1D(data);
        using var outBuf = accelerator.Allocate1D<float>(2);
        var reg = new OperatorRegistry(accelerator);
        var op = reg.Resolve("ReduceMin");
        op.Execute(new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 2, 3 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 2 }) },
            Attributes = new Dictionary<string, object> { ["axes"] = new long[] { 1 } },
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "x" },
        });
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 2);

        if (result[0] != 1f) throw new Exception($"ReduceMin[0]={result[0]}, expected 1");
        if (result[1] != 4f) throw new Exception($"ReduceMin[1]={result[1]}, expected 4");
        Console.WriteLine("[ReduceMin] PASS");
    });

    [TestMethod]
    public async Task Greater_Broadcast_MatchesCpu() => await RunTest(async accelerator =>
    {
        var a = new float[] { 1, 5, 3, 2 };
        var b = new float[] { 3, 3, 3, 3 };
        var expected = new float[] { 0, 1, 0, 0 }; // 1>3=F, 5>3=T, 3>3=F, 2>3=F

        using var aBuf = accelerator.Allocate1D(a);
        using var bBuf = accelerator.Allocate1D(b);
        using var outBuf = accelerator.Allocate1D<float>(4);
        new ElementWiseKernels(accelerator).Greater(aBuf.View, bBuf.View, outBuf.View, 4);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 4);

        for (int i = 0; i < 4; i++)
            if (result[i] != expected[i])
                throw new Exception($"Greater[{i}]: got {result[i]}, expected {expected[i]}");
        Console.WriteLine("[Greater] PASS");
    });

    [TestMethod]
    public async Task Less_MatchesCpu() => await RunTest(async accelerator =>
    {
        var a = new float[] { 1, 5, 3, 2 };
        var b = new float[] { 3, 3, 3, 3 };
        var expected = new float[] { 1, 0, 0, 1 }; // 1<3=T, 5<3=F, 3<3=F, 2<3=T

        using var aBuf = accelerator.Allocate1D(a);
        using var bBuf = accelerator.Allocate1D(b);
        using var outBuf = accelerator.Allocate1D<float>(4);
        new ElementWiseKernels(accelerator).Less(aBuf.View, bBuf.View, outBuf.View, 4);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 4);

        for (int i = 0; i < 4; i++)
            if (result[i] != expected[i])
                throw new Exception($"Less[{i}]: got {result[i]}, expected {expected[i]}");
        Console.WriteLine("[Less] PASS");
    });
}
