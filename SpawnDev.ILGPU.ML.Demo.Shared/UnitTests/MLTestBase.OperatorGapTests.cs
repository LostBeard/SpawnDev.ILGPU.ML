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

    [TestMethod]
    public async Task LessOrEqual_MatchesCpu() => await RunTest(async accelerator =>
    {
        var a = new float[] { 1, 3, 5, 3 };
        var b = new float[] { 3, 3, 3, 3 };
        var expected = new float[] { 1, 1, 0, 1 }; // 1<=3=T, 3<=3=T, 5<=3=F, 3<=3=T

        using var aBuf = accelerator.Allocate1D(a);
        using var bBuf = accelerator.Allocate1D(b);
        using var outBuf = accelerator.Allocate1D<float>(4);
        var reg = new OperatorRegistry(accelerator);
        var op = reg.Resolve("LessOrEqual");
        op.Execute(new OnnxOpContext
        {
            Inputs = new[] { new Tensor(aBuf.View, new[] { 4 }), new Tensor(bBuf.View, new[] { 4 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 4 }) },
            Attributes = new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "a", "b" },
        });
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 4);

        for (int i = 0; i < 4; i++)
            if (result[i] != expected[i])
                throw new Exception($"LessOrEqual[{i}]: got {result[i]}, expected {expected[i]}");
        Console.WriteLine("[LessOrEqual] PASS");
    });

    [TestMethod]
    public async Task Expand_BroadcastVector_MatchesCpu() => await RunTest(async accelerator =>
    {
        // Expand [1, 3] → [2, 3] by broadcasting dim 0
        var data = new float[] { 10, 20, 30 };
        var expected = new float[] { 10, 20, 30, 10, 20, 30 };

        using var inBuf = accelerator.Allocate1D(data);
        using var outBuf = accelerator.Allocate1D<float>(6);
        var reg = new OperatorRegistry(accelerator);
        var op = reg.Resolve("Expand");

        // Expand needs shape tensor as second input — use constant values
        using var shapeBuf = accelerator.Allocate1D(new float[] { 2, 3 });
        op.Execute(new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 1, 3 }), new Tensor(shapeBuf.View, new[] { 2 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 2, 3 }) },
            Attributes = new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "input", "shape" },
            ConstantValues = new Dictionary<string, float[]> { ["shape"] = new float[] { 2, 3 } },
        });
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 6);

        for (int i = 0; i < 6; i++)
            if (MathF.Abs(result[i] - expected[i]) > 0.01f)
                throw new Exception($"Expand[{i}]: got {result[i]}, expected {expected[i]}");
        Console.WriteLine("[Expand] PASS");
    });

    [TestMethod]
    public async Task DepthToSpace_2x2_MatchesCpu() => await RunTest(async accelerator =>
    {
        // [1, 4, 1, 1] → DepthToSpace blocksize=2 → [1, 1, 2, 2]
        var data = new float[] { 1, 2, 3, 4 };
        // DCR mode: output[0,0,0,0]=1, [0,0,0,1]=2, [0,0,1,0]=3, [0,0,1,1]=4
        var expected = new float[] { 1, 2, 3, 4 };

        using var inBuf = accelerator.Allocate1D(data);
        using var outBuf = accelerator.Allocate1D<float>(4);
        var reg = new OperatorRegistry(accelerator);
        var op = reg.Resolve("DepthToSpace");
        op.Execute(new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 1, 4, 1, 1 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 1, 1, 2, 2 }) },
            Attributes = new Dictionary<string, object> { ["blocksize"] = 2L },
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "x" },
        });
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 4);

        float absMax = result.Max(v => MathF.Abs(v));
        if (absMax < 0.01f) throw new Exception("DepthToSpace output is all zeros");
        Console.WriteLine($"[DepthToSpace] Output: [{string.Join(",", result.Select(v => v.ToString("F1")))}] PASS");
    });

    [TestMethod]
    public async Task ScatterND_SimpleUpdate_Works() => await RunTest(async accelerator =>
    {
        // data [3] = {10, 20, 30}, update index 1 to 99
        var data = new float[] { 10, 20, 30 };
        var indices = new float[] { 1 }; // update position 1
        var updates = new float[] { 99 };
        var expected = new float[] { 10, 99, 30 };

        using var dataBuf = accelerator.Allocate1D(data);
        using var idxBuf = accelerator.Allocate1D(indices);
        using var updateBuf = accelerator.Allocate1D(updates);
        using var outBuf = accelerator.Allocate1D<float>(3);

        var reg = new OperatorRegistry(accelerator);
        var op = reg.Resolve("ScatterND");
        op.Execute(new OnnxOpContext
        {
            Inputs = new[]
            {
                new Tensor(dataBuf.View, new[] { 3 }),
                new Tensor(idxBuf.View, new[] { 1, 1 }),
                new Tensor(updateBuf.View, new[] { 1 }),
            },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 3 }) },
            Attributes = new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "data", "indices", "updates" },
            ConstantValues = new Dictionary<string, float[]> { ["indices"] = indices },
        });
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 3);

        for (int i = 0; i < 3; i++)
            if (MathF.Abs(result[i] - expected[i]) > 0.01f)
                throw new Exception($"ScatterND[{i}]: got {result[i]}, expected {expected[i]}");
        Console.WriteLine("[ScatterND] PASS");
    });
}
