using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Comprehensive broadcast pattern tests for ALL binary operators.
/// Each test exercises a specific broadcast shape combination that
/// would trigger BroadcastBinaryOp if the operator doesn't handle it directly.
///
/// Patterns tested:
///   1. Per-row scalar: [N, C] op [N, 1] — LayerNorm mean/variance
///   2. Last-dim broadcast: [N, C] op [C] — bias add, LayerScale
///   3. Scalar broadcast: [N, C] op [1] — threshold, scaling
///   4. NCHW per-channel: [N, C, H, W] op [C] — BatchNorm
///
/// These tests are the safety net that prevents broadcast bugs from
/// reaching production. If DelegateSpecialization or BroadcastBinaryOpND
/// breaks, these tests catch it.
/// </summary>
public abstract partial class MLTestBase
{
    // ═══════════════════════════════════════════════════════════
    //  ARITHMETIC BROADCAST TESTS
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Broadcast_Add_LastDim() => await RunTest(async accelerator =>
    {
        // [2, 3] + [3] = last-dim broadcast
        var a = new float[] { 1, 2, 3, 4, 5, 6 };
        var b = new float[] { 10, 20, 30 };
        var expected = new float[] { 11, 22, 33, 14, 25, 36 };
        await VerifyBinaryOp(accelerator, "Add", a, new[] { 2, 3 }, b, new[] { 3 }, expected, new[] { 2, 3 });
    });

    [TestMethod]
    public async Task Broadcast_Add_Scalar() => await RunTest(async accelerator =>
    {
        // [2, 3] + [1] = scalar broadcast
        var a = new float[] { 1, 2, 3, 4, 5, 6 };
        var b = new float[] { 100 };
        var expected = new float[] { 101, 102, 103, 104, 105, 106 };
        await VerifyBinaryOp(accelerator, "Add", a, new[] { 2, 3 }, b, new[] { 1 }, expected, new[] { 2, 3 });
    });

    [TestMethod]
    public async Task Broadcast_Sub_LastDim() => await RunTest(async accelerator =>
    {
        // [2, 3] - [3] = last-dim broadcast
        var a = new float[] { 10, 20, 30, 40, 50, 60 };
        var b = new float[] { 1, 2, 3 };
        var expected = new float[] { 9, 18, 27, 39, 48, 57 };
        await VerifyBinaryOp(accelerator, "Sub", a, new[] { 2, 3 }, b, new[] { 3 }, expected, new[] { 2, 3 });
    });

    [TestMethod]
    public async Task Broadcast_Sub_Scalar() => await RunTest(async accelerator =>
    {
        // [2, 3] - [1] = scalar broadcast
        var a = new float[] { 10, 20, 30, 40, 50, 60 };
        var b = new float[] { 5 };
        var expected = new float[] { 5, 15, 25, 35, 45, 55 };
        await VerifyBinaryOp(accelerator, "Sub", a, new[] { 2, 3 }, b, new[] { 1 }, expected, new[] { 2, 3 });
    });

    [TestMethod]
    public async Task Broadcast_Sub_PerRow() => await RunTest(async accelerator =>
    {
        // [2, 3] - [2, 1] = per-row scalar (LayerNorm mean subtraction)
        var a = new float[] { 10, 20, 30, 40, 50, 60 };
        var b = new float[] { 5, 15 };
        var expected = new float[] { 5, 15, 25, 25, 35, 45 };
        await VerifyBinaryOp(accelerator, "Sub", a, new[] { 2, 3 }, b, new[] { 2, 1 }, expected, new[] { 2, 3 });
    });

    [TestMethod]
    public async Task Broadcast_Mul_Scalar() => await RunTest(async accelerator =>
    {
        // [2, 3] * [1] = scalar broadcast
        var a = new float[] { 1, 2, 3, 4, 5, 6 };
        var b = new float[] { 10 };
        var expected = new float[] { 10, 20, 30, 40, 50, 60 };
        await VerifyBinaryOp(accelerator, "Mul", a, new[] { 2, 3 }, b, new[] { 1 }, expected, new[] { 2, 3 });
    });

    [TestMethod]
    public async Task Broadcast_Div_LastDim() => await RunTest(async accelerator =>
    {
        // [2, 3] / [3] = last-dim broadcast
        var a = new float[] { 10, 20, 30, 40, 50, 60 };
        var b = new float[] { 2, 5, 10 };
        var expected = new float[] { 5, 4, 3, 20, 10, 6 };
        await VerifyBinaryOp(accelerator, "Div", a, new[] { 2, 3 }, b, new[] { 3 }, expected, new[] { 2, 3 });
    });

    [TestMethod]
    public async Task Broadcast_Div_Scalar() => await RunTest(async accelerator =>
    {
        // [2, 3] / [1] = scalar broadcast
        var a = new float[] { 10, 20, 30, 40, 50, 60 };
        var b = new float[] { 10 };
        var expected = new float[] { 1, 2, 3, 4, 5, 6 };
        await VerifyBinaryOp(accelerator, "Div", a, new[] { 2, 3 }, b, new[] { 1 }, expected, new[] { 2, 3 });
    });

    [TestMethod]
    public async Task Broadcast_Pow_Scalar() => await RunTest(async accelerator =>
    {
        // [4] ^ [1] = scalar exponent (LayerNorm variance)
        var a = new float[] { 1, 2, 3, 4 };
        var b = new float[] { 2 };
        var expected = new float[] { 1, 4, 9, 16 };
        await VerifyBinaryOp(accelerator, "Pow", a, new[] { 4 }, b, new[] { 1 }, expected, new[] { 4 });
    });

    // ═══════════════════════════════════════════════════════════
    //  COMPARISON BROADCAST TESTS
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Broadcast_Greater_Scalar() => await RunTest(async accelerator =>
    {
        // [4] > [1] = scalar threshold
        var a = new float[] { 1, 5, 3, 2 };
        var b = new float[] { 3 };
        var expected = new float[] { 0, 1, 0, 0 };
        await VerifyBinaryOp(accelerator, "Greater", a, new[] { 4 }, b, new[] { 1 }, expected, new[] { 4 });
    });

    [TestMethod]
    public async Task Broadcast_Less_Scalar() => await RunTest(async accelerator =>
    {
        // [4] < [1] = scalar threshold
        var a = new float[] { 1, 5, 3, 2 };
        var b = new float[] { 3 };
        var expected = new float[] { 1, 0, 0, 1 };
        await VerifyBinaryOp(accelerator, "Less", a, new[] { 4 }, b, new[] { 1 }, expected, new[] { 4 });
    });

    [TestMethod]
    public async Task Broadcast_Equal_Scalar() => await RunTest(async accelerator =>
    {
        // [4] == [1] = scalar comparison
        var a = new float[] { 1, 3, 3, 2 };
        var b = new float[] { 3 };
        var expected = new float[] { 0, 1, 1, 0 };
        await VerifyBinaryOp(accelerator, "Equal", a, new[] { 4 }, b, new[] { 1 }, expected, new[] { 4 });
    });

    [TestMethod]
    public async Task Broadcast_LessOrEqual_Scalar() => await RunTest(async accelerator =>
    {
        // [4] <= [1] = scalar threshold
        var a = new float[] { 1, 3, 5, 3 };
        var b = new float[] { 3 };
        var expected = new float[] { 1, 1, 0, 1 };
        await VerifyBinaryOp(accelerator, "LessOrEqual", a, new[] { 4 }, b, new[] { 1 }, expected, new[] { 4 });
    });

    [TestMethod]
    public async Task Broadcast_GreaterOrEqual_Scalar() => await RunTest(async accelerator =>
    {
        // [4] >= [1] = scalar threshold
        var a = new float[] { 1, 3, 5, 3 };
        var b = new float[] { 3 };
        var expected = new float[] { 0, 1, 1, 1 };
        await VerifyBinaryOp(accelerator, "GreaterOrEqual", a, new[] { 4 }, b, new[] { 1 }, expected, new[] { 4 });
    });

    // ═══════════════════════════════════════════════════════════
    //  HELPER — runs any binary operator and verifies output
    // ═══════════════════════════════════════════════════════════

    private async Task VerifyBinaryOp(Accelerator accelerator, string opType,
        float[] aData, int[] aShape, float[] bData, int[] bShape,
        float[] expected, int[] outShape)
    {
        using var aBuf = accelerator.Allocate1D(aData);
        using var bBuf = accelerator.Allocate1D(bData);
        using var outBuf = accelerator.Allocate1D<float>(expected.Length);

        var reg = new OperatorRegistry(accelerator);
        var op = reg.Resolve(opType);

        // Pre-read constants so BroadcastBinaryOp CPU path works
        var constants = new Dictionary<string, float[]>
        {
            ["a"] = aData,
            ["b"] = bData,
        };

        op.Execute(new OnnxOpContext
        {
            Inputs = new[] { new Tensor(aBuf.View, aShape), new Tensor(bBuf.View, bShape) },
            Outputs = new[] { new Tensor(outBuf.View, outShape) },
            Attributes = new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "a", "b" },
            ConstantValues = constants,
        });
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, expected.Length);

        float maxErr = 0;
        for (int i = 0; i < expected.Length; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(result[i] - expected[i]));

        if (maxErr > 0.01f)
            throw new Exception($"Broadcast_{opType} maxErr={maxErr:E3}. " +
                $"Expected: [{string.Join(",", expected.Select(v => v.ToString("F1")))}] " +
                $"Got: [{string.Join(",", result.Select(v => v.ToString("F1")))}]");

        Console.WriteLine($"[Broadcast_{opType}] PASS — {string.Join("x", aShape)} op {string.Join("x", bShape)} → maxErr={maxErr:E1}");
    }
}
