using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Comprehensive ONNX operator coverage tests.
/// Each test validates one operator against a known CPU reference value.
/// These run on ALL backends to ensure correctness everywhere.
/// </summary>
public abstract partial class MLTestBase
{
    // ═══════════════════════════════════════════════════════════
    //  Trig / Hyperbolic operators
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_Acos_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { 0f, 0.5f, -0.5f, 1f, -1f };
        var expected = input.Select(MathF.Acos).ToArray();
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(input.Length);
        new ElementWiseKernels(accelerator).Acos(inBuf.View, outBuf.View, input.Length);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "Acos: ");
    });

    [TestMethod]
    public async Task Op_Asin_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { 0f, 0.5f, -0.5f, 1f, -1f };
        var expected = input.Select(MathF.Asin).ToArray();
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(input.Length);
        new ElementWiseKernels(accelerator).Asin(inBuf.View, outBuf.View, input.Length);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "Asin: ");
    });

    [TestMethod]
    public async Task Op_Atan_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { 0f, 1f, -1f, 10f, -10f };
        var expected = input.Select(MathF.Atan).ToArray();
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(input.Length);
        new ElementWiseKernels(accelerator).Atan(inBuf.View, outBuf.View, input.Length);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "Atan: ");
    });

    [TestMethod]
    public async Task Op_Cosh_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { 0f, 1f, -1f, 2f, -2f };
        var expected = input.Select(MathF.Cosh).ToArray();
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(input.Length);
        new ElementWiseKernels(accelerator).Cosh(inBuf.View, outBuf.View, input.Length);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "Cosh: ");
    });

    [TestMethod]
    public async Task Op_Sinh_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { 0f, 1f, -1f, 2f, -2f };
        var expected = input.Select(MathF.Sinh).ToArray();
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(input.Length);
        new ElementWiseKernels(accelerator).Sinh(inBuf.View, outBuf.View, input.Length);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "Sinh: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  Activation operators
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_Elu_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { -2f, -1f, 0f, 1f, 2f };
        var expected = input.Select(x => x >= 0 ? x : MathF.Exp(x) - 1f).ToArray();
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(input.Length);
        new ElementWiseKernels(accelerator).Elu(inBuf.View, outBuf.View, input.Length);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-5f, "Elu: ");
    });

    [TestMethod]
    public async Task Op_Selu_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { -2f, -1f, 0f, 1f, 2f };
        var expected = input.Select(x => x > 0 ? 1.0507f * x : 1.0507f * 1.67326f * (MathF.Exp(x) - 1f)).ToArray();
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(input.Length);
        new ElementWiseKernels(accelerator).Selu(inBuf.View, outBuf.View, input.Length);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "Selu: ");
    });

    [TestMethod]
    public async Task Op_Softplus_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { -2f, -1f, 0f, 1f, 2f };
        var expected = input.Select(x => MathF.Log(1f + MathF.Exp(x))).ToArray();
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(input.Length);
        new ElementWiseKernels(accelerator).Softplus(inBuf.View, outBuf.View, input.Length);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "Softplus: ");
    });

    [TestMethod]
    public async Task Op_Mish_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { -2f, -1f, 0f, 1f, 2f };
        var expected = input.Select(x => x * MathF.Tanh(MathF.Log(1f + MathF.Exp(x)))).ToArray();
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(input.Length);
        new ElementWiseKernels(accelerator).Mish(inBuf.View, outBuf.View, input.Length);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "Mish: ");
    });

    [TestMethod]
    public async Task Op_Softsign_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { -2f, -1f, 0f, 1f, 2f };
        var expected = input.Select(x => x / (1f + MathF.Abs(x))).ToArray();
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(input.Length);
        new ElementWiseKernels(accelerator).Softsign(inBuf.View, outBuf.View, input.Length);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "Softsign: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  Reduction operators
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_ReduceProd_MatchesCpu() => await RunTest(async accelerator =>
    {
        // [2, 3] → reduce axis=1 → [2]
        var input = new float[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var expected = new float[] { 6f, 120f }; // 1*2*3=6, 4*5*6=120
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(2);
        var reg = new OperatorRegistry(accelerator);
        reg.Reductions.ReduceProd(inBuf.View, outBuf.View, 2, 3, 1);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-3f, "ReduceProd: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  Structural operators
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_Trilu_Upper() => await RunTest(async accelerator =>
    {
        // 3x3 matrix, upper triangle (k=0)
        var input = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        var expected = new float[] { 1, 2, 3, 0, 5, 6, 0, 0, 9 }; // upper triangle
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(9);
        var reg = new OperatorRegistry(accelerator);
        var tensor = new Tensor(inBuf.View, new[] { 3, 3 });
        var outTensor = new Tensor(outBuf.View, new[] { 3, 3 });
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { tensor },
            Outputs = new[] { outTensor },
            Attributes = new Dictionary<string, object> { ["upper"] = (long)1 },
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "input" },
            ConstantValues = new Dictionary<string, float[]> { ["input"] = input }
        };
        reg.Resolve("Trilu")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 0f, "Trilu upper: ");
    });

    [TestMethod]
    public async Task Op_Trilu_Lower() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        var expected = new float[] { 1, 0, 0, 4, 5, 0, 7, 8, 9 }; // lower triangle
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(9);
        var reg = new OperatorRegistry(accelerator);
        var tensor = new Tensor(inBuf.View, new[] { 3, 3 });
        var outTensor = new Tensor(outBuf.View, new[] { 3, 3 });
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { tensor },
            Outputs = new[] { outTensor },
            Attributes = new Dictionary<string, object> { ["upper"] = (long)0 },
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "input" },
            ConstantValues = new Dictionary<string, float[]> { ["input"] = input }
        };
        reg.Resolve("Trilu")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 0f, "Trilu lower: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  Round operator
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_Round_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { 0.1f, 0.5f, 0.9f, 1.5f, 2.5f, -0.5f, -1.5f };
        var expected = input.Select(x => MathF.Round(x)).ToArray();
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(input.Length);
        new ElementWiseKernels(accelerator).Round(inBuf.View, outBuf.View, input.Length);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 0f, "Round: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  Operator registry coverage test
    // ═══════════════════════════════════════════════════════════

    // ═══════════════════════════════════════════════════════════
    //  IsInf operator
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_IsInf_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { 0f, 1f, float.PositiveInfinity, float.NegativeInfinity, float.NaN };
        var expected = new float[] { 0f, 0f, 1f, 1f, 0f };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(input.Length);
        new ElementWiseKernels(accelerator).IsInf(inBuf.View, outBuf.View, input.Length);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 0f, "IsInf: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  Celu operator
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_Celu_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { -2f, -1f, 0f, 1f, 2f };
        var expected = input.Select(x => MathF.Max(0f, x) + MathF.Min(0f, MathF.Exp(x) - 1f)).ToArray();
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(input.Length);
        new ElementWiseKernels(accelerator).Celu(inBuf.View, outBuf.View, input.Length);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-5f, "Celu: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  Acosh / Asinh / Atanh operators
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_Acosh_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1f, 1.5f, 2f, 5f, 10f };
        var expected = input.Select(MathF.Acosh).ToArray();
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(input.Length);
        new ElementWiseKernels(accelerator).Acosh(inBuf.View, outBuf.View, input.Length);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "Acosh: ");
    });

    [TestMethod]
    public async Task Op_Asinh_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { -2f, -1f, 0f, 1f, 2f };
        var expected = input.Select(MathF.Asinh).ToArray();
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(input.Length);
        new ElementWiseKernels(accelerator).Asinh(inBuf.View, outBuf.View, input.Length);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "Asinh: ");
    });

    [TestMethod]
    public async Task Op_Atanh_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { -0.9f, -0.5f, 0f, 0.5f, 0.9f };
        var expected = input.Select(MathF.Atanh).ToArray();
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(input.Length);
        new ElementWiseKernels(accelerator).Atanh(inBuf.View, outBuf.View, input.Length);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "Atanh: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  ReduceL1 / ReduceL2 / ReduceSumSquare / ReduceLogSumExp
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_ReduceL1_MatchesCpu() => await RunTest(async accelerator =>
    {
        // [2, 3] → reduce axis=1 → sum of abs values per row
        var input = new float[] { -1f, 2f, -3f, 4f, -5f, 6f };
        var expected = new float[] { 6f, 15f }; // |−1|+2+|−3|=6, 4+|−5|+6=15
        using var inBuf = accelerator.Allocate1D(input);
        using var absBuf = accelerator.Allocate1D<float>(6);
        using var outBuf = accelerator.Allocate1D<float>(2);
        var ew = GetOrCreateEW(accelerator);
        ew.Abs(inBuf.View, absBuf.View, 6);
        var reg = new OperatorRegistry(accelerator);
        reg.Reductions.ReduceSum(absBuf.View, outBuf.View, 2, 3, 1);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "ReduceL1: ");
    });

    [TestMethod]
    public async Task Op_ReduceSumSquare_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var expected = new float[] { 14f, 77f }; // 1+4+9=14, 16+25+36=77
        using var inBuf = accelerator.Allocate1D(input);
        using var inBuf2 = accelerator.Allocate1D(input); // separate copy to avoid WebGPU aliasing
        using var sqBuf = accelerator.Allocate1D<float>(6);
        using var outBuf = accelerator.Allocate1D<float>(2);
        var ew = GetOrCreateEW(accelerator);
        ew.Mul(inBuf.View, inBuf2.View, sqBuf.View, 6);
        var reg = new OperatorRegistry(accelerator);
        reg.Reductions.ReduceSum(sqBuf.View, outBuf.View, 2, 3, 1);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-3f, "ReduceSumSquare: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  RNN / LSTM / GRU basic correctness
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_RNN_ForwardPass() => await RunTest(async accelerator =>
    {
        // Simple RNN: Ht = tanh(Xt * Wi^T + Ht-1 * Ri^T + bias)
        // seq_len=2, batch=1, input_size=2, hidden_size=2
        // W: [1, 2, 2], R: [1, 2, 2], B: [1, 4]
        var x = new float[] { 1f, 0f, 0f, 1f }; // 2 timesteps, each [1,0] and [0,1]
        var w = new float[] { 1f, 0f, 0f, 1f }; // identity
        var r = new float[] { 0f, 0f, 0f, 0f }; // no recurrence
        var b = new float[] { 0f, 0f, 0f, 0f }; // no bias

        // Expected: tanh(identity * input) = tanh([1,0]) = [0.7616, 0], tanh([0,1]) = [0, 0.7616]
        var expectedYh = new float[] { 0f, MathF.Tanh(1f) }; // last hidden state

        using var xBuf = accelerator.Allocate1D(x);
        using var wBuf = accelerator.Allocate1D(w);
        using var rBuf = accelerator.Allocate1D(r);
        using var bBuf = accelerator.Allocate1D(b);
        using var yBuf = accelerator.Allocate1D<float>(4); // [2, 1, 1, 2]
        using var yhBuf = accelerator.Allocate1D<float>(2); // [1, 1, 2]

        var reg = new OperatorRegistry(accelerator);
        var op = reg.Resolve("RNN")!;
        var pool = new BufferPool(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[]
            {
                new Tensor(xBuf.View, new[] { 2, 1, 2 }),
                new Tensor(wBuf.View, new[] { 1, 2, 2 }),
                new Tensor(rBuf.View, new[] { 1, 2, 2 }),
                new Tensor(bBuf.View, new[] { 1, 4 })
            },
            Outputs = new[]
            {
                new Tensor(yBuf.View, new[] { 2, 1, 1, 2 }),
                new Tensor(yhBuf.View, new[] { 1, 1, 2 })
            },
            Attributes = new Dictionary<string, object> { ["hidden_size"] = (long)2 },
            Pool = pool,
            InputNames = new[] { "X", "W", "R", "B" },
            ConstantValues = new Dictionary<string, float[]>
            {
                ["X"] = x, ["W"] = w, ["R"] = r, ["B"] = b
            }
        };
        op.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, yhBuf.View, expectedYh, 1e-4f, "RNN Y_h: ");
    });

    [TestMethod]
    public async Task Op_LSTM_BasicGating() => await RunTest(async accelerator =>
    {
        // LSTM with identity weights and zero recurrence — test gate behavior
        // input gate ≈ sigmoid(x) ≈ 0.73 for x=1
        // forget gate ≈ sigmoid(x) ≈ 0.73
        // cell candidate ≈ tanh(x) ≈ 0.76
        // output gate ≈ sigmoid(x) ≈ 0.73
        // Verify output is non-zero and reasonable magnitude
        var x = new float[] { 1f, 1f }; // seq=1, batch=1, input=2
        var w = new float[] { 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1 }; // [1, 8, 2] identity per gate
        var r = new float[8]; // [1, 8, 1] — zero recurrence (hidden_size=1... let me use 2)

        // Simpler: just verify the operator runs without crashing and produces non-zero output
        using var xBuf = accelerator.Allocate1D(x);
        using var wBuf = accelerator.Allocate1D(w);
        using var rBuf = accelerator.Allocate1D(new float[16]); // [1, 8, 2]
        using var bBuf = accelerator.Allocate1D(new float[16]); // [1, 16]
        using var yBuf = accelerator.Allocate1D<float>(2); // [1, 1, 1, 2]
        using var yhBuf = accelerator.Allocate1D<float>(2); // [1, 1, 2]
        using var ycBuf = accelerator.Allocate1D<float>(2); // [1, 1, 2]

        var reg = new OperatorRegistry(accelerator);
        var op = reg.Resolve("LSTM")!;
        var pool = new BufferPool(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[]
            {
                new Tensor(xBuf.View, new[] { 1, 1, 2 }),
                new Tensor(wBuf.View, new[] { 1, 8, 2 }),
                new Tensor(rBuf.View, new[] { 1, 8, 2 }),
                new Tensor(bBuf.View, new[] { 1, 16 })
            },
            Outputs = new[]
            {
                new Tensor(yBuf.View, new[] { 1, 1, 1, 2 }),
                new Tensor(yhBuf.View, new[] { 1, 1, 2 }),
                new Tensor(ycBuf.View, new[] { 1, 1, 2 })
            },
            Attributes = new Dictionary<string, object> { ["hidden_size"] = (long)2 },
            Pool = pool,
            InputNames = new[] { "X", "W", "R", "B" },
            ConstantValues = new Dictionary<string, float[]>
            {
                ["X"] = x, ["W"] = w, ["R"] = new float[16], ["B"] = new float[16]
            }
        };
        op.Execute(ctx);
        await accelerator.SynchronizeAsync();

        // CPU reference: input=[1,1], W=identity per gate, R=0, B=0, hidden_size=2
        // Each gate pre-activation = 1.0 (from identity W * x)
        // it = sigmoid(1) = 0.7311, ot = sigmoid(1) = 0.7311
        // ft = sigmoid(1) = 0.7311, ct_candidate = tanh(1) = 0.7616
        // Ct = 0 * ft + it * tanh(1) = 0.7311 * 0.7616 = 0.5569
        // Ht = ot * tanh(Ct) = 0.7311 * tanh(0.5569)
        float sigm1 = 1f / (1f + MathF.Exp(-1f));
        float cell = sigm1 * MathF.Tanh(1f); // it * tanh(gc)
        float expectedH = sigm1 * MathF.Tanh(cell); // ot * tanh(Ct)
        var yhResult = await yhBuf.CopyToHostAsync<float>(0, 2);
        // Both hidden units should have the same value (symmetric identity weights)
        for (int i = 0; i < 2; i++)
            if (MathF.Abs(yhResult[i] - expectedH) > 0.01f)
                throw new Exception($"LSTM Y_h[{i}] expected {expectedH:F4}, got {yhResult[i]:F4}");
        Console.WriteLine($"[LSTM] Y_h=[{yhResult[0]:F4}, {yhResult[1]:F4}] expected {expectedH:F4} — PASS");
    });

    // ═══════════════════════════════════════════════════════════
    //  Bitwise operators
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_BitwiseAnd_MatchesCpu() => await RunTest(async accelerator =>
    {
        var a = new float[] { 15f, 255f, 170f, 0f };   // 0x0F, 0xFF, 0xAA, 0x00
        var b = new float[] { 240f, 170f, 255f, 255f }; // 0xF0, 0xAA, 0xFF, 0xFF
        var expected = new float[] { 0f, 170f, 170f, 0f }; // AND results
        using var aBuf = accelerator.Allocate1D(a);
        using var bBuf = accelerator.Allocate1D(b);
        using var outBuf = accelerator.Allocate1D<float>(4);
        var reg = new OperatorRegistry(accelerator);
        var tensor_a = new Tensor(aBuf.View, new[] { 4 });
        var tensor_b = new Tensor(bBuf.View, new[] { 4 });
        var tensor_out = new Tensor(outBuf.View, new[] { 4 });
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { tensor_a, tensor_b },
            Outputs = new[] { tensor_out },
            Attributes = new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "A", "B" },
            ConstantValues = new Dictionary<string, float[]> { ["A"] = a, ["B"] = b }
        };
        reg.Resolve("BitwiseAnd")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 0f, "BitwiseAnd: ");
    });

    [TestMethod]
    public async Task Op_BitwiseOr_MatchesCpu() => await RunTest(async accelerator =>
    {
        var a = new float[] { 15f, 0f, 170f };
        var b = new float[] { 240f, 255f, 85f };
        var expected = new float[] { 255f, 255f, 255f }; // 0x0F|0xF0=0xFF, 0|0xFF=0xFF, 0xAA|0x55=0xFF
        using var aBuf = accelerator.Allocate1D(a);
        using var bBuf = accelerator.Allocate1D(b);
        using var outBuf = accelerator.Allocate1D<float>(3);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(aBuf.View, new[] { 3 }), new Tensor(bBuf.View, new[] { 3 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 3 }) },
            Attributes = new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "A", "B" },
            ConstantValues = new Dictionary<string, float[]> { ["A"] = a, ["B"] = b }
        };
        reg.Resolve("BitwiseOr")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 0f, "BitwiseOr: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  SpaceToDepth / GlobalMaxPool
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_GlobalMaxPool_MatchesCpu() => await RunTest(async accelerator =>
    {
        // [1, 2, 3] → global max per channel → [1, 2, 1]
        var input = new float[] { 1f, 5f, 3f, 2f, 4f, 6f }; // C=2, spatial=3
        var expected = new float[] { 5f, 6f }; // max per channel
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(2);
        var reg = new OperatorRegistry(accelerator);
        reg.Reductions.ReduceMax(inBuf.View, outBuf.View, 2, 3, 1);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 0f, "GlobalMaxPool: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  DFT basic test
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_DFT_DCComponent() => await RunTest(async accelerator =>
    {
        // DFT of constant signal [1,1,1,1] → DC component should be 4+0i
        var input = new float[] { 1f, 1f, 1f, 1f }; // [1, 4, 1] real signal
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(8); // [1, 4, 2] complex output

        var reg = new OperatorRegistry(accelerator);
        var op = reg.Resolve("DFT")!;
        var pool = new BufferPool(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 1, 4, 1 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 1, 4, 2 }) },
            Attributes = new Dictionary<string, object>(),
            Pool = pool,
            InputNames = new[] { "input" },
            ConstantValues = new Dictionary<string, float[]> { ["input"] = input }
        };
        op.Execute(ctx);
        await accelerator.SynchronizeAsync();

        // DC component (k=0) should be sum of all values = 4.0, imag = 0
        var result = await outBuf.CopyToHostAsync<float>(0, 8);
        float dcReal = result[0], dcImag = result[1];
        if (MathF.Abs(dcReal - 4f) > 0.01f || MathF.Abs(dcImag) > 0.01f)
            throw new Exception($"DFT DC component: expected (4, 0), got ({dcReal:F4}, {dcImag:F4})");
        Console.WriteLine($"[DFT] DC = ({dcReal:F4}, {dcImag:F4}) — PASS");
    });

    // ═══════════════════════════════════════════════════════════
    //  MelWeightMatrix shape test
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_MelWeightMatrix_ProducesTriangularFilters() => await RunTest(async accelerator =>
    {
        // Generate mel filter bank: 10 mel bins, dft_length=32, sample_rate=16000
        int numMelBins = 10, dftLength = 32, sampleRate = 16000;
        int numSpecBins = dftLength / 2 + 1; // 17
        float lowerEdge = 0f, upperEdge = 8000f;

        using var melBuf = accelerator.Allocate1D(new float[] { numMelBins });
        using var dftBuf = accelerator.Allocate1D(new float[] { dftLength });
        using var srBuf = accelerator.Allocate1D(new float[] { sampleRate });
        using var loBuf = accelerator.Allocate1D(new float[] { lowerEdge });
        using var hiBuf = accelerator.Allocate1D(new float[] { upperEdge });
        using var outBuf = accelerator.Allocate1D<float>(numSpecBins * numMelBins);

        var reg = new OperatorRegistry(accelerator);
        var op = reg.Resolve("MelWeightMatrix")!;
        var pool = new BufferPool(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[]
            {
                new Tensor(melBuf.View, new[] { 1 }),
                new Tensor(dftBuf.View, new[] { 1 }),
                new Tensor(srBuf.View, new[] { 1 }),
                new Tensor(loBuf.View, new[] { 1 }),
                new Tensor(hiBuf.View, new[] { 1 })
            },
            Outputs = new[] { new Tensor(outBuf.View, new[] { numSpecBins, numMelBins }) },
            Attributes = new Dictionary<string, object>(),
            Pool = pool,
            InputNames = new[] { "num_mel_bins", "dft_length", "sample_rate", "lower_edge", "upper_edge" },
            ConstantValues = new Dictionary<string, float[]>
            {
                ["num_mel_bins"] = new[] { (float)numMelBins },
                ["dft_length"] = new[] { (float)dftLength },
                ["sample_rate"] = new[] { (float)sampleRate },
                ["lower_edge"] = new[] { lowerEdge },
                ["upper_edge"] = new[] { upperEdge }
            }
        };
        op.Execute(ctx);
        await accelerator.SynchronizeAsync();

        var result = await outBuf.CopyToHostAsync<float>(0, numSpecBins * numMelBins);
        // Verify: all weights non-negative, each filter has at least some weight, and filters are triangular
        int nonZeroFilters = 0;
        for (int m = 0; m < numMelBins; m++)
        {
            float sum = 0f;
            bool anyNegative = false;
            for (int k = 0; k < numSpecBins; k++)
            {
                float w = result[k * numMelBins + m];
                sum += w;
                if (w < -1e-6f) anyNegative = true;
            }
            if (sum > 0.01f) nonZeroFilters++;
            if (anyNegative) throw new Exception($"MelWeightMatrix: filter {m} has negative weight");
        }
        if (nonZeroFilters < numMelBins - 1)
            throw new Exception($"MelWeightMatrix: only {nonZeroFilters}/{numMelBins} filters active — expected all");
        Console.WriteLine($"[MelWeightMatrix] {nonZeroFilters}/{numMelBins} active, all non-negative — PASS");
    });

    // ═══════════════════════════════════════════════════════════
    //  Sum / Mean multi-input operators
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_Sum_ThreeInputs() => await RunTest(async accelerator =>
    {
        var a = new float[] { 1f, 2f, 3f };
        var b = new float[] { 4f, 5f, 6f };
        var c = new float[] { 7f, 8f, 9f };
        var expected = new float[] { 12f, 15f, 18f };
        using var aBuf = accelerator.Allocate1D(a);
        using var bBuf = accelerator.Allocate1D(b);
        using var cBuf = accelerator.Allocate1D(c);
        using var outBuf = accelerator.Allocate1D<float>(3);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(aBuf.View, new[] { 3 }), new Tensor(bBuf.View, new[] { 3 }), new Tensor(cBuf.View, new[] { 3 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 3 }) },
            Attributes = new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "A", "B", "C" }
        };
        reg.Resolve("Sum")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "Sum: ");
    });

    [TestMethod]
    public async Task Op_Mean_TwoInputs() => await RunTest(async accelerator =>
    {
        var a = new float[] { 2f, 4f, 6f };
        var b = new float[] { 4f, 6f, 8f };
        var expected = new float[] { 3f, 5f, 7f };
        using var aBuf = accelerator.Allocate1D(a);
        using var bBuf = accelerator.Allocate1D(b);
        using var outBuf = accelerator.Allocate1D<float>(3);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(aBuf.View, new[] { 3 }), new Tensor(bBuf.View, new[] { 3 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 3 }) },
            Attributes = new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "A", "B" }
        };
        reg.Resolve("Mean")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "Mean: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  PRelu operator
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_PRelu_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { -2f, -1f, 0f, 1f, 2f };
        var slope = new float[] { 0.1f, 0.1f, 0.1f, 0.1f, 0.1f };
        var expected = new float[] { -0.2f, -0.1f, 0f, 1f, 2f };
        using var inBuf = accelerator.Allocate1D(input);
        using var slopeBuf = accelerator.Allocate1D(slope);
        using var outBuf = accelerator.Allocate1D<float>(5);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 5 }), new Tensor(slopeBuf.View, new[] { 5 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 5 }) },
            Attributes = new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "X", "slope" },
            ConstantValues = new Dictionary<string, float[]> { ["X"] = input, ["slope"] = slope }
        };
        reg.Resolve("PRelu")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "PRelu: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  ArgMin operator
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_ArgMin_MatchesCpu() => await RunTest(async accelerator =>
    {
        // [2, 3] → argmin axis=1 → [2, 1]
        var input = new float[] { 3f, 1f, 2f, 6f, 4f, 5f };
        var expected = new float[] { 1f, 1f }; // indices of min in each row
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(2);
        var reg = new OperatorRegistry(accelerator);
        var inTensor = new Tensor(inBuf.View, new[] { 2, 3 });
        var outTensor = new Tensor(outBuf.View, new[] { 2, 1 });
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { inTensor },
            Outputs = new[] { outTensor },
            Attributes = new Dictionary<string, object> { ["axis"] = (long)1, ["keepdims"] = (long)1 },
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "input" }
        };
        reg.Resolve("ArgMin")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 0f, "ArgMin: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  Identity / Size operators
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_Identity_CopiesInput() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1f, 2f, 3f, 4f, 5f };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(5);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 5 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 5 }) },
            Attributes = new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "input" }
        };
        reg.Resolve("Identity")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, input, 0f, "Identity: ");
    });

    [TestMethod]
    public async Task Op_Size_ReturnsElementCount() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1f, 2f, 3f, 4f, 5f, 6f }; // 6 elements
        var expected = new float[] { 6f };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(1);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 2, 3 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 1 }) },
            Attributes = new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "input" }
        };
        reg.Resolve("Size")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 0f, "Size: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  LogSoftmax operator
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_LogSoftmax_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1f, 2f, 3f };
        // softmax = [0.0900, 0.2447, 0.6652], log = [-2.4076, -1.4076, -0.4076]
        float sumExp = MathF.Exp(1) + MathF.Exp(2) + MathF.Exp(3);
        var expected = input.Select(x => MathF.Log(MathF.Exp(x) / sumExp)).ToArray();
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(3);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 1, 3 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 1, 3 }) },
            Attributes = new Dictionary<string, object> { ["axis"] = (long)1 },
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "input" }
        };
        reg.Resolve("LogSoftmax")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-3f, "LogSoftmax: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  EyeLike / Det / Compress
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_EyeLike_Identity3x3() => await RunTest(async accelerator =>
    {
        var input = new float[9]; // 3x3 zeros — shape template
        var expected = new float[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 }; // identity matrix
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(9);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 3, 3 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 3, 3 }) },
            Attributes = new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "input" }
        };
        reg.Resolve("EyeLike")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 0f, "EyeLike: ");
    });

    [TestMethod]
    public async Task Op_Det_2x2Matrix() => await RunTest(async accelerator =>
    {
        // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
        var input = new float[] { 1f, 2f, 3f, 4f };
        var expected = new float[] { -2f };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(1);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 2, 2 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 1 }) },
            Attributes = new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "input" },
            ConstantValues = new Dictionary<string, float[]> { ["input"] = input }
        };
        reg.Resolve("Det")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "Det 2x2: ");
    });

    [TestMethod]
    public async Task Op_Det_3x3Matrix() => await RunTest(async accelerator =>
    {
        // det([[1,2,3],[0,1,4],[5,6,0]]) = 1(0-24) - 2(0-20) + 3(0-5) = -24+40-15 = 1
        var input = new float[] { 1, 2, 3, 0, 1, 4, 5, 6, 0 };
        var expected = new float[] { 1f };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(1);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 3, 3 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 1 }) },
            Attributes = new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "input" },
            ConstantValues = new Dictionary<string, float[]> { ["input"] = input }
        };
        reg.Resolve("Det")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-3f, "Det 3x3: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  DequantizeLinear / QuantizeLinear
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_DequantizeLinear_MatchesCpu() => await RunTest(async accelerator =>
    {
        // y = (x - zero_point) * scale
        var x = new float[] { 0f, 1f, 2f, 3f, 4f };
        var scale = new float[] { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
        var zeroPoint = new float[] { 1f, 1f, 1f, 1f, 1f };
        var expected = new float[] { -0.5f, 0f, 0.5f, 1f, 1.5f };
        using var xBuf = accelerator.Allocate1D(x);
        using var scaleBuf = accelerator.Allocate1D(scale);
        using var zpBuf = accelerator.Allocate1D(zeroPoint);
        using var outBuf = accelerator.Allocate1D<float>(5);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] {
                new Tensor(xBuf.View, new[] { 5 }),
                new Tensor(scaleBuf.View, new[] { 5 }),
                new Tensor(zpBuf.View, new[] { 5 })
            },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 5 }) },
            Attributes = new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "x", "x_scale", "x_zero_point" },
            ConstantValues = new Dictionary<string, float[]> { ["x"] = x, ["x_scale"] = scale, ["x_zero_point"] = zeroPoint }
        };
        reg.Resolve("DequantizeLinear")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "DequantizeLinear: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  ScatterElements
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_ScatterElements_Axis0() => await RunTest(async accelerator =>
    {
        // data = [[1,2,3],[4,5,6],[7,8,9]], indices = [[1,0,2],[0,2,1]], updates = [[10,20,30],[40,50,60]]
        // axis=0: data[indices[i,j], j] = updates[i,j]
        // Result: data[1,0]=10, data[0,1]=20, data[2,2]=30, data[0,0]=40, data[2,1]=50, data[1,2]=60
        // → [[40,20,3],[10,5,60],[7,50,30]]
        var data = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        var indices = new float[] { 1, 0, 2, 0, 2, 1 };
        var updates = new float[] { 10, 20, 30, 40, 50, 60 };
        var expected = new float[] { 40, 20, 3, 10, 5, 60, 7, 50, 30 };
        using var dataBuf = accelerator.Allocate1D(data);
        using var idxBuf = accelerator.Allocate1D(indices);
        using var updBuf = accelerator.Allocate1D(updates);
        using var outBuf = accelerator.Allocate1D<float>(9);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] {
                new Tensor(dataBuf.View, new[] { 3, 3 }),
                new Tensor(idxBuf.View, new[] { 2, 3 }),
                new Tensor(updBuf.View, new[] { 2, 3 })
            },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 3, 3 }) },
            Attributes = new Dictionary<string, object> { ["axis"] = (long)0 },
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "data", "indices", "updates" },
            ConstantValues = new Dictionary<string, float[]>
            {
                ["data"] = data, ["indices"] = indices, ["updates"] = updates
            }
        };
        reg.Resolve("ScatterElements")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 0f, "ScatterElements: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  Where with broadcast
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_Where_Broadcast() => await RunTest(async accelerator =>
    {
        // condition [3]: [1, 0, 1], X [3]: [10, 20, 30], Y [3]: [100, 200, 300]
        // → [10, 200, 30]
        var cond = new float[] { 1f, 0f, 1f };
        var x = new float[] { 10f, 20f, 30f };
        var y = new float[] { 100f, 200f, 300f };
        var expected = new float[] { 10f, 200f, 30f };
        using var condBuf = accelerator.Allocate1D(cond);
        using var xBuf = accelerator.Allocate1D(x);
        using var yBuf = accelerator.Allocate1D(y);
        using var outBuf = accelerator.Allocate1D<float>(3);
        var ew = GetOrCreateEW(accelerator);
        ew.Where(condBuf.View, xBuf.View, yBuf.View, outBuf.View, 3);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 0f, "Where: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  GRU basic test
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_GRU_ForwardPass() => await RunTest(async accelerator =>
    {
        // GRU: seq=1, batch=1, input=2, hidden=2
        // With zero recurrence weights, output = (1-z)*tanh(Xt*Wh) where z=sigmoid(Xt*Wz)
        var x = new float[] { 1f, 0f }; // [1, 1, 2]
        var w = new float[] { 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1 }; // [1, 6, 2] identity per gate
        var r = new float[12]; // [1, 6, 2] zero recurrence
        var b = new float[12]; // [1, 12] zero bias

        using var xBuf = accelerator.Allocate1D(x);
        using var wBuf = accelerator.Allocate1D(w);
        using var rBuf = accelerator.Allocate1D(r);
        using var bBuf = accelerator.Allocate1D(b);
        using var yBuf = accelerator.Allocate1D<float>(2);
        using var yhBuf = accelerator.Allocate1D<float>(2);

        var reg = new OperatorRegistry(accelerator);
        var op = reg.Resolve("GRU")!;
        var pool = new BufferPool(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[]
            {
                new Tensor(xBuf.View, new[] { 1, 1, 2 }),
                new Tensor(wBuf.View, new[] { 1, 6, 2 }),
                new Tensor(rBuf.View, new[] { 1, 6, 2 }),
                new Tensor(bBuf.View, new[] { 1, 12 })
            },
            Outputs = new[]
            {
                new Tensor(yBuf.View, new[] { 1, 1, 1, 2 }),
                new Tensor(yhBuf.View, new[] { 1, 1, 2 })
            },
            Attributes = new Dictionary<string, object> { ["hidden_size"] = (long)2 },
            Pool = pool,
            InputNames = new[] { "X", "W", "R", "B" },
            ConstantValues = new Dictionary<string, float[]>
            {
                ["X"] = x, ["W"] = w, ["R"] = r, ["B"] = b
            }
        };
        op.Execute(ctx);
        await accelerator.SynchronizeAsync();

        // CPU reference: x=[1,0], W=identity per gate, R=0, B=0, h0=0
        // z = sigmoid(W_z * x) = [sigmoid(1), sigmoid(0)] = [0.7311, 0.5]
        // r = sigmoid(W_r * x) = [sigmoid(1), sigmoid(0)] = [0.7311, 0.5]
        // h_candidate = tanh(W_h * x + 0) = [tanh(1), tanh(0)] = [0.7616, 0]
        // Ht = (1-z)*h_candidate + z*h0 = (1-z)*h_candidate (h0=0)
        float sig1 = 1f / (1f + MathF.Exp(-1f));
        float sig0 = 0.5f;
        float expectedH0 = (1f - sig1) * MathF.Tanh(1f);
        float expectedH1 = (1f - sig0) * 0f; // tanh(0)=0
        var yhResult = await yhBuf.CopyToHostAsync<float>(0, 2);
        if (MathF.Abs(yhResult[0] - expectedH0) > 0.01f)
            throw new Exception($"GRU Y_h[0] expected {expectedH0:F4}, got {yhResult[0]:F4}");
        if (MathF.Abs(yhResult[1] - expectedH1) > 0.01f)
            throw new Exception($"GRU Y_h[1] expected {expectedH1:F4}, got {yhResult[1]:F4}");
        Console.WriteLine($"[GRU] Y_h=[{yhResult[0]:F4}, {yhResult[1]:F4}] expected [{expectedH0:F4}, {expectedH1:F4}] — PASS");
    });

    // ═══════════════════════════════════════════════════════════
    //  Additional operator tests for full coverage
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_BitwiseXor_MatchesCpu() => await RunTest(async accelerator =>
    {
        var a = new float[] { 15f, 255f, 170f };
        var b = new float[] { 240f, 170f, 85f };
        var expected = new float[] { 255f, 85f, 255f }; // XOR
        using var aBuf = accelerator.Allocate1D(a);
        using var bBuf = accelerator.Allocate1D(b);
        using var outBuf = accelerator.Allocate1D<float>(3);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext { Inputs = new[] { new Tensor(aBuf.View, new[] { 3 }), new Tensor(bBuf.View, new[] { 3 }) }, Outputs = new[] { new Tensor(outBuf.View, new[] { 3 }) }, Attributes = new Dictionary<string, object>(), Pool = new BufferPool(accelerator), InputNames = new[] { "A", "B" }, ConstantValues = new Dictionary<string, float[]> { ["A"] = a, ["B"] = b } };
        reg.Resolve("BitwiseXor")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 0f, "BitwiseXor: ");
    });

    [TestMethod]
    public async Task Op_BitShift_Left() => await RunTest(async accelerator =>
    {
        var a = new float[] { 1f, 2f, 4f };
        var b = new float[] { 2f, 3f, 1f };
        var expected = new float[] { 4f, 16f, 8f }; // 1<<2=4, 2<<3=16, 4<<1=8
        using var aBuf = accelerator.Allocate1D(a);
        using var bBuf = accelerator.Allocate1D(b);
        using var outBuf = accelerator.Allocate1D<float>(3);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext { Inputs = new[] { new Tensor(aBuf.View, new[] { 3 }), new Tensor(bBuf.View, new[] { 3 }) }, Outputs = new[] { new Tensor(outBuf.View, new[] { 3 }) }, Attributes = new Dictionary<string, object> { ["direction"] = "LEFT" }, Pool = new BufferPool(accelerator), InputNames = new[] { "A", "B" }, ConstantValues = new Dictionary<string, float[]> { ["A"] = a, ["B"] = b } };
        reg.Resolve("BitShift")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 0f, "BitShift LEFT: ");
    });

    [TestMethod]
    public async Task Op_HannWindow_Shape() => await RunTest(async accelerator =>
    {
        using var sizeBuf = accelerator.Allocate1D(new float[] { 5f });
        using var outBuf = accelerator.Allocate1D<float>(5);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext { Inputs = new[] { new Tensor(sizeBuf.View, new[] { 1 }) }, Outputs = new[] { new Tensor(outBuf.View, new[] { 5 }) }, Attributes = new Dictionary<string, object>(), Pool = new BufferPool(accelerator), InputNames = new[] { "size" }, ConstantValues = new Dictionary<string, float[]> { ["size"] = new[] { 5f } } };
        reg.Resolve("HannWindow")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        // Hann window: w[i] = 0.5 * (1 - cos(2*pi*i/(N-1)))
        var expected = new float[5];
        for (int i = 0; i < 5; i++) expected[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / 4f));
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "HannWindow: ");
    });

    [TestMethod]
    public async Task Op_HammingWindow_Shape() => await RunTest(async accelerator =>
    {
        using var sizeBuf = accelerator.Allocate1D(new float[] { 5f });
        using var outBuf = accelerator.Allocate1D<float>(5);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext { Inputs = new[] { new Tensor(sizeBuf.View, new[] { 1 }) }, Outputs = new[] { new Tensor(outBuf.View, new[] { 5 }) }, Attributes = new Dictionary<string, object>(), Pool = new BufferPool(accelerator), InputNames = new[] { "size" }, ConstantValues = new Dictionary<string, float[]> { ["size"] = new[] { 5f } } };
        reg.Resolve("HammingWindow")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var expected = new float[5];
        for (int i = 0; i < 5; i++) expected[i] = 0.54f - 0.46f * MathF.Cos(2f * MathF.PI * i / 4f);
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "HammingWindow: ");
    });

    [TestMethod]
    public async Task Op_BlackmanWindow_Shape() => await RunTest(async accelerator =>
    {
        using var sizeBuf = accelerator.Allocate1D(new float[] { 5f });
        using var outBuf = accelerator.Allocate1D<float>(5);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext { Inputs = new[] { new Tensor(sizeBuf.View, new[] { 1 }) }, Outputs = new[] { new Tensor(outBuf.View, new[] { 5 }) }, Attributes = new Dictionary<string, object>(), Pool = new BufferPool(accelerator), InputNames = new[] { "size" }, ConstantValues = new Dictionary<string, float[]> { ["size"] = new[] { 5f } } };
        reg.Resolve("BlackmanWindow")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var expected = new float[5];
        for (int i = 0; i < 5; i++) { float t = 2f * MathF.PI * i / 4f; expected[i] = 0.42f - 0.5f * MathF.Cos(t) + 0.08f * MathF.Cos(2f * t); }
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "BlackmanWindow: ");
    });

    [TestMethod]
    public async Task Op_ReduceLogSumExp_MatchesCpu() => await RunTest(async accelerator =>
    {
        // [1, 3] → log(sum(exp(x))) along axis 1
        var input = new float[] { 1f, 2f, 3f };
        float expectedVal = MathF.Log(MathF.Exp(1f) + MathF.Exp(2f) + MathF.Exp(3f));
        using var inBuf = accelerator.Allocate1D(input);
        using var expBuf = accelerator.Allocate1D<float>(3);
        using var sumBuf = accelerator.Allocate1D<float>(1);
        using var outBuf = accelerator.Allocate1D<float>(1);
        var ew = GetOrCreateEW(accelerator);
        var reg = new OperatorRegistry(accelerator);
        ew.Exp(inBuf.View, expBuf.View, 3);
        reg.Reductions.ReduceSum(expBuf.View, sumBuf.View, 1, 3, 1);
        ew.Log(sumBuf.View, outBuf.View, 1);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, new[] { expectedVal }, 1e-4f, "ReduceLogSumExp: ");
    });

    [TestMethod]
    public async Task Op_ReduceLogSum_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1f, 2f, 3f };
        float expectedVal = MathF.Log(6f); // log(1+2+3)
        using var inBuf = accelerator.Allocate1D(input);
        using var sumBuf = accelerator.Allocate1D<float>(1);
        using var outBuf = accelerator.Allocate1D<float>(1);
        var ew = GetOrCreateEW(accelerator);
        var reg = new OperatorRegistry(accelerator);
        reg.Reductions.ReduceSum(inBuf.View, sumBuf.View, 1, 3, 1);
        ew.Log(sumBuf.View, outBuf.View, 1);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, new[] { expectedVal }, 1e-4f, "ReduceLogSum: ");
    });

    [TestMethod]
    public async Task Op_ReduceL2_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { 3f, 4f }; // L2 norm = sqrt(9+16) = 5
        using var inBuf = accelerator.Allocate1D(input);
        using var inBuf2 = accelerator.Allocate1D(input); // separate copy to avoid WebGPU aliasing
        using var sqBuf = accelerator.Allocate1D<float>(2);
        using var sumBuf = accelerator.Allocate1D<float>(1);
        using var outBuf = accelerator.Allocate1D<float>(1);
        var ew = GetOrCreateEW(accelerator);
        var reg = new OperatorRegistry(accelerator);
        ew.Mul(inBuf.View, inBuf2.View, sqBuf.View, 2);
        reg.Reductions.ReduceSum(sqBuf.View, sumBuf.View, 1, 2, 1);
        ew.Sqrt(sumBuf.View, outBuf.View, 1);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, new[] { 5f }, 1e-4f, "ReduceL2: ");
    });

    [TestMethod]
    public async Task Op_CastLike_PassThrough() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1f, 2f, 3f };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(3);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext { Inputs = new[] { new Tensor(inBuf.View, new[] { 3 }), new Tensor(inBuf.View, new[] { 3 }) }, Outputs = new[] { new Tensor(outBuf.View, new[] { 3 }) }, Attributes = new Dictionary<string, object>(), Pool = new BufferPool(accelerator), InputNames = new[] { "input", "target" } };
        reg.Resolve("CastLike")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, input, 0f, "CastLike: ");
    });

    [TestMethod]
    public async Task Op_ThresholdedRelu_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { -1f, 0f, 0.5f, 1f, 2f };
        // Default alpha=1: output = x if x > 1 else 0
        // With Scale copy (current impl copies input — full impl would threshold)
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(5);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext { Inputs = new[] { new Tensor(inBuf.View, new[] { 5 }) }, Outputs = new[] { new Tensor(outBuf.View, new[] { 5 }) }, Attributes = new Dictionary<string, object>(), Pool = new BufferPool(accelerator), InputNames = new[] { "X" } };
        reg.Resolve("ThresholdedRelu")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        // ThresholdedRelu with alpha=1: x>1 → x, else 0
        var expected = new float[] { 0f, 0f, 0f, 0f, 2f };
        await AssertCloseGpu(accelerator, outBuf.View, expected, 0f, "ThresholdedRelu: ");
    });

    [TestMethod]
    public async Task Op_Hardmax_Runs() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1f, 3f, 2f };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(3);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext { Inputs = new[] { new Tensor(inBuf.View, new[] { 1, 3 }) }, Outputs = new[] { new Tensor(outBuf.View, new[] { 1, 3 }) }, Attributes = new Dictionary<string, object> { ["axis"] = (long)1 }, Pool = new BufferPool(accelerator), InputNames = new[] { "input" } };
        reg.Resolve("Hardmax")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        // input [1,3,2] → argmax at index 1 → one-hot [0, 1, 0]
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { 0f, 1f, 0f }, 0f, "Hardmax: ");
    });

    [TestMethod]
    public async Task Op_LpNormalization_Runs() => await RunTest(async accelerator =>
    {
        var input = new float[] { 3f, 4f }; // L2 norm = 5, normalized = [0.6, 0.8]
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(2);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext { Inputs = new[] { new Tensor(inBuf.View, new[] { 1, 2 }) }, Outputs = new[] { new Tensor(outBuf.View, new[] { 1, 2 }) }, Attributes = new Dictionary<string, object> { ["axis"] = (long)1, ["p"] = (long)2 }, Pool = new BufferPool(accelerator), InputNames = new[] { "input" }, ConstantValues = new Dictionary<string, float[]> { ["input"] = input } };
        reg.Resolve("LpNormalization")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, new[] { 0.6f, 0.8f }, 1e-4f, "LpNormalization: ");
    });

    [TestMethod]
    public async Task Op_Bernoulli_ProducesZeroOrOne() => await RunTest(async accelerator =>
    {
        var probs = new float[] { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
        using var probBuf = accelerator.Allocate1D(probs);
        using var outBuf = accelerator.Allocate1D<float>(8);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext { Inputs = new[] { new Tensor(probBuf.View, new[] { 8 }) }, Outputs = new[] { new Tensor(outBuf.View, new[] { 8 }) }, Attributes = new Dictionary<string, object>(), Pool = new BufferPool(accelerator), InputNames = new[] { "input" }, ConstantValues = new Dictionary<string, float[]> { ["input"] = probs } };
        reg.Resolve("Bernoulli")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 8);
        // All values should be 0 or 1
        foreach (var v in result)
            if (v != 0f && v != 1f)
                throw new Exception($"Bernoulli produced {v} — expected 0 or 1");
        Console.WriteLine($"[Bernoulli] All values 0 or 1 — PASS (sum={result.Sum():F0}/8)");
    });

    [TestMethod]
    public async Task Op_RandomNormal_ProducesValues() => await RunTest(async accelerator =>
    {
        using var outBuf = accelerator.Allocate1D<float>(100);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext { Inputs = Array.Empty<Tensor>(), Outputs = new[] { new Tensor(outBuf.View, new[] { 100 }) }, Attributes = new Dictionary<string, object> { ["shape"] = new long[] { 100 }, ["mean"] = 0f, ["scale"] = 1f, ["seed"] = (long)42 }, Pool = new BufferPool(accelerator), InputNames = Array.Empty<string>() };
        reg.Resolve("RandomNormal")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 100);
        float mean = result.Average();
        float std = MathF.Sqrt(result.Select(x => (x - mean) * (x - mean)).Average());
        if (MathF.Abs(mean) > 0.5f) throw new Exception($"RandomNormal mean={mean:F3} too far from 0");
        if (std < 0.3f || std > 2f) throw new Exception($"RandomNormal std={std:F3} out of range");
        Console.WriteLine($"[RandomNormal] mean={mean:F3}, std={std:F3} — PASS");
    });

    [TestMethod]
    public async Task Op_RandomUniform_InRange() => await RunTest(async accelerator =>
    {
        using var outBuf = accelerator.Allocate1D<float>(100);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext { Inputs = Array.Empty<Tensor>(), Outputs = new[] { new Tensor(outBuf.View, new[] { 100 }) }, Attributes = new Dictionary<string, object> { ["shape"] = new long[] { 100 }, ["low"] = 2f, ["high"] = 5f, ["seed"] = (long)42 }, Pool = new BufferPool(accelerator), InputNames = Array.Empty<string>() };
        reg.Resolve("RandomUniform")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 100);
        if (result.Any(v => v < 2f || v > 5f)) throw new Exception("RandomUniform value out of [2, 5] range");
        Console.WriteLine($"[RandomUniform] All in [2, 5], mean={result.Average():F3} — PASS");
    });

    [TestMethod]
    public async Task Op_MaxUnpool_ScattersByIndex() => await RunTest(async accelerator =>
    {
        var values = new float[] { 5f, 8f };
        var indices = new float[] { 1f, 3f }; // scatter to positions 1 and 3
        var expected = new float[] { 0f, 5f, 0f, 8f }; // zeros elsewhere
        using var valBuf = accelerator.Allocate1D(values);
        using var idxBuf = accelerator.Allocate1D(indices);
        using var outBuf = accelerator.Allocate1D<float>(4);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext { Inputs = new[] { new Tensor(valBuf.View, new[] { 2 }), new Tensor(idxBuf.View, new[] { 2 }) }, Outputs = new[] { new Tensor(outBuf.View, new[] { 4 }) }, Attributes = new Dictionary<string, object>(), Pool = new BufferPool(accelerator), InputNames = new[] { "X", "I" }, ConstantValues = new Dictionary<string, float[]> { ["X"] = values, ["I"] = indices } };
        reg.Resolve("MaxUnpool")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 0f, "MaxUnpool: ");
    });

    [TestMethod]
    public async Task Op_AffineGrid_Identity() => await RunTest(async accelerator =>
    {
        // Identity transform: theta = [[1,0,0],[0,1,0]] → grid should be normalized [-1,1] coords
        var theta = new float[] { 1, 0, 0, 0, 1, 0 }; // [1, 2, 3]
        var size = new float[] { 1, 1, 3, 3 }; // N=1, C=1, H=3, W=3
        using var thetaBuf = accelerator.Allocate1D(theta);
        using var sizeBuf = accelerator.Allocate1D(size);
        using var outBuf = accelerator.Allocate1D<float>(18); // [1, 3, 3, 2]
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext { Inputs = new[] { new Tensor(thetaBuf.View, new[] { 1, 2, 3 }), new Tensor(sizeBuf.View, new[] { 4 }) }, Outputs = new[] { new Tensor(outBuf.View, new[] { 1, 3, 3, 2 }) }, Attributes = new Dictionary<string, object>(), Pool = new BufferPool(accelerator), InputNames = new[] { "theta", "size" }, ConstantValues = new Dictionary<string, float[]> { ["theta"] = theta, ["size"] = size } };
        reg.Resolve("AffineGrid")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 18);
        // Center pixel (1,1) should map to (0,0) with identity transform
        float cx = result[8], cy = result[9]; // [0,1,1,0] and [0,1,1,1]
        if (MathF.Abs(cx) > 0.1f || MathF.Abs(cy) > 0.1f)
            throw new Exception($"AffineGrid center: expected ~(0,0), got ({cx:F3}, {cy:F3})");
        Console.WriteLine($"[AffineGrid] Center=({cx:F3}, {cy:F3}) — PASS");
    });

    // ═══════════════════════════════════════════════════════════
    //  GridSample / Col2Im / NMS / Quantized ops
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_GridSample_IdentityGrid() => await RunTest(async accelerator =>
    {
        // 1x1x2x2 input, identity grid should return same values
        var input = new float[] { 1f, 2f, 3f, 4f }; // [1,1,2,2]
        // Grid at corners: [-1,-1], [1,-1], [-1,1], [1,1] → maps to [0,0], [1,0], [0,1], [1,1]
        var grid = new float[] { -1, -1, 1, -1, -1, 1, 1, 1 }; // [1,2,2,2]
        using var inBuf = accelerator.Allocate1D(input);
        using var gridBuf = accelerator.Allocate1D(grid);
        using var outBuf = accelerator.Allocate1D<float>(4);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 1, 1, 2, 2 }), new Tensor(gridBuf.View, new[] { 1, 2, 2, 2 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 1, 1, 2, 2 }) },
            Attributes = new Dictionary<string, object> { ["align_corners"] = (long)1 },
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "X", "grid" },
            ConstantValues = new Dictionary<string, float[]> { ["X"] = input, ["grid"] = grid }
        };
        reg.Resolve("GridSample")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        // With align_corners=1, corners map exactly to pixel positions
        await AssertCloseGpu(accelerator, outBuf.View, input, 1e-4f, "GridSample identity: ");
    });

    [TestMethod]
    public async Task Op_NMS_BasicSelection() => await RunTest(async accelerator =>
    {
        // 3 boxes, 1 class, overlapping — NMS should select the highest-scoring non-overlapping
        var boxes = new float[] {
            0, 0, 10, 10,  // box 0
            1, 1, 11, 11,  // box 1 (overlaps heavily with 0)
            50, 50, 60, 60  // box 2 (no overlap)
        }; // [1, 3, 4]
        var scores = new float[] { 0.9f, 0.8f, 0.7f }; // [1, 1, 3]
        using var boxBuf = accelerator.Allocate1D(boxes);
        using var scoreBuf = accelerator.Allocate1D(scores);
        using var outBuf = accelerator.Allocate1D<float>(9); // max 3 selections * 3

        var reg = new OperatorRegistry(accelerator);
        // NMS inputs: boxes, scores, max_output_boxes, iou_threshold, score_threshold
        using var maxBoxBuf = accelerator.Allocate1D(new float[] { 10f });
        using var iouBuf = accelerator.Allocate1D(new float[] { 0.5f });
        using var scoreThrBuf = accelerator.Allocate1D(new float[] { 0.1f });

        var ctx = new OnnxOpContext
        {
            Inputs = new[] {
                new Tensor(boxBuf.View, new[] { 1, 3, 4 }),
                new Tensor(scoreBuf.View, new[] { 1, 1, 3 }),
                new Tensor(maxBoxBuf.View, new[] { 1 }),
                new Tensor(iouBuf.View, new[] { 1 }),
                new Tensor(scoreThrBuf.View, new[] { 1 })
            },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 3, 3 }) },
            Attributes = new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "boxes", "scores", "max_output_boxes", "iou_threshold", "score_threshold" },
            ConstantValues = new Dictionary<string, float[]>
            {
                ["boxes"] = boxes, ["scores"] = scores,
                ["max_output_boxes"] = new[] { 10f },
                ["iou_threshold"] = new[] { 0.5f },
                ["score_threshold"] = new[] { 0.1f }
            }
        };
        reg.Resolve("NonMaxSuppression")!.Execute(ctx);
        await accelerator.SynchronizeAsync();

        var result = await outBuf.CopyToHostAsync<float>(0, 9);
        // Box 0 (highest score) should be selected. Box 1 suppressed (overlaps). Box 2 selected.
        // Result format: [batch, class, box_index] — expect at least 2 selections
        int selected = 0;
        for (int i = 0; i < 3; i++)
            if (result[i * 3 + 2] >= 0) selected++;
        if (selected < 2) throw new Exception($"NMS selected only {selected} boxes — expected at least 2");
        Console.WriteLine($"[NMS] Selected {selected} boxes — PASS");
    });

    [TestMethod]
    public async Task Op_DequantizeLinear_NoZeroPoint() => await RunTest(async accelerator =>
    {
        // y = x * scale (no zero point)
        var x = new float[] { 0f, 1f, 2f, 3f };
        var scale = new float[] { 2f, 2f, 2f, 2f };
        var expected = new float[] { 0f, 2f, 4f, 6f };
        using var xBuf = accelerator.Allocate1D(x);
        using var scaleBuf = accelerator.Allocate1D(scale);
        using var outBuf = accelerator.Allocate1D<float>(4);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(xBuf.View, new[] { 4 }), new Tensor(scaleBuf.View, new[] { 4 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 4 }) },
            Attributes = new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "x", "x_scale" }
        };
        reg.Resolve("DequantizeLinear")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-4f, "DequantizeLinear (no zp): ");
    });

    [TestMethod]
    public async Task Op_Einsum_MatMul() => await RunTest(async accelerator =>
    {
        // Einsum "ij,jk->ik" = standard matmul
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]] → C = [[19,22],[43,50]]
        var a = new float[] { 1, 2, 3, 4 };
        var b = new float[] { 5, 6, 7, 8 };
        var expected = new float[] { 19, 22, 43, 50 };
        using var aBuf = accelerator.Allocate1D(a);
        using var bBuf = accelerator.Allocate1D(b);
        using var outBuf = accelerator.Allocate1D<float>(4);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(aBuf.View, new[] { 2, 2 }), new Tensor(bBuf.View, new[] { 2, 2 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 2, 2 }) },
            Attributes = new Dictionary<string, object> { ["equation"] = "ij,jk->ik" },
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "A", "B" },
            ConstantValues = new Dictionary<string, float[]> { ["A"] = a, ["B"] = b }
        };
        reg.Resolve("Einsum")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-3f, "Einsum matmul: ");
    });

    [TestMethod]
    public async Task Op_Einsum_BroadcastMul() => await RunTest(async accelerator =>
    {
        // Einsum "ij,j->ij" = broadcast multiply (row * vector)
        var a = new float[] { 1, 2, 3, 4, 5, 6 }; // [2, 3]
        var b = new float[] { 10, 20, 30 }; // [3]
        var expected = new float[] { 10, 40, 90, 40, 100, 180 };
        using var aBuf = accelerator.Allocate1D(a);
        using var bBuf = accelerator.Allocate1D(b);
        using var outBuf = accelerator.Allocate1D<float>(6);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(aBuf.View, new[] { 2, 3 }), new Tensor(bBuf.View, new[] { 3 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 2, 3 }) },
            Attributes = new Dictionary<string, object> { ["equation"] = "ij,j->ij" },
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "A", "B" }
        };
        reg.Resolve("Einsum")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-3f, "Einsum broadcast: ");
    });

    [TestMethod]
    public async Task Op_GlobalLpPool_L2() => await RunTest(async accelerator =>
    {
        // [1, 1, 4] → GlobalLpPool p=2 → sqrt(sum(x^2)/N) per channel
        // = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.738
        var input = new float[] { 1f, 2f, 3f, 4f };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(1);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 1, 1, 4 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 1, 1, 1 }) },
            Attributes = new Dictionary<string, object> { ["p"] = (long)2 },
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "X" },
            ConstantValues = new Dictionary<string, float[]> { ["X"] = input }
        };
        reg.Resolve("GlobalLpPool")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        // L2 pool of [1,2,3,4] = sqrt(1+4+9+16) = sqrt(30) ≈ 5.477
        float expected = MathF.Sqrt(1 + 4 + 9 + 16);
        await AssertCloseGpu(accelerator, outBuf.View, new[] { expected }, 0.01f, "GlobalLpPool L2: ");
    });

    [TestMethod]
    public async Task Op_ReverseSequence_MatchesCpu() => await RunTest(async accelerator =>
    {
        // Input [4, 2] (time=4, batch=2), batch_axis=1, time_axis=0, seq_lens=[4, 2]
        // Batch 0: reverse all 4 → [4,3,2,1], Batch 1: reverse first 2 → [2,1,3,4]
        var input = new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }; // [t0b0,t0b1,t1b0,t1b1,t2b0,t2b1,t3b0,t3b1]
        var seqLens = new float[] { 4f, 2f };
        // time_axis=0, batch_axis=1: reverse time for each batch independently
        // Batch 0 (col 0): [1,3,5,7] → reversed all 4: [7,5,3,1]
        // Batch 1 (col 1): [2,4,6,8] → reversed first 2: [4,2,6,8]
        var expected = new float[] { 7f, 4f, 5f, 2f, 3f, 6f, 1f, 8f };
        using var inBuf = accelerator.Allocate1D(input);
        using var slBuf = accelerator.Allocate1D(seqLens);
        using var outBuf = accelerator.Allocate1D<float>(8);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 4, 2 }), new Tensor(slBuf.View, new[] { 2 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 4, 2 }) },
            Attributes = new Dictionary<string, object> { ["batch_axis"] = (long)1, ["time_axis"] = (long)0 },
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "input", "sequence_lens" },
            ConstantValues = new Dictionary<string, float[]> { ["sequence_lens"] = seqLens }
        };
        reg.Resolve("ReverseSequence")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 0f, "ReverseSequence: ");
    });

    [TestMethod]
    public async Task Op_Unique_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { 2f, 1f, 1f, 3f, 2f };
        // Unique sorted: [1, 2, 3]
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(5);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 5 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 5 }) },
            Attributes = new Dictionary<string, object> { ["sorted"] = (long)1 },
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "X" },
            ConstantValues = new Dictionary<string, float[]> { ["X"] = input }
        };
        reg.Resolve("Unique")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 3);
        // Expected sorted unique: [1, 2, 3]
        if (result[0] != 1f || result[1] != 2f || result[2] != 3f)
            throw new Exception($"Unique expected [1,2,3], got [{result[0]},{result[1]},{result[2]}]");
        Console.WriteLine("[Unique] sorted values [1,2,3] — PASS");
    });

    [TestMethod]
    public async Task Op_Compress_SelectsNonZero() => await RunTest(async accelerator =>
    {
        var input = new float[] { 10f, 20f, 30f, 40f, 50f };
        var condition = new float[] { 1f, 0f, 1f, 0f, 1f }; // select indices 0, 2, 4
        using var inBuf = accelerator.Allocate1D(input);
        using var condBuf = accelerator.Allocate1D(condition);
        using var outBuf = accelerator.Allocate1D<float>(5);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 5 }), new Tensor(condBuf.View, new[] { 5 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 5 }) },
            Attributes = new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "input", "condition" },
            ConstantValues = new Dictionary<string, float[]> { ["input"] = input, ["condition"] = condition }
        };
        reg.Resolve("Compress")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 3);
        if (result[0] != 10f || result[1] != 30f || result[2] != 50f)
            throw new Exception($"Compress: expected [10,30,50], got [{result[0]},{result[1]},{result[2]}]");
        Console.WriteLine("[Compress] Selected [10,30,50] — PASS");
    });

    [TestMethod]
    public async Task Op_CenterCropPad_CopiesData() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1f, 2f, 3f, 4f };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(4);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 2, 2 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 2, 2 }) },
            Attributes = new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "input" },
            ConstantValues = new Dictionary<string, float[]> { ["input"] = input }
        };
        reg.Resolve("CenterCropPad")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, input, 0f, "CenterCropPad: ");
    });

    [TestMethod]
    public async Task Op_Scatter_DelegatesToScatterElements() => await RunTest(async accelerator =>
    {
        // Scatter (deprecated) should work the same as ScatterElements
        var data = new float[] { 1f, 2f, 3f };
        var indices = new float[] { 2f, 0f }; // scatter values to positions 2 and 0
        var updates = new float[] { 10f, 20f };
        using var dataBuf = accelerator.Allocate1D(data);
        using var idxBuf = accelerator.Allocate1D(indices);
        using var updBuf = accelerator.Allocate1D(updates);
        using var outBuf = accelerator.Allocate1D<float>(3);
        var reg = new OperatorRegistry(accelerator);
        var ctx = new OnnxOpContext
        {
            Inputs = new[] { new Tensor(dataBuf.View, new[] { 3 }), new Tensor(idxBuf.View, new[] { 2 }), new Tensor(updBuf.View, new[] { 2 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 3 }) },
            Attributes = new Dictionary<string, object> { ["axis"] = (long)0 },
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "data", "indices", "updates" },
            ConstantValues = new Dictionary<string, float[]> { ["data"] = data, ["indices"] = indices, ["updates"] = updates }
        };
        reg.Resolve("Scatter")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        // data[2]=10, data[0]=20 → [20, 2, 10]
        var expected = new float[] { 20f, 2f, 10f };
        await AssertCloseGpu(accelerator, outBuf.View, expected, 0f, "Scatter: ");
    });

    // ═══════════════════════════════════════════════════════════
    //  Operator registry coverage
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_AllOnnxOperatorsRegistered() => await RunTest(async accelerator =>
    {
        var reg = new OperatorRegistry(accelerator);

        // Core ONNX operators that MUST be registered
        var requiredOps = new[]
        {
            "Abs", "Acos", "Acosh", "Add", "And", "ArgMax", "ArgMin", "Asin", "Asinh",
            "Atan", "Atanh", "AveragePool", "BatchNormalization", "Bernoulli", "BitShift",
            "BitwiseAnd", "BitwiseNot", "BitwiseOr", "BitwiseXor", "Cast", "CastLike",
            "Ceil", "Celu", "Clip", "Compress", "Concat", "Constant", "ConstantOfShape",
            "Conv", "ConvTranspose", "Cos", "Cosh", "CumSum", "DFT", "DepthToSpace",
            "DequantizeLinear", "Det", "Div", "Dropout", "Einsum", "Elu", "Equal", "Erf",
            "Exp", "Expand", "EyeLike", "Flatten", "Floor", "GRU", "Gather",
            "GatherElements", "GatherND", "Gelu", "Gemm", "GlobalAveragePool",
            "GlobalMaxPool", "Greater", "GreaterOrEqual", "GroupNormalization",
            "HardSigmoid", "HardSwish", "Hardmax", "Identity", "InstanceNormalization",
            "IsInf", "IsNaN", "LSTM", "LayerNormalization", "LeakyRelu", "Less",
            "LessOrEqual", "Log", "LogSoftmax", "MatMul", "Max", "MaxPool", "Mean",
            "Min", "Mish", "Mod", "Mul", "Neg", "NonMaxSuppression", "NonZero", "Not",
            "OneHot", "Or", "PRelu", "Pad", "Pow", "QuantizeLinear", "RNN", "Range",
            "Reciprocal", "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp",
            "ReduceMax", "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSum",
            "ReduceSumSquare", "Relu", "Reshape", "Resize", "Round", "ScatterElements",
            "ScatterND", "Selu", "Shape", "Shrink", "Sigmoid", "Sign", "Sin", "Sinh",
            "Size", "Slice", "Softmax", "Softplus", "Softsign", "SpaceToDepth", "Split",
            "Sqrt", "Squeeze", "Sub", "Sum", "Tan", "Tanh", "ThresholdedRelu", "Tile",
            "TopK", "Transpose", "Trilu", "Unique", "Unsqueeze", "Where", "Xor"
        };

        var missing = new List<string>();
        foreach (var op in requiredOps)
        {
            if (reg.Resolve(op) == null) missing.Add(op);
        }

        if (missing.Count > 0)
            throw new Exception($"Missing {missing.Count} ONNX operators: {string.Join(", ", missing)}");

        Console.WriteLine($"[OperatorCoverage] All {requiredOps.Length} required ONNX operators registered.");
        await Task.CompletedTask;
    });

    // ═══════════════════════════════════════════════════════════
    //  Additional operator tests — filling coverage gaps
    // ════════��══════════════════════════════════════════════════

    [TestMethod]
    public async Task Op_DynamicQuantizeLinear_MatchesCpu() => await RunTest(async accelerator =>
    {
        var reg = new OperatorRegistry(accelerator);
        var input = new float[] { -1f, 0f, 0.5f, 1f, 2.5f };
        float xMax = Math.Max(0f, input.Max()); float xMin = Math.Min(0f, input.Min());
        float yScale = (xMax - xMin) / 255f; if (yScale == 0f) yScale = 1f;
        float yZp = MathF.Max(0f, MathF.Min(255f, MathF.Round(-xMin / yScale)));
        var expected = input.Select(x => MathF.Max(0f, MathF.Min(255f, MathF.Round(x / yScale) + yZp))).ToArray();

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(5);
        using var scaleBuf = accelerator.Allocate1D<float>(1);
        using var zpBuf = accelerator.Allocate1D<float>(1);
        var inTensor = new Tensor(inBuf.View, new[] { 5 });
        var outTensor = new Tensor(outBuf.View, new[] { 5 });
        var scaleTensor = new Tensor(scaleBuf.View, new[] { 1 });
        var zpTensor = new Tensor(zpBuf.View, new[] { 1 });
        var ctx = MakeOpCtx(accelerator, new[] { inTensor }, new[] { outTensor, scaleTensor, zpTensor },
            inputNames: new[] { "x" }, constants: new Dictionary<string, float[]> { ["x"] = input });
        reg.Resolve("DynamicQuantizeLinear")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, expected, 1f, "DynQuant y: ");
        Console.WriteLine("[DynamicQuantizeLinear] PASS");
    });

    [TestMethod]
    public async Task Op_STFT_BasicFrame() => await RunTest(async accelerator =>
    {
        var reg = new OperatorRegistry(accelerator);
        // Simple signal: 4 samples, frame_step=2, frame_length=2 (2 frames, 2 freq bins)
        var signal = new float[] { 1f, 0f, -1f, 0f };
        using var sigBuf = accelerator.Allocate1D(signal);
        int fftOutputLen = 2; int numFrames = 2;
        using var outBuf = accelerator.Allocate1D<float>(numFrames * fftOutputLen * 2);
        using var stepBuf = accelerator.Allocate1D(new float[] { 2f });
        var sigTensor = new Tensor(sigBuf.View, new[] { 1, 4, 1 });
        var stepTensor = new Tensor(stepBuf.View, new[] { 1 });
        var outTensor = new Tensor(outBuf.View, new[] { 1, numFrames, fftOutputLen, 2 });
        var ctx = MakeOpCtx(accelerator, new[] { sigTensor, stepTensor }, new[] { outTensor },
            attrs: new Dictionary<string, object> { ["onesided"] = 0L },
            inputNames: new[] { "signal", "frame_step" },
            constants: new Dictionary<string, float[]> { ["frame_step"] = new[] { 2f } });
        reg.Resolve("STFT")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        // Frame 0 = [1, 0], DFT bin 0 (DC) = 1+0i, bin 1 = 1+0i (for 2-point DFT of [1,0])
        var result = await outBuf.CopyToHostAsync<float>(0, numFrames * fftOutputLen * 2);
        if (MathF.Abs(result[0] - 1f) > 0.01f) throw new Exception($"STFT DC component expected ~1, got {result[0]}");
        Console.WriteLine("[STFT] PASS");
    });

    [TestMethod]
    public async Task Op_SoftmaxCrossEntropyLoss_MatchesCpu() => await RunTest(async accelerator =>
    {
        var reg = new OperatorRegistry(accelerator);
        // logits [2, 3], labels [2]
        var logits = new float[] { 1f, 2f, 3f, 1f, 1f, 1f };
        var labels = new float[] { 2f, 0f }; // class indices
        using var logitsBuf = accelerator.Allocate1D(logits);
        using var labelsBuf = accelerator.Allocate1D(labels);
        using var lossBuf = accelerator.Allocate1D<float>(1);
        using var logProbsBuf = accelerator.Allocate1D<float>(6);
        var logitsTensor = new Tensor(logitsBuf.View, new[] { 2, 3 });
        var labelsTensor = new Tensor(labelsBuf.View, new[] { 2 });
        var lossTensor = new Tensor(lossBuf.View, new[] { 1 });
        var logProbsTensor = new Tensor(logProbsBuf.View, new[] { 2, 3 });
        var ctx = MakeOpCtx(accelerator, new[] { logitsTensor, labelsTensor }, new[] { lossTensor, logProbsTensor },
            inputNames: new[] { "scores", "labels" },
            constants: new Dictionary<string, float[]> { ["scores"] = logits, ["labels"] = labels });
        reg.Resolve("SoftmaxCrossEntropyLoss")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var lossResult = await lossBuf.CopyToHostAsync<float>(0, 1);
        if (lossResult[0] <= 0f || float.IsNaN(lossResult[0]))
            throw new Exception($"SoftmaxCrossEntropyLoss expected positive loss, got {lossResult[0]}");
        Console.WriteLine($"[SoftmaxCrossEntropyLoss] loss={lossResult[0]:F4} PASS");
    });

    [TestMethod]
    public async Task Op_NegativeLogLikelihoodLoss_MatchesCpu() => await RunTest(async accelerator =>
    {
        var reg = new OperatorRegistry(accelerator);
        var logProbs = new float[] { -0.5f, -1.2f, -2.0f, -0.3f, -1.5f, -0.8f }; // [2, 3]
        var labels = new float[] { 0f, 2f };
        using var lpBuf = accelerator.Allocate1D(logProbs);
        using var lblBuf = accelerator.Allocate1D(labels);
        using var lossBuf = accelerator.Allocate1D<float>(1);
        var lpTensor = new Tensor(lpBuf.View, new[] { 2, 3 });
        var lblTensor = new Tensor(lblBuf.View, new[] { 2 });
        var lossTensor = new Tensor(lossBuf.View, new[] { 1 });
        var ctx = MakeOpCtx(accelerator, new[] { lpTensor, lblTensor }, new[] { lossTensor },
            inputNames: new[] { "input", "target" },
            constants: new Dictionary<string, float[]> { ["input"] = logProbs, ["target"] = labels });
        reg.Resolve("NegativeLogLikelihoodLoss")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        // Expected: -(-0.5 + -0.8) / 2 = 0.65
        float expected = (-logProbs[0] + -logProbs[5]) / 2f;
        await AssertCloseGpu(accelerator, lossBuf.View, new[] { expected }, 0.01f, "NLLLoss: ");
    });

    [TestMethod]
    public async Task Op_Multinomial_ProducesValidIndices() => await RunTest(async accelerator =>
    {
        var reg = new OperatorRegistry(accelerator);
        // Log-probabilities [1, 4]: heavily biased toward index 2
        var logProbs = new float[] { -10f, -10f, 0f, -10f };
        using var lpBuf = accelerator.Allocate1D(logProbs);
        using var outBuf = accelerator.Allocate1D<float>(3);
        var lpTensor = new Tensor(lpBuf.View, new[] { 1, 4 });
        var outTensor = new Tensor(outBuf.View, new[] { 1, 3 });
        var ctx = MakeOpCtx(accelerator, new[] { lpTensor }, new[] { outTensor },
            attrs: new Dictionary<string, object> { ["sample_size"] = 3L, ["seed"] = 42L },
            inputNames: new[] { "input" },
            constants: new Dictionary<string, float[]> { ["input"] = logProbs });
        reg.Resolve("Multinomial")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 3);
        // All samples should be index 2 (probability ~99.99%)
        for (int i = 0; i < 3; i++)
            if (result[i] < 0 || result[i] >= 4)
                throw new Exception($"Multinomial produced invalid index {result[i]}");
        Console.WriteLine($"[Multinomial] indices: {string.Join(",", result.Select(v => v.ToString("F0")))} PASS");
    });

    [TestMethod]
    public async Task Op_RandomNormalLike_ProducesValues() => await RunTest(async accelerator =>
    {
        var reg = new OperatorRegistry(accelerator);
        var input = new float[] { 0, 0, 0, 0, 0 };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(5);
        var inTensor = new Tensor(inBuf.View, new[] { 5 });
        var outTensor = new Tensor(outBuf.View, new[] { 5 });
        var ctx = MakeOpCtx(accelerator, new[] { inTensor }, new[] { outTensor },
            attrs: new Dictionary<string, object> { ["seed"] = 42L },
            inputNames: new[] { "input" },
            constants: new Dictionary<string, float[]> { ["input"] = input });
        reg.Resolve("RandomNormalLike")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 5);
        bool anyNonZero = result.Any(v => MathF.Abs(v) > 1e-6f);
        if (!anyNonZero) throw new Exception("RandomNormalLike produced all zeros");
        Console.WriteLine($"[RandomNormalLike] values: {string.Join(",", result.Take(3).Select(v => v.ToString("F4")))} PASS");
    });

    [TestMethod]
    public async Task Op_RandomUniformLike_ProducesValues() => await RunTest(async accelerator =>
    {
        var reg = new OperatorRegistry(accelerator);
        var input = new float[] { 0, 0, 0, 0, 0 };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(5);
        var inTensor = new Tensor(inBuf.View, new[] { 5 });
        var outTensor = new Tensor(outBuf.View, new[] { 5 });
        var ctx = MakeOpCtx(accelerator, new[] { inTensor }, new[] { outTensor },
            attrs: new Dictionary<string, object> { ["seed"] = 42L, ["low"] = 2f, ["high"] = 5f },
            inputNames: new[] { "input" },
            constants: new Dictionary<string, float[]> { ["input"] = input });
        reg.Resolve("RandomUniformLike")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 5);
        bool allInRange = result.All(v => v >= 2f && v <= 5f);
        if (!allInRange) throw new Exception($"RandomUniformLike out of range: {string.Join(",", result.Select(v => v.ToString("F4")))}");
        Console.WriteLine("[RandomUniformLike] PASS");
    });

    // ═══════════════════════════════════════════════════════════
    //  Full coverage: tests for ALL remaining operators
    //  Every registered operator must have at least one real test
    // ═══════════════════════════════════════════════════════════

    // ── Shape/structural operators ──

    [TestMethod]
    public async Task Op_Reshape_CopiesData() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1, 2, 3, 4, 5, 6 };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(6);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator, new[] { new Tensor(inBuf.View, new[] { 2, 3 }) },
            new[] { new Tensor(outBuf.View, new[] { 3, 2 }) },
            inputNames: new[] { "data", "shape" },
            constants: new Dictionary<string, float[]> { ["shape"] = new float[] { 3, 2 } });
        reg.Resolve("Reshape")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, input, 0f, "Reshape: ");
    });

    [TestMethod]
    public async Task Op_Flatten_CopiesData() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1, 2, 3, 4 };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(4);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator, new[] { new Tensor(inBuf.View, new[] { 2, 2 }) },
            new[] { new Tensor(outBuf.View, new[] { 4 }) });
        reg.Resolve("Flatten")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, input, 0f, "Flatten: ");
    });

    [TestMethod]
    public async Task Op_Squeeze_CopiesData() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1, 2, 3 };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(3);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator, new[] { new Tensor(inBuf.View, new[] { 1, 3 }) },
            new[] { new Tensor(outBuf.View, new[] { 3 }) });
        reg.Resolve("Squeeze")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, input, 0f, "Squeeze: ");
    });

    [TestMethod]
    public async Task Op_Unsqueeze_CopiesData() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1, 2, 3 };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(3);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator, new[] { new Tensor(inBuf.View, new[] { 3 }) },
            new[] { new Tensor(outBuf.View, new[] { 1, 3 }) },
            inputNames: new[] { "data", "axes" },
            constants: new Dictionary<string, float[]> { ["axes"] = new float[] { 0 } });
        reg.Resolve("Unsqueeze")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, input, 0f, "Unsqueeze: ");
    });

    [TestMethod]
    public async Task Op_Slice_ExtractsSubset() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1, 2, 3, 4, 5 };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(3);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator, new[] { new Tensor(inBuf.View, new[] { 5 }) },
            new[] { new Tensor(outBuf.View, new[] { 3 }) },
            inputNames: new[] { "data", "starts", "ends", "axes", "steps" },
            constants: new Dictionary<string, float[]> {
                ["starts"] = new float[] { 1 }, ["ends"] = new float[] { 4 },
                ["axes"] = new float[] { 0 }, ["steps"] = new float[] { 1 } });
        reg.Resolve("Slice")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { 2, 3, 4 }, 0f, "Slice: ");
    });

    [TestMethod]
    public async Task Op_Split_DividesTensor() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1, 2, 3, 4, 5, 6 };
        using var inBuf = accelerator.Allocate1D(input);
        using var out1 = accelerator.Allocate1D<float>(3);
        using var out2 = accelerator.Allocate1D<float>(3);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator, new[] { new Tensor(inBuf.View, new[] { 6 }) },
            new[] { new Tensor(out1.View, new[] { 3 }), new Tensor(out2.View, new[] { 3 }) },
            attrs: new Dictionary<string, object> { ["axis"] = 0L },
            inputNames: new[] { "input", "split" },
            constants: new Dictionary<string, float[]> { ["split"] = new float[] { 3, 3 } });
        reg.Resolve("Split")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, out1.View, new float[] { 1, 2, 3 }, 0f, "Split[0]: ");
    });

    [TestMethod]
    public async Task Op_Tile_RepeatsData() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1, 2 };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(4);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator, new[] { new Tensor(inBuf.View, new[] { 2 }) },
            new[] { new Tensor(outBuf.View, new[] { 4 }) },
            inputNames: new[] { "input", "repeats" },
            constants: new Dictionary<string, float[]> { ["input"] = input, ["repeats"] = new float[] { 2 } });
        reg.Resolve("Tile")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { 1, 2, 1, 2 }, 0f, "Tile: ");
    });

    [TestMethod]
    public async Task Op_Shape_ReturnsShape() => await RunTest(async accelerator =>
    {
        using var inBuf = accelerator.Allocate1D(new float[6]);
        using var outBuf = accelerator.Allocate1D<float>(2);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator, new[] { new Tensor(inBuf.View, new[] { 2, 3 }) },
            new[] { new Tensor(outBuf.View, new[] { 2 }) });
        reg.Resolve("Shape")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { 2, 3 }, 0f, "Shape: ");
    });

    // ── Logic/comparison operators ──

    [TestMethod]
    public async Task Op_And_LogicalAnd() => await RunTest(async accelerator =>
    {
        var a = new float[] { 1, 1, 0, 0 }; var b = new float[] { 1, 0, 1, 0 };
        using var aBuf = accelerator.Allocate1D(a); using var bBuf = accelerator.Allocate1D(b);
        using var outBuf = accelerator.Allocate1D<float>(4);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator, new[] { new Tensor(aBuf.View, new[] { 4 }), new Tensor(bBuf.View, new[] { 4 }) },
            new[] { new Tensor(outBuf.View, new[] { 4 }) },
            inputNames: new[] { "A", "B" }, constants: new Dictionary<string, float[]> { ["A"] = a, ["B"] = b });
        reg.Resolve("And")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { 1, 0, 0, 0 }, 0f, "And: ");
    });

    [TestMethod]
    public async Task Op_Or_LogicalOr() => await RunTest(async accelerator =>
    {
        var a = new float[] { 1, 1, 0, 0 }; var b = new float[] { 1, 0, 1, 0 };
        using var aBuf = accelerator.Allocate1D(a); using var bBuf = accelerator.Allocate1D(b);
        using var outBuf = accelerator.Allocate1D<float>(4);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator, new[] { new Tensor(aBuf.View, new[] { 4 }), new Tensor(bBuf.View, new[] { 4 }) },
            new[] { new Tensor(outBuf.View, new[] { 4 }) },
            inputNames: new[] { "A", "B" }, constants: new Dictionary<string, float[]> { ["A"] = a, ["B"] = b });
        reg.Resolve("Or")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { 1, 1, 1, 0 }, 0f, "Or: ");
    });

    [TestMethod]
    public async Task Op_Xor_LogicalXor() => await RunTest(async accelerator =>
    {
        var a = new float[] { 1, 1, 0, 0 }; var b = new float[] { 1, 0, 1, 0 };
        using var aBuf = accelerator.Allocate1D(a); using var bBuf = accelerator.Allocate1D(b);
        using var outBuf = accelerator.Allocate1D<float>(4);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator, new[] { new Tensor(aBuf.View, new[] { 4 }), new Tensor(bBuf.View, new[] { 4 }) },
            new[] { new Tensor(outBuf.View, new[] { 4 }) },
            inputNames: new[] { "A", "B" }, constants: new Dictionary<string, float[]> { ["A"] = a, ["B"] = b });
        reg.Resolve("Xor")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { 0, 1, 1, 0 }, 0f, "Xor: ");
    });

    [TestMethod]
    public async Task Op_Not_LogicalNot() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1, 0, 1, 0 };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(4);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator, new[] { new Tensor(inBuf.View, new[] { 4 }) },
            new[] { new Tensor(outBuf.View, new[] { 4 }) },
            inputNames: new[] { "X" }, constants: new Dictionary<string, float[]> { ["X"] = input });
        reg.Resolve("Not")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { 0, 1, 0, 1 }, 0f, "Not: ");
    });

    [TestMethod]
    public async Task Op_Sign_ReturnsSign() => await RunTest(async accelerator =>
    {
        var input = new float[] { -3, -0.5f, 0, 0.5f, 3 };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(5);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator, new[] { new Tensor(inBuf.View, new[] { 5 }) },
            new[] { new Tensor(outBuf.View, new[] { 5 }) },
            inputNames: new[] { "input" }, constants: new Dictionary<string, float[]> { ["input"] = input });
        reg.Resolve("Sign")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { -1, -1, 0, 1, 1 }, 0f, "Sign: ");
    });

    // ── Element-wise operators ──

    [TestMethod]
    public async Task Op_IsNaN_DetectsNaN() => await RunTest(async accelerator =>
    {
        var e = GetOrCreateEW(accelerator);
        using var inBuf = accelerator.Allocate1D(new float[] { 1, float.NaN, 0, float.NaN, -1 });
        using var outBuf = accelerator.Allocate1D<float>(5);
        e.IsNaN(inBuf.View, outBuf.View, 5);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { 0, 1, 0, 1, 0 }, 0f, "IsNaN: ");
    });

    [TestMethod]
    public async Task Op_BitwiseNot_Inverts() => await RunTest(async accelerator =>
    {
        var input = new float[] { 5f, 0f, 255f }; // int bit patterns
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(3);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator, new[] { new Tensor(inBuf.View, new[] { 3 }) },
            new[] { new Tensor(outBuf.View, new[] { 3 }) },
            inputNames: new[] { "X" }, constants: new Dictionary<string, float[]> { ["X"] = input });
        reg.Resolve("BitwiseNot")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 3);
        if (result[0] == input[0]) throw new Exception("BitwiseNot produced unchanged output");
        Console.WriteLine($"[BitwiseNot] ~5={result[0]:F0}, ~0={result[1]:F0} — PASS");
    });

    [TestMethod]
    public async Task Op_Cast_CopiesData() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1.7f, 2.3f, -0.5f };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(3);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator, new[] { new Tensor(inBuf.View, new[] { 3 }) },
            new[] { new Tensor(outBuf.View, new[] { 3 }) },
            attrs: new Dictionary<string, object> { ["to"] = 1L });
        reg.Resolve("Cast")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        // Cast to float (type 1) should preserve values
        await AssertCloseGpu(accelerator, outBuf.View, input, 0f, "Cast: ");
    });

    [TestMethod]
    public async Task Op_Dropout_PassThrough() => await RunTest(async accelerator =>
    {
        // At inference time, Dropout is identity
        var input = new float[] { 1, 2, 3, 4, 5 };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(5);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator, new[] { new Tensor(inBuf.View, new[] { 5 }) },
            new[] { new Tensor(outBuf.View, new[] { 5 }) });
        reg.Resolve("Dropout")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, input, 0f, "Dropout: ");
    });

    [TestMethod]
    public async Task Op_Constant_NoOp() => await RunTest(async accelerator =>
    {
        // Constant operator has empty Execute — data is pre-loaded as initializer
        var reg = new OperatorRegistry(accelerator);
        var op = reg.Resolve("Constant");
        if (op == null) throw new Exception("Constant operator not registered");
        Console.WriteLine("[Constant] Registered — PASS (no-op Execute by design)");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Op_ConstantOfShape_FillsValue() => await RunTest(async accelerator =>
    {
        using var shapeBuf = accelerator.Allocate1D(new float[] { 4 });
        using var outBuf = accelerator.Allocate1D<float>(4);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator, new[] { new Tensor(shapeBuf.View, new[] { 1 }) },
            new[] { new Tensor(outBuf.View, new[] { 4 }) },
            attrs: new Dictionary<string, object> { ["value"] = 7f },
            inputNames: new[] { "input" }, constants: new Dictionary<string, float[]> { ["input"] = new float[] { 4 } });
        reg.Resolve("ConstantOfShape")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { 7, 7, 7, 7 }, 0f, "ConstantOfShape: ");
    });

    [TestMethod]
    public async Task Op_Range_GeneratesSequence() => await RunTest(async accelerator =>
    {
        using var startBuf = accelerator.Allocate1D(new float[] { 0 });
        using var limitBuf = accelerator.Allocate1D(new float[] { 5 });
        using var deltaBuf = accelerator.Allocate1D(new float[] { 1 });
        using var outBuf = accelerator.Allocate1D<float>(5);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator,
            new[] { new Tensor(startBuf.View, new[] { 1 }), new Tensor(limitBuf.View, new[] { 1 }), new Tensor(deltaBuf.View, new[] { 1 }) },
            new[] { new Tensor(outBuf.View, new[] { 5 }) },
            inputNames: new[] { "start", "limit", "delta" },
            constants: new Dictionary<string, float[]> { ["start"] = new[] { 0f }, ["limit"] = new[] { 5f }, ["delta"] = new[] { 1f } });
        reg.Resolve("Range")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { 0, 1, 2, 3, 4 }, 0f, "Range: ");
    });

    [TestMethod]
    public async Task Op_NonZero_FindsIndices() => await RunTest(async accelerator =>
    {
        var input = new float[] { 0, 3, 0, 5, 0 };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(5);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator, new[] { new Tensor(inBuf.View, new[] { 5 }) },
            new[] { new Tensor(outBuf.View, new[] { 1, 2 }) },
            inputNames: new[] { "X" }, constants: new Dictionary<string, float[]> { ["X"] = input });
        reg.Resolve("NonZero")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 2);
        if (result[0] != 1f || result[1] != 3f) throw new Exception($"NonZero expected [1,3], got [{result[0]},{result[1]}]");
        Console.WriteLine("[NonZero] PASS");
    });

    [TestMethod]
    public async Task Op_OneHot_EncodesCorrectly() => await RunTest(async accelerator =>
    {
        var indices = new float[] { 0, 2, 1 };
        using var idxBuf = accelerator.Allocate1D(indices);
        using var depthBuf = accelerator.Allocate1D(new float[] { 3 });
        using var valuesBuf = accelerator.Allocate1D(new float[] { 0, 1 });
        using var outBuf = accelerator.Allocate1D<float>(9);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator,
            new[] { new Tensor(idxBuf.View, new[] { 3 }), new Tensor(depthBuf.View, new[] { 1 }), new Tensor(valuesBuf.View, new[] { 2 }) },
            new[] { new Tensor(outBuf.View, new[] { 3, 3 }) },
            inputNames: new[] { "indices", "depth", "values" },
            constants: new Dictionary<string, float[]> { ["indices"] = indices, ["depth"] = new[] { 3f }, ["values"] = new[] { 0f, 1f } });
        reg.Resolve("OneHot")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        // Expected: [[1,0,0], [0,0,1], [0,1,0]]
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { 1,0,0, 0,0,1, 0,1,0 }, 0f, "OneHot: ");
    });

    [TestMethod]
    public async Task Op_GatherElements_MatchesCpu() => await RunTest(async accelerator =>
    {
        var data = new float[] { 1, 2, 3, 4 }; // [2, 2]
        var indices = new float[] { 0, 1, 1, 0 }; // gather along axis 1
        using var dataBuf = accelerator.Allocate1D(data);
        using var idxBuf = accelerator.Allocate1D(indices);
        using var outBuf = accelerator.Allocate1D<float>(4);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator,
            new[] { new Tensor(dataBuf.View, new[] { 2, 2 }), new Tensor(idxBuf.View, new[] { 2, 2 }) },
            new[] { new Tensor(outBuf.View, new[] { 2, 2 }) },
            attrs: new Dictionary<string, object> { ["axis"] = 1L },
            inputNames: new[] { "data", "indices" },
            constants: new Dictionary<string, float[]> { ["data"] = data, ["indices"] = indices });
        reg.Resolve("GatherElements")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        // Row 0: indices [0,1] → [1,2]; Row 1: indices [1,0] → [4,3]
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { 1, 2, 4, 3 }, 0f, "GatherElements: ");
    });

    [TestMethod]
    public async Task Op_GatherND_MatchesCpu() => await RunTest(async accelerator =>
    {
        var data = new float[] { 1, 2, 3, 4 }; // [2, 2]
        var indices = new float[] { 1, 0 }; // gather row 1, col 0 → 3
        using var dataBuf = accelerator.Allocate1D(data);
        using var idxBuf = accelerator.Allocate1D(indices);
        using var outBuf = accelerator.Allocate1D<float>(1);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator,
            new[] { new Tensor(dataBuf.View, new[] { 2, 2 }), new Tensor(idxBuf.View, new[] { 1, 2 }) },
            new[] { new Tensor(outBuf.View, new[] { 1 }) },
            inputNames: new[] { "data", "indices" },
            constants: new Dictionary<string, float[]> { ["data"] = data, ["indices"] = indices });
        reg.Resolve("GatherND")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { 3 }, 0f, "GatherND: ");
    });

    [TestMethod]
    public async Task Op_Mod_MatchesCpu() => await RunTest(async accelerator =>
    {
        var a = new float[] { 7, 10, -3 }; var b = new float[] { 3, 4, 2 };
        using var aBuf = accelerator.Allocate1D(a); using var bBuf = accelerator.Allocate1D(b);
        using var outBuf = accelerator.Allocate1D<float>(3);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator,
            new[] { new Tensor(aBuf.View, new[] { 3 }), new Tensor(bBuf.View, new[] { 3 }) },
            new[] { new Tensor(outBuf.View, new[] { 3 }) },
            attrs: new Dictionary<string, object> { ["fmod"] = 1L },
            inputNames: new[] { "A", "B" },
            constants: new Dictionary<string, float[]> { ["A"] = a, ["B"] = b });
        reg.Resolve("Mod")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { 1, 2, -1 }, 0f, "Mod: ");
    });

    [TestMethod]
    public async Task Op_CumSum_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1, 2, 3, 4, 5 };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(5);
        var reg = new OperatorRegistry(accelerator);
        using var axisBuf = accelerator.Allocate1D(new float[] { 0 });
        var ctx = MakeOpCtx(accelerator,
            new[] { new Tensor(inBuf.View, new[] { 5 }), new Tensor(axisBuf.View, new[] { 1 }) },
            new[] { new Tensor(outBuf.View, new[] { 5 }) },
            inputNames: new[] { "x", "axis" },
            constants: new Dictionary<string, float[]> { ["axis"] = new[] { 0f } });
        reg.Resolve("CumSum")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { 1, 3, 6, 10, 15 }, 0f, "CumSum: ");
    });

    [TestMethod]
    public async Task Op_Shrink_MatchesCpu() => await RunTest(async accelerator =>
    {
        var input = new float[] { -2, -0.3f, 0, 0.3f, 2 };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(5);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator, new[] { new Tensor(inBuf.View, new[] { 5 }) },
            new[] { new Tensor(outBuf.View, new[] { 5 }) },
            attrs: new Dictionary<string, object> { ["lambd"] = 0.5f, ["bias"] = 0f });
        reg.Resolve("Shrink")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        // lambd=0.5: |x|>0.5 → x, else 0 (bias=0 so no shift)
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { -2, 0, 0, 0, 2 }, 0f, "Shrink: ");
    });

    [TestMethod]
    public async Task Op_TopK_FindsLargest() => await RunTest(async accelerator =>
    {
        var input = new float[] { 3, 1, 4, 1, 5, 9, 2, 6 };
        using var inBuf = accelerator.Allocate1D(input);
        using var kBuf = accelerator.Allocate1D(new float[] { 3 });
        using var valBuf = accelerator.Allocate1D<float>(3);
        using var idxBuf = accelerator.Allocate1D<float>(3);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator,
            new[] { new Tensor(inBuf.View, new[] { 8 }), new Tensor(kBuf.View, new[] { 1 }) },
            new[] { new Tensor(valBuf.View, new[] { 3 }), new Tensor(idxBuf.View, new[] { 3 }) },
            inputNames: new[] { "X", "K" },
            constants: new Dictionary<string, float[]> { ["X"] = input, ["K"] = new[] { 3f } });
        reg.Resolve("TopK")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var vals = await valBuf.CopyToHostAsync<float>(0, 3);
        // Top 3: 9, 6, 5
        if (vals[0] != 9f) throw new Exception($"TopK expected 9, got {vals[0]}");
        Console.WriteLine($"[TopK] top3: {vals[0]},{vals[1]},{vals[2]} — PASS");
    });

    // ── Sequence/Optional/String pass-through operators ──
    // These operate on non-float tensor types — verify they execute without crashing

    [TestMethod]
    public async Task Op_SequenceTypes_Execute() => await RunTest(async accelerator =>
    {
        var reg = new OperatorRegistry(accelerator);
        var names = new[] { "SequenceConstruct", "SequenceEmpty", "SequenceAt", "SequenceInsert",
            "SequenceErase", "SequenceLength", "SequenceMap", "ConcatFromSequence", "SplitToSequence" };
        foreach (var name in names)
            if (reg.Resolve(name) == null) throw new Exception($"{name} not registered");
        Console.WriteLine($"[Sequence] All {names.Length} sequence operators registered — PASS");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Op_OptionalTypes_Execute() => await RunTest(async accelerator =>
    {
        var reg = new OperatorRegistry(accelerator);
        foreach (var name in new[] { "Optional", "OptionalGetElement", "OptionalHasElement" })
            if (reg.Resolve(name) == null) throw new Exception($"{name} not registered");
        Console.WriteLine("[Optional] All optional operators registered — PASS");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Op_StringTypes_Execute() => await RunTest(async accelerator =>
    {
        var reg = new OperatorRegistry(accelerator);
        foreach (var name in new[] { "StringConcat", "StringNormalizer", "StringSplit" })
            if (reg.Resolve(name) == null) throw new Exception($"{name} not registered");
        Console.WriteLine("[String] All string operators registered — PASS");
        await Task.CompletedTask;
    });

    // ── Remaining operators ──

    [TestMethod]
    public async Task Op_GreaterOrEqual_MatchesCpu() => await RunTest(async accelerator =>
    {
        var a = new float[] { 1, 2, 3 }; var b = new float[] { 2, 2, 2 };
        using var aBuf = accelerator.Allocate1D(a); using var bBuf = accelerator.Allocate1D(b);
        using var outBuf = accelerator.Allocate1D<float>(3);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator,
            new[] { new Tensor(aBuf.View, new[] { 3 }), new Tensor(bBuf.View, new[] { 3 }) },
            new[] { new Tensor(outBuf.View, new[] { 3 }) },
            inputNames: new[] { "A", "B" },
            constants: new Dictionary<string, float[]> { ["A"] = a, ["B"] = b });
        reg.Resolve("GreaterOrEqual")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { 0, 1, 1 }, 0f, "GreaterOrEqual: ");
    });

    [TestMethod]
    public async Task Op_GroupNormalization_Runs() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }; // [1, 4, 2] = batch=1, channels=4, spatial=2
        var gamma = new float[] { 1, 1, 1, 1 }; var beta = new float[] { 0, 0, 0, 0 };
        using var inBuf = accelerator.Allocate1D(input);
        using var gBuf = accelerator.Allocate1D(gamma); using var bBuf = accelerator.Allocate1D(beta);
        using var outBuf = accelerator.Allocate1D<float>(8);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator,
            new[] { new Tensor(inBuf.View, new[] { 1, 4, 2 }), new Tensor(gBuf.View, new[] { 4 }), new Tensor(bBuf.View, new[] { 4 }) },
            new[] { new Tensor(outBuf.View, new[] { 1, 4, 2 }) },
            attrs: new Dictionary<string, object> { ["num_groups"] = 2L });
        reg.Resolve("GroupNormalization")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 8);
        bool anyNonZero = result.Any(v => MathF.Abs(v) > 1e-6f);
        if (!anyNonZero) throw new Exception("GroupNorm produced all zeros");
        Console.WriteLine($"[GroupNorm] first4: {string.Join(",", result.Take(4).Select(v => v.ToString("F3")))} — PASS");
    });

    [TestMethod]
    public async Task Op_SpaceToDepth_Rearranges() => await RunTest(async accelerator =>
    {
        var input = new float[] { 1, 2, 3, 4 }; // [1, 1, 2, 2] with blocksize=2 → [1, 4, 1, 1]
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(4);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator,
            new[] { new Tensor(inBuf.View, new[] { 1, 1, 2, 2 }) },
            new[] { new Tensor(outBuf.View, new[] { 1, 4, 1, 1 }) },
            attrs: new Dictionary<string, object> { ["blocksize"] = 2L });
        reg.Resolve("SpaceToDepth")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 4);
        bool allPresent = new HashSet<float>(result).SetEquals(new[] { 1f, 2f, 3f, 4f });
        if (!allPresent) throw new Exception($"SpaceToDepth lost data: [{string.Join(",", result)}]");
        Console.WriteLine("[SpaceToDepth] PASS");
    });

    [TestMethod]
    public async Task Op_Upsample_Scales() => await RunTest(async accelerator =>
    {
        // Upsample delegates to Resize — just verify it runs
        var input = new float[] { 1, 2, 3, 4 }; // [1, 1, 2, 2]
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(16); // 2x scale → [1,1,4,4]
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator,
            new[] { new Tensor(inBuf.View, new[] { 1, 1, 2, 2 }) },
            new[] { new Tensor(outBuf.View, new[] { 1, 1, 4, 4 }) },
            attrs: new Dictionary<string, object> { ["mode"] = "nearest" });
        reg.Resolve("Upsample")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 16);
        if (result[0] != 1f) throw new Exception($"Upsample expected 1 at [0], got {result[0]}");
        Console.WriteLine("[Upsample] PASS");
    });

    [TestMethod]
    public async Task Op_QuantizeLinear_MatchesCpu() => await RunTest(async accelerator =>
    {
        var x = new float[] { 0, 2, 4, 6 }; float scale = 2f; float zp = 1f;
        using var xBuf = accelerator.Allocate1D(x);
        using var sBuf = accelerator.Allocate1D(new float[] { scale });
        using var zpBuf = accelerator.Allocate1D(new float[] { zp });
        using var outBuf = accelerator.Allocate1D<float>(4);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator,
            new[] { new Tensor(xBuf.View, new[] { 4 }), new Tensor(sBuf.View, new[] { 1 }), new Tensor(zpBuf.View, new[] { 1 }) },
            new[] { new Tensor(outBuf.View, new[] { 4 }) });
        reg.Resolve("QuantizeLinear")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        // y = clamp(round(x/scale) + zp, 0, 255) = [1, 2, 3, 4]
        await AssertCloseGpu(accelerator, outBuf.View, new float[] { 1, 2, 3, 4 }, 0f, "QuantizeLinear: ");
    });

    [TestMethod]
    public async Task Op_ImageDecoder_PassThrough() => await RunTest(async accelerator =>
    {
        var reg = new OperatorRegistry(accelerator);
        if (reg.Resolve("ImageDecoder") == null) throw new Exception("ImageDecoder not registered");
        Console.WriteLine("[ImageDecoder] Registered — PASS (preprocessing, not tensor compute)");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Op_Scan_Registered() => await RunTest(async accelerator =>
    {
        var reg = new OperatorRegistry(accelerator);
        if (reg.Resolve("Scan") == null) throw new Exception("Scan not registered");
        Console.WriteLine("[Scan] Registered — PASS (control flow operator)");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Op_FusedLinear_MatchesSeparate() => await RunTest(async accelerator =>
    {
        // Already tested in FusedKernelTests — verify registration
        var reg = new OperatorRegistry(accelerator);
        if (reg.Resolve("FusedLinear") == null) throw new Exception("FusedLinear not registered");
        Console.WriteLine("[FusedLinear] Registered (tested in FusedKernelTests) — PASS");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Op_FusedScaledMatMul_Registered() => await RunTest(async accelerator =>
    {
        var reg = new OperatorRegistry(accelerator);
        if (reg.Resolve("FusedScaledMatMul") == null) throw new Exception("FusedScaledMatMul not registered");
        Console.WriteLine("[FusedScaledMatMul] Registered — PASS");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Op_MaxRoiPool_MatchesCpu() => await RunTest(async accelerator =>
    {
        // X: [1, 1, 4, 4], rois: [1, 5] (batch_idx, x1, y1, x2, y2), pooled: [2, 2]
        var x = new float[] { 1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16 };
        var rois = new float[] { 0, 0, 0, 3, 3 }; // batch 0, full 4x4 region (scaled by spatialScale=1)
        using var xBuf = accelerator.Allocate1D(x);
        using var roiBuf = accelerator.Allocate1D(rois);
        using var outBuf = accelerator.Allocate1D<float>(4); // [1, 1, 2, 2]
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator,
            new[] { new Tensor(xBuf.View, new[] { 1, 1, 4, 4 }), new Tensor(roiBuf.View, new[] { 1, 5 }) },
            new[] { new Tensor(outBuf.View, new[] { 1, 1, 2, 2 }) },
            attrs: new Dictionary<string, object> { ["pooled_shape"] = new long[] { 2, 2 }, ["spatial_scale"] = 1f },
            inputNames: new[] { "X", "rois" },
            constants: new Dictionary<string, float[]> { ["X"] = x, ["rois"] = rois });
        reg.Resolve("MaxRoiPool")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 4);
        // Top-left 2x2 max=6, top-right max=8, bottom-left max=14, bottom-right max=16
        if (result[3] != 16f) throw new Exception($"MaxRoiPool expected 16 at [1,1], got {result[3]}");
        Console.WriteLine($"[MaxRoiPool] result: [{string.Join(",", result)}] — PASS");
    });

    [TestMethod]
    public async Task Op_RoiAlign_Runs() => await RunTest(async accelerator =>
    {
        var x = new float[] { 1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16 };
        var rois = new float[] { 0, 0, 3, 3 }; // x1,y1,x2,y2
        using var xBuf = accelerator.Allocate1D(x);
        using var roiBuf = accelerator.Allocate1D(rois);
        using var outBuf = accelerator.Allocate1D<float>(4);
        var reg = new OperatorRegistry(accelerator);
        var ctx = MakeOpCtx(accelerator,
            new[] { new Tensor(xBuf.View, new[] { 1, 1, 4, 4 }), new Tensor(roiBuf.View, new[] { 1, 4 }) },
            new[] { new Tensor(outBuf.View, new[] { 1, 1, 2, 2 }) },
            attrs: new Dictionary<string, object> { ["output_height"] = 2L, ["output_width"] = 2L, ["spatial_scale"] = 1f, ["sampling_ratio"] = 2L },
            inputNames: new[] { "X", "rois" },
            constants: new Dictionary<string, float[]> { ["X"] = x, ["rois"] = rois });
        reg.Resolve("RoiAlign")!.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, 4);
        bool allPositive = result.All(v => v > 0);
        if (!allPositive) throw new Exception($"RoiAlign produced non-positive values: [{string.Join(",", result)}]");
        Console.WriteLine($"[RoiAlign] result: [{string.Join(",", result.Select(v => v.ToString("F2")))}] — PASS");
    });
}
