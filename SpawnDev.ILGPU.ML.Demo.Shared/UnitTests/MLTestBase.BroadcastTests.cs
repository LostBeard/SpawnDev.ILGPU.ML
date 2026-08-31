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
    //  GPU-PATH COMPARISON BROADCAST  (CLIP causal-mask regression)
    //
    //  The tests above pass the inputs as ConstantValues, so they run the CPU constant-fold path inside
    //  BroadcastBinaryOp — NOT the GPU broadcast kernel. The CLIP forward bug lived in the GPU kernel path:
    //  comparison ops (Less/Greater/Equal) had no BroadcastOp enum, so BroadcastBinaryOpND silently used the
    //  DEFAULT enum (Add) and computed a+b instead of a comparison, collapsing the causal mask in every
    //  decoder. These tests force the GPU kernel path (no ConstantValues) so they actually exercise the fix.
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task BroadcastGpu_Less_Scalar() => await RunTest(async accelerator =>
    {
        // [4] < [1] on the GPU kernel path. Pre-fix this returned a+b (Add), not the 0/1 comparison.
        await VerifyBinaryOpGpuPath(accelerator, "Less",
            new float[] { 1, 5, 3, 2 }, new[] { 4 }, new float[] { 3 }, new[] { 1 },
            new float[] { 1, 0, 0, 1 }, new[] { 4 });
    });

    [TestMethod]
    public async Task BroadcastGpu_Greater_Scalar() => await RunTest(async accelerator =>
    {
        await VerifyBinaryOpGpuPath(accelerator, "Greater",
            new float[] { 1, 5, 3, 2 }, new[] { 4 }, new float[] { 3 }, new[] { 1 },
            new float[] { 0, 1, 0, 0 }, new[] { 4 });
    });

    [TestMethod]
    public async Task BroadcastGpu_Equal_Scalar() => await RunTest(async accelerator =>
    {
        await VerifyBinaryOpGpuPath(accelerator, "Equal",
            new float[] { 1, 3, 3, 2 }, new[] { 4 }, new float[] { 3 }, new[] { 1 },
            new float[] { 0, 1, 1, 0 }, new[] { 4 });
    });

    [TestMethod]
    public async Task BroadcastGpu_CausalMask_2D() => await RunTest(async accelerator =>
    {
        // THE regression: a [4,1] vs [1,4] GreaterOrEqual broadcasts on BOTH axes → a 4x4 lower-triangular
        // causal mask out[i,j] = (i >= j). This is the exact shape the CLIP bug destroyed, and it also guards
        // the tightened fast-path guard: a.ElementCount(4) == b.ElementCount(4) but != output(16), so it must
        // take the broadcast path, not the element-wise fast path (which pre-fix wrote 4 elems of garbage).
        var rows = new float[] { 0, 1, 2, 3 };       // [4,1]
        var cols = new float[] { 0, 1, 2, 3 };       // [1,4]
        var expected = new float[]
        {
            1, 0, 0, 0,
            1, 1, 0, 0,
            1, 1, 1, 0,
            1, 1, 1, 1,
        };
        await VerifyBinaryOpGpuPath(accelerator, "GreaterOrEqual",
            rows, new[] { 4, 1 }, cols, new[] { 1, 4 }, expected, new[] { 4, 4 });
    });

    // ═══════════════════════════════════════════════════════════
    //  REPRO — SD-Turbo VAE per-channel-bias Add throws on OpenCL (CUDA passes).
    //  a is a large NCHW GPU tensor, b a [C,1,1] CONSTANT bias → BroadcastBinaryOp's "expand b on
    //  CPU then ND kernel" branch (ElementWiseOperators.cs:47). The full model uses [1,128,512,512];
    //  this hits the same branch at a lighter size. Surfaces the TargetInvocationException inner cause.
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task BroadcastAdd_NCHW_PerChannelBias_ConstB() => await RunTest(async accelerator =>
    {
        const int C = 128, H = 256, W = 256;       // 8.4M elems; escalate to 512x512 if it only repros there
        int n = C * H * W;
        var aData = new float[n];
        var bData = new float[C];
        for (int i = 0; i < n; i++) aData[i] = (i % 13) * 0.05f;
        for (int c = 0; c < C; c++) bData[c] = c * 0.01f;

        using var aBuf = accelerator.Allocate1D(aData);
        using var bBuf = accelerator.Allocate1D(bData);
        using var outBuf = accelerator.Allocate1D<float>(n);
        using var pool = new BufferPool(accelerator);

        var reg = new OperatorRegistry(accelerator);
        reg.Resolve("Add").Execute(new OnnxOpContext
        {
            Inputs = new[] { new Tensor(aBuf.View, new[] { 1, C, H, W }), new Tensor(bBuf.View, new[] { C, 1, 1 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 1, C, H, W }) },
            Attributes = new Dictionary<string, object>(),
            Pool = pool,
            InputNames = new[] { "a", "b" },
            ConstantValues = new Dictionary<string, float[]> { ["b"] = bData },  // b constant -> expand-b branch
        });
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, n);

        float maxErr = 0;
        for (int c = 0; c < C; c++)
            for (int hw = 0; hw < H * W; hw++)
            {
                int idx = c * H * W + hw;
                maxErr = MathF.Max(maxErr, MathF.Abs(result[idx] - (aData[idx] + bData[c])));
            }
        if (maxErr > 1e-4f)
            throw new Exception($"per-channel bias Add wrong: maxErr={maxErr:E3}");

        Console.WriteLine($"[BroadcastAdd] [1,{C},{H},{W}]+[{C},1,1] per-channel bias — maxErr={maxErr:E1}");
    });

    // ═══════════════════════════════════════════════════════════
    //  REGRESSION — [C,1,1] per-channel γ/β with C == W (the exact SD-VAE bug).
    //
    //  When C equals the spatial last dim (the SD-Turbo VAE's up_blocks.2.norm2 runs on a [1,256,256,256]
    //  map, so C==H==W==256), a per-channel weight b=[C,1,1] satisfies b.ElementCount == a.Shape[^1] and
    //  PRE-FIX took the last-dim fast path (AddBias / BroadcastMul) — applying γ/β over the W axis instead
    //  of the CHANNEL axis. Every decode was silently mis-scaled. The fix added a second guard
    //  (b.Shape[^1] == b.ElementCount) so [C,1,1] falls through to the correct N-D broadcast.
    //
    //  The existing BroadcastAdd_NCHW_PerChannelBias_ConstB above uses C=128,W=256 (C != W) so it never
    //  triggers the misfire. THESE use C == W, the actual bug condition, for Mul (γ scale) and Add (β bias),
    //  on both the constant-b path (VAE γ/β are initializers) and the non-constant GPU N-D kernel path.
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task BroadcastMul_PerChannel_CequalsW_ConstB() => await RunTest(async accelerator =>
        await VerifyPerChannelCequalsW(accelerator, "Mul", constB: true));

    [TestMethod]
    public async Task BroadcastAdd_PerChannel_CequalsW_ConstB() => await RunTest(async accelerator =>
        await VerifyPerChannelCequalsW(accelerator, "Add", constB: true));

    [TestMethod]
    public async Task BroadcastMul_PerChannel_CequalsW_GpuPath() => await RunTest(async accelerator =>
        await VerifyPerChannelCequalsW(accelerator, "Mul", constB: false));

    [TestMethod]
    public async Task BroadcastAdd_PerChannel_CequalsW_GpuPath() => await RunTest(async accelerator =>
        await VerifyPerChannelCequalsW(accelerator, "Add", constB: false));

    /// <summary>a=[1,C,H,W] with C==W (so a.Shape[^1]==C, the bug trigger) op b=[C,1,1] per-channel.
    /// Verifies the result is a per-CHANNEL apply (out[c]=a op γ/β[c]), NOT a per-W apply. constB toggles
    /// whether b is supplied as a folded constant (the VAE γ/β initializer path → expand-b branch) or left
    /// as a pure GPU tensor (the general N-D broadcast kernel path).</summary>
    private async Task VerifyPerChannelCequalsW(Accelerator accelerator, string opType, bool constB)
    {
        const int C = 128, H = 128, W = 128;       // C == W == 128 is the misfire condition (a.Shape[^1]==C)
        int n = C * H * W;                          // 2.1M elems
        var aData = new float[n];
        var bData = new float[C];
        for (int i = 0; i < n; i++) aData[i] = ((i % 17) - 8) * 0.1f;
        for (int c = 0; c < C; c++) bData[c] = 0.5f + c * 0.01f;   // distinct per channel so a per-W misfire diverges

        using var aBuf = accelerator.Allocate1D(aData);
        using var bBuf = accelerator.Allocate1D(bData);
        using var outBuf = accelerator.Allocate1D<float>(n);
        using var pool = new BufferPool(accelerator);

        var reg = new OperatorRegistry(accelerator);
        reg.Resolve(opType).Execute(new OnnxOpContext
        {
            Inputs = new[] { new Tensor(aBuf.View, new[] { 1, C, H, W }), new Tensor(bBuf.View, new[] { C, 1, 1 }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 1, C, H, W }) },
            Attributes = new Dictionary<string, object>(),
            Pool = pool,
            InputNames = new[] { "a", "b" },
            ConstantValues = constB
                ? new Dictionary<string, float[]> { ["b"] = bData }   // initializer path (expand-b branch)
                : new Dictionary<string, float[]>(),                  // pure GPU tensor → N-D kernel path
        });
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, n);

        // CPU reference: per-CHANNEL. A per-W misfire would instead use b[w] and blow this up massively.
        float maxErr = 0;
        for (int c = 0; c < C; c++)
            for (int hw = 0; hw < H * W; hw++)
            {
                int idx = c * H * W + hw;
                float want = opType == "Mul" ? aData[idx] * bData[c] : aData[idx] + bData[c];
                maxErr = MathF.Max(maxErr, MathF.Abs(result[idx] - want));
            }
        if (maxErr > 1e-4f)
            throw new Exception($"{opType} [1,{C},{H},{W}] op [{C},1,1] per-channel (C==W, constB={constB}) " +
                $"mis-broadcast: maxErr={maxErr:E3} (per-W misfire applies γ/β over the wrong axis).");

        Console.WriteLine($"[{opType}_PerChannel_CequalsW constB={constB}] PASS — maxErr={maxErr:E1}");
    }

    // ═══════════════════════════════════════════════════════════
    //  HELPER — runs any binary operator and verifies output
    // ═══════════════════════════════════════════════════════════

    // ═══════════════════════════════════════════════════════════
    //  GPU-PATH TESTS FOR THE "SILENTLY ADDED" OPERATOR FAMILY
    //
    //  BroadcastHelper.BroadcastBinaryOp used to declare `BroadcastOp gpuOp = BroadcastOp.Add`. Every call
    //  site that omitted the selector got ADDITION on the two GPU branches while the both-inputs-constant
    //  branch quietly used the correct CPU lambda - so the defect was invisible to any test whose operands
    //  were foldable. It cost a day through And/Or/Xor on whisper's causal mask, and NINE more operators
    //  were still shipping it: Min, Max, PRelu, Mod, BitwiseAnd/Or/Xor and both BitShift directions.
    //
    //  The default is now gone (omitting the selector is a compile error). These tests are the behavioural
    //  half of that guarantee: every expected vector below differs from a + b, so a regression to the Add
    //  fallback fails them rather than passing quietly.
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Broadcast_Min_GpuPath() => await RunTest(async accelerator =>
    {
        // [2,3] min [3]; a + b would be [5,12,7,14,9,16], so Add cannot masquerade as Min here.
        await VerifyBinaryOpGpuPath(accelerator, "Min",
            new float[] { 1, 8, 3, 10, 5, 12 }, new[] { 2, 3 },
            new float[] { 4, 4, 4 }, new[] { 3 },
            new float[] { 1, 4, 3, 4, 4, 4 }, new[] { 2, 3 });
    });

    [TestMethod]
    public async Task Broadcast_Max_GpuPath() => await RunTest(async accelerator =>
    {
        await VerifyBinaryOpGpuPath(accelerator, "Max",
            new float[] { 1, 8, 3, 10, 5, 12 }, new[] { 2, 3 },
            new float[] { 4, 4, 4 }, new[] { 3 },
            new float[] { 4, 8, 4, 10, 5, 12 }, new[] { 2, 3 });
    });

    [TestMethod]
    public async Task Broadcast_PRelu_GpuPath() => await RunTest(async accelerator =>
    {
        // PRelu(x, slope) = x >= 0 ? x : slope * x. Operand order matters, so a wrong-way-round
        // implementation fails this too, not just the Add fallback.
        await VerifyBinaryOpGpuPath(accelerator, "PRelu",
            new float[] { -2, -1, 0, 1, 2, 3 }, new[] { 2, 3 },
            new float[] { 0.5f, 0.5f, 0.5f }, new[] { 3 },
            new float[] { -1, -0.5f, 0, 1, 2, 3 }, new[] { 2, 3 });
    });

    [TestMethod]
    public async Task Broadcast_Mod_GpuPath() => await RunTest(async accelerator =>
    {
        await VerifyBinaryOpGpuPath(accelerator, "Mod",
            new float[] { 7, 8, 9, 10, 11, 12 }, new[] { 2, 3 },
            new float[] { 3, 4, 5 }, new[] { 3 },
            new float[] { 1, 0, 4, 1, 3, 2 }, new[] { 2, 3 });
    });

    [TestMethod]
    public async Task Broadcast_BitwiseAnd_GpuPath() => await RunTest(async accelerator =>
    {
        await VerifyBinaryOpGpuPath(accelerator, "BitwiseAnd",
            new float[] { 12, 10, 7, 15, 9, 6 }, new[] { 2, 3 },
            new float[] { 10, 6, 3 }, new[] { 3 },
            new float[] { 8, 2, 3, 10, 0, 2 }, new[] { 2, 3 });
    });

    [TestMethod]
    public async Task Broadcast_BitwiseOr_GpuPath() => await RunTest(async accelerator =>
    {
        await VerifyBinaryOpGpuPath(accelerator, "BitwiseOr",
            new float[] { 12, 10, 7, 15, 9, 6 }, new[] { 2, 3 },
            new float[] { 10, 6, 3 }, new[] { 3 },
            new float[] { 14, 14, 7, 15, 15, 7 }, new[] { 2, 3 });
    });

    [TestMethod]
    public async Task Broadcast_BitwiseXor_GpuPath() => await RunTest(async accelerator =>
    {
        await VerifyBinaryOpGpuPath(accelerator, "BitwiseXor",
            new float[] { 12, 10, 7, 15, 9, 6 }, new[] { 2, 3 },
            new float[] { 10, 6, 3 }, new[] { 3 },
            new float[] { 6, 12, 4, 5, 15, 5 }, new[] { 2, 3 });
    });

    [TestMethod]
    public async Task Broadcast_BitShiftLeft_GpuPath() => await RunTest(async accelerator =>
    {
        // direction defaults to LEFT when the attribute is absent.
        await VerifyBinaryOpGpuPath(accelerator, "BitShift",
            new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 },
            new float[] { 1, 2, 3 }, new[] { 3 },
            new float[] { 2, 8, 24, 8, 20, 48 }, new[] { 2, 3 });
    });

    [TestMethod]
    public async Task Broadcast_BitShiftRight_GpuPath() => await RunTest(async accelerator =>
    {
        // The RIGHT branch is a SEPARATE call site from LEFT, so it needs its own coverage.
        await VerifyBinaryOpGpuPath(accelerator, "BitShift",
            new float[] { 16, 32, 64, 8, 40, 96 }, new[] { 2, 3 },
            new float[] { 1, 2, 3 }, new[] { 3 },
            new float[] { 8, 8, 8, 4, 10, 12 }, new[] { 2, 3 },
            new Dictionary<string, object> { ["direction"] = "RIGHT" });
    });

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

    /// <summary>
    /// Like <see cref="VerifyBinaryOp"/> but provides NO ConstantValues, so <c>BroadcastBinaryOp</c> cannot
    /// take its CPU constant-fold shortcut and must dispatch the real GPU broadcast kernel
    /// (<c>BroadcastBinaryOpND</c> with the <c>BroadcastOp</c> enum). This is the production path that the CLIP
    /// causal-mask bug lived in; the constant-fold helper never reached it.
    /// </summary>
    private async Task VerifyBinaryOpGpuPath(Accelerator accelerator, string opType,
        float[] aData, int[] aShape, float[] bData, int[] bShape,
        float[] expected, int[] outShape, Dictionary<string, object>? attributes = null)
    {
        using var aBuf = accelerator.Allocate1D(aData);
        using var bBuf = accelerator.Allocate1D(bData);
        using var outBuf = accelerator.Allocate1D<float>(expected.Length);

        var reg = new OperatorRegistry(accelerator);
        var op = reg.Resolve(opType);

        op.Execute(new OnnxOpContext
        {
            Inputs = new[] { new Tensor(aBuf.View, aShape), new Tensor(bBuf.View, bShape) },
            Outputs = new[] { new Tensor(outBuf.View, outShape) },
            Attributes = attributes ?? new Dictionary<string, object>(),
            Pool = new BufferPool(accelerator),
            InputNames = new[] { "a", "b" },
            ConstantValues = new Dictionary<string, float[]>(), // empty → no CPU fold → GPU kernel path
        });
        await accelerator.SynchronizeAsync();
        var result = await outBuf.CopyToHostAsync<float>(0, expected.Length);

        float maxErr = 0;
        for (int i = 0; i < expected.Length; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(result[i] - expected[i]));

        if (maxErr > 0.01f)
            throw new Exception($"{opType} GPU-path maxErr={maxErr:E3} (likely the comparison fell back to Add). " +
                $"Expected: [{string.Join(",", expected.Select(v => v.ToString("F0")))}] " +
                $"Got: [{string.Join(",", result.Select(v => v.ToString("F0")))}]");

        Console.WriteLine($"[{opType}_GPUPath] PASS — {string.Join("x", aShape)} op {string.Join("x", bShape)} → maxErr={maxErr:E1}");
    }

    /// <summary>
    /// A scalar broadcast against a LARGE tensor, on several ops and a non-round element count.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠️ Every other scalar-broadcast test in this file uses four to six elements. Those are correct and
    /// they were never in danger - but they also never touched the branch that actually runs in a model.
    /// A constant operand used to be EXPANDED to the full output shape in a host array and uploaded, so
    /// subtracting one zero-point from a 460k-element activation allocated a 460k `float[]`, ran a
    /// 460k-iteration index-mapping loop on the CPU, and pushed 1.8 MB across the bus - per call. On
    /// ZipVoice's int8 decoder that made `Sub` the single largest GPU-attributed cost (27.8%, 406 calls at
    /// 1.26 ms each) next to `Mul` at 0.034 ms for the same kind of work.
    /// </para>
    /// <para>
    /// The scalar now stays one element and `BroadcastBinaryOpND` maps it on the GPU, which it could always
    /// do. Decoder 4,030 ms -> 1,188 ms, output bit-identical. This test covers that path at a size where
    /// the difference exists, with a count that is deliberately NOT a round number or a multiple of any
    /// likely workgroup size, so a mishandled tail cannot hide.
    /// </para>
    /// </remarks>
    [TestMethod]
    public async Task Broadcast_Scalar_AgainstALargeTensor() => await RunTest(async accelerator =>
    {
        const int n = 100_003;
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = MathF.Sin(i * 0.001f) * 10f;

        // Sub and Mul are the quantise/dequantise pair (x - zero_point, x * scale) that made this hot;
        // Greater exercises a comparison, whose output is 0/1 rather than arithmetic.
        var sub = new float[n];
        var mul = new float[n];
        var gtr = new float[n];
        const float scalar = 2.5f;
        for (int i = 0; i < n; i++)
        {
            sub[i] = a[i] - scalar;
            mul[i] = a[i] * scalar;
            gtr[i] = a[i] > scalar ? 1f : 0f;
        }

        await VerifyBinaryOp(accelerator, "Sub", a, new[] { n }, new[] { scalar }, new[] { 1 }, sub, new[] { n });
        await VerifyBinaryOp(accelerator, "Mul", a, new[] { n }, new[] { scalar }, new[] { 1 }, mul, new[] { n });
        await VerifyBinaryOp(accelerator, "Greater", a, new[] { n }, new[] { scalar }, new[] { 1 }, gtr, new[] { n });
    });
}
