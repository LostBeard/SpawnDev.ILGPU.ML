using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task OperatorRegistry_ResolvesBuiltinOps() => await RunTest(async accelerator =>
    {
        var registry = new OperatorRegistry(accelerator);

        // Check key ops are registered
        string[] requiredOps = { "MatMul", "Relu", "Gelu", "Add", "Mul", "Sub",
            "Reshape", "Transpose", "Softmax", "LayerNormalization",
            "Unsqueeze", "Squeeze", "Flatten", "Concat", "Sigmoid", "Tanh",
            "BatchNormalization", "GlobalAveragePool", "ReduceMean", "ReduceSum",
            "Neg", "Clip" };

        foreach (var op in requiredOps)
        {
            if (!registry.IsSupported(op))
                throw new Exception($"Op '{op}' not registered");
        }

        if (registry.SupportedOps.Count < 20)
            throw new Exception($"Expected 20+ ops, got {registry.SupportedOps.Count}");

        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Operator_MatMul_ViaRegistry() => await RunTest(async accelerator =>
    {
        var registry = new OperatorRegistry(accelerator);
        using var pool = new BufferPool(accelerator);

        int M = 4, K = 8, N = 3;
        var aData = RandomFloats(M * K, seed: 170);
        var bData = RandomFloats(K * N, seed: 171);
        var expected = CpuMatMul(aData, bData, M, K, N);

        var a = pool.AllocatePermanent(aData, new[] { M, K });
        var b = pool.AllocatePermanent(bData, new[] { K, N });
        var output = pool.Rent(new[] { M, N });

        var op = registry.Resolve("MatMul");
        op.Execute(new OnnxOpContext
        {
            Inputs = new[] { a, b },
            Outputs = new[] { output },
            Attributes = new Dictionary<string, object>(),
            Pool = pool,
        });
        await accelerator.SynchronizeAsync();

        var actual = await output.Data.SubView(0, M * N).CopyToAsync(accelerator, M * N);
        AssertClose(expected, actual, K * 2e-6f, "Operator MatMul: ");
    });

    [TestMethod]
    public async Task Operator_ReluAdd_Chain() => await RunTest(async accelerator =>
    {
        var registry = new OperatorRegistry(accelerator);
        using var pool = new BufferPool(accelerator);

        int count = 100;
        var aData = RandomFloats(count, seed: 172, scale: 2f);
        var bData = RandomFloats(count, seed: 173, scale: 0.5f);

        // Expected: ReLU(a) + b
        var reluA = new float[count];
        var expected = new float[count];
        for (int i = 0; i < count; i++)
        {
            reluA[i] = MathF.Max(0, aData[i]);
            expected[i] = reluA[i] + bData[i];
        }

        var a = pool.AllocatePermanent(aData, new[] { count });
        var b = pool.AllocatePermanent(bData, new[] { count });
        var reluOut = pool.Rent(new[] { count });
        var addOut = pool.Rent(new[] { count });

        var relu = registry.Resolve("Relu");
        relu.Execute(new OnnxOpContext
        {
            Inputs = new[] { a }, Outputs = new[] { reluOut },
            Attributes = new(), Pool = pool
        });

        var add = registry.Resolve("Add");
        add.Execute(new OnnxOpContext
        {
            Inputs = new[] { reluOut, b }, Outputs = new[] { addOut },
            Attributes = new(), Pool = pool
        });

        await accelerator.SynchronizeAsync();
        var actual = await addOut.Data.SubView(0, count).CopyToAsync(accelerator, count);
        AssertClose(expected, actual, 1e-5f, "Relu+Add chain: ");
    });

    [TestMethod]
    public async Task Operator_Conv_TFLiteDepthwiseSentinel() => await RunTest(async accelerator =>
    {
        // Regression for commit ce836a0: ConvOperator.InferOutputShapes must resolve
        // group=-1 (TFLite depthwise sentinel) by setting outC = inC. Before the fix,
        // group=-1 triggered the default outC = wOutC = 1 path (because wOutC was 1
        // for TFLite weight [1, kH, kW, inC] in NHWC), and the buffer pool allocated
        // outH*outW*1 elements while the kernel dispatched outH*outW*inC threads -
        // Wasm OOB; silent overrun on every other backend.
        // BlazeFace (24-channel depthwise) was the canonical repro.
        var registry = new OperatorRegistry(accelerator);
        var convOp = registry.Resolve("Conv");

        // BlazeFace ForwardDepthwiseNHWC #2 / #4 shape:
        //   x.Shape = [1, 64, 64, 24] (NHWC: N, H, W, inC=24)
        //   w.Shape = [1, 3, 3, 24]   (NHWC TFLite depthwise: [1, kH, kW, inC])
        //   group = -1 (TFLite sentinel)
        // Expected output: [1, 64, 64, 24] (98304 elements), NOT [1, 64, 64, 1] (4096).
        var attrs = new Dictionary<string, object>
        {
            ["_data_format"] = "NHWC",
            ["strides"] = new long[] { 1, 1 },
            ["pads"] = new long[] { 1, 1, 1, 1 },
            ["group"] = (long)-1,
        };
        var outShapes = convOp.InferOutputShapes(new[]
        {
            new[] { 1, 64, 64, 24 },
            new[] { 1, 3, 3, 24 },
        }, attrs);

        if (outShapes.Length != 1)
            throw new Exception($"Expected 1 output shape, got {outShapes.Length}");
        var outShape = outShapes[0];
        if (outShape.Length != 4)
            throw new Exception($"Expected rank-4 output, got rank {outShape.Length}: [{string.Join(",", outShape)}]");
        // NHWC: [N, outH, outW, outC]
        if (outShape[3] != 24)
            throw new Exception($"TFLite depthwise sentinel not resolved: outC={outShape[3]}, expected 24 (= inC). Output shape was [{string.Join(",", outShape)}]");
        var totalElements = outShape[0] * outShape[1] * outShape[2] * outShape[3];
        if (totalElements != 98304)
            throw new Exception($"Wrong total output elements: {totalElements}, expected 98304 (=1*64*64*24). Output shape was [{string.Join(",", outShape)}]");

        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Operator_Conv_NHWC_NonDepthwiseGroup1() => await RunTest(async accelerator =>
    {
        // Companion to the above: ensure the depthwise sentinel branch doesn't fire
        // for regular (non-depthwise) NHWC convs. Standard Conv2D group=1 inC=3 outC=24.
        var registry = new OperatorRegistry(accelerator);
        var convOp = registry.Resolve("Conv");

        var attrs = new Dictionary<string, object>
        {
            ["_data_format"] = "NHWC",
            ["strides"] = new long[] { 2, 2 },
            ["pads"] = new long[] { 1, 1, 1, 1 },
            ["group"] = (long)1,
        };
        var outShapes = convOp.InferOutputShapes(new[]
        {
            new[] { 1, 128, 128, 3 },     // NHWC input
            new[] { 24, 5, 5, 3 },        // NHWC weight [outC, kH, kW, inC]
        }, attrs);

        var outShape = outShapes[0];
        // NHWC: [N, outH, outW, outC] — outH/outW = (128 + 2 - 5)/2 + 1 = 63
        if (outShape[3] != 24)
            throw new Exception($"Non-depthwise NHWC outC wrong: {outShape[3]}, expected 24 (=wOutC)");
        if (outShape[1] != 63 || outShape[2] != 63)
            throw new Exception($"Non-depthwise NHWC spatial wrong: outH={outShape[1]} outW={outShape[2]}, expected 63x63. Output shape was [{string.Join(",", outShape)}]");

        await Task.CompletedTask;
    });
}

// Helper extension for reading tensor data
static file class TensorReadExtensions
{
    public static async Task<float[]> CopyToAsync(this ArrayView1D<float, Stride1D.Dense> view,
        Accelerator accelerator, int count)
    {
        using var temp = accelerator.Allocate1D<float>(count);
        var ew = new ElementWiseKernels(accelerator);
        ew.Scale(view, temp.View, count, 1f);
        await accelerator.SynchronizeAsync();
        return await temp.CopyToHostAsync<float>(0, count);
    }
}
