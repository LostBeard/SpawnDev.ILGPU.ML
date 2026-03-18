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
