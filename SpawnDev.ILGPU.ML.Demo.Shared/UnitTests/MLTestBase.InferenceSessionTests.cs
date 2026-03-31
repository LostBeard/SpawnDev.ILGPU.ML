using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task InferenceSession_CreateAndRun() => await RunTest(async accelerator =>
    {
        int M = 4, K = 8, N = 3;
        var inputData = RandomFloats(M * K, seed: 190);
        var weightData = RandomFloats(K * N, seed: 191, scale: 0.5f);
        var biasData = RandomFloats(N, seed: 192, scale: 0.1f);

        // CPU reference
        var mm = CpuMatMul(inputData, weightData, M, K, N);
        var expected = new float[M * N];
        for (int i = 0; i < M * N; i++)
            expected[i] = MathF.Max(0, mm[i] + biasData[i % N]);

        // Build model graph
        var graph = new ModelGraph
        {
            Name = "test_session",
            Inputs = new() { new() { Name = "input", Shape = new[] { M, K } } },
            Outputs = new() { new() { Name = "output", Shape = new[] { M, N } } },
            Initializers = new() { ["W"] = new[] { K, N }, ["b"] = new[] { N } },
            Nodes = new()
            {
                new() { OpType = "MatMul", Inputs = { "input", "W" }, Outputs = { "mm" } },
                new() { OpType = "Add", Inputs = { "mm", "b" }, Outputs = { "add" } },
                new() { OpType = "Relu", Inputs = { "add" }, Outputs = { "output" } },
            }
        };

        // Create weight tensors
        using var pool = new BufferPool(accelerator);
        var weights = new Dictionary<string, Tensor>
        {
            ["W"] = pool.AllocatePermanent(weightData, new[] { K, N }),
            ["b"] = pool.AllocatePermanent(biasData, new[] { N }),
        };

        // Create session and run
        using var session = InferenceSession.Create(accelerator, graph, weights);

        if (session.InputNames[0] != "input") throw new Exception("Wrong input name");
        if (session.OutputNames[0] != "output") throw new Exception("Wrong output name");
        if (session.SupportedOpCount < 20) throw new Exception($"Too few ops: {session.SupportedOpCount}");

        var inputTensor = pool.AllocatePermanent(inputData, new[] { M, K });
        var outputs = await session.RunAsync(new Dictionary<string, Tensor> { ["input"] = inputTensor });
        var output = outputs[session.OutputNames[0]];

        await accelerator.SynchronizeAsync();

        // Read output
        using var readBuf = accelerator.Allocate1D<float>(M * N);
        var ew = new ElementWiseKernels(accelerator);
        ew.Scale(output.Data.SubView(0, M * N), readBuf.View, M * N, 1f);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, readBuf.View.SubView(0, M * N), expected, K * 2e-6f, "InferenceSession: ");
    });
}
