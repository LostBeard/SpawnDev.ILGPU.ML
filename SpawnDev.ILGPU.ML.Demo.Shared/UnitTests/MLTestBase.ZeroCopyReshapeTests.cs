using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Verifies the executor's zero-copy Reshape (a large single-consumer Reshape hands off its input buffer as a
/// view instead of renting + copying). Runs a graph input → Relu → Reshape → Relu → output through the executor
/// and confirms the result equals the CPU reference (Reshape is metadata-only, so the data must survive the
/// buffer hand-off intact) on every backend. The chained Relu→Reshape→Relu exercises the single-consumer
/// ownership transfer (each Reshape's input is produced+consumed once). The SD-Turbo pipeline (peak −, image
/// bit-identical) is the real-model proof; this is the isolated regression guard.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task ZeroCopyReshape_PreservesData_AllBackends() => await RunTest(async accelerator =>
    {
        const int rows = 64, cols = 64, n = rows * cols;   // 4096 elems → hits the zero-copy floor
        var rng = new Random(717);
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 4 - 2);

        // input[64,64] → Relu(r1) → Reshape[1,4096] → Relu(r2) → Reshape[16,256] = output.
        // r1 is single-consumer of the first Reshape; the first Reshape output feeds Relu (single consumer of it),
        // etc. — the chain that triggers the zero-copy hand-off.
        var graph = new ModelGraph
        {
            Name = "zerocopy_reshape",
            Inputs = new() { new() { Name = "input", Shape = new[] { rows, cols } } },
            Outputs = new() { new() { Name = "output", Shape = new[] { 16, 256 } } },
            Nodes = new()
            {
                new() { OpType = "Relu", Inputs = { "input" }, Outputs = { "r1" } },
                new() { OpType = "Reshape", Inputs = { "r1" }, Outputs = { "rs1" },
                    Attributes = new() { ["shape"] = JsonSerializer.SerializeToElement(new[] { 1, 4096 }) } },
                new() { OpType = "Relu", Inputs = { "rs1" }, Outputs = { "r2" } },
                new() { OpType = "Reshape", Inputs = { "r2" }, Outputs = { "output" },
                    Attributes = new() { ["shape"] = JsonSerializer.SerializeToElement(new[] { 16, 256 }) } },
            },
        };

        var expected = new float[n];
        for (int i = 0; i < n; i++) { float v = x[i] > 0 ? x[i] : 0; expected[i] = v > 0 ? v : 0; } // relu∘relu = relu

        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);
        using var ex = new GraphExecutor(accelerator, compiled, new Dictionary<string, Tensor>());
        using var inBuf = accelerator.Allocate1D(x);
        using var host = accelerator.Allocate1D<float>(n);
        var outs = await ex.RunAsync(new Dictionary<string, Tensor> { ["input"] = new Tensor(inBuf.View, new[] { rows, cols }) });
        var o = outs["output"];
        if (o.ElementCount != n) throw new Exception($"output elem count {o.ElementCount} != {n} on {BackendName}");
        await host.View.CopyFromAsync(o.Data.SubView(0, n));
        await accelerator.SynchronizeAsync();
        var got = await host.CopyToHostAsync<float>(0, n);

        float worst = 0;
        for (int i = 0; i < n; i++) worst = MathF.Max(worst, MathF.Abs(got[i] - expected[i]));
        if (worst > 1e-6f)
            throw new Exception($"zero-copy Reshape corrupted data (worst |Δ|={worst:E3}) on {BackendName}");
        Console.WriteLine($"[ZeroCopyReshape] data intact through Reshape buffer hand-off on {BackendName}");
    });
}
