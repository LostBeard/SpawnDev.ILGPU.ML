using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// GraphExecutor's work-based submits (<see cref="GraphExecutor.SubmitWorkBudgetFlops"/>): on WebGPU a command buffer is submitted once the
/// estimated work queued in it reaches the budget, so no single command buffer can outrun a slow GPU's driver watchdog (FLUX.2 Klein's VAE
/// decode lost the device on an RX 590 in Chrome with 64 nodes per command buffer, 2026-11-23). A submit only cuts the command buffer, so the
/// results must not change by a single bit.
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>The estimator: Conv / MatMul from their shapes, everything else by element count.</summary>
    [TestMethod]
    public async Task WorkSubmit_EstimateNodeFlops_ConvMatMulElementwise() => await RunTest(async accelerator =>
    {
        // Conv: X [1,128,512,512], W [128,128,3,3] -> 2 * 1 * 128 * 512*512 * (128*3*3) = 77.3 GFLOP (one VAE up-block conv at 512)
        double conv = GraphExecutor.EstimateNodeFlops("Conv", new int[]?[] { new[] { 1, 128, 512, 512 }, new[] { 128, 128, 3, 3 }, new[] { 128 } });
        double convWant = 2.0 * 128 * 512 * 512 * 128 * 9;
        if (conv != convWant) throw new Exception($"Conv estimate {conv:E3}, expected {convWant:E3}");
        // MatMul: A [2,64,256], B [256,512] -> 2 * (2*64) * 256 * 512
        double mm = GraphExecutor.EstimateNodeFlops("MatMul", new int[]?[] { new[] { 2, 64, 256 }, new[] { 256, 512 } });
        double mmWant = 2.0 * (2 * 64) * 256 * 512;
        if (mm != mmWant) throw new Exception($"MatMul estimate {mm:E3}, expected {mmWant:E3}");
        // Add: the larger input's elements; an unknown (null) input is skipped
        double add = GraphExecutor.EstimateNodeFlops("Add", new int[]?[] { new[] { 1, 64, 32, 32 }, null, new[] { 64, 1, 1 } });
        if (add != 64 * 32 * 32) throw new Exception($"Add estimate {add}, expected {64 * 32 * 32}");
        await Task.CompletedTask;
    });

    /// <summary>
    /// The same conv stack run with a budget small enough to submit every node or two, and with work-based submits off: the outputs must be
    /// BIT-identical (a submit changes no math), WebGPU must actually have submitted (the path ran), and no other backend may (WebGPU only).
    /// </summary>
    [TestMethod]
    public async Task WorkSubmit_TinyBudget_SubmitsOnWebGPU_ResultsBitIdentical() => await RunTest(async accelerator =>
    {
        const int C = 32, H = 32, W = 32, Layers = 6;   // small enough for the interpreted Wasm lane inside the default 30 s
        var convAttrs = () => new Dictionary<string, System.Text.Json.JsonElement> {
            ["pads"] = System.Text.Json.JsonSerializer.SerializeToElement(new[] { 1, 1, 1, 1 }),
            ["strides"] = System.Text.Json.JsonSerializer.SerializeToElement(new[] { 1, 1 }) };
        var graph = new ModelGraph
        {
            Name = "work_submit_stack",
            Inputs = new() { new() { Name = "x", Shape = new[] { 1, C, H, W } } },
            Outputs = new() { new() { Name = $"r{Layers - 1}", Shape = new[] { 1, C, H, W } } },
            Initializers = new(),
            Nodes = new(),
        };
        string prev = "x";
        for (int l = 0; l < Layers; l++)
        {
            graph.Initializers[$"w{l}"] = new[] { C, C, 3, 3 };
            graph.Initializers[$"b{l}"] = new[] { C };
            graph.Nodes.Add(new() { OpType = "Conv", Inputs = { prev, $"w{l}", $"b{l}" }, Outputs = { $"c{l}" }, Attributes = convAttrs() });
            graph.Nodes.Add(new() { OpType = "Relu", Inputs = { $"c{l}" }, Outputs = { $"r{l}" } });
            prev = $"r{l}";
        }
        var compiled = new GraphCompiler(new OperatorRegistry(accelerator)).Compile(graph);
        using var pool = new BufferPool(accelerator);
        var weights = new Dictionary<string, Tensor>();
        for (int l = 0; l < Layers; l++)
        {
            weights[$"w{l}"] = pool.AllocatePermanent(RandomFloats(C * C * 9, seed: 900 + l, scale: 0.06f), new[] { C, C, 3, 3 });
            weights[$"b{l}"] = pool.AllocatePermanent(RandomFloats(C, seed: 950 + l, scale: 0.05f), new[] { C });
        }
        var inData = RandomFloats(C * H * W, seed: 999, scale: 1f);
        const int outN = C * H * W;
        double perConv = GraphExecutor.EstimateNodeFlops("Conv", new int[]?[] { new[] { 1, C, H, W }, new[] { C, C, 3, 3 } });

        async Task<(float[] Data, int Submits)> Run(double budget, bool sync = false)
        {
            using var ex = new GraphExecutor(accelerator, compiled, weights) { SubmitWorkBudgetFlopsOverride = budget };
            using var inBuf = accelerator.Allocate1D(inData);
            using var hostBuf = accelerator.Allocate1D<float>(outN);
            var feeds = new Dictionary<string, Tensor> { ["x"] = new Tensor(inBuf.View, new[] { 1, C, H, W }) };
            var outs = sync ? ex.Run(feeds) : await ex.RunAsync(feeds);
            int submits = GraphExecutor.LastRunWorkSubmitCount;
            await hostBuf.View.CopyFromAsync(outs[$"r{Layers - 1}"].Data.SubView(0, outN));
            await accelerator.SynchronizeAsync();
            return (await hostBuf.CopyToHostAsync<float>(0, outN), submits);
        }

        var off = await Run(0);                  // work-based submits disabled
        var tiny = await Run(perConv * 1.5);     // the budget holds one conv: a submit before nearly every next conv
        if (off.Submits != 0) throw new Exception($"{BackendName}: budget 0 still submitted {off.Submits} time(s)");
        bool webgpu = accelerator.AcceleratorType == global::ILGPU.Runtime.AcceleratorType.WebGPU;
        if (webgpu && tiny.Submits < Layers - 2)
            throw new Exception($"{BackendName}: a one-conv budget over {Layers} convs submitted only {tiny.Submits} time(s): the path did not run");
        if (!webgpu && tiny.Submits != 0) throw new Exception($"{BackendName}: work-based submits are WebGPU-only, but {tiny.Submits} happened");
        for (int i = 0; i < outN; i++)
            if (BitConverter.SingleToInt32Bits(off.Data[i]) != BitConverter.SingleToInt32Bits(tiny.Data[i]))
                throw new Exception($"{BackendName}: output[{i}] = {tiny.Data[i]} with submits, {off.Data[i]} without: a submit changed the math");
        // Negative control: the stack is not trivially constant (a broken graph returning zeros would compare equal).
        if (off.Data.All(v => v == 0f)) throw new Exception($"{BackendName}: the conv stack produced all zeros - the comparison proves nothing");

        // The synchronous Run path has its own loop and its own work-based submit: same contract.
        var syncTiny = await Run(perConv * 1.5, sync: true);
        if (webgpu && syncTiny.Submits < Layers - 2)
            throw new Exception($"{BackendName}: sync Run with a one-conv budget submitted only {syncTiny.Submits} time(s): the path did not run");
        if (!webgpu && syncTiny.Submits != 0) throw new Exception($"{BackendName}: sync Run submitted {syncTiny.Submits} time(s) off WebGPU");
        for (int i = 0; i < outN; i++)
            if (BitConverter.SingleToInt32Bits(off.Data[i]) != BitConverter.SingleToInt32Bits(syncTiny.Data[i]))
                throw new Exception($"{BackendName}: sync Run output[{i}] = {syncTiny.Data[i]}, async without submits {off.Data[i]}");
    });
}
