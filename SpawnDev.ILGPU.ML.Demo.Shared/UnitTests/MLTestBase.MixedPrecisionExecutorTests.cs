using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// CONTROLLED de-risk for the mixed-precision-activation executor cut
/// (Plans/fp16-bf16-mixed-precision-activations-2026-06-16.md): run a small real graph (Conv→Relu→Conv with
/// 64×64 feature maps ≥4096 elems, so the eligibility floor fires) through <see cref="GraphExecutor"/> with
/// <see cref="ActivationPrecision.F32"/> then <see cref="ActivationPrecision.F16"/>, and confirm the F16 run
/// (intermediates stored fp16, operators still fp32, convert at boundaries) PRODUCES A CORRECT result close
/// to F32 — not garbage — on EVERY backend. This de-risks the executor wiring on a 3-node graph BEFORE the
/// 227-node SD-Turbo VAE. F32 is byte-identical to today (the whole mechanism is guarded off), so existing
/// models can't regress; this proves the F16 path itself.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task MixedPrecision_Executor_F16MatchesF32_AllBackends() => await RunTest(async accelerator =>
    {
        // input [1,3,64,64] -> Conv(64,3,3,3) -> Relu -> Conv(64,64,3,3) -> Relu -> Conv(3,64,3,3) -> output.
        // The two interior feature maps are [1,64,64,64] (262144 elems = the BIG live buffers); the graph output
        // is small [1,3,64,64]. With the pass-through, the interior Conv/Relu read+write fp16 (half the bytes),
        // so the live working set (2 adjacent big maps) HALVES vs F32. The final Conv's output is the graph
        // output → it stays fp32 (one boundary), which is why the peak drops but not exactly to ½.
        var convAttrs = () => new Dictionary<string, System.Text.Json.JsonElement> {
            ["pads"] = System.Text.Json.JsonSerializer.SerializeToElement(new[] { 1, 1, 1, 1 }),
            ["strides"] = System.Text.Json.JsonSerializer.SerializeToElement(new[] { 1, 1 }) };
        var graph = new ModelGraph
        {
            Name = "mixedprec_mini",
            Inputs = new() { new() { Name = "input", Shape = new[] { 1, 3, 64, 64 } } },
            Outputs = new() { new() { Name = "output", Shape = new[] { 1, 3, 64, 64 } } },
            Initializers = new()
            {
                ["c1w"] = new[] { 64, 3, 3, 3 }, ["c1b"] = new[] { 64 },
                ["c2w"] = new[] { 64, 64, 3, 3 }, ["c2b"] = new[] { 64 },
                ["c3w"] = new[] { 3, 64, 3, 3 }, ["c3b"] = new[] { 3 },
            },
            Nodes = new()
            {
                new() { OpType = "Conv", Inputs = { "input", "c1w", "c1b" }, Outputs = { "c1" }, Attributes = convAttrs() },
                new() { OpType = "Relu", Inputs = { "c1" }, Outputs = { "r1" } },
                new() { OpType = "Conv", Inputs = { "r1", "c2w", "c2b" }, Outputs = { "c2" }, Attributes = convAttrs() },
                new() { OpType = "Relu", Inputs = { "c2" }, Outputs = { "r2" } },
                new() { OpType = "Conv", Inputs = { "r2", "c3w", "c3b" }, Outputs = { "output" }, Attributes = convAttrs() },
            }
        };

        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);

        using var pool = new BufferPool(accelerator);
        var weights = new Dictionary<string, Tensor>
        {
            ["c1w"] = pool.AllocatePermanent(RandomFloats(64 * 3 * 3 * 3, seed: 71, scale: 0.15f), new[] { 64, 3, 3, 3 }),
            ["c1b"] = pool.AllocatePermanent(RandomFloats(64, seed: 72, scale: 0.05f), new[] { 64 }),
            ["c2w"] = pool.AllocatePermanent(RandomFloats(64 * 64 * 3 * 3, seed: 73, scale: 0.04f), new[] { 64, 64, 3, 3 }),
            ["c2b"] = pool.AllocatePermanent(RandomFloats(64, seed: 74, scale: 0.05f), new[] { 64 }),
            ["c3w"] = pool.AllocatePermanent(RandomFloats(3 * 64 * 3 * 3, seed: 75, scale: 0.04f), new[] { 3, 64, 3, 3 }),
            ["c3b"] = pool.AllocatePermanent(RandomFloats(3, seed: 76, scale: 0.05f), new[] { 3 }),
        };
        var inData = RandomFloats(1 * 3 * 64 * 64, seed: 77, scale: 1f);
        const int outN = 1 * 3 * 64 * 64;

        async Task<(float[] data, long peakLive)> Run(ActivationPrecision prec)
        {
            BufferPool.ResetPeaks();
            using var ex = new GraphExecutor(accelerator, compiled, weights) { ActivationDtype = prec };
            using var inBuf = accelerator.Allocate1D(inData);
            using var hostBuf = accelerator.Allocate1D<float>(outN);
            // The shape here MUST match the graph's declared input and the data actually allocated above
            // (both [1,3,64,64]). It said [1,3,32,32] - a leftover from when this test used 32x32 maps -
            // which describes 3,072 elements over a 12,288-element buffer. The executor believed the tensor,
            // the Conv indexed past its extent, and an ILGPU bounds assert fired on an async continuation.
            // On desktop that failed the test; in the browser it EXITED THE WASM RUNTIME, which cost the
            // whole browser lane (see ProjectTest.PageRuntimeDied).
            var outs = await ex.RunAsync(new Dictionary<string, Tensor> { ["input"] = new Tensor(inBuf.View, new[] { 1, 3, 64, 64 }) });
            var o = outs["output"];
            await hostBuf.View.CopyFromAsync(o.Data.SubView(0, outN)); // GPU→GPU off the executor pool (ordered, all backends) before ex disposes
            await accelerator.SynchronizeAsync();
            long peak = BufferPool.PeakLiveBytes;
            return (await hostBuf.CopyToHostAsync<float>(0, outN), peak);
        }

        // Opt-in peak instrumentation so we can prove the F16 path actually CUTS the live working set (not just
        // matches numerically). Set after weights are allocated so only the executor's intermediates are measured.
        // Drain every node (SyncIntervalNodes=1) so the measured peak is the genuine live working set (2 adjacent
        // feature maps) rather than the whole deferred-release backlog — otherwise the fp32 boundary temp stacks
        // on top of the (already-freed-in-reality) half buffers and masks the win.
        bool prevTrack = BufferPool.TrackPeaks;
        int prevSync = GraphExecutor.SyncIntervalNodes;
        BufferPool.TrackPeaks = true;
        GraphExecutor.SyncIntervalNodes = 1;
        (float[] data, long peakLive) f32r, f16r;
        try
        {
            f32r = await Run(ActivationPrecision.F32);
            f16r = await Run(ActivationPrecision.F16);
        }
        finally { BufferPool.TrackPeaks = prevTrack; GraphExecutor.SyncIntervalNodes = prevSync; }
        var f32 = f32r.data; var f16 = f16r.data;

        // F16 stores the conv/relu feature maps fp16 → small rounding vs F32 (compute stayed fp32). Must be
        // CLOSE (fp16 quantization through 2 convs), emphatically NOT garbage (a mis-wire would be wildly off).
        float maxAbs = 0f; for (int i = 0; i < outN; i++) maxAbs = MathF.Max(maxAbs, MathF.Abs(f32[i]));
        float tol = MathF.Max(0.05f, maxAbs * 0.05f);   // 3 convs through fp16 storage accumulate quantization
        int bad = 0; float worst = 0;
        for (int i = 0; i < outN; i++) { float d = MathF.Abs(f16[i] - f32[i]); if (d > worst) worst = d; if (d > tol) bad++; }
        if (bad > 0)
            throw new Exception($"F16 executor output diverged from F32 at {bad}/{outN} elements (worst |Δ|={worst:F4}, tol={tol:F4}, maxAbs={maxAbs:F3}) — mixed-precision wiring wrong on {BackendName}.");

        // The whole point of approach (i): the precision-aware pass-through (Conv/Relu read+write fp16, no fp32
        // temp) must DROP the live working set below the all-fp32 run. If the peak did NOT drop, the pass-through
        // isn't firing (we'd be on the convert-around-node workaround, where fp32 temps are additive).
        if (f16r.peakLive >= f32r.peakLive)
            throw new Exception($"F16 peak live working set did NOT drop: F16={f16r.peakLive} >= F32={f32r.peakLive} bytes on {BackendName} — precision-aware pass-through not firing.");
        Console.WriteLine($"[MixedPrec] executor F16 matches F32 (worst |Δ|={worst:E2}, tol={tol:F4}); peak live F32={f32r.peakLive / 1024}KiB → F16={f16r.peakLive / 1024}KiB on {BackendName}");
    });
}
