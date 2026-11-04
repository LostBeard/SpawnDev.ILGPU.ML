using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    // Mirrors GraphExecutor.PlannedReuseEligible for a plain forward.
    private static bool PlannedReuseEligibleFor(Accelerator accelerator) => GraphExecutor.ImmediateRelease
        && (accelerator.AcceleratorType is AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.CPU or AcceleratorType.Velocity
            || (GraphExecutor.PlannedReuseOnWebGPU && accelerator.AcceleratorType == AcceleratorType.WebGPU));

    private static ModelGraph PlannedReuseGraph(int n)
    {
        static Dictionary<string, JsonElement> Axis0() => new() { ["axis"] = JsonSerializer.SerializeToElement(0) };
        return new()
        {
            Name = "planned_reuse",
            Inputs = new() { new() { Name = "x", Shape = new[] { n } } },
            Outputs = new() { new() { Name = "y", Shape = new[] { 4 * n } } },
            Initializers = new(),
            Nodes = new()
            {
                N("Mul", new[] { "x", "x" }, new[] { "a" }),
                N("Add", new[] { "a", "x" }, new[] { "b" }),
                N("Concat", new[] { "b", "a" }, new[] { "c" }, Axis0()),
                N("Tanh", new[] { "c" }, new[] { "d" }),
                N("Concat", new[] { "d", "c" }, new[] { "f" }, Axis0()),
                N("Sigmoid", new[] { "f" }, new[] { "g" }),
                N("Mul", new[] { "g", "f" }, new[] { "h" }),
                N("Add", new[] { "h", "g" }, new[] { "y" }),
            },
        };
    }

    /// <summary>
    /// Two executors (the same graph at two sizes, as a session recompiles per input shape) share one pool and alternate: both stay bit-identical to
    /// planning off, and the second plan reuses the first plan's slot buffers (runs never overlap) instead of adding its own, so the pool's slot bytes
    /// after both plans equal those after the first. Eligible backends only. Added 2026-11-03.
    /// </summary>
    [TestMethod]
    public async Task PlannedReuse_TwoExecutorsOnePool_ShareSlots() => await RunTest(async accelerator =>
    {
        if (!PlannedReuseEligibleFor(accelerator))
            throw new UnsupportedTestException("planned reuse runs on CUDA / OpenCL / CPU / Velocity (and WebGPU with GraphExecutor.PlannedReuseOnWebGPU) with ImmediateRelease");
        const int big = 1 << 16, small = 1 << 15;
        var rng = new Random(5);
        var xv = Enumerable.Range(0, big).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        using var xBuf = accelerator.Allocate1D(xv);
        async Task<float[]> RunOnce(GraphExecutor ex, int n)
        {
            var outs = await ex.RunAsync(new Dictionary<string, Tensor> { ["x"] = new Tensor(xBuf.View.SubView(0, n), new[] { n }) });
            return await outs["y"].Data.SubView(0, 4 * n).CopyToHostAsync();
        }
        GraphExecutor Make(int n, BufferPool? pool) => new(accelerator, new GraphCompiler(new OperatorRegistry(accelerator)).Compile(PlannedReuseGraph(n)), new Dictionary<string, Tensor>(), sharedPool: pool);
        bool saved = BufferPool.PlannedReuse;
        try
        {
            BufferPool.PlannedReuse = false;
            float[] wantBig, wantSmall;
            using (var r = Make(big, null)) wantBig = await RunOnce(r, big);
            using (var r = Make(small, null)) wantSmall = await RunOnce(r, small);

            BufferPool.PlannedReuse = true;
            using var pool = new BufferPool(accelerator);
            using var exBig = Make(big, pool);
            using var exSmall = Make(small, pool);
            void Check(float[] got, float[] want, string what)
            {
                for (int i = 0; i < got.Length; i++)
                    if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(want[i]))
                        throw new Exception($"{what}: y[{i}] = {got[i]:R}, planning off gives {want[i]:R}");
            }
            long plans0 = BufferPool.PlannedReusePlans, dev0 = BufferPool.PlannedReuseDeviations;
            for (int run = 1; run <= 4; run++) Check(await RunOnce(exBig, big), wantBig, $"big run {run}");
            long slotsAfterFirst = pool.PlannedSlotBytes;
            for (int run = 1; run <= 5; run++)
            {
                Check(await RunOnce(exSmall, small), wantSmall, $"small run {run}");
                Check(await RunOnce(exBig, big), wantBig, $"alternating big run {run}");
            }
            long plans = BufferPool.PlannedReusePlans - plans0, devs = BufferPool.PlannedReuseDeviations - dev0, slotsAfterBoth = pool.PlannedSlotBytes;
            if (plans != 2 || devs != 0 || slotsAfterFirst == 0 || slotsAfterBoth != slotsAfterFirst)
                throw new Exception($"plans +{plans}, deviations +{devs}, slot bytes {slotsAfterFirst} after the first plan, {slotsAfterBoth} after both (expected 2 plans, 0 deviations, equal slot bytes)");
            var line = $"{accelerator.AcceleratorType}: two executors alternating bit-identical; 2 plans sharing {slotsAfterBoth / 1024} KiB of slots";
            Console.WriteLine("[PlannedReuse] " + line);
            return line;
        }
        finally { BufferPool.PlannedReuse = saved; }
    });

    /// <summary>
    /// <see cref="BufferPool.PlannedReuse"/>: a graph whose intermediates have three sizes and overlapping lifetimes (x -> a, b -> concat [2N] -> tanh ->
    /// concat with a still-live [2N] -> [4N] -> sigmoid, mul, add) runs eight times on one executor with planning on; every output must equal the output
    /// of the same graph with planning off, bit for bit, while (on the eligible backends: CUDA, OpenCL, CPU, Velocity) a plan is built and later runs take
    /// their rents from its slots with no deviation. Red-check: let overlapping lifetimes share a slot (BuildPlanLocked's overlap test) - the output
    /// changes. Added 2026-11-03.
    /// </summary>
    [TestMethod]
    public async Task PlannedReuse_RepeatedRuns_BitIdenticalAndPlanned() => await RunTest(async accelerator =>
    {
        const int n = 1 << 16;
        var rng = new Random(11);
        var xv = Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        ModelGraph MakeGraph() => PlannedReuseGraph(n);
        using var xBuf = accelerator.Allocate1D(xv);
        var inputs = new Dictionary<string, Tensor> { ["x"] = new Tensor(xBuf.View, new[] { n }) };
        async Task<float[]> RunOnce(GraphExecutor ex)
        {
            var outs = await ex.RunAsync(inputs);
            return await outs["y"].Data.SubView(0, 4 * n).CopyToHostAsync();
        }

        bool saved = BufferPool.PlannedReuse;
        try
        {
            BufferPool.PlannedReuse = false;
            float[] want;
            using (var reference = new GraphExecutor(accelerator, new GraphCompiler(new OperatorRegistry(accelerator)).Compile(MakeGraph()), new Dictionary<string, Tensor>()))
                want = await RunOnce(reference);

            BufferPool.PlannedReuse = true;
            long plans0 = BufferPool.PlannedReusePlans, rents0 = BufferPool.PlannedReuseRents, dev0 = BufferPool.PlannedReuseDeviations;
            using var ex = new GraphExecutor(accelerator, new GraphCompiler(new OperatorRegistry(accelerator)).Compile(MakeGraph()), new Dictionary<string, Tensor>());
            for (int run = 1; run <= 8; run++)
            {
                var got = await RunOnce(ex);
                for (int i = 0; i < got.Length; i++)
                    if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(want[i]))
                        throw new Exception($"run {run}: y[{i}] = {got[i]:R}, planning off gives {want[i]:R} (plans +{BufferPool.PlannedReusePlans - plans0}, planned rents +{BufferPool.PlannedReuseRents - rents0})");
            }
            long plans = BufferPool.PlannedReusePlans - plans0, rents = BufferPool.PlannedReuseRents - rents0, devs = BufferPool.PlannedReuseDeviations - dev0;
            bool eligible = PlannedReuseEligibleFor(accelerator);
            if (eligible && (plans != 1 || rents == 0 || devs != 0))
                throw new Exception($"{accelerator.AcceleratorType}: expected one plan serving rents with no deviation; plans +{plans}, planned rents +{rents}, deviations +{devs}");
            if (!eligible && (plans != 0 || rents != 0))
                throw new Exception($"{accelerator.AcceleratorType} is not eligible but planned: plans +{plans}, planned rents +{rents}");
            var line = $"{accelerator.AcceleratorType}: 8 runs bit-identical; plans +{plans}, planned rents +{rents}, deviations +{devs}, slots {ex.Pool.PlannedSlotBytes / 1024} KiB";
            Console.WriteLine("[PlannedReuse] " + line);
            return line;
        }
        finally { BufferPool.PlannedReuse = saved; }
    });
}
