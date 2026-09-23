using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Tests;

/// <summary>
/// Graph outputs are pool-rented and pinned; nothing outside the decode loop ever gave them back, so
/// every forward leaked one set (SpawnScene DrJohnson: +90 MB per DAv3 joint pass, 14 passes, on top
/// of a 3.9 GB arena the pool kept through training; the GPU process died at 7.4 GB, 2026-09-23).
/// These pin the two APIs that fix it: <see cref="InferenceSession.ReturnOutputs"/> and
/// <see cref="InferenceSession.ReleaseWorkingMemory"/>.
/// </summary>
public class SessionOutputLifetimeTests : KernelTestBase
{
    // Large enough that a leaked output is unmistakable in bytes, small enough to run in milliseconds.
    const int N = 1 << 16;

    public SessionOutputLifetimeTests(AcceleratorFixture fixture) : base(fixture) { }

    /// <summary>y = (x * w) + w, two outputs so the leak is two buffers per run, not one.</summary>
    (InferenceSession session, BufferPool weightPool, float[] w) MakeSession()
    {
        var graph = new ModelGraph
        {
            Name = "output-lifetime",
            Inputs = { new GraphValueInfo { Name = "x", Shape = new[] { 1, N } } },
            Outputs =
            {
                new GraphValueInfo { Name = "m", Shape = new[] { 1, N } },
                new GraphValueInfo { Name = "y", Shape = new[] { 1, N } },
            },
            Nodes =
            {
                new GraphNode { OpType = "Mul", Inputs = { "x", "w" }, Outputs = { "m" } },
                new GraphNode { OpType = "Add", Inputs = { "m", "w" }, Outputs = { "y" } },
            },
            Initializers = { ["w"] = new[] { 1, N } },
        };
        var w = RandomFloats(N, seed: 7);
        var weightPool = new BufferPool(Accelerator);
        var weights = new Dictionary<string, Tensor> { ["w"] = weightPool.AllocatePermanent(w, new[] { 1, N }, "w") };
        return (InferenceSession.Create(Accelerator, graph, weights), weightPool, w);
    }

    Dictionary<string, Tensor> RunOnce(InferenceSession session, MemoryBuffer1D<float, Stride1D.Dense> xBuf)
        => session.Run(new Dictionary<string, Tensor> { ["x"] = new Tensor(xBuf.View, new[] { 1, N }) });

    float[] Read(Tensor t)
    {
        var host = new float[t.ElementCount];
        t.Data.SubView(0, t.ElementCount).CopyToCPU(host);
        Accelerator.Synchronize();
        return host;
    }

    [Fact]
    public void ReturnOutputs_KeepsPoolFlatAcrossRuns_AndLeaksWithoutIt()
    {
        var (session, weightPool, w) = MakeSession();
        using var _ = session;
        using var __ = weightPool;
        var x = RandomFloats(N, seed: 3);
        using var xBuf = Accelerator.Allocate1D(x);

        // Warm: the first run allocates the working set. Everything after must reuse it.
        var first = RunOnce(session, xBuf);
        var expectedY = Read(first["y"]);
        for (int i = 0; i < N; i++) Assert.Equal(x[i] * w[i] + w[i], expectedY[i], 1e-5f);
        session.ReturnOutputs(first);
        int baseline = session.Executor.AllocatedBufferCount;
        Assert.True(baseline > 0, "the warm run must have rented something");

        // WITHOUT returning: each run rents fresh output buffers because the prior ones are still
        // recorded live under their names. This is the leak, asserted so the fixed path below cannot
        // pass by accident (a pool that never allocated anything would also be 'flat').
        var leaked = new List<Dictionary<string, Tensor>>();
        for (int i = 0; i < 4; i++) leaked.Add(RunOnce(session, xBuf));
        int afterLeaking = session.Executor.AllocatedBufferCount;
        Assert.True(afterLeaking > baseline,
            $"expected un-returned outputs to grow the pool: {baseline} -> {afterLeaking}");
        foreach (var outs in leaked) session.ReturnOutputs(outs);
        int settled = session.Executor.AllocatedBufferCount;

        // WITH returning: flat, and the values are still right every time.
        for (int i = 0; i < 8; i++)
        {
            var outs = RunOnce(session, xBuf);
            var y = Read(outs["y"]);
            for (int k = 0; k < N; k += 4093) Assert.Equal(expectedY[k], y[k], 1e-6f);
            session.ReturnOutputs(outs);
            Assert.Equal(settled, session.Executor.AllocatedBufferCount);
        }

        // Returning twice is a no-op, not a corruption: the buffer cannot be pooled a second time.
        var again = RunOnce(session, xBuf);
        session.ReturnOutputs(again);
        session.ReturnOutputs(again);
        var check = RunOnce(session, xBuf);
        var m = Read(check["m"]);
        var y2 = Read(check["y"]);
        for (int k = 0; k < N; k += 997)
        {
            Assert.Equal(x[k] * w[k], m[k], 1e-6f);
            Assert.Equal(expectedY[k], y2[k], 1e-6f);
        }
        session.ReturnOutputs(check);
    }

    [Fact]
    public void ReleaseWorkingMemory_EmptiesFreeBuckets_NextRunStillCorrect()
    {
        var (session, weightPool, w) = MakeSession();
        using var _ = session;
        using var __ = weightPool;
        var x = RandomFloats(N, seed: 11);
        using var xBuf = Accelerator.Allocate1D(x);

        var outs = RunOnce(session, xBuf);
        var expectedY = Read(outs["y"]);
        session.ReturnOutputs(outs);

        long parked = session.PooledFreeBytes;
        Assert.True(parked >= 2L * N * sizeof(float),
            $"two returned outputs of {N} floats must sit in the free buckets, saw {parked} bytes");

        long freed = session.ReleaseWorkingMemory();
        Assert.Equal(parked, freed);
        Assert.Equal(0, session.PooledFreeBytes);

        // Weights survived and the executor re-allocates what it needs.
        var after = RunOnce(session, xBuf);
        var y = Read(after["y"]);
        for (int k = 0; k < N; k += 613) Assert.Equal(expectedY[k], y[k], 1e-6f);
        session.ReturnOutputs(after);
        Assert.True(session.PooledFreeBytes >= 2L * N * sizeof(float));
    }
}
