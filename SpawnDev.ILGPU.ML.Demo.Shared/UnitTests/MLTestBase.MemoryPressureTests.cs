using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Every <see cref="BufferPool"/> is a holder of reclaimable device memory registered with its accelerator (ILGPU
/// <see cref="MemoryPressure"/>): an allocation anywhere on the device takes idle buffers back from the least
/// recently used pool BEFORE the driver runs out. MEASURED 2026-10-12 (Vertex: a chat model, Whisper, Kokoro and a
/// sentence encoder in one process on a 12 GB RTX 4070): 7.3 GB sat idle across the pools, 1.4 GB was spilled into
/// system RAM by a driver that never fails an allocation, and a one-sentence reply took 5.5 minutes.
/// </summary>
public abstract partial class MLTestBase
{
    // 256 MiB of fp32 per tensor: a few of these fill a card fast without thousands of allocations.
    const int PressureChunkFloats = 64 * 1024 * 1024;
    const long PressureChunkBytes = PressureChunkFloats * 4L;

    /// <summary>
    /// One model's pool sits on idle buffers; another model's pool needs memory. The idle pool must give it back
    /// before the allocation, so the device never spills. Red-check: with the idle pool unregistered (as before
    /// this change) it keeps everything and free memory ends below the headroom.
    /// </summary>
    [TestMethod]
    public async Task MemoryPressure_IdlePoolGivesMemoryToAnotherPool() => await RunTest(async accelerator =>
    {
        if (!accelerator.TryGetFreeMemory(out long free))
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: cannot report free device memory");
        const long Headroom = 512L * 1024 * 1024;
        accelerator.DeviceMemoryHeadroomBytes = Headroom;
        using var idlePool = new BufferPool(accelerator);
        using var busyPool = new BufferPool(accelerator);
        try
        {
            // The idle model: rent until only headroom + 2 chunks are free, then return everything (a finished
            // forward's activation arena, sitting in the free buckets).
            var rented = new List<Tensor>();
            while (free > Headroom + 2 * PressureChunkBytes)
            {
                rented.Add(idlePool.Rent(new[] { PressureChunkFloats }, $"idle{rented.Count}"));
                accelerator.TryGetFreeMemory(out free);
            }
            if (rented.Count < 4)
                throw new UnsupportedTestException($"too little free device memory to test ({free / 1048576} MiB)");
            foreach (var t in rented) idlePool.Return(t);
            long idleBefore = idlePool.FreeBucketedBytes;

            // The busy model needs three chunks: they do not fit above the headroom.
            for (int i = 0; i < 3; i++) busyPool.Rent(new[] { PressureChunkFloats }, $"busy{i}");
            await accelerator.SynchronizeAsync();
            accelerator.TryGetFreeMemory(out long after);

            long given = idleBefore - idlePool.FreeBucketedBytes;
            if (given < PressureChunkBytes)
                throw new Exception($"the idle pool gave back {given / 1048576} MiB of {idleBefore / 1048576} MiB; expected at least one chunk");
            if (after < Headroom - PressureChunkBytes / 2)
                throw new Exception($"free device memory fell to {after / 1048576} MiB, below the {Headroom / 1048576} MiB headroom: the busy pool's allocations spilled");
        }
        finally
        {
            accelerator.DeviceMemoryHeadroomBytes = null;
        }
    });

    /// <summary>
    /// A name re-Rented while its previous tensor is live (the GGUF decode loop's 'logits', Whisper's per-step
    /// 'present.*'): returning the OLD tensor must free the old buffer, never the name's new, live one. Before,
    /// Return pooled whatever the name recorded, so a live buffer sat in the free bucket until a reclaim freed it
    /// under a running kernel ("an illegal memory access", Vertex 2026-10-12). Also: a second Return of the same
    /// tensor frees nothing, and a tensor that only VIEWS memory the pool never rented under that name (an alias)
    /// still frees the buffer rented under the name.
    /// </summary>
    [TestMethod]
    public async Task MemoryPressure_ReturnFreesTheReturnedTensorsBuffer_NotTheNamesNewOne() => await RunTest(async accelerator =>
    {
        using var pool = new BufferPool(accelerator);
        var shape = new[] { 4096 };
        var first = pool.Rent(shape, "logits");
        var second = pool.Rent(shape, "logits");   // same name, the first still live
        pool.Return(first);
        if (!pool.IsReturnedToPool(first)) throw new Exception("returning the stale tensor did not free its own buffer");
        if (pool.IsReturnedToPool(second)) throw new Exception("returning the stale tensor freed the name's NEW, live buffer");

        pool.Return(first);                          // a second Return of the same tensor
        if (pool.IsReturnedToPool(second)) throw new Exception("a double Return freed the name's live buffer");

        var third = pool.Rent(shape, "logits");      // takes the freed first buffer back
        pool.Return(second);
        if (!pool.IsReturnedToPool(second)) throw new Exception("the second tensor's own Return did not free it");
        if (pool.IsReturnedToPool(third)) throw new Exception("returning the second tensor freed the third");
        pool.Return(third);

        // Alias: a tensor named "out" viewing memory the pool never rented under that name frees the buffer that WAS
        // rented under "out" (an operator that replaced its output with a view).
        var rentedOut = pool.Rent(shape, "out");
        using var external = accelerator.Allocate1D<float>(shape[0]);
        pool.Return(new Tensor(external.View, shape, "out"));
        if (!pool.IsReturnedToPool(rentedOut)) throw new Exception("an aliased Return did not free the buffer rented under its name");
        await Task.CompletedTask;
    });

    /// <summary>A pinned pool (a captured graph replays its free buffers) gives nothing back until unpinned.</summary>
    [TestMethod]
    public async Task MemoryPressure_PinnedPoolKeepsItsFreeBuffers() => await RunTest(async accelerator =>
    {
        using var pool = new BufferPool(accelerator);
        var t = pool.Rent(new[] { 1024 * 1024 }, "kept");
        pool.Return(t);
        long idle = pool.FreeBucketedBytes;
        if (idle <= 0) throw new Exception("the returned buffer is not in the free buckets");

        pool.Pin();
        try
        {
            if (pool.ReclaimableBytes != 0) throw new Exception($"a pinned pool reports {pool.ReclaimableBytes} reclaimable bytes");
            long byPressure = pool.Reclaim(long.MaxValue / 4);
            long byRelease = pool.ReleaseFreeBuffers();
            if (byPressure != 0 || byRelease != 0 || pool.FreeBucketedBytes != idle)
                throw new Exception($"a pinned pool freed {byPressure} (pressure) + {byRelease} (release) bytes; its free buffers must stay");
        }
        finally { pool.Unpin(); }

        long freed = pool.Reclaim(1);
        if (freed <= 0 || pool.FreeBucketedBytes != idle - freed)
            throw new Exception($"after Unpin the pool freed {freed} bytes; expected its free buffer");
        await Task.CompletedTask;
    });

    /// <summary>
    /// A captured CUDA graph replays the intermediates it was recorded with, which sit in the session pool's free
    /// buckets between replays. A reclaim must not free them: the replay would write memory that now belongs to
    /// someone else. Here everything reclaimable is reclaimed, new buffers of the same size are allocated and filled
    /// (the driver hands freed addresses straight back), and the replay must neither disturb them nor go wrong
    /// itself. Red-check: without the capture's pins the reclaim frees the intermediates and the replay overwrites
    /// the new buffers.
    /// </summary>
    [TestMethod]
    public async Task MemoryPressure_ReclaimLeavesACapturedGraphsBuffersAlone() => await RunTest(async accelerator =>
    {
        if (accelerator is not CudaAccelerator)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: CUDA graph capture is CUDA-only");
        if (!CudaStream.SupportsGraphCapture)
            throw new UnsupportedTestException("driver does not expose the CUDA graph API");

        const int N = 1 << 20; // 4 MiB per intermediate
        // y = Relu(x * x + x): two pooled intermediates (a, b) the graph writes on every replay.
        var graph = new ModelGraph
        {
            Name = "capture-pins",
            Inputs = { new GraphValueInfo { Name = "x", Shape = new[] { N } } },
            Outputs = { new GraphValueInfo { Name = "y", Shape = new[] { N } } },
            Nodes =
            {
                new GraphNode { OpType = "Mul", Inputs = { "x", "x" }, Outputs = { "a" } },
                new GraphNode { OpType = "Add", Inputs = { "a", "x" }, Outputs = { "b" } },
                new GraphNode { OpType = "Relu", Inputs = { "b" }, Outputs = { "y" } },
            },
        };
        var x = RandomFloats(N, seed: 7);
        var expected = new float[N];
        for (int i = 0; i < N; i++) expected[i] = MathF.Max(0f, x[i] * x[i] + x[i]);

        using var session = InferenceSession.Create(accelerator, graph, new Dictionary<string, Tensor>());
        using var xBuf = accelerator.Allocate1D(x);
        var feed = new Dictionary<string, Tensor> { ["x"] = new Tensor(xBuf.View, new[] { N }, "x") };

        using var cap = await CudaGraphCapture.TryCaptureAsync(session, feed);
        if (cap == null) throw new UnsupportedTestException("capture refused: " + CudaGraphCapture.LastRefusalReason);

        // Reclaim everything reclaimable on the device, then take memory of the intermediates' size and fill it.
        accelerator.ReclaimDeviceMemory(long.MaxValue / 4);
        var junk = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
        try
        {
            for (int i = 0; i < 8; i++)
            {
                var j = accelerator.Allocate1D<float>(N);
                j.View.MemSet(0x5A);
                junk.Add(j);
            }
            await accelerator.SynchronizeAsync();

            var outs = await cap.ReplayAsync(feed);
            var y = outs["y"];
            var got = await y.Data.SubView(0, N).CopyToHostAsync();
            AssertClose(expected, got, 1e-5f, "replay after reclaim: ");

            for (int i = 0; i < junk.Count; i++)
            {
                var h = await junk[i].View.CopyToHostAsync();
                float pattern = BitConverter.Int32BitsToSingle(0x5A5A5A5A);
                for (int k = 0; k < h.Length; k += 4096)
                    if (BitConverter.SingleToInt32Bits(h[k]) != BitConverter.SingleToInt32Bits(pattern))
                        throw new Exception($"buffer {i} allocated after the reclaim was overwritten at {k} by the replay: " +
                                            "the reclaim freed memory the captured graph still writes");
            }
        }
        finally
        {
            foreach (var j in junk) j.Dispose();
        }
    });
}
