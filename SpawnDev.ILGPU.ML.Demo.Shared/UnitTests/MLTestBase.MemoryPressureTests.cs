using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Pipelines;
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

    /// <summary>
    /// <see cref="InferenceSession.OutputsLiveUntilNextRun"/>: a loop that feeds each run's output back in (a decoder's
    /// present K/V becoming past K/V) and never calls ReturnOutputs. The previous outputs go back to the pool at the
    /// next run, except the one passed back in, which must survive that run untouched. Without it every run left its
    /// outputs orphaned (the pool grew by a set per run: 3.4 GB of language-model logits in Vertex, 2026-10-12);
    /// returned too eagerly, the fed-back input would be recycled under the run reading it and the values corrupt.
    /// </summary>
    [TestMethod]
    public async Task MemoryPressure_OutputsLiveUntilNextRun_FeedbackLoopNeitherLeaksNorCorrupts() => await RunTest(async accelerator =>
    {
        const int N = 64 * 1024;
        // t = x + x is rented FIRST, then y = t - x reads x again: y == x. If the fed-back x had been given back before
        // this run it would be the only free buffer of its size, t would take it and overwrite it (y = 2x - 2x = 0).
        // Kept forever instead (no lifetime at all), every run's y would stay allocated: the pool grows.
        var graph = new ModelGraph
        {
            Name = "feedback",
            Inputs = { new GraphValueInfo { Name = "x", Shape = new[] { N } }, new GraphValueInfo { Name = "one", Shape = new[] { N } } },
            Outputs = { new GraphValueInfo { Name = "y", Shape = new[] { N } } },
            Nodes =
            {
                new GraphNode { OpType = "Add", Inputs = { "x", "x" }, Outputs = { "t" } },
                new GraphNode { OpType = "Sub", Inputs = { "t", "x" }, Outputs = { "y" } },
            },
        };
        using var session = InferenceSession.Create(accelerator, graph, new Dictionary<string, Tensor>());
        session.OutputsLiveUntilNextRun = true;
        var start = RandomFloats(N, seed: 3);
        using var xBuf = accelerator.Allocate1D(start);
        using var oneBuf = accelerator.Allocate1D(Enumerable.Repeat(1f, N).ToArray());
        var one = new Tensor(oneBuf.View, new[] { N }, "one");
        var x = new Tensor(xBuf.View, new[] { N }, "x");
        int settled = -1;
        const int Runs = 40;
        for (int run = 0; run < Runs; run++)
        {
            var input = x;
            var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["x"] = input, ["one"] = one });
            // The run just read `input`; it is only given back at the NEXT run, once nothing passes it in any more.
            if (run > 0 && session.Executor.Pool.IsReturnedToPool(input))
                throw new Exception($"run {run}: the fed-back input was given back to the pool before the run that reads it finished");
            // Given back too early, the input is free when this run rents its output, and the output lands ON the input
            // (elementwise Sub still computes correctly in place on CPU/CUDA; WebGPU refuses the aliased binding).
            if (run > 0 && ReferenceEquals(((IArrayView)outs["y"].Data).Buffer, ((IArrayView)input.Data).Buffer))
                throw new Exception($"run {run}: the output was rented over the fed-back input it was reading");
            x = outs["y"];
            await accelerator.SynchronizeAsync();
            AssertClose(start, await x.Data.SubView(0, N).CopyToHostAsync(), 1e-4f, $"run {run}: ");
            if (run == 4) settled = session.Executor.AllocatedBufferCount;
            if (run > 4 && session.Executor.AllocatedBufferCount > settled)
                throw new Exception($"the pool grew from {settled} to {session.Executor.AllocatedBufferCount} buffers by run {run}: outputs are not given back");
        }
        await accelerator.SynchronizeAsync();
        var got = await x.Data.SubView(0, N).CopyToHostAsync();
        AssertClose(start, got, 1e-4f, "after the feedback loop: ");
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

    /// <summary>
    /// Kokoro speaking line after line with the device budget so tight that EVERY allocation first reclaims every idle
    /// pooled buffer. A buffer the pool has in a free bucket while a tensor still uses it is freed under it here, and
    /// the next kernel touching it is CUDA_ERROR_ILLEGAL_ADDRESS, which poisons the context for every model in the
    /// process. MEASURED 2026-10-12 in Vertex (llama3.1:8b + Kokoro + Whisper + Silero on a 12 GB card, so reclaims
    /// were real): the third Kokoro line after a reclaim faulted; Kokoro alone under forced pressure reproduces it.
    /// The same budget leaves the audio identical to an unpressured run.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy")]
    public async Task MemoryPressure_KokoroSpeaksLineAfterLine_UnderAReclaimOnEveryAllocation() => await RunTest(async accelerator =>
    {
        if (accelerator is not CudaAccelerator)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: the device budget is read from CUDA (cuMemGetInfo)");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        byte[] modelBytes, voiceBytes;
        try
        {
            modelBytes = await http.GetByteArrayAsync($"{KokoroModelsBase}/onnx/model.onnx");
            voiceBytes = await http.GetByteArrayAsync($"{KokoroModelsBase}/voices/af_heart.bin");
        }
        catch (HttpRequestException ex)
        {
            throw new UnsupportedTestException($"the hub did not serve Kokoro: {ex.Message}");
        }
        var pack = KokoroVoicePack.FromBytes("af_heart", voiceBytes);
        // The STREAMING loader, as SpawnDev.AI loads it: weights uploaded while the file is read.
        // The reference, from a pipeline of its own with no pressure.
        float[] reference;
        using (var clean = await KokoroPipeline.CreateFromStreamAsync(accelerator, new MemoryStream(modelBytes)))
            reference = (await clean.SpeakTokensAsync(KokoroReferenceTokens, pack)).Samples;

        // As Vertex used it: warmed on a short line, then a sentence length seen for the FIRST time under pressure
        // (its shape executor built while reclaims fire) and said again and again. Vertex faulted on the third.
        using var pipeline = await KokoroPipeline.CreateFromStreamAsync(accelerator, new MemoryStream(modelBytes));
        await pipeline.SpeakTokensAsync(KokoroShorterTokens, pack);
        var headroom = accelerator.DeviceMemoryHeadroomBytes;
        var reclaimsBefore = MemoryPressure.ProactiveReclaimCount;
        accelerator.DeviceMemoryHeadroomBytes = accelerator.MemorySize; // never "enough": every allocation reclaims first
        try
        {
            for (int line = 0; line < 4; line++)
            {
                var audio = (await pipeline.SpeakTokensAsync(KokoroReferenceTokens, pack)).Samples;
                await accelerator.SynchronizeAsync();
                if (audio.Length != reference.Length)
                    throw new Exception($"line {line}: {audio.Length} samples, {reference.Length} unpressured");
                float maxDiff = 0f;
                for (int i = 0; i < audio.Length; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(audio[i] - reference[i]));
                if (maxDiff > 1e-3f)
                    throw new Exception($"line {line}: the audio differs from the unpressured run by up to {maxDiff:G4} - a reclaimed buffer was still in use");
            }
        }
        finally
        {
            accelerator.DeviceMemoryHeadroomBytes = headroom;
        }
        if (MemoryPressure.ProactiveReclaimCount == reclaimsBefore)
            throw new Exception("no reclaim fired: the test did not put the budget under pressure");
    });

    /// <summary>
    /// No buffer goes back to the pool while a later node still reads it. With <see cref="BufferPool.PoisonFreedBuffers"/>
    /// every buffer is filled with NaN as it is released, so a premature release turns the audio NaN - deterministically,
    /// with no dependence on whether a reclaim happens to run between the release and the late read. Kokoro, both a
    /// first run of a shape and cached runs, against the same lines spoken without the poison.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy")]
    public async Task Pool_NoBufferIsReleasedBeforeItsLastReader_Kokoro() => await RunTest(async accelerator =>
    {
        RequireShippableTtsBackend(accelerator);
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        byte[] modelBytes, voiceBytes;
        try
        {
            modelBytes = await http.GetByteArrayAsync($"{KokoroModelsBase}/onnx/model.onnx");
            voiceBytes = await http.GetByteArrayAsync($"{KokoroModelsBase}/voices/af_heart.bin");
        }
        catch (HttpRequestException ex)
        {
            throw new UnsupportedTestException($"the hub did not serve Kokoro: {ex.Message}");
        }
        var pack = KokoroVoicePack.FromBytes("af_heart", voiceBytes);
        long[][] lines = { KokoroReferenceTokens, KokoroShorterTokens, KokoroReferenceTokens, KokoroShorterTokens };

        float[][] clean;
        using (var pipeline = await KokoroPipeline.CreateFromStreamAsync(accelerator, new MemoryStream(modelBytes)))
        {
            clean = new float[lines.Length][];
            for (int i = 0; i < lines.Length; i++) clean[i] = (await pipeline.SpeakTokensAsync(lines[i], pack)).Samples;
        }

        var was = BufferPool.PoisonFreedBuffers;
        BufferPool.PoisonFreedBuffers = true;
        try
        {
            using var pipeline = await KokoroPipeline.CreateFromStreamAsync(accelerator, new MemoryStream(modelBytes));
            for (int i = 0; i < lines.Length; i++)
            {
                var got = (await pipeline.SpeakTokensAsync(lines[i], pack)).Samples;
                int nan = got.Count(float.IsNaN);
                if (nan > 0)
                    throw new Exception($"{BackendName}: line {i} ({lines[i].Length} tokens): {nan} of {got.Length} samples NaN - "
                        + "a buffer was released to the pool while a later node still read it");
                if (got.Length != clean[i].Length)
                    throw new Exception($"{BackendName}: line {i}: {got.Length} samples, {clean[i].Length} without the poison");
                float maxDiff = 0f;
                for (int k = 0; k < got.Length; k++) maxDiff = MathF.Max(maxDiff, MathF.Abs(got[k] - clean[i][k]));
                if (maxDiff > 1e-3f)
                    throw new Exception($"{BackendName}: line {i}: differs from the unpoisoned run by up to {maxDiff:G4} - "
                        + "a released buffer was read after its release");
            }
        }
        finally
        {
            BufferPool.PoisonFreedBuffers = was;
        }
    });

    /// <summary>
    /// The same check as <see cref="Pool_NoBufferIsReleasedBeforeItsLastReader_Kokoro"/>, on the sentence sequence that
    /// faulted in Vertex (2026-10-12): two voices, and lines whose lengths change from one to the next, so every line
    /// may meet a new shape executor. Under <see cref="BufferPool.PoisonFreedBuffers"/> and nothing else, Vertex's third
    /// "The number N..." line hit CUDA_ERROR_ILLEGAL_ADDRESS in Pad.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy")]
    public async Task Pool_NoBufferIsReleasedBeforeItsLastReader_KokoroSentences() => await RunTest(async accelerator =>
    {
        RequireShippableTtsBackend(accelerator);
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        byte[] modelBytes, heart, michael;
        try
        {
            modelBytes = await http.GetByteArrayAsync($"{KokoroModelsBase}/onnx/model.onnx");
            heart = await http.GetByteArrayAsync($"{KokoroModelsBase}/voices/af_heart.bin");
            michael = await http.GetByteArrayAsync($"{KokoroModelsBase}/voices/am_michael.bin");
        }
        catch (HttpRequestException ex)
        {
            throw new UnsupportedTestException($"the hub did not serve Kokoro: {ex.Message}");
        }
        var heartPack = KokoroVoicePack.FromBytes("af_heart", heart);
        var michaelPack = KokoroVoicePack.FromBytes("am_michael", michael);
        var phonemizer = SpawnDev.Phonemizer.EmbeddedData.CreatePhonemizer();
        var lines = new List<(string Text, KokoroVoicePack Voice)>
        {
            ("Hello there.", heartPack),
            ("Vertex, can you build us a house by the river?", michaelPack),
        };
        for (int i = 0; i < 4; i++) lines.Add(($"The number {i} is a placeholder used to show nothing.", heartPack));

        var was = BufferPool.PoisonFreedBuffers;
        BufferPool.PoisonFreedBuffers = true;
        try
        {
            using var pipeline = await KokoroPipeline.CreateFromStreamAsync(accelerator, new MemoryStream(modelBytes));
            for (int i = 0; i < lines.Count; i++)
            {
                var audio = await pipeline.SpeakAsync(phonemizer.ToSymbols(lines[i].Text), lines[i].Voice);
                await accelerator.SynchronizeAsync();
                int nan = audio.Samples.Count(float.IsNaN);
                Console.WriteLine($"[pool-sentences] line {i} '{lines[i].Text}': {audio.Tokens} tokens, {audio.Samples.Length} samples, {nan} NaN");
                if (nan > 0)
                    throw new Exception($"{BackendName}: line {i} ({audio.Tokens} tokens): {nan} NaN samples - "
                        + "a buffer was released to the pool while a later node still read it");
            }
        }
        finally
        {
            BufferPool.PoisonFreedBuffers = was;
        }
    });
}
