using ILGPU;
using System.Collections.Generic;
using System.Threading;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// GPU→Stream SAVE primitive (<see cref="Tensor.CopyToStreamAsync"/>, added 2026-07-01 on SpawnDev.ILGPU 4.17.0's
/// <c>ArrayView&lt;T&gt;.CopyToStreamAsync</c>). The save-side mirror of the streaming model-load path — lets a
/// large tensor / GPU buffer be exported to OPFS one bounded chunk at a time (browser: GPU→JS Uint8Array→stream,
/// no managed-heap copy), instead of reading the whole thing into a .NET array and OOMing (SpawnScene's need).
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task Tensor_CopyToStreamAsync_RoundTrip() => await RunTest(async accelerator =>
    {
        // Known fp32 payload on the GPU.
        const int n = 5000;
        var data = new float[n];
        var rng = new Random(1234);
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 200 - 100);

        using var buf = accelerator.Allocate1D(data);
        var tensor = new Tensor(buf.View, new[] { n });

        // SAVE: stream the tensor's raw fp32 bytes out. Small 4 KiB chunk to exercise the multi-chunk loop
        // (20,000 bytes → ~5 chunks) without a 16 MiB allocation; 4-byte aligned for the WebGPU copy rule.
        using var ms = new System.IO.MemoryStream();
        await tensor.CopyToStreamAsync(ms, chunkSizeInBytes: 4096);
        await accelerator.SynchronizeAsync();

        var bytes = ms.ToArray();
        int expectedBytes = n * sizeof(float);
        if (bytes.Length < expectedBytes)
            throw new Exception($"CopyToStreamAsync wrote {bytes.Length} bytes, expected >= {expectedBytes}");

        // Verify the fp32 payload is byte-exact (allow trailing 4-byte-alignment padding on browser backends).
        var got = new float[n];
        Buffer.BlockCopy(bytes, 0, got, 0, expectedBytes);
        // ⚠️ Report the SHAPE of the corruption, not just the first bad index. This failure is
        // intermittent on OpenCL and the first-mismatch-only message could not distinguish "one chunk is
        // zero" from "everything after chunk k is zero" from "scattered" - and those have different causes.
        // 1,024 floats per 4 KiB chunk.
        int bad = 0, zeroBad = 0, firstBad = -1, lastBad = -1;
        var badChunks = new SortedSet<int>();
        for (int i = 0; i < n; i++)
        {
            if (got[i] == data[i]) continue;
            bad++;
            if (got[i] == 0f) zeroBad++;
            if (firstBad < 0) firstBad = i;
            lastBad = i;
            badChunks.Add(i / 1024);
        }
        if (bad > 0)
            throw new Exception(
                $"CopyToStreamAsync mismatch: {bad} of {n} values wrong ({zeroBad} of them ZERO), "
              + $"first [{firstBad}] (chunk {firstBad / 1024}), last [{lastBad}] (chunk {lastBad / 1024}), "
              + $"chunks affected [{string.Join(",", badChunks)}] of {(n + 1023) / 1024}. "
              + $"got {got[firstBad]} expected {data[firstBad]}");

        Console.WriteLine($"[TensorStream] CopyToStreamAsync round-trip byte-exact: {n} floats, {bytes.Length} bytes (~{(bytes.Length + 4095) / 4096} chunks)");
    });

    /// <summary>
    /// The same round-trip with the device DELIBERATELY BUSY - the case that actually caught the bug.
    /// </summary>
    /// <remarks>
    /// <para>
    /// 🔴 WHY THIS EXISTS. <c>MemoryBuffer.CopyToRawAsync</c> drained the stream BEFORE issuing its
    /// <c>CopyTo</c> and then returned without waiting for the copy itself. On CUDA and OpenCL that
    /// <c>CopyTo</c> is an asynchronous DMA, so the method handed back a still-zeroed array - and
    /// <c>using</c> unpinned the CPU buffer the DMA was writing into.
    /// </para>
    /// <para>
    /// MEASURED on OpenCL: <c>Tensor_CopyToStreamAsync_RoundTrip</c> failed ONE full sweep with
    /// "byte mismatch at [3072]: got 0, expected -55.31856" - 3,072 is exactly the start of the fourth
    /// 1,024-float chunk, so chunks 1-3 won the race and 4-5 came back zeros - and passed 6/6 when re-run
    /// scoped. A test that only fails when the machine happens to be loaded is not a gate; it is a thing
    /// people learn to ignore.
    /// </para>
    /// <para>
    /// ⚠️ So this one makes the device busy on purpose before reading back, and uses many small chunks so
    /// there are many chances to lose the race. Zeros rather than garbage is the signature to assert on:
    /// the destination is a fresh array, so a copy that never landed reads as a clean run of 0.
    /// </para>
    /// </remarks>
    [TestMethod(Timeout = 180000)]
    public async Task Tensor_CopyToStreamAsync_SurvivesABusyDevice() => await RunTest(async accelerator =>
    {
        const int n = 65536;                       // 256 KiB -> 64 chunks at 4 KiB
        var data = new float[n];
        var rng = new Random(99);
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 200 - 100);

        using var buf = accelerator.Allocate1D(data);
        var tensor = new Tensor(buf.View, new[] { n });

        // Put REAL work in front of the readback. Without this the DMA usually completes before anyone
        // looks, which is exactly why the original defect survived so long.
        int M = 256, K = 256, N = 256;
        using var a = accelerator.Allocate1D(RandomFloats(M * K, seed: 7));
        using var b = accelerator.Allocate1D(RandomFloats(K * N, seed: 8));
        using var c = accelerator.Allocate1D<float>(M * N);
        var matMul = new MatMulKernel(accelerator);
        for (int rep = 0; rep < 24; rep++)
            matMul.MatMul(a.View, b.View, c.View, M, K, N);

        // 🔴 GC PRESSURE IS THE POINT, not device load. The defect this gates against was NOT an
        // unfinished queue: MemoryBuffer.CopyToRawAsync handed clEnqueueReadBuffer (non-blocking) the raw
        // address of a MOVABLE managed byte[] via Unsafe.AsPointer, which does not pin. Every await inside
        // the chunk loop is a point where the GC may compact the heap and relocate that array, after which
        // the DMA lands on the old address and the caller reads a clean run of zeros.
        //
        // MEASURED before the fix: "1024 of 5000 values wrong (1024 of them ZERO), chunks affected [2] of
        // 5" - exactly ONE chunk zeroed with the chunks after it intact, which is the signature of a moved
        // destination rather than a late one.
        //
        // So this hammers the allocator DURING the copy. Collections are forced rather than hoped for,
        // because "it usually reproduces" is how this survived in the first place.
        using var gcPressure = new CancellationTokenSource();
        var churn = Task.Run(async () =>
        {
            var rnd = new Random(4242);
            while (!gcPressure.IsCancellationRequested)
            {
                // Gen-0 churn plus periodic compacting collections - what actually moves a pinned-less
                // buffer. The arrays are deliberately kept briefly so they survive into gen 1.
                var keep = new List<byte[]>();
                for (int k = 0; k < 64; k++) keep.Add(new byte[rnd.Next(4096, 65536)]);
                GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
                await Task.Yield();
            }
        });

        using var ms = new System.IO.MemoryStream();
        try
        {
            await tensor.CopyToStreamAsync(ms, chunkSizeInBytes: 4096);
            await accelerator.SynchronizeAsync();
        }
        finally
        {
            gcPressure.Cancel();
            try { await churn; } catch { /* cancellation is the expected exit */ }
        }

        var bytes = ms.ToArray();
        int expectedBytes = n * sizeof(float);
        if (bytes.Length < expectedBytes)
            throw new Exception($"CopyToStreamAsync wrote {bytes.Length} bytes, expected >= {expectedBytes}");

        var got = new float[n];
        Buffer.BlockCopy(bytes, 0, got, 0, expectedBytes);
        int firstBad = -1, zeros = 0;
        for (int i = 0; i < n; i++)
        {
            if (got[i] == data[i]) continue;
            if (firstBad < 0) firstBad = i;
            if (got[i] == 0f) zeros++;
        }
        if (firstBad >= 0)
            throw new Exception(
                $"CopyToStreamAsync under load lost data: first mismatch at [{firstBad}] "
              + $"(chunk {firstBad / 1024}), got {got[firstBad]} expected {data[firstBad]}; "
              + $"{zeros} of the mismatches are ZERO, which means the DMA had not landed when the "
              + "readback returned - the copy is not being awaited.");

        Console.WriteLine($"[TensorStream] busy-device round-trip byte-exact: {n} floats over "
                        + $"{(bytes.Length + 4095) / 4096} chunks, behind 24 queued matmuls");
    });
}
