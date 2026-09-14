using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.SpawnJS.Toolbox;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Measures OPFS read and write throughput as a function of chunk size, PAST the range that has been
/// tested before.
/// </summary>
/// <remarks>
/// <para>
/// What was known: 64 KiB reads ran at ~84 MB/s and 4 MiB reads at ~985 MB/s. 4 MiB was simply the largest
/// size tried, NOT a measured optimum - so every "4 MB is enough" choice in the stack rests on the top of a
/// curve nobody has seen the end of. ILGPU already defaults to 16 MiB
/// (<c>MemoryBuffer.DefaultStreamChunkSizeInBytes</c>) and <c>BufferPool</c>'s quantized path passes 4 MiB,
/// so the two differ by 4x with no measurement behind either.
/// </para>
/// <para>
/// This sweeps 64 KiB -&gt; 64 MiB against a real OPFS file, on the exact stream type model loading uses
/// (<see cref="OPFSStream"/>), and prints MB/s per size. It is a MEASUREMENT, not a pass/fail guard: the
/// only assertion is that the biggest chunk is not dramatically slower than the smallest, which would mean
/// the sweep itself is broken. Read the printed table to choose sizes.
/// </para>
/// <para>Category "Benchmark" so it is excluded from the fast loop. Run it with
/// <c>PMT_EXCLUDE_CATEGORIES=__none__ PMT_FILTER=Opfs_ChunkSizeSweep</c>.</para>
/// </remarks>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 600000, Category = "Benchmark")]
    public async Task Opfs_ChunkSizeSweep() => await RunTest(async accelerator =>
    {
        var js = SpawnJSRuntime.Instance;
        if (js == null || !js.IsBrowser)
            throw new UnsupportedTestException("OPFS is browser-only");

        const string dirName = "ilgpu-ml-test-chunksweep";
        const string fileName = "sweep.bin";
        // 192 MiB: big enough that a 64 MiB chunk still takes several chunks, small enough to stay polite
        // about disk. Must be a multiple of every chunk size tested so the last read is never a short tail.
        const long fileSize = 192L * 1024 * 1024;
        int[] chunkSizes = { 64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024, 16 * 1024 * 1024, 32 * 1024 * 1024, 64 * 1024 * 1024 };

        using var navigator = js.Get<Navigator>("navigator");
        using var storage = navigator.Storage;
        using var root = await storage.GetDirectory();
        using var dir = await root.GetDirectoryHandle(dirName, create: true);

        try { await dir.RemoveEntry(fileName); } catch { /* absent */ }

        // ── Build the file, measuring WRITE throughput at a fixed large chunk ────────────────────────
        // The payload is one JS-side Uint8Array written repeatedly: the bytes never enter the managed heap,
        // so this measures OPFS, not marshalling.
        const int buildChunk = 16 * 1024 * 1024;
        using (var payload = new Uint8Array(buildChunk))
        {
            var swWrite = System.Diagnostics.Stopwatch.StartNew();
            var w = await OPFSStream.OpenPath(dir, fileName, FileMode.Create, FileAccess.Write);
            await using (w.ConfigureAwait(false))
            {
                for (long done = 0; done < fileSize; done += buildChunk)
                    await w.WriteUint8ArrayAsync(payload);
                await w.FlushAsync();
            }
            swWrite.Stop();
            Console.WriteLine($"[ChunkSweep] wrote {fileSize / 1048576} MiB at {buildChunk / 1048576} MiB chunks: " +
                              $"{fileSize / 1048576.0 / swWrite.Elapsed.TotalSeconds,8:F1} MB/s");
        }

        // ── Read the whole file at each chunk size ───────────────────────────────────────────────────
        Console.WriteLine("[ChunkSweep] chunk        MB/s     elapsed   reads");
        double best = 0; int bestSize = 0; double at4MiB = 0, atLargest = 0, atSmallest = 0;
        foreach (var chunk in chunkSizes)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            long total = 0; int reads = 0;
            var r = await OPFSStream.OpenPath(dir, fileName, FileMode.Open, FileAccess.Read);
            await using (r.ConfigureAwait(false))
            {
                while (total < fileSize)
                {
                    // ReadUint8ArrayAsync: the bytes stay JS-side, exactly as the weight-upload path reads.
                    using var u8 = await r.ReadUint8ArrayAsync(chunk);
                    var got = u8?.Length ?? 0;
                    if (got <= 0) break;
                    total += got;
                    reads++;
                }
            }
            sw.Stop();
            var mbps = total / 1048576.0 / sw.Elapsed.TotalSeconds;
            Console.WriteLine($"[ChunkSweep] {chunk / 1024,7} KiB {mbps,9:F1}  {sw.Elapsed.TotalMilliseconds,8:F0} ms {reads,7}");
            if (total != fileSize)
                throw new Exception($"Read {total} of {fileSize} bytes at chunk {chunk} - the sweep is measuring the wrong thing.");
            if (mbps > best) { best = mbps; bestSize = chunk; }
            if (chunk == 4 * 1024 * 1024) at4MiB = mbps;
            if (chunk == chunkSizes[^1]) atLargest = mbps;
            if (chunk == chunkSizes[0]) atSmallest = mbps;
        }

        Console.WriteLine($"[ChunkSweep] FASTEST: {bestSize / 1024} KiB at {best:F1} MB/s");
        if (at4MiB > 0)
            Console.WriteLine($"[ChunkSweep] vs 4 MiB ({at4MiB:F1} MB/s): best is {best / at4MiB:F2}x; " +
                              $"largest ({chunkSizes[^1] / 1048576} MiB) is {atLargest / at4MiB:F2}x");

        // The ONLY assertion: a sweep where the largest chunk is slower than the smallest means the harness
        // is broken (or OPFS is pathological), not that big chunks are bad.
        if (atLargest < atSmallest)
            throw new Exception(
                $"Largest chunk ({chunkSizes[^1] / 1048576} MiB, {atLargest:F1} MB/s) was SLOWER than the " +
                $"smallest ({chunkSizes[0] / 1024} KiB, {atSmallest:F1} MB/s). Sweep is suspect.");

        try { await dir.RemoveEntry(fileName); } catch { }
    });
}
