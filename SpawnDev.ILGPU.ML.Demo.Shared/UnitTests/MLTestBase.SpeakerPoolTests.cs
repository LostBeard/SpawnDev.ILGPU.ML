using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// A measurement (Data, 2026-11-03: ReDimNet2-B6, 51 MB of weights, held a 2,141 MiB working pool after warm-up in SpawnDev.AI): the buffer pool's peak
    /// TOTAL allocation against its peak LIVE bytes (rented and not yet returned) over two 6 s speaker embeddings, the pool's free bytes after them, and the
    /// largest live buffers at the live peak. Total far above live means the pool's exact-size buckets keep idle buffers of one size that later rents of other
    /// sizes cannot use; live itself near the total means the intermediates really are that large or are held too long. Desktop CUDA lane (the model loads
    /// from Hugging Face directly).
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task Bench_SpeakerEmbedding_PoolPeak() => await RunTest(async accelerator =>
    {
        if (OperatingSystem.IsBrowser() || accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL))
            throw new UnsupportedTestException("a desktop measurement (CUDA / OpenCL)");
        using var http = CreateHuggingFaceHttpClient();
        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http, HuggingFaceClient.GetDownloadUrl("soniqo/ReDimNet2-B6-ONNX-FP32", "ReDimNet2B6.onnx"));
        bool trackSaved = BufferPool.TrackPeaks, compSaved = BufferPool.TrackLivePeakComposition;
        BufferPool.TrackPeaks = true; BufferPool.TrackLivePeakComposition = true;
        try
        {
            using var session = InferenceSession.CreateFromFile(accelerator, modelBytes);
            using var pipeline = new SpeakerEmbeddingPipeline(session, accelerator);
            var rng = new Random(3);
            var clip = Enumerable.Range(0, 96000).Select(i => (float)(0.3 * Math.Sin(i * 0.05) + 0.05 * (rng.NextDouble() * 2 - 1))).ToArray();
            // Five embeddings, peaks reset before each: with BufferPool.PlannedReuse the runs record until two consecutive rent sequences match, then plan;
            // the later ones replay the plan. Each must
            // equal the first bit for bit: which buffer serves a tensor never changes a result.
            const double MiB = 1048576.0;
            var pool = session.Executor.Pool;
            long plansBefore = BufferPool.PlannedReusePlans, rentsBefore = BufferPool.PlannedReuseRents, devBefore = BufferPool.PlannedReuseDeviations;
            var parts = new List<string>();
            float[]? first = null;
            for (int run = 1; run <= 5; run++)
            {
                BufferPool.ResetPeaks();
                var emb = await pipeline.EmbedAsync(clip, 16000);
                if (first == null) first = emb;
                else if (!emb.SequenceEqual(first)) throw new Exception($"embedding {run} differs from embedding 1 (max |d| {emb.Zip(first).Max(z => Math.Abs(z.First - z.Second)):G3})");
                parts.Add($"embedding {run}: peak total {BufferPool.PeakTotalBytes / MiB:F0} MiB, peak live {BufferPool.PeakLiveBytes / MiB:F0} MiB, pool free {session.PooledFreeBytes / MiB:F0} MiB, planned slots {pool.PlannedSlotBytes / MiB:F0} MiB");
            }
            var top = (BufferPool.PeakLiveSnapshot ?? new()).OrderByDescending(s => s.bytes).Take(12)
                .Select(s => $"{s.name} {s.bytes / 1048576.0:F1} MiB");
            var report = $"{accelerator.AcceleratorType}: weights {modelBytes.Length / MiB:F0} MiB, planned reuse {(BufferPool.PlannedReuse ? "on" : "off")} "
                + $"(plans +{BufferPool.PlannedReusePlans - plansBefore}, planned rents +{BufferPool.PlannedReuseRents - rentsBefore}, deviations +{BufferPool.PlannedReuseDeviations - devBefore}) | "
                + string.Join(" | ", parts) + $" | largest live at the last peak: {string.Join(", ", top)}";
            Console.WriteLine("[SpeakerPool] " + report);
            return report;
        }
        finally { BufferPool.TrackPeaks = trackSaved; BufferPool.TrackLivePeakComposition = compSaved; }
    });

    /// <summary>
    /// <see cref="Graph.GraphExecutor.ImmediateRelease"/> under reclaim, in Vertex's shape (Data, 2026-11-03: chat, speech and speaker engines share ONE
    /// accelerator, and a memory-pressure reclaim can free another session's free buckets): two ReDimNet2-B6 sessions on one accelerator embed alternately
    /// while the pool frees every free bucket every few rents (<see cref="BufferPool.ForceReclaimEveryNRents"/> = 3, then 7). With immediate release a free
    /// bucket can hold a buffer a queued kernel still reads, so the reclaim must wait for the queue first (BufferPool.DrainBeforeFree on CUDA / OpenCL); every
    /// embedding must equal the one computed without forced reclaims, bit for bit. Desktop CUDA / OpenCL.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task ImmediateRelease_TwoSessionsUnderForcedReclaim_MatchUndisturbed() => await RunTest(async accelerator =>
    {
        if (OperatingSystem.IsBrowser() || accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL))
            throw new UnsupportedTestException("desktop CUDA / OpenCL (the ordered-queue backends whose reclaim waits)");
        if (!Graph.GraphExecutor.ImmediateRelease) throw new UnsupportedTestException("ImmediateRelease is off (ML_IMMEDIATE_RELEASE=0)");
        using var http = CreateHuggingFaceHttpClient();
        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http, HuggingFaceClient.GetDownloadUrl("soniqo/ReDimNet2-B6-ONNX-FP32", "ReDimNet2B6.onnx"));
        using var sA = InferenceSession.CreateFromFile(accelerator, modelBytes);
        using var sB = InferenceSession.CreateFromFile(accelerator, modelBytes);
        using var pA = new SpeakerEmbeddingPipeline(sA, accelerator);
        using var pB = new SpeakerEmbeddingPipeline(sB, accelerator);
        var rng = new Random(9);
        float[] Clip(double f) => Enumerable.Range(0, 96000).Select(i => (float)(0.3 * Math.Sin(i * f) + 0.05 * (rng.NextDouble() * 2 - 1))).ToArray();
        var clipA = Clip(0.05); var clipB = Clip(0.031);
        var refA = await pA.EmbedAsync(clipA, 16000);
        var refB = await pB.EmbedAsync(clipB, 16000);
        int saved = BufferPool.ForceReclaimEveryNRents;
        var notes = new List<string>();
        try
        {
            foreach (int every in new[] { 3, 7 })
            {
                BufferPool.ForceReclaimEveryNRents = every;
                for (int round = 0; round < 2; round++)
                {
                    var a = await pA.EmbedAsync(clipA, 16000);
                    var b = await pB.EmbedAsync(clipB, 16000);
                    int diffA = a.Zip(refA).Count(z => z.First != z.Second), diffB = b.Zip(refB).Count(z => z.First != z.Second);
                    if (diffA > 0 || diffB > 0)
                        throw new Exception($"forced reclaim every {every} rents, round {round}: {diffA} of {a.Length} values of session A and {diffB} of session B differ from the undisturbed embeddings (cosine A {SpeakerEmbeddingPipeline.Cosine(a, refA):F6}, B {SpeakerEmbeddingPipeline.Cosine(b, refB):F6})");
                }
                notes.Add($"reclaim every {every} rents: both sessions bit-identical over 2 rounds");
            }
        }
        finally { BufferPool.ForceReclaimEveryNRents = saved; }
        return $"{accelerator.AcceleratorType}: " + string.Join("; ", notes);
    });
}
