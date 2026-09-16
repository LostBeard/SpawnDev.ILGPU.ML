using System.Diagnostics;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.SpawnJS;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// End-to-end load time for a real ~1.8 GB GGUF through the HTTP + OPFS path, split into COLD (download +
/// load) and WARM (load from cache).
/// </summary>
/// <remarks>
/// <para>
/// This exists because the whole delivery rework was driven by one number - a 1.6-1.8 GB model taking ~300s
/// to become resident in the SpawnDev.AI demo - and every measurement so far has been of a COMPONENT (OPFS
/// read throughput, managed allocation, chunk sizes). A component measurement cannot answer "how long does
/// the model take to load", and quoting one as if it did would be the same mistake as reporting a desktop
/// figure for a browser lane.
/// </para>
/// <para>
/// The split matters more than the total: if the time is in DOWNLOAD, delivery is the lever and the network
/// is the floor; if it is in the WARM load, delivery was never the problem and the cost is OPFS read + GPU
/// upload + graph compile. <c>TraceWeightLoad</c> prints that second breakdown (stream READ vs GPU WRITE).
/// </para>
/// <para>
/// Category "HeavyModel" - run explicitly:
/// <c>PMT_EXCLUDE_CATEGORIES=__none__ PMT_FILTER=LargeModel_LoadBenchmark PMT_CONSOLE_LOG=LoadBench</c>
/// </para>
/// </remarks>
public abstract partial class MLTestBase
{
    /// <summary>Qwen3 1.7B Q8_0 - the SpawnDev.AI demo's RECOMMENDED model, and the size class behind the ~300s report.</summary>
    private const string BenchRepo = "Qwen/Qwen3-1.7B-GGUF";
    private const string BenchFile = "Qwen3-1.7B-Q8_0.gguf";
    private const long BenchApproxBytes = 1_834_426_016;

    [TestMethod(Timeout = 1_800_000, Category = "HeavyModel")]
    public async Task LargeModel_LoadBenchmark() => await RunTest(async accelerator =>
    {
        var js = SpawnJSRuntime.Instance;
        if (js == null || !js.IsBrowser)
            throw new UnsupportedTestException("Delivery benchmark is browser-only (OPFS)");

        // WebGPU only. On WebGL/Wasm a 1.8 GB model is not a delivery measurement, it is a backend
        // measurement - and per the standing backend-priority rule those lanes do not gate progress.
        if (accelerator.AcceleratorType.ToString().IndexOf("WebGPU", StringComparison.OrdinalIgnoreCase) < 0)
            throw new UnsupportedTestException($"WebGPU only; this lane is {accelerator.AcceleratorType}");

        using var source = new HubModelSource(js);
        var key = source.CacheKey(BenchRepo, BenchFile);

        // Start genuinely cold so the download half is real.
        await source.Store.RemoveAsync(key);

        var progressReports = 0;
        long lastBytes = 0, totalBytes = -1;
        var downloadWatch = Stopwatch.StartNew();
        var sink = new SyncProgress();
        var relay = new RelayProgress(p =>
        {
            progressReports++; lastBytes = p.BytesReceived; totalBytes = p.TotalBytes;
            sink.Report(p);
        });

        // ── COLD: download + open ────────────────────────────────────────────────────────────────────
        var cold = await source.OpenAsync(BenchRepo, BenchFile, relay);
        downloadWatch.Stop();
        long fileBytes;
        await using (cold.ConfigureAwait(false)) fileBytes = cold.Length;

        var mb = fileBytes / 1048576.0;
        var dlSec = downloadWatch.Elapsed.TotalSeconds;
        Console.WriteLine($"[LoadBench] file {fileBytes:N0} B ({mb:F0} MiB), expected ~{BenchApproxBytes:N0}");
        Console.WriteLine($"[LoadBench] DOWNLOAD to OPFS: {dlSec:F1}s = {mb / Math.Max(0.001, dlSec):F1} MB/s " +
                          $"({progressReports} progress reports, last {lastBytes:N0}/{totalBytes:N0})");

        // ── WARM: load from the cache onto the GPU ───────────────────────────────────────────────────
        // TraceWeightLoad prints the stream-READ vs GPU-WRITE split, which is what says whether the
        // remaining time is delivery or upload.
        var prevTrace = InferenceSession.TraceWeightLoad;
        InferenceSession.TraceWeightLoad = true;

        // 🔴 THE SPLIT THIS BENCHMARK KEPT PROMISING AND NEVER PRINTED. The remarks above say
        // "TraceWeightLoad prints the stream-READ vs GPU-WRITE split", but TraceWeightLoad only logs
        // per-tensor anomalies - the actual read/write accumulators live on BrowserStreamUpload behind
        // their OWN flag, and nothing ever set it. So the one question this test exists to answer -
        // is a slow load the OPFS read or the queue.writeBuffer upload - has never had an answer.
        //
        // TJ 2026-09-15: "turn that trace on and get the split." The open number it is aimed at: a
        // recorded 92.8 s load of which 71 s was attributed to "upload", on a path that streams 16 MiB
        // chunks JS-side and should move 1.8 GB in a couple of seconds.
        var prevStreamTrace = SpawnDev.ILGPU.BrowserBufferPolicy.TraceStreamUploadTiming;
        SpawnDev.ILGPU.BrowserBufferPolicy.TraceStreamUploadTiming = true;
        SpawnDev.ILGPU.BrowserBufferPolicy.ResetStreamUploadTiming();
        try
        {
            var stages = new List<(string Stage, double Sec)>();
            var stageWatch = Stopwatch.StartNew();
            var lastStageAt = 0.0;

            var loadWatch = Stopwatch.StartNew();
            var warm = await source.OpenAsync(BenchRepo, BenchFile);
            var openSec = loadWatch.Elapsed.TotalSeconds;

            GgufTextGenerationPipeline pipe;
            await using (warm.ConfigureAwait(false))
            {
                pipe = await GgufTextGenerationPipeline.CreateFromStreamAsync(accelerator, warm,
                    maxSeqLen: 4096,
                    onProgress: (stage, pct) =>
                    {
                        if (pct != 100) return;
                        var now = stageWatch.Elapsed.TotalSeconds;
                        stages.Add((stage, now - lastStageAt));
                        lastStageAt = now;
                    });
            }
            loadWatch.Stop();

            using (pipe)
            {
                var loadSec = loadWatch.Elapsed.TotalSeconds;
                Console.WriteLine($"[LoadBench] WARM LOAD (cached -> GPU): {loadSec:F1}s " +
                                  $"({mb / Math.Max(0.001, loadSec):F1} MB/s effective), of which " +
                                  $"cache open {openSec:F2}s");
                foreach (var (stage, sec) in stages)
                    Console.WriteLine($"[LoadBench]   stage {stage,-12} {sec,7:F1}s");

                // ── THE SPLIT ────────────────────────────────────────────────────────────────────
                // readMs  = IJSReadStream.ReadUint8ArrayAsync  (OPFS / delivery)
                // writeMs = IBrowserMemoryBuffer.CopyFromJS    (queue.writeBuffer, JS -> GPU)
                // Whichever dominates names the lever. If NEITHER does, the time is in the graph
                // walk around them and chunk size is irrelevant.
                var readMs = SpawnDev.ILGPU.BrowserBufferPolicy.StreamReadMs;
                var writeMs = SpawnDev.ILGPU.BrowserBufferPolicy.StreamWriteMs;
                var sBytes = SpawnDev.ILGPU.BrowserBufferPolicy.StreamBytes;
                var sChunks = SpawnDev.ILGPU.BrowserBufferPolicy.StreamChunks;
                var sMiB = sBytes / 1048576.0;
                Console.WriteLine($"[LoadBench] STREAM SPLIT: {sMiB:F0} MiB over {sChunks:N0} chunks "
                    + $"({(sChunks > 0 ? sMiB / sChunks : 0):F1} MiB/chunk) | "
                    + $"read {readMs:F0} ms ({(readMs > 0 ? sMiB / (readMs / 1000.0) : 0):F0} MB/s) | "
                    + $"write {writeMs:F0} ms ({(writeMs > 0 ? sMiB / (writeMs / 1000.0) : 0):F0} MB/s) | "
                    + $"read+write {readMs + writeMs:F0} ms of a {loadSec * 1000:F0} ms load "
                    + $"({(loadSec > 0 ? (readMs + writeMs) / (loadSec * 1000) * 100 : 0):F0}%)");
                if (sChunks == 0)
                    Console.WriteLine("[LoadBench] ⚠️ ZERO chunks streamed - this load did NOT take the "
                        + "JS-side streaming path, so the split above describes nothing. Check that the "
                        + "source is an IJSReadStream (the .NET fallback CopyFromCPUs instead).");

                Console.WriteLine($"[LoadBench] COLD TOTAL (download + load): {dlSec + loadSec:F1}s");

                // Prove the model actually works - a load benchmark that measured a broken pipeline would
                // be worse than no number at all.
                var text = await pipe.GenerateAsync(
                    new[] { ("user", "Reply with exactly one word: ready") },
                    config: new GenerationConfig { MaxNewTokens = 8, Strategy = "greedy" });
                Console.WriteLine($"[LoadBench] generation check: \"{text?.Trim()}\"");
                if (string.IsNullOrWhiteSpace(text))
                    throw new Exception("Model loaded but generated nothing - the timing above is of a broken load.");
            }

            if (fileBytes < BenchApproxBytes / 2)
                throw new Exception($"Downloaded {fileBytes:N0} bytes, expected ~{BenchApproxBytes:N0} - wrong file?");
        }
        finally
        {
            InferenceSession.TraceWeightLoad = prevTrace;
            SpawnDev.ILGPU.BrowserBufferPolicy.TraceStreamUploadTiming = prevStreamTrace;
        }
    });

    /// <summary>Synchronous relay so progress counts are the downloader's, not the scheduler's.</summary>
    private sealed class RelayProgress : IProgress<ModelDownloadProgress>
    {
        private readonly Action<ModelDownloadProgress> _onReport;
        public RelayProgress(Action<ModelDownloadProgress> onReport) => _onReport = onReport;
        public void Report(ModelDownloadProgress value) => _onReport(value);
    }
}
