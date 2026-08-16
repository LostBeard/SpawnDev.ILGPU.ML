using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// WEBGPU SD-Turbo image-gen REGRESSION GATE (Tuvok, preview.8 node-256 "external Instance reference no
/// longer exists" hunt). The cross-backend <c>SDTurbo_Generate_E2E</c> proves ONE cold generation on
/// every backend; it is HeavyModel (PMT-excluded by default) so it never re-ran on the WebGPU lane after
/// preview.8 — which is exactly how the regression shipped. THIS test is the missing WebGPU gate:
///   - WebGPU-only (the crash is WebGPU-specific: a disposed JS buffer instance at a batched-dispatch submit).
///   - MULTIPLE generations (cold gen-1 + several warm) with DISTINCT prompts/seeds, because the failure is
///     INTERMITTENT — a single render can pass. More generations + the warm shape-readback recycling path +
///     accumulated GPU memory pressure raise the reproduction probability toward certainty.
///   - <see cref="BufferPool.TraceReclaim"/> ON: if the under-pressure <c>AllocateWithReclaim</c> reclaim
///     (which disposes bucketed buffers after only a SYNC flush, not a drain, on WebGPU) fires mid-forward,
///     it logs the node + bytes, and the drain-fail message reports the reclaim count. A non-zero reclaim at
///     the crash confirms the disposed-while-referenced mechanism.
/// Each generation must produce a valid, non-degenerate 512x512 image; a crash or flat image fails the test
/// with the full diagnostic (op tail + reclaim count + node). Run:
///   PMT_EXCLUDE_CATEGORIES= PMT_FILTER=SDTurbo_WebGPU_ImageGen_MultiGen dotnet test PlaywrightMultiTest/...
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> SDTurbo_WebGPU_ImageGen_MultiGen() => await RunTestWithResult(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: WebGPU-only regression gate (the disposed-instance crash is WebGPU-specific; cross-backend coverage is SDTurbo_Generate_E2E)");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // OPFS-backed pieces, COLD (the DA-gate lesson, mirrored from SDTurbo_Generate_E2E): wipe the torrent
        // dir, then a FRESH client over the same IAsyncFS so no restored state pulls pieces into the .NET heap.
        var fs = GetAsyncFS();
        if (fs != null && await fs.DirectoryExists("webtorrent")) await fs.Remove("webtorrent", true);
        var client = fs != null
            ? new SpawnDev.WebTorrent.WebTorrentClient(new SpawnDev.WebTorrent.WebTorrentClientOptions { AsyncFileSystem = fs })
            : new SpawnDev.WebTorrent.WebTorrentClient();

        // Diagnostic: name any under-pressure reclaim fire (node + MiB). Reset the process-cumulative counters
        // so the numbers below are THIS test's. Restored in finally so we leave no cross-test global state.
        bool _savedTrace = BufferPool.TraceReclaim;
        bool _savedPeaks = BufferPool.TrackPeaks;
        BufferPool.TraceReclaim = true;
        BufferPool.TrackPeaks = true;   // per-gen peak-TOTAL leak guard (see the assert in the loop)
        BufferPool.ResetReclaimTrace();
        try
        {
            var hub = new Hub.HubModelStream(client, http);
            var pipe = await ImageGenerationPipeline.CreateAsync(accelerator, hub,
                Hub.ModelHub.KnownModels.SDTurbo,
                onProgress: (stage, pct) => Console.WriteLine($"[SDTurbo/load] {stage} {pct}%"));
            using (pipe)
            {
                if (!pipe.IsReady)
                    throw new Exception("SD-Turbo pipeline not ready after CreateAsync (a sub-model failed to load).");
                pipe.NumInferenceSteps = 1; // SD-Turbo is single-step
                pipe.GuidanceScale = 0f;    // SD-Turbo uses no classifier-free guidance

                // Distinct prompts + seeds per generation. TJ hit the crash on the FIRST cold gen; the warm
                // shape-readback cache activates from gen 3, so 4 gens exercise cold + probe + warm paths. Each
                // gen fully reads its image back (drained) before the next — matching the demo's usage.
                var prompts = new[] { "a chicken boxing match", "a nice house", "a watercolor fox", "a photo of a cat" };
                var results = new System.Text.StringBuilder();
                long gen1PeakTotal = 0;
                for (int g = 0; g < prompts.Length; g++)
                {
                    pipe.Seed = 42 + g;
                    long reclaimsBefore = BufferPool.ReclaimFireCount;
                    BufferPool.ResetPeaks();   // measure THIS gen's peak-TOTAL for the leak guard
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    // A crash surfaces here as the GraphExecutor drain-fail exception (op tail + reclaim count +
                    // node) — let it propagate so the test fails WITH the diagnostic rather than swallowing it.
                    var result = await pipe.RunAsync(new ImageGenerationInput { Prompt = prompts[g] });
                    sw.Stop();

                    int px = result.Width * result.Height;
                    if (result.Width != 512 || result.Height != 512)
                        throw new Exception($"gen {g + 1}: expected 512x512, got {result.Width}x{result.Height}.");
                    if (result.ImageRGBA.Length != 4 * px)
                        throw new Exception($"gen {g + 1}: image byte length {result.ImageRGBA.Length} != expected {4 * px}.");

                    long nonZero = 0; double sum = 0, sumSq = 0;
                    for (int i = 0; i < px; i++)
                    {
                        byte r = result.ImageRGBA[i * 4], gg = result.ImageRGBA[i * 4 + 1],
                             b = result.ImageRGBA[i * 4 + 2], a = result.ImageRGBA[i * 4 + 3];
                        if (a != 255) throw new Exception($"gen {g + 1}: alpha at px {i} = {a}, expected 255.");
                        if (r != 0 || gg != 0 || b != 0) nonZero++;
                        double lum = r + gg + b; sum += lum; sumSq += lum * lum;
                    }
                    double mean = sum / px, std = Math.Sqrt(Math.Max(0, sumSq / px - mean * mean));
                    long reclaimsThisGen = BufferPool.ReclaimFireCount - reclaimsBefore;
                    string tag = g == 0 ? "COLD" : g >= 2 ? "WARM" : "PROBE";
                    Console.WriteLine($"[SDTurbo-WebGPU] gen {g + 1}/{prompts.Length} ({tag}) \"{prompts[g]}\" {result.InferenceTimeMs:F0}ms " +
                        $"nonZero={nonZero}/{px} lumStd={std:F1} reclaims(this gen)={reclaimsThisGen} (total={BufferPool.ReclaimFireCount})");

                    if (nonZero < px / 100)
                        throw new Exception($"gen {g + 1}: image essentially all-black ({nonZero}/{px} non-zero) — diffusion produced no image.");
                    if (std < 5.0)
                        throw new Exception($"gen {g + 1}: image near-constant (lumStd={std:F1}) — flat/degenerate.");

                    // PER-GEN PEAK-TOTAL LEAK GUARD (Tuvok 2026-07-11): the tiled VAE decode's break-based partial
                    // run used to strand up_blocks.2 conv_shortcut + norm2 (128 MiB/gen) live -> pool couldn't
                    // reuse them -> peak-TOTAL grew +128 MiB EVERY gen -> browser GPU-process OOM on the 2nd gen
                    // ("external Instance reference no longer exists"). Fixed by returning break-stranded buffers.
                    // Guard: after the cold gen 1 warms the pool, no later gen may grow peak-TOTAL beyond a small
                    // margin (pre-fix this grew by a full 128 MiB feature-map per gen).
                    long peakTotalMiB = BufferPool.PeakTotalBytes / 1048576;
                    if (g == 0) gen1PeakTotal = peakTotalMiB;
                    else if (peakTotalMiB > gen1PeakTotal + 32)
                        throw new Exception($"gen {g + 1}: peak-TOTAL GPU memory grew to {peakTotalMiB} MiB (gen 1 = {gen1PeakTotal} MiB) " +
                            $"— a per-gen buffer leak (the SD-Turbo 2nd-gen OOM regression). Must stay flat on a warm resident pipeline.");
                    results.Append($"g{g + 1}:std={std:F1},peakTot={peakTotalMiB}MiB,rc={reclaimsThisGen} ");
                }
                var report = $"{prompts.Length} WebGPU generations OK | totalReclaims={BufferPool.ReclaimFireCount} " +
                    $"({BufferPool.ReclaimFreedBytes / 1048576.0:F0} MiB) | {results}";
                Console.WriteLine($"[SDTurbo-WebGPU] {report}");
                return report;
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network") || ex.Message.Contains("magnet"))
        {
            throw new UnsupportedTestException($"SD-Turbo hub/network unavailable: {ex.Message}");
        }
        finally
        {
            BufferPool.TraceReclaim = _savedTrace;
            BufferPool.TrackPeaks = _savedPeaks;
            await client.DisposeAsync();
        }
    });

    /// <summary>
    /// DETERMINISTIC node-256 REGRESSION GATE. A fat GPU (12GB) fits SD-Turbo with room to spare, so the
    /// under-pressure <c>AllocateWithReclaim</c> reclaim NEVER fires here (SDTurbo_WebGPU_ImageGen_MultiGen ran 4
    /// gens with totalReclaims=0). TJ's live failure needs GPU pressure this clean environment lacks. So this
    /// gate FORCES the reclaim disposal directly (<see cref="BufferPool.ForceReclaimEveryNRents"/>=8) through a
    /// real SD-Turbo WebGPU generation — injecting the exact production event without needing an OOM.
    ///
    /// CONFIRMED 2026-07-06: with an UN-fixed reclaim this crashed on the FIRST forced reclaim with
    /// "[Buffer] used in submit while destroyed" / "external Instance reference no longer exists" — a bucketed
    /// buffer disposed while still referenced by an UN-SUBMITTED WebGPU dispatch (AllocateWithReclaim's
    /// pre-reclaim flush is <c>Synchronize()</c>, which throws+swallowed on WebGPU, so the encoder was never
    /// submitted). The fix (<see cref="BufferPool.DisposeBucketedBuffers"/> now <c>Flush()</c>es before
    /// disposing) makes it GREEN: 238 forced reclaims disposing ~15GB mid-forward, image still valid. This gate
    /// LOCKS that fix in — it must stay green. Run:
    ///   PMT_EXCLUDE_CATEGORIES= PMT_FILTER=SDTurbo_WebGPU_ForcedReclaim_Probe dotnet test PlaywrightMultiTest/...
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> SDTurbo_WebGPU_ForcedReclaim_Probe() => await RunTestWithResult(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: WebGPU-only forced-reclaim probe");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var fs = GetAsyncFS();
        if (fs != null && await fs.DirectoryExists("webtorrent")) await fs.Remove("webtorrent", true);
        var client = fs != null
            ? new SpawnDev.WebTorrent.WebTorrentClient(new SpawnDev.WebTorrent.WebTorrentClientOptions { AsyncFileSystem = fs })
            : new SpawnDev.WebTorrent.WebTorrentClient();

        bool _savedTrace = BufferPool.TraceReclaim;
        int _savedForce = BufferPool.ForceReclaimEveryNRents;
        BufferPool.TraceReclaim = true;
        BufferPool.ResetReclaimTrace();
        try
        {
            var hub = new Hub.HubModelStream(client, http);
            var pipe = await ImageGenerationPipeline.CreateAsync(accelerator, hub,
                Hub.ModelHub.KnownModels.SDTurbo,
                onProgress: (stage, pct) => Console.WriteLine($"[SDTurbo/load] {stage} {pct}%"));
            using (pipe)
            {
                if (!pipe.IsReady)
                    throw new Exception("SD-Turbo pipeline not ready after CreateAsync.");
                pipe.NumInferenceSteps = 1;
                pipe.GuidanceScale = 0f;
                pipe.Seed = 42;

                // Turn on forced reclaim ONLY for the generation (not the weight load) so it injects reclaims
                // through CLIP + UNet + the VAE decode (where node-256 lives).
                BufferPool.ForceReclaimEveryNRents = 8;
                ImageGenerationResult result;
                try { result = await pipe.RunAsync(new ImageGenerationInput { Prompt = "a chicken boxing match" }); }
                finally { BufferPool.ForceReclaimEveryNRents = _savedForce; }

                int px = result.Width * result.Height;
                long nonZero = 0; double sum = 0, sumSq = 0;
                for (int i = 0; i < px; i++)
                {
                    byte r = result.ImageRGBA[i * 4], g = result.ImageRGBA[i * 4 + 1], b = result.ImageRGBA[i * 4 + 2];
                    if (r != 0 || g != 0 || b != 0) nonZero++;
                    double lum = r + g + b; sum += lum; sumSq += lum * lum;
                }
                double mean = sum / px, std = Math.Sqrt(Math.Max(0, sumSq / px - mean * mean));
                if (nonZero < px / 100) throw new Exception($"forced-reclaim: image all-black ({nonZero}/{px}).");
                if (std < 5.0) throw new Exception($"forced-reclaim: image flat (lumStd={std:F1}).");

                var report = $"GREEN with forced reclaim: {BufferPool.ReclaimFireCount} reclaims " +
                    $"({BufferPool.ReclaimFreedBytes / 1048576.0:F0} MiB disposed mid-forward) | image std={std:F1} " +
                    $"→ Flush-before-dispose fix holds: reclaim disposes bucketed buffers safely (was: crash at reclaim 1)";
                Console.WriteLine($"[SDTurbo-ForcedReclaim] {report}");
                return report;
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network") || ex.Message.Contains("magnet"))
        {
            throw new UnsupportedTestException($"SD-Turbo hub/network unavailable: {ex.Message}");
        }
        finally
        {
            BufferPool.TraceReclaim = _savedTrace;
            BufferPool.ForceReclaimEveryNRents = _savedForce;
            await client.DisposeAsync();
        }
    });

    /// <summary>
    /// WEBGPU SD-Turbo CAPTURE/REPLAY measurement (Tuvok 2026-07-10). Enables
    /// <see cref="ImageGenerationPipeline.EnableGraphCapture"/> (which WASM env cannot set) and runs 3
    /// generations at a FIXED 512x512 shape/seed: gen-1 pays the capture (2 warm forwards + record the
    /// WebGPUDispatchPlan), gen-2/3 REPLAY. Reports per-gen InferenceTimeMs so the replay speedup is
    /// visible, and validates every image is non-degenerate + no capture crash. Capture is BEST-EFFORT
    /// (a capture-unsafe op degrades to the direct forward), so this does NOT hard-assert a speedup — it
    /// measures it. A replay time ~= capture time means capture fell back (investigate); a large drop
    /// confirms the replay lever. Run:
    ///   PMT_EXCLUDE_CATEGORIES= PMT_FILTER=SDTurbo_WebGPU_Capture_Replay dotnet test PlaywrightMultiTest/...
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> SDTurbo_WebGPU_Capture_Replay() => await RunTestWithResult(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: WebGPU-only capture/replay measurement");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var fs = GetAsyncFS();
        if (fs != null && await fs.DirectoryExists("webtorrent")) await fs.Remove("webtorrent", true);
        var client = fs != null
            ? new SpawnDev.WebTorrent.WebTorrentClient(new SpawnDev.WebTorrent.WebTorrentClientOptions { AsyncFileSystem = fs })
            : new SpawnDev.WebTorrent.WebTorrentClient();
        try
        {
            var hub = new Hub.HubModelStream(client, http);
            var pipe = await ImageGenerationPipeline.CreateAsync(accelerator, hub,
                Hub.ModelHub.KnownModels.SDTurbo,
                onProgress: (stage, pct) => Console.WriteLine($"[SDTurbo/load] {stage} {pct}%"));
            using (pipe)
            {
                if (!pipe.IsReady) throw new Exception("SD-Turbo pipeline not ready after CreateAsync.");
                pipe.NumInferenceSteps = 1;
                pipe.GuidanceScale = 0f;
                pipe.EnableGraphCapture = true;   // the thing under measurement (WASM env can't set it)

                // FIXED prompt + seed → fixed shapes → gen-1 captures, gen-2/3 replay the same plan.
                const string prompt = "a lighthouse in a storm";
                pipe.Seed = 42;
                var times = new List<double>();
                for (int g = 0; g < 3; g++)
                {
                    var result = await pipe.RunAsync(new ImageGenerationInput { Prompt = prompt });
                    times.Add(result.InferenceTimeMs);
                    int px = result.Width * result.Height;
                    if (result.Width != 512 || result.Height != 512)
                        throw new Exception($"gen {g + 1}: expected 512x512, got {result.Width}x{result.Height}.");
                    long nonZero = 0; double sum = 0, sumSq = 0;
                    for (int i = 0; i < px; i++)
                    {
                        byte r = result.ImageRGBA[i * 4], gg = result.ImageRGBA[i * 4 + 1], b = result.ImageRGBA[i * 4 + 2];
                        if (r != 0 || gg != 0 || b != 0) nonZero++;
                        double lum = r + gg + b; sum += lum; sumSq += lum * lum;
                    }
                    double mean = sum / px, std = Math.Sqrt(Math.Max(0, sumSq / px - mean * mean));
                    string tag = g == 0 ? "CAPTURE" : "REPLAY";
                    Console.WriteLine($"[SDTurbo-WebGPU-Capture] gen {g + 1} ({tag}) {result.InferenceTimeMs:F0}ms nonZero={nonZero}/{px} lumStd={std:F1}");
                    if (nonZero < px / 100) throw new Exception($"gen {g + 1}: image all-black ({nonZero}/{px}) — no image produced.");
                    if (std < 5.0) throw new Exception($"gen {g + 1}: image near-constant (lumStd={std:F1}) — flat/degenerate.");
                }
                double capMs = times[0], replayMs = (times[1] + times[2]) / 2.0;
                var report = $"capture(gen1)={capMs:F0}ms replay(gen2-3 avg)={replayMs:F0}ms speedup={capMs / Math.Max(1, replayMs):F2}x";
                Console.WriteLine($"[SDTurbo-WebGPU-Capture] {report}");
                return report;
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network") || ex.Message.Contains("magnet"))
        {
            throw new UnsupportedTestException($"SD-Turbo hub/network unavailable: {ex.Message}");
        }
        finally { await client.DisposeAsync(); }
    });

    /// <summary>
    /// FULL-RES VAE DECODE PROBE (Tuvok 2026-07-12). Design experiment for Plans/vae-decode-gpu-resident-
    /// fix-2026-07-12.md: does a FULL-RES GPU-resident VAE decode (NO .NET tiling) survive on WebGPU when the
    /// buffer pool reclaims aggressively? CUDA measured the full-res VAE working set at 896 MiB LIVE but the
    /// pool HOARDED 3224 MiB (freed-but-retained buckets + deferred-release backlog) — THAT pool bloat crossed
    /// the browser per-process GPU budget, not genuine need. The current "fix" shoves whole feature maps into
    /// the .NET managed heap (a Rule-4 violation that starves the GPU and OOMs the WASM heap). This probe forces
    /// <see cref="ImageGenerationPipeline.VaeTileGrid"/>=-1 (full-res, WASM env can't set it) + proactive reclaim
    /// (<see cref="BufferPool.ForceReclaimEveryNRents"/> bucket-trim + a low
    /// <see cref="Graph.GraphExecutor.MaxPendingReleaseBytes"/> backlog cap) and generates once on WebGPU.
    ///   - GREEN + a peak-TOTAL far below 3224 MiB => Option A: delete the .NET tiling, run full-res GPU-resident.
    ///   - OOM ("out of memory" / device lost) => Option B: GPU-RESIDENT tiling (tiles stay GPU buffers) is needed.
    /// Either outcome decides the fix. Run:
    ///   PMT_EXCLUDE_CATEGORIES= PMT_FILTER=SDTurbo_WebGPU_FullResReclaim_Probe dotnet test PlaywrightMultiTest/...
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> SDTurbo_WebGPU_FullResReclaim_Probe() => await RunTestWithResult(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: WebGPU-only full-res-decode design probe");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var fs = GetAsyncFS();
        if (fs != null && await fs.DirectoryExists("webtorrent")) await fs.Remove("webtorrent", true);
        var client = fs != null
            ? new SpawnDev.WebTorrent.WebTorrentClient(new SpawnDev.WebTorrent.WebTorrentClientOptions { AsyncFileSystem = fs })
            : new SpawnDev.WebTorrent.WebTorrentClient();

        bool _savedTrace = BufferPool.TraceReclaim, _savedPeaks = BufferPool.TrackPeaks;
        int _savedForce = BufferPool.ForceReclaimEveryNRents;
        long _savedCap = SpawnDev.ILGPU.ML.Graph.GraphExecutor.MaxPendingReleaseBytes;
        BufferPool.TraceReclaim = true;
        BufferPool.TrackPeaks = true;
        BufferPool.ResetReclaimTrace();
        try
        {
            var hub = new Hub.HubModelStream(client, http);
            var pipe = await ImageGenerationPipeline.CreateAsync(accelerator, hub,
                Hub.ModelHub.KnownModels.SDTurbo,
                onProgress: (stage, pct) => Console.WriteLine($"[SDTurbo/load] {stage} {pct}%"));
            using (pipe)
            {
                if (!pipe.IsReady)
                    throw new Exception("SD-Turbo pipeline not ready after CreateAsync.");
                pipe.NumInferenceSteps = 1;
                pipe.GuidanceScale = 0f;
                pipe.Seed = 42;
                pipe.VaeTileGrid = -1;   // FORCE full-res GPU-resident VAE decode (no .NET tiling)

                BufferPool.ResetPeaks();
                ImageGenerationResult result;
                // Proactive reclaim ONLY for the generation: bucket-trim every 8 rents + a 64 MiB deferred-release
                // backlog cap, so the resident total tracks the working set instead of hoarding to 3224 MiB.
                SpawnDev.ILGPU.ML.Graph.GraphExecutor.MaxPendingReleaseBytes = 64L * 1024 * 1024;
                BufferPool.ForceReclaimEveryNRents = 8;
                try { result = await pipe.RunAsync(new ImageGenerationInput { Prompt = "a photo of a cat" }); }
                finally
                {
                    BufferPool.ForceReclaimEveryNRents = _savedForce;
                    SpawnDev.ILGPU.ML.Graph.GraphExecutor.MaxPendingReleaseBytes = _savedCap;
                }

                int px = result.Width * result.Height;
                if (result.Width != 512 || result.Height != 512)
                    throw new Exception($"expected 512x512, got {result.Width}x{result.Height}.");
                long nonZero = 0; double sum = 0, sumSq = 0;
                for (int i = 0; i < px; i++)
                {
                    byte r = result.ImageRGBA[i * 4], g = result.ImageRGBA[i * 4 + 1], b = result.ImageRGBA[i * 4 + 2];
                    if (r != 0 || g != 0 || b != 0) nonZero++;
                    double lum = r + g + b; sum += lum; sumSq += lum * lum;
                }
                double mean = sum / px, std = Math.Sqrt(Math.Max(0, sumSq / px - mean * mean));
                if (nonZero < px / 100) throw new Exception($"full-res: image all-black ({nonZero}/{px}) — no image produced.");
                if (std < 5.0) throw new Exception($"full-res: image flat (lumStd={std:F1}) — degenerate.");

                long peakTotalMiB = BufferPool.PeakTotalBytes / 1048576, peakLiveMiB = BufferPool.PeakLiveBytes / 1048576;
                var report = $"FULL-RES GPU-resident VAE decode SURVIVED on WebGPU: peakTOTAL={peakTotalMiB} MiB " +
                    $"peakLIVE={peakLiveMiB} MiB reclaims={BufferPool.ReclaimFireCount} " +
                    $"({BufferPool.ReclaimFreedBytes / 1048576.0:F0} MiB) image std={std:F1} " +
                    $"=> Option A viable (delete .NET tiling). CUDA baseline was 896 LIVE / 3224 TOTAL (default cap).";
                Console.WriteLine($"[SDTurbo-FullResReclaim] {report}");
                return report;
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network") || ex.Message.Contains("magnet"))
        {
            throw new UnsupportedTestException($"SD-Turbo hub/network unavailable: {ex.Message}");
        }
        finally
        {
            BufferPool.TraceReclaim = _savedTrace;
            BufferPool.TrackPeaks = _savedPeaks;
            BufferPool.ForceReclaimEveryNRents = _savedForce;
            SpawnDev.ILGPU.ML.Graph.GraphExecutor.MaxPendingReleaseBytes = _savedCap;
            await client.DisposeAsync();
        }
    });

    /// <summary>
    /// PER-STEP UNet COST measurement (Tuvok 2026-07-12, TJ request). Now that the VAE-decode memory path is
    /// wide open (full-res GPU-resident), measure the marginal cost of one SD-Turbo denoise step on WebGPU by
    /// running the SAME generation at NumInferenceSteps=1 vs =4. Only the denoise loop scales with steps
    /// (CLIP text-encode + VAE decode + RGBA pack are one-time), so per-step UNet ~ (t4 - t1) / 3, and the fixed
    /// (non-UNet) overhead ~ t1 - perStep. Warms the pipeline first (kernel compile + shape caches) so the
    /// numbers are steady-state; capture is OFF (EnableGraphCapture default false) so this is the raw per-node
    /// .NET-orchestration + GPU-exec cost. Reports the medians; does not hard-assert a threshold — it measures.
    ///   PMT_EXCLUDE_CATEGORIES= PMT_FILTER=SDTurbo_WebGPU_PerStepUnetCost dotnet test PlaywrightMultiTest/...
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> SDTurbo_WebGPU_PerStepUnetCost() => await RunTestWithResult(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: WebGPU-only per-step UNet cost measurement");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var fs = GetAsyncFS();
        if (fs != null && await fs.DirectoryExists("webtorrent")) await fs.Remove("webtorrent", true);
        var client = fs != null
            ? new SpawnDev.WebTorrent.WebTorrentClient(new SpawnDev.WebTorrent.WebTorrentClientOptions { AsyncFileSystem = fs })
            : new SpawnDev.WebTorrent.WebTorrentClient();
        try
        {
            var hub = new Hub.HubModelStream(client, http);
            var pipe = await ImageGenerationPipeline.CreateAsync(accelerator, hub,
                Hub.ModelHub.KnownModels.SDTurbo,
                onProgress: (stage, pct) => Console.WriteLine($"[SDTurbo/load] {stage} {pct}%"));
            using (pipe)
            {
                if (!pipe.IsReady)
                    throw new Exception("SD-Turbo pipeline not ready after CreateAsync.");
                pipe.GuidanceScale = 0f;
                pipe.Seed = 42;

                async Task<double> Gen(int steps)
                {
                    pipe.NumInferenceSteps = steps;
                    var r = await pipe.RunAsync(new ImageGenerationInput { Prompt = "a photo of a cat" });
                    if (r.Width != 512 || r.Height != 512)
                        throw new Exception($"steps={steps}: expected 512x512, got {r.Width}x{r.Height}.");
                    return r.InferenceTimeMs;
                }

                // Warm: kernel compile + shape-readback caches (the UNet graph is identical for steps=1 and =4,
                // so one warm-up covers both). Discard.
                await Gen(1); await Gen(1);

                var t1 = new List<double>(); for (int i = 0; i < 3; i++) t1.Add(await Gen(1));
                var t4 = new List<double>(); for (int i = 0; i < 3; i++) t4.Add(await Gen(4));
                t1.Sort(); t4.Sort();
                double m1 = t1[1], m4 = t4[1];                 // median of 3
                double perStep = (m4 - m1) / 3.0;              // 3 extra UNet steps between steps=1 and steps=4
                double fixedCost = m1 - perStep;               // CLIP text-encode + VAE decode + RGBA pack

                var report = $"steps=1 median={m1:F0}ms (runs {string.Join("/", t1.Select(x => x.ToString("F0")))}) | " +
                    $"steps=4 median={m4:F0}ms ({string.Join("/", t4.Select(x => x.ToString("F0")))}) => " +
                    $"per-step UNet ~{perStep:F0}ms; fixed CLIP+VAE+pack ~{fixedCost:F0}ms";
                // JS.LogError => console.error, which PMT captures (dumps the text) WITHOUT tripping
                // #blazor-error-ui (info-level Console.WriteLine is summarized-away by PMT). Same pattern as
                // SDTurboProfile. Does not fail the test.
                SpawnDev.SpawnJS.SpawnJSRuntime.Instance?.LogError("[SDTurbo-PerStep] " + report);
                return report;
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network") || ex.Message.Contains("magnet"))
        {
            throw new UnsupportedTestException($"SD-Turbo hub/network unavailable: {ex.Message}");
        }
        finally { await client.DisposeAsync(); }
    });
}
