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
                    throw new Exception("SD-Turbo pipeline not ready after CreateAsync (a sub-model failed to load).");
                pipe.NumInferenceSteps = 1; // SD-Turbo is single-step
                pipe.GuidanceScale = 0f;    // SD-Turbo uses no classifier-free guidance

                // Distinct prompts + seeds per generation. TJ hit the crash on the FIRST cold gen; the warm
                // shape-readback cache activates from gen 3, so 4 gens exercise cold + probe + warm paths. Each
                // gen fully reads its image back (drained) before the next — matching the demo's usage.
                var prompts = new[] { "a chicken boxing match", "a nice house", "a watercolor fox", "a photo of a cat" };
                var results = new System.Text.StringBuilder();
                for (int g = 0; g < prompts.Length; g++)
                {
                    pipe.Seed = 42 + g;
                    long reclaimsBefore = BufferPool.ReclaimFireCount;
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
                    results.Append($"g{g + 1}:std={std:F1},rc={reclaimsThisGen} ");
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
}
