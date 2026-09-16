using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Gate for the download segment sizing law (<see cref="HttpModelDownloader.SegmentBytesForRate"/>).
///
/// 🔴 WHY A SEPARATE GATE. A segment is simultaneously the download's throughput unit, its PROGRESS
/// increment, and its RESUME checkpoint. A size picked for the first of those silently sets the other
/// two, and that shipped: a flat 64 MiB segment meant a progress bar that froze for 32 s at a time on a
/// 2 MB/s link and re-downloaded up to 64 MiB after a drop - worst precisely on the connections that
/// drop. `OpfsModelCache_ProgressFiresDuringDownload` caught the extreme case (a 2 MB file inside one
/// segment reported only 0% and 100%) but it downloads 2 MB over a fast LAN, so it CANNOT observe the
/// behaviour on a slow connection, which is the case TJ raised and the case this covers.
///
/// MEASURED on the 1.83 GB model before settling on the rule - fixed sizes cannot satisfy both ends:
///   64 MiB fixed            35.8 s  48.8 MB/s    31 reports
///    4 MiB fixed            46.3 s  37.8 MB/s   441 reports   (29% slower: per-request latency + ramp)
///   time-sized, 16 MiB cap  35.9 s  48.7 MB/s   113 reports   (shipped)
/// </summary>
public abstract partial class MLTestBase
{
    private const long SegMin = 512 * 1024;
    private const long SegMax = 16 * 1024 * 1024;
    private const double SegTargetSec = 1.5;

    [TestMethod]
    public async Task DownloadSegmentSizing_HoldsTheUpdateIntervalAcrossConnectionSpeeds() => await RunTest(async accelerator =>
    {
        // (label, MB/s on ONE connection)
        var links = new (string Name, double MBs)[]
        {
            ("3G-ish",      0.5),
            ("slow DSL",    2.0),
            ("cable",      12.0),
            ("fast LAN",   50.0),
            ("gigabit",   120.0),
        };

        foreach (var (name, mbs) in links)
        {
            var bytesPerMs = mbs * 1_000_000.0 / 1000.0;
            var seg = HttpModelDownloader.SegmentBytesForRate(bytesPerMs, SegTargetSec, SegMin, SegMax);
            var seconds = seg / (mbs * 1_000_000.0);
            Console.WriteLine($"[SegSizing] {name,-10} {mbs,6:F1} MB/s -> segment {seg / 1024.0 / 1024.0,6:F2} MiB "
                + $"= {seconds:F2}s per update");

            if (seg < SegMin || seg > SegMax)
                throw new Exception($"{name}: segment {seg} outside [{SegMin}, {SegMax}].");

            // THE PROPERTY: an update interval a human reads as a live bar. Only the clamps may break it,
            // and only in the directions the clamps exist for - never a 30-second freeze in the middle.
            if (seg > SegMin && seg < SegMax && Math.Abs(seconds - SegTargetSec) > 0.01)
                throw new Exception($"{name}: unclamped segment should take {SegTargetSec}s, takes {seconds:F2}s.");
            if (seconds > 4.0)
                throw new Exception($"{name}: {seconds:F1}s between progress updates - that reads as a frozen bar. "
                    + "This is the 64 MiB regression returning.");
        }
        Console.WriteLine("[SegSizing] update interval holds across 0.5-120 MB/s");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task DownloadSegmentSizing_ClampsAndDegradesSafely() => await RunTest(async accelerator =>
    {
        // A stalled or unmeasured connection must not ask for a zero/negative range.
        foreach (var bad in new[] { 0.0, -1.0, double.NaN })
            if (HttpModelDownloader.SegmentBytesForRate(bad, SegTargetSec, SegMin, SegMax) != SegMin)
                throw new Exception($"rate {bad} did not fall back to the minimum segment.");
        if (HttpModelDownloader.SegmentBytesForRate(1e12, SegTargetSec, SegMin, SegMax) != SegMax)
            throw new Exception("An absurd rate was not clamped to the ceiling.");
        if (HttpModelDownloader.SegmentBytesForRate(1.0, SegTargetSec, SegMin, SegMax) != SegMin)
            throw new Exception("A crawling rate was not clamped to the floor.");

        // Monotonic: a faster link never asks for less.
        long prev = 0;
        for (double mbs = 0.25; mbs <= 256; mbs *= 2)
        {
            var seg = HttpModelDownloader.SegmentBytesForRate(mbs * 1000.0, SegTargetSec, SegMin, SegMax);
            if (seg < prev) throw new Exception($"Segment shrank as the link got faster: {prev} -> {seg}.");
            prev = seg;
        }
        Console.WriteLine("[SegSizing] clamps + monotonicity OK");
        await Task.CompletedTask;
    });
}
