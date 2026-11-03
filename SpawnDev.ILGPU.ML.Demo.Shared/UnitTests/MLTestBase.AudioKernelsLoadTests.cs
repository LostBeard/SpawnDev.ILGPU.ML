using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// A measurement (Riker's question, 2026-11-02): <see cref="AudioKernels"/> calls LoadAutoGroupedStreamKernel inside each method, and ApplyHannWindow's
    /// kernel is a lambda capturing piTerm / frameSize. Times four consecutive calls of ApplyHannWindow (capturing lambda) and PowerSpectrum (non-capturing)
    /// on a 4,000-sample buffer, each synchronised: a kernel this small runs in microseconds, so later calls in the millisecond range mean a load or compile per
    /// call. Desktop lanes (the browser's timers are coarse and a sync there costs the same order).
    /// </summary>
    [TestMethod]
    public async Task Bench_AudioKernels_PerCallLoadCost() => await RunTest(async accelerator =>
    {
        if (OperatingSystem.IsBrowser()) throw new UnsupportedTestException("a desktop timing");
        var audio = new AudioKernels(accelerator);
        using var buf = accelerator.Allocate1D(Enumerable.Range(0, 4000).Select(i => (float)Math.Sin(i * 0.01)).ToArray());
        string Time(string name, Action call)
        {
            var ms = new double[4];
            for (int i = 0; i < 4; i++)
            {
                var sw = Stopwatch.StartNew();
                call();
                accelerator.Synchronize();
                ms[i] = sw.Elapsed.TotalMilliseconds;
            }
            return $"{name}: " + string.Join(" / ", ms.Select(m => $"{m:F2}")) + " ms";
        }
        var report = $"{accelerator.AcceleratorType}: " + Time("ApplyHannWindow (capturing lambda)", () => audio.ApplyHannWindow(buf.View, 400))
            + "; " + Time("PowerSpectrum", () => audio.PowerSpectrum(buf.View, 4000));
        await Task.CompletedTask;
        Console.WriteLine("[AudioLoad] " + report);
        return report;
    });
}
