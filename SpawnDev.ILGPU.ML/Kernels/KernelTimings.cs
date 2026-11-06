using System.Diagnostics;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Opt-in GPU time per piece of a composite operation (env GGUF_KERNEL_TIMING=1, desktop backends only: every mark synchronizes the device).
/// <see cref="Start"/> begins an operation, <see cref="Mark"/> charges the time since the previous mark to a name. For splitting an op's cost
/// on a machine without a profiler (2026-11-05: the training input gradient on the RTX 2060: dequant-transpose vs scale vs pack vs GEMM).
/// Off: both calls return immediately.
/// </summary>
public static class KernelTimings
{
    /// <summary>On when GGUF_KERNEL_TIMING=1 at startup; settable.</summary>
    public static bool Enabled { get; set; } = Environment.GetEnvironmentVariable("GGUF_KERNEL_TIMING") == "1";

    private static readonly Dictionary<string, (double Ms, long Count)> _totals = new();
    private static readonly Stopwatch _sw = new();

    /// <summary>Synchronize and start timing a composite operation.</summary>
    public static void Start(Accelerator acc)
    {
        if (!Enabled) return;
        acc.Synchronize();
        _sw.Restart();
    }

    /// <summary>Synchronize and charge the time since the previous mark (or <see cref="Start"/>) to <paramref name="name"/>.</summary>
    public static void Mark(Accelerator acc, string name)
    {
        if (!Enabled) return;
        acc.Synchronize();
        double ms = _sw.Elapsed.TotalMilliseconds;
        lock (_totals)
        {
            _totals.TryGetValue(name, out var t);
            _totals[name] = (t.Ms + ms, t.Count + 1);
        }
        _sw.Restart();
    }

    /// <summary>The totals so far, largest first.</summary>
    public static IReadOnlyList<(string Name, double Ms, long Count)> Snapshot()
    {
        lock (_totals) return _totals.Select(kv => (kv.Key, kv.Value.Ms, kv.Value.Count)).OrderByDescending(x => x.Ms).ToList();
    }

    /// <summary>Clear the totals.</summary>
    public static void Reset()
    {
        lock (_totals) _totals.Clear();
    }
}
