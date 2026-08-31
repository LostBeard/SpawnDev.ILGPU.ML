using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// One device buffer per DISTINCT set of packed kernel parameters, reused for the life of the kernel.
/// </summary>
/// <remarks>
/// <para>
/// Kernels here pack their scalars into a small device array to stay under the launch parameter limit, and
/// the pattern they all grew was "allocate a FRESH buffer per call, retire the previous one for deferred
/// disposal". That is wrong in three ways at once:
/// </para>
/// <list type="number">
/// <item>
/// 🔴 <b>It makes CUDA graph capture impossible.</b> <c>Allocate1D</c> is a <c>cuMemAlloc</c>, which is
/// illegal inside a capture window and faults with an ACCESS VIOLATION (0xC0000005) that no try/catch can
/// observe - the process simply dies, and the managed stack shows only async plumbing.
/// MEASURED via <c>GraphExecutor.CaptureTraceFile</c> on Silero VAD: the capture pass died at node 13,
/// <c>Conv /feature_extractor/Conv_output_0</c>, with <c>cuMemAlloc_v2</c> at the top of the native stack.
/// </item>
/// <item>The retired buffers accumulate until the kernel is disposed - a session-lifetime growth.</item>
/// <item>Every call pays an allocation plus a host-to-device copy for values that are a pure function of
/// the shapes and therefore constant.</item>
/// </list>
/// <para>
/// ⚠️ A SINGLE "last values" slot is NOT sufficient, and fails in a way worth recording because it looks
/// correct: one kernel instance serves EVERY node of its type in the graph, and their shapes differ. Silero
/// VAD has eighteen Conv1D nodes, so a warm pass ends with the last node's params resident and the next
/// pass misses on its FIRST node - which under capture is exactly the fatal allocation. Per-set caching
/// means two warm passes make every set resident and the capture pass allocates NOTHING.
/// </para>
/// <para>
/// ⚠️ Why reuse is safe where the fresh-per-call pattern was defensive: rewriting a params buffer that a
/// PENDING dispatch still reads would corrupt it (the command-batching hazard on the browser backends).
/// This performs NO WRITE on a hit - the values are already the ones resident - so there is nothing to
/// corrupt. A miss allocates a new buffer and never touches the old one.
/// </para>
/// <para>
/// A linear scan over a handful of short arrays beats hashing here and allocates nothing on the hot path.
/// Kernels see a few distinct sets at most; if one ever saw thousands, the scan is still cheaper than the
/// allocation it replaces.
/// </para>
/// </remarks>
public sealed class ParamBufferCache<T> : IDisposable where T : unmanaged, IEquatable<T>
{
    private readonly List<(T[] Values, MemoryBuffer1D<T, Stride1D.Dense> Buffer)> _entries = new();

    /// <summary>Distinct parameter sets currently resident. Diagnostic - a number that keeps growing is a bug.</summary>
    public int Count => _entries.Count;

    /// <summary>
    /// The device buffer holding <paramref name="values"/>, allocating only if this exact set is new.
    /// </summary>
    /// <remarks>
    /// ⚠️ Takes ownership of <paramref name="values"/> on a miss (it is retained as the cache key), so the
    /// caller must not mutate the array afterwards. Every current caller builds a fresh array per call.
    /// </remarks>
    public ArrayView1D<T, Stride1D.Dense> Get(Accelerator accelerator, T[] values)
    {
        foreach (var (cachedValues, buffer) in _entries)
            if (values.AsSpan().SequenceEqual(cachedValues)) return buffer.View;

        var fresh = accelerator.Allocate1D(values);
        _entries.Add((values, fresh));
        return fresh.View;
    }

    public void Dispose()
    {
        foreach (var (_, b) in _entries) { try { b.Dispose(); } catch { } }
        _entries.Clear();
    }
}
