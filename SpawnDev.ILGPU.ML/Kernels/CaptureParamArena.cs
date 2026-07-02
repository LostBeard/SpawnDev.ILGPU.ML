using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Per-kernel stable param-buffer arena for CUDA-graph capture.
///
/// Many operators upload a tiny <c>int[]</c> params array (shapes/strides/axes) to the GPU per dispatch.
/// The production path allocates a FRESH buffer every call (deferred-dispose so an un-submitted WebGPU/Wasm
/// batch never reads a recycled buffer). That is correct for normal inference but FATAL during CUDA-graph
/// capture: a per-call <c>cuMemAlloc</c> is illegal mid-capture (native access violation), and even if it
/// weren't, the captured kernel node would bake a device pointer that moves every forward, so replay would
/// read a stale slot.
///
/// This arena solves both. During capture (<see cref="GraphExecutor.UseCaptureParamSlots"/>), the k-th
/// <see cref="RentStableSlot"/> call of every forward returns the SAME device buffer (the per-forward
/// counter resets whenever <see cref="GraphExecutor.ForwardGeneration"/> ticks). The warm pass populates
/// each slot with the fixed-shape params; the capture pass then SKIPS the H2D upload while
/// <see cref="GraphExecutor.SuppressDrains"/> is set (a <c>CopyFromCPU</c> synchronizes = illegal during
/// capture). The slot already holds the immediately-preceding warm pass's identical params.
///
/// It also removes the per-op alloc + H2D re-upload from the warm forward itself, so it is a real
/// launch-prep win even independent of capture. When capture is NOT in use the arena is untouched, so the
/// kernels' normal fresh-alloc + defer-dispose path (and thus the WebGPU/Wasm backends) is byte-identical.
///
/// ONE shared arena per accelerator (<see cref="Shared"/>) → a single global per-forward cursor across every
/// operator. The forward executes nodes in a fixed order and each operator rents its params in a fixed order,
/// so for a given input shape the k-th rent is always the same (op, call) → the same stable slot every forward.
/// Kernels add a one-line capture branch; no per-kernel slot state. CUDA-only capability; harmless elsewhere.
/// </summary>
public sealed class CaptureParamArena : IDisposable
{
    private static readonly ConditionalWeakTable<Accelerator, CaptureParamArena> _perAccelerator = new();

    /// <summary>The shared arena for <paramref name="accelerator"/> (one global capture cursor across all ops).</summary>
    public static CaptureParamArena Shared(Accelerator accelerator)
        => _perAccelerator.GetValue(accelerator, a => new CaptureParamArena(a));

    private readonly Accelerator _accelerator;

    // Distinct stable slot per per-forward call ordinal. Grows on demand; each slot is allocated once and
    // reused across forwards (fixed shapes → the k-th rent is the same op with the same length every time).
    private readonly List<MemoryBuffer1D<int, Stride1D.Dense>> _slots = new();
    private int _next;
    private long _gen = -1;

    public CaptureParamArena(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Returns a stable device view of <paramref name="data"/> for the current forward's next param rent.
    /// Only valid to call while <see cref="GraphExecutor.UseCaptureParamSlots"/> is set (the kernel keeps
    /// its normal path otherwise). The H2D upload is skipped while <see cref="GraphExecutor.SuppressDrains"/>
    /// is set — the warm pass already wrote the (identical, fixed-shape) params into this slot.
    /// </summary>
    public ArrayView1D<int, Stride1D.Dense> RentStableSlot(int[] data)
    {
        long gen = GraphExecutor.ForwardGeneration;
        if (gen != _gen) { _gen = gen; _next = 0; }   // new forward → restart the per-forward cursor
        int i = _next++;

        MemoryBuffer1D<int, Stride1D.Dense> slot;
        if (i < _slots.Count && _slots[i].Length >= data.Length)
        {
            slot = _slots[i];
        }
        else
        {
            // First time this cursor is used, or a longer params array than the slot was sized for.
            slot = _accelerator.Allocate1D<int>(data.Length);
            if (i < _slots.Count) { _slots[i].Dispose(); _slots[i] = slot; }
            else _slots.Add(slot);
        }

        var view = slot.View.SubView(0, data.Length);
        // Skip the synchronizing H2D during the capture pass; the warm pass already populated this slot.
        if (!GraphExecutor.SuppressDrains)
            view.CopyFromCPU(data);
        return view;
    }

    public void Dispose()
    {
        foreach (var s in _slots) s.Dispose();
        _slots.Clear();
    }
}
