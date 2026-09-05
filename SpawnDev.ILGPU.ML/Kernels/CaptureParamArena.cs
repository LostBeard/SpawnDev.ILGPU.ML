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
/// ONE arena PER CAPTURE (<see cref="BeginCaptureScope"/>; <see cref="Shared"/> falls back to a
/// per-accelerator arena outside a capture) → a single per-forward cursor across every operator, private to
/// the plan that stages it. 🔴 It used to be one arena per ACCELERATOR, and that silently corrupted every
/// earlier plan the moment a second capture's warm pass rented the same cursor - see
/// <see cref="BeginCaptureScope"/>. The forward executes nodes in a fixed order and each operator rents its params in a fixed order,
/// so for a given input shape the k-th rent is always the same (op, call) → the same stable slot every forward.
/// Kernels add a one-line capture branch; no per-kernel slot state. CUDA-only capability; harmless elsewhere.
/// </summary>
public sealed class CaptureParamArena : IDisposable
{
    private static readonly ConditionalWeakTable<Accelerator, CaptureParamArena> _perAccelerator = new();

    /// <summary>The arena the innermost open capture scope owns; null outside a capture. See
    /// <see cref="BeginCaptureScope"/>.</summary>
    private static CaptureParamArena? _active;

    /// <summary>
    /// The arena to rent from: the open capture scope's arena when there is one, else the per-accelerator
    /// fallback (one global cursor across all ops).
    /// </summary>
    public static CaptureParamArena Shared(Accelerator accelerator)
    {
        var active = _active;
        if (active != null && ReferenceEquals(active._accelerator, accelerator)) return active;
        return _perAccelerator.GetValue(accelerator, a => new CaptureParamArena(a));
    }

    /// <summary>
    /// Open a capture scope with its OWN arena, so a later capture cannot write over the slots a recorded
    /// plan binds. Returns the arena; the CALLER OWNS IT and must dispose it when - and not before - the
    /// plan it staged is disposed. Pair with <see cref="EndCaptureScope"/> in a <c>finally</c>.
    /// </summary>
    /// <remarks>
    /// 🔴 A SLOT A RECORDED PLAN BINDS MUST NEVER BE WRITTEN AGAIN. A captured plan (WebGPU bind groups,
    /// a CUDA graph's baked device pointers) reads these buffers at REPLAY time, long after the capture
    /// that staged them. While one arena was shared per ACCELERATOR, the next capture's warm passes rented
    /// the same cursors and <c>CopyFromCPU</c>'d DIFFERENT params straight into buffers that every earlier
    /// plan still reads - silently, with no error anywhere, because overwriting a live buffer is a
    /// perfectly legal thing to do. The earlier plan then replays with the later capture's shapes/strides.
    /// <para>
    /// ⚠️ THIS IS NOT THE CAUSE OF THE 2026-09-04 HISTORY-DEPENDENT AUDIO, though it looks exactly like
    /// it and I fixed it believing it was. MEASURED 2026-09-05 on the SpawnDev.AI seven-line voice gate:
    /// the fix changed nothing (100/100/100/67/48/73/48, identical to the run before it), and the browser
    /// console says why - ZipVoice's decoder capture is REFUSED on every synthesis
    /// ("graph contains control flow (If) ... running direct forward"), so a synthesis performs no capture,
    /// no replay and no arena rent at all. The only live capture in that process is Whisper's encoder, one
    /// plan, 176 dispatches, captured once. Fix a defect because it is a defect; do not let it inherit
    /// another bug's evidence.
    /// </para>
    /// <para>
    /// What it IS: the measured 2026-09-04 "[Buffer ...] used in submit while destroyed" failure, where
    /// Whisper's encoder plan and a ZipVoice forward shared one accelerator's arena.
    /// </para>
    /// <para>
    /// ⚠️ This is the same defect class as the <see cref="_retired"/> note below, which fixed only the
    /// half where a GROWN slot's old buffer was DISPOSED. Retiring stopped the buffer being destroyed; it
    /// did nothing about the far more common case where the slot is big enough and is simply overwritten.
    /// </para>
    /// </remarks>
    public static CaptureParamArena BeginCaptureScope(Accelerator accelerator)
    {
        var arena = new CaptureParamArena(accelerator) { _previous = _lastScoped };
        _lastScoped = new WeakReference<CaptureParamArena>(arena);
        _active = arena;
        ScopeId++;
        return arena;
    }

    /// <summary>
    /// Monotonic id of the most recently opened capture scope. Other holders of stable capture slots
    /// (<c>FusedAttentionKernel</c>) watch this to take a fresh slot set per capture, for the same reason
    /// this arena is now per-capture.
    /// </summary>
    public static long ScopeId { get; private set; }

    /// <summary>Close the scope opened by <see cref="BeginCaptureScope"/>. Does NOT dispose the arena.</summary>
    public static void EndCaptureScope() => _active = null;

    /// <summary>The previous capture scope's arena, weakly - only so the diagnostic below can count.</summary>
    private static WeakReference<CaptureParamArena>? _lastScoped;
    private WeakReference<CaptureParamArena>? _previous;

    /// <summary>
    /// How many slot writes this process has made that WOULD have landed in a buffer the previous
    /// capture's plan still binds, had the arena still been shared per-accelerator. Every one of these is
    /// a silently corrupted replay under the old scheme; zero means the sequence never re-captured.
    /// </summary>
    /// <remarks>Counted, not prevented - the per-scope arena already prevents it. This exists so the fix
    /// can be shown to have had something to fix, on the exact sequence that reproduced the defect.</remarks>
    public static int CrossCaptureSlotOverwrites { get; private set; }

    /// <summary>Reset <see cref="CrossCaptureSlotOverwrites"/> (call before a measured sequence).</summary>
    public static void ResetCrossCaptureTrace() => CrossCaptureSlotOverwrites = 0;

    // Last host data written to each slot, kept only to answer the question above.
    private readonly List<int[]?> _slotData = new();
    private readonly List<float[]?> _floatSlotData = new();

    private void NoteIntWrite(int slot, int[] data)
    {
        while (_slotData.Count <= slot) _slotData.Add(null);
        if (_previous != null && _previous.TryGetTarget(out var prev)
            && slot < prev._slotData.Count && prev._slotData[slot] is { } pd
            && slot < prev._slots.Count && prev._slots[slot].Length >= data.Length
            && !pd.AsSpan().SequenceEqual(data))
            CrossCaptureSlotOverwrites++;
        _slotData[slot] = (int[])data.Clone();
    }

    private void NoteFloatWrite(int slot, float[] data)
    {
        while (_floatSlotData.Count <= slot) _floatSlotData.Add(null);
        if (_previous != null && _previous.TryGetTarget(out var prev)
            && slot < prev._floatSlotData.Count && prev._floatSlotData[slot] is { } pd
            && slot < prev._floatSlots.Count && prev._floatSlots[slot].Length >= data.Length
            && !pd.AsSpan().SequenceEqual(data))
            CrossCaptureSlotOverwrites++;
        _floatSlotData[slot] = (float[])data.Clone();
    }

    /// <summary>
    /// Capture-safe write of a CONSTANT CPU-computed result into a GPU node-output view. Ops that write a
    /// small constant to their output via <c>CopyFromCPU</c> (Cast/Expand/BroadcastBinaryOp/Slice CPU paths)
    /// MUST use this in capture-mode: a plain CopyFromCPU is a synchronous H2D (illegal mid-capture), and
    /// SKIPPING it is WRONG — the pooled output buffer is reused between the warm and capture passes, so the
    /// captured graph would contain no write for this node and REPLAY would read stale data. Instead we stage
    /// the constant in a STABLE arena slot (written on the warm pass, held thereafter) and, on the capture
    /// pass, issue a GPU→GPU <c>CopyFrom</c> — which IS capturable — so the write lands in the graph and
    /// replay reproduces it. Call in BOTH warm and capture (the rent keeps the float cursor deterministic).
    /// </summary>
    public static void CaptureConstWrite(Accelerator accelerator,
        ArrayView1D<float, Stride1D.Dense> output, float[] result)
    {
        var slot = Shared(accelerator).RentStableSlotFloat(result);   // warm: writes slot; capture: warm value
        if (GraphExecutor.SuppressDrains)
            output.CopyFrom(slot);            // capture pass: captured GPU→GPU copy (in the graph → replay-safe)
        else
            output.CopyFromCPU(result);       // warm pass: direct host upload
    }

    private readonly Accelerator _accelerator;

    // Distinct stable slot per per-forward call ordinal. Grows on demand; each slot is allocated once and
    // reused across forwards (fixed shapes → the k-th rent is the same op with the same length every time).
    private readonly List<MemoryBuffer1D<int, Stride1D.Dense>> _slots = new();
    private int _next;
    private long _gen = -1;

    // Float-typed param slots (e.g. BroadcastBinaryOpND packs strides as float) — an INDEPENDENT cursor, since
    // the int-rent and float-rent sequences are each deterministic per forward but interleave arbitrarily.
    private readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _floatSlots = new();
    private int _floatNext;
    private long _floatGen = -1;

    public CaptureParamArena(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Patch-point discovery observers (parameterized replay - WebGPUDecodeCapture): fired on every
    /// stable-slot rent that CARRIES data (warm passes; not during SuppressDrains, where data is
    /// skipped) with (slotIndex, data, slotView). A driver records two probe forwards at two values
    /// of the loop variable and diffs the data per slot to find variable-dependent params. Static -
    /// set only around probe forwards, null in production.
    /// </summary>
    public static Action<int, int[], ArrayView1D<int, Stride1D.Dense>>? IntSlotObserver;
    /// <summary>Float analogue of <see cref="IntSlotObserver"/>.</summary>
    public static Action<int, float[], ArrayView1D<float, Stride1D.Dense>>? FloatSlotObserver;

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
            if (GraphExecutor.SuppressDrains && GraphExecutor.CaptureTraceFile != null)
            { try { System.IO.File.AppendAllText(GraphExecutor.CaptureTraceFile, $"   -> ARENA-ALLOC int slot={i} len={data.Length}  (capture sizing gap)\n"); } catch { } }
            // First time this cursor is used, or a longer params array than the slot was sized for.
            slot = _accelerator.Allocate1D<int>(data.Length);
            // RETIRE, DO NOT DISPOSE - see _retired.
            if (i < _slots.Count) { Retire(_slots[i]); _slots[i] = slot; }
            else _slots.Add(slot);
        }

        var view = slot.View.SubView(0, data.Length);
        // Skip the synchronizing H2D during the capture pass; the warm pass already populated this slot.
        if (!GraphExecutor.SuppressDrains)
        {
            NoteIntWrite(i, data);
            view.CopyFromCPU(data);
            IntSlotObserver?.Invoke(i, data, view);
        }
        return view;
    }

    /// <summary>Float-typed analogue of <see cref="RentStableSlot"/> (independent cursor).</summary>
    public ArrayView1D<float, Stride1D.Dense> RentStableSlotFloat(float[] data)
    {
        long gen = GraphExecutor.ForwardGeneration;
        if (gen != _floatGen) { _floatGen = gen; _floatNext = 0; }
        int i = _floatNext++;

        MemoryBuffer1D<float, Stride1D.Dense> slot;
        if (i < _floatSlots.Count && _floatSlots[i].Length >= data.Length)
        {
            slot = _floatSlots[i];
        }
        else
        {
            if (GraphExecutor.SuppressDrains && GraphExecutor.CaptureTraceFile != null)
            { try { System.IO.File.AppendAllText(GraphExecutor.CaptureTraceFile, $"   -> ARENA-ALLOC float slot={i} len={data.Length}  (capture sizing gap)\n"); } catch { } }
            slot = _accelerator.Allocate1D<float>(data.Length);
            // RETIRE, DO NOT DISPOSE - see _retired.
            if (i < _floatSlots.Count) { Retire(_floatSlots[i]); _floatSlots[i] = slot; }
            else _floatSlots.Add(slot);
        }

        var view = slot.View.SubView(0, data.Length);
        if (!GraphExecutor.SuppressDrains)
        {
            NoteFloatWrite(i, data);
            view.CopyFromCPU(data);
            FloatSlotObserver?.Invoke(i, data, view);
        }
        return view;
    }

    /// <summary>
    /// Slots that have been GROWN out of use but must stay alive until this arena is disposed.
    /// </summary>
    /// <remarks>
    /// 🔴 A GROWN SLOT'S OLD BUFFER IS STILL INSIDE SOMEONE'S RECORDED PLAN. This arena is a per-
    /// accelerator SINGLETON (<see cref="Shared"/>), shared by every pipeline on that device, and a
    /// captured WebGPU dispatch plan holds raw <c>GPUBuffer</c> references in its bind groups. Disposing
    /// the old buffer when a later, longer params array grows the slot therefore destroys memory a
    /// previously recorded plan still binds - and WebGPU reports that only at the NEXT submit, as
    /// "[Buffer ...] used in submit while destroyed", from whichever innocent caller happens to
    /// synchronize next.
    /// <para>
    /// MEASURED 2026-09-04 in the SpawnDev.AI demo: Whisper's encoder capture recorded a plan against
    /// float slot i (669 floats, "Storage#5888:2676B"); a ZipVoice synthesis then rented the same cursor
    /// with a longer array, the old buffer was disposed here, and the next transcription's replay failed
    /// with that error - which is why turning Whisper's capture off appeared to "fix" it. The stack that
    /// finally named this came from <c>WebGPUBackend.TraceBufferDestroy</c>.
    /// </para>
    /// Retiring costs a handful of small buffers per process; freeing one early costs correctness.
    /// </remarks>
    private readonly List<IDisposable> _retired = new();

    private void Retire(IDisposable slot) => _retired.Add(slot);

    public void Dispose()
    {
        foreach (var s in _slots) s.Dispose();
        _slots.Clear();
        foreach (var s in _floatSlots) s.Dispose();
        _floatSlots.Clear();
        foreach (var s in _retired) { try { s.Dispose(); } catch { } }
        _retired.Clear();
    }
}
