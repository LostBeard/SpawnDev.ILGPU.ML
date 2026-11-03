using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Tensors;

// ── Planned reuse: a run's recorded rent / return intervals -> shared slot buffers ──
//
// The size buckets keep each size's own peak: a free 64 MiB buffer cannot serve a 32 MiB rent, so a pool settles at the SUM of every size's peak, not
// the live working set. MEASURED 2026-11-03, ReDimNet2-B6 on CUDA: 682 MiB for 256 MiB peak live; finer size classes and best-fit fallbacks only moved
// that to 593-848 MiB (Plans/activation-arena-2026-11-03.md). A compiled executor rents and returns in the same order with the same sizes every run, so
// the first run is RECORDED (each fp32 rent's bucket and its [rent, return] interval on one clock), and every rent returned within that run is assigned a
// SLOT by greedy-by-size interval packing (largest first, into the first slot none of whose intervals overlap): a slot is one real buffer, the size of
// its largest rent, shared by rents whose lifetimes never overlap. Later runs hand rent k its slot (a sub-view of exactly its bucket length, as a bucket
// buffer is), so the pool holds about the packing's peak instead of every size's.
//
// Why slots and not offsets in one arena: every pool record is keyed by the raw buffer (_live, _namedBuffers, _rentable), and WebGPU forbids two
// read_write views of one buffer in a dispatch. A slot is held by at most one rent at a time, so all of that holds unchanged.
// Safety is the pool's own: a slot is handed out only after the recorded Return that freed it, exactly when the pool would have re-rented that buffer.
// A run that deviates (a different size at rent k, a slot still busy, more or fewer rents) falls back to the buckets for the rest of that run; three
// deviations drop the plan. Runs with capture flags (CUDA graph / capture param slots) never plan and drop any plan their executor had (its slots go back
// to the buckets), so capture's warm-pass priming is unchanged. A reclaim drops every plan first, so its slots are disposable like any free buffer.
public partial class BufferPool
{
    /// <summary>
    /// Plan each executor's activation buffers from its first run (see the notes in BufferPool.PlannedReuse.cs). Env <c>ML_PLANNED_REUSE</c> (1 = on).
    /// </summary>
    public static bool PlannedReuse = Environment.GetEnvironmentVariable("ML_PLANNED_REUSE") is "1" or "true";

    /// <summary>Rents served from a plan's slots (cumulative, diagnostic).</summary>
    public static long PlannedReuseRents;
    /// <summary>Planned runs that deviated from their recording (cumulative, diagnostic).</summary>
    public static long PlannedReuseDeviations;
    /// <summary>
    /// Device allocations made while BUILDING plans (cumulative): once per plan at most, when the pool never held enough free buffers of a slot's size
    /// at once. Counted in <see cref="TotalDeviceAllocations"/> too; a per-forward leak check subtracts these (they are not a forward's rents).
    /// </summary>
    public static long PlannedReuseAllocations;
    /// <summary>Plans built (cumulative, diagnostic).</summary>
    public static long PlannedReusePlans;

    private sealed class ReusePlan
    {
        public int[] Bucket = Array.Empty<int>();     // per rent: the bucket size recorded
        public int[] Slot = Array.Empty<int>();       // per rent: its slot, or -1 (not returned within the run: stays on the buckets)
        public SlotEntry[] Slots = Array.Empty<SlotEntry>();
        public int Deviations;
        public bool Dropped;
    }

    // A slot buffer, SHARED by every plan of this pool that fits in it: only one outermost run is active at a time, so between runs all slots are
    // free, and a session alternating two input shapes holds about the larger plan instead of the sum of both. Refs = plans using it; Busy = rented now.
    private sealed class SlotEntry
    {
        public MemoryBuffer1D<float, Stride1D.Dense> Buffer = null!;
        public bool Busy;
        public int Refs;
    }

    private readonly Dictionary<object, ReusePlan> _plans = new(ReferenceEqualityComparer.Instance);
    private readonly HashSet<object> _neverPlan = new(ReferenceEqualityComparer.Instance);
    // A recording becomes a plan only when the NEXT run's rent sequence equals it: an executor's first run is not its steady state (one-time rents:
    // shape readbacks, cached constants). MEASURED 2026-11-03, ReDimNet2-B6: a plan built from run 1 deviated at its 3rd rent on every later run.
    private readonly Dictionary<object, (int[] Buckets, int Attempts)> _pendingSequence = new(ReferenceEqualityComparer.Instance);
    // A steady recording waits here for its owner's NEXT run and is built when that run begins: at the end of a run its outputs are still live (the
    // caller returns them after), and a slot whose recorded buffer was re-rented as an output would need a fresh allocation in a warm forward
    // (ControlFlow_IfSqueeze_NoLeakAcrossForwards). By the next run they are back in the buckets.
    private readonly Dictionary<object, List<(int Bucket, int RentAt, int ReturnAt, MemoryBuffer1D<float, Stride1D.Dense> Buffer)>> _readyToPlan = new(ReferenceEqualityComparer.Instance);

    /// <summary>Log a planned run's first deviation (rent index, recorded vs requested bucket) to the console. Diagnostic.</summary>
    public static bool TracePlannedReuse = Environment.GetEnvironmentVariable("ML_TRACE_PLANNED_REUSE") is "1" or "true";
    private readonly Dictionary<MemoryBuffer, SlotEntry> _slotOf = new(ReferenceEqualityComparer.Instance);
    private int _planDepth;
    private object? _recOwner;
    private List<(int Bucket, int RentAt, int ReturnAt, MemoryBuffer1D<float, Stride1D.Dense> Buffer)>? _rec;
    private Dictionary<MemoryBuffer, int>? _recLive;
    private int _recClock;
    private ReusePlan? _replay;
    private int _replaySeq;
    private bool _replayOff;

    /// <summary>Bytes held by planned slots (busy or not).</summary>
    public long PlannedSlotBytes
    {
        get
        {
            lock (_sync)
            {
                long total = 0;
                foreach (var e in _slotOf.Values) total += e.Buffer.LengthInBytes;
                return total;
            }
        }
    }

    /// <summary>
    /// A graph run starts. Only the outermost run plans (If / Loop / Scan bodies run inside it and are part of its sequence).
    /// <paramref name="eligible"/> false (a capture-regime run) drops <paramref name="owner"/>'s plan and never plans it again.
    /// </summary>
    internal void BeginPlannedRun(object owner, bool eligible)
    {
        lock (_sync)
        {
            if (_planDepth++ > 0 || !PlannedReuse) return;
            if (!eligible) { DropPlanLocked(owner); _readyToPlan.Remove(owner); _neverPlan.Add(owner); return; }
            if (_neverPlan.Contains(owner)) return;
            if (_readyToPlan.Remove(owner, out var ready)) BuildPlanLocked(owner, ready);
            if (_plans.TryGetValue(owner, out var plan))
            {
                _replay = plan; _replaySeq = 0; _replayOff = false;
                return;
            }
            _recOwner = owner;
            _rec = new();
            _recLive = new(ReferenceEqualityComparer.Instance);
            _recClock = 0;
        }
    }

    /// <summary>The run started by the matching <see cref="BeginPlannedRun"/> ended: a recording becomes a plan; a replay is checked for deviation.</summary>
    internal void EndPlannedRun(object owner)
    {
        lock (_sync)
        {
            if (_planDepth == 0 || --_planDepth > 0) return;
            if (_replay != null)
            {
                var plan = _replay;
                _replay = null;
                if (_replayOff || _replaySeq != plan.Bucket.Length)
                {
                    if (TracePlannedReuse && !_replayOff)
                        Console.WriteLine($"[PlannedReuse] deviation: run made {_replaySeq} rents, plan has {plan.Bucket.Length}");
                    plan.Deviations++; PlannedReuseDeviations++;
                    if (plan.Deviations >= 3) { DropPlanLocked(owner); _neverPlan.Add(owner); }
                }
            }
            if (_rec != null)
            {
                var rec = _rec; var who = _recOwner!;
                _rec = null; _recLive = null; _recOwner = null;
                if (ReferenceEquals(who, owner))
                {
                    var seq = new int[rec.Count];
                    for (int i = 0; i < seq.Length; i++) seq[i] = rec[i].Bucket;
                    if (_pendingSequence.TryGetValue(owner, out var prev) && prev.Buckets.AsSpan().SequenceEqual(seq))
                    {
                        _pendingSequence.Remove(owner);
                        _readyToPlan[owner] = rec;
                    }
                    else if (prev.Attempts >= 4) { _pendingSequence.Remove(owner); _neverPlan.Add(owner); }   // never steady: leave it on the buckets
                    else _pendingSequence[owner] = (seq, prev.Attempts + 1);
                }
            }
        }
    }

    /// <summary>Replay: the planned slot for the next rent, if it has one and nothing has deviated.</summary>
    private bool TryRentPlanned(int bucketSize, out MemoryBuffer1D<float, Stride1D.Dense> buffer)
    {
        buffer = null!;
        var plan = _replay;
        if (plan == null || _replayOff) return false;
        int seq = _replaySeq++;
        if (seq >= plan.Bucket.Length || plan.Bucket[seq] != bucketSize || plan.Dropped)
        {
            if (TracePlannedReuse)
                Console.WriteLine($"[PlannedReuse] deviation at rent {seq} of {plan.Bucket.Length}: recorded {(seq < plan.Bucket.Length ? plan.Bucket[seq] : -1)}, " +
                    $"requested {bucketSize}, dropped={plan.Dropped} (node {Graph.GraphExecutor.CurrentRunNodeIndex} {Graph.GraphExecutor.CurrentRunOpType})");
            _replayOff = true;
            return false;
        }
        int k = plan.Slot[seq];
        if (k < 0) return false;
        var entry = plan.Slots[k];
        if (entry.Busy)
        {
            if (TracePlannedReuse) Console.WriteLine($"[PlannedReuse] deviation at rent {seq}: slot {k} still busy (node {Graph.GraphExecutor.CurrentRunNodeIndex})");
            _replayOff = true;
            return false;
        }
        entry.Busy = true;
        buffer = entry.Buffer;
        PlannedReuseRents++;
        return true;
    }

    /// <summary>Recording: note a rent (any path: bucket reuse or fresh allocation).</summary>
    private void NoteRecordedRent(int bucketSize, MemoryBuffer1D<float, Stride1D.Dense> buffer)
    {
        if (_rec == null) return;
        _recLive![buffer.Buffer] = _rec.Count;
        _rec.Add((bucketSize, _recClock++, -1, buffer));
    }

    /// <summary>
    /// A buffer is being freed (PushFree). A busy slot is just marked free (it never enters a bucket) - true. While recording, the rent it ends is
    /// stamped with its return time. Otherwise false: the caller pushes it into its bucket.
    /// </summary>
    private bool NotePlannedFree(MemoryBuffer1D<float, Stride1D.Dense> buffer)
    {
        if (_recLive != null && _recLive.Remove(buffer.Buffer, out int seq))
        {
            var r = _rec![seq];
            _rec[seq] = (r.Bucket, r.RentAt, _recClock++, r.Buffer);
        }
        if (!_slotOf.TryGetValue(buffer.Buffer, out var entry)) return false;
        entry.Busy = false;
        if (entry.Refs > 0) return true;
        _slotOf.Remove(buffer.Buffer);   // every plan using it was dropped while it was busy: it goes to the buckets like any buffer
        return false;
    }

    private void BuildPlanLocked(object owner, List<(int Bucket, int RentAt, int ReturnAt, MemoryBuffer1D<float, Stride1D.Dense> Buffer)> rec)
    {
        var planned = new List<int>();
        for (int i = 0; i < rec.Count; i++) if (rec[i].ReturnAt >= 0) planned.Add(i);
        if (planned.Count == 0) return;
        planned.Sort((a, b) => rec[a].Bucket != rec[b].Bucket ? rec[b].Bucket.CompareTo(rec[a].Bucket) : rec[a].RentAt.CompareTo(rec[b].RentAt));

        // Greedy by size: largest rent first, into the first slot whose intervals it does not overlap; a slot's size is its first (largest) rent's.
        var slotOf = new int[rec.Count];
        Array.Fill(slotOf, -1);
        var slotFirst = new List<int>();
        var slotIntervals = new List<List<(int A, int B)>>();
        foreach (int i in planned)
        {
            int a = rec[i].RentAt, b = rec[i].ReturnAt;
            int chosen = -1;
            for (int k = 0; k < slotIntervals.Count && chosen < 0; k++)
            {
                bool overlap = false;
                foreach (var (sa, sb) in slotIntervals[k]) if (a < sb && sa < b) { overlap = true; break; }
                if (!overlap) chosen = k;
            }
            if (chosen < 0) { chosen = slotIntervals.Count; slotIntervals.Add(new()); slotFirst.Add(i); }
            slotIntervals[chosen].Add((a, b));
            slotOf[i] = chosen;
        }

        // Rents that outlive the run (outputs, ownership handoffs) stay on the buckets and recur every forward. The recorded buffers of those that are
        // FREE now (the caller returned them) are what the next forward's outlived rents will take back: RESERVE them - never claimed, never disposed
        // (claiming them made the next forward's outputs allocate: ControlFlow_IfSqueeze_NoLeakAcrossForwards, 2026-11-03). One still live (an output
        // the executor recycles inside the next run) comes back by itself and needs no reserve.
        var reserved = new HashSet<MemoryBuffer>(ReferenceEqualityComparer.Instance);
        foreach (var r in rec)
            if (r.ReturnAt < 0 && !r.Buffer.IsDisposed && _buckets.TryGetValue((int)r.Buffer.Length, out var rs) && rs.Contains(r.Buffer))
                reserved.Add(r.Buffer.Buffer);

        // Each slot, largest first, takes: the smallest slot buffer another plan of this pool holds that fits (runs never overlap); else the smallest
        // unreserved free bucket buffer that fits; else (rare: the pool never held enough buffers of the size at once) a new one.
        var taken = new HashSet<MemoryBuffer>(ReferenceEqualityComparer.Instance);
        var slots = new SlotEntry[slotFirst.Count];
        for (int k = 0; k < slots.Length; k++)
        {
            int need = rec[slotFirst[k]].Bucket;
            SlotEntry? best = null;
            foreach (var e in _slotOf.Values)
                if (e.Refs > 0 && e.Buffer.Length >= need && !taken.Contains(e.Buffer.Buffer) && (best == null || e.Buffer.Length < best.Buffer.Length))
                    best = e;
            if (best == null)
            {
                MemoryBuffer1D<float, Stride1D.Dense>? pick = null;
                foreach (var kv in _buckets)
                {
                    if (kv.Key < need || (pick != null && kv.Key >= pick.Length)) continue;
                    foreach (var b in kv.Value) if (!reserved.Contains(b.Buffer)) { pick = b; break; }
                }
                if (pick != null)
                {
                    _buckets[(int)pick.Length].Remove(pick);
                    Interlocked.Add(ref _freeBucketedBytesCounter, -pick.LengthInBytes);
                }
                else
                {
                    TotalDeviceAllocations++; PlannedReuseAllocations++;
                    pick = _accelerator.Allocate1D<float>(need);
                    _allBuffers.Add(pick);
                    _rentable.Add(pick.Buffer);
                }
                best = new SlotEntry { Buffer = pick };
                _slotOf[pick.Buffer] = best;
            }
            best.Refs++;
            slots[k] = best;
            taken.Add(best.Buffer.Buffer);
        }

        // The recording's other free buffers (not slots, not reserved) are what the plan replaces: dispose them (after the queue, as a reclaim does).
        var surplus = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
        var seen = new HashSet<MemoryBuffer>(ReferenceEqualityComparer.Instance);
        foreach (var r in rec)
        {
            var buf = r.Buffer;
            if (!seen.Add(buf.Buffer) || buf.IsDisposed || _slotOf.ContainsKey(buf.Buffer) || reserved.Contains(buf.Buffer)) continue;
            if (_buckets.TryGetValue((int)buf.Length, out var stack) && stack.Remove(buf)) surplus.Add(buf);
        }
        if (surplus.Count > 0)
        {
            DrainBeforeFree();
            foreach (var buf in surplus)
            {
                Interlocked.Add(ref _freeBucketedBytesCounter, -buf.LengthInBytes);
                _allBuffers.Remove(buf); _rentable.Remove(buf.Buffer);
                buf.Dispose();
            }
        }

        var plan = new ReusePlan
        {
            Bucket = rec.Select(r => r.Bucket).ToArray(),
            Slot = slotOf,
            Slots = slots,
        };
        _plans[owner] = plan;
        PlannedReusePlans++;
    }

    /// <summary>An executor sharing this pool is being disposed: drop its plan (slots back to the buckets) and every record of it.</summary>
    internal void ForgetPlannedReuse(object owner)
    {
        lock (_sync)
        {
            if (IsDisposed) return;
            DropPlanLocked(owner);
            _neverPlan.Remove(owner);
            _pendingSequence.Remove(owner);
            _readyToPlan.Remove(owner);
            if (ReferenceEquals(_recOwner, owner)) { _rec = null; _recLive = null; _recOwner = null; }
        }
    }

    /// <summary>Drop <paramref name="owner"/>'s plan: its free slots go back to the buckets; a busy one follows when it is returned.</summary>
    private void DropPlanLocked(object owner)
    {
        if (!_plans.Remove(owner, out var plan)) return;
        ReleasePlanSlotsLocked(plan);
    }

    /// <summary>Drop every plan (a reclaim is about to dispose the free buffers: slots must be among them).</summary>
    private void DropAllPlansLocked()
    {
        if (_plans.Count == 0) return;
        foreach (var plan in _plans.Values) ReleasePlanSlotsLocked(plan);
        _plans.Clear();
        _replayOff = true;
    }

    /// <summary>A plan lets go of its slots; a slot no plan uses any more goes to the buckets (now, or when its current rent is returned).</summary>
    private void ReleasePlanSlotsLocked(ReusePlan plan)
    {
        plan.Dropped = true;
        foreach (var entry in plan.Slots)
        {
            if (--entry.Refs > 0 || entry.Busy) continue;
            _slotOf.Remove(entry.Buffer.Buffer);
            PushFree(entry.Buffer);
        }
    }
}
