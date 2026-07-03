using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Backend;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// PARAMETERIZED WebGPU dispatch-plan replay of the GGUF DECODE STEP - the browser answer to the
/// measured 60x decode gap (686ms/tok of which ~570ms is per-node dispatch orchestration; the
/// same-state replay probe proved the GPU floor is ~21ms/tok, bit-exact).
///
/// The decode graph at fixed shape [1,1] is IDENTICAL per token except for a handful of values that
/// depend on the KV cursor (pastLen): attention SKV/kvOffset (stable param slots), RoPE
/// startPosition (packed _scalar_params bytes), materialized shape values (CaptureParamArena slots),
/// and the KV-cache append destinations (plan copy entries). Rather than hard-coding where those
/// live per architecture, this driver DISCOVERS them: it captures TWO plans at pastLen P0 and P0+1
/// (identical structure - only values differ), plus two observer probes of every stable-slot write,
/// and DIFFS them. Every difference is fitted as value(P) = valueAtP0 + slope*(P - P0) (decode
/// state advances by exactly 1 token, so every cursor-dependent value is affine in pastLen; a
/// non-int or structurally-mismatched diff throws - fail loud, never silently wrong).
///
/// Per token, <see cref="PatchAndReplayAsync"/> then: writes the token id into the stable input
/// buffer, patches the discovered scalar bytes / slot arrays / copy offsets for the current
/// pastLen, replays plan A with ONE interop crossing, and returns the stable logits tensor.
/// Correctness gate: token-identical greedy decode vs the uncaptured path
/// (GGUF_WebGPU_DecodeCapture_TokenIdentical).
/// </summary>
public sealed class WebGPUDecodeCapture : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly InferenceSession _session;
    private readonly WebGPUDispatchPlan _plan;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _inputBuf;
    private readonly Dictionary<string, Tensor> _inputDict;
    private readonly Dictionary<string, Tensor> _outputs;
    private readonly int _p0;

    // Discovered patch points (affine in pastLen, anchored at P0). Slot + input patches write via
    // RAW queue.WriteBuffer on the resolved GPUBuffer (measured ~0.02ms/write) - the ILGPU
    // ArrayView.CopyFromCPU chain costs ~14ms/call on WebGPU (flush + marshal + wrapper churn;
    // library root-cause tracked separately) and made the first patched replay 456ms/tok.
    private readonly List<(int SnapIdx, int ByteOfs, int ValAtP0, int Slope)> _scalarPatches;
    private readonly List<(int EntryIdx, double DstAtP0, double Slope)> _copyPatches;
    private sealed record SlotPatch(SpawnDev.BlazorJS.JSObjects.GPUBuffer Buf, long ByteOfs,
        System.Array Template, int[] Indices, long[] Slopes, bool IsFloat, byte[] Scratch);
    private readonly List<SlotPatch> _slotPatches;
    private readonly int[] _copyEntryIdx;
    private readonly double[] _copyOfsScratch;
    private readonly SpawnDev.BlazorJS.JSObjects.GPUQueue _queue;
    private readonly SpawnDev.BlazorJS.JSObjects.GPUBuffer _inputRawBuf;
    private readonly long _inputRawOfs;

    /// <summary>Raw (GPUBuffer, byteOffset) of a contiguous WebGPU view - the fast-write handle.</summary>
    private static (SpawnDev.BlazorJS.JSObjects.GPUBuffer Buf, long ByteOfs) RawOf(IContiguousArrayView view)
    {
        var mb = view.Buffer as WebGPUMemoryBuffer
            ?? throw new InvalidOperationException("view is not backed by a WebGPU buffer");
        return (mb.NativeBuffer.NativeBuffer!, view.IndexInBytes);
    }

    /// <summary>Ops in the captured plan (diagnostics).</summary>
    public int DispatchCount => _plan.DispatchCount;
    /// <summary>Discovered patch-point counts (diagnostics): scalar ints / copy offsets / slot arrays.</summary>
    public (int Scalars, int Copies, int Slots) PatchCounts => (_scalarPatches.Count, _copyPatches.Count, _slotPatches.Count);

    private WebGPUDecodeCapture(Accelerator acc, InferenceSession session, WebGPUDispatchPlan plan,
        MemoryBuffer1D<float, Stride1D.Dense> inputBuf, Dictionary<string, Tensor> inputDict,
        Dictionary<string, Tensor> outputs, int p0,
        List<(int, int, int, int)> scalarPatches, List<(int, double, double)> copyPatches, List<SlotPatch> slotPatches)
    {
        _accelerator = acc; _session = session; _plan = plan; _inputBuf = inputBuf;
        _inputDict = inputDict; _outputs = outputs; _p0 = p0;
        _scalarPatches = scalarPatches; _copyPatches = copyPatches; _slotPatches = slotPatches;
        _copyEntryIdx = copyPatches.Select(c => c.Item1).ToArray();
        _copyOfsScratch = new double[copyPatches.Count];
        _queue = ((WebGPUAccelerator)acc).NativeAccelerator.Queue
            ?? throw new InvalidOperationException("WebGPU queue unavailable");
        (_inputRawBuf, _inputRawOfs) = RawOf((IContiguousArrayView)inputBuf.View.BaseView);
    }

    // One observer probe forward: records every stable-slot write (attention + arena int/float).
    private sealed class SlotProbe
    {
        public readonly List<(int Slot, int[] Data, ArrayView1D<int, Stride1D.Dense> View)> Attn = new();
        public readonly List<(int Slot, int[] Data, ArrayView1D<int, Stride1D.Dense> View)> ArenaInt = new();
        public readonly List<(int Slot, float[] Data, ArrayView1D<float, Stride1D.Dense> View)> ArenaFloat = new();
        public void Attach()
        {
            FusedAttentionKernel.CaptureSlotObserver = (s, d, v) => Attn.Add((s, (int[])d.Clone(), v));
            CaptureParamArena.IntSlotObserver = (s, d, v) => ArenaInt.Add((s, (int[])d.Clone(), v));
            CaptureParamArena.FloatSlotObserver = (s, d, v) => ArenaFloat.Add((s, (float[])d.Clone(), v));
        }
        public static void Detach()
        {
            FusedAttentionKernel.CaptureSlotObserver = null;
            CaptureParamArena.IntSlotObserver = null;
            CaptureParamArena.FloatSlotObserver = null;
        }
    }

    /// <summary>
    /// Capture the decode step at the session's CURRENT cursor (pastLen = P0) with
    /// <paramref name="tokenId"/> as the step's input, then probe pastLen = P0+1 with the same token
    /// to discover the cursor-dependent patch points. On return the session cursor is at P0+1 and
    /// the CAPTURE PASS's logits (the step's real output) are in <see cref="Outputs"/> - the caller
    /// treats the capture exactly like a normal decode step and replays from the NEXT token on.
    /// NOTE: the P0+1 probe writes throwaway K/V into cache row P0+1; the first
    /// <see cref="PatchAndReplayAsync"/> at pastLen P0+1 rewrites that row with the real token
    /// before attention reads it (the decode step always writes its own row first), so no stale
    /// data survives. Returns null on non-WebGPU accelerators.
    /// </summary>
    private GpuArgMax? _argmax;
    private int _vocab;

    /// <summary>
    /// Capture with the greedy ARGMAX folded into the plan: the partial-argmax kernel over the
    /// step's last-position logits is recorded as the plan's final dispatch, so
    /// <see cref="PatchAndDecodeGreedyAsync"/> needs exactly ONE GPU round-trip per token (the
    /// partials readback's own mapAsync fence - no separate SynchronizeAsync, no separate argmax
    /// dispatch). <paramref name="argmax"/> is the caller's (generator's) instance - shared so the
    /// capture pass and non-captured steps reduce identically (lowest-index tie-break preserved).
    /// </summary>
    public static async Task<WebGPUDecodeCapture?> TryCaptureAsync(InferenceSession session, float tokenId, GpuArgMax? argmax = null)
    {
        var acc = session.Accelerator;
        if (acc is not WebGPUAccelerator webGpu) return null;

        int p0 = session.DecodePastLen;
        session.CacheShapeReadbacks = true;

        bool prevFold = GraphCompiler.ShapeSubgraphFoldEnabled;
        bool prevElide = GraphExecutor.ShapeInterpElideDispatch;
        bool prevValidate = GraphExecutor.ShapeInterpValidate;
        bool prevBgCache = WebGPUBackend.EnableBindGroupCaching;
        GraphCompiler.ShapeSubgraphFoldEnabled = true;
        GraphExecutor.ShapeInterpElideDispatch = true;
        GraphExecutor.ShapeInterpValidate = false;
        WebGPUBackend.EnableBindGroupCaching = false;
        FusedAttentionKernel.UseStableCaptureSlots = true;
        GraphExecutor.UseCaptureParamSlots = true;

        var inputBuf = acc.Allocate1D(new[] { tokenId });
        var inputDict = new Dictionary<string, Tensor> { ["input_ids"] = new Tensor(inputBuf.View, new[] { 1, 1 }, "input_ids") };
        try
        {
            async Task<Dictionary<string, Tensor>> StepAt(int p, bool suppress)
            {
                session.SetGGUFDecodePastLen(p);
                if (suppress) GraphExecutor.SuppressDrains = true;
                try { return await session.RunDecodeStepAsync(inputDict); }
                finally { GraphExecutor.SuppressDrains = false; }
            }

            // ── Plan A at P0: warm A (observer probe) + warm B (suppressed) + capture (snapshots on) ──
            var probeA = new SlotProbe(); probeA.Attach();
            await StepAt(p0, false); await acc.SynchronizeAsync();
            SlotProbe.Detach();
            await StepAt(p0, true); await acc.SynchronizeAsync();

            // Last-position logits view of a step's outputs + the argmax fold: recorded INTO the
            // plan as its final dispatch (constant params - the logits tensor is stable, vocab
            // fixed), so replays carry their own argmax. Dispatched in BOTH captures (B's discarded)
            // to keep the A/B structural parity check exact.
            (ArrayView1D<float, Stride1D.Dense> View, int Vocab) LastLogits(Dictionary<string, Tensor> o)
            {
                var t = o.TryGetValue("logits", out var l) ? l : o.Values.First();
                int vocab = t.Shape[^1];
                int so = t.ElementCount / vocab;
                return (t.Data.SubView((long)(so - 1) * vocab, vocab), vocab);
            }

            Dictionary<string, Tensor> capOut;
            var planA = webGpu.BeginDispatchCapture();
            planA.CaptureScalarSnapshots = true;
            try
            {
                capOut = await StepAt(p0, true);
                if (argmax != null) { var (lv, vc) = LastLogits(capOut); argmax.DispatchPartials(lv, vc); }
            }
            catch { webGpu.EndDispatchCapture().Dispose(); throw; }
            webGpu.EndDispatchCapture();
            await acc.SynchronizeAsync();

            // ── Plan B at P0+1 (value probe only - discarded after the diff) ──
            var probeB = new SlotProbe(); probeB.Attach();
            await StepAt(p0 + 1, false); await acc.SynchronizeAsync();
            SlotProbe.Detach();
            await StepAt(p0 + 1, true); await acc.SynchronizeAsync();

            var planB = webGpu.BeginDispatchCapture();
            planB.CaptureScalarSnapshots = true;
            try
            {
                var outB = await StepAt(p0 + 1, true);
                if (argmax != null) { var (lv, vc) = LastLogits(outB); argmax.DispatchPartials(lv, vc); }
            }
            catch { webGpu.EndDispatchCapture().Dispose(); throw; }
            webGpu.EndDispatchCapture();
            await acc.SynchronizeAsync();

            List<(int, int, int, int)> scalarPatches;
            List<(int, double, double)> copyPatches;
            List<SlotPatch> slotPatches;
            using (planB)
            {
                // Structural parity - a mismatch means the two captures took different graph paths
                // and NO diff is trustworthy. Fail loud.
                if (planA.DispatchCount != planB.DispatchCount
                    || planA.ScalarSnapshots.Count != planB.ScalarSnapshots.Count
                    || planA.CopyEntries.Count != planB.CopyEntries.Count)
                    throw new InvalidOperationException(
                        $"decode capture parity mismatch: ops {planA.DispatchCount}/{planB.DispatchCount}, " +
                        $"scalars {planA.ScalarSnapshots.Count}/{planB.ScalarSnapshots.Count}, " +
                        $"copies {planA.CopyEntries.Count}/{planB.CopyEntries.Count}");

                // Scalar bytes: diff as 4-byte ints (all cursor-dependent scalars are ints - a
                // non-int-aligned diff throws).
                scalarPatches = new();
                for (int s = 0; s < planA.ScalarSnapshots.Count; s++)
                {
                    var (entryA, _, a) = planA.ScalarSnapshots[s];
                    var (entryB, _, b) = planB.ScalarSnapshots[s];
                    if (entryA != entryB || a.Length != b.Length)
                        throw new InvalidOperationException($"scalar snapshot {s} parity mismatch (entry {entryA}/{entryB}, len {a.Length}/{b.Length})");
                    for (int i = 0; i + 4 <= a.Length; i += 4)
                    {
                        int va = BitConverter.ToInt32(a, i), vb = BitConverter.ToInt32(b, i);
                        if (va != vb) scalarPatches.Add((s, i, va, vb - va));
                    }
                    // Guard the 4-byte-int assumption: any residual byte-level diff outside aligned
                    // int words would have been caught above only if int-visible; verify totals.
                    for (int i = a.Length & ~3; i < a.Length; i++)
                        if (a[i] != b[i]) throw new InvalidOperationException($"scalar snapshot {s}: non-int-aligned diff at byte {i}");
                }

                // Copy destinations (KV appends): affine dstOffset.
                copyPatches = new();
                for (int c = 0; c < planA.CopyEntries.Count; c++)
                {
                    var ea = planA.CopyEntries[c]; var eb = planB.CopyEntries[c];
                    if (ea.EntryIndex != eb.EntryIndex || ea.Size != eb.Size)
                        throw new InvalidOperationException($"copy entry {c} parity mismatch");
                    if (ea.DstOffset != eb.DstOffset)
                        copyPatches.Add((ea.EntryIndex, ea.DstOffset, (double)eb.DstOffset - ea.DstOffset));
                    if (ea.SrcOffset != eb.SrcOffset)
                        throw new InvalidOperationException($"copy entry {c}: srcOffset is cursor-dependent ({ea.SrcOffset} vs {eb.SrcOffset}) - unsupported");
                }

                // Stable slots (attention params + arena int/float): diff element-wise per slot.
                slotPatches = new();
                void DiffInt(List<(int Slot, int[] Data, ArrayView1D<int, Stride1D.Dense> View)> la,
                             List<(int Slot, int[] Data, ArrayView1D<int, Stride1D.Dense> View)> lb, string family)
                {
                    if (la.Count != lb.Count) throw new InvalidOperationException($"{family} slot count mismatch {la.Count}/{lb.Count}");
                    for (int k = 0; k < la.Count; k++)
                    {
                        var (sa, da, va) = la[k]; var (sb, db, _) = lb[k];
                        if (sa != sb || da.Length != db.Length) throw new InvalidOperationException($"{family} slot {k} parity mismatch");
                        var idx = new List<int>(); var slopes = new List<long>();
                        for (int i = 0; i < da.Length; i++) if (da[i] != db[i]) { idx.Add(i); slopes.Add((long)db[i] - da[i]); }
                        if (idx.Count > 0)
                        {
                            var (rb, ro) = RawOf((IContiguousArrayView)va.BaseView);
                            slotPatches.Add(new SlotPatch(rb, ro, (int[])da.Clone(), idx.ToArray(), slopes.ToArray(), false, new byte[da.Length * 4]));
                        }
                    }
                }
                DiffInt(probeA.Attn, probeB.Attn, "attention");
                DiffInt(probeA.ArenaInt, probeB.ArenaInt, "arena-int");
                if (probeA.ArenaFloat.Count != probeB.ArenaFloat.Count)
                    throw new InvalidOperationException($"arena-float slot count mismatch {probeA.ArenaFloat.Count}/{probeB.ArenaFloat.Count}");
                for (int k = 0; k < probeA.ArenaFloat.Count; k++)
                {
                    var (sa, da, va) = probeA.ArenaFloat[k]; var (sb, db, _) = probeB.ArenaFloat[k];
                    if (sa != sb || da.Length != db.Length) throw new InvalidOperationException($"arena-float slot {k} parity mismatch");
                    var idx = new List<int>(); var slopes = new List<long>();
                    for (int i = 0; i < da.Length; i++)
                        if (da[i] != db[i])
                        {
                            // Cursor-dependent floats are integer-valued lengths/positions; anything else is unsupported.
                            long ia = (long)da[i], ib = (long)db[i];
                            if (ia != da[i] || ib != db[i])
                                throw new InvalidOperationException($"arena-float slot {k}[{i}]: non-integer cursor-dependent float {da[i]} -> {db[i]}");
                            idx.Add(i); slopes.Add(ib - ia);
                        }
                    if (idx.Count > 0)
                    {
                        var (rb, ro) = RawOf((IContiguousArrayView)va.BaseView);
                        slotPatches.Add(new SlotPatch(rb, ro, (float[])da.Clone(), idx.ToArray(), slopes.ToArray(), true, new byte[da.Length * 4]));
                    }
                }
            }

            // The P0+1 probe left stale slot/scalar values (for P0+1 with the SAME dummy relation) -
            // consistent with the FIRST PatchAndReplay being at pastLen P0+1, which rewrites all of
            // them anyway. Cursor: leave at P0+1 (the capture pass consumed the real token at P0).
            session.SetGGUFDecodePastLen(p0 + 1);
            var cap = new WebGPUDecodeCapture(acc, session, planA, inputBuf, inputDict, capOut, p0,
                scalarPatches, copyPatches, slotPatches);
            cap._argmax = argmax;
            cap._vocab = argmax != null ? LastLogits(capOut).Vocab : 0;
            return cap;
        }
        catch
        {
            inputBuf.Dispose();
            throw;
        }
        finally
        {
            SlotProbe.Detach();
            FusedAttentionKernel.UseStableCaptureSlots = false;
            GraphExecutor.UseCaptureParamSlots = false;
            GraphExecutor.SuppressDrains = false;
            GraphCompiler.ShapeSubgraphFoldEnabled = prevFold;
            GraphExecutor.ShapeInterpElideDispatch = prevElide;
            GraphExecutor.ShapeInterpValidate = prevValidate;
            WebGPUBackend.EnableBindGroupCaching = prevBgCache;
        }
    }

    /// <summary>The capture pass's outputs (the P0 step's REAL logits) - stable tensors, also
    /// returned by every <see cref="PatchAndReplayAsync"/>.</summary>
    public IReadOnlyDictionary<string, Tensor> Outputs => _outputs;

    /// <summary>
    /// Decode ONE token via patched plan replay: writes <paramref name="tokenId"/> into the stable
    /// input buffer, patches every discovered cursor-dependent value for
    /// <paramref name="pastLen"/>, replays with one interop crossing, and advances the session
    /// cursor. The returned dict's logits are valid until the next call.
    /// </summary>
    /// <summary>Per-phase ms of the most recent <see cref="PatchAndReplayAsync"/> (diagnostics).</summary>
    public double LastPatchMs { get; private set; }
    /// <summary>Patch sub-phases (diagnostics): input write / scalar patches / slot patches / copy patches.</summary>
    public (double Input, double Scalars, double Slots, double Copies) LastPatchSplitMs { get; private set; }
    /// <summary>ms of the last replay's plan call (interop + JS encode + submit).</summary>
    public double LastReplayMs { get; private set; }
    /// <summary>ms of the last replay's GPU wait.</summary>
    public double LastSyncMs { get; private set; }

    public async Task<IReadOnlyDictionary<string, Tensor>> PatchAndReplayAsync(float tokenId, int pastLen)
    {
        var t1 = await PatchAllAsync(tokenId, pastLen);
        await _plan.ReplayAsync();
        var t2 = System.Diagnostics.Stopwatch.GetTimestamp();
        await _accelerator.SynchronizeAsync();
        var t3 = System.Diagnostics.Stopwatch.GetTimestamp();
        LastReplayMs = System.Diagnostics.Stopwatch.GetElapsedTime(t1, t2).TotalMilliseconds;
        LastSyncMs = System.Diagnostics.Stopwatch.GetElapsedTime(t2, t3).TotalMilliseconds;
        _session.SetGGUFDecodePastLen(pastLen + 1);
        return _outputs;
    }

    /// <summary>
    /// Greedy decode ONE token with a SINGLE GPU round-trip: patch + replay (the plan's final
    /// dispatch is the folded argmax partial kernel) + read the partial pairs - the readback's own
    /// mapAsync fence orders after the replay, so no separate SynchronizeAsync is needed. Requires
    /// the capture to have been built with an argmax instance.
    /// </summary>
    public async Task<int> PatchAndDecodeGreedyAsync(float tokenId, int pastLen)
    {
        if (_argmax == null) throw new InvalidOperationException("capture was built without an argmax fold");
        var t1 = await PatchAllAsync(tokenId, pastLen);
        await _plan.ReplayAsync();
        var t2 = System.Diagnostics.Stopwatch.GetTimestamp();
        int token = await _argmax.ReadPartialsAsync(_vocab);   // the ONLY per-token fence
        var t3 = System.Diagnostics.Stopwatch.GetTimestamp();
        LastReplayMs = System.Diagnostics.Stopwatch.GetElapsedTime(t1, t2).TotalMilliseconds;
        LastSyncMs = System.Diagnostics.Stopwatch.GetElapsedTime(t2, t3).TotalMilliseconds;
        _session.SetGGUFDecodePastLen(pastLen + 1);
        return token;
    }

    // Shared patch phase: input token + scalar bytes + slot arrays + copy destinations, with the
    // sub-phase diagnostics. Returns the end-of-patch timestamp.
    private async Task<long> PatchAllAsync(float tokenId, int pastLen)
    {
        var t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        int dp = pastLen - _p0;
        _queue.WriteBuffer(_inputRawBuf, (ulong)_inputRawOfs, BitConverter.GetBytes(tokenId));
        var ta = System.Diagnostics.Stopwatch.GetTimestamp();

        foreach (var (snap, ofs, v0, slope) in _scalarPatches)
            _plan.PatchScalarInt(snap, ofs, v0 + slope * dp);
        var tb = System.Diagnostics.Stopwatch.GetTimestamp();

        // Slot templates are anchored at P0 - recompute absolutely (not incrementally) each call.
        // RAW queue.WriteBuffer (bytes built in the preallocated scratch) - see the field comment.
        foreach (var sp in _slotPatches)
        {
            if (sp.IsFloat)
            {
                var t = (float[])sp.Template;
                System.Buffer.BlockCopy(t, 0, sp.Scratch, 0, sp.Scratch.Length);
                for (int i = 0; i < sp.Indices.Length; i++)
                    BitConverter.GetBytes(t[sp.Indices[i]] + sp.Slopes[i] * dp).CopyTo(sp.Scratch, sp.Indices[i] * 4);
            }
            else
            {
                var t = (int[])sp.Template;
                System.Buffer.BlockCopy(t, 0, sp.Scratch, 0, sp.Scratch.Length);
                for (int i = 0; i < sp.Indices.Length; i++)
                    BitConverter.GetBytes((int)(t[sp.Indices[i]] + sp.Slopes[i] * dp)).CopyTo(sp.Scratch, sp.Indices[i] * 4);
            }
            _queue.WriteBuffer(sp.Buf, (ulong)sp.ByteOfs, sp.Scratch);
        }

        var tc = System.Diagnostics.Stopwatch.GetTimestamp();
        if (_copyPatches.Count > 0)
        {
            for (int i = 0; i < _copyPatches.Count; i++)
                _copyOfsScratch[i] = _copyPatches[i].DstAtP0 + _copyPatches[i].Slope * dp;
            await _plan.PatchCopyDstOffsetsAsync(_copyEntryIdx, _copyOfsScratch);
        }

        var t1 = System.Diagnostics.Stopwatch.GetTimestamp();
        LastPatchSplitMs = (
            System.Diagnostics.Stopwatch.GetElapsedTime(t0, ta).TotalMilliseconds,
            System.Diagnostics.Stopwatch.GetElapsedTime(ta, tb).TotalMilliseconds,
            System.Diagnostics.Stopwatch.GetElapsedTime(tb, tc).TotalMilliseconds,
            System.Diagnostics.Stopwatch.GetElapsedTime(tc, t1).TotalMilliseconds);
        LastPatchMs = System.Diagnostics.Stopwatch.GetElapsedTime(t0, t1).TotalMilliseconds;
        return t1;
    }

    public void Dispose()
    {
        try { _plan.Dispose(); } catch { }
        try { _inputBuf.Dispose(); } catch { }
    }
}
