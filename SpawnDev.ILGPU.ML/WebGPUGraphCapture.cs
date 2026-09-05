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
/// A captured, replayable WebGPU dispatch plan of one <see cref="InferenceSession"/> forward at a FIXED
/// input shape - the browser twin of <see cref="CudaGraphCapture"/>.
///
/// A warm WebGPU forward pays ~2,500 per-node .NET-&gt;JS dispatch encodings plus per-node drains - the
/// dominant cost of the DAv3 77s-vs-ORT-Web-73ms gap (measured: SpawnJS interop is 14-64us/call and each
/// dispatch spends ~100 calls' worth of crossings + payloads). Recording the dispatches ONCE into a
/// <see cref="WebGPUDispatchPlan"/> (Dawn has no graph API, but the command encoder IS the graph recorder)
/// and replaying them with a SINGLE interop crossing removes that entire term: per-forward cost becomes
/// writeBuffer(inputs) + one JS encode loop + one submit + the output readback.
///
/// Same regime and same validity contract as the CUDA capture: two warm forwards (JIT + stable attention/
/// param slots + finalized readback cache + pre-warmed buffer pool so the capture pass allocates nothing),
/// then the third forward records under the dispatch-elide regime with drains suppressed. The captured
/// plan reads the SAME device input buffer(s) and writes the SAME output buffer(s) - keep them alive, do
/// not interleave normal <see cref="InferenceSession.RunAsync"/> calls on the session between replays.
/// Capture-once, replay-many (the video / repeat-inference path).
/// </summary>
public sealed class WebGPUGraphCapture : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly WebGPUDispatchPlan _plan;
    private readonly Dictionary<string, Tensor> _inputBuffers;   // stable device inputs the plan reads
    private readonly Dictionary<string, Tensor> _outputs;        // stable device outputs the plan writes
    private readonly CaptureParamArena _arena;                   // kernel-params slots THIS plan binds

    /// <summary>The input shapes this plan was captured for. A replay is only valid at the SAME shapes.</summary>
    public IReadOnlyDictionary<string, int[]> InputShapes { get; }

    /// <summary>Number of GPU dispatches the captured plan replays.</summary>
    public int DispatchCount => _plan.DispatchCount;

    /// <summary>
    /// CPU-&gt;GPU writes that happened inside the most recent capture window - work a replay cannot repeat.
    /// </summary>
    /// <remarks>See the note at the end of <see cref="TryCaptureAsync"/>. Non-zero is the first thing to
    /// check when a replay does not reproduce its own capture.</remarks>
    public static int HostWritesDuringCapture { get; private set; }

    /// <summary>
    /// The subset of <see cref="HostWritesDuringCapture"/> that is the per-dispatch packed scalar-params
    /// upload. See <c>WebGPUDispatchPlan.ScalarParamWriteCount</c>.
    /// </summary>
    /// <remarks>
    /// ⚠️ Reported separately because until 2026-09-04 it was reported NOWHERE. The scalar-params
    /// <c>queue.writeBuffer</c> is issued once per dispatch and was invisible to the host-write census,
    /// which only hooked <c>WebGPUBuffer</c>'s upload paths - so this class printed "0 host writes" for a
    /// capture window containing one per dispatch, and that zero was cited as evidence the plan was
    /// complete while the replay disagreed with its own capture in every value.
    /// </remarks>
    public static int ScalarParamWritesDuringCapture { get; private set; }

    /// <summary>
    /// When set, <see cref="TryCaptureAsync"/> copies the capture pass's FIRST output to the host before
    /// anything else can touch it, into <see cref="CapturePassOutput"/>. Diagnostic; default off.
    /// </summary>
    /// <remarks>
    /// 🔴 WHY THIS HAD TO EXIST. Nothing could observe the capture pass's own result. The obvious way to
    /// look at it - call the session and read what comes back - does NOT return it:
    /// <c>SessionGraphCapture.RunAsync</c> ends with
    /// <c>if (_webGpu != null) return await _webGpu.ReplayAsync(inputs)</c>, so the call that performs the
    /// capture returns a REPLAY. Every value anyone has called "the capture pass" since 2026-09-03 was a
    /// replay result, including the one the fidelity test prints under the heading "both are ordinary
    /// forwards, so a difference here is not about replay at all". It is entirely about replay.
    /// <para>
    /// The consequence was a whole day spent looking for arithmetic that goes wrong under the capture
    /// regime. It does not: MEASURED 2026-09-04, all 4,873 probed node outputs of the real capture pass
    /// match a plain forward, and its LAST node agrees to the digit
    /// (<c>[0.238362,0.349662,0.372184,0.181907]</c>) while the value handed back differs
    /// (<c>[-0.797333,0.443003,0.288282,-0.534283]</c>). The capture pass computes the right answer.
    /// </para>
    /// <para>
    /// Read here, between the capture forward and the <c>finally</c> that lifts SuppressDrains, so the
    /// sample cannot be blamed on pool recycling afterwards.
    /// </para>
    /// </remarks>
    public static bool RecordCapturePassOutput { get; set; }

    /// <summary>The capture pass's own first output, host-side. See <see cref="RecordCapturePassOutput"/>.</summary>
    public static float[]? CapturePassOutput { get; private set; }

    /// <summary>
    /// Whether the capture pass runs with <see cref="GraphExecutor.ShapeInterpElideDispatch"/> ON.
    /// Default true (the shipping behaviour). Set false to A/B whether ELIDED dispatches are the work a
    /// replay is missing.
    /// </summary>
    /// <remarks>
    /// ⚠️ WHY THIS IS THE NEXT EXPERIMENT. As of 2026-09-04 the capture pass is PROVEN correct (0 of
    /// 16,900) and unreplayable host writes are PROVEN zero, so the replay is missing work that is neither
    /// a host write nor bad arithmetic. Dispatch-elide is the one mechanism that removes work from the
    /// plan BY DESIGN: a CPU-resolved shape op does not dispatch, so nothing is recorded for it, and its
    /// buffer holds a value some earlier pass put there. That is fine for the capture pass - the value is
    /// still sitting in the buffer - and it is exactly what a replay cannot reproduce once the pool has
    /// handed that memory to something else.
    /// <para>
    /// Turning it off makes the captured plan the FULL forward (~1200 more dispatches per frame on
    /// ZipVoice, so it costs replay speed). If the replay becomes faithful with it off, elided dispatches
    /// are the missing work and the fix is to record a write for each elided value - the pattern
    /// <c>CaptureParamArena.CaptureConstWrite</c> already implements for Range and Einsum. If the replay
    /// is STILL wrong, elide is exonerated and the remaining suspects are the named/pooled buffers a
    /// warm pass populated.
    /// </para>
    /// </remarks>
    public static bool ElideDispatchDuringCapture { get; set; } = true;

    /// <summary>The stable output tensors the captured plan writes (also returned by ReplayAsync).</summary>
    public IReadOnlyDictionary<string, Tensor> Outputs => _outputs;

    /// <summary>
    /// When true, each <see cref="ReplayAsync"/> also fetches the JS-side encode/submit split into
    /// <see cref="LastJsEncodeMs"/>/<see cref="LastJsSubmitMs"/> (two extra interop reads per replay -
    /// diagnostic only, leave off in production). The .NET-side splits below are recorded always
    /// (Stopwatch timestamps, effectively free).
    /// </summary>
    public bool CollectTimings { get; set; }

    /// <summary>ms the last replay spent copying fresh inputs into the stable input buffers (0 when the caller reuses the stable tensor - the video path).</summary>
    public double LastInputCopyMs { get; private set; }
    /// <summary>ms the last replay spent in the plan call: one .NET->JS crossing + the JS encode loop + queue.submit.</summary>
    public double LastPlanCallMs { get; private set; }
    /// <summary>ms the last replay spent awaiting GPU completion (SynchronizeAsync = onSubmittedWorkDone).</summary>
    public double LastSyncMs { get; private set; }
    /// <summary>JS-side ms of the last replay's encode loop (only when <see cref="CollectTimings"/>).</summary>
    public double LastJsEncodeMs { get; private set; }
    /// <summary>JS-side ms of the last replay's enc.finish()+queue.submit (only when <see cref="CollectTimings"/>).</summary>
    public double LastJsSubmitMs { get; private set; }

    private WebGPUGraphCapture(Accelerator accelerator, WebGPUDispatchPlan plan,
        Dictionary<string, Tensor> inputBuffers, Dictionary<string, Tensor> outputs,
        Dictionary<string, int[]> inputShapes, CaptureParamArena arena)
    {
        _accelerator = accelerator;
        _plan = plan;
        _arena = arena;
        _inputBuffers = inputBuffers;
        _outputs = outputs;
        InputShapes = inputShapes;
        ReclaimGenerationAtCapture = Tensors.BufferPool.ReclaimFireCount;
    }

    /// <summary>
    /// <see cref="Tensors.BufferPool.ReclaimFireCount"/> as it stood when this plan was recorded.
    /// </summary>
    /// <remarks>
    /// 🔴 THE PLAN'S BIND GROUPS REFERENCE POOLED BUFFERS, AND THE POOL CAN FREE THEM. This class's own
    /// summary says the captured plan reads and writes the SAME device buffers and that the caller must
    /// "keep them alive" - but a pooled intermediate goes back to the pool when the capture run ends, and
    /// <c>BufferPool.DisposeBucketedBuffers</c> (the under-pressure <c>AllocateWithReclaim</c> path, also
    /// reachable deterministically via <c>BufferPool.ForceReclaimEveryNRents</c>) then DESTROYS it. The
    /// recorded bind group still points at it, and the next replay submits a command buffer referencing
    /// destroyed memory.
    /// <para>
    /// MEASURED 2026-09-04 in the SpawnDev.AI demo: <c>/api/speak</c> returned 500 with
    /// "[Buffer (unlabeled)] used in submit while destroyed. - While calling [Queue].Submit(...)", thrown
    /// from <c>WebGPUGraphCapture.ReplayAsync</c> under
    /// <c>SpeechRecognitionPipeline.TranscribeAsync</c> - Whisper's encoder capture - as soon as a
    /// transcription was run in the middle of a ZipVoice synthesis. Disabling that capture made it go
    /// away, which is a diagnosis, not a fix.
    /// </para>
    /// Comparing this against the live count says whether a reclaim happened since the recording, which
    /// is the difference between "the plan is stale" and some other cause.
    /// </remarks>
    public int ReclaimGenerationAtCapture { get; }

    /// <summary>
    /// True when the pool has freed buffers since this plan was recorded, so replaying it may submit
    /// against destroyed memory. See <see cref="ReclaimGenerationAtCapture"/>.
    /// </summary>
    public bool InvalidatedByReclaim => Tensors.BufferPool.ReclaimFireCount != ReclaimGenerationAtCapture;

    /// <summary>
    /// Capture the session's forward for <paramref name="inputs"/> into a replayable WebGPU dispatch plan.
    /// Runs two WARM forwards (shader JIT + stable slots + readback-cache finalize + pool pre-warm) then
    /// records a third forward. Returns null when the accelerator is not WebGPU. The <paramref name="inputs"/>
    /// device buffers become this capture's stable inputs.
    /// </summary>
    public static async Task<WebGPUGraphCapture?> TryCaptureAsync(InferenceSession session, Dictionary<string, Tensor> inputs)
    {
        var acc = session.Accelerator;
        if (acc is not WebGPUAccelerator webGpu) return null;

        // 🔴 SAVED AND RESTORED, like the CUDA path - it was set here and never put back, and the CUDA
        // path's own remarks already measured what that costs.
        //
        // While the flag stays on, the frozen readback values are seeded into runtimeConstants on EVERY
        // subsequent run of this session - including a plain direct forward. So a capture that failed and
        // fell through, or any later uncaptured call, silently inherits capture-time constants. MEASURED on
        // CUDA: ZipVoice then rendered at rms 0.0021 instead of 0.0761 - audio that is quietly wrong rather
        // than absent, from a fallback whose entire purpose is to degrade safely. The WebGPU path had the
        // same set with no restore, which is the asymmetry this fixes.
        bool prevCacheReadbacks = session.CacheShapeReadbacks;
        session.CacheShapeReadbacks = true;   // finalize a stable readback cache → the capture pass syncs nothing

        // Same regime as the CUDA capture: dispatch-elide keeps shape ops CPU-resolved so the captured plan
        // is the pure GPU-compute forward; stable param slots keep kernel params buffers alive + addressed;
        // bind-group caching must be OFF (a cache hit rewrites its owned scalar buffer, which would corrupt
        // earlier plan entries - BeginDispatchCapture enforces it, we save/restore around the whole warm-up
        // so the warm passes match the capture pass exactly).
        bool prevFold = GraphCompiler.ShapeSubgraphFoldEnabled;
        bool prevElide = GraphExecutor.ShapeInterpElideDispatch;
        bool prevValidate = GraphExecutor.ShapeInterpValidate;
        bool prevBgCache = WebGPUBackend.EnableBindGroupCaching;
        long prevReleaseCap = GraphExecutor.MaxPendingReleaseBytes;
        GraphCompiler.ShapeSubgraphFoldEnabled = true;
        // Dispatch-elide ON (same as the CUDA capture): CPU-resolved shape ops don't dispatch, so the
        // captured plan is the pure compute forward - ~1200 fewer per-frame GPU passes on replay.
        // (The old WebGPU gap - "Tensor not found" when a GPU consumer read an elided EMPTY value -
        // was fixed 2026-07-03: zero-length values are excluded from elide in GraphExecutor.elideSafe
        // and dispatch normally; gate = DA3_WebGPU_ElideOn_Forward.)
        GraphExecutor.ShapeInterpElideDispatch = ElideDispatchDuringCapture;
        GraphExecutor.ShapeInterpValidate = false;
        WebGPUBackend.EnableBindGroupCaching = false;
        // ⚠️ THE STABLE-SLOT ARENA IS A CUDA REQUIREMENT, NOT A WEBGPU ONE, and it is not free here.
        //
        // On CUDA a per-call parameter buffer means a cuMemAlloc, which is ILLEGAL inside a capture window,
        // so kernel parameters are staged into arena slots by the warm passes and the capture pass re-reads
        // them without uploading. That works only while the warm and capture passes rent the SAME cursor
        // sequence. WebGPU has no allocation restriction inside a recording - the plan simply retains the
        // bind groups, and a per-call parameter buffer is retained with them - so the arena buys nothing and
        // adds a cross-pass invariant that has to hold exactly.
        //
        // MEASURED 2026-09-03 on ZipVoice's fm_decoder: with the arena OFF, a forward under the rest of the
        // capture regime is bit-identical to a plain forward at every one of 4,873 node outputs and in all
        // 16,900 output values; with it ON, the capture pass disagreed in all 16,900
        // (Pipeline_ZipVoice_CaptureReplayFidelity's regime bisect).
        // ⚠️ KEEP THE ARENA ON. It looks like a CUDA-only device (it exists so a capture window contains no
        // cuMemAlloc), but on WebGPU it is doing a second job that IS needed here: with the arena off,
        // kernel parameters are uploaded per call with `queue.writeBuffer`, and a host write is not ordered
        // against command-encoder work that has not been submitted yet. MEASURED 2026-09-03 with the arena
        // off under this regime: the capture pass produced NaN. With it on, every one of 4,873 node outputs
        // matches a plain forward.
        FusedAttentionKernel.UseStableCaptureSlots = true;
        GraphExecutor.UseCaptureParamSlots = true;
        // Bound the warm passes' deferred-release backlog (default 512MB) so warm's pool footprint stays
        // near the true live set and doesn't trip AllocateWithReclaim mid-warm, which would leave the
        // capture pass under-primed. Mirror of the CUDA capture fix; the guard below is the safety gate.
        GraphExecutor.MaxPendingReleaseBytes = 64L * 1024 * 1024;
        // 🔴 THIS CAPTURE GETS ITS OWN PARAM ARENA. The plan recorded below binds the arena's slot buffers
        // and reads them at every REPLAY. While the arena was shared per-accelerator, the warm passes of
        // the NEXT capture - a different shape, or a different model entirely, since Whisper and ZipVoice
        // share one accelerator - rented the same cursors and uploaded their own params straight into
        // those buffers, so every earlier plan silently replayed with someone else's shapes and strides.
        // See CaptureParamArena.BeginCaptureScope. The arena is disposed with this capture, below.
        int overwritesBefore = CaptureParamArena.CrossCaptureSlotOverwrites;
        var arena = CaptureParamArena.BeginCaptureScope(acc);
        bool arenaHandedOff = false;
        try
        {
            // Warm A: shader JIT + populate stable attention/param slots + finalize the readback cache +
            // snapshot runtimeConstants for the capture pass to seed.
            await session.RunAsync(inputs);
            await acc.SynchronizeAsync();
            // Warm B: primes every size-bucket. Reset the reclaim trace first so the guard below can PROVE
            // this pass held resident without disposing bucketed buffers.
            BufferPool.ResetReclaimTrace();
            await session.RunAsync(inputs);
            await acc.SynchronizeAsync();
            // Provable priming guard: if Warm B tripped AllocateWithReclaim, the pool's buckets are no
            // longer a superset of the capture pass's rentals → an allocation mid-dispatch-capture would
            // perturb the recorded plan. Do NOT capture; degrade to the direct forward.
            if (BufferPool.ReclaimFireCount > 0)
            {
                Console.WriteLine($"[WebGPUGraphCapture] warm reclaimed {BufferPool.ReclaimFireCount}x " +
                    $"({BufferPool.ReclaimFreedBytes / 1048576.0:F0} MiB) - working set not resident, " +
                    "capture not provably safe; running direct forward.");
                return null;
            }

            // Capture: record the forward. Drains suppressed → no mid-forward submit-and-wait or
            // buffer-return perturbs the stable regime; the plan records every dispatch as it encodes.
            Dictionary<string, Tensor> capOut;
            GraphExecutor.SuppressDrains = true;
            var plan = webGpu.BeginDispatchCapture();
            try
            {
                capOut = await session.RunAsync(inputs);
            }
            catch
            {
                webGpu.EndDispatchCapture().Dispose();
                throw;
            }
            finally
            {
                GraphExecutor.SuppressDrains = false;
            }
            webGpu.EndDispatchCapture();
            await acc.SynchronizeAsync();

            // Read the capture pass's OWN result while it is still the only thing that has run. See
            // RecordCapturePassOutput for why no caller can otherwise see this value: the enclosing
            // SessionGraphCapture.RunAsync returns a REPLAY, not this.
            CapturePassOutput = null;
            if (RecordCapturePassOutput && capOut.Count > 0)
            {
                try
                {
                    var first = capOut.Values.First();
                    int n = first.ElementCount;
                    if (n > 0)
                    {
                        using var sample = acc.Allocate1D<float>(n);
                        await sample.View.CopyFromAsync(first.Data.SubView(0, n));
                        await acc.SynchronizeAsync();
                        CapturePassOutput = await sample.CopyToHostAsync<float>(0, n);
                    }
                }
                catch (Exception ex)
                {
                    // A diagnostic must never decide whether a capture succeeds.
                    Console.WriteLine($"[WebGPUGraphCapture] capture-pass output sample failed: {ex.Message}");
                }
            }

            // ⚠️ HOST WRITES INSIDE THE WINDOW ARE MISSING WORK, and they are silent. A plan records
            // dispatches, buffer-to-buffer copies and clears - all command-encoder work. A
            // `queue.writeBuffer` is none of those: it moves bytes the CPU is holding, so a replay never
            // performs it and the destination keeps whatever the capture pass last left there. Constant
            // bytes are harmless; input-dependent ones make the replay confidently wrong.
            //
            // MEASURED 2026-09-03 on ZipVoice's fm_decoder: a replay did not reproduce the forward it
            // recorded AT THE INPUTS IT CAPTURED - 16,900 of 16,900 values differ, worst 0.711702. Saying so
            // HERE, with a count, is the difference between a number and two days of hypotheses.
            if (plan.HostWriteCount > 0)
                Console.WriteLine($"[WebGPUGraphCapture] ⚠️ {plan.HostWriteCount} host write(s) "
                    + $"({plan.HostWriteBytes / 1024.0:F1} KiB) happened INSIDE the capture window, of which "
                    + $"{plan.ScalarParamWriteCount} are the PER-DISPATCH packed scalar-params upload. A "
                    + "queue.writeBuffer is not part of a dispatch plan, so a replay does not repeat it - "
                    + "any of these carrying per-call data makes the replay wrong. ⚠️ READ THE SUBTRACTION, "
                    + "NOT THE TOTAL: the scalar-params writes are BENIGN (the plan retains each dispatch's "
                    + "scalar buffer via RetainScalarBuffers, so it is never recycled and the recorded bind "
                    + $"group keeps its own parameters). Unreplayable work here = {plan.HostWriteCount - plan.ScalarParamWriteCount}.");
            HostWritesDuringCapture = plan.HostWriteCount;
            ScalarParamWritesDuringCapture = plan.ScalarParamWriteCount;

            var shapes = new Dictionary<string, int[]>();
            foreach (var (name, t) in inputs) shapes[name] = (int[])t.Shape.Clone();
            arenaHandedOff = true;
            return new WebGPUGraphCapture(acc, plan, inputs, capOut, shapes, arena);
        }
        finally
        {
            CaptureParamArena.EndCaptureScope();
            // What the per-capture arena actually bought, on THIS sequence. Non-zero = the warm passes of
            // THIS attempt wrote params that, under the old per-accelerator arena, would have landed in
            // buffers the PREVIOUS capture's plan still reads on every replay - a silently wrong replay
            // from then on. ⚠️ NOT the cause of the history-dependent ZipVoice audio - see the note on
            // CaptureParamArena.BeginCaptureScope; that pipeline captures nothing.
            // ⚠️ Reported on EVERY exit, including the refused/failed ones: an attempt that ends up
            // returning null still ran two full warm forwards through the arena, so it did just as much
            // damage as a successful capture.
            int wouldHaveCorrupted = CaptureParamArena.CrossCaptureSlotOverwrites - overwritesBefore;
            if (wouldHaveCorrupted > 0)
                Console.WriteLine($"[WebGPUGraphCapture] param arena: {wouldHaveCorrupted} slot write(s) in "
                    + "this capture attempt would have overwritten the previous capture's live plan params "
                    + "(per-capture arena prevented it).");
            // No plan was recorded (guard tripped, or the capture threw) → nothing binds these slots.
            // DRAIN FIRST: the warm passes' dispatches read them, and an unsubmitted command encoder
            // referencing a destroyed buffer is the very failure this class documents.
            if (!arenaHandedOff)
            {
                try { await acc.SynchronizeAsync(); } catch { }
                try { arena.Dispose(); } catch { }
            }
            FusedAttentionKernel.UseStableCaptureSlots = false;
            GraphExecutor.UseCaptureParamSlots = false;
            GraphExecutor.SuppressDrains = false;
            GraphCompiler.ShapeSubgraphFoldEnabled = prevFold;
            GraphExecutor.ShapeInterpElideDispatch = prevElide;
            GraphExecutor.ShapeInterpValidate = prevValidate;
            WebGPUBackend.EnableBindGroupCaching = prevBgCache;
            GraphExecutor.MaxPendingReleaseBytes = prevReleaseCap;
            session.CacheShapeReadbacks = prevCacheReadbacks;
        }
    }

    /// <summary>
    /// Replay the captured forward with fresh input data. Copies each of <paramref name="newInputs"/> into
    /// the capture's stable input buffer (async device copy - drains the producer, then a queue-ordered
    /// CopyBufferToBuffer that lands before the replay submit), replays the plan with ONE interop crossing,
    /// awaits completion, and returns the stable output tensors (valid until the next replay).
    /// </summary>
    public async Task<Dictionary<string, Tensor>> ReplayAsync(Dictionary<string, Tensor> newInputs)
    {
        // Say it BEFORE the submit that would fail. A replay against a pool that has freed buffers since
        // the recording surfaces as "[Buffer (unlabeled)] used in submit while destroyed", which names
        // neither the buffer nor the reason; this names the reason. See ReclaimGenerationAtCapture.
        if (InvalidatedByReclaim)
            Console.WriteLine("[WebGPUGraphCapture] ⚠️ replaying a plan recorded before "
                + $"{Tensors.BufferPool.ReclaimFireCount - ReclaimGenerationAtCapture} pool reclaim(s) "
                + $"({Tensors.BufferPool.ReclaimFreedBytes / 1048576.0:F0} MiB freed since process start) - "
                + "its bind groups may reference destroyed buffers.");

        var t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        foreach (var (name, t) in newInputs)
        {
            if (!_inputBuffers.TryGetValue(name, out var stable)) continue;
            // The common video-path case: the caller reuses the capture's own stable input tensor
            // (fresh data written into it upstream). WebGPU forbids same-buffer CopyBufferToBuffer,
            // and there is nothing to copy anyway.
            if (ReferenceEquals(t, stable) || t.Data.Equals(stable.Data)) continue;
            int n = Math.Min(t.ElementCount, stable.ElementCount);
            if (n > 0)
                await stable.Data.SubView(0, n).CopyFromAsync(t.Data.SubView(0, n));
        }
        var t1 = System.Diagnostics.Stopwatch.GetTimestamp();
        await _plan.ReplayAsync();
        var t2 = System.Diagnostics.Stopwatch.GetTimestamp();
        await _accelerator.SynchronizeAsync();
        var t3 = System.Diagnostics.Stopwatch.GetTimestamp();
        LastInputCopyMs = System.Diagnostics.Stopwatch.GetElapsedTime(t0, t1).TotalMilliseconds;
        LastPlanCallMs = System.Diagnostics.Stopwatch.GetElapsedTime(t1, t2).TotalMilliseconds;
        LastSyncMs = System.Diagnostics.Stopwatch.GetElapsedTime(t2, t3).TotalMilliseconds;
        if (CollectTimings)
            (LastJsEncodeMs, LastJsSubmitMs) = WebGPUDispatchPlan.GetLastReplayTimings();
        return _outputs;
    }

    /// <summary>
    /// Replays the captured forward with per-pass GPU timestamps and returns the JSON kernel
    /// attribution from <see cref="WebGPUDispatchPlan.ReplayTimedAsync"/> (GPU ms by kernel label,
    /// sorted descending; <c>{"supported":false}</c> when the device lacks 'timestamp-query').
    /// Same validity contract as <see cref="ReplayAsync"/>; runs against the current stable input
    /// buffer contents and waits for completion internally. Diagnostic - do not frame-time with it.
    /// </summary>
    public Task<string> ReplayTimedAsync() => _plan.ReplayTimedAsync();

    public void Dispose()
    {
        try { _plan.Dispose(); } catch { }
        // Only now: the plan's bind groups read these slots on every replay.
        try { _arena.Dispose(); } catch { }
    }
}
