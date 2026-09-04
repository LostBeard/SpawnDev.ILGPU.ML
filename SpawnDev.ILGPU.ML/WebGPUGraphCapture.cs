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
        Dictionary<string, int[]> inputShapes)
    {
        _accelerator = accelerator;
        _plan = plan;
        _inputBuffers = inputBuffers;
        _outputs = outputs;
        InputShapes = inputShapes;
    }

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
        GraphExecutor.ShapeInterpElideDispatch = true;
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
            return new WebGPUGraphCapture(acc, plan, inputs, capOut, shapes);
        }
        finally
        {
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
    }
}
