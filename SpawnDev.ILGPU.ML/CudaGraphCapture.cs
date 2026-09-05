using System.Linq;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// A captured, replayable CUDA graph of one <see cref="InferenceSession"/> forward at a FIXED input shape.
///
/// Recording the ~thousands of per-node kernel launches ONCE into a <c>CudaGraph</c> and replaying them with a
/// single <c>cuGraphLaunch</c> removes the per-node CPU launch-prep (param-buffer alloc + H2D upload +
/// cuLaunchKernel) that dominates a warm forward. On DAv3-Small (RTX 4070) this is ~3x: a 355.6 ms non-graph
/// forward replays in 116.8 ms, BIT-IDENTICAL. The real win is the video / repeat-inference path: capture on the
/// first frame at a resolution, then every subsequent frame at that resolution just replays.
///
/// The captured graph reads the SAME device input buffer(s) it was recorded with and writes the SAME output
/// buffer(s). <see cref="ReplayAsync"/> copies fresh input data into those buffers, replays, and returns the
/// (stable) output tensors. CUDA-only (graph capture is a CUDA driver feature); <see cref="TryCaptureAsync"/>
/// returns null on any other backend or when the driver lacks the graph API.
///
/// LIFETIME: the input buffers passed to <see cref="TryCaptureAsync"/> and the returned output buffers are the
/// exact device buffers the graph references — keep them alive (do not dispose or let the session's pool recycle
/// them) for as long as this capture is used. Do NOT interleave a normal <see cref="InferenceSession.RunAsync"/>
/// on the same session between replays: it would recycle the captured output buffers. Capture-once, replay-many.
/// </summary>
public sealed class CudaGraphCapture : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly CudaStream _stream;
    private readonly CudaGraphExec _exec;
    private readonly Dictionary<string, Tensor> _inputBuffers;   // stable device inputs the graph reads
    private readonly Dictionary<string, Tensor> _outputs;        // stable device outputs the graph writes
    private readonly Kernels.CaptureParamArena _arena;           // kernel-params slots THIS graph binds

    /// <summary>The input shapes this graph was captured for. A replay is only valid at the SAME shapes.</summary>
    public IReadOnlyDictionary<string, int[]> InputShapes { get; }

    private CudaGraphCapture(Accelerator accelerator, CudaStream stream, CudaGraphExec exec,
        Dictionary<string, Tensor> inputBuffers, Dictionary<string, Tensor> outputs,
        Dictionary<string, int[]> inputShapes, Kernels.CaptureParamArena arena)
    {
        _accelerator = accelerator;
        _stream = stream;
        _exec = exec;
        _arena = arena;
        _inputBuffers = inputBuffers;
        _outputs = outputs;
        InputShapes = inputShapes;
    }

    /// <summary>
    /// Capture the session's forward for <paramref name="inputs"/> into a replayable CUDA graph. Runs two WARM
    /// forwards (JIT + finalize the readback cache + prime the buffer pool so the capture pass allocates nothing)
    /// then records a third forward under capture. Returns null if the accelerator is not CUDA or the driver has
    /// no graph API. The <paramref name="inputs"/> device buffers become this capture's stable inputs.
    /// </summary>
    /// <summary>DIAGNOSTIC ONLY: keep pool drains suppressed after capture, to test whether a
    /// post-capture drain is what frees the buffers a replay then reads.</summary>
    internal static bool ExperimentKeepDrainsSuppressed;

    /// <param name="allowControlFlow">
    /// The CALLER's decision, when it has one. <see cref="Graph.SessionGraphCapture"/> can establish by
    /// OBSERVATION that a full forward runs no control-flow body (see its remarks), and that observation is
    /// strictly better evidence than the blanket refusal below - so when it passes true, honour it.
    /// </param>
    /// <remarks>
    /// ⚠️ WHY THE PARAMETER EXISTS. This guard used to read the STATIC
    /// <c>SessionGraphCapture.RefuseControlFlow</c>, while the opt-in ZipVoice actually uses is the
    /// PER-INSTANCE <c>SessionGraphCapture.AllowControlFlow</c>. So the session-level capture would announce
    /// "observing: a full forward ran NO control-flow body, so recording is safe" and this method would then
    /// refuse anyway - MEASURED 2026-09-03 on CUDA, where the sample-level capture A/B consequently reported
    /// "SKIPPED: capture is not live" and had never once run.
    /// </remarks>
    public static async Task<CudaGraphCapture?> TryCaptureAsync(InferenceSession session,
        Dictionary<string, Tensor> inputs, bool? allowControlFlow = null)
    {
        var acc = session.Accelerator;
        if (acc is not CudaAccelerator) return null;
        if (!CudaStream.SupportsGraphCapture) return null;

        // ⚠️ REFUSE control flow. If/Loop/Scan run their bodies through SubgraphRunner, which calls
        // BuildExecutor on EVERY execution and allocates permanent buffers there - a device allocation
        // inside the capture window. That is not a catchable failure: a mid-capture cuMemAlloc is an
        // UNCATCHABLE 0xC0000005 that takes the process down, so the usual try/catch-and-degrade cannot
        // save it. Measured on ZipVoice's decoder: segfault, exit 139, from
        // BufferPool.AllocatePermanent <- SubgraphRunner.BuildExecutor <- IfOperator.ExecuteAsync.
        //
        // Refusing here degrades to the direct forward, which is the whole point of a best-effort capture.
        // The real fix is for SubgraphRunner to build its executor ONCE and reuse it - that would remove a
        // per-call allocation from every control-flow graph as well as unblocking capture - but a guard
        // that turns a process crash into a graceful fallback should not wait on that work.
        var controlFlow = new[] { "If", "Loop", "Scan" };
        bool refuse = allowControlFlow is bool allow ? !allow : Graph.SessionGraphCapture.RefuseControlFlow;
        var present = refuse
            ? session.OperatorTypes.Where(o => controlFlow.Contains(o)).ToArray()
            : Array.Empty<string>();
        if (present.Length > 0)
        {
            Console.WriteLine($"[CudaGraphCapture] graph contains control flow ({string.Join(", ", present)}), "
                + "whose subgraph executors allocate per call - a mid-capture allocation is an uncatchable "
                + "access violation; running direct forward.");
            return null;
        }

        // ⚠️ SAVED and restored in the finally below. This is not tidiness - leaving it on silently
        // CORRUPTS every later direct forward on this session.
        //
        // The readback cache auto-detects which captured tensors are safe to reuse by probing two runs and
        // keeping only the values that MATCH across them. Its own documentation calls that "correct by
        // construction", and it is - given two runs with DIFFERENT data. The capture path below runs its two
        // warm passes with the SAME `inputs`, because that is what capturing a fixed-shape graph means. So
        // every readback compares equal and gets cached as "stable", INCLUDING data-derived ones that
        // genuinely change per call.
        //
        // While the flag stays on, those frozen values are seeded into runtimeConstants on every subsequent
        // run. Measured: after a capture that FAILED and fell through to the direct forward, ZipVoice
        // rendered at rms 0.0021 instead of 0.0761 - audio that is quietly wrong rather than absent, from a
        // fallback whose entire purpose is to degrade safely.
        bool prevCacheReadbacks = session.CacheShapeReadbacks;
        session.CacheShapeReadbacks = true;   // finalize a stable readback cache → the capture pass syncs nothing
        var capStream = (CudaStream)acc.CreateStream();
        Dictionary<string, Tensor> capOut;
        CudaGraph graph;

        // The capture path is built for the dispatch-ELIDE regime: shape ops are CPU-resolved (not dispatched),
        // so the captured graph is the pure GPU-compute forward. Force it here (save + restore) so a caller that
        // has elide off still gets a working capture. Elide is bit-identical to non-elide, so the captured result
        // matches a normal forward regardless of the caller's prior setting.
        bool prevFold = GraphCompiler.ShapeSubgraphFoldEnabled;
        bool prevElide = GraphExecutor.ShapeInterpElideDispatch;
        bool prevValidate = GraphExecutor.ShapeInterpValidate;
        long prevReleaseCap = GraphExecutor.MaxPendingReleaseBytes;
        GraphCompiler.ShapeSubgraphFoldEnabled = true;
        GraphExecutor.ShapeInterpElideDispatch = true;
        GraphExecutor.ShapeInterpValidate = false;
        FusedAttentionKernel.UseStableCaptureSlots = true;
        GraphExecutor.UseCaptureParamSlots = true;
        // Bound the warm passes' deferred-release backlog so the pool footprint stays near the TRUE
        // simultaneous-live set instead of (live + 512MB pending). On a memory-tight card the default
        // 512MB backlog inflates warm VRAM enough to trip AllocateWithReclaim mid-warm, whose bucket
        // disposal leaves the capture pass under-primed → cuMemAlloc mid-capture (0xC0000005). A small
        // cap = frequent warm drains = buckets returned promptly = primed to the real peak. One-time
        // warm cost; the provable guard below refuses capture if a warm reclaim fired anyway.
        GraphExecutor.MaxPendingReleaseBytes = 64L * 1024 * 1024;
        // 🔴 THIS CAPTURE GETS ITS OWN PARAM ARENA. The graph baked below holds DEVICE POINTERS into the
        // arena's slots and reads them at every replay, so a later capture's warm passes must not upload
        // their own params into those same buffers. See CaptureParamArena.BeginCaptureScope.
        var arena = Kernels.CaptureParamArena.BeginCaptureScope(acc);
        bool arenaHandedOff = false;
        try
        {
            using (acc.WithDefaultStream(capStream))   // reroute *StreamKernel launches → capStream
            {
                // Warm A: JIT + populate stable attention/param slots + finalize the readback cache + snapshot
                // runtimeConstants for the capture pass to seed.
                await session.RunAsync(inputs);
                await acc.SynchronizeAsync();
                // Warm B: primes every size-bucket. Reset the reclaim trace first so the guard below can
                // PROVE this pass held resident without disposing bucketed buffers.
                BufferPool.ResetReclaimTrace();
                await session.RunAsync(inputs);
                await acc.SynchronizeAsync();
                // Provable priming guard: if Warm B tripped AllocateWithReclaim, the pool's bucket contents
                // are no longer a superset of the capture pass's rentals → a mid-capture cuMemAlloc would AV
                // (0xC0000005, uncatchable). Do NOT enter the capture window; degrade to the direct forward.
                if (BufferPool.ReclaimFireCount > 0)
                {
                    Console.WriteLine($"[CudaGraphCapture] warm reclaimed {BufferPool.ReclaimFireCount}x " +
                        $"({BufferPool.ReclaimFreedBytes / 1048576.0:F0} MiB) - working set not resident, " +
                        "capture not provably safe; running direct forward.");
                    capStream.Dispose();
                    return null;
                }
                // ⚠️ WARM UNTIL NOTHING NEW REGISTERS, then quiesce ILGPU's collector.
                //
                // ILGPU runs a BACKGROUND GC THREAD (Accelerator.GC.cs): it waits on a monitor and, when
                // pulsed, disposes collected child objects and evicts cached kernels - i.e. it calls
                // cuModuleUnload FROM ANOTHER THREAD AT AN ARBITRARY MOMENT. The pulse comes from
                // RegisterChildObject, every Nth registration.
                //
                // CUDA forbids driver work on a capturing stream, so that thread firing inside the capture
                // window is fatal and uncatchable: 0xC0000005 in cuModuleUnload (measured, on ZipVoice's
                // decoder). It is also why capture works at all for the graphs that already use it - a
                // fully warm forward registers nothing, so the collector never wakes.
                //
                // Two warm passes are not enough when a graph has CONTROL FLOW: a branch body builds its
                // own executor and kernels the first time that branch is taken, so a body first entered on
                // the capture pass registers objects exactly where it must not. Warming to a FIXED POINT -
                // repeat until the child-object count stops moving - is what makes "no registrations during
                // capture" provable rather than hoped for.
                //
                // The GC.Collect + WaitForPendingFinalizers then leaves nothing dead for the collector to
                // find even if it does wake: a pulse it cannot act on is harmless.
                int prevChildren = -1;
                for (int warm = 0; warm < 6 && acc.NumberChildObjects != prevChildren; warm++)
                {
                    prevChildren = acc.NumberChildObjects;
                    await session.RunAsync(inputs);
                    await acc.SynchronizeAsync();
                }
                if (acc.NumberChildObjects != prevChildren)
                {
                    Console.WriteLine($"[CudaGraphCapture] accelerator objects still growing after 6 warm "
                        + $"passes ({prevChildren} -> {acc.NumberChildObjects}); a registration inside the "
                        + "capture window would unload a module on ILGPU's GC thread and take the process "
                        + "down. Running direct forward.");
                    capStream.Dispose();
                    return null;
                }
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                await acc.SynchronizeAsync();

                // Capture: record the forward. Drains suppressed → no periodic drain / final sync / buffer-return
                // aborts the capture; the seeded runtimeConstants keep eliding identical to warm.
                GraphExecutor.SuppressDrains = true;
                capStream.BeginCapture(CudaStreamCaptureMode.Global);
                try
                {
                    capOut = await session.RunAsync(inputs);
                    graph = capStream.EndCapture();
                    // A graph now exists and bakes pointers into the arena's slots → it owns the arena.
                    arenaHandedOff = true;
                }
                catch
                {
                    // A capture-illegal op (e.g. a mid-capture stream sync in a dynamic Resize) INVALIDATES the
                    // capture. EndCapture MUST still run to take the stream OUT of capture mode - otherwise the
                    // stream stays capturing, the CUDA context is poisoned, and the caller's direct-forward
                    // fallback AVs on its next cuMemAlloc (0xC0000005). The invalidated EndCapture reports
                    // failure itself; swallow it - we only need the capture-mode reset. Mirrors the WebGPU path.
                    try { capStream.EndCapture(); } catch { /* expected: capture was invalidated */ }
                    // The caller degrades to a direct forward and never sees this stream again, so it would
                    // otherwise leak for the life of the accelerator - once per capture-incompatible graph.
                    try { capStream.Dispose(); } catch { }
                    throw;
                }
                finally { GraphExecutor.SuppressDrains = ExperimentKeepDrainsSuppressed; }
            }
        }
        finally
        {
            Kernels.CaptureParamArena.EndCaptureScope();
            // No graph was instantiated (guard tripped, or the capture threw) → nothing binds these slots.
            // DRAIN FIRST: the warm/probe passes' dispatches read these slots.
            if (!arenaHandedOff)
            {
                try { await acc.SynchronizeAsync(); } catch { }
                try { arena.Dispose(); } catch { }
            }
            session.CacheShapeReadbacks = prevCacheReadbacks;
            FusedAttentionKernel.UseStableCaptureSlots = false;
            GraphExecutor.UseCaptureParamSlots = false;
            GraphExecutor.SuppressDrains = ExperimentKeepDrainsSuppressed;
            GraphCompiler.ShapeSubgraphFoldEnabled = prevFold;
            GraphExecutor.ShapeInterpElideDispatch = prevElide;
            GraphExecutor.ShapeInterpValidate = prevValidate;
            GraphExecutor.MaxPendingReleaseBytes = prevReleaseCap;
        }

        CudaGraphExec exec;
        using (graph) { exec = graph.Instantiate(); exec.Upload(capStream); }

        var shapes = new Dictionary<string, int[]>();
        foreach (var (name, t) in inputs) shapes[name] = (int[])t.Shape.Clone();
        return new CudaGraphCapture(acc, capStream, exec, inputs, capOut, shapes, arena);
    }

    /// <summary>
    /// Replay the captured forward with fresh input data. Copies each of <paramref name="newInputs"/> into the
    /// capture's stable input buffer (device→device, stream-ordered before the launch), replays the graph with a
    /// single <c>cuGraphLaunch</c>, and returns the stable output tensors (valid until the next replay). The
    /// input shapes must match <see cref="InputShapes"/>.
    /// </summary>
    public async Task<Dictionary<string, Tensor>> ReplayAsync(Dictionary<string, Tensor> newInputs)
    {
        using (_accelerator.WithDefaultStream(_stream))   // order the input copies before the launch on _stream
        {
            foreach (var (name, t) in newInputs)
            {
                if (!_inputBuffers.TryGetValue(name, out var stable)) continue;
                int n = Math.Min(t.ElementCount, stable.ElementCount);
                if (n > 0) stable.Data.SubView(0, n).CopyFrom(t.Data.SubView(0, n));
            }
            _exec.Launch(_stream);
        }
        await _stream.SynchronizeAsync();
        return _outputs;
    }

    public void Dispose()
    {
        try { _exec.Dispose(); } catch { }
        try { _stream.Dispose(); } catch { }
        // Only now: the baked graph reads these slots on every replay.
        try { _arena.Dispose(); } catch { }
    }
}
