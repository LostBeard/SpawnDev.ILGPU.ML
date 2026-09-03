using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Graph;

/// <summary>
/// Reusable capture-once/replay-many wrapper around one <see cref="InferenceSession"/> at FIXED
/// input shapes: CUDA graphs on CUDA, dispatch plans on WebGPU (<see cref="WebGPUGraphCapture"/>),
/// transparent <c>RunAsync</c> fall-through everywhere else (and when capture fails). The
/// DepthEstimationPipeline proved the pattern (6s direct forward → ~300ms page estimates); this
/// class makes it one field + one call for the OTHER multi-model pipelines (SD-Turbo's CLIP/UNet/
/// VAE, and any future repeat-inference session) instead of a third inline copy of the dance.
/// Callers must keep input tensor SHAPES fixed across calls (contents are free to change); a shape
/// change throws - construct a new instance per shape.
/// </summary>
public sealed class SessionGraphCapture : IDisposable
{
    private readonly InferenceSession _session;
    private readonly Accelerator _accelerator;
    private CudaGraphCapture? _cuda;
    private WebGPUGraphCapture? _webGpu;
    private bool _captureAttempted;
    private int[][]? _capturedShapes;
    private readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _ownedInputs = new();

    /// <summary>False disables capture entirely (plain RunAsync) - the per-pipeline opt-out.</summary>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Refuse to capture graphs containing If/Loop/Scan. Default true.
    /// </summary>
    /// <remarks>
    /// ⚠️ The reason to keep this ON is that the failure it prevents is UNRECOVERABLE, not merely wrong:
    /// a device allocation inside a capture window is an uncatchable 0xC0000005 on CUDA and a HUNG DEVICE
    /// on WebGPU (DXGI_ERROR_DEVICE_HUNG - a TDR that resets the display driver, which is felt outside the
    /// process). No try/catch reaches either.
    ///
    /// It became liftable once SubgraphRunner started caching its compiled plans, because the allocation
    /// it guards against now happens on the FIRST execution - during capture's warm passes - instead of
    /// every execution. Verify per graph before trusting it: a body that allocates for some other reason
    /// still bites exactly as hard.
    /// </remarks>
    public static bool RefuseControlFlow { get; set; } = true;

    /// <summary>
    /// Per-instance override of <see cref="RefuseControlFlow"/>. Null follows the static default.
    /// </summary>
    /// <remarks>
    /// ⚠️ WHY PER INSTANCE. The refusal is a property of ONE graph - whether ITS subgraph bodies allocate
    /// per call - so a global switch is the wrong shape twice over: it cannot express "this graph is
    /// verified and that one is not", and flipping it for one pipeline silently changes every other
    /// pipeline in the process, including ones nobody has checked.
    /// <para>
    /// ⚠️ It also has to be a PROPERTY rather than an environment variable. The old opt-in was
    /// <c>ML_CF_CAPTURE=1</c>, which cannot work where it matters: environment variables do not reach the
    /// Blazor WASM runtime, so in a browser lane that switch was read as unset no matter what was exported
    /// - and the browser is the only place the refusal costs ~20x. MEASURED 2026-09-03: a run with
    /// <c>ML_CF_CAPTURE=1</c> exported still reported <c>refused: graph contains control flow (If)</c>.
    /// </para>
    /// </remarks>
    public bool? AllowControlFlow { get; set; }

    public SessionGraphCapture(InferenceSession session, Accelerator accelerator)
    {
        _session = session;
        _accelerator = accelerator;
    }

    /// <summary>True once a capture is live (calls replay instead of a full graph walk).</summary>
    public bool IsCaptured => _cuda != null || _webGpu != null;

    /// <summary>WHY capture is or is not live, in words.</summary>
    /// <remarks>
    /// ⚠️ <see cref="IsCaptured"/> alone is a dead end when the answer is "false": there are five separate
    /// ways to get there - disabled by the caller, an ineligible backend, the control-flow refusal, a
    /// TryCapture that returned null, and a thrown capture - and they call for completely different work.
    /// MEASURED 2026-09-03: the ZipVoice decoder reported <c>capture LIVE: False</c> on WebGPU while
    /// costing 8.4 s per Euler step, of which the class docs put ~62% in per-node HOST work that a
    /// recorded plan skips. Knowing it was false cost a whole measurement cycle that knowing WHY would
    /// have saved. Two of those five reasons print nothing at all on their own.
    /// </remarks>
    public string CaptureStatus { get; private set; } = "not attempted";

    /// <summary>
    /// The WebGPU replay's own split of the last frame - input copies, the single plan crossing, and the
    /// wait for GPU completion. Null on CUDA or before a capture is live.
    /// </summary>
    /// <remarks>
    /// Surfaced because without it a caller measuring a replayed frame can only see the total and has to
    /// GUESS which term dominates - and the terms are wildly different in kind: the plan call is one
    /// interop crossing, while the sync is a round trip through the JS event loop.
    /// </remarks>
    public (double InputCopyMs, double PlanCallMs, double SyncMs)? LastReplaySplit
        => _webGpu == null ? null
         : (_webGpu.LastInputCopyMs, _webGpu.LastPlanCallMs, _webGpu.LastSyncMs);

    /// <summary>Number of GPU dispatches the captured WebGPU plan replays. 0 when no plan is live.</summary>
    public int DispatchCount => _webGpu?.DispatchCount ?? 0;

    /// <summary>
    /// Run the session: first eligible call captures (paying a few warm forwards), subsequent calls
    /// replay. Input tensor shapes must match the captured shapes exactly.
    /// </summary>
    public async Task<Dictionary<string, Tensor>> RunAsync(Dictionary<string, Tensor> inputs)
    {
        bool eligible = Enabled
            && (_accelerator.AcceleratorType == AcceleratorType.Cuda
                || _accelerator.AcceleratorType == AcceleratorType.WebGPU);
        if (!eligible)
        {
            CaptureStatus = Enabled
                ? $"ineligible backend {_accelerator.AcceleratorType} (capture is CUDA and WebGPU only)"
                : "disabled by the caller";
            return await _session.RunAsync(inputs).ConfigureAwait(false);
        }

        var shapes = inputs.Values.Select(t => t.Shape).ToArray();
        // ⚠️ Only enforced when a capture is actually LIVE. The shapes are baked into a recorded graph, so
        // they matter to a replay and to nothing else - and a wrapper that never captured must behave
        // exactly like the plain session it is standing in for.
        //
        // This previously recorded the shapes on the FIRST call and enforced them forever, including when
        // capture had been refused and every call was a direct forward. ZipVoice's decoder is shaped by the
        // utterance length, so the second thing it said - being a different length - threw
        // "input shape changed after capture" from a wrapper that had captured nothing.
        if (_capturedShapes != null && IsCaptured)
        {
            if (shapes.Length != _capturedShapes.Length
                || shapes.Where((s, i) => !s.AsSpan().SequenceEqual(_capturedShapes[i])).Any())
                throw new InvalidOperationException(
                    "SessionGraphCapture: input shapes changed after capture - use one instance per fixed shape set.");
        }

        if (!_captureAttempted)
        {
            _captureAttempted = true;

            // ⚠️ Control flow is refused HERE, for EVERY backend - not inside the CUDA path.
            //
            // If/Loop/Scan run their bodies through SubgraphRunner, which calls BuildExecutor on every
            // execution and allocates permanent buffers there. A device allocation inside a capture window
            // is fatal in a way no try/catch reaches: on CUDA a mid-capture cuMemAlloc is an uncatchable
            // 0xC0000005 (segfault, exit 139), and on WebGPU it HUNG THE GPU - DXGI_ERROR_DEVICE_HUNG, a
            // D3D12 TDR, reproducibly, which surfaces later as "device has been lost" on an innocent node.
            //
            // I first put this guard in CudaGraphCapture alone. That fixed CUDA and left WebGPU - the one
            // backend this project actually ships to - hanging the device instead. The eligibility decision
            // is made in this class, so the guard belongs in this class.
            bool refuseControlFlow = AllowControlFlow is bool allow ? !allow : RefuseControlFlow;
            if (refuseControlFlow && _session.OperatorTypes.Any(o => o is "If" or "Loop" or "Scan"))
            {
                var cf = string.Join(", ", _session.OperatorTypes.Where(o => o is "If" or "Loop" or "Scan"));
                CaptureStatus = $"refused: graph contains control flow ({cf}); "
                              + "set SessionGraphCapture.RefuseControlFlow = false to lift, after verifying "
                              + "the subgraph bodies do not allocate per call";
                Console.WriteLine($"[SessionGraphCapture] graph contains control flow ({cf}), whose subgraph "
                    + "executors allocate per call - a mid-capture allocation is unrecoverable on both CUDA "
                    + "and WebGPU; running direct forward.");
                return await _session.RunAsync(inputs).ConfigureAwait(false);
            }
            // The capture BINDS its input buffers into the recorded graph (CUDA graph nodes hold
            // device pointers; WebGPU plans hold GPUBuffer refs) - so the capture-pass inputs must
            // OUTLIVE the capture. Callers pass per-step transients (disposing one crashed cuMemFree
            // with 0xC0000005, SD-Turbo 2026-07-03), so this wrapper OWNS stable clones: copy the
            // caller's data in, capture against the clones; ReplayAsync copies every later call's
            // tensors into these same stable buffers.
            var stable = new Dictionary<string, Tensor>();
            foreach (var (name, t) in inputs)
            {
                var buf = _accelerator.Allocate1D<float>(Math.Max(1, t.ElementCount));
                _ownedInputs.Add(buf);
                if (t.ElementCount > 0)
                    await buf.View.SubView(0, t.ElementCount).CopyFromAsync(t.Data.SubView(0, t.ElementCount));
                stable[name] = new Tensor(buf.View.SubView(0, Math.Max(1, t.ElementCount)), (int[])t.Shape.Clone());
            }
            await _accelerator.SynchronizeAsync();
            try
            {
                if (_accelerator.AcceleratorType == AcceleratorType.Cuda)
                    _cuda = await CudaGraphCapture.TryCaptureAsync(_session, stable).ConfigureAwait(false);
                else
                    _webGpu = await WebGPUGraphCapture.TryCaptureAsync(_session, stable).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                // Capture is BEST-EFFORT: an over-VRAM model (pool reclaim is forbidden mid-capture)
                // or a capture-unsafe op must degrade to the direct forward, not fail generation.
                CaptureStatus = $"capture threw {ex.GetType().Name}: {ex.Message}";
                Console.WriteLine($"[SessionGraphCapture] capture failed - running direct: {ex.Message}");
            }

            // ⚠️ TryCapture returning NULL is the one outcome that prints nothing and throws nothing, so
            // without this line the loudest signal for the commonest silent failure is no signal at all.
            if (CaptureStatus == "not attempted")
                CaptureStatus = IsCaptured
                    ? $"live on {_accelerator.AcceleratorType} ({DispatchCount} dispatches)"
                    : $"TryCapture returned null on {_accelerator.AcceleratorType} "
                    + "(no exception, no message - the graph was ineligible to record)";
        }

        // Recorded only once a capture is live, so it describes a graph that actually exists.
        if (IsCaptured) _capturedShapes ??= shapes;
        if (_cuda != null) return await _cuda.ReplayAsync(inputs).ConfigureAwait(false);
        if (_webGpu != null) return await _webGpu.ReplayAsync(inputs).ConfigureAwait(false);
        return await _session.RunAsync(inputs).ConfigureAwait(false);   // capture unavailable - direct
    }

    public void Dispose()
    {
        _cuda?.Dispose(); _cuda = null;
        _webGpu?.Dispose(); _webGpu = null;
        foreach (var b in _ownedInputs) { try { b.Dispose(); } catch { } }
        _ownedInputs.Clear();
    }
}
