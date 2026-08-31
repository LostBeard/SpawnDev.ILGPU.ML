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

    public SessionGraphCapture(InferenceSession session, Accelerator accelerator)
    {
        _session = session;
        _accelerator = accelerator;
    }

    /// <summary>True once a capture is live (calls replay instead of a full graph walk).</summary>
    public bool IsCaptured => _cuda != null || _webGpu != null;

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
        if (!eligible) return await _session.RunAsync(inputs).ConfigureAwait(false);

        var shapes = inputs.Values.Select(t => t.Shape).ToArray();
        if (_capturedShapes != null)
        {
            if (shapes.Length != _capturedShapes.Length
                || shapes.Where((s, i) => !s.AsSpan().SequenceEqual(_capturedShapes[i])).Any())
                throw new InvalidOperationException(
                    "SessionGraphCapture: input shapes changed after capture - use one instance per fixed shape set.");
        }

        if (!_captureAttempted)
        {
            _captureAttempted = true;
            _capturedShapes = shapes;
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
                Console.WriteLine($"[SessionGraphCapture] capture failed - running direct: {ex.Message}");
            }
        }

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
