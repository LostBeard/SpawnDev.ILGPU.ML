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
/// dominant cost of the DAv3 77s-vs-ORT-Web-73ms gap (measured: BlazorJS interop is 14-64us/call and each
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

    /// <summary>The stable output tensors the captured plan writes (also returned by ReplayAsync).</summary>
    public IReadOnlyDictionary<string, Tensor> Outputs => _outputs;

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
        GraphCompiler.ShapeSubgraphFoldEnabled = true;
        // NOTE: dispatch-elide stays OFF here (unlike the CUDA capture): elide has a WebGPU-specific
        // gap (an elided shape value consumed by a GPU-tensor reader is missing from runtimeConstants
        // -> "Tensor not found"; tracked, executor lane). The plan does not need elide - the shape-op
        // dispatches are simply recorded as extra tiny entries, and a REPLAY never needs the shape
        // VALUES at all (they only steer .NET-side orchestration during the capture pass itself).
        GraphExecutor.ShapeInterpElideDispatch = false;
        GraphExecutor.ShapeInterpValidate = false;
        WebGPUBackend.EnableBindGroupCaching = false;
        FusedAttentionKernel.UseStableCaptureSlots = true;
        GraphExecutor.UseCaptureParamSlots = true;
        try
        {
            // Warm A: shader JIT + populate stable attention/param slots + finalize the readback cache +
            // snapshot runtimeConstants for the capture pass to seed.
            await session.RunAsync(inputs);
            await acc.SynchronizeAsync();
            // Warm B (normal drains): over-provision the buffer pool to the deferred-release peak AND fully
            // return every size-bucket, so the capture pass finds a warm buffer in every bucket.
            await session.RunAsync(inputs);
            await acc.SynchronizeAsync();

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
        await _plan.ReplayAsync();
        await _accelerator.SynchronizeAsync();
        return _outputs;
    }

    public void Dispose()
    {
        try { _plan.Dispose(); } catch { }
    }
}
