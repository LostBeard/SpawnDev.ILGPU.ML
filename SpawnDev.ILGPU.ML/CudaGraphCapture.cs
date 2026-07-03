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

    /// <summary>The input shapes this graph was captured for. A replay is only valid at the SAME shapes.</summary>
    public IReadOnlyDictionary<string, int[]> InputShapes { get; }

    private CudaGraphCapture(Accelerator accelerator, CudaStream stream, CudaGraphExec exec,
        Dictionary<string, Tensor> inputBuffers, Dictionary<string, Tensor> outputs,
        Dictionary<string, int[]> inputShapes)
    {
        _accelerator = accelerator;
        _stream = stream;
        _exec = exec;
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
    public static async Task<CudaGraphCapture?> TryCaptureAsync(InferenceSession session, Dictionary<string, Tensor> inputs)
    {
        var acc = session.Accelerator;
        if (acc is not CudaAccelerator) return null;
        if (!CudaStream.SupportsGraphCapture) return null;

        session.CacheShapeReadbacks = true;   // finalize a stable readback cache → the capture pass syncs nothing
        var capStream = (CudaStream)acc.CreateStream();
        Dictionary<string, Tensor> capOut;
        CudaGraph graph;
        FusedAttentionKernel.UseStableCaptureSlots = true;
        GraphExecutor.UseCaptureParamSlots = true;
        try
        {
            using (acc.WithDefaultStream(capStream))   // reroute *StreamKernel launches → capStream
            {
                // Warm A: JIT + populate stable attention/param slots + finalize the readback cache + snapshot
                // runtimeConstants for the capture pass to seed.
                await session.RunAsync(inputs);
                await acc.SynchronizeAsync();
                // Warm B (normal drains): over-provision the buffer pool to the deferred-release peak AND fully
                // return every size-bucket, so the capture pass finds a warm buffer in every bucket (no cuMemAlloc,
                // which is illegal mid-capture).
                await session.RunAsync(inputs);
                await acc.SynchronizeAsync();
                // Capture: record the forward. Drains suppressed → no periodic drain / final sync / buffer-return
                // aborts the capture; the seeded runtimeConstants keep eliding identical to warm.
                GraphExecutor.SuppressDrains = true;
                capStream.BeginCapture(CudaStreamCaptureMode.Global);
                capOut = await session.RunAsync(inputs);
                graph = capStream.EndCapture();
                GraphExecutor.SuppressDrains = false;
            }
        }
        finally
        {
            FusedAttentionKernel.UseStableCaptureSlots = false;
            GraphExecutor.UseCaptureParamSlots = false;
            GraphExecutor.SuppressDrains = false;
        }

        CudaGraphExec exec;
        using (graph) { exec = graph.Instantiate(); exec.Upload(capStream); }

        var shapes = new Dictionary<string, int[]>();
        foreach (var (name, t) in inputs) shapes[name] = (int[])t.Shape.Clone();
        return new CudaGraphCapture(acc, capStream, exec, inputs, capOut, shapes);
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
    }
}
