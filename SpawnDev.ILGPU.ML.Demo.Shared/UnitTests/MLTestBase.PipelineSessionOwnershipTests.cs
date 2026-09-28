using System.IO;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// A <see cref="DepthEstimationPipeline"/> that CREATED its session (the CreateFromStreamsAsync / CreateFromHubAsync
/// factories) must dispose it - and with it the model's GPU weights - when the pipeline is disposed. Before, nothing
/// could: SpawnScene unloaded DAv3 to make room for training and ~100 MB of weights in 335 buffers stayed on the
/// GPU (2026-09-27). A session handed to the constructor stays the caller's, as it always was.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 120000)]
    public async Task DepthPipeline_Dispose_FreesTheSessionItCreated() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");
        var onnxBytes = await http.GetByteArrayAsync("models/squeezenet/model.onnx");

        // Factory-created: the pipeline owns the session.
        var owned = await DepthEstimationPipeline.CreateFromStreamsAsync(accelerator, new MemoryStream(onnxBytes));
        var ownedSession = owned.Session;
        owned.Dispose();
        if (!ownedSession.IsDisposed)
            throw new Exception("disposing a factory-created pipeline left its session (and the model weights) alive");

        // Caller-supplied: the caller owns the session, and it must still work after the pipeline is gone.
        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes);
        var borrowed = new DepthEstimationPipeline(session, accelerator);
        borrowed.Dispose();
        if (session.IsDisposed)
            throw new Exception("disposing a pipeline disposed a session it did not create");
        Console.WriteLine($"[PipelineOwnership] owned session freed; borrowed session intact ({session.NodeCount} nodes)");
    });
}
