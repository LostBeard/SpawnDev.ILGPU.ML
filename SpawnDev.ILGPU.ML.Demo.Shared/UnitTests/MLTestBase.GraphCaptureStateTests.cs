using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// A graph capture must not change what the session computes afterwards - especially when it FAILS.
/// </summary>
/// <remarks>
/// <para>
/// ⚠️ This exists because of a bug that produced no error at all. <c>CudaGraphCapture.TryCaptureAsync</c>
/// sets <c>session.CacheShapeReadbacks = true</c> and, before this test, never restored it. That cache
/// decides which captured tensors are safe to reuse by probing two runs and keeping the values that MATCH
/// across them - which its own documentation calls "correct by construction", and is, GIVEN two runs with
/// different data. The capture path runs its two warm passes with the SAME inputs, because that is what
/// capturing a fixed-shape graph means. So every readback compared equal, every one was cached as "stable"
/// including the data-derived ones, and the flag stayed on afterwards.
/// </para>
/// <para>
/// The consequence landed on the FALLBACK path - the one whose entire purpose is to degrade safely. A
/// capture-incompatible graph (ZipVoice's decoder: a <c>GreaterOrEqual</c> syncs the stream mid-capture,
/// which CUDA forbids) failed, fell through to the direct forward, and then rendered at rms 0.0021 instead
/// of 0.0761. Quietly wrong audio, not absent audio, from a code path that reported it was running direct.
/// </para>
/// <para>
/// So the assertion here is deliberately end-to-end rather than a flag check: run the session, put it
/// through a capture attempt, run it again, and require BIT-IDENTICAL output. That catches any state a
/// capture leaves behind, not only the one flag that caused this.
/// </para>
/// </remarks>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 600000)]
    public async Task GraphCapture_LeavesTheSessionComputingTheSameThing() => await RunTest(async accelerator =>
    {
        // Capture is a CUDA/WebGPU feature and the corruption was CUDA-specific. Elsewhere
        // SessionGraphCapture is a documented pass-through, so there is nothing here to prove.
        if (accelerator.AcceleratorType != AcceleratorType.Cuda)
            throw new UnsupportedTestException(
                $"graph capture state is a CUDA concern; this lane is {accelerator.AcceleratorType}");

        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        using var session = await InferenceSession.CreateAsync(accelerator, http, "models/squeezenet");
        bool flagBefore = session.CacheShapeReadbacks;

        // A fixed, arbitrary input. The VALUES do not matter - only that the same input gives the same
        // answer before and after a capture attempt.
        var name = session.InputNames[0];
        var shape = new[] { 1, 3, 224, 224 };
        int count = shape.Aggregate(1, (a, b) => a * b);
        var data = new float[count];
        for (int i = 0; i < count; i++) data[i] = MathF.Sin(i * 0.0013f);

        using var inBuf = accelerator.Allocate1D(data);
        Dictionary<string, Tensor> MakeInputs() =>
            new() { [name] = new Tensor(inBuf.View, shape) };

        var before = await session.RunAsync(MakeInputs());
        var beforeVals = await ReadFirstOutputAsync(accelerator, session, before);

        // The capture attempt itself. Success and failure are BOTH acceptable outcomes here - a graph that
        // cannot be captured is a fact about the graph, not a defect. What is never acceptable is the
        // session computing something different afterwards.
        var cap = await CudaGraphCapture.TryCaptureAsync(session, MakeInputs());
        bool captured = cap != null;
        // Disposed before re-running directly: a live capture owns the session's output buffers, and the
        // class documents that interleaving a direct RunAsync would recycle them.
        cap?.Dispose();

        if (session.CacheShapeReadbacks != flagBefore)
            throw new Exception(
                $"CudaGraphCapture left session.CacheShapeReadbacks = {session.CacheShapeReadbacks} "
              + $"(was {flagBefore}). That cache is finalised from the capture's two IDENTICAL warm passes, "
              + "so it marks data-derived readbacks as 'stable' and freezes them into every later run.");

        var after = await session.RunAsync(MakeInputs());
        var afterVals = await ReadFirstOutputAsync(accelerator, session, after);

        if (beforeVals.Length != afterVals.Length)
            throw new Exception($"output length changed across a capture attempt: "
                              + $"{beforeVals.Length} -> {afterVals.Length}");
        for (int i = 0; i < beforeVals.Length; i++)
            if (beforeVals[i] != afterVals[i])
                throw new Exception(
                    $"the SAME input produced a different result after a capture attempt "
                  + $"(capture {(captured ? "succeeded" : "failed and fell through")}): index {i} "
                  + $"{beforeVals[i]} -> {afterVals[i]}. A capture must leave the session computing exactly "
                  + "what it computed before - a fallback that returns different numbers is worse than one "
                  + "that throws, because nothing reports it.");

        Console.WriteLine($"[GraphCapture] capture {(captured ? "succeeded" : "failed (fell through)")}; "
                        + $"{beforeVals.Length} outputs bit-identical across the attempt");
    });

    /// <summary>Read the session's first graph output back to the host.</summary>
    private static async Task<float[]> ReadFirstOutputAsync(
        Accelerator accelerator, InferenceSession session, Dictionary<string, Tensor> outputs)
    {
        await accelerator.SynchronizeAsync();
        var t = outputs[session.OutputNames[0]];
        var host = new float[t.ElementCount];
        t.Data.SubView(0, t.ElementCount).CopyToCPU(host);
        return host;
    }
}
