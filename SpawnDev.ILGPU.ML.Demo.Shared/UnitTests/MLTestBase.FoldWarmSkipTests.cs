using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// <see cref="Graph.GraphExecutor.FoldWarmSkipFolded"/>: a warm folded forward visits only the non-folded nodes plus the
/// folded ones that release a varying input, instead of walking every folded node (DAv3: 1,678 of 2,524). The skip must
/// change NOTHING observable - the same outputs bit for bit, and the same buffers back in the pool.
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// models/tests/fold_shape_last_consumer.onnx (tools/gen_fold_shape_last_consumer.py): v = Relu(x) is consumed ONLY
    /// by s = Shape(v), which folds and is therefore v's last consumer. If the warm skip dropped that folded node, v
    /// would never go back to the pool. Skip off vs on: y and z bit-identical, the pool's free bytes identical, and the
    /// folded Shape node really visited (else the model no longer exercises the case).
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task FoldWarmSkip_ReleasesAndOutputsUnchanged() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available for this backend");
        var bytes = await http.GetByteArrayAsync("models/tests/fold_shape_last_consumer.onnx");
        var x = RandomFloats(1 * 4 * 8 * 8, seed: 91, scale: 2f);   // negatives included: Relu is not the identity
        using var xB = accelerator.Allocate1D(x);
        var inputs = new Dictionary<string, Tensor> { ["x"] = new Tensor(xB.View, new[] { 1, 4, 8, 8 }) };

        async Task<(float[] Y, float[] Z, long FreeBytes, int Folded, int Visited, string State)> Run(bool skip)
        {
            // A fresh session per arm: identical pool history, so free-bucket bytes are comparable.
            using var session = InferenceSession.CreateFromOnnx(accelerator, bytes);
            Graph.GraphExecutor.FoldWarmSkipFolded = skip;
            Dictionary<string, Tensor> outs = null!;
            for (int run = 0; run < 3; run++)   // record the fold, then warm forwards
            {
                if (run > 0) session.ReturnOutputs(outs);
                outs = await session.RunAsync(inputs);
            }
            int folded = Graph.GraphExecutor.LastRunFoldedNodes, visited = Graph.GraphExecutor.LastRunFoldedVisited;
            string state = Graph.GraphExecutor.LastRunFoldState;
            await accelerator.SynchronizeAsync();
            var y = await Read(outs["y"]);
            var z = await Read(outs["z"]);
            session.ReturnOutputs(outs);
            return (y, z, session.PooledFreeBytes, folded, visited, state);
        }
        async Task<float[]> Read(Tensor t)
        {
            using var h = accelerator.Allocate1D<float>(t.ElementCount);
            await h.View.CopyFromAsync(t.Data.SubView(0, t.ElementCount));
            await accelerator.SynchronizeAsync();
            return await h.CopyToHostAsync<float>(0, t.ElementCount);
        }

        try
        {
            var off = await Run(false);
            var on = await Run(true);
            if (!on.State.StartsWith("warm"))
                throw new Exception($"{BackendName}: the third forward is not a warm folded one ({on.State}) - nothing was skipped");
            if (on.Folded != off.Folded)
                throw new Exception($"{BackendName}: folded count {on.Folded} with the skip, {off.Folded} without");
            for (int i = 0; i < x.Length; i++)
            {
                if (BitConverter.SingleToInt32Bits(on.Y[i]) != BitConverter.SingleToInt32Bits(off.Y[i]))
                    throw new Exception($"{BackendName}: y[{i}] = {on.Y[i]} with the skip, {off.Y[i]} without");
                if (BitConverter.SingleToInt32Bits(on.Z[i]) != BitConverter.SingleToInt32Bits(off.Z[i]))
                    throw new Exception($"{BackendName}: z[{i}] = {on.Z[i]} with the skip, {off.Z[i]} without");
            }
            if (on.FreeBytes != off.FreeBytes)
                throw new Exception($"{BackendName}: pool free bytes {on.FreeBytes} with the skip, {off.FreeBytes} without - a buffer was not released");
            // Last: a model that no longer puts a folded node last on a varying input would pass the checks above vacuously.
            if (on.Visited < 1)
                throw new Exception($"{BackendName}: no folded node was visited ({on.Folded} folded) - the model no longer puts a folded node last on a varying input");
            Console.WriteLine($"[FoldWarmSkip] {BackendName}: {on.State}, folded {on.Folded} (visited {on.Visited}), pool free {on.FreeBytes} B - identical");
        }
        finally
        {
            Graph.GraphExecutor.FoldWarmSkipFolded = true;
        }
    });

    /// <summary>
    /// The same A/B on a real graph: Depth Anything V2 Small (224) - folded forwards with the skip off and on must give
    /// bit-identical depth. Gated in the profile test for DAv3 on WebGPU too (see DirectForwardProfile).
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task FoldWarmSkip_DAv2_BitIdentical() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available for this backend");
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl("onnx-community/depth-anything-v2-small", "onnx/model.onnx"));
        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 3, 224, 224 } });
        using var pipeline = new Pipelines.DepthEstimationPipeline(session, accelerator);
        pipeline.EnableGraphCapture = false;
        int w = 96, h = 64;
        var rgba = new int[w * h];
        var rng = new Random(5);
        for (int i = 0; i < rgba.Length; i++) rgba[i] = rng.Next(0, 0x1000000) | unchecked((int)0xFF000000);

        async Task<(float[] Depth, int Folded, int Visited, string State)> Run(bool skip)
        {
            Graph.GraphExecutor.FoldWarmSkipFolded = skip;
            var r = await pipeline.EstimateGpuRawAsync(rgba, w, h, w, h);
            int folded = Graph.GraphExecutor.LastRunFoldedNodes, visited = Graph.GraphExecutor.LastRunFoldedVisited;
            string state = Graph.GraphExecutor.LastRunFoldState;
            using (r.RawDepth) return (await r.RawDepth.CopyToHostAsync<float>(0, w * h), folded, visited, state);
        }

        try
        {
            await Run(true);   // records the fold
            var off = await Run(false);
            var on = await Run(true);
            if (!on.State.StartsWith("warm") || on.Folded == 0)
                throw new Exception($"{BackendName}: not a warm folded forward ({on.State}, folded {on.Folded}) - the skip was not exercised");
            if (on.Folded != off.Folded)
                throw new Exception($"{BackendName}: folded count {on.Folded} with the skip, {off.Folded} without");
            for (int i = 0; i < on.Depth.Length; i++)
                if (BitConverter.SingleToInt32Bits(on.Depth[i]) != BitConverter.SingleToInt32Bits(off.Depth[i]))
                    throw new Exception($"{BackendName}: depth[{i}] = {on.Depth[i]} with the skip, {off.Depth[i]} without");
            Console.WriteLine($"[FoldWarmSkip] {BackendName} DAv2: {on.State}, folded {on.Folded} (visited {on.Visited}) - bit-identical");
        }
        finally
        {
            Graph.GraphExecutor.FoldWarmSkipFolded = true;
        }
    });
}
