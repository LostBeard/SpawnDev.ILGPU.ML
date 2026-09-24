using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;
using System.Text;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// WHERE the large joint multi-view DAv3 forward spends its time. MEASURED 2026-09-23 (RTX 4070):
/// 6 views @518 replays in 6,243 ms on WebGPU vs 713 ms on CUDA and 598 ms in Transformers.js,
/// and the replay is SLOWER than the direct forward (3,407 ms) - while 4 views @504x378 replays in
/// 144 ms. Replay removes per-node orchestration, so what is left is GPU time; this names the nodes.
///
/// Per-op attribution needs <see cref="Graph.GraphExecutor.PerOpSync"/>: without it WebGPU kernel
/// time surfaces at the next sync point, on an innocent node. The per-op table is written to the
/// PMT sink as <c>_mldump/test-out/dav3/&lt;backend&gt;/profile-&lt;case&gt;.json</c>.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> DA3_MultiView_Profile() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.WebGPU or AcceleratorType.Cuda))
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: profile runs on WebGPU (the slow one) and CUDA (its control)");

        var (http, manifest) = await Da3Manifest();
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl(ModelHub.KnownModels.DepthAnythingV3Small, "onnx/model.onnx"));
        var extBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl(ModelHub.KnownModels.DepthAnythingV3Small, "onnx/model.onnx_data"));
        string backend = accelerator.AcceleratorType.ToString();
        var report = new StringBuilder();

        // The slow shape and a fast one: the difference between their tables is the defect.
        foreach (var name in new[] { "mv6_temple_518", "mv4_temple_off" })
        {
            var c = manifest.GetProperty("cases").GetProperty(name);
            var shape = Da3Ints(c.GetProperty("input_shape"));
            var input = await F32(http, $"test-refs/dav3/{name}.input.f32");
            using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
                inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = shape }, externalData: extBytes);
            using var inBuf = accelerator.Allocate1D(input);
            var feed = new Dictionary<string, Tensor> { [session.InputNames[0]] = new Tensor(inBuf.View, shape) };

            // Warm (kernel compile), then one timed direct forward WITHOUT per-op sync for the baseline.
            session.ReturnOutputs(await session.RunAsync(feed)); await accelerator.SynchronizeAsync();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            session.ReturnOutputs(await session.RunAsync(feed)); await accelerator.SynchronizeAsync();
            long directMs = sw.ElapsedMilliseconds;

            // Attributed forward.
            var timings = new Dictionary<string, double>();
            Graph.GraphExecutor.CapturedNodeTimingsMs = timings;
            Graph.GraphExecutor.PerOpSync = true;
            try
            {
                sw.Restart();
                session.ReturnOutputs(await session.RunAsync(feed)); await accelerator.SynchronizeAsync();
            }
            finally
            {
                Graph.GraphExecutor.CapturedNodeTimingsMs = null;
                Graph.GraphExecutor.PerOpSync = false;
            }
            long syncedMs = sw.ElapsedMilliseconds;

            // key = NNN_OpType_output
            var byOp = new Dictionary<string, (double ms, int n)>();
            foreach (var (k, ms) in timings)
            {
                var parts = k.Split('_', 3);
                string op = parts.Length > 1 ? parts[1] : k;
                byOp[op] = byOp.TryGetValue(op, out var a) ? (a.ms + ms, a.n + 1) : (ms, 1);
            }
            var opRows = byOp.OrderByDescending(kv => kv.Value.ms).Take(12).ToList();
            var topNodes = timings.OrderByDescending(kv => kv.Value).Take(15).ToList();

            // Replay split (WebGPU: encode / submit / GPU wait; CUDA: launch + sync).
            string replaySplit = "";
            if (accelerator.AcceleratorType == AcceleratorType.WebGPU)
            {
                using var cap = await WebGPUGraphCapture.TryCaptureAsync(session, feed);
                if (cap != null)
                {
                    cap.CollectTimings = true;
                    await cap.ReplayAsync(feed); await accelerator.SynchronizeAsync();
                    sw.Restart();
                    await cap.ReplayAsync(feed); await accelerator.SynchronizeAsync();
                    replaySplit = $"replay={sw.ElapsedMilliseconds}ms (dispatches={cap.DispatchCount} planCall={cap.LastPlanCallMs:F0} " +
                        $"jsEncode={cap.LastJsEncodeMs:F0} jsSubmit={cap.LastJsSubmitMs:F0} gpuWait={cap.LastSyncMs:F0})";
                }
                else replaySplit = "capture refused";
            }

            report.AppendLine($"[{name} {string.Join("x", shape)}] direct={directMs}ms perOpSynced={syncedMs}ms nodes={timings.Count} {replaySplit}");
            report.AppendLine("  by op: " + string.Join(", ", opRows.Select(r => $"{r.Key} {r.Value.ms:F0}ms/{r.Value.n}")));
            report.AppendLine("  top: " + string.Join(", ", topNodes.Select(t => $"{t.Key} {t.Value:F0}")));
            Console.WriteLine($"[DA3-ORT] profile {backend} {report}");

            try
            {
                var json = JsonSerializer.Serialize(new
                {
                    name, shape, directMs, syncedMs, replaySplit,
                    byOp = byOp.OrderByDescending(kv => kv.Value.ms).Select(kv => new { op = kv.Key, ms = kv.Value.ms, count = kv.Value.n }),
                    nodes = timings.OrderByDescending(kv => kv.Value).Select(kv => new { node = kv.Key, ms = kv.Value }),
                });
                using var resp = await http.PostAsync($"__pmt/out/dav3/{backend}/profile-{name}.json", new StringContent(json));
            }
            catch { /* no sink outside PMT */ }
        }
        // A measurement, not a gate: it reports, and a throw above is the only failure.
        return $"PASSED. {backend}\n{report}";
    });
}
