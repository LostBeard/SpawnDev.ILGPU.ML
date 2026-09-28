using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;
using System.Text;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// A warm DAv3 forward must allocate NOTHING: once the first passes have sized the pool, every Rent of a later
/// pass has to be served from a free bucket. SpawnScene's 83-pass TruckFull cascade measured 16 fresh buffers
/// per pass after warm-up (2026-09-27, <c>&amp;pooltrace=1</c>): six unnamed 1024-bucket rents and ten named
/// ones, never returned, ~1,300 buffers per run. Each pass is also checked against onnxruntime, so a fix that
/// recycles a buffer too early (the other way to get a flat count) fails on the values.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> DA3_WarmForward_AllocatesNothing() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: DAv3 runs on CUDA/OpenCL/WebGPU");

        var (http, manifest) = await Da3Manifest();
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl(ModelHub.KnownModels.DepthAnythingV3Small, "onnx/model.onnx"));
        var extBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl(ModelHub.KnownModels.DepthAnythingV3Small, "onnx/model.onnx_data"));

        var report = new StringBuilder();
        var failures = new List<string>();
        // Single view (SpawnScene's per-image depth) and a joint multi-view shape (its chunked cascade).
        foreach (var name in new[] { "s518_truck", "mv4_temple_off" })
        {
            var c = manifest.GetProperty("cases").GetProperty(name);
            var shape = Da3Ints(c.GetProperty("input_shape"));
            var input = await F32(http, $"test-refs/dav3/{name}.input.f32");
            var reference = await F32(http, $"test-refs/dav3/{name}.predicted_depth.f32");

            using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
                inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = shape },
                externalData: extBytes);
            using var inBuf = accelerator.Allocate1D(input);
            var feed = new Dictionary<string, Tensor> { [session.InputNames[0]] = new Tensor(inBuf.View, shape) };

            async Task<double> Pass()
            {
                var outs = await session.RunAsync(feed);
                await accelerator.SynchronizeAsync();
                var t = outs["predicted_depth"];
                var ours = await t.Data.SubView(0, t.ElementCount).CopyToHostAsync();
                session.ReturnOutputs(outs);
                var (rel, _, _) = Da3Compare(ours, reference);
                return rel;
            }

            // Warm: the first pass sizes the pool, the second settles any first-run-only paths (readback
            // caches, lazily built kernels' params).
            await Pass();
            await Pass();

            BufferPool.RecentFreshAllocNames.Clear();
            BufferPool.TraceFreshAllocNames = true;
            var line = new StringBuilder($"[{name} {string.Join("x", shape)}]");
            try
            {
                for (int p = 0; p < 3; p++)
                {
                    int before = BufferPool.TotalDeviceAllocations;
                    double rel = await Pass();
                    int fresh = BufferPool.TotalDeviceAllocations - before;
                    line.Append($" pass{p + 3}: fresh={fresh} rel={rel:E1};");
                    if (!(rel <= Da3DepthRelRmsGate))
                        failures.Add($"{name} pass {p + 3}: predicted_depth relRMS {rel:E2} > {Da3DepthRelRmsGate:E0}");
                    if (fresh != 0)
                        failures.Add($"{name} pass {p + 3}: {fresh} fresh device allocations");
                }
            }
            finally { BufferPool.TraceFreshAllocNames = false; }

            if (BufferPool.RecentFreshAllocNames.Count > 0)
            {
                // name#bucket@node:op, grouped, so a failure names the operators rather than a count.
                var groups = BufferPool.RecentFreshAllocNames.GroupBy(s => s)
                    .OrderByDescending(g => g.Count()).Select(g => $"{g.Key} x{g.Count()}");
                line.Append(" misses: ").Append(string.Join(", ", groups));
            }
            Console.WriteLine($"[DA3-POOL] {accelerator.AcceleratorType} {line}");
            report.AppendLine(line.ToString());
        }

        if (failures.Count > 0)
            throw new Exception($"DAv3 warm forward is not allocation-free ({failures.Count}):\n  " +
                                $"{string.Join("\n  ", failures)}\n{report}");
        return $"PASSED. {report}";
    });
}
