using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;
using System.Text;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Where does large joint multi-view fall off the cliff, and does the cliff track memory?
/// MEASURED 2026-09-23 (RTX 4070, WebGPU): a captured replay is 147 ms of GPU at 4 views x 504x378 and
/// 6,706 ms at 6 views x 518 - 45x for ~2.1x the pixels and tokens - while CUDA scales ~2x. The replay's
/// encode + submit is ~5 ms, so all of it is GPU execution. Views 1..6 of the same TempleRing tensor at
/// 518, a fresh session per N (a cached per-shape executor keeps its own pool, which would inflate every
/// later N's footprint): direct time, replay GPU time, and the pool's peak bytes for each.
/// </summary>
public abstract partial class MLTestBase
{
    // The regime DepthEstimationPipeline records DAv3 with (see WebGPUGraphCapture.KeepDrainsDuringCapture).
    // Flip to false to reproduce the pin-everything capture and its cliff.
    const bool SweepKeepDrains = true;

    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task<string> DA3_MultiView_ScalingSweep() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.WebGPU or AcceleratorType.Cuda))
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: sweep runs on WebGPU and its CUDA control");

        var (http, manifest) = await Da3Manifest();
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl(ModelHub.KnownModels.DepthAnythingV3Small, "onnx/model.onnx"));
        var extBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl(ModelHub.KnownModels.DepthAnythingV3Small, "onnx/model.onnx_data"));
        var all = await F32(http, "test-refs/dav3/mv6_temple_518.input.f32");   // [1,6,3,518,518]
        const int S = 518; int chw = 3 * S * S;
        string backend = accelerator.AcceleratorType.ToString();
        var rows = new List<object>();
        var report = new StringBuilder($"{backend} N | direct ms | replay ms (gpuWait) | peak pool MB (direct / +capture)\n");

        bool trackWas = BufferPool.TrackPeaks;
        BufferPool.TrackPeaks = true;
        try
        {
            for (int n = 1; n <= 6; n++)
            {
                var shape = new[] { 1, n, 3, S, S };
                using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
                    inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = shape }, externalData: extBytes);
                using var inBuf = accelerator.Allocate1D(all.AsSpan(0, n * chw).ToArray());
                var feed = new Dictionary<string, Tensor> { [session.InputNames[0]] = new Tensor(inBuf.View, shape) };

                BufferPool.ResetPeaks();
                session.ReturnOutputs(await session.RunAsync(feed)); await accelerator.SynchronizeAsync();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                session.ReturnOutputs(await session.RunAsync(feed)); await accelerator.SynchronizeAsync();
                long directMs = sw.ElapsedMilliseconds;
                long peakDirect = BufferPool.PeakTotalBytes;

                long replayMs = -1; double gpuWait = -1; long peakCapture = -1;
                if (accelerator.AcceleratorType == AcceleratorType.WebGPU)
                {
                    using var cap = await WebGPUGraphCapture.TryCaptureAsync(session, feed, keepDrains: SweepKeepDrains);
                    if (cap != null)
                    {
                        peakCapture = BufferPool.PeakTotalBytes;
                        await cap.ReplayAsync(feed); await accelerator.SynchronizeAsync();
                        var rt = new List<(long, double)>();
                        for (int i = 0; i < 3; i++)
                        {
                            sw.Restart();
                            await cap.ReplayAsync(feed); await accelerator.SynchronizeAsync();
                            rt.Add((sw.ElapsedMilliseconds, cap.LastSyncMs));
                        }
                        rt.Sort();
                        (replayMs, gpuWait) = rt[1];
                    }
                }
                else
                {
                    using var cg = await CudaGraphCapture.TryCaptureAsync(session, feed);
                    if (cg != null)
                    {
                        peakCapture = BufferPool.PeakTotalBytes;
                        await cg.ReplayAsync(feed); await accelerator.SynchronizeAsync();
                        var rt = new List<long>();
                        for (int i = 0; i < 3; i++)
                        {
                            sw.Restart();
                            await cg.ReplayAsync(feed); await accelerator.SynchronizeAsync();
                            rt.Add(sw.ElapsedMilliseconds);
                        }
                        rt.Sort();
                        replayMs = rt[1];
                    }
                }
                report.AppendLine($"  {n} | {directMs} | {replayMs} ({gpuWait:F0}) | {peakDirect / 1048576} / {peakCapture / 1048576}"
                    + (accelerator.AcceleratorType == AcceleratorType.WebGPU
                        ? $" | keepDrains={SweepKeepDrains}" : ""));
                rows.Add(new { n, directMs, replayMs, gpuWait, peakDirectBytes = peakDirect, peakCaptureBytes = peakCapture });
                Console.WriteLine($"[DA3-ORT] sweep {backend} N={n} direct={directMs} replay={replayMs} gpuWait={gpuWait:F0} peakMB={peakDirect / 1048576}/{peakCapture / 1048576}"
                    + (accelerator.AcceleratorType == AcceleratorType.WebGPU
                        ? $" keepDrains={SweepKeepDrains}" : ""));
            }
        }
        finally { BufferPool.TrackPeaks = trackWas; }

        try
        {
            using var resp = await http.PostAsync($"__pmt/out/dav3/{backend}/sweep-518.json", new StringContent(JsonSerializer.Serialize(rows)));
        }
        catch { /* no sink outside PMT */ }
        return $"PASSED. {report}";
    });
}
