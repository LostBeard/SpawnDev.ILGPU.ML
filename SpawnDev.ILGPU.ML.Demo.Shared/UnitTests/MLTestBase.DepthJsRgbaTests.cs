using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// DepthEstimationPipeline's TypedArray and GPU-view entry points must match the managed <c>int[]</c> path.
/// </summary>
/// <remarks>
/// SpawnScene was forced to <c>Read&lt;int&gt;()</c> (and even GPU→JS→.NET round-trip for GpuImage) solely
/// because the pipeline only accepted <c>int[]</c>. These gates pin the zero-copy overloads: identical
/// packed RGBA uploaded three ways must produce the same raw depth map.
/// </remarks>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task Depth_JsAndGpuViewPaths_MatchManaged() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        // Small compiled resolution keeps the gate under a few seconds while still exercising
        // preprocess → forward → resize → min/max for every upload path.
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl("onnx-community/depth-anything-v2-small", "onnx/model.onnx"));
        using var session = InferenceSession.CreateFromOnnx(
            accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["pixel_values"] = new[] { 1, 3, 224, 224 }
            });
        using var pipeline = new DepthEstimationPipeline(session, accelerator);
        pipeline.EnableGraphCapture = false; // compare direct forwards only

        int w = 64, h = 64;
        var managed = new int[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                managed[y * w + x] = (int)(x * 255f / w)
                    | ((int)(y * 255f / h) << 8)
                    | (128 << 16)
                    | (unchecked((int)0xFF000000));

        // Warm once so compile/capture cost does not land on the first compared path.
        {
            var (warm, _, _, _, _) = await pipeline.EstimateGpuRawAsync(managed, w, h);
            warm.Dispose();
        }

        var (rawManaged, minM, maxM, outW, outH) = await pipeline.EstimateGpuRawAsync(managed, w, h);
        try
        {
            using var gpuViewBuf = accelerator.Allocate1D(managed);
            var (rawView, minV, maxV, outWV, outHV) = await pipeline.EstimateGpuRawAsync(
                gpuViewBuf.View, w, h);
            try
            {
                if (outW != outWV || outH != outHV)
                    throw new Exception(
                        $"{BackendName}: GPU-view path returned {outWV}x{outHV}, managed returned {outW}x{outH}");
                if (minM != minV || maxM != maxV)
                    throw new Exception(
                        $"{BackendName}: GPU-view min/max [{minV},{maxV}] != managed [{minM},{maxM}]");

                await AssertCloseGpu(accelerator, rawView.View, await rawManaged.CopyToHostAsync<float>(0, outW * outH),
                    0f, "Depth GPU-view vs managed: ");
            }
            finally
            {
                rawView.Dispose();
            }

            var js = SpawnJSRuntime.Instance;
            if (js == null || !js.IsBrowser)
            {
                Console.WriteLine($"[DepthJs] {BackendName}: GPU-view matched managed; TypedArray skipped (no JS heap)");
                return;
            }

            // Fixture only: seed a JS Uint8ClampedArray from the same packed bytes (outside the timed path).
            var bytes = new byte[managed.Length * 4];
            Buffer.BlockCopy(managed, 0, bytes, 0, bytes.Length);
            using var jsRgba = new Uint8ClampedArray(bytes.Length);
            jsRgba.Set(bytes);

            var (rawJs, minJ, maxJ, outWJ, outHJ) = await pipeline.EstimateGpuRawAsync(jsRgba, w, h);
            try
            {
                if (outW != outWJ || outH != outHJ)
                    throw new Exception(
                        $"{BackendName}: TypedArray path returned {outWJ}x{outHJ}, managed returned {outW}x{outH}");
                if (minJ != minM || maxJ != maxM)
                    throw new Exception(
                        $"{BackendName}: TypedArray min/max [{minJ},{maxJ}] != managed [{minM},{maxM}]");

                await AssertCloseGpu(accelerator, rawJs.View, await rawManaged.CopyToHostAsync<float>(0, outW * outH),
                    0f, "Depth TypedArray vs managed: ");
                Console.WriteLine(
                    $"[DepthJs] {BackendName}: TypedArray + GPU-view matched managed ({outW}x{outH}, min={minM:F4}, max={maxM:F4})");
            }
            finally
            {
                rawJs.Dispose();
            }
        }
        finally
        {
            rawManaged.Dispose();
        }
    });
}
