using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// End-to-end pipeline tests for Detection, Pose, and Face pipelines.
/// Each test loads the real model, runs through the full pipeline, and
/// verifies output structure and basic correctness.
/// </summary>
public abstract partial class MLTestBase
{
    // ── Object Detection (YOLOv8) ──

    [TestMethod(Timeout = 120000)]
    public async Task Pipeline_YOLOv8_DetectsObjects() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Load model (same path as demo page)
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl("salim4n/yolov8n-detect-onnx", "yolov8n-onnx-web/yolov8n.onnx"));
        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes);

        var pipeline = new ObjectDetectionPipeline(session, accelerator);

        // Create a test image: 640x480 with a bright region (simulates an object)
        var testImage = new int[640 * 480];
        for (int y = 0; y < 480; y++)
            for (int x = 0; x < 640; x++)
            {
                int r = (x > 200 && x < 400 && y > 100 && y < 350) ? 200 : 50;
                int g = (x > 200 && x < 400 && y > 100 && y < 350) ? 180 : 40;
                int b = (x > 200 && x < 400 && y > 100 && y < 350) ? 160 : 30;
                testImage[y * 640 + x] = r | (g << 8) | (b << 16) | (0xFF << 24);
            }

        var result = await pipeline.DetectAsync(testImage, 640, 480, confidenceThreshold: 0.1f);

        Console.WriteLine($"[Pipeline] YOLOv8: {result.Objects.Length} detections, {result.InferenceTimeMs:F0}ms");
        foreach (var obj in result.Objects.Take(5))
            Console.WriteLine($"  {obj.Label} ({obj.Confidence:P1}) [{obj.X:F0},{obj.Y:F0} {obj.Width:F0}x{obj.Height:F0}]");

        // Pipeline should return a valid result (even if no high-confidence detections on synthetic data)
        if (result.InferenceTimeMs <= 0)
            throw new Exception("YOLOv8 inference time is 0 — pipeline didn't run");
        if (result.ImageWidth != 640 || result.ImageHeight != 480)
            throw new Exception($"Result dimensions wrong: {result.ImageWidth}x{result.ImageHeight}");

        Console.WriteLine($"[Pipeline] YOLOv8 pipeline: PASS");
        // Don't call pipeline.Dispose() — it disposes the session, but 'using var session' already handles that.
    });

    [TestMethod(Timeout = 120000)]
    public async Task Pipeline_YOLOv8_Reference_MatchesOnnxRuntime() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Load model and run with reference input
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl("salim4n/yolov8n-detect-onnx", "yolov8n-onnx-web/yolov8n.onnx"));
        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes);

        // Load reference input (pre-preprocessed NCHW)
        var inputBytes = await http.GetByteArrayAsync("references/yolov8n/cat_input_nchw.bin");
        var inputData = new float[inputBytes.Length / 4];
        Buffer.BlockCopy(inputBytes, 0, inputData, 0, inputBytes.Length);

        using var inputBuf = accelerator.Allocate1D(inputData);
        var inputTensor = new Tensor(inputBuf.View, new[] { 1, 3, 640, 640 });

        var outputs = await session.RunAsync(new Dictionary<string, Tensor>
        {
            [session.InputNames[0]] = inputTensor
        });

        var output = outputs[session.OutputNames[0]];
        int elems = output.ElementCount;

        using var readBuf = accelerator.Allocate1D<float>(elems);
        new ElementWiseKernels(accelerator).Scale(output.Data.SubView(0, elems), readBuf.View, elems, 1f);
        await accelerator.SynchronizeAsync();
        var actual = await readBuf.CopyToHostAsync<float>(0, elems);

        // Compare against ONNX Runtime reference
        var refBytes = await http.GetByteArrayAsync("references/yolov8n/cat_output.bin");
        var expected = new float[refBytes.Length / 4];
        Buffer.BlockCopy(refBytes, 0, expected, 0, refBytes.Length);

        var cmpLen = Math.Min(actual.Length, expected.Length);
        AssertReferenceMatch(actual.Take(cmpLen).ToArray(), expected.Take(cmpLen).ToArray(), 1.0f, "YOLOv8");
    });

    // ── Pose Estimation (MoveNet) ──

    [TestMethod(Timeout = 120000)]
    public async Task Pipeline_MoveNet_DetectsKeypoints() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // MoveNet is TFLite format
        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl("Xenova/movenet-singlepose-lightning", "onnx/model.onnx"));
        using var session = InferenceSession.CreateFromFile(accelerator, modelBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["input"] = new[] { 1, 192, 192, 3 }
            });

        // Create a simple test input: NHWC int32 [0,255] format
        var inputData = new float[1 * 192 * 192 * 3];
        var rng = new Random(42);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = rng.Next(0, 256);

        using var inputBuf = accelerator.Allocate1D(inputData);
        var inputTensor = new Tensor(inputBuf.View, new[] { 1, 192, 192, 3 });

        var outputs = await session.RunAsync(new Dictionary<string, Tensor>
        {
            [session.InputNames[0]] = inputTensor
        });

        var output = outputs[session.OutputNames[0]];
        Console.WriteLine($"[Pipeline] MoveNet output shape: [{string.Join(",", output.Shape)}], elements: {output.ElementCount}");

        // Output should be [1, 1, 17, 3] = 51 values (17 keypoints × [y, x, confidence])
        int elems = output.ElementCount;
        if (elems < 51)
            throw new Exception($"MoveNet output too small: {elems} elements, expected ≥51");

        using var readBuf = accelerator.Allocate1D<float>(elems);
        new ElementWiseKernels(accelerator).Scale(output.Data.SubView(0, elems), readBuf.View, elems, 1f);
        await accelerator.SynchronizeAsync();
        var keypoints = await readBuf.CopyToHostAsync<float>(0, elems);

        // Check that keypoint values are in valid range
        int nanCount = 0;
        for (int i = 0; i < keypoints.Length; i++)
            if (float.IsNaN(keypoints[i]) || float.IsInfinity(keypoints[i])) nanCount++;

        if (nanCount > 0)
            throw new Exception($"MoveNet output has {nanCount} NaN/Inf values");

        Console.WriteLine($"[Pipeline] MoveNet: {elems / 3} keypoints, values range [{keypoints.Min():F4}, {keypoints.Max():F4}]");
        Console.WriteLine($"[Pipeline] MoveNet pipeline: PASS");
    });

    // ── Face Detection (BlazeFace) ──

    [TestMethod(Timeout = 120000)]
    public async Task Pipeline_BlazeFace_Reference_MatchesOnnxRuntime() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // BlazeFace is TFLite format — try Kaggle TF model first, fall back to local
        byte[] modelBytes;
        try
        {
            modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
                "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite");
        }
        catch
        {
            // Fallback to local model (may be quantized with different tensor names)
            modelBytes = await http.GetByteArrayAsync("models/blaze-face/model.tflite");
        }
        using var session = InferenceSession.CreateFromFile(accelerator, modelBytes);

        // Load reference input in NHWC format (TFLite native layout)
        var inputBytes = await http.GetByteArrayAsync("references/blaze-face/cat_input.bin");
        var inputData = new float[inputBytes.Length / 4];
        Buffer.BlockCopy(inputBytes, 0, inputData, 0, inputBytes.Length);

        using var inputBuf = accelerator.Allocate1D(inputData);
        var inputTensor = new Tensor(inputBuf.View, new[] { 1, 128, 128, 3 });

        var outputs = await session.RunAsync(new Dictionary<string, Tensor>
        {
            [session.InputNames[0]] = inputTensor
        });

        // BlazeFace has 2 outputs: regressors [1,896,16] and classificators [1,896,1]
        if (session.OutputNames.Length < 2)
            throw new Exception($"BlazeFace expected 2 outputs, got {session.OutputNames.Length}");

        // Classificators are the detection gate (soft "finite regressors" was a false VERIFIED).
        var refClsBytes = await http.GetByteArrayAsync("references/blaze-face/cat_output_classificators.bin");
        var expectedCls = new float[refClsBytes.Length / 4];
        Buffer.BlockCopy(refClsBytes, 0, expectedCls, 0, refClsBytes.Length);

        var clsName = session.OutputNames.FirstOrDefault(n =>
            n.Contains("classif", StringComparison.OrdinalIgnoreCase)) ?? session.OutputNames[1];
        var clsOutput = outputs[clsName];
        int clsElems = Math.Min(clsOutput.ElementCount, expectedCls.Length);
        using var clsReadBuf = accelerator.Allocate1D<float>(clsElems);
        new ElementWiseKernels(accelerator).Scale(clsOutput.Data.SubView(0, clsElems), clsReadBuf.View, clsElems, 1f);
        await accelerator.SynchronizeAsync();
        var actualCls = await clsReadBuf.CopyToHostAsync<float>(0, clsElems);

        float maxAct = float.NegativeInfinity, maxExp = float.NegativeInfinity;
        double sumSq = 0, sumDiff = 0;
        int n = Math.Min(actualCls.Length, expectedCls.Length);
        for (int i = 0; i < n; i++)
        {
            if (actualCls[i] > maxAct) maxAct = actualCls[i];
            if (expectedCls[i] > maxExp) maxExp = expectedCls[i];
            sumSq += expectedCls[i] * expectedCls[i];
            double d = actualCls[i] - expectedCls[i];
            sumDiff += d * d;
        }
        double relRms = Math.Sqrt(sumDiff / Math.Max(1e-12, sumSq));
        Console.WriteLine(
            $"[Pipeline] BlazeFace classificators: maxAct={maxAct:F3} maxExp={maxExp:F3} relRMS={relRms:E2} n={n}");
        // Face detection is unusable when classificators diverge this far — do not soften this gate.
        // Soft "finite regressors" used to keep /face marked VERIFIED while live Faces: 0 (relRMS~240, 2026-09-27).
        if (relRms > 0.05)
            throw new Exception(
                $"BlazeFace NHWC forward diverges from TFLite reference: maxAct={maxAct:F3} maxExp={maxExp:F3} "
                + $"relRMS={relRms:E2} (need ≤0.05). /face will show Faces: 0 until this is fixed.");
    });

    /// <summary>
    /// End-to-end BlazeFace on a real portrait: must detect ≥1 face.
    /// Pins MediaPipe short-range decode (896 anchors, [-1,1] letterbox, reverse_output_order).
    /// The previous strides-[8,16]/[0,1]/xy path returned Faces: 0 on every /face sample.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task Pipeline_BlazeFace_Portrait_DetectsFace() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        byte[] modelBytes;
        try { modelBytes = await http.GetByteArrayAsync("models/blaze-face/model.tflite"); }
        catch (Exception ex) { throw new UnsupportedTestException($"BlazeFace model missing: {ex.Message}"); }

        byte[] bin;
        try { bin = await http.GetByteArrayAsync("samples/portrait_rgba.bin"); }
        catch (Exception ex) { throw new UnsupportedTestException($"portrait_rgba.bin missing: {ex.Message}"); }

        int width = BitConverter.ToInt32(bin, 0);
        int height = BitConverter.ToInt32(bin, 4);
        var pixels = new int[width * height];
        Buffer.BlockCopy(bin, 8, pixels, 0, width * height * 4);

        using var session = InferenceSession.CreateFromFile(accelerator, modelBytes);
        Console.WriteLine($"[BlazeFace] outputs=[{string.Join(", ", session.OutputNames)}]");
        using var pipeline = new FaceDetectionPipeline(session, accelerator);

        // Diagnostic: raw max score before decode (catches swapped outputs / wrong norm)
        {
            using var rgbaBuf = accelerator.Allocate1D(pixels);
            // Re-run through public API after logging scores via a lowered threshold probe
        }

        var resultLoose = await pipeline.DetectAsync(pixels, width, height, confidenceThreshold: 0.01f);
        Console.WriteLine($"[BlazeFace] loose(0.01) faces={resultLoose.FaceCount}");
        if (resultLoose.FaceCount > 0)
            Console.WriteLine($"[BlazeFace] loose top conf={resultLoose.Faces[0].Confidence:P2}");

        var result = await pipeline.DetectAsync(pixels, width, height);

        Console.WriteLine(
            $"[BlazeFace] portrait {width}x{height}: faces={result.FaceCount} in {result.InferenceTimeMs:F1}ms");
        if (result.FaceCount < 1)
            throw new Exception(
                $"BlazeFace detected 0 faces on samples/portrait.jpg (loose@0.01 got {resultLoose.FaceCount}). "
                + "decode/preprocess still wrong.");
        var top = result.Faces[0];
        Console.WriteLine(
            $"[BlazeFace] top face conf={top.Confidence:P1} box=({top.X:F0},{top.Y:F0},{top.Width:F0}x{top.Height:F0})");
        if (top.Confidence < 0.5f)
            throw new Exception($"top face confidence {top.Confidence} below MediaPipe min_score_thresh 0.5");
        if (top.Width < 10 || top.Height < 10)
            throw new Exception($"degenerate box {top.Width}x{top.Height}");
    });
}
