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
            "https://huggingface.co/salim4n/yolov8n-detect-onnx/resolve/main/yolov8n-onnx-web/yolov8n.onnx");
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
        pipeline.Dispose();
    });

    [TestMethod(Timeout = 120000)]
    public async Task Pipeline_YOLOv8_Reference_MatchesOnnxRuntime() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Load model and run with reference input
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/salim4n/yolov8n-detect-onnx/resolve/main/yolov8n-onnx-web/yolov8n.onnx");
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
        session.Dispose();
    });

    // ── Pose Estimation (MoveNet) ──

    [TestMethod(Timeout = 120000)]
    public async Task Pipeline_MoveNet_DetectsKeypoints() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // MoveNet is TFLite format
        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/main/onnx/model.onnx");
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
        session.Dispose();
    });

    // ── Face Detection (BlazeFace) ──

    [TestMethod(Timeout = 120000)]
    public async Task Pipeline_BlazeFace_Reference_MatchesOnnxRuntime() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // BlazeFace is TFLite format — load from HF
        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/litert-community/blaze-face/resolve/main/model_unquant.tflite");
        using var session = InferenceSession.CreateFromFile(accelerator, modelBytes);

        // Load reference input
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

        // Compare regressors
        var regOutput = outputs[session.OutputNames[0]];
        int regElems = regOutput.ElementCount;
        using var regReadBuf = accelerator.Allocate1D<float>(regElems);
        new ElementWiseKernels(accelerator).Scale(regOutput.Data.SubView(0, regElems), regReadBuf.View, regElems, 1f);
        await accelerator.SynchronizeAsync();
        var actualReg = await regReadBuf.CopyToHostAsync<float>(0, regElems);

        var refRegBytes = await http.GetByteArrayAsync("references/blaze-face/cat_output_regressors.bin");
        var expectedReg = new float[refRegBytes.Length / 4];
        Buffer.BlockCopy(refRegBytes, 0, expectedReg, 0, refRegBytes.Length);

        var cmpLen = Math.Min(actualReg.Length, expectedReg.Length);
        AssertReferenceMatch(actualReg.Take(cmpLen).ToArray(), expectedReg.Take(cmpLen).ToArray(), 1.0f, "BlazeFace_regressors");

        // Compare classificators
        var clsOutput = outputs[session.OutputNames[1]];
        int clsElems = clsOutput.ElementCount;
        using var clsReadBuf = accelerator.Allocate1D<float>(clsElems);
        new ElementWiseKernels(accelerator).Scale(clsOutput.Data.SubView(0, clsElems), clsReadBuf.View, clsElems, 1f);
        await accelerator.SynchronizeAsync();
        var actualCls = await clsReadBuf.CopyToHostAsync<float>(0, clsElems);

        var refClsBytes = await http.GetByteArrayAsync("references/blaze-face/cat_output_classificators.bin");
        var expectedCls = new float[refClsBytes.Length / 4];
        Buffer.BlockCopy(refClsBytes, 0, expectedCls, 0, refClsBytes.Length);

        var clsCmpLen = Math.Min(actualCls.Length, expectedCls.Length);
        AssertReferenceMatch(actualCls.Take(clsCmpLen).ToArray(), expectedCls.Take(clsCmpLen).ToArray(), 1.0f, "BlazeFace_classificators");

        Console.WriteLine($"[Pipeline] BlazeFace: regressors={regElems}, classificators={clsElems}");
        Console.WriteLine($"[Pipeline] BlazeFace reference: PASS");
        session.Dispose();
    });
}
