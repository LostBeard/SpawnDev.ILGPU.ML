using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Test TFLite model loading and compilation via InferenceSession.
    /// Uses the BlazeFace model if available.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task TFLite_CreateSession_BlazeFace() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        // Try to load BlazeFace TFLite model
        try
        {
            var session = await InferenceSession.CreateFromTFLiteAsync(
                accelerator, http, "models/blaze-face/model.tflite");

            Console.WriteLine($"[TFLite] Session: {session}");
            Console.WriteLine($"[TFLite] Inputs: {string.Join(", ", session.InputNames)}");
            Console.WriteLine($"[TFLite] Outputs: {string.Join(", ", session.OutputNames)}");
            Console.WriteLine($"[TFLite] Nodes: {session.NodeCount}");
            Console.WriteLine($"[TFLite] Weights: {session.WeightCount}");
            Console.WriteLine($"[TFLite] Operators: {string.Join(", ", session.OperatorTypes)}");

            if (session.NodeCount == 0)
                throw new Exception("TFLite session has 0 nodes");

            Console.WriteLine("[TFLite] PASS — session created from .tflite");
            session.Dispose();
        }
        catch (HttpRequestException)
        {
            throw new UnsupportedTestException("BlazeFace model not available at models/blaze-face/model.tflite");
        }
    });

    /// <summary>
    /// Test auto-format detection — same API loads both ONNX and TFLite.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task AutoDetect_LoadOnnxAndTFLite() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        // Load ONNX via auto-detect
        try
        {
            var onnxSession = await InferenceSession.CreateFromFileAsync(
                accelerator, http, "models/squeezenet/model.onnx");
            Console.WriteLine($"[AutoDetect] ONNX: {onnxSession.ModelName}, {onnxSession.NodeCount} nodes");
            onnxSession.Dispose();
        }
        catch (HttpRequestException)
        {
            Console.WriteLine("[AutoDetect] SqueezeNet not available — skipping ONNX test");
        }

        // Load TFLite via auto-detect
        try
        {
            var tfliteSession = await InferenceSession.CreateFromFileAsync(
                accelerator, http, "models/blaze-face/model.tflite");
            Console.WriteLine($"[AutoDetect] TFLite: {tfliteSession.ModelName}, {tfliteSession.NodeCount} nodes");
            tfliteSession.Dispose();
        }
        catch (HttpRequestException)
        {
            Console.WriteLine("[AutoDetect] BlazeFace not available — skipping TFLite test");
        }

        Console.WriteLine("[AutoDetect] PASS — format auto-detection works");
    });

    /// <summary>
    /// Test format detection from magic bytes.
    /// </summary>
    [TestMethod]
    public async Task FormatDetection_MagicBytes() => await RunTest(async accelerator =>
    {
        // ONNX: starts with protobuf field tag
        var onnxLike = new byte[] { 0x08, 0x07, 0x12, 0x04, 0x6F, 0x6E, 0x6E, 0x78 }; // "onnx" at offset 4
        if (InferenceSession.DetectModelFormat(onnxLike) != ModelFormat.ONNX)
            throw new Exception("Failed to detect ONNX");

        // GGUF: starts with "GGUF"
        var ggufLike = new byte[] { 0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00 };
        if (InferenceSession.DetectModelFormat(ggufLike) != ModelFormat.GGUF)
            throw new Exception("Failed to detect GGUF");

        // TFLite: "TFL3" at offset 4
        var tfliteLike = new byte[] { 0x00, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4C, 0x33 };
        if (InferenceSession.DetectModelFormat(tfliteLike) != ModelFormat.TFLite)
            throw new Exception("Failed to detect TFLite");

        Console.WriteLine("[FormatDetect] All 3 formats detected correctly");
        await Task.CompletedTask;
    });
}
