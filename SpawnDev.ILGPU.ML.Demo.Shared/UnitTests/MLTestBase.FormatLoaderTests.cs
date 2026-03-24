using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    // ──────────────────────────────────────────────────────────────
    //  Format loader tests: verify all 7 README-claimed formats parse correctly.
    //  Rule #5: unit test everything, not just what the demos use.
    // ──────────────────────────────────────────────────────────────

    [TestMethod]
    public async Task FormatLoader_ONNX_ParsesCorrectly() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var bytes = await http.GetByteArrayAsync("models/squeezenet/model.onnx");
        if (InferenceSession.DetectModelFormat(bytes) != ModelFormat.ONNX)
            throw new Exception("Format detection failed for ONNX");
        var model = SpawnDev.ILGPU.ML.Onnx.OnnxParser.Parse(bytes);
        if (model.Graph.Nodes.Count < 10) throw new Exception($"ONNX: expected many nodes, got {model.Graph.Nodes.Count}");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task FormatLoader_TFLite_ParsesCorrectly() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var bytes = await http.GetByteArrayAsync("models/blaze-face/model.tflite");
        if (InferenceSession.DetectModelFormat(bytes) != ModelFormat.TFLite)
            throw new Exception("Format detection failed for TFLite");
        var model = SpawnDev.ILGPU.ML.TFLite.TFLiteParser.Parse(bytes);
        if (model.Subgraphs.Length < 1) throw new Exception($"TFLite: expected subgraphs, got {model.Subgraphs.Length}");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task FormatLoader_SafeTensors_ParsesCorrectly() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var bytes = await http.GetByteArrayAsync("test-models/test.safetensors");
        if (InferenceSession.DetectModelFormat(bytes) != ModelFormat.SafeTensors)
            throw new Exception("Format detection failed for SafeTensors");
        var model = SpawnDev.ILGPU.ML.SafeTensors.SafeTensorsParser.Parse(bytes);
        if (model.Tensors.Length != 2) throw new Exception($"SafeTensors: expected 2 tensors, got {model.Tensors.Length}");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task FormatLoader_GGUF_ParsesCorrectly() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var bytes = await http.GetByteArrayAsync("test-models/test.gguf");
        if (InferenceSession.DetectModelFormat(bytes) != ModelFormat.GGUF)
            throw new Exception("Format detection failed for GGUF");
        var model = SpawnDev.ILGPU.ML.GGUF.GGUFParser.Parse(bytes);
        if (model.Tensors.Length != 2) throw new Exception($"GGUF: expected 2 tensors, got {model.Tensors.Length}");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task FormatLoader_TFGraphDef_ParsesCorrectly() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var bytes = await http.GetByteArrayAsync("test-models/test.pb");
        var model = SpawnDev.ILGPU.ML.TensorFlow.TFGraphDefParser.Parse(bytes);
        if (model.Nodes.Count < 1) throw new Exception($"TFGraphDef: expected nodes, got {model.Nodes.Count}");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task FormatLoader_PyTorch_ParsesCorrectly() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var bytes = await http.GetByteArrayAsync("test-models/test.pt");
        if (InferenceSession.DetectModelFormat(bytes) != ModelFormat.PyTorch)
            throw new Exception("Format detection failed for PyTorch");
        var model = SpawnDev.ILGPU.ML.PyTorch.PyTorchLoader.Parse(bytes);
        if (model.DataFiles.Count < 1) throw new Exception($"PyTorch: expected data files, got {model.DataFiles.Count}");
        if (model.PickleData == null || model.PickleData.Length == 0) throw new Exception("PyTorch: no pickle data");
        if (model.TensorNames.Count < 1) throw new Exception($"PyTorch: expected tensor names, got {model.TensorNames.Count}");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task FormatLoader_CoreML_ParsesCorrectly() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var bytes = await http.GetByteArrayAsync("test-models/test.mlmodel");
        var model = SpawnDev.ILGPU.ML.CoreML.CoreMLParser.Parse(bytes);
        if (model.SpecVersion < 1) throw new Exception($"CoreML: expected spec version >= 1, got {model.SpecVersion}");
        if (model.InputNames.Count < 1) throw new Exception($"CoreML: expected inputs, got {model.InputNames.Count}");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task FormatDetection_AllFormats() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var tests = new (string path, ModelFormat expected)[]
        {
            ("models/squeezenet/model.onnx", ModelFormat.ONNX),
            ("models/blaze-face/model.tflite", ModelFormat.TFLite),
            ("test-models/test.gguf", ModelFormat.GGUF),
            ("test-models/test.safetensors", ModelFormat.SafeTensors),
            ("test-models/test.pt", ModelFormat.PyTorch),
        };

        foreach (var (path, expected) in tests)
        {
            var bytes = await http.GetByteArrayAsync(path);
            var detected = InferenceSession.DetectModelFormat(bytes);
            if (detected != expected)
                throw new Exception($"{path}: expected {expected}, got {detected}");
        }
        await Task.CompletedTask;
    });
}
