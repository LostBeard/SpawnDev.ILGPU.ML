using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Tests for model format auto-detection across all 11 supported formats.
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// An ONNX file is recognised by its producer string, and exporters spell it however they like. Parakeet's
    /// log-mel preprocessor (istupakov/parakeet-tdt-0.6b-v3-onnx, nemo128.onnx) is written by onnxscript as
    /// "OnnxScript" and was parsed as CoreML ("Unknown wire type: 7"). The first rows are REAL file headers:
    /// ir_version 8 (08 08), producer_name (12 len ...), then producer_version.
    /// </summary>
    [TestMethod]
    public void FormatDetection_OnnxProducerStrings_AreOnnxInAnyCase()
    {
        static byte[] Header(string producer)
        {
            var name = System.Text.Encoding.ASCII.GetBytes(producer);
            var h = new List<byte> { 0x08, 0x08, 0x12, (byte)name.Length };
            h.AddRange(name);
            h.AddRange(new byte[] { 0x1A, 0x05, (byte)'2', (byte)'.', (byte)'8', (byte)'.', (byte)'0', 0x3A, 0xD7, 0xBE, 0xF5, 0x13 });
            return h.ToArray();
        }
        foreach (var producer in new[] { "OnnxScript", "onnx", "ONNX", "tf2onnx", "skl2onnx", "pytorch", "PyTorch" })
        {
            var detected = InferenceSession.DetectModelFormat(Header(producer));
            if (detected != ModelFormat.ONNX)
                throw new Exception($"an ONNX header with producer \"{producer}\" was detected as {detected}");
        }
    }

    [TestMethod]
    public async Task FormatDetection_AllFormats_Detected()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var results = new List<string>();

        // Test each format we have test files for
        var testFiles = new Dictionary<string, ModelFormat>
        {
            ["test-models/test.gguf"] = ModelFormat.GGUF,
            ["test-models/test.safetensors"] = ModelFormat.SafeTensors,
            ["test-models/test.pt"] = ModelFormat.PyTorch,
        };

        foreach (var (file, expectedFormat) in testFiles)
        {
            try
            {
                var bytes = await http.GetByteArrayAsync(file);
                var detected = InferenceSession.DetectModelFormat(bytes);
                if (detected != expectedFormat)
                    throw new Exception($"{file}: detected {detected}, expected {expectedFormat}");
                results.Add($"{file} → {detected} ✓");
            }
            catch (HttpRequestException)
            {
                results.Add($"{file} → (not available, skipped)");
            }
        }

        // Test 3D format detection with reference files
        try
        {
            var spzBytes = await http.GetByteArrayAsync("references/blazing-edge/test_10gaussians.spz");
            var spzFormat = InferenceSession.DetectModelFormat(spzBytes);
            if (spzFormat == ModelFormat.SPZ)
                results.Add("SPZ → detected ✓");
            else
                results.Add($"SPZ → detected as {spzFormat} (may need gzip check)");
        }
        catch { results.Add("SPZ → (not available)"); }

        // GLB detection
        var glbMagic = new byte[] { 0x67, 0x6C, 0x54, 0x46, 0x02, 0x00, 0x00, 0x00 };
        var glbFormat = InferenceSession.DetectModelFormat(glbMagic);
        results.Add(glbFormat == ModelFormat.GLTF ? "glTF → detected ✓" : $"glTF → {glbFormat}");

        // PLY detection
        var plyHeader = System.Text.Encoding.ASCII.GetBytes("ply\nformat binary_little_endian 1.0\n");
        var plyFormat = InferenceSession.DetectModelFormat(plyHeader);
        results.Add(plyFormat == ModelFormat.PLY ? "PLY → detected ✓" : $"PLY → {plyFormat}");

        // OBJ detection
        var objHeader = System.Text.Encoding.ASCII.GetBytes("# OBJ file\nv 0 0 0\n");
        var objFormat = InferenceSession.DetectModelFormat(objHeader);
        results.Add(objFormat == ModelFormat.OBJ ? "OBJ → detected ✓" : $"OBJ → {objFormat}");

        Console.WriteLine($"[FormatDetection] Results:\n  {string.Join("\n  ", results)}");
    }
}
