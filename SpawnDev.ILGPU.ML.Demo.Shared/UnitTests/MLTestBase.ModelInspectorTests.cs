using System.Text;
using System.Text.Json;
using SpawnDev.ILGPU.ML.Onnx;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Tests for the Model Inspector demo (/inspector). These exercise the EXACT code path the
/// demo page runs in HandleFileSelected: ModelInspectorHelper.Inspect(bytes) +
/// ModelInspectorHelper.CheckCompatibility(bytes), against real model files. Pure in-browser
/// parsing — no accelerator needed (parallels FormatDetectionTests). Proving these here means TJ
/// only has to CONFIRM the demo, not discover whether it works.
/// </summary>
public abstract partial class MLTestBase
{
    // Every model the Inspector demo accepts (the demo's accept= list is .onnx/.tflite/.gguf/.safetensors).
    private static readonly string[] AllInspectableModels =
    {
        "models/squeezenet/model.onnx",
        "models/mobilenetv2/model.onnx",
        "models/movenet-lightning/model.onnx",
        "models/yolov8n/model.onnx",
        "models/depth-anything-v2-small/model.onnx",
        "models/distilbert-sst2/model.onnx",
        "models/gpt2/model.onnx",
        "models/whisper-tiny/encoder_model.onnx",
        "models/whisper-tiny/decoder_model.onnx",
        "models/super-resolution/model.onnx",
        "models/style-mosaic/model.onnx",
        "models/style-candy/model.onnx",
        "models/style-pointilism/model.onnx",
        "models/style-rain-princess/model.onnx",
        "models/style-udnie/model.onnx",
        "models/blaze-face/model.tflite",
        "models/efficientnet-lite0/model.tflite",
        "test-models/test.gguf",
        "test-models/test.safetensors",
    };

    /// <summary>ONNX inspection: SqueezeNet parses into a meaningful architecture summary.</summary>
    [TestMethod]
    public async Task ModelInspector_Onnx_SqueezeNet_Inspects()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var bytes = await http.GetByteArrayAsync("models/squeezenet/model.onnx");

        var r = ModelInspectorHelper.Inspect(bytes);
        if (r.NodeCount <= 0) throw new Exception($"NodeCount={r.NodeCount}, expected > 0");
        if (r.Inputs.Length < 1) throw new Exception("Expected >= 1 input");
        if (r.Outputs.Length < 1) throw new Exception("Expected >= 1 output");
        if (r.TotalParameters <= 0) throw new Exception($"TotalParameters={r.TotalParameters}, expected > 0");
        if (r.LargestWeights.Length < 1) throw new Exception("Expected weights listed");
        if (!r.Operators.Any(o => o.OpType == "Conv")) throw new Exception("SqueezeNet must use Conv");
        if (string.IsNullOrEmpty(r.GraphName) && string.IsNullOrEmpty(r.ProducerName))
            throw new Exception("Expected graph/producer metadata");
    }

    /// <summary>ONNX inspection: transformer models (GPT-2, DistilBERT) report millions of params + MatMul/Gemm.</summary>
    [TestMethod]
    public async Task ModelInspector_Onnx_Transformers_Inspect()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        foreach (var path in new[] { "models/gpt2/model.onnx", "models/distilbert-sst2/model.onnx" })
        {
            var bytes = await http.GetByteArrayAsync(path);
            var r = ModelInspectorHelper.Inspect(bytes);
            if (r.NodeCount <= 0) throw new Exception($"{path}: NodeCount={r.NodeCount}");
            if (r.TotalParameters < 1_000_000) throw new Exception($"{path}: params={r.TotalParameters}, expected millions");
            if (!r.Operators.Any(o => o.OpType is "MatMul" or "Gemm"))
                throw new Exception($"{path}: transformer must use MatMul/Gemm");
        }
    }

    /// <summary>TFLite inspection: BlazeFace + EfficientNet parse with ops and weights.</summary>
    [TestMethod]
    public async Task ModelInspector_TFLite_Models_Inspect()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        foreach (var path in new[] { "models/blaze-face/model.tflite", "models/efficientnet-lite0/model.tflite" })
        {
            var bytes = await http.GetByteArrayAsync(path);
            var r = ModelInspectorHelper.Inspect(bytes);
            if (r.NodeCount <= 0) throw new Exception($"{path}: NodeCount={r.NodeCount}");
            if (r.Operators.Length < 1) throw new Exception($"{path}: expected operators");
            if (r.TotalParameters <= 0) throw new Exception($"{path}: params={r.TotalParameters}");
            if (!r.ProducerName.Contains("TensorFlow", StringComparison.OrdinalIgnoreCase))
                throw new Exception($"{path}: ProducerName='{r.ProducerName}', expected TensorFlow Lite");
        }
    }

    /// <summary>GGUF inspection: tensor metadata parses (weights-only LLM format).</summary>
    [TestMethod]
    public async Task ModelInspector_GGUF_Inspects()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        byte[] bytes;
        try { bytes = await http.GetByteArrayAsync("test-models/test.gguf"); }
        catch (HttpRequestException) { throw new UnsupportedTestException("test.gguf not available"); }

        var r = ModelInspectorHelper.Inspect(bytes);
        if (r.InitializerCount <= 0 && r.NodeCount <= 0) throw new Exception("GGUF: expected tensors");
        if (string.IsNullOrEmpty(r.GraphName)) throw new Exception("GGUF: expected a graph/arch name");
        if (!r.ProducerName.Contains("GGUF", StringComparison.OrdinalIgnoreCase))
            throw new Exception($"GGUF: ProducerName='{r.ProducerName}'");
    }

    /// <summary>SafeTensors inspection: weights-only file lists tensors, reports no graph.</summary>
    [TestMethod]
    public async Task ModelInspector_SafeTensors_Inspects()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        byte[] bytes;
        try { bytes = await http.GetByteArrayAsync("test-models/test.safetensors"); }
        catch (HttpRequestException) { throw new UnsupportedTestException("test.safetensors not available"); }

        var r = ModelInspectorHelper.Inspect(bytes);
        if (r.NodeCount != 0) throw new Exception($"SafeTensors has no graph; NodeCount={r.NodeCount}, expected 0");
        if (r.InitializerCount <= 0) throw new Exception("SafeTensors: expected tensor count > 0");
        if (!r.ProducerName.Contains("SafeTensors", StringComparison.OrdinalIgnoreCase))
            throw new Exception($"SafeTensors: ProducerName='{r.ProducerName}'");
    }

    /// <summary>
    /// Compatibility check is meaningful for ONNX: ops are partitioned into supported/unsupported,
    /// the counts are self-consistent, and a fully-supported model reports IsFullySupported.
    /// </summary>
    [TestMethod]
    public async Task ModelInspector_Compatibility_Onnx_Meaningful()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // SqueezeNet uses only supported ops → fully compatible.
        var sq = ModelInspectorHelper.CheckCompatibility(await http.GetByteArrayAsync("models/squeezenet/model.onnx"));
        if (!sq.Applicable) throw new Exception("ONNX compat must be Applicable");
        if (sq.TotalOpsUsed <= 0) throw new Exception("Expected ops");
        if (sq.SupportedOps.Length + sq.UnsupportedOps.Length != sq.TotalOpsUsed)
            throw new Exception($"Op partition mismatch: {sq.SupportedOps.Length}+{sq.UnsupportedOps.Length} != {sq.TotalOpsUsed}");
        if (!sq.IsFullySupported) throw new Exception($"SqueezeNet expected fully supported; missing: {string.Join(",", sq.UnsupportedOps)}");

        // GPT-2 has some unsupported ops → partial but self-consistent (guards the % math).
        var gpt2 = ModelInspectorHelper.CheckCompatibility(await http.GetByteArrayAsync("models/gpt2/model.onnx"));
        if (gpt2.SupportedOps.Length + gpt2.UnsupportedOps.Length != gpt2.TotalOpsUsed)
            throw new Exception("GPT-2 op partition mismatch");
        if (gpt2.CompatibilityPercent < 0 || gpt2.CompatibilityPercent > 100)
            throw new Exception($"GPT-2 CompatibilityPercent={gpt2.CompatibilityPercent} out of range");
    }

    /// <summary>
    /// REGRESSION GUARD: CheckCompatibility must NOT throw for non-ONNX formats. The demo calls it
    /// for every dropped file; before the fix, OnnxParser.Parse on TFLite/GGUF/SafeTensors bytes
    /// threw InvalidOperationException and broke the page. Non-ONNX → Applicable==false, no throw.
    /// </summary>
    [TestMethod]
    public async Task ModelInspector_Compatibility_NonOnnx_DoesNotThrow()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var nonOnnx = new[]
        {
            "models/blaze-face/model.tflite",
            "models/efficientnet-lite0/model.tflite",
            "test-models/test.gguf",
            "test-models/test.safetensors",
        };
        foreach (var path in nonOnnx)
        {
            byte[] bytes;
            try { bytes = await http.GetByteArrayAsync(path); }
            catch (HttpRequestException) { continue; }

            // Must not throw.
            var c = ModelInspectorHelper.CheckCompatibility(bytes);
            if (c.Applicable)
                throw new Exception($"{path}: op-compat is ONNX-only; expected Applicable==false");
            if (string.IsNullOrEmpty(c.Summary))
                throw new Exception($"{path}: expected a non-empty Summary for the demo to display");
        }
    }

    /// <summary>
    /// Full-coverage smoke: every model the demo accepts must Inspect() without throwing and
    /// produce a populated result. This is the demo's full use case (drop ANY supported model).
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task ModelInspector_Inspect_AllDemoModels_NoThrow()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var failures = new List<string>();
        int reachable = 0;
        foreach (var path in AllInspectableModels)
        {
            byte[] bytes;
            try { bytes = await http.GetByteArrayAsync(path); }
            catch (HttpRequestException) { continue; } // optional model not deployed — skip
            reachable++;

            try
            {
                var r = ModelInspectorHelper.Inspect(bytes);
                if (string.IsNullOrEmpty(r.GraphName) && string.IsNullOrEmpty(r.ProducerName))
                    failures.Add($"{path}: empty metadata");
                if (r.NodeCount <= 0 && r.InitializerCount <= 0)
                    failures.Add($"{path}: no nodes and no initializers");
                // CheckCompatibility must also never throw for any accepted file (demo calls both).
                _ = ModelInspectorHelper.CheckCompatibility(bytes);
            }
            catch (Exception ex)
            {
                failures.Add($"{path}: {ex.GetType().Name}: {ex.Message}");
            }
        }

        if (reachable == 0) throw new UnsupportedTestException("No inspectable models reachable");
        if (failures.Count > 0)
            throw new Exception($"Inspect/CheckCompatibility failed for {failures.Count}/{reachable}: {string.Join(" | ", failures)}");
    }

    // ── Stream-based inspection (no full-model-in-memory) ──

    /// <summary>InspectAsync(Stream) for SafeTensors must match Inspect(byte[]) field-for-field.</summary>
    [TestMethod]
    public async Task ModelInspector_Stream_SafeTensors_MatchesByteArray()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        byte[] bytes;
        try { bytes = await http.GetByteArrayAsync("test-models/test.safetensors"); }
        catch (HttpRequestException) { throw new UnsupportedTestException("test.safetensors not available"); }

        var fromBytes = ModelInspectorHelper.Inspect(bytes);
        using var ms = new MemoryStream(bytes);
        var fromStream = await ModelInspectorHelper.InspectAsync(ms);

        if (fromStream.InitializerCount != fromBytes.InitializerCount)
            throw new Exception($"InitializerCount stream={fromStream.InitializerCount} bytes={fromBytes.InitializerCount}");
        if (fromStream.TotalParameters != fromBytes.TotalParameters)
            throw new Exception($"TotalParameters stream={fromStream.TotalParameters} bytes={fromBytes.TotalParameters}");
        if (fromStream.LargestWeights.Length != fromBytes.LargestWeights.Length)
            throw new Exception("LargestWeights length mismatch");
        if (fromStream.NodeCount != fromBytes.NodeCount)
            throw new Exception("NodeCount mismatch");
    }

    /// <summary>
    /// PROOF that SafeTensors stream inspection reads only the header, never the weight blob:
    /// a stream that serves [8-byte len][JSON header] and THROWS on any read into the data section.
    /// If InspectAsync succeeds, it provably never touched the (here, multi-GB-claimed) tensor data.
    /// This is the whole point of streaming — a giant checkpoint inspects from a few KB.
    /// </summary>
    [TestMethod]
    public async Task ModelInspector_Stream_SafeTensors_ReadsHeaderOnly()
    {
        // Build a SafeTensors header for 4 tensors whose data_offsets claim 1 GB of data we will
        // NOT provide. Header JSON is > the 256-byte detection prefix so detection stays in-header.
        var entries = new List<string>();
        long off = 0;
        const long perTensor = 256L * 1024 * 1024; // 256 MB each → ~1 GB total claimed
        for (int i = 0; i < 4; i++)
        {
            long end = off + perTensor;
            entries.Add($"\"layer_{i}.weight\":{{\"dtype\":\"F32\",\"shape\":[8192,8192],\"data_offsets\":[{off},{end}]}}");
            off = end;
        }
        var json = "{" + string.Join(",", entries) + "}";
        var jsonBytes = Encoding.UTF8.GetBytes(json);
        var header = new byte[8 + jsonBytes.Length];
        BitConverter.GetBytes((long)jsonBytes.Length).CopyTo(header, 0);
        jsonBytes.CopyTo(header, 8);
        if (header.Length <= 256) throw new Exception($"test setup: header {header.Length} must exceed 256-byte prefix");

        // Stream that yields exactly the header bytes, then throws if read further (data unavailable).
        using var stream = new HeaderThenThrowStream(header, claimedTotalLength: 8 + off);

        var r = await ModelInspectorHelper.InspectAsync(stream); // must NOT throw
        if (r.InitializerCount != 4) throw new Exception($"Expected 4 tensors, got {r.InitializerCount}");
        if (r.NodeCount != 0) throw new Exception("SafeTensors has no graph");
        // 4 * 8192 * 8192 = 268,435,456 params
        if (r.TotalParameters != 4L * 8192 * 8192)
            throw new Exception($"params={r.TotalParameters}, expected {4L * 8192 * 8192}");
        if (stream.BytesReadPastHeader != 0)
            throw new Exception($"Read {stream.BytesReadPastHeader} bytes into the weight section — should be 0");
    }

    /// <summary>GGUF header-only stream inspection (seekable) matches Inspect(byte[]).</summary>
    [TestMethod]
    public async Task ModelInspector_Stream_GGUF_Seekable_Matches()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        byte[] bytes;
        try { bytes = await http.GetByteArrayAsync("test-models/test.gguf"); }
        catch (HttpRequestException) { throw new UnsupportedTestException("test.gguf not available"); }

        var fromBytes = ModelInspectorHelper.Inspect(bytes);
        using var ms = new MemoryStream(bytes); // seekable → GGUF header-only path
        var fromStream = await ModelInspectorHelper.InspectAsync(ms);

        if (fromStream.NodeCount != fromBytes.NodeCount)
            throw new Exception($"NodeCount stream={fromStream.NodeCount} bytes={fromBytes.NodeCount}");
        if (fromStream.InitializerCount != fromBytes.InitializerCount)
            throw new Exception("InitializerCount mismatch");
        if (fromStream.TotalParameters != fromBytes.TotalParameters)
            throw new Exception($"TotalParameters stream={fromStream.TotalParameters} bytes={fromBytes.TotalParameters}");
        if (fromStream.Operators.Length != fromBytes.Operators.Length)
            throw new Exception("Operators mismatch");
    }

    /// <summary>InspectAsync fallback path (ONNX) over a non-seekable stream matches Inspect(byte[]).</summary>
    [TestMethod]
    public async Task ModelInspector_Stream_Onnx_NonSeekable_Matches()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var bytes = await http.GetByteArrayAsync("models/squeezenet/model.onnx");

        var fromBytes = ModelInspectorHelper.Inspect(bytes);
        // Forward-only, non-seekable wrapper — simulates browser OpenReadStream / HttpClient stream.
        using var stream = new ForwardOnlyStream(bytes);
        var fromStream = await ModelInspectorHelper.InspectAsync(stream);

        if (fromStream.NodeCount != fromBytes.NodeCount)
            throw new Exception($"NodeCount stream={fromStream.NodeCount} bytes={fromBytes.NodeCount}");
        if (fromStream.Operators.Length != fromBytes.Operators.Length)
            throw new Exception("Operators mismatch");
        if (fromStream.TotalParameters != fromBytes.TotalParameters)
            throw new Exception("TotalParameters mismatch");
    }
}

/// <summary>Serves a fixed header, then throws on any read past it. Tracks bytes read past header.</summary>
file sealed class HeaderThenThrowStream : Stream
{
    private readonly byte[] _header;
    private readonly long _claimedTotalLength;
    private long _pos;
    public long BytesReadPastHeader { get; private set; }

    public HeaderThenThrowStream(byte[] header, long claimedTotalLength)
    {
        _header = header;
        _claimedTotalLength = claimedTotalLength;
    }

    public override int Read(byte[] buffer, int offset, int count)
    {
        if (_pos >= _header.Length)
        {
            BytesReadPastHeader += count;
            throw new IOException("Read past header into the weight-data section (data not available / would be expensive).");
        }
        int n = (int)Math.Min(count, _header.Length - _pos);
        Array.Copy(_header, _pos, buffer, offset, n);
        _pos += n;
        return n;
    }

    public override bool CanRead => true;
    public override bool CanSeek => true;             // seekable, but Length is the CLAIMED full size
    public override bool CanWrite => false;
    public override long Length => _claimedTotalLength;
    public override long Position { get => _pos; set => _pos = value; }
    public override long Seek(long offset, SeekOrigin origin) => _pos;
    public override void Flush() { }
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
}

/// <summary>Forward-only, non-seekable stream over a byte[] (simulates browser/HTTP streams).</summary>
file sealed class ForwardOnlyStream : Stream
{
    private readonly byte[] _data;
    private int _pos;
    public ForwardOnlyStream(byte[] data) => _data = data;

    public override int Read(byte[] buffer, int offset, int count)
    {
        int n = Math.Min(count, _data.Length - _pos);
        if (n <= 0) return 0;
        Array.Copy(_data, _pos, buffer, offset, n);
        _pos += n;
        return n;
    }

    public override bool CanRead => true;
    public override bool CanSeek => false;
    public override bool CanWrite => false;
    public override long Length => throw new NotSupportedException();
    public override long Position { get => _pos; set => throw new NotSupportedException(); }
    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
    public override void Flush() { }
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
}
