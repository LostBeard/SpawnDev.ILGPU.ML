using System.Text;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Onnx;
using SpawnDev.UnitTesting;
using SpawnDev.WebTorrent;

namespace SpawnDev.ILGPU.ML.Demo.UnitTests;

/// <summary>
/// Tests for the Model Inspector demo (/inspector). These exercise the EXACT code path the demo page
/// runs in HandleFileSelected: ModelInspectorHelper.InspectWithCompatibilityAsync over a stream, against
/// real model files.
///
/// Model inspection is PURE CPU protobuf/header parsing — it NEVER touches a GPU accelerator. So these
/// tests are NOT part of MLTestBase (which fans every method out across all 6 backend lanes); they live
/// in their own class registered once in Program.cs, so they run a SINGLE time in the browser runtime
/// (the meaningful environment: fetch-stream behavior, the sync-read ban, WASM memory limits) instead of
/// 6× redundantly. Proving these here means TJ only has to CONFIRM the demo, not discover whether it works.
/// </summary>
public class ModelInspectorTests
{
    private readonly HttpClient _http;

    public ModelInspectorTests(HttpClient http)
    {
        _http = http;
    }

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
        var bytes = await _http.GetByteArrayAsync("models/squeezenet/model.onnx");

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

    /// <summary>
    /// ONNX inspection: transformer models (GPT-2 ~623MB, DistilBERT ~256MB) report millions of params
    /// + MatMul/Gemm. Inspected via STREAMING (InspectAsync over the HTTP response stream) — the weights
    /// are skipped, never buffered, so this never materializes the multi-hundred-MB file in memory. This
    /// is the exact path the demo runs when a user drops a large model. I/O-bound (large transfer), so a
    /// generous timeout, not the 30s compute default.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task ModelInspector_Onnx_Transformers_Inspect()
    {
        foreach (var path in new[] { "models/gpt2/model.onnx", "models/distilbert-sst2/model.onnx" })
        {
            using var stream = await _http.GetStreamAsync(path);
            var r = await ModelInspectorHelper.InspectAsync(stream);
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
        foreach (var path in new[] { "models/blaze-face/model.tflite", "models/efficientnet-lite0/model.tflite" })
        {
            var bytes = await _http.GetByteArrayAsync(path);
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
        byte[] bytes;
        try { bytes = await _http.GetByteArrayAsync("test-models/test.gguf"); }
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
        byte[] bytes;
        try { bytes = await _http.GetByteArrayAsync("test-models/test.safetensors"); }
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
    [TestMethod(Timeout = 120000)]
    public async Task ModelInspector_Compatibility_Onnx_Meaningful()
    {
        // SqueezeNet uses only supported ops → fully compatible. Streamed (CheckCompatibilityAsync reads
        // only the graph structure; weights are skipped).
        using (var sqStream = await _http.GetStreamAsync("models/squeezenet/model.onnx"))
        {
            var sq = await ModelInspectorHelper.CheckCompatibilityAsync(sqStream);
            if (!sq.Applicable) throw new Exception("ONNX compat must be Applicable");
            if (sq.TotalOpsUsed <= 0) throw new Exception("Expected ops");
            if (sq.SupportedOps.Length + sq.UnsupportedOps.Length != sq.TotalOpsUsed)
                throw new Exception($"Op partition mismatch: {sq.SupportedOps.Length}+{sq.UnsupportedOps.Length} != {sq.TotalOpsUsed}");
            if (!sq.IsFullySupported) throw new Exception($"SqueezeNet expected fully supported; missing: {string.Join(",", sq.UnsupportedOps)}");
        }

        // GPT-2 (~623MB) has some unsupported ops → partial but self-consistent (guards the % math).
        // Streamed — its compatibility is checked without ever loading its weights.
        using (var gpt2Stream = await _http.GetStreamAsync("models/gpt2/model.onnx"))
        {
            var gpt2 = await ModelInspectorHelper.CheckCompatibilityAsync(gpt2Stream);
            if (gpt2.SupportedOps.Length + gpt2.UnsupportedOps.Length != gpt2.TotalOpsUsed)
                throw new Exception("GPT-2 op partition mismatch");
            if (gpt2.CompatibilityPercent < 0 || gpt2.CompatibilityPercent > 100)
                throw new Exception($"GPT-2 CompatibilityPercent={gpt2.CompatibilityPercent} out of range");
        }
    }

    /// <summary>
    /// REGRESSION GUARD: CheckCompatibility must NOT throw for non-ONNX formats. The demo calls it
    /// for every dropped file; before the fix, OnnxParser.Parse on TFLite/GGUF/SafeTensors bytes
    /// threw InvalidOperationException and broke the page. Non-ONNX → Applicable==false, no throw.
    /// </summary>
    [TestMethod]
    public async Task ModelInspector_Compatibility_NonOnnx_DoesNotThrow()
    {
        var nonOnnx = new[]
        {
            "models/blaze-face/model.tflite",
            "models/efficientnet-lite0/model.tflite",
            "test-models/test.gguf",
            "test-models/test.safetensors",
        };
        foreach (var path in nonOnnx)
        {
            Stream stream;
            try { stream = await _http.GetStreamAsync(path); }
            catch (HttpRequestException) { continue; }

            // Must not throw (streamed — non-ONNX is detected from the prefix, weights never read).
            CompatibilityResult c;
            using (stream) c = await ModelInspectorHelper.CheckCompatibilityAsync(stream);
            if (c.Applicable)
                throw new Exception($"{path}: op-compat is ONNX-only; expected Applicable==false");
            if (string.IsNullOrEmpty(c.Summary))
                throw new Exception($"{path}: expected a non-empty Summary for the demo to display");
        }
    }

    /// <summary>
    /// Full-coverage smoke: every model the demo accepts must inspect without throwing and produce a
    /// populated result. This is the demo's full use case (drop ANY supported model), streamed one at a
    /// time so memory stays bounded regardless of model size.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task ModelInspector_Inspect_AllDemoModels_NoThrow()
    {
        var failures = new List<string>();
        int reachable = 0;
        foreach (var path in AllInspectableModels)
        {
            reachable++;
            try
            {
                // ONE stream pass yields BOTH inspection and compatibility (the demo calls both) — the
                // model is fetched once, not twice, and weights are skipped. This is exactly the demo's
                // per-file path.
                using var stream = await _http.GetStreamAsync(path);
                var (r, _) = await ModelInspectorHelper.InspectWithCompatibilityAsync(stream);
                if (string.IsNullOrEmpty(r.GraphName) && string.IsNullOrEmpty(r.ProducerName))
                    failures.Add($"{path}: empty metadata");
                if (r.NodeCount <= 0 && r.InitializerCount <= 0)
                    failures.Add($"{path}: no nodes and no initializers");
            }
            catch (HttpRequestException) { reachable--; continue; } // optional model not deployed — skip
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
        byte[] bytes;
        try { bytes = await _http.GetByteArrayAsync("test-models/test.safetensors"); }
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
        byte[] bytes;
        try { bytes = await _http.GetByteArrayAsync("test-models/test.gguf"); }
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
        var bytes = await _http.GetByteArrayAsync("models/squeezenet/model.onnx");

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

    /// <summary>
    /// Streaming ONNX inspection over a SEEKABLE stream must match the in-memory Inspect(byte[])
    /// field-for-field on every structural metric (the demo path for a seekable source).
    /// </summary>
    [TestMethod]
    public async Task ModelInspector_Stream_Onnx_Seekable_Matches()
    {
        var bytes = await _http.GetByteArrayAsync("models/squeezenet/model.onnx");

        var fromBytes = ModelInspectorHelper.Inspect(bytes);
        using var ms = new MemoryStream(bytes); // seekable → exercises the Seek-past-weights path
        var fromStream = await ModelInspectorHelper.InspectAsync(ms);

        if (fromStream.NodeCount != fromBytes.NodeCount)
            throw new Exception($"NodeCount stream={fromStream.NodeCount} bytes={fromBytes.NodeCount}");
        if (fromStream.InitializerCount != fromBytes.InitializerCount)
            throw new Exception($"InitializerCount stream={fromStream.InitializerCount} bytes={fromBytes.InitializerCount}");
        if (fromStream.TotalParameters != fromBytes.TotalParameters)
            throw new Exception($"TotalParameters stream={fromStream.TotalParameters} bytes={fromBytes.TotalParameters}");
        if (fromStream.TotalWeightBytes != fromBytes.TotalWeightBytes)
            throw new Exception($"TotalWeightBytes stream={fromStream.TotalWeightBytes} bytes={fromBytes.TotalWeightBytes}");
        if (fromStream.Operators.Length != fromBytes.Operators.Length)
            throw new Exception($"Operators stream={fromStream.Operators.Length} bytes={fromBytes.Operators.Length}");
        if (fromStream.Inputs.Length != fromBytes.Inputs.Length)
            throw new Exception($"Inputs stream={fromStream.Inputs.Length} bytes={fromBytes.Inputs.Length}");
        if (fromStream.Outputs.Length != fromBytes.Outputs.Length)
            throw new Exception($"Outputs stream={fromStream.Outputs.Length} bytes={fromBytes.Outputs.Length}");
        if (fromStream.LargestWeights.Length != fromBytes.LargestWeights.Length)
            throw new Exception($"LargestWeights stream={fromStream.LargestWeights.Length} bytes={fromBytes.LargestWeights.Length}");
        // The two op-type sets must be identical (order among equal counts may differ).
        var sSet = fromStream.Operators.Select(o => o.OpType).OrderBy(o => o).ToArray();
        var bSet = fromBytes.Operators.Select(o => o.OpType).OrderBy(o => o).ToArray();
        if (!sSet.SequenceEqual(bSet))
            throw new Exception($"Op-type sets differ: stream=[{string.Join(",", sSet)}] bytes=[{string.Join(",", bSet)}]");
    }

    /// <summary>
    /// PROOF that seekable ONNX inspection NEVER reads the weight blobs: a counting seekable stream
    /// over a real ONNX model. The bytes actually Read() must be a small fraction of the file — the
    /// raw_data weight sections are Seek()-ed past, not read. This is the whole point: inspect any-size
    /// model from its structure alone.
    /// </summary>
    [TestMethod]
    public async Task ModelInspector_Stream_Onnx_SeekSkipsWeights()
    {
        // SqueezeNet (~5MB) is mostly Conv weights; structure is a small fraction of the file.
        var bytes = await _http.GetByteArrayAsync("models/squeezenet/model.onnx");

        using var counting = new CountingSeekableStream(bytes);
        var r = await ModelInspectorHelper.InspectAsync(counting);

        if (r.NodeCount <= 0) throw new Exception("Expected a parsed graph");
        if (r.InitializerCount <= 0) throw new Exception("Expected initializers (weights) listed by metadata");
        // Weights were skipped via Seek, so bytes READ must be far below the file size. Generous bound:
        // structure + dim/name fields are well under a quarter of a weight-heavy model.
        if (counting.BytesRead >= bytes.Length / 4)
            throw new Exception($"Read {counting.BytesRead} of {bytes.Length} bytes — weights were NOT skipped (expected << {bytes.Length / 4})");
        if (counting.BytesSeeked <= 0)
            throw new Exception("Expected Seek() calls skipping weight blobs; none occurred");
    }

    // ── Inspect-by-URL via the live SpawnDev hub (the original #1 goal) ──

    /// <summary>
    /// Inspect a HuggingFace model BY URL via the live SpawnDev hub (hub.spawndev.com): HubModelStream
    /// asks the hub for a magnet, resolves metadata PEER-FREE via the magnet's HTTP exact-source (xs=),
    /// opens the torrent DESELECTED, and the inspector seeks past every weight blob. So inspecting the
    /// model fetches only the structure pieces it touches — NOT the weights. Asserts (a) a meaningful
    /// architecture parsed, and (b) the torrent did NOT download the whole file. This is the production
    /// inspect-by-URL path the demo exposes; requires internet (cold hub cache → generous timeout).
    /// </summary>
    [TestMethod(Timeout = 240000, RetryCount = 2)]
    public async Task ModelInspector_Hub_InspectByUrl_StructureOnly()
    {
        const string repoId = "onnx-community/mobilenetv3_small_100.lamb_in1k";
        const string filePath = "onnx/model.onnx";

        var client = new WebTorrentClient();
        try
        {
            var hub = new HubModelStream(client, _http);
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(180));

            // Open DESELECTED so only touched (structure) pieces download.
            var model = await hub.OpenAsync(repoId, filePath, deselect: true, cts.Token);
            if (model.File.Length <= 0) throw new Exception($"hub model file length={model.File.Length}");

            InspectionResult r;
            await using (model.Stream)
                r = await ModelInspectorHelper.InspectAsync(model.Stream, cts.Token);

            // (a) Structure must be meaningful — proves on-demand deselected reads fetch the right pieces.
            if (r.NodeCount <= 0) throw new Exception($"NodeCount={r.NodeCount}, expected > 0");
            if (r.TotalParameters <= 0) throw new Exception($"TotalParameters={r.TotalParameters}, expected > 0");
            if (!r.Operators.Any(o => o.OpType == "Conv")) throw new Exception("mobilenet must use Conv");

            // (b) Inspecting structure must NOT pull the whole model. Without deselect + seek-past-weights
            // the default select-all would download every byte. Degree of saving is layout/piece-size
            // dependent, so the robust, non-flaky claim is simply: strictly less than the full file.
            long downloaded = model.Torrent.Downloaded;
            if (downloaded >= model.File.Length)
                throw new Exception(
                    $"inspect-by-URL downloaded {downloaded} of {model.File.Length} bytes (the whole file) — " +
                    "deselect / seek-past-weights was not effective; weights were pulled");
        }
        finally
        {
            await client.DisposeAsync();
        }
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

/// <summary>Seekable stream over a byte[] that counts bytes actually READ vs bytes SEEKED past.
/// Proves the streaming inspector skips weight blobs (Seek) instead of reading them.</summary>
file sealed class CountingSeekableStream : Stream
{
    private readonly byte[] _data;
    private long _pos;
    public long BytesRead { get; private set; }
    public long BytesSeeked { get; private set; }

    public CountingSeekableStream(byte[] data) => _data = data;

    public override int Read(byte[] buffer, int offset, int count)
    {
        int n = (int)Math.Min(count, _data.Length - _pos);
        if (n <= 0) return 0;
        Array.Copy(_data, _pos, buffer, offset, n);
        _pos += n;
        BytesRead += n;
        return n;
    }

    public override long Seek(long offset, SeekOrigin origin)
    {
        long target = origin switch
        {
            SeekOrigin.Begin => offset,
            SeekOrigin.Current => _pos + offset,
            SeekOrigin.End => _data.Length + offset,
            _ => _pos,
        };
        if (target > _pos) BytesSeeked += target - _pos;
        _pos = target;
        return _pos;
    }

    public override bool CanRead => true;
    public override bool CanSeek => true;
    public override bool CanWrite => false;
    public override long Length => _data.Length;
    public override long Position { get => _pos; set => _pos = value; }
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
