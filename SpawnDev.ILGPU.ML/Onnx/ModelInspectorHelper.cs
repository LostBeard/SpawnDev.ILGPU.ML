using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace SpawnDev.ILGPU.ML.Onnx;

/// <summary>
/// Helper for the Model Inspector demo page.
/// Drop any .onnx file → see architecture, operators, shapes, weights.
/// A developer tool that showcases our native ONNX parser.
/// </summary>
public static partial class ModelInspectorHelper
{
    // Bytes read up-front to detect the format. Covers every magic-byte check plus the ONNX
    // 64-byte producer-string scan and the SafeTensors 16-byte probe, while staying small enough
    // that header-front formats (SafeTensors/GGUF) with a non-trivial header never over-read into
    // the weight section during detection.
    private const int DetectPrefixBytes = 256;

    /// <summary>
    /// Inspect a model from a <see cref="Stream"/> WITHOUT requiring the whole model in memory.
    /// For header-front formats (SafeTensors, GGUF) only the metadata header is read — a multi-GB
    /// weights file inspects from a few KB and the weight blobs are never materialized. Works with
    /// ANY stream source: browser IBrowserFile.OpenReadStream, HttpClient response streams, a
    /// desktop FileStream, or a WebTorrent seekable stream. Other formats fall back to a full read
    /// (functionally identical to <see cref="Inspect(byte[])"/>).
    /// </summary>
    public static async Task<InspectionResult> InspectAsync(Stream stream, CancellationToken ct = default)
    {
        var (result, _) = await InspectCoreAsync(stream, ct).ConfigureAwait(false);
        return result;
    }

    /// <summary>
    /// Inspect AND check operator compatibility from a SINGLE stream pass — the structure is walked
    /// once and reused for both results. This avoids re-opening / re-reading the source, which matters
    /// for expensive streams (browser file, HTTP, WebTorrent piece stream) where a second pass would
    /// re-download the whole model. The demo and the all-models smoke test use this.
    /// </summary>
    public static async Task<(InspectionResult Inspection, CompatibilityResult Compatibility)> InspectWithCompatibilityAsync(
        Stream stream, Operators.OperatorRegistry? registry = null, CancellationToken ct = default)
    {
        var (result, format) = await InspectCoreAsync(stream, ct).ConfigureAwait(false);
        CompatibilityResult compat;
        if (format == ModelFormat.ONNX)
        {
            // Op types already collected by the structure walk — no second read.
            var opsUsed = result.Operators.Select(o => o.OpType).Distinct().OrderBy(o => o).ToArray();
            compat = PartitionCompatibility(opsUsed, registry);
        }
        else
        {
            compat = NonApplicableCompatibility(format);
        }
        return (result, compat);
    }

    /// <summary>Shared streaming-inspection core: detects format and returns the inspection result plus
    /// the detected <see cref="ModelFormat"/> (so callers can derive compatibility without a second pass).</summary>
    private static async Task<(InspectionResult Result, ModelFormat Format)> InspectCoreAsync(Stream stream, CancellationToken ct)
    {
        if (stream == null) throw new ArgumentNullException(nameof(stream));

        // Read a small detection prefix without consuming the whole stream.
        var prefix = new byte[DetectPrefixBytes];
        int prefixLen = await ReadUpToAsync(stream, prefix, 0, prefix.Length, ct).ConfigureAwait(false);

        // SafeTensors must be probed prefix-tolerantly: its header (and thus the whole file) is
        // typically far larger than the detection prefix, and DetectModelFormat's SafeTensors check
        // requires `headerSize < data.Length - 8`, which a short prefix fails — that would send a
        // multi-GB SafeTensors down the full-read fallback, defeating streaming. Probe directly:
        // a plausible uint64 header length followed by '{'.
        if (IsSafeTensorsPrefix(prefix, prefixLen))
            return (await InspectSafeTensorsStreamAsync(stream, prefix, prefixLen, ct).ConfigureAwait(false), ModelFormat.SafeTensors);

        // DetectModelFormat is happy with a short buffer (it bounds all its probes by length).
        var detectBuf = prefixLen == prefix.Length ? prefix : prefix.AsSpan(0, prefixLen).ToArray();
        var format = InferenceSession.DetectModelFormat(detectBuf);

        switch (format)
        {
            case ModelFormat.SafeTensors:
                return (await InspectSafeTensorsStreamAsync(stream, prefix, prefixLen, ct).ConfigureAwait(false), format);

            case ModelFormat.GGUF:
                // GGUF header is front-loaded (metadata + tensor infos, then the aligned data section).
                // ParseHeaderAsync reads ONLY up to the data boundary — never the weight blob — using
                // ReadAsync exclusively, so it works on every stream including async-only sources
                // (BlobStream, browser OpenReadStream, WebTorrent). The header is forward-only, so no
                // seeking is required; we replay the already-read detection prefix via PrefixedReadStream.
                var ggufModel = await GGUF.GGUFParser.ParseHeaderAsync(
                    new PrefixedReadStream(prefix, prefixLen, stream), ct).ConfigureAwait(false);
                return (BuildGGUFResult(ggufModel, stream.CanSeek ? stream.Length : 0), format);

            case ModelFormat.ONNX:
            {
                // ONNX weights (initializer raw_data) are interleaved throughout the graph, so there
                // is no contiguous header to range-fetch — but they are never needed for inspection.
                // Walk the protobuf from the stream, keep graph structure, and seek past every
                // raw_data blob. A seekable source rewinds to 0 (true Seek-over-weights); a forward
                // stream replays the detection prefix and reads-and-discards weights. Either way the
                // whole file is never resident — a 600 MB GPT-2 inspects from its structure alone.
                Stream onnxStream;
                if (stream.CanSeek)
                {
                    stream.Seek(0, SeekOrigin.Begin);
                    onnxStream = stream;
                }
                else
                {
                    onnxStream = new PrefixedReadStream(prefix, prefixLen, stream);
                }
                var reader = new StreamProtoReader(onnxStream, ct);
                return (await InspectOnnxStreamAsync(reader, stream.CanSeek ? stream.Length : 0, ct).ConfigureAwait(false), format);
            }

            default:
                // Remaining formats (TFLite FlatBuffers, etc.) are random-access; the demo's are small.
                // Assemble the full bytes (already-read prefix + the rest) and reuse the in-memory path.
                var all = await DrainWithPrefixAsync(stream, prefix, prefixLen, ct).ConfigureAwait(false);
                return (Inspect(all), format);
        }
    }

    /// <summary>
    /// Replays an already-read prefix, then delegates to the underlying stream — lets a parser read
    /// from byte 0 even though the format-detection prefix was already consumed from the source.
    /// </summary>
    private sealed class PrefixedReadStream : Stream
    {
        private readonly byte[] _prefix;
        private readonly int _prefixLen;
        private readonly Stream _inner;
        private int _prefixPos;

        public PrefixedReadStream(byte[] prefix, int prefixLen, Stream inner)
        {
            _prefix = prefix; _prefixLen = prefixLen; _inner = inner;
        }

        public override int Read(byte[] buffer, int offset, int count)
        {
            if (_prefixPos < _prefixLen)
            {
                int n = Math.Min(count, _prefixLen - _prefixPos);
                Array.Copy(_prefix, _prefixPos, buffer, offset, n);
                _prefixPos += n;
                return n;
            }
            return _inner.Read(buffer, offset, count);
        }

        // Browser HTTP/file streams reject synchronous Read (net_http_synchronous_reads_not_supported).
        // The base Stream.ReadAsync would fall back to sync Read on _inner, so override the async path
        // explicitly: serve the buffered prefix first, then delegate to the inner stream's ReadAsync.
        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            if (_prefixPos < _prefixLen)
            {
                int n = Math.Min(buffer.Length, _prefixLen - _prefixPos);
                _prefix.AsMemory(_prefixPos, n).CopyTo(buffer);
                _prefixPos += n;
                return n;
            }
            return await _inner.ReadAsync(buffer, cancellationToken).ConfigureAwait(false);
        }

        public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken)
            => ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();

        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void Flush() { }
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    /// <summary>
    /// SafeTensors stream inspection: reads only [8-byte header length] + [JSON header], then stops.
    /// The header carries every tensor's dtype/shape/data_offsets, so sizes are computed without
    /// touching the tensor-data section (which may be hundreds of GB).
    /// </summary>
    private static async Task<InspectionResult> InspectSafeTensorsStreamAsync(
        Stream stream, byte[] prefix, int prefixLen, CancellationToken ct)
    {
        if (prefixLen < 8) throw new InvalidOperationException("Stream too small for SafeTensors format");
        long headerSize = BitConverter.ToInt64(prefix, 0);
        if (headerSize <= 0 || headerSize > 1_000_000_000L)
            throw new InvalidOperationException($"Invalid SafeTensors header size: {headerSize}");

        long needed = 8 + headerSize;
        // SafeTensorsParser.Parse validates headerSize <= data.Length - 8, so a header-only buffer
        // (length == 8 + headerSize) is exactly sufficient; it never reads the tensor-data section.
        var headerBuf = new byte[needed];
        int have = (int)Math.Min(prefixLen, needed);
        Array.Copy(prefix, headerBuf, have);
        if (have < needed)
            await ReadExactAsync(stream, headerBuf, have, (int)(needed - have), ct).ConfigureAwait(false);

        var result = InspectSafeTensors(headerBuf);
        // FileSizeBytes from the header-only buffer would be wrong; report the true length when known.
        result.FileSizeBytes = stream.CanSeek ? stream.Length : 0;
        return result;
    }

    /// <summary>
    /// Prefix-tolerant SafeTensors probe: first 8 bytes are a plausible uint64 header length and the
    /// 9th byte is '{' (start of the header JSON). Unlike InferenceSession.DetectModelFormat this
    /// does NOT compare headerSize to the buffer length, so it correctly identifies a large
    /// SafeTensors from a short stream prefix.
    /// </summary>
    private static bool IsSafeTensorsPrefix(byte[] prefix, int prefixLen)
    {
        if (prefixLen < 9) return false;
        long headerSize = BitConverter.ToInt64(prefix, 0);
        return headerSize > 0 && headerSize <= 1_000_000_000L && prefix[8] == (byte)'{';
    }

    /// <summary>Read up to <paramref name="count"/> bytes; returns the number actually read (may be &lt; count at EOF).</summary>
    private static async Task<int> ReadUpToAsync(Stream s, byte[] buf, int offset, int count, CancellationToken ct)
    {
        int total = 0;
        while (total < count)
        {
            int n = await s.ReadAsync(buf.AsMemory(offset + total, count - total), ct).ConfigureAwait(false);
            if (n == 0) break;
            total += n;
        }
        return total;
    }

    /// <summary>Read EXACTLY <paramref name="count"/> bytes or throw on premature EOF.</summary>
    private static async Task ReadExactAsync(Stream s, byte[] buf, int offset, int count, CancellationToken ct)
    {
        int got = await ReadUpToAsync(s, buf, offset, count, ct).ConfigureAwait(false);
        if (got < count)
            throw new EndOfStreamException($"Expected {count} bytes, got {got} (truncated model stream).");
    }

    /// <summary>Concatenate the already-read prefix with the remainder of the stream into one byte[].</summary>
    private static async Task<byte[]> DrainWithPrefixAsync(Stream s, byte[] prefix, int prefixLen, CancellationToken ct)
    {
        using var ms = new MemoryStream();
        ms.Write(prefix, 0, prefixLen);
        await s.CopyToAsync(ms, ct).ConfigureAwait(false);
        return ms.ToArray();
    }

    /// <summary>
    /// Inspect a model file and return a structured summary.
    /// Auto-detects format (ONNX or TFLite) from magic bytes.
    /// </summary>
    public static InspectionResult Inspect(byte[] modelBytes)
    {
        var format = InferenceSession.DetectModelFormat(modelBytes);
        return format switch
        {
            ModelFormat.TFLite => InspectTFLite(modelBytes),
            ModelFormat.GGUF => InspectGGUF(modelBytes),
            ModelFormat.SafeTensors => InspectSafeTensors(modelBytes),
            ModelFormat.SPZ => InspectSPZ(modelBytes),
            ModelFormat.PLY => InspectPLY(modelBytes),
            ModelFormat.GLTF => InspectGLTF(modelBytes),
            ModelFormat.OBJ => InspectOBJ(modelBytes),
            _ => InspectOnnx(modelBytes),
        };
    }

    /// <summary>Inspect an ONNX model.</summary>
    public static InspectionResult InspectOnnx(byte[] onnxBytes)
    {
        var model = OnnxParser.Parse(onnxBytes);
        var graph = model.Graph;

        var opset = model.OpsetImports.FirstOrDefault(o => o.Domain == "")?.Version ?? 0;

        // Operator usage
        var opCounts = graph.Nodes
            .GroupBy(n => n.OpType)
            .OrderByDescending(g => g.Count())
            .Select(g => new OpUsage { OpType = g.Key, Count = g.Count() })
            .ToArray();

        // Weight statistics
        long totalParams = 0;
        long totalBytes = 0;
        var largestWeights = new List<WeightInfo>();

        foreach (var init in graph.Initializers)
        {
            long elements = init.ElementCount;
            long bytes = init.RawData?.Length ?? (elements * DataTypeSize(init.DataType));
            totalParams += elements;
            totalBytes += bytes;

            largestWeights.Add(new WeightInfo
            {
                Name = init.Name,
                Shape = init.Dims.Select(d => (int)d).ToArray(),
                Elements = elements,
                SizeBytes = bytes,
                DataType = DataTypeName(init.DataType),
            });
        }

        largestWeights.Sort((a, b) => b.SizeBytes.CompareTo(a.SizeBytes));

        // Inputs and outputs
        var inputs = graph.Inputs
            .Where(i => !graph.Initializers.Any(init => init.Name == i.Name))
            .Select(i => new TensorInfo
            {
                Name = i.Name,
                Shape = i.Shape.Select(d => d.ToString()).ToArray(),
                DataType = DataTypeName(i.ElemType),
            }).ToArray();

        var outputs = graph.Outputs.Select(o => new TensorInfo
        {
            Name = o.Name,
            Shape = o.Shape.Select(d => d.ToString()).ToArray(),
            DataType = DataTypeName(o.ElemType),
        }).ToArray();

        return new InspectionResult
        {
            GraphName = graph.Name,
            ProducerName = model.ProducerName,
            ProducerVersion = model.ProducerVersion,
            IrVersion = model.IrVersion,
            OpsetVersion = opset,
            NodeCount = graph.Nodes.Count,
            InitializerCount = graph.Initializers.Count,
            TotalParameters = totalParams,
            TotalWeightBytes = totalBytes,
            Operators = opCounts,
            Inputs = inputs,
            Outputs = outputs,
            LargestWeights = largestWeights.Take(20).ToArray(),
            FileSizeBytes = onnxBytes.Length,
        };
    }

    /// <summary>Inspect a TFLite model.</summary>
    public static InspectionResult InspectTFLite(byte[] tfliteBytes)
    {
        var model = TFLite.TFLiteParser.Parse(tfliteBytes);
        if (model.Subgraphs.Length == 0)
            return new InspectionResult { GraphName = "Empty TFLite model", FileSizeBytes = tfliteBytes.Length };

        var sg = model.Subgraphs[0];

        // Operator usage
        var opCounts = sg.Operators
            .Select(o => model.GetOperatorName(o.OpcodeIndex))
            .GroupBy(n => n)
            .OrderByDescending(g => g.Count())
            .Select(g => new OpUsage { OpType = g.Key, Count = g.Count() })
            .ToArray();

        // Weight statistics
        long totalParams = 0;
        long totalBytes = 0;
        var largestWeights = new List<WeightInfo>();
        for (int i = 0; i < sg.Tensors.Length; i++)
        {
            var tensor = sg.Tensors[i];
            var buffer = model.Buffers[tensor.BufferIndex];
            if (buffer.DataLength == 0) continue;

            long elems = 1;
            foreach (var d in tensor.Shape) elems *= d;
            totalParams += elems;
            totalBytes += buffer.DataLength;

            largestWeights.Add(new WeightInfo
            {
                Name = tensor.Name,
                Shape = tensor.Shape,
                Elements = elems,
                SizeBytes = buffer.DataLength,
                DataType = tensor.Type.ToString()
            });
        }
        largestWeights.Sort((a, b) => b.SizeBytes.CompareTo(a.SizeBytes));

        // Inputs/outputs
        var inputs = sg.Inputs.Where(i => model.Buffers[sg.Tensors[i].BufferIndex].DataLength == 0)
            .Select(i => new TensorInfo
            {
                Name = sg.Tensors[i].Name,
                Shape = sg.Tensors[i].Shape.Select(d => d.ToString()).ToArray(),
                DataType = sg.Tensors[i].Type.ToString()
            }).ToArray();

        var outputs = sg.Outputs.Select(i => new TensorInfo
        {
            Name = sg.Tensors[i].Name,
            Shape = sg.Tensors[i].Shape.Select(d => d.ToString()).ToArray(),
            DataType = sg.Tensors[i].Type.ToString()
        }).ToArray();

        return new InspectionResult
        {
            GraphName = model.Description.Length > 0 ? model.Description : "TFLite Model",
            ProducerName = "TensorFlow Lite",
            ProducerVersion = $"v{model.Version}",
            IrVersion = model.Version,
            NodeCount = sg.Operators.Length,
            InitializerCount = largestWeights.Count,
            TotalParameters = totalParams,
            TotalWeightBytes = totalBytes,
            Operators = opCounts,
            Inputs = inputs,
            Outputs = outputs,
            LargestWeights = largestWeights.Take(20).ToArray(),
            FileSizeBytes = tfliteBytes.Length,
        };
    }

    /// <summary>Inspect a GGUF model (LLM weights).</summary>
    public static InspectionResult InspectGGUF(byte[] ggufBytes)
        => BuildGGUFResult(GGUF.GGUFParser.Parse(ggufBytes), ggufBytes.Length);

    /// <summary>Build a GGUF InspectionResult from a parsed model (shared by byte[] and stream paths).</summary>
    private static InspectionResult BuildGGUFResult(GGUF.GGUFModel model, long fileSize)
    {
        // Count tensor types as "operators"
        var typeCounts = model.Tensors
            .GroupBy(t => t.Type.ToString())
            .OrderByDescending(g => g.Count())
            .Select(g => new OpUsage { OpType = g.Key, Count = g.Count() })
            .ToArray();

        long totalParams = 0;
        long totalBytes = 0;
        var largestWeights = new List<WeightInfo>();

        foreach (var tensor in model.Tensors)
        {
            long elems = model.GetTensorElementCount(tensor);
            long bytes = GGUF.GGMLTypes.TypeSize(tensor.Type, elems);
            totalParams += elems;
            totalBytes += bytes;

            largestWeights.Add(new WeightInfo
            {
                Name = tensor.Name,
                Shape = tensor.Shape,
                Elements = elems,
                SizeBytes = bytes,
                DataType = tensor.Type.ToString()
            });
        }
        largestWeights.Sort((a, b) => b.SizeBytes.CompareTo(a.SizeBytes));

        // Full metadata KV map. LargestWeights caps at 20 (all big quantized matmuls), so the
        // architecturally-decisive small tensors and the metadata that describes them (sliding-window
        // pattern, per-layer head_count_kv, dual RoPE base, logit soft-cap) would otherwise be invisible.
        // Array values (e.g. 262K-token tokenizer lists) are summarized so they never bloat the result.
        var metadata = model.Metadata
            .OrderBy(kv => kv.Key, StringComparer.Ordinal)
            .Select(kv => new MetadataEntry { Key = kv.Key, Value = FormatMetadataValue(kv.Value) })
            .ToArray();

        // Tensor-name templates with blk.N collapsed to blk.* — surfaces EVERY distinct tensor shape
        // (incl. the small norms/scales hidden by LargestWeights.Take(20)) without listing all N hundred.
        // Group by (collapsed-name, SHAPE, dtype) — NOT name alone: a `blk.*` name whose layers carry
        // DIFFERENT shapes (e.g. gemma4 attn_q is [3840,4096] on sliding layers but [3840,8192] on the 8
        // global layers) must split into one row PER distinct shape. Collapsing on name and keeping only
        // First()'s shape silently hid that variance and made the gemma4 global-layer geometry invisible.
        var templates = model.Tensors
            .GroupBy(t => (Name: CollapseBlock(t.Name), Shape: string.Join(",", t.Shape), Type: t.Type.ToString()))
            .Select(g => new TensorTemplate
            {
                Name = g.Key.Name,
                DataType = g.Key.Type,
                ExampleShape = g.First().Shape,
                Count = g.Count(),
            })
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .ThenByDescending(t => t.Count) // same name, distinct shapes: most-common variant first
            .ToArray();

        return new InspectionResult
        {
            GraphName = $"{model.Name} ({model.Architecture})",
            ProducerName = "GGUF / llama.cpp",
            ProducerVersion = $"v{model.Version}",
            IrVersion = model.Version,
            OpsetVersion = model.ContextLength,
            NodeCount = model.Tensors.Length,
            InitializerCount = model.Tensors.Length,
            TotalParameters = totalParams,
            TotalWeightBytes = totalBytes,
            Operators = typeCounts,
            Inputs = new[] { new TensorInfo
            {
                Name = "Architecture",
                Shape = new[] { model.Architecture, $"{model.BlockCount} layers", $"{model.EmbeddingLength}d", $"{model.AttentionHeadCount} heads" },
                DataType = $"ctx={model.ContextLength}"
            }},
            Outputs = new[] { new TensorInfo
            {
                Name = "Vocab",
                Shape = new[] { $"{model.VocabSize} tokens" },
                DataType = "text"
            }},
            LargestWeights = largestWeights.Take(20).ToArray(),
            Metadata = metadata,
            TensorTemplates = templates,
            FileSizeBytes = fileSize,
        };
    }

    /// <summary>Collapse a per-layer tensor name's block index so all layers share one template
    /// (e.g. "blk.7.attn_q_norm.weight" → "blk.*.attn_q_norm.weight").</summary>
    private static string CollapseBlock(string name)
        => System.Text.RegularExpressions.Regex.Replace(name, @"\bblk\.\d+\.", "blk.*.");

    /// <summary>Render a GGUF metadata value for display. Scalars/strings pass through (long strings
    /// truncated); arrays are summarized as "[N × elemType] {first few…}" so a 262K-entry tokenizer
    /// list never materializes into the inspection output.</summary>
    private static string FormatMetadataValue(object? v)
    {
        if (v is null) return "null";
        if (v is string s) return s.Length > 120 ? s[..120] + "…" : s;
        if (v is Array arr)
        {
            // Boxed arrays (bool[]/int[] come through as object[]) report element type "Object" — infer the
            // real element type from the first non-null entry so the reader sees "[48 × bool]" not "[48 × Object]".
            var declared = arr.GetType().GetElementType();
            var elemType = declared != null && declared != typeof(object)
                ? declared
                : arr.Cast<object?>().FirstOrDefault(x => x != null)?.GetType();
            var shown = arr.Cast<object?>().Take(8).Select(FormatMetadataElement);
            return $"[{arr.Length} × {FriendlyTypeName(elemType)}] {{{string.Join(", ", shown)}{(arr.Length > 8 ? ", …" : "")}}}";
        }
        return v.ToString() ?? "";
    }

    private static string FriendlyTypeName(Type? t) => t?.Name switch
    {
        null => "?",
        "Boolean" => "bool", "String" => "str", "Single" => "f32", "Double" => "f64",
        "Byte" => "u8", "SByte" => "i8", "Int16" => "i16", "UInt16" => "u16",
        "Int32" => "i32", "UInt32" => "u32", "Int64" => "i64", "UInt64" => "u64",
        var n => n,
    };

    private static string FormatMetadataElement(object? e)
    {
        if (e is null) return "null";
        if (e is string s) return "\"" + (s.Length > 24 ? s[..24] + "…" : s) + "\"";
        return e.ToString() ?? "";
    }

    /// <summary>Inspect a SafeTensors file (weights only, no graph).</summary>
    public static InspectionResult InspectSafeTensors(byte[] stBytes)
    {
        var file = SafeTensors.SafeTensorsParser.Parse(stBytes);

        long totalParams = 0;
        long totalBytes = 0;
        var dtypeCounts = file.Tensors
            .GroupBy(t => t.DType)
            .OrderByDescending(g => g.Count())
            .Select(g => new OpUsage { OpType = g.Key, Count = g.Count() })
            .ToArray();

        var largestWeights = new List<WeightInfo>();
        foreach (var tensor in file.Tensors)
        {
            long elems = tensor.Shape.Aggregate(1L, (a, b) => a * b);
            totalParams += elems;
            totalBytes += tensor.DataLength;
            largestWeights.Add(new WeightInfo
            {
                Name = tensor.Name,
                Shape = tensor.Shape,
                Elements = elems,
                SizeBytes = tensor.DataLength,
                DataType = tensor.DType
            });
        }
        largestWeights.Sort((a, b) => b.SizeBytes.CompareTo(a.SizeBytes));

        return new InspectionResult
        {
            GraphName = "SafeTensors (weights only)",
            ProducerName = "HuggingFace SafeTensors",
            NodeCount = 0, // no graph
            InitializerCount = file.Tensors.Length,
            TotalParameters = totalParams,
            TotalWeightBytes = totalBytes,
            Operators = dtypeCounts,
            Inputs = Array.Empty<TensorInfo>(),
            Outputs = Array.Empty<TensorInfo>(),
            LargestWeights = largestWeights.Take(20).ToArray(),
            FileSizeBytes = stBytes.Length,
        };
    }

    // ═══════════════════════════════════════════════════════════
    //  3D Format Inspectors
    // ═══════════════════════════════════════════════════════════

    public static InspectionResult InspectSPZ(byte[] spzBytes)
    {
        var cloud = Formats.SPZParser.Parse(spzBytes);
        return new InspectionResult
        {
            GraphName = $"SPZ Gaussian Splat ({cloud.NumPoints:N0} points)",
            ProducerName = $"SPZ v{cloud.Version}",
            NodeCount = 0,
            InitializerCount = cloud.NumPoints,
            TotalParameters = cloud.NumPoints * 14, // pos(3)+alpha(1)+color(3)+scale(3)+rot(4)
            TotalWeightBytes = spzBytes.Length,
            Inputs = new[] { new TensorInfo { Name = "gaussians", Shape = new[] { cloud.NumPoints.ToString(), "14" } } },
            Outputs = Array.Empty<TensorInfo>(),
            Operators = Array.Empty<OpUsage>(),
            LargestWeights = Array.Empty<WeightInfo>(),
        };
    }

    public static InspectionResult InspectPLY(byte[] plyBytes)
    {
        var ply = Formats.PLYParser.Parse(plyBytes);
        bool isGaussian = ply.Gaussians != null;
        return new InspectionResult
        {
            GraphName = isGaussian ? $"PLY Gaussian Splat ({ply.VertexCount:N0} points)" : $"PLY Mesh ({ply.VertexCount:N0} vertices, {ply.FaceCount} faces)",
            ProducerName = $"PLY {ply.Format}",
            NodeCount = 0,
            InitializerCount = ply.VertexCount,
            TotalParameters = ply.VertexCount * ply.Properties.Length,
            TotalWeightBytes = plyBytes.Length,
            Inputs = new[] { new TensorInfo { Name = "vertices", Shape = new[] { ply.VertexCount.ToString(), ply.Properties.Length.ToString() } } },
            Outputs = Array.Empty<TensorInfo>(),
            Operators = ply.Properties.Select(p => new OpUsage { OpType = p, Count = ply.VertexCount }).ToArray(),
            LargestWeights = Array.Empty<WeightInfo>(),
        };
    }

    public static InspectionResult InspectGLTF(byte[] glbBytes)
    {
        var mesh = Formats.GLTFLoader.LoadGLB(glbBytes);
        return new InspectionResult
        {
            GraphName = $"glTF Mesh ({mesh.VertexCount:N0} vertices, {mesh.TriangleCount:N0} triangles)",
            ProducerName = "glTF 2.0",
            NodeCount = mesh.TriangleCount,
            InitializerCount = mesh.VertexCount,
            TotalParameters = mesh.VertexCount * 3,
            TotalWeightBytes = glbBytes.Length,
            Inputs = new[] { new TensorInfo { Name = "vertices", Shape = new[] { mesh.VertexCount.ToString(), "3" } } },
            Outputs = new[] { new TensorInfo { Name = "triangles", Shape = new[] { mesh.TriangleCount.ToString(), "3" } } },
            Operators = Array.Empty<OpUsage>(),
            LargestWeights = Array.Empty<WeightInfo>(),
        };
    }

    public static InspectionResult InspectOBJ(byte[] objBytes)
    {
        var mesh = Formats.OBJExporter.Load(objBytes);
        return new InspectionResult
        {
            GraphName = $"OBJ Mesh ({mesh.VertexCount:N0} vertices, {mesh.TriangleCount:N0} triangles)",
            ProducerName = "Wavefront OBJ",
            NodeCount = mesh.TriangleCount,
            InitializerCount = mesh.VertexCount,
            TotalParameters = mesh.VertexCount * 3,
            TotalWeightBytes = objBytes.Length,
            Inputs = new[] { new TensorInfo { Name = "vertices", Shape = new[] { mesh.VertexCount.ToString(), "3" } } },
            Outputs = new[] { new TensorInfo { Name = "triangles", Shape = new[] { mesh.TriangleCount.ToString(), "3" } } },
            Operators = Array.Empty<OpUsage>(),
            LargestWeights = Array.Empty<WeightInfo>(),
        };
    }

    /// <summary>
    /// Check which operators the model needs that we support vs don't support.
    /// Answers "Can I run this model?" instantly. Operator-level compatibility is an ONNX-graph
    /// concept: GGUF/SafeTensors are weights only (no executable graph) and TFLite uses a different
    /// operator vocabulary, so for non-ONNX formats this returns a non-applicable result instead of
    /// throwing (the Model Inspector demo calls this for every dropped file).
    /// </summary>
    public static CompatibilityResult CheckCompatibility(byte[] modelBytes, Operators.OperatorRegistry? registry = null)
    {
        var format = InferenceSession.DetectModelFormat(modelBytes);
        if (format != ModelFormat.ONNX)
            return NonApplicableCompatibility(format);

        var model = OnnxParser.Parse(modelBytes);
        var opsUsed = model.Graph.Nodes.Select(n => n.OpType).Distinct().OrderBy(o => o).ToArray();
        return PartitionCompatibility(opsUsed, registry);
    }

    /// <summary>Compatibility result for a non-ONNX (weights-only / different-vocabulary) format.</summary>
    private static CompatibilityResult NonApplicableCompatibility(ModelFormat format) => new()
    {
        Format = format.ToString(),
        Applicable = false,
        TotalOpsUsed = 0,
        SupportedOps = Array.Empty<string>(),
        UnsupportedOps = Array.Empty<string>(),
        IsFullySupported = true,
        CompatibilityPercent = 100,
    };

    /// <summary>Operators the engine supports when no live registry is supplied. Single source of truth
    /// for both the byte[] and streaming compatibility checks.
    ///
    /// This is NOT a hand-maintained list — it points at <see cref="Operators.OperatorRegistry.BuiltinOpTypes"/>,
    /// the single source of truth for every registered op. A drift test keeps that manifest locked to the
    /// live registry, so the inspector can never again under-report support the way it did when this was a
    /// stale local copy (GPT-2 falsely showed 90% because And/IsNaN/LessOrEqual were registered but absent here).</summary>
    private static readonly IReadOnlySet<string> KnownSupportedOps = Operators.OperatorRegistry.BuiltinOpTypes;

    /// <summary>Partition a distinct, sorted op-type list into supported/unsupported against the
    /// registry (if provided) or the built-in known-supported set.</summary>
    private static CompatibilityResult PartitionCompatibility(string[] opsUsed, Operators.OperatorRegistry? registry)
    {
        IReadOnlySet<string> supportedOps;
        if (registry != null)
        {
            var live = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var op in opsUsed)
                if (registry.IsSupported(op)) live.Add(op);
            supportedOps = live;
        }
        else
        {
            supportedOps = KnownSupportedOps;
        }

        var supported = opsUsed.Where(o => supportedOps.Contains(o)).ToArray();
        var unsupported = opsUsed.Where(o => !supportedOps.Contains(o)).ToArray();

        return new CompatibilityResult
        {
            TotalOpsUsed = opsUsed.Length,
            SupportedOps = supported,
            UnsupportedOps = unsupported,
            IsFullySupported = unsupported.Length == 0,
            CompatibilityPercent = opsUsed.Length > 0
                ? (float)supported.Length / opsUsed.Length * 100
                : 100,
        };
    }

    private static string DataTypeName(int dt) => dt switch
    {
        0 => "undefined", 1 => "float32", 2 => "uint8", 3 => "int8",
        4 => "uint16", 5 => "int16", 6 => "int32", 7 => "int64",
        8 => "string", 9 => "bool", 10 => "float16", 11 => "float64",
        12 => "uint32", 13 => "uint64", 16 => "bfloat16",
        _ => $"type_{dt}",
    };

    private static long DataTypeSize(int dt) => dt switch
    {
        1 => 4, 2 => 1, 3 => 1, 4 => 2, 5 => 2, 6 => 4, 7 => 8,
        9 => 1, 10 => 2, 11 => 8, 12 => 4, 13 => 8, 16 => 2,
        _ => 4,
    };
}

public class InspectionResult
{
    public string GraphName { get; set; } = "";
    public string ProducerName { get; set; } = "";
    public string ProducerVersion { get; set; } = "";
    public long IrVersion { get; set; }
    public long OpsetVersion { get; set; }
    public int NodeCount { get; set; }
    public int InitializerCount { get; set; }
    public long TotalParameters { get; set; }
    public long TotalWeightBytes { get; set; }
    public long FileSizeBytes { get; set; }
    public OpUsage[] Operators { get; set; } = Array.Empty<OpUsage>();
    public TensorInfo[] Inputs { get; set; } = Array.Empty<TensorInfo>();
    public TensorInfo[] Outputs { get; set; } = Array.Empty<TensorInfo>();
    public WeightInfo[] LargestWeights { get; set; } = Array.Empty<WeightInfo>();

    /// <summary>Full key/value metadata map. Populated for header formats that carry one (GGUF today);
    /// empty otherwise. Array values are summarized so large embedded arrays (tokenizer lists) never
    /// bloat the result. This is where frontier-arch detail lives (sliding-window pattern, per-layer
    /// head_count_kv, dual RoPE base, logit soft-cap) that LargestWeights cannot show.</summary>
    public MetadataEntry[] Metadata { get; set; } = Array.Empty<MetadataEntry>();

    /// <summary>Distinct tensor-name templates (per-layer block index collapsed to blk.*). Populated for
    /// GGUF; empty otherwise. Surfaces every tensor SHAPE including the small norms/scales hidden by the
    /// LargestWeights top-20 cap, without enumerating all N hundred tensors.</summary>
    public TensorTemplate[] TensorTemplates { get; set; } = Array.Empty<TensorTemplate>();

    public string TotalParametersFormatted => TotalParameters switch
    {
        >= 1_000_000_000 => $"{TotalParameters / 1_000_000_000.0:F1}B",
        >= 1_000_000 => $"{TotalParameters / 1_000_000.0:F1}M",
        >= 1_000 => $"{TotalParameters / 1_000.0:F1}K",
        _ => TotalParameters.ToString(),
    };

    public string TotalWeightMB => $"{TotalWeightBytes / 1024.0 / 1024.0:F1} MB";
    public string FileSizeMB => $"{FileSizeBytes / 1024.0 / 1024.0:F1} MB";
}

public class OpUsage
{
    public string OpType { get; set; } = "";
    public int Count { get; set; }
}

public class TensorInfo
{
    public string Name { get; set; } = "";
    public string[] Shape { get; set; } = Array.Empty<string>();
    public string DataType { get; set; } = "";
    public string ShapeStr => $"[{string.Join(", ", Shape)}]";
}

public class WeightInfo
{
    public string Name { get; set; } = "";
    public int[] Shape { get; set; } = Array.Empty<int>();
    public long Elements { get; set; }
    public long SizeBytes { get; set; }
    public string DataType { get; set; } = "";
    public string ShapeStr => $"[{string.Join(", ", Shape)}]";
    public string SizeFormatted => SizeBytes switch
    {
        >= 1_048_576 => $"{SizeBytes / 1048576.0:F1} MB",
        >= 1024 => $"{SizeBytes / 1024.0:F1} KB",
        _ => $"{SizeBytes} B",
    };
}

/// <summary>One GGUF metadata key/value pair (array values pre-summarized for display).</summary>
public class MetadataEntry
{
    public string Key { get; set; } = "";
    public string Value { get; set; } = "";
}

/// <summary>A distinct tensor-name template (per-layer block index collapsed to blk.*) with an example
/// shape/dtype and how many tensors share it.</summary>
public class TensorTemplate
{
    public string Name { get; set; } = "";
    public string DataType { get; set; } = "";
    public int[] ExampleShape { get; set; } = Array.Empty<int>();
    public int Count { get; set; }
    public string ShapeStr => $"[{string.Join(", ", ExampleShape)}]";
}

/// <summary>Result of checking model compatibility with our engine.</summary>
public class CompatibilityResult
{
    public int TotalOpsUsed { get; set; }
    public string[] SupportedOps { get; set; } = Array.Empty<string>();
    public string[] UnsupportedOps { get; set; } = Array.Empty<string>();
    public bool IsFullySupported { get; set; }
    public float CompatibilityPercent { get; set; }

    /// <summary>Detected model format ("ONNX", "TFLite", ...). Set for all results.</summary>
    public string Format { get; set; } = "ONNX";
    /// <summary>True when operator-level compatibility applies (ONNX graphs). False for
    /// weights-only / non-ONNX formats, where the op check is not meaningful.</summary>
    public bool Applicable { get; set; } = true;

    public string Summary => !Applicable
        ? $"{Format} model — operator compatibility check applies to ONNX graphs"
        : IsFullySupported
            ? $"Fully compatible ({TotalOpsUsed} operators supported)"
            : $"{CompatibilityPercent:F0}% compatible ({SupportedOps.Length}/{TotalOpsUsed} operators). Missing: {string.Join(", ", UnsupportedOps)}";
}
