using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace SpawnDev.ILGPU.ML.Onnx;

/// <summary>
/// Streaming, structure-only ONNX inspection. Walks the protobuf graph (nodes, initializer metadata,
/// inputs, outputs) directly from a <see cref="Stream"/> while SKIPPING every tensor's raw_data weight
/// blob — so a 600 MB GPT-2 or a multi-GB Llama inspects from a few hundred KB of structure, never
/// holding the weights in memory. Weight byte-size is computed from dims × dtype (or the recorded
/// raw_data length), exactly as the in-memory <see cref="InspectOnnx"/> path does.
/// </summary>
public static partial class ModelInspectorHelper
{
    /// <summary>
    /// Operator-compatibility check from a stream, without materializing weights. Mirrors
    /// <see cref="CheckCompatibility(byte[], Operators.OperatorRegistry?)"/> but reads only the graph
    /// structure (op types) from the stream — so a 600 MB GPT-2's compatibility is checked without
    /// ever loading its weights. Non-ONNX formats return a non-applicable result (op-level
    /// compatibility is an ONNX-graph concept), matching the byte[] overload.
    /// </summary>
    public static async Task<CompatibilityResult> CheckCompatibilityAsync(
        Stream stream, Operators.OperatorRegistry? registry = null, CancellationToken ct = default)
    {
        if (stream == null) throw new ArgumentNullException(nameof(stream));

        var prefix = new byte[DetectPrefixBytes];
        int prefixLen = await ReadUpToAsync(stream, prefix, 0, prefix.Length, ct).ConfigureAwait(false);

        ModelFormat format;
        if (IsSafeTensorsPrefix(prefix, prefixLen))
        {
            format = ModelFormat.SafeTensors;
        }
        else
        {
            var detectBuf = prefixLen == prefix.Length ? prefix : prefix.AsSpan(0, prefixLen).ToArray();
            format = InferenceSession.DetectModelFormat(detectBuf);
        }

        if (format != ModelFormat.ONNX)
            return NonApplicableCompatibility(format);

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
        var inspection = await InspectOnnxStreamAsync(reader, stream.CanSeek ? stream.Length : 0, ct).ConfigureAwait(false);
        var opsUsed = inspection.Operators.Select(o => o.OpType).Distinct().OrderBy(o => o).ToArray();
        return PartitionCompatibility(opsUsed, registry);
    }

    /// <summary>
    /// Inspect an ONNX model from a stream without materializing weights. Uses <see cref="Stream.Seek"/>
    /// to bypass raw_data when the stream is seekable, otherwise a bounded read-and-discard.
    /// </summary>
    private static async Task<InspectionResult> InspectOnnxStreamAsync(
        StreamProtoReader r, long fileSizeBytes, CancellationToken ct)
    {
        long irVersion = 0;
        string producerName = "", producerVersion = "";
        var opsets = new List<OnnxOpsetImport>();
        InspectionResult? graphResult = null;

        // ── ModelProto (top level, bounded by EOF) ──
        while (await r.HasMoreAsync().ConfigureAwait(false))
        {
            var (field, wire) = await r.ReadTagAsync().ConfigureAwait(false);
            switch (field)
            {
                case 1: irVersion = (long)await r.ReadVarintAsync().ConfigureAwait(false); break;  // ir_version
                case 2: producerName = await r.ReadStringAsync().ConfigureAwait(false); break;     // producer_name
                case 3: producerVersion = await r.ReadStringAsync().ConfigureAwait(false); break;  // producer_version
                case 7:                                                                            // graph
                {
                    long graphLen = (long)await r.ReadVarintAsync().ConfigureAwait(false);
                    graphResult = await ParseGraphStreamAsync(r, graphLen, ct).ConfigureAwait(false);
                    break;
                }
                case 8:                                                                            // opset_import
                {
                    int opLen = checked((int)await r.ReadVarintAsync().ConfigureAwait(false));
                    var opBytes = await r.ReadBytesAsync(opLen).ConfigureAwait(false);
                    opsets.Add(ParseOpsetImport(opBytes));
                    break;
                }
                default: await r.SkipFieldAsync(wire).ConfigureAwait(false); break;
            }
        }

        var result = graphResult ?? new InspectionResult();
        result.IrVersion = irVersion;
        result.ProducerName = producerName;
        result.ProducerVersion = producerVersion;
        result.OpsetVersion = opsets.FirstOrDefault(o => o.Domain == "")?.Version ?? 0;
        // For a non-seekable stream we read-through (and discard) the entire file, so Position is the
        // true byte length; for a seekable stream the caller passes the authoritative Length.
        result.FileSizeBytes = fileSizeBytes > 0 ? fileSizeBytes : r.Position;
        return result;
    }

    /// <summary>Parse a GraphProto submessage (bounded by <paramref name="graphLen"/>) into an
    /// <see cref="InspectionResult"/>, skipping all weight data.</summary>
    private static async Task<InspectionResult> ParseGraphStreamAsync(
        StreamProtoReader r, long graphLen, CancellationToken ct)
    {
        long end = r.Position + graphLen;
        string graphName = "";
        var opCounts = new Dictionary<string, int>();
        int nodeCount = 0;
        long totalParams = 0, totalBytes = 0;
        var weights = new List<WeightInfo>();
        var initializerNames = new HashSet<string>();
        var inputs = new List<TensorInfo>();
        var outputs = new List<TensorInfo>();

        while (r.Position < end)
        {
            var (field, wire) = await r.ReadTagAsync().ConfigureAwait(false);
            switch (field)
            {
                case 1:                                                                    // node (repeated)
                {
                    long len = (long)await r.ReadVarintAsync().ConfigureAwait(false);
                    string opType = await ParseNodeOpTypeStreamAsync(r, len).ConfigureAwait(false);
                    nodeCount++;
                    if (opType.Length > 0)
                        opCounts[opType] = opCounts.TryGetValue(opType, out var c) ? c + 1 : 1;
                    break;
                }
                case 2: graphName = await r.ReadStringAsync().ConfigureAwait(false); break;  // name
                case 5:                                                                     // initializer — WEIGHTS
                {
                    long len = (long)await r.ReadVarintAsync().ConfigureAwait(false);
                    var (name, dims, dtype, rawLen) = await ParseInitializerMetaStreamAsync(r, len).ConfigureAwait(false);
                    long elements = ElementsOf(dims);
                    long bytes = rawLen >= 0 ? rawLen : elements * DataTypeSize(dtype);
                    totalParams += elements;
                    totalBytes += bytes;
                    initializerNames.Add(name);
                    weights.Add(new WeightInfo
                    {
                        Name = name,
                        Shape = Array.ConvertAll(dims, d => (int)d),
                        Elements = elements,
                        SizeBytes = bytes,
                        DataType = DataTypeName(dtype),
                    });
                    break;
                }
                case 11:                                                                    // input (repeated)
                {
                    int len = checked((int)await r.ReadVarintAsync().ConfigureAwait(false));
                    var b = await r.ReadBytesAsync(len).ConfigureAwait(false);
                    inputs.Add(ParseValueInfo(b));
                    break;
                }
                case 12:                                                                    // output (repeated)
                {
                    int len = checked((int)await r.ReadVarintAsync().ConfigureAwait(false));
                    var b = await r.ReadBytesAsync(len).ConfigureAwait(false);
                    outputs.Add(ParseValueInfo(b));
                    break;
                }
                default: await r.SkipFieldAsync(wire).ConfigureAwait(false); break;
            }
        }

        weights.Sort((a, b) => b.SizeBytes.CompareTo(a.SizeBytes));
        var operators = opCounts
            .OrderByDescending(kv => kv.Value)
            .Select(kv => new OpUsage { OpType = kv.Key, Count = kv.Value })
            .ToArray();

        // Graph inputs exclude those that are also initializers (constants), matching InspectOnnx.
        var realInputs = inputs.Where(i => !initializerNames.Contains(i.Name)).ToArray();

        return new InspectionResult
        {
            GraphName = graphName,
            NodeCount = nodeCount,
            InitializerCount = weights.Count,
            TotalParameters = totalParams,
            TotalWeightBytes = totalBytes,
            Operators = operators,
            Inputs = realInputs,
            Outputs = outputs.ToArray(),
            LargestWeights = weights.Take(20).ToArray(),
        };
    }

    /// <summary>Parse a NodeProto submessage extracting only op_type (field 4); all else is skipped —
    /// including any large embedded Constant-tensor attributes, which are bypassed via raw skips.</summary>
    private static async Task<string> ParseNodeOpTypeStreamAsync(StreamProtoReader r, long len)
    {
        long end = r.Position + len;
        string opType = "";
        while (r.Position < end)
        {
            var (field, wire) = await r.ReadTagAsync().ConfigureAwait(false);
            if (field == 4 && wire == 2)
                opType = await r.ReadStringAsync().ConfigureAwait(false);   // op_type
            else
                await r.SkipFieldAsync(wire).ConfigureAwait(false);
        }
        return opType;
    }

    /// <summary>Parse a TensorProto initializer for inspection metadata only: name, dims, data_type, and
    /// the raw_data BYTE LENGTH (the bytes themselves are skipped, never read). Packed numeric data
    /// fields (float_data/int32_data/...) are also skipped — element count comes from dims.</summary>
    private static async Task<(string Name, long[] Dims, int DataType, long RawLen)> ParseInitializerMetaStreamAsync(
        StreamProtoReader r, long len)
    {
        long end = r.Position + len;
        var dims = new List<long>();
        int dtype = 0;
        string name = "";
        long rawLen = -1;

        while (r.Position < end)
        {
            var (field, wire) = await r.ReadTagAsync().ConfigureAwait(false);
            switch (field)
            {
                case 1:                                                         // dims (packed or single)
                    if (wire == 2)
                    {
                        int plen = checked((int)await r.ReadVarintAsync().ConfigureAwait(false));
                        var pb = await r.ReadBytesAsync(plen).ConfigureAwait(false);
                        var pr = new ProtobufReader(pb);
                        while (pr.HasMore) dims.Add(pr.ReadInt64());
                    }
                    else
                    {
                        dims.Add((long)await r.ReadVarintAsync().ConfigureAwait(false));
                    }
                    break;
                case 2: dtype = (int)await r.ReadVarintAsync().ConfigureAwait(false); break;  // data_type
                case 8: name = await r.ReadStringAsync().ConfigureAwait(false); break;        // name
                case 9:                                                                       // raw_data — SKIP
                {
                    rawLen = (long)await r.ReadVarintAsync().ConfigureAwait(false);
                    await r.SkipAsync(rawLen).ConfigureAwait(false);
                    break;
                }
                default: await r.SkipFieldAsync(wire).ConfigureAwait(false); break;
            }
        }
        return (name, dims.ToArray(), dtype, rawLen);
    }

    /// <summary>Parse a small ValueInfoProto (name + tensor type + shape) from a buffered byte[].</summary>
    private static TensorInfo ParseValueInfo(byte[] bytes)
    {
        var r = new ProtobufReader(bytes);
        string name = "";
        int elemType = 0;
        var shape = new List<string>();

        while (r.HasMore)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1: name = r.ReadString(); break;          // name
                case 2:                                         // type (TypeProto)
                    var typeSub = r.ReadSubMessage();
                    ParseTypeForValueInfo(ref typeSub, ref elemType, shape);
                    break;
                default: r.SkipField(wire); break;
            }
        }
        return new TensorInfo
        {
            Name = name,
            DataType = DataTypeName(elemType),
            Shape = shape.ToArray(),
        };
    }

    private static void ParseTypeForValueInfo(ref ProtobufReader r, ref int elemType, List<string> shape)
    {
        while (r.HasMore)
        {
            var (field, wire) = r.ReadTag();
            if (field == 1) // tensor_type
            {
                var tt = r.ReadSubMessage();
                while (tt.HasMore)
                {
                    var (f2, w2) = tt.ReadTag();
                    switch (f2)
                    {
                        case 1: elemType = tt.ReadInt32(); break;   // elem_type
                        case 2:                                      // shape
                            var sh = tt.ReadSubMessage();
                            while (sh.HasMore)
                            {
                                var (f3, w3) = sh.ReadTag();
                                if (f3 == 1) // dim
                                {
                                    var dim = sh.ReadSubMessage();
                                    shape.Add(ParseDimString(ref dim));
                                }
                                else sh.SkipField(w3);
                            }
                            break;
                        default: tt.SkipField(w2); break;
                    }
                }
            }
            else r.SkipField(wire);
        }
    }

    private static string ParseDimString(ref ProtobufReader r)
    {
        long? dimValue = null;
        string? dimParam = null;
        while (r.HasMore)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1: dimValue = r.ReadInt64(); break;    // dim_value
                case 2: dimParam = r.ReadString(); break;   // dim_param
                default: r.SkipField(wire); break;
            }
        }
        return dimValue.HasValue ? dimValue.Value.ToString() : (dimParam ?? "?");
    }

    private static OnnxOpsetImport ParseOpsetImport(byte[] bytes)
    {
        var r = new ProtobufReader(bytes);
        var op = new OnnxOpsetImport();
        while (r.HasMore)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1: op.Domain = r.ReadString(); break;   // domain
                case 2: op.Version = r.ReadInt64(); break;   // version
                default: r.SkipField(wire); break;
            }
        }
        return op;
    }

    private static long ElementsOf(long[] dims)
    {
        long n = 1;
        foreach (var d in dims) n *= d;
        return n;
    }
}
