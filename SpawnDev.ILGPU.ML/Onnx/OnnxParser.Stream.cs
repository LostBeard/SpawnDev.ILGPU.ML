using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace SpawnDev.ILGPU.ML.Onnx;

/// <summary>
/// Streaming ONNX parse for LOADING (execution), companion to the byte[] <see cref="OnnxParser.Parse(byte[])"/>.
/// Produces the SAME <see cref="OnnxModelProto"/> the execution path consumes, but the large weight blobs
/// (initializer raw_data) are NOT held in memory — instead each records its absolute byte offset in the
/// load stream (<see cref="OnnxTensorProto.RawDataStreamOffset"/>). The caller then seeks + uploads each
/// weight to the GPU in chunks (<c>BufferPool.AllocatePermanentFromStreamAsync</c>), so a multi-GB model
/// loads with a CPU peak of one chunk, and a peer can SEEK to only the tensors of its shard.
///
/// Small structural submessages (nodes, value_info, opset, and small inline-data initializers) are read
/// whole and parsed by the EXACT byte[] sub-parsers used by the in-memory path — so the execution graph is
/// identical and there is no parser divergence. Only large initializers are parsed field-by-field over the
/// stream to capture the raw_data offset. Requires a SEEKABLE stream (the loader seeks back to weights).
/// </summary>
public static partial class OnnxParser
{
    /// <summary>
    /// Parse an ONNX model's structure from a seekable stream, recording each large initializer's raw_data
    /// stream offset instead of materializing it. Initializers whose submessage is &lt;= <paramref name="streamThreshold"/>
    /// bytes are read whole (parsed by the byte[] path); larger ones record an offset and skip the weight.
    /// </summary>
    public static async Task<OnnxModelProto> ParseFromStreamAsync(
        Stream stream, int streamThreshold = 1024 * 1024, CancellationToken ct = default)
    {
        if (stream == null) throw new ArgumentNullException(nameof(stream));
        if (!stream.CanSeek)
            throw new NotSupportedException(
                "Streaming ONNX load requires a seekable stream (the loader seeks back to each weight). " +
                "Use a seekable source (TorrentReadStream, HTTP-Range/Blob-backed stream, or MemoryStream).");

        // The recorded weight offsets are StreamProtoReader.Position values, which are absolute file
        // offsets only if the reader starts at position 0 — ensure it (the upload pass Seeks to them).
        if (stream.Position != 0) stream.Seek(0, SeekOrigin.Begin);

        var r = new StreamProtoReader(stream, ct);
        var model = new OnnxModelProto();

        // ── ModelProto (top level, bounded by EOF) ──
        while (await r.HasMoreAsync().ConfigureAwait(false))
        {
            var (field, wire) = await r.ReadTagAsync().ConfigureAwait(false);
            switch (field)
            {
                case 1: model.IrVersion = (long)await r.ReadVarintAsync().ConfigureAwait(false); break;       // ir_version
                case 2: model.ProducerName = await r.ReadStringAsync().ConfigureAwait(false); break;          // producer_name
                case 3: model.ProducerVersion = await r.ReadStringAsync().ConfigureAwait(false); break;       // producer_version
                case 4: model.Domain = await r.ReadStringAsync().ConfigureAwait(false); break;                // domain
                case 5: model.ModelVersion = (long)await r.ReadVarintAsync().ConfigureAwait(false); break;    // model_version
                case 7:                                                                                       // graph
                {
                    long graphLen = (long)await r.ReadVarintAsync().ConfigureAwait(false);
                    model.Graph = await ParseGraphFromStreamAsync(r, graphLen, streamThreshold).ConfigureAwait(false);
                    break;
                }
                case 8:                                                                                       // opset_import
                {
                    int opLen = checked((int)await r.ReadVarintAsync().ConfigureAwait(false));
                    var opBytes = await r.ReadBytesAsync(opLen).ConfigureAwait(false);
                    model.OpsetImports.Add(ParseOpsetFromBytes(opBytes));
                    break;
                }
                default: await r.SkipFieldAsync(wire).ConfigureAwait(false); break;
            }
        }
        return model;
    }

    private static async Task<OnnxGraphProto> ParseGraphFromStreamAsync(
        StreamProtoReader r, long graphLen, int streamThreshold)
    {
        long end = r.Position + graphLen;
        var graph = new OnnxGraphProto();

        while (r.Position < end)
        {
            var (field, wire) = await r.ReadTagAsync().ConfigureAwait(false);
            switch (field)
            {
                case 1:                                                                    // node (repeated)
                {
                    int len = checked((int)await r.ReadVarintAsync().ConfigureAwait(false));
                    var bytes = await r.ReadBytesAsync(len).ConfigureAwait(false);
                    graph.Nodes.Add(ParseNodeFromBytes(bytes));
                    break;
                }
                case 2: graph.Name = await r.ReadStringAsync().ConfigureAwait(false); break;  // name
                case 5:                                                                      // initializer (WEIGHTS)
                {
                    long len = (long)await r.ReadVarintAsync().ConfigureAwait(false);
                    graph.Initializers.Add(await ParseInitializerFromStreamAsync(r, len, streamThreshold).ConfigureAwait(false));
                    break;
                }
                case 11:                                                                     // input (repeated)
                {
                    int len = checked((int)await r.ReadVarintAsync().ConfigureAwait(false));
                    var bytes = await r.ReadBytesAsync(len).ConfigureAwait(false);
                    graph.Inputs.Add(ParseValueInfoFromBytes(bytes));
                    break;
                }
                case 12:                                                                     // output (repeated)
                {
                    int len = checked((int)await r.ReadVarintAsync().ConfigureAwait(false));
                    var bytes = await r.ReadBytesAsync(len).ConfigureAwait(false);
                    graph.Outputs.Add(ParseValueInfoFromBytes(bytes));
                    break;
                }
                case 13:                                                                     // value_info (repeated)
                {
                    int len = checked((int)await r.ReadVarintAsync().ConfigureAwait(false));
                    var bytes = await r.ReadBytesAsync(len).ConfigureAwait(false);
                    graph.ValueInfo.Add(ParseValueInfoFromBytes(bytes));
                    break;
                }
                default: await r.SkipFieldAsync(wire).ConfigureAwait(false); break;
            }
        }
        return graph;
    }

    /// <summary>
    /// Parse one initializer. Small submessages (&lt;= threshold) are read whole and parsed by the exact
    /// byte[] <see cref="ParseTensorProto"/>. Large ones are parsed field-by-field so the raw_data offset
    /// can be recorded and the weight skipped (never materialized).
    /// </summary>
    private static async Task<OnnxTensorProto> ParseInitializerFromStreamAsync(
        StreamProtoReader r, long len, int streamThreshold)
    {
        // ALWAYS field-walk, at ANY size. A raw_data / packed-float_data weight BLOB records its stream
        // OFFSET and is NEVER materialized (JS-side zero-copy upload straight to the GPU), regardless of
        // size. The old `len <= streamThreshold` early-return read every sub-1MB weight WHOLE into a .NET
        // byte[] (RawDataStreamOffset stayed -1 → CopyFromCPU) — SD-Turbo's ~4651 small fp16 weights all
        // fell into .NET, which was the entire ~26x WebGPU load gap (Geordi root-caused 2026-07-05, from
        // the CUDA-vs-WebGPU 7.6s-vs-200s measurement). `streamThreshold` is now vestigial (kept for the
        // caller signature). The stream-vs-materialize decision is by ELEMENT COUNT, matching the CPU
        // constant-extraction threshold (<=64 elems): a float WEIGHT (dtype 1/10, >64 elems) streams
        // zero-copy; everything <=64 elems (CPU shape/scalar constants) and all non-float raw_data +
        // inline int32/int64/double_data materialize into .NET — those genuinely ARE small .NET-side
        // constants the graph reads on the CPU for shape inference.
        _ = streamThreshold;
        long end = r.Position + len;
        var tensor = new OnnxTensorProto();
        var dims = new List<long>();
        List<int>? int32s = null; List<long>? int64s = null; List<double>? doubles = null;

        while (r.Position < end)
        {
            var (field, wire) = await r.ReadTagAsync().ConfigureAwait(false);
            switch (field)
            {
                case 1:                                                                    // dims (packed int64)
                    if (wire == 2)
                    {
                        int plen = checked((int)await r.ReadVarintAsync().ConfigureAwait(false));
                        var pb = await r.ReadBytesAsync(plen).ConfigureAwait(false);
                        AppendPackedInt64s(pb, dims);
                    }
                    else dims.Add((long)await r.ReadVarintAsync().ConfigureAwait(false));
                    break;
                case 2: tensor.DataType = (int)await r.ReadVarintAsync().ConfigureAwait(false); break;  // data_type
                case 4:                                                                                 // float_data (packed f32)
                    if (wire == 2)
                    {
                        long fLen = (long)await r.ReadVarintAsync().ConfigureAwait(false);
                        if (tensor.DataType == 0) tensor.DataType = 1;   // float_data is always f32
                        if (TensorElems(dims) > 64)                      // WEIGHT: stream (contiguous LE f32 == raw_data layout)
                        {
                            tensor.RawDataStreamOffset = r.Position;
                            tensor.RawDataLength = checked((int)fLen);
                            await r.SkipAsync(fLen).ConfigureAwait(false);
                        }
                        else                                             // small float CONSTANT (<=64): materialize its CPU value
                        {
                            var pb = await r.ReadBytesAsync(checked((int)fLen)).ConfigureAwait(false);
                            var fl = new List<float>();
                            AppendPackedFloats(pb, fl);
                            tensor.FloatData = fl.ToArray();
                        }
                    }
                    else await r.SkipFieldAsync(wire).ConfigureAwait(false);
                    break;
                case 5:                                                                                 // int32_data — inline (small constant)
                    if (wire == 2)
                    {
                        int plen = checked((int)await r.ReadVarintAsync().ConfigureAwait(false));
                        var pb = await r.ReadBytesAsync(plen).ConfigureAwait(false);
                        AppendPackedInt32s(pb, int32s ??= new());
                    }
                    else (int32s ??= new()).Add((int)(long)await r.ReadVarintAsync().ConfigureAwait(false));
                    break;
                case 7:                                                                                 // int64_data — inline
                    if (wire == 2)
                    {
                        int plen = checked((int)await r.ReadVarintAsync().ConfigureAwait(false));
                        var pb = await r.ReadBytesAsync(plen).ConfigureAwait(false);
                        AppendPackedInt64s(pb, int64s ??= new());
                    }
                    else (int64s ??= new()).Add((long)await r.ReadVarintAsync().ConfigureAwait(false));
                    break;
                case 10:                                                                                // double_data — inline
                    if (wire == 2)
                    {
                        int plen = checked((int)await r.ReadVarintAsync().ConfigureAwait(false));
                        var pb = await r.ReadBytesAsync(plen).ConfigureAwait(false);
                        AppendPackedDoubles(pb, doubles ??= new());
                    }
                    else
                    {
                        var db = await r.ReadBytesAsync(8).ConfigureAwait(false); // fixed64
                        (doubles ??= new()).Add(BitConverter.ToDouble(db, 0));
                    }
                    break;
                case 8: tensor.Name = await r.ReadStringAsync().ConfigureAwait(false); break;           // name
                case 9:                                                                                 // raw_data
                {
                    long rawLen = (long)await r.ReadVarintAsync().ConfigureAwait(false);
                    // Stream ONLY a float WEIGHT: dtype FLOAT32(1)/FLOAT16(10) AND >64 elements. Everything
                    // else materializes into .NET (identical to the old byte[] path; downstream ToFloatArray/
                    // constant extraction consumes RawData): (a) non-float raw_data — INT64/INT32 shape
                    // constants, bool masks — the GPU uploader is float-only; (b) a SMALL float tensor
                    // (<=64 elems) is a CPU constant (shape/scalar the graph reads on the CPU for shape
                    // inference) — streaming it to GPU-only would lose that value and corrupt downstream
                    // shapes (a Conv saw rank-3 instead of rank-4, 2026-07-05). data_type + dims precede
                    // raw_data in standard ONNX, so both are known here; if unset we materialize (safe).
                    // ⚠️ INT8(3)/UINT8(2) stream too. The "non-float raw_data materialises" rule above was
                    // written for INT64/INT32 shape constants and bool masks - small metadata - and an int8
                    // QUANTISED WEIGHT is the opposite of that: on an int8 export it is most of the model.
                    // Excluding them sent every one through the managed heap, which is what the browser
                    // bulk-data guard caught on ZipVoice. The >64-element rule still protects small integer
                    // constants that CPU shape inference has to read.
                    if ((tensor.DataType == 1 || tensor.DataType == 10 || tensor.DataType == 3)
                        && TensorElems(dims) > 64)
                    {
                        tensor.RawDataStreamOffset = r.Position;      // absolute offset of the weight blob
                        tensor.RawDataLength = checked((int)rawLen);
                        await r.SkipAsync(rawLen).ConfigureAwait(false);
                    }
                    else
                    {
                        tensor.RawData = await r.ReadBytesAsync(checked((int)rawLen)).ConfigureAwait(false);
                    }
                    break;
                }
                case 14: tensor.DataLocation = (int)await r.ReadVarintAsync().ConfigureAwait(false); break; // data_location (1 = external → loader skips)
                default: await r.SkipFieldAsync(wire).ConfigureAwait(false); break;                     // string_data(6)/uint64(11)/doc_string(12)/external_data(13)
            }
        }

        tensor.Dims = dims.ToArray();
        tensor.Int32Data = int32s?.ToArray();
        tensor.Int64Data = int64s?.ToArray();
        tensor.DoubleData = doubles?.ToArray();

        // Unsupported ONLY if the tensor has NO data in any recognized field AND is not external.
        if (tensor.RawDataStreamOffset < 0 && tensor.RawData == null && tensor.FloatData == null
            && tensor.Int32Data == null && tensor.Int64Data == null && tensor.DoubleData == null
            && tensor.DataLocation != 1)
            throw new NotSupportedException(
                $"Streaming load: initializer '{tensor.Name}' (dtype {tensor.DataType}) has no raw_data, packed " +
                "float_data, or int32/int64/double_data. Load this model via CreateFromOnnx(byte[]).");
        return tensor;
    }

    // ── sync byte[] adapters (ref struct ProtobufReader cannot cross an await) ──
    private static OnnxNodeProto ParseNodeFromBytes(byte[] bytes)
    {
        var r = new ProtobufReader(bytes);
        return ParseNodeProto(ref r);
    }

    private static OnnxValueInfoProto ParseValueInfoFromBytes(byte[] bytes)
    {
        var r = new ProtobufReader(bytes);
        return ParseValueInfoProto(ref r);
    }

    private static OnnxOpsetImport ParseOpsetFromBytes(byte[] bytes)
    {
        var r = new ProtobufReader(bytes);
        return ParseOpsetImport(ref r);
    }

    private static OnnxTensorProto ParseTensorFromBytes(byte[] bytes)
    {
        var r = new ProtobufReader(bytes);
        return ParseTensorProto(ref r);
    }

    private static void AppendPackedInt64s(byte[] packed, List<long> into)
    {
        var pr = new ProtobufReader(packed);
        while (pr.HasMore) into.Add(pr.ReadInt64());
    }

    private static void AppendPackedInt32s(byte[] packed, List<int> into)
    {
        var pr = new ProtobufReader(packed);
        while (pr.HasMore) into.Add(pr.ReadInt32());
    }

    private static void AppendPackedDoubles(byte[] packed, List<double> into)
    {
        var pr = new ProtobufReader(packed);
        while (pr.HasMore) into.Add(pr.ReadDouble());
    }

    private static void AppendPackedFloats(byte[] packed, List<float> into)
    {
        var pr = new ProtobufReader(packed);
        while (pr.HasMore) into.Add(pr.ReadFloat());
    }

    /// <summary>Element count from parsed dims (empty dims = scalar = 1). Used to tell a small CPU
    /// constant (&lt;=64 elems, must materialize its value for shape inference) from a streamable weight.</summary>
    private static long TensorElems(List<long> dims)
    {
        if (dims.Count == 0) return 1;
        long e = 1;
        foreach (var d in dims) e *= d;
        return e;
    }
}
