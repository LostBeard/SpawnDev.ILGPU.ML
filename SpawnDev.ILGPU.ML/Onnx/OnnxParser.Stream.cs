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
        // Small initializer (incl. all inline-data constants): read whole, reuse the byte[] parser verbatim.
        if (len <= streamThreshold)
        {
            var bytes = await r.ReadBytesAsync(checked((int)len)).ConfigureAwait(false);
            return ParseTensorFromBytes(bytes);
        }

        // Large initializer: dominated by raw_data. Walk fields; record raw_data offset + skip the bytes.
        long end = r.Position + len;
        var tensor = new OnnxTensorProto();
        var dims = new List<long>();

        while (r.Position < end)
        {
            var (field, wire) = await r.ReadTagAsync().ConfigureAwait(false);
            switch (field)
            {
                case 1:                                                                    // dims
                    if (wire == 2)
                    {
                        int plen = checked((int)await r.ReadVarintAsync().ConfigureAwait(false));
                        var pb = await r.ReadBytesAsync(plen).ConfigureAwait(false);
                        AppendPackedInt64s(pb, dims);
                    }
                    else dims.Add((long)await r.ReadVarintAsync().ConfigureAwait(false));
                    break;
                case 2: tensor.DataType = (int)await r.ReadVarintAsync().ConfigureAwait(false); break;  // data_type
                case 4:                                                                                 // float_data (packed)
                    // Packed repeated float is contiguous little-endian float32 — the SAME byte layout as
                    // raw_data for a FLOAT tensor, so it streams identically (offset + skip, upload as f32).
                    // Some exporters (e.g. SqueezeNet) store large weights here instead of raw_data.
                    if (wire == 2)
                    {
                        long fLen = (long)await r.ReadVarintAsync().ConfigureAwait(false);
                        tensor.RawDataStreamOffset = r.Position;
                        tensor.RawDataLength = checked((int)fLen);
                        if (tensor.DataType == 0) tensor.DataType = 1;   // FLOAT (float_data is always f32)
                        await r.SkipAsync(fLen).ConfigureAwait(false);
                    }
                    else await r.SkipFieldAsync(wire).ConfigureAwait(false);
                    break;
                case 8: tensor.Name = await r.ReadStringAsync().ConfigureAwait(false); break;           // name
                case 9:                                                                                 // raw_data
                {
                    long rawLen = (long)await r.ReadVarintAsync().ConfigureAwait(false);
                    tensor.RawDataStreamOffset = r.Position;          // absolute offset of the weight blob
                    tensor.RawDataLength = checked((int)rawLen);
                    await r.SkipAsync(rawLen).ConfigureAwait(false);
                    break;
                }
                default: await r.SkipFieldAsync(wire).ConfigureAwait(false); break;
            }
        }

        tensor.Dims = dims.ToArray();
        if (tensor.RawDataStreamOffset < 0)
            throw new NotSupportedException(
                $"Streaming load: large initializer '{tensor.Name}' (dtype {tensor.DataType}) stores its data in a " +
                "field other than raw_data or packed float_data (e.g. int32/int64/double_data); not yet supported " +
                "by the stream parser. Load this model via CreateFromOnnx(byte[]).");
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
}
