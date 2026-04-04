namespace SpawnDev.ILGPU.ML.Onnx;

/// <summary>
/// Hand-written ONNX protobuf parser. Zero external dependencies.
/// Parses .onnx files (serialized Protocol Buffers) into OnnxModelProto.
/// Only reads fields needed for inference — skips doc_strings, training_info, metadata.
/// ~350 lines. Compiles to ~10KB of IL (vs ~700KB for Google.Protobuf).
/// </summary>
public static class OnnxParser
{
    /// <summary>
    /// Parse an ONNX model from raw bytes.
    /// </summary>
    public static OnnxModelProto Parse(byte[] onnxBytes) => Parse(onnxBytes, zeroCopyThreshold: -1);

    /// <summary>
    /// Parse with zero-copy mode for large tensor raw data.
    /// Tensors with raw_data larger than zeroCopyThreshold bytes store a reference
    /// to onnxBytes instead of copying. Saves ~147MB per large GPT-2 tensor.
    /// </summary>
    public static OnnxModelProto Parse(byte[] onnxBytes, int zeroCopyThreshold)
    {
        _zeroCopySource = zeroCopyThreshold >= 0 ? onnxBytes : null;
        _zeroCopyThreshold = zeroCopyThreshold;
        try
        {
            var reader = new ProtobufReader(onnxBytes);
            return ParseModelProto(ref reader);
        }
        finally
        {
            _zeroCopySource = null;
        }
    }

    // Thread-local zero-copy state (safe for single-threaded WASM)
    [ThreadStatic] private static byte[]? _zeroCopySource;
    [ThreadStatic] private static int _zeroCopyThreshold;

    /// <summary>
    /// Parse an ONNX model from a ReadOnlySpan.
    /// </summary>
    public static OnnxModelProto Parse(ReadOnlySpan<byte> onnxBytes)
    {
        var reader = new ProtobufReader(onnxBytes);
        return ParseModelProto(ref reader);
    }

    // ──────────────────────────────────────────────
    //  Per-message-type parsers
    // ──────────────────────────────────────────────

    private static OnnxModelProto ParseModelProto(ref ProtobufReader r)
    {
        var model = new OnnxModelProto();
        while (r.HasMore)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1: model.IrVersion = r.ReadInt64(); break;              // ir_version
                case 2: model.ProducerName = r.ReadString(); break;          // producer_name
                case 3: model.ProducerVersion = r.ReadString(); break;       // producer_version
                case 4: model.Domain = r.ReadString(); break;                // domain
                case 5: model.ModelVersion = r.ReadInt64(); break;           // model_version
                case 6: model.DocString = r.ReadString(); break;             // doc_string
                case 7:                                                       // graph
                    var sub = r.ReadSubMessage();
                    model.Graph = ParseGraphProto(ref sub);
                    break;
                case 8:                                                       // opset_import
                    var opSub = r.ReadSubMessage();
                    model.OpsetImports.Add(ParseOpsetImport(ref opSub));
                    break;
                case 14: r.SkipField(wire); break;                           // metadata_props (repeated StringStringEntryProto)
                case 15: r.SkipField(wire); break;                           // training_info (repeated TrainingInfoProto)
                case 20: r.SkipField(wire); break;                           // functions (repeated FunctionProto)
                default: r.SkipField(wire); break;
            }
        }
        return model;
    }

    private static OnnxGraphProto ParseGraphProto(ref ProtobufReader r)
    {
        var graph = new OnnxGraphProto();
        while (r.HasMore)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1:                                                       // node (repeated)
                    var nodeSub = r.ReadSubMessage();
                    graph.Nodes.Add(ParseNodeProto(ref nodeSub));
                    break;
                case 2: graph.Name = r.ReadString(); break;                  // name
                case 5:                                                       // initializer (repeated) — WEIGHTS
                    var initSub = r.ReadSubMessage();
                    graph.Initializers.Add(ParseTensorProto(ref initSub));
                    break;
                case 11:                                                      // input (repeated)
                    var inSub = r.ReadSubMessage();
                    graph.Inputs.Add(ParseValueInfoProto(ref inSub));
                    break;
                case 12:                                                      // output (repeated)
                    var outSub = r.ReadSubMessage();
                    graph.Outputs.Add(ParseValueInfoProto(ref outSub));
                    break;
                case 13:                                                      // value_info (repeated)
                    var viSub = r.ReadSubMessage();
                    graph.ValueInfo.Add(ParseValueInfoProto(ref viSub));
                    break;
                default: r.SkipField(wire); break;
            }
        }
        return graph;
    }

    private static OnnxNodeProto ParseNodeProto(ref ProtobufReader r)
    {
        var node = new OnnxNodeProto();
        while (r.HasMore)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1: node.Inputs.Add(r.ReadString()); break;              // input (repeated string)
                case 2: node.Outputs.Add(r.ReadString()); break;             // output (repeated string)
                case 3: node.Name = r.ReadString(); break;                   // name
                case 4: node.OpType = r.ReadString(); break;                 // op_type
                case 5:                                                       // attribute (repeated)
                    var attrSub = r.ReadSubMessage();
                    node.Attributes.Add(ParseAttributeProto(ref attrSub));
                    break;
                case 7: node.Domain = r.ReadString(); break;                 // domain
                default: r.SkipField(wire); break;
            }
        }
        return node;
    }

    private static OnnxTensorProto ParseTensorProto(ref ProtobufReader r)
    {
        var tensor = new OnnxTensorProto();
        List<long>? dims = null;
        List<float>? floats = null;
        List<int>? int32s = null;
        List<long>? int64s = null;
        List<double>? doubles = null;

        while (r.HasMore)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1: // dims (packed repeated int64)
                    if (wire == 2) // packed
                    {
                        var packed = r.ReadPackedInt64s();
                        dims ??= new List<long>();
                        dims.AddRange(packed);
                    }
                    else // unpacked (single varint)
                    {
                        dims ??= new List<long>();
                        dims.Add(r.ReadInt64());
                    }
                    break;
                case 2: tensor.DataType = r.ReadInt32(); break;              // data_type
                case 4: // float_data (packed repeated float)
                    if (wire == 2)
                    {
                        var packed = r.ReadPackedFloats();
                        floats ??= new List<float>();
                        floats.AddRange(packed);
                    }
                    else // unpacked (single fixed32)
                    {
                        floats ??= new List<float>();
                        floats.Add(r.ReadFloat());
                    }
                    break;
                case 5: // int32_data (packed repeated int32)
                    if (wire == 2)
                    {
                        var packed = r.ReadPackedInt32s();
                        int32s ??= new List<int>();
                        int32s.AddRange(packed);
                    }
                    else
                    {
                        int32s ??= new List<int>();
                        int32s.Add(r.ReadInt32());
                    }
                    break;
                case 7: // int64_data (packed repeated int64)
                    if (wire == 2)
                    {
                        var packed = r.ReadPackedInt64s();
                        int64s ??= new List<long>();
                        int64s.AddRange(packed);
                    }
                    else
                    {
                        int64s ??= new List<long>();
                        int64s.Add(r.ReadInt64());
                    }
                    break;
                case 6: // string_data (repeated bytes — string tensors)
                    r.SkipField(wire); // String tensors stored as bytes — skip for float inference
                    break;
                case 8: tensor.Name = r.ReadString(); break;                 // name
                case 9: // raw_data
                    if (_zeroCopySource != null)
                    {
                        // Zero-copy: store reference into source bytes instead of copying
                        var (refOffset, refLength) = r.ReadBytesReference();
                        if (refLength > _zeroCopyThreshold)
                        {
                            tensor.RawDataSource = _zeroCopySource;
                            tensor.RawDataOffset = refOffset;
                            tensor.RawDataLength = refLength;
                        }
                        else
                        {
                            tensor.RawData = new byte[refLength];
                            Buffer.BlockCopy(_zeroCopySource, refOffset, tensor.RawData, 0, refLength);
                        }
                    }
                    else
                    {
                        tensor.RawData = r.ReadByteArray();
                    }
                    break;
                case 10: // double_data (packed repeated double)
                    if (wire == 2)
                    {
                        var packed = r.ReadPackedDoubles();
                        doubles ??= new List<double>();
                        doubles.AddRange(packed);
                    }
                    else
                    {
                        doubles ??= new List<double>();
                        doubles.Add(r.ReadDouble());
                    }
                    break;
                case 13: // external_data (repeated StringStringEntryProto)
                    var extSub = r.ReadSubMessage();
                    var (key, val) = ParseStringStringEntry(ref extSub);
                    tensor.ExternalData ??= new Dictionary<string, string>();
                    tensor.ExternalData[key] = val;
                    break;
                case 14: tensor.DataLocation = r.ReadInt32(); break;         // data_location
                case 11: // uint64_data (packed repeated uint64)
                    r.SkipField(wire); // uint64 tensors — skip for float inference
                    break;
                case 12: // doc_string
                    r.SkipField(wire); // documentation — skip
                    break;
                default: r.SkipField(wire); break;
            }
        }

        tensor.Dims = dims?.ToArray() ?? Array.Empty<long>();
        tensor.FloatData = floats?.ToArray();
        tensor.Int32Data = int32s?.ToArray();
        tensor.Int64Data = int64s?.ToArray();
        tensor.DoubleData = doubles?.ToArray();

        return tensor;
    }

    private static OnnxAttributeProto ParseAttributeProto(ref ProtobufReader r)
    {
        var attr = new OnnxAttributeProto();
        List<float>? floats = null;
        List<long>? ints = null;
        List<byte[]>? strings = null;

        while (r.HasMore)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1: attr.Name = r.ReadString(); break;                   // name
                case 2: attr.F = r.ReadFloat(); break;                       // f (float, wire type I32)
                case 3: attr.I = r.ReadInt64(); break;                       // i (int64, varint)
                case 4: attr.S = r.ReadByteArray(); break;                   // s (bytes)
                case 5:                                                       // t (TensorProto)
                    var tSub = r.ReadSubMessage();
                    attr.T = ParseTensorProto(ref tSub);
                    break;
                case 6:                                                       // g (GraphProto)
                    var gSub = r.ReadSubMessage();
                    attr.G = ParseGraphProto(ref gSub);
                    break;
                case 7: // floats (packed repeated float)
                    if (wire == 2)
                    {
                        var packed = r.ReadPackedFloats();
                        floats ??= new List<float>();
                        floats.AddRange(packed);
                    }
                    else
                    {
                        floats ??= new List<float>();
                        floats.Add(r.ReadFloat());
                    }
                    break;
                case 8: // ints (packed repeated int64)
                    if (wire == 2)
                    {
                        var packed = r.ReadPackedInt64s();
                        ints ??= new List<long>();
                        ints.AddRange(packed);
                    }
                    else
                    {
                        ints ??= new List<long>();
                        ints.Add(r.ReadInt64());
                    }
                    break;
                case 9: // strings (repeated bytes — NOT packed)
                    strings ??= new List<byte[]>();
                    strings.Add(r.ReadByteArray());
                    break;
                case 20: attr.Type = (OnnxAttributeType)r.ReadInt32(); break; // type
                default: r.SkipField(wire); break;
            }
        }

        attr.Floats = floats?.ToArray();
        attr.Ints = ints?.ToArray();
        attr.Strings = strings;

        return attr;
    }

    private static OnnxValueInfoProto ParseValueInfoProto(ref ProtobufReader r)
    {
        var vi = new OnnxValueInfoProto();
        while (r.HasMore)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1: vi.Name = r.ReadString(); break;                    // name
                case 2:                                                       // type (TypeProto)
                    var typeSub = r.ReadSubMessage();
                    ParseTypeProto(ref typeSub, vi);
                    break;
                default: r.SkipField(wire); break;
            }
        }
        return vi;
    }

    private static void ParseTypeProto(ref ProtobufReader r, OnnxValueInfoProto vi)
    {
        while (r.HasMore)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1: // tensor_type (TypeProto.Tensor)
                    var tensorSub = r.ReadSubMessage();
                    ParseTensorTypeProto(ref tensorSub, vi);
                    break;
                default: r.SkipField(wire); break;
            }
        }
    }

    private static void ParseTensorTypeProto(ref ProtobufReader r, OnnxValueInfoProto vi)
    {
        while (r.HasMore)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1: vi.ElemType = r.ReadInt32(); break;                  // elem_type
                case 2:                                                       // shape (TensorShapeProto)
                    var shapeSub = r.ReadSubMessage();
                    ParseTensorShapeProto(ref shapeSub, vi);
                    break;
                default: r.SkipField(wire); break;
            }
        }
    }

    private static void ParseTensorShapeProto(ref ProtobufReader r, OnnxValueInfoProto vi)
    {
        while (r.HasMore)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1: // dim (repeated Dimension)
                    var dimSub = r.ReadSubMessage();
                    vi.Shape.Add(ParseDimension(ref dimSub));
                    break;
                default: r.SkipField(wire); break;
            }
        }
    }

    private static OnnxDimension ParseDimension(ref ProtobufReader r)
    {
        var dim = new OnnxDimension();
        while (r.HasMore)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1: dim.DimValue = r.ReadInt64(); break;                 // dim_value
                case 2: dim.DimParam = r.ReadString(); break;                // dim_param
                default: r.SkipField(wire); break;
            }
        }
        return dim;
    }

    private static OnnxOpsetImport ParseOpsetImport(ref ProtobufReader r)
    {
        var op = new OnnxOpsetImport();
        while (r.HasMore)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1: op.Domain = r.ReadString(); break;                   // domain
                case 2: op.Version = r.ReadInt64(); break;                   // version
                default: r.SkipField(wire); break;
            }
        }
        return op;
    }

    private static (string Key, string Value) ParseStringStringEntry(ref ProtobufReader r)
    {
        string key = "", value = "";
        while (r.HasMore)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1: key = r.ReadString(); break;
                case 2: value = r.ReadString(); break;
                default: r.SkipField(wire); break;
            }
        }
        return (key, value);
    }
}
