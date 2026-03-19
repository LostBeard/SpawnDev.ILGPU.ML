using SpawnDev.ILGPU.ML.Onnx; // Reuse ProtobufReader

namespace SpawnDev.ILGPU.ML.TensorFlow;

/// <summary>
/// Zero-dependency TensorFlow GraphDef (.pb) parser.
/// Reuses the ProtobufReader from our ONNX parser to read the protocol buffers format.
///
/// GraphDef protobuf schema:
///   GraphDef { node: NodeDef[], versions: VersionDef }
///   NodeDef { name, op, input[], device, attr: map[string, AttrValue] }
///   AttrValue { type, shape, tensor, list, ... }
///
/// TensorFlow frozen graphs store everything in a single .pb file:
/// - Placeholder nodes = model inputs
/// - Const nodes = weights/biases (tensor data embedded in attr["value"])
/// - Operation nodes = computation graph
/// </summary>
public static class TFGraphDefParser
{
    /// <summary>
    /// Parse a TensorFlow frozen graph (.pb) from raw bytes.
    /// </summary>
    public static TFGraphDef Parse(byte[] data)
    {
        var graph = new TFGraphDef();
        var reader = new ProtobufReader(data);

        while (reader.HasMore)
        {
            var (fieldNumber, wireType) = reader.ReadTag();
            switch (fieldNumber)
            {
                case 1: // node (repeated NodeDef)
                    var nodeBytes = reader.ReadBytes().ToArray();
                    graph.Nodes.Add(ParseNodeDef(nodeBytes));
                    break;
                case 4: // versions
                    reader.SkipField(wireType);
                    break;
                default:
                    reader.SkipField(wireType);
                    break;
            }
        }

        return graph;
    }

    /// <summary>Check if data looks like a TF GraphDef protobuf.</summary>
    public static bool IsGraphDef(byte[] data)
    {
        if (data.Length < 4) return false;
        // GraphDef starts with field 1 (node), wire type 2 (length-delimited)
        // Tag byte = (1 << 3) | 2 = 0x0A
        return data[0] == 0x0A;
    }

    /// <summary>Get a summary string.</summary>
    public static string GetSummary(TFGraphDef graph)
    {
        var ops = graph.Nodes.Where(n => n.Op != "Const" && n.Op != "Placeholder")
            .Select(n => n.Op).Distinct().OrderBy(s => s);
        int constCount = graph.Nodes.Count(n => n.Op == "Const");
        int inputCount = graph.Nodes.Count(n => n.Op == "Placeholder");
        return $"TF GraphDef: {graph.Nodes.Count} nodes ({inputCount} inputs, {constCount} consts), ops: {string.Join(", ", ops)}";
    }

    private static TFNodeDef ParseNodeDef(byte[] data)
    {
        var node = new TFNodeDef();
        var reader = new ProtobufReader(data);

        while (reader.HasMore)
        {
            var (fieldNumber, wireType) = reader.ReadTag();
            switch (fieldNumber)
            {
                case 1: // name (string)
                    node.Name = reader.ReadString();
                    break;
                case 2: // op (string)
                    node.Op = reader.ReadString();
                    break;
                case 3: // input (repeated string)
                    node.Inputs.Add(reader.ReadString());
                    break;
                case 4: // device (string)
                    reader.ReadString(); // skip device
                    break;
                case 5: // attr (map<string, AttrValue>)
                    var attrBytes = reader.ReadBytes().ToArray();
                    var (key, value) = ParseAttrEntry(attrBytes);
                    if (key != null)
                        node.Attributes[key] = value;
                    break;
                default:
                    reader.SkipField(wireType);
                    break;
            }
        }

        return node;
    }

    private static (string? Key, TFAttrValue Value) ParseAttrEntry(byte[] data)
    {
        string? key = null;
        var value = new TFAttrValue();
        var reader = new ProtobufReader(data);

        while (reader.HasMore)
        {
            var (fieldNumber, wireType) = reader.ReadTag();
            switch (fieldNumber)
            {
                case 1: // key (string)
                    key = reader.ReadString();
                    break;
                case 2: // value (AttrValue)
                    var valueBytes = reader.ReadBytes().ToArray();
                    value = ParseAttrValue(valueBytes);
                    break;
                default:
                    reader.SkipField(wireType);
                    break;
            }
        }

        return (key, value);
    }

    private static TFAttrValue ParseAttrValue(byte[] data)
    {
        var attr = new TFAttrValue();
        var reader = new ProtobufReader(data);

        while (reader.HasMore)
        {
            var (fieldNumber, wireType) = reader.ReadTag();
            switch (fieldNumber)
            {
                case 1: // s (bytes — string value)
                    attr.StringValue = reader.ReadString();
                    break;
                case 2: // i (int64)
                    attr.IntValue = (long)reader.ReadVarint();
                    break;
                case 3: // f (float)
                    attr.FloatValue = reader.ReadFloat();
                    break;
                case 4: // b (bool)
                    attr.BoolValue = reader.ReadVarint() != 0;
                    break;
                case 5: // type (DataType enum)
                    attr.DataType = (int)reader.ReadVarint();
                    break;
                case 7: // shape (TensorShapeProto)
                    var shapeBytes = reader.ReadBytes().ToArray();
                    attr.Shape = ParseShape(shapeBytes);
                    break;
                case 8: // tensor (TensorProto — for Const nodes, contains weight data)
                    attr.TensorBytes = reader.ReadBytes().ToArray();
                    break;
                default:
                    reader.SkipField(wireType);
                    break;
            }
        }

        return attr;
    }

    private static int[] ParseShape(byte[] data)
    {
        var dims = new List<int>();
        var reader = new ProtobufReader(data);

        while (reader.HasMore)
        {
            var (fieldNumber, wireType) = reader.ReadTag();
            switch (fieldNumber)
            {
                case 2: // dim (repeated Dim)
                    var dimBytes2 = reader.ReadBytes().ToArray();
                    var dimReader = new ProtobufReader(dimBytes2);
                    while (dimReader.HasMore)
                    {
                        var (df, dw) = dimReader.ReadTag();
                        if (df == 1) // size (int64)
                            dims.Add((int)dimReader.ReadVarint());
                        else
                            dimReader.SkipField(dw);
                    }
                    break;
                default:
                    reader.SkipField(wireType);
                    break;
            }
        }

        return dims.ToArray();
    }
}
