using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Onnx;

namespace SpawnDev.ILGPU.ML.TensorFlow;

/// <summary>
/// Converts a parsed TFGraphDef into the shared ModelGraph IR.
/// TF frozen graphs use NHWC layout by default. Placeholder nodes become inputs,
/// Const nodes become initializers (weights), and operation nodes become graph nodes.
/// </summary>
public static class TFGraphDefGraphBuilder
{
    /// <summary>
    /// Build a ModelGraph from a parsed TensorFlow GraphDef.
    /// Returns the graph and a dictionary of constant tensor data (name → float[]).
    /// </summary>
    public static (ModelGraph Graph, Dictionary<string, float[]> Constants) BuildGraph(TFGraphDef tfGraph)
    {
        var graph = new ModelGraph { Name = "TFGraphDef" };
        var constants = new Dictionary<string, float[]>();

        // First pass: collect all Placeholder (input) and Const (weight) nodes
        var constShapes = new Dictionary<string, int[]>();
        foreach (var node in tfGraph.Nodes)
        {
            switch (node.Op)
            {
                case "Placeholder":
                {
                    var shape = node.GetAttrShape("shape") ?? new[] { 1 };
                    // Replace -1 (dynamic) dimensions with 1
                    for (int i = 0; i < shape.Length; i++)
                        if (shape[i] <= 0) shape[i] = 1;
                    graph.Inputs.Add(new GraphValueInfo { Name = node.Name, Shape = shape });
                    break;
                }
                case "Const":
                {
                    // Extract tensor data from the "value" attribute
                    if (node.Attributes.TryGetValue("value", out var valAttr) && valAttr.TensorBytes != null)
                    {
                        var (data, shape) = ParseTensorProto(valAttr.TensorBytes);
                        if (data != null)
                        {
                            constants[node.Name] = data;
                            constShapes[node.Name] = shape;
                            graph.Initializers[node.Name] = shape;
                        }
                    }
                    else
                    {
                        // Const node with shape attribute but no tensor data (scalar or type-only)
                        var shape = node.GetAttrShape("shape") ?? new[] { 1 };
                        graph.Initializers[node.Name] = shape;
                    }
                    break;
                }
            }
        }

        // Second pass: convert operation nodes to graph nodes
        foreach (var node in tfGraph.Nodes)
        {
            if (node.Op == "Placeholder" || node.Op == "Const" || node.Op == "NoOp" || node.Op == "Identity")
            {
                // Identity nodes are pass-throughs — add them as Identity ops to preserve naming
                if (node.Op == "Identity" && node.Inputs.Count > 0)
                {
                    graph.Nodes.Add(new GraphNode
                    {
                        OpType = "Identity",
                        Inputs = CleanInputNames(node.Inputs),
                        Outputs = new List<string> { node.Name }
                    });
                }
                continue;
            }

            var onnxOp = TFOpMapping.ToOnnxOpType(node.Op);
            if (onnxOp == null) continue; // Unmappable op — skip

            var inputs = CleanInputNames(node.Inputs);
            var graphNode = new GraphNode
            {
                OpType = onnxOp,
                Inputs = inputs,
                Outputs = new List<string> { node.Name }
            };

            // Extract attributes specific to the TF op
            var attrs = ExtractAttributes(node, onnxOp);
            if (attrs.Count > 0)
            {
                graphNode.Attributes = new Dictionary<string, System.Text.Json.JsonElement>();
                foreach (var (key, value) in attrs)
                {
                    var json = System.Text.Json.JsonSerializer.Serialize(value);
                    graphNode.Attributes[key] = System.Text.Json.JsonDocument.Parse(json).RootElement.Clone();
                }
            }

            graph.Nodes.Add(graphNode);
        }

        // Determine output: use the last non-const, non-placeholder node
        if (graph.Outputs.Count == 0 && graph.Nodes.Count > 0)
        {
            var lastNode = graph.Nodes[^1];
            var outputName = lastNode.Outputs[0];
            graph.Outputs.Add(new GraphValueInfo { Name = outputName, Shape = new[] { 1 } });
        }

        return (graph, constants);
    }

    /// <summary>
    /// Clean TF input names: remove ":0" suffix and "^" control dependency prefix.
    /// </summary>
    private static List<string> CleanInputNames(List<string> inputs)
    {
        var clean = new List<string>();
        foreach (var inp in inputs)
        {
            if (inp.StartsWith('^')) continue; // Skip control dependencies
            var name = inp.Contains(':') ? inp[..inp.IndexOf(':')] : inp;
            clean.Add(name);
        }
        return clean;
    }

    /// <summary>
    /// Extract ONNX-compatible attributes from a TF NodeDef.
    /// </summary>
    private static Dictionary<string, object> ExtractAttributes(TFNodeDef node, string onnxOp)
    {
        var attrs = new Dictionary<string, object>();

        switch (onnxOp)
        {
            case "Conv":
            {
                // TF Conv2D has strides, padding, data_format, dilations
                if (node.Attributes.TryGetValue("strides", out var s) && s.Shape != null)
                {
                    // TF strides are [1, sH, sW, 1] for NHWC
                    var st = s.Shape;
                    if (st.Length == 4)
                        attrs["strides"] = new long[] { st[1], st[2] };
                }
                if (node.Attributes.TryGetValue("dilations", out var d) && d.Shape != null)
                {
                    var dl = d.Shape;
                    if (dl.Length == 4)
                        attrs["dilations"] = new long[] { dl[1], dl[2] };
                }
                var padding = node.GetAttrString("padding", "VALID");
                if (padding == "SAME")
                    attrs["auto_pad"] = "SAME_UPPER";
                // Mark as NHWC for downstream handling
                attrs["_data_format"] = "NHWC";
                break;
            }
            case "MaxPool" or "AveragePool":
            {
                if (node.Attributes.TryGetValue("ksize", out var k) && k.Shape != null)
                {
                    var ks = k.Shape;
                    if (ks.Length == 4)
                        attrs["kernel_shape"] = new long[] { ks[1], ks[2] };
                }
                if (node.Attributes.TryGetValue("strides", out var s) && s.Shape != null)
                {
                    var st = s.Shape;
                    if (st.Length == 4)
                        attrs["strides"] = new long[] { st[1], st[2] };
                }
                var padding = node.GetAttrString("padding", "VALID");
                if (padding == "SAME")
                    attrs["auto_pad"] = "SAME_UPPER";
                attrs["_data_format"] = "NHWC";
                break;
            }
            case "Clip":
            {
                // Relu6 → Clip [0, 6]
                if (node.Op == "Relu6")
                {
                    attrs["min"] = 0f;
                    attrs["max"] = 6f;
                }
                break;
            }
            case "ReduceMean" or "ReduceSum" or "ReduceMax" or "ReduceMin":
            {
                var keepDims = node.GetAttrBool("keep_dims", false) || node.GetAttrBool("keepdims", false);
                attrs["keepdims"] = keepDims ? 1L : 0L;
                break;
            }
            case "Concat":
            {
                // TF ConcatV2 has axis as last input; TF Concat has axis as first
                if (node.Op == "ConcatV2")
                    attrs["_axis_is_last_input"] = true;
                break;
            }
            case "Softmax":
            {
                attrs["axis"] = -1L;
                break;
            }
            case "LeakyRelu":
            {
                attrs["alpha"] = node.GetAttrFloat("alpha", 0.2f);
                break;
            }
            case "Resize":
            {
                if (node.Op == "ResizeBilinear")
                    attrs["mode"] = "linear";
                else if (node.Op == "ResizeNearestNeighbor")
                    attrs["mode"] = "nearest";
                break;
            }
        }

        return attrs;
    }

    /// <summary>
    /// Parse a TensorProto from raw protobuf bytes.
    /// TF TensorProto: field 1 = dtype, field 2 = tensor_shape, field 4 = tensor_content (raw bytes)
    /// Also handles float_val (field 5), int_val (field 7), etc.
    /// </summary>
    private static (float[]? Data, int[] Shape) ParseTensorProto(byte[] data)
    {
        var reader = new ProtobufReader(data);
        int dtype = 0;
        int[] shape = new[] { 1 };
        byte[]? tensorContent = null;
        var floatVals = new List<float>();
        var intVals = new List<int>();

        while (reader.HasMore)
        {
            var (fieldNumber, wireType) = reader.ReadTag();
            switch (fieldNumber)
            {
                case 1: // dtype
                    dtype = (int)reader.ReadVarint();
                    break;
                case 2: // tensor_shape
                    var shapeBytes = reader.ReadBytes().ToArray();
                    shape = ParseTensorShape(shapeBytes);
                    break;
                case 4: // tensor_content (raw bytes — most efficient)
                    tensorContent = reader.ReadBytes().ToArray();
                    break;
                case 5: // float_val (repeated float, packed or unpacked)
                    if (wireType == 2) // packed
                    {
                        var packed = reader.ReadBytes();
                        for (int i = 0; i + 3 < packed.Length; i += 4)
                            floatVals.Add(BitConverter.ToSingle(packed.Slice(i, 4)));
                    }
                    else
                    {
                        floatVals.Add(reader.ReadFloat());
                    }
                    break;
                case 7: // int_val (repeated int32)
                    if (wireType == 2) // packed
                    {
                        var packed = reader.ReadBytes();
                        var pr = new ProtobufReader(packed.ToArray());
                        while (pr.HasMore)
                            intVals.Add((int)pr.ReadVarint());
                    }
                    else
                    {
                        intVals.Add((int)reader.ReadVarint());
                    }
                    break;
                default:
                    reader.SkipField(wireType);
                    break;
            }
        }

        int totalElements = 1;
        foreach (int d in shape) totalElements *= Math.Max(d, 1);

        // Convert to float array based on dtype
        float[]? result = null;

        if (tensorContent != null && tensorContent.Length > 0)
        {
            switch (dtype)
            {
                case 1: // DT_FLOAT
                    result = new float[tensorContent.Length / 4];
                    Buffer.BlockCopy(tensorContent, 0, result, 0, tensorContent.Length);
                    break;
                case 3: // DT_INT32
                    result = new float[tensorContent.Length / 4];
                    for (int i = 0; i < result.Length; i++)
                        result[i] = BitConverter.ToInt32(tensorContent, i * 4);
                    break;
                case 19: // DT_HALF (FP16)
                    result = new float[tensorContent.Length / 2];
                    for (int i = 0; i < result.Length; i++)
                        result[i] = (float)BitConverter.ToHalf(tensorContent, i * 2);
                    break;
                case 2: // DT_DOUBLE
                    result = new float[tensorContent.Length / 8];
                    for (int i = 0; i < result.Length; i++)
                        result[i] = (float)BitConverter.ToDouble(tensorContent, i * 8);
                    break;
                default:
                    // Unknown dtype — try as float
                    if (tensorContent.Length >= totalElements * 4)
                    {
                        result = new float[totalElements];
                        Buffer.BlockCopy(tensorContent, 0, result, 0, totalElements * 4);
                    }
                    break;
            }
        }
        else if (floatVals.Count > 0)
        {
            result = floatVals.ToArray();
            // TF broadcasts scalar float_val to fill the tensor
            if (result.Length == 1 && totalElements > 1)
            {
                var expanded = new float[totalElements];
                Array.Fill(expanded, result[0]);
                result = expanded;
            }
        }
        else if (intVals.Count > 0)
        {
            result = intVals.Select(v => (float)v).ToArray();
            if (result.Length == 1 && totalElements > 1)
            {
                var expanded = new float[totalElements];
                Array.Fill(expanded, result[0]);
                result = expanded;
            }
        }

        return (result, shape);
    }

    private static int[] ParseTensorShape(byte[] data)
    {
        var dims = new List<int>();
        var reader = new ProtobufReader(data);

        while (reader.HasMore)
        {
            var (fieldNumber, wireType) = reader.ReadTag();
            if (fieldNumber == 2) // dim (repeated Dim)
            {
                var dimBytes = reader.ReadBytes().ToArray();
                var dimReader = new ProtobufReader(dimBytes);
                while (dimReader.HasMore)
                {
                    var (df, dw) = dimReader.ReadTag();
                    if (df == 1) // size (int64)
                        dims.Add((int)dimReader.ReadVarint());
                    else
                        dimReader.SkipField(dw);
                }
            }
            else
            {
                reader.SkipField(wireType);
            }
        }

        return dims.Count > 0 ? dims.ToArray() : new[] { 1 };
    }
}
