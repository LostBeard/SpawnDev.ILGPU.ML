using System.Text.Json;
using System.Text.Json.Serialization;

namespace SpawnDev.ILGPU.ML.Graph;

/// <summary>
/// JSON-serializable description of an ONNX computation graph.
/// Extracted from .onnx protobuf by a desktop tool or Python script.
/// Loaded at runtime for graph compilation and execution.
/// </summary>
public class ModelGraph
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = "";

    [JsonPropertyName("inputs")]
    public List<GraphValueInfo> Inputs { get; set; } = new();

    [JsonPropertyName("outputs")]
    public List<GraphValueInfo> Outputs { get; set; } = new();

    [JsonPropertyName("nodes")]
    public List<GraphNode> Nodes { get; set; } = new();

    /// <summary>
    /// Constant/initializer tensor shapes. Maps tensor name → shape.
    /// The actual data is loaded separately via WeightLoader.
    /// </summary>
    [JsonPropertyName("initializers")]
    public Dictionary<string, int[]> Initializers { get; set; } = new();

    /// <summary>
    /// Constant data values for small tensors (e.g., Reshape target shapes).
    /// Maps tensor name → integer values. Populated from weights during
    /// InferenceSession creation for use in shape inference.
    /// Not serialized — populated at runtime.
    /// </summary>
    [JsonIgnore]
    public Dictionary<string, int[]>? ConstantData { get; set; }

    /// <summary>
    /// Float-precision constant data for compile-time arithmetic.
    /// Unlike ConstantData (int), this preserves fractional values like 0.5
    /// needed for Upsample scale factor computation (Mul, Div chains).
    /// Not serialized — populated at runtime.
    /// </summary>
    [JsonIgnore]
    public Dictionary<string, float[]>? FloatConstantData { get; set; }

    /// <summary>
    /// ONNX-declared data type per initializer / Constant-node output
    /// (see <see cref="Onnx.OnnxDataType"/> codes). Lets the runtime apply
    /// integer-vs-float semantic differences (e.g. ONNX Div truncates toward
    /// zero on integer dtypes but does float division on FP dtypes) even
    /// though all storage in this pipeline is float32.
    /// Not serialized — populated at runtime from OnnxModelInfo.
    /// </summary>
    [JsonIgnore]
    public Dictionary<string, int>? InitializerDataTypes { get; set; }

    public static ModelGraph FromJson(string json)
        => JsonSerializer.Deserialize<ModelGraph>(json) ?? throw new InvalidOperationException("Failed to parse model graph JSON");

    public string ToJson()
        => JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true });
}

/// <summary>Value (tensor) metadata: name and shape.</summary>
public class GraphValueInfo
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = "";

    [JsonPropertyName("shape")]
    public int[] Shape { get; set; } = Array.Empty<int>();
}

/// <summary>A single operation node in the graph.</summary>
public class GraphNode
{
    [JsonPropertyName("opType")]
    public string OpType { get; set; } = "";

    [JsonPropertyName("inputs")]
    public List<string> Inputs { get; set; } = new();

    [JsonPropertyName("outputs")]
    public List<string> Outputs { get; set; } = new();

    [JsonPropertyName("attributes")]
    public Dictionary<string, JsonElement>? Attributes { get; set; }

    /// <summary>
    /// Attributes that cannot survive a JSON round trip, keyed by attribute name. Subgraphs
    /// (<c>then_branch</c>, <c>else_branch</c>, <c>body</c>) are the only ones so far.
    /// </summary>
    /// <remarks>
    /// ⚠️ <see cref="Attributes"/> is <see cref="JsonElement"/> so a graph can be serialised, and the
    /// ONNX path reaches it by serialising each typed attribute and re-parsing. An <c>OnnxGraphProto</c>
    /// serialises to a JSON OBJECT, and <see cref="GetTypedAttributes"/> has no case for an object, so it
    /// fell through to <c>GetRawText()</c> - the subgraph arrived at the operator as a STRING.
    /// <para>
    /// That silently disabled ALL ONNX control flow. <c>IfOperator.Execute</c> tests
    /// <c>branchObj is OnnxGraphProto</c>, which was never true, so neither branch ever ran and the node
    /// emitted whatever its output buffer already held. MEASURED on ZipVoice's text encoder, whose
    /// relative positional-encoding table comes through a single If: onnxruntime returns [1999, 48] of
    /// sin/cos values, we returned the scalar 1.0, and every relative-position bias in all four layers
    /// was computed from it. <c>If</c> was listed in <c>BuiltinOpTypes</c> and had a complete-looking
    /// implementation the whole time.
    /// </para>
    /// </remarks>
    [JsonIgnore]
    public Dictionary<string, object>? RawAttributes { get; set; }

    /// <summary>Convert JSON attributes to typed dictionary for operator execution.</summary>
    public Dictionary<string, object> GetTypedAttributes()
    {
        var result = new Dictionary<string, object>();
        if (Attributes == null) return result;
        foreach (var (key, elem) in Attributes)
        {
            result[key] = elem.ValueKind switch
            {
                JsonValueKind.Number when elem.TryGetInt64(out var l) => l,
                JsonValueKind.Number => elem.GetDouble(),
                JsonValueKind.String => elem.GetString()!,
                JsonValueKind.Array => elem.EnumerateArray().All(e => e.ValueKind == JsonValueKind.Number)
                    ? elem.EnumerateArray().Select(e => e.GetInt64()).ToArray()
                    : elem.EnumerateArray().Select(e => e.GetString()!).ToArray(),
                _ => elem.GetRawText()
            };
        }

        // Out-of-band attributes win: they are the ones JSON could not represent, and the JSON copy of a
        // subgraph is the raw text that used to be mistaken for its value.
        if (RawAttributes != null)
            foreach (var (key, value) in RawAttributes)
                result[key] = value;

        return result;
    }
}
