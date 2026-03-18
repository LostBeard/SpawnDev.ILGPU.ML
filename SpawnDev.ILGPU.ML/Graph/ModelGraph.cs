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
        return result;
    }
}
