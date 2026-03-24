using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Onnx;

/// <summary>
/// High-level ONNX model loader. Bridges the parsed protobuf model to
/// the existing ModelGraph + WeightStore architecture.
///
/// Usage:
/// <code>
/// var onnxBytes = await httpClient.GetByteArrayAsync("model.onnx");
/// var (graph, weights) = OnnxLoader.Load(onnxBytes, accelerator);
/// var session = new InferenceSession(accelerator, graph, weights);
/// </code>
///
/// This eliminates the need for the Python extract_onnx.py tool.
/// </summary>
public static class OnnxLoader
{
    /// <summary>
    /// Load an ONNX model from raw bytes.
    /// Returns the model graph info and weight store ready for InferenceSession.
    /// </summary>
    /// <param name="onnxBytes">Raw .onnx file bytes</param>
    /// <param name="accelerator">GPU accelerator for weight upload</param>
    /// <returns>Parsed model info + GPU weight store</returns>
    public static (OnnxModelInfo ModelInfo, Dictionary<string, float[]> Weights) LoadModel(byte[] onnxBytes)
    {
        // Parse protobuf
        var model = OnnxParser.Parse(onnxBytes);

        // Extract graph info
        var modelInfo = ExtractModelInfo(model);

        // Extract weights as float arrays
        var weights = ExtractWeights(model.Graph);

        return (modelInfo, weights);
    }

    /// <summary>
    /// Parse an ONNX file and return just the model info (no weights).
    /// Useful for inspecting a model before loading weights to GPU.
    /// </summary>
    public static OnnxModelInfo ParseModelInfo(byte[] onnxBytes)
    {
        var model = OnnxParser.Parse(onnxBytes);
        return ExtractModelInfo(model);
    }

    /// <summary>
    /// Get a summary of an ONNX model (for display/logging).
    /// </summary>
    public static string GetModelSummary(byte[] onnxBytes)
    {
        var model = OnnxParser.Parse(onnxBytes);
        var graph = model.Graph;

        var opset = model.OpsetImports.FirstOrDefault(o => o.Domain == "")?.Version ?? 0;
        int totalParams = 0;
        long totalBytes = 0;
        foreach (var init in graph.Initializers)
        {
            totalParams += (int)init.ElementCount;
            totalBytes += init.RawData?.Length ?? 0;
        }

        var lines = new List<string>
        {
            $"Model: {graph.Name} (opset {opset}, IR v{model.IrVersion})",
            $"Producer: {model.ProducerName} {model.ProducerVersion}",
            $"Nodes: {graph.Nodes.Count}",
            $"Initializers: {graph.Initializers.Count} ({totalParams:N0} params, {totalBytes / 1024.0 / 1024.0:F1} MB)",
            $"Inputs: {string.Join(", ", graph.Inputs.Select(FormatValueInfo))}",
            $"Outputs: {string.Join(", ", graph.Outputs.Select(FormatValueInfo))}",
            "",
            "Op types used:",
        };

        var opCounts = graph.Nodes.GroupBy(n => n.OpType).OrderByDescending(g => g.Count());
        foreach (var group in opCounts)
        {
            lines.Add($"  {group.Key}: {group.Count()}");
        }

        return string.Join("\n", lines);
    }

    // ──────────────────────────────────────────────
    //  Internal conversion
    // ──────────────────────────────────────────────

    private static OnnxModelInfo ExtractModelInfo(OnnxModelProto model)
    {
        var graph = model.Graph;

        // Build node list
        var nodes = graph.Nodes.Select(n => new OnnxNodeInfo
        {
            Name = n.Name,
            OpType = n.OpType,
            Domain = n.Domain,
            Inputs = n.Inputs.ToArray(),
            Outputs = n.Outputs.ToArray(),
            Attributes = n.Attributes.ToDictionary(
                a => a.Name,
                a => ConvertAttribute(a)),
        }).ToList();

        // Graph inputs that are NOT initializers (true model inputs)
        var initNames = new HashSet<string>(graph.Initializers.Select(i => i.Name));
        var inputNames = graph.Inputs
            .Where(i => !initNames.Contains(i.Name))
            .Select(i => i.Name)
            .ToArray();

        var outputNames = graph.Outputs.Select(o => o.Name).ToArray();

        // Known shapes from value_info + inputs + outputs
        var shapes = new Dictionary<string, int[]>();
        // Model inputs: default dynamic dims to 1 (batch=1 is standard)
        foreach (var vi in graph.Inputs)
        {
            if (vi.Shape.Count > 0)
            {
                shapes[vi.Name] = vi.Shape.Select(d =>
                    d.DimValue.HasValue ? (int)d.DimValue.Value : 1).ToArray();
            }
        }
        // Outputs and intermediates: only include fully-resolved shapes
        // (let the compiler infer shapes for tensors with dynamic dims)
        foreach (var vi in graph.Outputs.Concat(graph.ValueInfo))
        {
            if (vi.Shape.Count > 0 && vi.Shape.All(d => d.DimValue.HasValue))
            {
                shapes[vi.Name] = vi.Shape.Select(d => (int)d.DimValue!.Value).ToArray();
            }
        }

        // Initializer shapes
        foreach (var init in graph.Initializers)
        {
            shapes[init.Name] = init.Dims.Select(d => (int)d).ToArray();
        }

        // Register Constant node outputs as initializers (with shapes from their tensor data).
        // This ensures constant extraction picks them up for ConstantData propagation.
        foreach (var node in graph.Nodes)
        {
            if (node.OpType == "Constant" && node.Outputs.Count > 0)
            {
                var valueAttr = node.Attributes.FirstOrDefault(a => a.Name == "value");
                if (valueAttr?.T != null)
                {
                    var outputName = node.Outputs[0];
                    var tensorShape = valueAttr.T.Dims.Select(d => (int)d).ToArray();
                    if (!shapes.ContainsKey(outputName))
                        shapes[outputName] = tensorShape;
                    if (!initNames.Contains(outputName))
                        initNames.Add(outputName);
                }
            }
        }

        var opset = model.OpsetImports.FirstOrDefault(o => o.Domain == "")?.Version ?? 0;

        return new OnnxModelInfo
        {
            Name = graph.Name,
            Nodes = nodes,
            InputNames = inputNames,
            OutputNames = outputNames,
            ValueShapes = shapes,
            OpsetVersion = (int)opset,
            InitializerNames = initNames.ToArray(),
        };
    }

    private static Dictionary<string, float[]> ExtractWeights(OnnxGraphProto graph)
    {
        var weights = new Dictionary<string, float[]>();
        foreach (var init in graph.Initializers)
        {
            if (init.DataLocation == 1) continue; // External data — skip for now
            weights[init.Name] = init.ToFloatArray();
        }

        // Extract Constant node tensor data into weights.
        // ONNX Constant nodes store data as a 'value' attribute (TensorProto).
        // Without this, Constant node outputs are missing from the weight dictionary
        // and ConstantData, breaking compile-time constant propagation chains
        // (e.g., Upsample scale factor computation: Shape→Gather→Mul→Floor→Concat).
        foreach (var node in graph.Nodes)
        {
            if (node.OpType == "Constant" && node.Outputs.Count > 0)
            {
                var valueAttr = node.Attributes.FirstOrDefault(a => a.Name == "value");
                if (valueAttr?.T != null)
                {
                    var outputName = node.Outputs[0];
                    if (!weights.ContainsKey(outputName))
                        weights[outputName] = valueAttr.T.ToFloatArray();
                }
            }
        }

        return weights;
    }

    private static object ConvertAttribute(OnnxAttributeProto attr)
    {
        return attr.Type switch
        {
            OnnxAttributeType.FLOAT => attr.F,
            OnnxAttributeType.INT => attr.I,
            OnnxAttributeType.STRING => attr.StringValue,
            OnnxAttributeType.FLOATS => attr.Floats ?? Array.Empty<float>(),
            OnnxAttributeType.INTS => attr.Ints ?? Array.Empty<long>(),
            OnnxAttributeType.STRINGS => attr.Strings?.Select(s => System.Text.Encoding.UTF8.GetString(s)).ToArray() ?? Array.Empty<string>(),
            _ => attr.I, // Default to int for unknown types
        };
    }

    private static string FormatValueInfo(OnnxValueInfoProto vi)
    {
        var shape = vi.Shape.Count > 0
            ? $"[{string.Join(",", vi.Shape.Select(d => d.ToString()))}]"
            : "[]";
        return $"{vi.Name}:{shape}";
    }
}

/// <summary>
/// Backend-agnostic model description extracted from ONNX protobuf.
/// Compatible with the existing ModelGraph JSON format — can be used
/// interchangeably with extract_onnx.py output.
/// </summary>
public class OnnxModelInfo
{
    public string Name { get; set; } = "";
    public List<OnnxNodeInfo> Nodes { get; set; } = new();
    public string[] InputNames { get; set; } = Array.Empty<string>();
    public string[] OutputNames { get; set; } = Array.Empty<string>();
    public Dictionary<string, int[]> ValueShapes { get; set; } = new();
    public int OpsetVersion { get; set; }
    public string[] InitializerNames { get; set; } = Array.Empty<string>();
}

/// <summary>A node in the model graph.</summary>
public class OnnxNodeInfo
{
    public string Name { get; set; } = "";
    public string OpType { get; set; } = "";
    public string Domain { get; set; } = "";
    public string[] Inputs { get; set; } = Array.Empty<string>();
    public string[] Outputs { get; set; } = Array.Empty<string>();
    public Dictionary<string, object> Attributes { get; set; } = new();
}
