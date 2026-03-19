namespace SpawnDev.ILGPU.ML.Onnx;

/// <summary>
/// Helper for the Model Inspector demo page.
/// Drop any .onnx file → see architecture, operators, shapes, weights.
/// A developer tool that showcases our native ONNX parser.
/// </summary>
public static class ModelInspectorHelper
{
    /// <summary>
    /// Inspect an ONNX model and return a structured summary.
    /// Uses our zero-dependency ONNX parser.
    /// </summary>
    public static InspectionResult Inspect(byte[] onnxBytes)
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

    /// <summary>
    /// Check which operators the model needs that we support vs don't support.
    /// Answers "Can I run this model?" instantly.
    /// </summary>
    public static CompatibilityResult CheckCompatibility(byte[] onnxBytes, Operators.OperatorRegistry? registry = null)
    {
        var model = OnnxParser.Parse(onnxBytes);
        var graph = model.Graph;

        var opsUsed = graph.Nodes.Select(n => n.OpType).Distinct().OrderBy(o => o).ToArray();

        // Known supported operators (hardcoded fallback if no registry provided)
        var knownSupported = new HashSet<string>
        {
            "Abs", "Add", "AveragePool", "BatchNormalization", "Cast", "Ceil", "Clip",
            "Concat", "Constant", "ConstantOfShape", "Conv", "ConvTranspose",
            "DepthToSpace", "Div", "Dropout", "Equal", "Erf", "Exp", "Expand",
            "Flatten", "Floor", "Gather", "Gelu", "Gemm", "GlobalAveragePool",
            "Greater", "HardSigmoid", "HardSwish", "Identity", "InstanceNormalization",
            "LayerNormalization", "LeakyRelu", "Less", "Log", "MatMul", "Max", "MaxPool",
            "Min", "Mul", "Neg", "Not", "Pad", "Pow", "Range", "Reciprocal",
            "ReduceMax", "ReduceMean", "ReduceMin", "ReduceSum",
            "Relu", "Reshape", "Resize", "Shape", "Sigmoid", "Sign", "SiLU",
            "Slice", "Softmax", "Split", "Sqrt", "Squeeze", "Sub",
            "Tanh", "TopK", "Transpose", "Unsqueeze", "Upsample", "Where",
        };

        // If registry provided, use its actual supported ops
        HashSet<string> supportedOps;
        if (registry != null)
        {
            supportedOps = new HashSet<string>();
            foreach (var op in opsUsed)
            {
                if (registry.IsSupported(op))
                    supportedOps.Add(op);
            }
        }
        else
        {
            supportedOps = knownSupported;
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

/// <summary>Result of checking model compatibility with our engine.</summary>
public class CompatibilityResult
{
    public int TotalOpsUsed { get; set; }
    public string[] SupportedOps { get; set; } = Array.Empty<string>();
    public string[] UnsupportedOps { get; set; } = Array.Empty<string>();
    public bool IsFullySupported { get; set; }
    public float CompatibilityPercent { get; set; }

    public string Summary => IsFullySupported
        ? $"Fully compatible ({TotalOpsUsed} operators supported)"
        : $"{CompatibilityPercent:F0}% compatible ({SupportedOps.Length}/{TotalOpsUsed} operators). Missing: {string.Join(", ", UnsupportedOps)}";
}
