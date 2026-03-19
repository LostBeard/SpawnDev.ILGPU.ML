namespace SpawnDev.ILGPU.ML.TensorFlow;

/// <summary>
/// Parsed TensorFlow GraphDef.
/// </summary>
public class TFGraphDef
{
    public List<TFNodeDef> Nodes { get; set; } = new();
}

/// <summary>
/// A single node in a TensorFlow graph.
/// </summary>
public class TFNodeDef
{
    public string Name { get; set; } = "";
    public string Op { get; set; } = "";
    public List<string> Inputs { get; set; } = new();
    public Dictionary<string, TFAttrValue> Attributes { get; set; } = new();

    /// <summary>Get an attribute as int, or default.</summary>
    public long GetAttrInt(string name, long defaultValue = 0) =>
        Attributes.TryGetValue(name, out var v) ? v.IntValue : defaultValue;

    /// <summary>Get an attribute as float, or default.</summary>
    public float GetAttrFloat(string name, float defaultValue = 0) =>
        Attributes.TryGetValue(name, out var v) ? v.FloatValue : defaultValue;

    /// <summary>Get an attribute as string, or default.</summary>
    public string GetAttrString(string name, string defaultValue = "") =>
        Attributes.TryGetValue(name, out var v) && v.StringValue != null ? v.StringValue : defaultValue;

    /// <summary>Get an attribute as bool, or default.</summary>
    public bool GetAttrBool(string name, bool defaultValue = false) =>
        Attributes.TryGetValue(name, out var v) ? v.BoolValue : defaultValue;

    /// <summary>Get shape attribute.</summary>
    public int[]? GetAttrShape(string name) =>
        Attributes.TryGetValue(name, out var v) ? v.Shape : null;
}

/// <summary>
/// Attribute value in a TF NodeDef.
/// </summary>
public class TFAttrValue
{
    public string? StringValue { get; set; }
    public long IntValue { get; set; }
    public float FloatValue { get; set; }
    public bool BoolValue { get; set; }
    public int DataType { get; set; }
    public int[]? Shape { get; set; }
    public byte[]? TensorBytes { get; set; }
}

/// <summary>
/// Maps TF operation names to ONNX equivalents.
/// </summary>
public static class TFOpMapping
{
    public static string? ToOnnxOpType(string tfOp) => tfOp switch
    {
        "Placeholder" => null, // Input — not an op
        "Const" => null,        // Constant/weight — not an op
        "Identity" => "Identity",
        "NoOp" => null,

        // Math
        "Add" or "AddV2" => "Add",
        "Sub" => "Sub",
        "Mul" => "Mul",
        "RealDiv" or "Div" => "Div",
        "Neg" => "Neg",
        "Abs" => "Abs",
        "Sqrt" => "Sqrt",
        "Rsqrt" => "Reciprocal", // 1/sqrt
        "Exp" => "Exp",
        "Log" => "Log",
        "Pow" => "Pow",
        "Floor" => "Floor",
        "Ceil" => "Ceil",
        "Round" => "Identity", // approximate
        "Maximum" => "Max",
        "Minimum" => "Min",
        "Square" => "Mul", // x*x

        // Comparison
        "Equal" => "Equal",
        "Greater" => "Greater",
        "Less" => "Less",
        "GreaterEqual" => "Greater", // approximate
        "LessEqual" => "Less",       // approximate

        // Convolution
        "Conv2D" => "Conv",
        "DepthwiseConv2dNative" => "Conv", // with groups
        "Conv2DBackpropInput" => "ConvTranspose",

        // Pooling
        "MaxPool" => "MaxPool",
        "AvgPool" => "AveragePool",

        // Activation
        "Relu" => "Relu",
        "Relu6" => "Clip", // Clip [0, 6]
        "Sigmoid" or "Logistic" => "Sigmoid",
        "Tanh" => "Tanh",
        "Elu" => "Identity", // approximate
        "LeakyRelu" => "LeakyRelu",
        "Selu" => "Identity", // approximate
        "Softmax" => "Softmax",
        "LogSoftmax" => "Softmax", // approximate

        // Linear
        "MatMul" or "BatchMatMul" or "BatchMatMulV2" => "MatMul",
        "BiasAdd" => "Add",

        // Shape
        "Reshape" => "Reshape",
        "Squeeze" => "Squeeze",
        "ExpandDims" => "Unsqueeze",
        "Transpose" => "Transpose",
        "Shape" => "Shape",
        "Slice" => "Slice",
        "StridedSlice" => "Slice",
        "Concat" or "ConcatV2" => "Concat",
        "Pack" => "Concat", // approximate
        "Unpack" => "Split",
        "Split" or "SplitV" => "Split",
        "Tile" => "Expand",
        "Pad" or "PadV2" or "MirrorPad" => "Pad",
        "Gather" or "GatherV2" => "Gather",
        "GatherNd" => "GatherND",

        // Normalization
        "FusedBatchNorm" or "FusedBatchNormV2" or "FusedBatchNormV3" => "BatchNormalization",
        "LRN" => "Identity", // approximate

        // Reduction
        "Mean" => "ReduceMean",
        "Sum" => "ReduceSum",
        "Max" => "ReduceMax",
        "Min" => "ReduceMin",
        "Prod" => "Identity", // approximate

        // Resize
        "ResizeBilinear" => "Resize",
        "ResizeNearestNeighbor" => "Resize",

        // Cast
        "Cast" => "Cast",
        "Range" => "Range",

        // Other
        "Fill" => "ConstantOfShape",
        "Where" => "Where",
        "TopKV2" => "TopK",
        "ArgMax" => "ArgMax",

        _ => null
    };
}
