namespace SpawnDev.ILGPU.ML.TFLite;

/// <summary>
/// TFLite BuiltinOperator enum mapping.
/// Maps builtin_code integers to human-readable names and to ONNX-equivalent op types.
/// Values from the TFLite schema (tensorflow/lite/schema/schema.fbs).
/// </summary>
public static class TFLiteBuiltinOps
{
    /// <summary>Get the TFLite operator name for a builtin code.</summary>
    public static string GetName(int builtinCode) =>
        builtinCode < Names.Length && builtinCode >= 0 ? Names[builtinCode] : $"OP_{builtinCode}";

    /// <summary>
    /// Map a TFLite operator to the equivalent ONNX op type name.
    /// Returns null if no direct mapping exists.
    /// </summary>
    public static string? ToOnnxOpType(int builtinCode) => builtinCode switch
    {
        0 => "Add",                 // ADD
        1 => "AveragePool",         // AVERAGE_POOL_2D
        2 => "Concat",              // CONCATENATION
        3 => "Conv",                // CONV_2D
        4 => "Conv",                // DEPTHWISE_CONV_2D (with groups)
        5 => "DepthToSpace",        // DEPTH_TO_SPACE
        6 => "Identity",            // DEQUANTIZE — in float mode this is a no-op
        7 => "Gather",              // EMBEDDING_LOOKUP → Gather on embedding matrix
        8 => "Floor",               // FLOOR
        9 => "MatMul",              // FULLY_CONNECTED → decomposed to MatMul+Add in loader
        10 => "Gather",              // HASHTABLE_LOOKUP → Gather approximation
        11 => "LpNormalization",    // L2_NORMALIZATION
        12 => "LpPool",             // L2_POOL_2D
        13 => "LRN",                // LOCAL_RESPONSE_NORMALIZATION
        14 => "Sigmoid",            // LOGISTIC
        15 => "Identity",            // LSH_PROJECTION → pass-through
        16 => "LSTM",               // LSTM
        17 => "MaxPool",            // MAX_POOL_2D
        18 => "Mul",                // MUL
        19 => "Relu",               // RELU
        20 => "Clip",               // RELU_N1_TO_1 → Clip(min=-1, max=1)
        21 => "Clip",               // RELU6 → Clip(min=0, max=6)
        22 => "Reshape",            // RESHAPE
        23 => "Resize",             // RESIZE_BILINEAR
        24 => "RNN",                // RNN
        25 => "Softmax",            // SOFTMAX
        26 => "SpaceToDepth",       // SPACE_TO_DEPTH
        27 => "RNN",                 // SVDF → RNN approximation (single-value decomposition filter)
        28 => "Tanh",               // TANH
        29 => "Concat",             // CONCAT_EMBEDDINGS
        30 => "Identity",            // SKIP_GRAM → pass-through
        31 => "Identity",            // CALL → pass-through (subgraph invocation)
        32 => "Identity",            // CUSTOM → pass-through (model-specific custom op)
        33 => "Gather",              // EMBEDDING_LOOKUP_SPARSE → Gather approximation
        34 => "Pad",                // PAD
        35 => "RNN",                // UNIDIRECTIONAL_SEQUENCE_RNN
        36 => "Gather",             // GATHER
        37 => "Reshape",            // BATCH_TO_SPACE_ND → reshape approximation
        38 => "Reshape",            // SPACE_TO_BATCH_ND → reshape approximation
        39 => "Transpose",          // TRANSPOSE
        40 => "ReduceMean",         // MEAN
        41 => "Sub",                // SUB
        42 => "Div",                // DIV
        43 => "Squeeze",            // SQUEEZE
        44 => "LSTM",               // UNIDIRECTIONAL_SEQUENCE_LSTM
        45 => "Slice",              // STRIDED_SLICE
        46 => "RNN",                // BIDIRECTIONAL_SEQUENCE_RNN
        47 => "Exp",                // EXP
        48 => "TopK",               // TOPK_V2
        49 => "Split",              // SPLIT
        50 => "LogSoftmax",         // LOG_SOFTMAX
        51 => "Identity",            // DELEGATE → pass-through (runtime delegation)
        52 => "LSTM",               // BIDIRECTIONAL_SEQUENCE_LSTM
        53 => "Cast",               // CAST
        54 => "PRelu",              // PRELU — per-channel learned alpha
        55 => "Max",                // MAXIMUM
        56 => "ArgMax",             // ARG_MAX
        57 => "Min",                // MINIMUM
        58 => "Less",               // LESS
        59 => "Neg",                // NEG
        60 => "Pad",                // PADV2 → same as Pad
        61 => "Greater",            // GREATER
        62 => "GreaterOrEqual",     // GREATER_EQUAL
        63 => "LessOrEqual",        // LESS_EQUAL
        64 => "Where",              // SELECT → Where(condition, x, y)
        65 => "Slice",              // SLICE
        66 => "Sin",                // SIN
        67 => "ConvTranspose",      // TRANSPOSE_CONV
        68 => "ScatterND",          // SPARSE_TO_DENSE → ScatterND approximation
        69 => "Tile",               // TILE
        70 => "Unsqueeze",          // EXPAND_DIMS → Unsqueeze
        71 => "Equal",              // EQUAL
        72 => "Not",                // NOT_EQUAL → decompose: Equal + Not
        73 => "Log",                // LOG
        74 => "ReduceSum",          // SUM
        75 => "Sqrt",               // SQRT
        76 => "Sqrt",               // RSQRT → decompose: Sqrt + Reciprocal in loader
        77 => "Shape",              // SHAPE
        78 => "Pow",                // POW
        79 => "Abs",                // ABS
        80 => "Identity",           // FAKE_QUANT → pass-through in float mode
        81 => "ReduceMax",          // REDUCE_MAX
        82 => "OneHot",             // ONE_HOT
        83 => "Or",                 // LOGICAL_OR
        84 => "And",                // LOGICAL_AND
        85 => "Not",                // LOGICAL_NOT
        86 => "Split",              // UNPACK → Split along axis
        87 => "ReduceMin",          // REDUCE_MIN
        88 => "Div",                // FLOOR_DIV → Div + Floor (approximate as Div)
        89 => "ReduceMax",          // REDUCE_ANY → ReduceMax approximation (any nonzero → max > 0)
        90 => "Mul",                // SQUARE → Mul(x, x) — handled by decomposition
        91 => "ConstantOfShape",    // ZEROS_LIKE → ConstantOfShape with value 0
        92 => "ConstantOfShape",    // FILL → ConstantOfShape
        93 => "Mod",                // FLOOR_MOD
        94 => "Range",              // RANGE
        95 => "Resize",             // RESIZE_NEAREST_NEIGHBOR
        96 => "LeakyRelu",          // LEAKY_RELU
        97 => "Sub",                // SQUARED_DIFFERENCE → decompose: Sub then Mul(x,x)
        98 => "Pad",                // MIRROR_PAD → Pad approximation
        99 => "Unique",             // UNIQUE
        100 => "Ceil",              // CEIL
        101 => "ReverseSequence",   // REVERSE_V2
        102 => "Add",               // ADD_N → multi-input Add
        103 => "GatherND",          // GATHER_ND
        104 => "Cos",               // COS
        105 => "Where",             // WHERE
        106 => "Shape",             // RANK → Shape + Size composition
        107 => "Elu",               // ELU
        108 => "ReverseSequence",   // REVERSE_SEQUENCE
        109 => "EyeLike",           // MATRIX_DIAG → EyeLike approximation
        110 => "QuantizeLinear",    // QUANTIZE
        111 => "Identity",          // MATRIX_SET_DIAG → pass-through approximation
        112 => "Round",             // ROUND
        113 => "HardSwish",         // HARD_SWISH
        114 => "If",                // IF
        115 => "Loop",              // WHILE → Loop
        116 => "NonMaxSuppression", // NON_MAX_SUPPRESSION_V4
        117 => "NonMaxSuppression", // NON_MAX_SUPPRESSION_V5
        118 => "ScatterND",         // SCATTER_ND
        119 => "Where",             // SELECT_V2 → Where
        120 => "Identity",          // DENSIFY → pass-through in float mode
        121 => "ReduceSum",         // SEGMENT_SUM → ReduceSum approximation
        122 => "MatMul",            // BATCH_MATMUL
        _ => null
    };

    // Operator names indexed by builtin code
    private static readonly string[] Names = new[]
    {
        "ADD", "AVERAGE_POOL_2D", "CONCATENATION", "CONV_2D",
        "DEPTHWISE_CONV_2D", "DEPTH_TO_SPACE", "DEQUANTIZE", "EMBEDDING_LOOKUP",
        "FLOOR", "FULLY_CONNECTED", "HASHTABLE_LOOKUP", "L2_NORMALIZATION",
        "L2_POOL_2D", "LOCAL_RESPONSE_NORMALIZATION", "LOGISTIC", "LSH_PROJECTION",
        "LSTM", "MAX_POOL_2D", "MUL", "RELU",
        "RELU_N1_TO_1", "RELU6", "RESHAPE", "RESIZE_BILINEAR",
        "RNN", "SOFTMAX", "SPACE_TO_DEPTH", "SVDF",
        "TANH", "CONCAT_EMBEDDINGS", "SKIP_GRAM", "CALL",
        "CUSTOM", "EMBEDDING_LOOKUP_SPARSE", "PAD", "UNIDIRECTIONAL_SEQUENCE_RNN",
        "GATHER", "BATCH_TO_SPACE_ND", "SPACE_TO_BATCH_ND", "TRANSPOSE",
        "MEAN", "SUB", "DIV", "SQUEEZE",
        "UNIDIRECTIONAL_SEQUENCE_LSTM", "STRIDED_SLICE", "BIDIRECTIONAL_SEQUENCE_RNN", "EXP",
        "TOPK_V2", "SPLIT", "LOG_SOFTMAX", "DELEGATE",
        "BIDIRECTIONAL_SEQUENCE_LSTM", "CAST", "PRELU", "MAXIMUM",
        "ARG_MAX", "MINIMUM", "LESS", "NEG",
        "PADV2", "GREATER", "GREATER_EQUAL", "LESS_EQUAL",
        "SELECT", "SLICE", "SIN", "TRANSPOSE_CONV",
        "SPARSE_TO_DENSE", "TILE", "EXPAND_DIMS", "EQUAL",
        "NOT_EQUAL", "LOG", "SUM", "SQRT",
        "RSQRT", "SHAPE", "POW", "ABS",
        "FAKE_QUANT", "REDUCE_MAX", "ONE_HOT", "LOGICAL_OR",
        "LOGICAL_AND", "LOGICAL_NOT", "UNPACK", "REDUCE_MIN",
        "FLOOR_DIV", "REDUCE_ANY", "SQUARE", "ZEROS_LIKE",
        "FILL", "FLOOR_MOD", "RANGE", "RESIZE_NEAREST_NEIGHBOR",
        "LEAKY_RELU", "SQUARED_DIFFERENCE", "MIRROR_PAD", "UNIQUE",
        "CEIL", "REVERSE_V2", "ADD_N", "GATHER_ND",
        "COS", "WHERE", "RANK", "ELU",
        "REVERSE_SEQUENCE", "MATRIX_DIAG", "QUANTIZE", "MATRIX_SET_DIAG",
        "ROUND", "HARD_SWISH", "IF", "WHILE",
        "NON_MAX_SUPPRESSION_V4", "NON_MAX_SUPPRESSION_V5", "SCATTER_ND", "SELECT_V2",
        "DENSIFY", "SEGMENT_SUM", "BATCH_MATMUL",
    };
}
