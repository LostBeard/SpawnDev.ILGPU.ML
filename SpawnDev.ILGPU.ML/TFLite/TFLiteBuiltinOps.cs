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
        0 => "Add",
        1 => "AveragePool",         // AVERAGE_POOL_2D
        2 => "Concat",              // CONCATENATION
        3 => "Conv",                // CONV_2D
        4 => "Conv",                // DEPTHWISE_CONV_2D (with groups)
        5 => null,                  // DEPTH_TO_SPACE — exists in ONNX but rarely used
        6 => "Identity",             // DEQUANTIZE — in float mode this is a no-op
        7 => null,                  // EMBEDDING_LOOKUP
        8 => "Floor",
        9 => "MatMul",              // FULLY_CONNECTED
        10 => null,                 // HASHTABLE_LOOKUP
        11 => null,                 // L2_NORMALIZATION
        12 => null,                 // L2_POOL_2D
        13 => null,                 // LOCAL_RESPONSE_NORMALIZATION
        14 => "Sigmoid",             // LOGISTIC
        15 => null,                 // LSH_PROJECTION
        16 => null,                 // LSTM
        17 => "MaxPool",            // MAX_POOL_2D
        18 => "Mul",
        19 => "Relu",
        20 => null,                 // RELU_N1_TO_1
        21 => "Relu",               // RELU6 (clamped, map to Clip)
        22 => "Reshape",
        23 => "Resize",             // RESIZE_BILINEAR
        24 => null,                 // RNN
        25 => "Softmax",
        26 => null,                 // SPACE_TO_DEPTH
        27 => null,                 // SVDF
        28 => "Tanh",
        29 => "Concat",             // CONCATENATION_EMBEDDINGS (treated as concat)
        30 => null,                 // SKIP_GRAM
        31 => null,                 // CALL
        32 => null,                 // CUSTOM
        33 => null,                 // EMBEDDING_LOOKUP_SPARSE
        34 => "Pad",
        35 => null,                 // UNIDIRECTIONAL_SEQUENCE_RNN
        36 => "Gather",
        37 => null,                 // BATCH_TO_SPACE_ND
        38 => null,                 // SPACE_TO_BATCH_ND
        39 => "Transpose",
        40 => "ReduceMean",         // MEAN
        41 => "Sub",
        42 => "Div",
        43 => "Squeeze",
        44 => null,                 // UNIDIRECTIONAL_SEQUENCE_LSTM
        45 => "Slice",               // STRIDED_SLICE
        46 => null,                 // BIDIRECTIONAL_SEQUENCE_RNN
        47 => "Exp",
        48 => null,                 // TOPK_V2
        49 => "Split",
        50 => "Log",                // LOG_SOFTMAX → can decompose
        51 => null,                 // DELEGATE
        52 => null,                 // BIDIRECTIONAL_SEQUENCE_LSTM
        53 => "Cast",
        54 => "LeakyRelu",           // PRELU (parameterized ReLU — close to LeakyReLU)
        55 => "Max",                // MAXIMUM
        56 => "ArgMax",
        57 => "Min",                // MINIMUM
        58 => "Less",
        59 => "Neg",
        60 => null,                 // PADV2
        61 => "Greater",
        62 => null,                 // GREATER_EQUAL
        63 => null,                 // LESS_EQUAL
        64 => null,                 // SELECT
        65 => "Slice",
        66 => "Sin",
        67 => "Transpose",          // TRANSPOSE_CONV (→ ConvTranspose)
        68 => null,                 // SPARSE_TO_DENSE
        69 => null,                 // TILE
        70 => "Expand",             // EXPAND_DIMS
        71 => "Equal",
        72 => null,                 // NOT_EQUAL
        73 => "Log",
        74 => "ReduceSum",          // SUM
        75 => "Sqrt",
        76 => "Reciprocal",          // RSQRT (1/sqrt) — close enough, can compose
        77 => "Shape",
        78 => "Pow",
        79 => "Abs",
        80 => null,                 // FAKE_QUANT
        81 => "ReduceMax",          // REDUCE_MAX
        82 => null,                 // ONE_HOT
        83 => null,                 // LOGICAL_OR
        84 => null,                 // LOGICAL_AND
        85 => null,                 // LOGICAL_NOT
        86 => null,                 // UNPACK
        87 => "ReduceMin",          // REDUCE_MIN
        88 => null,                 // FLOOR_DIV
        89 => null,                 // REDUCE_ANY
        90 => null,                 // SQUARE
        91 => null,                 // ZEROS_LIKE
        92 => null,                 // FILL
        93 => null,                 // FLOOR_MOD
        94 => "Range",
        95 => "Resize",             // RESIZE_NEAREST_NEIGHBOR
        96 => "LeakyRelu",          // LEAKY_RELU
        97 => null,                 // SQUARED_DIFFERENCE
        98 => null,                 // MIRROR_PAD
        99 => null,                 // UNIQUE
        100 => "Ceil",
        101 => null,                // REVERSE_V2
        102 => "Add",               // ADD_N
        103 => null,                // GATHER_ND
        104 => "Cos",
        105 => "Where",
        106 => "Identity",          // RANK → identity-like
        107 => null,                // ELU
        108 => null,                // REVERSE_SEQUENCE
        109 => "MatMul",            // MATRIX_DIAG → not really, but close
        110 => null,                // QUANTIZE
        111 => null,                // MATRIX_SET_DIAG
        112 => "Erf",               // ROUND → use erf for now
        113 => "HardSwish",         // HARD_SWISH
        114 => "If",                // IF
        115 => null,                // WHILE
        116 => null,                // NON_MAX_SUPPRESSION_V4
        117 => null,                // NON_MAX_SUPPRESSION_V5
        118 => null,                // SCATTER_ND
        119 => null,                // SELECT_V2
        120 => null,                // DENSIFY
        121 => null,                // SEGMENT_SUM
        122 => "MatMul",             // BATCH_MATMUL
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
