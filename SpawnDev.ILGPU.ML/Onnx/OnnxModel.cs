namespace SpawnDev.ILGPU.ML.Onnx;

/// <summary>
/// Minimal ONNX data model classes for inference.
/// Only includes fields needed to load and execute models.
/// </summary>

/// <summary>Top-level ONNX model container.</summary>
public class OnnxModelProto
{
    public long IrVersion { get; set; }
    public OnnxGraphProto Graph { get; set; } = new();
    public List<OnnxOpsetImport> OpsetImports { get; set; } = new();
    public string ProducerName { get; set; } = "";
    public string ProducerVersion { get; set; } = "";
}

/// <summary>The computation graph containing nodes, weights, and I/O specs.</summary>
public class OnnxGraphProto
{
    public string Name { get; set; } = "";
    public List<OnnxNodeProto> Nodes { get; set; } = new();
    public List<OnnxTensorProto> Initializers { get; set; } = new();
    public List<OnnxValueInfoProto> Inputs { get; set; } = new();
    public List<OnnxValueInfoProto> Outputs { get; set; } = new();
    public List<OnnxValueInfoProto> ValueInfo { get; set; } = new();
}

/// <summary>A single operation (node) in the graph.</summary>
public class OnnxNodeProto
{
    public List<string> Inputs { get; set; } = new();
    public List<string> Outputs { get; set; } = new();
    public string Name { get; set; } = "";
    public string OpType { get; set; } = "";
    public string Domain { get; set; } = "";
    public List<OnnxAttributeProto> Attributes { get; set; } = new();
}

/// <summary>A tensor (weight/initializer) in the graph.</summary>
public class OnnxTensorProto
{
    public string Name { get; set; } = "";
    public long[] Dims { get; set; } = Array.Empty<long>();
    public int DataType { get; set; }

    /// <summary>Raw tensor data (most common storage format). Little-endian, fixed-width.</summary>
    public byte[]? RawData { get; set; }

    /// <summary>Float data (packed repeated float, proto field 4). Used when raw_data is absent.</summary>
    public float[]? FloatData { get; set; }

    /// <summary>Int32 data (packed repeated int32, proto field 5). Also stores int8/uint8/fp16.</summary>
    public int[]? Int32Data { get; set; }

    /// <summary>Int64 data (packed repeated int64, proto field 7).</summary>
    public long[]? Int64Data { get; set; }

    /// <summary>Double data (packed repeated double, proto field 10).</summary>
    public double[]? DoubleData { get; set; }

    /// <summary>External data key-value pairs (for models with separate weight files).</summary>
    public Dictionary<string, string>? ExternalData { get; set; }

    /// <summary>0=DEFAULT (data in this message), 1=EXTERNAL (data in separate file).</summary>
    public int DataLocation { get; set; }

    /// <summary>Total number of elements (product of dims).</summary>
    public long ElementCount => Dims.Length > 0 ? Dims.Aggregate(1L, (a, b) => a * b) : 0;

    /// <summary>
    /// Get the tensor data as a float array, converting from the stored format.
    /// Handles: raw_data (FLOAT, FLOAT16), float_data, int32_data (FLOAT16 packed).
    /// </summary>
    public float[] ToFloatArray()
    {
        int count = (int)ElementCount;
        if (count == 0) return Array.Empty<float>();

        // raw_data is the most common format
        if (RawData != null && RawData.Length > 0)
        {
            return DataType switch
            {
                1 => ReadRawFloats(RawData, count),           // FLOAT
                10 => ReadRawFloat16s(RawData, count),        // FLOAT16
                11 => ReadRawDoublesAsFloats(RawData, count), // DOUBLE
                6 => ReadRawInt32sAsFloats(RawData, count),   // INT32
                7 => ReadRawInt64sAsFloats(RawData, count),   // INT64
                2 => ReadRawUint8sAsFloats(RawData, count),   // UINT8
                3 => ReadRawInt8sAsFloats(RawData, count),    // INT8
                16 => ReadRawBFloat16sAsFloats(RawData, count), // BFLOAT16
                _ => throw new NotSupportedException($"Unsupported tensor data type: {DataType}")
            };
        }

        if (FloatData != null) return FloatData;
        if (Int32Data != null && DataType == 10) return ConvertInt32PackedFloat16(Int32Data, count);
        if (Int32Data != null) return Int32Data.Select(x => (float)x).ToArray();
        if (Int64Data != null) return Int64Data.Select(x => (float)x).ToArray();
        if (DoubleData != null) return DoubleData.Select(x => (float)x).ToArray();

        return new float[count]; // Empty tensor
    }

    // ── Raw data converters ──

    private static float[] ReadRawFloats(byte[] raw, int count)
    {
        var result = new float[count];
        Buffer.BlockCopy(raw, 0, result, 0, count * 4);
        return result;
    }

    private static float[] ReadRawFloat16s(byte[] raw, int count)
    {
        var result = new float[count];
        for (int i = 0; i < count; i++)
        {
            ushort h = (ushort)(raw[i * 2] | (raw[i * 2 + 1] << 8));
            result[i] = HalfToFloat(h);
        }
        return result;
    }

    private static float[] ReadRawBFloat16sAsFloats(byte[] raw, int count)
    {
        var result = new float[count];
        for (int i = 0; i < count; i++)
        {
            // BFloat16: upper 16 bits of float32
            ushort bf = (ushort)(raw[i * 2] | (raw[i * 2 + 1] << 8));
            uint f32bits = (uint)bf << 16;
            result[i] = BitConverter.Int32BitsToSingle((int)f32bits);
        }
        return result;
    }

    private static float[] ReadRawDoublesAsFloats(byte[] raw, int count)
    {
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = (float)BitConverter.ToDouble(raw, i * 8);
        return result;
    }

    private static float[] ReadRawInt32sAsFloats(byte[] raw, int count)
    {
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = BitConverter.ToInt32(raw, i * 4);
        return result;
    }

    private static float[] ReadRawInt64sAsFloats(byte[] raw, int count)
    {
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = BitConverter.ToInt64(raw, i * 8);
        return result;
    }

    private static float[] ReadRawUint8sAsFloats(byte[] raw, int count)
    {
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = raw[i];
        return result;
    }

    private static float[] ReadRawInt8sAsFloats(byte[] raw, int count)
    {
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = (sbyte)raw[i];
        return result;
    }

    private static float[] ConvertInt32PackedFloat16(int[] data, int count)
    {
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = HalfToFloat((ushort)(data[i] & 0xFFFF));
        return result;
    }

    /// <summary>IEEE 754 half-precision (16-bit) to single-precision (32-bit) conversion.</summary>
    private static float HalfToFloat(ushort h)
    {
        int sign = (h >> 15) & 1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;

        if (exp == 0)
        {
            if (mant == 0) return sign == 0 ? 0f : -0f;
            // Denormalized
            float val = mant / 1024f * (1f / 16384f);
            return sign == 0 ? val : -val;
        }
        if (exp == 31)
        {
            if (mant == 0) return sign == 0 ? float.PositiveInfinity : float.NegativeInfinity;
            return float.NaN;
        }

        float result = (1f + mant / 1024f) * MathF.Pow(2, exp - 15);
        return sign == 0 ? result : -result;
    }
}

/// <summary>An attribute on a node (e.g., kernel_shape, strides, pads).</summary>
public class OnnxAttributeProto
{
    public string Name { get; set; } = "";
    public OnnxAttributeType Type { get; set; }
    public float F { get; set; }
    public long I { get; set; }
    public byte[]? S { get; set; }
    public OnnxTensorProto? T { get; set; }
    public OnnxGraphProto? G { get; set; }
    public float[]? Floats { get; set; }
    public long[]? Ints { get; set; }
    public List<byte[]>? Strings { get; set; }

    /// <summary>Get string value (S decoded as UTF-8).</summary>
    public string StringValue => S != null ? System.Text.Encoding.UTF8.GetString(S) : "";
}

public enum OnnxAttributeType
{
    UNDEFINED = 0, FLOAT = 1, INT = 2, STRING = 3, TENSOR = 4,
    GRAPH = 5, SPARSE_TENSOR = 11, TYPE_PROTO = 13,
    FLOATS = 6, INTS = 7, STRINGS = 8, TENSORS = 9,
    GRAPHS = 10, SPARSE_TENSORS = 12, TYPE_PROTOS = 14,
}

/// <summary>Input/output value description.</summary>
public class OnnxValueInfoProto
{
    public string Name { get; set; } = "";
    public int ElemType { get; set; }
    public List<OnnxDimension> Shape { get; set; } = new();
}

/// <summary>A dimension in a tensor shape (fixed value or symbolic name).</summary>
public class OnnxDimension
{
    public long? DimValue { get; set; }
    public string? DimParam { get; set; }

    public override string ToString() => DimValue.HasValue ? DimValue.Value.ToString() : DimParam ?? "?";
}

/// <summary>Opset version import.</summary>
public class OnnxOpsetImport
{
    public string Domain { get; set; } = "";
    public long Version { get; set; }
}

/// <summary>ONNX tensor data types.</summary>
public static class OnnxDataType
{
    public const int UNDEFINED = 0;
    public const int FLOAT = 1;
    public const int UINT8 = 2;
    public const int INT8 = 3;
    public const int UINT16 = 4;
    public const int INT16 = 5;
    public const int INT32 = 6;
    public const int INT64 = 7;
    public const int STRING = 8;
    public const int BOOL = 9;
    public const int FLOAT16 = 10;
    public const int DOUBLE = 11;
    public const int UINT32 = 12;
    public const int UINT64 = 13;
    public const int BFLOAT16 = 16;
}
