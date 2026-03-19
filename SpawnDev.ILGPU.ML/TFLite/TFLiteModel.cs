namespace SpawnDev.ILGPU.ML.TFLite;

/// <summary>
/// Parsed TFLite model structure.
/// </summary>
public class TFLiteModel
{
    public int Version { get; set; }
    public string Description { get; set; } = "";
    public TFLiteOperatorCode[] OperatorCodes { get; set; } = Array.Empty<TFLiteOperatorCode>();
    public TFLiteSubGraph[] Subgraphs { get; set; } = Array.Empty<TFLiteSubGraph>();
    public TFLiteBuffer[] Buffers { get; set; } = Array.Empty<TFLiteBuffer>();

    /// <summary>Raw model bytes — buffers reference offsets into this.</summary>
    public byte[] RawData { get; set; } = Array.Empty<byte>();

    /// <summary>Get the operator name for an opcode index.</summary>
    public string GetOperatorName(int opcodeIndex)
    {
        if (opcodeIndex < 0 || opcodeIndex >= OperatorCodes.Length) return $"unknown_{opcodeIndex}";
        var oc = OperatorCodes[opcodeIndex];
        if (!string.IsNullOrEmpty(oc.CustomCode)) return $"CUSTOM:{oc.CustomCode}";
        return TFLiteBuiltinOps.GetName(oc.BuiltinCode);
    }

    /// <summary>Get tensor buffer data as a float array (handles quantization).</summary>
    public float[]? GetTensorData(TFLiteTensor tensor)
    {
        if (tensor.BufferIndex < 0 || tensor.BufferIndex >= Buffers.Length) return null;
        var buffer = Buffers[tensor.BufferIndex];
        if (buffer.DataLength == 0) return null;

        int elements = 1;
        foreach (var d in tensor.Shape) elements *= d;
        if (elements <= 0) return null;

        return tensor.Type switch
        {
            TFLiteTensorType.Float32 => ReadFloat32Buffer(buffer, elements),
            TFLiteTensorType.Float16 => ReadFloat16Buffer(buffer, elements),
            TFLiteTensorType.Int8 => DequantizeInt8(buffer, tensor, elements),
            TFLiteTensorType.UInt8 => DequantizeUInt8(buffer, tensor, elements),
            TFLiteTensorType.Int32 => ReadInt32AsFloat(buffer, elements),
            _ => null
        };
    }

    private float[] ReadFloat32Buffer(TFLiteBuffer buf, int count)
    {
        var result = new float[count];
        Buffer.BlockCopy(RawData, buf.DataOffset, result, 0, count * 4);
        return result;
    }

    private float[] ReadFloat16Buffer(TFLiteBuffer buf, int count)
    {
        var result = new float[count];
        for (int i = 0; i < count; i++)
        {
            ushort fp16 = (ushort)(RawData[buf.DataOffset + i * 2] | (RawData[buf.DataOffset + i * 2 + 1] << 8));
            result[i] = HalfToFloat(fp16);
        }
        return result;
    }

    private float[] DequantizeInt8(TFLiteBuffer buf, TFLiteTensor tensor, int count)
    {
        var result = new float[count];
        float scale = tensor.Quantization?.Scale?.Length > 0 ? tensor.Quantization.Scale[0] : 1f;
        long zp = tensor.Quantization?.ZeroPoint?.Length > 0 ? tensor.Quantization.ZeroPoint[0] : 0;
        for (int i = 0; i < count; i++)
            result[i] = ((sbyte)RawData[buf.DataOffset + i] - zp) * scale;
        return result;
    }

    private float[] DequantizeUInt8(TFLiteBuffer buf, TFLiteTensor tensor, int count)
    {
        var result = new float[count];
        float scale = tensor.Quantization?.Scale?.Length > 0 ? tensor.Quantization.Scale[0] : 1f / 255f;
        long zp = tensor.Quantization?.ZeroPoint?.Length > 0 ? tensor.Quantization.ZeroPoint[0] : 0;
        for (int i = 0; i < count; i++)
            result[i] = (RawData[buf.DataOffset + i] - zp) * scale;
        return result;
    }

    private float[] ReadInt32AsFloat(TFLiteBuffer buf, int count)
    {
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = BitConverter.ToInt32(RawData, buf.DataOffset + i * 4);
        return result;
    }

    private static float HalfToFloat(ushort h)
    {
        int sign = (h >> 15) & 1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0) return sign == 0 ? 0f : -0f;
        if (exp == 31) return mant == 0 ? (sign == 0 ? float.PositiveInfinity : float.NegativeInfinity) : float.NaN;
        float val = MathF.Pow(2, exp - 15) * (1f + mant / 1024f);
        return sign == 0 ? val : -val;
    }
}

public class TFLiteSubGraph
{
    public string Name { get; set; } = "";
    public TFLiteTensor[] Tensors { get; set; } = Array.Empty<TFLiteTensor>();
    public int[] Inputs { get; set; } = Array.Empty<int>();
    public int[] Outputs { get; set; } = Array.Empty<int>();
    public TFLiteOperator[] Operators { get; set; } = Array.Empty<TFLiteOperator>();
}

public class TFLiteTensor
{
    public string Name { get; set; } = "";
    public int[] Shape { get; set; } = Array.Empty<int>();
    public TFLiteTensorType Type { get; set; }
    public int BufferIndex { get; set; }
    public TFLiteQuantization? Quantization { get; set; }
}

public class TFLiteOperator
{
    public int OpcodeIndex { get; set; }
    public int[] Inputs { get; set; } = Array.Empty<int>();
    public int[] Outputs { get; set; } = Array.Empty<int>();
    public byte BuiltinOptionsType { get; set; }
    public int BuiltinOptionsOffset { get; set; }
}

public class TFLiteOperatorCode
{
    public int BuiltinCode { get; set; }
    public string CustomCode { get; set; } = "";
    public int Version { get; set; } = 1;
}

public class TFLiteBuffer
{
    public int DataOffset { get; set; }
    public int DataLength { get; set; }
}

public class TFLiteQuantization
{
    public float[]? Scale { get; set; }
    public long[]? ZeroPoint { get; set; }
}

/// <summary>
/// TFLite tensor data types.
/// </summary>
public enum TFLiteTensorType : byte
{
    Float32 = 0,
    Float16 = 1,
    Int32 = 2,
    UInt8 = 3,
    Int64 = 4,
    String = 5,
    Bool = 6,
    Int16 = 7,
    Complex64 = 8,
    Int8 = 9,
    Float64 = 10,
    Complex128 = 11,
    UInt64 = 12,
    UInt32 = 14,
    UInt16 = 15,
    Int4 = 16,
}
