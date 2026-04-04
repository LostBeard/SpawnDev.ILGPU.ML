namespace SpawnDev.ILGPU.ML.SafeTensors;

/// <summary>
/// Parsed SafeTensors file.
/// Contains tensor metadata and raw data for weight extraction.
/// SafeTensors is weights-only — graph construction requires a config.json (architecture).
/// </summary>
public class SafeTensorsFile
{
    public SafeTensorInfo[] Tensors { get; set; } = Array.Empty<SafeTensorInfo>();

    /// <summary>Check if a tensor with the given name exists.</summary>
    public bool HasTensor(string name) => Tensors.Any(t => t.Name == name);
    public Dictionary<string, object> Metadata { get; set; } = new();
    public byte[] RawData { get; set; } = Array.Empty<byte>();
    /// <summary>Offset of tensor data section within RawData (after header).</summary>
    public long DataOffset { get; set; }

    /// <summary>Get tensor data as float32 (converts from stored dtype).</summary>
    public float[]? GetTensorFloat32(SafeTensorInfo tensor)
    {
        long elements = tensor.Shape.Aggregate(1L, (a, b) => a * b);
        if (elements <= 0) return null;

        return tensor.DType switch
        {
            "F32" => ReadF32(tensor),
            "F16" => ReadF16(tensor),
            "BF16" => ReadBF16(tensor),
            "F64" => ReadF64(tensor),
            "I32" => ReadI32AsFloat(tensor),
            "I16" => ReadI16AsFloat(tensor),
            "I8" => ReadI8AsFloat(tensor),
            "U8" => ReadU8AsFloat(tensor),
            "I64" => ReadI64AsFloat(tensor),
            "BOOL" => ReadBoolAsFloat(tensor),
            "U16" => ReadU16AsFloat(tensor),
            "U32" => ReadU32AsFloat(tensor),
            "F8_E4M3" => ReadF16(tensor), // Approximate as F16 (same size)
            "F8_E5M2" => ReadF16(tensor), // Approximate as F16 (same size)
            _ => null
        };
    }

    /// <summary>Extract all tensors as a name → float[] dictionary.</summary>
    public Dictionary<string, float[]> ExtractAllWeights()
    {
        var weights = new Dictionary<string, float[]>();
        foreach (var tensor in Tensors)
        {
            var data = GetTensorFloat32(tensor);
            if (data != null)
                weights[tensor.Name] = data;
        }
        return weights;
    }

    /// <summary>Get tensor shapes as a name → int[] dictionary (for ModelGraph.Initializers).</summary>
    public Dictionary<string, int[]> GetShapes()
    {
        var shapes = new Dictionary<string, int[]>();
        foreach (var tensor in Tensors)
            shapes[tensor.Name] = tensor.Shape;
        return shapes;
    }

    /// <summary>Get the raw data bytes for a tensor (handles multi-shard models).</summary>
    private byte[] GetData(SafeTensorInfo t) => t.ShardData ?? RawData;
    private long GetDataStart(SafeTensorInfo t) => t.ShardData != null ? t.ShardDataOffset + t.DataOffset : t.DataOffset;

    private float[] ReadF32(SafeTensorInfo t)
    {
        int count = (int)(t.DataLength / 4);
        var result = new float[count];
        Buffer.BlockCopy(GetData(t), (int)GetDataStart(t), result, 0, count * 4);
        return result;
    }

    private float[] ReadF16(SafeTensorInfo t)
    {
        int count = (int)(t.DataLength / 2);
        var result = new float[count];
        for (int i = 0; i < count; i++)
        {
            int pos = (int)t.DataOffset + i * 2;
            ushort fp16 = (ushort)(RawData[pos] | (RawData[pos + 1] << 8));
            result[i] = HalfToFloat(fp16);
        }
        return result;
    }

    private float[] ReadBF16(SafeTensorInfo t)
    {
        int count = (int)(t.DataLength / 2);
        var result = new float[count];
        for (int i = 0; i < count; i++)
        {
            int pos = (int)t.DataOffset + i * 2;
            // BF16: same exponent as F32, just truncated mantissa
            // Reconstruct by shifting to upper 16 bits of float32
            ushort bf16 = (ushort)(RawData[pos] | (RawData[pos + 1] << 8));
            uint f32bits = (uint)bf16 << 16;
            result[i] = BitConverter.Int32BitsToSingle((int)f32bits);
        }
        return result;
    }

    private float[] ReadF64(SafeTensorInfo t)
    {
        int count = (int)(t.DataLength / 8);
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = (float)BitConverter.ToDouble(GetData(t), (int)GetDataStart(t) + i * 8);
        return result;
    }

    private float[] ReadI32AsFloat(SafeTensorInfo t)
    {
        int count = (int)(t.DataLength / 4);
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = BitConverter.ToInt32(GetData(t), (int)GetDataStart(t) + i * 4);
        return result;
    }

    private float[] ReadI16AsFloat(SafeTensorInfo t)
    {
        int count = (int)(t.DataLength / 2);
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = BitConverter.ToInt16(GetData(t), (int)GetDataStart(t) + i * 2);
        return result;
    }

    private float[] ReadI8AsFloat(SafeTensorInfo t)
    {
        int count = (int)t.DataLength;
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = (sbyte)GetData(t)[(int)GetDataStart(t) + i];
        return result;
    }

    private float[] ReadU8AsFloat(SafeTensorInfo t)
    {
        int count = (int)t.DataLength;
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = GetData(t)[(int)GetDataStart(t) + i];
        return result;
    }

    private float[] ReadI64AsFloat(SafeTensorInfo t)
    {
        int count = (int)(t.DataLength / 8);
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = BitConverter.ToInt64(GetData(t), (int)GetDataStart(t) + i * 8);
        return result;
    }

    private float[] ReadBoolAsFloat(SafeTensorInfo t)
    {
        int count = (int)t.DataLength;
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = GetData(t)[(int)GetDataStart(t) + i] != 0 ? 1f : 0f;
        return result;
    }

    private float[] ReadU16AsFloat(SafeTensorInfo t)
    {
        int count = (int)(t.DataLength / 2);
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = BitConverter.ToUInt16(GetData(t), (int)GetDataStart(t) + i * 2);
        return result;
    }

    private float[] ReadU32AsFloat(SafeTensorInfo t)
    {
        int count = (int)(t.DataLength / 4);
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = BitConverter.ToUInt32(GetData(t), (int)GetDataStart(t) + i * 4);
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

/// <summary>Describes a single tensor in a SafeTensors file.</summary>
public class SafeTensorInfo
{
    public string Name { get; set; } = "";
    public string DType { get; set; } = "F32";
    public int[] Shape { get; set; } = Array.Empty<int>();
    public long DataOffset { get; set; }
    public long DataLength { get; set; }
    /// <summary>For multi-shard models: the shard's raw data bytes. Null for single-file models.</summary>
    public byte[]? ShardData { get; set; }
    /// <summary>For multi-shard models: offset of tensor data section in ShardData.</summary>
    public long ShardDataOffset { get; set; }
}
