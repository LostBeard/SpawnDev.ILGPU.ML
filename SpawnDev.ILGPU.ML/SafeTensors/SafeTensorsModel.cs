namespace SpawnDev.ILGPU.ML.SafeTensors;

/// <summary>
/// Parsed SafeTensors file.
/// Contains tensor metadata and raw data for weight extraction.
/// SafeTensors is weights-only — graph construction requires a config.json (architecture).
/// </summary>
public class SafeTensorsFile
{
    public SafeTensorInfo[] Tensors { get; set; } = Array.Empty<SafeTensorInfo>();
    public Dictionary<string, object> Metadata { get; set; } = new();
    public byte[] RawData { get; set; } = Array.Empty<byte>();

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

    private float[] ReadF32(SafeTensorInfo t)
    {
        int count = (int)(t.DataLength / 4);
        var result = new float[count];
        Buffer.BlockCopy(RawData, (int)t.DataOffset, result, 0, count * 4);
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
            result[i] = (float)BitConverter.ToDouble(RawData, (int)t.DataOffset + i * 8);
        return result;
    }

    private float[] ReadI32AsFloat(SafeTensorInfo t)
    {
        int count = (int)(t.DataLength / 4);
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = BitConverter.ToInt32(RawData, (int)t.DataOffset + i * 4);
        return result;
    }

    private float[] ReadI16AsFloat(SafeTensorInfo t)
    {
        int count = (int)(t.DataLength / 2);
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = BitConverter.ToInt16(RawData, (int)t.DataOffset + i * 2);
        return result;
    }

    private float[] ReadI8AsFloat(SafeTensorInfo t)
    {
        int count = (int)t.DataLength;
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = (sbyte)RawData[(int)t.DataOffset + i];
        return result;
    }

    private float[] ReadU8AsFloat(SafeTensorInfo t)
    {
        int count = (int)t.DataLength;
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = RawData[(int)t.DataOffset + i];
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
}
