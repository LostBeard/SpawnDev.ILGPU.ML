using System.Text;

namespace SpawnDev.ILGPU.ML.GGUF;

/// <summary>
/// Zero-dependency GGUF model parser.
/// GGUF is the llama.cpp format for LLM weights — simple binary with metadata + tensor data.
/// Supports Llama, Mistral, Phi, Qwen, Gemma, SmolLM, TinyLlama, and any llama.cpp-compatible model.
///
/// Format: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
///
/// Layout:
///   Magic "GGUF" (4 bytes)
///   Version (uint32)
///   Tensor count (uint64)
///   Metadata KV count (uint64)
///   Metadata KV pairs (variable)
///   Tensor info entries (variable)
///   Alignment padding
///   Tensor data (bulk binary)
/// </summary>
public static class GGUFParser
{
    private const uint GGUF_MAGIC = 0x46554747; // "GGUF" in little-endian

    /// <summary>
    /// Parse a GGUF model from raw bytes.
    /// </summary>
    public static GGUFModel Parse(byte[] data)
    {
        var model = new GGUFModel { RawData = data };
        int pos = 0;

        // Magic
        uint magic = ReadUInt32(data, ref pos);
        if (magic != GGUF_MAGIC)
            throw new InvalidOperationException($"Not a GGUF file (magic: 0x{magic:X8}, expected 0x{GGUF_MAGIC:X8})");

        // Version
        model.Version = ReadUInt32(data, ref pos);
        if (model.Version < 2 || model.Version > 3)
            throw new InvalidOperationException($"Unsupported GGUF version: {model.Version} (expected 2 or 3)");

        // Counts
        ulong tensorCount = ReadUInt64(data, ref pos);
        ulong metadataCount = ReadUInt64(data, ref pos);

        // Parse metadata KV pairs
        model.Metadata = new Dictionary<string, object>();
        for (ulong i = 0; i < metadataCount; i++)
        {
            var key = ReadString(data, ref pos);
            var valueType = (GGUFValueType)ReadUInt32(data, ref pos);
            var value = ReadValue(data, ref pos, valueType);
            model.Metadata[key] = value;
        }

        // Parse tensor info entries
        model.Tensors = new GGUFTensorInfo[tensorCount];
        for (ulong i = 0; i < tensorCount; i++)
        {
            var name = ReadString(data, ref pos);
            uint nDims = ReadUInt32(data, ref pos);
            var dims = new long[nDims];
            for (int d = 0; d < (int)nDims; d++)
                dims[d] = (long)ReadUInt64(data, ref pos);

            var type = (GGMLType)ReadUInt32(data, ref pos);
            ulong offset = ReadUInt64(data, ref pos);

            model.Tensors[i] = new GGUFTensorInfo
            {
                Name = name,
                Dimensions = dims,
                Type = type,
                DataOffset = offset
            };
        }

        // Calculate alignment and data start
        uint alignment = 32; // default
        if (model.Metadata.TryGetValue("general.alignment", out var alignVal) && alignVal is long a)
            alignment = (uint)a;
        model.Alignment = alignment;

        // Data starts after tensor info, aligned to alignment boundary
        long dataStart = pos;
        dataStart = (dataStart + alignment - 1) / alignment * alignment;
        model.DataStartOffset = dataStart;

        return model;
    }

    /// <summary>Check if a byte array is a GGUF file.</summary>
    public static bool IsGGUF(byte[] data) =>
        data.Length >= 4 && data[0] == 'G' && data[1] == 'G' && data[2] == 'U' && data[3] == 'F';

    /// <summary>Get a quick summary string.</summary>
    public static string GetSummary(GGUFModel model)
    {
        var arch = model.GetMetadataString("general.architecture") ?? "unknown";
        var name = model.GetMetadataString("general.name") ?? "unnamed";
        return $"GGUF v{model.Version}: {name} ({arch}), {model.Tensors.Length} tensors, " +
               $"{model.Metadata.Count} metadata keys";
    }

    // ── Binary readers ──

    private static uint ReadUInt32(byte[] data, ref int pos)
    {
        uint v = (uint)(data[pos] | (data[pos + 1] << 8) | (data[pos + 2] << 16) | (data[pos + 3] << 24));
        pos += 4;
        return v;
    }

    private static ulong ReadUInt64(byte[] data, ref int pos)
    {
        ulong lo = ReadUInt32(data, ref pos);
        ulong hi = ReadUInt32(data, ref pos);
        return lo | (hi << 32);
    }

    private static float ReadFloat32(byte[] data, ref int pos)
    {
        float v = BitConverter.ToSingle(data, pos);
        pos += 4;
        return v;
    }

    private static double ReadFloat64(byte[] data, ref int pos)
    {
        double v = BitConverter.ToDouble(data, pos);
        pos += 8;
        return v;
    }

    private static string ReadString(byte[] data, ref int pos)
    {
        ulong len = ReadUInt64(data, ref pos);
        var s = Encoding.UTF8.GetString(data, pos, (int)len);
        pos += (int)len;
        return s;
    }

    private static bool ReadBool(byte[] data, ref int pos)
    {
        bool v = data[pos] != 0;
        pos += 1;
        return v;
    }

    private static object ReadValue(byte[] data, ref int pos, GGUFValueType type)
    {
        return type switch
        {
            GGUFValueType.UInt8 => (object)data[pos++],
            GGUFValueType.Int8 => (object)(sbyte)data[pos++],
            GGUFValueType.UInt16 => ReadUInt16(data, ref pos),
            GGUFValueType.Int16 => (short)ReadUInt16(data, ref pos),
            GGUFValueType.UInt32 => ReadUInt32(data, ref pos),
            GGUFValueType.Int32 => (int)ReadUInt32(data, ref pos),
            GGUFValueType.UInt64 => ReadUInt64(data, ref pos),
            GGUFValueType.Int64 => (long)ReadUInt64(data, ref pos),
            GGUFValueType.Float32 => ReadFloat32(data, ref pos),
            GGUFValueType.Float64 => ReadFloat64(data, ref pos),
            GGUFValueType.Bool => ReadBool(data, ref pos),
            GGUFValueType.String => ReadString(data, ref pos),
            GGUFValueType.Array => ReadArray(data, ref pos),
            _ => throw new NotSupportedException($"Unknown GGUF value type: {type}")
        };
    }

    private static ushort ReadUInt16(byte[] data, ref int pos)
    {
        ushort v = (ushort)(data[pos] | (data[pos + 1] << 8));
        pos += 2;
        return v;
    }

    private static object ReadArray(byte[] data, ref int pos)
    {
        var elemType = (GGUFValueType)ReadUInt32(data, ref pos);
        ulong count = ReadUInt64(data, ref pos);

        // For string arrays (common for tokenizer vocab), read as string[]
        if (elemType == GGUFValueType.String)
        {
            var arr = new string[count];
            for (ulong i = 0; i < count; i++)
                arr[i] = ReadString(data, ref pos);
            return arr;
        }

        // For numeric arrays, read as object[]
        var result = new object[count];
        for (ulong i = 0; i < count; i++)
            result[i] = ReadValue(data, ref pos, elemType);
        return result;
    }
}
