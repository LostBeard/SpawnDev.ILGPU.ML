namespace SpawnDev.ILGPU.ML.GGUF;

/// <summary>
/// Parsed GGUF model. Contains metadata (architecture, hyperparameters, tokenizer)
/// and tensor descriptors that reference into the raw data blob.
/// </summary>
public class GGUFModel
{
    public uint Version { get; set; }
    public Dictionary<string, object> Metadata { get; set; } = new();
    public GGUFTensorInfo[] Tensors { get; set; } = Array.Empty<GGUFTensorInfo>();
    public uint Alignment { get; set; } = 32;
    public long DataStartOffset { get; set; }
    public byte[] RawData { get; set; } = Array.Empty<byte>();

    // ── Metadata helpers ──

    public string? GetMetadataString(string key) =>
        Metadata.TryGetValue(key, out var v) && v is string s ? s : null;

    public long GetMetadataInt(string key, long defaultValue = 0) =>
        Metadata.TryGetValue(key, out var v) ? Convert.ToInt64(v) : defaultValue;

    public float GetMetadataFloat(string key, float defaultValue = 0) =>
        Metadata.TryGetValue(key, out var v) ? Convert.ToSingle(v) : defaultValue;

    public string[]? GetMetadataStringArray(string key) =>
        Metadata.TryGetValue(key, out var v) && v is string[] arr ? arr : null;

    // ── Architecture info ──

    /// <summary>Model architecture (llama, mistral, phi, qwen, gemma, etc.)</summary>
    public string Architecture => GetMetadataString("general.architecture") ?? "unknown";

    /// <summary>Model name.</summary>
    public string Name => GetMetadataString("general.name") ?? "unnamed";

    /// <summary>Context length (max sequence length).</summary>
    public long ContextLength => GetMetadataInt($"{Architecture}.context_length", 2048);

    /// <summary>Hidden dimension (embedding size).</summary>
    public long EmbeddingLength => GetMetadataInt($"{Architecture}.embedding_length", 0);

    /// <summary>Number of transformer layers.</summary>
    public long BlockCount => GetMetadataInt($"{Architecture}.block_count", 0);

    /// <summary>Number of attention heads.</summary>
    public long AttentionHeadCount => GetMetadataInt($"{Architecture}.attention.head_count", 0);

    /// <summary>Number of KV heads (for GQA).</summary>
    public long AttentionHeadCountKV => GetMetadataInt($"{Architecture}.attention.head_count_kv", 0);

    /// <summary>Vocabulary size.</summary>
    public long VocabSize => GetMetadataInt($"tokenizer.ggml.tokens", 0) > 0
        ? (GetMetadataStringArray("tokenizer.ggml.tokens")?.Length ?? 0)
        : GetMetadataInt($"{Architecture}.vocab_size", 0);

    /// <summary>Get the absolute data offset for a tensor.</summary>
    public long GetTensorDataOffset(GGUFTensorInfo tensor)
    {
        return DataStartOffset + (long)tensor.DataOffset;
    }

    /// <summary>Get the total element count for a tensor.</summary>
    public long GetTensorElementCount(GGUFTensorInfo tensor)
    {
        long count = 1;
        foreach (var d in tensor.Dimensions) count *= d;
        return count;
    }

    /// <summary>Get raw tensor bytes without dequantizing. For Q4/Q8 weights that will
    /// be dequantized on-the-fly during MatMul via FusedDequantMatMul.</summary>
    public byte[]? GetTensorRawBytes(GGUFTensorInfo tensor)
    {
        long elements = GetTensorElementCount(tensor);
        if (elements <= 0) return null;
        long offset = GetTensorDataOffset(tensor);
        long byteSize = GGMLTypes.TypeSize(tensor.Type, elements);
        if (offset + byteSize > RawData.Length) return null;
        var result = new byte[byteSize];
        Buffer.BlockCopy(RawData, (int)offset, result, 0, (int)byteSize);
        return result;
    }

    /// <summary>True if tensor type is a quantized format that FusedDequantMatMul supports.</summary>
    public static bool IsQuantized(GGMLType type) => type is GGMLType.Q4_0 or GGMLType.Q4_1
        or GGMLType.Q5_0 or GGMLType.Q5_1 or GGMLType.Q8_0;

    /// <summary>Get tensor data as float32 (dequantizes if needed).</summary>
    public float[]? GetTensorFloat32(GGUFTensorInfo tensor)
    {
        long elements = GetTensorElementCount(tensor);
        if (elements <= 0) return null;

        long offset = GetTensorDataOffset(tensor);
        if (offset + GGMLTypes.TypeSize(tensor.Type, elements) > RawData.Length) return null;

        return tensor.Type switch
        {
            GGMLType.F32 => ReadF32(offset, elements),
            GGMLType.F16 => ReadF16(offset, elements),
            GGMLType.Q8_0 => DequantizeQ8_0(offset, elements),
            GGMLType.Q4_0 => DequantizeQ4_0(offset, elements),
            GGMLType.Q4_1 => DequantizeQ4_1(offset, elements),
            GGMLType.Q5_0 => DequantizeQ5_0(offset, elements),
            GGMLType.Q5_1 => DequantizeQ5_1(offset, elements),
            _ => null // K-quant types (Q2_K through Q6_K) and IQ types not yet supported
        };
    }

    private float[] ReadF32(long offset, long count)
    {
        var result = new float[count];
        Buffer.BlockCopy(RawData, (int)offset, result, 0, (int)count * 4);
        return result;
    }

    /// <summary>
    /// Dequantize Q8_0: 32 elements per block.
    /// Block layout: [scale:float16] [quants:int8 × 32]
    /// Block size: 2 + 32 = 34 bytes
    /// </summary>
    private float[] DequantizeQ8_0(long offset, long elements)
    {
        var result = new float[elements];
        int numBlocks = (int)(elements / 32);
        for (int block = 0; block < numBlocks; block++)
        {
            int blockOffset = (int)offset + block * 34;
            ushort scaleHalf = (ushort)(RawData[blockOffset] | (RawData[blockOffset + 1] << 8));
            float scale = HalfToFloat(scaleHalf);

            int resultBase = block * 32;
            for (int i = 0; i < 32; i++)
            {
                sbyte q = (sbyte)RawData[blockOffset + 2 + i];
                result[resultBase + i] = q * scale;
            }
        }
        return result;
    }

    /// <summary>
    /// Dequantize Q4_0: 32 elements per block.
    /// Block layout: [scale:float16] [quants:uint8 × 16] (4-bit packed, 2 per byte)
    /// Block size: 2 + 16 = 18 bytes
    /// Each byte holds two 4-bit values (low nibble first, unsigned, offset by -8)
    /// </summary>
    private float[] DequantizeQ4_0(long offset, long elements)
    {
        var result = new float[elements];
        int numBlocks = (int)(elements / 32);
        for (int block = 0; block < numBlocks; block++)
        {
            int blockOffset = (int)offset + block * 18;
            ushort scaleHalf = (ushort)(RawData[blockOffset] | (RawData[blockOffset + 1] << 8));
            float scale = HalfToFloat(scaleHalf);

            int resultBase = block * 32;
            for (int i = 0; i < 16; i++)
            {
                byte packed = RawData[blockOffset + 2 + i];
                int lo = (packed & 0x0F) - 8; // unsigned 4-bit, offset by -8
                int hi = (packed >> 4) - 8;
                result[resultBase + i] = lo * scale;
                result[resultBase + i + 16] = hi * scale;
            }
        }
        return result;
    }

    /// <summary>
    /// Dequantize Q4_1: 32 elements per block.
    /// Block layout: [scale:float16] [min:float16] [quants:uint8 × 16]
    /// Block size: 2 + 2 + 16 = 20 bytes
    /// Each byte holds two 4-bit unsigned values (no offset)
    /// value = quant * scale + min
    /// </summary>
    private float[] DequantizeQ4_1(long offset, long elements)
    {
        var result = new float[elements];
        int numBlocks = (int)(elements / 32);
        for (int block = 0; block < numBlocks; block++)
        {
            int blockOffset = (int)offset + block * 20;
            ushort scaleHalf = (ushort)(RawData[blockOffset] | (RawData[blockOffset + 1] << 8));
            ushort minHalf = (ushort)(RawData[blockOffset + 2] | (RawData[blockOffset + 3] << 8));
            float scale = HalfToFloat(scaleHalf);
            float min = HalfToFloat(minHalf);

            int resultBase = block * 32;
            for (int i = 0; i < 16; i++)
            {
                byte packed = RawData[blockOffset + 4 + i];
                int lo = packed & 0x0F;
                int hi = packed >> 4;
                result[resultBase + i] = lo * scale + min;
                result[resultBase + i + 16] = hi * scale + min;
            }
        }
        return result;
    }

    /// <summary>
    /// Dequantize Q5_0: 32 elements per block.
    /// Block layout: [scale:float16] [high_bits:uint8 × 4] [quants:uint8 × 16]
    /// Block size: 2 + 4 + 16 = 22 bytes
    /// 4-bit base + 1 high bit per element
    /// </summary>
    private float[] DequantizeQ5_0(long offset, long elements)
    {
        var result = new float[elements];
        int numBlocks = (int)(elements / 32);
        for (int block = 0; block < numBlocks; block++)
        {
            int blockOffset = (int)offset + block * 22;
            ushort scaleHalf = (ushort)(RawData[blockOffset] | (RawData[blockOffset + 1] << 8));
            float scale = HalfToFloat(scaleHalf);

            // High bits packed in 4 bytes (32 bits for 32 elements)
            uint highBits = (uint)(RawData[blockOffset + 2] | (RawData[blockOffset + 3] << 8)
                | (RawData[blockOffset + 4] << 16) | (RawData[blockOffset + 5] << 24));

            int resultBase = block * 32;
            for (int i = 0; i < 16; i++)
            {
                byte packed = RawData[blockOffset + 6 + i];
                int lo = packed & 0x0F;
                int hi = packed >> 4;

                // Add high bit
                lo |= (int)((highBits >> i) & 1) << 4;
                hi |= (int)((highBits >> (i + 16)) & 1) << 4;

                result[resultBase + i] = (lo - 16) * scale;
                result[resultBase + i + 16] = (hi - 16) * scale;
            }
        }
        return result;
    }

    /// <summary>
    /// Dequantize Q5_1: 32 elements per block.
    /// Block layout: [scale:float16] [min:float16] [high_bits:uint8 × 4] [quants:uint8 × 16]
    /// Block size: 2 + 2 + 4 + 16 = 24 bytes
    /// value = quant * scale + min (unsigned 5-bit)
    /// </summary>
    private float[] DequantizeQ5_1(long offset, long elements)
    {
        var result = new float[elements];
        int numBlocks = (int)(elements / 32);
        for (int block = 0; block < numBlocks; block++)
        {
            int blockOffset = (int)offset + block * 24;
            ushort scaleHalf = (ushort)(RawData[blockOffset] | (RawData[blockOffset + 1] << 8));
            ushort minHalf = (ushort)(RawData[blockOffset + 2] | (RawData[blockOffset + 3] << 8));
            float scale = HalfToFloat(scaleHalf);
            float min = HalfToFloat(minHalf);

            uint highBits = (uint)(RawData[blockOffset + 4] | (RawData[blockOffset + 5] << 8)
                | (RawData[blockOffset + 6] << 16) | (RawData[blockOffset + 7] << 24));

            int resultBase = block * 32;
            for (int i = 0; i < 16; i++)
            {
                byte packed = RawData[blockOffset + 8 + i];
                int lo = packed & 0x0F;
                int hi = packed >> 4;

                lo |= (int)((highBits >> i) & 1) << 4;
                hi |= (int)((highBits >> (i + 16)) & 1) << 4;

                result[resultBase + i] = lo * scale + min;
                result[resultBase + i + 16] = hi * scale + min;
            }
        }
        return result;
    }

    private float[] ReadF16(long offset, long count)
    {
        var result = new float[count];
        for (long i = 0; i < count; i++)
        {
            int pos = (int)(offset + i * 2);
            ushort fp16 = (ushort)(RawData[pos] | (RawData[pos + 1] << 8));
            result[i] = HalfToFloat(fp16);
        }
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

/// <summary>
/// Describes a single tensor in a GGUF file.
/// </summary>
public class GGUFTensorInfo
{
    public string Name { get; set; } = "";
    public long[] Dimensions { get; set; } = Array.Empty<long>();
    public GGMLType Type { get; set; }
    public ulong DataOffset { get; set; }

    public int[] Shape => Dimensions.Select(d => (int)d).ToArray();
}

/// <summary>GGML quantization types.</summary>
public enum GGMLType : uint
{
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
}

/// <summary>GGUF metadata value types.</summary>
public enum GGUFValueType : uint
{
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

/// <summary>GGML type size calculations.</summary>
public static class GGMLTypes
{
    /// <summary>Get the total byte size for a tensor of the given type and element count.</summary>
    public static long TypeSize(GGMLType type, long elements) => type switch
    {
        GGMLType.F32 => elements * 4,
        GGMLType.F16 => elements * 2,
        GGMLType.Q4_0 => elements / 32 * 18,   // 32 elements per block, 18 bytes per block
        GGMLType.Q4_1 => elements / 32 * 20,
        GGMLType.Q5_0 => elements / 32 * 22,
        GGMLType.Q5_1 => elements / 32 * 24,
        GGMLType.Q8_0 => elements / 32 * 34,
        GGMLType.Q8_1 => elements / 32 * 36,
        GGMLType.Q2_K => elements / 256 * 84,
        GGMLType.Q3_K => elements / 256 * 110,
        GGMLType.Q4_K => elements / 256 * 144,
        GGMLType.Q5_K => elements / 256 * 176,
        GGMLType.Q6_K => elements / 256 * 210,
        GGMLType.I8 => elements,
        GGMLType.I16 => elements * 2,
        GGMLType.I32 => elements * 4,
        GGMLType.I64 => elements * 8,
        GGMLType.F64 => elements * 8,
        _ => elements * 2 // conservative estimate for exotic types
    };
}
