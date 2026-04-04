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
        or GGMLType.Q5_0 or GGMLType.Q5_1 or GGMLType.Q8_0 or GGMLType.Q8_1
        or GGMLType.Q2_K or GGMLType.Q3_K or GGMLType.Q4_K or GGMLType.Q5_K or GGMLType.Q6_K;

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
            GGMLType.Q8_1 => DequantizeQ8_1(offset, elements),
            GGMLType.Q2_K => DequantizeQ2_K(offset, elements),
            GGMLType.Q3_K => DequantizeQ3_K(offset, elements),
            GGMLType.Q4_K => DequantizeQ4_K(offset, elements),
            GGMLType.Q5_K => DequantizeQ5_K(offset, elements),
            GGMLType.Q6_K => DequantizeQ6_K(offset, elements),
            _ => null // IQ types not yet supported
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

    /// <summary>
    /// Dequantize Q8_1: 32 elements per block.
    /// Block layout: [scale:float16] [min:float16] [quants:int8 × 32]
    /// Block size: 2 + 2 + 32 = 36 bytes
    /// value = quant * scale + min
    /// </summary>
    private float[] DequantizeQ8_1(long offset, long elements)
    {
        var result = new float[elements];
        int numBlocks = (int)(elements / 32);
        for (int block = 0; block < numBlocks; block++)
        {
            int blockOffset = (int)offset + block * 36;
            ushort scaleHalf = (ushort)(RawData[blockOffset] | (RawData[blockOffset + 1] << 8));
            ushort minHalf = (ushort)(RawData[blockOffset + 2] | (RawData[blockOffset + 3] << 8));
            float scale = HalfToFloat(scaleHalf);
            float min = HalfToFloat(minHalf);
            int resultBase = block * 32;
            for (int i = 0; i < 32; i++)
            {
                sbyte q = (sbyte)RawData[blockOffset + 4 + i];
                result[resultBase + i] = q * scale + min;
            }
        }
        return result;
    }

    // ═══════════════════════════════════════════════════════════
    //  K-quant dequantization (256 elements per super-block)
    //  Based on llama.cpp quantization format
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Q4_K: 256 elements per super-block, 144 bytes per block.
    /// Layout: [d:fp16(2)] [dmin:fp16(2)] [scales:uint8×12] [quants:uint8×128]
    /// Each byte holds two 4-bit values. Scales encode per-sub-block scale and min.
    /// </summary>
    private float[] DequantizeQ4_K(long offset, long elements)
    {
        var result = new float[elements];
        int numBlocks = (int)(elements / 256);
        for (int block = 0; block < numBlocks; block++)
        {
            int bOff = (int)offset + block * 144;
            float d = HalfToFloat((ushort)(RawData[bOff] | (RawData[bOff + 1] << 8)));
            float dmin = HalfToFloat((ushort)(RawData[bOff + 2] | (RawData[bOff + 3] << 8)));
            int scaleOff = bOff + 4;
            int quantOff = bOff + 16; // 4 + 12 scales = 16

            for (int j = 0; j < 8; j++) // 8 sub-blocks of 32
            {
                // Decode 6-bit scales from packed 12 bytes
                int sc, m;
                if (j < 4)
                {
                    sc = RawData[scaleOff + j] & 0x3F;
                    m = RawData[scaleOff + j + 4] & 0x3F;
                }
                else
                {
                    sc = ((RawData[scaleOff + j + 4] & 0xF) | ((RawData[scaleOff + j - 4] >> 6) << 4));
                    m = ((RawData[scaleOff + j + 4] >> 4) | ((RawData[scaleOff + j] >> 6) << 4));
                }
                float scale = d * sc;
                float min = dmin * m;
                int rBase = block * 256 + j * 32;

                for (int i = 0; i < 16; i++)
                {
                    byte packed = RawData[quantOff + j * 16 + i];
                    result[rBase + i] = (packed & 0x0F) * scale - min;
                    result[rBase + i + 16] = (packed >> 4) * scale - min;
                }
            }
        }
        return result;
    }

    /// <summary>
    /// Q6_K: 256 elements per super-block, 210 bytes per block.
    /// Layout: [ql:uint8×128] [qh:uint8×64] [scales:int8×16] [d:fp16(2)]
    /// 6-bit quantization: 4 bits from ql + 2 bits from qh = 6 bits, signed offset by -32.
    /// </summary>
    private float[] DequantizeQ6_K(long offset, long elements)
    {
        var result = new float[elements];
        int numBlocks = (int)(elements / 256);
        for (int block = 0; block < numBlocks; block++)
        {
            int bOff = (int)offset + block * 210;
            int qlOff = bOff;
            int qhOff = bOff + 128;
            int scOff = bOff + 192;
            float d = HalfToFloat((ushort)(RawData[bOff + 208] | (RawData[bOff + 209] << 8)));
            int rBase = block * 256;

            for (int j = 0; j < 16; j++) // 16 sub-blocks of 16
            {
                float scale = d * (sbyte)RawData[scOff + j];
                int subOff = j * 16;
                for (int i = 0; i < 16; i++)
                {
                    int ql4 = (i < 8)
                        ? (RawData[qlOff + subOff / 2 + i] & 0x0F)
                        : (RawData[qlOff + subOff / 2 + i - 8] >> 4);
                    int qh2 = (RawData[qhOff + subOff / 4 + i % 8] >> ((i / 8) * 2 + (subOff / 16 % 2) * 4)) & 0x03;
                    int q = (ql4 | (qh2 << 4)) - 32;
                    result[rBase + subOff + i] = q * scale;
                }
            }
        }
        return result;
    }

    /// <summary>Q2_K: 256 elements, 84 bytes per block.</summary>
    private float[] DequantizeQ2_K(long offset, long elements)
    {
        var result = new float[elements];
        int numBlocks = (int)(elements / 256);
        for (int block = 0; block < numBlocks; block++)
        {
            int bOff = (int)offset + block * 84;
            float d = HalfToFloat((ushort)(RawData[bOff + 80] | (RawData[bOff + 81] << 8)));
            float dmin = HalfToFloat((ushort)(RawData[bOff + 82] | (RawData[bOff + 83] << 8)));
            int rBase = block * 256;

            for (int j = 0; j < 16; j++)
            {
                int sc = RawData[bOff + j] & 0x0F;
                int m = RawData[bOff + j] >> 4;
                float scale = d * sc;
                float min = dmin * m;
                for (int i = 0; i < 16; i++)
                {
                    int byteIdx = bOff + 16 + j * 4 + i / 4;
                    int bitShift = (i % 4) * 2;
                    int q = (RawData[byteIdx] >> bitShift) & 0x03;
                    result[rBase + j * 16 + i] = q * scale - min;
                }
            }
        }
        return result;
    }

    /// <summary>Q3_K: 256 elements, 110 bytes per block.</summary>
    private float[] DequantizeQ3_K(long offset, long elements)
    {
        var result = new float[elements];
        int numBlocks = (int)(elements / 256);
        for (int block = 0; block < numBlocks; block++)
        {
            int bOff = (int)offset + block * 110;
            int hmOff = bOff;          // high bit mask: 32 bytes
            int qOff = bOff + 32;       // quants: 64 bytes (2 bits each, 4 per byte)
            int scOff = bOff + 96;      // scales: 12 bytes
            float d = HalfToFloat((ushort)(RawData[bOff + 108] | (RawData[bOff + 109] << 8)));
            int rBase = block * 256;

            for (int j = 0; j < 16; j++)
            {
                // Decode scale from packed 12 bytes
                int scByte = j < 8 ? RawData[scOff + j] : RawData[scOff + j - 4];
                int sc = j < 8 ? (scByte & 0x0F) : (scByte >> 4);
                sc = sc >= 8 ? sc - 16 : sc; // sign extend 4-bit
                float scale = d * sc;

                for (int i = 0; i < 16; i++)
                {
                    int idx = j * 16 + i;
                    int q2 = (RawData[qOff + idx / 4] >> ((idx % 4) * 2)) & 0x03;
                    int hm = (RawData[hmOff + idx / 8] >> (idx % 8)) & 0x01;
                    int q = (q2 | (hm << 2)) - 4; // 3 bits, offset by -4
                    result[rBase + idx] = q * scale;
                }
            }
        }
        return result;
    }

    /// <summary>Q5_K: 256 elements, 176 bytes per block.</summary>
    private float[] DequantizeQ5_K(long offset, long elements)
    {
        var result = new float[elements];
        int numBlocks = (int)(elements / 256);
        for (int block = 0; block < numBlocks; block++)
        {
            int bOff = (int)offset + block * 176;
            float d = HalfToFloat((ushort)(RawData[bOff] | (RawData[bOff + 1] << 8)));
            float dmin = HalfToFloat((ushort)(RawData[bOff + 2] | (RawData[bOff + 3] << 8)));
            int scOff = bOff + 4;
            int qhOff = bOff + 16;  // 12 scale bytes + 4 header = 16
            int qlOff = bOff + 48;  // 16 + 32 high bits = 48
            int rBase = block * 256;

            for (int j = 0; j < 8; j++)
            {
                int sc, m;
                if (j < 4)
                {
                    sc = RawData[scOff + j] & 0x3F;
                    m = RawData[scOff + j + 4] & 0x3F;
                }
                else
                {
                    sc = ((RawData[scOff + j + 4] & 0xF) | ((RawData[scOff + j - 4] >> 6) << 4));
                    m = ((RawData[scOff + j + 4] >> 4) | ((RawData[scOff + j] >> 6) << 4));
                }
                float scale = d * sc;
                float min = dmin * m;

                for (int i = 0; i < 32; i++)
                {
                    int idx = j * 32 + i;
                    int q4 = i < 16
                        ? (RawData[qlOff + j * 16 + i] & 0x0F)
                        : (RawData[qlOff + j * 16 + i - 16] >> 4);
                    int qh = (RawData[qhOff + idx / 8] >> (idx % 8)) & 0x01;
                    int q = (q4 | (qh << 4));
                    result[rBase + idx] = q * scale - min;
                }
            }
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
