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

    // Real-world GGUF metadata is heterogeneous: a key may hold a scalar, a string, a bool, or an
    // ARRAY (e.g. tokenizer.ggml.tokens is a string[] of 100k+ tokens). Convert.ToInt64/ToSingle THROWS
    // InvalidCastException on an array and FormatException on a non-numeric string. For inspection we
    // want the default, never a crash — a single odd metadata value must not abort inspecting a 9 GB model.
    public long GetMetadataInt(string key, long defaultValue = 0)
    {
        if (!Metadata.TryGetValue(key, out var v) || v is null) return defaultValue;
        try { return Convert.ToInt64(v); }
        catch (Exception ex) when (ex is InvalidCastException or FormatException or OverflowException)
        { return defaultValue; }
    }

    public float GetMetadataFloat(string key, float defaultValue = 0)
    {
        if (!Metadata.TryGetValue(key, out var v) || v is null) return defaultValue;
        try { return Convert.ToSingle(v); }
        catch (Exception ex) when (ex is InvalidCastException or FormatException or OverflowException)
        { return defaultValue; }
    }

    public string[]? GetMetadataStringArray(string key) =>
        Metadata.TryGetValue(key, out var v) && v is string[] arr ? arr : null;

    public float[]? GetMetadataFloatArray(string key)
    {
        if (!Metadata.TryGetValue(key, out var v)) return null;
        if (v is float[] farr) return farr;
        if (v is object[] oarr) return oarr.Select(o => Convert.ToSingle(o)).ToArray();
        return null;
    }

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
    public long VocabSize
    {
        get
        {
            // tokenizer.ggml.tokens is a STRING ARRAY (the token list); its length IS the vocab size.
            // Must read it via GetMetadataStringArray — the old code called GetMetadataInt on this key,
            // which did Convert.ToInt64(string[]) and threw InvalidCastException on every real LLM GGUF
            // that carries an embedded tokenizer (the 9 GB model exposed it; test.gguf has no tokenizer).
            var tokens = GetMetadataStringArray("tokenizer.ggml.tokens");
            if (tokens is { Length: > 0 }) return tokens.Length;
            return GetMetadataInt($"{Architecture}.vocab_size", 0);
        }
    }

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
    /// Block layout: [d:float16] [s:float16] [quants:int8 × 32]
    /// Block size: 2 + 2 + 32 = 36 bytes
    /// value = quant * d. The second fp16 is s = d * sum(quants), a dot-product
    /// optimization term in ggml - NOT a minimum; ggml has no dequantize_row_q8_1
    /// and Q8_1 never appears as a GGUF weight type (it is an activation format).
    /// Kept for completeness; previously this wrongly added s as a per-element min.
    /// </summary>
    private float[] DequantizeQ8_1(long offset, long elements)
    {
        var result = new float[elements];
        int numBlocks = (int)(elements / 32);
        for (int block = 0; block < numBlocks; block++)
        {
            int blockOffset = (int)offset + block * 36;
            ushort scaleHalf = (ushort)(RawData[blockOffset] | (RawData[blockOffset + 1] << 8));
            float scale = HalfToFloat(scaleHalf);
            int resultBase = block * 32;
            for (int i = 0; i < 32; i++)
            {
                sbyte q = (sbyte)RawData[blockOffset + 4 + i];
                result[resultBase + i] = q * scale;
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
    /// Direct port of ggml dequantize_row_q4_K: the 128 quant bytes are consumed in
    /// FOUR 32-byte chunks; each chunk's LOW nibbles are elements 64t..64t+31 and its
    /// HIGH nibbles are elements 64t+32..64t+63 (two consecutive 6-bit scale/min pairs
    /// per chunk via get_scale_min_k4). NOT a per-16-byte lo/hi split - getting this
    /// order wrong permutes the weights inside every super-block.
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
            int q = bOff + 16; // 4 + 12 scales = 16
            int y = block * 256;
            int isIdx = 0;

            for (int j = 0; j < 256; j += 64) // four 64-element chunks per super-block
            {
                GetScaleMinK4(isIdx + 0, scaleOff, out int sc1, out int m1);
                float d1 = d * sc1; float min1 = dmin * m1;
                GetScaleMinK4(isIdx + 1, scaleOff, out int sc2, out int m2);
                float d2 = d * sc2; float min2 = dmin * m2;

                for (int l = 0; l < 32; l++)
                {
                    byte packed = RawData[q + l];
                    result[y + l] = d1 * (packed & 0x0F) - min1;
                    result[y + 32 + l] = d2 * (packed >> 4) - min2;
                }
                y += 64; q += 32; isIdx += 2;
            }
        }
        return result;
    }

    /// <summary>
    /// ggml get_scale_min_k4: decode the j-th 6-bit (scale, min) pair from the
    /// 12-byte packed scales array of a Q4_K/Q5_K super-block.
    /// </summary>
    private void GetScaleMinK4(int j, int scaleOff, out int sc, out int m)
    {
        if (j < 4)
        {
            sc = RawData[scaleOff + j] & 63;
            m = RawData[scaleOff + j + 4] & 63;
        }
        else
        {
            sc = (RawData[scaleOff + j + 4] & 0xF) | ((RawData[scaleOff + j - 4] >> 6) << 4);
            m = (RawData[scaleOff + j + 4] >> 4) | ((RawData[scaleOff + j] >> 6) << 4);
        }
    }

    /// <summary>
    /// Q6_K: 256 elements per super-block, 210 bytes per block.
    /// Layout: [ql:uint8×128] [qh:uint8×64] [scales:int8×16] [d:fp16(2)]
    /// Direct port of ggml dequantize_row_q6_K: the super-block is TWO 128-element
    /// halves (ql advances 64, qh 32, scales 8 per half). Within a half, for l in
    /// 0..31: element l = ql[l].lo + qh[l] bits 0-1; element l+32 = ql[l+32].lo +
    /// qh[l] bits 2-3; element l+64 = ql[l].hi + qh[l] bits 4-5; element l+96 =
    /// ql[l+32].hi + qh[l] bits 6-7; all minus 32, scale = sc[l/16 + {0,2,4,6}].
    /// </summary>
    private float[] DequantizeQ6_K(long offset, long elements)
    {
        var result = new float[elements];
        int numBlocks = (int)(elements / 256);
        for (int block = 0; block < numBlocks; block++)
        {
            int bOff = (int)offset + block * 210;
            float d = HalfToFloat((ushort)(RawData[bOff + 208] | (RawData[bOff + 209] << 8)));
            int ql = bOff;
            int qh = bOff + 128;
            int sc = bOff + 192;
            int y = block * 256;

            for (int n = 0; n < 256; n += 128) // two 128-element halves
            {
                for (int l = 0; l < 32; l++)
                {
                    int isIdx = l / 16;
                    int hbits = RawData[qh + l];
                    int q1 = ((RawData[ql + l] & 0x0F) | (((hbits >> 0) & 3) << 4)) - 32;
                    int q2 = ((RawData[ql + l + 32] & 0x0F) | (((hbits >> 2) & 3) << 4)) - 32;
                    int q3 = ((RawData[ql + l] >> 4) | (((hbits >> 4) & 3) << 4)) - 32;
                    int q4 = ((RawData[ql + l + 32] >> 4) | (((hbits >> 6) & 3) << 4)) - 32;
                    result[y + l] = d * (sbyte)RawData[sc + isIdx] * q1;
                    result[y + l + 32] = d * (sbyte)RawData[sc + isIdx + 2] * q2;
                    result[y + l + 64] = d * (sbyte)RawData[sc + isIdx + 4] * q3;
                    result[y + l + 96] = d * (sbyte)RawData[sc + isIdx + 6] * q4;
                }
                y += 128; ql += 64; qh += 32; sc += 8;
            }
        }
        return result;
    }

    /// <summary>
    /// Q2_K: 256 elements, 84 bytes per block.
    /// Layout: [scales:uint8×16] [quants:uint8×64] [d:fp16(2)] [dmin:fp16(2)]
    /// Direct port of ggml dequantize_row_q2_K: two 128-element halves; within a half
    /// the 2-bit quants come from the SAME 32 bytes at increasing shifts (0,2,4,6),
    /// 16-element runs alternating bytes q[0..15] / q[16..31]. Scale byte per run:
    /// low nibble × d, high nibble × dmin.
    /// </summary>
    private float[] DequantizeQ2_K(long offset, long elements)
    {
        var result = new float[elements];
        int numBlocks = (int)(elements / 256);
        for (int block = 0; block < numBlocks; block++)
        {
            int bOff = (int)offset + block * 84;
            float d = HalfToFloat((ushort)(RawData[bOff + 80] | (RawData[bOff + 81] << 8)));
            float dmin = HalfToFloat((ushort)(RawData[bOff + 82] | (RawData[bOff + 83] << 8)));
            int q = bOff + 16;
            int y = block * 256;
            int isIdx = 0;

            for (int n = 0; n < 256; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    byte sc = RawData[bOff + isIdx++];
                    float dl = d * (sc & 0x0F); float ml = dmin * (sc >> 4);
                    for (int l = 0; l < 16; l++)
                        result[y++] = dl * ((RawData[q + l] >> shift) & 3) - ml;

                    sc = RawData[bOff + isIdx++];
                    dl = d * (sc & 0x0F); ml = dmin * (sc >> 4);
                    for (int l = 0; l < 16; l++)
                        result[y++] = dl * ((RawData[q + l + 16] >> shift) & 3) - ml;

                    shift += 2;
                }
                q += 32;
            }
        }
        return result;
    }

    /// <summary>
    /// Q3_K: 256 elements, 110 bytes per block.
    /// Layout: [hmask:uint8×32] [quants:uint8×64] [scales:uint8×12] [d:fp16(2)]
    /// Direct port of ggml dequantize_row_q3_K: quant runs mirror Q2_K (same-32-bytes
    /// at shifts 0/2/4/6 per half); the high bit comes from hmask with a bit-plane mask
    /// m that doubles every run ACROSS the whole super-block (1..128, never resets);
    /// value = 2-bit quant MINUS 4 when the hmask bit is NOT set. The 16 6-bit signed
    /// scales unpack via the ggml kmask aux-word scheme (low 4 bits from bytes 0..7,
    /// high 2 bits from bytes 8..11), then -32.
    /// </summary>
    private float[] DequantizeQ3_K(long offset, long elements)
    {
        var result = new float[elements];
        int numBlocks = (int)(elements / 256);
        Span<int> scales = stackalloc int[16];
        Span<uint> aux = stackalloc uint[4];
        for (int block = 0; block < numBlocks; block++)
        {
            int bOff = (int)offset + block * 110;
            int hm = bOff;          // high-bit mask: 32 bytes
            int q = bOff + 32;      // quants: 64 bytes (2 bits each)
            int scOff = bOff + 96;  // scales: 12 bytes
            float d = HalfToFloat((ushort)(RawData[bOff + 108] | (RawData[bOff + 109] << 8)));

            // ggml kmask scale unpack: aux[0..1] = low nibbles source, aux[2] = high-2-bit source
            uint a0 = (uint)(RawData[scOff] | (RawData[scOff + 1] << 8) | (RawData[scOff + 2] << 16) | (RawData[scOff + 3] << 24));
            uint a1 = (uint)(RawData[scOff + 4] | (RawData[scOff + 5] << 8) | (RawData[scOff + 6] << 16) | (RawData[scOff + 7] << 24));
            uint tmp = (uint)(RawData[scOff + 8] | (RawData[scOff + 9] << 8) | (RawData[scOff + 10] << 16) | (RawData[scOff + 11] << 24));
            const uint kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
            aux[2] = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
            for (int i = 0; i < 16; i++)
                scales[i] = (sbyte)(byte)(aux[i / 4] >> ((i % 4) * 8));

            int y = block * 256;
            int isIdx = 0;
            int mBit = 1;
            for (int n = 0; n < 256; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    float dl = d * (scales[isIdx++] - 32);
                    for (int l = 0; l < 16; l++)
                        result[y++] = dl * (((RawData[q + l] >> shift) & 3) - ((RawData[hm + l] & mBit) != 0 ? 0 : 4));

                    dl = d * (scales[isIdx++] - 32);
                    for (int l = 0; l < 16; l++)
                        result[y++] = dl * (((RawData[q + l + 16] >> shift) & 3) - ((RawData[hm + l + 16] & mBit) != 0 ? 0 : 4));

                    shift += 2;
                    mBit <<= 1;
                }
                q += 32;
            }
        }
        return result;
    }

    /// <summary>
    /// Q5_K: 256 elements, 176 bytes per block.
    /// Layout: [d:fp16(2)] [dmin:fp16(2)] [scales:uint8×12] [qh:uint8×32] [ql:uint8×128]
    /// Direct port of ggml dequantize_row_q5_K: four 64-element chunks like Q4_K
    /// (lo nibbles of 32 bytes, then hi nibbles, ql advances 32/chunk); the fifth bit
    /// for element l of a chunk comes from qh[l] (qh does NOT advance) under bit-plane
    /// masks u1/u2 that shift left 2 each chunk (lo: 1,4,16,64; hi: 2,8,32,128).
    /// </summary>
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
            int qh = bOff + 16;  // 4 header + 12 scales = 16
            int ql = bOff + 48;  // 16 + 32 high bits = 48
            int y = block * 256;
            int isIdx = 0;
            int u1 = 1, u2 = 2;

            for (int j = 0; j < 256; j += 64)
            {
                GetScaleMinK4(isIdx + 0, scOff, out int sc1, out int m1);
                float d1 = d * sc1; float min1 = dmin * m1;
                GetScaleMinK4(isIdx + 1, scOff, out int sc2, out int m2);
                float d2 = d * sc2; float min2 = dmin * m2;

                for (int l = 0; l < 32; l++)
                {
                    int hbits = RawData[qh + l];
                    result[y + l] = d1 * ((RawData[ql + l] & 0x0F) + ((hbits & u1) != 0 ? 16 : 0)) - min1;
                    result[y + 32 + l] = d2 * ((RawData[ql + l] >> 4) + ((hbits & u2) != 0 ? 16 : 0)) - min2;
                }
                y += 64; ql += 32; isIdx += 2;
                u1 <<= 2; u2 <<= 2;
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
