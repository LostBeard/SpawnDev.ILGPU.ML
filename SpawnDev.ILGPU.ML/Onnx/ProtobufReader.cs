namespace SpawnDev.ILGPU.ML.Onnx;

/// <summary>
/// Minimal protobuf wire format reader. Zero dependencies, zero reflection.
/// Reads directly from a byte span — no allocations except result objects.
/// Implements only what's needed for ONNX model parsing.
/// </summary>
public ref struct ProtobufReader
{
    private readonly ReadOnlySpan<byte> _data;
    private int _pos;
    private readonly int _absoluteBase; // offset of _data[0] within the root byte array

    public ProtobufReader(ReadOnlySpan<byte> data)
    {
        _data = data;
        _pos = 0;
        _absoluteBase = 0;
    }

    public ProtobufReader(byte[] data) : this(data.AsSpan()) { }

    private ProtobufReader(ReadOnlySpan<byte> data, int absoluteBase)
    {
        _data = data;
        _pos = 0;
        _absoluteBase = absoluteBase;
    }

    /// <summary>Whether there's more data to read.</summary>
    public bool HasMore => _pos < _data.Length;

    /// <summary>Current position in the buffer.</summary>
    public int Position => _pos;

    /// <summary>Total length of the buffer.</summary>
    public int Length => _data.Length;

    /// <summary>Remaining bytes.</summary>
    public int Remaining => _data.Length - _pos;

    // ──────────────────────────────────────────────
    //  Wire format primitives
    // ──────────────────────────────────────────────

    /// <summary>Read a varint (unsigned, up to 64 bits).</summary>
    public ulong ReadVarint()
    {
        ulong result = 0;
        int shift = 0;
        while (true)
        {
            if (_pos >= _data.Length) throw new InvalidOperationException("Unexpected end of protobuf data");
            byte b = _data[_pos++];
            result |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) return result;
            shift += 7;
            if (shift > 63) throw new InvalidOperationException("Varint too long");
        }
    }

    /// <summary>Read a signed varint as int32.</summary>
    public int ReadInt32() => (int)ReadVarint();

    /// <summary>Read a signed varint as int64.</summary>
    public long ReadInt64() => (long)ReadVarint();

    /// <summary>Read a tag and decompose into field number + wire type.</summary>
    public (int FieldNumber, int WireType) ReadTag()
    {
        uint tag = (uint)ReadVarint();
        return ((int)(tag >> 3), (int)(tag & 0x07));
    }

    /// <summary>Read a length-delimited field as raw bytes.</summary>
    public ReadOnlySpan<byte> ReadBytes()
    {
        int length = (int)ReadVarint();
        if (_pos + length > _data.Length) throw new InvalidOperationException($"Length-delimited field extends past end of data: need {length} bytes at pos {_pos}, have {_data.Length - _pos}");
        var result = _data.Slice(_pos, length);
        _pos += length;
        return result;
    }

    /// <summary>Read a length-delimited field as a byte array (heap copy).</summary>
    public byte[] ReadByteArray()
    {
        return ReadBytes().ToArray();
    }

    /// <summary>Read a length-delimited field length and skip past it, returning the ABSOLUTE offset
    /// within the root byte array and the length. Zero-copy alternative to ReadByteArray for large
    /// fields — caller reads from source array directly. The absolute offset is correct even when
    /// this reader is a submessage reader (e.g., TensorProto inside AttributeProto).</summary>
    public (int offset, int length) ReadBytesReference()
    {
        int length = (int)ReadVarint();
        if (_pos + length > _data.Length) throw new InvalidOperationException($"Length-delimited field extends past end of data");
        int absoluteOffset = _absoluteBase + _pos;
        _pos += length;
        return (absoluteOffset, length);
    }

    /// <summary>Read a length-delimited field as a UTF-8 string.</summary>
    public string ReadString()
    {
        var bytes = ReadBytes();
        return System.Text.Encoding.UTF8.GetString(bytes);
    }

    /// <summary>Read a fixed 32-bit value as float (IEEE 754).</summary>
    public float ReadFloat()
    {
        if (_pos + 4 > _data.Length) throw new InvalidOperationException("Not enough data for float");
        float result = BitConverter.ToSingle(_data.Slice(_pos, 4));
        _pos += 4;
        return result;
    }

    /// <summary>Read a fixed 64-bit value as double (IEEE 754).</summary>
    public double ReadDouble()
    {
        if (_pos + 8 > _data.Length) throw new InvalidOperationException("Not enough data for double");
        double result = BitConverter.ToDouble(_data.Slice(_pos, 8));
        _pos += 8;
        return result;
    }

    /// <summary>Read a fixed 32-bit unsigned integer.</summary>
    public uint ReadFixed32()
    {
        if (_pos + 4 > _data.Length) throw new InvalidOperationException("Not enough data for fixed32");
        uint result = BitConverter.ToUInt32(_data.Slice(_pos, 4));
        _pos += 4;
        return result;
    }

    /// <summary>Read a fixed 64-bit unsigned integer.</summary>
    public ulong ReadFixed64()
    {
        if (_pos + 8 > _data.Length) throw new InvalidOperationException("Not enough data for fixed64");
        ulong result = BitConverter.ToUInt64(_data.Slice(_pos, 8));
        _pos += 8;
        return result;
    }

    /// <summary>
    /// Read a length-delimited field as a sub-message reader.
    /// The returned reader operates on the sub-span but tracks its absolute
    /// position within the root byte array for zero-copy ReadBytesReference.
    /// </summary>
    public ProtobufReader ReadSubMessage()
    {
        int length = (int)ReadVarint();
        if (_pos + length > _data.Length) throw new InvalidOperationException("Sub-message extends past end of data");
        int subAbsoluteBase = _absoluteBase + _pos;
        var subSpan = _data.Slice(_pos, length);
        _pos += length;
        return new ProtobufReader(subSpan, subAbsoluteBase);
    }

    /// <summary>Skip an unknown field based on its wire type.</summary>
    public void SkipField(int wireType)
    {
        switch (wireType)
        {
            case 0: ReadVarint(); break;           // VARINT
            case 1: _pos += 8; break;              // I64
            case 2:                                 // LEN
                int len = (int)ReadVarint();
                _pos += len;
                break;
            case 5: _pos += 4; break;              // I32
            default:
                throw new InvalidOperationException($"Unknown wire type: {wireType}");
        }
    }

    // ──────────────────────────────────────────────
    //  Packed repeated field readers
    // ──────────────────────────────────────────────

    /// <summary>Read a packed repeated float field.</summary>
    public float[] ReadPackedFloats()
    {
        var bytes = ReadBytes();
        int count = bytes.Length / 4;
        var result = new float[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = BitConverter.ToSingle(bytes.Slice(i * 4, 4));
        }
        return result;
    }

    /// <summary>Read a packed repeated double field.</summary>
    public double[] ReadPackedDoubles()
    {
        var bytes = ReadBytes();
        int count = bytes.Length / 8;
        var result = new double[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = BitConverter.ToDouble(bytes.Slice(i * 8, 8));
        }
        return result;
    }

    /// <summary>Read a packed repeated int32 (varint) field.</summary>
    public int[] ReadPackedInt32s()
    {
        var bytes = ReadBytes();
        var sub = new ProtobufReader(bytes);
        var result = new List<int>();
        while (sub.HasMore)
        {
            result.Add(sub.ReadInt32());
        }
        return result.ToArray();
    }

    /// <summary>Read a packed repeated int64 (varint) field.</summary>
    public long[] ReadPackedInt64s()
    {
        var bytes = ReadBytes();
        var sub = new ProtobufReader(bytes);
        var result = new List<long>();
        while (sub.HasMore)
        {
            result.Add(sub.ReadInt64());
        }
        return result.ToArray();
    }

    /// <summary>Read a packed repeated uint64 (varint) field.</summary>
    public ulong[] ReadPackedUInt64s()
    {
        var bytes = ReadBytes();
        var sub = new ProtobufReader(bytes);
        var result = new List<ulong>();
        while (sub.HasMore)
        {
            result.Add(sub.ReadVarint());
        }
        return result.ToArray();
    }
}
