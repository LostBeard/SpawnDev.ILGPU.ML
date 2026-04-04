namespace SpawnDev.ILGPU.ML.TFLite;

/// <summary>
/// Minimal FlatBuffers reader for TFLite model parsing. Zero dependencies.
/// FlatBuffers uses relative offsets from positions in the buffer.
/// All multi-byte values are little-endian.
/// </summary>
public class FlatBufferReader
{
    private readonly byte[] _data;

    public FlatBufferReader(byte[] data) => _data = data;

    public byte[] Data => _data;
    public int Length => _data.Length;

    // ── Primitive reads ──

    public byte ReadByte(int offset) => _data[offset];

    public short ReadInt16(int offset) =>
        (short)(_data[offset] | (_data[offset + 1] << 8));

    public ushort ReadUInt16(int offset) =>
        (ushort)(_data[offset] | (_data[offset + 1] << 8));

    public int ReadInt32(int offset) =>
        _data[offset] | (_data[offset + 1] << 8) | (_data[offset + 2] << 16) | (_data[offset + 3] << 24);

    public uint ReadUInt32(int offset) => (uint)ReadInt32(offset);

    public long ReadInt64(int offset) =>
        (long)(uint)ReadInt32(offset) | ((long)(uint)ReadInt32(offset + 4) << 32);

    public float ReadFloat32(int offset) => BitConverter.ToSingle(_data, offset);

    // ── FlatBuffers navigation ──

    /// <summary>
    /// Read the root table offset. FlatBuffers files start with a 4-byte offset to the root table.
    /// </summary>
    public int GetRootTableOffset() => ReadInt32(0);

    /// <summary>
    /// Given a table offset, read the vtable offset and return it.
    /// The vtable offset is stored as a negative soffset_t at the table position.
    /// </summary>
    public int GetVTableOffset(int tableOffset)
    {
        int vtableRelative = ReadInt32(tableOffset);
        return tableOffset - vtableRelative;
    }

    /// <summary>
    /// Read a field offset from a vtable. Returns 0 if the field is not present.
    /// VTable layout: [vtable_size:uint16, table_size:uint16, field0_offset:uint16, field1_offset:uint16, ...]
    /// </summary>
    public int GetFieldOffset(int vtableOffset, int fieldIndex)
    {
        int vtableSize = ReadUInt16(vtableOffset);
        int fieldByteOffset = 4 + fieldIndex * 2; // skip vtable_size + table_size
        if (fieldByteOffset >= vtableSize) return 0; // field not present
        return ReadUInt16(vtableOffset + fieldByteOffset);
    }

    /// <summary>
    /// Read a scalar field from a table. Returns defaultValue if not present.
    /// </summary>
    public int ReadFieldInt32(int tableOffset, int fieldIndex, int defaultValue = 0)
    {
        int vtable = GetVTableOffset(tableOffset);
        int fieldOff = GetFieldOffset(vtable, fieldIndex);
        return fieldOff == 0 ? defaultValue : ReadInt32(tableOffset + fieldOff);
    }

    public byte ReadFieldByte(int tableOffset, int fieldIndex, byte defaultValue = 0)
    {
        int vtable = GetVTableOffset(tableOffset);
        int fieldOff = GetFieldOffset(vtable, fieldIndex);
        return fieldOff == 0 ? defaultValue : ReadByte(tableOffset + fieldOff);
    }

    public float ReadFieldFloat(int tableOffset, int fieldIndex, float defaultValue = 0f)
    {
        int vtable = GetVTableOffset(tableOffset);
        int fieldOff = GetFieldOffset(vtable, fieldIndex);
        return fieldOff == 0 ? defaultValue : BitConverter.ToSingle(_data, tableOffset + fieldOff);
    }

    public sbyte ReadFieldSByte(int tableOffset, int fieldIndex, sbyte defaultValue = 0)
    {
        int vtable = GetVTableOffset(tableOffset);
        int fieldOff = GetFieldOffset(vtable, fieldIndex);
        return fieldOff == 0 ? defaultValue : (sbyte)ReadByte(tableOffset + fieldOff);
    }

    /// <summary>
    /// Read an offset field (reference to another table or vector).
    /// Returns the absolute offset, or 0 if field not present.
    /// </summary>
    public int ReadFieldOffset(int tableOffset, int fieldIndex)
    {
        int vtable = GetVTableOffset(tableOffset);
        int fieldOff = GetFieldOffset(vtable, fieldIndex);
        if (fieldOff == 0) return 0;
        int relativeOffset = ReadInt32(tableOffset + fieldOff);
        return tableOffset + fieldOff + relativeOffset;
    }

    // ── Vector operations ──

    /// <summary>
    /// Read a vector length. Vectors start with a 4-byte length prefix.
    /// </summary>
    public int VectorLength(int vectorOffset)
    {
        return vectorOffset == 0 ? 0 : ReadInt32(vectorOffset);
    }

    /// <summary>
    /// Get the offset of element i in a vector of offsets (tables).
    /// Each element is stored as an offset_t (4 bytes).
    /// </summary>
    public int VectorTableElement(int vectorOffset, int index)
    {
        int elemStart = vectorOffset + 4 + index * 4;
        int relOffset = ReadInt32(elemStart);
        return elemStart + relOffset;
    }

    /// <summary>
    /// Get the offset of element i in a vector of scalars.
    /// </summary>
    public int VectorScalarOffset(int vectorOffset, int index, int elemSize)
    {
        return vectorOffset + 4 + index * elemSize;
    }

    /// <summary>
    /// Read an int32 vector element.
    /// </summary>
    public int VectorInt32(int vectorOffset, int index)
    {
        return ReadInt32(vectorOffset + 4 + index * 4);
    }

    /// <summary>
    /// Read a byte vector as a byte array (copy).
    /// </summary>
    public byte[] VectorBytes(int vectorOffset)
    {
        int len = VectorLength(vectorOffset);
        if (len == 0) return Array.Empty<byte>();
        var result = new byte[len];
        Array.Copy(_data, vectorOffset + 4, result, 0, len);
        return result;
    }

    /// <summary>
    /// Read a byte vector as a span (no copy).
    /// </summary>
    public ReadOnlySpan<byte> VectorBytesSpan(int vectorOffset)
    {
        int len = VectorLength(vectorOffset);
        if (len == 0) return ReadOnlySpan<byte>.Empty;
        return _data.AsSpan(vectorOffset + 4, len);
    }

    /// <summary>
    /// Read a string (FlatBuffers strings are length-prefixed UTF-8).
    /// </summary>
    public string ReadString(int stringOffset)
    {
        if (stringOffset == 0) return "";
        int len = ReadInt32(stringOffset);
        return System.Text.Encoding.UTF8.GetString(_data, stringOffset + 4, len);
    }

    /// <summary>
    /// Read a string field from a table.
    /// </summary>
    public string ReadFieldString(int tableOffset, int fieldIndex)
    {
        int off = ReadFieldOffset(tableOffset, fieldIndex);
        return off == 0 ? "" : ReadString(off);
    }
}
