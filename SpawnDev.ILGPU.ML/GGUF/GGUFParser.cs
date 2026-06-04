using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

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

    /// <summary>
    /// Parse ONLY the GGUF header (metadata + tensor infos) from a stream, WITHOUT reading the
    /// tensor-data section. For inspecting large LLM weights from a stream (HttpClient, FileStream,
    /// WebTorrent) — reads a few KB-MB of header and stops at the data boundary; the multi-GB weight
    /// blob is never touched. The returned model has Metadata, Tensors, Version and DataStartOffset
    /// but no RawData (so it is for inspection, not execution).
    /// </summary>
    public static GGUFModel ParseHeader(Stream s)
    {
        long pos = 0;
        uint magic = SReadUInt32(s, ref pos);
        if (magic != GGUF_MAGIC)
            throw new InvalidOperationException($"Not a GGUF file (magic: 0x{magic:X8}, expected 0x{GGUF_MAGIC:X8})");

        var model = new GGUFModel { Version = SReadUInt32(s, ref pos) };
        if (model.Version < 2 || model.Version > 3)
            throw new InvalidOperationException($"Unsupported GGUF version: {model.Version} (expected 2 or 3)");

        ulong tensorCount = SReadUInt64(s, ref pos);
        ulong metadataCount = SReadUInt64(s, ref pos);

        model.Metadata = new Dictionary<string, object>();
        for (ulong i = 0; i < metadataCount; i++)
        {
            var key = SReadString(s, ref pos);
            var valueType = (GGUFValueType)SReadUInt32(s, ref pos);
            model.Metadata[key] = SReadValue(s, valueType, ref pos);
        }

        model.Tensors = new GGUFTensorInfo[tensorCount];
        for (ulong i = 0; i < tensorCount; i++)
        {
            var name = SReadString(s, ref pos);
            uint nDims = SReadUInt32(s, ref pos);
            var dims = new long[nDims];
            for (int d = 0; d < (int)nDims; d++)
                dims[d] = (long)SReadUInt64(s, ref pos);
            var type = (GGMLType)SReadUInt32(s, ref pos);
            ulong offset = SReadUInt64(s, ref pos);
            model.Tensors[i] = new GGUFTensorInfo { Name = name, Dimensions = dims, Type = type, DataOffset = offset };
        }

        uint alignment = 32;
        if (model.Metadata.TryGetValue("general.alignment", out var alignVal) && alignVal is long a)
            alignment = (uint)a;
        model.Alignment = alignment;
        long dataStart = (pos + alignment - 1) / alignment * alignment;
        model.DataStartOffset = dataStart;
        return model;
    }

    /// <summary>
    /// Async, forward-only twin of <see cref="ParseHeader"/>. Reads ONLY the GGUF header (metadata +
    /// tensor infos) using <see cref="Stream.ReadAsync(Memory{byte},CancellationToken)"/> exclusively —
    /// never a synchronous <see cref="Stream.Read(byte[],int,int)"/>. This is mandatory for async-only
    /// stream sources (Blazor WASM BlobStream, browser HTTP streams, desktop WebTorrent), where sync
    /// Read throws. The header is forward-only so no seeking is required; the multi-GB weight blob is
    /// never touched.
    /// </summary>
    public static async ValueTask<GGUFModel> ParseHeaderAsync(Stream s, CancellationToken ct = default)
    {
        var r = new AsyncByteSource(s);

        uint magic = await r.ReadUInt32Async(ct).ConfigureAwait(false);
        if (magic != GGUF_MAGIC)
            throw new InvalidOperationException($"Not a GGUF file (magic: 0x{magic:X8}, expected 0x{GGUF_MAGIC:X8})");

        var model = new GGUFModel { Version = await r.ReadUInt32Async(ct).ConfigureAwait(false) };
        if (model.Version < 2 || model.Version > 3)
            throw new InvalidOperationException($"Unsupported GGUF version: {model.Version} (expected 2 or 3)");

        ulong tensorCount = await r.ReadUInt64Async(ct).ConfigureAwait(false);
        ulong metadataCount = await r.ReadUInt64Async(ct).ConfigureAwait(false);

        model.Metadata = new Dictionary<string, object>();
        for (ulong i = 0; i < metadataCount; i++)
        {
            var key = await r.ReadStringAsync(ct).ConfigureAwait(false);
            var valueType = (GGUFValueType)await r.ReadUInt32Async(ct).ConfigureAwait(false);
            model.Metadata[key] = await r.ReadValueAsync(valueType, ct).ConfigureAwait(false);
        }

        model.Tensors = new GGUFTensorInfo[tensorCount];
        for (ulong i = 0; i < tensorCount; i++)
        {
            var name = await r.ReadStringAsync(ct).ConfigureAwait(false);
            uint nDims = await r.ReadUInt32Async(ct).ConfigureAwait(false);
            var dims = new long[nDims];
            for (int d = 0; d < (int)nDims; d++)
                dims[d] = (long)await r.ReadUInt64Async(ct).ConfigureAwait(false);
            var type = (GGMLType)await r.ReadUInt32Async(ct).ConfigureAwait(false);
            ulong offset = await r.ReadUInt64Async(ct).ConfigureAwait(false);
            model.Tensors[i] = new GGUFTensorInfo { Name = name, Dimensions = dims, Type = type, DataOffset = offset };
        }

        uint alignment = 32;
        if (model.Metadata.TryGetValue("general.alignment", out var alignVal) && alignVal is long a)
            alignment = (uint)a;
        model.Alignment = alignment;
        long dataStart = (r.Position + alignment - 1) / alignment * alignment;
        model.DataStartOffset = dataStart;
        return model;
    }

    /// <summary>
    /// Buffered, forward-only ASYNC byte source over a stream. Mirrors the sequential SRead* helpers but
    /// fills its buffer via <see cref="Stream.ReadAsync(Memory{byte},CancellationToken)"/> only, so it
    /// works on async-only streams. Position tracks total bytes consumed (for the data-start alignment).
    /// </summary>
    private sealed class AsyncByteSource
    {
        private readonly Stream _s;
        private readonly byte[] _buf = new byte[64 * 1024];
        private int _bufPos, _bufLen;
        public long Position { get; private set; }

        public AsyncByteSource(Stream s) => _s = s;

        public async ValueTask<byte[]> ReadBytesAsync(int n, CancellationToken ct)
        {
            var outBuf = new byte[n];
            int got = 0;
            while (got < n)
            {
                if (_bufPos >= _bufLen)
                {
                    _bufPos = 0;
                    _bufLen = await _s.ReadAsync(_buf.AsMemory(0, _buf.Length), ct).ConfigureAwait(false);
                    if (_bufLen == 0)
                        throw new EndOfStreamException($"GGUF header truncated: wanted {n} bytes at pos {Position}, got {got}.");
                }
                int take = Math.Min(n - got, _bufLen - _bufPos);
                Array.Copy(_buf, _bufPos, outBuf, got, take);
                _bufPos += take;
                got += take;
            }
            Position += n;
            return outBuf;
        }

        public async ValueTask<uint> ReadUInt32Async(CancellationToken ct)
        {
            var b = await ReadBytesAsync(4, ct).ConfigureAwait(false);
            return (uint)(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24));
        }

        public async ValueTask<ulong> ReadUInt64Async(CancellationToken ct)
        {
            ulong lo = await ReadUInt32Async(ct).ConfigureAwait(false);
            ulong hi = await ReadUInt32Async(ct).ConfigureAwait(false);
            return lo | (hi << 32);
        }

        public async ValueTask<ushort> ReadUInt16Async(CancellationToken ct)
        {
            var b = await ReadBytesAsync(2, ct).ConfigureAwait(false);
            return (ushort)(b[0] | (b[1] << 8));
        }

        public async ValueTask<string> ReadStringAsync(CancellationToken ct)
        {
            ulong len = await ReadUInt64Async(ct).ConfigureAwait(false);
            var b = await ReadBytesAsync((int)len, ct).ConfigureAwait(false);
            return Encoding.UTF8.GetString(b);
        }

        public async ValueTask<object> ReadValueAsync(GGUFValueType type, CancellationToken ct)
        {
            switch (type)
            {
                case GGUFValueType.UInt8: return (await ReadBytesAsync(1, ct).ConfigureAwait(false))[0];
                case GGUFValueType.Int8: return (sbyte)(await ReadBytesAsync(1, ct).ConfigureAwait(false))[0];
                case GGUFValueType.UInt16: return await ReadUInt16Async(ct).ConfigureAwait(false);
                case GGUFValueType.Int16: return (short)await ReadUInt16Async(ct).ConfigureAwait(false);
                case GGUFValueType.UInt32: return await ReadUInt32Async(ct).ConfigureAwait(false);
                case GGUFValueType.Int32: return (int)await ReadUInt32Async(ct).ConfigureAwait(false);
                case GGUFValueType.UInt64: return await ReadUInt64Async(ct).ConfigureAwait(false);
                case GGUFValueType.Int64: return (long)await ReadUInt64Async(ct).ConfigureAwait(false);
                case GGUFValueType.Float32: return BitConverter.ToSingle(await ReadBytesAsync(4, ct).ConfigureAwait(false), 0);
                case GGUFValueType.Float64: return BitConverter.ToDouble(await ReadBytesAsync(8, ct).ConfigureAwait(false), 0);
                case GGUFValueType.Bool: return (await ReadBytesAsync(1, ct).ConfigureAwait(false))[0] != 0;
                case GGUFValueType.String: return await ReadStringAsync(ct).ConfigureAwait(false);
                case GGUFValueType.Array: return await ReadArrayAsync(ct).ConfigureAwait(false);
                default: throw new NotSupportedException($"Unknown GGUF value type: {type}");
            }
        }

        public async ValueTask<object> ReadArrayAsync(CancellationToken ct)
        {
            var elemType = (GGUFValueType)await ReadUInt32Async(ct).ConfigureAwait(false);
            ulong count = await ReadUInt64Async(ct).ConfigureAwait(false);
            if (elemType == GGUFValueType.String)
            {
                var arr = new string[count];
                for (ulong i = 0; i < count; i++) arr[i] = await ReadStringAsync(ct).ConfigureAwait(false);
                return arr;
            }
            var result = new object[count];
            for (ulong i = 0; i < count; i++) result[i] = await ReadValueAsync(elemType, ct).ConfigureAwait(false);
            return result;
        }
    }

    // ── Stream binary readers (sequential, header-only) ──

    private static byte[] SReadBytes(Stream s, int n, ref long pos)
    {
        var buf = new byte[n];
        int total = 0;
        while (total < n)
        {
            int r = s.Read(buf, total, n - total);
            if (r == 0) throw new EndOfStreamException($"GGUF header truncated: wanted {n} bytes at pos {pos}, got {total}.");
            total += r;
        }
        pos += n;
        return buf;
    }

    private static uint SReadUInt32(Stream s, ref long pos)
    {
        var b = SReadBytes(s, 4, ref pos);
        return (uint)(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24));
    }

    private static ulong SReadUInt64(Stream s, ref long pos)
    {
        ulong lo = SReadUInt32(s, ref pos);
        ulong hi = SReadUInt32(s, ref pos);
        return lo | (hi << 32);
    }

    private static ushort SReadUInt16(Stream s, ref long pos)
    {
        var b = SReadBytes(s, 2, ref pos);
        return (ushort)(b[0] | (b[1] << 8));
    }

    private static string SReadString(Stream s, ref long pos)
    {
        ulong len = SReadUInt64(s, ref pos);
        var b = SReadBytes(s, (int)len, ref pos);
        return Encoding.UTF8.GetString(b);
    }

    private static object SReadValue(Stream s, GGUFValueType type, ref long pos)
    {
        switch (type)
        {
            case GGUFValueType.UInt8: return SReadBytes(s, 1, ref pos)[0];
            case GGUFValueType.Int8: return (sbyte)SReadBytes(s, 1, ref pos)[0];
            case GGUFValueType.UInt16: return SReadUInt16(s, ref pos);
            case GGUFValueType.Int16: return (short)SReadUInt16(s, ref pos);
            case GGUFValueType.UInt32: return SReadUInt32(s, ref pos);
            case GGUFValueType.Int32: return (int)SReadUInt32(s, ref pos);
            case GGUFValueType.UInt64: return SReadUInt64(s, ref pos);
            case GGUFValueType.Int64: return (long)SReadUInt64(s, ref pos);
            case GGUFValueType.Float32: return BitConverter.ToSingle(SReadBytes(s, 4, ref pos), 0);
            case GGUFValueType.Float64: return BitConverter.ToDouble(SReadBytes(s, 8, ref pos), 0);
            case GGUFValueType.Bool: return SReadBytes(s, 1, ref pos)[0] != 0;
            case GGUFValueType.String: return SReadString(s, ref pos);
            case GGUFValueType.Array: return SReadArray(s, ref pos);
            default: throw new NotSupportedException($"Unknown GGUF value type: {type}");
        }
    }

    private static object SReadArray(Stream s, ref long pos)
    {
        var elemType = (GGUFValueType)SReadUInt32(s, ref pos);
        ulong count = SReadUInt64(s, ref pos);
        if (elemType == GGUFValueType.String)
        {
            var arr = new string[count];
            for (ulong i = 0; i < count; i++) arr[i] = SReadString(s, ref pos);
            return arr;
        }
        var result = new object[count];
        for (ulong i = 0; i < count; i++) result[i] = SReadValue(s, elemType, ref pos);
        return result;
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
