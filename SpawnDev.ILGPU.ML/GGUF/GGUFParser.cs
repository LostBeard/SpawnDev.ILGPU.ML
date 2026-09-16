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

    // ── Header parsing: READ THE REGION ONCE, THEN PARSE IT IN MEMORY ────────────────────────────────
    //
    // 🔴 MEASURED 2026-09-15, Qwen3-1.7B-Q8_0: the "parse" stage was 2.2 s of a 4.5 s warm load - MORE
    // than the OPFS read (1.6 s) and the GPU write (0.15 s) combined, so the load was never I/O bound and
    // tuning chunk size would have optimised the half that was not the problem.
    //
    // It is not the bytes. The header is 5.7 MiB of a 1,749 MiB file, and the tensor table inside it costs
    // 9 ms. It is the COUNT: the tokenizer metadata is 303,323 length-prefixed strings
    // (tokenizer.ggml.tokens 151,936 + merges 151,387 + token_type 151,936), and BOTH old stream paths
    // walked them one field at a time:
    //   - ParseHeader (sync)  : SReadBytes did a raw Stream.Read PER FIELD, allocating a byte[] each time.
    //   - ParseHeaderAsync    : every field was an `async` state machine over a 64 KiB buffer, and one
    //                           u64 is 3 async frames + 2 four-byte allocations, one string ~6 awaits and
    //                           3 allocations -> ~1.8M state-machine transitions and ~900K tiny byte[]
    //                           allocations, on the single-threaded WASM managed heap.
    // A/B on the real file, same bytes, same 303K strings, same UTF8 decode (MEASURED):
    //   per-field over a FileStream .......... 2,657 ms
    //   parse from a byte[] already in memory ..... 16 ms      <- 162x, and the parse was never the cost
    //
    // So both entry points now do what TJ asked for on 2026-09-15 - "sometimes just reading the entire
    // data model into memory and single reading and writing all at once is faster" - and then run the SAME
    // byte[] cursor that Parse(byte[]) uses. That is deliberate: the boxing rules (UInt8 -> byte, Int8 ->
    // sbyte, Int64 -> long ... callers pattern-match on them with `is long a`) now exist in exactly ONE
    // place instead of the three copies that were here, so they cannot drift apart.
    //
    // ⚠️ We do not know the header's length until we have parsed it, so this reads a chunk, tries, and
    // grows on overrun. Growth is geometric and the buffer is reused, so total bytes pulled from the
    // stream is just the final capacity (< 2x the header) - and the retry re-parses only the prefix it
    // already has, which the 16 ms figure above says is free. The stream is only ever read FORWARD and
    // never seeked, preserving the forward-only contract that the HTTP/WebTorrent sources rely on.
    //
    // ⚠️ Over-reading PAST the header is safe here and always was: CreateFromGGUFStreamAsync requires a
    // seekable stream and every later read addresses an ABSOLUTE offset (DataStartOffset + tensor offset,
    // and SourceBytesAsync seeks). The old 64 KiB buffer over-read for the same reason.

    /// <summary>First header read, and the growth step. Covers a 5.7 MiB Qwen3-class header in two reads.</summary>
    private const int HeaderReadChunkBytes = 4 * 1024 * 1024;

    /// <summary>
    /// Refuse to grow past this. A header is megabytes; anything demanding more is a corrupt or non-GGUF
    /// stream, and without this bound a bad length field would pull a multi-GB weight blob into memory
    /// looking for an end that never comes.
    /// </summary>
    private const int MaxHeaderBytes = 256 * 1024 * 1024;

    /// <summary>
    /// Parse ONLY the GGUF header (metadata + tensor infos) from a stream, WITHOUT reading the
    /// tensor-data section. For inspecting large LLM weights from a stream (HttpClient, FileStream,
    /// WebTorrent) — reads a few KB-MB of header and stops at the data boundary; the multi-GB weight
    /// blob is never touched. The returned model has Metadata, Tensors, Version and DataStartOffset
    /// but no RawData (so it is for inspection, not execution).
    /// </summary>
    public static GGUFModel ParseHeader(Stream s)
    {
        var buf = new byte[HeaderReadChunkBytes];
        int len = 0;
        while (true)
        {
            while (len < buf.Length)
            {
                int n = s.Read(buf, len, buf.Length - len);
                if (n == 0) break;
                len += n;
            }
            if (TryParseHeaderFromBuffer(Exact(buf, len), out var model)) return model;
            GrowOrThrow(ref buf, len);
        }
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
        var buf = new byte[HeaderReadChunkBytes];
        int len = 0;
        while (true)
        {
            // ONE await per multi-MiB read instead of one per field. This is the whole fix.
            while (len < buf.Length)
            {
                int n = await s.ReadAsync(buf.AsMemory(len, buf.Length - len), ct).ConfigureAwait(false);
                if (n == 0) break;
                len += n;
            }
            if (TryParseHeaderFromBuffer(Exact(buf, len), out var model)) return model;
            GrowOrThrow(ref buf, len);
        }
    }

    /// <summary>
    /// A byte[] whose Length is EXACTLY the bytes we hold, so that any read past the end of what we have
    /// throws instead of quietly parsing the uninitialised tail as data. That exception is the "need more
    /// bytes" signal, so the exact length is load-bearing, not tidiness.
    /// </summary>
    private static byte[] Exact(byte[] buf, int len) => len == buf.Length ? buf : buf[..len];

    /// <summary>
    /// Double the buffer for another attempt, or throw if the stream is exhausted (we read less than we
    /// asked for) or the header has grown past all reason.
    /// </summary>
    private static void GrowOrThrow(ref byte[] buf, int len)
    {
        if (len < buf.Length)
            throw new EndOfStreamException(
                $"GGUF header truncated: the stream ended after {len:N0} bytes and the header did not parse.");
        if (buf.Length >= MaxHeaderBytes)
            throw new InvalidOperationException(
                $"GGUF header did not parse within {MaxHeaderBytes:N0} bytes - corrupt file or bad length field.");
        Array.Resize(ref buf, Math.Min(buf.Length * 2, MaxHeaderBytes));
    }

    /// <summary>
    /// Parse the header out of an in-memory buffer. Returns false ONLY when it ran off the end of the
    /// buffer (i.e. we need more bytes). A bad magic, an unsupported version or an unknown value type is
    /// a real defect in the data and propagates immediately - without that distinction, pointing this at
    /// a non-GGUF stream would keep doubling the buffer and pull the entire file into memory before
    /// admitting the first four bytes were already wrong.
    /// </summary>
    private static bool TryParseHeaderFromBuffer(byte[] data, out GGUFModel model)
    {
        model = null!;
        try
        {
            int pos = 0;
            uint magic = ReadUInt32(data, ref pos);
            if (magic != GGUF_MAGIC)
                throw new InvalidOperationException($"Not a GGUF file (magic: 0x{magic:X8}, expected 0x{GGUF_MAGIC:X8})");

            var m = new GGUFModel { Version = ReadUInt32(data, ref pos) };
            if (m.Version < 2 || m.Version > 3)
                throw new InvalidOperationException($"Unsupported GGUF version: {m.Version} (expected 2 or 3)");

            ulong tensorCount = ReadUInt64(data, ref pos);
            ulong metadataCount = ReadUInt64(data, ref pos);

            m.Metadata = new Dictionary<string, object>();
            for (ulong i = 0; i < metadataCount; i++)
            {
                var key = ReadString(data, ref pos);
                var valueType = (GGUFValueType)ReadUInt32(data, ref pos);
                m.Metadata[key] = ReadValue(data, ref pos, valueType);
            }

            m.Tensors = new GGUFTensorInfo[tensorCount];
            for (ulong i = 0; i < tensorCount; i++)
            {
                var name = ReadString(data, ref pos);
                uint nDims = ReadUInt32(data, ref pos);
                var dims = new long[nDims];
                for (int d = 0; d < (int)nDims; d++)
                    dims[d] = (long)ReadUInt64(data, ref pos);
                var type = (GGMLType)ReadUInt32(data, ref pos);
                ulong offset = ReadUInt64(data, ref pos);
                m.Tensors[i] = new GGUFTensorInfo { Name = name, Dimensions = dims, Type = type, DataOffset = offset };
            }

            uint alignment = 32;
            if (m.Metadata.TryGetValue("general.alignment", out var alignVal) && alignVal is long a)
                alignment = (uint)a;
            m.Alignment = alignment;
            // pos is the absolute file offset of the end of the header, because the buffer starts at
            // byte 0 of the file. Over-read beyond `pos` is irrelevant - callers seek by absolute offset.
            m.DataStartOffset = (pos + alignment - 1) / alignment * alignment;
            model = m;
            return true;
        }
        catch (Exception ex) when (ex is IndexOutOfRangeException or ArgumentOutOfRangeException or ArgumentException)
        {
            return false; // ran off the end of what we have: the caller reads more and retries
        }
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
