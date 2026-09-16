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

    // ── The retry no longer re-materialises the vocab ───────────────────────────────────────────────
    //
    // 🔴 MEASURED 2026-09-16, Qwen3-1.7B-Q8_0 header (5,951,136 B = 5.68 MiB, 151,936 tokens +
    // 151,387 merges), desktop .NET Release. `ModelInspector.Console --parsebench --only=stream`,
    // min of 15, THREE separate processes per arm, off a MemoryStream so no disk is in the number:
    //
    //   ParseHeaderAsync, before ....... 37.5 / 37.9 / 38.7 ms
    //   ParseHeaderAsync, after ........ 18.4 / 18.5 / 18.8 ms      <- 2.0x
    //
    // Where the old time went. The first read is 4 MiB against a 5.68 MiB header, so the first attempt
    // MATERIALISED about 70% of a 303K-string vocab - every string allocated and UTF8-decoded, into a
    // Dictionary - and then threw all of it away on the overrun.
    //
    // Two things were removed:
    //   1. The wasted materialise. TryScanHeaderEnd walks the SAME structure allocation-free: no string
    //      decoded, no dictionary built, no array materialised, and a numeric array skipped in ONE add
    //      rather than a loop. The grow-and-retry loop now retries the CHEAP walk - both scans together
    //      cost 3.1 ms - and the materialising parse runs exactly once.
    //   2. The exact-length copy. `Exact(buf, len)` sliced the 8 MiB buffer down to the 5.95 MiB actually
    //      held, purely so that an overrun would throw; that is a large-object allocation, a zeroing and
    //      a memcpy to say what an int already says. The scan bounds-checks against an explicit `limit`
    //      instead. MEASURED at 5.7 ms of a 39.9 ms parse before it was removed.
    //
    // ⚠️ MEASURE THE ARMS IN SEPARATE PROCESSES. Interleaving them in one process looked fair and was
    // not: the stream path's multi-MiB buffers move the GC schedule under the floor path, and the same
    // untouched Parse(byte[]) call measured 8.9 ms in one configuration and 14.3 ms in another. A shared
    // managed heap is shared state.
    //
    // ⚠️ This is a SECOND walker over the same format, which is exactly the duplication the 09-15
    // rewrite removed. It knows only SIZES - values still have exactly one reader - and it is pinned by
    // the EXISTING MLTestBase.GGUFHeaderParseTests rather than by care:
    //   - GGUFHeader_StreamParsersMatchInMemoryOracle: both stream paths vs Parse(byte[]) over a fixture
    //     carrying all 13 value types, comparing DataStartOffset and every value's RUNTIME TYPE. A
    //     FixedSizeOf entry that is wrong by one byte desyncs the scan and fails here.
    //   - GGUFHeader_GrowAndRetry_IsActuallyExercised: proves the fixture really does overrun the first
    //     read, so the scan runs TWICE and the second scan's agreement is what is being checked.
    //   - GGUFHeader_TruncatedStream_Throws: cuts the header mid-vocab. This is the guard for the new
    //     risk introduced by dropping the exact-length slice - the parse now runs over a buffer that is
    //     LARGER than the content, so a scan that over-reported would hand it an uninitialised tail.
    //   - GGUFHeader_NonGGUF_FailsFastWithoutBufferingTheStream: the scan keeps the fail-fast contract,
    //     propagating bad magic/version/value-type instead of treating them as "need more bytes".

    /// <summary>First header read, and the growth step. Covers a 5.7 MiB Qwen3-class header in two reads.</summary>
    private const int HeaderReadChunkBytes = 4 * 1024 * 1024;

    // ── Diagnostics for the header stage ────────────────────────────────────────────────────────────
    // Same shape as GGUFModel.LastHydrateMs and the GraphExecutor.LastRun* counters. A stage that cannot
    // report its own split gets optimised by argument instead of by measurement.

    /// <summary>How many times the cheap scan ran for the last stream header parse (1 = no grow).</summary>
    public static int LastHeaderScanCount { get; private set; }

    /// <summary>Total time in the cheap scans for the last stream header parse.</summary>
    public static double LastHeaderScanMs { get; private set; }

    /// <summary>Time in the single materialising parse for the last stream header parse.</summary>
    public static double LastHeaderParseMs { get; private set; }

    /// <summary>Time spent reading from the stream for the last stream header parse.</summary>
    public static double LastHeaderReadMs { get; private set; }

    /// <summary>Time spent resizing and slicing buffers for the last stream header parse.</summary>
    public static double LastHeaderBufferMs { get; private set; }

    /// <summary>Bytes pulled from the stream for the last stream header parse.</summary>
    public static long LastHeaderBytesRead { get; private set; }

    /// <summary>
    /// How many times a stream header parse has been performed, process-wide, and their total cost.
    /// CUMULATIVE - never reset by a parse.
    /// <para>
    /// 🔴 The Last* counters above describe ONE call, which is exactly why they could not see that a
    /// single model load parses the header TWICE: the second call overwrites the first one's numbers and
    /// the total looks like the cost of one parse. Counting calls is what made the duplicate visible.
    /// </para>
    /// </summary>
    public static int TotalHeaderParses { get; private set; }

    /// <summary>Total time in stream header parses, process-wide. See <see cref="TotalHeaderParses"/>.</summary>
    public static double TotalHeaderMs { get; private set; }

    /// <summary>Zero the cumulative counters, so one load can be measured on its own.</summary>
    public static void ResetHeaderTotals() { TotalHeaderParses = 0; TotalHeaderMs = 0; }

    private static void ResetHeaderStats()
    {
        LastHeaderScanCount = 0;
        LastHeaderScanMs = LastHeaderParseMs = LastHeaderReadMs = LastHeaderBufferMs = 0;
        LastHeaderBytesRead = 0;
    }

    private static double MsSince(long t0) =>
        (System.Diagnostics.Stopwatch.GetTimestamp() - t0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

    /// <summary>
    /// Everything the sync and async entry points do AFTER a read: decide whether the header is all
    /// here, and materialise it if so.
    /// <para>
    /// The two entry points can only differ in the read itself (<c>Read</c> vs <c>await ReadAsync</c>);
    /// sharing the decision keeps them from drifting, which is the failure the 09-15 rewrite was about -
    /// it deleted three copies of the value-reading rules for the same reason.
    /// </para>
    /// </summary>
    private static bool TryCompleteHeader(byte[] buf, int len, out GGUFModel model)
    {
        var tScan = System.Diagnostics.Stopwatch.GetTimestamp();
        bool complete = TryScanHeaderEnd(buf, len, out _);
        LastHeaderScanMs += MsSince(tScan);
        LastHeaderScanCount++;

        if (!complete) { model = null!; return false; }

        // No exact-length copy: the scan already proved the header ends inside `len`, so the
        // materialising parse cannot run off into the buffer's uninitialised tail.
        var tParse = System.Diagnostics.Stopwatch.GetTimestamp();
        model = ParseWholeHeader(buf, len);
        LastHeaderParseMs = MsSince(tParse);

        TotalHeaderParses++;
        TotalHeaderMs += LastHeaderScanMs + LastHeaderParseMs + LastHeaderReadMs + LastHeaderBufferMs;
        return true;
    }

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
        // Reset here too, or a sync parse leaves the previous async parse's numbers standing and the
        // next reader of LastHeader* is looking at a different call than the one they just made.
        ResetHeaderStats();
        var buf = new byte[HeaderReadChunkBytes];
        int len = 0;
        while (true)
        {
            var tRead = System.Diagnostics.Stopwatch.GetTimestamp();
            while (len < buf.Length)
            {
                int n = s.Read(buf, len, buf.Length - len);
                if (n == 0) break;
                len += n;
            }
            LastHeaderReadMs += MsSince(tRead);
            LastHeaderBytesRead = len;

            // Cheap walk decides whether we have the whole header; only then do we materialise it.
            if (TryCompleteHeader(buf, len, out var model)) return model;

            var tGrow = System.Diagnostics.Stopwatch.GetTimestamp();
            GrowOrThrow(ref buf, len);
            LastHeaderBufferMs += MsSince(tGrow);
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
        ResetHeaderStats();
        var buf = new byte[HeaderReadChunkBytes];
        int len = 0;
        while (true)
        {
            // ONE await per multi-MiB read instead of one per field. This is the whole fix.
            var tRead = System.Diagnostics.Stopwatch.GetTimestamp();
            while (len < buf.Length)
            {
                int n = await s.ReadAsync(buf.AsMemory(len, buf.Length - len), ct).ConfigureAwait(false);
                if (n == 0) break;
                len += n;
            }
            LastHeaderReadMs += MsSince(tRead);
            LastHeaderBytesRead = len;

            // Cheap walk decides whether we have the whole header; only then do we materialise it.
            if (TryCompleteHeader(buf, len, out var model)) return model;

            var tGrow = System.Diagnostics.Stopwatch.GetTimestamp();
            GrowOrThrow(ref buf, len);
            LastHeaderBufferMs += MsSince(tGrow);
        }
    }

    // `Exact(buf, len)` used to live here: it sliced the buffer down to the bytes actually held, so that a
    // read past the end threw and became the "need more bytes" signal. The scan carries that signal now -
    // it bounds-checks against an explicit `limit` - so the slice was pure cost and is gone.

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
    /// The materialising parse, run ONCE on a buffer the scan has already proved holds the whole header.
    /// An overrun here is not "need more bytes" - the scan just walked the same structure to the end of
    /// it - so it is a disagreement between the two walkers and must be loud rather than an infinite
    /// grow loop. <c>GGUFHeader_ScanEndMatchesParseEnd</c> is what keeps it from happening.
    /// </summary>
    private static GGUFModel ParseWholeHeader(byte[] data, int limit)
    {
        if (TryParseHeaderFromBuffer(data, out var model)) return model;
        throw new InvalidOperationException(
            "GGUF header scan and parse disagree: the scan found a complete header within "
            + $"{limit:N0} bytes but the parse ran off the end of it. This is a parser defect, "
            + "not a bad file.");
    }

    /// <summary>
    /// Walk the header WITHOUT materialising anything, to find where it ends.
    /// No string is decoded, no dictionary or array is built, and a numeric array is skipped in one
    /// add rather than per element - so this costs a fraction of a real parse and is what the
    /// grow-and-retry loop repeats.
    /// <para>
    /// Same contract as <see cref="TryParseHeaderFromBuffer"/>: false means ONLY "ran off the end, need
    /// more bytes". A bad magic, an unsupported version or an unknown value type is a defect in the data
    /// and propagates immediately, so a non-GGUF stream still fails on its first four bytes instead of
    /// being buffered in full.
    /// </para>
    /// </summary>
    /// <param name="limit">
    /// How many bytes of <paramref name="data"/> actually hold stream content. The buffer is usually
    /// LARGER than that (it is sized in 4 MiB steps), and passing the count instead of slicing the array
    /// down to it is what keeps this allocation-free - MEASURED at 5.7 ms of a 39.9 ms header parse for a
    /// single 5.95 MiB exact-length copy, which is a large-object allocation, a zeroing and a memcpy to
    /// tell the walker something an int already says.
    /// </param>
    private static bool TryScanHeaderEnd(byte[] data, int limit, out int end)
    {
        end = 0;
        try
        {
            int pos = 0;
            uint magic = ReadU32(data, limit, ref pos);
            if (magic != GGUF_MAGIC)
                throw new InvalidOperationException($"Not a GGUF file (magic: 0x{magic:X8}, expected 0x{GGUF_MAGIC:X8})");

            uint version = ReadU32(data, limit, ref pos);
            if (version < 2 || version > 3)
                throw new InvalidOperationException($"Unsupported GGUF version: {version} (expected 2 or 3)");

            ulong tensorCount = ReadU64(data, limit, ref pos);
            ulong metadataCount = ReadU64(data, limit, ref pos);

            for (ulong i = 0; i < metadataCount; i++)
            {
                SkipString(data, limit, ref pos);
                SkipValue(data, limit, ref pos, (GGUFValueType)ReadU32(data, limit, ref pos));
            }

            for (ulong i = 0; i < tensorCount; i++)
            {
                SkipString(data, limit, ref pos);
                uint nDims = ReadU32(data, limit, ref pos);
                Advance(limit, ref pos, (long)nDims * 8);   // dims
                Advance(limit, ref pos, 4);                 // GGMLType
                Advance(limit, ref pos, 8);                 // data offset
            }

            end = pos;
            return true;
        }
        catch (Exception ex) when (ex is IndexOutOfRangeException or ArgumentOutOfRangeException or ArgumentException)
        {
            return false; // ran off the end of what we have: the caller reads more and retries
        }
    }

    /// <summary>
    /// Move the cursor <paramref name="n"/> bytes, refusing to pass <paramref name="limit"/>. Skipping is
    /// not reading, so nothing else would notice an overrun - without this bounds check the scan would
    /// sail past the end on a bad length field and report a header that is not there.
    /// </summary>
    private static void Advance(int limit, ref int pos, long n)
    {
        if (n < 0 || pos + n > limit) throw new IndexOutOfRangeException();
        pos += (int)n;
    }

    /// <summary>
    /// Fixed-width reads for the SCAN path. They bounds-check against <paramref name="limit"/> rather
    /// than the array, because the array is deliberately bigger than the content and its own bounds
    /// check would happily read uninitialised tail bytes as header data.
    /// </summary>
    private static uint ReadU32(byte[] data, int limit, ref int pos)
    {
        if (pos + 4 > limit) throw new IndexOutOfRangeException();
        return ReadUInt32(data, ref pos);
    }

    private static ulong ReadU64(byte[] data, int limit, ref int pos)
    {
        if (pos + 8 > limit) throw new IndexOutOfRangeException();
        return ReadUInt64(data, ref pos);
    }

    private static void SkipString(byte[] data, int limit, ref int pos)
    {
        ulong len = ReadU64(data, limit, ref pos);
        // A corrupt length can be anything up to 2^64-1. Compare in ulong BEFORE narrowing, or the cast
        // to long goes negative (or wraps small) and the cursor leaves the buffer unnoticed.
        if (len > (ulong)(limit - pos)) throw new IndexOutOfRangeException();
        pos += (int)len;
    }

    /// <summary>
    /// Byte width of a fixed-size GGUF scalar, or -1 for the variable-length types (String, Array).
    /// ⚠️ This is the scanner's ONLY piece of format knowledge that <see cref="ReadValue"/> also holds,
    /// so it is the one thing that can drift. An entry that is wrong by a single byte desyncs the scan
    /// and fails <c>GGUFHeader_StreamParsersMatchInMemoryOracle</c>, whose fixture carries all 13 types.
    /// A NEW value type added to <see cref="ReadValue"/> and not here throws <see cref="NotSupportedException"/>
    /// on the first scan rather than silently mis-sizing anything.
    /// </summary>
    private static int FixedSizeOf(GGUFValueType type) => type switch
    {
        GGUFValueType.UInt8 or GGUFValueType.Int8 or GGUFValueType.Bool => 1,
        GGUFValueType.UInt16 or GGUFValueType.Int16 => 2,
        GGUFValueType.UInt32 or GGUFValueType.Int32 or GGUFValueType.Float32 => 4,
        GGUFValueType.UInt64 or GGUFValueType.Int64 or GGUFValueType.Float64 => 8,
        GGUFValueType.String or GGUFValueType.Array => -1,
        _ => throw new NotSupportedException($"Unknown GGUF value type: {type}")
    };

    private static void SkipValue(byte[] data, int limit, ref int pos, GGUFValueType type)
    {
        int fixedSize = FixedSizeOf(type);
        if (fixedSize > 0) { Advance(limit, ref pos, fixedSize); return; }
        if (type == GGUFValueType.String) { SkipString(data, limit, ref pos); return; }

        // Array
        var elemType = (GGUFValueType)ReadU32(data, limit, ref pos);
        ulong count = ReadU64(data, limit, ref pos);
        int elemSize = FixedSizeOf(elemType);
        if (elemSize > 0)
        {
            // The whole point: a 151,936-element numeric array costs one add, not 151,936 iterations.
            // Divide rather than multiply so a hostile count cannot overflow into a small positive
            // number and let the cursor walk off the end while still reporting a complete header.
            if (count > (ulong)(limit - pos) / (ulong)elemSize) throw new IndexOutOfRangeException();
            pos += (int)(count * (ulong)elemSize);
            return;
        }
        for (ulong i = 0; i < count; i++) SkipValue(data, limit, ref pos, elemType);
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
