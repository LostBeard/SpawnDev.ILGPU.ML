using System.Text;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Gates for the GGUF header stream parsers after the 2026-09-15 rewrite that made them read the header
/// region ONCE and parse it in memory (see the block comment in GGUFParser.cs).
///
/// 🔴 WHY THESE EXIST. The rewrite deleted two hand-written per-field readers and pointed both stream
/// entry points at the SAME byte[] cursor that <see cref="GGUFParser.Parse(byte[])"/> uses. That removes
/// a whole class of drift, but it buys the removal with a read-and-retry loop that did not exist before,
/// and that loop has exactly two ways to be wrong:
///   1. the buffer grows but the retry parses the prefix differently -> silently wrong metadata;
///   2. a stream that never parses is retried until it is entirely in memory -> a non-GGUF URL pulls
///      gigabytes into the WASM heap before admitting the first four bytes were already wrong.
/// Both are gated below, and <see cref="GGUFHeader_NonGGUF_FailsFastWithoutBufferingTheStream"/> counts
/// the bytes actually pulled, because "it threw" would pass even if it read the entire file first.
///
/// The in-memory <see cref="GGUFParser.Parse(byte[])"/> is the oracle: it is the untouched path, and it
/// is what the boxed metadata types have always been (UInt8 -> byte, Int8 -> sbyte, Int64 -> long ...),
/// which callers pattern-match on with `is long a`. Comparing VALUE AND RUNTIME TYPE is the point - an
/// equality-only check passes while a long quietly becomes a ulong and every `is long` in the codebase
/// starts missing.
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// Target size when BUILDING the fixture - it must exceed GGUFParser.HeaderReadChunkBytes so the
    /// header forces at least one grow-and-retry.
    ///
    /// ⚠️ This is only a build target and is deliberately NOT what the grow test asserts against. A test
    /// that checked its own copy of a private constant would keep passing after someone raised the real
    /// one, while the retry path it claims to cover quietly stopped running - the exact "a test that
    /// cannot fail" shape. <see cref="GGUFHeader_GrowAndRetry_IsActuallyExercised"/> therefore OBSERVES
    /// the parser's real first read off the stream instead.
    /// </summary>
    private const int GrowPathHeaderMinBytes = 4 * 1024 * 1024;

    /// <summary>
    /// A GGUF with a header bigger than the first read, every metadata value type represented, and a
    /// tokenizer-shaped string array. Returns the full file bytes.
    /// </summary>
    private static byte[] BuildBigHeaderGguf(out int vocabCount)
    {
        vocabCount = 20_000;
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        void Str(string v) { var b = Encoding.UTF8.GetBytes(v); bw.Write((ulong)b.Length); bw.Write(b); }

        bw.Write((byte)'G'); bw.Write((byte)'G'); bw.Write((byte)'U'); bw.Write((byte)'F');
        bw.Write((uint)3);              // version
        bw.Write((ulong)2);             // tensor count
        bw.Write((ulong)14);            // metadata count

        // One of every scalar type, so the shared ReadValue boxing is compared end to end.
        Str("t.u8");   bw.Write((uint)GGUFValueType.UInt8);   bw.Write((byte)200);
        Str("t.i8");   bw.Write((uint)GGUFValueType.Int8);    bw.Write((sbyte)-99);
        Str("t.u16");  bw.Write((uint)GGUFValueType.UInt16);  bw.Write((ushort)65000);
        Str("t.i16");  bw.Write((uint)GGUFValueType.Int16);   bw.Write((short)-30000);
        Str("t.u32");  bw.Write((uint)GGUFValueType.UInt32);  bw.Write((uint)4000000000);
        Str("t.i32");  bw.Write((uint)GGUFValueType.Int32);   bw.Write(-2000000000);
        Str("t.u64");  bw.Write((uint)GGUFValueType.UInt64);  bw.Write((ulong)18000000000000000000);
        Str("t.i64");  bw.Write((uint)GGUFValueType.Int64);   bw.Write(-9000000000000000000L);
        Str("t.f32");  bw.Write((uint)GGUFValueType.Float32); bw.Write(3.5f);
        Str("t.f64");  bw.Write((uint)GGUFValueType.Float64); bw.Write(-1.25);
        Str("t.bool"); bw.Write((uint)GGUFValueType.Bool);    bw.Write((byte)1);
        Str("t.str");  bw.Write((uint)GGUFValueType.String);  Str("hello gguf");

        // A numeric array (object[] path).
        Str("t.arr.i32"); bw.Write((uint)GGUFValueType.Array);
        bw.Write((uint)GGUFValueType.Int32); bw.Write((ulong)4);
        bw.Write(10); bw.Write(-20); bw.Write(30); bw.Write(-40);

        // The tokenizer-shaped string array. Padded so the HEADER alone clears the first read and the
        // parser is forced to grow: this is the only reason the file is this big.
        var filler = new string('x', 220);
        Str("tokenizer.ggml.tokens"); bw.Write((uint)GGUFValueType.Array);
        bw.Write((uint)GGUFValueType.String); bw.Write((ulong)vocabCount);
        for (int i = 0; i < vocabCount; i++) Str($"tok{i}_{filler}");

        // Two tensors, one F32 one Q8_0, with distinct dims so a mixed-up tensor loop is visible.
        Str("norm.weight"); bw.Write((uint)1); bw.Write((ulong)8);  bw.Write((uint)0); bw.Write((ulong)0);
        Str("w.q");         bw.Write((uint)2); bw.Write((ulong)64); bw.Write((ulong)32);
        bw.Write((uint)8);  bw.Write((ulong)64);

        while (ms.Position % 32 != 0) bw.Write((byte)0);
        for (int i = 0; i < 4096; i++) bw.Write((byte)(i & 0xFF));   // stand-in tensor data
        bw.Flush();
        return ms.ToArray();
    }

    /// <summary>Forward-only, non-seekable, byte-counting wrapper. Proves no seeking and measures reads.</summary>
    private sealed class ForwardOnlyCountingStream : Stream
    {
        private readonly byte[] _data;
        private int _pos;
        public long BytesRead { get; private set; }
        /// <summary>
        /// Size of the FIRST Read the parser asked for = its real first-chunk constant, observed rather
        /// than copied. This is what makes the grow test unable to pass vacuously.
        /// </summary>
        public int FirstReadRequest { get; private set; } = -1;
        public ForwardOnlyCountingStream(byte[] data) => _data = data;
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count)
        {
            if (FirstReadRequest < 0) FirstReadRequest = count;
            int n = Math.Min(count, _data.Length - _pos);
            Array.Copy(_data, _pos, buffer, offset, n);
            _pos += n; BytesRead += n;
            return n;
        }
        public override long Seek(long o, SeekOrigin r) => throw new NotSupportedException("seek attempted");
        public override void SetLength(long v) => throw new NotSupportedException();
        public override void Write(byte[] b, int o, int c) => throw new NotSupportedException();
    }

    private static void AssertSameHeader(string label, GGUFModel expected, GGUFModel actual)
    {
        if (actual.Version != expected.Version)
            throw new Exception($"{label}: Version {actual.Version} != {expected.Version}");
        if (actual.Alignment != expected.Alignment)
            throw new Exception($"{label}: Alignment {actual.Alignment} != {expected.Alignment}");
        if (actual.DataStartOffset != expected.DataStartOffset)
            throw new Exception($"{label}: DataStartOffset {actual.DataStartOffset} != {expected.DataStartOffset}");
        if (actual.Metadata.Count != expected.Metadata.Count)
            throw new Exception($"{label}: metadata count {actual.Metadata.Count} != {expected.Metadata.Count}");

        foreach (var kv in expected.Metadata)
        {
            if (!actual.Metadata.TryGetValue(kv.Key, out var got))
                throw new Exception($"{label}: missing metadata key '{kv.Key}'");
            // RUNTIME TYPE first - callers pattern-match with `is long a`, so a long that became a ulong
            // is a real break that an equality-only comparison would wave through.
            if (got.GetType() != kv.Value.GetType())
                throw new Exception($"{label}: '{kv.Key}' boxed as {got.GetType().Name}, expected {kv.Value.GetType().Name}");
            if (kv.Value is string[] se)
            {
                var ge = (string[])got;
                if (ge.Length != se.Length)
                    throw new Exception($"{label}: '{kv.Key}' length {ge.Length} != {se.Length}");
                for (int i = 0; i < se.Length; i++)
                    if (ge[i] != se[i]) throw new Exception($"{label}: '{kv.Key}'[{i}] '{ge[i]}' != '{se[i]}'");
            }
            else if (kv.Value is object[] oe)
            {
                var ge = (object[])got;
                if (ge.Length != oe.Length)
                    throw new Exception($"{label}: '{kv.Key}' length {ge.Length} != {oe.Length}");
                for (int i = 0; i < oe.Length; i++)
                {
                    if (ge[i].GetType() != oe[i].GetType())
                        throw new Exception($"{label}: '{kv.Key}'[{i}] boxed as {ge[i].GetType().Name}, expected {oe[i].GetType().Name}");
                    if (!ge[i].Equals(oe[i])) throw new Exception($"{label}: '{kv.Key}'[{i}] {ge[i]} != {oe[i]}");
                }
            }
            else if (!got.Equals(kv.Value))
                throw new Exception($"{label}: '{kv.Key}' {got} != {kv.Value}");
        }

        if (actual.Tensors.Length != expected.Tensors.Length)
            throw new Exception($"{label}: tensor count {actual.Tensors.Length} != {expected.Tensors.Length}");
        for (int i = 0; i < expected.Tensors.Length; i++)
        {
            var e = expected.Tensors[i]; var g = actual.Tensors[i];
            if (g.Name != e.Name) throw new Exception($"{label}: tensor[{i}] name '{g.Name}' != '{e.Name}'");
            if (g.Type != e.Type) throw new Exception($"{label}: tensor[{i}] type {g.Type} != {e.Type}");
            if (g.DataOffset != e.DataOffset) throw new Exception($"{label}: tensor[{i}] offset {g.DataOffset} != {e.DataOffset}");
            if (g.Dimensions.Length != e.Dimensions.Length)
                throw new Exception($"{label}: tensor[{i}] rank {g.Dimensions.Length} != {e.Dimensions.Length}");
            for (int d = 0; d < e.Dimensions.Length; d++)
                if (g.Dimensions[d] != e.Dimensions[d])
                    throw new Exception($"{label}: tensor[{i}] dim[{d}] {g.Dimensions[d]} != {e.Dimensions[d]}");
        }
    }

    [TestMethod]
    public async Task GGUFHeader_StreamParsersMatchInMemoryOracle() => await RunTest(async accelerator =>
    {
        var bytes = BuildBigHeaderGguf(out var vocabCount);
        var oracle = GGUFParser.Parse(bytes);

        // Forward-only + non-seekable on BOTH: the HTTP and WebTorrent sources cannot seek, and a Seek
        // in the parser would throw from the wrapper rather than quietly working on a MemoryStream.
        var syncSrc = new ForwardOnlyCountingStream(bytes);
        AssertSameHeader("ParseHeader(sync)", oracle, GGUFParser.ParseHeader(syncSrc));

        var asyncSrc = new ForwardOnlyCountingStream(bytes);
        AssertSameHeader("ParseHeaderAsync", oracle, await GGUFParser.ParseHeaderAsync(asyncSrc));

        Console.WriteLine($"[GGUFHeader] oracle-match OK: {oracle.Metadata.Count} keys (vocab {vocabCount:N0}), "
            + $"{oracle.Tensors.Length} tensors, dataStart {oracle.DataStartOffset:N0}");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task GGUFHeader_GrowAndRetry_IsActuallyExercised() => await RunTest(async accelerator =>
    {
        // 🔴 Without this the equivalence test above could pass entirely on the first-read path and the
        // grow-and-retry loop - the only genuinely new code in the rewrite - would never run once.
        var bytes = BuildBigHeaderGguf(out _);
        var model = GGUFParser.Parse(bytes);

        var src = new ForwardOnlyCountingStream(bytes);
        var got = await GGUFParser.ParseHeaderAsync(src);
        AssertSameHeader("ParseHeaderAsync(grown)", model, got);

        // The parser's own first-chunk size, as it asked the stream for it - so raising
        // HeaderReadChunkBytes past the fixture FAILS here instead of silently skipping the retry.
        if (src.FirstReadRequest <= 0)
            throw new Exception("The parser never read from the stream.");
        if (model.DataStartOffset <= src.FirstReadRequest)
            throw new Exception(
                $"Fixture header is {model.DataStartOffset:N0} B but the parser's first read asked for "
                + $"{src.FirstReadRequest:N0} B, so it was satisfied in one go and the grow-and-retry path "
                + "never ran. Enlarge BuildBigHeaderGguf (GrowPathHeaderMinBytes) past the new chunk size.");

        // Growth is geometric with buffer reuse, so the stream should give up roughly the final capacity,
        // never the whole file over and over. Catches a retry that re-reads from the start.
        if (src.BytesRead > model.DataStartOffset * 3)
            throw new Exception($"Grow path read {src.BytesRead:N0} B for a {model.DataStartOffset:N0} B "
                + "header - the retry is re-reading the stream instead of reusing the buffer.");
        Console.WriteLine($"[GGUFHeader] grow path OK: header {model.DataStartOffset:N0} B > first read "
            + $"{src.FirstReadRequest:N0} B (observed), pulled {src.BytesRead:N0} B");
    });

    [TestMethod]
    public async Task GGUFHeader_NonGGUF_FailsFastWithoutBufferingTheStream() => await RunTest(async accelerator =>
    {
        // 🔴 THE RUNAWAY GUARD. "It threw" is not enough: the failure mode that matters is throwing only
        // AFTER doubling the buffer all the way to EOF, which on a real model URL means pulling a
        // multi-GB blob into the WASM heap to discover the first four bytes were wrong.
        var junk = new byte[48 * 1024 * 1024];
        new Random(1234).NextBytes(junk);
        junk[0] = (byte)'N'; junk[1] = (byte)'O'; junk[2] = (byte)'P'; junk[3] = (byte)'E';

        var src = new ForwardOnlyCountingStream(junk);
        var threw = false;
        try { await GGUFParser.ParseHeaderAsync(src); }
        catch (InvalidOperationException) { threw = true; }
        if (!threw) throw new Exception("A non-GGUF stream parsed as a GGUF header.");
        if (src.BytesRead > 8 * 1024 * 1024)
            throw new Exception($"Non-GGUF stream was read {src.BytesRead:N0} B before failing - the magic "
                + "check is not short-circuiting the grow loop.");
        Console.WriteLine($"[GGUFHeader] non-GGUF fail-fast OK: threw after {src.BytesRead:N0} B of {junk.Length:N0}");
    });

    [TestMethod]
    public async Task GGUFHeader_TruncatedStream_Throws() => await RunTest(async accelerator =>
    {
        // A header cut off mid-vocab must report truncation, not loop forever and not return a model
        // holding whatever happened to be in the uninitialised tail of the buffer.
        var bytes = BuildBigHeaderGguf(out _);
        var cut = bytes[..(bytes.Length / 2)];
        var threw = false;
        try { await GGUFParser.ParseHeaderAsync(new ForwardOnlyCountingStream(cut)); }
        catch (EndOfStreamException) { threw = true; }
        if (!threw) throw new Exception("A truncated GGUF header did not throw EndOfStreamException.");
        Console.WriteLine("[GGUFHeader] truncated-stream OK");
        await Task.CompletedTask;
    });
}
