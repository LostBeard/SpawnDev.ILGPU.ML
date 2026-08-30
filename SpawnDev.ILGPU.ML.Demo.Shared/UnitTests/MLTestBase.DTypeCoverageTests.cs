using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Onnx;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Every ONNX dtype this library claims to load, actually loaded and verified.
/// </summary>
/// <remarks>
/// A library's supported-dtype list is a promise to callers, and we control only what it supports and what
/// we test - never how a dev will use it. Before these tests, FLOAT32 and FLOAT16 were exercised heavily,
/// BFLOAT16 through the low-p weight path, and UINT8 / INT8 / INT32 / INT64 / BOOL / DOUBLE not at all, even
/// though <c>OnnxTensorProto.ToFloatArray</c> decodes all nine. Edge dtypes are exactly where a silent
/// mis-decode hides: an INT64 read as INT32, or a BFLOAT16 assembled from the wrong half of the word, still
/// produces plausible-looking numbers.
/// <para>
/// Each case is checked EXACTLY. The values are chosen to be representable without loss in their format
/// (bf16 values have zero low mantissa bits, fp16 values are small dyadic rationals), so any tolerance would
/// only hide an encoding bug rather than accommodate one.
/// </para>
/// </remarks>
public abstract partial class MLTestBase
{
    private static byte[] EncodeRaw(int dtype, float[] values)
    {
        switch (dtype)
        {
            case 1: // FLOAT
                var f = new byte[values.Length * 4];
                Buffer.BlockCopy(values, 0, f, 0, f.Length);
                return f;
            case 2: // UINT8
                return values.Select(v => (byte)v).ToArray();
            case 3: // INT8
                return values.Select(v => unchecked((byte)(sbyte)v)).ToArray();
            case 6: // INT32
                return values.SelectMany(v => BitConverter.GetBytes((int)v)).ToArray();
            case 7: // INT64
                return values.SelectMany(v => BitConverter.GetBytes((long)v)).ToArray();
            case 9: // BOOL
                return values.Select(v => (byte)(v != 0f ? 1 : 0)).ToArray();
            case 10: // FLOAT16
                return values.SelectMany(v => BitConverter.GetBytes((System.Half)v)).ToArray();
            case 11: // DOUBLE
                return values.SelectMany(v => BitConverter.GetBytes((double)v)).ToArray();
            case 16: // BFLOAT16 — the TOP 16 bits of the fp32 word (ToFloatArray reassembles bf << 16)
                return values.SelectMany(v => BitConverter.GetBytes((ushort)(BitConverter.SingleToInt32Bits(v) >> 16))).ToArray();
            default:
                throw new NotSupportedException($"test encoder has no case for dtype {dtype}");
        }
    }

    /// <summary>
    /// Load a small tensor of EVERY claimed dtype through the proto path and verify the GPU floats exactly.
    /// </summary>
    [TestMethod]
    public Task DType_AllClaimedProtoDtypes_LoadExactly() => RunTest(async accelerator =>
    {
        // (dtype, name, values) — each value exactly representable in that dtype AND in fp32.
        var cases = new (int DType, string Name, float[] Values)[]
        {
            (1,  "FLOAT",     new[] {  1.5f, -2.25f, 0f,  3.0f }),
            (2,  "UINT8",     new[] {  0f,    1f,    200f, 255f }),
            (3,  "INT8",      new[] { -128f, -1f,    0f,  127f }),
            (6,  "INT32",     new[] { -70000f, -1f,  0f,  65536f }),
            (7,  "INT64",     new[] { -70000f, -1f,  0f,  65536f }),
            (9,  "BOOL",      new[] {  0f,    1f,    1f,   0f }),
            (10, "FLOAT16",   new[] {  1.5f, -2.25f, 0f,  3.0f }),
            (11, "DOUBLE",    new[] {  1.5f, -2.25f, 0f,  3.0f }),
            (16, "BFLOAT16",  new[] {  1.5f, -2.0f,  0f,  3.0f }),  // low 16 mantissa bits zero
        };

        using var pool = new BufferPool(accelerator);
        var failures = new List<string>();

        foreach (var (dtype, dtypeName, values) in cases)
        {
            var proto = new OnnxTensorProto
            {
                Name = $"w_{dtypeName}",
                DataType = dtype,
                Dims = new long[] { values.Length },
                RawData = EncodeRaw(dtype, values),
            };

            Tensor t;
            try { t = pool.AllocatePermanentChunked(proto, new[] { values.Length }, proto.Name); }
            catch (Exception ex) { failures.Add($"{dtypeName} (dtype {dtype}) THREW: {ex.GetType().Name}: {ex.Message}"); continue; }

            await accelerator.SynchronizeAsync();
            var got = await t.Data.SubView(0, values.Length).CopyToHostAsync();
            for (int i = 0; i < values.Length; i++)
            {
                if (MathF.Abs(got[i] - values[i]) > 1e-6f)
                {
                    failures.Add($"{dtypeName} (dtype {dtype}) element {i}: got {got[i]}, expected {values[i]}");
                    break;
                }
            }
        }

        if (failures.Count > 0)
            throw new Exception($"{failures.Count} of {cases.Length} claimed dtypes did not load correctly:\n  " +
                                string.Join("\n  ", failures));

        Console.WriteLine($"[DType] all {cases.Length} claimed proto dtypes load exactly (1,2,3,6,7,9,10,11,16)");
    });

    /// <summary>
    /// A LARGE non-FLOAT tensor, to exercise the chunked upload rather than the small-tensor fast path.
    /// </summary>
    /// <remarks>
    /// <c>AllocatePermanentChunked</c> takes a whole-array shortcut below 262,144 elements, so the case
    /// matrix above never reaches the chunking loop. The non-float branch used to skip chunking entirely -
    /// one whole-tensor <c>CopyFromCPU</c> regardless of size - so this asserts the boundary case the matrix
    /// cannot: values must stay correct ACROSS chunk boundaries, not merely be decoded correctly.
    /// </remarks>
    [TestMethod]
    public Task DType_LargeNonFloatTensor_ChunksCorrectly() => RunTest(async accelerator =>
    {
        const int Count = 300_000;              // > the 262,144 fast-path threshold
        var values = new float[Count];
        for (int i = 0; i < Count; i++) values[i] = (i % 251) - 125;   // spans negatives, crosses chunk edges

        var proto = new OnnxTensorProto
        {
            Name = "big_int32",
            DataType = 6,
            Dims = new long[] { Count },
            RawData = EncodeRaw(6, values),
        };

        using var pool = new BufferPool(accelerator);
        var t = pool.AllocatePermanentChunked(proto, new[] { Count }, proto.Name);
        await accelerator.SynchronizeAsync();
        var got = await t.Data.SubView(0, Count).CopyToHostAsync();

        // Check the chunk boundaries specifically, then everything.
        int[] edges = { 0, 262143, 262144, 262145, Count - 1 };
        foreach (var e in edges)
            if (MathF.Abs(got[e] - values[e]) > 1e-6f)
                throw new Exception($"chunk-boundary element {e}: got {got[e]}, expected {values[e]}");

        for (int i = 0; i < Count; i++)
            if (MathF.Abs(got[i] - values[i]) > 1e-6f)
                throw new Exception($"element {i}: got {got[i]}, expected {values[i]}");

        Console.WriteLine($"[DType] {Count:N0}-element INT32 tensor uploaded correctly across chunk boundaries");
    });

    /// <summary>
    /// Every dtype the STREAMING loader claims, streamed through it and verified exactly.
    /// </summary>
    /// <remarks>
    /// <c>AllocatePermanentFromStreamAsync</c> advertises FLOAT32 (1), FLOAT16 (10), BFLOAT16 (16),
    /// FLOAT8E4M3 (17) and FLOAT8E5M2 (19). The last three were added to that switch without being exercised
    /// through it - the native low-p loader covers them, but that is a DIFFERENT method, and a dtype listed in
    /// a switch nobody ran is a claim rather than a capability. This closes the matrix so the advertised list
    /// and the tested list are the same list.
    /// <para>
    /// FP8 values are chosen from the exactly-representable grid of each format (E4M3: 3 mantissa bits;
    /// E5M2: 2), so these stay exact comparisons rather than tolerance checks.
    /// </para>
    /// </remarks>
    [TestMethod]
    public Task DType_AllClaimedStreamingDtypes_LoadExactly() => RunTest(async accelerator =>
    {
        var cases = new (int DType, string Name, float[] Values, Func<float[], byte[]> Encode)[]
        {
            (1,  "FLOAT",    new[] { 1.5f, -2.25f, 0f, 3.0f },  v => EncodeRaw(1, v)),
            (10, "FLOAT16",  new[] { 1.5f, -2.25f, 0f, 3.0f },  v => EncodeRaw(10, v)),
            (16, "BFLOAT16", new[] { 1.5f, -2.0f,  0f, 3.0f },  v => EncodeRaw(16, v)),
            (17, "FP8E4M3",  new[] { 1.5f, -2.0f,  0f, 3.0f },
                 v => v.Select(x => ((global::ILGPU.Float8E4M3)x).RawValue).ToArray()),
            (19, "FP8E5M2",  new[] { 1.5f, -2.0f,  0f, 3.0f },
                 v => v.Select(x => ((global::ILGPU.Float8E5M2)x).RawValue).ToArray()),
        };

        using var pool = new BufferPool(accelerator);
        var failures = new List<string>();

        foreach (var (dtype, dtypeName, values, encode) in cases)
        {
            var raw = encode(values);
            using var ms = new MemoryStream(raw);
            Tensor t;
            try
            {
                t = await pool.AllocatePermanentFromStreamAsync(ms, 0, raw.Length, dtype, new[] { values.Length }, $"s_{dtypeName}");
            }
            catch (Exception ex) { failures.Add($"{dtypeName} (dtype {dtype}) THREW: {ex.GetType().Name}: {ex.Message}"); continue; }

            await pool.FlushPendingFp16ConvertsAsync();
            await accelerator.SynchronizeAsync();
            var got = await t.Data.SubView(0, values.Length).CopyToHostAsync();
            for (int i = 0; i < values.Length; i++)
                if (MathF.Abs(got[i] - values[i]) > 1e-6f)
                {
                    failures.Add($"{dtypeName} (dtype {dtype}) element {i}: got {got[i]}, expected {values[i]}");
                    break;
                }
        }
        if (failures.Count > 0)
            throw new Exception($"{failures.Count} of {cases.Length} claimed STREAMING dtypes failed: " +
                                string.Join(" | ", failures));

        Console.WriteLine($"[DType] all {cases.Length} claimed streaming dtypes load exactly (1,10,16,17,19)");
    });
}
