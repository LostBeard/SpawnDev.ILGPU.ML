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
            // ⚠️ INT8 carries NEGATIVES and the extremes on purpose. Reading an int8 -1 as a byte yields
            // 255, and a quantised weight off by 256 is not subtly wrong - it is noise. -128/127 pin the
            // ends of the range where a sign error shows up largest.
            (3,  "INT8",     new[] { -1f, 127f, -128f, 0f },    v => EncodeRaw(3, v)),
            // ⚠️ UINT8 (2) is deliberately ABSENT. It was claimed briefly and this very test caught it
            // returning 255 as -1 - the convert lowers as signed, wrapping the top half of the range. It
            // materialises instead, so nothing is wrong today; when the lowering is fixed in ILGPU, add
            // (2, "UINT8", new[] { 0f, 255f, 128f, 1f }, v => EncodeRaw(2, v)) here FIRST and let it fail.
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

        Console.WriteLine($"[DType] all {cases.Length} claimed streaming dtypes load exactly "
                        + "(1,10,16,17,19,3)");
    });

    /// <summary>
    /// The FNUZ FP8 formats - FLOAT8E4M3FNUZ (18) and FLOAT8E5M2FNUZ (20) - through both load paths.
    /// </summary>
    /// <remarks>
    /// These were previously unsupported EVERYWHERE, and deliberately so: they are not the OCP formats
    /// ILGPU carries. The exponent bias is one higher (8 vs 7, 16 vs 15), there are no infinities, and 0x80
    /// is the only NaN rather than negative zero. Aliasing them onto Float8E4M3/Float8E5M2 would have
    /// mis-decoded every weight by a factor of two while still producing plausible numbers.
    /// <para>
    /// ⚠️ The expected values are derived BY HAND from the format definition, not produced by our own
    /// decoder - a round-trip through the code under test proves only self-consistency. The
    /// <c>0x38</c> case is the load-bearing one: it is 1.0 in OCP E4M3 and 0.5 in E4M3FNUZ, so aliasing
    /// would fail exactly there.
    /// </para>
    /// </remarks>
    [TestMethod]
    public Task DType_Fp8FnuzVariants_DecodeToSpec() => RunTest(async accelerator =>
    {
        // (raw byte, expected value) — hand-derived from the FNUZ definitions.
        // E4M3FNUZ: 1 sign, 4 exp, 3 mant, bias 8.
        var e4m3 = new (byte Raw, float Expect)[]
        {
            (0x00, 0f),        // zero
            (0x38, 0.5f),      // exp 7, mant 0 -> 2^(7-8)          ⚠️ 1.0 in OCP E4M3
            (0x40, 1.0f),      // exp 8, mant 0 -> 2^0
            (0x48, 2.0f),      // exp 9, mant 0 -> 2^1
            (0x41, 1.125f),    // exp 8, mant 1 -> (1+1/8)*2^0
            (0xC0, -1.0f),     // sign 1, exp 8 -> -1
        };
        // E5M2FNUZ: 1 sign, 5 exp, 2 mant, bias 16.
        var e5m2 = new (byte Raw, float Expect)[]
        {
            (0x00, 0f),
            (0x3C, 0.5f),      // exp 15 -> 2^-1
            (0x40, 1.0f),      // exp 16 -> 2^0
            (0x44, 2.0f),      // exp 17 -> 2^1
            (0x41, 1.25f),     // exp 16, mant 1 -> (1+1/4)
            (0xC0, -1.0f),
        };

        using var pool = new BufferPool(accelerator);
        var failures = new List<string>();

        foreach (var (dtype, label, cases) in new (int, string, (byte Raw, float Expect)[])[]
                 { (18, "E4M3FNUZ", e4m3), (20, "E5M2FNUZ", e5m2) })
        {
            var raw = cases.Select(c => c.Raw).ToArray();
            var want = cases.Select(c => c.Expect).ToArray();

            // Path 1: the proto / host path (ToFloatArray).
            var proto = new OnnxTensorProto
            {
                Name = $"fnuz_{label}", DataType = dtype,
                Dims = new long[] { raw.Length }, RawData = raw,
            };
            var t1 = pool.AllocatePermanentChunked(proto, new[] { raw.Length }, proto.Name);
            await accelerator.SynchronizeAsync();
            var got1 = await t1.Data.SubView(0, raw.Length).CopyToHostAsync();

            // Path 2: the streaming loader.
            using var ms = new MemoryStream(raw);
            var t2 = await pool.AllocatePermanentFromStreamAsync(ms, 0, raw.Length, dtype, new[] { raw.Length }, proto.Name);
            await pool.FlushPendingFp16ConvertsAsync();
            await accelerator.SynchronizeAsync();
            var got2 = await t2.Data.SubView(0, raw.Length).CopyToHostAsync();

            for (int i = 0; i < want.Length; i++)
            {
                if (MathF.Abs(got1[i] - want[i]) > 1e-6f)
                    failures.Add($"{label} proto 0x{cases[i].Raw:X2}: got {got1[i]}, expected {want[i]}");
                if (MathF.Abs(got2[i] - want[i]) > 1e-6f)
                    failures.Add($"{label} stream 0x{cases[i].Raw:X2}: got {got2[i]}, expected {want[i]}");
            }
        }

        if (failures.Count > 0)
            throw new Exception($"{failures.Count} FNUZ decode mismatch(es): " + string.Join(" | ", failures));

        Console.WriteLine("[DType] FNUZ FP8 (18, 20) decode to spec on both the proto and streaming paths");
    });

    /// <summary>
    /// MaskKernels: GPU-generated triangular masks match a CPU reference, and the detector accepts ONLY
    /// genuinely triangular data.
    /// </summary>
    /// <remarks>
    /// This exists because distilgpt2's causal mask (onnx::Slice_260) is a 1024x1024 BOOL tensor that gets
    /// expanded to fp32 (1 MiB -&gt; 4 MiB) and host-copied, tripping the browser host-copy guard - and a
    /// triangular mask is cheaper to GENERATE than to ship over the bus.
    /// <para>
    /// ⚠️ The detector half matters more than the generator half. Substituting a generated mask for data
    /// that is only MOSTLY triangular would silently change what the model computes - far worse than the
    /// copy it avoids - so the negative cases here (one flipped bit, and a non-square shape) are the point.
    /// </para>
    /// </remarks>
    [TestMethod]
    public Task Mask_GeneratedTriangular_MatchesCpuAndDetectsExactly() => RunTest(async accelerator =>
    {
        const int N = 64;
        var mk = new SpawnDev.ILGPU.ML.Kernels.MaskKernels(accelerator);

        foreach (var mode in new[]
                 {
                     SpawnDev.ILGPU.ML.Kernels.MaskKernels.TriangleMode.LowerInclusive,
                     SpawnDev.ILGPU.ML.Kernels.MaskKernels.TriangleMode.LowerExclusive,
                     SpawnDev.ILGPU.ML.Kernels.MaskKernels.TriangleMode.UpperInclusive,
                     SpawnDev.ILGPU.ML.Kernels.MaskKernels.TriangleMode.UpperExclusive,
                 })
        {
            using var buf = accelerator.Allocate1D<float>(N * N);
            mk.FillTriangular(buf.View, N, N, mode);
            await accelerator.SynchronizeAsync();
            var got = await buf.View.SubView(0, N * N).CopyToHostAsync();

            int m = (int)mode;
            var raw = new byte[N * N];
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                {
                    bool admit = m == 0 ? c <= r : m == 1 ? c < r : m == 2 ? c >= r : c > r;
                    float want = admit ? 1f : 0f;
                    if (got[r * N + c] != want)
                        throw new Exception($"{mode} at ({r},{c}): got {got[r * N + c]}, expected {want}");
                    raw[r * N + c] = (byte)(admit ? 1 : 0);
                }

            // Round-trip: the detector must recognise the very pattern the kernel produces, as that mode.
            if (!SpawnDev.ILGPU.ML.Kernels.MaskKernels.TryDetectTriangular(raw, N, N, out var found))
                throw new Exception($"{mode}: detector rejected a mask the kernel generated");
            if (found != mode)
                throw new Exception($"{mode}: detector reported {found}");

            // NEGATIVE: one flipped bit must disqualify it. This is the assertion that stops a
            // nearly-triangular tensor being replaced by a generated one.
            raw[(N / 2) * N + (N / 4)] ^= 1;
            if (SpawnDev.ILGPU.ML.Kernels.MaskKernels.TryDetectTriangular(raw, N, N, out _))
                throw new Exception($"{mode}: detector ACCEPTED data with one flipped element - it would " +
                                    "substitute a generated mask and silently change the model");
        }

        // Degenerate shapes must be refused rather than guessed at.
        if (SpawnDev.ILGPU.ML.Kernels.MaskKernels.TryDetectTriangular(new byte[] { 1 }, 1, 1, out _))
            throw new Exception("detector accepted a 1x1 tensor");

        Console.WriteLine("[DType] MaskKernels: 4 triangle modes match CPU, detector exact (1 flipped bit rejected)");
    });

    /// <summary>
    /// Quantized int8/uint8 weights stored NATIVE (a quarter of the fp32 bytes) and widened correctly at use.
    /// </summary>
    /// <remarks>
    /// Storage is the claim: an int8 weight that gets expanded to fp32 on load costs 4x the memory, which
    /// defeats the reason a model was quantized at all. So the assertions are DType, a native view of the
    /// right length, and Data.Length == 0 proving no float buffer was allocated alongside - the widening
    /// happens in DequantizeLinear at use, into that op's own output, never as a resident copy.
    /// <para>
    /// The conversion is checked over the FULL 8-bit range, not a sample: sbyte spans -128..127 and byte
    /// 0..255, so every representable value is verified rather than a handful of convenient ones. Sign
    /// handling is exactly where an 8-bit widening goes wrong (0x80 as -128 vs 128).
    /// </para>
    /// </remarks>
    [TestMethod]
    public Task DType_Int8Weights_StayNative_AndWidenExactly() => RunTest(async accelerator =>
    {
        using var pool = new BufferPool(accelerator);
        var conv = new SpawnDev.ILGPU.ML.Kernels.IntConvertKernels(accelerator);

        // ── INT8 (dtype 3): full signed range ──
        {
            var vals = Enumerable.Range(-128, 256).Select(i => (sbyte)i).ToArray();
            var raw = vals.Select(v => unchecked((byte)v)).ToArray();
            using var ms = new MemoryStream(raw);
            var w = await pool.AllocateLowPWeightFromStreamAsync<sbyte>(ms, 0, raw.Length, 3, new[] { raw.Length }, "q8");

            if (w.DType != SpawnDev.ILGPU.ML.Tensors.TensorDataType.Int8)
                throw new Exception($"int8 loaded as DType={w.DType}");
            if (w.Data.Length != 0)
                throw new Exception($"int8 weight allocated a float buffer (Data.Length={w.Data.Length}) - expanded 4x on load");
            var wv = w.AsView<sbyte>();
            if (wv.Length != raw.Length)
                throw new Exception($"native int8 view length {wv.Length}, expected {raw.Length}");

            using var outBuf = accelerator.Allocate1D<float>(raw.Length);
            conv.Int8ToFloat(wv, outBuf.View, raw.Length);
            await accelerator.SynchronizeAsync();
            var got = await outBuf.View.SubView(0, raw.Length).CopyToHostAsync();
            for (int i = 0; i < vals.Length; i++)
                if (got[i] != vals[i])
                    throw new Exception($"int8 widen: raw 0x{raw[i]:X2} -> {got[i]}, expected {vals[i]}");
        }

        // ── UINT8 (dtype 2): full unsigned range ──
        {
            var raw = Enumerable.Range(0, 256).Select(i => (byte)i).ToArray();
            using var ms = new MemoryStream(raw);
            var w = await pool.AllocateLowPWeightFromStreamAsync<byte>(ms, 0, raw.Length, 2, new[] { raw.Length }, "qu8");

            if (w.DType != SpawnDev.ILGPU.ML.Tensors.TensorDataType.UInt8)
                throw new Exception($"uint8 loaded as DType={w.DType}");
            if (w.Data.Length != 0)
                throw new Exception($"uint8 weight allocated a float buffer - expanded 4x on load");
            var wv = w.AsView<byte>();

            using var outBuf = accelerator.Allocate1D<float>(raw.Length);
            conv.UInt8ToFloat(wv, outBuf.View, raw.Length);
            await accelerator.SynchronizeAsync();
            var got = await outBuf.View.SubView(0, raw.Length).CopyToHostAsync();
            for (int i = 0; i < raw.Length; i++)
                if (got[i] != raw[i])
                    throw new Exception($"uint8 widen: raw {raw[i]} -> {got[i]}");
        }

        // Refusing a mismatched dtype matters as much as accepting a matching one.
        bool refused = false;
        try
        {
            using var bad = new MemoryStream(new byte[4]);
            await pool.AllocateLowPWeightFromStreamAsync<sbyte>(bad, 0, 4, 1 /* FLOAT32 */, new[] { 4 }, "bad");
        }
        catch (NotSupportedException) { refused = true; }
        if (!refused) throw new Exception("FLOAT32 was accepted into an int8 buffer - it must refuse, not convert");

        Console.WriteLine("[DType] int8/uint8 stay native (no float buffer) and widen exactly over the full 8-bit range");
    });
}
