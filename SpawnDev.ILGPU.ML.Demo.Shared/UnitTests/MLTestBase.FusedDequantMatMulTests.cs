using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// GGUF quantization correctness suite. THE ORACLE IS THE GGML SPEC: the test-local
/// reference decoders below are line-for-line ports of ggml-quants.c dequantize_row_*
/// (fetched verbatim 2026-06-11; see the seven P1 DevComms thread). Synthetic blocks are
/// encoded in the REAL GGUF byte layout - never in "whatever the kernel reads" order;
/// the previous version of this file encoded test data in the GPU kernel's own
/// (incorrect, interleaved) nibble order, so kernel and test were consistently wrong
/// together while every real GGUF file decoded as garbage.
///
/// Layers locked here:
///  1. GGUFModel CPU dequant == ggml reference, for ALL TEN quantized types.
///  2. FusedDequantMatMul GPU == reference-dequant + CPU matmul, per fused type.
///  3. FusedDequantGather GPU == reference rows, per fused type.
///  4. MatMulOperator routing carries the GGMLType (and refuses to guess without one).
///  5. GGUFGraphBuilder declares orientation/typing/tied-head per the contracts.
/// </summary>
public abstract partial class MLTestBase
{
    // ═════════════════════════════════════════════════════════════════════
    //  1. CPU dequant family lock - every type GGUFModel claims to support
    // ═════════════════════════════════════════════════════════════════════

    [TestMethod]
    public async Task GGUFDequant_CPU_MatchesGgmlReference_AllTypes() => await RunTest(async accelerator =>
    {
        // 512 elements = 16 legacy blocks or 2 K-quant super-blocks.
        var types = new[]
        {
            GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0,
            GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K,
            GGMLType.MXFP4,
        };
        foreach (var type in types)
        {
            const int elements = 512;
            var bytes = MakeBlocks(type, elements, new Random(7 + (int)type));
            var model = new GGUFModel
            {
                RawData = bytes,
                DataStartOffset = 0,
                Tensors = new[] { new GGUFTensorInfo { Name = "t", Dimensions = new long[] { elements }, Type = type, DataOffset = 0 } },
            };
            var got = model.GetTensorFloat32(model.Tensors[0])
                ?? throw new Exception($"{type}: GetTensorFloat32 returned null");
            var want = ReferenceDequant(type, bytes, elements);
            for (int i = 0; i < elements; i++)
                if (MathF.Abs(got[i] - want[i]) > 1e-6f)
                    throw new Exception(
                        $"{type}: CPU dequant diverges from the ggml reference at element {i}: " +
                        $"got {got[i]}, want {want[i]} (first divergence shown).");
        }
        Console.WriteLine("[GGUFDequant] CPU dequant matches the ggml reference for all 11 types");
    });

    // ═════════════════════════════════════════════════════════════════════
    //  2. Fused dequant MatMul GPU oracle - per fused type
    // ═════════════════════════════════════════════════════════════════════

    [TestMethod]
    public async Task FusedDequantMatMul_MatchesOracle_Q4_0() => await FusedMatMulOracle(GGMLType.Q4_0);
    [TestMethod]
    public async Task FusedDequantMatMul_MatchesOracle_Q8_0() => await FusedMatMulOracle(GGMLType.Q8_0);
    [TestMethod]
    public async Task FusedDequantMatMul_MatchesOracle_Q4_K() => await FusedMatMulOracle(GGMLType.Q4_K);
    [TestMethod]
    public async Task FusedDequantMatMul_MatchesOracle_Q6_K() => await FusedMatMulOracle(GGMLType.Q6_K);
    [TestMethod]
    public async Task FusedDequantMatMul_MatchesOracle_MXFP4() => await FusedMatMulOracle(GGMLType.MXFP4);

    private async Task FusedMatMulOracle(GGMLType type) => await RunTest(async accelerator =>
    {
        // K is a multiple of 256 so the same dims serve legacy and K-quant blocks.
        const int M = 2, K = 512, N = 3;
        var rng = new Random(42 + (int)type);

        var input = new float[M * K];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        // Weight = GGUF storage orientation: N rows, each a quantized run of K elements.
        int bytesPerRow = RowBytes(type, K);
        var weightBytes = new byte[N * bytesPerRow];
        var wRows = new float[N][];
        for (int n = 0; n < N; n++)
        {
            var rowBytes = MakeBlocks(type, K, rng);
            Buffer.BlockCopy(rowBytes, 0, weightBytes, n * bytesPerRow, bytesPerRow);
            wRows[n] = ReferenceDequant(type, rowBytes, K);
        }

        // Oracle: out[m,n] = sum_k input[m,k] * W[n,k] (the fused transposed-read contract).
        var expected = new float[M * N];
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                float sum = 0f;
                for (int k = 0; k < K; k++) sum += input[m * K + k] * wRows[n][k];
                expected[m * N + n] = sum;
            }

        using var inputBuf = accelerator.Allocate1D(input);
        using var weightBuf = AllocatePadded(accelerator, weightBytes);
        using var outputBuf = accelerator.Allocate1D<float>(M * N);

        using var fused = new FusedDequantMatMul(accelerator);
        fused.Forward(inputBuf.View, weightBuf.View, outputBuf.View, M, K, N, type);
        await accelerator.SynchronizeAsync();
        var gpuOut = await outputBuf.CopyToHostAsync<float>(0, M * N);

        AssertCloseQuant(gpuOut, expected, 2e-3f, $"FusedDequantMatMul[{type}]");
        Console.WriteLine($"[FusedDequantMatMul] {type}: GPU matches the ggml-reference oracle");
    });

    // ═════════════════════════════════════════════════════════════════════
    //  3. Fused dequant Gather GPU oracle (quantized embedding lookup)
    // ═════════════════════════════════════════════════════════════════════

    [TestMethod]
    public async Task FusedDequantGather_MatchesOracle_Q4_K() => await FusedGatherOracle(GGMLType.Q4_K);
    [TestMethod]
    public async Task FusedDequantGather_MatchesOracle_Q6_K() => await FusedGatherOracle(GGMLType.Q6_K);
    [TestMethod]
    public async Task FusedDequantGather_MatchesOracle_MXFP4() => await FusedGatherOracle(GGMLType.MXFP4);

    private async Task FusedGatherOracle(GGMLType type) => await RunTest(async accelerator =>
    {
        const int rows = 8, rowLength = 256;
        var rng = new Random(11 + (int)type);
        int bytesPerRow = RowBytes(type, rowLength);
        var tableBytes = new byte[rows * bytesPerRow];
        var refRows = new float[rows][];
        for (int r = 0; r < rows; r++)
        {
            var rowBytes = MakeBlocks(type, rowLength, rng);
            Buffer.BlockCopy(rowBytes, 0, tableBytes, r * bytesPerRow, bytesPerRow);
            refRows[r] = ReferenceDequant(type, rowBytes, rowLength);
        }

        var indices = new float[] { 3, 0, 7, 3, 5 };
        int numIdx = indices.Length;
        var expected = new float[numIdx * rowLength];
        for (int i = 0; i < numIdx; i++)
            Array.Copy(refRows[(int)indices[i]], 0, expected, i * rowLength, rowLength);

        using var tableBuf = AllocatePadded(accelerator, tableBytes);
        using var idxBuf = accelerator.Allocate1D(indices);
        using var outBuf = accelerator.Allocate1D<float>(numIdx * rowLength);

        using var gather = new FusedDequantGather(accelerator);
        gather.GatherAxis0(tableBuf.View, idxBuf.View, outBuf.View, numIdx, rowLength, rows, type);
        await accelerator.SynchronizeAsync();
        var gpuOut = await outBuf.CopyToHostAsync<float>(0, numIdx * rowLength);

        AssertCloseQuant(gpuOut, expected, 1e-5f, $"FusedDequantGather[{type}]");
        Console.WriteLine($"[FusedDequantGather] {type}: gathered rows match the reference");
    });

    // ═════════════════════════════════════════════════════════════════════
    //  4. Operator routing: type travels with the bytes; no type = refuse
    // ═════════════════════════════════════════════════════════════════════

    [TestMethod]
    public async Task QuantDequantRouting_ViaOperator_TypeRequired() => await RunTest(async accelerator =>
    {
        const int M = 2, K = 256, N = 2;
        var type = GGMLType.Q4_K;
        var rng = new Random(99);

        var input = new float[M * K];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        int bytesPerRow = RowBytes(type, K);
        var weightBytes = new byte[N * bytesPerRow];
        for (int n = 0; n < N; n++)
            Buffer.BlockCopy(MakeBlocks(type, K, rng), 0, weightBytes, n * bytesPerRow, bytesPerRow);

        using var inputBuf = accelerator.Allocate1D(input);
        using var weightBuf = AllocatePadded(accelerator, weightBytes);
        using var directOut = accelerator.Allocate1D<float>(M * N);
        using var routedOut = accelerator.Allocate1D<float>(M * N);

        using var registry = new OperatorRegistry(accelerator);
        registry.FusedDequant.Forward(inputBuf.View, weightBuf.View, directOut.View, M, K, N, type);
        await accelerator.SynchronizeAsync();
        var direct = await directOut.CopyToHostAsync<float>(0, M * N);

        var matmulOp = registry.Resolve("MatMul");
        var pool = new BufferPool(accelerator);
        OnnxOpContext MakeCtx() => new OnnxOpContext
        {
            Inputs = new[]
            {
                new Tensor(inputBuf.View, new[] { M, K }),
                Tensor.ShapeOnly(new[] { K, N }, "w_q"),
            },
            Outputs = new[] { new Tensor(routedOut.View, new[] { M, N }) },
            Attributes = new Dictionary<string, object>(),
            Pool = pool,
            InputNames = new[] { "input", "w_q" },
            QuantizedWeights = new Dictionary<string, ArrayView1D<byte, Stride1D.Dense>> { ["w_q"] = weightBuf.View },
        };

        // Without a registered type the operator must refuse - guessing a block layout
        // is the exact bug class this suite exists to kill.
        registry.QuantizedWeightTypes = null;
        bool threw = false;
        try { matmulOp.Execute(MakeCtx()); }
        catch (InvalidOperationException) { threw = true; }
        if (!threw) throw new Exception("MatMul accepted a quantized weight WITHOUT a GGML type.");

        registry.QuantizedWeightTypes = new Dictionary<string, GGMLType> { ["w_q"] = type };
        matmulOp.Execute(MakeCtx());
        await accelerator.SynchronizeAsync();
        var routed = await routedOut.CopyToHostAsync<float>(0, M * N);

        AssertCloseQuant(routed, direct, 0f, "QuantMatMul routed-vs-direct");
        Console.WriteLine("[QuantRouting] MatMulOperator routes by GGMLType and refuses untyped quantized weights");
    });

    // ═════════════════════════════════════════════════════════════════════
    //  5. Graph-builder contracts: typing, orientation, tied head
    // ═════════════════════════════════════════════════════════════════════

    [TestMethod]
    public async Task GGUFDequant_GraphBuilder_Typing_Orientation_TiedHead() => await RunTest(async accelerator =>
    {
        const int embd = 256, vocab = 32, ffn = 320;
        var rng = new Random(5);

        var raw = new List<byte>();
        var tensors = new List<GGUFTensorInfo>();
        void Add(string name, long[] ne, GGMLType type, byte[] bytes)
        {
            tensors.Add(new GGUFTensorInfo { Name = name, Dimensions = ne, Type = type, DataOffset = (ulong)raw.Count });
            raw.AddRange(bytes);
        }
        byte[] F32(int count)
        {
            var b = new byte[count * 4];
            rng.NextBytes(b);
            // Re-write as small sane floats
            for (int i = 0; i < count; i++)
                BitConverter.GetBytes((float)(rng.NextDouble() - 0.5)).CopyTo(b, i * 4);
            return b;
        }
        byte[] QRows(GGMLType t, int k, int n)
        {
            var all = new byte[n * RowBytes(t, k)];
            for (int r = 0; r < n; r++)
                Buffer.BlockCopy(MakeBlocks(t, k, rng), 0, all, r * RowBytes(t, k), RowBytes(t, k));
            return all;
        }

        // GGUF ne order is fastest-dim-first: linear ne=[K,N] (storage [N][K]),
        // embedding ne=[embd, vocab] (storage [vocab][embd]).
        Add("token_embd.weight", new long[] { embd, vocab }, GGMLType.Q6_K, QRows(GGMLType.Q6_K, embd, vocab));
        Add("blk.0.attn_norm.weight", new long[] { embd }, GGMLType.F32, F32(embd));
        Add("blk.0.attn_q.weight", new long[] { embd, embd }, GGMLType.Q4_K, QRows(GGMLType.Q4_K, embd, embd));
        Add("blk.0.attn_k.weight", new long[] { embd, embd }, GGMLType.Q4_K, QRows(GGMLType.Q4_K, embd, embd));
        Add("blk.0.attn_v.weight", new long[] { embd, embd }, GGMLType.Q4_K, QRows(GGMLType.Q4_K, embd, embd));
        Add("blk.0.attn_output.weight", new long[] { embd, embd }, GGMLType.Q4_K, QRows(GGMLType.Q4_K, embd, embd));
        Add("blk.0.ffn_norm.weight", new long[] { embd }, GGMLType.F32, F32(embd));
        Add("blk.0.ffn_gate.weight", new long[] { embd, ffn }, GGMLType.Q4_K, QRows(GGMLType.Q4_K, embd, ffn));
        Add("blk.0.ffn_up.weight", new long[] { embd, ffn }, GGMLType.Q4_K, QRows(GGMLType.Q4_K, embd, ffn));
        // F32 linear: must land in TransposeOnUpload (storage [N][K] -> declared [K,N]).
        Add("blk.0.ffn_down.weight", new long[] { ffn, embd }, GGMLType.F32, F32(ffn * embd));
        Add("output_norm.weight", new long[] { embd }, GGMLType.F32, F32(embd));
        // NO output.weight -> tied-embedding LM head.

        var model = new GGUFModel
        {
            RawData = raw.ToArray(),
            DataStartOffset = 0,
            Tensors = tensors.ToArray(),
            Metadata = new Dictionary<string, object>
            {
                ["general.architecture"] = "llama",
                ["general.name"] = "tiny-test",
                ["llama.embedding_length"] = (long)embd,
                ["llama.block_count"] = 1L,
                ["llama.attention.head_count"] = 4L,
                ["llama.attention.head_count_kv"] = 4L,
                ["llama.vocab_size"] = (long)vocab,
                ["llama.feed_forward_length"] = (long)ffn,
                ["llama.context_length"] = 64L,
            },
        };

        var (graph, weights, quantized, transposeOnUpload) = GGUFGraphBuilder.BuildGraph(model);

        // Embedding declared in PHYSICAL order [vocab, embd] (reversed ne) for Gather.
        var embedDecl = graph.Initializers["token_embd.weight"];
        if (embedDecl[0] != vocab || embedDecl[1] != embd)
            throw new Exception($"Embedding declared [{embedDecl[0]},{embedDecl[1]}], want [{vocab},{embd}] (physical order).");

        // Quantized types recorded; linears declared [K, N] = ne order.
        if (quantized["blk.0.attn_q.weight"].Type != GGMLType.Q4_K)
            throw new Exception("attn_q type not recorded as Q4_K.");
        var qDecl = graph.Initializers["blk.0.attn_q.weight"];
        if (qDecl[0] != embd || qDecl[1] != embd)
            throw new Exception("attn_q declaration is not [K, N].");

        // F32 linear marked for GPU transpose at upload; quantized linears are NOT.
        if (!transposeOnUpload.Contains("blk.0.ffn_down.weight"))
            throw new Exception("F32 linear ffn_down missing from TransposeOnUpload.");
        if (transposeOnUpload.Contains("blk.0.attn_q.weight"))
            throw new Exception("Quantized linear attn_q must NOT be transposed (fused kernels read storage order).");

        // Tied head: alias shares the SAME bytes (single upload), declared [embd, vocab],
        // consumed by a direct MatMul - no Transpose node on the embedding.
        var head = quantized["token_embd.weight#lm_head"];
        if (!ReferenceEquals(head.Bytes, quantized["token_embd.weight"].Bytes))
            throw new Exception("Tied head alias does not share the embedding's byte array.");
        var headDecl = graph.Initializers["token_embd.weight#lm_head"];
        if (headDecl[0] != embd || headDecl[1] != vocab)
            throw new Exception($"Tied head declared [{headDecl[0]},{headDecl[1]}], want [{embd},{vocab}].");
        if (graph.Nodes.Any(n => n.OpType == "Transpose" && n.Inputs.Contains("token_embd.weight")))
            throw new Exception("Embedding must not be consumed by a Transpose node (tied head reads storage directly).");
        if (!graph.Nodes.Any(n => n.OpType == "MatMul" && n.Inputs.Contains("token_embd.weight#lm_head")
                && n.Outputs.Contains("logits")))
            throw new Exception("Tied-head MatMul against the alias not found.");

        Console.WriteLine("[GGUFGraphBuilder] typing, orientation, and tied-head contracts hold");
        await Task.CompletedTask;
    });

    // ═════════════════════════════════════════════════════════════════════
    //  Synthetic block encoding (REAL GGUF layout) + ggml reference decoders
    // ═════════════════════════════════════════════════════════════════════

    private static int BlockBytes(GGMLType t) => t switch
    {
        GGMLType.Q4_0 => 18, GGMLType.Q4_1 => 20, GGMLType.Q5_0 => 22, GGMLType.Q5_1 => 24,
        GGMLType.Q8_0 => 34,
        GGMLType.Q2_K => 84, GGMLType.Q3_K => 110, GGMLType.Q4_K => 144,
        GGMLType.Q5_K => 176, GGMLType.Q6_K => 210,
        GGMLType.MXFP4 => 17,
        _ => throw new ArgumentException($"no block size for {t}"),
    };

    private static int BlockElems(GGMLType t) =>
        t is GGMLType.Q2_K or GGMLType.Q3_K or GGMLType.Q4_K or GGMLType.Q5_K or GGMLType.Q6_K ? 256 : 32;

    private static int RowBytes(GGMLType t, int k) => k / BlockElems(t) * BlockBytes(t);

    /// <summary>Random-but-valid quantized blocks: random quant/scale-index bytes with the
    /// fp16 scale fields patched to small sane values (random fp16 bits could be NaN/Inf).
    /// Any byte pattern decodes deterministically, so random bytes exercise every bit path.</summary>
    private static byte[] MakeBlocks(GGMLType type, int elements, Random rng)
    {
        int nBlocks = elements / BlockElems(type);
        int bs = BlockBytes(type);
        var bytes = new byte[nBlocks * bs];
        rng.NextBytes(bytes);
        // MXFP4: the per-block scale is a single E8M0 byte (no fp16 fields). Random bytes decode fine, but
        // patch the scale byte to a sane exponent (~127 -> scale ~O(1)) so magnitudes stay comparable to the
        // other types (a random e up to 255 would give 2^127). The 16 nibble bytes stay fully random.
        if (type == GGMLType.MXFP4)
        {
            for (int b = 0; b < nBlocks; b++)
                bytes[b * bs] = (byte)(126 + rng.Next(0, 4)); // e in 126..129 -> 2^(e-128) in 0.25..2
            return bytes;
        }
        int[] halfOffsets = type switch
        {
            GGMLType.Q4_0 or GGMLType.Q5_0 or GGMLType.Q8_0 => new[] { 0 },
            GGMLType.Q4_1 or GGMLType.Q5_1 or GGMLType.Q4_K or GGMLType.Q5_K => new[] { 0, 2 },
            GGMLType.Q2_K => new[] { 80, 82 },
            GGMLType.Q3_K => new[] { 108 },
            GGMLType.Q6_K => new[] { 208 },
            _ => throw new ArgumentException($"{type}"),
        };
        for (int b = 0; b < nBlocks; b++)
            foreach (int off in halfOffsets)
            {
                int h = FloatToHalf(0.02f + (float)rng.NextDouble() * 0.2f);
                bytes[b * bs + off] = (byte)(h & 0xFF);
                bytes[b * bs + off + 1] = (byte)((h >> 8) & 0xFF);
            }
        return bytes;
    }

    /// <summary>Dispatch to the per-type ggml reference decoder.</summary>
    private static float[] ReferenceDequant(GGMLType type, byte[] data, int elements) => type switch
    {
        GGMLType.Q4_0 => RefQ4_0(data, elements),
        GGMLType.Q4_1 => RefQ4_1(data, elements),
        GGMLType.Q5_0 => RefQ5_0(data, elements),
        GGMLType.Q5_1 => RefQ5_1(data, elements),
        GGMLType.Q8_0 => RefQ8_0(data, elements),
        GGMLType.Q2_K => RefQ2_K(data, elements),
        GGMLType.Q3_K => RefQ3_K(data, elements),
        GGMLType.Q4_K => RefQ4_K(data, elements),
        GGMLType.Q5_K => RefQ5_K(data, elements),
        GGMLType.Q6_K => RefQ6_K(data, elements),
        GGMLType.MXFP4 => RefMXFP4(data, elements),
        _ => throw new ArgumentException($"{type}"),
    };

    private static float Half(byte[] d, int off) => HalfToFloatCPU(d[off] | (d[off + 1] << 8));

    /// <summary>ggml dequantize_row_mxfp4: 17 B/block ([e:E8M0][16 nibble bytes]); low nibble of byte j is
    /// element j, high nibble is element j+16; value = kvalues_mxfp4[nibble] * 2^(e-128). kvalues table
    /// {0,1,2,3,4,6,8,12, 0,-1,..,-12} written out literally here (independent of the production bit-math).</summary>
    private static readonly int[] KvaluesMXFP4 = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
    private static float[] RefMXFP4(byte[] d, int n)
    {
        var y = new float[n];
        for (int i = 0; i < n / 32; i++)
        {
            int o = i * 17;
            float s = MathF.Pow(2f, d[o] - 128f);
            for (int j = 0; j < 16; j++)
            {
                y[i * 32 + j] = KvaluesMXFP4[d[o + 1 + j] & 0xF] * s;
                y[i * 32 + j + 16] = KvaluesMXFP4[d[o + 1 + j] >> 4] * s;
            }
        }
        return y;
    }

    private static float[] RefQ4_0(byte[] d, int n)
    {
        var y = new float[n];
        for (int i = 0; i < n / 32; i++)
        {
            int o = i * 18; float s = Half(d, o);
            for (int j = 0; j < 16; j++)
            {
                y[i * 32 + j] = ((d[o + 2 + j] & 0xF) - 8) * s;
                y[i * 32 + j + 16] = ((d[o + 2 + j] >> 4) - 8) * s;
            }
        }
        return y;
    }

    private static float[] RefQ4_1(byte[] d, int n)
    {
        var y = new float[n];
        for (int i = 0; i < n / 32; i++)
        {
            int o = i * 20; float s = Half(d, o); float m = Half(d, o + 2);
            for (int j = 0; j < 16; j++)
            {
                y[i * 32 + j] = (d[o + 4 + j] & 0xF) * s + m;
                y[i * 32 + j + 16] = (d[o + 4 + j] >> 4) * s + m;
            }
        }
        return y;
    }

    private static float[] RefQ5_0(byte[] d, int n)
    {
        var y = new float[n];
        for (int i = 0; i < n / 32; i++)
        {
            int o = i * 22; float s = Half(d, o);
            uint qh = (uint)(d[o + 2] | (d[o + 3] << 8) | (d[o + 4] << 16) | (d[o + 5] << 24));
            for (int j = 0; j < 16; j++)
            {
                int xh0 = (int)((qh >> j) << 4) & 0x10;
                int xh1 = (int)(qh >> (j + 12)) & 0x10;
                y[i * 32 + j] = (((d[o + 6 + j] & 0xF) | xh0) - 16) * s;
                y[i * 32 + j + 16] = (((d[o + 6 + j] >> 4) | xh1) - 16) * s;
            }
        }
        return y;
    }

    private static float[] RefQ5_1(byte[] d, int n)
    {
        var y = new float[n];
        for (int i = 0; i < n / 32; i++)
        {
            int o = i * 24; float s = Half(d, o); float m = Half(d, o + 2);
            uint qh = (uint)(d[o + 4] | (d[o + 5] << 8) | (d[o + 6] << 16) | (d[o + 7] << 24));
            for (int j = 0; j < 16; j++)
            {
                int xh0 = (int)((qh >> j) << 4) & 0x10;
                int xh1 = (int)(qh >> (j + 12)) & 0x10;
                y[i * 32 + j] = ((d[o + 8 + j] & 0xF) | xh0) * s + m;
                y[i * 32 + j + 16] = ((d[o + 8 + j] >> 4) | xh1) * s + m;
            }
        }
        return y;
    }

    private static float[] RefQ8_0(byte[] d, int n)
    {
        var y = new float[n];
        for (int i = 0; i < n / 32; i++)
        {
            int o = i * 34; float s = Half(d, o);
            for (int j = 0; j < 32; j++) y[i * 32 + j] = (sbyte)d[o + 2 + j] * s;
        }
        return y;
    }

    private static float[] RefQ2_K(byte[] d, int n)
    {
        var y = new float[n]; int yi = 0;
        for (int i = 0; i < n / 256; i++)
        {
            int o = i * 84;
            float dd = Half(d, o + 80); float dm = Half(d, o + 82);
            int q = o + 16; int isx = 0;
            for (int half = 0; half < 2; half++)
            {
                for (int shift = 0; shift <= 6; shift += 2)
                {
                    byte sc = d[o + isx++];
                    float dl = dd * (sc & 0xF); float ml = dm * (sc >> 4);
                    for (int l = 0; l < 16; l++) y[yi++] = dl * ((d[q + l] >> shift) & 3) - ml;
                    sc = d[o + isx++];
                    dl = dd * (sc & 0xF); ml = dm * (sc >> 4);
                    for (int l = 0; l < 16; l++) y[yi++] = dl * ((d[q + l + 16] >> shift) & 3) - ml;
                }
                q += 32;
            }
        }
        return y;
    }

    private static float[] RefQ3_K(byte[] d, int n)
    {
        var y = new float[n]; int yi = 0;
        for (int i = 0; i < n / 256; i++)
        {
            int o = i * 110;
            float dAll = Half(d, o + 108);
            int hm = o, q = o + 32, scOff = o + 96;
            uint a0 = (uint)(d[scOff] | (d[scOff + 1] << 8) | (d[scOff + 2] << 16) | (d[scOff + 3] << 24));
            uint a1 = (uint)(d[scOff + 4] | (d[scOff + 5] << 8) | (d[scOff + 6] << 16) | (d[scOff + 7] << 24));
            uint tmp = (uint)(d[scOff + 8] | (d[scOff + 9] << 8) | (d[scOff + 10] << 16) | (d[scOff + 11] << 24));
            const uint k1 = 0x03030303, k2 = 0x0f0f0f0f;
            var aux = new uint[4];
            aux[2] = ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4);
            aux[3] = ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4);
            aux[0] = (a0 & k2) | (((tmp >> 0) & k1) << 4);
            aux[1] = (a1 & k2) | (((tmp >> 2) & k1) << 4);
            int Scale(int j) => (sbyte)(byte)(aux[j / 4] >> ((j % 4) * 8));
            int isx = 0; int mBit = 1;
            for (int half = 0; half < 2; half++)
            {
                for (int shift = 0; shift <= 6; shift += 2)
                {
                    float dl = dAll * (Scale(isx++) - 32);
                    for (int l = 0; l < 16; l++)
                        y[yi++] = dl * (((d[q + l] >> shift) & 3) - ((d[hm + l] & mBit) != 0 ? 0 : 4));
                    dl = dAll * (Scale(isx++) - 32);
                    for (int l = 0; l < 16; l++)
                        y[yi++] = dl * (((d[q + l + 16] >> shift) & 3) - ((d[hm + l + 16] & mBit) != 0 ? 0 : 4));
                    mBit <<= 1;
                }
                q += 32;
            }
        }
        return y;
    }

    private static (int sc, int m) ScaleMinK4(byte[] d, int scOff, int j)
    {
        if (j < 4) return (d[scOff + j] & 63, d[scOff + j + 4] & 63);
        return ((d[scOff + j + 4] & 0xF) | ((d[scOff + j - 4] >> 6) << 4),
                (d[scOff + j + 4] >> 4) | ((d[scOff + j] >> 6) << 4));
    }

    private static float[] RefQ4_K(byte[] d, int n)
    {
        var y = new float[n]; int yi = 0;
        for (int i = 0; i < n / 256; i++)
        {
            int o = i * 144;
            float dd = Half(d, o); float dm = Half(d, o + 2);
            int q = o + 16; int isx = 0;
            for (int j = 0; j < 256; j += 64)
            {
                var (sc1, m1) = ScaleMinK4(d, o + 4, isx + 0);
                var (sc2, m2) = ScaleMinK4(d, o + 4, isx + 1);
                float d1 = dd * sc1, mm1 = dm * m1, d2 = dd * sc2, mm2 = dm * m2;
                for (int l = 0; l < 32; l++) y[yi + l] = d1 * (d[q + l] & 0xF) - mm1;
                for (int l = 0; l < 32; l++) y[yi + 32 + l] = d2 * (d[q + l] >> 4) - mm2;
                yi += 64; q += 32; isx += 2;
            }
        }
        return y;
    }

    private static float[] RefQ5_K(byte[] d, int n)
    {
        var y = new float[n]; int yi = 0;
        for (int i = 0; i < n / 256; i++)
        {
            int o = i * 176;
            float dd = Half(d, o); float dm = Half(d, o + 2);
            int qh = o + 16, ql = o + 48; int isx = 0; int u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64)
            {
                var (sc1, m1) = ScaleMinK4(d, o + 4, isx + 0);
                var (sc2, m2) = ScaleMinK4(d, o + 4, isx + 1);
                float d1 = dd * sc1, mm1 = dm * m1, d2 = dd * sc2, mm2 = dm * m2;
                for (int l = 0; l < 32; l++)
                    y[yi + l] = d1 * ((d[ql + l] & 0xF) + ((d[qh + l] & u1) != 0 ? 16 : 0)) - mm1;
                for (int l = 0; l < 32; l++)
                    y[yi + 32 + l] = d2 * ((d[ql + l] >> 4) + ((d[qh + l] & u2) != 0 ? 16 : 0)) - mm2;
                yi += 64; ql += 32; isx += 2; u1 <<= 2; u2 <<= 2;
            }
        }
        return y;
    }

    private static float[] RefQ6_K(byte[] d, int n)
    {
        var y = new float[n]; int yi = 0;
        for (int i = 0; i < n / 256; i++)
        {
            int o = i * 210;
            float dd = Half(d, o + 208);
            int ql = o, qh = o + 128, sc = o + 192;
            for (int half = 0; half < 2; half++)
            {
                for (int l = 0; l < 32; l++)
                {
                    int isx = l / 16;
                    int q1 = ((d[ql + l] & 0xF) | (((d[qh + l] >> 0) & 3) << 4)) - 32;
                    int q2 = ((d[ql + l + 32] & 0xF) | (((d[qh + l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((d[ql + l] >> 4) | (((d[qh + l] >> 4) & 3) << 4)) - 32;
                    int q4 = ((d[ql + l + 32] >> 4) | (((d[qh + l] >> 6) & 3) << 4)) - 32;
                    y[yi + l] = dd * (sbyte)d[sc + isx] * q1;
                    y[yi + l + 32] = dd * (sbyte)d[sc + isx + 2] * q2;
                    y[yi + l + 64] = dd * (sbyte)d[sc + isx + 4] * q3;
                    y[yi + l + 96] = dd * (sbyte)d[sc + isx + 6] * q4;
                }
                yi += 128; ql += 64; qh += 32; sc += 8;
            }
        }
        return y;
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Small shared helpers
    // ═════════════════════════════════════════════════════════════════════

    /// <summary>Upload bytes padded to a 4-byte multiple (the fused kernels read packed
    /// int words; Cast&lt;byte,int&gt; truncates a ragged tail) - mirrors the loader.</summary>
    private static MemoryBuffer1D<byte, Stride1D.Dense> AllocatePadded(Accelerator acc, byte[] bytes)
    {
        int padded = (bytes.Length + 3) & ~3;
        var buf = acc.Allocate1D<byte>(padded);
        buf.View.SubView(0, bytes.Length).CopyFromCPU(bytes);
        return buf;
    }

    private static void AssertCloseQuant(float[] got, float[] want, float relTol, string what)
    {
        for (int i = 0; i < want.Length; i++)
        {
            float tol = relTol == 0f ? 0f : MathF.Max(relTol, MathF.Abs(want[i]) * relTol);
            if (MathF.Abs(got[i] - want[i]) > tol || float.IsNaN(got[i]))
                throw new Exception(
                    $"{what} mismatch @{i}: got {got[i]}, want {want[i]} (tol {tol}). " +
                    $"got=[{string.Join(", ", got.Take(8).Select(v => v.ToString("F4")))}...] " +
                    $"want=[{string.Join(", ", want.Take(8).Select(v => v.ToString("F4")))}...]");
        }
    }

    /// <summary>FP32 → FP16 bits (test data generation; normal range only).</summary>
    private static int FloatToHalf(float f)
    {
        if (f == 0) return 0;
        int sign = f < 0 ? 1 : 0;
        f = MathF.Abs(f);
        int exp = (int)MathF.Floor(MathF.Log2(f));
        float frac = f / MathF.Pow(2, exp) - 1f;
        int biasedExp = exp + 15;
        if (biasedExp <= 0) return (sign << 15);
        if (biasedExp >= 31) return (sign << 15) | 0x7C00;
        int mant = (int)(frac * 1024f + 0.5f);
        if (mant > 1023) mant = 1023;
        return (sign << 15) | (biasedExp << 10) | mant;
    }

    /// <summary>FP16 bits → FP32 (matches the kernel/CPU HalfToFloat formula).</summary>
    private static float HalfToFloatCPU(int h)
    {
        int sign = (h >> 15) & 1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0) return mant == 0 ? (sign == 1 ? -0f : 0f) : (sign == 1 ? -1 : 1) * mant / 1024f * (1f / 16384f);
        if (exp == 31) return mant == 0 ? (sign == 1 ? float.NegativeInfinity : float.PositiveInfinity) : float.NaN;
        float result = (1f + mant / 1024f) * MathF.Pow(2, exp - 15);
        return sign == 1 ? -result : result;
    }
}
