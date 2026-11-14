using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Preprocessing;

namespace SpawnDev.ILGPU.ML.Pipelines;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// Several continuations of one prompt decoded together (a Qwen2-family GGUF: qwen2 / qwen2.5, NeoX rotary, biased q/k/v): the prompt is prefilled once, its K/V copied
/// into one cache region per sequence, then every step feeds the B sequences' last tokens as one [B, hidden] batch, so each quantised matrix is read once per step for all B
/// (<see cref="FusedDequantMatMul"/> with M = B) instead of B times. All sequences sit at the same position (one new token each per step), so they share the rotary angle
/// and the attention length; only their caches differ. The building block for self-consistency (sample N answers, vote: Plans/test-time-search-2026-10-31.md), best-of-N,
/// beam search and parallel candidates.
/// <para>
/// Weights stay as the GGUF stores them: quantised matrices as raw blocks on the device, norms and biases as floats; the token embedding's blocks are on the device too
/// (its rows are gathered and dequantised there; a tied LM head is the same blocks).
/// </para>
/// <para>
/// Each forward ends on the device with every row's <see cref="Candidates"/> best tokens (exact top-k, ties to the lower id: the K-th best chunk maximum is a floor that
/// at least K tokens reach, the tokens at or above it are gathered and sorted) and, when a temperature is given, the row's softmax mass in partial sums, so a step reads
/// back a few thousand floats instead of [B, Vocab] logits. <see cref="SampleAsync"/> draws from
/// the candidates and is <see cref="SampleLogits(float[], int, int, float, float, Random)"/> on the full row (the same nucleus, the same draw); when the nucleus is wider than the candidates it reads that row.
/// </para>
/// <para>
/// Device draws (<see cref="Draw"/>, <see cref="StepDrawnAsync"/>, <see cref="ReadDrawnAsync"/>; <see cref="GenerateAsync"/> by default): the draw is a kernel at the
/// end of the step (from the candidates, or exactly from the whole row when the nucleus is wider), its uniform a hash of (seed, sequence, position)
/// (<see cref="DrawUniform"/>), the token written where the next step reads its input. No step waits for the host: <see cref="GenerateAsync"/> queues step P before it
/// reads the tokens of P, so the device always has work while the host streams and stops sequences one step behind.
/// </para>
/// </summary>
public sealed class GgufBatchDecoder : IDisposable
{
    private readonly Accelerator _acc;
    private readonly GGUFModel _model;
    private readonly Stream _stream;
    private readonly FusedDequantMatMul _mm;
    // OverlapV's second stream gets its OWN matmul: FusedDequantMatMul stages prefill weights and activations in per-instance scratch (the f16 scratch GEMM
    // from 512 rows, the int8 GEMM always), so v's GEMM on one instance with q|k's overwrote q|k's weights mid-GEMM (2026-11-10, ML 5.8.11 - 5.8.12, CUDA,
    // prompts over 512 tokens where v's block type differs from q|k; GgufBatchDecoder_OverlapV_LongPrefill_BitIdentical)
    private FusedDequantMatMul? _mmV;
    private readonly NormalizationKernels _norm;
    private readonly List<IDisposable> _owned = new();
    private readonly Dictionary<string, (MemoryBuffer1D<byte, Stride1D.Dense> Data, GGMLType Type)> _mats = new();
    private readonly Dictionary<string, MemoryBuffer1D<float, Stride1D.Dense>> _vecs = new();
    private readonly GGUFTensorInfo _embd;
    private readonly string _head;

    /// <summary>The parsed GGUF header (metadata, tokenizer, chat template).</summary>
    public GGUFModel Model => _model;

    /// <summary>Model width, layers, heads, key/value heads, head width, feed-forward width, vocabulary.</summary>
    public int Hidden { get; }
    public int Layers { get; }
    public int Heads { get; }
    public int KvHeads { get; }
    public int HeadDim { get; }
    /// <summary>Query width: heads x head width (qwen3: not the model width).</summary>
    public int QDim { get; }
    private readonly bool _normRope, _qkNorm;
    public int FfnDim { get; }
    public int Vocab { get; }
    /// <summary>Longest sequence (prompt + generation) and most sequences the caches hold.</summary>
    public int MaxSeq { get; }
    public int MaxBatch { get; }
    /// <summary>The position the next token goes to (the prompt length after <see cref="PrefillAsync"/>, plus one per <see cref="StepAsync"/>).</summary>
    public int Position { get; private set; }
    /// <summary>
    /// Where the time of all decode steps since <see cref="PrefillAsync"/> went, in ms: the token ids and sequence map up, the forward queued (WebGPU: a replay;
    /// desktop launches overlap the device), then the wait for the device and the candidates read (<see cref="StepAsync(int[], float, CancellationToken)"/>; a
    /// <see cref="StepDrawnAsync"/> does not wait, its tokens are read by <see cref="ReadDrawnAsync"/>).
    /// </summary>
    public (double Embed, double Device, double Readback, int Steps) DecodeTimings { get; private set; }
    /// <summary>
    /// Diagnostics (desktop accelerators: it synchronises after every kernel): when set, each forward adds every kernel's time, its own GPU drain included, to
    /// <see cref="KernelTimes"/> (ms, by kernel). Off: no cost.
    /// </summary>
    public bool ProfileKernels { get; set; }
    /// <summary>Accumulated kernel times while <see cref="ProfileKernels"/> is on (ms by kernel name; cleared by <see cref="PrefillAsync"/>).</summary>
    public Dictionary<string, double> KernelTimes { get; } = new();
    private long _profT;
    private void Prof(string name)
    {
        if (!ProfileKernels) return;
        _acc.Synchronize();
        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        KernelTimes[name] = KernelTimes.GetValueOrDefault(name) + (now - _profT) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        _profT = now;
    }
    /// <summary>
    /// Attention as two kernels (split-K: per row, head and chunk of 256 keys the scores, their max and exp-sum and P.V in one group; then the chunks merged) instead
    /// of three (scores, a thread-per-row softmax, P.V with a thread per output walking every key). On by default where a group can be a head wide; set false for the
    /// three-kernel path (the A/B and the fallback).
    /// </summary>
    public bool FusedAttention { get => _fusedAttention; set { if (_fusedAttention != value) { _fusedAttention = value; _bufGen++; } } }
    private bool _fusedAttention;
    private const int AttChunk = 256;
    /// <summary>
    /// The split-K attention over the K / V caches grouped by kv head (<c>AttnPartialGqaImpl</c>): the query heads sharing a kv head (grouped-query attention:
    /// 4 on llama3 / qwen3 8B) in one group, so each cached K / V row is read once a step instead of once per query head. Where the heads are grouped (at most
    /// 8 a kv head, head width at most 128). Default on; GGUF_BATCH_GQA=0 (or false) is the A/B.
    /// </summary>
    // a captured WebGPU step replays the kernels it recorded: changing the route bumps _bufGen so the next step captures again (before 2026-11-02 a decoder
    // that had captured kept replaying the old route after the property changed)
    public bool GqaAttention { get => _gqaAttention; set { if (_gqaAttention != value) { _gqaAttention = value; _bufGen++; } } }
    private bool _gqaAttention = Environment.GetEnvironmentVariable("GGUF_BATCH_GQA") != "0";
    /// <summary>
    /// Keys per chunk of the grouped-query decode attention (0: with <see cref="AttentionIlp"/> 64 or fewer, else AttChunk / G rounded up to a power of two, so a step has as many groups as the per-head kernel's and
    /// each group the same work). Settable for the A/B (env GGUF_GQA_CHUNK; AttChunk = the chunking before 2026-11-02); clamped to [16, AttChunk] (the shared
    /// scores hold GqaMaxG x AttChunk).
    /// </summary>
    public int GqaChunk { get => _gqaChunk; set { if (_gqaChunk != value) { _gqaChunk = value; _bufGen++; } } }
    /// <summary>
    /// The decode attention on <c>AttnPartialGqaIlpImpl</c> (chunks past the position leave at once; K / V loads issued ahead of their FMAs; bit-identical) instead of
    /// <c>AttnPartialGqaImpl</c>. Default: on for CUDA and OpenCL, and for WebGPU with subgroups (where <see cref="AttentionVec"/> runs instead; see the
    /// constructor), off for WebGPU without them, per the A/B below (env GGUF_ATTN_ILP=0 / 1 forces it). With <see cref="AttentionVec"/>
    /// it picks the vector kernel's P.V loop: four V rows loaded ahead of their FMAs (bit-identical) or one at a time.
    /// </summary>
    public bool AttentionIlp { get => _attIlp; set { if (_attIlp != value) { _attIlp = value; _bufGen++; } } }
    /// <summary>
    /// The grouped-query decode attention (<see cref="AttentionIlp"/>'s kernel or <see cref="AttentionVec"/>'s) with each head's chunk max and exp-sum reduced by
    /// Warp.ShuffleXor (2 group barriers in place of 17; bit-identical, see IGqaReduce). Needs a real 32-lane warp and a head width of 32, 64 or 128. Default: on for CUDA and for WebGPU with subgroups (measured in the constructor); env GGUF_ATTN_SHUFFLE=0 / 1
    /// forces it.
    /// </summary>
    public bool AttentionShuffle { get => _attShuffle; set { if (_attShuffle != value) { _attShuffle = value; _bufGen++; } } }
    /// <summary>
    /// The split-K decode attention's chunk merge on <c>AttnCombineGroupImpl</c> (a group per row and head: the chunk weights found once in shared memory;
    /// bit-identical) instead of <c>AttnCombineImpl</c> (a thread per value, each re-reading every chunk). Head widths a power of two up to 256. Env
    /// GGUF_ATTN_COMBINE_GROUP=0 / 1 forces it. Default: on for CUDA and WebGPU (measured in the constructor).
    /// </summary>
    public bool AttentionCombineGroup { get => _attCombineGroupOn; set { if (_attCombineGroupOn != value) { _attCombineGroupOn = value; _bufGen++; } } }
    private bool _attCombineGroupOn;
    /// <summary>The chunk merge the last fused decode step ran: "group" or "per-value" (null before one).</summary>
    public string? LastAttentionMergeKernel { get; private set; }
    private bool _attShuffle;
    private bool ShuffleAttentionFits(int d) => global::SpawnDev.ILGPU.AcceleratorRequirementsExtensions.SupportsWarpShuffle(_acc, 32) && (d == 32 || d == 64 || d == 128);
    /// <summary>
    /// The grouped-query decode attention on <c>AttnPartialGqaVecImpl</c> (dead chunks leave at once; scores from 16-byte K loads reduced across lanes with
    /// Warp.ShuffleXor). Needs an fp16 cache, a 32-wide warp / subgroup and a head width whose D / 8 is a power of two up to 32. Default: on for WebGPU and CUDA
    /// (CUDA since 2026-11-09, with <see cref="AttentionIlp"/>'s four-ahead P.V; env GGUF_ATTN_VEC=0 / 1 forces it); see the remarks in the kernel for what it replaces.
    /// </summary>
    public bool AttentionVec { get => _attVec; set { if (_attVec != value) { _attVec = value; _bufGen++; } } }
    private bool _attVec;
    // a real 32-lane warp: a WebGPU accelerator without subgroups reports WarpSize 32 but its "warp" is the whole workgroup (SpawnDev.ILGPU 5.4.22)
    private bool VecAttentionFits(int d) => global::SpawnDev.ILGPU.AcceleratorRequirementsExtensions.SupportsWarpShuffle(_acc, 32) && d % 8 == 0 && (d / 8 & (d / 8 - 1)) == 0 && d / 8 <= 32 && d <= GqaMaxD;
    private bool _attIlp;
    /// <summary>The next decode step records its kernels again (WebGPU plans, CUDA graphs): call after changing a static kernel switch (e.g. FusedDequantMatMul.Q8Vec)
    /// that a captured step would otherwise keep replaying.</summary>
    public void InvalidateCapturedSteps() => _bufGen++;
    private int _gqaChunk = int.TryParse(Environment.GetEnvironmentVariable("GGUF_GQA_CHUNK"), out var gc) ? gc : 0;
    private int GqaChunkKeys(int G)
    {
        int gp = 1; while (gp < G) gp <<= 1;
        int max = AttChunk / gp;
        // with AttentionIlp (chunks past the position cost nothing, the merge reads only live chunks) 64-key chunks: MEASURED 2026-11-06 (CUDA, RTX 4070,
        // qwen3:1.7b Q8_0, G = 2): 128 / 64 / 32 keys = 163.1 / 164.8 / 166.8 tok/s at a short prompt, 138.1 / 142.5 / 134.6 at 1,830 tokens
        return _gqaChunk <= 0 ? (_attIlp ? Math.Min(max, 64) : max) : Math.Clamp(_gqaChunk, 16, AttChunk);
    }
    /// <summary>
    /// Load-time option (static: read when a decoder is created; env GGUF_BATCH_SEPARATE=1 turns it off): join q, k (and v) into one matrix and gate and up into
    /// another where their block types agree, and fuse the bias, rotary and cache store into one kernel and SwiGLU into its GEMV's output: about half the kernel
    /// launches a layer (WebGPU pays per dispatch).
    /// </summary>
    public static bool JoinProjections = Environment.GetEnvironmentVariable("GGUF_BATCH_SEPARATE") != "1";
    /// <summary>
    /// WebGL: the forward ends at the logits (no candidate kernels: their gather takes slots from an atomic add, which WebGL cannot return), and a draw reads its row.
    /// </summary>
    public bool NoCandidates { get; }
    /// <summary>Tokens per row the device keeps after each forward (the nucleus is drawn from these unless it is wider).</summary>
    public const int Candidates = SpawnDev.ILGPU.ML.Kernels.TopKKernels.K;
    /// <summary>Draws since <see cref="PrefillAsync"/> that read their full logits row: the nucleus was wider than the candidates, or more than <see cref="SelectCap"/> tokens reached the floor.</summary>
    public int RowReads { get; private set; }
    /// <summary>Most tokens per row the candidate gather keeps (a flatter row is read whole).</summary>
    public const int SelectCap = SpawnDev.ILGPU.ML.Kernels.TopKKernels.SelectCap;

    /// <summary>Sequences live since the last <see cref="Fork"/>.</summary>
    public int Batch { get; private set; } = 1;

    private readonly float _eps, _theta;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _invFreq;
    private readonly KvStore _kv;
    private bool _hasFlash, _hasFlashGqa, _hasFlashMma;
    private MemoryBuffer1D<float, Stride1D.Dense>? _hid, _hid2, _hn, _q, _k, _v, _att, _proj, _gate, _up, _scores, _attPart, _attMl, _qkv, _gateUp, _logits, _maxV, _selV, _cand, _softStats;
    private MemoryBuffer1D<int, Stride1D.Dense>? _maxI, _selI, _selN, _seqBuf, _stepIds, _posBuf;
    // device draws: the token each sequence drew last (the next step's input, by sequence), its copy for the captures' restores, the draw parameters
    // (seed, force the full-row path), top-p, and two read rings (WebGPU keeps one staging buffer per source buffer, so two reads in flight need two sources)
    private MemoryBuffer1D<int, Stride1D.Dense>? _seqIds, _seqIdsSnap, _drawI;
    // the repetition penalty's window: per sequence the tokens it drew, [MaxBatch counts][MaxBatch rings of _repCap], appended by the draw kernel (so it follows
    // pipelined and captured steps without the host); its copy for the captures' restores
    private MemoryBuffer1D<int, Stride1D.Dense>? _rep, _repSnap;
    private int _repCap;
    private readonly MemoryBuffer1D<int, Stride1D.Dense>?[] _ring = new MemoryBuffer1D<int, Stride1D.Dense>?[2];
    private int _ringAt;
    private MemoryBuffer1D<float, Stride1D.Dense>? _drawF;
    private FV _lastLogits, _lastCandView;
    private float[]? _lastCand = Array.Empty<float>();
    private int _lastRows, _lastP;
    private int[] _lastSeqs = Array.Empty<int>();
    private float _lastT;
    private const int Chunk = SpawnDev.ILGPU.ML.Kernels.TopKKernels.GroupChunk;    // 256: the group kernels take a thread per value of a chunk
    /// <summary>
    /// Each step's candidate selection with a group per chunk of logits (<c>TopKKernels.ChunkMaxGroupImpl</c> / <c>SelectGroupImpl</c>: coalesced reads, tree
    /// reductions) and a group per row for the floor (<c>FloorGroupImpl</c>: the chunk maxima ranked from shared memory) instead of a thread per chunk. The same candidates; the chunks' softmax mass summed pairwise. Env GGUF_TOPK_GROUP=0 / 1
    /// forces it; default on for CUDA and WebGPU (<see cref="TopKGroupDefault"/>, also <see cref="DrawRowsAsync"/>'s).
    /// </summary>
    public bool TopKGroup { get => _topKGroup; set { if (_topKGroup != value) { _topKGroup = value; _bufGen++; } } }
    private bool _topKGroup;
    /// <summary>
    /// <see cref="TopKGroup"/>'s default on an accelerator: env GGUF_TOPK_GROUP=0 / 1, else on for CUDA. MEASURED 2026-11-09 (RTX 4070, qwen3:1.7b
    /// Q4_K_M, 151,936 logits, nsys per step): ChunkMax 61.6 -> 4.3 us, Floor 28.8 -> 8.4, Select 76.3 -> 4.3 (Final 2.8 either way); decode (quiet,
    /// interleaved x2, greedy and sampled alike) 215.6 / 215.8 -> 222.5 / 222.7 tok/s short, 202.5 / 202.7 -> 208.2 / 209.0 long. WebGPU since
    /// 2026-11-09 (Chrome, Qwen3-0.6B Q8_0, the same 151,936 logits, GPU timestamps): ChunkMax + Floor + Select 0.091 -> 0.029 ms a step.
    /// </summary>
    public static bool TopKGroupDefault(Accelerator accelerator) => Environment.GetEnvironmentVariable("GGUF_TOPK_GROUP") switch
    {
        "0" => false, "1" => true, _ => accelerator.AcceleratorType is AcceleratorType.Cuda or AcceleratorType.WebGPU,
    };
    private static bool TopKGroupFits(Accelerator a) => a.AcceleratorType != AcceleratorType.WebGL && a.MaxNumThreadsPerGroup >= Chunk;
    private int Chunks => (Vocab + Chunk - 1) / Chunk;
    private int CandStride => 2 * Candidates + 1 + Chunks;      // per row: values, ids (as floats), tokens at the floor, mass partial sums

    private GgufBatchDecoder(Accelerator accelerator, GGUFModel model, Stream stream, int maxSeq, int maxBatch)
    {
        _acc = accelerator; _model = model; _stream = stream; _mm = new FusedDequantMatMul(accelerator); _norm = new NormalizationKernels(accelerator);
        string a = model.Architecture;
        if (a is not ("qwen2" or "qwen3" or "llama"))
            throw new NotSupportedException($"GgufBatchDecoder: architecture '{a}' (supported: qwen2, qwen3, llama)");
        Hidden = (int)model.EmbeddingLength; Layers = (int)model.BlockCount; Heads = (int)model.AttentionHeadCount; KvHeads = (int)model.AttentionHeadCountKV;
        // the head width is its own metadata (qwen3: heads x head width need not be the model width)
        HeadDim = (int)model.GetMetadataInt($"{a}.attention.key_length", Hidden / Heads); FfnDim = (int)model.GetMetadataInt($"{a}.feed_forward_length", 0);
        QDim = Heads * HeadDim;
        // llama's GGUF q/k are permuted for rotary on adjacent pairs; qwen's rotate split halves (GGUFGraphBuilder.UsesNormRope)
        _normRope = SpawnDev.ILGPU.ML.GGUF.GGUFGraphBuilder.UsesNormRope(a);
        _qkNorm = model.Tensors.Any(t => t.Name == "blk.0.attn_q_norm.weight");
        if ((int)model.GetMetadataInt($"{a}.rope.dimension_count", HeadDim) != HeadDim)
            throw new NotSupportedException($"GgufBatchDecoder: rotary over part of the head ({a}.rope.dimension_count)");
        if (!RopeFactorsFit(model, HeadDim, out var rfWhy)) throw new NotSupportedException("GgufBatchDecoder: " + rfWhy);
        _eps = model.GetMetadataFloat($"{a}.attention.layer_norm_rms_epsilon", 1e-6f);
        _theta = model.GetMetadataFloat($"{a}.rope.freq_base", 10000f);
        _embd = Tensor("token_embd.weight");
        Vocab = (int)_embd.Dimensions[1];
        if (Vocab >= 1 << 24) throw new NotSupportedException($"vocabulary {Vocab}: candidate ids travel as floats (exact below 2^24)");
        _head = model.Tensors.Any(t => t.Name == "output.weight") ? "output.weight" : "token_embd.weight";
        MaxSeq = maxSeq; MaxBatch = maxBatch;
        CaptureWebGPU = accelerator is SpawnDev.ILGPU.WebGPU.WebGPUAccelerator;
        // WebGPU: on where the vector kernel runs (subgroups), for its four-ahead P.V and 64-key chunks; the ILP kernel itself stays off there (it lost the
        // 2026-11-05 A/B to the plain one, which a browser without subgroups keeps). MEASURED 2026-11-09 (Chrome, qwen3 1.7B Q8_0, vector kernel, attention GPU
        // ms a token short / long): 0.509 / 0.807 -> 0.420 / 0.617, + shuffle 0.401 / ~0.60; decode 170.2 / 154.9 -> 170.7 / 157.5 tok/s.
        bool webGpuVec = accelerator.AcceleratorType is AcceleratorType.WebGPU && global::SpawnDev.ILGPU.AcceleratorRequirementsExtensions.SupportsWarpShuffle(accelerator, 32);
        _attIlp = Environment.GetEnvironmentVariable("GGUF_ATTN_ILP") switch { "0" => false, "1" => true, _ => accelerator.AcceleratorType is AcceleratorType.Cuda or AcceleratorType.OpenCL || webGpuVec };
        // CUDA: the vector kernel with four-ahead P.V and shuffle reductions. MEASURED 2026-11-09 (RTX 4070, qwen3:1.7b Q4_K_M, nsys per call, short / long
        // prompt): ILP kernel 16.97 / 38.52 us, + shuffle 16.79 / 38.55, vector + ILP P.V 15.66 / 23.42, + shuffle 15.17 / 23.01; decode (quiet, interleaved
        // x2) 212.9 / 211.9 -> 215.3 / 212.9 tok/s short, 174.2 / 174.4 -> 188.8 / 188.5 long. Bit-identical per GgufBatchDecoder_ShuffleAttention_BitIdentical.
        _attShuffle = Environment.GetEnvironmentVariable("GGUF_ATTN_SHUFFLE") switch { "0" => false, "1" => true, _ => accelerator.AcceleratorType is AcceleratorType.Cuda || webGpuVec };
        // MEASURED 2026-11-09 (RTX 4070, qwen3:1.7b Q4_K_M, 64-key chunks, nsys per call short / long prompt): per-value merge 2.62 / 15.97 us, group 2.00 /
        // 3.07 (at 128-key chunks 2.14 / 9.79 vs 1.95 / 2.41, but the attention kernel itself is 24 / 34 us there); decode (quiet, interleaved x2) 214.5 / 212.5
        // -> 213.0 / 211.4 tok/s short (noise), 186.4 / 186.0 -> 200.2 / 199.6 long (+7.4%).
        _topKGroup = TopKGroupDefault(accelerator);
        // WebGPU since 2026-11-09 (Chrome, Qwen3-0.6B Q8_0, GPU timestamps, interleaved x2): merge 0.275 -> 0.136 ms a token at 1,826 tokens, 0.075 -> 0.080 at 25;
        // the step's timestamped total 3.39 -> 3.26 ms long, 2.99 -> 3.00 short
        _attCombineGroupOn = Environment.GetEnvironmentVariable("GGUF_ATTN_COMBINE_GROUP") switch { "0" => false, "1" => true, _ => accelerator.AcceleratorType is AcceleratorType.Cuda or AcceleratorType.WebGPU };
        _attVec = Environment.GetEnvironmentVariable("GGUF_ATTN_VEC") switch { "0" => false, "1" => true, _ => accelerator.AcceleratorType is AcceleratorType.WebGPU or AcceleratorType.Cuda };
        CaptureCuda = accelerator is global::ILGPU.Runtime.Cuda.CudaAccelerator && global::ILGPU.Runtime.Cuda.CudaStream.SupportsGraphCapture
            && Environment.GetEnvironmentVariable("GGUF_BATCH_CUDA_GRAPH") != "0";
        DeviceSampling = SupportsDeviceDraws(accelerator);
        NoCandidates = accelerator.AcceleratorType == AcceleratorType.WebGL;
        // the fused attention runs a group per head (a power of two up to 256 threads; WebGL has no groups)
        FusedAttention = accelerator.AcceleratorType != AcceleratorType.WebGL && HeadDim <= AttChunk && (HeadDim & (HeadDim - 1)) == 0 && HeadDim <= accelerator.MaxNumThreadsPerGroup;
        var inv = new float[HeadDim / 2];
        for (int k = 0; k < inv.Length; k++) inv[k] = 1f / MathF.Pow(_theta, 2f * k / HeadDim);
        _invFreq = Own(_acc.Allocate1D(inv));
        int kvC = KvHeads * HeadDim;
        long kvElems = (long)maxBatch * maxSeq * kvC;
        _kv = Own(HalfKvCache ? new KvStore<global::ILGPU.Half>(_acc, Layers, kvElems) : (KvStore)new KvStore<float>(_acc, Layers, kvElems));
        KvCacheBytes = _kv.Bytes;
    }

    private T Own<T>(T d) where T : IDisposable { _owned.Add(d); return d; }
    private GGUFTensorInfo Tensor(string name) => _model.Tensors.FirstOrDefault(t => t.Name == name) ?? throw new KeyNotFoundException($"GGUF tensor '{name}' not found");

    /// <summary>
    /// Load-time option (static: read when a decoder is created; env GGUF_BATCH_KV_F32=1 turns it off): the K / V caches in 16-bit float (half the memory and
    /// the bytes each step's attention reads, llama.cpp's default) instead of 32-bit. They take layers x 2 x maxBatch x maxSeq x kvHeads x headDim values:
    /// llama3.1:8b at maxSeq 8192 is 2.1 GB in F32, 1.1 GB in F16 (Data 2026-11-02: an RTX 2060 ran 17.7 tok/s with 4.9 GB of weights next to the F32 cache).
    /// The attention accumulates in float either way.
    /// </summary>
    public static bool HalfKvCache = Environment.GetEnvironmentVariable("GGUF_BATCH_KV_F32") != "1";

    /// <summary>This decoder's K / V caches are F16 (<see cref="HalfKvCache"/> when it was created).</summary>
    public bool KvCacheIsHalf => _kv.IsHalf;
    /// <summary>The K / V caches' device memory (bytes), F16 or F32 per <see cref="HalfKvCache"/> when the decoder was created.</summary>
    public long KvCacheBytes { get; }

    // the K / V caches and the kernels that read or write them, for one element type (float or half): the kernels are generic over it (PrecisionConvert)
    private abstract class KvStore : IDisposable
    {
        public abstract long Bytes { get; }
        public abstract void QkvPost(int l, int n, FV qkv, FV vb, FV bias, FV invFreq, FV qOut, IV seqs, IV pos, int heads, int D, int sa, int flags, int maxSeq);
        public abstract void Store(int l, int n, FV k, FV v, IV seqs, IV pos, int kvC, int maxSeq, int mode);
        public abstract void Flash(int l, KernelConfig cfg, FV q, FV att, int R, int H, int kvH, int D, int p, float scale);
        public abstract void FlashGqa(int l, KernelConfig cfg, FV q, FV att, int R, int H, int kvH, int D, int p, float scale);
        /// <summary>The tensor-core flash prefill (<see cref="MmaFlashPrefill"/>; F16 caches only, head dim 128): false when this store cannot take it.</summary>
        public abstract bool FlashMma(int l, FV q, FV att, int R, int H, int kvH, int p);
        public abstract void AttPartial(int l, KernelConfig cfg, FV q, IV seqs, IV pos, FV part, FV ml, int H, int kvH, int maxSeq, int mode, int nch, float scale);
        public abstract void AttPartialGqa(int l, KernelConfig cfg, FV q, IV seqs, IV pos, FV part, FV ml, int H, int kvH, int maxSeq, int mode, int nch, int ck, float scale, bool ilp, bool shuffle);
        public abstract bool IsHalf { get; }
        public abstract void AttPartialGqaVec(int l, KernelConfig cfg, FV q, IV seqs, IV pos, FV part, FV ml, int H, int kvH, int maxSeq, int mode, int nch, int ck, float scale, bool shuffle, bool ilp);
        public abstract void Scores(int l, int n, FV q, IV seqs, FV scores, int R, int nk, int H, int D, int kvH, int maxSeq, int p, int mode, float scale);
        public abstract void Pv(int l, int n, FV scores, IV seqs, FV o, int R, int nk, int H, int D, int kvH, int maxSeq, int mode);
        public abstract void Fork(int l, int n, long count, long region);
        public abstract void Dispose();
    }
    private sealed class KvStore<T> : KvStore where T : unmanaged, System.Numerics.INumber<T>
    {
        private readonly Accelerator _a;
        private readonly MemoryBuffer1D<T, Stride1D.Dense>[] _k, _v;
        private Action<Index1D, FV, FV, FV, FV, FV, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, IV, IV, int, int, int, int, int>? _qkvPost;
        private Action<Index1D, FV, FV, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, IV, IV, int, int, int>? _store;
        private Action<KernelConfig, FV, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, FV, int, int, int, int, int, float>? _flash, _flashGqa;
        private Action<KernelConfig, FV, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, IV, IV, FV, FV, int, int, int, int, int, float>? _att;
        private readonly Action<KernelConfig, FV, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, IV, IV, FV, FV, int, int, int, int, int, int, float>?[] _attGqa =
            new Action<KernelConfig, FV, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, IV, IV, FV, FV, int, int, int, int, int, int, float>?[GqaMaxG + 1];
        private readonly Action<KernelConfig, FV, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, IV, IV, FV, FV, int, int, int, int, int, int, float>?[] _attGqaIlp =
            new Action<KernelConfig, FV, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, IV, IV, FV, FV, int, int, int, int, int, int, float>?[GqaMaxG + 1];
        private readonly Action<KernelConfig, FV, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, IV, IV, FV, FV, int, int, int, int, int, int, float>?[] _attGqaIlpShuffle =
            new Action<KernelConfig, FV, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, IV, IV, FV, FV, int, int, int, int, int, int, float>?[GqaMaxG + 1];
        // [variant][G], variant = (shuffle ? 2 : 0) | (ilp ? 1 : 0)
        private readonly Action<KernelConfig, FV, ArrayView<FusedDequantMatMul.W16>, ArrayView<global::ILGPU.Half>, IV, IV, FV, FV, int, int, int, int, int, int, float>?[][] _attGqaVec = [new Action<KernelConfig, FV, ArrayView<FusedDequantMatMul.W16>, ArrayView<global::ILGPU.Half>, IV, IV, FV, FV, int, int, int, int, int, int, float>?[GqaMaxG + 1], new Action<KernelConfig, FV, ArrayView<FusedDequantMatMul.W16>, ArrayView<global::ILGPU.Half>, IV, IV, FV, FV, int, int, int, int, int, int, float>?[GqaMaxG + 1], new Action<KernelConfig, FV, ArrayView<FusedDequantMatMul.W16>, ArrayView<global::ILGPU.Half>, IV, IV, FV, FV, int, int, int, int, int, int, float>?[GqaMaxG + 1], new Action<KernelConfig, FV, ArrayView<FusedDequantMatMul.W16>, ArrayView<global::ILGPU.Half>, IV, IV, FV, FV, int, int, int, int, int, int, float>?[GqaMaxG + 1]];
        private Action<KernelConfig, FV, ArrayView<FusedDequantMatMul.W16>, ArrayView<global::ILGPU.Half>, IV, IV, FV, FV, int, int, int, int, int, int, float> LoadGqaVec<TG>(int variant) where TG : struct, IGqaGroup => variant switch
        {
            0 => _a.LoadStreamKernel<FV, ArrayView<FusedDequantMatMul.W16>, ArrayView<global::ILGPU.Half>, IV, IV, FV, FV, int, int, int, int, int, int, float>(AttnPartialGqaVecImpl<TG, GqaReduceTree, GqaPvPlain>),
            1 => _a.LoadStreamKernel<FV, ArrayView<FusedDequantMatMul.W16>, ArrayView<global::ILGPU.Half>, IV, IV, FV, FV, int, int, int, int, int, int, float>(AttnPartialGqaVecImpl<TG, GqaReduceTree, GqaPvIlp>),
            2 => _a.LoadStreamKernel<FV, ArrayView<FusedDequantMatMul.W16>, ArrayView<global::ILGPU.Half>, IV, IV, FV, FV, int, int, int, int, int, int, float>(AttnPartialGqaVecImpl<TG, GqaReduceShuffle, GqaPvPlain>),
            _ => _a.LoadStreamKernel<FV, ArrayView<FusedDequantMatMul.W16>, ArrayView<global::ILGPU.Half>, IV, IV, FV, FV, int, int, int, int, int, int, float>(AttnPartialGqaVecImpl<TG, GqaReduceShuffle, GqaPvIlp>),
        };
        /// <summary>The fp16 cache can take AttnPartialGqaVecImpl (16-byte K loads).</summary>
        public override bool IsHalf => typeof(T) == typeof(global::ILGPU.Half);
        public override void AttPartialGqaVec(int l, KernelConfig cfg, FV q, IV seqs, IV pos, FV part, FV ml, int H, int kvH, int maxSeq, int mode, int nch, int ck, float scale, bool shuffle, bool ilp)
        {
            if (!IsHalf) throw new InvalidOperationException("the vector decode attention reads an fp16 cache");
            int variant = (shuffle ? 2 : 0) | (ilp ? 1 : 0);
            (_attGqaVec[variant][H / kvH] ??= (H / kvH) switch
            {
                2 => LoadGqaVec<Gqa2>(variant), 3 => LoadGqaVec<Gqa3>(variant), 4 => LoadGqaVec<Gqa4>(variant), 5 => LoadGqaVec<Gqa5>(variant), 6 => LoadGqaVec<Gqa6>(variant), 7 => LoadGqaVec<Gqa7>(variant), 8 => LoadGqaVec<Gqa8>(variant),
                _ => throw new ArgumentOutOfRangeException(nameof(H), $"grouped-query attention takes 2..{GqaMaxG} query heads a kv head, not {H / kvH}"),
            })(cfg, q, _k[l].View.BaseView.Cast<FusedDequantMatMul.W16>(), _v[l].View.BaseView.Cast<global::ILGPU.Half>(), seqs, pos, part, ml, H, kvH, maxSeq, mode, nch, ck, scale);
        }
        private Action<KernelConfig, FV, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, IV, IV, FV, FV, int, int, int, int, int, int, float> LoadGqa<TG>(bool ilp, bool shuffle) where TG : struct, IGqaGroup =>
            shuffle ? _a.LoadStreamKernel<FV, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, IV, IV, FV, FV, int, int, int, int, int, int, float>(AttnPartialGqaIlpImpl<T, TG, GqaReduceShuffle>)
            : ilp ? _a.LoadStreamKernel<FV, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, IV, IV, FV, FV, int, int, int, int, int, int, float>(AttnPartialGqaIlpImpl<T, TG, GqaReduceTree>)
                : _a.LoadStreamKernel<FV, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, IV, IV, FV, FV, int, int, int, int, int, int, float>(AttnPartialGqaImpl<T, TG>);
        private Action<Index1D, FV, ArrayView1D<T, Stride1D.Dense>, IV, FV, int, int, int, int, int, int, int, int, float>? _scores;
        private Action<Index1D, FV, ArrayView1D<T, Stride1D.Dense>, IV, FV, int, int, int, int, int, int, int>? _pv;
        private Action<Index1D, ArrayView1D<T, Stride1D.Dense>, long, long>? _fork;
        public KvStore(Accelerator a, int layers, long elems)
        {
            _a = a;
            _k = new MemoryBuffer1D<T, Stride1D.Dense>[layers]; _v = new MemoryBuffer1D<T, Stride1D.Dense>[layers];
            try { for (int l = 0; l < layers; l++) { _k[l] = a.Allocate1D<T>(elems); _v[l] = a.Allocate1D<T>(elems); } }
            catch { Dispose(); throw; }
            Bytes = 2L * layers * elems * System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        }
        public override long Bytes { get; }
        public override void QkvPost(int l, int n, FV qkv, FV vb, FV bias, FV invFreq, FV qOut, IV seqs, IV pos, int heads, int D, int sa, int flags, int maxSeq) =>
            (_qkvPost ??= _a.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, FV, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, IV, IV, int, int, int, int, int>(QkvPostImpl<T>))(
                n, qkv, vb, bias, invFreq, qOut, _k[l].View, _v[l].View, seqs, pos, heads, D, sa, flags, maxSeq);
        public override void Store(int l, int n, FV k, FV v, IV seqs, IV pos, int kvC, int maxSeq, int mode) =>
            (_store ??= _a.LoadAutoGroupedStreamKernel<Index1D, FV, FV, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, IV, IV, int, int, int>(StoreImpl<T>))(
                n, k, v, _k[l].View, _v[l].View, seqs, pos, kvC, maxSeq, mode);
        public override void Flash(int l, KernelConfig cfg, FV q, FV att, int R, int H, int kvH, int D, int p, float scale) =>
            (_flash ??= _a.LoadStreamKernel<FV, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, FV, int, int, int, int, int, float>(FlashPrefillImpl<T>))(
                cfg, q, _k[l].View, _v[l].View, att, R, H, kvH, D, p, scale);
        private Action<KernelConfig, FV, ArrayView<uint>, ArrayView<uint>, FV, int, int, int, int>? _flashMma, _flashMmaSm75;
        public override bool FlashMma(int l, FV q, FV att, int R, int H, int kvH, int p)
        {
            if (typeof(T) != typeof(global::ILGPU.Half)) return false;
            // sm_75 (Turing) on the m16n8k8 variant; FusedDequantMatMul.ForceSm75Mma runs it on a newer card (the tests)
            bool turing = !CudaCaps.TensorCores(_a) || FusedDequantMatMul.ForceSm75Mma;
            (turing
                ? _flashMmaSm75 ??= _a.LoadStreamKernel<FV, ArrayView<uint>, ArrayView<uint>, FV, int, int, int, int>(MmaFlashPrefill.KernelLdmSm75)
                : _flashMma ??= _a.LoadStreamKernel<FV, ArrayView<uint>, ArrayView<uint>, FV, int, int, int, int>(MmaFlashPrefill.KernelLdm))(
                new KernelConfig(((R + MmaFlashPrefill.QTile - 1) / MmaFlashPrefill.QTile) * H, MmaFlashPrefill.Threads),
                q, _k[l].View.BaseView.Cast<uint>(), _v[l].View.BaseView.Cast<uint>(), att, R, H, kvH, p);
            return true;
        }
        public override void FlashGqa(int l, KernelConfig cfg, FV q, FV att, int R, int H, int kvH, int D, int p, float scale) =>
            (_flashGqa ??= _a.LoadStreamKernel<FV, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, FV, int, int, int, int, int, float>(FlashPrefillGqaImpl<T>))(
                cfg, q, _k[l].View, _v[l].View, att, R, H, kvH, D, p, scale);
        public override void AttPartial(int l, KernelConfig cfg, FV q, IV seqs, IV pos, FV part, FV ml, int H, int kvH, int maxSeq, int mode, int nch, float scale) =>
            (_att ??= _a.LoadStreamKernel<FV, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, IV, IV, FV, FV, int, int, int, int, int, float>(AttnPartialImpl<T>))(
                cfg, q, _k[l].View, _v[l].View, seqs, pos, part, ml, H, kvH, maxSeq, mode, nch, scale);
        // shuffle runs the ILP kernel (the caller passes it only with ilp)
        public override void AttPartialGqa(int l, KernelConfig cfg, FV q, IV seqs, IV pos, FV part, FV ml, int H, int kvH, int maxSeq, int mode, int nch, int ck, float scale, bool ilp, bool shuffle) =>
            ((shuffle ? _attGqaIlpShuffle : ilp ? _attGqaIlp : _attGqa)[H / kvH] ??= (H / kvH) switch
            {
                2 => LoadGqa<Gqa2>(ilp, shuffle), 3 => LoadGqa<Gqa3>(ilp, shuffle), 4 => LoadGqa<Gqa4>(ilp, shuffle), 5 => LoadGqa<Gqa5>(ilp, shuffle), 6 => LoadGqa<Gqa6>(ilp, shuffle), 7 => LoadGqa<Gqa7>(ilp, shuffle), 8 => LoadGqa<Gqa8>(ilp, shuffle),
                _ => throw new ArgumentOutOfRangeException(nameof(H), $"grouped-query attention takes 2..{GqaMaxG} query heads a kv head, not {H / kvH}"),
            })(cfg, q, _k[l].View, _v[l].View, seqs, pos, part, ml, H, kvH, maxSeq, mode, nch, ck, scale);
        public override void Scores(int l, int n, FV q, IV seqs, FV scores, int R, int nk, int H, int D, int kvH, int maxSeq, int p, int mode, float scale) =>
            (_scores ??= _a.LoadAutoGroupedStreamKernel<Index1D, FV, ArrayView1D<T, Stride1D.Dense>, IV, FV, int, int, int, int, int, int, int, int, float>(ScoresImpl<T>))(
                n, q, _k[l].View, seqs, scores, R, nk, H, D, kvH, maxSeq, p, mode, scale);
        public override void Pv(int l, int n, FV scores, IV seqs, FV o, int R, int nk, int H, int D, int kvH, int maxSeq, int mode) =>
            (_pv ??= _a.LoadAutoGroupedStreamKernel<Index1D, FV, ArrayView1D<T, Stride1D.Dense>, IV, FV, int, int, int, int, int, int, int>(PvImpl<T>))(
                n, scores, _v[l].View, seqs, o, R, nk, H, D, kvH, maxSeq, mode);
        public override void Fork(int l, int n, long count, long region)
        {
            _fork ??= _a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<T, Stride1D.Dense>, long, long>(ForkImpl<T>);
            _fork(n, _k[l].View, count, region);
            _fork(n, _v[l].View, count, region);
        }
        public override void Dispose()
        {
            foreach (var b in _k) b?.Dispose();
            foreach (var b in _v) b?.Dispose();
        }
    }

    /// <summary>
    /// Load a GGUF from a seekable stream (kept open for the embedding rows; disposed with the decoder). <paramref name="maxSeq"/>: prompt plus generation;
    /// <paramref name="maxBatch"/>: most sequences at once (the caches take layers x 2 x maxBatch x maxSeq x kvHeads x headDim floats).
    /// </summary>
    public static Task<GgufBatchDecoder> CreateAsync(Accelerator accelerator, Stream gguf, int maxSeq = 2048, int maxBatch = 8, CancellationToken ct = default)
        => CreateAsync(accelerator, gguf, maxSeq, maxBatch, null, ct);

    /// <summary><see cref="CreateAsync(Accelerator, Stream, int, int, CancellationToken)"/> with load progress: <paramref name="onProgress"/>("parse" | "weights", percent).</summary>
    public static async Task<GgufBatchDecoder> CreateAsync(Accelerator accelerator, Stream gguf, int maxSeq, int maxBatch, Action<string, int>? onProgress, CancellationToken ct = default)
    {
        if (!gguf.CanSeek) throw new ArgumentException("the GGUF stream must be seekable");
        gguf.Seek(0, SeekOrigin.Begin);
        onProgress?.Invoke("parse", 0);
        var model = await GGUFParser.ParseHeaderAsync(gguf, ct).ConfigureAwait(false);
        onProgress?.Invoke("parse", 100);
        return await CreateAsync(accelerator, gguf, model, maxSeq, maxBatch, onProgress, ct).ConfigureAwait(false);
    }

    /// <summary>Load from a stream whose header was already parsed into <paramref name="model"/> (the bytes are not parsed twice).</summary>
    public static async Task<GgufBatchDecoder> CreateAsync(Accelerator accelerator, Stream gguf, GGUFModel model, int maxSeq, int maxBatch, Action<string, int>? onProgress = null,
        CancellationToken ct = default)
    {
        if (!gguf.CanSeek) throw new ArgumentException("the GGUF stream must be seekable");
        model.SourceStream = gguf;
        var d = new GgufBatchDecoder(accelerator, model, gguf, maxSeq, maxBatch) { _progress = onProgress };
        foreach (var t in model.Tensors) d._totalBytes += GGMLTypes.TypeSize(t.Type, model.GetTensorElementCount(t));
        try { await d.LoadAsync(ct).ConfigureAwait(false); }
        catch { d.Dispose(); throw; }
        onProgress?.Invoke("weights", 100);
        return d;
    }

    /// <summary>How 16-bit float (BF16 / F16) matrices load: as stored, or converted to Q8_0 (8 bits + one fp16 scale per 32 weights, ggml's block) as they stream in.</summary>
    public enum Float16Matrices
    {
        /// <summary>Keep them 16-bit unless they do not fit: when the matrices would take more than <see cref="AutoQ8Fraction"/> of the device's memory, Q8_0 (CUDA and OpenCL, which report
        /// their real memory; the browser backends keep them 16-bit unless <see cref="Q8"/> is asked for).</summary>
        Auto,
        /// <summary>Always as stored.</summary>
        Keep,
        /// <summary>Always Q8_0 (half the memory, the Q8 GEMV path; not bit-identical to the 16-bit model).</summary>
        Q8,
    }

    /// <summary>
    /// The load-time choice for 16-bit matrices (a safetensors model seen through <see cref="GGUF.SafeTensorsGguf"/>: ComfyUI's qwen_3_4b.safetensors is 8.05 GB of bf16, more
    /// than an 8 GB card holds). Env ML_GGUF_F16_MATRICES=keep / q8 / auto. The decoder reports what it did in <see cref="ConvertedToQ8"/>.
    /// </summary>
    public static Float16Matrices Float16MatrixLoad { get; set; } = Environment.GetEnvironmentVariable("ML_GGUF_F16_MATRICES")?.ToLowerInvariant() switch
    {
        "keep" => Float16Matrices.Keep, "q8" => Float16Matrices.Q8, _ => Float16Matrices.Auto,
    };
    /// <summary>The share of the device's memory 16-bit matrices may take before <see cref="Float16Matrices.Auto"/> converts them (the rest is for the KV cache and activations).</summary>
    public static double AutoQ8Fraction { get; set; } = 0.75;
    /// <summary>True when this decoder converted its 16-bit matrices to Q8_0 at load (<see cref="Float16MatrixLoad"/>).</summary>
    public bool ConvertedToQ8 { get; private set; }

    // the type a matrix is held in on the device: its own, or Q8_0 when 16-bit matrices are converted (whole 32-value blocks along each row)
    private GGMLType LoadType(GGUFTensorInfo t) =>
        ConvertedToQ8 && t.Type is GGMLType.BF16 or GGMLType.F16 && t.Dimensions.Length == 2 && t.Dimensions[0] % 32 == 0 ? GGMLType.Q8_0 : t.Type;

    private void ChooseFloat16Load(IEnumerable<string> matrices)
    {
        if (Float16MatrixLoad == Float16Matrices.Keep) return;
        long bytes = matrices.Distinct().Select(Tensor).Where(t => t.Type is GGMLType.BF16 or GGMLType.F16).Sum(t => GGMLTypes.TypeSize(t.Type, _model.GetTensorElementCount(t)));
        if (bytes == 0) return;
        // Auto trusts only a device that reports its real memory (CUDA, OpenCL): the browser backends report an estimate or a buffer limit (WebGL: a fixed 256 MB),
        // which converted a 0.5B model's 16-bit matrices on WebGL
        bool realMemory = _acc.AcceleratorType is AcceleratorType.Cuda or AcceleratorType.OpenCL;
        ConvertedToQ8 = Float16MatrixLoad == Float16Matrices.Q8 || (realMemory && bytes > AutoQ8Fraction * _acc.MemorySize);
        if (ConvertedToQ8) Console.WriteLine($"[GgufBatchDecoder] 16-bit matrices ({bytes / 1e9:F2} GB) load as Q8_0 ({(Float16MatrixLoad == Float16Matrices.Q8 ? "requested" : $"more than {AutoQ8Fraction:P0} of the device's {_acc.MemorySize / 1e9:F2} GB")})");
    }

    // ggml's block_q8_0 of 16-bit values: per 32 values, d = max|x| / 127 (stored fp16) then round(x / d) as int8 (half away from zero, as roundf)
    private static byte[] QuantizeQ8_0(byte[] raw, GGMLType type, long count)
    {
        long blocks = count / 32;
        var o = new byte[(blocks * 34 + 3) / 4 * 4];
        Parallel.For(0L, blocks, b =>
        {
            Span<float> v = stackalloc float[32];
            float amax = 0f;
            for (int i = 0; i < 32; i++)
            {
                ushort bits = BitConverter.ToUInt16(raw, (int)((b * 32 + i) * 2));
                v[i] = type == GGMLType.BF16 ? BitConverter.Int32BitsToSingle(bits << 16) : (float)BitConverter.UInt16BitsToHalf(bits);
                amax = Math.Max(amax, Math.Abs(v[i]));
            }
            float d = amax / 127f, id = d != 0f ? 1f / d : 0f;
            long at = b * 34;
            BitConverter.TryWriteBytes(o.AsSpan((int)at, 2), BitConverter.HalfToUInt16Bits((System.Half)d));
            for (int i = 0; i < 32; i++) o[at + 2 + i] = (byte)(sbyte)MathF.Round(v[i] * id, MidpointRounding.AwayFromZero);
        });
        return o;
    }

    // a matrix's bytes as held on the device (LoadType): the stored bytes, or their Q8_0 conversion
    private async Task<byte[]> ReadLoadedAsync(GGUFTensorInfo t, CancellationToken ct)
    {
        var raw = await ReadAsync(t, ct).ConfigureAwait(false);
        return LoadType(t) == t.Type ? raw : QuantizeQ8_0(raw, t.Type, _model.GetTensorElementCount(t));
    }

    // load progress: bytes read over the file's tensor bytes
    private Action<string, int>? _progress;
    private long _readBytes, _totalBytes;
    private int _lastPct = -1;

    private async Task<byte[]> ReadAsync(GGUFTensorInfo t, CancellationToken ct)
    {
        long n = GGMLTypes.TypeSize(t.Type, _model.GetTensorElementCount(t));
        var buf = new byte[(n + 3) / 4 * 4];                                      // the GEMM reads the blocks as int words: pad to 4
        _stream.Seek(_model.GetTensorDataOffset(t), SeekOrigin.Begin);
        int got = 0;
        while (got < n) { int r = await _stream.ReadAsync(buf.AsMemory(got, (int)(n - got)), ct).ConfigureAwait(false); if (r == 0) throw new EndOfStreamException(t.Name); got += r; }
        Progress(n);
        return buf;
    }

    // a tensor's bytes into dst (at least its size, padded to 4) straight from the stream: SpawnDev.ILGPU's stream copy takes a browser JS stream JS -> GPU
    // (queue.writeBuffer) without the bytes entering the .NET heap (the weight census: a chat model's matrices used to arrive as managed arrays, ~145 MB for
    // Qwen2.5-0.5B's embedding alone). writeBuffer moves 4-byte words: the up to 3 trailing bytes of a size that is not a multiple of 4 go through one word.
    private async Task UploadAsync(GGUFTensorInfo t, ArrayView1D<byte, Stride1D.Dense> dst, CancellationToken ct)
    {
        if (LoadType(t) != t.Type)
        {
            var q = await ReadLoadedAsync(t, ct).ConfigureAwait(false);                     // converted on the host as it streams in
            dst.SubView(0, q.Length).CopyFromCPU(q);
            return;
        }
        long n = GGMLTypes.TypeSize(t.Type, _model.GetTensorElementCount(t));
        _stream.Seek(_model.GetTensorDataOffset(t), SeekOrigin.Begin);
        long aligned = n & ~3L;
        if (aligned > 0) await dst.SubView(0, aligned).CopyFromStreamAsync(_stream, cancellationToken: ct).ConfigureAwait(false);
        if (n > aligned)
        {
            var w = new byte[4];
            int got = 0;
            while (got < n - aligned) { int r = await _stream.ReadAsync(w.AsMemory(got, (int)(n - aligned - got)), ct).ConfigureAwait(false); if (r == 0) throw new EndOfStreamException(t.Name); got += r; }
            dst.SubView(aligned, 4).CopyFromCPU(w);
        }
        Progress(n);
    }

    private void Progress(long bytes)
    {
        if (_progress == null || _totalBytes <= 0) return;
        _readBytes += bytes;
        int pct = (int)Math.Min(99, _readBytes * 100 / _totalBytes);
        if (pct != _lastPct) { _lastPct = pct; _progress("weights", pct); }
    }

    // a tensor in its own device buffer (padded to 4 bytes: the GEMM reads the blocks as int words)
    private async Task<MemoryBuffer1D<byte, Stride1D.Dense>> UploadAsync(GGUFTensorInfo t, CancellationToken ct)
    {
        long n = GGMLTypes.TypeSize(LoadType(t), _model.GetTensorElementCount(t));
        var buf = _acc.Allocate1D<byte>((n + 3) / 4 * 4);
        try { await UploadAsync(t, buf.View, ct).ConfigureAwait(false); }
        catch { buf.Dispose(); throw; }
        return buf;
    }

    // rope_freqs (llama 3.x): per-frequency factors that divide each rotary pair's angle, ggml's theta / freq_factors[i] (the session engine's RoPE kernel)
    private static bool RopeFactorsFit(GGUFModel model, int headDim, out string reason)
    {
        reason = "";
        var rf = model.Tensors.FirstOrDefault(t => t.Name == "rope_freqs.weight");
        if (rf == null) return true;
        if (rf.Type != GGMLType.F32) { reason = $"rope_freqs.weight: {rf.Type} (expected F32)"; return false; }
        if (model.GetTensorElementCount(rf) != headDim / 2) { reason = $"rope_freqs.weight: {model.GetTensorElementCount(rf)} factors (expected one per rotary pair, {headDim / 2})"; return false; }
        return true;
    }

    private async Task LoadAsync(CancellationToken ct)
    {
        if (_model.Tensors.Any(t => t.Name == "rope_freqs.weight"))
        {
            // the rotary table divided by llama 3's per-frequency factors: angle = pos * invFreq[k] / factor[k]
            var factors = await VecAsync("rope_freqs.weight", ct).ConfigureAwait(false);
            var inv = new float[HeadDim / 2];
            for (int k = 0; k < inv.Length; k++) inv[k] = 1f / MathF.Pow(_theta, 2f * k / HeadDim) / factors[k];
            _invFreq.View.CopyFromCPU(inv);
        }
        ChooseFloat16Load(_model.Tensors.Where(t => t.Dimensions.Length == 2 && (t.Name.StartsWith("blk.", StringComparison.Ordinal) || t.Name == _head || t.Name == "token_embd.weight")).Select(t => t.Name));
        var mats = new List<string> { _head, "token_embd.weight" };          // the embedding rows are gathered on the device (a tied head is the same blocks)
        var vecs = new List<string> { "output_norm.weight" };
        _qkvAll = new bool[Layers];
        for (int l = 0; l < Layers; l++)
        {
            string p = $"blk.{l}.";
            var (tq, tk, tv) = (LoadType(Tensor(p + "attn_q.weight")), LoadType(Tensor(p + "attn_k.weight")), LoadType(Tensor(p + "attn_v.weight")));
            // the projections that share a block type are joined into one matrix (one GEMV a step instead of two or three); v alone when it differs (Q4_K_M)
            if (JoinProjections && tq == tk)
            {
                _qkvAll[l] = tv == tq;
                await JoinAsync(p + "attn_qkv", _qkvAll[l] ? new[] { p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight" } : new[] { p + "attn_q.weight", p + "attn_k.weight" }, ct).ConfigureAwait(false);
                if (!_qkvAll[l]) mats.Add(p + "attn_v.weight");
                int kvC = KvHeads * HeadDim;
                _vecs[p + "attn_qkv.bias"] = Own(_acc.Allocate1D((await BiasAsync(p + "attn_q.bias", QDim, ct)).Concat(await BiasAsync(p + "attn_k.bias", kvC, ct)).Concat(await BiasAsync(p + "attn_v.bias", kvC, ct)).ToArray()));
            }
            else
            {
                mats.AddRange(new[] { p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight" });
                int kvC = KvHeads * HeadDim;
                _vecs[p + "attn_q.bias"] = Own(_acc.Allocate1D(await BiasAsync(p + "attn_q.bias", QDim, ct)));
                _vecs[p + "attn_k.bias"] = Own(_acc.Allocate1D(await BiasAsync(p + "attn_k.bias", kvC, ct)));
                _vecs[p + "attn_v.bias"] = Own(_acc.Allocate1D(await BiasAsync(p + "attn_v.bias", kvC, ct)));
            }
            if (JoinProjections && LoadType(Tensor(p + "ffn_gate.weight")) == LoadType(Tensor(p + "ffn_up.weight")))
                await JoinAsync(p + "ffn_gate_up", new[] { p + "ffn_gate.weight", p + "ffn_up.weight" }, ct).ConfigureAwait(false);
            else mats.AddRange(new[] { p + "ffn_gate.weight", p + "ffn_up.weight" });
            mats.AddRange(new[] { p + "attn_output.weight", p + "ffn_down.weight" });
            vecs.AddRange(new[] { p + "attn_norm.weight", p + "ffn_norm.weight" });
            if (_qkNorm) vecs.AddRange(new[] { p + "attn_q_norm.weight", p + "attn_k_norm.weight" });
        }
        foreach (var name in mats.Distinct())
        {
            var t = Tensor(name);
            if (!FusedDequantMatMul.Accepts(LoadType(t))) throw new NotSupportedException($"{name}: {LoadType(t)} is not a type the fused GEMM reads");
            _mats[name] = (Own(await UploadAsync(t, ct).ConfigureAwait(false)), LoadType(t));
        }
        foreach (var name in vecs) _vecs[name] = Own(_acc.Allocate1D(await VecAsync(name, ct).ConfigureAwait(false)));
        // Q8_0 matrices regrouped in place for the faster GEMVs (GGMLType.Q8_0G; FusedDequantMatMul.RepackQ8_0Grouped). The joined q|k|v and gate|up read the
        // hidden width; every other matrix its own first dimension.
        if (GroupQ8)
            foreach (var name in _mats.Keys.ToList())
            {
                var (b, ty) = _mats[name];
                if (ty != GGMLType.Q8_0) continue;
                int K = name.EndsWith(".weight", StringComparison.Ordinal) ? (int)Tensor(name).Dimensions[0] : Hidden;
                if (K % 256 != 0) continue;
                _mm.RepackQ8_0Grouped(b.View, b.Length / 272);
                _mats[name] = (b, GGMLType.Q8_0G);
            }
    }

    /// <summary>
    /// Regroup Q8_0 weights in place at load (<see cref="GGMLType.Q8_0G"/>: per 256 values the 8 block scales, then the int8 values; 16-byte loads). MEASURED
    /// 2026-11-06, see FusedDequantMatMul.RepackQ8_0Grouped. Read when a decoder loads; env GGUF_Q8_GROUPED=0 keeps the native layout (the A/B).
    /// </summary>
    public static bool GroupQ8 { get; set; } = Environment.GetEnvironmentVariable("GGUF_Q8_GROUPED") != "0";

    // matrices of one block type stacked along the output dimension: their rows (the blocks of each output) follow each other, so the bytes simply concatenate
    private async Task JoinAsync(string name, string[] parts, CancellationToken ct)
    {
        var type = LoadType(Tensor(parts[0]));
        if (!FusedDequantMatMul.Accepts(type)) throw new NotSupportedException($"{parts[0]}: {type} is not a type the fused GEMM reads");
        var sizes = parts.Select(n => GGMLTypes.TypeSize(LoadType(Tensor(n)), _model.GetTensorElementCount(Tensor(n)))).ToArray();
        if (sizes.Take(sizes.Length - 1).All(z => z % 4 == 0))
        {
            // each part streamed into its place (the offsets stay word aligned)
            var joined = Own(_acc.Allocate1D<byte>((sizes.Sum() + 3) / 4 * 4));
            long off = 0;
            for (int i = 0; i < parts.Length; i++) { await UploadAsync(Tensor(parts[i]), joined.View.SubView(off, joined.Length - off), ct).ConfigureAwait(false); off += sizes[i]; }
            _mats[name] = (joined, type);
            return;
        }
        var chunks = new List<byte[]>();
        long total = 0;
        foreach (var part in parts)
        {
            var t = Tensor(part);
            long n = GGMLTypes.TypeSize(LoadType(t), _model.GetTensorElementCount(t));
            var raw = await ReadLoadedAsync(t, ct).ConfigureAwait(false);
            chunks.Add(raw.AsSpan(0, (int)n).ToArray()); total += n;
        }
        var all = new byte[(total + 3) / 4 * 4];
        long at = 0;
        foreach (var c in chunks) { Buffer.BlockCopy(c, 0, all, (int)at, c.Length); at += c.Length; }
        _mats[name] = (Own(_acc.Allocate1D(all)), type);
    }
    // a projection's bias, or zeros when the GGUF has none (qwen3, llama)
    private async Task<float[]> BiasAsync(string name, int length, CancellationToken ct) =>
        _model.Tensors.Any(t => t.Name == name) ? await VecAsync(name, ct).ConfigureAwait(false) : new float[length];
    // a norm or bias vector as F32: stored F32, or F16 / BF16 (a safetensors model seen through GGUF.SafeTensorsGguf) widened on the host
    private async Task<float[]> VecAsync(string name, CancellationToken ct)
    {
        var t = Tensor(name);
        if (!IsVectorType(t.Type)) throw new NotSupportedException($"{name}: {t.Type} (vectors are read as F32, F16 or BF16)");
        var raw = await ReadAsync(t, ct).ConfigureAwait(false);
        var f = new float[_model.GetTensorElementCount(t)];
        switch (t.Type)
        {
            case GGMLType.F32: Buffer.BlockCopy(raw, 0, f, 0, f.Length * 4); break;
            case GGMLType.F16: for (int i = 0; i < f.Length; i++) f[i] = (float)BitConverter.UInt16BitsToHalf(BitConverter.ToUInt16(raw, 2 * i)); break;
            default: for (int i = 0; i < f.Length; i++) f[i] = BitConverter.Int32BitsToSingle(BitConverter.ToUInt16(raw, 2 * i) << 16); break;
        }
        return f;
    }
    private static bool IsVectorType(GGMLType t) => t is GGMLType.F32 or GGMLType.F16 or GGMLType.BF16;
    private bool[] _qkvAll = Array.Empty<bool>();

    private readonly List<MemoryBuffer1D<int, Stride1D.Dense>> _idBufs = new();
    private int _idCall;
    // token ids: a fresh buffer per call until reused after a sync (a queued gather may still read the last one); ForwardAsync syncs every call, so two alternate
    private IV IntBuf(int n)
    {
        int slot = _idCall++ & 1;
        while (_idBufs.Count <= slot) { _idBufs.Add(Own(_acc.Allocate1D<int>(Math.Max(64, MaxSeq)))); _bufGen++; }
        return _idBufs[slot].View.SubView(0, n);
    }

    private FV Buf(ref MemoryBuffer1D<float, Stride1D.Dense>? b, long n)
    {
        if (b == null || b.Length < n) { b = _acc.Allocate1D<float>(Math.Max(n, 64)); _owned.Add(b); _bufGen++; }     // a replaced buffer stays owned (a queued dispatch may read it) and is freed with the decoder
        return b.View.SubView(0, n);
    }

    // sum = a + b (the residual stream), normed = rmsnorm(sum) * w; one fused pass, or (WebGL: no shared memory) an add then the two-pass norm
    private void AddNorm(FV a, FV b, FV sum, FV normed, FV w, int rows, int C)
    {
        _hnQ8 = null;
        if (_mm.Int8ActFor(rows, C))
        {
            // opt-in int8 activations: the norm also writes hn as q8_1 for the Q4_K GEMVs that read it (no quantize launch of their own)
            // (prefill rows too: the int8 tensor-core GEMM reads them); a grown buffer's predecessor stays owned (a queued launch may read it)
            long n8 = (long)rows * C;
            if (_hnQs == null || _hnQs.Length < n8) { _hnQs = _acc.Allocate1D<byte>(Math.Max(n8, 256)); _owned.Add(_hnQs); _hnDs = _acc.Allocate1D<float>(Math.Max(n8 / 16, 16)); _owned.Add(_hnDs); _bufGen++; }
            if (_norm.AddRMSNormFusedQ8(a, b, sum, normed, w, _hnQs.View, _hnDs!.View, rows, C, _eps)) { _hnQ8 = new FusedDequantMatMul.Q8Act(_hnQs.View, _hnDs.View); return; }
        }
        if (_norm.AddRMSNormFused(a, b, sum, normed, w, rows, C, _eps, 1)) return;
        _add3!((int)((long)rows * C), a, b, sum);
        _norm.RMSNorm(sum, normed, w, rows, C, _eps);
    }

    /// <summary>
    /// LoRA adapters applied unmerged after each adapted projection (<see cref="Training.LoraRuntime"/>; PEFT tensors). Null = the base model.
    /// Changing it forgets the reusable prefix: cached K/V were computed with the old adapters. It also drops the recorded decode steps (WebGPU plans,
    /// CUDA graphs): they replay the kernels and buffers of the adapters they were recorded with. Found by Data (SpawnDev.AI lora-test, 2026-11-06):
    /// after an in-place change a recorded step kept replaying the old set (1-2 of 24 greedy tokens right); gate
    /// Training_DecoderLora_SwapInPlace_DecodeMatchesFresh. LoraRuntime's static route switches (FuseAdjacent, GroupedDown) need
    /// <see cref="InvalidateCapturedSteps"/> the same way.
    /// </summary>
    public Training.LoraRuntime? Lora
    {
        get => _lora;
        set { if (!ReferenceEquals(_lora, value)) { _lora = value; _cacheIds = null; _bufGen++; } }
    }
    private Training.LoraRuntime? _lora;
    private static readonly string[] LoraQkv = { "attn_q", "attn_k", "attn_v" }, LoraQk = { "attn_q", "attn_k" }, LoraGateUp = { "ffn_gate", "ffn_up" };
    private int[]? _loraQkvOff, _loraQkOff, _loraGateUpOff;

    // the opt-in int8 activations' q8_1 forms of hn (written by AddNorm) and of the SwiGLU output (the down projection's input); null when not written this step
    private MemoryBuffer1D<byte, Stride1D.Dense>? _hnQs, _ffQs;
    private MemoryBuffer1D<float, Stride1D.Dense>? _hnDs, _ffDs;
    private FusedDequantMatMul.Q8Act? _hnQ8;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _attQs;
    private MemoryBuffer1D<float, Stride1D.Dense>? _attDs;
    private Action<KernelConfig, FV, FV, FV, IV, int, int, int, int, ArrayView1D<byte, Stride1D.Dense>, FV>? _attCombineGroupQ8;

    // decode steps (a few rows) read each weight once for all rows (ForwardRows); a prefill is a real GEMM. pre: x as q8_1, when its producer wrote it (opt-in int8 activations)
    private void Gemm(FV x, string name, FV o, int rows, int K, int N, FusedDequantMatMul.Q8Act? pre = null, FusedDequantMatMul? mm = null)
    {
        var (w, t) = _mats[name];
        mm ??= _mm;
        if (rows <= FusedDequantMatMul.RowsMax) mm.ForwardRows(x, w.View, o, rows, K, N, t, pre);
        else mm.Forward(x, w.View, o, rows, K, N, t, pre);
    }

    /// <summary>
    /// WebGPU: capture a decode step's dispatches once per (live sequences, temperature) and replay them each step after with one interop crossing, the position
    /// patched into the scalars that carry it (found by diffing captures at two positions; WebGPUDispatchPlan). On by default on WebGPU with the fused attention;
    /// off: every step encodes its kernels one by one.
    /// </summary>
    public bool CaptureWebGPU { get; set; }
    private bool CaptureActive => CaptureWebGPU && FusedAttention && !ProfileKernels && _acc is SpawnDev.ILGPU.WebGPU.WebGPUAccelerator;
    private bool _fixedChunks;
    private int _bufGen;
    private sealed class StepPlan
    {
        public SpawnDev.ILGPU.WebGPU.WebGPUDispatchPlan? Plan;
        public int P0, Gen;
        public FV Logits, Cand;
        public int OutRows;
        public readonly List<(int Snap, int Ofs, int ValAtP0, int Slope)> Patches = new();
        public bool Failed;
    }
    private readonly Dictionary<(int R, float T, bool Dev, bool Pen), StepPlan> _plans = new();
    // the drawn steps run the penalty kernel (Draw with a penalty other than 1); off, it is not dispatched at all (WebGPU pays per dispatch)
    private bool _penOn;
    private SpawnDev.ILGPU.WebGPU.WebGPUDispatchPlan? _lastReplayedPlan;

    /// <summary>
    /// Diagnostic (WebGPU): re-runs the most recently replayed decode-step plan <paramref name="replays"/> times (the per-token GPU floor, ms) and once
    /// with timestamps (<see cref="SpawnDev.ILGPU.WebGPU.WebGPUDispatchPlan.ReplayTimedAsync"/>: GPU ms by kernel label, JSON). Null when no plan has
    /// replayed. ⚠️ Each replay rewrites the K/V row of the last step's position from the newest drawn token: call it after a generation, not between steps.
    /// </summary>
    public async Task<(double FloorMs, string Attribution)?> ProfileRecordedStepAsync(int replays = 20)
    {
        var plan = _lastReplayedPlan;
        if (plan == null) return null;   // (a plan replaced by a recapture is disposed: its replay throws ObjectDisposedException)
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int r = 0; r < replays; r++) { await plan.ReplayAsync().ConfigureAwait(false); await _acc.SynchronizeAsync().ConfigureAwait(false); }
        double floor = sw.Elapsed.TotalMilliseconds / replays;
        return (floor, await plan.ReplayTimedAsync().ConfigureAwait(false));
    }

    /// <summary>Diagnostics: decode steps replayed from a captured plan, and plans captured, since the decoder was created.</summary>
    public (int Replays, int Captures) CaptureStats { get; private set; }

    private async Task<(FV Logits, FV Cand, int OutRows)> CapturedStepAsync(IV idBuf, int R, float temperature, int p, bool dev)
    {
        _fixedChunks = true;
        if (_plans.TryGetValue((R, temperature, dev, dev && _penOn), out var sp) && sp.Plan != null && sp.Gen == _bufGen)
        {
            _posBuf!.View.SubView(0, 1).CopyFromCPU(new[] { p });            // the kernels read the position from this buffer
            foreach (var (snap, ofs, v0, slope) in sp.Patches) sp.Plan.PatchScalarInt(snap, ofs, v0 + slope * (p - sp.P0));   // none left now; kept for any scalar that carries it
            await sp.Plan.ReplayAsync().ConfigureAwait(false);
            _lastReplayedPlan = sp.Plan;
            CaptureStats = (CaptureStats.Replays + 1, CaptureStats.Captures);
            return (sp.Logits, sp.Cand, sp.OutRows);
        }
        if (sp is { Failed: true }) return EncodeForward(idBuf, R, true, temperature, p, dev);
        sp?.Plan?.Dispose();
        // warm: a real step (allocates whatever the shape needs), then the captures: one at p + 1 (only for the diff: it writes cache row p + 1, which the real step
        // at p + 1 overwrites; the cache is position-addressed), then the real one at p, whose results stand.
        // Device draws: every pass draws, and a draw overwrites the tokens the step reads (_seqIds), so the tokens are kept aside and put back before each pass,
        // outside the recording windows (a restore recorded into the plan would pin the input to these tokens forever)
        // (the penalty windows likewise: every pass appends its draws)
        if (dev)
        {
            _seqIdsSnap ??= Own(_acc.Allocate1D<int>(Math.Max(64, MaxBatch))); _seqIdsSnap.View.CopyFrom(_seqIds!.View);
            if (_repSnap == null || _repSnap.Length != _rep!.Length) _repSnap = Own(_acc.Allocate1D<int>(_rep!.Length));
            _repSnap.View.CopyFrom(_rep.View);
        }
        var warm = EncodeForward(idBuf, R, true, temperature, p, dev);
        await _acc.SynchronizeAsync().ConfigureAwait(false);
        int gen = _bufGen;
        var wacc = (SpawnDev.ILGPU.WebGPU.WebGPUAccelerator)_acc;
        bool caching = SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableBindGroupCaching;
        SpawnDev.ILGPU.WebGPU.WebGPUDispatchPlan? a = null, b = null;
        (FV Logits, FV Cand, int OutRows) res;
        SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableBindGroupCaching = false;
        try
        {
            if (dev) { _seqIds!.View.CopyFrom(_seqIdsSnap!.View); _rep!.View.CopyFrom(_repSnap!.View); await _acc.SynchronizeAsync().ConfigureAwait(false); }
            b = wacc.BeginDispatchCapture(); b.CaptureScalarSnapshots = true;
            EncodeForward(idBuf, R, true, temperature, p + 1, dev);
            wacc.EndDispatchCapture();
            await _acc.SynchronizeAsync().ConfigureAwait(false);
            if (dev) { _seqIds!.View.CopyFrom(_seqIdsSnap!.View); _rep!.View.CopyFrom(_repSnap!.View); await _acc.SynchronizeAsync().ConfigureAwait(false); }
            a = wacc.BeginDispatchCapture(); a.CaptureScalarSnapshots = true;
            res = EncodeForward(idBuf, R, true, temperature, p, dev);
            wacc.EndDispatchCapture();
            await _acc.SynchronizeAsync().ConfigureAwait(false);
        }
        finally { SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableBindGroupCaching = caching; }
        sp = new StepPlan { P0 = p, Gen = gen, Logits = res.Logits, Cand = res.Cand, OutRows = res.OutRows };
        bool ok = _bufGen == gen && a.DispatchCount == b.DispatchCount && a.ScalarSnapshots.Count == b.ScalarSnapshots.Count && a.CopyEntries.Count == 0;
        for (int i = 0; ok && i < a.ScalarSnapshots.Count; i++)
        {
            var (ea, _, ba) = a.ScalarSnapshots[i];
            var (eb, _, bb) = b.ScalarSnapshots[i];
            if (ea != eb || ba.Length != bb.Length) { ok = false; break; }
            for (int o = 0; o + 4 <= ba.Length; o += 4)
            {
                int va = BitConverter.ToInt32(ba, o), vb = BitConverter.ToInt32(bb, o);
                if (va != vb) sp.Patches.Add((i, o, va, vb - va));               // a value of the position: affine, value(p) = value(p0) + slope (p - p0)
            }
        }
        b.Dispose();
        if (!ok) { a.Dispose(); sp.Failed = true; }
        else { sp.Plan = a; CaptureStats = (CaptureStats.Replays, CaptureStats.Captures + 1); }
        _plans[(R, temperature, dev, dev && _penOn)] = sp;
        return ok ? res : warm;
    }

    /// <summary>
    /// CUDA: record a decode step's kernels once per (live sequences, temperature) into a CUDA graph and replay it each step after with one
    /// <c>cuGraphLaunch</c>, instead of launching every kernel from the host (a 28-layer model is ~300 launches a token). The kernels read the position
    /// from a device buffer and the attention keeps one shape for the whole generation (as for <see cref="CaptureWebGPU"/>), so the graph has nothing to
    /// patch: a step is the position write and the launch. On by default on CUDA where the driver has the graph API (env GGUF_BATCH_CUDA_GRAPH=0: off).
    /// Bit-identical: the same kernels with the same arguments in the same order.
    /// </summary>
    public bool CaptureCuda { get; set; }
    private bool CaptureCudaActive => CaptureCuda && FusedAttention && !ProfileKernels && _acc is global::ILGPU.Runtime.Cuda.CudaAccelerator;
    private bool _cudaCapturing;                                              // EncodeForward is recording: no host writes (illegal mid-capture)
    private global::ILGPU.Runtime.Cuda.CudaStream? _capStream;
    private global::ILGPU.Runtime.Cuda.CudaStream? _vStream;
    /// <summary>
    /// CUDA: run v's GEMV (when v's block type differs from q / k, as in Q4_K_M) on a second stream concurrently with the q|k GEMV and head norms
    /// (parallel branches of the captured step graph). Default on (it applies on CUDA only); env GGUF_OVERLAP_V=0 / 1. MEASURED 2026-11-10 (RTX 4070,
    /// qwen3:1.7b Q4_K_M, SpawnDev.ILGPU 5.4.24, three interleaved rounds, quiet machine): decode 224.9 / 230.7 / 221.1 -> 234.9 / 233.8 / 235.1 tok/s
    /// short (+4.0%), 212.8 / 216.1 / 210.7 -> 219.4 / 216.6 / 220.8 long (+2.7%), prefill ~+3.5%; greedy text identical every round.
    /// </summary>
    public bool OverlapV { get => _overlapV; set { if (_overlapV != value) { _overlapV = value; _bufGen++; } } }
    private bool _overlapV = Environment.GetEnvironmentVariable("GGUF_OVERLAP_V") != "0";
    private sealed class CudaStepPlan
    {
        public global::ILGPU.Runtime.Cuda.CudaGraphExec? Exec;
        public int Gen;
        public FV Logits, Cand;
        public int OutRows;
        public string? Refusal;
    }
    private readonly Dictionary<(int R, float T, bool Dev, bool Pen), CudaStepPlan> _cudaPlans = new();
    /// <summary>Why the last CUDA step capture was refused (null: none was).</summary>
    public string? LastCudaCaptureRefusal { get; private set; }

    private (FV Logits, FV Cand, int OutRows) CudaCapturedStep(IV idBuf, int R, float temperature, int p, bool dev)
    {
        _fixedChunks = true;
        var key = (R, temperature, dev, dev && _penOn);
        var defStream = (global::ILGPU.Runtime.Cuda.CudaStream)_acc.DefaultStream;   // the legacy stream: in order with the position write and the reads
        if (_cudaPlans.TryGetValue(key, out var cp) && cp.Exec != null && cp.Gen == _bufGen)
        {
            _posBuf!.View.SubView(0, 1).CopyFromCPU(new[] { p });
            cp.Exec.Launch(defStream);
            CaptureStats = (CaptureStats.Replays + 1, CaptureStats.Captures);
            return (cp.Logits, cp.Cand, cp.OutRows);
        }
        // the real step (it also loads and allocates whatever the shape needs); its results stand. A stream capture RECORDS without running, so the capture
        // below changes nothing on the device: no tokens to keep aside, no second pass.
        var res = EncodeForward(idBuf, R, true, temperature, p, dev);
        if (cp is { Refusal: not null } && cp.Gen == _bufGen) return res;
        cp?.Exec?.Dispose();
        // nothing may register with or be freed by the accelerator inside the capture window: settle first
        _acc.Synchronize();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        int gen = _bufGen;
        _capStream ??= (global::ILGPU.Runtime.Cuda.CudaStream)_acc.CreateStream();
        global::ILGPU.Runtime.Cuda.CudaGraph? graph = null;
        string? why = null;
        (FV Logits, FV Cand, int OutRows) rec = default;
        _cudaCapturing = true;
        try
        {
            using (_acc.WithDefaultStream(_capStream))
            {
                _capStream.BeginCapture(global::ILGPU.Runtime.Cuda.CudaStreamCaptureMode.ThreadLocal);
                try { rec = EncodeForward(idBuf, R, true, temperature, p, dev); }
                catch (Exception ex) { why = ex.GetType().Name + ": " + ex.Message; }
                try { graph = _capStream.EndCapture(); }
                catch (Exception ex) { why ??= "EndCapture: " + ex.Message; graph = null; }
            }
        }
        finally { _cudaCapturing = false; }
        if (why == null && _bufGen != gen) why = "a buffer grew during the capture pass";
        cp = new CudaStepPlan { Gen = gen, Logits = rec.Logits, Cand = rec.Cand, OutRows = rec.OutRows };
        if (why == null && graph != null)
        {
            try
            {
                using (graph) { cp.Exec = graph.Instantiate(); cp.Exec.Upload(defStream); }
                CaptureStats = (CaptureStats.Replays, CaptureStats.Captures + 1);
            }
            catch (Exception ex) { why = "Instantiate: " + ex.Message; cp.Exec = null; }
        }
        else graph?.Dispose();
        cp.Refusal = why;
        LastCudaCaptureRefusal = why;
        if (why != null) Console.WriteLine($"[GgufBatchDecoder] CUDA step capture refused, launching kernels one by one: {why}");
        _cudaPlans[key] = cp;
        return res;
    }

    // every kernel of one forward (no sync): the embedding gather through the candidate kernels. Decode steps on WebGPU are captured from here and replayed.
    private (FV Logits, FV Cand, int OutRows) EncodeForward(IV idBuf, int R, bool decode, float temperature, int p, bool dev = false)
    {
        int C = Hidden, kvC = KvHeads * HeadDim;
        var seqView = _seqBuf!.View;
        // the position the kernels read (a buffer, not a scalar: a captured step then has nothing to patch, one int written per step). The write is ordered
        // with the queued work on every backend (WebGPU: writeBuffer on the queue timeline after the submits before it; desktop: a synchronous copy), and a
        // pipelined step's kernels read it before the next step's write lands.
        _posBuf ??= Own(_acc.Allocate1D<int>(64));
        var posView = _posBuf.View.SubView(0, 1);
        if (!_cudaCapturing) posView.CopyFromCPU(new[] { p });              // a CUDA capture records the kernels; the replay writes the position
        // device draws: row r's input is the token its sequence drew last
        if (dev) _gather!(R, _seqIds!.View, seqView, idBuf);
        var hid = Buf(ref _hid, (long)R * C);
        var (embW, embT) = _mats["token_embd.weight"];
        int code = FusedDequantMatMul.TypeCode(embT);
        _embed!((int)((long)R * C), idBuf, embW.View.Cast<byte, int>(), hid, C, code, FusedDequantMatMul.RowBytesOf(code, C)); Prof("embed");
        var hid2 = Buf(ref _hid2, (long)R * C);
        int Q = QDim, rope = _normRope ? 2 : 0;                               // rotary style, passed in bit 1 of the rotary kernels' mode
        var hn = Buf(ref _hn, (long)R * C); var q = Buf(ref _q, (long)R * Q); var k = Buf(ref _k, (long)R * kvC); var v = Buf(ref _v, (long)R * kvC);
        var att = Buf(ref _att, (long)R * Q); var proj = Buf(ref _proj, (long)R * C);
        var gate = Buf(ref _gate, (long)R * FfnDim); var up = Buf(ref _up, (long)R * FfnDim);
        int nk = p + (decode ? 1 : R);
        int mode = decode ? 1 : 0;
        // the residual stream alternates between hid and hid2; every residual add is fused with the norm after it (a layer's FFN add with the next layer's
        // attention norm, the last one with the output norm): one group per row, not a thread per row walking the row twice
        _norm.RMSNorm(hid, hn, _vecs["blk.0.attn_norm.weight"].View, R, C, _eps); Prof("rms");
        _hnQ8 = null;
        for (int l = 0; l < Layers; l++)
        {
            string b = $"blk.{l}.";
            FusedDequantMatMul.Q8Act? attQ8 = null;                     // att as q8_1 when the merge wrote it (opt-in int8 activations)
            if (_mats.ContainsKey(b + "attn_qkv"))
            {
                // one GEMV for q|k(|v), then bias + rotary + the cache store in one kernel (q to its buffer, k and v to the caches)
                bool all = _qkvAll[l];
                int n = all ? Q + 2 * kvC : Q + kvC;
                var qkv = Buf(ref _qkv, (long)R * (Q + 2 * kvC));
                // CUDA: v's GEMV (its own block type, so not in qkv) on a second stream beside the q|k GEMV and the head norms - two small latency-bound
                // GEMVs on the same input; captured as parallel branches of the step's graph (SpawnDev.ILGPU CudaStream.WaitFor: fork here, join before
                // QkvPost reads v). Off when LoRA applies (its deltas are not stream-aware) or kernels are profiled (Prof syncs).
                bool overlapV = !all && OverlapV && Lora == null && !ProfileKernels && _acc is global::ILGPU.Runtime.Cuda.CudaAccelerator;
                global::ILGPU.Runtime.Cuda.CudaStream? mainStream = null;
                if (overlapV)
                {
                    mainStream = (global::ILGPU.Runtime.Cuda.CudaStream)_acc.DefaultStream;
                    _vStream ??= (global::ILGPU.Runtime.Cuda.CudaStream)_acc.CreateStream();
                    _vStream.WaitFor(mainStream);                                // fork: v reads hn, written on the main stream
                    using (_acc.WithDefaultStream(_vStream)) Gemm(hn, b + "attn_v.weight", v, R, C, kvC, _hnQ8, _mmV ??= new FusedDequantMatMul(_acc));
                }
                Gemm(hn, b + "attn_qkv", qkv, R, C, n, _hnQ8); Prof("qkv");
                if (Lora != null)
                {
                    // LoRA deltas into the fused q|k|v columns, before the bias / rotary / cache-store kernel reads them (one packed launch)
                    if (all) Lora.ApplyAdjacent(l, LoraQkv, _loraQkvOff ??= new[] { 0, Q, Q + kvC }, hn, qkv, R, n);
                    else Lora.ApplyAdjacent(l, LoraQk, _loraQkOff ??= new[] { 0, Q }, hn, qkv, R, n);
                }
                var vb = att;                                                // unread when v is in qkv (a distinct buffer: no aliasing)
                if (overlapV) vb = v;
                else if (!all) { Gemm(hn, b + "attn_v.weight", v, R, C, kvC, _hnQ8); Lora?.Apply(l, "attn_v", hn, v, R, 0, kvC); vb = v; Prof("v"); }
                if (_qkNorm)
                {
                    // qwen3: an RMS norm per head of q and of k (weight [head width]) before the rotary; no biases, so the post kernel's zero bias changes nothing
                    // one launch for both (the q heads then the k heads are contiguous in each qkv row); bit-identical to the two-launch form
                    _headNormQk!(new KernelConfig(R * (Heads + KvHeads), HeadDim), qkv, _vecs[b + "attn_q_norm.weight"].View, _vecs[b + "attn_k_norm.weight"].View,
                        n, Heads, Heads + KvHeads, _eps); Prof("qk-norm");
                }
                if (overlapV) mainStream!.WaitFor(_vStream!);                    // join: QkvPost reads v
                int U = Heads * (HeadDim / 2) + KvHeads * (HeadDim / 2) + kvC;
                _kv.QkvPost(l, R * U, qkv, vb, _vecs[b + "attn_qkv.bias"].View, _invFreq.View, q, seqView, posView, Heads | (KvHeads << 16), HeadDim, n, (all ? 1 : 0) | (mode << 1) | (rope << 1), MaxSeq); Prof("qkv-post");
            }
            else
            {
                Gemm(hn, b + "attn_q.weight", q, R, C, Q, _hnQ8); Lora?.Apply(l, "attn_q", hn, q, R, 0, Q); _bias!((int)((long)R * Q), q, _vecs[b + "attn_q.bias"].View, Q); Prof("q");
                Gemm(hn, b + "attn_k.weight", k, R, C, kvC, _hnQ8); Lora?.Apply(l, "attn_k", hn, k, R, 0, kvC); _bias!((int)((long)R * kvC), k, _vecs[b + "attn_k.bias"].View, kvC); Prof("k");
                Gemm(hn, b + "attn_v.weight", v, R, C, kvC, _hnQ8); Lora?.Apply(l, "attn_v", hn, v, R, 0, kvC); _bias!((int)((long)R * kvC), v, _vecs[b + "attn_v.bias"].View, kvC); Prof("v");
                if (_qkNorm)
                {
                    _headNorm!(new KernelConfig(R * Heads, HeadDim), q, _vecs[b + "attn_q_norm.weight"].View, Q, 0, Heads, _eps);
                    _headNorm(new KernelConfig(R * KvHeads, HeadDim), k, _vecs[b + "attn_k_norm.weight"].View, kvC, 0, KvHeads, _eps); Prof("qk-norm");
                }
                _rope!(R * Heads * (HeadDim / 2), q, _invFreq.View, posView, Heads, HeadDim, mode | rope);
                _rope!(R * KvHeads * (HeadDim / 2), k, _invFreq.View, posView, KvHeads, HeadDim, mode | rope); Prof("rope");
                _kv.Store(l, (int)((long)R * kvC), k, v, seqView, posView, kvC, MaxSeq, mode); Prof("store");
            }
            // split-K attention for decode rows only: it gives each (row, head, chunk of keys) a group that walks its keys one by one, which is right for a few decode
            // rows and ruinous for a prefill's hundreds (MEASURED 2026-11-01, CUDA, qwen3-0.6b, 600 rows over a 3k cache: attention 3.1 s of a 5.8 s prefill). A
            // prefill takes the three-kernel path (scores, softmax, P.V: a thread per score / per output)
            if (!decode && _hasFlash && FlashPrefill)
            {
                if (_hasFlashMma && FlashMma && _kv.FlashMma(l, q, att, R, Heads, KvHeads, p))
                    LastFlashKernel = "mma";
                else if (_hasFlashGqa && FlashGqa)
                {
                    int rowsPer = FgQ / (Heads / KvHeads);
                    _kv.FlashGqa(l, new KernelConfig(((R + rowsPer - 1) / rowsPer) * KvHeads, FgThreads), q, att, R, Heads, KvHeads, HeadDim, p, 1f / MathF.Sqrt(HeadDim));
                    LastFlashKernel = "gqa-tiled";
                }
                else
                {
                    _kv.Flash(l, new KernelConfig(((R + FaRows - 1) / FaRows) * Heads, FaThreads), q, att, R, Heads, KvHeads, HeadDim, p, 1f / MathF.Sqrt(HeadDim));
                    LastFlashKernel = "tiled";
                }
                Prof("attn-flash");
            }
            else if (FusedAttention && decode)
            {
                // split-K attention: a group per (row, head, chunk of keys) computes the chunk's scores, max, exp-sum and P.V; a second kernel merges the chunks
                // a captured step keeps one shape for the whole generation: every chunk the cache can hold (the ones past the position mask to nothing)
                bool gqa = GqaAttention && Heads > KvHeads && Heads / KvHeads <= GqaMaxG && HeadDim <= GqaMaxD && HeadDim % 8 == 0;
                int ck = gqa ? GqaChunkKeys(Heads / KvHeads) : AttChunk;
                int ach = _fixedChunks ? (MaxSeq + ck - 1) / ck : (nk + ck - 1) / ck;
                var part = Buf(ref _attPart, (long)R * Heads * ach * HeadDim); var ml = Buf(ref _attMl, (long)R * Heads * ach * 2);
                if (gqa && AttentionVec && _kv.IsHalf && VecAttentionFits(HeadDim))
                {
                    bool shuffle = AttentionShuffle && ShuffleAttentionFits(HeadDim);
                    _kv.AttPartialGqaVec(l, new KernelConfig(R * KvHeads * ach, HeadDim), q, seqView, posView, part, ml, Heads, KvHeads, MaxSeq, mode, ach, ck, 1f / MathF.Sqrt(HeadDim), shuffle, AttentionIlp);
                    LastDecodeAttentionKernel = "gqa-vec" + (AttentionIlp ? "-ilp" : "") + (shuffle ? "-shuffle" : "");
                }
                else if (gqa)
                {
                    bool shuffle = AttentionIlp && AttentionShuffle && ShuffleAttentionFits(HeadDim);
                    _kv.AttPartialGqa(l, new KernelConfig(R * KvHeads * ach, HeadDim), q, seqView, posView, part, ml, Heads, KvHeads, MaxSeq, mode, ach, ck, 1f / MathF.Sqrt(HeadDim), AttentionIlp, shuffle);
                    LastDecodeAttentionKernel = shuffle ? "gqa-ilp-shuffle" : AttentionIlp ? "gqa-ilp" : "gqa";
                }
                else
                {
                    _kv.AttPartial(l, new KernelConfig(R * Heads * ach, HeadDim), q, seqView, posView, part, ml, Heads, KvHeads, MaxSeq, mode, ach, 1f / MathF.Sqrt(HeadDim));
                    LastDecodeAttentionKernel = "per-head";
                }
                Prof("attn");
                if (AttentionCombineGroup && _attCombineGroup != null && ach <= CombineMaxChunks && HeadDim % 32 == 0 && _mm.Int8ActFor(R, Q))
                {
                    // opt-in int8 activations: the merge also writes att as q8_1 for the o projection
                    if (_attQs == null) { _attQs = _acc.Allocate1D<byte>(Math.Max(Q, 256)); _owned.Add(_attQs); _attDs = _acc.Allocate1D<float>(Math.Max(Q / 16, 16)); _owned.Add(_attDs); _bufGen++; }
                    _attCombineGroupQ8 ??= _acc.LoadStreamKernel<FV, FV, FV, IV, int, int, int, int, ArrayView1D<byte, Stride1D.Dense>, FV>(AttnCombineGroupQ8Impl);
                    _attCombineGroupQ8(new KernelConfig(R * Heads, HeadDim), part, ml, att, posView, ach, gqa ? ck : AttChunk, Heads, mode, _attQs.View, _attDs!.View);
                    attQ8 = new FusedDequantMatMul.Q8Act(_attQs.View, _attDs.View);
                    LastAttentionMergeKernel = "group-q8";
                }
                else if (AttentionCombineGroup && _attCombineGroup != null && ach <= CombineMaxChunks)
                {
                    _attCombineGroup(new KernelConfig(R * Heads, HeadDim), part, ml, att, posView, ach, gqa ? ck : AttChunk, Heads, mode);
                    LastAttentionMergeKernel = "group";
                }
                else
                {
                    _attCombine!((int)((long)R * Q), part, ml, att, posView, HeadDim, ach, gqa ? ck : AttChunk, Heads, mode);
                    LastAttentionMergeKernel = "per-value";
                }
                Prof("attn-merge");
            }
            else
            {
                var scores = Buf(ref _scores, (long)Heads * R * nk);
                _kv.Scores(l, (int)((long)Heads * R * nk), q, seqView, scores, R, nk, Heads, HeadDim, KvHeads, MaxSeq, p, mode, 1f / MathF.Sqrt(HeadDim)); Prof("scores");
                var stats = Buf(ref _softStats, 2L * Heads * R);
                _softmaxStats!(Heads * R, scores, stats, nk);
                _softmaxNorm!((int)((long)Heads * R * nk), scores, stats, nk); Prof("softmax");
                _kv.Pv(l, (int)((long)R * Q), scores, seqView, att, R, nk, Heads, HeadDim, KvHeads, MaxSeq, mode); Prof("pv");
            }
            Gemm(att, b + "attn_output.weight", proj, R, Q, C, attQ8); Lora?.Apply(l, "attn_output", att, proj, R, 0, C); Prof("o");
            AddNorm(hid, proj, hid2, hn, _vecs[b + "ffn_norm.weight"].View, R, C); Prof("add+rms");
            FusedDequantMatMul.Q8Act? ffQ8 = null;
            if (_mats.ContainsKey(b + "ffn_gate_up"))
            {
                var gu = Buf(ref _gateUp, (long)R * 2 * FfnDim);
                var (wgu, tgu) = _mats[b + "ffn_gate_up"];
                // opt-in int8 activations, prefill rows: gate | up with SwiGLU in the int8 GEMM's epilogue (gate | up never reaches memory); down quantizes its input
                if (Lora == null && R > FusedDequantMatMul.RowsMax && _mm.ForwardGateUpSwiGLUInt8(hn, wgu.View, tgu, gate, R, C, FfnDim, _hnQ8))
                {
                    Prof("gate-up+swiglu");
                    goto FfnDown;
                }
                Gemm(hn, b + "ffn_gate_up", gu, R, C, 2 * FfnDim, _hnQ8); Prof("gate-up");
                Lora?.ApplyAdjacent(l, LoraGateUp, _loraGateUpOff ??= new[] { 0, FfnDim }, hn, gu, R, 2 * FfnDim);
                if (_mm.Int8ActFor(R, FfnDim) && FfnDim % 32 == 0)
                {
                    // opt-in int8 activations: the SwiGLU also writes its output as q8_1 for the down projection (a thread a value, warps aligned with blocks)
                    long nf = (long)R * FfnDim;
                    if (_ffQs == null || _ffQs.Length < nf) { _ffQs = _acc.Allocate1D<byte>(Math.Max(nf, 256)); _owned.Add(_ffQs); _ffDs = _acc.Allocate1D<float>(Math.Max(nf / 16, 16)); _owned.Add(_ffDs); _bufGen++; }
                    _swigluJoinedQ8 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, ArrayView1D<byte, Stride1D.Dense>, FV, int>(SwigluJoinedQ8Impl);
                    _swigluJoinedQ8((int)nf, gu, gate, _ffQs.View, _ffDs!.View, FfnDim); Prof("swiglu");
                    ffQ8 = new FusedDequantMatMul.Q8Act(_ffQs.View, _ffDs.View);
                }
                else { _swigluJoined!((int)((long)R * FfnDim), gu, gate, FfnDim); Prof("swiglu"); }
            }
            else
            {
                Gemm(hn, b + "ffn_gate.weight", gate, R, C, FfnDim, _hnQ8); Lora?.Apply(l, "ffn_gate", hn, gate, R, 0, FfnDim); Prof("gate");
                Gemm(hn, b + "ffn_up.weight", up, R, C, FfnDim, _hnQ8); Lora?.Apply(l, "ffn_up", hn, up, R, 0, FfnDim); Prof("up");
                _swiglu!((int)((long)R * FfnDim), gate, up); Prof("swiglu");
            }
            FfnDown:
            Gemm(gate, b + "ffn_down.weight", proj, R, FfnDim, C, ffQ8); Lora?.Apply(l, "ffn_down", gate, proj, R, 0, C); Prof("down");
            AddNorm(hid2, proj, hid, hn, _vecs[l + 1 < Layers ? $"blk.{l + 1}.attn_norm.weight" : "output_norm.weight"].View, R, C); Prof("add+rms");
            // a prefill submits each layer on its own (WebGPU): one submission is then at most one layer of one PrefillChunk, whatever the model's size (SpawnDev.AI
            // lost the device on qwen3:4b with a prefill encoded as one command buffer, 2026-11-01). A decode step is untouched (its captured plan replays it).
            if (!decode) (_acc as SpawnDev.ILGPU.WebGPU.WebGPUAccelerator)?.FlushPendingCommands();
        }
        // the LM head on the output norm's rows: every row when decoding, the last row of a prefill
        int outRows = decode ? R : 1;
        var hnOut = decode ? hn : hn.SubView((long)(R - 1) * C, C);
        var logits = Buf(ref _logits, (long)outRows * Vocab);
        Gemm(hnOut, _head, logits, outRows, C, Vocab, decode ? _hnQ8 : null); Prof("head");
        // WebGL: no candidates (the gather takes its slots from Atomic.Add, which WebGL cannot return: SpawnDev.ILGPU's vote emulation; 5.4.15 refuses the kernel),
        // a draw reads the row
        if (NoCandidates) return (logits, default, outRows);
        // each row's best tokens and (sampling) its softmax mass, on the device
        int nch = Chunks, S = CandStride;
        var maxV = Buf(ref _maxV, (long)outRows * nch);
        var selV = Buf(ref _selV, (long)outRows * SelectCap);
        _maxI ??= Own(_acc.Allocate1D<int>((long)MaxBatch * nch));
        _selI ??= Own(_acc.Allocate1D<int>((long)MaxBatch * SelectCap));
        _selN ??= Own(_acc.Allocate1D<int>(Math.Max(64, MaxBatch)));
        var maxI = _maxI.View.SubView(0, (long)outRows * nch);
        var cand = Buf(ref _cand, (long)outRows * S);
        // device draws: each row's repetition penalty on its logits first (a no-op at penalty 1), so the candidates and the draw see the penalised row
        if (dev && _penOn) { _penalize!(R * _repCap, logits, seqView, _drawI!.View, _drawF!.View, _rep!.View, Vocab); Prof("penalty"); }
        bool group = TopKGroup && _chunkMaxGroup != null;
        if (group) _chunkMaxGroup!(new KernelConfig(outRows * nch, Chunk), logits, maxV, maxI, Vocab, nch);
        else _chunkMax!(outRows * nch, logits, maxV, maxI, Vocab, nch, Chunk);
        Prof("chunkMax");
        if (group && _floorGroup != null) _floorGroup(new KernelConfig(SpawnDev.ILGPU.ML.Kernels.TopKKernels.FloorGroups(outRows, nch), SpawnDev.ILGPU.ML.Kernels.TopKKernels.FloorGroupThreads), maxV, maxI, cand, _selN.View, nch, S);
        else _floor!(SpawnDev.ILGPU.ML.Kernels.TopKKernels.FloorThreads(outRows, nch), maxV, maxI, cand, _selN.View, nch, S);
        Prof("floor");
        if (group) _selectGroup!(new KernelConfig(outRows * nch, Chunk), logits, cand, _selN.View, selV, _selI.View, Vocab, nch, S, temperature);
        else _select!(outRows * nch, logits, cand, _selN.View, selV, _selI.View, Vocab, nch, S, temperature, Chunk);
        Prof("select");
        _final!(SpawnDev.ILGPU.ML.Kernels.TopKKernels.FinalThreads(outRows), selV, _selI.View, _selN.View, cand, S); Prof("final");
        // device draws: each row's next token, into its sequence's slot of _seqIds (the next step's input), then a copy for the host to read
        if (dev) QueueDraw(R, rowZero: false, tAdd: 1, temperature, logits, cand);
        return (logits, cand, outRows);
    }

    private void EnsureDrawBuffers()
    {
        // loaded on first use, not with the rest: a group kernel (WebGL, which has no groups, never draws on the device)
        _draw ??= _acc.LoadStreamKernel<FV, FV, IV, IV, FV, IV, IV, IV, int, int, int, int, int, float>(DrawImpl);
        _gather ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV>(GatherIdsImpl);
        _penalize ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, FV, IV, int>(PenalizeImpl);
        EnsurePenaltyWindow(64);
        _seqIds ??= Own(_acc.Allocate1D<int>(Math.Max(64, MaxBatch)));
        _drawI ??= Own(_acc.Allocate1D<int>(64));
        _drawF ??= Own(_acc.Allocate1D<float>(64));
        for (int i = 0; i < 2; i++) _ring[i] ??= Own(_acc.Allocate1D<int>(Math.Max(64, MaxBatch)));
    }

    // the draw kernel for `slots` sequences (seqs[0..slots), _seqBuf): slot s draws from row s of the forward's logits / candidates, or (rowZero) all from row 0
    // (the prefill's one row); the token goes to _seqIds[seq]; its position (what the uniform is drawn for) is the forward's position + tAdd
    private void QueueDraw(int slots, bool rowZero, int tAdd, float temperature, FV logits, FV cand)
    {
        EnsureDrawBuffers();
        _draw!(new KernelConfig(slots, DrawGroup), logits, cand, _seqBuf!.View, _posBuf!.View, _drawF!.View, _drawI!.View, _seqIds!.View, _rep!.View, Vocab, Chunks, CandStride, rowZero ? 1 : 0, tAdd, temperature);
        Prof("draw");
    }

    // the penalty windows hold at least `window` tokens per sequence (a larger buffer replaces the old one: captured plans bound it, so they are recaptured)
    private void EnsurePenaltyWindow(int window)
    {
        if (_rep != null && _repCap >= window) return;
        _repCap = Math.Max(window, _repCap);
        _rep = Own(_acc.Allocate1D<int>((long)MaxBatch * (_repCap + 1)));       // the old one stays owned: a queued dispatch may read it
        _bufGen++;                                                              // captured steps (WebGPU plans, CUDA graphs) bound the old one: capture again
        _rep.MemSetToZero();
        _bufGen++;
    }

    /// <summary>
    /// Run <paramref name="ids"/> through the model: <paramref name="decode"/> = false is a prefill of sequence 0 at positions Position.. (causal over its cache), true is one
    /// token per live sequence at Position. Leaves the final hidden rows' logits (prefill: only the last row's) on the device and their candidates on the host
    /// (with the softmax mass at <paramref name="temperature"/> when it is above 0).
    /// </summary>
    /// <param name="ids">The tokens (null: device draws, each row's input is the token its sequence drew last, and the step ends with the rows' next draws).</param>
    /// <param name="readBack">Read the candidates now (the forward's one wait); false leaves the step queued (they are read when <see cref="SampleAsync"/> needs them).</param>
    private async Task ForwardAsync(int[]? ids, int[]? seqs, bool decode, float temperature, bool readBack, CancellationToken ct)
    {
        bool dev = ids == null;
        int R = dev ? seqs!.Length : ids!.Length, p = Position;
        if (p + (decode ? 1 : R) > MaxSeq) throw new InvalidOperationException($"past the cache length ({MaxSeq})");
        LoadKernels();
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        if (ProfileKernels) { _acc.Synchronize(); _profT = System.Diagnostics.Stopwatch.GetTimestamp(); }
        // the token ids (decode: one stable buffer, the one a captured step binds; device draws: gathered there on the device) and the sequence map; the embedding
        // rows are gathered on the device
        IV idBuf;
        if (decode) { _stepIds ??= Own(_acc.Allocate1D<int>(Math.Max(64, MaxBatch))); idBuf = _stepIds.View.SubView(0, R); }
        else idBuf = IntBuf(R);
        if (!dev) idBuf.CopyFromCPU(ids!);
        // decode row r advances sequence seqs[r] (its cache region); sized for a prefill's rows too, which the kernels do not read it for. Written only when it
        // changes (each write is a queue operation; desktop: a stream sync)
        _seqBuf ??= Own(_acc.Allocate1D<int>((long)MaxSeq + MaxBatch));
        if (decode && !seqs!.AsSpan().SequenceEqual(_seqsOnDevice)) { _seqBuf.View.SubView(0, R).CopyFromCPU(seqs); _seqsOnDevice = (int[])seqs.Clone(); }
        long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
        FV logits, cand; int outRows;
        if (decode && CaptureActive) (logits, cand, outRows) = await CapturedStepAsync(idBuf, R, temperature, p, dev).ConfigureAwait(false);
        else if (decode && CaptureCudaActive) (logits, cand, outRows) = CudaCapturedStep(idBuf, R, temperature, p, dev);
        else (logits, cand, outRows) = EncodeForward(idBuf, R, decode, temperature, p, dev);
        long t2 = System.Diagnostics.Stopwatch.GetTimestamp();
        _lastP = p;
        Position += decode ? 1 : R;
        _lastLogits = logits; _lastCandView = cand; _lastRows = outRows; _lastT = temperature; _lastSeqs = decode ? (int[])seqs!.Clone() : new[] { 0 }; _lastPrefill = !decode;
        _lastCand = null;
        // the readback is the forward's one wait: it is ordered after the queued kernels on every backend (WebGPU: flush, copy, mapAsync; desktop: a synchronous
        // copy), so a SynchronizeAsync before it only added a second round trip
        if (readBack && !NoCandidates) _lastCand = await cand.CopyToHostAsync().ConfigureAwait(false);
        long t3 = System.Diagnostics.Stopwatch.GetTimestamp();
        if (decode)
        {
            double ms(long a, long b) => (b - a) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            var d = DecodeTimings;
            DecodeTimings = (d.Embed + ms(t0, t1), d.Device + ms(t1, t2), d.Readback + ms(t2, t3), d.Steps + 1);
        }
    }
    private int[] _seqsOnDevice = Array.Empty<int>();

    /// <summary>
    /// Prefill the prompt into sequence 0 from position 0 (one live sequence afterwards). <paramref name="temperature"/>: what the next draw (<see cref="SampleAsync"/>
    /// or <see cref="Draw"/>) uses, 0 = greedy (the device then skips the softmax mass). Queued: nothing is read back until a draw needs it.
    /// </summary>
    public Task PrefillAsync(int[] promptIds, float temperature = 0f, CancellationToken ct = default) => PrefillAsync(promptIds, temperature, false, ct);

    /// <summary>
    /// <see cref="PrefillAsync(int[], float, CancellationToken)"/>; <paramref name="reusePrefix"/>: keep the cache rows of the longest prefix this prompt shares
    /// with the last <see cref="GenerateAsync"/>'s prompt and sequence 0's tokens, and run only the rest (a chat's next turn re-sends the whole conversation:
    /// only the new part costs a prefill). At least the prompt's last token always runs (its logits start the answer). <see cref="LastReusedPrefix"/> says
    /// how many rows were kept.
    /// </summary>
    public async Task PrefillAsync(int[] promptIds, float temperature, bool reusePrefix, CancellationToken ct = default)
    {
        if (promptIds.Length == 0) throw new ArgumentException("an empty prompt");
        int keep = 0;
        if (reusePrefix && _cacheIds != null)
        {
            int max = Math.Min(_cacheIds.Length, promptIds.Length - 1);
            while (keep < max && _cacheIds[keep] == promptIds[keep]) keep++;
        }
        LastReusedPrefix = keep;
        _cacheIds = null;                                                        // GenerateAsync records what the rows hold when it ends
        Position = keep; Batch = 1; DecodeTimings = default; RowReads = 0; KernelTimes.Clear(); _drawPos = -1;
        // in chunks of at most PrefillChunk rows: a chunk is a prefill at the next position (causal over the cache rows before it, as a reused prefix is), so the
        // result is the same; it bounds the row buffers (one submission's work is bounded by the per-layer submit in EncodeForward)
        int chunk = Math.Max(1, PrefillChunk);
        for (int at = keep; at < promptIds.Length; at += chunk)
        {
            int n = Math.Min(chunk, promptIds.Length - at);
            await ForwardAsync(promptIds[at..(at + n)], null, decode: false, temperature, readBack: false, ct).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Most prompt rows one prefill forward runs; a longer prompt (or the new part of one) runs in chunks of this many. It bounds the row buffers; one
    /// submission's work is bounded by the per-layer submit of a WebGPU prefill (SpawnDev.AI lost the WebGPU device on a ~600-token turn over a ~3k-token
    /// history encoded as one command buffer while other applications loaded the GPU: Windows resets a GPU whose command buffer runs past the driver's
    /// timeout; 2026-11-01. The cause was the per-element GEMM and the split-K attention: TiledGemmImpl / FlashPrefillImpl). The chunks give
    /// the same rows as one forward (GgufBatchDecoder_ChunkedPrefill_MatchesOneForward). Default 1024: a WebGPU prefill also submits after every layer, which is what
    /// bounds one submission; smaller chunks only cost (MEASURED: 600 rows over 3k, 811 ms in one forward, 1061 ms in chunks of 256).
    /// </summary>
    public int PrefillChunk { get; set; } = 1024;

    /// <summary>A prefill's attention on the tiled kernel (default; groups with shared memory, head width up to 128): off, the three-kernel path (scores, softmax,
    /// P.V), the reference the tiled kernel is gated against (GgufBatchDecoder_FlashPrefill_MatchesThreeKernelPath).</summary>
    public bool FlashPrefill { get; set; } = true;
    /// <summary>
    /// A prefill's attention on <c>FlashPrefillGqaImpl</c> (default; env GGUF_FLASH_GQA=0 for the older tiled kernel): 32 query vectors a group (the rows of a block
    /// times the query heads that share one kv head), 32 keys a tile, so each K / V row is read once for all of them. Taken when the head width is a multiple of 32 up
    /// to 128 and a kv head has at most 32 query heads. MEASURED 2026-11-02 before it (CUDA, RTX 4070, llama3.1:8b): the older kernel spent 8.5 s of an 18.3 s
    /// 4,400-token prefill (~0.6 TFLOPS) and 59% of a 194-row prefill over a reused 4,206-row prefix.
    /// </summary>
    public bool FlashGqa { get; set; } = Environment.GetEnvironmentVariable("GGUF_FLASH_GQA") != "0";
    /// <summary>
    /// A prefill's attention on the tensor cores (<see cref="MmaFlashPrefill"/>, default; env GGUF_FLASH_MMA=0 for <see cref="FlashGqa"/>'s kernel): CUDA sm_75+,
    /// head dim 128, F16 caches. MEASURED 2026-11-02 before it (RTX 4070, llama3.1:8b, 4,400-token prefill): the tiled SIMT kernel's 1,643 ms were 40% of it.
    /// </summary>
    public bool FlashMma { get; set; } = Environment.GetEnvironmentVariable("GGUF_FLASH_MMA") != "0";
    /// <summary>The kernel the last prefill's attention ran on: "mma", "gqa-tiled", "tiled", or null before any flash prefill.</summary>
    public string? LastFlashKernel { get; private set; }
    /// <summary>The split-K decode attention kernel the last fused decode step ran: "gqa-vec[-ilp][-shuffle]", "gqa-ilp[-shuffle]", "gqa" or "per-head" (null before one).</summary>
    public string? LastDecodeAttentionKernel { get; private set; }
    /// <summary>Whether the last step's final norm wrote its rows as q8_1 for the opt-in int8-activation GEMVs (<see cref="FusedDequantMatMul.Int8ActFor"/>): tests check the fused route ran.</summary>
    public bool LastStepInt8Fused => _hnQ8 != null;

    /// <summary>Cache rows the last <see cref="PrefillAsync(int[], float, bool, CancellationToken)"/> kept from the generation before it (0: a full prefill).</summary>
    public int LastReusedPrefix { get; private set; }

    /// <summary>The accelerator the decoder runs on.</summary>
    public Accelerator Accelerator => _acc;

    /// <summary>Per sequence of the last <see cref="GenerateAsync"/>: the stop token that ended it, or -1 (the length, <c>shouldContinue</c>, or a cancel).</summary>
    public int[] LastStopTokens { get; private set; } = Array.Empty<int>();

    /// <summary>
    /// True when <paramref name="model"/> can load here (checked on the header, before any weight is read): the architecture (qwen2, qwen3, llama), rotary over
    /// the whole head (per-frequency factors, llama 3's rope_freqs, as F32 with one per pair), a vocabulary below 2^24, and every matrix in a block type the fused GEMM reads
    /// (norms F32). <paramref name="reason"/> says what does not fit.
    /// </summary>
    public static bool Supports(GGUFModel model, out string reason)
    {
        string a = model.Architecture;
        if (a is not ("qwen2" or "qwen3" or "llama")) { reason = $"architecture '{a}' (supported: qwen2, qwen3, llama)"; return false; }
        int hidden = (int)model.EmbeddingLength, heads = (int)model.AttentionHeadCount;
        int headDim = (int)model.GetMetadataInt($"{a}.attention.key_length", heads > 0 ? hidden / heads : 0);
        if ((int)model.GetMetadataInt($"{a}.rope.dimension_count", headDim) != headDim) { reason = "rotary over part of the head"; return false; }
        if (!RopeFactorsFit(model, headDim, out reason)) return false;
        var embd = model.Tensors.FirstOrDefault(t => t.Name == "token_embd.weight");
        if (embd == null) { reason = "no token_embd.weight"; return false; }
        if (embd.Dimensions[1] >= 1 << 24) { reason = $"vocabulary {embd.Dimensions[1]} (candidate ids travel as floats, exact below 2^24)"; return false; }
        foreach (var t in model.Tensors)
        {
            bool block = t.Name.StartsWith("blk.", StringComparison.Ordinal), vec = t.Dimensions.Length == 1 || t.Name.EndsWith(".bias", StringComparison.Ordinal);
            bool head = t.Name is "output.weight" or "token_embd.weight";
            if (!(block || head)) continue;
            if (vec) { if (!IsVectorType(t.Type)) { reason = $"{t.Name}: {t.Type} (vectors are read as F32, F16 or BF16)"; return false; } continue; }
            if (!FusedDequantMatMul.Accepts(t.Type)) { reason = $"{t.Name}: {t.Type} is not a type the fused GEMM reads"; return false; }
        }
        reason = "";
        return true;
    }

    // the tokens whose K/V rows sequence 0's cache region holds, rows 0..length-1 (set by GenerateAsync when it ends; null: nothing to reuse). A generation's
    // last token is never in it: drawing token k needs token k - 1 as an input, so k tokens prove k - 1 of them reached the cache (a step queued ahead may have
    // written one more row, which a later prefill simply overwrites)
    private int[]? _cacheIds;

    /// <summary>Copy sequence 0's cache (the prompt) into sequences 1..count-1: <paramref name="count"/> live sequences from now on, all at the same position.</summary>
    public void Fork(int count)
    {
        if (count < 1 || count > MaxBatch) throw new ArgumentOutOfRangeException(nameof(count), $"1..{MaxBatch}");
        // a kernel, not CopyFrom: the regions share one buffer, and WebGPU refuses a buffer-to-buffer copy within one buffer
        LoadKernels();
        long n = (long)Position * KvHeads * HeadDim, region = (long)MaxSeq * KvHeads * HeadDim;
        if (count > 1 && n > 0)
            for (int l = 0; l < Layers; l++)
            {
                _kv.Fork(l, (int)((count - 1) * n), n, region);
            }
        Batch = count;
    }

    /// <summary>One token per live sequence (<paramref name="tokens"/>.Length == <see cref="Batch"/>) at <see cref="Position"/>; <paramref name="temperature"/> as for <see cref="PrefillAsync"/>.</summary>
    public Task StepAsync(int[] tokens, float temperature = 0f, CancellationToken ct = default)
    {
        if (tokens.Length != Batch) throw new ArgumentException($"{Batch} tokens expected (one per live sequence)");
        return ForwardAsync(tokens, Enumerable.Range(0, Batch).ToArray(), decode: true, temperature, readBack: true, ct);
    }

    /// <summary>
    /// One token for each of <paramref name="sequences"/> (distinct, each below <see cref="Batch"/>) at <see cref="Position"/>: row r of the step is sequence
    /// <paramref name="sequences"/>[r] (its <see cref="SampleAsync"/> row). The others are left out of the step (finished sequences stop costing compute), their caches
    /// untouched; a sequence left out falls behind <see cref="Position"/> and cannot be resumed.
    /// </summary>
    public Task StepAsync(int[] tokens, int[] sequences, float temperature = 0f, CancellationToken ct = default)
    {
        if (tokens.Length != sequences.Length) throw new ArgumentException("one token per sequence");
        CheckSequences(sequences);
        return ForwardAsync(tokens, sequences, decode: true, temperature, readBack: true, ct);
    }

    private void CheckSequences(int[] sequences)
    {
        if (sequences.Length == 0) throw new ArgumentException("at least one sequence");
        if (sequences.Any(s => s < 0 || s >= Batch) || sequences.Distinct().Count() != sequences.Length) throw new ArgumentException($"distinct sequences below {Batch}");
    }

    /// <summary>The last forward's logits [rows, Vocab] row-major (a prefill: one row), read from the device.</summary>
    public async Task<float[]> ReadLogitsAsync() => await _lastLogits.CopyToHostAsync().ConfigureAwait(false);

    /// <summary>
    /// One token for row <paramref name="row"/> of the last forward, at the temperature that forward was given: argmax when 0, else <see cref="SampleLogits(float[], int, int, float, float, Random)"/>'s
    /// draw, taken from the device's candidates (the same set and the same draw; the row is read only when its nucleus is wider than <see cref="Candidates"/>).
    /// The first draw after a forward reads its candidates (that forward's one wait).
    /// </summary>
    public ValueTask<int> SampleAsync(int row, float topP, Random rng) => SampleAsync(row, topP, rng, 0);

    /// <summary>
    /// <see cref="SampleAsync(int, float, Random)"/> with top-k first (<see cref="SampleLogits(float[], int, int, float, float, double, int)"/>'s rule): k up to
    /// <see cref="Candidates"/> is drawn from the candidates; a larger k, or a row whose candidates overflowed, reads the row.
    /// </summary>
    public ValueTask<int> SampleAsync(int row, float topP, Random rng, int topK) => SampleAsync(row, topP, rng, topK, 0f);

    /// <summary><see cref="SampleAsync(int, float, Random, int)"/> then min-p (<see cref="SampleLogits(float[], int, int, float, float, double, int, float)"/>'s rule).</summary>
    public async ValueTask<int> SampleAsync(int row, float topP, Random rng, int topK, float minP)
    {
        if ((uint)row >= (uint)_lastRows) throw new ArgumentOutOfRangeException(nameof(row));
        if (NoCandidates) return SampleLogits(await _lastLogits.SubView((long)row * Vocab, Vocab).CopyToHostAsync().ConfigureAwait(false), 0, Vocab, _lastT, topP, rng.NextDouble(), topK, minP);
        int K = Candidates, o = row * CandStride;
        var c = _lastCand ??= await _lastCandView.CopyToHostAsync().ConfigureAwait(false);
        float T = _lastT, max = c[o], floor = max - 20f * T;
        double total = 0, keep = 0; int m = 0;
        bool whole = c[o + 2 * K] > SelectCap || topK > K;                       // too many tokens reached the floor (the candidates are not the top K), or k past them
        if (!whole)
        {
            if (T <= 0f) return (int)c[o + K];
            if (topK > 0)
            {
                // the k best within the window are the first candidates: the nucleus runs over their mass only
                int kk = 0;
                while (kk < topK && c[o + kk] >= floor) total += Math.Exp((c[o + kk++] - max) / T);
                while (m < kk && keep < topP * total) { keep += Math.Exp((c[o + m] - max) / T); m++; }
            }
            else
            {
                for (int ch = 0; ch < Chunks; ch++) total += c[o + 2 * K + 1 + ch];
                while (m < K && c[o + m] >= floor && keep < topP * total) { keep += Math.Exp((c[o + m] - max) / T); m++; }
                whole = keep < topP * total;                                     // the nucleus reaches past the candidates (or the device's float sums end it a hair later)
            }
        }
        if (whole)
        {
            RowReads++;
            var full = await _lastLogits.SubView((long)row * Vocab, Vocab).CopyToHostAsync().ConfigureAwait(false);
            return SampleLogits(full, 0, Vocab, T, topP, rng.NextDouble(), topK, minP);
        }
        float minFloor = MinPFloor(max, T, minP);
        while (m > 1 && c[o + m - 1] < minFloor) m--;
        if (minP > 0f) { keep = 0; for (int i = 0; i < m; i++) keep += Math.Exp((c[o + i] - max) / T); }
        double u = rng.NextDouble() * keep, acc = 0;
        for (int i = 0; i < m; i++) { acc += Math.Exp((c[o + i] - max) / T); if (u < acc) return (int)c[o + K + i]; }
        return (int)c[o + K + m - 1];
    }

    // ---------------------------------------------------------------------------------------------------
    // device draws: the token is drawn where the logits are, so a step's input never crosses to the host and back; the host reads the tokens to stream and stop
    // them, one step behind (GenerateAsync)
    // ---------------------------------------------------------------------------------------------------

    /// <summary>
    /// <see cref="GenerateAsync"/> draws on the device and keeps the next step queued while it reads a step's tokens (no wait on the step's critical path); false:
    /// every token is drawn on the host by <see cref="SampleAsync"/> with a <see cref="Random"/> per sequence (seed + index), a round trip per step. On by default
    /// where the draw kernel's groups exist (not WebGL).
    /// </summary>
    public bool DeviceSampling { get; set; }
    /// <summary>Diagnostics: device draws take the full-row path (the bisection over every logit) even when the candidates hold the nucleus. Read by <see cref="Draw"/>.</summary>
    public bool ForceFullRowDraws { get; set; }
    private int _drawPos = -1;
    private bool _lastPrefill;

    /// <summary>
    /// The uniform sequence <paramref name="sequence"/> draws its token for <paramref name="position"/> with under <paramref name="seed"/>: a counter hash (three
    /// rounds of lowbias32), 24 bits, in [0, 1). The device computes the same float; passed to <see cref="SampleLogits(float[], int, int, float, float, double)"/> it
    /// gives the host's draw from the same logits.
    /// </summary>
    public static float DrawUniform(int seed, int sequence, int position)
    {
        uint h = Mix32((uint)seed ^ 0x9E3779B9u);
        h = Mix32(h ^ ((uint)sequence * 0x85EBCA6Bu));
        h = Mix32(h ^ ((uint)position * 0xC2B2AE35u));
        return (h >> 8) * (1f / 16777216f);
    }
    private static uint Mix32(uint x)
    {
        x ^= x >> 16; x *= 0x7FEB352Du;
        x ^= x >> 15; x *= 0x846CA68Bu;
        x ^= x >> 16;
        return x;
    }

    /// <summary>
    /// Draw every live sequence's next token on the device from the last forward (after <see cref="PrefillAsync"/> and <see cref="Fork"/>: all of them from the
    /// prompt's row; after a step: each row for its sequence), at the temperature that forward was given (0: the argmax), nucleus <paramref name="topP"/> (0, 1],
    /// uniform <see cref="DrawUniform"/>(<paramref name="seed"/>, sequence, <see cref="Position"/>). Queued, no wait: the tokens stay on the device as the next
    /// <see cref="StepDrawnAsync"/>'s input, <see cref="ReadDrawnAsync"/> reads them. <paramref name="seed"/> and <paramref name="topP"/> hold for the steps after.
    /// <para>
    /// <paramref name="repetitionPenalty"/> (1 = off) applies to the drawn steps after this one, per sequence, over its last <paramref name="repeatLastN"/> drawn tokens
    /// (0 or less: all of them; this draw starts the window): each distinct token's logit is divided by the penalty when positive, multiplied when not
    /// (<see cref="TextGenerationSampler.ApplyRepetitionPenalty"/>), before the candidates and the draw.
    /// </para>
    /// </summary>
    public void Draw(float topP, int seed, float repetitionPenalty = 1f, int repeatLastN = 64, int topK = 0, float minP = 0f, float presencePenalty = 0f)
    {
        if (_lastRows == 0) throw new InvalidOperationException("no forward to draw from");
        if (!SupportsDeviceDraws(_acc)) throw new NotSupportedException($"device draws need groups of {DrawGroup} threads ({_acc.AcceleratorType}): use SampleAsync");
        if (!(topP > 0f && topP <= 1f)) throw new ArgumentOutOfRangeException(nameof(topP), "(0, 1]");
        if (!(repetitionPenalty > 0f)) throw new ArgumentOutOfRangeException(nameof(repetitionPenalty), "above 0");
        if (topK < 0) throw new ArgumentOutOfRangeException(nameof(topK), "0 (off) or more");
        if (!(minP >= 0f && minP < 1f)) throw new ArgumentOutOfRangeException(nameof(minP), "[0, 1)");
        LoadKernels(); EnsureDrawBuffers();
        int window = repeatLastN > 0 ? repeatLastN : MaxSeq;
        EnsurePenaltyWindow(window);
        _rep!.View.SubView(0, MaxBatch).MemSetToZero();                          // every sequence's window starts empty
        _drawI!.View.SubView(0, 6).CopyFromCPU(new[] { seed, ForceFullRowDraws ? 1 : 0, window, _repCap, MaxBatch, topK });
        _drawF!.View.SubView(0, 4).CopyFromCPU(new[] { topP, repetitionPenalty, minP, presencePenalty });
        _penOn = repetitionPenalty != 1f || presencePenalty != 0f;
        var seqs = _lastPrefill ? Enumerable.Range(0, Batch).ToArray() : _lastSeqs;
        if (!seqs.AsSpan().SequenceEqual(_seqsOnDevice)) { _seqBuf!.View.SubView(0, seqs.Length).CopyFromCPU(seqs); _seqsOnDevice = (int[])seqs.Clone(); }
        QueueDraw(seqs.Length, rowZero: _lastPrefill, tAdd: Position - _lastP, _lastT, _lastLogits, _lastCandView);
        CopyDrawnToRing();
    }

    /// <summary>
    /// One step for <paramref name="sequences"/> (distinct, below <see cref="Batch"/>) whose inputs are the tokens they drew last (<see cref="Draw"/> or the step before),
    /// ending with their next draws (as <see cref="Draw"/>: its seed and top-p, this <paramref name="temperature"/>). Queued, no wait (WebGPU: a captured step's replay).
    /// </summary>
    public async Task StepDrawnAsync(int[] sequences, float temperature, CancellationToken ct = default)
    {
        CheckSequences(sequences);
        if (_drawPos != Position) throw new InvalidOperationException("the sequences have no drawn token at this position: Draw after the prefill");
        await ForwardAsync(null, sequences, decode: true, temperature, readBack: false, ct).ConfigureAwait(false);
        CopyDrawnToRing();
    }

    // diagnostics (the gates): sequence's penalty window as the device holds it, oldest first (the last min(drawn, repeatLastN, ring) tokens it drew)
    internal async Task<int[]> ReadPenaltyWindowAsync(int sequence)
    {
        var all = await _rep!.View.CopyToHostAsync().ConfigureAwait(false);
        var cfg = await _drawI!.View.SubView(0, 5).CopyToHostAsync().ConfigureAwait(false);
        int W = cfg[3], MB = cfg[4], c = all[sequence], n = Math.Min(Math.Min(c, cfg[2]), W);
        var w = new int[n];
        for (int k = 0; k < n; k++) w[n - 1 - k] = all[MB + sequence * W + (c - 1 - k) % W];
        return w;
    }

    // the drawn tokens into the next read ring (a device copy, queued after the draw)
    private void CopyDrawnToRing()
    {
        _ringAt ^= 1;
        _ring[_ringAt]!.View.SubView(0, MaxBatch).CopyFrom(_seqIds!.View.SubView(0, MaxBatch));
        _drawPos = Position;
    }

    /// <summary>
    /// The tokens the last draw gave, by sequence (length <see cref="MaxBatch"/>; a sequence that was not drawn keeps its last token). The read is queued when this is
    /// called and waits only for the draw: it may be awaited after the next <see cref="StepDrawnAsync"/> is queued (at most two reads unawaited).
    /// </summary>
    public Task<int[]> ReadDrawnAsync() => _ring[_ringAt]!.View.SubView(0, MaxBatch).CopyToHostAsync();

    /// <summary>
    /// <paramref name="n"/> continuations of <paramref name="promptIds"/> decoded together: prefill once, fork, then one batched step per token: greedy when
    /// <paramref name="temperature"/> &lt;= 0, else temperature then nucleus (top-p), each sequence with its own uniforms (<see cref="DeviceSampling"/>:
    /// <see cref="DrawUniform"/>(<paramref name="seed"/>, sequence, position) on the device; off: a <see cref="Random"/>(seed + sequence) on the host). A sequence ends
    /// at a stop token or after <paramref name="maxNew"/> tokens and then leaves the batch (the steps after it run the live sequences only). Returns the generated token
    /// ids per sequence (stop token excluded). <paramref name="onToken"/> gets (sequence, token) as they are sampled, <paramref name="onFinished"/> the sequence when it ends.
    /// <paramref name="repetitionPenalty"/> / <paramref name="repeatLastN"/>: as for <see cref="Draw"/>, over each sequence's own generated tokens (the host-draw path
    /// reads the rows to apply it).
    /// </summary>
    public async Task<int[][]> GenerateAsync(int[] promptIds, int n, int maxNew, float temperature, float topP, int seed, IReadOnlyCollection<int> stopIds,
        Action<int, int>? onToken = null, Action<int>? onFinished = null, CancellationToken ct = default, float repetitionPenalty = 1f, int repeatLastN = 64,
        bool reusePrefix = false, Func<int, int, ValueTask<bool>>? shouldContinue = null, int topK = 0, float minP = 0f, float presencePenalty = 0f)
    {
        await PrefillAsync(promptIds, temperature, reusePrefix, ct).ConfigureAwait(false);
        Fork(n);
        var outs = Enumerable.Range(0, n).Select(_ => new List<int>()).ToArray();
        try { return await GenerateForkedAsync(outs, n, maxNew, temperature, topP, seed, stopIds, onToken, onFinished, ct, repetitionPenalty, repeatLastN, shouldContinue, topK, minP, presencePenalty).ConfigureAwait(false); }
        finally
        {
            // what sequence 0's cache rows now hold, for the next reusePrefix prefill (also after a cancel: the rows written so far are valid)
            var o = outs[0];
            _cacheIds = o.Count > 1 ? promptIds.Concat(o.Take(o.Count - 1)).ToArray() : promptIds;
        }
    }

    /// <summary>
    /// The device-draw loop hands out a token read that has already completed before it queues the next step (the desktop backends' reads are synchronous):
    /// the first token no longer waits for the first decode step's encode and run. Env GGUF_HANDOUT_FIRST=0 restores queue-first (the A/B).
    /// </summary>
    public static bool HandOutCompletedFirst = Environment.GetEnvironmentVariable("GGUF_HANDOUT_FIRST") != "0";

    private async Task<int[][]> GenerateForkedAsync(List<int>[] outs, int n, int maxNew, float temperature, float topP, int seed, IReadOnlyCollection<int> stopIds,
        Action<int, int>? onToken, Action<int>? onFinished, CancellationToken ct, float repetitionPenalty, int repeatLastN, Func<int, int, ValueTask<bool>>? shouldContinue, int topK,
        float minP = 0f, float presencePenalty = 0f)
    {
        var done = new bool[n];
        LastStopTokens = Enumerable.Repeat(-1, n).ToArray();
        // a token for sequence s: true when the sequence goes on (a stop token, the length, or shouldContinue saying no ends it; the stop token is not kept)
        async ValueTask<bool> Take(int s, int token)
        {
            if (stopIds.Contains(token)) { LastStopTokens[s] = token; done[s] = true; onFinished?.Invoke(s); return false; }
            if (outs[s].Count >= maxNew) { done[s] = true; onFinished?.Invoke(s); return false; }
            outs[s].Add(token); onToken?.Invoke(s, token);
            if (shouldContinue != null && !await shouldContinue(s, token).ConfigureAwait(false)) { done[s] = true; onFinished?.Invoke(s); return false; }
            return true;
        }
        void FinishAll() { for (int s = 0; s < n; s++) if (!done[s]) { done[s] = true; onFinished?.Invoke(s); } }
        // host draws when the device does not draw (WebGL; DeviceSampling off): the candidates are read each step (~KB), the row only when the nucleus passes them
        if (!DeviceSampling)
        {
            var rngs = Enumerable.Range(0, n).Select(s => new Random(seed + s)).ToArray();
            var next = new int[n];
            for (int s = 0; s < n; s++) next[s] = await SampleAsync(0, topP, rngs[s], topK, minP).ConfigureAwait(false);
            while (true)
            {
                for (int s = 0; s < n; s++) if (!done[s]) await Take(s, next[s]).ConfigureAwait(false);
                if (done.All(d => d)) break;
                if (Position + 1 > MaxSeq) { FinishAll(); break; }
                ct.ThrowIfCancellationRequested();
                // only the live sequences step: a finished one leaves the batch (it used to ride along to the end of the longest)
                var live = Enumerable.Range(0, n).Where(s => !done[s]).ToArray();
                await StepAsync(live.Select(s => next[s]).ToArray(), live, temperature, ct).ConfigureAwait(false);
                if (repetitionPenalty == 1f && presencePenalty == 0f)
                    for (int r = 0; r < live.Length; r++) next[live[r]] = await SampleAsync(r, topP, rngs[live[r]], topK, minP).ConfigureAwait(false);
                else
                {
                    // the penalty changes the row: read it, penalise each sequence's own window, draw from the whole row (the device path's reference)
                    var rowsAll = await ReadLogitsAsync().ConfigureAwait(false);
                    for (int r = 0; r < live.Length; r++)
                    {
                        var o = outs[live[r]];
                        int take = repeatLastN > 0 ? Math.Min(repeatLastN, o.Count) : o.Count;
                        var row = new float[Vocab];
                        Array.Copy(rowsAll, (long)r * Vocab, row, 0, Vocab);
                        var window = o.GetRange(o.Count - take, take).ToArray();
                        TextGenerationSampler.ApplyRepetitionPenalty(row, window, repetitionPenalty);
                        TextGenerationSampler.ApplyPresencePenalty(row, window, presencePenalty);
                        next[live[r]] = SampleLogits(row, 0, Vocab, temperature, topP, rngs[live[r]].NextDouble(), topK, minP);
                    }
                }
            }
            return outs.Select(o => o.ToArray()).ToArray();
        }
        // device draws, one step ahead: step P (its inputs the tokens drawn for P, its draws the tokens for P + 1) is queued before the tokens for P are read, so the
        // device always has the next step while the host streams the last one (an asynchronous read: WebGPU; a read that has already completed is handed out first,
        // below). A sequence that drew a stop token rides one step more on that overlapped path (its row is thrown away), then leaves the batch.
        Draw(temperature > 0f ? topP : 1f, seed, repetitionPenalty, repeatLastN, temperature > 0f ? topK : 0, temperature > 0f ? minP : 0f, presencePenalty);
        var live2 = Enumerable.Range(0, n).ToArray();
        Task<int[]>? pending = ReadDrawnAsync(), ahead = null;
        try
        {
            while (true)
            {
                ct.ThrowIfCancellationRequested();
                // a read that has ALREADY completed is handed out before the next step is queued: the desktop backends read synchronously (the read IS the wait),
                // so queuing first made every token wait for the next step's encode AND its run, the first token included: time to first text paid a whole
                // uncaptured decode step (nsys 2026-11-11, RTX 4070, qwen3:1.7b, 29-token prompt: ~3.6 ms of a 12.5 ms start). The GPU idles across the
                // hand-out either way there; an asynchronous read (WebGPU) that is still in flight keeps the one-step-ahead overlap below. The draws are keyed
                // by (seed, sequence, position), so stepping only the still-live sequences one step sooner changes no token.
                bool handed = false;
                if (HandOutCompletedFirst && pending!.IsCompleted)
                {
                    var t0 = await pending.ConfigureAwait(false);
                    pending = null; handed = true;
                    foreach (int s in live2) await Take(s, t0[s]).ConfigureAwait(false);
                    live2 = live2.Where(s => !done[s]).ToArray();
                    if (live2.Length == 0) break;
                }
                ahead = null;
                if (Position + 1 <= MaxSeq) { await StepDrawnAsync(live2, temperature, ct).ConfigureAwait(false); ahead = ReadDrawnAsync(); }
                if (!handed)
                {
                    var tokens = await pending!.ConfigureAwait(false);
                    pending = null;
                    foreach (int s in live2) await Take(s, tokens[s]).ConfigureAwait(false);
                    live2 = live2.Where(s => !done[s]).ToArray();
                    if (live2.Length == 0) break;
                }
                if (ahead == null) { FinishAll(); break; }
                pending = ahead; ahead = null;
            }
        }
        finally
        {
            // the step queued ahead is still on the device: wait for its read, so the decoder is idle (and its staging unmapped) when this returns
            foreach (var t in new[] { pending, ahead })
                if (t != null) try { await t.ConfigureAwait(false); } catch { }
        }
        return outs.Select(o => o.ToArray()).ToArray();
    }

    /// <summary>
    /// One token from row <paramref name="row"/> of [rows, Vocab] logits: argmax when <paramref name="temperature"/> &lt;= 0; else softmax(logits / temperature) over the
    /// tokens within e^20 of the best (the rest carry under 2e-9 of the mass each), taken best first (ties: the lower id), the smallest set whose mass reaches
    /// <paramref name="topP"/>, renormalised and drawn. <paramref name="vocab"/>: the row length.
    /// </summary>
    public static int SampleLogits(float[] logits, int row, int vocab, float temperature, float topP, Random rng) =>
        SampleLogits(logits, row, vocab, temperature, topP, rng.NextDouble());

    /// <summary>
    /// <see cref="SampleLogits(float[], int, int, float, float, Random)"/> with the uniform given: <paramref name="u"/> in [0, 1) (the device draws: <see cref="DrawUniform"/>).
    /// </summary>
    public static int SampleLogits(float[] logits, int row, int vocab, float temperature, float topP, double u) => SampleLogits(logits, row, vocab, temperature, topP, u, 0);

    /// <summary>
    /// <see cref="SampleLogits(float[], int, int, float, float, double)"/> with top-k first: <paramref name="topK"/> &gt; 0 keeps the k best of the tokens within e^20
    /// of the best (ties: the lower id), then the nucleus of <paramref name="topP"/> over their mass (1: all k), renormalised and drawn (Hugging Face's order).
    /// </summary>
    public static int SampleLogits(float[] logits, int row, int vocab, float temperature, float topP, double u, int topK) => SampleLogits(logits, row, vocab, temperature, topP, u, topK, 0f);

    /// <summary>The min-p floor at temperature <paramref name="T"/>: p / p_max &gt;= minP exactly when logit &gt;= max + T ln(minP) (the device computes the same float).</summary>
    public static float MinPFloor(float max, float T, float minP) => minP > 0f ? max + T * MathF.Log(minP) : float.NegativeInfinity;

    /// <summary>
    /// <see cref="SampleLogits(float[], int, int, float, float, double, int)"/> then min-p: of the kept tokens, only those with logit at or above <see cref="MinPFloor"/> stay
    /// (the best one always), renormalised and drawn (Hugging Face's order: temperature, top-k, top-p, min-p).
    /// </summary>
    public static int SampleLogits(float[] logits, int row, int vocab, float temperature, float topP, double u, int topK, float minP)
    {
        int V = vocab, o = row * V, best = 0;
        for (int i = 1; i < V; i++) if (logits[o + i] > logits[o + best]) best = i;
        if (temperature <= 0f) return best;
        float max = logits[o + best], floor = max - 20f * temperature;
        var cand = new List<(int Id, float L, double P)>();
        for (int i = 0; i < V; i++) if (logits[o + i] >= floor) cand.Add((i, logits[o + i], Math.Exp((logits[o + i] - max) / temperature)));
        cand.Sort((a, b) => a.L != b.L ? b.L.CompareTo(a.L) : a.Id.CompareTo(b.Id));
        if (topK > 0 && cand.Count > topK) cand.RemoveRange(topK, cand.Count - topK);
        double total = cand.Sum(c => c.P), keep = 0; int m = 0;
        while (m < cand.Count && keep < topP * total) keep += cand[m++].P;
        float minFloor = MinPFloor(max, temperature, minP);
        while (m > 1 && cand[m - 1].L < minFloor) m--;
        if (minP > 0f) { keep = 0; for (int i = 0; i < m; i++) keep += cand[i].P; }       // summed again in order (the running sum's rounding)
        double x = u * keep, acc = 0;
        for (int i = 0; i < m; i++) { acc += cand[i].P; if (x < acc) return cand[i].Id; }
        return cand[m - 1].Id;
    }

    // ---------------------------------------------------------------------------------------------------
    // kernels. Row r: prefill (mode 0) = sequence 0 at position p + r; decode (mode 1) = sequence seqs[r] at position p.
    // Caches: [sequence][MaxSeq][kvHeads * headDim].
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, FV, FV>? _add3;
    private Action<Index1D, FV, FV, int>? _bias;
    private Action<Index1D, FV, FV, IV, int, int, int>? _rope;
    private Action<Index1D, FV, FV, int>? _softmaxStats, _softmaxNorm;
    private Action<Index1D, FV, FV>? _swiglu;
    private Action<Index1D, FV, FV, int>? _swigluJoined;
    private Action<Index1D, FV, FV, ArrayView1D<byte, Stride1D.Dense>, FV, int>? _swigluJoinedQ8;
    private Action<KernelConfig, FV, FV, int, int, int, float>? _headNorm;
    private Action<KernelConfig, FV, FV, FV, int, int, int, float>? _headNormQk;
    private Action<Index1D, FV, FV, FV, IV, int, int, int, int, int>? _attCombine;
    private Action<KernelConfig, FV, FV, FV, IV, int, int, int, int>? _attCombineGroup;
    private Action<Index1D, IV, ArrayView1D<int, Stride1D.Dense>, FV, int, int, int>? _embed;
    private Action<Index1D, FV, FV, IV, int, int, int>? _chunkMax;
    private Action<KernelConfig, FV, FV, IV, int, int>? _chunkMaxGroup;
    private Action<KernelConfig, FV, FV, IV, FV, IV, int, int, int, float>? _selectGroup;
    private Action<KernelConfig, FV, IV, FV, IV, int, int>? _floorGroup;
    private Action<Index1D, FV, IV, FV, IV, int, int>? _floor;
    private Action<Index1D, FV, FV, IV, FV, IV, int, int, int, float, int>? _select;
    private Action<Index1D, FV, IV, IV, FV, int>? _final;
    private Action<KernelConfig, FV, FV, IV, IV, FV, IV, IV, IV, int, int, int, int, int, float>? _draw;
    private Action<Index1D, FV, IV, IV, FV, IV, int>? _penalize;
    private Action<Index1D, IV, IV, IV>? _gather;

    private void LoadKernels()
    {
        if (_add3 != null) return;
        _add3 = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV>(Add3Impl);
        _bias = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int>(BiasImpl);
        _rope = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, int, int, int>(RopeImpl);
        _softmaxStats = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int>(SoftmaxStatsImpl);
        _softmaxNorm = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int>(SoftmaxNormImpl);
        _swiglu = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(SwigluImpl);
        _swigluJoined = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int>(SwigluJoinedImpl);
        if (_qkNorm)
        {
            _headNorm = _acc.LoadStreamKernel<FV, FV, int, int, int, float>(HeadNormImpl);
            _headNormQk = _acc.LoadStreamKernel<FV, FV, FV, int, int, int, float>(HeadNormQkImpl);
        }
        if (FusedAttention)
        {
            _attCombine = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, IV, int, int, int, int, int>(AttnCombineImpl);
            if (HeadDim <= CombineMaxD && (HeadDim & (HeadDim - 1)) == 0 && HeadDim <= _acc.MaxNumThreadsPerGroup)
                _attCombineGroup = _acc.LoadStreamKernel<FV, FV, FV, IV, int, int, int, int>(AttnCombineGroupImpl);
        }
        _hasFlash = _acc.AcceleratorType != AcceleratorType.WebGL && HeadDim <= FaMaxD && FaThreads <= _acc.MaxNumThreadsPerGroup;
        // the tensor-core flash prefill: CUDA sm_75+, head dim 128 (the F16-cache check is the store's: FlashMma returns false on F32 caches)
        _hasFlashMma = _hasFlash && HeadDim == MmaFlashPrefill.HeadDim && Heads % KvHeads == 0 && CudaCaps.TuringTensorCores(_acc);
        _hasFlashGqa = _hasFlash && HeadDim % FgDc == 0 && HeadDim <= FgMaxD && Heads % KvHeads == 0 && Heads / KvHeads <= FgQ
            && FgThreads <= _acc.MaxNumThreadsPerGroup && FgShared * sizeof(float) <= _acc.MaxSharedMemoryPerGroup;
        _embed = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, ArrayView1D<int, Stride1D.Dense>, FV, int, int, int>(EmbedImpl);
        if (!NoCandidates)
        {
            _chunkMax = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, int, int, int>(SpawnDev.ILGPU.ML.Kernels.TopKKernels.ChunkMaxImpl);
            if (TopKGroupFits(_acc))
            {
                _chunkMaxGroup = _acc.LoadStreamKernel<FV, FV, IV, int, int>(SpawnDev.ILGPU.ML.Kernels.TopKKernels.ChunkMaxGroupImpl);
                _selectGroup = _acc.LoadStreamKernel<FV, FV, IV, FV, IV, int, int, int, float>(SpawnDev.ILGPU.ML.Kernels.TopKKernels.SelectGroupImpl);
                if (Chunks <= SpawnDev.ILGPU.ML.Kernels.TopKKernels.FloorGroupMaxChunks) _floorGroup = _acc.LoadStreamKernel<FV, IV, FV, IV, int, int>(SpawnDev.ILGPU.ML.Kernels.TopKKernels.FloorGroupImpl);
            }
            _floor = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, FV, IV, int, int>(SpawnDev.ILGPU.ML.Kernels.TopKKernels.FloorImpl);
            _select = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, FV, IV, int, int, int, float, int>(SpawnDev.ILGPU.ML.Kernels.TopKKernels.SelectImpl);
            _final = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, FV, int>(SpawnDev.ILGPU.ML.Kernels.TopKKernels.FinalImpl);
        }
    }

    // o[r] = x[r] / rms(x[r]) * w (Qwen2RMSNorm: mean of squares + eps)
    private static void Add3Impl(Index1D i, FV a, FV b, FV o) => o[i] = a[i] + b[i];
    private static void BiasImpl(Index1D i, FV x, FV bias, int C) => x[i] += bias[(int)i % C];
    // gate[i] = silu(gate[i]) * up[i]
    private static void SwigluImpl(Index1D i, FV g, FV u) { float x = g[i]; g[i] = x / (1f + MathF.Exp(-x)) * u[i]; }
    // o[r, f] = silu(gu[r, f]) * gu[r, F + f] (the joined gate|up GEMV's rows)
    private static void SwigluJoinedImpl(Index1D i, FV gu, FV o, int F)
    {
        int ii = i, r = ii / F, f = ii % F;
        long b = (long)r * 2 * F;
        float x = gu[b + f];
        o[ii] = x / (1f + MathF.Exp(-x)) * gu[b + F + f];
    }
    // SwigluJoinedImpl that also writes the output as q8_1 (opt-in int8 activations): F a multiple of 32, an auto-grouped launch of R * F threads, so a warp's
    // lanes are one 32-value block of one row; the same float as SwigluJoinedImpl, the same bytes as the consumer's own quantize
    private static void SwigluJoinedQ8Impl(Index1D i, FV gu, FV o, ArrayView1D<byte, Stride1D.Dense> qs, FV ds, int F)
    {
        int ii = i, r = ii / F, f = ii - r * F;
        long b = (long)r * 2 * F;
        float x = gu[b + f];
        float y = x / (1f + MathF.Exp(-x)) * gu[b + F + f];
        o[ii] = y;
        FusedDequantMatMul.Q8_1WarpStore(y, ii, qs, ds);
    }
    // after the joined q|k(|v) GEMV (row stride sa): bias, rotary (NeoX pairs d, d + D/2) and the store, the arithmetic of BiasImpl + RopeImpl + StoreImpl. Per row the
    // threads are the q pairs, the k pairs, then v's values; q goes to qOut [r, C], k and v to their sequence's cache row; v comes from qkv (flags bit 0) or from vb
    // (row kvC); flags bit 1 = decode (row r is sequence seqs[r] at p; prefill: sequence 0 at p + r), bit 2 = rotary on adjacent pairs (llama); heads = H | kvH << 16
    private static void QkvPostImpl<T>(Index1D i, FV qkv, FV vb, FV bias, FV invFreq, FV qOut, ArrayView1D<T, Stride1D.Dense> kc, ArrayView1D<T, Stride1D.Dense> vc, IV seqs, IV posBuf, int heads, int D, int sa, int flags, int maxSeq) where T : unmanaged, System.Numerics.INumber<T>
    {
        int p = posBuf[0];
        int H = heads & 0xFFFF, kvH = heads >> 16, vInA = flags & 1, mode = (flags >> 1) & 1, adj = (flags >> 2) & 1;
        int half = D / 2, C = H * D, kvC = kvH * D, qPairs = H * half, kPairs = kvH * half;
        int U = qPairs + kPairs + kvC, ii = i, r = ii / U, u = ii % U;
        int seq = mode == 0 ? 0 : seqs[r], pos = mode == 0 ? p + r : p;
        long cacheRow = ((long)seq * maxSeq + pos) * kvC, row = (long)r * sa;
        if (u < qPairs + kPairs)
        {
            bool isQ = u < qPairs;
            int uu = isQ ? u : u - qPairs, hh = uu / half, kk = uu % half;
            int i0 = adj != 0 ? 2 * kk : kk, i1 = adj != 0 ? 2 * kk + 1 : kk + half;      // the pair: adjacent (llama) or split halves (qwen)
            int col = (isQ ? 0 : C) + hh * D;
            float x0 = qkv[row + col + i0] + bias[col + i0], x1 = qkv[row + col + i1] + bias[col + i1];
            float ang = pos * invFreq[kk], c = MathF.Cos(ang), s = MathF.Sin(ang);
            // the rotation with its fused multiply-adds written out (one product, then one FMA), identical in RopeImpl and QkvPostImpl: with FP contraction on
            // (CUDA, ML_FP_CONTRACT) the compiler fused `x0 * c - x1 * s` differently in the two kernels, so joined and separate projections drifted 2.6e-4
            // (GgufBatchDecoder_GreedyMatchesGenerator's joinOk, measured 2026-11-10; passes with ML_FP_CONTRACT=0)
            float y0 = float.MultiplyAddEstimate(x0, c, -(x1 * s)), y1 = float.MultiplyAddEstimate(x1, c, x0 * s);
            if (isQ) { long qo = (long)r * C + hh * D; qOut[qo + i0] = y0; qOut[qo + i1] = y1; }
            else { long ko = cacheRow + hh * D; kc[ko + i0] = PrecisionConvert.ConvertFromSingle<T>(y0); kc[ko + i1] = PrecisionConvert.ConvertFromSingle<T>(y1); }
        }
        else
        {
            int c2 = u - qPairs - kPairs;
            float x = vInA != 0 ? qkv[row + C + kvC + c2] : vb[(long)r * kvC + c2];
            vc[cacheRow + c2] = PrecisionConvert.ConvertFromSingle<T>(x + bias[C + kvC + c2]);
        }
    }

    // o[r, c] = token_embd[ids[r], c], decoded from its blocks
    private static void EmbedImpl(Index1D i, IV ids, ArrayView1D<int, Stride1D.Dense> w, FV o, int C, int code, int rowBytes)
    {
        int ii = i, r = ii / C, c = ii % C;
        o[ii] = FusedDequantMatMul.DecodeOf(code, w, ids[r] * rowBytes, c);
    }

    // candidate selection: Kernels/TopKKernels (shared with DeviceVectorIndex), at Chunk = 256

    // ---- device draws ----
    private const int DrawGroup = 256;

    /// <summary>True when <paramref name="accelerator"/> can run the draw kernel (a group of 256 threads per row; WebGL has no groups).</summary>
    public static bool SupportsDeviceDraws(Accelerator accelerator) =>
        accelerator.AcceleratorType != AcceleratorType.WebGL && accelerator.MaxNumThreadsPerGroup >= DrawGroup;

    /// <summary>
    /// The decoder's device draw on given logits, without a model (its gate): [<paramref name="rows"/>, <paramref name="vocab"/>] row-major through the candidate
    /// kernels and the draw kernel. Row r draws as sequence r for <paramref name="position"/> (uniform <see cref="DrawUniform"/>(<paramref name="seed"/>, r,
    /// <paramref name="position"/>)); <paramref name="forceFullRow"/> takes the whole-row path for every row. <paramref name="windows"/>[r]: row r's previous
    /// tokens, oldest first, penalised by <paramref name="repetitionPenalty"/> (the decoder's penalty kernel) before the draw: the last <paramref name="repeatLastN"/> of them (0: all
    /// the ring holds), the ring <paramref name="ringCapacity"/> tokens long (0: the longest window; shorter, the ring wraps as the decoder's does). Returns the token per row.
    /// </summary>
    public static async Task<int[]> DrawRowsAsync(Accelerator accelerator, float[] logits, int rows, int vocab, float temperature, float topP, int seed, int position, bool forceFullRow = false,
        int[][]? windows = null, float repetitionPenalty = 1f, int repeatLastN = 0, int ringCapacity = 0, int topK = 0, float minP = 0f, float presencePenalty = 0f)
    {
        if (!SupportsDeviceDraws(accelerator)) throw new NotSupportedException($"device draws need groups of {DrawGroup} threads");
        if (logits.Length != (long)rows * vocab) throw new ArgumentException("rows x vocab logits");
        if (vocab >= 1 << 24) throw new NotSupportedException("candidate ids travel as floats (exact below 2^24)");
        int nch = (vocab + Chunk - 1) / Chunk, S = 2 * Candidates + 1 + nch;
        var acc = accelerator;
        using var lg = acc.Allocate1D(logits);
        using var maxV = acc.Allocate1D<float>((long)rows * nch);
        using var maxI = acc.Allocate1D<int>((long)rows * nch);
        using var selV = acc.Allocate1D<float>((long)rows * SelectCap);
        using var selI = acc.Allocate1D<int>((long)rows * SelectCap);
        using var selN = acc.Allocate1D<int>(Math.Max(64, rows));
        using var cand = acc.Allocate1D<float>((long)rows * S);
        using var seqs = acc.Allocate1D(Enumerable.Range(0, rows).ToArray());
        using var pos = acc.Allocate1D(new[] { position - 1, 0 });
        int W = ringCapacity > 0 ? ringCapacity : Math.Max(1, windows?.Max(w => w.Length) ?? 1), active = repeatLastN > 0 ? repeatLastN : W;
        var repHost = new int[rows + rows * W];
        for (int r = 0; windows != null && r < rows; r++)
        {
            repHost[r] = windows[r].Length;
            for (int j = 0; j < windows[r].Length; j++) repHost[rows + r * W + j % W] = windows[r][j];
        }
        using var rep = acc.Allocate1D(repHost);
        using var dF2 = acc.Allocate1D(new[] { topP, repetitionPenalty, minP, presencePenalty });
        using var dI = acc.Allocate1D(new[] { seed, forceFullRow ? 1 : 0, active, W, rows, topK });
        using var ids = acc.Allocate1D<int>(Math.Max(64, rows));
        bool group = TopKGroupDefault(acc) && TopKGroupFits(acc);
        var chunkMax = acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, int, int, int>(SpawnDev.ILGPU.ML.Kernels.TopKKernels.ChunkMaxImpl);
        var floor = acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, FV, IV, int, int>(SpawnDev.ILGPU.ML.Kernels.TopKKernels.FloorImpl);
        var select = acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, FV, IV, int, int, int, float, int>(SpawnDev.ILGPU.ML.Kernels.TopKKernels.SelectImpl);
        var chunkMaxGroup = group ? acc.LoadStreamKernel<FV, FV, IV, int, int>(SpawnDev.ILGPU.ML.Kernels.TopKKernels.ChunkMaxGroupImpl) : null;
        var selectGroup = group ? acc.LoadStreamKernel<FV, FV, IV, FV, IV, int, int, int, float>(SpawnDev.ILGPU.ML.Kernels.TopKKernels.SelectGroupImpl) : null;
        var floorGroup = group && nch <= SpawnDev.ILGPU.ML.Kernels.TopKKernels.FloorGroupMaxChunks ? acc.LoadStreamKernel<FV, IV, FV, IV, int, int>(SpawnDev.ILGPU.ML.Kernels.TopKKernels.FloorGroupImpl) : null;
        var final = acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, FV, int>(SpawnDev.ILGPU.ML.Kernels.TopKKernels.FinalImpl);
        var draw = acc.LoadStreamKernel<FV, FV, IV, IV, FV, IV, IV, IV, int, int, int, int, int, float>(DrawImpl);
        var penalize = acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, FV, IV, int>(PenalizeImpl);
        penalize(rows * W, lg.View, seqs.View, dI.View, dF2.View, rep.View, vocab);
        if (chunkMaxGroup != null) chunkMaxGroup(new KernelConfig(rows * nch, Chunk), lg.View, maxV.View, maxI.View, vocab, nch);
        else chunkMax(rows * nch, lg.View, maxV.View, maxI.View, vocab, nch, Chunk);
        if (floorGroup != null) floorGroup(new KernelConfig(SpawnDev.ILGPU.ML.Kernels.TopKKernels.FloorGroups(rows, nch), SpawnDev.ILGPU.ML.Kernels.TopKKernels.FloorGroupThreads), maxV.View, maxI.View, cand.View, selN.View, nch, S);
        else floor(SpawnDev.ILGPU.ML.Kernels.TopKKernels.FloorThreads(rows, nch), maxV.View, maxI.View, cand.View, selN.View, nch, S);
        if (selectGroup != null) selectGroup(new KernelConfig(rows * nch, Chunk), lg.View, cand.View, selN.View, selV.View, selI.View, vocab, nch, S, temperature);
        else select(rows * nch, lg.View, cand.View, selN.View, selV.View, selI.View, vocab, nch, S, temperature, Chunk);
        final(SpawnDev.ILGPU.ML.Kernels.TopKKernels.FinalThreads(rows), selV.View, selI.View, selN.View, cand.View, S);
        draw(new KernelConfig(rows, DrawGroup), lg.View, cand.View, seqs.View, pos.View, dF2.View, dI.View, ids.View, rep.View, vocab, nch, S, 0, 1, temperature);
        var got = await ids.View.SubView(0, rows).CopyToHostAsync().ConfigureAwait(false);      // the read waits for the kernels (every backend), so the buffers may go
        return got;
    }

    // a step's input under device draws: row r is the token its sequence drew last
    private static void GatherIdsImpl(Index1D r, IV seqIds, IV seqs, IV ids) => ids[r] = seqIds[seqs[r]];

    // the draw: group g = slot g (sequence seqs[g], from row g of the forward, or row 0 when rowZero), DrawGroup threads; the token goes to seqIds[sequence]. The
    // uniform is DrawUniform(seed, sequence, position + tAdd); drawF = [topP, penalty], drawI = [seed, force the full-row path, penalty window, ring, MaxBatch, topK].
    //  top-k (drawI[5] > 0): the k best within the window (ties: the lower id), then the nucleus over their mass, as SampleLogits; on the candidates when
    //  k is at most K and they are the top K, else on the whole row (kappa / tieCut below).
    //  T <= 0: the argmax, the floor kernel's best chunk maximum (the lowest id among equals: exact even when the candidates are not the top K).
    //  T > 0, the nucleus inside the candidates: thread 0 walks them as SampleAsync does.
    //  Otherwise (more than SelectCap tokens at the floor, a nucleus wider than the candidates, or forced) the whole row, exactly: the nucleus' last token is the
    //  largest logit tau whose mass at or above it reaches topP x the total, found by bisecting the float's order key (33 passes over the row, each a group sum),
    //  the ties at tau taken in id order (each thread holds a contiguous id range, so a scan over the threads is the id order); the draw is the same search for
    //  u x keep. Float sums can disagree with themselves by a hair: a search that finds no tie takes the last one, a draw past the nucleus its last token.
    // The full-row branch holds barriers: its condition comes from Group.BarrierOr (uniform in WGSL: workgroupUniformLoad), its loops run a fixed count.
    private static void DrawImpl(FV logits, FV cand, IV seqs, IV posBuf, FV drawF, IV drawI, IV seqIds, IV rep, int V, int nch, int S, int rowZero, int tAdd, float T)
    {
        const int K = Candidates;
        int slot = Grid.IdxX, t = Group.IdxX;
        int row = rowZero != 0 ? 0 : slot, seq = seqs[slot];
        long o = (long)row * S, lb = (long)row * V;
        var red = SharedMemory.Allocate<float>(DrawGroup + 1);
        var redI = SharedMemory.Allocate<int>(DrawGroup);
        float topP = drawF[0], max = cand[o];
        float u = DrawUniform(drawI[0], seq, posBuf[0] + tAdd);
        bool forced = drawI[1] != 0, sampling = T > 0f;
        int topK = drawI[5];
        int tok = (int)cand[o + K];
        bool needFull = false;
        if (t == 0 && sampling)
        {
            bool wide = forced | (cand[o + 2 * K] > SelectCap) | (topK > K);
            if (wide) needFull = true;
            else
            {
                float floorT = max - 20f * T, keep = 0f, target;
                int m = 0;
                if (topK > 0)
                {
                    // top-k: the k best within the window are the first candidates, and the nucleus runs over their mass only (SampleAsync's order)
                    float total = 0f;
                    int kk = 0;
                    for (int j = 0; j < K; j++)
                    {
                        float cv = cand[o + j];
                        bool take = (cv >= floorT) & (kk < topK);
                        if (!take) break;
                        total += MathF.Exp((cv - max) / T);
                        kk = j + 1;
                    }
                    target = topP * total;
                    for (int j = 0; j < kk; j++)
                    {
                        if (keep >= target) break;
                        keep += MathF.Exp((cand[o + j] - max) / T);
                        m = j + 1;
                    }
                }
                else
                {
                    float total = 0f;
                    for (int ch = 0; ch < nch; ch++) total += cand[o + 2 * K + 1 + ch];
                    target = topP * total;
                    for (int j = 0; j < K; j++)
                    {
                        float cv = cand[o + j];
                        bool take = (cv >= floorT) & (keep < target);
                        if (!take) break;
                        keep += MathF.Exp((cv - max) / T);
                        m = j + 1;
                    }
                }
                if (keep < target & topK <= 0) needFull = true;                  // the nucleus reaches past the candidates
                else
                {
                    // min-p (drawF[2]): of the nucleus, only the tokens at or above max + T ln(minP) stay (the best always); the kept mass summed again in order
                    float minP = drawF[2];
                    if (minP > 0f)
                    {
                        float minFloor = max + T * MathF.Log(minP);
                        while ((m > 1) && (cand[o + m - 1] < minFloor)) m--;
                        keep = 0f;
                        for (int j = 0; j < m; j++) keep += MathF.Exp((cand[o + j] - max) / T);
                    }
                    float x = u * keep, acc = 0f;
                    tok = (int)cand[o + K + m - 1];
                    for (int j = 0; j < m; j++)
                    {
                        acc += MathF.Exp((cand[o + j] - max) / T);
                        if (x < acc) { tok = (int)cand[o + K + j]; break; }
                    }
                }
            }
        }
        bool full = Group.BarrierOr(needFull);
        if (full)
        {
            float floorT = max - 20f * T;
            int span = (V + DrawGroup - 1) / DrawGroup, i0 = t * span, i1 = Math.Min(V, i0 + span);
            // the kept set: logits above kappa, and those equal to kappa up to id tieCut. No top-k: kappa = the e^20 floor and every tie (l >= floorT).
            // Top-k: kappa = the k-th best value (the largest key with at least k tokens at or above it), the ties at kappa taken in id order up to the k-th token
            // (SampleLogits' order: the lower id first); the nucleus and the draw below run over this set.
            float kappa = floorT;
            int tieCut = int.MaxValue;
            // the branch holds barriers, so its condition must be uniform in WGSL: topK is read from a storage buffer and the count from shared memory, neither
            // uniform there, so the decision goes through Group.BarrierOr (workgroupUniformLoad) and the count is taken unconditionally (one pass, whole-row path only)
            float inWindow = GroupSum(red, t, CountPart(logits, lb, V, floorT, floorT, false, t));
            bool cutK = Group.BarrierOr((topK > 0) & (inWindow > topK));
            {
                if (cutK)
                {
                    uint klo = OrderKey(floorT), khi = OrderKey(max);
                    for (int it = 0; it < 33; it++)
                    {
                        uint mid = klo + ((khi - klo + 1u) >> 1);
                        float c = GroupSum(red, t, CountPart(logits, lb, V, floorT, FromOrderKey(mid), false, t));
                        if (c >= topK) klo = mid; else khi = mid - 1u;
                    }
                    kappa = FromOrderKey(klo);
                    float above = GroupSum(red, t, CountPart(logits, lb, V, floorT, kappa, true, t));
                    float need = topK - above;                                    // 1 or more ties at kappa join (exact: counts below 2^24)
                    float ownTies = 0f;
                    for (int i = i0; i < i1; i++) if (logits[lb + i] == kappa) ownTies += 1f;
                    float rank = GroupExclusive(red, t, ownTies);
                    int cutHere = int.MaxValue;
                    for (int i = i0; i < i1; i++)
                    {
                        if (logits[lb + i] == kappa)
                        {
                            rank += 1f;
                            bool hit = (rank >= need) & (cutHere == int.MaxValue);
                            if (hit) cutHere = i;
                        }
                    }
                    tieCut = GroupMin(redI, t, cutHere);
                }
            }
            float total = GroupSum(red, t, MassPart(logits, lb, V, max, T, kappa, tieCut, kappa, false, t));
            float target = topP * total;
            // tau: the largest key whose mass at or above reaches the target (true at kappa: its mass is the total)
            uint lo = OrderKey(kappa), hi = OrderKey(max);
            for (int it = 0; it < 33; it++)
            {
                uint mid = lo + ((hi - lo + 1u) >> 1);
                float s = GroupSum(red, t, MassPart(logits, lb, V, max, T, kappa, tieCut, FromOrderKey(mid), false, t));
                if (s >= target) lo = mid; else hi = mid - 1u;
            }
            float tau = FromOrderKey(lo);
            float gt = GroupSum(red, t, MassPart(logits, lb, V, max, T, kappa, tieCut, tau, true, t));
            float own = 0f; int lastTie = -1;
            for (int i = i0; i < i1; i++)
            {
                float l = logits[lb + i];
                bool tie = (l == tau) & ((l > kappa) | (i <= tieCut));
                if (tie) { own += MathF.Exp((l - max) / T); lastTie = i; }
            }
            float before = GroupExclusive(red, t, own);
            float tiesTotal = red[DrawGroup];
            float run = gt + before, cutKeep = 0f; int cut = int.MaxValue;
            for (int i = i0; i < i1; i++)
            {
                float l = logits[lb + i];
                bool tie = (l == tau) & ((l > kappa) | (i <= tieCut));
                if (tie)
                {
                    run += MathF.Exp((l - max) / T);
                    bool hit = (run >= target) & (cut == int.MaxValue);
                    if (hit) { cut = i; cutKeep = run; }
                }
            }
            int nucLast = GroupMin(redI, t, cut);
            int lastTau = GroupMax(redI, t, lastTie);
            Group.Barrier();
            bool mine = (cut == nucLast) & (cut != int.MaxValue);
            if (mine) red[0] = cutKeep;
            bool none = (nucLast == int.MaxValue) & (t == 0);
            if (none) red[0] = gt + tiesTotal;
            Group.Barrier();
            float keep = red[0];
            if (nucLast == int.MaxValue) nucLast = lastTau;
            // min-p (drawF[2]) after the nucleus: when its floor max + T ln(minP) lies above tau, the kept set ends at the lowest logit at or above the floor (all its ties,
            // the last of them by id); the mass is summed again over that set. The condition is uniform (Group.BarrierOr): the branch holds group reductions.
            float minPF = drawF[2];
            float minFloorF = minPF > 0f ? max + T * MathF.Log(minPF) : float.NegativeInfinity;
            bool cutMinP = Group.BarrierOr((minPF > 0f) & (minFloorF > tau));
            if (cutMinP)
            {
                keep = GroupSum(red, t, MassPart(logits, lb, V, max, T, kappa, tieCut, minFloorF, false, t));
                int ownLow = int.MaxValue;
                for (int i = i0; i < i1; i++)
                {
                    float l = logits[lb + i];
                    bool kept = ((l > kappa) | ((l == kappa) & (i <= tieCut))) & (l >= minFloorF);
                    if (kept) { int key = (int)(OrderKey(l) ^ 0x80000000u); if (key < ownLow) ownLow = key; }
                }
                int lowKey = GroupMin(redI, t, ownLow);
                float lowest = FromOrderKey((uint)lowKey ^ 0x80000000u);
                int ownLast = -1;
                for (int i = i0; i < i1; i++)
                {
                    float l = logits[lb + i];
                    bool tieLow = (l == lowest) & ((l > kappa) | (i <= tieCut));
                    if (tieLow) ownLast = i;
                }
                nucLast = GroupMax(redI, t, ownLast);
                tau = lowest;
            }
            // the draw: the first token (best first, ties by id) whose running mass passes x = u * keep
            float x = u * keep;
            lo = OrderKey(kappa); hi = OrderKey(max);
            for (int it = 0; it < 33; it++)
            {
                uint mid = lo + ((hi - lo + 1u) >> 1);
                float s = GroupSum(red, t, MassPart(logits, lb, V, max, T, kappa, tieCut, FromOrderKey(mid), false, t));
                if (s > x) lo = mid; else hi = mid - 1u;
            }
            float sigma = FromOrderKey(lo);
            float gt2 = GroupSum(red, t, MassPart(logits, lb, V, max, T, kappa, tieCut, sigma, true, t));
            float own2 = 0f; int lastTie2 = -1;
            for (int i = i0; i < i1; i++)
            {
                float l = logits[lb + i];
                bool tie = (l == sigma) & ((l > kappa) | (i <= tieCut));
                if (tie) { own2 += MathF.Exp((l - max) / T); lastTie2 = i; }
            }
            float before2 = GroupExclusive(red, t, own2);
            float run2 = gt2 + before2; int pick = int.MaxValue;
            for (int i = i0; i < i1; i++)
            {
                float l = logits[lb + i];
                bool tie = (l == sigma) & ((l > kappa) | (i <= tieCut));
                if (tie)
                {
                    run2 += MathF.Exp((l - max) / T);
                    bool hit = (run2 > x) & (pick == int.MaxValue);
                    if (hit) pick = i;
                }
            }
            int drawn = GroupMin(redI, t, pick);
            int lastSigma = GroupMax(redI, t, lastTie2);
            if (drawn == int.MaxValue) drawn = lastSigma;
            bool outside = (sigma < tau) | ((sigma == tau) & (drawn > nucLast));
            if (outside) drawn = nucLast;
            tok = drawn;
        }
        if (t == 0)
        {
            seqIds[seq] = tok;
            // the token joins its sequence's penalty window (rep: [MB counts][MB rings of W]; drawI[3] = W, drawI[4] = MB)
            int W = drawI[3], MB = drawI[4], c = rep[seq];
            rep[MB + seq * W + c % W] = tok;
            rep[seq] = c + 1;
        }
    }

    // the repetition penalty on row r (sequence seqs[r]) of the logits: thread (r, k) takes the sequence's k-th most recent drawn token (k below the window
    // drawI[2] and the tokens drawn), skips it when a more recent slot holds the same token (so each distinct token is penalised once, by one thread), and
    // divides its logit by the penalty drawF[1] when positive, multiplies it when not (TextGenerationSampler.ApplyRepetitionPenalty). Threads per row: W = drawI[3].
    // Then the presence penalty drawF[3] is subtracted (TextGenerationSampler.ApplyPresencePenalty), on the same distinct tokens.
    private static void PenalizeImpl(Index1D i, FV logits, IV seqs, IV drawI, FV drawF, IV rep, int V)
    {
        float pen = drawF[1], pres = drawF[3];
        if ((pen == 1f) & (pres == 0f)) return;
        int W = drawI[3], MB = drawI[4], ii = i, r = ii / W, k = ii % W;
        int seq = seqs[r], c = rep[seq], n = Math.Min(Math.Min(c, drawI[2]), W);
        if (k >= n) return;
        int b = MB + seq * W, tok = rep[b + (c - 1 - k) % W];
        bool dup = false;
        for (int k2 = 0; k2 < k; k2++)
        {
            bool same = rep[b + (c - 1 - k2) % W] == tok;
            if (same) { dup = true; break; }
        }
        if (dup) return;
        long at = (long)r * V + tok;
        float v = logits[at];
        logits[at] = (v > 0f ? v / pen : v * pen) - pres;
    }

    // this thread's share of the row's softmax mass at T over the tokens within e^20 of the best (logit >= floorT) and at or above v (strict: above v)
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static float MassPart(FV logits, long lb, int V, float max, float T, float kappa, int tieCut, float v, bool strict, int t)
    {
        // the kept set (DrawImpl): above kappa, or equal to it up to id tieCut
        float s = 0f;
        for (int i = t; i < V; i += DrawGroup)
        {
            float l = logits[lb + i];
            bool inc = ((l > kappa) | ((l == kappa) & (i <= tieCut))) & (strict ? l > v : l >= v);
            if (inc) s += MathF.Exp((l - max) / T);
        }
        return s;
    }
    // this thread's count of the window's tokens (at or above floorT) at or above v (strict: above)
    private static float CountPart(FV logits, long lb, int V, float floorT, float v, bool strict, int t)
    {
        float c = 0f;
        for (int i = t; i < V; i += DrawGroup)
        {
            float l = logits[lb + i];
            bool inc = (l >= floorT) & (strict ? l > v : l >= v);
            if (inc) c += 1f;
        }
        return c;
    }
    // the group's sum of x (every thread gets it)
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static float GroupSum(ArrayView<float> red, int t, float x)
    {
        Group.Barrier();                                                       // the last reads of red are done
        red[t] = x;
        Group.Barrier();
        for (int w = DrawGroup / 2; w > 0; w >>= 1)
        {
            if (t < w) red[t] += red[t + w];
            Group.Barrier();
        }
        return red[0];
    }
    // the sum of x over the threads below this one; red[DrawGroup] = the sum over all
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static float GroupExclusive(ArrayView<float> red, int t, float x)
    {
        Group.Barrier();
        red[t] = x;
        Group.Barrier();
        if (t == 0)
        {
            float acc = 0f;
            for (int i = 0; i < DrawGroup; i++) { float v = red[i]; red[i] = acc; acc += v; }
            red[DrawGroup] = acc;
        }
        Group.Barrier();
        return red[t];
    }
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static int GroupMin(ArrayView<int> red, int t, int x)
    {
        Group.Barrier();
        red[t] = x;
        Group.Barrier();
        for (int w = DrawGroup / 2; w > 0; w >>= 1)
        {
            if (t < w) red[t] = Math.Min(red[t], red[t + w]);
            Group.Barrier();
        }
        return red[0];
    }
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static int GroupMax(ArrayView<int> red, int t, int x)
    {
        Group.Barrier();
        red[t] = x;
        Group.Barrier();
        for (int w = DrawGroup / 2; w > 0; w >>= 1)
        {
            if (t < w) red[t] = Math.Max(red[t], red[t + w]);
            Group.Barrier();
        }
        return red[0];
    }
    // a float's order as an unsigned key (a < b exactly when OrderKey(a) < OrderKey(b)), and back
    private static uint OrderKey(float f) { uint b = Interop.FloatAsInt(f); return (b & 0x80000000u) != 0 ? ~b : b | 0x80000000u; }
    private static float FromOrderKey(uint k) => Interop.IntAsFloat((k & 0x80000000u) != 0 ? k & 0x7FFFFFFFu : ~k);

    private const int FaRows = 8, FaKeys = 8, FaThreads = FaRows * FaKeys, FaMaxD = 128;

    // prefill attention, tiled (flash). Group g = (block of FaRows query rows, head h), FaThreads threads. The block's queries (scaled) sit in shared memory; each
    // tile of FaKeys keys is loaded into shared memory once (coalesced) and serves all FaRows rows (thread (row, key) one score), each row's running max and
    // exp-sum take the tile (online softmax: the output so far is rescaled), then the tile's values replace its keys in the same buffer and the FaRows x D outputs
    // in shared memory take them. Causal: row r (position p + r) sees keys 0..p + r; a prefill is sequence 0. 12.6 KB of shared memory (WebGPU allows 16 KB).
    // MEASURED 2026-11-01 before it: a prefill on the split-K kernel (built for decode rows) spent 3.1 s in attention for 600 rows over a 3k cache (CUDA).
    private static void FlashPrefillImpl<T>(FV q, ArrayView1D<T, Stride1D.Dense> kc, ArrayView1D<T, Stride1D.Dense> vc, FV att, int R, int H, int kvH, int D, int p, float scale) where T : unmanaged, System.Numerics.INumber<T>
    {
        int g = Grid.IdxX, t = Group.IdxX;
        int h = g % H, r0 = (g / H) * FaRows;
        int kvC = kvH * D, hk = h / (H / kvH);
        var qs = SharedMemory.Allocate<float>(FaRows * FaMaxD);
        var kv = SharedMemory.Allocate<float>(FaKeys * FaMaxD);
        var os = SharedMemory.Allocate<float>(FaRows * FaMaxD);
        var ps = SharedMemory.Allocate<float>(FaRows * FaKeys);
        var st = SharedMemory.Allocate<float>(3 * FaRows);          // per row: running max, exp-sum, this tile's rescale of the output so far
        for (int idx = t; idx < FaRows * D; idx += FaThreads)
        {
            int i = idx / D, d = idx % D, r = r0 + i;
            qs[idx] = r < R ? q[((long)r * H + h) * D + d] * scale : 0f;
            os[idx] = 0f;
        }
        if (t < FaRows) { st[t] = -1e30f; st[FaRows + t] = 0f; st[2 * FaRows + t] = 0f; }
        Group.Barrier();
        int last = p + Math.Min(R, r0 + FaRows) - 1;                 // the block's last query position: keys past it are masked for every row of the block
        int ti = t / FaKeys, tc = t % FaKeys;
        for (int j0 = 0; j0 <= last; j0 += FaKeys)
        {
            for (int idx = t; idx < FaKeys * D; idx += FaThreads)
            {
                int jj = idx / D, d = idx % D, j = j0 + jj;
                kv[idx] = j <= last ? PrecisionConvert.ConvertToSingle(kc[(long)j * kvC + hk * D + d]) : 0f;
            }
            Group.Barrier();
            {
                int r = r0 + ti, j = j0 + tc;
                float sc = -1e30f;
                bool live = (r < R) & (j <= p + r);
                if (live)
                {
                    float acc = 0f;
                    for (int d = 0; d < D; d++) acc = float.MultiplyAddEstimate(qs[ti * D + d], kv[tc * D + d], acc);
                    sc = acc;
                }
                ps[ti * FaKeys + tc] = sc;
            }
            Group.Barrier();
            if (t < FaRows)
            {
                float mo = st[t], mt = -1e30f;
                for (int c = 0; c < FaKeys; c++) mt = MathF.Max(mt, ps[t * FaKeys + c]);
                float mn = MathF.Max(mo, mt), sum = 0f;
                for (int c = 0; c < FaKeys; c++)
                {
                    float sv = ps[t * FaKeys + c];
                    float e = sv <= -1e29f ? 0f : MathF.Exp(sv - mn);
                    ps[t * FaKeys + c] = e; sum += e;
                }
                float corr = mo <= -1e29f ? 0f : MathF.Exp(mo - mn);
                st[t] = mn; st[FaRows + t] = st[FaRows + t] * corr + sum; st[2 * FaRows + t] = corr;
            }
            Group.Barrier();
            for (int idx = t; idx < FaKeys * D; idx += FaThreads)
            {
                int jj = idx / D, d = idx % D, j = j0 + jj;
                kv[idx] = j <= last ? PrecisionConvert.ConvertToSingle(vc[(long)j * kvC + hk * D + d]) : 0f;
            }
            Group.Barrier();
            for (int idx = t; idx < FaRows * D; idx += FaThreads)
            {
                int i = idx / D, d = idx % D;
                float o = os[idx] * st[2 * FaRows + i];
                for (int c = 0; c < FaKeys; c++) o = float.MultiplyAddEstimate(ps[i * FaKeys + c], kv[c * D + d], o);
                os[idx] = o;
            }
            Group.Barrier();
        }
        for (int idx = t; idx < FaRows * D; idx += FaThreads)
        {
            int i = idx / D, d = idx % D, r = r0 + i;
            if (r < R) { float l = st[FaRows + i]; att[((long)r * H + h) * D + d] = l > 0f ? os[idx] / l : 0f; }
        }
    }

    private const int FgQ = 32, FgKeys = 32, FgThreads = 256, FgDc = 32, FgPad = FgDc + 1, FgHalf = FgKeys / 2, FgMaxD = 128;
    // one shared array, regions: U (the Q and K chunks of the score phase, then half a tile of V rows), P (the tile's probabilities), PM / PS (per query the eight
    // partial maxima / sums of its score threads), SM / SL (running max, exp-sum), CR (this tile's rescale), MN (the new max)
    private const int FgU = 0, FgP = 2 * FgQ * FgPad, FgPM = FgP + FgQ * FgPad, FgPS = FgPM + FgQ * 8, FgSM = FgPS + FgQ * 8, FgSL = FgSM + FgQ, FgCR = FgSL + FgQ,
        FgMN = FgCR + FgQ, FgShared = FgMN + FgQ;                     // 3,808 floats = 15,232 bytes (WebGPU guarantees 16,384)
    private struct FgAcc { public float X, Y, Z, W; }

    // prefill attention, tiled for grouped-query heads. Group g = (block of rowsPer = 32 / G query rows, kv head hk), 256 threads, 32 query vectors: vector v is
    // row r0 + v / G of query head hk * G + v % G (G = H / kvH; vectors past rowsPer * G idle). Per tile of 32 keys: the scores in four 32-wide D chunks (the Q and K
    // chunks staged in shared memory, rows padded to 33 floats; thread t scores vector t / 8 against keys 4 (t % 8) .. +3), the tile's softmax (online: each query's
    // max from its eight score threads' partial maxima, the probabilities written over the scores, the exp-sum kept per query), then P.V with V staged 16 keys at a
    // time (thread t accumulates vectors 4 (t / 32) .. +3 at d = t % 32 + 32 x in registers). Causal: vector at position p + row sees keys 0..p + row; a prefill is
    // sequence 0. Each K / V row is read once per block for all the G heads that share it (the older FlashPrefillImpl read it once per query head).
    private static void FlashPrefillGqaImpl<T>(FV q, ArrayView1D<T, Stride1D.Dense> kc, ArrayView1D<T, Stride1D.Dense> vc, FV att, int R, int H, int kvH, int D, int p, float scale) where T : unmanaged, System.Numerics.INumber<T>
    {
        int g = Grid.IdxX, t = Group.IdxX;
        int G = H / kvH, rowsPer = FgQ / G, hk = g % kvH, r0 = (g / kvH) * rowsPer;
        int nv = rowsPer * G, kvC = kvH * D;
        int rowsHere = Math.Min(rowsPer, R - r0);
        int last = p + r0 + rowsHere - 1;                            // the block's last query position: no key past it is live for any of its vectors
        var sh = SharedMemory.Allocate<float>(FgShared);
        if (t < FgQ) { sh[FgSM + t] = -1e30f; sh[FgSL + t] = 0f; }
        int sq = t >> 3, sk = (t & 7) * 4;                             // score phase: vector sq, keys sk..sk+3 of the tile
        int sqRow = sq / G;
        bool sqLive = sq < nv && sqRow < rowsHere;
        int sqPos = p + r0 + sqRow;
        int oq = (t >> 5) * 4, dl = t & 31;                            // P.V phase: vectors oq..oq+3, d = dl + 32 x
        var o0 = new FgAcc(); var o1 = new FgAcc(); var o2 = new FgAcc(); var o3 = new FgAcc();
        for (int j0 = 0; j0 <= last; j0 += FgKeys)
        {
            float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
            for (int dc = 0; dc < D; dc += FgDc)
            {
                for (int e = t; e < FgQ * FgDc; e += FgThreads)
                {
                    int v = e >> 5, d = e & 31, i = v / G, j = j0 + v;
                    sh[FgU + v * FgPad + d] = v < nv && i < rowsHere ? q[((long)(r0 + i) * H + hk * G + v % G) * D + dc + d] * scale : 0f;
                    sh[FgU + FgQ * FgPad + v * FgPad + d] = j <= last ? PrecisionConvert.ConvertToSingle(kc[(long)j * kvC + hk * D + dc + d]) : 0f;
                }
                Group.Barrier();
                int qb = FgU + sq * FgPad, kb = FgU + FgQ * FgPad + sk * FgPad;
                for (int d = 0; d < FgDc; d++)
                {
                    float qv = sh[qb + d];
                    s0 = float.MultiplyAddEstimate(qv, sh[kb + d], s0);
                    s1 = float.MultiplyAddEstimate(qv, sh[kb + FgPad + d], s1);
                    s2 = float.MultiplyAddEstimate(qv, sh[kb + 2 * FgPad + d], s2);
                    s3 = float.MultiplyAddEstimate(qv, sh[kb + 3 * FgPad + d], s3);
                }
                Group.Barrier();
            }
            int jb = j0 + sk;
            bool live = sqLive;
            if (!(live && jb <= sqPos)) s0 = -1e30f;
            if (!(live && jb + 1 <= sqPos)) s1 = -1e30f;
            if (!(live && jb + 2 <= sqPos)) s2 = -1e30f;
            if (!(live && jb + 3 <= sqPos)) s3 = -1e30f;
            sh[FgPM + sq * 8 + (t & 7)] = MathF.Max(MathF.Max(s0, s1), MathF.Max(s2, s3));
            Group.Barrier();
            float mt = -1e30f;
            for (int c = 0; c < 8; c++) mt = MathF.Max(mt, sh[FgPM + sq * 8 + c]);
            float mo = sh[FgSM + sq], mn = MathF.Max(mo, mt);
            float e0 = s0 <= -1e29f ? 0f : MathF.Exp(s0 - mn), e1 = s1 <= -1e29f ? 0f : MathF.Exp(s1 - mn);
            float e2 = s2 <= -1e29f ? 0f : MathF.Exp(s2 - mn), e3 = s3 <= -1e29f ? 0f : MathF.Exp(s3 - mn);
            int pb = FgP + sq * FgPad + sk;
            sh[pb] = e0; sh[pb + 1] = e1; sh[pb + 2] = e2; sh[pb + 3] = e3;
            sh[FgPS + sq * 8 + (t & 7)] = e0 + e1 + e2 + e3;
            if ((t & 7) == 0) { sh[FgCR + sq] = mo <= -1e29f ? 0f : MathF.Exp(mo - mn); sh[FgMN + sq] = mn; }
            Group.Barrier();
            if (t < FgQ)
            {
                float sum = 0f;
                for (int c = 0; c < 8; c++) sum += sh[FgPS + t * 8 + c];
                sh[FgSL + t] = sh[FgSL + t] * sh[FgCR + t] + sum;
                sh[FgSM + t] = sh[FgMN + t];
            }
            o0 = FgScale(o0, sh[FgCR + oq]); o1 = FgScale(o1, sh[FgCR + oq + 1]); o2 = FgScale(o2, sh[FgCR + oq + 2]); o3 = FgScale(o3, sh[FgCR + oq + 3]);
            for (int half = 0; half < 2; half++)
            {
                int jh = j0 + half * FgHalf;
                for (int e = t; e < FgHalf * D; e += FgThreads)
                {
                    int jj = e / D, d = e - jj * D, j = jh + jj;
                    sh[FgU + e] = j <= last ? PrecisionConvert.ConvertToSingle(vc[(long)j * kvC + hk * D + d]) : 0f;
                }
                Group.Barrier();
                for (int c = 0; c < FgHalf; c++)
                {
                    int vb = FgU + c * D + dl, col = half * FgHalf + c;
                    float v0 = sh[vb], v1 = D > 32 ? sh[vb + 32] : 0f, v2 = D > 64 ? sh[vb + 64] : 0f, v3 = D > 96 ? sh[vb + 96] : 0f;
                    o0 = FgMac(o0, sh[FgP + oq * FgPad + col], v0, v1, v2, v3);
                    o1 = FgMac(o1, sh[FgP + (oq + 1) * FgPad + col], v0, v1, v2, v3);
                    o2 = FgMac(o2, sh[FgP + (oq + 2) * FgPad + col], v0, v1, v2, v3);
                    o3 = FgMac(o3, sh[FgP + (oq + 3) * FgPad + col], v0, v1, v2, v3);
                }
                Group.Barrier();
            }
        }
        FgStore(att, o0, sh[FgSL + oq], oq, G, nv, rowsHere, r0, hk, H, D, dl);
        FgStore(att, o1, sh[FgSL + oq + 1], oq + 1, G, nv, rowsHere, r0, hk, H, D, dl);
        FgStore(att, o2, sh[FgSL + oq + 2], oq + 2, G, nv, rowsHere, r0, hk, H, D, dl);
        FgStore(att, o3, sh[FgSL + oq + 3], oq + 3, G, nv, rowsHere, r0, hk, H, D, dl);
    }
    private static FgAcc FgScale(FgAcc a, float c) { a.X *= c; a.Y *= c; a.Z *= c; a.W *= c; return a; }
    private static FgAcc FgMac(FgAcc a, float pr, float v0, float v1, float v2, float v3) { a.X = float.MultiplyAddEstimate(pr, v0, a.X); a.Y = float.MultiplyAddEstimate(pr, v1, a.Y); a.Z = float.MultiplyAddEstimate(pr, v2, a.Z); a.W = float.MultiplyAddEstimate(pr, v3, a.W); return a; }
    private static void FgStore(FV att, FgAcc a, float l, int v, int G, int nv, int rowsHere, int r0, int hk, int H, int D, int dl)
    {
        int row = v / G;
        if (v >= nv || row >= rowsHere) return;
        float inv = l > 0f ? 1f / l : 0f;
        long b = ((long)(r0 + row) * H + hk * G + v % G) * D + dl;
        att[b] = a.X * inv;
        if (D > 32) att[b + 32] = a.Y * inv;
        if (D > 64) att[b + 64] = a.Z * inv;
        if (D > 96) att[b + 96] = a.W * inv;
    }

    // split-K attention, part 1. Group g = (row r, head h, chunk ch of AttChunk keys), a thread per head dimension (D = group size, a power of two). The chunk's
    // scores go to shared memory, then its max m and exp-sum l (masked keys contribute 0), then thread d sums p_j * v[j, d] over the chunk (coalesced across d).
    // part[g, d] = that sum, ml[g] = (m, l).
    private static void AttnPartialImpl<T>(FV q, ArrayView1D<T, Stride1D.Dense> kc, ArrayView1D<T, Stride1D.Dense> vc, IV seqs, IV posBuf, FV part, FV ml, int H, int kvH, int maxSeq, int mode, int nch, float scale) where T : unmanaged, System.Numerics.INumber<T>
    {
        int p = posBuf[0];
        int g = Grid.IdxX, t = Group.IdxX, D = Group.DimX;
        int ch = g % nch, rest = g / nch, h = rest % H, r = rest / H;
        int seq = mode == 0 ? 0 : seqs[r], pos = mode == 0 ? p + r : p;
        int kvC = kvH * D, hk = h / (H / kvH), j0 = ch * AttChunk;
        var sc = SharedMemory.Allocate<float>(AttChunk);
        var red = SharedMemory.Allocate<float>(AttChunk);
        int qo = (r * H + h) * D;
        // this sequence's cache rows: 32-bit offsets in the loops (see AttnPartialGqaImpl)
        var kq = kc.SubView((long)seq * maxSeq * kvC, (long)maxSeq * kvC);
        var vq = vc.SubView((long)seq * maxSeq * kvC, (long)maxSeq * kvC);
        for (int jj = t; jj < AttChunk; jj += D)
        {
            int j = j0 + jj;
            float s = -1e30f;
            if (j <= pos)
            {
                int ko = j * kvC + hk * D;
                float acc = 0f;
                for (int d = 0; d < D; d++) acc = float.MultiplyAddEstimate(q[qo + d], PrecisionConvert.ConvertToSingle(kq[ko + d]), acc);
                s = acc * scale;
            }
            sc[jj] = s;
        }
        Group.Barrier();
        float m = -1e30f;
        for (int jj = t; jj < AttChunk; jj += D) m = MathF.Max(m, sc[jj]);
        red[t] = m;
        Group.Barrier();
        for (int s2 = D / 2; s2 > 0; s2 >>= 1)
        {
            if (t < s2) red[t] = MathF.Max(red[t], red[t + s2]);
            Group.Barrier();
        }
        m = red[0];
        Group.Barrier();
        float l = 0f;
        for (int jj = t; jj < AttChunk; jj += D)
        {
            float e = sc[jj] <= -1e29f ? 0f : MathF.Exp(sc[jj] - m);
            sc[jj] = e; l += e;
        }
        red[t] = l;
        Group.Barrier();
        for (int s2 = D / 2; s2 > 0; s2 >>= 1)
        {
            if (t < s2) red[t] += red[t + s2];
            Group.Barrier();
        }
        l = red[0];
        int keys = Math.Min(AttChunk, pos + 1 - j0);
        float o = 0f;
        int vo = hk * D + t;
        for (int jj = 0; jj < keys; jj++) o = float.MultiplyAddEstimate(sc[jj], PrecisionConvert.ConvertToSingle(vq[vo + (j0 + jj) * kvC]), o);
        part[(long)g * D + t] = o;
        if (t == 0) { ml[2L * g] = m; ml[2L * g + 1] = l; }
    }
    // split-K attention, part 1, grouped-query: group g = (row r, kv head hk, chunk ch of AttChunk keys), a thread per head dimension (D = group size, at most
    // GqaMaxD). The G = H / kvH query heads that share kv head hk are staged in shared memory; each key's K row is read ONCE and dotted against all G of them,
    // each V row read once for all G (the per-head kernel reads every K / V row G times: 4x on llama3 / qwen3, 7x on qwen2.5). Writes the same per-head partials
    // (part, ml) as AttnPartialImpl, so AttnCombineImpl merges them unchanged. G <= GqaMaxG.
    private static void AttnPartialGqaImpl<T, TG>(FV q, ArrayView1D<T, Stride1D.Dense> kc, ArrayView1D<T, Stride1D.Dense> vc, IV seqs, IV posBuf, FV part, FV ml, int H, int kvH, int maxSeq, int mode, int nch, int ck, float scale)
        where T : unmanaged, System.Numerics.INumber<T> where TG : struct, IGqaGroup
    {
        int p = posBuf[0];
        int g = Grid.IdxX, t = Group.IdxX, D = Group.DimX;
        int G = default(TG).G;                                      // a compile-time constant: the per-head guards below fold away (H / kvH == G)
        int ch = g % nch, rest = g / nch, hk = rest % kvH, r = rest / kvH;
        int seq = mode == 0 ? 0 : seqs[r], pos = mode == 0 ? p + r : p;
        int kvC = kvH * D, j0 = ch * ck;                             // ck keys a chunk (at most AttChunk: sc holds G x AttChunk scores)
        var qs = SharedMemory.Allocate<float>(GqaMaxG * GqaMaxD);
        var sc = SharedMemory.Allocate<float>(GqaMaxG * AttChunk);
        var mxs = SharedMemory.Allocate<float>(GqaMaxG);              // each head's chunk max (the reductions run in qs, free after the scores)
        // this sequence's rows of the caches: the offsets below stay 32-bit (a sequence's maxSeq x kvC is far below 2^31), so the inner loops carry no 64-bit
        // arithmetic (emulated on WebGPU: an i64 add per K element and an i64 multiply per V row before 2026-11-02)
        var kq = kc.SubView((long)seq * maxSeq * kvC, (long)maxSeq * kvC);
        var vq = vc.SubView((long)seq * maxSeq * kvC, (long)maxSeq * kvC);
        for (int gi = 0; gi < G; gi++) qs[gi * D + t] = q[((long)r * H + hk * G + gi) * D + t];
        Group.Barrier();
        // scores: each key's row once, against the G heads
        for (int jj = t; jj < ck; jj += D)
        {
            int j = j0 + jj;
            float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f, s4 = 0f, s5 = 0f, s6 = 0f, s7 = 0f;
            bool live = j <= pos;
            if (live)
            {
                int ko = j * kvC + hk * D;
                for (int d = 0; d < D; d++)
                {
                    float kv = PrecisionConvert.ConvertToSingle(kq[ko + d]);
                    s0 = float.MultiplyAddEstimate(qs[d], kv, s0);
                    if (G > 1) s1 = float.MultiplyAddEstimate(qs[D + d], kv, s1);
                    if (G > 2) s2 = float.MultiplyAddEstimate(qs[2 * D + d], kv, s2);
                    if (G > 3) s3 = float.MultiplyAddEstimate(qs[3 * D + d], kv, s3);
                    if (G > 4) s4 = float.MultiplyAddEstimate(qs[4 * D + d], kv, s4);
                    if (G > 5) s5 = float.MultiplyAddEstimate(qs[5 * D + d], kv, s5);
                    if (G > 6) s6 = float.MultiplyAddEstimate(qs[6 * D + d], kv, s6);
                    if (G > 7) s7 = float.MultiplyAddEstimate(qs[7 * D + d], kv, s7);
                }
            }
            sc[jj] = live ? s0 * scale : -1e30f;
            if (G > 1) sc[AttChunk + jj] = live ? s1 * scale : -1e30f;
            if (G > 2) sc[2 * AttChunk + jj] = live ? s2 * scale : -1e30f;
            if (G > 3) sc[3 * AttChunk + jj] = live ? s3 * scale : -1e30f;
            if (G > 4) sc[4 * AttChunk + jj] = live ? s4 * scale : -1e30f;
            if (G > 5) sc[5 * AttChunk + jj] = live ? s5 * scale : -1e30f;
            if (G > 6) sc[6 * AttChunk + jj] = live ? s6 * scale : -1e30f;
            if (G > 7) sc[7 * AttChunk + jj] = live ? s7 * scale : -1e30f;
        }
        Group.Barrier();
        int keys = Math.Min(ck, pos + 1 - j0);
        int vo = hk * D + t;
        // every head's chunk max, then exp-sum, in ONE tree reduction each (the probabilities replace the scores in place); the G heads reduce side by side, so
        // the barriers do not grow with G. MEASURED 2026-11-02 before it (WebGPU, qwen3-0.6b, G = 2, short prompt, captured steps): a reduction pair per head
        // made this kernel 14% slower per token than the per-head kernel (216 against 251 tok/s): half the groups, each with G times the barrier chain.
        for (int gi = 0; gi < G; gi++)
        {
            float m = -1e30f;
            for (int jj = t; jj < ck; jj += D) m = MathF.Max(m, sc[gi * AttChunk + jj]);
            qs[gi * D + t] = m;
        }
        Group.Barrier();
        for (int w = D / 2; w > 0; w >>= 1)
        {
            if (t < w)
                for (int gi = 0; gi < G; gi++) qs[gi * D + t] = MathF.Max(qs[gi * D + t], qs[gi * D + t + w]);
            Group.Barrier();
        }
        if (t < G) mxs[t] = qs[t * D];
        Group.Barrier();
        for (int gi = 0; gi < G; gi++)
        {
            float m = mxs[gi], l = 0f;
            for (int jj = t; jj < ck; jj += D)
            {
                int si = gi * AttChunk + jj;
                float e = sc[si] <= -1e29f ? 0f : MathF.Exp(sc[si] - m);
                sc[si] = e; l += e;
            }
            qs[gi * D + t] = l;
        }
        Group.Barrier();
        for (int w = D / 2; w > 0; w >>= 1)
        {
            if (t < w)
                for (int gi = 0; gi < G; gi++) qs[gi * D + t] += qs[gi * D + t + w];
            Group.Barrier();
        }
        if (t < G)
        {
            long gq = ((long)r * H + hk * G + t) * nch + ch;
            ml[2 * gq] = mxs[t]; ml[2 * gq + 1] = qs[t * D];
        }
        float o0 = 0f, o1 = 0f, o2 = 0f, o3 = 0f, o4 = 0f, o5 = 0f, o6 = 0f, o7 = 0f;
        for (int jj = 0; jj < keys; jj++)
        {
            float v = PrecisionConvert.ConvertToSingle(vq[vo + (j0 + jj) * kvC]);
            o0 = float.MultiplyAddEstimate(sc[jj], v, o0);
            if (G > 1) o1 = float.MultiplyAddEstimate(sc[AttChunk + jj], v, o1);
            if (G > 2) o2 = float.MultiplyAddEstimate(sc[2 * AttChunk + jj], v, o2);
            if (G > 3) o3 = float.MultiplyAddEstimate(sc[3 * AttChunk + jj], v, o3);
            if (G > 4) o4 = float.MultiplyAddEstimate(sc[4 * AttChunk + jj], v, o4);
            if (G > 5) o5 = float.MultiplyAddEstimate(sc[5 * AttChunk + jj], v, o5);
            if (G > 6) o6 = float.MultiplyAddEstimate(sc[6 * AttChunk + jj], v, o6);
            if (G > 7) o7 = float.MultiplyAddEstimate(sc[7 * AttChunk + jj], v, o7);
        }
        long pb = ((long)r * H + hk * G) * nch + ch;
        part[pb * D + t] = o0;
        if (G > 1) part[(pb + nch) * D + t] = o1;
        if (G > 2) part[(pb + 2L * nch) * D + t] = o2;
        if (G > 3) part[(pb + 3L * nch) * D + t] = o3;
        if (G > 4) part[(pb + 4L * nch) * D + t] = o4;
        if (G > 5) part[(pb + 5L * nch) * D + t] = o5;
        if (G > 6) part[(pb + 6L * nch) * D + t] = o6;
        if (G > 7) part[(pb + 7L * nch) * D + t] = o7;
    }

    // AttnPartialGqaImpl with (1) the dead-chunk exit of the ILP kernel (a captured step launches every chunk the cache holds: MEASURED 2026-11-08, Chrome
    // WebGPU, qwen3:1.7b Q8_0, MaxSeq 4096: 0.53 ms of attention a token at a 25-token prompt, 31 of every 32 chunks dead) and (2) the scores from 16-byte K
    // loads: L = D / 8 lanes share a key row, each lane one 16-byte unit (8 halves) dotted with its 8 query values per head, the row's dot reduced across the
    // L lanes with Warp.ShuffleXor (offsets < L stay inside the row). The old score loop had each thread walk a whole row a half at a time: D scalar loads a
    // key, rows 2 KB apart across a warp. fp16 caches, WarpSize 32 (subgroups on WebGPU), D / 8 a power of two up to 32. P.V and the reductions are unchanged.
    private static void AttnPartialGqaVecImpl<TG, TR, TP>(FV q, ArrayView<FusedDequantMatMul.W16> k16, ArrayView<global::ILGPU.Half> vc, IV seqs, IV posBuf, FV part, FV ml, int H, int kvH, int maxSeq, int mode, int nch, int ck, float scale)
        where TG : struct, IGqaGroup where TR : struct, IGqaReduce where TP : struct, IGqaPv
    {
        int p = posBuf[0];
        int g = Grid.IdxX, t = Group.IdxX, D = Group.DimX;
        int G = default(TG).G;
        int ch = g % nch, rest = g / nch, hk = rest % kvH, r = rest / kvH;
        int seq = mode == 0 ? 0 : seqs[r], pos = mode == 0 ? p + r : p;
        int kvC = kvH * D, j0 = ch * ck;
        var qs = SharedMemory.Allocate<float>(GqaMaxG * GqaMaxD);
        var sc = SharedMemory.Allocate<float>(GqaMaxG * AttChunk);
        var mxs = SharedMemory.Allocate<float>(GqaMaxG);
        var wmx = SharedMemory.Allocate<float>(GqaMaxG * GqaMaxWarps);
        if (Group.BarrierOr(j0 > pos))
        {
            if (t < G) { long gq = ((long)r * H + hk * G + t) * nch + ch; ml[2 * gq] = -1e30f; ml[2 * gq + 1] = 0f; }
            return;
        }
        var vq = vc.SubView((long)seq * maxSeq * kvC, (long)maxSeq * kvC);
        for (int gi = 0; gi < G; gi++) qs[gi * D + t] = q[((long)r * H + hk * G + gi) * D + t];
        Group.Barrier();
        int L = D >> 3;                                   // lanes a key row (16 at D = 128)
        int lane = t & 31, u = lane & (L - 1);            // this lane's 16-byte unit of the row
        int kps = D / L;                                  // keys a step (8 at D = 128)
        int keyOfThread = t / L;                          // which key of the step this lane's row is
        int qb = u * 8;
        int rowUnit0 = (seq * maxSeq * kvC + hk * D) / 8 + u;   // this sequence's key-row units: 32-bit (a cache layer is far below 2^31 units)
        int unitsPerKey = kvC / 8;
        for (int s0 = 0; s0 < ck; s0 += kps)
        {
            int jj = s0 + keyOfThread;
            int j = j0 + jj;
            bool inChunk = jj < ck;
            bool live = inChunk & (j <= pos);
            float a0 = 0f, a1 = 0f, a2 = 0f, a3 = 0f, a4 = 0f, a5 = 0f, a6 = 0f, a7 = 0f;
            if (live)
            {
                var w = k16.AsAligned16()[rowUnit0 + j * unitsPerKey];
                float k0 = FusedDequantMatMul.HalfBitsAll(w.a & 0xFFFF), k1 = FusedDequantMatMul.HalfBitsAll((w.a >> 16) & 0xFFFF);
                float k2 = FusedDequantMatMul.HalfBitsAll(w.b & 0xFFFF), k3 = FusedDequantMatMul.HalfBitsAll((w.b >> 16) & 0xFFFF);
                float k4 = FusedDequantMatMul.HalfBitsAll(w.c & 0xFFFF), k5 = FusedDequantMatMul.HalfBitsAll((w.c >> 16) & 0xFFFF);
                float k6 = FusedDequantMatMul.HalfBitsAll(w.d & 0xFFFF), k7 = FusedDequantMatMul.HalfBitsAll((w.d >> 16) & 0xFFFF);
                a0 = qs[qb] * k0 + qs[qb + 1] * k1 + qs[qb + 2] * k2 + qs[qb + 3] * k3 + qs[qb + 4] * k4 + qs[qb + 5] * k5 + qs[qb + 6] * k6 + qs[qb + 7] * k7;
                if (G > 1) { int o = D + qb; a1 = qs[o] * k0 + qs[o + 1] * k1 + qs[o + 2] * k2 + qs[o + 3] * k3 + qs[o + 4] * k4 + qs[o + 5] * k5 + qs[o + 6] * k6 + qs[o + 7] * k7; }
                if (G > 2) { int o = 2 * D + qb; a2 = qs[o] * k0 + qs[o + 1] * k1 + qs[o + 2] * k2 + qs[o + 3] * k3 + qs[o + 4] * k4 + qs[o + 5] * k5 + qs[o + 6] * k6 + qs[o + 7] * k7; }
                if (G > 3) { int o = 3 * D + qb; a3 = qs[o] * k0 + qs[o + 1] * k1 + qs[o + 2] * k2 + qs[o + 3] * k3 + qs[o + 4] * k4 + qs[o + 5] * k5 + qs[o + 6] * k6 + qs[o + 7] * k7; }
                if (G > 4) { int o = 4 * D + qb; a4 = qs[o] * k0 + qs[o + 1] * k1 + qs[o + 2] * k2 + qs[o + 3] * k3 + qs[o + 4] * k4 + qs[o + 5] * k5 + qs[o + 6] * k6 + qs[o + 7] * k7; }
                if (G > 5) { int o = 5 * D + qb; a5 = qs[o] * k0 + qs[o + 1] * k1 + qs[o + 2] * k2 + qs[o + 3] * k3 + qs[o + 4] * k4 + qs[o + 5] * k5 + qs[o + 6] * k6 + qs[o + 7] * k7; }
                if (G > 6) { int o = 6 * D + qb; a6 = qs[o] * k0 + qs[o + 1] * k1 + qs[o + 2] * k2 + qs[o + 3] * k3 + qs[o + 4] * k4 + qs[o + 5] * k5 + qs[o + 6] * k6 + qs[o + 7] * k7; }
                if (G > 7) { int o = 7 * D + qb; a7 = qs[o] * k0 + qs[o + 1] * k1 + qs[o + 2] * k2 + qs[o + 3] * k3 + qs[o + 4] * k4 + qs[o + 5] * k5 + qs[o + 6] * k6 + qs[o + 7] * k7; }
            }
            // the row's dot across its L lanes; every lane takes part (the shuffles stay in uniform control flow)
            for (int off = L >> 1; off > 0; off >>= 1)
            {
                a0 += Warp.ShuffleXor(a0, off);
                if (G > 1) a1 += Warp.ShuffleXor(a1, off);
                if (G > 2) a2 += Warp.ShuffleXor(a2, off);
                if (G > 3) a3 += Warp.ShuffleXor(a3, off);
                if (G > 4) a4 += Warp.ShuffleXor(a4, off);
                if (G > 5) a5 += Warp.ShuffleXor(a5, off);
                if (G > 6) a6 += Warp.ShuffleXor(a6, off);
                if (G > 7) a7 += Warp.ShuffleXor(a7, off);
            }
            bool writer = inChunk & (u == 0);
            if (writer)
            {
                sc[jj] = live ? a0 * scale : -1e30f;
                if (G > 1) sc[AttChunk + jj] = live ? a1 * scale : -1e30f;
                if (G > 2) sc[2 * AttChunk + jj] = live ? a2 * scale : -1e30f;
                if (G > 3) sc[3 * AttChunk + jj] = live ? a3 * scale : -1e30f;
                if (G > 4) sc[4 * AttChunk + jj] = live ? a4 * scale : -1e30f;
                if (G > 5) sc[5 * AttChunk + jj] = live ? a5 * scale : -1e30f;
                if (G > 6) sc[6 * AttChunk + jj] = live ? a6 * scale : -1e30f;
                if (G > 7) sc[7 * AttChunk + jj] = live ? a7 * scale : -1e30f;
            }
        }
        Group.Barrier();
        int keys = Math.Min(ck, pos + 1 - j0);
        int vo = hk * D + t;
        GqaChunkSoftmax<TG, TR>(qs, sc, mxs, wmx, ml, t, D, ck, ((long)r * H + hk * G) * nch + ch, nch);
        float o0 = 0f, o1 = 0f, o2 = 0f, o3 = 0f, o4 = 0f, o5 = 0f, o6 = 0f, o7 = 0f;
        if (default(TP).Ilp)
        {
            // four V rows in flight before their FMAs, the sums in the same key order; then the remainder one key at a time
            int kj = 0;
            for (; kj + 4 <= keys; kj += 4)
            {
                int vb = vo + (j0 + kj) * kvC;
                float v0 = PrecisionConvert.ConvertToSingle(vq[vb]), v1 = PrecisionConvert.ConvertToSingle(vq[vb + kvC]), v2 = PrecisionConvert.ConvertToSingle(vq[vb + 2 * kvC]), v3 = PrecisionConvert.ConvertToSingle(vq[vb + 3 * kvC]);
                o0 = float.MultiplyAddEstimate(sc[kj + 0], v0, o0); o0 = float.MultiplyAddEstimate(sc[kj + 1], v1, o0); o0 = float.MultiplyAddEstimate(sc[kj + 2], v2, o0); o0 = float.MultiplyAddEstimate(sc[kj + 3], v3, o0);
                if (G > 1) { o1 = float.MultiplyAddEstimate(sc[AttChunk + kj + 0], v0, o1); o1 = float.MultiplyAddEstimate(sc[AttChunk + kj + 1], v1, o1); o1 = float.MultiplyAddEstimate(sc[AttChunk + kj + 2], v2, o1); o1 = float.MultiplyAddEstimate(sc[AttChunk + kj + 3], v3, o1); }
                if (G > 2) { o2 = float.MultiplyAddEstimate(sc[2 * AttChunk + kj + 0], v0, o2); o2 = float.MultiplyAddEstimate(sc[2 * AttChunk + kj + 1], v1, o2); o2 = float.MultiplyAddEstimate(sc[2 * AttChunk + kj + 2], v2, o2); o2 = float.MultiplyAddEstimate(sc[2 * AttChunk + kj + 3], v3, o2); }
                if (G > 3) { o3 = float.MultiplyAddEstimate(sc[3 * AttChunk + kj + 0], v0, o3); o3 = float.MultiplyAddEstimate(sc[3 * AttChunk + kj + 1], v1, o3); o3 = float.MultiplyAddEstimate(sc[3 * AttChunk + kj + 2], v2, o3); o3 = float.MultiplyAddEstimate(sc[3 * AttChunk + kj + 3], v3, o3); }
                if (G > 4) { o4 = float.MultiplyAddEstimate(sc[4 * AttChunk + kj + 0], v0, o4); o4 = float.MultiplyAddEstimate(sc[4 * AttChunk + kj + 1], v1, o4); o4 = float.MultiplyAddEstimate(sc[4 * AttChunk + kj + 2], v2, o4); o4 = float.MultiplyAddEstimate(sc[4 * AttChunk + kj + 3], v3, o4); }
                if (G > 5) { o5 = float.MultiplyAddEstimate(sc[5 * AttChunk + kj + 0], v0, o5); o5 = float.MultiplyAddEstimate(sc[5 * AttChunk + kj + 1], v1, o5); o5 = float.MultiplyAddEstimate(sc[5 * AttChunk + kj + 2], v2, o5); o5 = float.MultiplyAddEstimate(sc[5 * AttChunk + kj + 3], v3, o5); }
                if (G > 6) { o6 = float.MultiplyAddEstimate(sc[6 * AttChunk + kj + 0], v0, o6); o6 = float.MultiplyAddEstimate(sc[6 * AttChunk + kj + 1], v1, o6); o6 = float.MultiplyAddEstimate(sc[6 * AttChunk + kj + 2], v2, o6); o6 = float.MultiplyAddEstimate(sc[6 * AttChunk + kj + 3], v3, o6); }
                if (G > 7) { o7 = float.MultiplyAddEstimate(sc[7 * AttChunk + kj + 0], v0, o7); o7 = float.MultiplyAddEstimate(sc[7 * AttChunk + kj + 1], v1, o7); o7 = float.MultiplyAddEstimate(sc[7 * AttChunk + kj + 2], v2, o7); o7 = float.MultiplyAddEstimate(sc[7 * AttChunk + kj + 3], v3, o7); }
            }
            for (; kj < keys; kj++)
            {
                float v = PrecisionConvert.ConvertToSingle(vq[vo + (j0 + kj) * kvC]);
                o0 = float.MultiplyAddEstimate(sc[kj], v, o0);
                if (G > 1) o1 = float.MultiplyAddEstimate(sc[AttChunk + kj], v, o1);
                if (G > 2) o2 = float.MultiplyAddEstimate(sc[2 * AttChunk + kj], v, o2);
                if (G > 3) o3 = float.MultiplyAddEstimate(sc[3 * AttChunk + kj], v, o3);
                if (G > 4) o4 = float.MultiplyAddEstimate(sc[4 * AttChunk + kj], v, o4);
                if (G > 5) o5 = float.MultiplyAddEstimate(sc[5 * AttChunk + kj], v, o5);
                if (G > 6) o6 = float.MultiplyAddEstimate(sc[6 * AttChunk + kj], v, o6);
                if (G > 7) o7 = float.MultiplyAddEstimate(sc[7 * AttChunk + kj], v, o7);
            }
        }
        else
        {
            for (int jj = 0; jj < keys; jj++)
            {
                float v = PrecisionConvert.ConvertToSingle(vq[vo + (j0 + jj) * kvC]);
                o0 = float.MultiplyAddEstimate(sc[jj], v, o0);
                if (G > 1) o1 = float.MultiplyAddEstimate(sc[AttChunk + jj], v, o1);
                if (G > 2) o2 = float.MultiplyAddEstimate(sc[2 * AttChunk + jj], v, o2);
                if (G > 3) o3 = float.MultiplyAddEstimate(sc[3 * AttChunk + jj], v, o3);
                if (G > 4) o4 = float.MultiplyAddEstimate(sc[4 * AttChunk + jj], v, o4);
                if (G > 5) o5 = float.MultiplyAddEstimate(sc[5 * AttChunk + jj], v, o5);
                if (G > 6) o6 = float.MultiplyAddEstimate(sc[6 * AttChunk + jj], v, o6);
                if (G > 7) o7 = float.MultiplyAddEstimate(sc[7 * AttChunk + jj], v, o7);
            }
        }
        long pb = ((long)r * H + hk * G) * nch + ch;
        part[pb * D + t] = o0;
        if (G > 1) part[(pb + nch) * D + t] = o1;
        if (G > 2) part[(pb + 2L * nch) * D + t] = o2;
        if (G > 3) part[(pb + 3L * nch) * D + t] = o3;
        if (G > 4) part[(pb + 4L * nch) * D + t] = o4;
        if (G > 5) part[(pb + 5L * nch) * D + t] = o5;
        if (G > 6) part[(pb + 6L * nch) * D + t] = o6;
        if (G > 7) part[(pb + 7L * nch) * D + t] = o7;
    }


    /// <summary>
    /// A chunk's softmax statistics for the G heads of a grouped-query attention group: each head's max over its chunk scores (sc, G rows of AttChunk), the scores
    /// replaced in place by exp(score - max) (masked scores, -1e30, by 0), the exp-sum, and (max, sum) written to ml at gq0 + head * nch. qs (G x D) is scratch:
    /// its queries must already be read. Either path's barriers publish sc for the P.V that follows. Shared by AttnPartialGqaIlpImpl and AttnPartialGqaVecImpl
    /// (inlined into each kernel); TR picks the tree or the warp-shuffle reduction (bit-identical, see IGqaReduce).
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static void GqaChunkSoftmax<TG, TR>(ArrayView<float> qs, ArrayView<float> sc, ArrayView<float> mxs, ArrayView<float> wmx, FV ml, int t, int D, int ck, long gq0, int nch)
        where TG : struct, IGqaGroup where TR : struct, IGqaReduce
    {
        int G = default(TG).G;
        if (default(TR).Shuffle)
        {
            // the same max and exp-sum with 2 barriers in place of 17 (see IGqaReduce). Max: each warp's lanes by Warp.ShuffleXor, the D / 32 warp maxima
            // through shared memory (a max takes no rounding: any order gives the tree's value). Sum: the per-thread partials through shared memory, then
            // every lane adds its column of warps in the tree's pairing ((c, c + 64) + (c + 32, c + 96) at D = 128) and the 32 columns by ShuffleXor 16, 8,
            // 4, 2, 1: the tree's last five levels, the same operands in the same pairs. Every lane holds the tree's sum; every thread runs every shuffle.
            int lane = t & 31, nw = D >> 5;
            for (int gi = 0; gi < G; gi++)
            {
                float m = -1e30f;
                for (int jj = t; jj < ck; jj += D) m = MathF.Max(m, sc[gi * AttChunk + jj]);
                m = default(TR).WarpMax(m);
                if (lane == 0) wmx[gi * GqaMaxWarps + (t >> 5)] = m;
            }
            Group.Barrier();
            for (int gi = 0; gi < G; gi++)
            {
                float m = wmx[gi * GqaMaxWarps];
                for (int w = 1; w < nw; w++) m = MathF.Max(m, wmx[gi * GqaMaxWarps + w]);
                float l = 0f;
                for (int jj = t; jj < ck; jj += D)
                {
                    int si = gi * AttChunk + jj;
                    float e = sc[si] <= -1e29f ? 0f : MathF.Exp(sc[si] - m);
                    sc[si] = e; l += e;
                }
                qs[gi * D + t] = l;                                  // qs is free: the scores were the last to read the queries
                if (t == gi) ml[2 * (gq0 + (long)t * nch)] = m;
            }
            Group.Barrier();                                         // the partials, and the probabilities the caller's P.V reads
            for (int gi = 0; gi < G; gi++)
            {
                int b = gi * D + lane;
                float s = nw == 4 ? (qs[b] + qs[b + 64]) + (qs[b + 32] + qs[b + 96]) : nw == 2 ? qs[b] + qs[b + 32] : qs[b];
                s = default(TR).WarpSum(s);
                if (t == gi) ml[2 * (gq0 + (long)t * nch) + 1] = s;
            }
        }
        else
        {
            // every head's chunk max, then exp-sum, in ONE tree reduction each (the probabilities replace the scores in place); the G heads reduce side by side, so
            // the barriers do not grow with G. MEASURED 2026-11-02 before it (WebGPU, qwen3-0.6b, G = 2, short prompt, captured steps): a reduction pair per head
            // made this kernel 14% slower per token than the per-head kernel (216 against 251 tok/s): half the groups, each with G times the barrier chain.
            for (int gi = 0; gi < G; gi++)
            {
                float m = -1e30f;
                for (int jj = t; jj < ck; jj += D) m = MathF.Max(m, sc[gi * AttChunk + jj]);
                qs[gi * D + t] = m;
            }
            Group.Barrier();
            for (int w = D / 2; w > 0; w >>= 1)
            {
                if (t < w)
                    for (int gi = 0; gi < G; gi++) qs[gi * D + t] = MathF.Max(qs[gi * D + t], qs[gi * D + t + w]);
                Group.Barrier();
            }
            if (t < G) mxs[t] = qs[t * D];
            Group.Barrier();
            for (int gi = 0; gi < G; gi++)
            {
                float m = mxs[gi], l = 0f;
                for (int jj = t; jj < ck; jj += D)
                {
                    int si = gi * AttChunk + jj;
                    float e = sc[si] <= -1e29f ? 0f : MathF.Exp(sc[si] - m);
                    sc[si] = e; l += e;
                }
                qs[gi * D + t] = l;
            }
            Group.Barrier();
            for (int w = D / 2; w > 0; w >>= 1)
            {
                if (t < w)
                    for (int gi = 0; gi < G; gi++) qs[gi * D + t] += qs[gi * D + t + w];
                Group.Barrier();
            }
            if (t < G)
            {
                long gq = gq0 + (long)t * nch;
                ml[2 * gq] = mxs[t]; ml[2 * gq + 1] = qs[t * D];
            }
        }
    }

    // AttnPartialGqaImpl with (1) a chunk wholly past the position leaving at once (Group.BarrierOr: uniform in WGSL) and (2) the K / V loads issued ahead of
    // their FMAs (eight K, four V; the same sums in the same order: bit-identical). MEASURED 2026-11-05 on CUDA (RTX 4070): see AttentionIlp.
    private static void AttnPartialGqaIlpImpl<T, TG, TR>(FV q, ArrayView1D<T, Stride1D.Dense> kc, ArrayView1D<T, Stride1D.Dense> vc, IV seqs, IV posBuf, FV part, FV ml, int H, int kvH, int maxSeq, int mode, int nch, int ck, float scale)
        where T : unmanaged, System.Numerics.INumber<T> where TG : struct, IGqaGroup where TR : struct, IGqaReduce
    {
        int p = posBuf[0];
        int g = Grid.IdxX, t = Group.IdxX, D = Group.DimX;
        int G = default(TG).G;                                      // a compile-time constant: the per-head guards below fold away (H / kvH == G)
        int ch = g % nch, rest = g / nch, hk = rest % kvH, r = rest / kvH;
        int seq = mode == 0 ? 0 : seqs[r], pos = mode == 0 ? p + r : p;
        int kvC = kvH * D, j0 = ch * ck;                             // ck keys a chunk (at most AttChunk: sc holds G x AttChunk scores)
        var qs = SharedMemory.Allocate<float>(GqaMaxG * GqaMaxD);
        var sc = SharedMemory.Allocate<float>(GqaMaxG * AttChunk);
        var mxs = SharedMemory.Allocate<float>(GqaMaxG);              // each head's chunk max (the reductions run in qs, free after the scores)
        var wmx = SharedMemory.Allocate<float>(GqaMaxG * GqaMaxWarps);       // the shuffle reduction's per-warp maxima
        // this sequence's rows of the caches: the offsets below stay 32-bit (a sequence's maxSeq x kvC is far below 2^31), so the inner loops carry no 64-bit
        // arithmetic (emulated on WebGPU: an i64 add per K element and an i64 multiply per V row before 2026-11-02)
        // a chunk wholly past the position (a captured step launches every chunk the cache holds): the empty partial it would compute (max -1e30, sum 0: the
        // merge skips it) and nothing else. Bit-identical to running it. The decision is the group's (j0, pos are), taken through Group.BarrierOr so WGSL sees
        // a uniform branch around the barriers below.
        if (Group.BarrierOr(j0 > pos))
        {
            if (t < G) { long gq = ((long)r * H + hk * G + t) * nch + ch; ml[2 * gq] = -1e30f; ml[2 * gq + 1] = 0f; }
            return;
        }
        var kq = kc.SubView((long)seq * maxSeq * kvC, (long)maxSeq * kvC);
        var vq = vc.SubView((long)seq * maxSeq * kvC, (long)maxSeq * kvC);
        for (int gi = 0; gi < G; gi++) qs[gi * D + t] = q[((long)r * H + hk * G + gi) * D + t];
        Group.Barrier();
        // scores: each key's row once, against the G heads
        for (int jj = t; jj < ck; jj += D)
        {
            int j = j0 + jj;
            float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f, s4 = 0f, s5 = 0f, s6 = 0f, s7 = 0f;
            bool live = j <= pos;
            if (live)
            {
                int ko = j * kvC + hk * D;
                // eight loads in flight before their FMAs (D is a multiple of 8: the route requires it); the sums take the same terms in the same order
                for (int d = 0; d < D; d += 8)
                {
                    float k0 = PrecisionConvert.ConvertToSingle(kq[ko + d + 0]); float k1 = PrecisionConvert.ConvertToSingle(kq[ko + d + 1]); float k2 = PrecisionConvert.ConvertToSingle(kq[ko + d + 2]); float k3 = PrecisionConvert.ConvertToSingle(kq[ko + d + 3]);
                    float k4 = PrecisionConvert.ConvertToSingle(kq[ko + d + 4]); float k5 = PrecisionConvert.ConvertToSingle(kq[ko + d + 5]); float k6 = PrecisionConvert.ConvertToSingle(kq[ko + d + 6]); float k7 = PrecisionConvert.ConvertToSingle(kq[ko + d + 7]);
                    s0 = float.MultiplyAddEstimate(qs[d + 0], k0, s0); s0 = float.MultiplyAddEstimate(qs[d + 1], k1, s0); s0 = float.MultiplyAddEstimate(qs[d + 2], k2, s0); s0 = float.MultiplyAddEstimate(qs[d + 3], k3, s0); s0 = float.MultiplyAddEstimate(qs[d + 4], k4, s0); s0 = float.MultiplyAddEstimate(qs[d + 5], k5, s0); s0 = float.MultiplyAddEstimate(qs[d + 6], k6, s0); s0 = float.MultiplyAddEstimate(qs[d + 7], k7, s0);
                    if (G > 1) { s1 = float.MultiplyAddEstimate(qs[D + d + 0], k0, s1); s1 = float.MultiplyAddEstimate(qs[D + d + 1], k1, s1); s1 = float.MultiplyAddEstimate(qs[D + d + 2], k2, s1); s1 = float.MultiplyAddEstimate(qs[D + d + 3], k3, s1); s1 = float.MultiplyAddEstimate(qs[D + d + 4], k4, s1); s1 = float.MultiplyAddEstimate(qs[D + d + 5], k5, s1); s1 = float.MultiplyAddEstimate(qs[D + d + 6], k6, s1); s1 = float.MultiplyAddEstimate(qs[D + d + 7], k7, s1); }
                    if (G > 2) { s2 = float.MultiplyAddEstimate(qs[2 * D + d + 0], k0, s2); s2 = float.MultiplyAddEstimate(qs[2 * D + d + 1], k1, s2); s2 = float.MultiplyAddEstimate(qs[2 * D + d + 2], k2, s2); s2 = float.MultiplyAddEstimate(qs[2 * D + d + 3], k3, s2); s2 = float.MultiplyAddEstimate(qs[2 * D + d + 4], k4, s2); s2 = float.MultiplyAddEstimate(qs[2 * D + d + 5], k5, s2); s2 = float.MultiplyAddEstimate(qs[2 * D + d + 6], k6, s2); s2 = float.MultiplyAddEstimate(qs[2 * D + d + 7], k7, s2); }
                    if (G > 3) { s3 = float.MultiplyAddEstimate(qs[3 * D + d + 0], k0, s3); s3 = float.MultiplyAddEstimate(qs[3 * D + d + 1], k1, s3); s3 = float.MultiplyAddEstimate(qs[3 * D + d + 2], k2, s3); s3 = float.MultiplyAddEstimate(qs[3 * D + d + 3], k3, s3); s3 = float.MultiplyAddEstimate(qs[3 * D + d + 4], k4, s3); s3 = float.MultiplyAddEstimate(qs[3 * D + d + 5], k5, s3); s3 = float.MultiplyAddEstimate(qs[3 * D + d + 6], k6, s3); s3 = float.MultiplyAddEstimate(qs[3 * D + d + 7], k7, s3); }
                    if (G > 4) { s4 = float.MultiplyAddEstimate(qs[4 * D + d + 0], k0, s4); s4 = float.MultiplyAddEstimate(qs[4 * D + d + 1], k1, s4); s4 = float.MultiplyAddEstimate(qs[4 * D + d + 2], k2, s4); s4 = float.MultiplyAddEstimate(qs[4 * D + d + 3], k3, s4); s4 = float.MultiplyAddEstimate(qs[4 * D + d + 4], k4, s4); s4 = float.MultiplyAddEstimate(qs[4 * D + d + 5], k5, s4); s4 = float.MultiplyAddEstimate(qs[4 * D + d + 6], k6, s4); s4 = float.MultiplyAddEstimate(qs[4 * D + d + 7], k7, s4); }
                    if (G > 5) { s5 = float.MultiplyAddEstimate(qs[5 * D + d + 0], k0, s5); s5 = float.MultiplyAddEstimate(qs[5 * D + d + 1], k1, s5); s5 = float.MultiplyAddEstimate(qs[5 * D + d + 2], k2, s5); s5 = float.MultiplyAddEstimate(qs[5 * D + d + 3], k3, s5); s5 = float.MultiplyAddEstimate(qs[5 * D + d + 4], k4, s5); s5 = float.MultiplyAddEstimate(qs[5 * D + d + 5], k5, s5); s5 = float.MultiplyAddEstimate(qs[5 * D + d + 6], k6, s5); s5 = float.MultiplyAddEstimate(qs[5 * D + d + 7], k7, s5); }
                    if (G > 6) { s6 = float.MultiplyAddEstimate(qs[6 * D + d + 0], k0, s6); s6 = float.MultiplyAddEstimate(qs[6 * D + d + 1], k1, s6); s6 = float.MultiplyAddEstimate(qs[6 * D + d + 2], k2, s6); s6 = float.MultiplyAddEstimate(qs[6 * D + d + 3], k3, s6); s6 = float.MultiplyAddEstimate(qs[6 * D + d + 4], k4, s6); s6 = float.MultiplyAddEstimate(qs[6 * D + d + 5], k5, s6); s6 = float.MultiplyAddEstimate(qs[6 * D + d + 6], k6, s6); s6 = float.MultiplyAddEstimate(qs[6 * D + d + 7], k7, s6); }
                    if (G > 7) { s7 = float.MultiplyAddEstimate(qs[7 * D + d + 0], k0, s7); s7 = float.MultiplyAddEstimate(qs[7 * D + d + 1], k1, s7); s7 = float.MultiplyAddEstimate(qs[7 * D + d + 2], k2, s7); s7 = float.MultiplyAddEstimate(qs[7 * D + d + 3], k3, s7); s7 = float.MultiplyAddEstimate(qs[7 * D + d + 4], k4, s7); s7 = float.MultiplyAddEstimate(qs[7 * D + d + 5], k5, s7); s7 = float.MultiplyAddEstimate(qs[7 * D + d + 6], k6, s7); s7 = float.MultiplyAddEstimate(qs[7 * D + d + 7], k7, s7); }
                }
            }
            sc[jj] = live ? s0 * scale : -1e30f;
            if (G > 1) sc[AttChunk + jj] = live ? s1 * scale : -1e30f;
            if (G > 2) sc[2 * AttChunk + jj] = live ? s2 * scale : -1e30f;
            if (G > 3) sc[3 * AttChunk + jj] = live ? s3 * scale : -1e30f;
            if (G > 4) sc[4 * AttChunk + jj] = live ? s4 * scale : -1e30f;
            if (G > 5) sc[5 * AttChunk + jj] = live ? s5 * scale : -1e30f;
            if (G > 6) sc[6 * AttChunk + jj] = live ? s6 * scale : -1e30f;
            if (G > 7) sc[7 * AttChunk + jj] = live ? s7 * scale : -1e30f;
        }
        Group.Barrier();
        int keys = Math.Min(ck, pos + 1 - j0);
        int vo = hk * D + t;
        GqaChunkSoftmax<TG, TR>(qs, sc, mxs, wmx, ml, t, D, ck, ((long)r * H + hk * G) * nch + ch, nch);
        float o0 = 0f, o1 = 0f, o2 = 0f, o3 = 0f, o4 = 0f, o5 = 0f, o6 = 0f, o7 = 0f;
        // four V rows in flight before their FMAs, the sums in the same key order; then the remainder one key at a time
        int kj = 0;
        for (; kj + 4 <= keys; kj += 4)
        {
            int vb = vo + (j0 + kj) * kvC;
            float v0 = PrecisionConvert.ConvertToSingle(vq[vb]), v1 = PrecisionConvert.ConvertToSingle(vq[vb + kvC]), v2 = PrecisionConvert.ConvertToSingle(vq[vb + 2 * kvC]), v3 = PrecisionConvert.ConvertToSingle(vq[vb + 3 * kvC]);
            o0 = float.MultiplyAddEstimate(sc[kj + 0], v0, o0); o0 = float.MultiplyAddEstimate(sc[kj + 1], v1, o0); o0 = float.MultiplyAddEstimate(sc[kj + 2], v2, o0); o0 = float.MultiplyAddEstimate(sc[kj + 3], v3, o0);
            if (G > 1) { o1 = float.MultiplyAddEstimate(sc[AttChunk + kj + 0], v0, o1); o1 = float.MultiplyAddEstimate(sc[AttChunk + kj + 1], v1, o1); o1 = float.MultiplyAddEstimate(sc[AttChunk + kj + 2], v2, o1); o1 = float.MultiplyAddEstimate(sc[AttChunk + kj + 3], v3, o1); }
            if (G > 2) { o2 = float.MultiplyAddEstimate(sc[2 * AttChunk + kj + 0], v0, o2); o2 = float.MultiplyAddEstimate(sc[2 * AttChunk + kj + 1], v1, o2); o2 = float.MultiplyAddEstimate(sc[2 * AttChunk + kj + 2], v2, o2); o2 = float.MultiplyAddEstimate(sc[2 * AttChunk + kj + 3], v3, o2); }
            if (G > 3) { o3 = float.MultiplyAddEstimate(sc[3 * AttChunk + kj + 0], v0, o3); o3 = float.MultiplyAddEstimate(sc[3 * AttChunk + kj + 1], v1, o3); o3 = float.MultiplyAddEstimate(sc[3 * AttChunk + kj + 2], v2, o3); o3 = float.MultiplyAddEstimate(sc[3 * AttChunk + kj + 3], v3, o3); }
            if (G > 4) { o4 = float.MultiplyAddEstimate(sc[4 * AttChunk + kj + 0], v0, o4); o4 = float.MultiplyAddEstimate(sc[4 * AttChunk + kj + 1], v1, o4); o4 = float.MultiplyAddEstimate(sc[4 * AttChunk + kj + 2], v2, o4); o4 = float.MultiplyAddEstimate(sc[4 * AttChunk + kj + 3], v3, o4); }
            if (G > 5) { o5 = float.MultiplyAddEstimate(sc[5 * AttChunk + kj + 0], v0, o5); o5 = float.MultiplyAddEstimate(sc[5 * AttChunk + kj + 1], v1, o5); o5 = float.MultiplyAddEstimate(sc[5 * AttChunk + kj + 2], v2, o5); o5 = float.MultiplyAddEstimate(sc[5 * AttChunk + kj + 3], v3, o5); }
            if (G > 6) { o6 = float.MultiplyAddEstimate(sc[6 * AttChunk + kj + 0], v0, o6); o6 = float.MultiplyAddEstimate(sc[6 * AttChunk + kj + 1], v1, o6); o6 = float.MultiplyAddEstimate(sc[6 * AttChunk + kj + 2], v2, o6); o6 = float.MultiplyAddEstimate(sc[6 * AttChunk + kj + 3], v3, o6); }
            if (G > 7) { o7 = float.MultiplyAddEstimate(sc[7 * AttChunk + kj + 0], v0, o7); o7 = float.MultiplyAddEstimate(sc[7 * AttChunk + kj + 1], v1, o7); o7 = float.MultiplyAddEstimate(sc[7 * AttChunk + kj + 2], v2, o7); o7 = float.MultiplyAddEstimate(sc[7 * AttChunk + kj + 3], v3, o7); }
        }
        for (; kj < keys; kj++)
        {
            float v = PrecisionConvert.ConvertToSingle(vq[vo + (j0 + kj) * kvC]);
            o0 = float.MultiplyAddEstimate(sc[kj], v, o0);
            if (G > 1) o1 = float.MultiplyAddEstimate(sc[AttChunk + kj], v, o1);
            if (G > 2) o2 = float.MultiplyAddEstimate(sc[2 * AttChunk + kj], v, o2);
            if (G > 3) o3 = float.MultiplyAddEstimate(sc[3 * AttChunk + kj], v, o3);
            if (G > 4) o4 = float.MultiplyAddEstimate(sc[4 * AttChunk + kj], v, o4);
            if (G > 5) o5 = float.MultiplyAddEstimate(sc[5 * AttChunk + kj], v, o5);
            if (G > 6) o6 = float.MultiplyAddEstimate(sc[6 * AttChunk + kj], v, o6);
            if (G > 7) o7 = float.MultiplyAddEstimate(sc[7 * AttChunk + kj], v, o7);
        }
        long pb = ((long)r * H + hk * G) * nch + ch;
        part[pb * D + t] = o0;
        if (G > 1) part[(pb + nch) * D + t] = o1;
        if (G > 2) part[(pb + 2L * nch) * D + t] = o2;
        if (G > 3) part[(pb + 3L * nch) * D + t] = o3;
        if (G > 4) part[(pb + 4L * nch) * D + t] = o4;
        if (G > 5) part[(pb + 5L * nch) * D + t] = o5;
        if (G > 6) part[(pb + 6L * nch) * D + t] = o6;
        if (G > 7) part[(pb + 7L * nch) * D + t] = o7;
    }
    private const int GqaMaxG = 8, GqaMaxD = 128, GqaMaxWarps = GqaMaxD / 32;
    /// <summary>
    /// How <see cref="AttnPartialGqaIlpImpl"/> reduces each head's chunk max and exp-sum, as a type (compiled per choice, the other path folds away): a shared-memory
    /// tree over the D threads (two 7-level chains at D = 128, a barrier a level) or Warp.ShuffleXor inside each warp with one shared hop between warps. The two
    /// give bit-identical results (the shuffle path keeps the tree's pairing for the sum). The shuffle path needs a real 32-lane warp and D in {32, 64, 128}.
    /// The shuffles live in the struct, not the kernel: ILGPU keeps the untaken <c>if (default(TR).Shuffle)</c> branch, so a WarpShuffle written in the kernel
    /// reached the tree variant's code and OpenCL without subgroup shuffles refused to compile it (MEASURED 2026-11-09, GqaAttention_MatchesPerHead on OpenCL).
    /// The tree struct's WarpMax / WarpSum are never called (its kernel takes the tree branch) and hold no shuffle.
    /// </summary>
    internal interface IGqaReduce { bool Shuffle { get; } float WarpMax(float v); float WarpSum(float v); }
    internal struct GqaReduceTree : IGqaReduce { public bool Shuffle => false; public float WarpMax(float v) => v; public float WarpSum(float v) => v; }
    /// <summary>How <see cref="AttnPartialGqaVecImpl"/> walks P.V: one V row at a time, or four rows loaded ahead of their FMAs (AttnPartialGqaIlpImpl's loop; the
    /// same sums in the same order: bit-identical).</summary>
    internal interface IGqaPv { bool Ilp { get; } }
    internal struct GqaPvPlain : IGqaPv { public bool Ilp => false; }
    internal struct GqaPvIlp : IGqaPv { public bool Ilp => true; }
    internal struct GqaReduceShuffle : IGqaReduce
    {
        public bool Shuffle => true;
        /// <summary>The warp's max in every lane (xor butterfly 16, 8, 4, 2, 1).</summary>
        public float WarpMax(float v) { for (int off = 16; off > 0; off >>= 1) v = MathF.Max(v, Warp.ShuffleXor(v, off)); return v; }
        /// <summary>The warp's sum in every lane, paired as a shared-memory tree's last five levels pair it (lane c with c + 16, then + 8, ...).</summary>
        public float WarpSum(float v) { for (int off = 16; off > 0; off >>= 1) v += Warp.ShuffleXor(v, off); return v; }
    }
    /// <summary>The query heads per kv head of <see cref="AttnPartialGqaImpl"/>, as a type: the kernel is compiled per G, so its G-way guards are constants.</summary>
    internal interface IGqaGroup { int G { get; } }
    internal struct Gqa2 : IGqaGroup { public int G => 2; }
    internal struct Gqa3 : IGqaGroup { public int G => 3; }
    internal struct Gqa4 : IGqaGroup { public int G => 4; }
    internal struct Gqa5 : IGqaGroup { public int G => 5; }
    internal struct Gqa6 : IGqaGroup { public int G => 6; }
    internal struct Gqa7 : IGqaGroup { public int G => 7; }
    internal struct Gqa8 : IGqaGroup { public int G => 8; }

    // split-K attention, part 2: out[r, h * D + d] = sum_ch e^(m_ch - M) part[ch, d] / sum_ch e^(m_ch - M) l_ch (chunks without a key skipped)
    // merges the chunks a row's keys reach: chunk c holds keys [c ck, (c + 1) ck), live when c ck <= the row's position (the rest hold sum 0 and were
    // skipped by the test below anyway: bounding the loops is bit-identical). A captured step launches every chunk the cache holds (MaxSeq / ck), so an
    // unbounded loop read all of them for every output.
    private static void AttnCombineImpl(Index1D i, FV part, FV ml, FV o, IV posBuf, int D, int nch, int ck, int H, int mode)
    {
        int ii = i, d = ii % D, rh = ii / D;
        int pos = mode == 0 ? posBuf[0] + rh / H : posBuf[0];
        int live = Math.Min(nch, pos / ck + 1);
        long g0 = (long)rh * nch;
        float M = -1e30f;
        for (int c = 0; c < live; c++) if (ml[2 * (g0 + c) + 1] > 0f) M = MathF.Max(M, ml[2 * (g0 + c)]);
        float num = 0f, den = 0f;
        for (int c = 0; c < live; c++)
        {
            float lc = ml[2 * (g0 + c) + 1];
            if (lc > 0f) { float w = MathF.Exp(ml[2 * (g0 + c)] - M); num = float.MultiplyAddEstimate(w, part[(g0 + c) * D + d], num); den = float.MultiplyAddEstimate(w, lc, den); }
        }
        o[ii] = num / den;
    }

    /// <summary>
    /// <see cref="AttnCombineImpl"/> with a group per (row, head) and a thread per value (D threads, D a power of two up to CombineMaxD): the live chunks'
    /// max M, their weights e^(m_c - M) and sums l_c are found ONCE into shared memory (the per-value kernel had every thread re-read every chunk's
    /// (m, l) twice and take every exp: two serial chains of global loads, 29 links each at 1,830 tokens), then each thread sums its value over the chunks
    /// in the same order with the same fused multiply-adds, four partials loaded ahead: bit-identical (a max takes no rounding; each weight is the
    /// same exp of the same operands). Needs nch <= CombineMaxChunks.
    /// </summary>
    private static void AttnCombineGroupImpl(FV part, FV ml, FV o, IV posBuf, int nch, int ck, int H, int mode)
    {
        var wc = SharedMemory.Allocate<float>(CombineMaxChunks);    // e^(m_c - M), 0 for a chunk without a key
        var lcs = SharedMemory.Allocate<float>(CombineMaxChunks);   // l_c
        var red = SharedMemory.Allocate<float>(CombineMaxD);
        int rh = Grid.IdxX, d = Group.IdxX, D = Group.DimX;
        o[(long)rh * D + d] = AttnCombineGroupValue(part, ml, posBuf, nch, ck, H, mode, wc, lcs, red);
    }

    // AttnCombineGroupImpl that also writes the merged rows as q8_1 for the o projection's opt-in int8-activation GEMV (FusedDequantMatMul.Int8ActGemv): the
    // same value, and a warp's lanes are 32 consecutive values of a head (D a multiple of 32), so the bytes equal the GEMV's own quantize. CUDA.
    private static void AttnCombineGroupQ8Impl(FV part, FV ml, FV o, IV posBuf, int nch, int ck, int H, int mode, ArrayView1D<byte, Stride1D.Dense> qs, FV ds)
    {
        var wc = SharedMemory.Allocate<float>(CombineMaxChunks);
        var lcs = SharedMemory.Allocate<float>(CombineMaxChunks);
        var red = SharedMemory.Allocate<float>(CombineMaxD);
        int rh = Grid.IdxX, d = Group.IdxX, D = Group.DimX;
        float y = AttnCombineGroupValue(part, ml, posBuf, nch, ck, H, mode, wc, lcs, red);
        int idx = rh * D + d;
        o[idx] = y;
        FusedDequantMatMul.Q8_1WarpStore(y, idx, qs, ds);
    }

    // the body of the group merge: this thread's merged value (row-head Grid.IdxX, value Group.IdxX); wc / lcs / red are the group's shared arrays
    private static float AttnCombineGroupValue(FV part, FV ml, IV posBuf, int nch, int ck, int H, int mode, ArrayView<float> wc, ArrayView<float> lcs, ArrayView<float> red)
    {
        int rh = Grid.IdxX, d = Group.IdxX, D = Group.DimX;
        int pos = mode == 0 ? posBuf[0] + rh / H : posBuf[0];
        int live = Math.Min(nch, pos / ck + 1);
        long g0 = (long)rh * nch;
        float m = -1e30f;
        for (int c = d; c < live; c += D) if (ml[2 * (g0 + c) + 1] > 0f) m = MathF.Max(m, ml[2 * (g0 + c)]);
        red[d] = m;
        Group.Barrier();
        for (int w = D / 2; w > 0; w >>= 1)
        {
            if (d < w) red[d] = MathF.Max(red[d], red[d + w]);
            Group.Barrier();
        }
        float M = red[0];
        for (int c = d; c < live; c += D)
        {
            float lc = ml[2 * (g0 + c) + 1];
            lcs[c] = lc;
            wc[c] = lc > 0f ? MathF.Exp(ml[2 * (g0 + c)] - M) : 0f;
        }
        Group.Barrier();
        float num = 0f, den = 0f;
        long pb = g0 * D + d;
        int c4 = 0;
        for (; c4 + 4 <= live; c4 += 4)
        {
            float p0 = part[pb + (long)c4 * D], p1 = part[pb + (long)(c4 + 1) * D], p2 = part[pb + (long)(c4 + 2) * D], p3 = part[pb + (long)(c4 + 3) * D];
            if (lcs[c4] > 0f) { num = float.MultiplyAddEstimate(wc[c4], p0, num); den = float.MultiplyAddEstimate(wc[c4], lcs[c4], den); }
            if (lcs[c4 + 1] > 0f) { num = float.MultiplyAddEstimate(wc[c4 + 1], p1, num); den = float.MultiplyAddEstimate(wc[c4 + 1], lcs[c4 + 1], den); }
            if (lcs[c4 + 2] > 0f) { num = float.MultiplyAddEstimate(wc[c4 + 2], p2, num); den = float.MultiplyAddEstimate(wc[c4 + 2], lcs[c4 + 2], den); }
            if (lcs[c4 + 3] > 0f) { num = float.MultiplyAddEstimate(wc[c4 + 3], p3, num); den = float.MultiplyAddEstimate(wc[c4 + 3], lcs[c4 + 3], den); }
        }
        for (; c4 < live; c4++)
            if (lcs[c4] > 0f) { num = float.MultiplyAddEstimate(wc[c4], part[pb + (long)c4 * D], num); den = float.MultiplyAddEstimate(wc[c4], lcs[c4], den); }
        return num / den;
    }
    private const int CombineMaxChunks = 1024, CombineMaxD = 256;

    // region 0's first n values into regions 1, 2, ... (sequence s + 1 gets element i of index s * n + i)
    private static void ForkImpl<T>(Index1D i, ArrayView1D<T, Stride1D.Dense> cache, long n, long region) where T : unmanaged, System.Numerics.INumber<T>
    {
        long ii = i, s = ii / n + 1, e = ii % n;
        cache[s * region + e] = cache[e];
    }

    // NeoX / rotate_half rotary in place: element k pairs with k + D/2, at the row's position
    // mode bit 0: decode (row r at p) or prefill (row r at p + r); bit 1: rotary on adjacent pairs (2k, 2k + 1, llama) instead of split halves (k, k + D/2)
    private static void RopeImpl(Index1D i, FV buf, FV invFreq, IV posBuf, int H, int D, int mode)
    {
        int p = posBuf[0];
        int half = D / 2, ii = i;
        int k = ii % half, rest = ii / half, h = rest % H, r = rest / H;
        int pos = (mode & 1) == 0 ? p + r : p;
        float ang = pos * invFreq[k], c = MathF.Cos(ang), s = MathF.Sin(ang);
        long b = (long)r * H * D + h * D;
        int i0 = (mode & 2) != 0 ? 2 * k : k, i1 = (mode & 2) != 0 ? 2 * k + 1 : half + k;
        float x0 = buf[b + i0], x1 = buf[b + i1];
        // the same explicit FMAs as QkvPostImpl (see there)
        buf[b + i0] = float.MultiplyAddEstimate(x0, c, -(x1 * s));
        buf[b + i1] = float.MultiplyAddEstimate(x1, c, x0 * s);
    }
    // an RMS norm of each head's D values in place (qwen3's q / k norms): group g = (row, head), a thread per value; rows are stride apart, the heads start at off
    private static void HeadNormImpl(FV buf, FV w, int stride, int off, int heads, float eps)
    {
        int g = Grid.IdxX, t = Group.IdxX, D = Group.DimX;
        long b = (long)(g / heads) * stride + off + (long)(g % heads) * D;
        var red = SharedMemory.Allocate<float>(AttChunk);
        float x = buf[b + t];
        red[t] = x * x;
        Group.Barrier();
        for (int s2 = D / 2; s2 > 0; s2 >>= 1)
        {
            if (t < s2) red[t] += red[t + s2];
            Group.Barrier();
        }
        buf[b + t] = x * (1f / MathF.Sqrt(red[0] / D + eps)) * w[t];
    }

    // HeadNormImpl for the joined q|k rows in ONE launch (qwen3's q and k norms were two): group g = (row, head) over the row's qHeads q heads then its k
    // heads, contiguous from the row's start (k begins at qHeads x D), each head normed with its own weight vector (q's or k's): the same math per head
    private static void HeadNormQkImpl(FV buf, FV wq, FV wk, int stride, int qHeads, int heads, float eps)
    {
        int g = Grid.IdxX, t = Group.IdxX, D = Group.DimX, hh = g % heads;
        long b = (long)(g / heads) * stride + (long)hh * D;
        var red = SharedMemory.Allocate<float>(AttChunk);
        float x = buf[b + t];
        red[t] = x * x;
        Group.Barrier();
        for (int s2 = D / 2; s2 > 0; s2 >>= 1)
        {
            if (t < s2) red[t] += red[t + s2];
            Group.Barrier();
        }
        float wt = hh < qHeads ? wq[t] : wk[t];
        buf[b + t] = x * (1f / MathF.Sqrt(red[0] / D + eps)) * wt;
    }

    // the row's k / v into its sequence's cache at its position
    private static void StoreImpl<T>(Index1D i, FV k, FV v, ArrayView1D<T, Stride1D.Dense> kc, ArrayView1D<T, Stride1D.Dense> vc, IV seqs, IV posBuf, int kvC, int maxSeq, int mode) where T : unmanaged, System.Numerics.INumber<T>
    {
        int p = posBuf[0];
        int ii = i, r = ii / kvC, c = ii % kvC;
        int seq = mode == 0 ? 0 : seqs[r], pos = mode == 0 ? p + r : p;
        long at = ((long)seq * maxSeq + pos) * kvC + c;
        kc[at] = PrecisionConvert.ConvertFromSingle<T>(k[ii]); vc[at] = PrecisionConvert.ConvertFromSingle<T>(v[ii]);
    }

    // scores[h, r, j] = q[r, h] . k[seq(r), j, h / rep] * scale for j <= pos(r), else masked
    private static void ScoresImpl<T>(Index1D i, FV q, ArrayView1D<T, Stride1D.Dense> kc, IV seqs, FV scores, int R, int nk, int H, int D, int kvH, int maxSeq, int p, int mode, float scale) where T : unmanaged, System.Numerics.INumber<T>
    {
        int ii = i, j = ii % nk, rest = ii / nk, r = rest % R, h = rest / R;
        int seq = mode == 0 ? 0 : seqs[r], pos = mode == 0 ? p + r : p;
        float s = -1e30f;
        if (j <= pos)
        {
            int kvC = kvH * D, hk = h / (H / kvH);
            long qo = (long)r * H * D + h * D, ko = ((long)seq * maxSeq + j) * kvC + hk * D;
            float acc = 0f;
            for (int d = 0; d < D; d++) acc = float.MultiplyAddEstimate(q[qo + d], PrecisionConvert.ConvertToSingle(kc[ko + d]), acc);
            s = acc * scale;
        }
        scores[ii] = s;
    }
    // the softmax of each score row in two kernels, so every thread stores one element per store site (WebGL's transform feedback captures no more; the
    // one-kernel version rewrote the row from a loop): the row's max and the inverse of its exp-sum, then each score normalised
    private static void SoftmaxStatsImpl(Index1D row, FV scores, FV stats, int nk)
    {
        long o = (long)(int)row * nk;
        float max = scores[o];
        for (int j = 1; j < nk; j++) max = MathF.Max(max, scores[o + j]);
        float sum = 0f;
        for (int j = 0; j < nk; j++) sum += MathF.Exp(scores[o + j] - max);
        stats[2L * row] = max;
        stats[2L * row + 1] = 1f / sum;
    }
    private static void SoftmaxNormImpl(Index1D i, FV scores, FV stats, int nk)
    {
        long ii = i, row = ii / nk;
        scores[ii] = MathF.Exp(scores[ii] - stats[2 * row]) * stats[2 * row + 1];
    }
    // o[r, h * D + d] = sum_j scores[h, r, j] v[seq(r), j, h / rep, d]
    private static void PvImpl<T>(Index1D i, FV scores, ArrayView1D<T, Stride1D.Dense> vc, IV seqs, FV o, int R, int nk, int H, int D, int kvH, int maxSeq, int mode) where T : unmanaged, System.Numerics.INumber<T>
    {
        int ii = i, C = H * D, r = ii / C, c = ii % C, h = c / D, d = c % D;
        int seq = mode == 0 ? 0 : seqs[r], kvC = kvH * D, hk = h / (H / kvH);
        long so = ((long)h * R + r) * nk, vo = (long)seq * maxSeq * kvC + hk * D + d;
        float acc = 0f;
        for (int j = 0; j < nk; j++) acc = float.MultiplyAddEstimate(scores[so + j], PrecisionConvert.ConvertToSingle(vc[vo + (long)j * kvC]), acc);
        o[ii] = acc;
    }

    public void Dispose()
    {
        foreach (var o in _owned) o.Dispose();
        _owned.Clear();
        foreach (var sp in _plans.Values) sp.Plan?.Dispose();
        _plans.Clear();
        foreach (var cp in _cudaPlans.Values) cp.Exec?.Dispose();
        _cudaPlans.Clear();
        _capStream?.Dispose();
        _vStream?.Dispose();
        _capStream = null;
        _mm.Dispose();
        _mmV?.Dispose();
        _norm.Dispose();
        _stream.Dispose();
    }
}
