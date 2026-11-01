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
    public bool FusedAttention { get; set; }
    private const int AttChunk = 256;
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
    private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _kc, _vc;
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
    private const int Chunk = 256;
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
        if (model.Tensors.Any(t => t.Name == "rope_freqs.weight"))
            throw new NotSupportedException("GgufBatchDecoder: per-frequency rotary factors (rope_freqs, llama 3 scaling)");
        _eps = model.GetMetadataFloat($"{a}.attention.layer_norm_rms_epsilon", 1e-6f);
        _theta = model.GetMetadataFloat($"{a}.rope.freq_base", 10000f);
        _embd = Tensor("token_embd.weight");
        Vocab = (int)_embd.Dimensions[1];
        if (Vocab >= 1 << 24) throw new NotSupportedException($"vocabulary {Vocab}: candidate ids travel as floats (exact below 2^24)");
        _head = model.Tensors.Any(t => t.Name == "output.weight") ? "output.weight" : "token_embd.weight";
        MaxSeq = maxSeq; MaxBatch = maxBatch;
        CaptureWebGPU = accelerator is SpawnDev.ILGPU.WebGPU.WebGPUAccelerator;
        DeviceSampling = SupportsDeviceDraws(accelerator);
        NoCandidates = accelerator.AcceleratorType == AcceleratorType.WebGL;
        // the fused attention runs a group per head (a power of two up to 256 threads; WebGL has no groups)
        FusedAttention = accelerator.AcceleratorType != AcceleratorType.WebGL && HeadDim <= AttChunk && (HeadDim & (HeadDim - 1)) == 0 && HeadDim <= accelerator.MaxNumThreadsPerGroup;
        var inv = new float[HeadDim / 2];
        for (int k = 0; k < inv.Length; k++) inv[k] = 1f / MathF.Pow(_theta, 2f * k / HeadDim);
        _invFreq = Own(_acc.Allocate1D(inv));
        int kvC = KvHeads * HeadDim;
        _kc = new MemoryBuffer1D<float, Stride1D.Dense>[Layers]; _vc = new MemoryBuffer1D<float, Stride1D.Dense>[Layers];
        for (int l = 0; l < Layers; l++) { _kc[l] = Own(_acc.Allocate1D<float>((long)maxBatch * maxSeq * kvC)); _vc[l] = Own(_acc.Allocate1D<float>((long)maxBatch * maxSeq * kvC)); }
    }

    private T Own<T>(T d) where T : IDisposable { _owned.Add(d); return d; }
    private GGUFTensorInfo Tensor(string name) => _model.Tensors.FirstOrDefault(t => t.Name == name) ?? throw new KeyNotFoundException($"GGUF tensor '{name}' not found");

    /// <summary>
    /// Load a GGUF from a seekable stream (kept open for the embedding rows; disposed with the decoder). <paramref name="maxSeq"/>: prompt plus generation;
    /// <paramref name="maxBatch"/>: most sequences at once (the caches take layers x 2 x maxBatch x maxSeq x kvHeads x headDim floats).
    /// </summary>
    public static async Task<GgufBatchDecoder> CreateAsync(Accelerator accelerator, Stream gguf, int maxSeq = 2048, int maxBatch = 8, CancellationToken ct = default)
    {
        if (!gguf.CanSeek) throw new ArgumentException("the GGUF stream must be seekable");
        gguf.Seek(0, SeekOrigin.Begin);
        var model = await GGUFParser.ParseHeaderAsync(gguf, ct).ConfigureAwait(false);
        model.SourceStream = gguf;
        var d = new GgufBatchDecoder(accelerator, model, gguf, maxSeq, maxBatch);
        await d.LoadAsync(ct).ConfigureAwait(false);
        return d;
    }

    private async Task<byte[]> ReadAsync(GGUFTensorInfo t, CancellationToken ct)
    {
        long n = GGMLTypes.TypeSize(t.Type, _model.GetTensorElementCount(t));
        var buf = new byte[(n + 3) / 4 * 4];                                      // the GEMM reads the blocks as int words: pad to 4
        _stream.Seek(_model.GetTensorDataOffset(t), SeekOrigin.Begin);
        int got = 0;
        while (got < n) { int r = await _stream.ReadAsync(buf.AsMemory(got, (int)(n - got)), ct).ConfigureAwait(false); if (r == 0) throw new EndOfStreamException(t.Name); got += r; }
        return buf;
    }

    private async Task LoadAsync(CancellationToken ct)
    {
        var mats = new List<string> { _head, "token_embd.weight" };          // the embedding rows are gathered on the device (a tied head is the same blocks)
        var vecs = new List<string> { "output_norm.weight" };
        _qkvAll = new bool[Layers];
        for (int l = 0; l < Layers; l++)
        {
            string p = $"blk.{l}.";
            var (tq, tk, tv) = (Tensor(p + "attn_q.weight").Type, Tensor(p + "attn_k.weight").Type, Tensor(p + "attn_v.weight").Type);
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
            if (JoinProjections && Tensor(p + "ffn_gate.weight").Type == Tensor(p + "ffn_up.weight").Type)
                await JoinAsync(p + "ffn_gate_up", new[] { p + "ffn_gate.weight", p + "ffn_up.weight" }, ct).ConfigureAwait(false);
            else mats.AddRange(new[] { p + "ffn_gate.weight", p + "ffn_up.weight" });
            mats.AddRange(new[] { p + "attn_output.weight", p + "ffn_down.weight" });
            vecs.AddRange(new[] { p + "attn_norm.weight", p + "ffn_norm.weight" });
            if (_qkNorm) vecs.AddRange(new[] { p + "attn_q_norm.weight", p + "attn_k_norm.weight" });
        }
        foreach (var name in mats.Distinct())
        {
            var t = Tensor(name);
            if (!FusedDequantMatMul.Supports(t.Type)) throw new NotSupportedException($"{name}: {t.Type} is not a block type the fused GEMM reads");
            _mats[name] = (Own(_acc.Allocate1D(await ReadAsync(t, ct).ConfigureAwait(false))), t.Type);
        }
        foreach (var name in vecs)
        {
            var t = Tensor(name);
            if (t.Type != GGMLType.F32) throw new NotSupportedException($"{name}: {t.Type} (expected F32)");
            var raw = await ReadAsync(t, ct).ConfigureAwait(false);
            var f = new float[_model.GetTensorElementCount(t)];
            Buffer.BlockCopy(raw, 0, f, 0, f.Length * 4);
            _vecs[name] = Own(_acc.Allocate1D(f));
        }
    }

    // matrices of one block type stacked along the output dimension: their rows (the blocks of each output) follow each other, so the bytes simply concatenate
    private async Task JoinAsync(string name, string[] parts, CancellationToken ct)
    {
        var type = Tensor(parts[0]).Type;
        if (!FusedDequantMatMul.Supports(type)) throw new NotSupportedException($"{parts[0]}: {type} is not a block type the fused GEMM reads");
        var chunks = new List<byte[]>();
        long total = 0;
        foreach (var part in parts)
        {
            var t = Tensor(part);
            long n = GGMLTypes.TypeSize(t.Type, _model.GetTensorElementCount(t));
            var raw = await ReadAsync(t, ct).ConfigureAwait(false);
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
    private async Task<float[]> VecAsync(string name, CancellationToken ct)
    {
        var t = Tensor(name);
        if (t.Type != GGMLType.F32) throw new NotSupportedException($"{name}: {t.Type} (expected F32)");
        var raw = await ReadAsync(t, ct).ConfigureAwait(false);
        var f = new float[_model.GetTensorElementCount(t)];
        Buffer.BlockCopy(raw, 0, f, 0, f.Length * 4);
        return f;
    }
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
        if (_norm.AddRMSNormFused(a, b, sum, normed, w, rows, C, _eps, 1)) return;
        _add3!((int)((long)rows * C), a, b, sum);
        _norm.RMSNorm(sum, normed, w, rows, C, _eps);
    }

    // decode steps (a few rows) read each weight once for all rows (ForwardRows); a prefill is a real GEMM
    private void Gemm(FV x, string name, FV o, int rows, int K, int N)
    {
        var (w, t) = _mats[name];
        if (rows <= FusedDequantMatMul.RowsMax) _mm.ForwardRows(x, w.View, o, rows, K, N, t);
        else _mm.Forward(x, w.View, o, rows, K, N, t);
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
        posView.CopyFromCPU(new[] { p });
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
        for (int l = 0; l < Layers; l++)
        {
            string b = $"blk.{l}.";
            if (_mats.ContainsKey(b + "attn_qkv"))
            {
                // one GEMV for q|k(|v), then bias + rotary + the cache store in one kernel (q to its buffer, k and v to the caches)
                bool all = _qkvAll[l];
                int n = all ? Q + 2 * kvC : Q + kvC;
                var qkv = Buf(ref _qkv, (long)R * (Q + 2 * kvC));
                Gemm(hn, b + "attn_qkv", qkv, R, C, n); Prof("qkv");
                var vb = att;                                                // unread when v is in qkv (a distinct buffer: no aliasing)
                if (!all) { Gemm(hn, b + "attn_v.weight", v, R, C, kvC); vb = v; Prof("v"); }
                if (_qkNorm)
                {
                    // qwen3: an RMS norm per head of q and of k (weight [head width]) before the rotary; no biases, so the post kernel's zero bias changes nothing
                    _headNorm!(new KernelConfig(R * Heads, HeadDim), qkv, _vecs[b + "attn_q_norm.weight"].View, n, 0, Heads, _eps);
                    _headNorm(new KernelConfig(R * KvHeads, HeadDim), qkv, _vecs[b + "attn_k_norm.weight"].View, n, Q, KvHeads, _eps); Prof("qk-norm");
                }
                int U = Heads * (HeadDim / 2) + KvHeads * (HeadDim / 2) + kvC;
                _qkvPost!(R * U, qkv, vb, _vecs[b + "attn_qkv.bias"].View, _invFreq.View, q, _kc[l].View, _vc[l].View, seqView, posView, Heads | (KvHeads << 16), HeadDim, n, (all ? 1 : 0) | (mode << 1) | (rope << 1), MaxSeq); Prof("qkv-post");
            }
            else
            {
                Gemm(hn, b + "attn_q.weight", q, R, C, Q); _bias!((int)((long)R * Q), q, _vecs[b + "attn_q.bias"].View, Q); Prof("q");
                Gemm(hn, b + "attn_k.weight", k, R, C, kvC); _bias!((int)((long)R * kvC), k, _vecs[b + "attn_k.bias"].View, kvC); Prof("k");
                Gemm(hn, b + "attn_v.weight", v, R, C, kvC); _bias!((int)((long)R * kvC), v, _vecs[b + "attn_v.bias"].View, kvC); Prof("v");
                if (_qkNorm)
                {
                    _headNorm!(new KernelConfig(R * Heads, HeadDim), q, _vecs[b + "attn_q_norm.weight"].View, Q, 0, Heads, _eps);
                    _headNorm(new KernelConfig(R * KvHeads, HeadDim), k, _vecs[b + "attn_k_norm.weight"].View, kvC, 0, KvHeads, _eps); Prof("qk-norm");
                }
                _rope!(R * Heads * (HeadDim / 2), q, _invFreq.View, posView, Heads, HeadDim, mode | rope);
                _rope!(R * KvHeads * (HeadDim / 2), k, _invFreq.View, posView, KvHeads, HeadDim, mode | rope); Prof("rope");
                _store!((int)((long)R * kvC), k, v, _kc[l].View, _vc[l].View, seqView, posView, kvC, MaxSeq, mode); Prof("store");
            }
            if (FusedAttention)
            {
                // split-K attention: a group per (row, head, chunk of keys) computes the chunk's scores, max, exp-sum and P.V; a second kernel merges the chunks
                // a captured step keeps one shape for the whole generation: every chunk the cache can hold (the ones past the position mask to nothing)
                int ach = _fixedChunks ? (MaxSeq + AttChunk - 1) / AttChunk : (nk + AttChunk - 1) / AttChunk;
                var part = Buf(ref _attPart, (long)R * Heads * ach * HeadDim); var ml = Buf(ref _attMl, (long)R * Heads * ach * 2);
                _attPartial!(new KernelConfig(R * Heads * ach, HeadDim), q, _kc[l].View, _vc[l].View, seqView, posView, part, ml, Heads, KvHeads, MaxSeq, mode, ach, 1f / MathF.Sqrt(HeadDim)); Prof("attn");
                _attCombine!((int)((long)R * Q), part, ml, att, HeadDim, ach); Prof("attn-merge");
            }
            else
            {
                var scores = Buf(ref _scores, (long)Heads * R * nk);
                _scoresK!((int)((long)Heads * R * nk), q, _kc[l].View, seqView, scores, R, nk, Heads, HeadDim, KvHeads, MaxSeq, p, mode, 1f / MathF.Sqrt(HeadDim)); Prof("scores");
                var stats = Buf(ref _softStats, 2L * Heads * R);
                _softmaxStats!(Heads * R, scores, stats, nk);
                _softmaxNorm!((int)((long)Heads * R * nk), scores, stats, nk); Prof("softmax");
                _pv!((int)((long)R * Q), scores, _vc[l].View, seqView, att, R, nk, Heads, HeadDim, KvHeads, MaxSeq, mode); Prof("pv");
            }
            Gemm(att, b + "attn_output.weight", proj, R, Q, C); Prof("o");
            AddNorm(hid, proj, hid2, hn, _vecs[b + "ffn_norm.weight"].View, R, C); Prof("add+rms");
            if (_mats.ContainsKey(b + "ffn_gate_up"))
            {
                var gu = Buf(ref _gateUp, (long)R * 2 * FfnDim);
                Gemm(hn, b + "ffn_gate_up", gu, R, C, 2 * FfnDim); Prof("gate-up");
                _swigluJoined!((int)((long)R * FfnDim), gu, gate, FfnDim); Prof("swiglu");
            }
            else
            {
                Gemm(hn, b + "ffn_gate.weight", gate, R, C, FfnDim); Prof("gate");
                Gemm(hn, b + "ffn_up.weight", up, R, C, FfnDim); Prof("up");
                _swiglu!((int)((long)R * FfnDim), gate, up); Prof("swiglu");
            }
            Gemm(gate, b + "ffn_down.weight", proj, R, FfnDim, C); Prof("down");
            AddNorm(hid2, proj, hid, hn, _vecs[l + 1 < Layers ? $"blk.{l + 1}.attn_norm.weight" : "output_norm.weight"].View, R, C); Prof("add+rms");
        }
        // the LM head on the output norm's rows: every row when decoding, the last row of a prefill
        int outRows = decode ? R : 1;
        var hnOut = decode ? hn : hn.SubView((long)(R - 1) * C, C);
        var logits = Buf(ref _logits, (long)outRows * Vocab);
        Gemm(hnOut, _head, logits, outRows, C, Vocab); Prof("head");
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
        _chunkMax!(outRows * nch, logits, maxV, maxI, Vocab, nch, Chunk); Prof("chunkMax");
        _floor!(SpawnDev.ILGPU.ML.Kernels.TopKKernels.FloorThreads(outRows, nch), maxV, maxI, cand, _selN.View, nch, S); Prof("floor");
        _select!(outRows * nch, logits, cand, _selN.View, selV, _selI.View, Vocab, nch, S, temperature, Chunk); Prof("select");
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
        if (ProfileKernels) { _acc.Synchronize(); _profT = t0; }
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
    public Task PrefillAsync(int[] promptIds, float temperature = 0f, CancellationToken ct = default)
    {
        if (promptIds.Length == 0) throw new ArgumentException("an empty prompt");
        Position = 0; Batch = 1; DecodeTimings = default; RowReads = 0; KernelTimes.Clear(); _drawPos = -1;
        return ForwardAsync(promptIds, null, decode: false, temperature, readBack: false, ct);
    }

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
                _fork!((int)((count - 1) * n), _kc[l].View, n, region);
                _fork((int)((count - 1) * n), _vc[l].View, n, region);
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
    public async ValueTask<int> SampleAsync(int row, float topP, Random rng)
    {
        if ((uint)row >= (uint)_lastRows) throw new ArgumentOutOfRangeException(nameof(row));
        if (NoCandidates) return SampleLogits(await _lastLogits.SubView((long)row * Vocab, Vocab).CopyToHostAsync().ConfigureAwait(false), 0, Vocab, _lastT, topP, rng);
        int K = Candidates, o = row * CandStride;
        var c = _lastCand ??= await _lastCandView.CopyToHostAsync().ConfigureAwait(false);
        float T = _lastT, max = c[o], floor = max - 20f * T;
        double total = 0, keep = 0; int m = 0;
        bool whole = c[o + 2 * K] > SelectCap;                                   // too many tokens reached the floor: the candidates are not the top K
        if (!whole)
        {
            if (T <= 0f) return (int)c[o + K];
            for (int ch = 0; ch < Chunks; ch++) total += c[o + 2 * K + 1 + ch];
            while (m < K && c[o + m] >= floor && keep < topP * total) { keep += Math.Exp((c[o + m] - max) / T); m++; }
            whole = keep < topP * total;                                         // the nucleus reaches past the candidates (or the device's float sums end it a hair later)
        }
        if (whole)
        {
            RowReads++;
            var full = await _lastLogits.SubView((long)row * Vocab, Vocab).CopyToHostAsync().ConfigureAwait(false);
            return SampleLogits(full, 0, Vocab, T, topP, rng);
        }
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
    public void Draw(float topP, int seed, float repetitionPenalty = 1f, int repeatLastN = 64)
    {
        if (_lastRows == 0) throw new InvalidOperationException("no forward to draw from");
        if (!SupportsDeviceDraws(_acc)) throw new NotSupportedException($"device draws need groups of {DrawGroup} threads ({_acc.AcceleratorType}): use SampleAsync");
        if (!(topP > 0f && topP <= 1f)) throw new ArgumentOutOfRangeException(nameof(topP), "(0, 1]");
        if (!(repetitionPenalty > 0f)) throw new ArgumentOutOfRangeException(nameof(repetitionPenalty), "above 0");
        LoadKernels(); EnsureDrawBuffers();
        int window = repeatLastN > 0 ? repeatLastN : MaxSeq;
        EnsurePenaltyWindow(window);
        _rep!.View.SubView(0, MaxBatch).MemSetToZero();                          // every sequence's window starts empty
        _drawI!.View.SubView(0, 5).CopyFromCPU(new[] { seed, ForceFullRowDraws ? 1 : 0, window, _repCap, MaxBatch });
        _drawF!.View.SubView(0, 2).CopyFromCPU(new[] { topP, repetitionPenalty });
        _penOn = repetitionPenalty != 1f;
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
        Action<int, int>? onToken = null, Action<int>? onFinished = null, CancellationToken ct = default, float repetitionPenalty = 1f, int repeatLastN = 64)
    {
        await PrefillAsync(promptIds, temperature, ct).ConfigureAwait(false);
        Fork(n);
        var outs = Enumerable.Range(0, n).Select(_ => new List<int>()).ToArray();
        var done = new bool[n];
        // a token for sequence s: true when the sequence goes on
        bool Take(int s, int token)
        {
            if (stopIds.Contains(token) || outs[s].Count >= maxNew) { done[s] = true; onFinished?.Invoke(s); return false; }
            outs[s].Add(token); onToken?.Invoke(s, token);
            return true;
        }
        void FinishAll() { for (int s = 0; s < n; s++) if (!done[s]) { done[s] = true; onFinished?.Invoke(s); } }
        if (!DeviceSampling)
        {
            var rngs = Enumerable.Range(0, n).Select(s => new Random(seed + s)).ToArray();
            var next = new int[n];
            for (int s = 0; s < n; s++) next[s] = await SampleAsync(0, topP, rngs[s]).ConfigureAwait(false);
            while (true)
            {
                for (int s = 0; s < n; s++) if (!done[s]) Take(s, next[s]);
                if (done.All(d => d)) break;
                if (Position + 1 > MaxSeq) { FinishAll(); break; }
                ct.ThrowIfCancellationRequested();
                // only the live sequences step: a finished one leaves the batch (it used to ride along to the end of the longest)
                var live = Enumerable.Range(0, n).Where(s => !done[s]).ToArray();
                await StepAsync(live.Select(s => next[s]).ToArray(), live, temperature, ct).ConfigureAwait(false);
                if (repetitionPenalty == 1f)
                    for (int r = 0; r < live.Length; r++) next[live[r]] = await SampleAsync(r, topP, rngs[live[r]]).ConfigureAwait(false);
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
                        TextGenerationSampler.ApplyRepetitionPenalty(row, o.GetRange(o.Count - take, take).ToArray(), repetitionPenalty);
                        next[live[r]] = SampleLogits(row, 0, Vocab, temperature, topP, rngs[live[r]]);
                    }
                }
            }
            return outs.Select(o => o.ToArray()).ToArray();
        }
        // device draws, one step ahead: step P (its inputs the tokens drawn for P, its draws the tokens for P + 1) is queued before the tokens for P are read, so the
        // device always has the next step while the host streams the last one. A sequence that drew a stop token rides one step more (its row is thrown away), then
        // leaves the batch.
        Draw(temperature > 0f ? topP : 1f, seed, repetitionPenalty, repeatLastN);
        var live2 = Enumerable.Range(0, n).ToArray();
        Task<int[]>? pending = ReadDrawnAsync(), ahead = null;
        try
        {
            while (true)
            {
                ct.ThrowIfCancellationRequested();
                ahead = null;
                if (Position + 1 <= MaxSeq) { await StepDrawnAsync(live2, temperature, ct).ConfigureAwait(false); ahead = ReadDrawnAsync(); }
                var tokens = await pending!.ConfigureAwait(false);
                pending = null;
                foreach (int s in live2) Take(s, tokens[s]);
                live2 = live2.Where(s => !done[s]).ToArray();
                if (live2.Length == 0) break;
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
    public static int SampleLogits(float[] logits, int row, int vocab, float temperature, float topP, double u)
    {
        int V = vocab, o = row * V, best = 0;
        for (int i = 1; i < V; i++) if (logits[o + i] > logits[o + best]) best = i;
        if (temperature <= 0f) return best;
        float max = logits[o + best], floor = max - 20f * temperature;
        var cand = new List<(int Id, float L, double P)>();
        for (int i = 0; i < V; i++) if (logits[o + i] >= floor) cand.Add((i, logits[o + i], Math.Exp((logits[o + i] - max) / temperature)));
        cand.Sort((a, b) => a.L != b.L ? b.L.CompareTo(a.L) : a.Id.CompareTo(b.Id));
        double total = cand.Sum(c => c.P), keep = 0; int m = 0;
        while (m < cand.Count && keep < topP * total) keep += cand[m++].P;
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
    private Action<Index1D, FV, FV, FV, FV, IV, IV, int, int, int>? _store;
    private Action<Index1D, FV, FV, IV, FV, int, int, int, int, int, int, int, int, float>? _scoresK;
    private Action<Index1D, FV, FV, int>? _softmaxStats, _softmaxNorm;
    private Action<Index1D, FV, FV, IV, FV, int, int, int, int, int, int, int>? _pv;
    private Action<Index1D, FV, FV>? _swiglu;
    private Action<Index1D, FV, FV, FV, FV, FV, FV, FV, IV, IV, int, int, int, int, int>? _qkvPost;
    private Action<Index1D, FV, FV, int>? _swigluJoined;
    private Action<KernelConfig, FV, FV, int, int, int, float>? _headNorm;
    private Action<KernelConfig, FV, FV, FV, IV, IV, FV, FV, int, int, int, int, int, float>? _attPartial;
    private Action<Index1D, FV, FV, FV, int, int>? _attCombine;
    private Action<Index1D, FV, long, long>? _fork;
    private Action<Index1D, IV, ArrayView1D<int, Stride1D.Dense>, FV, int, int, int>? _embed;
    private Action<Index1D, FV, FV, IV, int, int, int>? _chunkMax;
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
        _store = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, IV, IV, int, int, int>(StoreImpl);
        _scoresK = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, FV, int, int, int, int, int, int, int, int, float>(ScoresImpl);
        _softmaxStats = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int>(SoftmaxStatsImpl);
        _softmaxNorm = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int>(SoftmaxNormImpl);
        _pv = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, FV, int, int, int, int, int, int, int>(PvImpl);
        _swiglu = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(SwigluImpl);
        _qkvPost = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, FV, FV, FV, IV, IV, int, int, int, int, int>(QkvPostImpl);
        _swigluJoined = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int>(SwigluJoinedImpl);
        if (_qkNorm) _headNorm = _acc.LoadStreamKernel<FV, FV, int, int, int, float>(HeadNormImpl);
        if (FusedAttention)
        {
            _attPartial = _acc.LoadStreamKernel<FV, FV, FV, IV, IV, FV, FV, int, int, int, int, int, float>(AttnPartialImpl);
            _attCombine = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int>(AttnCombineImpl);
        }
        _fork = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, long, long>(ForkImpl);
        _embed = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, ArrayView1D<int, Stride1D.Dense>, FV, int, int, int>(EmbedImpl);
        if (!NoCandidates)
        {
            _chunkMax = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, int, int, int>(SpawnDev.ILGPU.ML.Kernels.TopKKernels.ChunkMaxImpl);
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
    // after the joined q|k(|v) GEMV (row stride sa): bias, rotary (NeoX pairs d, d + D/2) and the store, the arithmetic of BiasImpl + RopeImpl + StoreImpl. Per row the
    // threads are the q pairs, the k pairs, then v's values; q goes to qOut [r, C], k and v to their sequence's cache row; v comes from qkv (flags bit 0) or from vb
    // (row kvC); flags bit 1 = decode (row r is sequence seqs[r] at p; prefill: sequence 0 at p + r), bit 2 = rotary on adjacent pairs (llama); heads = H | kvH << 16
    private static void QkvPostImpl(Index1D i, FV qkv, FV vb, FV bias, FV invFreq, FV qOut, FV kc, FV vc, IV seqs, IV posBuf, int heads, int D, int sa, int flags, int maxSeq)
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
            float y0 = x0 * c - x1 * s, y1 = x1 * c + x0 * s;
            if (isQ) { long qo = (long)r * C + hh * D; qOut[qo + i0] = y0; qOut[qo + i1] = y1; }
            else { long ko = cacheRow + hh * D; kc[ko + i0] = y0; kc[ko + i1] = y1; }
        }
        else
        {
            int c2 = u - qPairs - kPairs;
            float x = vInA != 0 ? qkv[row + C + kvC + c2] : vb[(long)r * kvC + c2];
            vc[cacheRow + c2] = x + bias[C + kvC + c2];
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
        int[][]? windows = null, float repetitionPenalty = 1f, int repeatLastN = 0, int ringCapacity = 0)
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
        using var dF2 = acc.Allocate1D(new[] { topP, repetitionPenalty });
        using var dI = acc.Allocate1D(new[] { seed, forceFullRow ? 1 : 0, active, W, rows });
        using var ids = acc.Allocate1D<int>(Math.Max(64, rows));
        var chunkMax = acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, int, int, int>(SpawnDev.ILGPU.ML.Kernels.TopKKernels.ChunkMaxImpl);
        var floor = acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, FV, IV, int, int>(SpawnDev.ILGPU.ML.Kernels.TopKKernels.FloorImpl);
        var select = acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, FV, IV, int, int, int, float, int>(SpawnDev.ILGPU.ML.Kernels.TopKKernels.SelectImpl);
        var final = acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, FV, int>(SpawnDev.ILGPU.ML.Kernels.TopKKernels.FinalImpl);
        var draw = acc.LoadStreamKernel<FV, FV, IV, IV, FV, IV, IV, IV, int, int, int, int, int, float>(DrawImpl);
        var penalize = acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, FV, IV, int>(PenalizeImpl);
        penalize(rows * W, lg.View, seqs.View, dI.View, dF2.View, rep.View, vocab);
        chunkMax(rows * nch, lg.View, maxV.View, maxI.View, vocab, nch, Chunk);
        floor(SpawnDev.ILGPU.ML.Kernels.TopKKernels.FloorThreads(rows, nch), maxV.View, maxI.View, cand.View, selN.View, nch, S);
        select(rows * nch, lg.View, cand.View, selN.View, selV.View, selI.View, vocab, nch, S, temperature, Chunk);
        final(SpawnDev.ILGPU.ML.Kernels.TopKKernels.FinalThreads(rows), selV.View, selI.View, selN.View, cand.View, S);
        draw(new KernelConfig(rows, DrawGroup), lg.View, cand.View, seqs.View, pos.View, dF2.View, dI.View, ids.View, rep.View, vocab, nch, S, 0, 1, temperature);
        var got = await ids.View.SubView(0, rows).CopyToHostAsync().ConfigureAwait(false);      // the read waits for the kernels (every backend), so the buffers may go
        return got;
    }

    // a step's input under device draws: row r is the token its sequence drew last
    private static void GatherIdsImpl(Index1D r, IV seqIds, IV seqs, IV ids) => ids[r] = seqIds[seqs[r]];

    // the draw: group g = slot g (sequence seqs[g], from row g of the forward, or row 0 when rowZero), DrawGroup threads; the token goes to seqIds[sequence]. The
    // uniform is DrawUniform(seed, sequence, position + tAdd); drawF = [topP], drawI = [seed, force the full-row path].
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
        int tok = (int)cand[o + K];
        bool needFull = false;
        if (t == 0 && sampling)
        {
            bool wide = forced | (cand[o + 2 * K] > SelectCap);
            if (wide) needFull = true;
            else
            {
                float total = 0f;
                for (int ch = 0; ch < nch; ch++) total += cand[o + 2 * K + 1 + ch];
                float floorT = max - 20f * T, target = topP * total, keep = 0f;
                int m = 0;
                for (int j = 0; j < K; j++)
                {
                    float cv = cand[o + j];
                    bool take = (cv >= floorT) & (keep < target);
                    if (!take) break;
                    keep += MathF.Exp((cv - max) / T);
                    m = j + 1;
                }
                if (keep < target) needFull = true;                               // the nucleus reaches past the candidates
                else
                {
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
            float total = GroupSum(red, t, MassPart(logits, lb, V, max, T, floorT, floorT, false, t));
            float target = topP * total;
            // tau: the largest key whose mass at or above reaches the target (true at floorT: its mass is the total)
            uint lo = OrderKey(floorT), hi = OrderKey(max);
            for (int it = 0; it < 33; it++)
            {
                uint mid = lo + ((hi - lo + 1u) >> 1);
                float s = GroupSum(red, t, MassPart(logits, lb, V, max, T, floorT, FromOrderKey(mid), false, t));
                if (s >= target) lo = mid; else hi = mid - 1u;
            }
            float tau = FromOrderKey(lo);
            float gt = GroupSum(red, t, MassPart(logits, lb, V, max, T, floorT, tau, true, t));
            int span = (V + DrawGroup - 1) / DrawGroup, i0 = t * span, i1 = Math.Min(V, i0 + span);
            float own = 0f; int lastTie = -1;
            for (int i = i0; i < i1; i++)
            {
                float l = logits[lb + i];
                if (l == tau) { own += MathF.Exp((l - max) / T); lastTie = i; }
            }
            float before = GroupExclusive(red, t, own);
            float tiesTotal = red[DrawGroup];
            float run = gt + before, cutKeep = 0f; int cut = int.MaxValue;
            for (int i = i0; i < i1; i++)
            {
                float l = logits[lb + i];
                if (l == tau)
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
            // the draw: the first token (best first, ties by id) whose running mass passes x = u * keep
            float x = u * keep;
            lo = OrderKey(floorT); hi = OrderKey(max);
            for (int it = 0; it < 33; it++)
            {
                uint mid = lo + ((hi - lo + 1u) >> 1);
                float s = GroupSum(red, t, MassPart(logits, lb, V, max, T, floorT, FromOrderKey(mid), false, t));
                if (s > x) lo = mid; else hi = mid - 1u;
            }
            float sigma = FromOrderKey(lo);
            float gt2 = GroupSum(red, t, MassPart(logits, lb, V, max, T, floorT, sigma, true, t));
            float own2 = 0f; int lastTie2 = -1;
            for (int i = i0; i < i1; i++)
            {
                float l = logits[lb + i];
                if (l == sigma) { own2 += MathF.Exp((l - max) / T); lastTie2 = i; }
            }
            float before2 = GroupExclusive(red, t, own2);
            float run2 = gt2 + before2; int pick = int.MaxValue;
            for (int i = i0; i < i1; i++)
            {
                float l = logits[lb + i];
                if (l == sigma)
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
    private static void PenalizeImpl(Index1D i, FV logits, IV seqs, IV drawI, FV drawF, IV rep, int V)
    {
        float pen = drawF[1];
        if (pen == 1f) return;
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
        logits[at] = v > 0f ? v / pen : v * pen;
    }

    // this thread's share of the row's softmax mass at T over the tokens within e^20 of the best (logit >= floorT) and at or above v (strict: above v)
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static float MassPart(FV logits, long lb, int V, float max, float T, float floorT, float v, bool strict, int t)
    {
        float s = 0f;
        for (int i = t; i < V; i += DrawGroup)
        {
            float l = logits[lb + i];
            bool inc = (l >= floorT) & (strict ? l > v : l >= v);
            if (inc) s += MathF.Exp((l - max) / T);
        }
        return s;
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

    // split-K attention, part 1. Group g = (row r, head h, chunk ch of AttChunk keys), a thread per head dimension (D = group size, a power of two). The chunk's
    // scores go to shared memory, then its max m and exp-sum l (masked keys contribute 0), then thread d sums p_j * v[j, d] over the chunk (coalesced across d).
    // part[g, d] = that sum, ml[g] = (m, l).
    private static void AttnPartialImpl(FV q, FV kc, FV vc, IV seqs, IV posBuf, FV part, FV ml, int H, int kvH, int maxSeq, int mode, int nch, float scale)
    {
        int p = posBuf[0];
        int g = Grid.IdxX, t = Group.IdxX, D = Group.DimX;
        int ch = g % nch, rest = g / nch, h = rest % H, r = rest / H;
        int seq = mode == 0 ? 0 : seqs[r], pos = mode == 0 ? p + r : p;
        int kvC = kvH * D, hk = h / (H / kvH), j0 = ch * AttChunk;
        var sc = SharedMemory.Allocate<float>(AttChunk);
        var red = SharedMemory.Allocate<float>(AttChunk);
        long qo = ((long)r * H + h) * D;
        for (int jj = t; jj < AttChunk; jj += D)
        {
            int j = j0 + jj;
            float s = -1e30f;
            if (j <= pos)
            {
                long ko = ((long)seq * maxSeq + j) * kvC + hk * D;
                float acc = 0f;
                for (int d = 0; d < D; d++) acc += q[qo + d] * kc[ko + d];
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
        long vo = (long)seq * maxSeq * kvC + hk * D + t;
        for (int jj = 0; jj < keys; jj++) o += sc[jj] * vc[vo + (long)(j0 + jj) * kvC];
        part[(long)g * D + t] = o;
        if (t == 0) { ml[2L * g] = m; ml[2L * g + 1] = l; }
    }
    // split-K attention, part 2: out[r, h * D + d] = sum_ch e^(m_ch - M) part[ch, d] / sum_ch e^(m_ch - M) l_ch (chunks without a key skipped)
    private static void AttnCombineImpl(Index1D i, FV part, FV ml, FV o, int D, int nch)
    {
        int ii = i, d = ii % D, rh = ii / D;
        long g0 = (long)rh * nch;
        float M = -1e30f;
        for (int c = 0; c < nch; c++) if (ml[2 * (g0 + c) + 1] > 0f) M = MathF.Max(M, ml[2 * (g0 + c)]);
        float num = 0f, den = 0f;
        for (int c = 0; c < nch; c++)
        {
            float lc = ml[2 * (g0 + c) + 1];
            if (lc > 0f) { float w = MathF.Exp(ml[2 * (g0 + c)] - M); num += w * part[(g0 + c) * D + d]; den += w * lc; }
        }
        o[ii] = num / den;
    }

    // region 0's first n values into regions 1, 2, ... (sequence s + 1 gets element i of index s * n + i)
    private static void ForkImpl(Index1D i, FV cache, long n, long region)
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
        buf[b + i0] = x0 * c - x1 * s;
        buf[b + i1] = x1 * c + x0 * s;
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

    // the row's k / v into its sequence's cache at its position
    private static void StoreImpl(Index1D i, FV k, FV v, FV kc, FV vc, IV seqs, IV posBuf, int kvC, int maxSeq, int mode)
    {
        int p = posBuf[0];
        int ii = i, r = ii / kvC, c = ii % kvC;
        int seq = mode == 0 ? 0 : seqs[r], pos = mode == 0 ? p + r : p;
        long at = ((long)seq * maxSeq + pos) * kvC + c;
        kc[at] = k[ii]; vc[at] = v[ii];
    }

    // scores[h, r, j] = q[r, h] . k[seq(r), j, h / rep] * scale for j <= pos(r), else masked
    private static void ScoresImpl(Index1D i, FV q, FV kc, IV seqs, FV scores, int R, int nk, int H, int D, int kvH, int maxSeq, int p, int mode, float scale)
    {
        int ii = i, j = ii % nk, rest = ii / nk, r = rest % R, h = rest / R;
        int seq = mode == 0 ? 0 : seqs[r], pos = mode == 0 ? p + r : p;
        float s = -1e30f;
        if (j <= pos)
        {
            int kvC = kvH * D, hk = h / (H / kvH);
            long qo = (long)r * H * D + h * D, ko = ((long)seq * maxSeq + j) * kvC + hk * D;
            float acc = 0f;
            for (int d = 0; d < D; d++) acc += q[qo + d] * kc[ko + d];
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
    private static void PvImpl(Index1D i, FV scores, FV vc, IV seqs, FV o, int R, int nk, int H, int D, int kvH, int maxSeq, int mode)
    {
        int ii = i, C = H * D, r = ii / C, c = ii % C, h = c / D, d = c % D;
        int seq = mode == 0 ? 0 : seqs[r], kvC = kvH * D, hk = h / (H / kvH);
        long so = ((long)h * R + r) * nk, vo = (long)seq * maxSeq * kvC + hk * D + d;
        float acc = 0f;
        for (int j = 0; j < nk; j++) acc += scores[so + j] * vc[vo + (long)j * kvC];
        o[ii] = acc;
    }

    public void Dispose()
    {
        foreach (var o in _owned) o.Dispose();
        _owned.Clear();
        foreach (var sp in _plans.Values) sp.Plan?.Dispose();
        _plans.Clear();
        _mm.Dispose();
        _norm.Dispose();
        _stream.Dispose();
    }
}
