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
/// the candidates and is <see cref="SampleLogits"/> on the full row (the same nucleus, the same draw); when the nucleus is wider than the candidates it reads that row.
/// </para>
/// </summary>
public sealed class GgufBatchDecoder : IDisposable
{
    private readonly Accelerator _acc;
    private readonly GGUFModel _model;
    private readonly Stream _stream;
    private readonly FusedDequantMatMul _mm;
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
    public int FfnDim { get; }
    public int Vocab { get; }
    /// <summary>Longest sequence (prompt + generation) and most sequences the caches hold.</summary>
    public int MaxSeq { get; }
    public int MaxBatch { get; }
    /// <summary>The position the next token goes to (the prompt length after <see cref="PrefillAsync"/>, plus one per <see cref="StepAsync"/>).</summary>
    public int Position { get; private set; }
    /// <summary>Where the time of all decode steps since <see cref="PrefillAsync"/> went, in ms: the token ids up and the gather queued, the forward on the device (to its sync), the candidates read back.</summary>
    public (double Embed, double Device, double Readback, int Steps) DecodeTimings { get; private set; }
    /// <summary>Tokens per row the device keeps after each forward (the nucleus is drawn from these unless it is wider).</summary>
    public const int Candidates = 64;
    /// <summary>Draws since <see cref="PrefillAsync"/> that read their full logits row: the nucleus was wider than the candidates, or more than <see cref="SelectCap"/> tokens reached the floor.</summary>
    public int RowReads { get; private set; }
    /// <summary>Most tokens per row the candidate gather keeps (a flatter row is read whole).</summary>
    public const int SelectCap = 4096;

    /// <summary>Sequences live since the last <see cref="Fork"/>.</summary>
    public int Batch { get; private set; } = 1;

    private readonly float _eps, _theta;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _invFreq;
    private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _kc, _vc;
    private MemoryBuffer1D<float, Stride1D.Dense>? _hid, _hn, _q, _k, _v, _att, _proj, _gate, _up, _scores, _logits, _maxV, _selV, _cand;
    private MemoryBuffer1D<int, Stride1D.Dense>? _maxI, _selI, _selN;
    private FV _lastLogits;
    private float[] _lastCand = Array.Empty<float>();
    private int _lastRows;
    private float _lastT;
    private const int Chunk = 256;
    private int Chunks => (Vocab + Chunk - 1) / Chunk;
    private int CandStride => 2 * Candidates + 1 + Chunks;      // per row: values, ids (as floats), tokens at the floor, mass partial sums

    private GgufBatchDecoder(Accelerator accelerator, GGUFModel model, Stream stream, int maxSeq, int maxBatch)
    {
        _acc = accelerator; _model = model; _stream = stream; _mm = new FusedDequantMatMul(accelerator);
        string a = model.Architecture;
        if (a is not ("qwen2"))
            throw new NotSupportedException($"GgufBatchDecoder: architecture '{a}' (supported: qwen2: NeoX rotary, biased q/k/v)");
        Hidden = (int)model.EmbeddingLength; Layers = (int)model.BlockCount; Heads = (int)model.AttentionHeadCount; KvHeads = (int)model.AttentionHeadCountKV;
        HeadDim = Hidden / Heads; FfnDim = (int)model.GetMetadataInt($"{a}.feed_forward_length", 0);
        _eps = model.GetMetadataFloat($"{a}.attention.layer_norm_rms_epsilon", 1e-6f);
        _theta = model.GetMetadataFloat($"{a}.rope.freq_base", 10000f);
        _embd = Tensor("token_embd.weight");
        Vocab = (int)_embd.Dimensions[1];
        if (Vocab >= 1 << 24) throw new NotSupportedException($"vocabulary {Vocab}: candidate ids travel as floats (exact below 2^24)");
        _head = model.Tensors.Any(t => t.Name == "output.weight") ? "output.weight" : "token_embd.weight";
        MaxSeq = maxSeq; MaxBatch = maxBatch;
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
        for (int l = 0; l < Layers; l++)
        {
            string p = $"blk.{l}.";
            mats.AddRange(new[] { p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight", p + "attn_output.weight", p + "ffn_gate.weight", p + "ffn_up.weight", p + "ffn_down.weight" });
            vecs.AddRange(new[] { p + "attn_norm.weight", p + "ffn_norm.weight", p + "attn_q.bias", p + "attn_k.bias", p + "attn_v.bias" });
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

    private readonly List<MemoryBuffer1D<int, Stride1D.Dense>> _idBufs = new();
    private int _idCall;
    // token ids: a fresh buffer per call until reused after a sync (a queued gather may still read the last one); ForwardAsync syncs every call, so two alternate
    private IV IntBuf(int n)
    {
        int slot = _idCall++ & 1;
        while (_idBufs.Count <= slot) _idBufs.Add(Own(_acc.Allocate1D<int>(Math.Max(64, MaxSeq))));
        return _idBufs[slot].View.SubView(0, n);
    }

    private FV Buf(ref MemoryBuffer1D<float, Stride1D.Dense>? b, long n)
    {
        if (b == null || b.Length < n) { b = _acc.Allocate1D<float>(Math.Max(n, 64)); _owned.Add(b); }     // a replaced buffer stays owned (a queued dispatch may read it) and is freed with the decoder
        return b.View.SubView(0, n);
    }

    // decode steps (a few rows) read each weight once for all rows (ForwardRows); a prefill is a real GEMM
    private void Gemm(FV x, string name, FV o, int rows, int K, int N)
    {
        var (w, t) = _mats[name];
        if (rows <= FusedDequantMatMul.RowsMax) _mm.ForwardRows(x, w.View, o, rows, K, N, t);
        else _mm.Forward(x, w.View, o, rows, K, N, t);
    }

    /// <summary>
    /// Run <paramref name="ids"/> through the model: <paramref name="decode"/> = false is a prefill of sequence 0 at positions Position.. (causal over its cache), true is one
    /// token per live sequence at Position. Leaves the final hidden rows' logits (prefill: only the last row's) on the device and their candidates on the host
    /// (with the softmax mass at <paramref name="temperature"/> when it is above 0).
    /// </summary>
    private async Task ForwardAsync(int[] ids, bool decode, float temperature, CancellationToken ct)
    {
        int R = ids.Length, C = Hidden, kvC = KvHeads * HeadDim, p = Position;
        if (p + (decode ? 1 : R) > MaxSeq) throw new InvalidOperationException($"past the cache length ({MaxSeq})");
        LoadKernels();
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        // the embedding rows, gathered and dequantised on the device from the token embedding's blocks
        var idBuf = IntBuf(R); idBuf.CopyFromCPU(ids);
        var hid = Buf(ref _hid, (long)R * C);
        var (embW, embT) = _mats["token_embd.weight"];
        int code = FusedDequantMatMul.TypeCode(embT);
        _embed!((int)((long)R * C), idBuf, embW.View.Cast<byte, int>(), hid, C, code, FusedDequantMatMul.RowBytesOf(code, C));
        long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
        var hn = Buf(ref _hn, (long)R * C); var q = Buf(ref _q, (long)R * C); var k = Buf(ref _k, (long)R * kvC); var v = Buf(ref _v, (long)R * kvC);
        var att = Buf(ref _att, (long)R * C); var proj = Buf(ref _proj, (long)R * C);
        var gate = Buf(ref _gate, (long)R * FfnDim); var up = Buf(ref _up, (long)R * FfnDim);
        int nk = p + (decode ? 1 : R);
        var scores = Buf(ref _scores, (long)Heads * R * nk);
        int mode = decode ? 1 : 0;
        for (int l = 0; l < Layers; l++)
        {
            string b = $"blk.{l}.";
            _rms!(R, hid, hn, _vecs[b + "attn_norm.weight"].View, C, _eps);
            Gemm(hn, b + "attn_q.weight", q, R, C, C); _bias!((int)((long)R * C), q, _vecs[b + "attn_q.bias"].View, C);
            Gemm(hn, b + "attn_k.weight", k, R, C, kvC); _bias!((int)((long)R * kvC), k, _vecs[b + "attn_k.bias"].View, kvC);
            Gemm(hn, b + "attn_v.weight", v, R, C, kvC); _bias!((int)((long)R * kvC), v, _vecs[b + "attn_v.bias"].View, kvC);
            _rope!(R * Heads * (HeadDim / 2), q, _invFreq.View, Heads, HeadDim, p, mode);
            _rope!(R * KvHeads * (HeadDim / 2), k, _invFreq.View, KvHeads, HeadDim, p, mode);
            _store!((int)((long)R * kvC), k, v, _kc[l].View, _vc[l].View, kvC, MaxSeq, p, mode);
            _scoresK!((int)((long)Heads * R * nk), q, _kc[l].View, scores, R, nk, Heads, HeadDim, KvHeads, MaxSeq, p, mode, 1f / MathF.Sqrt(HeadDim));
            _softmax!(Heads * R, scores, nk);
            _pv!((int)((long)R * C), scores, _vc[l].View, att, R, nk, Heads, HeadDim, KvHeads, MaxSeq, mode);
            Gemm(att, b + "attn_output.weight", proj, R, C, C);
            _add!((int)((long)R * C), hid, proj);
            _rms(R, hid, hn, _vecs[b + "ffn_norm.weight"].View, C, _eps);
            Gemm(hn, b + "ffn_gate.weight", gate, R, C, FfnDim);
            Gemm(hn, b + "ffn_up.weight", up, R, C, FfnDim);
            _swiglu!((int)((long)R * FfnDim), gate, up);
            Gemm(gate, b + "ffn_down.weight", proj, R, FfnDim, C);
            _add((int)((long)R * C), hid, proj);
        }
        // final norm and the LM head: every row when decoding, the last row of a prefill
        int outRows = decode ? R : 1;
        var last = decode ? hid : hid.SubView((long)(R - 1) * C, C);
        var hnOut = hn.SubView(0, (long)outRows * C);
        _rms(outRows, last, hnOut, _vecs["output_norm.weight"].View, C, _eps);
        var logits = Buf(ref _logits, (long)outRows * Vocab);
        Gemm(hnOut, _head, logits, outRows, C, Vocab);
        // each row's best tokens and (sampling) its softmax mass, on the device
        int nch = Chunks, S = CandStride;
        var maxV = Buf(ref _maxV, (long)outRows * nch);
        var selV = Buf(ref _selV, (long)outRows * SelectCap);
        _maxI ??= Own(_acc.Allocate1D<int>((long)MaxBatch * nch));
        _selI ??= Own(_acc.Allocate1D<int>((long)MaxBatch * SelectCap));
        _selN ??= Own(_acc.Allocate1D<int>(Math.Max(64, MaxBatch)));
        var maxI = _maxI.View.SubView(0, (long)outRows * nch);
        var cand = Buf(ref _cand, (long)outRows * S);
        _chunkMax!(outRows * nch, logits, maxV, maxI, Vocab, nch);
        _floor!(outRows, maxV, maxI, cand, _selN.View, nch, S);
        _select!(outRows * nch, logits, cand, _selN.View, selV, _selI.View, Vocab, nch, S, temperature);
        _final!(outRows, selV, _selI.View, _selN.View, cand, S);
        await _acc.SynchronizeAsync().ConfigureAwait(false);
        long t2 = System.Diagnostics.Stopwatch.GetTimestamp();
        Position += decode ? 1 : R;
        _lastCand = await cand.CopyToHostAsync().ConfigureAwait(false);
        _lastLogits = logits; _lastRows = outRows; _lastT = temperature;
        long t3 = System.Diagnostics.Stopwatch.GetTimestamp();
        if (decode)
        {
            double ms(long a, long b) => (b - a) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            var d = DecodeTimings;
            DecodeTimings = (d.Embed + ms(t0, t1), d.Device + ms(t1, t2), d.Readback + ms(t2, t3), d.Steps + 1);
        }
    }

    /// <summary>
    /// Prefill the prompt into sequence 0 from position 0 (one live sequence afterwards). <paramref name="temperature"/>: what the next draw (<see cref="SampleAsync"/>) uses,
    /// 0 = greedy (the device then skips the softmax mass).
    /// </summary>
    public Task PrefillAsync(int[] promptIds, float temperature = 0f, CancellationToken ct = default)
    {
        if (promptIds.Length == 0) throw new ArgumentException("an empty prompt");
        Position = 0; Batch = 1; DecodeTimings = default; RowReads = 0;
        return ForwardAsync(promptIds, decode: false, temperature, ct);
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
        return ForwardAsync(tokens, decode: true, temperature, ct);
    }

    /// <summary>The last forward's logits [rows, Vocab] row-major (a prefill: one row), read from the device.</summary>
    public async Task<float[]> ReadLogitsAsync() => await _lastLogits.CopyToHostAsync().ConfigureAwait(false);

    /// <summary>
    /// One token for row <paramref name="row"/> of the last forward, at the temperature that forward was given: argmax when 0, else <see cref="SampleLogits"/>'s draw,
    /// taken from the device's candidates (the same set and the same draw; the row is read only when its nucleus is wider than <see cref="Candidates"/>).
    /// </summary>
    public async ValueTask<int> SampleAsync(int row, float topP, Random rng)
    {
        if ((uint)row >= (uint)_lastRows) throw new ArgumentOutOfRangeException(nameof(row));
        int K = Candidates, o = row * CandStride;
        var c = _lastCand;
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

    /// <summary>
    /// <paramref name="n"/> continuations of <paramref name="promptIds"/> decoded together: prefill once, fork, then one batched step per token. Each sequence samples from
    /// its own logits with its own generator (seed + index): greedy when <paramref name="temperature"/> &lt;= 0, else temperature then nucleus (top-p). A sequence ends at a
    /// stop token or after <paramref name="maxNew"/> tokens; finished sequences ride along until all are done. Returns the generated token ids per sequence (stop token
    /// excluded). <paramref name="onToken"/> gets (sequence, token) as they are sampled, <paramref name="onFinished"/> the sequence when it ends.
    /// </summary>
    public async Task<int[][]> GenerateAsync(int[] promptIds, int n, int maxNew, float temperature, float topP, int seed, IReadOnlyCollection<int> stopIds,
        Action<int, int>? onToken = null, Action<int>? onFinished = null, CancellationToken ct = default)
    {
        await PrefillAsync(promptIds, temperature, ct).ConfigureAwait(false);
        Fork(n);
        var rngs = Enumerable.Range(0, n).Select(s => new Random(seed + s)).ToArray();
        var outs = Enumerable.Range(0, n).Select(_ => new List<int>()).ToArray();
        var done = new bool[n];
        var next = new int[n];
        for (int s = 0; s < n; s++) next[s] = await SampleAsync(0, topP, rngs[s]).ConfigureAwait(false);
        for (int step = 0; ; step++)
        {
            for (int s = 0; s < n; s++)
            {
                if (done[s]) continue;
                if (stopIds.Contains(next[s]) || outs[s].Count >= maxNew) { done[s] = true; onFinished?.Invoke(s); continue; }
                outs[s].Add(next[s]); onToken?.Invoke(s, next[s]);
            }
            if (done.All(d => d)) break;
            if (Position + 1 > MaxSeq)
            {
                for (int s = 0; s < n; s++) if (!done[s]) { done[s] = true; onFinished?.Invoke(s); }
                break;
            }
            ct.ThrowIfCancellationRequested();
            await StepAsync(next, temperature, ct).ConfigureAwait(false);
            for (int s = 0; s < n; s++) if (!done[s]) next[s] = await SampleAsync(s, topP, rngs[s]).ConfigureAwait(false);
        }
        return outs.Select(o => o.ToArray()).ToArray();
    }

    /// <summary>
    /// One token from row <paramref name="row"/> of [rows, Vocab] logits: argmax when <paramref name="temperature"/> &lt;= 0; else softmax(logits / temperature) over the
    /// tokens within e^20 of the best (the rest carry under 2e-9 of the mass each), taken best first (ties: the lower id), the smallest set whose mass reaches
    /// <paramref name="topP"/>, renormalised and drawn. <paramref name="vocab"/>: the row length.
    /// </summary>
    public static int SampleLogits(float[] logits, int row, int vocab, float temperature, float topP, Random rng)
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
        double u = rng.NextDouble() * keep, acc = 0;
        for (int i = 0; i < m; i++) { acc += cand[i].P; if (u < acc) return cand[i].Id; }
        return cand[m - 1].Id;
    }

    // ---------------------------------------------------------------------------------------------------
    // kernels. Row r: prefill (mode 0) = sequence 0 at position p + r; decode (mode 1) = sequence r at position p.
    // Caches: [sequence][MaxSeq][kvHeads * headDim].
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, FV, FV, int, float>? _rms;
    private Action<Index1D, FV, FV, int>? _bias;
    private Action<Index1D, FV, FV, int, int, int, int>? _rope;
    private Action<Index1D, FV, FV, FV, FV, int, int, int, int>? _store;
    private Action<Index1D, FV, FV, FV, int, int, int, int, int, int, int, int, float>? _scoresK;
    private Action<Index1D, FV, int>? _softmax;
    private Action<Index1D, FV, FV, FV, int, int, int, int, int, int, int>? _pv;
    private Action<Index1D, FV, FV>? _add;
    private Action<Index1D, FV, FV>? _swiglu;
    private Action<Index1D, FV, long, long>? _fork;
    private Action<Index1D, IV, ArrayView1D<int, Stride1D.Dense>, FV, int, int, int>? _embed;
    private Action<Index1D, FV, FV, IV, int, int>? _chunkMax;
    private Action<Index1D, FV, IV, FV, IV, int, int>? _floor;
    private Action<Index1D, FV, FV, IV, FV, IV, int, int, int, float>? _select;
    private Action<Index1D, FV, IV, IV, FV, int>? _final;

    private void LoadKernels()
    {
        if (_rms != null) return;
        _rms = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, float>(RmsImpl);
        _bias = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int>(BiasImpl);
        _rope = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int, int>(RopeImpl);
        _store = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, int, int, int, int>(StoreImpl);
        _scoresK = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, int, int, int, int, int, float>(ScoresImpl);
        _softmax = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, int>(SoftmaxImpl);
        _pv = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, int, int, int, int>(PvImpl);
        _add = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(AddImpl);
        _swiglu = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(SwigluImpl);
        _fork = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, long, long>(ForkImpl);
        _embed = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, ArrayView1D<int, Stride1D.Dense>, FV, int, int, int>(EmbedImpl);
        _chunkMax = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, int, int>(ChunkMaxImpl);
        _floor = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, FV, IV, int, int>(FloorImpl);
        _select = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, FV, IV, int, int, int, float>(SelectImpl);
        _final = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, FV, int>(FinalImpl);
    }

    // o[r] = x[r] / rms(x[r]) * w (Qwen2RMSNorm: mean of squares + eps)
    private static void RmsImpl(Index1D r, FV x, FV o, FV w, int C, float eps)
    {
        long b = (long)(int)r * C;
        float s = 0f;
        for (int c = 0; c < C; c++) { float v = x[b + c]; s += v * v; }
        float inv = 1f / MathF.Sqrt(s / C + eps);
        for (int c = 0; c < C; c++) o[b + c] = x[b + c] * inv * w[c];
    }
    private static void BiasImpl(Index1D i, FV x, FV bias, int C) => x[i] += bias[(int)i % C];
    private static void AddImpl(Index1D i, FV h, FV d) => h[i] += d[i];
    // gate[i] = silu(gate[i]) * up[i]
    private static void SwigluImpl(Index1D i, FV g, FV u) { float x = g[i]; g[i] = x / (1f + MathF.Exp(-x)) * u[i]; }

    // o[r, c] = token_embd[ids[r], c], decoded from its blocks
    private static void EmbedImpl(Index1D i, IV ids, ArrayView1D<int, Stride1D.Dense> w, FV o, int C, int code, int rowBytes)
    {
        int ii = i, r = ii / C, c = ii % C;
        o[ii] = FusedDequantMatMul.DecodeOf(code, w, ids[r] * rowBytes, c);
    }

    // candidate selection. Per row of the output o (stride S): [0, K) values best first, [K, 2K) their ids as floats, 2K the tokens at or above the floor,
    // 2K + 1 + chunk the chunk's softmax mass. A token is better than another when its logit is larger, or equal with the lower id.

    // each Chunk's largest logit and its first id
    private static void ChunkMaxImpl(Index1D i, FV logits, FV maxV, IV maxI, int V, int nch)
    {
        int ii = i, r = ii / nch, ch = ii % nch, end = Math.Min(V, (ch + 1) * Chunk), bi = ch * Chunk;
        float best = logits[(long)r * V + bi];
        for (int idx = bi + 1; idx < end; idx++) { float v = logits[(long)r * V + idx]; if (v > best) { best = v; bi = idx; } }
        maxV[ii] = best; maxI[ii] = bi;
    }
    // the row's K best chunk maxima into o's value / id slots (o[0] is then the row's best logit); the K-th of them is the floor: those K tokens all reach it.
    // The gather count is zeroed here, before the gather runs.
    private static void FloorImpl(Index1D row, FV maxV, IV maxI, FV o, IV selN, int nch, int S)
    {
        const int K = Candidates;
        int r = row;
        long ob = (long)r * S;
        for (int j = 0; j < K; j++) { o[ob + j] = float.MinValue; o[ob + K + j] = -1f; }
        for (int ch = 0; ch < nch; ch++)
        {
            float v = maxV[(long)r * nch + ch];
            if (v <= o[ob + K - 1]) continue;
            int pos = K - 1;
            while (pos > 0 && o[ob + pos - 1] < v) { o[ob + pos] = o[ob + pos - 1]; o[ob + K + pos] = o[ob + K + pos - 1]; pos--; }
            o[ob + pos] = v; o[ob + K + pos] = maxI[(long)r * nch + ch];
        }
        selN[r] = 0;
    }
    // each Chunk's tokens at or above the floor appended to the row's list (in any order), and (T > 0) the chunk's softmax mass at T over the tokens within e^20 of
    // the best (the set SampleLogits draws from)
    private static void SelectImpl(Index1D i, FV logits, FV o, IV selN, FV selV, IV selI, int V, int nch, int S, float T)
    {
        const int K = Candidates;
        int ii = i, r = ii / nch, ch = ii % nch, end = Math.Min(V, (ch + 1) * Chunk);
        long ob = (long)r * S;
        float floorK = o[ob + K - 1], max = o[ob], floorT = max - 20f * T, sum = 0f;
        for (int idx = ch * Chunk; idx < end; idx++)
        {
            float v = logits[(long)r * V + idx];
            if (v >= floorK)
            {
                int slot = Atomic.Add(ref selN[r], 1);
                if (slot < SelectCap) { selV[(long)r * SelectCap + slot] = v; selI[(long)r * SelectCap + slot] = idx; }
            }
            if (T > 0f && v >= floorT) sum += MathF.Exp((v - max) / T);
        }
        o[ob + 2 * K + 1 + ch] = sum;
    }
    // the row's gathered tokens sorted into its K best (skipped when more than SelectCap reached the floor: the host reads that row)
    private static void FinalImpl(Index1D row, FV selV, IV selI, IV selN, FV o, int S)
    {
        const int K = Candidates;
        int r = row, n = selN[r];
        long ob = (long)r * S, sb = (long)r * SelectCap;
        o[ob + 2 * K] = n;
        if (n > SelectCap) return;
        for (int j = 0; j < K; j++) { o[ob + j] = float.MinValue; o[ob + K + j] = -1f; }
        for (int s = 0; s < n; s++)
        {
            float v = selV[sb + s]; int id = selI[sb + s];
            float lv = o[ob + K - 1];
            if (v < lv || (v == lv && o[ob + 2 * K - 1] >= 0f && id > (int)o[ob + 2 * K - 1])) continue;
            int pos = K - 1;
            while (pos > 0)
            {
                float pv = o[ob + pos - 1];
                if (pv > v || (pv == v && (int)o[ob + K + pos - 1] < id)) break;
                o[ob + pos] = pv; o[ob + K + pos] = o[ob + K + pos - 1]; pos--;
            }
            o[ob + pos] = v; o[ob + K + pos] = id;
        }
    }

    // region 0's first n values into regions 1, 2, ... (sequence s + 1 gets element i of index s * n + i)
    private static void ForkImpl(Index1D i, FV cache, long n, long region)
    {
        long ii = i, s = ii / n + 1, e = ii % n;
        cache[s * region + e] = cache[e];
    }

    // NeoX / rotate_half rotary in place: element k pairs with k + D/2, at the row's position
    private static void RopeImpl(Index1D i, FV buf, FV invFreq, int H, int D, int p, int mode)
    {
        int half = D / 2, ii = i;
        int k = ii % half, rest = ii / half, h = rest % H, r = rest / H;
        int pos = mode == 0 ? p + r : p;
        float ang = pos * invFreq[k], c = MathF.Cos(ang), s = MathF.Sin(ang);
        long b = (long)r * H * D + h * D;
        float x0 = buf[b + k], x1 = buf[b + half + k];
        buf[b + k] = x0 * c - x1 * s;
        buf[b + half + k] = x1 * c + x0 * s;
    }

    // the row's k / v into its sequence's cache at its position
    private static void StoreImpl(Index1D i, FV k, FV v, FV kc, FV vc, int kvC, int maxSeq, int p, int mode)
    {
        int ii = i, r = ii / kvC, c = ii % kvC;
        int seq = mode == 0 ? 0 : r, pos = mode == 0 ? p + r : p;
        long at = ((long)seq * maxSeq + pos) * kvC + c;
        kc[at] = k[ii]; vc[at] = v[ii];
    }

    // scores[h, r, j] = q[r, h] . k[seq(r), j, h / rep] * scale for j <= pos(r), else masked
    private static void ScoresImpl(Index1D i, FV q, FV kc, FV scores, int R, int nk, int H, int D, int kvH, int maxSeq, int p, int mode, float scale)
    {
        int ii = i, j = ii % nk, rest = ii / nk, r = rest % R, h = rest / R;
        int seq = mode == 0 ? 0 : r, pos = mode == 0 ? p + r : p;
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
    private static void SoftmaxImpl(Index1D row, FV scores, int nk)
    {
        long o = (long)(int)row * nk;
        float max = scores[o];
        for (int j = 1; j < nk; j++) max = MathF.Max(max, scores[o + j]);
        float sum = 0f;
        for (int j = 0; j < nk; j++) { float e = MathF.Exp(scores[o + j] - max); scores[o + j] = e; sum += e; }
        float inv = 1f / sum;
        for (int j = 0; j < nk; j++) scores[o + j] *= inv;
    }
    // o[r, h * D + d] = sum_j scores[h, r, j] v[seq(r), j, h / rep, d]
    private static void PvImpl(Index1D i, FV scores, FV vc, FV o, int R, int nk, int H, int D, int kvH, int maxSeq, int mode)
    {
        int ii = i, C = H * D, r = ii / C, c = ii % C, h = c / D, d = c % D;
        int seq = mode == 0 ? 0 : r, kvC = kvH * D, hk = h / (H / kvH);
        long so = ((long)h * R + r) * nk, vo = (long)seq * maxSeq * kvC + hk * D + d;
        float acc = 0f;
        for (int j = 0; j < nk; j++) acc += scores[so + j] * vc[vo + (long)j * kvC];
        o[ii] = acc;
    }

    public void Dispose()
    {
        foreach (var o in _owned) o.Dispose();
        _owned.Clear();
        _mm.Dispose();
        _stream.Dispose();
    }
}
