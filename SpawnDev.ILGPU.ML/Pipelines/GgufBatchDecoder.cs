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
/// Weights stay as the GGUF stores them: quantised matrices as raw blocks on the device, norms and biases as floats; the token embedding rows are dequantised on the host from
/// the stream as they are needed (B rows a step), and a tied LM head reuses the embedding blocks.
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
    /// <summary>Sequences live since the last <see cref="Fork"/>.</summary>
    public int Batch { get; private set; } = 1;

    private readonly float _eps, _theta;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _invFreq;
    private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _kc, _vc;
    private MemoryBuffer1D<float, Stride1D.Dense>? _hid, _hn, _q, _k, _v, _att, _proj, _gate, _up, _scores, _logits;
    private int _rows;

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
        var mats = new List<string> { _head };
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

    private FV Buf(ref MemoryBuffer1D<float, Stride1D.Dense>? b, long n)
    {
        if (b == null || b.Length < n) { b = _acc.Allocate1D<float>(Math.Max(n, 64)); _owned.Add(b); }     // a replaced buffer stays owned (a queued dispatch may read it) and is freed with the decoder
        return b.View.SubView(0, n);
    }

    // decode steps (a few rows) read each weight once for all rows (ForwardRows); a prefill is a real GEMM
    private void Gemm(FV x, string name, FV o, int rows, int K, int N)
    {
        var (w, t) = _mats[name];
        if (rows > 1 && rows <= FusedDequantMatMul.RowsMax) _mm.ForwardRows(x, w.View, o, rows, K, N, t);
        else _mm.Forward(x, w.View, o, rows, K, N, t);
    }

    /// <summary>
    /// Run <paramref name="ids"/> through the model: <paramref name="decode"/> = false is a prefill of sequence 0 at positions Position.. (causal over its cache), true is one
    /// token per live sequence at Position. Returns the final hidden rows' logits (prefill: only the last row's).
    /// </summary>
    private async Task<float[]> ForwardAsync(int[] ids, bool decode, CancellationToken ct)
    {
        int R = ids.Length, C = Hidden, kvC = KvHeads * HeadDim, p = Position;
        if (p + (decode ? 1 : R) > MaxSeq) throw new InvalidOperationException($"past the cache length ({MaxSeq})");
        LoadKernels();
        // the embedding rows, dequantised on the host
        var emb = new float[(long)R * C];
        for (int r = 0; r < R; r++) Array.Copy((await _model.GetTensorRowFloat32Async(_embd, ids[r]).ConfigureAwait(false))!, 0, emb, (long)r * C, C);
        var hid = Buf(ref _hid, (long)R * C); hid.CopyFromCPU(emb);
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
        await _acc.SynchronizeAsync().ConfigureAwait(false);
        Position += decode ? 1 : R;
        return await logits.CopyToHostAsync().ConfigureAwait(false);
    }

    /// <summary>Prefill the prompt into sequence 0 from position 0 (one live sequence afterwards); returns the last prompt token's logits [Vocab].</summary>
    public Task<float[]> PrefillAsync(int[] promptIds, CancellationToken ct = default)
    {
        if (promptIds.Length == 0) throw new ArgumentException("an empty prompt");
        Position = 0; Batch = 1;
        return ForwardAsync(promptIds, decode: false, ct);
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

    /// <summary>One token per live sequence (<paramref name="tokens"/>.Length == <see cref="Batch"/>) at <see cref="Position"/>; returns the logits [Batch, Vocab] row-major.</summary>
    public Task<float[]> StepAsync(int[] tokens, CancellationToken ct = default)
    {
        if (tokens.Length != Batch) throw new ArgumentException($"{Batch} tokens expected (one per live sequence)");
        return ForwardAsync(tokens, decode: true, ct);
    }

    /// <summary>
    /// <paramref name="n"/> continuations of <paramref name="promptIds"/> decoded together: prefill once, fork, then one batched step per token. Each sequence samples from
    /// its own logits with its own generator (seed + index): greedy when <paramref name="temperature"/> &lt;= 0, else temperature then nucleus (top-p). A sequence ends at a
    /// stop token or after <paramref name="maxNew"/> tokens; finished sequences ride along until all are done. Returns the generated token ids per sequence (stop token
    /// excluded). <paramref name="onToken"/> gets (sequence, token) as they are sampled.
    /// </summary>
    public async Task<int[][]> GenerateAsync(int[] promptIds, int n, int maxNew, float temperature, float topP, int seed, IReadOnlyCollection<int> stopIds,
        Action<int, int>? onToken = null, CancellationToken ct = default)
    {
        var logits = await PrefillAsync(promptIds, ct).ConfigureAwait(false);
        Fork(n);
        var rngs = Enumerable.Range(0, n).Select(s => new Random(seed + s)).ToArray();
        var outs = Enumerable.Range(0, n).Select(_ => new List<int>()).ToArray();
        var done = new bool[n];
        var next = new int[n];
        for (int s = 0; s < n; s++) next[s] = Sample(logits, 0, temperature, topP, rngs[s]);
        for (int step = 0; ; step++)
        {
            for (int s = 0; s < n; s++)
            {
                if (done[s]) continue;
                if (stopIds.Contains(next[s]) || outs[s].Count >= maxNew) { done[s] = true; continue; }
                outs[s].Add(next[s]); onToken?.Invoke(s, next[s]);
            }
            if (done.All(d => d) || Position + 1 > MaxSeq) break;
            ct.ThrowIfCancellationRequested();
            logits = await StepAsync(next, ct).ConfigureAwait(false);
            for (int s = 0; s < n; s++) if (!done[s]) next[s] = Sample(logits, s, temperature, topP, rngs[s]);
        }
        return outs.Select(o => o.ToArray()).ToArray();
    }

    /// <summary>
    /// One token from row <paramref name="row"/> of [rows, Vocab] logits: argmax when <paramref name="temperature"/> &lt;= 0; else softmax(logits / temperature) over the
    /// tokens within e^20 of the best (the rest carry under 2e-9 of the mass each), the smallest set whose mass reaches <paramref name="topP"/>, renormalised and drawn.
    /// </summary>
    public int Sample(float[] logits, int row, float temperature, float topP, Random rng)
    {
        int V = Vocab, o = row * V, best = 0;
        for (int i = 1; i < V; i++) if (logits[o + i] > logits[o + best]) best = i;
        if (temperature <= 0f) return best;
        float max = logits[o + best], floor = max - 20f * temperature;
        var cand = new List<(int Id, double P)>();
        for (int i = 0; i < V; i++) if (logits[o + i] >= floor) cand.Add((i, Math.Exp((logits[o + i] - max) / temperature)));
        cand.Sort((a, b) => b.P.CompareTo(a.P));
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
