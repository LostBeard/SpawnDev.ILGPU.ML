using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.Flux2;

/// <summary>
/// A Qwen3 causal language model with a key / value cache, sized from its weights (<see cref="Flux2Weights"/> of a Hugging Face Qwen3 file, the adapter's layout: q | k | v
/// and gate | up fused, <see cref="Flux2Weights.FusedQkvSplit"/>), tied embeddings: <c>logits = rms(h) E^T</c>. <see cref="Prefill"/> consumes a prompt, <see cref="Step"/>
/// one token; both leave the last row's final-norm state in <see cref="LastHidden"/> and its logits over a chosen slice of the vocabulary in <see cref="Logits"/>
/// (<see cref="SetLogitRange"/>: ACE-Step's code LM reads only its 64000 audio-code tokens). Several decoders may share one <see cref="Flux2Weights"/> (CFG: two sequences).
/// The kernels are the SkinTokens decoder's (<see cref="SkinTokensKernels.StoreKv"/>, <see cref="SkinTokensKernels.CausalGqa"/>, GEMV for single tokens) with exact host
/// rotary tables.
/// </summary>
public sealed class Qwen3KvDecoder : IDisposable
{
    public const int HeadDim = 128;
    public const float RmsEps = 1e-6f;

    private readonly Accelerator _acc;
    private readonly Flux2Weights _w;
    private readonly DiTKernels _k;
    private readonly SkinTokensKernels _u;
    private readonly int _layers, _hidden, _qCols, _kvCols, _qkvCols, _qHeads, _kvHeads, _inter, _maxPositions, _maxPrefill;
    private readonly List<IDisposable> _owned = new();
    private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _kc, _vc;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _h, _hn, _qkv, _attn, _proj, _gateUp, _act, _scores, _logits, _last, _dummy, _cos, _sin;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _ids;
    private int _pos, _logitStart, _logitCount;

    public int Position => _pos;
    public int Hidden => _hidden;
    /// <summary>The last row's state after the final norm, [hidden] (valid after <see cref="Prefill"/> / <see cref="Step"/>).</summary>
    public FV LastHidden => _last.View.SubView(0, _hidden);
    /// <summary>The last row's logits over the current range, [<see cref="LogitCount"/>].</summary>
    public FV Logits => _logits.View.SubView(0, _logitCount);
    public int LogitStart => _logitStart;
    public int LogitCount => _logitCount;

    /// <param name="maxPositions">Cache length: prompt plus generated tokens.</param>
    /// <param name="maxPrefill">Most tokens one <see cref="Prefill"/> may carry.</param>
    public Qwen3KvDecoder(Accelerator accelerator, Flux2Weights weights, int layers, int maxPositions, int maxPrefill, float ropeTheta = 1_000_000f)
    {
        _acc = accelerator; _w = weights; _layers = layers; _maxPositions = maxPositions; _maxPrefill = maxPrefill;
        if (!weights.Has("layers.0.self_attn.qkv_proj.weight") || !weights.Has("layers.0.mlp.gate_up_proj.weight"))
            throw new NotSupportedException("expects the Qwen3 adapter's fused q | k | v and gate | up (bf16 Hugging Face Qwen3 files)");
        _hidden = weights.Shape("layers.0.input_layernorm.weight")[0];
        int rows = weights.Shape("layers.0.self_attn.qkv_proj.weight")[0];
        (_qCols, _kvCols) = weights.FusedQkvSplit ?? throw new InvalidDataException("the q | k | v split was not recorded (load the file through Flux2Weights.LoadAsync)");
        if (_qCols + 2 * _kvCols != rows) throw new InvalidDataException($"fused q|k|v has {rows} rows, the split {_qCols} | {_kvCols}");
        _qkvCols = rows; _qHeads = _qCols / HeadDim; _kvHeads = _kvCols / HeadDim;
        _inter = weights.Shape("layers.0.mlp.gate_up_proj.weight")[0] / 2;
        _k = new DiTKernels(accelerator) { PreciseActivations = true }; _u = new SkinTokensKernels(accelerator);
        _kc = new MemoryBuffer1D<float, Stride1D.Dense>[layers]; _vc = new MemoryBuffer1D<float, Stride1D.Dense>[layers];
        for (int l = 0; l < layers; l++) { _kc[l] = Alloc<float>((long)maxPositions * _kvCols); _vc[l] = Alloc<float>((long)maxPositions * _kvCols); }
        long T = maxPrefill;
        _h = Alloc<float>(T * _hidden); _hn = Alloc<float>(T * _hidden); _qkv = Alloc<float>(T * _qkvCols); _attn = Alloc<float>(T * _qCols);
        _proj = Alloc<float>(T * _hidden); _gateUp = Alloc<float>(T * 2 * _inter); _act = Alloc<float>(T * _inter);
        _scores = Alloc<float>(Math.Max((long)_qHeads * T * T, (long)_qHeads * maxPositions));
        int vocab = weights.Shape("embed_tokens.weight")[0];
        _logitStart = 0; _logitCount = vocab;
        _logits = Alloc<float>(vocab); _last = Alloc<float>(_hidden); _dummy = Alloc<float>(1); _ids = Alloc<int>(T);
        // exact rotate_half tables for every cache position, from the host (float32 angles as the reference computes them; SkinTokensKernels.RopeTable rounds the
        // frequencies to bf16 - SkinTokens' own reference - which is 1.7e-2 off a float32 Qwen3 over a 478-token prompt)
        int half = HeadDim / 2;
        var c = new float[(long)maxPositions * half]; var sn = new float[(long)maxPositions * half];
        for (int k = 0; k < half; k++)
        {
            float inv = 1f / MathF.Pow(ropeTheta, 2f * k / HeadDim);
            for (int p = 0; p < maxPositions; p++) { float ang = p * inv; c[(long)p * half + k] = (float)Math.Cos(ang); sn[(long)p * half + k] = (float)Math.Sin(ang); }
        }
        _cos = _acc.Allocate1D(c); _owned.Add(_cos); _sin = _acc.Allocate1D(sn); _owned.Add(_sin);
    }

    private MemoryBuffer1D<T, Stride1D.Dense> Alloc<T>(long n) where T : unmanaged { var b = _acc.Allocate1D<T>(Math.Max(1, n)); _owned.Add(b); return b; }

    /// <summary>Compute logits only for tokens [<paramref name="start"/>, <paramref name="start"/> + <paramref name="count"/>) (rows of the tied embedding table).</summary>
    public void SetLogitRange(int start, int count)
    {
        int vocab = _w.Shape("embed_tokens.weight")[0];
        if (start < 0 || count < 1 || start + count > vocab) throw new ArgumentOutOfRangeException(nameof(count), $"[{start}, {start + count}) is outside the {vocab}-token vocabulary");
        _logitStart = start; _logitCount = count;
    }

    /// <summary>Start a sequence with <paramref name="tokenIds"/> (positions 0 ..); the last row's state and logits follow.</summary>
    public void Prefill(int[] tokenIds)
    {
        if (tokenIds.Length < 1 || tokenIds.Length > _maxPrefill) throw new ArgumentException($"1 to {_maxPrefill} tokens");
        if (tokenIds.Length > _maxPositions) throw new ArgumentException("longer than the cache");
        _pos = 0;
        _ids.View.SubView(0, tokenIds.Length).CopyFromCPU(tokenIds);
        _u.Embed(_ids.View, _w.Vector("embed_tokens.weight"), _h.View, tokenIds.Length, _hidden);
        Forward(tokenIds.Length);
    }

    /// <summary>Append one token.</summary>
    public void Step(int tokenId)
    {
        if (_pos == 0) throw new InvalidOperationException("Prefill first");
        if (_pos + 1 > _maxPositions) throw new InvalidOperationException("the cache is full");
        _ids.View.SubView(0, 1).CopyFromCPU(new[] { tokenId });
        _u.Embed(_ids.View, _w.Vector("embed_tokens.weight"), _h.View, 1, _hidden);
        Forward(1);
    }

    private void Lin(FV x, string weight, FV o, int rows, int K, int N)
    {
        if (rows == 1) _u.Gemv(x, _w.Vector(weight), null, o, K, N);
        else _k.Linear(x, _w.Vector(weight), null, o, rows, K, N);
    }

    private void Forward(int T)
    {
        int C = _hidden, p0 = _pos, rep = _qHeads / _kvHeads;
        for (int l = 0; l < _layers; l++)
        {
            // browser: a command buffer per layer for a PREFILL (DeviceSync.Submit; a no-op on desktop). A whole multi-token forward in ONE command
            // buffer can outrun a slow GPU's 2 s watchdog (2026-11-23, Lore: FLUX.2 on an RX 590 in Chrome). A one-token decode step is small: it stays
            // one command buffer, so generation pays no per-layer submits.
            if (T > 1) _acc.Submit();
            string b = $"layers.{l}.";
            _u.RmsNorm(_h.View, _hn.View, _w.Vector(b + "input_layernorm.weight"), T, C, RmsEps);
            Lin(_hn.View, b + "self_attn.qkv_proj.weight", _qkv.View, T, C, _qkvCols);
            _u.RmsNormHeads(_qkv.View, _w.Vector(b + "self_attn.q_norm.weight"), T, _qkvCols, 0, _qHeads, HeadDim, RmsEps);
            _u.RmsNormHeads(_qkv.View, _w.Vector(b + "self_attn.k_norm.weight"), T, _qkvCols, _qCols, _kvHeads, HeadDim, RmsEps);
            var cos = _cos.View.SubView((long)p0 * (HeadDim / 2), (long)T * (HeadDim / 2)); var sin = _sin.View.SubView((long)p0 * (HeadDim / 2), (long)T * (HeadDim / 2));
            _u.RopeTables(_qkv.View, cos, sin, T, _qkvCols, 0, _qHeads, HeadDim);
            _u.RopeTables(_qkv.View, cos, sin, T, _qkvCols, _qCols, _kvHeads, HeadDim);
            _u.StoreKv(_qkv.View, _kc[l].View, _vc[l].View, p0, T, _qCols, _kvCols);
            _u.CausalGqa(_qkv.View, _kc[l].View, _vc[l].View, _scores.View, _attn.View, p0, T, _qHeads, HeadDim, rep);
            Lin(_attn.View, b + "self_attn.o_proj.weight", _proj.View, T, _qCols, C);
            _k.GateResidual(_h.View, _proj.View, _dummy.View, -1, T * C, C);
            _u.RmsNorm(_h.View, _hn.View, _w.Vector(b + "post_attention_layernorm.weight"), T, C, RmsEps);
            Lin(_hn.View, b + "mlp.gate_up_proj.weight", _gateUp.View, T, C, 2 * _inter);
            _u.Swiglu(_gateUp.View, _act.View, T, _inter);
            Lin(_act.View, b + "mlp.down_proj.weight", _proj.View, T, _inter, C);
            _k.GateResidual(_h.View, _proj.View, _dummy.View, -1, T * C, C);
        }
        _u.RmsNorm(_h.View.SubView((long)(T - 1) * C, C), _last.View, _w.Vector("norm.weight"), 1, C, RmsEps);
        _u.Gemv(_last.View, _w.Vector("embed_tokens.weight").SubView((long)_logitStart * C, (long)_logitCount * C), null, _logits.View, C, _logitCount);
        _pos += T;
    }

    public void Dispose()
    {
        foreach (var b in _owned) b.Dispose();
        _owned.Clear();
    }
}
