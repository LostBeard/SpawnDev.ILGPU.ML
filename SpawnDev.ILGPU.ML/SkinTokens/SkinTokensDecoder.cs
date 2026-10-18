using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.SkinTokens;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>
/// The language model of TokenRig (SkinTokens): a Qwen3-style decoder (28 layers, width 896, 16 query heads and 8 key/value heads of 128, per-head RMS norm on q and k,
/// rotary positions with theta 1e6, SwiGLU of 3072, tied 33036-token embedding) that reads the mesh encoder's 512 prefix embeddings and then the rig tokens: skeleton
/// tokens, a switch token, 4 skin tokens per bone, end.
/// <code>
/// h = [prefix rows | embed(bos) embed(cls) ...]                                      one causal sequence
/// 28 x { h += o_proj(attention(rope(qk_norm(qkv(rms(h))))))  ;  h += down(swiglu(gate_up(rms'(h)))) }     per-layer KV cache
/// logits = lm_head(rms(h))
/// </code>
/// bf16 weights, fp32 activations (<see cref="DiTKernels"/>). <see cref="PrefillAsync"/> consumes the prefix and the first tokens in one pass, <see cref="StepAsync"/> one
/// token at a time. Checked against the reference's teacher-forced logits (tools/unirig-harness/reference/golden_skintokens.py).
/// </summary>
public sealed class SkinTokensDecoder : IDisposable
{
    public const int Layers = 28, Hidden = 896, QHeads = 16, KvHeads = 8, HeadDim = 128, Rep = QHeads / KvHeads, Inter = 3072, Vocab = 33036, MaxPositions = 3192;
    public const int QCols = QHeads * HeadDim, KvCols = KvHeads * HeadDim, QkvCols = QCols + 2 * KvCols;
    public const float RopeTheta = 1_000_000f, RmsEps = 1e-6f;

    private readonly Accelerator _acc;
    private readonly GpuWeights _w;
    private readonly DiTKernels _k;
    private readonly SkinTokensKernels _u;
    private readonly int _maxRows;
    private readonly List<IDisposable> _owned = new();
    private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _kc = new MemoryBuffer1D<float, Stride1D.Dense>[Layers], _vc = new MemoryBuffer1D<float, Stride1D.Dense>[Layers];
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _h, _hn, _qkv, _attn, _proj, _gateUp, _act, _scores, _logits, _dummy;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _ids;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _invFreq;
    private int _pos;

    /// <param name="weights">The SkinTokens export (export_weights.py skintokens), at least the transformer.* tensors.</param>
    /// <param name="maxPrefillRows">Most rows one <see cref="PrefillAsync"/> may carry: prefix tokens plus the first rig tokens.</param>
    public SkinTokensDecoder(Accelerator accelerator, GpuWeights weights, int maxPrefillRows = 512 + 64)
    {
        _acc = accelerator; _w = weights; _maxRows = maxPrefillRows;
        _k = new DiTKernels(accelerator); _u = new SkinTokensKernels(accelerator);
        for (int l = 0; l < Layers; l++) { _kc[l] = Alloc<float>((long)MaxPositions * KvCols); _vc[l] = Alloc<float>((long)MaxPositions * KvCols); }
        long rows = maxPrefillRows;
        _h = Alloc<float>(rows * Hidden); _hn = Alloc<float>(rows * Hidden); _qkv = Alloc<float>(rows * QkvCols); _attn = Alloc<float>(rows * QCols);
        _proj = Alloc<float>(rows * Hidden); _gateUp = Alloc<float>(rows * 2 * Inter); _act = Alloc<float>(rows * Inter);
        _scores = Alloc<float>(Math.Max((long)QHeads * rows * rows, (long)QHeads * MaxPositions));
        _logits = Alloc<float>(rows * Vocab); _dummy = Alloc<float>(1); _ids = Alloc<int>(rows);
        _invFreq = _acc.Allocate1D(SkinTokensKernels.RopeTable(RopeTheta, HeadDim)); _owned.Add(_invFreq);
    }

    private MemoryBuffer1D<T, Stride1D.Dense> Alloc<T>(long n) where T : unmanaged { var b = _acc.Allocate1D<T>(Math.Max(1, n)); _owned.Add(b); return b; }

    /// <summary>Tokens consumed so far (the next token's position).</summary>
    public int Position => _pos;

    /// <summary>
    /// Start a sequence. <paramref name="prefix"/> is [prefixRows, 896] on the device (the mesh encoder's output after output_proj), followed by <paramref name="tokenIds"/>
    /// (bos, class, ...). Returns the logits of the last <paramref name="lastRows"/> rows, [lastRows, 33036].
    /// </summary>
    public async Task<float[]> PrefillAsync(FV prefix, int prefixRows, int[] tokenIds, int lastRows = 1)
    {
        int T = prefixRows + tokenIds.Length;
        if (T > _maxRows) throw new ArgumentException($"{T} rows exceed maxPrefillRows {_maxRows}");
        if (T + 1 > MaxPositions) throw new ArgumentException("sequence longer than the model's positions");
        if (lastRows < 1 || lastRows > tokenIds.Length) throw new ArgumentOutOfRangeException(nameof(lastRows));
        _pos = 0;
        _h.View.SubView(0, (long)prefixRows * Hidden).CopyFrom(prefix.SubView(0, (long)prefixRows * Hidden));
        _ids.View.SubView(0, tokenIds.Length).CopyFromCPU(tokenIds);
        _u.Embed(_ids.View, _w.BF16("transformer.model.embed_tokens.weight"), _h.View.SubView((long)prefixRows * Hidden, (long)tokenIds.Length * Hidden), tokenIds.Length, Hidden);
        return await ForwardAsync(T, lastRows);
    }

    /// <summary>Append one token and return the distribution of the next, [33036].</summary>
    public async Task<float[]> StepAsync(int tokenId)
    {
        if (_pos == 0) throw new InvalidOperationException("PrefillAsync first");
        if (_pos + 1 > MaxPositions) throw new InvalidOperationException("sequence reached the model's last position");
        _ids.View.SubView(0, 1).CopyFromCPU(new[] { tokenId });
        _u.Embed(_ids.View, _w.BF16("transformer.model.embed_tokens.weight"), _h.View, 1, Hidden);
        return await ForwardAsync(1, 1);
    }

    /// <summary>x[rows, K] times w[N, K]^T: the general GEMM for a prefill, the coalesced GEMV for a single token.</summary>
    private void Lin(FV x, string weight, FV o, int rows, int K, int N)
    {
        if (rows == 1 && UseGemv) _u.Gemv(x, _w.BF16(weight), null, o, K, N);
        else _k.Linear(x, _w.BF16(weight), null, o, rows, K, N);
    }

    /// <summary>Single-token steps use <see cref="SkinTokensKernels.Gemv"/> (default); turn off to compare with the general GEMM.</summary>
    public static bool UseGemv = Environment.GetEnvironmentVariable("SKINTOKENS_GEMV") != "0";

    /// <summary>T rows already embedded in _h at positions _pos.._pos+T-1.</summary>
    private async Task<float[]> ForwardAsync(int T, int lastRows)
    {
        int C = Hidden, p0 = _pos;
        for (int l = 0; l < Layers; l++)
        {
            string b = $"transformer.model.layers.{l}.";
            _u.RmsNorm(_h.View, _hn.View, _w.BF16(b + "input_layernorm.weight"), T, C, RmsEps);
            Lin(_hn.View, b + "self_attn.qkv_proj.weight", _qkv.View, T, C, QkvCols);
            _u.RmsNormHeads(_qkv.View, _w.BF16(b + "self_attn.q_norm.weight"), T, QkvCols, 0, QHeads, HeadDim, RmsEps);
            _u.RmsNormHeads(_qkv.View, _w.BF16(b + "self_attn.k_norm.weight"), T, QkvCols, QCols, KvHeads, HeadDim, RmsEps);
            _u.Rope(_qkv.View, _invFreq.View, T, QkvCols, 0, QHeads, HeadDim, p0);
            _u.Rope(_qkv.View, _invFreq.View, T, QkvCols, QCols, KvHeads, HeadDim, p0);
            _u.StoreKv(_qkv.View, _kc[l].View, _vc[l].View, p0, T, QCols, KvCols);
            _u.CausalGqa(_qkv.View, _kc[l].View, _vc[l].View, _scores.View, _attn.View, p0, T, QHeads, HeadDim, Rep);
            Lin(_attn.View, b + "self_attn.o_proj.weight", _proj.View, T, QCols, C);
            _k.GateResidual(_h.View, _proj.View, _dummy.View, -1, T * C, C);
            _u.RmsNorm(_h.View, _hn.View, _w.BF16(b + "post_attention_layernorm.weight"), T, C, RmsEps);
            Lin(_hn.View, b + "mlp.gate_up_proj.weight", _gateUp.View, T, C, 2 * Inter);
            _u.Swiglu(_gateUp.View, _act.View, T, Inter);
            Lin(_act.View, b + "mlp.down_proj.weight", _proj.View, T, Inter, C);
            _k.GateResidual(_h.View, _proj.View, _dummy.View, -1, T * C, C);
        }
        var last = _h.View.SubView((long)(T - lastRows) * C, (long)lastRows * C);
        _u.RmsNorm(last, _hn.View, _w.BF16("transformer.model.norm.weight"), lastRows, C, RmsEps);
        Lin(_hn.View, "transformer.lm_head.weight", _logits.View, lastRows, C, Vocab);
        _pos += T;
        return await _logits.View.SubView(0, (long)lastRows * Vocab).CopyToHostAsync();
    }

    public void Dispose()
    {
        foreach (var o in _owned) o.Dispose();
        _owned.Clear();
    }
}
