using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.UniRig;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>
/// The language model of UniRig's skeleton stage: a pre-LN OPT-350m decoder (24 layers, width 1024, 16 heads of 64, ReLU feed-forward of 4096, learned positions with
/// OPT's offset of 2, no embedding scaling, a separate untied lm_head of 267 tokens) that reads the 1024 encoder tokens as a prefix and then the skeleton tokens.
/// <code>
/// h = [encoder tokens | embed(bos) embed(cls) ...] + positions                  one causal sequence
/// 24 x { h += out_proj(attention(ln(h)))  ;  h += fc2(relu(fc1(ln'(h)))) }        qkv from one fused projection, per-layer KV cache
/// logits = lm_head(final_layer_norm(h))
/// </code>
/// bf16 weights, fp32 activations (<see cref="DiTKernels"/>). <see cref="PrefillAsync"/> consumes the prefix and the first tokens in one pass,
/// <see cref="StepAsync"/> one token at a time. Checked against the reference's teacher-forced logits (tools/unirig-harness, decoder gate).
/// </summary>
public sealed class UniRigDecoder : IDisposable
{
    public const int Layers = 24, Hidden = 1024, Heads = 16, HeadDim = Hidden / Heads, Ffn = 4096, Vocab = UniRigTokenizer.VocabSize, MaxPositions = 3076;

    private readonly Accelerator _acc;
    private readonly GpuWeights _w;
    private readonly DiTKernels _k;
    private readonly UniRigKernels _u;
    private readonly int _maxRows;
    private readonly List<IDisposable> _owned = new();
    private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _kc = new MemoryBuffer1D<float, Stride1D.Dense>[Layers], _vc = new MemoryBuffer1D<float, Stride1D.Dense>[Layers];
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _h, _hn, _qkv, _attn, _proj, _ffn, _scores, _logits, _dummy;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _ids;
    private int _pos;

    /// <param name="weights">The skeleton checkpoint (export_weights.py), at least the transformer.* tensors.</param>
    /// <param name="maxPrefillRows">Most rows one <see cref="PrefillAsync"/> may carry: encoder tokens plus the first skeleton tokens.</param>
    public UniRigDecoder(Accelerator accelerator, GpuWeights weights, int maxPrefillRows = 1024 + 256)
    {
        _acc = accelerator; _w = weights; _maxRows = maxPrefillRows;
        _k = new DiTKernels(accelerator); _u = new UniRigKernels(accelerator);
        for (int l = 0; l < Layers; l++) { _kc[l] = Alloc<float>((long)MaxPositions * Hidden); _vc[l] = Alloc<float>((long)MaxPositions * Hidden); }
        long rows = maxPrefillRows;
        _h = Alloc<float>(rows * Hidden); _hn = Alloc<float>(rows * Hidden); _qkv = Alloc<float>(rows * 3 * Hidden); _attn = Alloc<float>(rows * Hidden);
        _proj = Alloc<float>(rows * Hidden); _ffn = Alloc<float>(rows * Ffn);
        _scores = Alloc<float>(Math.Max((long)Heads * rows * rows, (long)Heads * MaxPositions));
        _logits = Alloc<float>(rows * Vocab); _dummy = Alloc<float>(1); _ids = Alloc<int>(rows);
    }

    private MemoryBuffer1D<T, Stride1D.Dense> Alloc<T>(long n) where T : unmanaged { var b = _acc.Allocate1D<T>(Math.Max(1, n)); _owned.Add(b); return b; }

    /// <summary>Tokens consumed so far (the next token's position).</summary>
    public int Position => _pos;

    private void LayerNorm(FV x, FV o, string name, int rows)
        => _k.LayerNormAffine(x, o, _w.BF16(name + ".weight"), _w.BF16(name + ".bias"), rows, Hidden, 1e-5f);

    /// <summary>
    /// Start a sequence. <paramref name="cond"/> is [condTokens, 1024] on the device (the encoder's output), followed by <paramref name="tokenIds"/> (bos, class, ...).
    /// Returns the logits of the last <paramref name="lastRows"/> rows, [lastRows, 267], row r predicting the token after the r-th of those positions.
    /// </summary>
    public async Task<float[]> PrefillAsync(FV cond, int condTokens, int[] tokenIds, int lastRows = 1)
    {
        int T = condTokens + tokenIds.Length;
        if (T > _maxRows) throw new ArgumentException($"{T} rows exceed maxPrefillRows {_maxRows}");
        if (T + 1 > MaxPositions) throw new ArgumentException("sequence longer than the model's positions");
        if (lastRows < 1 || lastRows > tokenIds.Length) throw new ArgumentOutOfRangeException(nameof(lastRows));
        _pos = 0;
        _h.View.SubView(0, (long)condTokens * Hidden).CopyFrom(cond.SubView(0, (long)condTokens * Hidden));
        _u.AddPositions(_h.View, _w.BF16("transformer.model.decoder.embed_positions.weight"), 0, condTokens, Hidden);
        _ids.View.SubView(0, tokenIds.Length).CopyFromCPU(tokenIds);
        _u.Embed(_ids.View, _w.BF16("transformer.model.decoder.embed_tokens.weight"), _w.BF16("transformer.model.decoder.embed_positions.weight"),
            _h.View.SubView((long)condTokens * Hidden, (long)tokenIds.Length * Hidden), condTokens, tokenIds.Length, Hidden);
        return await ForwardAsync(T, lastRows);
    }

    /// <summary>Append one token and return the distribution of the next, [267].</summary>
    public async Task<float[]> StepAsync(int tokenId)
    {
        if (_pos == 0) throw new InvalidOperationException("PrefillAsync first");
        if (_pos + 1 > MaxPositions) throw new InvalidOperationException("sequence reached the model's last position");
        _ids.View.SubView(0, 1).CopyFromCPU(new[] { tokenId });
        _u.Embed(_ids.View, _w.BF16("transformer.model.decoder.embed_tokens.weight"), _w.BF16("transformer.model.decoder.embed_positions.weight"), _h.View, _pos, 1, Hidden);
        return await ForwardAsync(1, 1);
    }

    /// <summary>T rows already embedded in _h at positions _pos.._pos+T-1.</summary>
    private async Task<float[]> ForwardAsync(int T, int lastRows)
    {
        int C = Hidden, p0 = _pos;
        for (int l = 0; l < Layers; l++)
        {
            string b = $"transformer.model.decoder.layers.{l}.";
            LayerNorm(_h.View, _hn.View, b + "self_attn_layer_norm", T);
            _k.Linear(_hn.View, _w.BF16(b + "self_attn.qkv_proj.weight"), _w.BF16(b + "self_attn.qkv_proj.bias"), _qkv.View, T, C, 3 * C);
            _u.StoreKv(_qkv.View, _kc[l].View, _vc[l].View, p0, T, C);
            _u.CausalAttention(_qkv.View, _kc[l].View, _vc[l].View, _scores.View, _attn.View, p0, T, Heads, HeadDim);
            _k.Linear(_attn.View, _w.BF16(b + "self_attn.out_proj.weight"), _w.BF16(b + "self_attn.out_proj.bias"), _proj.View, T, C, C);
            _k.GateResidual(_h.View, _proj.View, _dummy.View, -1, T * C, C);
            LayerNorm(_h.View, _hn.View, b + "final_layer_norm", T);
            _k.Linear(_hn.View, _w.BF16(b + "fc1.weight"), _w.BF16(b + "fc1.bias"), _ffn.View, T, C, Ffn);
            _u.Relu(_ffn.View, T * Ffn);
            _k.Linear(_ffn.View, _w.BF16(b + "fc2.weight"), _w.BF16(b + "fc2.bias"), _proj.View, T, Ffn, C);
            _k.GateResidual(_h.View, _proj.View, _dummy.View, -1, T * C, C);
        }
        var last = _h.View.SubView((long)(T - lastRows) * C, (long)lastRows * C);
        LayerNorm(last, _hn.View, "transformer.model.decoder.final_layer_norm", lastRows);
        _k.Linear(_hn.View, _w.BF16("transformer.lm_head.weight"), null, _logits.View, lastRows, C, Vocab);
        _pos += T;
        return await _logits.View.SubView(0, (long)lastRows * Vocab).CopyToHostAsync();
    }

    public void Dispose()
    {
        foreach (var o in _owned) o.Dispose();
        _owned.Clear();
    }
}
