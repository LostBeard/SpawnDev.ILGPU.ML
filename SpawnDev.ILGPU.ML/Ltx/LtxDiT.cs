using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.Wan;

namespace SpawnDev.ILGPU.ML.Ltx;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>
/// The LTX-Video 2B diffusion transformer (<c>LTXVideoTransformer3DModel</c>): a linear patch projection of the 128 latent channels (patch 1x1x1: every latent position is a token),
/// the caption projection MLP, a time embedding with a 6-way adaLN projection, 28 blocks of (RMS norm + adaLN modulation, self-attention with qk RMS norm over the whole q / k vector
/// and RoPE over the whole 2048 width, cross-attention to the T5 text on the raw residual stream, GELU-tanh feed-forward with a gate) and a modulated LayerNorm head. Batch 1.
/// Weights are the bf16 export of <c>tools/ltx-harness/reference/export_dit_bf16.py</c> (Wan layout and names; q/k/v and k/v fused), activations fp32, accumulation fp32.
/// The T5 padding mask is a pure suffix (the tokenizer pads on the right) and the reference adds -10000 to the padded keys, which is exactly zero attention weight, so the
/// cross-attention here just runs over the real text rows. The text half of every cross-attention (k and v of the caption) does not change between steps: <see cref="SetContext"/>
/// computes it once per prompt slot and block. The latent is [128, frames, h, w] (= [128, N] channel-major, the VAE layout); the DiT transposes to tokens and back.
/// Work buffers are allocated once per latent grid (<see cref="SetGrid"/>); <see cref="Dispose"/> frees them.
/// </summary>
public sealed class LtxDiT : IDisposable
{
    private readonly Accelerator _acc;
    private readonly DiTKernels _k;
    private readonly WanKernels _wk;
    private readonly LtxKernels _lk;
    private readonly GpuWeights _w;
    private readonly Flux2.Flux2Weights _mat;
    private readonly int _dim, _ffn, _heads, _headDim, _blocks, _inCh, _outCh, _textLen, _textDim, _freqDim;

    private int _fz, _h, _wd, _n;
    private readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _ws = new();
    private readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _ctxBufs = new();
    private MemoryBuffer1D<float, Stride1D.Dense>? _phases;
    private FV _tokens, _hid, _hn, _qkv, _q, _attn, _proj, _mlp, _scores, _headOut, _outTokens;
    private FV _tFreq, _tHidden, _e, _eAct, _e0, _mod, _hm, _ctxIn, _ctxHid;
    private readonly FV[] _textEmb = new FV[2];
    private readonly FV[][] _crossKv = new FV[2][];          // [slot][block]: k | v of the caption, [rows, 2C], k already RMS-normalised
    private readonly int[] _rows = new int[2];

    public int Dim => _dim;
    public int Layers => _blocks;
    public int LatentChannels => _inCh;
    public int TokenCount => _n;
    public int TextLength => _textLen;

    /// <summary>Test hook: (stage name, activation view, element count); valid only during the call.</summary>
    public Action<string, FV, int>? Tap { get; set; }

    /// <summary>Blocks whose output is tapped (as "block_N").</summary>
    public int[] TapBlocks { get; set; } = { 0, 1, 14, 27 };

    public LtxDiT(Accelerator accelerator, DiTKernels kernels, GpuWeights weights)
    {
        _acc = accelerator; _k = kernels; _wk = new WanKernels(accelerator); _lk = new LtxKernels(accelerator); _w = weights;
        _mat = new Flux2.Flux2Weights(accelerator, weights, ownsWeights: false);
        _dim = weights.Shape("patch_embedding.weight")[0];
        _inCh = int.Parse(weights.Metadata.GetValueOrDefault("in_dim", "128"));
        _outCh = int.Parse(weights.Metadata.GetValueOrDefault("out_dim", "128"));
        _ffn = int.Parse(weights.Metadata.GetValueOrDefault("ffn_dim", "8192"));
        _textDim = 4096;                          // the T5-v1.1-XXL width
        _freqDim = 256;
        _blocks = 0; while (weights.Has($"blocks.{_blocks}.modulation")) _blocks++;
        _heads = int.Parse(weights.Metadata.GetValueOrDefault("num_heads", "32"));
        _textLen = int.Parse(weights.Metadata.GetValueOrDefault("text_len", "128"));
        _headDim = _dim / _heads;
        if (_headDim != 64 && _headDim != 128) throw new NotSupportedException("LtxDiT: head dim 64 or 128 (the flash attention kernels) only");
    }

    private FV Alloc(long n)
    {
        var b = _acc.Allocate1D<float>(n);
        _ws.Add(b);
        return b.View;
    }

    private void FreeWorkspace()
    {
        foreach (var b in _ws) b.Dispose();
        _ws.Clear();
        _phases?.Dispose(); _phases = null;
    }

    /// <summary>
    /// Fix the latent grid ([128, <paramref name="frames"/>, <paramref name="h"/>, <paramref name="w"/>]; N = frames * h * w tokens) and the pipeline's
    /// <c>rope_interpolation_scale</c> (temporal VAE ratio / frame rate, spatial ratio, spatial ratio): builds the RoPE phase table and sizes the work buffers.
    /// </summary>
    public void SetGrid(int frames, int h, int w, float ropeScaleT, float ropeScaleH, float ropeScaleW)
    {
        FreeWorkspace();
        _fz = frames; _h = h; _wd = w; _n = frames * h * w;
        int N = _n, C = _dim;
        var table = LtxKernels.RopePhases(frames, h, w, C, ropeScaleT, ropeScaleH, ropeScaleW);
        _phases = _acc.Allocate1D<float>(table.Length);
        _phases.View.CopyFromCPU(table);
        _tFreq = Alloc(_freqDim); _tHidden = Alloc(C); _e = Alloc(C); _eAct = Alloc(C); _e0 = Alloc(6 * C); _mod = Alloc(6 * C); _hm = Alloc(2 * C);
        _tokens = Alloc((long)N * _inCh); _hid = Alloc((long)N * C); _hn = Alloc((long)N * C); _qkv = Alloc((long)N * 3 * C); _q = Alloc((long)N * C);
        _attn = Alloc((long)N * C); _proj = Alloc((long)N * C); _mlp = Alloc((long)N * _ffn);
        _scores = Alloc(_k.UsesFlashAttention(_headDim) ? 1 : (long)_heads * N * Math.Max(N, _textLen));
        _headOut = Alloc((long)N * _outCh); _outTokens = _headOut;
        _ctxIn = Alloc((long)_textLen * _textDim); _ctxHid = Alloc((long)_textLen * C);
    }

    /// <summary>
    /// The text context of prompt slot <paramref name="slot"/>: <paramref name="context"/> is [rows, 4096] row-major (the T5 output for the real tokens, rows &lt;= the text length). It is run
    /// through the caption projection (Linear, GELU tanh, Linear), then every block's cross-attention k (RMS-normalised) and v are computed once. Call again to change the prompt.
    /// </summary>
    public void SetContext(int slot, float[] context, int rows)
    {
        if (slot is < 0 or > 1) throw new ArgumentOutOfRangeException(nameof(slot));
        if (rows > _textLen || rows < 1 || context.Length < (long)rows * _textDim) throw new ArgumentException("context rows exceed the text length");
        if (_n == 0) throw new InvalidOperationException("SetGrid must be called before SetContext");
        int C = _dim;
        _ctxIn.SubView(0, (long)rows * _textDim).CopyFromCPU(context.AsSpan(0, rows * _textDim).ToArray());
        if (_textEmb[slot].Length == 0)
        {
            var b = _acc.Allocate1D<float>((long)_textLen * C); _ctxBufs.Add(b); _textEmb[slot] = b.View;
            _crossKv[slot] = new FV[_blocks];
            for (int i = 0; i < _blocks; i++) { var kv = _acc.Allocate1D<float>((long)_textLen * 2 * C); _ctxBufs.Add(kv); _crossKv[slot][i] = kv.View; }
        }
        _rows[slot] = rows;
        _k.Linear(_ctxIn, _mat.Matrix("text_embedding.0.weight"), _w.BF16("text_embedding.0.bias"), _ctxHid, rows, _textDim, C, DiTKernels.Act.GeluTanh);
        _k.Linear(_ctxHid, _mat.Matrix("text_embedding.2.weight"), _w.BF16("text_embedding.2.bias"), _textEmb[slot], rows, C, C);
        for (int b = 0; b < _blocks; b++)
        {
            string p = $"blocks.{b}.";
            var kv = _crossKv[slot][b];
            _k.Linear(_textEmb[slot], _mat.Matrix(p + "cross_attn.kv.weight"), _w.BF16(p + "cross_attn.kv.bias"), kv, rows, C, 2 * C);
            _wk.RowRmsNorm(kv, _w.BF16(p + "cross_attn.norm_k.weight"), rows, 2 * C, 0, C, 1e-5f);
        }
        _acc.Fence();
    }

    /// <summary>The time embedding e = time_embedding(sinusoid(t)) and the 6-way adaLN projection e0 = time_projection(silu(e)).</summary>
    private void TimeEmbed(double t)
    {
        int C = _dim;
        _tFreq.CopyFromCPU(WanKernels.SinusoidalEmbedding(t, _freqDim));
        _k.Linear(_tFreq, _mat.Matrix("time_embedding.0.weight"), _w.BF16("time_embedding.0.bias"), _tHidden, 1, _freqDim, C, DiTKernels.Act.Silu);
        _k.Linear(_tHidden, _mat.Matrix("time_embedding.2.weight"), _w.BF16("time_embedding.2.bias"), _e, 1, C, C);
        _k.Silu(_e, _eAct, C);
        _k.Linear(_eAct, _mat.Matrix("time_projection.1.weight"), _w.BF16("time_projection.1.bias"), _e0, 1, C, 6 * C);
    }

    /// <summary>
    /// One noise (velocity) prediction. <paramref name="latent"/> is [128, frames, h, w] (channel-major), <paramref name="o"/> receives the same shape. <paramref name="t"/> is the model
    /// timestep as the reference passes it. <paramref name="slot"/> selects the prompt set with <see cref="SetContext"/>.
    /// </summary>
    public void Forward(FV latent, double t, int slot, FV o)
    {
        if (_n == 0) throw new InvalidOperationException("SetGrid must be called before Forward");
        if (_textEmb[slot].Length == 0) throw new InvalidOperationException($"SetContext({slot}) must be called before Forward");
        int N = _n, C = _dim, H = _heads, D = _headDim, L = _rows[slot];

        TimeEmbed(t);
        Tap?.Invoke("time_embedding", _e, C);
        Tap?.Invoke("time_projection", _e0, 6 * C);
        Tap?.Invoke("text_embedding", _textEmb[slot], L * C);

        // tokens: [128, N] -> [N, 128], then the patch projection (a linear layer)
        _k.Transpose(latent, _tokens, _inCh, N);
        _k.Linear(_tokens, _mat.Matrix("patch_embedding.weight"), _w.BF16("patch_embedding.bias"), _hid, N, _inCh, C);
        Tap?.Invoke("patch_embedding", _hid, N * C);

        for (int b = 0; b < _blocks; b++)
        {
            string p = $"blocks.{b}.";
            _k.AddBf16(_w.BF16(p + "modulation"), _e0, _mod, 6 * C);          // scale_shift_table + temb: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

            // self attention
            _lk.RmsNormModulate(_hid, _hn, _mod, 0 * C, 1 * C, N, C, 1e-6f);
            _k.Linear(_hn, _mat.Matrix(p + "self_attn.qkv.weight"), _w.BF16(p + "self_attn.qkv.bias"), _qkv, N, C, 3 * C);
            _wk.RowRmsNorm(_qkv, _w.BF16(p + "self_attn.norm_q.weight"), N, 3 * C, 0 * C, C, 1e-5f);
            _wk.RowRmsNorm(_qkv, _w.BF16(p + "self_attn.norm_k.weight"), N, 3 * C, 1 * C, C, 1e-5f);
            _wk.RopeApply(_qkv, _phases!.View, N, 3 * C, 0 * C, 1, C);          // the rotation runs over the whole q / k vector: one "head" of width C
            _wk.RopeApply(_qkv, _phases.View, N, 3 * C, 1 * C, 1, C);
            _k.Attention(_qkv, 3 * C, 0, _qkv, 3 * C, 1 * C, 2 * C, _scores, _attn, N, N, H, D);
            _k.Linear(_attn, _mat.Matrix(p + "self_attn.o.weight"), _w.BF16(p + "self_attn.o.bias"), _proj, N, C, C);
            _k.GateResidual(_hid, _proj, _mod, 2 * C, N * C, C);

            // cross attention: on the raw residual stream (no norm, no modulation, no RoPE), plain residual
            _k.Linear(_hid, _mat.Matrix(p + "cross_attn.q.weight"), _w.BF16(p + "cross_attn.q.bias"), _q, N, C, C);
            _wk.RowRmsNorm(_q, _w.BF16(p + "cross_attn.norm_q.weight"), N, C, 0, C, 1e-5f);
            _k.Attention(_q, C, 0, _crossKv[slot][b], 2 * C, 0, C, _scores, _attn, N, L, H, D);
            _k.Linear(_attn, _mat.Matrix(p + "cross_attn.o.weight"), _w.BF16(p + "cross_attn.o.bias"), _proj, N, C, C);
            _k.GateResidual(_hid, _proj, default, -1, N * C, C);

            // feed forward
            _lk.RmsNormModulate(_hid, _hn, _mod, 3 * C, 4 * C, N, C, 1e-6f);
            _k.Linear(_hn, _mat.Matrix(p + "ffn.0.weight"), _w.BF16(p + "ffn.0.bias"), _mlp, N, C, _ffn, DiTKernels.Act.GeluTanh);
            _k.Linear(_mlp, _mat.Matrix(p + "ffn.2.weight"), _w.BF16(p + "ffn.2.bias"), _proj, N, _ffn, C);
            _k.GateResidual(_hid, _proj, _mod, 5 * C, N * C, C);

            if (Tap != null && Array.IndexOf(TapBlocks, b) >= 0) Tap($"block_{b}", _hid, N * C);
        }

        // head: LayerNorm (no affine) modulated by (scale_shift_table + e): shift = row 0, scale = row 1, then the output projection
        _wk.AddBroadcastBf16(_w.BF16("head.modulation"), _e, _hm, 2 * C, C);
        _k.LayerNormModulate(_hid, _hn, _hm, 0, C, N, C, 1e-6f);
        _k.Linear(_hn, _mat.Matrix("head.head.weight"), _w.BF16("head.head.bias"), _headOut, N, C, _outCh);
        Tap?.Invoke("head", _headOut, N * _outCh);
        _k.Transpose(_headOut, o, N, _outCh);                                    // [N, 128] -> [128, N]
        _acc.Fence();
    }

    public void Dispose()
    {
        _mat.Dispose();
        FreeWorkspace();
        foreach (var b in _ctxBufs) b.Dispose();
        _ctxBufs.Clear();
    }
}
