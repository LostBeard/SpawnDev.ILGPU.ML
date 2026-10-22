using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Wan;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>
/// The Wan 2.2 TI2V-5B diffusion transformer (<c>WanModel</c>, model_type ti2v): patch embedding (a linear layer on 1x2x2 patches of the 48-channel latent),
/// the text embedding MLP, a time embedding and projection, 30 blocks of (adaLN self-attention with 3-D RoPE and RMS norm over the whole q and k vector,
/// cross-attention to the umT5 text, GELU-tanh feed-forward), and a modulated head. Batch 1.
/// Weights are the bf16 export of <c>tools/wan-harness/reference/export_dit_bf16.py</c> (q/k/v and k/v fused), activations are fp32, accumulation is fp32.
/// Per-token timesteps: in the image-to-video mode the first latent frame is the conditioning picture and has timestep 0, every other token has t. The reference
/// materialises a [tokens, 6, dim] modulation tensor for that (2 GB at 27k tokens); only two values ever occur, so the modulation is computed twice and
/// applied to two row segments. The text embedding of a prompt does not change between steps: <see cref="SetContext"/> computes it once per prompt slot.
/// Work buffers are allocated once per latent grid (<see cref="SetGrid"/>); <see cref="Dispose"/> frees them.
/// </summary>
public sealed class WanDiT : IDisposable
{
    private readonly Accelerator _acc;
    private readonly DiTKernels _k;
    private readonly WanKernels _wk;
    private readonly GpuWeights _w;
    private readonly int _dim, _ffn, _heads, _headDim, _blocks, _inCh, _outCh, _textLen, _textDim, _freqDim;
    public const int PatchTime = 1, PatchH = 2, PatchW = 2;

    private int _fz, _h, _wd, _n, _n0;
    private readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _ws = new();
    private readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _ctxBufs = new();
    private MemoryBuffer1D<float, Stride1D.Dense>? _phases;
    private FV _patches, _hid, _hn, _qkv, _q, _kv, _attn, _proj, _mlp, _scores, _headOut;
    private FV _tFreq, _tHidden, _eA, _eB, _eAct, _e0A, _e0B, _modA, _modB, _hmA, _hmB, _ctxIn, _ctxHid;
    private readonly FV[] _textEmb = new FV[2];

    public int Dim => _dim;
    public int Layers => _blocks;
    public int LatentChannels => _inCh;
    public int TokenCount => _n;
    public int TextLength => _textLen;

    /// <summary>Test hook: (stage name, activation view, element count); valid only during the call.</summary>
    public Action<string, FV, int>? Tap { get; set; }

    /// <summary>Blocks whose output is tapped (as "block_N").</summary>
    public int[] TapBlocks { get; set; } = { 0, 1, 14, 29 };

    public WanDiT(Accelerator accelerator, DiTKernels kernels, GpuWeights weights)
    {
        _acc = accelerator; _k = kernels; _wk = new WanKernels(accelerator); _w = weights;
        _dim = weights.Shape("patch_embedding.weight")[0];
        _inCh = weights.Shape("patch_embedding.weight")[1] / (PatchTime * PatchH * PatchW);
        _outCh = weights.Shape("head.head.weight")[0] / (PatchTime * PatchH * PatchW);
        _ffn = weights.Shape("blocks.0.ffn.0.weight")[0];
        _textDim = weights.Shape("text_embedding.0.weight")[1];
        _freqDim = weights.Shape("time_embedding.0.weight")[1];
        _blocks = 0; while (weights.Has($"blocks.{_blocks}.modulation")) _blocks++;
        _heads = int.Parse(weights.Metadata.GetValueOrDefault("num_heads", "24"));
        _textLen = int.Parse(weights.Metadata.GetValueOrDefault("text_len", "512"));
        _headDim = _dim / _heads;
        if (_headDim != 128) throw new NotSupportedException("WanDiT: head dim 128 (the flash attention kernels) only");
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
    /// Fix the latent grid ([48, <paramref name="frames"/>, <paramref name="h"/>, <paramref name="w"/>]; h and w even): builds the RoPE phase table and sizes the
    /// work buffers. Tokens N = frames * h/2 * w/2.
    /// </summary>
    public void SetGrid(int frames, int h, int w)
    {
        if (h % 2 != 0 || w % 2 != 0) throw new ArgumentException("the latent height and width must be even (1x2x2 patches)");
        FreeWorkspace();
        _fz = frames; _h = h; _wd = w; _n = frames * (h / 2) * (w / 2); _n0 = (h / 2) * (w / 2);
        int N = _n, C = _dim;
        var table = WanKernels.RopePhases3D(frames, h / 2, w / 2, _headDim);
        _phases = _acc.Allocate1D<float>(table.Length);
        _phases.View.CopyFromCPU(table);
        _tFreq = Alloc(_freqDim); _tHidden = Alloc(C); _eA = Alloc(C); _eB = Alloc(C); _eAct = Alloc(C);
        _e0A = Alloc(6 * C); _e0B = Alloc(6 * C); _modA = Alloc(6 * C); _modB = Alloc(6 * C); _hmA = Alloc(2 * C); _hmB = Alloc(2 * C);
        _patches = Alloc((long)N * _inCh * 4); _hid = Alloc((long)N * C); _hn = Alloc((long)N * C); _qkv = Alloc((long)N * 3 * C); _q = Alloc((long)N * C);
        _kv = Alloc((long)_textLen * 2 * C); _attn = Alloc((long)N * C); _proj = Alloc((long)N * C); _mlp = Alloc((long)N * _ffn);
        _scores = Alloc(_k.UsesFlashAttention(_headDim) ? 1 : (long)_heads * N * Math.Max(N, _textLen));
        _headOut = Alloc((long)N * _outCh * 4);
        _ctxIn = Alloc((long)_textLen * _textDim); _ctxHid = Alloc((long)_textLen * C);
    }

    /// <summary>
    /// The text context of prompt slot <paramref name="slot"/> (0 = conditional, 1 = the negative prompt): <paramref name="context"/> is [rows, 4096] row-major
    /// (the umT5 output for the real tokens); it is zero-padded to the text length (the reference does too, and attends over all 512 rows) and run through the text
    /// embedding MLP (Linear, GELU tanh, Linear) once. Call again to change the prompt.
    /// </summary>
    public void SetContext(int slot, float[] context, int rows)
    {
        if (slot is < 0 or > 1) throw new ArgumentOutOfRangeException(nameof(slot));
        if (rows > _textLen || context.Length < (long)rows * _textDim) throw new ArgumentException("context rows exceed the text length");
        if (_n == 0) throw new InvalidOperationException("SetGrid must be called before SetContext");
        int C = _dim;
        var padded = new float[(long)_textLen * _textDim];
        Array.Copy(context, padded, (long)rows * _textDim);
        _ctxIn.CopyFromCPU(padded);
        if (_textEmb[slot].Length == 0) { var b = _acc.Allocate1D<float>((long)_textLen * C); _ctxBufs.Add(b); _textEmb[slot] = b.View; }
        _k.Linear(_ctxIn, _w.BF16("text_embedding.0.weight"), _w.BF16("text_embedding.0.bias"), _ctxHid, _textLen, _textDim, C, DiTKernels.Act.GeluTanh);
        _k.Linear(_ctxHid, _w.BF16("text_embedding.2.weight"), _w.BF16("text_embedding.2.bias"), _textEmb[slot], _textLen, C, C);
        _acc.Fence();
    }

    /// <summary>The time embedding e = time_embedding(sinusoid(t)) and the 6-way modulation e0 = time_projection(e) for one timestep value.</summary>
    private void TimeEmbed(double t, FV e, FV e0)
    {
        int C = _dim;
        _tFreq.CopyFromCPU(WanKernels.SinusoidalEmbedding(t, _freqDim));
        _k.Linear(_tFreq, _w.BF16("time_embedding.0.weight"), _w.BF16("time_embedding.0.bias"), _tHidden, 1, _freqDim, C, DiTKernels.Act.Silu);
        _k.Linear(_tHidden, _w.BF16("time_embedding.2.weight"), _w.BF16("time_embedding.2.bias"), e, 1, C, C);
        _k.Silu(e, _eAct, C);
        _k.Linear(_eAct, _w.BF16("time_projection.1.weight"), _w.BF16("time_projection.1.bias"), e0, 1, C, 6 * C);
    }

    /// <summary>
    /// One noise prediction. <paramref name="latent"/> is [48, frames, h, w], <paramref name="o"/> receives the same shape. <paramref name="t"/> is the model timestep
    /// (the scheduler's, 0..1000). <paramref name="firstFrameClean"/>: the first latent frame is the conditioning picture (timestep 0 for its tokens).
    /// <paramref name="slot"/> selects the prompt set with <see cref="SetContext"/>.
    /// </summary>
    public void Forward(FV latent, double t, bool firstFrameClean, int slot, FV o)
    {
        if (_n == 0) throw new InvalidOperationException("SetGrid must be called before Forward");
        if (_textEmb[slot].Length == 0) throw new InvalidOperationException($"SetContext({slot}) must be called before Forward");
        int N = _n, C = _dim, H = _heads, D = _headDim, L = _textLen;
        int n0 = firstFrameClean ? _n0 : 0;           // rows [0, n0) use the clean-frame modulation (timestep 0), rows [n0, N) the step's
        FV tx = _textEmb[slot];

        // time: two embeddings when the first frame is clean (timestep 0 for its tokens), else one
        if (firstFrameClean) TimeEmbed(0.0, _eA, _e0A);
        TimeEmbed(t, _eB, _e0B);
        Tap?.Invoke("time_embedding", _eB, C);
        Tap?.Invoke("time_projection", _e0B, 6 * C);
        Tap?.Invoke("text_embedding", tx, L * C);

        // patch embedding: gather the 1x2x2 patches, then the linear layer
        _wk.Patchify(latent, _patches, _inCh, _fz, _h, _wd);
        _k.Linear(_patches, _w.BF16("patch_embedding.weight"), _w.BF16("patch_embedding.bias"), _hid, N, _inCh * 4, C);
        Tap?.Invoke("patch_embedding", _hid, N * C);

        void Segments(Action<int, int, FV> body) { if (n0 > 0) body(0, n0, _modA); body(n0, N, _modB); }
        FV Rows(FV v, int row0, int rows) => v.SubView((long)row0 * C, (long)rows * C);

        for (int b = 0; b < _blocks; b++)
        {
            string p = $"blocks.{b}.";
            if (n0 > 0) _k.AddBf16(_w.BF16(p + "modulation"), _e0A, _modA, 6 * C);          // (self.modulation + e): shift/scale/gate msa, shift/scale/gate mlp
            _k.AddBf16(_w.BF16(p + "modulation"), _e0B, _modB, 6 * C);

            // self attention
            Segments((r0, r1, mod) => _k.LayerNormModulate(Rows(_hid, r0, r1 - r0), Rows(_hn, r0, r1 - r0), mod, 0 * C, 1 * C, r1 - r0, C, 1e-6f));
            _k.Linear(_hn, _w.BF16(p + "self_attn.qkv.weight"), _w.BF16(p + "self_attn.qkv.bias"), _qkv, N, C, 3 * C);
            _wk.RowRmsNorm(_qkv, _w.BF16(p + "self_attn.norm_q.weight"), N, 3 * C, 0 * C, C, 1e-6f);
            _wk.RowRmsNorm(_qkv, _w.BF16(p + "self_attn.norm_k.weight"), N, 3 * C, 1 * C, C, 1e-6f);
            _wk.RopeApply(_qkv, _phases!.View, N, 3 * C, 0 * C, H, D);
            _wk.RopeApply(_qkv, _phases.View, N, 3 * C, 1 * C, H, D);
            _k.Attention(_qkv, 3 * C, 0, _qkv, 3 * C, 1 * C, 2 * C, _scores, _attn, N, N, H, D);
            _k.Linear(_attn, _w.BF16(p + "self_attn.o.weight"), _w.BF16(p + "self_attn.o.bias"), _proj, N, C, C);
            Segments((r0, r1, mod) => _k.GateResidual(Rows(_hid, r0, r1 - r0), Rows(_proj, r0, r1 - r0), mod, 2 * C, (r1 - r0) * C, C));

            // cross attention (norm3 is the affine LayerNorm, no modulation, no RoPE, plain residual)
            _k.LayerNormAffine(_hid, _hn, _w.BF16(p + "norm3.weight"), _w.BF16(p + "norm3.bias"), N, C, 1e-6f);
            _k.Linear(_hn, _w.BF16(p + "cross_attn.q.weight"), _w.BF16(p + "cross_attn.q.bias"), _q, N, C, C);
            _k.Linear(tx, _w.BF16(p + "cross_attn.kv.weight"), _w.BF16(p + "cross_attn.kv.bias"), _kv, L, C, 2 * C);
            _wk.RowRmsNorm(_q, _w.BF16(p + "cross_attn.norm_q.weight"), N, C, 0, C, 1e-6f);
            _wk.RowRmsNorm(_kv, _w.BF16(p + "cross_attn.norm_k.weight"), L, 2 * C, 0, C, 1e-6f);
            _k.Attention(_q, C, 0, _kv, 2 * C, 0, C, _scores, _attn, N, L, H, D);
            _k.Linear(_attn, _w.BF16(p + "cross_attn.o.weight"), _w.BF16(p + "cross_attn.o.bias"), _proj, N, C, C);
            _k.GateResidual(_hid, _proj, _modB, -1, N * C, C);

            // feed forward
            Segments((r0, r1, mod) => _k.LayerNormModulate(Rows(_hid, r0, r1 - r0), Rows(_hn, r0, r1 - r0), mod, 3 * C, 4 * C, r1 - r0, C, 1e-6f));
            _k.Linear(_hn, _w.BF16(p + "ffn.0.weight"), _w.BF16(p + "ffn.0.bias"), _mlp, N, C, _ffn, DiTKernels.Act.GeluTanh);
            _k.Linear(_mlp, _w.BF16(p + "ffn.2.weight"), _w.BF16(p + "ffn.2.bias"), _proj, N, _ffn, C);
            Segments((r0, r1, mod) => _k.GateResidual(Rows(_hid, r0, r1 - r0), Rows(_proj, r0, r1 - r0), mod, 5 * C, (r1 - r0) * C, C));

            if (Tap != null && Array.IndexOf(TapBlocks, b) >= 0) Tap($"block_{b}", _hid, N * C);
        }

        // head: LayerNorm (no affine) modulated by (head.modulation + e), Linear to 4 * out channels, unpatchify
        if (n0 > 0) _wk.AddBroadcastBf16(_w.BF16("head.modulation"), _eA, _hmA, 2 * C, C);
        _wk.AddBroadcastBf16(_w.BF16("head.modulation"), _eB, _hmB, 2 * C, C);
        if (n0 > 0) _k.LayerNormModulate(Rows(_hid, 0, n0), Rows(_hn, 0, n0), _hmA, 0, C, n0, C, 1e-6f);
        _k.LayerNormModulate(Rows(_hid, n0, N - n0), Rows(_hn, n0, N - n0), _hmB, 0, C, N - n0, C, 1e-6f);
        _k.Linear(_hn, _w.BF16("head.head.weight"), _w.BF16("head.head.bias"), _headOut, N, C, _outCh * 4);
        Tap?.Invoke("head", _headOut, N * _outCh * 4);
        _wk.Unpatchify(_headOut, o, _outCh, _fz, _h, _wd);
        _acc.Fence();
    }

    public void Dispose()
    {
        FreeWorkspace();
        foreach (var b in _ctxBufs) b.Dispose();
        _ctxBufs.Clear();
    }
}
