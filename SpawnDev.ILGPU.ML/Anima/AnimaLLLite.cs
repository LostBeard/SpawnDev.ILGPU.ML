using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Anima;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>
/// ControlNet-LLLite v2 for <see cref="AnimaDiT"/> (kohya-ss sd-scripts networks/control_net_lllite_anima.py, Apache-2.0; e.g. anima-lllite-inpainting-v2).
/// A conditioning trunk turns a picture [C_in, 16 Hp, 16 Wp] into one row per DiT token ([Hp * Wp, cond_emb]); then each module perturbs the INPUT of one
/// DiT Linear (self-attention q / k / v, the MLP's first layer) of one block: c = cx + depth_embed, h = SiLU(down(x)), (gamma, beta) = cond_to_film(c),
/// m = SiLU(mid(cat[c, h]) * (1 + gamma) + beta), x' = x + up(m) * <see cref="Strength"/>.
/// <para>
/// Trunk: conv 4x4 s4, GroupNorm, SiLU; conv 3x3; GN SiLU; conv 4x4 s4; GN SiLU; residual blocks (GN SiLU conv3x3 GN SiLU conv3x3 + skip); conv 1x1;
/// LayerNorm. GroupNorm groups = the largest of 8 / 4 / 2 / 1 dividing the channels, eps 1e-5 (as the LayerNorm; 1e-6 costs 4.3e-6 on block 0).
/// MEASURED against ComfyUI's AnimaLLLiteApply (black box; tools/anima-ref/check_lllite.py, the float64 reference of this class): block 0 5.1e-7, block 27
/// 2.0e-7, the denoised output of both sampler calls 1.3e-6 / 1.0e-6 (the LLLite's own effect on call 0: 1.23).
/// </para>
/// Apply it by setting <see cref="AnimaDiT.LLLite"/>; it then acts on every <see cref="AnimaDiT.Forward"/> (both CFG slots). A start / end percent is the
/// caller's: set or clear <see cref="AnimaDiT.LLLite"/> per step. The weights are the LLLite file's (bf16), not owned.
/// </summary>
public sealed class AnimaLLLite : IDisposable
{
    /// <summary>The DiT Linear whose input a module perturbs.</summary>
    public enum Target
    {
        /// <summary>self_attn.q_proj</summary>
        Q,
        /// <summary>self_attn.k_proj</summary>
        K,
        /// <summary>self_attn.v_proj</summary>
        V,
        /// <summary>mlp.layer1 (the MLP's first Linear)</summary>
        MlpFc1,
    }

    private static readonly string[] TargetNames = { "self_attn_q_proj", "self_attn_k_proj", "self_attn_v_proj", "mlp_layer1" };
    private const string Cond = "lllite_conditioning1.";
    private readonly Accelerator _acc;
    private readonly DiTKernels _k;
    private readonly AnimaKernels _ak;
    private readonly GpuWeights _w;
    private readonly Flux2.Flux2Weights _mat;
    private readonly int _cin, _half, _condDim, _emb, _mlp, _res, _dim;
    private readonly bool[,] _has;

    private int _tokens, _capTokens;
    private readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _ws = new();
    private MemoryBuffer1D<float, Stride1D.Dense>? _cx;
    private FV _c, _gb, _h, _cat, _m, _up, _outQ, _outK, _outV;

    /// <summary>Picture channels the trunk takes (3 RGB; 4 = inpainting: RGB + mask, see <see cref="AnimaKernels.InpaintPicture"/>).</summary>
    public int PictureChannels => _cin;
    /// <summary>Width of the per-token conditioning rows.</summary>
    public int CondEmbDim => _emb;
    /// <summary>Hidden width of each module.</summary>
    public int MlpDim => _mlp;
    /// <summary>Residual blocks of the trunk.</summary>
    public int ResBlocks => _res;
    /// <summary>Modules in the file.</summary>
    public int Modules { get; }
    /// <summary>Tokens of the picture set by <see cref="SetPictureAsync"/> (it must equal the DiT's <see cref="AnimaDiT.TokenCount"/>); 0 before it.</summary>
    public int Tokens => _tokens;
    /// <summary>The modules' multiplier (ComfyUI's strength). Default 1.</summary>
    public float Strength { get; set; } = 1f;

    /// <summary>The LLLite over its weights already on the device (not owned).</summary>
    /// <param name="accelerator">the application's accelerator (never disposed here)</param>
    /// <param name="kernels">the transformer kernels (the DiT's own instance is fine)</param>
    /// <param name="weights">the LLLite file (lllite_conditioning1.*, lllite_dit_blocks_{i}_{target}.*)</param>
    /// <param name="dim">the DiT width (2048)</param>
    public AnimaLLLite(Accelerator accelerator, DiTKernels kernels, GpuWeights weights, int dim = 2048)
    {
        _acc = accelerator; _k = kernels; _ak = new AnimaKernels(accelerator); _w = weights; _dim = dim;
        _mat = new Flux2.Flux2Weights(accelerator, weights, ownsWeights: false);
        if (weights.Has(Cond + "aspp.proj.0.weight")) throw new NotSupportedException("AnimaLLLite: ASPP conditioning (use_aspp) is not supported");
        var c1 = weights.Shape(Cond + "conv1.weight");
        if (c1[2] != 4 || c1[3] != 4) throw new NotSupportedException($"AnimaLLLite: conv1 is {c1[2]}x{c1[3]}, expected 4x4 (LLLite v2)");
        _half = c1[0]; _cin = c1[1];
        _condDim = weights.Shape(Cond + "conv3.weight")[0];
        _emb = weights.Shape(Cond + "proj.weight")[0];
        while (weights.Has($"{Cond}resblocks.{_res}.conv1.weight")) _res++;
        int blocks = 0;
        while (Enumerable.Range(0, 4).Any(t => weights.Has(Name(blocks, (Target)t) + ".down.weight"))) blocks++;
        _has = new bool[Math.Max(blocks, 1), 4];
        int n = 0;
        for (int b = 0; b < blocks; b++)
            for (int t = 0; t < 4; t++)
                if (_has[b, t] = weights.Has(Name(b, (Target)t) + ".down.weight"))
                {
                    n++;
                    _mlp = weights.Shape(Name(b, (Target)t) + ".down.weight")[0];
                    if (weights.Shape(Name(b, (Target)t) + ".down.weight")[1] != dim) throw new NotSupportedException($"AnimaLLLite: {Name(b, (Target)t)} is not {dim} wide");
                }
        if (n == 0) throw new InvalidDataException("AnimaLLLite: no lllite_dit_blocks_* modules in the weights");
        Modules = n;
    }

    private static string Name(int block, Target t) => $"lllite_dit_blocks_{block}_{TargetNames[(int)t]}";

    /// <summary>Whether block <paramref name="block"/> has a module on <paramref name="target"/>.</summary>
    public bool Has(int block, Target target) => block >= 0 && block < _has.GetLength(0) && _has[block, (int)target];

    private FV Alloc(long n) { var b = _acc.Allocate1D<float>(n); _ws.Add(b); return b.View; }

    private async Task EnsureTokensAsync(int tokens)
    {
        if (tokens <= _capTokens) return;
        await _acc.SynchronizeAsync();                       // earlier dispatches may still read these
        foreach (var b in _ws) b.Dispose();
        _ws.Clear();
        _capTokens = tokens;
        long T = tokens;
        _c = Alloc(T * _emb); _gb = Alloc(T * 2 * _mlp); _h = Alloc(T * _mlp); _cat = Alloc(T * (_emb + _mlp)); _m = Alloc(T * _mlp);
        _up = Alloc(T * _dim); _outQ = Alloc(T * _dim); _outK = Alloc(T * _dim); _outV = Alloc(T * _dim);
    }

    // GroupNorm (+ SiLU) of a channel-first [C, HW] tensor into o: each group of channels is contiguous, so it is a LayerNorm over (C / g) * HW values
    private void GroupNorm(FV x, FV o, string name, int C, int HW, bool silu)
    {
        int g = 8;
        while (g > 1 && C % g != 0) g /= 2;
        _k.LayerNormPlain(x, o, g, C / g * HW, 1e-5f);
        _ak.ChannelAffine(o, _w.BF16(name + ".weight"), _w.BF16(name + ".bias"), C, HW, silu);
    }

    /// <summary>
    /// Run the conditioning trunk on <paramref name="picture"/> ([<see cref="PictureChannels"/>, height, width] in the model's range: [-1, 1], plus the mask
    /// channel for inpainting, see <see cref="AnimaKernels.InpaintPicture"/>; height and width multiples of 16 = 8 x the latent's). The rows stay on the
    /// device for every later <see cref="AnimaDiT.Forward"/> until the next call.
    /// </summary>
    public async Task SetPictureAsync(FV picture, int height, int width)
    {
        if (height % 16 != 0 || width % 16 != 0) throw new ArgumentException("the picture's height and width must be multiples of 16 (8 x the latent's, which is even)");
        if (picture.Length < (long)_cin * height * width) throw new ArgumentException($"the picture holds fewer than {_cin} x {height} x {width} values");
        int h4 = height / 4, w4 = width / 4, hp = height / 16, wp = width / 16, S = hp * wp;
        await EnsureTokensAsync(S);
        _tokens = S;
        var tmp = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
        FV T(long n) { var b = _acc.Allocate1D<float>(n); tmp.Add(b); return b.View; }
        try
        {
            FV a = T((long)_half * h4 * w4), bq = T((long)_half * h4 * w4);
            FV r0 = T((long)_condDim * S), r1 = T((long)_condDim * S), r2 = T((long)_condDim * S), pj = T((long)_emb * S);
            _ak.Conv2d(picture, _w.BF16(Cond + "conv1.weight"), _w.BF16(Cond + "conv1.bias"), a, _cin, height, width, _half, 4, 4, 0);
            GroupNorm(a, bq, Cond + "norm1", _half, h4 * w4, true);
            _ak.Conv2d(bq, _w.BF16(Cond + "conv2.weight"), _w.BF16(Cond + "conv2.bias"), a, _half, h4, w4, _half, 3, 1, 1);
            GroupNorm(a, bq, Cond + "norm2", _half, h4 * w4, true);
            _ak.Conv2d(bq, _w.BF16(Cond + "conv3.weight"), _w.BF16(Cond + "conv3.bias"), r1, _half, h4, w4, _condDim, 4, 4, 0);
            GroupNorm(r1, r0, Cond + "norm3", _condDim, S, true);                           // r0 = the residual stream
            for (int r = 0; r < _res; r++)
            {
                string p = $"{Cond}resblocks.{r}.";
                GroupNorm(r0, r1, p + "norm1", _condDim, S, true);
                _ak.Conv2d(r1, _w.BF16(p + "conv1.weight"), _w.BF16(p + "conv1.bias"), r2, _condDim, hp, wp, _condDim, 3, 1, 1);
                GroupNorm(r2, r1, p + "norm2", _condDim, S, true);
                _ak.Conv2d(r1, _w.BF16(p + "conv2.weight"), _w.BF16(p + "conv2.bias"), r2, _condDim, hp, wp, _condDim, 3, 1, 1);
                _k.GateResidual(r0, r2, default, -1, _condDim * S, _condDim);              // r0 += r2
            }
            _ak.Conv2d(r0, _w.BF16(Cond + "proj.weight"), _w.BF16(Cond + "proj.bias"), pj, _condDim, hp, wp, _emb, 1, 1, 0);
            if (_cx == null || _cx.Length < (long)S * _emb) { _cx?.Dispose(); _cx = _acc.Allocate1D<float>((long)S * _emb); }
            _k.Transpose(pj, r1, _emb, S);                                                   // [emb, S] -> [S, emb] (row, col token order)
            _k.LayerNormAffine(r1, _cx.View, _w.BF16(Cond + "out_norm.weight"), _w.BF16(Cond + "out_norm.bias"), S, _emb, 1e-5f);
        }
        finally
        {
            await _acc.SynchronizeAsync();                   // the temporaries are read by queued dispatches until here
            foreach (var b in tmp) b.Dispose();
        }
    }

    /// <summary>
    /// The perturbed input of <paramref name="target"/> in block <paramref name="block"/>: x + up(...) * <see cref="Strength"/> for <paramref name="x"/>
    /// [tokens, 2048], in this LLLite's own buffer (valid until the next call for the same target; <see cref="Target.MlpFc1"/> shares q's). Null when the
    /// file has no module there (the Linear then reads x).
    /// </summary>
    public FV? Apply(int block, Target target, FV x, int tokens)
    {
        if (!Has(block, target)) return null;
        if (_cx == null || tokens != _tokens) throw new InvalidOperationException($"AnimaLLLite: the picture gives {_tokens} tokens, the DiT has {tokens} (call SetPictureAsync with 8 x the latent's size)");
        string p = Name(block, target);
        int S = tokens, E = _emb, D = _mlp;
        FV c = _c.SubView(0, (long)S * E), gb = _gb.SubView(0, (long)S * 2 * D), h = _h.SubView(0, (long)S * D);
        FV cat = _cat.SubView(0, (long)S * (E + D)), m = _m.SubView(0, (long)S * D), up = _up.SubView(0, (long)S * _dim);
        FV dst = (target switch { Target.K => _outK, Target.V => _outV, _ => _outQ }).SubView(0, (long)S * _dim);
        _ak.AddRow(_cx.View.SubView(0, (long)S * E), _w.BF16(p + ".depth_embed"), c, S, E);
        _mat.Linear(_k, c, p + ".cond_to_film.weight", _w.BF16(p + ".cond_to_film.bias"), gb, S, E, 2 * D);
        _mat.Linear(_k, x, p + ".down.weight", _w.BF16(p + ".down.bias"), h, S, _dim, D, DiTKernels.Act.Silu);
        _ak.Concat(c, h, cat, S, E, D);
        _mat.Linear(_k, cat, p + ".mid.weight", _w.BF16(p + ".mid.bias"), m, S, E + D, D);
        _ak.FilmSilu(m, gb, S, D);
        _mat.Linear(_k, m, p + ".up.weight", _w.BF16(p + ".up.bias"), up, S, D, _dim);
        _ak.AddScaled(x, up, dst, Strength, S * _dim);
        return dst;
    }

    /// <summary>Frees the work buffers (not the weights, not the accelerator). Drain the queue first (SynchronizeAsync) on browser backends.</summary>
    public void Dispose()
    {
        _mat.Dispose();
        foreach (var b in _ws) b.Dispose();
        _ws.Clear();
        _cx?.Dispose();
    }
}
