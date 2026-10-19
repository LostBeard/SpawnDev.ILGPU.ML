using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Flux3;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>The shape of a FLUX 3 image DiT (the released Klein is the default; the numbers are parameters so a small copy can be gated against the reference code on random weights).</summary>
/// <param name="Dim">hidden width (3072)</param>
/// <param name="Heads">attention heads; <c>Dim / Heads</c> must be 128 (the rotary axes are 4 x 32)</param>
/// <param name="EarlyDepth">blocks every token mode runs on its own before the shared stream (5)</param>
/// <param name="SingleDepth">blocks of the shared stream (28)</param>
/// <param name="MlpHidden">SwiGLU width (Dim * 3 = 9216)</param>
/// <param name="TextChannels">width of a text token: 8 Qwen3-VL layers of 2560 (20480)</param>
/// <param name="Theta">rotary base (10000)</param>
/// <param name="FlowMap">the distilled model adds a second time embedding of the time each step jumps to</param>
public sealed record Flux3Config(int Dim = 3072, int Heads = 24, int EarlyDepth = 5, int SingleDepth = 28, int MlpHidden = 9216, int TextChannels = 20480, float Theta = 10000f, bool FlowMap = true)
{
    public const int HeadDim = 128, InChannels = 128, TimeChannels = 256;
}

/// <summary>
/// FLUX 3 Image Klein's diffusion transformer (black-forest-labs/flux-3-image-klein, <c>src/flux3/dit.py</c>), no biases anywhere. Tokens come in three MODES that sit in one sequence per sample in the order
/// <b>image | image_cond (the reference pictures) | text</b>, each with its own input projection and its own modulation vectors:
/// <code>
/// early  : EarlyDepth x { each mode alone: x += gate * (attn_out(attention(rope(qk_norm(qkv(ln_mod(x)))))) + mlp_out(swiglu(mlp_in(ln_mod(x))))) }   (attention only inside the mode)
/// single : SingleDepth x the same block over the whole sequence (joint attention), per-mode modulation
/// final  : image rows -> ln_mod -> Linear(Dim, 128)
/// </code>
/// ln_mod(x) = layernorm(x, no affine, eps 1e-6) * (1 + scale) + shift, and (shift, scale, gate) = Linear(silu(time vector)) of the token's mode. The time vector is
/// time_in(sinusoid(t * 1000)) (+ ref_time_in(sinusoid(t_next * 1000)) for the flow-map model); only the picture's tokens see the real t, the other modes see 0.
/// The blocks are the FLUX.2 single-stream block ([q|k|v|gate|up] in one matrix, [attn_out|mlp_out] in one) with the weights fused by <c>convert_flux3.py</c>:
/// <c>early.{image|image_cond|text}.{d}.{qkv_mlp|out|norm_q|norm_k}</c>, <c>single.{i}.*</c>, <c>emb_in.image</c>, <c>emb_in.image_cond</c>, <c>txt_in</c>, <c>time_in.{in_layer|out_layer}</c>, <c>ref_time_in.*</c>,
/// <c>mod_early.{image|image_cond|txt}</c>, <c>mod_single.*</c>, <c>final.linear</c>, <c>final.mod</c> (all <c>.weight</c>).
/// </summary>
public sealed class Flux3Transformer : IDisposable
{
    private readonly Accelerator _acc;
    private readonly Flux2Weights _w;
    private readonly Flux3Config _c;
    private readonly DiTKernels _k;
    private readonly SkinTokensKernels _u;
    private readonly Flux2Kernels _f;
    private readonly List<IDisposable> _owned = new();
    private readonly int _nImg, _nCond, _nTxt, _n, _dim, _heads, _mlp, _cols;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _x, _norm, _fused, _attn, _gateUp, _scores, _cos, _sin, _out, _inImg, _inCond, _inTxt;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _tin, _thid, _tact, _vecImg, _vecOther, _vecRef, _vecAct, _modBuf, _finalMod;
    private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _early = new MemoryBuffer1D<float, Stride1D.Dense>[3], _single = new MemoryBuffer1D<float, Stride1D.Dense>[3];

    /// <summary>When true (default) the timestep is rounded to bf16 the way the reference's bf16 tensors do; off feeds the exact float (the fp32 gates).</summary>
    public bool RoundTimestepToBf16 { get; set; } = true;

    public static readonly string[] Modes = { "image", "image_cond", "text" };
    private static readonly string[] ModKeys = { "image", "image_cond", "txt" };           // the checkpoint calls the text mode "txt" in its modulation tables

    public Flux3Transformer(Accelerator accelerator, Flux2Weights weights, Flux3Config config, int imageTokens, int condTokens, int textTokens)
    {
        if (config.Dim / config.Heads != Flux3Config.HeadDim) throw new ArgumentException("Dim / Heads must be 128");
        _acc = accelerator; _w = weights; _c = config; _nImg = imageTokens; _nCond = condTokens; _nTxt = textTokens; _n = imageTokens + condTokens + textTokens;
        _dim = config.Dim; _heads = config.Heads; _mlp = config.MlpHidden; _cols = 3 * _dim + 2 * _mlp;
        _k = new DiTKernels(accelerator); _u = new SkinTokensKernels(accelerator); _f = new Flux2Kernels(accelerator);
        long N = _n, D = _dim;
        _x = Alloc(N * D); _norm = Alloc(N * D); _fused = Alloc(N * _cols); _attn = Alloc(N * D); _gateUp = Alloc(N * (D + _mlp));
        _scores = Alloc(_k.UsesFlashAttention(Flux3Config.HeadDim) ? 1 : (long)_heads * N * N);
        _cos = Alloc(N * Flux3Config.HeadDim); _sin = Alloc(N * Flux3Config.HeadDim); _out = Alloc((long)Math.Max(1, _nImg) * Flux3Config.InChannels);
        _inImg = Alloc((long)Math.Max(1, _nImg) * Flux3Config.InChannels); _inCond = Alloc((long)Math.Max(1, _nCond) * Flux3Config.InChannels); _inTxt = Alloc((long)Math.Max(1, _nTxt) * _c.TextChannels);
        _tin = Alloc(Flux3Config.TimeChannels); _thid = Alloc(D); _tact = Alloc(D); _vecImg = Alloc(D); _vecOther = Alloc(D); _vecRef = Alloc(D); _vecAct = Alloc(D); _modBuf = Alloc(3 * D); _finalMod = Alloc(2 * D);
        for (int m = 0; m < 3; m++) { _early[m] = Alloc(3 * D); _single[m] = Alloc(3 * D); }
    }

    private MemoryBuffer1D<float, Stride1D.Dense> Alloc(long n) { var b = _acc.Allocate1D<float>(Math.Max(1, n)); _owned.Add(b); return b; }

    /// <summary>The prediction for the picture's tokens of the last <see cref="Forward"/>, [imageTokens, 128].</summary>
    public FV Output => _out.View;

    /// <summary>Intermediate checkpoints for tests: name, device view, rows, columns.</summary>
    public Action<string, FV, int, int>? Probe;

    private static float BF16Round(float f)
    {
        uint u = BitConverter.SingleToUInt32Bits(f);
        if ((u & 0x7FFFFFFFu) > 0x7F800000u) return f;
        u += 0x7FFFu + ((u >> 16) & 1u);
        return BitConverter.UInt32BitsToSingle(u & 0xFFFF0000u);
    }

    /// <summary>
    /// Rotary tables for (t, row, column, l) ids [rows, 4] (floats: the picture's rows and columns are halved): cos and sin [rows, 128], each axis' 16 frequencies repeated for the interleaved pairs.
    /// </summary>
    public static (float[] Cos, float[] Sin) RopeTables(float[] ids, int rows, float theta)
    {
        const int HeadDim = Flux3Config.HeadDim;
        var cos = new float[rows * HeadDim]; var sin = new float[rows * HeadDim];
        for (int r = 0; r < rows; r++)
        {
            int o = 0;
            for (int a = 0; a < 4; a++)
                for (int j = 0; j < 16; j++)
                {
                    double ang = ids[r * 4 + a] / Math.Pow(theta, 2.0 * j / 32);
                    float c = (float)Math.Cos(ang), s = (float)Math.Sin(ang);
                    cos[r * HeadDim + o] = c; cos[r * HeadDim + o + 1] = c; sin[r * HeadDim + o] = s; sin[r * HeadDim + o + 1] = s;
                    o += 2;
                }
        }
        return (cos, sin);
    }

    private void Lin(FV x, string weight, FV o, int rows, int K, int N)
    {
        if (rows == 1) _u.Gemv(x, _w.Matrix(weight), null, o, K, N);
        else _k.Linear(x, _w.Matrix(weight), null, o, rows, K, N);
    }

    /// <summary>Sinusoid of t * 1000 as the reference computes it (cos | sin of 128 frequencies), through an MLPEmbedder (in_layer, silu, out_layer) into <paramref name="o"/> [Dim].</summary>
    private void TimeVector(float t, string name, FV o)
    {
        float tt = t;
        if (RoundTimestepToBf16) { tt = BF16Round(t); tt = BF16Round(1000f * tt); } else tt = 1000f * t;
        var s = new float[Flux3Config.TimeChannels];
        for (int i = 0; i < s.Length / 2; i++)
        {
            double f = Math.Exp(-Math.Log(10000.0) * i / (s.Length / 2)), a = tt * f;
            s[i] = RoundTimestepToBf16 ? BF16Round((float)Math.Cos(a)) : (float)Math.Cos(a);
            s[s.Length / 2 + i] = RoundTimestepToBf16 ? BF16Round((float)Math.Sin(a)) : (float)Math.Sin(a);
        }
        _tin.View.CopyFromCPU(s);
        Lin(_tin.View, name + ".in_layer.weight", _thid.View, 1, Flux3Config.TimeChannels, _dim);
        _k.Silu(_thid.View, _tact.View, _dim);
        Lin(_tact.View, name + ".out_layer.weight", o, 1, _dim, _dim);
    }

    /// <summary>
    /// One denoising step's model call. <paramref name="image"/> [imageTokens, 128], <paramref name="cond"/> [condTokens, 128] (the reference pictures' tokens), <paramref name="text"/> [textTokens, TextChannels]
    /// are host arrays, <paramref name="t"/> the time x is at (1 = noise), <paramref name="tNext"/> the time the step jumps to (flow-map model), the ids [rows, 4] as the reference builds them
    /// (picture and references: (0, row / 2, col / 2, 0), reference i with l = -(1 + i); text (0, 0, 0, index)). The velocity is left in <see cref="Output"/>.
    /// </summary>
    public void Forward(float[] image, float[] cond, float[] text, float t, float tNext, float[] imageIds, float[] condIds, float[] textIds)
    {
        _inImg.View.SubView(0, image.Length).CopyFromCPU(image);
        if (_nCond > 0) _inCond.View.SubView(0, cond.Length).CopyFromCPU(cond);
        _inTxt.View.SubView(0, text.Length).CopyFromCPU(text);
        var ids = new float[_n * 4];
        Array.Copy(imageIds, 0, ids, 0, _nImg * 4); Array.Copy(condIds, 0, ids, _nImg * 4, _nCond * 4); Array.Copy(textIds, 0, ids, (_nImg + _nCond) * 4, _nTxt * 4);
        var (cos, sin) = RopeTables(ids, _n, _c.Theta);
        _cos.View.SubView(0, cos.Length).CopyFromCPU(cos); _sin.View.SubView(0, sin.Length).CopyFromCPU(sin);
        Forward(t, tNext);
    }

    private void Forward(float t, float tNext)
    {
        int D = _dim;
        // ---- time vectors and the modulations of every mode ---------------------------------------------------------------------------
        TimeVector(t, "time_in", _vecImg.View);
        TimeVector(0f, "time_in", _vecOther.View);
        if (_c.FlowMap)
        {
            TimeVector(tNext, "ref_time_in", _vecRef.View);
            var a = _vecImg.View.GetAsArray1D(); var b = _vecOther.View.GetAsArray1D(); var r = _vecRef.View.GetAsArray1D();
            for (int i = 0; i < D; i++) { a[i] += r[i]; b[i] += r[i]; }
            _vecImg.View.CopyFromCPU(a); _vecOther.View.CopyFromCPU(b);
        }
        for (int m = 0; m < 3; m++)
        {
            var vec = m == 0 ? _vecImg.View : _vecOther.View;
            _k.Silu(vec, _vecAct.View, D);
            Lin(_vecAct.View, $"mod_early.{ModKeys[m]}.weight", _early[m].View, 1, D, 3 * D);
            Lin(_vecAct.View, $"mod_single.{ModKeys[m]}.weight", _single[m].View, 1, D, 3 * D);
            if (m == 0) Lin(_vecAct.View, "final.mod.weight", _finalMod.View, 1, D, 2 * D);
        }
        // ---- the token projections into one sequence [image | image_cond | text] ---------------------------------------------------------
        if (_nImg > 0) Lin(_inImg.View, "emb_in.image.weight", _x.View, _nImg, Flux3Config.InChannels, D);
        if (_nCond > 0) Lin(_inCond.View, "emb_in.image_cond.weight", _x.View.SubView((long)_nImg * D, (long)_nCond * D), _nCond, Flux3Config.InChannels, D);
        Lin(_inTxt.View, "txt_in.weight", _x.View.SubView((long)(_nImg + _nCond) * D, (long)_nTxt * D), _nTxt, _c.TextChannels, D);
        Probe?.Invoke("embedded", _x.View, _n, D);

        var starts = new[] { 0, _nImg, _nImg + _nCond }; var counts = new[] { _nImg, _nCond, _nTxt };
        // ---- early blocks: every mode on its own ---------------------------------------------------------------------------------------------
        for (int d = 0; d < _c.EarlyDepth; d++)
        {
            for (int m = 0; m < 3; m++)
            {
                if (counts[m] == 0) continue;
                Block($"early.{Modes[m]}.{d}", new[] { (starts[m], counts[m], _early[m].View) }, starts[m], counts[m]);
            }
            Probe?.Invoke($"early{d + 1}", _x.View, _n, D);
        }
        // ---- single blocks: the whole sequence, per-mode modulation ------------------------------------------------------------------------
        var segments = new List<(int Row, int Rows, FV Mod)>();
        for (int m = 0; m < 3; m++) if (counts[m] > 0) segments.Add((starts[m], counts[m], _single[m].View));
        var segArray = segments.ToArray();
        for (int l = 0; l < _c.SingleDepth; l++)
        {
            Block($"single.{l}", segArray, 0, _n);
            if (Probe != null && (l == 0 || l == _c.SingleDepth - 1)) Probe($"single{l + 1}", _x.View, _n, D);
        }
        // ---- the picture's rows -> final layer -------------------------------------------------------------------------------------------------
        var img = _x.View.SubView(0, (long)_nImg * D);
        _k.LayerNormModulate(img, _norm.View, _finalMod.View, 0, D, _nImg, D, 1e-6f);        // chunk order (shift, scale)
        Lin(_norm.View, "final.linear.weight", _out.View, _nImg, D, Flux3Config.InChannels);
    }

    /// <summary>
    /// One block over the rows [<paramref name="attnRow"/>, + <paramref name="attnRows"/>) of the sequence: <paramref name="segments"/> are the row ranges that share a modulation vector (shift | scale | gate);
    /// the projections run on the whole span, attention only inside it.
    /// </summary>
    private void Block(string prefix, (int Row, int Rows, FV Mod)[] segments, int attnRow, int attnRows)
    {
        int D = _dim, M = _mlp, cols = _cols; long r0 = attnRow, n = attnRows;
        const float Eps = 1e-6f;
        foreach (var (row, rows, mod) in segments)
            _k.LayerNormModulate(_x.View.SubView((long)row * D, (long)rows * D), _norm.View.SubView((long)row * D, (long)rows * D), mod, 0, D, rows, D, Eps);   // shift, scale
        var fused = _fused.View.SubView(0, n * cols);
        Lin(_norm.View.SubView(r0 * D, n * D), prefix + ".qkv_mlp.weight", fused, attnRows, D, cols);                              // q | k | v | gate | up
        _u.RmsNormHeads(fused, _w.Vector(prefix + ".norm_q.weight"), attnRows, cols, 0, _heads, Flux3Config.HeadDim, Eps);
        _u.RmsNormHeads(fused, _w.Vector(prefix + ".norm_k.weight"), attnRows, cols, D, _heads, Flux3Config.HeadDim, Eps);
        var cos = _cos.View.SubView(r0 * Flux3Config.HeadDim, n * Flux3Config.HeadDim); var sin = _sin.View.SubView(r0 * Flux3Config.HeadDim, n * Flux3Config.HeadDim);
        _f.RopeInterleaved(fused, cos, sin, attnRows, cols, 0, _heads, Flux3Config.HeadDim);
        _f.RopeInterleaved(fused, cos, sin, attnRows, cols, D, _heads, Flux3Config.HeadDim);
        var attn = _attn.View.SubView(0, n * D);
        _k.Attention(fused, cols, 0, fused, cols, D, 2 * D, _scores.View, attn, attnRows, attnRows, _heads, Flux3Config.HeadDim);
        var gu = _gateUp.View.SubView(0, n * (D + M));
        _f.SwigluCols(fused, cols, 3 * D, gu, D + M, D, attnRows, M);
        _f.CopyCols(attn, gu, attnRows, D, D, D + M, 0);
        Lin(gu, prefix + ".out.weight", attn, attnRows, D + M, D);                                                                   // attn_out | mlp_out
        foreach (var (row, rows, mod) in segments)
            _k.GateResidual(_x.View.SubView((long)row * D, (long)rows * D), attn.SubView((long)(row - attnRow) * D, (long)rows * D), mod, 2 * D, rows * D, D);
    }

    public void Dispose()
    {
        foreach (var o in _owned) o.Dispose();
        _owned.Clear();
    }
}
