using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Flux1;

using FV = ArrayView1D<float, Stride1D.Dense>;
using BV = ArrayView1D<BFloat16, Stride1D.Dense>;

/// <summary>
/// FLUX.1's flow transformer (Black Forest Labs' 12B MMDiT: dev and schnell): width 3072 = 24 heads of 128, 19 double-stream blocks then 38 single-stream blocks, adaLN modulation PER
/// BLOCK from one conditioning vector, biases on every linear, GELU (tanh) MLPs of 12288, 3-axis rotary positions (16 / 56 / 56 dims, theta 10000, interleaved pairs) over
/// (0, row, column) for the picture and zeros for the text, fp32 activations.
/// <code>
/// vec = time_in(sinusoid(1000 sigma)) + guidance_in(sinusoid(1000 guidance)) [dev] + vector_in(pooled CLIP-L)      each an MLP: linear, silu, linear
/// img = img_in(2 x 2 patches of the latent, 64 wide),  txt = txt_in(T5 [rows, 4096])
/// 19 x double { (shift, scale, gate) x 2 = img_mod / txt_mod(silu(vec)); joint attention of [txt | img] (qkv + bias, RMS q / k norms, rope); img += gate * proj; img += gate * mlp }
/// h = [txt | img];  38 x single { (shift, scale, gate) = modulation(silu(vec)); (q, k, v | mlp) = linear1(adaln(h)); h += gate * linear2([attention | gelu_tanh(mlp)]) }
/// velocity = final_layer.linear(LN(h[img]) * (1 + scale) + shift),  (shift, scale) = final_layer.adaLN_modulation(silu(vec))
/// </code>
/// ComfyUI calls it with the latent in MODEL space ((raw - 0.1159) * 0.3611), timestep = sigma, and turns the velocity into the denoised estimate x - sigma * v
/// (<see cref="Flux1Kernels.Denoised"/>): measured by SpawnGen on ComfyUI's FLUX.1 dev (its dumps gate this class).
/// </summary>
public sealed class Flux1Transformer : IDisposable
{
    public const int Dim = 3072, Heads = 24, HeadDim = 128, MlpHidden = 12288, InChannels = 64, TxtDim = 4096, VecDim = 768, TimeChannels = 256;
    public const float Eps = 1e-6f, RopeTheta = 10000f;
    public static readonly int[] RopeAxes = { 16, 56, 56 };

    private readonly Accelerator _acc;
    private readonly Flux1Weights _w;
    private readonly DiTKernels _k;
    private readonly SkinTokensKernels _u;
    private readonly Flux2Kernels _f;
    private readonly Flux1Kernels _f1;
    private readonly int _nTxt, _nImg, _n;
    private readonly List<IDisposable> _owned = new();
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _img, _txt, _normI, _normT, _qkv, _attn, _attnImg, _attnTxt, _proj, _mlp, _cat, _scores;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _vec, _vecAct, _tproj, _hid, _part, _modImg, _modTxt, _modSingle, _modOut, _cosT, _sinT, _out, _xs;

    public Flux1Transformer(Accelerator accelerator, Flux1Weights weights, int textTokens, int imageTokens)
    {
        _acc = accelerator; _w = weights; _nTxt = textTokens; _nImg = imageTokens; _n = textTokens + imageTokens;
        _k = new DiTKernels(accelerator) { PreciseActivations = true }; _u = new SkinTokensKernels(accelerator); _f = new Flux2Kernels(accelerator); _f1 = new Flux1Kernels(accelerator);
        long N = _n, I = _nImg, T = _nTxt;
        _img = Alloc(I * Dim); _txt = Alloc(T * Dim); _normI = Alloc(I * Dim); _normT = Alloc(T * Dim);
        _qkv = Alloc(N * 3 * Dim); _attn = Alloc(N * Dim); _attnImg = Alloc(I * Dim); _attnTxt = Alloc(T * Dim);
        _proj = Alloc(N * Dim); _mlp = Alloc(N * MlpHidden); _cat = Alloc(N * (Dim + MlpHidden));
        _scores = Alloc(_k.UsesFlashAttention(HeadDim) ? 1 : (long)Heads * N * N);
        _vec = Alloc(Dim); _vecAct = Alloc(Dim); _tproj = Alloc(TimeChannels); _hid = Alloc(Dim); _part = Alloc(Dim);
        _modImg = Alloc(6 * Dim); _modTxt = Alloc(6 * Dim); _modSingle = Alloc(3 * Dim); _modOut = Alloc(2 * Dim);
        _cosT = Alloc(N * HeadDim); _sinT = Alloc(N * HeadDim); _out = Alloc(I * InChannels);
        _xs = Alloc(weights.HasScales ? N * (Dim + MlpHidden) : 1);                     // the widest scaled input: linear2's [rows, 3072 + 12288]
    }

    private MemoryBuffer1D<float, Stride1D.Dense> Alloc(long n) { var b = _acc.Allocate1D<float>(Math.Max(1, n)); _owned.Add(b); return b; }

    /// <summary>
    /// CUDA: fp32-class activations through the tensor cores (<see cref="DiTKernels.PreciseActivations"/>: split hi + lo bf16 GEMMs, fp32 flash attention). ON by default: this port is
    /// checked against ComfyUI's fp32 runs, and bf16 activations cost ~5e-3 per call (FLUX.1 4.7e-3 -> 3.3e-5, Z-Image 5.6e-3 -> 3.2e-5, Qwen-Image 4.3e-2 -> 2.4e-4 measured on an
    /// RTX 4070; no slower at 256 px). Other backends compute in fp32 either way.
    /// </summary>
    public bool PreciseActivations { get => _k.PreciseActivations; set => _k.PreciseActivations = value; }

    /// <summary>The velocity of the last <see cref="ForwardAsync"/>, [imageTokens, 64] (2 x 2 patches of the 16-channel latent).</summary>
    public FV Output => _out.View;

    /// <summary>Intermediate checkpoints for tests: name, device view, rows, columns.</summary>
    public Action<string, FV, int, int>? Probe;

    /// <summary>Rotary tables for [rows, 3] ids: cos and sin [rows, 128], each axis' frequencies repeated for the interleaved pairs (double precision on the host, as FLUX.2's).</summary>
    public static (float[] Cos, float[] Sin) RopeTables(int[] ids, int rows)
    {
        var cos = new float[rows * HeadDim]; var sin = new float[rows * HeadDim];
        for (int r = 0; r < rows; r++)
        {
            int o = 0;
            for (int a = 0; a < RopeAxes.Length; a++)
            {
                int dim = RopeAxes[a];
                for (int j = 0; j < dim / 2; j++)
                {
                    double ang = ids[r * 3 + a] / Math.Pow(RopeTheta, 2.0 * j / dim);
                    float c = (float)Math.Cos(ang), s = (float)Math.Sin(ang);
                    cos[r * HeadDim + o] = c; cos[r * HeadDim + o + 1] = c;
                    sin[r * HeadDim + o] = s; sin[r * HeadDim + o + 1] = s;
                    o += 2;
                }
            }
        }
        return (cos, sin);
    }

    /// <summary>The position ids of a picture of rows x cols tokens after textTokens text rows: text (0, 0, 0), picture (0, row, column); [textTokens + rows * cols, 3].</summary>
    public static int[] Ids(int textTokens, int rows, int cols)
    {
        var ids = new int[(textTokens + rows * cols) * 3];
        for (int y = 0; y < rows; y++) for (int x = 0; x < cols; x++) { int r = textTokens + y * cols + x; ids[r * 3 + 1] = y; ids[r * 3 + 2] = x; }
        return ids;
    }

    /// <summary>Set the rotary tables for this run (a function of the ids alone: once per run, not per step).</summary>
    public void SetPositions(int rows, int cols)
    {
        if (rows * cols != _nImg) throw new ArgumentException($"{rows} x {cols} tokens, the transformer was built for {_nImg}");
        var (cos, sin) = RopeTables(Ids(_nTxt, rows, cols), _n);
        _cosT.View.SubView(0, cos.Length).CopyFromCPU(cos); _sinT.View.SubView(0, sin.Length).CopyFromCPU(sin);
    }

    private static float[] Sinusoid(float value)
    {
        // Black Forest Labs' timestep_embedding(1000 * value, 256): [cos | sin] over 128 frequencies exp(-ln(10000) i / 128)
        var e = new float[TimeChannels];
        double v = 1000.0 * value;
        for (int i = 0; i < TimeChannels / 2; i++)
        {
            double a = v * Math.Exp(-Math.Log(10000.0) * i / (TimeChannels / 2));
            e[i] = (float)Math.Cos(a); e[TimeChannels / 2 + i] = (float)Math.Sin(a);
        }
        return e;
    }


    // an fp8_scaled file: the matrix runs on its exact unscaled decode with its input times the scale (StreamedModelWeights.MatrixExact); _xs holds the scaled input
    private FV Scaled(GpuWeights w, string weight, FV x, long n)
    {
        float s = _w.ActivationScale(w, weight);
        if (s == 1f) return x;
        var xs = _xs.View.SubView(0, n);
        _f.ScaleInto(x, xs, s, (int)n);
        return xs;
    }
    private void Gemv(GpuWeights w, FV x, string name, FV o, int K, int N)
    {
        _u.Gemv(Scaled(w, name + ".weight", x, K), _w.MatrixExact(w, name + ".weight"), w.BF16(name + ".bias"), o, K, N);
        if (w.Has(name + ".bias.lo")) _f1.AddRowVector(o, w.BF16(name + ".bias.lo"), 1, N, 0);              // an fp16 bias held exactly (hi + lo)
    }
    private void Lin(GpuWeights w, FV x, string name, FV o, int rows, int K, int N, DiTKernels.Act act = DiTKernels.Act.None)
    {
        bool lo = w.Has(name + ".bias.lo");                                                  // an fp16 bias held exactly: the remainder goes in before the activation
        _k.Linear(Scaled(w, name + ".weight", x, (long)rows * K), _w.MatrixExact(w, name + ".weight"), w.BF16(name + ".bias"), o, rows, K, N, lo ? DiTKernels.Act.None : act);
        if (lo) _f1.AddRowVector(o, w.BF16(name + ".bias.lo"), rows, N, (int)act);
    }
    // q / k RMS norm per head; an fp16 scale held exactly (hi + lo) when the file had one
    private void NormHeads(GpuWeights w, FV buf, string scale, int rows, int stride, int off)
    {
        if (w.Has(scale + ".lo")) _u.RmsNormHeads(buf, w.BF16(scale), w.BF16(scale + ".lo"), rows, stride, off, Heads, HeadDim, Eps);
        else _u.RmsNormHeads(buf, w.BF16(scale), rows, stride, off, Heads, HeadDim, Eps);
    }

    /// <summary>
    /// One model call: <paramref name="imageTokens"/> [imageTokens, 64] (the model-space latent's 2 x 2 patches: <see cref="Flux1Kernels.Patchify"/>), <paramref name="txt"/> the T5 output
    /// [textTokens, 4096], <paramref name="pooled"/> the pooled CLIP-L vector [768], <paramref name="sigma"/> the timestep as ComfyUI passes it, <paramref name="guidance"/> FluxGuidance's
    /// value (dev; ignored when the model has no guidance embedder). The velocity is left in <see cref="Output"/>. Streamed blocks are read from the file as the pass reaches them.
    /// </summary>
    public async Task ForwardAsync(FV imageTokens, FV txt, FV pooled, float sigma, float guidance, CancellationToken ct = default)
    {
        var b0 = _w.Base;
        // the conditioning vector
        _tproj.View.CopyFromCPU(Sinusoid(sigma));
        Gemv(b0, _tproj.View, "time_in.in_layer", _hid.View, TimeChannels, Dim); _k.Silu(_hid.View, _vecAct.View, Dim);
        Gemv(b0, _vecAct.View, "time_in.out_layer", _vec.View, Dim, Dim);
        if (_w.HasGuidance)
        {
            _tproj.View.CopyFromCPU(Sinusoid(guidance));
            Gemv(b0, _tproj.View, "guidance_in.in_layer", _hid.View, TimeChannels, Dim); _k.Silu(_hid.View, _vecAct.View, Dim);
            Gemv(b0, _vecAct.View, "guidance_in.out_layer", _part.View, Dim, Dim);
            _f1.AddInto(_vec.View, _part.View, Dim);
        }
        Gemv(b0, pooled, "vector_in.in_layer", _hid.View, VecDim, Dim); _k.Silu(_hid.View, _vecAct.View, Dim);
        Gemv(b0, _vecAct.View, "vector_in.out_layer", _part.View, Dim, Dim);
        _f1.AddInto(_vec.View, _part.View, Dim);
        Probe?.Invoke("vec", _vec.View, 1, Dim);
        _k.Silu(_vec.View, _vecAct.View, Dim);                                       // every modulation reads silu(vec)

        Lin(b0, imageTokens, "img_in", _img.View, _nImg, InChannels, Dim);
        Lin(b0, txt, "txt_in", _txt.View, _nTxt, TxtDim, Dim);

        for (int k = 0; k < Flux1Weights.DoubleBlocks; k++)
        {
            var (w, streamed) = await _w.BlockAsync(k, ct);
            try { DoubleBlock(w, Flux1Weights.BlockPrefix(k)); }
            finally { if (streamed) { await _acc.SynchronizeAsync(); w.Dispose(); } }
            Probe?.Invoke($"double{k}_img", _img.View, _nImg, Dim);
        }
        _proj.View.SubView(0, (long)_nTxt * Dim).CopyFrom(_txt.View.SubView(0, (long)_nTxt * Dim));
        _proj.View.SubView((long)_nTxt * Dim, (long)_nImg * Dim).CopyFrom(_img.View.SubView(0, (long)_nImg * Dim));
        for (int k = 0; k < Flux1Weights.SingleBlocks; k++)
        {
            var (w, streamed) = await _w.BlockAsync(Flux1Weights.DoubleBlocks + k, ct);
            try { SingleBlock(w, Flux1Weights.BlockPrefix(Flux1Weights.DoubleBlocks + k)); }
            finally { if (streamed) { await _acc.SynchronizeAsync(); w.Dispose(); } }
        }
        Probe?.Invoke("single_out", _proj.View, _n, Dim);
        Gemv(b0, _vecAct.View, "final_layer.adaLN_modulation.1", _modOut.View, Dim, 2 * Dim);       // (shift, scale)
        var imgRows = _proj.View.SubView((long)_nTxt * Dim, (long)_nImg * Dim);
        _k.LayerNormModulate(imgRows, _normI.View, _modOut.View, 0, Dim, _nImg, Dim, Eps);
        Lin(b0, _normI.View, "final_layer.linear", _out.View, _nImg, Dim, InChannels);
    }

    private void DoubleBlock(GpuWeights w, string b)
    {
        // (shift, scale, gate) for attention, then for the MLP, per stream
        Gemv(w, _vecAct.View, b + "img_mod.lin", _modImg.View, Dim, 6 * Dim);
        Gemv(w, _vecAct.View, b + "txt_mod.lin", _modTxt.View, Dim, 6 * Dim);
        _k.LayerNormModulate(_img.View, _normI.View, _modImg.View, 0, Dim, _nImg, Dim, Eps);
        _k.LayerNormModulate(_txt.View, _normT.View, _modTxt.View, 0, Dim, _nTxt, Dim, Eps);
        var qkvT = _qkv.View.SubView(0, (long)_nTxt * 3 * Dim); var qkvI = _qkv.View.SubView((long)_nTxt * 3 * Dim, (long)_nImg * 3 * Dim);
        Lin(w, _normT.View, b + "txt_attn.qkv", qkvT, _nTxt, Dim, 3 * Dim);
        Lin(w, _normI.View, b + "img_attn.qkv", qkvI, _nImg, Dim, 3 * Dim);
        NormHeads(w, qkvT, b + "txt_attn.norm.query_norm.scale", _nTxt, 3 * Dim, 0);
        NormHeads(w, qkvT, b + "txt_attn.norm.key_norm.scale", _nTxt, 3 * Dim, Dim);
        NormHeads(w, qkvI, b + "img_attn.norm.query_norm.scale", _nImg, 3 * Dim, 0);
        NormHeads(w, qkvI, b + "img_attn.norm.key_norm.scale", _nImg, 3 * Dim, Dim);
        _f.RopeInterleaved(_qkv.View, _cosT.View, _sinT.View, _n, 3 * Dim, 0, Heads, HeadDim);
        _f.RopeInterleaved(_qkv.View, _cosT.View, _sinT.View, _n, 3 * Dim, Dim, Heads, HeadDim);
        _k.Attention(_qkv.View, 3 * Dim, 0, _qkv.View, 3 * Dim, Dim, 2 * Dim, _scores.View, _attn.View, _n, _n, Heads, HeadDim);
        Lin(w, _attn.View.SubView(0, (long)_nTxt * Dim), b + "txt_attn.proj", _attnTxt.View, _nTxt, Dim, Dim);
        Lin(w, _attn.View.SubView((long)_nTxt * Dim, (long)_nImg * Dim), b + "img_attn.proj", _attnImg.View, _nImg, Dim, Dim);
        _k.GateResidual(_img.View, _attnImg.View, _modImg.View, 2 * Dim, _nImg * Dim, Dim);
        _k.GateResidual(_txt.View, _attnTxt.View, _modTxt.View, 2 * Dim, _nTxt * Dim, Dim);
        // MLPs: linear (+ bias, GELU tanh), linear (+ bias)
        _k.LayerNormModulate(_img.View, _normI.View, _modImg.View, 3 * Dim, 4 * Dim, _nImg, Dim, Eps);
        Lin(w, _normI.View, b + "img_mlp.0", _mlp.View, _nImg, Dim, MlpHidden, DiTKernels.Act.GeluTanh);
        Lin(w, _mlp.View, b + "img_mlp.2", _attnImg.View, _nImg, MlpHidden, Dim);
        _k.GateResidual(_img.View, _attnImg.View, _modImg.View, 5 * Dim, _nImg * Dim, Dim);
        _k.LayerNormModulate(_txt.View, _normT.View, _modTxt.View, 3 * Dim, 4 * Dim, _nTxt, Dim, Eps);
        Lin(w, _normT.View, b + "txt_mlp.0", _mlp.View, _nTxt, Dim, MlpHidden, DiTKernels.Act.GeluTanh);
        Lin(w, _mlp.View, b + "txt_mlp.2", _attnTxt.View, _nTxt, MlpHidden, Dim);
        _k.GateResidual(_txt.View, _attnTxt.View, _modTxt.View, 5 * Dim, _nTxt * Dim, Dim);
    }

    private void SingleBlock(GpuWeights w, string b)
    {
        var h = _proj.View;
        Gemv(w, _vecAct.View, b + "modulation.lin", _modSingle.View, Dim, 3 * Dim);          // (shift, scale, gate)
        _k.LayerNormModulate(h, _attn.View, _modSingle.View, 0, Dim, _n, Dim, Eps);
        // linear1 = [q | k | v | mlp] rows of one matrix: two GEMMs on its row ranges (the MLP part with GELU tanh straight into its own buffer)
        var x1 = Scaled(w, b + "linear1.weight", _attn.View, (long)_n * Dim);               // one matrix, one scale: both row ranges read the same scaled input
        var w1 = _w.MatrixExact(w, b + "linear1.weight"); var b1 = w.BF16(b + "linear1.bias");
        bool lo1 = w.Has(b + "linear1.bias.lo");
        _k.Linear(x1, w1.SubView(0, 3L * Dim * Dim), b1.SubView(0, 3 * Dim), _qkv.View, _n, Dim, 3 * Dim);
        _k.Linear(x1, w1.SubView(3L * Dim * Dim, (long)MlpHidden * Dim), b1.SubView(3 * Dim, MlpHidden), _mlp.View, _n, Dim, MlpHidden, lo1 ? DiTKernels.Act.None : DiTKernels.Act.GeluTanh);
        if (lo1)
        {
            var b1lo = w.BF16(b + "linear1.bias.lo");
            _f1.AddRowVector(_qkv.View, b1lo.SubView(0, 3 * Dim), _n, 3 * Dim, 0);
            _f1.AddRowVector(_mlp.View, b1lo.SubView(3 * Dim, MlpHidden), _n, MlpHidden, (int)DiTKernels.Act.GeluTanh);
        }
        NormHeads(w, _qkv.View, b + "norm.query_norm.scale", _n, 3 * Dim, 0);
        NormHeads(w, _qkv.View, b + "norm.key_norm.scale", _n, 3 * Dim, Dim);
        _f.RopeInterleaved(_qkv.View, _cosT.View, _sinT.View, _n, 3 * Dim, 0, Heads, HeadDim);
        _f.RopeInterleaved(_qkv.View, _cosT.View, _sinT.View, _n, 3 * Dim, Dim, Heads, HeadDim);
        _k.Attention(_qkv.View, 3 * Dim, 0, _qkv.View, 3 * Dim, Dim, 2 * Dim, _scores.View, _attn.View, _n, _n, Heads, HeadDim);
        // linear2 over [attention | mlp]
        _f.CopyCols(_attn.View, _cat.View, _n, Dim, Dim, Dim + MlpHidden, 0);
        _f.CopyCols(_mlp.View, _cat.View, _n, MlpHidden, MlpHidden, Dim + MlpHidden, Dim);
        Lin(w, _cat.View, b + "linear2", _attn.View, _n, Dim + MlpHidden, Dim);        // _attn is free again (copied into _cat)
        _k.GateResidual(h, _attn.View, _modSingle.View, 2 * Dim, _n * Dim, Dim);
    }

    public void Dispose()
    {
        foreach (var o in _owned) o.Dispose();
        _owned.Clear();
    }
}
