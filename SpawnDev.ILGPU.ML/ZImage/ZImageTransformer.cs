using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux1;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.ZImage;

using FV = ArrayView1D<float, Stride1D.Dense>;
using BV = ArrayView1D<BFloat16, Stride1D.Dense>;

/// <summary>
/// Z-Image's single-stream DiT (6B: Tongyi's Z-Image / Z-Image-Turbo), ported from the reference implementation (diffusers' <c>ZImageTransformer2DModel</c>, Apache-2.0)
/// and loaded from ComfyUI's file (<c>z_image_turbo_bf16.safetensors</c>, the original names): width 3840 = 30 heads of 128, 2 noise-refiner blocks on the picture's
/// tokens and 2 context-refiner blocks (no modulation) on the caption's, then 30 blocks over [picture | caption], sandwich RMS norms, SwiGLU of 10240, modulation from a
/// 256-wide timestep embedding (scale 1 + s, gate tanh), 3-axis rotary positions (32 / 48 / 48, theta 256, interleaved pairs), sequences padded to multiples of 32 with
/// learned pad tokens. Its blocks stream from the file past a residency budget (<see cref="StreamedModelWeights"/>).
/// </summary>
public sealed class ZImageTransformer : IDisposable
{
    public const int Dim = 3840, Heads = 30, HeadDim = 128, Ffn = 10240, InChannels = 64, CapDim = 2560, AdaDim = 256, TimeChannels = 256, Layers = 30, Refiners = 2, SeqMultiple = 32;
    public const float NormEps = 1e-5f, FinalEps = 1e-6f, RopeTheta = 256f;
    public static readonly int[] RopeAxes = { 32, 48, 48 };

    private readonly Accelerator _acc;
    private readonly StreamedModelWeights _w;
    private readonly DiTKernels _k;
    private readonly SkinTokensKernels _u;
    private readonly Flux2Kernels _f;
    private readonly Flux1Kernels _f1;
    private readonly ZImageKernels _z;
    private readonly int _nImg, _nImgPad, _maxCap;
    private int _nCap, _nCapPad, _n;                                                  // this call's caption length (a prompt and its negative differ)
    private readonly List<IDisposable> _owned = new();
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _h, _norm, _qkv, _attn, _proj, _gateUp, _act, _scores, _cosT, _sinT, _out, _tproj, _thid, _c, _cAct, _mod, _modOut, _capIn, _xs;

    private static int Padded(int n) => n + (-n % SeqMultiple + SeqMultiple) % SeqMultiple;

    private ZImageTransformer(Accelerator acc, StreamedModelWeights w, int imageTokens, int maxCaptionTokens)
    {
        _acc = acc; _w = w;
        _k = new DiTKernels(acc) { PreciseActivations = true }; _u = new SkinTokensKernels(acc); _f = new Flux2Kernels(acc); _f1 = new Flux1Kernels(acc); _z = new ZImageKernels(acc);
        _nImg = imageTokens; _nImgPad = Padded(imageTokens); _maxCap = maxCaptionTokens;
        _nCap = maxCaptionTokens; _nCapPad = Padded(maxCaptionTokens); _n = _nImgPad + _nCapPad;
        long N = _n;                                                                        // every buffer holds the longest caption
        _h = Alloc(N * Dim); _norm = Alloc(N * Dim); _qkv = Alloc(N * 3 * Dim); _attn = Alloc(N * Dim); _proj = Alloc(N * Dim);
        _gateUp = Alloc(N * 2 * Ffn); _act = Alloc(N * Ffn);
        _scores = Alloc(_k.UsesFlashAttention(HeadDim) ? 1 : (long)Heads * N * N);
        _cosT = Alloc(N * HeadDim); _sinT = Alloc(N * HeadDim); _out = Alloc(N * InChannels);
        _tproj = Alloc(TimeChannels); _thid = Alloc(1024); _c = Alloc(AdaDim); _cAct = Alloc(AdaDim); _mod = Alloc(4 * Dim); _modOut = Alloc(2 * Dim);
        _capIn = Alloc((long)_nCapPad * CapDim);
        _xs = Alloc(w.HasScales ? N * Ffn : 1);                                                // the widest scaled input: w2's [rows, 10240]
        SetPositions();
    }

    private MemoryBuffer1D<float, Stride1D.Dense> Alloc(long n) { var b = _acc.Allocate1D<float>(Math.Max(1, n)); _owned.Add(b); return b; }

    /// <summary>Block 0..1 = noise_refiner, 2..3 = context_refiner, 4..33 = layers.</summary>
    private static int BlockOf(string n)
    {
        static int Idx(string n, int at) => int.Parse(n.AsSpan(at, n.IndexOf('.', at) - at));
        if (n.StartsWith("noise_refiner.", StringComparison.Ordinal)) return Idx(n, 14);
        if (n.StartsWith("context_refiner.", StringComparison.Ordinal)) return Refiners + Idx(n, 16);
        if (n.StartsWith("layers.", StringComparison.Ordinal)) return 2 * Refiners + Idx(n, 7);
        return -1;
    }
    private static string BlockPrefix(int k) => k < Refiners ? $"noise_refiner.{k}." : k < 2 * Refiners ? $"context_refiner.{k - Refiners}." : $"layers.{k - 2 * Refiners}.";

    /// <summary>w1 | w3 of every feed-forward fused into one gate | up matrix (one GEMM, then SwiGLU), as they load.</summary>
    private static async Task FuseFfnAsync(Accelerator acc, GpuWeights w)
    {
        foreach (var n in w.Names.Where(n => n.EndsWith(".feed_forward.w1.weight", StringComparison.Ordinal)).ToList())
        {
            string b = n[..^"w1.weight".Length], n3 = b + "w3.weight";
            if (!w.IsBF16(n) || !w.IsBF16(n3)) throw new NotSupportedException($"{b}: expected bf16 w1 / w3");
            var s1 = w.Shape(n); long c1 = (long)s1[0] * s1[1];
            var dst = acc.Allocate1D<BFloat16>(2 * c1);
            dst.View.SubView(0, c1).CopyFrom(w.BF16(n)); dst.View.SubView(c1, c1).CopyFrom(w.BF16(n3));
            await acc.SynchronizeAsync();
            w.Remove(n)?.Dispose(); w.Remove(n3)?.Dispose();
            w.SetBF16(b + "w13.weight", dst, new[] { 2 * s1[0], s1[1] });
        }
    }

    /// <summary>
    /// Open the transformer from <paramref name="file"/> (ComfyUI's <c>z_image_turbo_bf16.safetensors</c>) for a picture of <paramref name="rows"/> x <paramref name="cols"/> tokens (latent / 2)
    /// and up to <paramref name="maxCaptionTokens"/> caption rows (each call passes its own count: a prompt and its negative may differ); blocks beyond <paramref name="residentBytes"/>
    /// stream from the file every call.
    /// </summary>
    public static async Task<ZImageTransformer> OpenAsync(Accelerator accelerator, IModelSource source, string repo, string file, int rows, int cols, int maxCaptionTokens, long residentBytes, CancellationToken ct = default)
    {
        var w = await StreamedModelWeights.OpenAsync(accelerator, source, repo, file, "layers.0.attention.qkv.weight", BlockOf, 2 * Refiners + Layers, residentBytes, w2 => FuseFfnAsync(accelerator, w2), ct);
        try { var t = new ZImageTransformer(accelerator, w, rows * cols, maxCaptionTokens) { _rows = rows, _cols = cols }; t.SetPositions(); return t; }
        catch { w.Dispose(); throw; }
    }
    private int _rows, _cols;

    public long ResidentBytes => _w.ResidentBytes;
    public int ResidentBlocks => _w.ResidentBlocks;
    public long StreamedBytesPerForward => _w.StreamedBytesPerForward;
    /// <summary>
    /// CUDA: fp32-class activations through the tensor cores (<see cref="DiTKernels.PreciseActivations"/>: split hi + lo bf16 GEMMs, fp32 flash attention). ON by default: this port is
    /// checked against ComfyUI's fp32 runs, and bf16 activations cost ~5e-3 per call (FLUX.1 4.7e-3 -> 3.3e-5, Z-Image 5.6e-3 -> 3.2e-5, Qwen-Image 4.3e-2 -> 2.4e-4 measured on an
    /// RTX 4070; no slower at 256 px). Other backends compute in fp32 either way.
    /// </summary>
    public bool PreciseActivations { get => _k.PreciseActivations; set => _k.PreciseActivations = value; }

    /// <summary>The longest caption this transformer was opened for.</summary>
    public int MaxCaptionTokens => _maxCap;
    /// <summary>The last output [paddedImage + paddedCaption, 64]; the picture's tokens are the first rows.</summary>
    public FV Output => _out.View;

    /// <summary>Rotary tables for the unified [picture | caption] rows: picture (capPadded + 1, h, w), its pad rows (0, 0, 0); caption (1 + i, 0, 0).</summary>
    private void SetPositions()
    {
        if (_rows == 0) return;
        var ids = new int[_n * 3];
        for (int i = 0; i < _nImg; i++) { ids[i * 3] = _nCapPad + 1; ids[i * 3 + 1] = i / _cols; ids[i * 3 + 2] = i % _cols; }
        for (int i = 0; i < _nCapPad; i++) ids[(_nImgPad + i) * 3] = 1 + i;
        var cos = new float[_n * HeadDim]; var sin = new float[_n * HeadDim];
        for (int r = 0; r < _n; r++)
        {
            int o = 0;
            for (int a = 0; a < 3; a++)
                for (int j = 0; j < RopeAxes[a] / 2; j++)
                {
                    double ang = ids[r * 3 + a] / Math.Pow(RopeTheta, 2.0 * j / RopeAxes[a]);
                    float c = (float)Math.Cos(ang), s = (float)Math.Sin(ang);
                    cos[r * HeadDim + o] = c; cos[r * HeadDim + o + 1] = c; sin[r * HeadDim + o] = s; sin[r * HeadDim + o + 1] = s;
                    o += 2;
                }
        }
        _cosT.View.SubView(0, cos.Length).CopyFromCPU(cos); _sinT.View.SubView(0, sin.Length).CopyFromCPU(sin);
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
    private void Lin(GpuWeights w, FV x, string name, FV o, int rows, int K, int N, bool bias)
        => _k.Linear(Scaled(w, name + ".weight", x, (long)rows * K), _w.MatrixExact(w, name + ".weight"), bias ? w.BF16(name + ".bias") : null, o, rows, K, N);
    private void Gemv(GpuWeights w, FV x, string name, FV o, int K, int N) => _u.Gemv(Scaled(w, name + ".weight", x, K), _w.MatrixExact(w, name + ".weight"), w.BF16(name + ".bias"), o, K, N);

    /// <summary>
    /// One model call: <paramref name="latent"/> the model-space latent [16, 2 rows, 2 cols], <paramref name="caption"/> the text encoder's rows [captionTokens, 2560], <paramref name="modelTime"/>
    /// the reference's model time (1 - sigma). <paramref name="captionTokens"/>: this call's caption rows, 1 to <see cref="MaxCaptionTokens"/> (-1: the maximum); the weights and buffers are
    /// shared, only the rotary table is rebuilt when the length changes. The output tokens are left in <see cref="Output"/> (<see cref="ZImageKernels.Denoised"/> makes the denoised estimate).
    /// </summary>
    public async Task ForwardAsync(FV latent, FV caption, float modelTime, int captionTokens = -1, CancellationToken ct = default)
    {
        if (captionTokens < 0) captionTokens = _maxCap;
        if (captionTokens < 1 || captionTokens > _maxCap) throw new ArgumentOutOfRangeException(nameof(captionTokens), $"1 to {_maxCap} (the transformer's MaxCaptionTokens)");
        if (captionTokens != _nCap) { _nCap = captionTokens; _nCapPad = Padded(captionTokens); _n = _nImgPad + _nCapPad; SetPositions(); }
        var b0 = _w.Base;
        // timestep embedding: sinusoid(1000 t) -> 1024 -> silu -> 256
        var e = new float[TimeChannels];
        for (int i = 0; i < TimeChannels / 2; i++) { double a = 1000.0 * modelTime * Math.Exp(-Math.Log(10000.0) * i / (TimeChannels / 2)); e[i] = (float)Math.Cos(a); e[TimeChannels / 2 + i] = (float)Math.Sin(a); }
        _tproj.View.CopyFromCPU(e);
        Gemv(b0, _tproj.View, "t_embedder.mlp.0", _thid.View, TimeChannels, 1024); _k.Silu(_thid.View, _thid.View, 1024);
        Gemv(b0, _thid.View, "t_embedder.mlp.2", _c.View, 1024, AdaDim);
        // picture tokens: patchify (pad rows repeat the last token), embed, pad rows := x_pad_token
        var img = _h.View.SubView(0, (long)_nImgPad * Dim);
        _z.Patchify(latent, _proj.View, _rows, _cols, _nImgPad);
        Lin(b0, _proj.View, "x_embedder", img, _nImgPad, InChannels, Dim, true);
        _z.FillRows(_h.View, b0.BF16("x_pad_token"), _nImg, _nImgPad - _nImg, Dim);
        // caption: pad by repeating the last row, RMS norm + linear, pad rows := cap_pad_token
        _capIn.View.SubView(0, (long)_nCap * CapDim).CopyFrom(caption.SubView(0, (long)_nCap * CapDim));
        for (int r = _nCap; r < _nCapPad; r++) _capIn.View.SubView((long)r * CapDim, CapDim).CopyFrom(caption.SubView((long)(_nCap - 1) * CapDim, CapDim));
        _u.RmsNorm(_capIn.View, _capIn.View, b0.BF16("cap_embedder.0.weight"), _nCapPad, CapDim, NormEps);
        var cap = _h.View.SubView((long)_nImgPad * Dim, (long)_nCapPad * Dim);
        Lin(b0, _capIn.View, "cap_embedder.1", cap, _nCapPad, CapDim, Dim, true);
        _z.FillRows(_h.View, b0.BF16("cap_pad_token"), _nImgPad + _nCap, _nCapPad - _nCap, Dim);
        // refiners: the picture's (modulated) and the caption's (not), each over its own rows
        for (int k = 0; k < Refiners; k++) await RunBlockAsync(k, 0, _nImgPad, true, ct);
        for (int k = 0; k < Refiners; k++) await RunBlockAsync(Refiners + k, _nImgPad, _nCapPad, false, ct);
        // the main stack over [picture | caption]
        for (int k = 0; k < Layers; k++) await RunBlockAsync(2 * Refiners + k, 0, _n, true, ct);
        // final layer: LN (no affine) * (1 + adaLN(silu(c))) -> linear
        _k.Silu(_c.View, _cAct.View, AdaDim);
        _modOut.View.SubView(0, Dim).MemSetToZero();                                        // no shift
        Gemv(b0, _cAct.View, "final_layer.adaLN_modulation.1", _modOut.View.SubView(Dim, Dim), AdaDim, Dim);
        _k.LayerNormModulate(_h.View, _norm.View, _modOut.View, 0, Dim, _n, Dim, FinalEps);
        Lin(b0, _norm.View, "final_layer.linear", _out.View, _n, Dim, InChannels, true);
    }

    private async Task RunBlockAsync(int k, int row0, int rows, bool modulated, CancellationToken ct)
    {
        var (w, streamed) = await _w.BlockAsync(k, ct);
        try { Block(w, BlockPrefix(k), row0, rows, modulated); }
        finally { if (streamed) { await _acc.SynchronizeAsync(); w.Dispose(); } }
    }

    private void Block(GpuWeights w, string b, int row0, int rows, bool modulated)
    {
        long o = (long)row0 * Dim;
        var x = _h.View.SubView(o, (long)rows * Dim);
        var n = _norm.View.SubView(0, (long)rows * Dim);
        var qkv = _qkv.View.SubView(0, (long)rows * 3 * Dim);
        var attn = _attn.View.SubView(0, (long)rows * Dim);
        var proj = _proj.View.SubView(0, (long)rows * Dim);
        var cos = _cosT.View.SubView((long)row0 * HeadDim, (long)rows * HeadDim); var sin = _sinT.View.SubView((long)row0 * HeadDim, (long)rows * HeadDim);
        if (modulated) { Gemv(w, _c.View, b + "adaLN_modulation.0", _mod.View, AdaDim, 4 * Dim); _z.AdjustMod(_mod.View, Dim); }    // (1 + scale_msa, tanh gate_msa, 1 + scale_mlp, tanh gate_mlp)
        // attention
        _u.RmsNorm(x, n, w.BF16(b + "attention_norm1.weight"), rows, Dim, NormEps);
        if (modulated) _z.ScaleCols(n, _mod.View, 0, rows, Dim);
        Lin(w, n, b + "attention.qkv", qkv, rows, Dim, 3 * Dim, false);
        _u.RmsNormHeads(qkv, w.BF16(b + "attention.q_norm.weight"), rows, 3 * Dim, 0, Heads, HeadDim, NormEps);
        _u.RmsNormHeads(qkv, w.BF16(b + "attention.k_norm.weight"), rows, 3 * Dim, Dim, Heads, HeadDim, NormEps);
        _f.RopeInterleaved(qkv, cos, sin, rows, 3 * Dim, 0, Heads, HeadDim);
        _f.RopeInterleaved(qkv, cos, sin, rows, 3 * Dim, Dim, Heads, HeadDim);
        _k.Attention(qkv, 3 * Dim, 0, qkv, 3 * Dim, Dim, 2 * Dim, _scores.View, attn, rows, rows, Heads, HeadDim);
        Lin(w, attn, b + "attention.out", proj, rows, Dim, Dim, false);
        _u.RmsNorm(proj, proj, w.BF16(b + "attention_norm2.weight"), rows, Dim, NormEps);
        if (modulated) _k.GateResidual(x, proj, _mod.View, Dim, rows * Dim, Dim); else _f1.AddInto(x, proj, rows * Dim);
        // feed-forward: w2(silu(w1 h) * w3 h) with h = ffn_norm1(x) * scale_mlp
        _u.RmsNorm(x, n, w.BF16(b + "ffn_norm1.weight"), rows, Dim, NormEps);
        if (modulated) _z.ScaleCols(n, _mod.View, 2 * Dim, rows, Dim);
        var gu = _gateUp.View.SubView(0, (long)rows * 2 * Ffn); var act = _act.View.SubView(0, (long)rows * Ffn);
        Lin(w, n, b + "feed_forward.w13", gu, rows, Dim, 2 * Ffn, false);
        _u.Swiglu(gu, act, rows, Ffn);
        Lin(w, act, b + "feed_forward.w2", proj, rows, Ffn, Dim, false);
        _u.RmsNorm(proj, proj, w.BF16(b + "ffn_norm2.weight"), rows, Dim, NormEps);
        if (modulated) _k.GateResidual(x, proj, _mod.View, 3 * Dim, rows * Dim, Dim); else _f1.AddInto(x, proj, rows * Dim);
    }

    public void Dispose()
    {
        foreach (var b in _owned) b.Dispose();
        _owned.Clear();
        _w.Dispose();
    }
}
