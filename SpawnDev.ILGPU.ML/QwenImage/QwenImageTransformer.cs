using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux1;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.QwenImage;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>
/// Qwen-Image's flow transformer (20B MMDiT), ported from the reference implementation (diffusers' <c>QwenImageTransformer2DModel</c>, Apache-2.0) and loaded from ComfyUI's
/// file (<c>qwen_image_fp8_e4m3fn.safetensors</c>: diffusers names, every tensor 8-bit floats without scales). Width 3072 = 24 heads of 128, 60 double-stream blocks (FLUX.1's
/// double block: per-stream (shift, scale, gate) x 2 modulation, joint attention of [text | picture] with RMS q / k norms and interleaved rotary pairs, GELU (tanh) MLPs of 12288),
/// no single-stream blocks, conditioned on the timestep alone.
/// <code>
/// temb = linear_2(silu(linear_1(sinusoid(1000 t))))                    sinusoid: [cos | sin], 128 frequencies exp(-ln(10000) i / 128)
/// img = img_in(2 x 2 patches, 64 wide),  txt = txt_in(rms(cond [rows, 3584]))
/// 60 x block (img_mod / txt_mod(silu(temb)) -> (shift, scale, gate) x 2)
/// velocity = proj_out(LN(img) * (1 + scale) + shift),  (scale, shift) = norm_out.linear(silu(temb))      (diffusers' order: scale first)
/// </code>
/// Rotary positions (16 / 56 / 56 dims, theta 10000), the reference's centred layout: picture token (y, x) of an h x w grid sits at (0, y - ceil(h/2), x - ceil(w/2));
/// text token j at j + max(h/2, w/2) on all three axes. Blocks past the residency budget stream from the file every call (<see cref="StreamedModelWeights"/>); q | k | v and
/// the text stream's q | k | v are fused as they load (one GEMM each, still 8-bit).
/// </summary>
public sealed class QwenImageTransformer : IDisposable
{
    public const int Dim = 3072, Heads = 24, HeadDim = 128, MlpHidden = 12288, InChannels = 64, TxtDim = 3584, Layers = 60, TimeChannels = 256;
    public const float Eps = 1e-6f, RopeTheta = 10000f;

    private readonly Accelerator _acc;
    private readonly StreamedModelWeights _w;
    private readonly DiTKernels _k;
    private readonly SkinTokensKernels _u;
    private readonly Flux2Kernels _f;
    private readonly int _maxTxt, _nImg, _nOut;                                      // _nImg: the noise latent's tokens plus every reference's; _nOut: the noise latent's
    private readonly (int Rows, int Cols)[] _shapes;
    private int _nTxt, _n;                                                           // this call's text rows (a prompt and its negative differ)
    private readonly List<IDisposable> _owned = new();
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _img, _txt, _normI, _normT, _txtIn, _qkv, _attn, _attnImg, _attnTxt, _mlp, _scores;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _temb, _tembAct, _tproj, _hid, _modImg, _modTxt, _modOut, _cosT, _sinT, _out, _xs;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _temb0, _tembAct0, _modImg0;      // the timestep-zero modulation of the reference tokens (ReferencesAtTimestepZero)

    /// <summary>
    /// Qwen-Image Edit 2511 (ComfyUI's FluxKontextMultiReferenceLatentMethod "index_timestep_zero", diffusers' <c>zero_cond_t</c>): the reference tokens are modulated with the
    /// time embedding of t = 0 while the noise latent's tokens and the text use the call's t (measured on SpawnGen's 2511 taps: img_mod [2, 3072] -> [2, 18432], txt_mod [1, ..]).
    /// Off for 2509; <see cref="OpenAsync(Accelerator, IModelSource, string, string, int, int, int, IReadOnlyList{ValueTuple{int, int}}, long, CancellationToken)"/> turns it on when
    /// the file carries ComfyUI's <c>__index_timestep_zero__</c> marker (its 2511 files do). No effect without reference pictures.
    /// </summary>
    public bool ReferencesAtTimestepZero { get; set; }

    public long ResidentBytes => _w.ResidentBytes;
    public int ResidentBlocks => _w.ResidentBlocks;
    public long StreamedBytesPerForward => _w.StreamedBytesPerForward;
    /// <summary>The weights (streaming statistics: <see cref="StreamedModelWeights.StreamSeconds"/>).</summary>
    public StreamedModelWeights Weights => _w;
    /// <summary>The velocity of the last <see cref="ForwardAsync"/>, [noise latent tokens, 64] (2 x 2 patches of the 16-channel latent, channel c * 4 + dy * 2 + dx).</summary>
    public FV Output => _out.View;
    /// <summary>Tokens each call takes: the noise latent's plus every reference picture's (<see cref="ForwardAsync"/>'s imageTokens rows).</summary>
    public int InputTokens => _nImg;
    /// <summary>Intermediate checkpoints for tests: name, device view, rows, columns.</summary>
    public Action<string, FV, int, int>? Probe;

    private QwenImageTransformer(Accelerator acc, StreamedModelWeights w, int maxTextTokens, (int Rows, int Cols)[] shapes)
    {
        _acc = acc; _w = w; _maxTxt = _nTxt = maxTextTokens; _shapes = shapes;
        _nOut = shapes[0].Rows * shapes[0].Cols; _nImg = shapes.Sum(s => s.Rows * s.Cols); _n = _nTxt + _nImg;   // every buffer holds the longest text
        _k = new DiTKernels(acc) { PreciseActivations = true }; _u = new SkinTokensKernels(acc); _f = new Flux2Kernels(acc);
        long N = _n, I = _nImg, T = _nTxt;
        _img = Alloc(I * Dim); _txt = Alloc(T * Dim); _normI = Alloc(I * Dim); _normT = Alloc(T * Dim); _txtIn = Alloc(T * TxtDim);
        _qkv = Alloc(N * 3 * Dim); _attn = Alloc(N * Dim); _attnImg = Alloc(I * Dim); _attnTxt = Alloc(T * Dim); _mlp = Alloc(Math.Max(I, T) * MlpHidden);
        _scores = Alloc(_k.UsesFlashAttention(HeadDim) ? 1 : (long)Heads * N * N);
        _temb = Alloc(Dim); _tembAct = Alloc(Dim); _tproj = Alloc(TimeChannels); _hid = Alloc(Dim);
        _modImg = Alloc(6 * Dim); _modTxt = Alloc(6 * Dim); _modOut = Alloc(2 * Dim);
        _temb0 = Alloc(Dim); _tembAct0 = Alloc(Dim); _modImg0 = Alloc(6 * Dim);
        _cosT = Alloc(N * HeadDim); _sinT = Alloc(N * HeadDim); _out = Alloc((long)_nOut * InChannels);
        _xs = Alloc(w.HasScales ? Math.Max(I, T) * MlpHidden : 1);                             // the widest scaled input: an MLP's second matrix [rows, 12288]
    }

    private MemoryBuffer1D<float, Stride1D.Dense> Alloc(long n) { var b = _acc.Allocate1D<float>(Math.Max(1, n)); _owned.Add(b); return b; }

    private static int BlockOf(string n)
    {
        const string P = "transformer_blocks.";
        if (!n.StartsWith(P, StringComparison.Ordinal)) return -1;
        return int.Parse(n.AsSpan(P.Length, n.IndexOf('.', P.Length) - P.Length));
    }

    /// <summary>
    /// Open the transformer from <paramref name="file"/> for up to <paramref name="maxTextTokens"/> conditioning rows (each call passes its own count) and a picture of
    /// <paramref name="rows"/> x <paramref name="cols"/> tokens (latent / 2); blocks beyond <paramref name="residentBytes"/> stream from the file every call.
    /// </summary>
    public static Task<QwenImageTransformer> OpenAsync(Accelerator accelerator, IModelSource source, string repo, string file, int maxTextTokens, int rows, int cols, long residentBytes, CancellationToken ct = default)
        => OpenAsync(accelerator, source, repo, file, maxTextTokens, rows, cols, Array.Empty<(int, int)>(), residentBytes, ct);

    /// <summary>
    /// The same with reference pictures (Qwen-Image Edit: ComfyUI's <c>reference_latents</c>, diffusers' <c>image_latents</c>): <paramref name="references"/> are their token grids
    /// (ceil(latent / 2): odd sides are wrapped by <see cref="Flux1Kernels.PatchifyWrap"/>). Each call then takes the noise latent's tokens followed by every reference's tokens,
    /// each reference NORMALISED like the noise ((VAE encode - mean) / std), and the velocity is the noise latent's rows only.
    /// </summary>
    public static async Task<QwenImageTransformer> OpenAsync(Accelerator accelerator, IModelSource source, string repo, string file, int maxTextTokens, int rows, int cols,
        IReadOnlyList<(int Rows, int Cols)> references, long residentBytes, CancellationToken ct = default)
    {
        if (rows <= 0 || cols <= 0 || references.Any(r => r.Rows <= 0 || r.Cols <= 0)) throw new ArgumentException("token grids must be positive");
        var w = await StreamedModelWeights.OpenWithScalesAsync(accelerator, source, repo, file, "transformer_blocks.0.attn.to_q.weight", BlockOf, Layers, residentBytes,
            (w2, scales) => FuseQkvAsync(accelerator, w2, scales), ct);
        try
        {
            var t = new QwenImageTransformer(accelerator, w, maxTextTokens, new[] { (rows, cols) }.Concat(references).ToArray());
            t.ReferencesAtTimestepZero = w.HeaderNames.Contains("__index_timestep_zero__");   // ComfyUI's marker in Qwen-Image Edit 2511 files
            t.SetPositions(); return t;
        }
        catch { w.Dispose(); throw; }
    }

    /// <summary>
    /// to_q | to_k | to_v into attn.qkv and add_q_proj | add_k_proj | add_v_proj into attn.add_qkv (rows stacked; 8-bit matrices stay 8-bit, the bias vectors are bf16 by now).
    /// 8-bit parts fuse only when they share one decode scale (unscaled files, or equal scales), which the fused matrix then takes; parts with their own scales (ComfyUI's
    /// fp8mixed 2511 file) stay separate and run as three GEMMs (<see cref="Qkv"/>): one matrix cannot carry three scales exactly.
    /// </summary>
    private static async Task FuseQkvAsync(Accelerator acc, GpuWeights w, IDictionary<string, float> scales)
    {
        foreach (var q in w.Names.Where(n => n.EndsWith(".attn.to_q.weight", StringComparison.Ordinal)).ToList())
        {
            string b = q[..^"to_q.weight".Length];
            await FuseAsync(acc, w, scales, b + "qkv", b + "to_q", b + "to_k", b + "to_v");
            await FuseAsync(acc, w, scales, b + "add_qkv", b + "add_q_proj", b + "add_k_proj", b + "add_v_proj");
        }
    }

    private static async Task FuseAsync(Accelerator acc, GpuWeights w, IDictionary<string, float> scales, string to, params string[] parts)
    {
        var shapes = parts.Select(p => w.Shape(p + ".weight")).ToArray();
        int rowsOut = shapes.Sum(s => s[0]), cols = shapes[0][1];
        var retire = new List<IDisposable?>();
        if (parts.All(p => w.IsF8(p + ".weight")))
        {
            var partScales = parts.Select(p => scales.TryGetValue(p + ".weight", out var s) ? s : (float?)null).ToArray();
            if (partScales.Distinct().Count() > 1) return;                                  // different scales: leave the parts (Qkv runs them one by one)
            if (partScales[0] is float shared) { foreach (var p in parts) scales.Remove(p + ".weight"); scales[to + ".weight"] = shared; }
            var words = acc.Allocate1D<uint>(parts.Sum(p => w.F8Words(p + ".weight").Length));
            long off = 0;
            foreach (var p in parts) { var src = w.F8Words(p + ".weight"); words.View.SubView(off, src.Length).CopyFrom(src); off += src.Length; }
            foreach (var p in parts) retire.Add(w.Remove(p + ".weight"));
            retire.Add(w.SetF8(to + ".weight", words, new[] { rowsOut, cols }));
        }
        else if (parts.All(p => w.IsBF16(p + ".weight")))
        {
            var m = acc.Allocate1D<BFloat16>((long)rowsOut * cols);
            long off = 0;
            foreach (var p in parts) { var src = w.BF16(p + ".weight"); m.View.SubView(off, src.Length).CopyFrom(src); off += src.Length; }
            foreach (var p in parts) retire.Add(w.Remove(p + ".weight"));
            retire.Add(w.SetBF16(to + ".weight", m, new[] { rowsOut, cols }));
        }
        else throw new NotSupportedException($"{to}: the parts must all be 8-bit floats or all bf16");
        var bias = acc.Allocate1D<BFloat16>(rowsOut);
        long boff = 0;
        foreach (var p in parts) { var src = w.BF16(p + ".bias"); bias.View.SubView(boff, src.Length).CopyFrom(src); boff += src.Length; }
        foreach (var p in parts) retire.Add(w.Remove(p + ".bias"));
        retire.Add(w.SetBF16(to + ".bias", bias, new[] { rowsOut }));
        await acc.SynchronizeAsync();
        foreach (var r in retire) r?.Dispose();
    }

    /// <summary>The position ids [textTokens + rows * cols, 3] in the reference's order of the joint sequence: text first, then the picture (see the class summary).</summary>
    public static int[] Ids(int textTokens, int rows, int cols) => Ids(textTokens, new[] { (rows, cols) });

    /// <summary>
    /// The position ids of text followed by several pictures, ComfyUI's layout (measured on its 2509 edit runs, SpawnGen's pe_embedder taps): picture k (0 = the noise latent, then
    /// the references) has frame index k; a grid of n rows (columns) runs from -(n / 2) to n - 1 - n / 2 (integer division: 83 columns -41..41); the text starts at
    /// max(rows / 2, cols / 2) of the NOISE LATENT (text row 0 at (8, 8, 8) for a 12 x 16 latent with 56 x 74 and 50 x 83 references). <paramref name="diffusers"/>: diffusers'
    /// <c>QwenEmbedRope</c> instead, which differs in two places: an odd grid runs from -(n - n / 2) (83 columns -42..40), and the text starts after the largest
    /// max(rows / 2, cols / 2) over every picture. Both agree for one picture of even sides (the base model's gates).
    /// </summary>
    public static int[] Ids(int textTokens, IReadOnlyList<(int Rows, int Cols)> pictures, bool diffusers = false)
    {
        int total = textTokens, start = 0;
        for (int q = 0; q < pictures.Count; q++)
        {
            var (r, c) = pictures[q]; total += r * c;
            if (q == 0 || diffusers) start = Math.Max(start, Math.Max(r / 2, c / 2));
        }
        var ids = new int[total * 3];
        for (int j = 0; j < textTokens; j++) { ids[j * 3] = ids[j * 3 + 1] = ids[j * 3 + 2] = start + j; }
        int row = textTokens;
        for (int k = 0; k < pictures.Count; k++)
        {
            var (rows, cols) = pictures[k];
            int y0 = diffusers ? rows - rows / 2 : rows / 2, x0 = diffusers ? cols - cols / 2 : cols / 2;
            for (int y = 0; y < rows; y++) for (int x = 0; x < cols; x++) { ids[row * 3] = k; ids[row * 3 + 1] = y - y0; ids[row * 3 + 2] = x - x0; row++; }
        }
        return ids;
    }

    /// <summary>
    /// CUDA: fp32-class activations through the tensor cores (<see cref="DiTKernels.PreciseActivations"/>: split hi + lo bf16 GEMMs, fp32 flash attention). ON by default: this port is
    /// checked against ComfyUI's fp32 runs, and bf16 activations cost ~5e-3 per call (FLUX.1 4.7e-3 -> 3.3e-5, Z-Image 5.6e-3 -> 3.2e-5, Qwen-Image 4.3e-2 -> 2.4e-4 measured on an
    /// RTX 4070; no slower at 256 px). Other backends compute in fp32 either way.
    /// </summary>
    public bool PreciseActivations { get => _k.PreciseActivations; set => _k.PreciseActivations = value; }

    /// <summary>The longest text this transformer was opened for.</summary>
    public int MaxTextTokens => _maxTxt;

    private int[]? _customIds;

    private void SetPositions()
    {
        var ids = _customIds ?? Ids(_nTxt, _shapes);
        var (cos, sin) = Flux1Transformer.RopeTables(ids, _n);                                  // the same axes (16 / 56 / 56) and theta as FLUX.1
        _cosT.View.SubView(0, cos.Length).CopyFromCPU(cos); _sinT.View.SubView(0, sin.Length).CopyFromCPU(sin);
    }

    /// <summary>
    /// Use these position ids [(textTokens + <see cref="InputTokens"/>) * 3] (text first, then the pictures; (frame, row, column) per token) instead of <see cref="Ids(int, IReadOnlyList{ValueTuple{int, int}})"/>
    /// for calls with <paramref name="textTokens"/> rows, until <see cref="ResetPositionIds"/>. For measuring a reference implementation's layout and for callers that place pictures
    /// their own way.
    /// </summary>
    public void UsePositionIds(int[] ids, int textTokens)
    {
        if (textTokens < 1 || textTokens > _maxTxt) throw new ArgumentOutOfRangeException(nameof(textTokens), $"1 to {_maxTxt}");
        if (ids.Length != (textTokens + _nImg) * 3) throw new ArgumentException($"{(textTokens + _nImg) * 3} ids (text {textTokens} + pictures {_nImg}, 3 each)");
        _customIds = (int[])ids.Clone(); _nTxt = textTokens; _n = _nTxt + _nImg; SetPositions();
    }

    /// <summary>Back to <see cref="Ids(int, IReadOnlyList{ValueTuple{int, int}})"/>.</summary>
    public void ResetPositionIds() { _customIds = null; SetPositions(); }

    private static float[] Sinusoid(float t)
    {
        // diffusers' Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000): [cos | sin] of 1000 t over exp(-ln(10000) i / 128)
        var e = new float[TimeChannels];
        double v = 1000.0 * t;
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
    private void Gemv(GpuWeights w, FV x, string name, FV o, int K, int N) => _u.Gemv(Scaled(w, name + ".weight", x, K), _w.MatrixExact(w, name + ".weight"), w.BF16(name + ".bias"), o, K, N);
    private void Lin(GpuWeights w, FV x, string name, FV o, int rows, int K, int N, DiTKernels.Act act = DiTKernels.Act.None)
        => _k.Linear(Scaled(w, name + ".weight", x, (long)rows * K), _w.MatrixExact(w, name + ".weight"), w.BF16(name + ".bias"), o, rows, K, N, act);

    /// <summary>
    /// One model call: <paramref name="imageTokens"/> [<see cref="InputTokens"/>, 64] (the model-space latent's 2 x 2 patches, <see cref="Flux1Kernels.Patchify"/>, then each reference
    /// latent's, in the order the transformer was opened with), <paramref name="cond"/> the text
    /// encoder's rows [textTokens, 3584], <paramref name="timestep"/> the model's time (how ComfyUI maps sigma to it is measured by the gate). <paramref name="textTokens"/>: this call's
    /// rows, 1 to <see cref="MaxTextTokens"/> (-1: the maximum); only the rotary table is rebuilt when it changes. The velocity is left in <see cref="Output"/>.
    /// </summary>
    public async Task ForwardAsync(FV imageTokens, FV cond, float timestep, int textTokens = -1, CancellationToken ct = default)
    {
        if (textTokens < 0) textTokens = _maxTxt;
        if (textTokens < 1 || textTokens > _maxTxt) throw new ArgumentOutOfRangeException(nameof(textTokens), $"1 to {_maxTxt} (the transformer's MaxTextTokens)");
        if (textTokens != _nTxt) { _nTxt = textTokens; _n = _nTxt + _nImg; SetPositions(); }
        var b0 = _w.Base;
        _tproj.View.CopyFromCPU(Sinusoid(timestep));
        Gemv(b0, _tproj.View, "time_text_embed.timestep_embedder.linear_1", _hid.View, TimeChannels, Dim); _k.Silu(_hid.View, _tembAct.View, Dim);
        Gemv(b0, _tembAct.View, "time_text_embed.timestep_embedder.linear_2", _temb.View, Dim, Dim);
        _k.Silu(_temb.View, _tembAct.View, Dim);                                     // every modulation reads silu(temb)
        _zero = ReferencesAtTimestepZero && _nImg > _nOut;
        if (_zero)
        {
            // the same time MLP at t = 0 (diffusers: timestep concatenated with timestep * 0)
            _tproj.View.CopyFromCPU(Sinusoid(0f));
            Gemv(b0, _tproj.View, "time_text_embed.timestep_embedder.linear_1", _hid.View, TimeChannels, Dim); _k.Silu(_hid.View, _tembAct0.View, Dim);
            Gemv(b0, _tembAct0.View, "time_text_embed.timestep_embedder.linear_2", _temb0.View, Dim, Dim);
            _k.Silu(_temb0.View, _tembAct0.View, Dim);
        }
        Lin(b0, imageTokens, "img_in", _img.View, _nImg, InChannels, Dim);
        _u.RmsNorm(cond, _txtIn.View, b0.BF16("txt_norm.weight"), _nTxt, TxtDim, Eps);
        Lin(b0, _txtIn.View, "txt_in", _txt.View, _nTxt, TxtDim, Dim);
        for (int k = 0; k < Layers; k++)
        {
            var (w, streamed) = await _w.BlockAsync(k, ct);
            try { Block(w, $"transformer_blocks.{k}."); }
            finally { if (streamed) { await _acc.SynchronizeAsync(); w.Dispose(); } }
            Probe?.Invoke($"block{k}_img", _img.View, _nImg, Dim);
            Probe?.Invoke($"block{k}_txt", _txt.View, _nTxt, Dim);
        }
        Gemv(b0, _tembAct.View, "norm_out.linear", _modOut.View, Dim, 2 * Dim);       // (scale, shift): the call's t only, also in 2511 (ComfyUI's norm_out gets one [1, 3072] row)
        _k.LayerNormModulate(_img.View, _normI.View, _modOut.View, Dim, 0, _nOut, Dim, Eps);   // the head runs per token: only the noise latent's rows are the prediction
        Probe?.Invoke("norm_out", _normI.View, _nOut, Dim);
        Lin(b0, _normI.View, "proj_out", _out.View, _nOut, Dim, InChannels);
        Probe?.Invoke("proj_out", _out.View, _nOut, InChannels);
    }

    private bool _zero;                                                              // this call modulates the reference tokens at t = 0

    // q | k | v into dst [rows, 3 Dim]: one GEMM when the load fused them (FuseQkvAsync), else each part's own GEMM (its own 8-bit scale) through tmp [rows, Dim], copied into its
    // column range. tmp is the attention output buffer of the stream, free until the attention runs.
    private void Qkv(GpuWeights w, FV x, string fused, string[] parts, FV dst, FV tmp, int rows)
    {
        if (w.Has(fused + ".weight")) { Lin(w, x, fused, dst, rows, Dim, 3 * Dim); return; }
        for (int p = 0; p < 3; p++)
        {
            Lin(w, x, parts[p], tmp, rows, Dim, Dim);
            _f.CopyCols(tmp, dst, rows, Dim, Dim, 3 * Dim, p * Dim);
        }
    }

    // the image stream's modulate / gate: one modulation for every image token, or (ReferencesAtTimestepZero) the noise latent's rows with t and the reference rows with 0
    private void ImgModulate(int shiftOff, int scaleOff)
    {
        if (!_zero) { _k.LayerNormModulate(_img.View, _normI.View, _modImg.View, shiftOff, scaleOff, _nImg, Dim, Eps); return; }
        long split = (long)_nOut * Dim, rest = (long)(_nImg - _nOut) * Dim;
        _k.LayerNormModulate(_img.View.SubView(0, split), _normI.View.SubView(0, split), _modImg.View, shiftOff, scaleOff, _nOut, Dim, Eps);
        _k.LayerNormModulate(_img.View.SubView(split, rest), _normI.View.SubView(split, rest), _modImg0.View, shiftOff, scaleOff, _nImg - _nOut, Dim, Eps);
    }
    private void ImgGate(int gateOff)
    {
        if (!_zero) { _k.GateResidual(_img.View, _attnImg.View, _modImg.View, gateOff, _nImg * Dim, Dim); return; }
        long split = (long)_nOut * Dim, rest = (long)(_nImg - _nOut) * Dim;
        _k.GateResidual(_img.View.SubView(0, split), _attnImg.View.SubView(0, split), _modImg.View, gateOff, _nOut * Dim, Dim);
        _k.GateResidual(_img.View.SubView(split, rest), _attnImg.View.SubView(split, rest), _modImg0.View, gateOff, (_nImg - _nOut) * Dim, Dim);
    }

    private void Block(GpuWeights w, string b)
    {
        Gemv(w, _tembAct.View, b + "img_mod.1", _modImg.View, Dim, 6 * Dim);           // (shift, scale, gate) for attention, then for the MLP
        if (_zero) Gemv(w, _tembAct0.View, b + "img_mod.1", _modImg0.View, Dim, 6 * Dim);
        Gemv(w, _tembAct.View, b + "txt_mod.1", _modTxt.View, Dim, 6 * Dim);
        ImgModulate(0, Dim);
        _k.LayerNormModulate(_txt.View, _normT.View, _modTxt.View, 0, Dim, _nTxt, Dim, Eps);
        var qkvT = _qkv.View.SubView(0, (long)_nTxt * 3 * Dim); var qkvI = _qkv.View.SubView((long)_nTxt * 3 * Dim, (long)_nImg * 3 * Dim);
        Qkv(w, _normT.View, b + "attn.add_qkv", new[] { b + "attn.add_q_proj", b + "attn.add_k_proj", b + "attn.add_v_proj" }, qkvT, _attnTxt.View, _nTxt);
        Qkv(w, _normI.View, b + "attn.qkv", new[] { b + "attn.to_q", b + "attn.to_k", b + "attn.to_v" }, qkvI, _attnImg.View, _nImg);
        _u.RmsNormHeads(qkvT, w.BF16(b + "attn.norm_added_q.weight"), _nTxt, 3 * Dim, 0, Heads, HeadDim, Eps);
        _u.RmsNormHeads(qkvT, w.BF16(b + "attn.norm_added_k.weight"), _nTxt, 3 * Dim, Dim, Heads, HeadDim, Eps);
        _u.RmsNormHeads(qkvI, w.BF16(b + "attn.norm_q.weight"), _nImg, 3 * Dim, 0, Heads, HeadDim, Eps);
        _u.RmsNormHeads(qkvI, w.BF16(b + "attn.norm_k.weight"), _nImg, 3 * Dim, Dim, Heads, HeadDim, Eps);
        _f.RopeInterleaved(_qkv.View, _cosT.View, _sinT.View, _n, 3 * Dim, 0, Heads, HeadDim);
        _f.RopeInterleaved(_qkv.View, _cosT.View, _sinT.View, _n, 3 * Dim, Dim, Heads, HeadDim);
        Probe?.Invoke(b + "qkv", _qkv.View, _n, 3 * Dim);                                // the attention's inputs: [tokens][q | k | v], text tokens first
        _k.Attention(_qkv.View, 3 * Dim, 0, _qkv.View, 3 * Dim, Dim, 2 * Dim, _scores.View, _attn.View, _n, _n, Heads, HeadDim);
        Lin(w, _attn.View.SubView(0, (long)_nTxt * Dim), b + "attn.to_add_out", _attnTxt.View, _nTxt, Dim, Dim);
        Lin(w, _attn.View.SubView((long)_nTxt * Dim, (long)_nImg * Dim), b + "attn.to_out.0", _attnImg.View, _nImg, Dim, Dim);
        if (Probe != null) { Probe(b + "attn_img", _attnImg.View, _nImg, Dim); Probe(b + "attn_txt", _attnTxt.View, _nTxt, Dim); }
        ImgGate(2 * Dim);
        _k.GateResidual(_txt.View, _attnTxt.View, _modTxt.View, 2 * Dim, _nTxt * Dim, Dim);
        ImgModulate(3 * Dim, 4 * Dim);
        Lin(w, _normI.View, b + "img_mlp.net.0.proj", _mlp.View, _nImg, Dim, MlpHidden, DiTKernels.Act.GeluTanh);
        Lin(w, _mlp.View, b + "img_mlp.net.2", _attnImg.View, _nImg, MlpHidden, Dim);
        ImgGate(5 * Dim);
        _k.LayerNormModulate(_txt.View, _normT.View, _modTxt.View, 3 * Dim, 4 * Dim, _nTxt, Dim, Eps);
        Lin(w, _normT.View, b + "txt_mlp.net.0.proj", _mlp.View, _nTxt, Dim, MlpHidden, DiTKernels.Act.GeluTanh);
        Lin(w, _mlp.View, b + "txt_mlp.net.2", _attnTxt.View, _nTxt, MlpHidden, Dim);
        _k.GateResidual(_txt.View, _attnTxt.View, _modTxt.View, 5 * Dim, _nTxt * Dim, Dim);
    }

    public void Dispose()
    {
        foreach (var o in _owned) o.Dispose();
        _owned.Clear();
        _w.Dispose();
    }
}
