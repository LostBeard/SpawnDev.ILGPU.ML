using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Anima;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>
/// Anima's diffusion transformer: Cosmos-Predict2-2B MiniTrainDIT (28 blocks, width 2048, 16 heads x 128). Every convention is the one
/// tools/anima-ref/ANIMA-SPEC.md measured against ComfyUI's black-box taps (float64 reference within 6.4e-6 at every stage):
/// patch embed of the latent plus a zero padding-mask channel (1x2x2 patches, 68 -> 2048, no bias); timestep = sigma (cos-first sinusoid ->
/// RMSNorm for the block conditioning, -> Linear SiLU Linear for the shared adaLN-LoRA vector [6144]); per block three adaLN-LoRA-modulated
/// sub-layers (self-attention with per-head q/k RMSNorm eps 1e-6 and rotate-half 2-D RoPE, cross-attention to the 512-row context, exact-GELU
/// MLP), each gated; a modulated final layer and the (p_h, p_w, p_t, c) unpatchify. No biases anywhere.
/// <para>
/// Returns the VELOCITY v (rectified flow); ComfyUI's denoised prediction is x - sigma * v. The context is padded to exactly 512 rows with zeros
/// and the padding is NOT masked (attending over only the real rows gives 1.5 relRMS): <see cref="SetContext(int, FV, int)"/> does the padding.
/// The residual stream reaches 2.5e5, so it stays fp32 (LayerNorm two-pass in DiTKernels). Batch 1; CFG = one call per prompt slot.
/// </para>
/// </summary>
public sealed class AnimaDiT : IDisposable
{
    /// <summary>The DiT's tensor-name prefix in the Anima file.</summary>
    public const string Prefix = "net.";
    /// <summary>Context rows every cross-attention reads (the prompt's rows plus zero padding) and their width.</summary>
    public const int ContextRows = 512, ContextDim = 1024;
    private readonly Accelerator _acc;
    private readonly DiTKernels _k;
    private readonly Wan.WanKernels _wk;
    private readonly AnimaKernels _ak;
    private readonly GpuWeights _w;
    private readonly Flux2.Flux2Weights _mat;
    private readonly int _dim, _ffn, _heads, _headDim, _blocks, _rank;

    private int _t, _h, _wd, _n;
    private readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _ws = new();
    private readonly MemoryBuffer1D<float, Stride1D.Dense>?[] _ctx = new MemoryBuffer1D<float, Stride1D.Dense>?[2];
    private MemoryBuffer1D<float, Stride1D.Dense>? _phases;
    private FV _feat, _hid, _hn, _q, _kv, _attn, _proj, _mlp, _scores, _out, _tProj, _tHid, _adaln, _emb, _eAct, _lo, _mod;

    /// <summary>Transformer width (2048).</summary>
    public int Dim => _dim;
    /// <summary>Transformer blocks in the file (28).</summary>
    public int Layers => _blocks;
    /// <summary>Image tokens of the grid set by <see cref="SetGrid"/> (frames x h/2 x w/2); 0 before it.</summary>
    public int TokenCount => _n;

    /// <summary>CUDA: fp32-class activations through the tensor-core GEMMs (<see cref="DiTKernels.PreciseActivations"/>). Other backends compute in fp32 either way.</summary>
    public bool PreciseActivations { get => _k.PreciseActivations; set => _k.PreciseActivations = value; }

    /// <summary>Test hook: (stage name, activation view, element count); valid only during the call.</summary>
    public Action<string, FV, int>? Tap { get; set; }

    /// <summary>
    /// A ControlNet-LLLite applied to every <see cref="Forward"/> while set (both CFG slots; set it to null for the steps outside its start / end range).
    /// Its picture must give this grid's token count (<see cref="AnimaLLLite.SetPictureAsync"/> with 8 x the latent's size).
    /// </summary>
    public AnimaLLLite? LLLite { get; set; }

    /// <summary>Blocks whose output is tapped (as "block_N").</summary>
    public int[] TapBlocks { get; set; } = { 0, 13, 27 };

    /// <summary>The DiT over weights already on the device (not owned: the caller disposes them).</summary>
    /// <param name="accelerator">the application's accelerator (never disposed here)</param>
    /// <param name="kernels">the transformer kernels (one instance per model; <see cref="PreciseActivations"/> sets them)</param>
    /// <param name="weights">the Anima file (net.*); the adapter's tensors in the same file are left to <see cref="AnimaLlmAdapter"/></param>
    public AnimaDiT(Accelerator accelerator, DiTKernels kernels, GpuWeights weights)
    {
        _acc = accelerator; _k = kernels; _wk = new Wan.WanKernels(accelerator); _ak = new AnimaKernels(accelerator); _w = weights;
        _mat = new Flux2.Flux2Weights(accelerator, weights, ownsWeights: false);
        _dim = weights.Shape(Prefix + "x_embedder.proj.1.weight")[0];
        if (weights.Shape(Prefix + "x_embedder.proj.1.weight")[1] != AnimaKernels.PatchFeatures)
            throw new NotSupportedException($"AnimaDiT: x_embedder takes {weights.Shape(Prefix + "x_embedder.proj.1.weight")[1]} features, expected {AnimaKernels.PatchFeatures} (16 + 1 mask channel, 1x2x2)");
        _ffn = weights.Shape(Prefix + "blocks.0.mlp.layer1.weight")[0];
        _rank = weights.Shape(Prefix + "blocks.0.adaln_modulation_self_attn.1.weight")[0];
        _headDim = weights.Shape(Prefix + "blocks.0.self_attn.q_norm.weight")[0];
        _heads = _dim / _headDim;
        _blocks = 0; while (weights.Has($"{Prefix}blocks.{_blocks}.self_attn.q_proj.weight")) _blocks++;
        if (_headDim != 128) throw new NotSupportedException("AnimaDiT: head dim 128 only");
    }

    private FV Alloc(long n) { var b = _acc.Allocate1D<float>(n); _ws.Add(b); return b.View; }

    private void FreeWorkspace()
    {
        foreach (var b in _ws) b.Dispose();
        _ws.Clear();
        _phases?.Dispose(); _phases = null;
    }

    /// <summary>Fix the latent grid ([16, frames, h, w], h and w even): RoPE table and work buffers. Tokens N = frames * h/2 * w/2.</summary>
    public void SetGrid(int frames, int h, int w)
    {
        if (h % 2 != 0 || w % 2 != 0) throw new ArgumentException("the latent height and width must be even (1x2x2 patches)");
        FreeWorkspace();
        _t = frames; _h = h; _wd = w; _n = frames * (h / 2) * (w / 2);
        int N = _n, C = _dim, kvRows = Math.Max(N, ContextRows);
        var table = AnimaKernels.DitRopePhases(frames, h / 2, w / 2);
        _phases = _acc.Allocate1D<float>(table.Length);
        _phases.View.CopyFromCPU(table);
        _feat = Alloc((long)N * AnimaKernels.PatchFeatures); _hid = Alloc((long)N * C); _hn = Alloc((long)N * C); _q = Alloc((long)N * C);
        _kv = Alloc(2L * kvRows * C); _attn = Alloc((long)N * C); _proj = Alloc((long)N * C); _mlp = Alloc((long)N * _ffn);
        _scores = Alloc(_k.UsesFlashAttention(_headDim) ? 1 : (long)_heads * N * kvRows);
        _out = Alloc((long)N * AnimaKernels.OutFeatures);
        _tProj = Alloc(C); _tHid = Alloc(C); _adaln = Alloc(3 * C); _emb = Alloc(C); _eAct = Alloc(C); _lo = Alloc(_rank); _mod = Alloc(3 * C);
    }

    /// <summary>
    /// The context of prompt slot <paramref name="slot"/> (0 / 1, e.g. negative / positive): <paramref name="rows"/> rows of [rows, 1024] (the
    /// adapter's output, unpadded), zero-padded here to exactly 512 rows. Stays on the device; call again to change the prompt.
    /// </summary>
    public void SetContext(int slot, FV context, int rows)
    {
        if (slot is < 0 or > 1) throw new ArgumentOutOfRangeException(nameof(slot));
        if (rows < 0 || rows > ContextRows || context.Length < (long)rows * ContextDim) throw new ArgumentException($"the context has {rows} rows; at most {ContextRows}");
        var buf = _ctx[slot] ??= _acc.Allocate1D<float>((long)ContextRows * ContextDim);
        _ak.CopyPadded(context, buf.View, (long)rows * ContextDim);
    }

    /// <summary>Host overload of <see cref="SetContext(int, FV, int)"/> for tests (the pipeline passes the adapter's device output).</summary>
    public void SetContext(int slot, float[] context, int rows)
    {
        if (slot is < 0 or > 1) throw new ArgumentOutOfRangeException(nameof(slot));
        if (rows < 0 || rows > ContextRows || context.Length < rows * ContextDim) throw new ArgumentException($"the context has {rows} rows; at most {ContextRows}");
        var padded = new float[ContextRows * ContextDim];
        Array.Copy(context, padded, rows * ContextDim);
        var buf = _ctx[slot] ??= _acc.Allocate1D<float>((long)ContextRows * ContextDim);
        buf.View.CopyFromCPU(padded);
    }

    // m = Linear(Linear(SiLU(emb), name.1 [rank, C]), name.2 [width, rank]) + adaln[0 .. width)
    private void Modulation(string name, int width)
    {
        _mat.Linear(_k, _eAct, name + ".1.weight", null, _lo, 1, _dim, _rank);
        _mat.Linear(_k, _lo, name + ".2.weight", null, _mod, 1, _rank, width);
        _k.GateResidual(_mod, _adaln, default, -1, width, width);
    }

    /// <summary>
    /// One velocity prediction: <paramref name="latent"/> [16, frames, h, w] -> <paramref name="v"/> (same shape) at timestep <paramref name="sigma"/>
    /// (the flow sigma itself), with the context of <paramref name="slot"/>.
    /// </summary>
    public void Forward(FV latent, double sigma, int slot, FV v)
    {
        if (_n == 0) throw new InvalidOperationException("SetGrid must be called before Forward");
        var ctx = _ctx[slot] ?? throw new InvalidOperationException($"SetContext({slot}) must be called before Forward");
        int N = _n, C = _dim, H = _heads, D = _headDim, L = ContextRows;
        string P = Prefix;

        // timestep: t_proj (cos, sin), emb = RMSNorm(t_proj), adaln = Linear(SiLU(Linear(t_proj)))
        _ak.TimestepSinusoid(_tProj, (float)sigma, C);
        _emb.CopyFrom(_tProj);
        _wk.RowRmsNorm(_emb, _w.BF16(P + "t_embedding_norm.weight"), 1, C, 0, C, 1e-6f);
        _mat.Linear(_k, _tProj, P + "t_embedder.1.linear_1.weight", null, _tHid, 1, C, C, DiTKernels.Act.Silu);
        _mat.Linear(_k, _tHid, P + "t_embedder.1.linear_2.weight", null, _adaln, 1, C, 3 * C);
        _k.Silu(_emb, _eAct, C);
        Tap?.Invoke("t_proj", _tProj, C); Tap?.Invoke("emb", _emb, C); Tap?.Invoke("adaln", _adaln, 3 * C);

        // patch embedding
        _ak.Patchify(latent, _feat, _t, _h, _wd);
        _mat.Linear(_k, _feat, P + "x_embedder.proj.1.weight", null, _hid, N, AnimaKernels.PatchFeatures, C);
        Tap?.Invoke("x_embedder", _hid, N * C);

        FV k = _kv.SubView(0, (long)N * C), vv = _kv.SubView((long)N * C, (long)N * C);
        FV kc = _kv.SubView(0, (long)L * C), vc = _kv.SubView((long)L * C, (long)L * C);
        var ll = LLLite;
        if (ll != null && ll.Tokens != N) throw new InvalidOperationException($"the LLLite's picture gives {ll.Tokens} tokens, this grid has {N}: SetPictureAsync with 8 x the latent's size");
        for (int b = 0; b < _blocks; b++)
        {
            string p = $"{P}blocks.{b}.";

            // self attention: per-head RMSNorm on q and k, rotate-half RoPE, no mask
            Modulation(p + "adaln_modulation_self_attn", 3 * C);
            _k.LayerNormModulate(_hid, _hn, _mod, 0, C, N, C, 1e-6f);
            // an LLLite perturbs the INPUT of each projection it has a module for (its own buffers; null = the plain input)
            FV qIn = _hn, kIn = _hn, vIn = _hn;
            if (ll != null)
            {
                qIn = ll.Apply(b, AnimaLLLite.Target.Q, _hn, N) ?? _hn;
                kIn = ll.Apply(b, AnimaLLLite.Target.K, _hn, N) ?? _hn;
                vIn = ll.Apply(b, AnimaLLLite.Target.V, _hn, N) ?? _hn;
            }
            _mat.Linear(_k, qIn, p + "self_attn.q_proj.weight", null, _q, N, C, C);
            _mat.Linear(_k, kIn, p + "self_attn.k_proj.weight", null, k, N, C, C);
            _mat.Linear(_k, vIn, p + "self_attn.v_proj.weight", null, vv, N, C, C);
            _wk.RowRmsNorm(_q, _w.BF16(p + "self_attn.q_norm.weight"), N * H, D, 0, D, 1e-6f);
            _wk.RowRmsNorm(k, _w.BF16(p + "self_attn.k_norm.weight"), N * H, D, 0, D, 1e-6f);
            _ak.RopeHalf(_q, _phases!.View, N, C, 0, H, D);
            _ak.RopeHalf(k, _phases.View, N, C, 0, H, D);
            if (b == 0) { Tap?.Invoke("b0_mod_sa", _mod, 3 * C); Tap?.Invoke("b0_q", _q, N * C); Tap?.Invoke("b0_k", k, N * C); Tap?.Invoke("b0_v", vv, N * C); }
            _k.Attention(_q, C, 0, _kv, C, 0, N * C, _scores, _attn, N, N, H, D);
            if (b == 0) Tap?.Invoke("b0_attn_sa", _attn, N * C);
            _mat.Linear(_k, _attn, p + "self_attn.output_proj.weight", null, _proj, N, C, C);
            _k.GateResidual(_hid, _proj, _mod, 2 * C, N * C, C);

            // cross attention over all 512 context rows (zero padding unmasked), no RoPE
            Modulation(p + "adaln_modulation_cross_attn", 3 * C);
            _k.LayerNormModulate(_hid, _hn, _mod, 0, C, N, C, 1e-6f);
            _mat.Linear(_k, _hn, p + "cross_attn.q_proj.weight", null, _q, N, C, C);
            _mat.Linear(_k, ctx.View, p + "cross_attn.k_proj.weight", null, kc, L, ContextDim, C);
            _mat.Linear(_k, ctx.View, p + "cross_attn.v_proj.weight", null, vc, L, ContextDim, C);
            _wk.RowRmsNorm(_q, _w.BF16(p + "cross_attn.q_norm.weight"), N * H, D, 0, D, 1e-6f);
            _wk.RowRmsNorm(kc, _w.BF16(p + "cross_attn.k_norm.weight"), L * H, D, 0, D, 1e-6f);
            if (b == 0) { Tap?.Invoke("b0_after_sa", _hid, N * C); Tap?.Invoke("b0_kc", kc, L * C); Tap?.Invoke("b0_vc", vc, L * C); }
            _k.Attention(_q, C, 0, _kv, C, 0, L * C, _scores, _attn, N, L, H, D);
            if (b == 0) Tap?.Invoke("b0_attn_ca", _attn, N * C);
            _mat.Linear(_k, _attn, p + "cross_attn.output_proj.weight", null, _proj, N, C, C);
            _k.GateResidual(_hid, _proj, _mod, 2 * C, N * C, C);

            // MLP, exact GELU
            Modulation(p + "adaln_modulation_mlp", 3 * C);
            _k.LayerNormModulate(_hid, _hn, _mod, 0, C, N, C, 1e-6f);
            FV fc1In = ll?.Apply(b, AnimaLLLite.Target.MlpFc1, _hn, N) ?? _hn;
            _mat.Linear(_k, fc1In, p + "mlp.layer1.weight", null, _mlp, N, C, _ffn, DiTKernels.Act.GeluErf);
            if (b == 0) Tap?.Invoke("b0_mlp_hidden", _mlp, N * _ffn);
            _mat.Linear(_k, _mlp, p + "mlp.layer2.weight", null, _proj, N, _ffn, C);
            _k.GateResidual(_hid, _proj, _mod, 2 * C, N * C, C);

            if (Tap != null && Array.IndexOf(TapBlocks, b) >= 0) Tap($"block_{b}", _hid, N * C);
            _acc.Submit();                                  // one command buffer per block in the browser (the GPU watchdog; WanDiT)
        }

        // final layer: 2-way modulation (+ adaln[0 .. 2C)), LayerNorm, Linear 2048 -> 64, unpatchify
        Modulation(P + "final_layer.adaln_modulation", 2 * C);
        _k.LayerNormModulate(_hid, _hn, _mod, 0, C, N, C, 1e-6f);
        _mat.Linear(_k, _hn, P + "final_layer.linear.weight", null, _out, N, C, AnimaKernels.OutFeatures);
        Tap?.Invoke("final_layer", _out, N * AnimaKernels.OutFeatures);
        _ak.Unpatchify(_out, v, _t, _h, _wd);
        _acc.Fence();
    }

    /// <summary>Frees the work buffers and contexts (not the weights, not the accelerator). Drain the queue first (SynchronizeAsync) on browser backends.</summary>
    public void Dispose()
    {
        _mat.Dispose();
        FreeWorkspace();
        foreach (var c in _ctx) c?.Dispose();
    }
}
