using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// The TRELLIS.2 flow transformer on a token sequence: input Linear, N blocks of (adaLN self-attention with RoPE-3D and
/// qk RMS norm, cross-attention to the image features, GELU feed-forward), shared adaLN modulation, plain LayerNorm and
/// output Linear. One class serves all three flow checkpoints: the dense 16^3 sparse-structure model (tokens = grid cells),
/// and the shape / texture SLat models (tokens = occupied voxels), because the reference blocks are the same maths
/// (trellis2/modules/transformer/modulated.py and modules/sparse/transformer/modulated.py). Batch 1.
/// Weights stay bf16 on the device, activations are fp32; work buffers are allocated once per token count and reused
/// across the sampling steps (call <see cref="Dispose"/> to free them).
/// </summary>
public sealed class FlowDiT : IDisposable
{
    private readonly Accelerator _acc;
    private readonly DiTKernels _k;
    private readonly GpuWeights _w;
    private readonly int _inCh, _outCh, _dim, _condCh, _blocks, _heads, _headDim, _hidden;

    private int _n = -1, _condTokens = -1;
    private MemoryBuffer1D<float, Stride1D.Dense>? _phases;
    private readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _ws = new();
    private FV _h, _hn, _qkv, _q, _kv, _attn, _proj, _mlp, _scores, _modv, _mod, _tFreq, _tHidden, _tEmb, _tAct;

    public int InChannels => _inCh;
    public int OutChannels => _outCh;
    public int CondChannels => _condCh;
    public int ModelChannels => _dim;
    public int TokenCount => _n;

    /// <summary>TRELLIS2_PROFILE=1: per-op device time of the NEXT forward (synchronises around every op), summed by name in <see cref="ProfileMs"/>.</summary>
    public static readonly bool Profile = Environment.GetEnvironmentVariable("TRELLIS2_PROFILE") == "1";
    public readonly Dictionary<string, double> ProfileMs = new();
    private void T(string name, Action a)
    {
        if (!Profile) { a(); return; }
        _acc.Synchronize();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        a();
        _acc.Synchronize();
        ProfileMs[name] = ProfileMs.GetValueOrDefault(name) + sw.Elapsed.TotalMilliseconds;
    }

    /// <summary>Test hook: (stage name, activation view, element count); valid only during the call.</summary>
    public Action<string, FV, int>? Tap { get; set; }

    public FlowDiT(Accelerator accelerator, DiTKernels kernels, GpuWeights weights)
    {
        _acc = accelerator; _k = kernels; _w = weights;
        var a = weights.Args;
        _inCh = a.GetProperty("in_channels").GetInt32();
        _outCh = a.GetProperty("out_channels").GetInt32();
        _dim = a.GetProperty("model_channels").GetInt32();
        _condCh = a.GetProperty("cond_channels").GetInt32();
        _blocks = a.GetProperty("num_blocks").GetInt32();
        _heads = a.GetProperty("num_heads").GetInt32();
        _headDim = _dim / _heads;
        _hidden = (int)(_dim * a.GetProperty("mlp_ratio").GetDouble());
        // Only the configuration the released checkpoints use is implemented; anything else must fail loudly.
        if (a.GetProperty("pe_mode").GetString() != "rope" || !a.GetProperty("share_mod").GetBoolean()
            || !a.GetProperty("qk_rms_norm").GetBoolean() || !a.GetProperty("qk_rms_norm_cross").GetBoolean())
            throw new NotSupportedException("FlowDiT supports pe_mode=rope, share_mod, qk_rms_norm, qk_rms_norm_cross only");
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
    /// Fix the token set: <paramref name="coords"/> is [N,3] integer voxel coordinates (the RoPE positions). Builds the
    /// RoPE phase table and sizes the work buffers; the cross-attention length is set by the first Forward.
    /// </summary>
    public void SetTokens(IV coords, int n)
    {
        FreeWorkspace();
        _n = n; _condTokens = -1;
        _phases = _acc.Allocate1D<float>((long)n * (_headDim / 2) * 2);
        _k.RopePhases(coords, _phases.View, n, _headDim);
        _acc.Fence();   // the caller may free `coords` right after this returns
    }

    private void EnsureWorkspace(int condTokens)
    {
        if (_n < 0) throw new InvalidOperationException("SetTokens must be called before Forward");
        if (condTokens == _condTokens) return;
        // keep the phase table; rebuild the activations for the new cross-attention length
        foreach (var b in _ws) b.Dispose();
        _ws.Clear();
        _condTokens = condTokens;
        int N = _n, C = _dim;
        _tFreq = Alloc(256); _tHidden = Alloc(C); _tEmb = Alloc(C); _tAct = Alloc(C); _mod = Alloc(6 * C); _modv = Alloc(6 * C);
        _h = Alloc((long)N * C); _hn = Alloc((long)N * C); _qkv = Alloc((long)N * 3 * C); _q = Alloc((long)N * C);
        _kv = Alloc((long)condTokens * 2 * C); _attn = Alloc((long)N * C); _proj = Alloc((long)N * C);
        _mlp = Alloc((long)N * _hidden);
        // the reference attention materialises H*N*max(N, condTokens) scores; the flash kernel needs none
        _scores = Alloc(_k.UsesFlashAttention(_headDim) ? 1 : (long)_heads * N * Math.Max(N, condTokens));
    }

    /// <summary>
    /// One velocity prediction. <paramref name="x"/> is [N, inCh] token-major, <paramref name="cond"/> [condTokens, condCh],
    /// <paramref name="o"/> receives [N, outCh]. <paramref name="t"/> is the model time (the sampler passes 1000 * t).
    /// </summary>
    public void Forward(FV x, float t, FV cond, int condTokens, FV o)
    {
        EnsureWorkspace(condTokens);
        int N = _n, C = _dim, H = _heads, D = _headDim;

        // timestep -> shared modulation (TimestepEmbedder, then adaLN_modulation = SiLU -> Linear)
        _tFreq.CopyFromCPU(SinusoidalEmbedding(t, 256));
        _k.Linear(_tFreq, _w.BF16("t_embedder.mlp.0.weight"), _w.BF16("t_embedder.mlp.0.bias"), _tHidden, 1, 256, C, DiTKernels.Act.Silu);
        _k.Linear(_tHidden, _w.BF16("t_embedder.mlp.2.weight"), _w.BF16("t_embedder.mlp.2.bias"), _tEmb, 1, C, C);
        Tap?.Invoke("t_emb", _tEmb, C);
        _k.Silu(_tEmb, _tAct, C);
        _k.Linear(_tAct, _w.BF16("adaLN_modulation.1.weight"), _w.BF16("adaLN_modulation.1.bias"), _mod, 1, C, 6 * C);

        _k.Linear(x, _w.BF16("input_layer.weight"), _w.BF16("input_layer.bias"), _h, N, _inCh, C);
        Tap?.Invoke("input_layer", _h, N * C);

        for (int b = 0; b < _blocks; b++)
        {
            string p = $"blocks.{b}.";
            _k.AddBf16(_w.BF16(p + "modulation"), _mod, _modv, 6 * C);   // shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

            // self attention
            T("norm+mod", () => _k.LayerNormModulate(_h, _hn, _modv, 0 * C, 1 * C, N, C, 1e-6f));
            T("linear qkv", () => _k.Linear(_hn, _w.BF16(p + "self_attn.to_qkv.weight"), _w.BF16(p + "self_attn.to_qkv.bias"), _qkv, N, C, 3 * C));
            T("head norm", () => { _k.HeadNorm(_qkv, _w.BF16(p + "self_attn.q_rms_norm.gamma"), _phases!.View, N, 3 * C, 0 * C, H, D);
                                   _k.HeadNorm(_qkv, _w.BF16(p + "self_attn.k_rms_norm.gamma"), _phases.View, N, 3 * C, 1 * C, H, D); });
            T("self attention", () => _k.Attention(_qkv, 3 * C, 0, _qkv, 3 * C, 1 * C, 2 * C, _scores, _attn, N, N, H, D));
            T("linear out", () => _k.Linear(_attn, _w.BF16(p + "self_attn.to_out.weight"), _w.BF16(p + "self_attn.to_out.bias"), _proj, N, C, C));
            T("gate residual", () => _k.GateResidual(_h, _proj, _modv, 2 * C, N * C, C));

            // cross attention (no modulation, no RoPE; norm2 is the affine one)
            T("norm affine", () => _k.LayerNormAffine(_h, _hn, _w.BF16(p + "norm2.weight"), _w.BF16(p + "norm2.bias"), N, C, 1e-6f));
            T("linear q", () => _k.Linear(_hn, _w.BF16(p + "cross_attn.to_q.weight"), _w.BF16(p + "cross_attn.to_q.bias"), _q, N, C, C));
            T("linear kv", () => _k.Linear(cond, _w.BF16(p + "cross_attn.to_kv.weight"), _w.BF16(p + "cross_attn.to_kv.bias"), _kv, condTokens, _condCh, 2 * C));
            T("head norm", () => { _k.HeadNorm(_q, _w.BF16(p + "cross_attn.q_rms_norm.gamma"), null, N, C, 0, H, D);
                                   _k.HeadNorm(_kv, _w.BF16(p + "cross_attn.k_rms_norm.gamma"), null, condTokens, 2 * C, 0, H, D); });
            T("cross attention", () => _k.Attention(_q, C, 0, _kv, 2 * C, 0, C, _scores, _attn, N, condTokens, H, D));
            T("linear out", () => _k.Linear(_attn, _w.BF16(p + "cross_attn.to_out.weight"), _w.BF16(p + "cross_attn.to_out.bias"), _proj, N, C, C));
            T("gate residual", () => _k.GateResidual(_h, _proj, _modv, -1, N * C, C));

            // feed forward
            T("norm+mod", () => _k.LayerNormModulate(_h, _hn, _modv, 3 * C, 4 * C, N, C, 1e-6f));
            T("linear mlp1", () => _k.Linear(_hn, _w.BF16(p + "mlp.mlp.0.weight"), _w.BF16(p + "mlp.mlp.0.bias"), _mlp, N, C, _hidden, DiTKernels.Act.GeluTanh));
            T("linear mlp2", () => _k.Linear(_mlp, _w.BF16(p + "mlp.mlp.2.weight"), _w.BF16(p + "mlp.mlp.2.bias"), _proj, N, _hidden, C));
            T("gate residual", () => _k.GateResidual(_h, _proj, _modv, 5 * C, N * C, C));

            if (Tap != null)
            {
                if (b == 0) Tap("block0", _h, N * C);
                if (b == _blocks - 1) Tap("block29", _h, N * C);
            }
        }

        // out: F.layer_norm (torch default eps 1e-5, no affine), then out_layer
        _k.LayerNormPlain(_h, _hn, N, C, 1e-5f);
        _k.Linear(_hn, _w.BF16("out_layer.weight"), _w.BF16("out_layer.bias"), o, N, C, _outCh);
        _acc.Fence();
    }

    /// <summary>TimestepEmbedder.timestep_embedding: [cos(t*f), sin(t*f)], f = exp(-ln(10000) * i / half). Host scalar math (256 floats).</summary>
    internal static float[] SinusoidalEmbedding(float t, int dim, float maxPeriod = 10000f)
    {
        int half = dim / 2;
        var e = new float[dim];
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-MathF.Log(maxPeriod) * i / half);
            float a = t * freq;
            e[i] = MathF.Cos(a);
            e[half + i] = MathF.Sin(a);
        }
        return e;
    }

    public void Dispose() => FreeWorkspace();
}
