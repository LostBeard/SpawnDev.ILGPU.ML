using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>
/// TRELLIS.2 sparse-structure flow DiT (trellis2/models/sparse_structure_flow.py): a dense 16^3 latent of 8
/// channels, 30 blocks of (adaLN self-attention with RoPE-3D and qk RMS norm, cross-attention to the image
/// features, GELU feed-forward), shared adaLN modulation. Batch 1 (classifier-free guidance runs the model twice).
/// Weights stay bf16 on the device; activations are fp32.
/// </summary>
public sealed class SparseStructureFlow : IDisposable
{
    private readonly Accelerator _acc;
    private readonly DiTKernels _k;
    private readonly GpuWeights _w;
    private readonly int _res, _inCh, _outCh, _dim, _condCh, _blocks, _heads, _headDim, _hidden;
    private MemoryBuffer1D<float, Stride1D.Dense>? _phases;

    /// <summary>Latent resolution per axis (16).</summary>
    public int Resolution => _res;
    public int InChannels => _inCh;
    public int OutChannels => _outCh;
    public int CondChannels => _condCh;

    /// <summary>Test hook: called with a stage name and the activation after it (valid only during the call).</summary>
    public Action<string, FV, int>? Tap { get; set; }

    public SparseStructureFlow(Accelerator accelerator, DiTKernels kernels, GpuWeights weights)
    {
        _acc = accelerator; _k = kernels; _w = weights;
        var a = weights.Args;
        _res = a.GetProperty("resolution").GetInt32();
        _inCh = a.GetProperty("in_channels").GetInt32();
        _outCh = a.GetProperty("out_channels").GetInt32();
        _dim = a.GetProperty("model_channels").GetInt32();
        _condCh = a.GetProperty("cond_channels").GetInt32();
        _blocks = a.GetProperty("num_blocks").GetInt32();
        _heads = a.GetProperty("num_heads").GetInt32();
        _headDim = _dim / _heads;
        _hidden = (int)(_dim * a.GetProperty("mlp_ratio").GetDouble());
        // The reference only implements the configuration the released checkpoint uses; refuse anything else loudly.
        if (a.GetProperty("pe_mode").GetString() != "rope" || !a.GetProperty("share_mod").GetBoolean()
            || !a.GetProperty("qk_rms_norm").GetBoolean() || !a.GetProperty("qk_rms_norm_cross").GetBoolean())
            throw new NotSupportedException("SparseStructureFlow supports pe_mode=rope, share_mod, qk_rms_norm, qk_rms_norm_cross only");
    }

    private MemoryBuffer1D<float, Stride1D.Dense> Alloc(long n) => _acc.Allocate1D<float>(n);

    /// <summary>
    /// One velocity prediction. <paramref name="x"/> is [inCh, res^3] (the reference's [1, C, R, R, R]),
    /// <paramref name="cond"/> is [condTokens, condCh]. <paramref name="o"/> receives [outCh, res^3].
    /// </summary>
    public void Forward(FV x, float t, FV cond, int condTokens, FV o)
    {
        int N = _res * _res * _res, C = _dim, H = _heads, D = _headDim;

        if (_phases == null)
        {
            using var coords = _acc.Allocate1D<int>(N * 3);
            _k.GridCoords(coords.View, _res);
            _phases = Alloc((long)N * (D / 2) * 2);
            _k.RopePhases(coords.View, _phases.View, N, D);
        }

        // --- timestep -> shared modulation (TimestepEmbedder, then adaLN_modulation = SiLU -> Linear) ---
        using var tFreq = _acc.Allocate1D(SinusoidalEmbedding(t, 256));
        using var tHidden = Alloc(C);
        using var tEmb = Alloc(C);
        using var tAct = Alloc(C);
        using var mod = Alloc(6 * C);
        _k.Linear(tFreq.View, _w.BF16("t_embedder.mlp.0.weight"), _w.BF16("t_embedder.mlp.0.bias"), tHidden.View, 1, 256, C, DiTKernels.Act.Silu);
        _k.Linear(tHidden.View, _w.BF16("t_embedder.mlp.2.weight"), _w.BF16("t_embedder.mlp.2.bias"), tEmb.View, 1, C, C);
        Tap?.Invoke("t_emb", tEmb.View, C);
        _k.Silu(tEmb.View, tAct.View, C);
        _k.Linear(tAct.View, _w.BF16("adaLN_modulation.1.weight"), _w.BF16("adaLN_modulation.1.bias"), mod.View, 1, C, 6 * C);

        // --- input layer: [inCh, N] -> tokens [N, inCh] -> [N, C] ---
        using var xTok = Alloc((long)N * _inCh);
        using var h = Alloc((long)N * C);
        _k.Transpose(x, xTok.View, _inCh, N);
        _k.Linear(xTok.View, _w.BF16("input_layer.weight"), _w.BF16("input_layer.bias"), h.View, N, _inCh, C);
        Tap?.Invoke("input_layer", h.View, N * C);

        using var modv = Alloc(6 * C);
        using var hn = Alloc((long)N * C);
        using var qkv = Alloc((long)N * 3 * C);
        using var q = Alloc((long)N * C);
        using var kv = Alloc((long)condTokens * 2 * C);
        using var attn = Alloc((long)N * C);
        using var proj = Alloc((long)N * C);
        using var mlp = Alloc((long)N * _hidden);
        using var scores = Alloc(_k.UsesFlashAttention(D) ? 1 : (long)H * N * Math.Max(N, condTokens));

        for (int b = 0; b < _blocks; b++)
        {
            string p = $"blocks.{b}.";
            _k.AddBf16(_w.BF16(p + "modulation"), mod.View, modv.View, 6 * C);   // shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

            // self attention
            _k.LayerNormModulate(h.View, hn.View, modv.View, 0 * C, 1 * C, N, C, 1e-6f);
            _k.Linear(hn.View, _w.BF16(p + "self_attn.to_qkv.weight"), _w.BF16(p + "self_attn.to_qkv.bias"), qkv.View, N, C, 3 * C);
            _k.HeadNorm(qkv.View, _w.BF16(p + "self_attn.q_rms_norm.gamma"), _phases.View, N, 3 * C, 0 * C, H, D);
            _k.HeadNorm(qkv.View, _w.BF16(p + "self_attn.k_rms_norm.gamma"), _phases.View, N, 3 * C, 1 * C, H, D);
            _k.Attention(qkv.View, 3 * C, 0, qkv.View, 3 * C, 1 * C, 2 * C, scores.View, attn.View, N, N, H, D);
            _k.Linear(attn.View, _w.BF16(p + "self_attn.to_out.weight"), _w.BF16(p + "self_attn.to_out.bias"), proj.View, N, C, C);
            _k.GateResidual(h.View, proj.View, modv.View, 2 * C, N * C, C);

            // cross attention (no modulation, no RoPE; norm2 is the affine one)
            _k.LayerNormAffine(h.View, hn.View, _w.BF16(p + "norm2.weight"), _w.BF16(p + "norm2.bias"), N, C, 1e-6f);
            _k.Linear(hn.View, _w.BF16(p + "cross_attn.to_q.weight"), _w.BF16(p + "cross_attn.to_q.bias"), q.View, N, C, C);
            _k.Linear(cond, _w.BF16(p + "cross_attn.to_kv.weight"), _w.BF16(p + "cross_attn.to_kv.bias"), kv.View, condTokens, _condCh, 2 * C);
            _k.HeadNorm(q.View, _w.BF16(p + "cross_attn.q_rms_norm.gamma"), null, N, C, 0, H, D);
            _k.HeadNorm(kv.View, _w.BF16(p + "cross_attn.k_rms_norm.gamma"), null, condTokens, 2 * C, 0, H, D);
            _k.Attention(q.View, C, 0, kv.View, 2 * C, 0, C, scores.View, attn.View, N, condTokens, H, D);
            _k.Linear(attn.View, _w.BF16(p + "cross_attn.to_out.weight"), _w.BF16(p + "cross_attn.to_out.bias"), proj.View, N, C, C);
            _k.GateResidual(h.View, proj.View, modv.View, -1, N * C, C);

            // feed forward
            _k.LayerNormModulate(h.View, hn.View, modv.View, 3 * C, 4 * C, N, C, 1e-6f);
            _k.Linear(hn.View, _w.BF16(p + "mlp.mlp.0.weight"), _w.BF16(p + "mlp.mlp.0.bias"), mlp.View, N, C, _hidden, DiTKernels.Act.GeluTanh);
            _k.Linear(mlp.View, _w.BF16(p + "mlp.mlp.2.weight"), _w.BF16(p + "mlp.mlp.2.bias"), proj.View, N, _hidden, C);
            _k.GateResidual(h.View, proj.View, modv.View, 5 * C, N * C, C);

            if (b == 0) Tap?.Invoke("block0", h.View, N * C);
            if (b == _blocks - 1) Tap?.Invoke("block29", h.View, N * C);
        }

        // --- out: F.layer_norm (torch default eps 1e-5, no affine), out_layer, back to [outCh, N] ---
        _k.LayerNormPlain(h.View, hn.View, N, C, 1e-5f);
        using var outTok = Alloc((long)N * _outCh);
        _k.Linear(hn.View, _w.BF16("out_layer.weight"), _w.BF16("out_layer.bias"), outTok.View, N, C, _outCh);
        _k.Transpose(outTok.View, o, N, _outCh);
        _acc.Synchronize();
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

    public void Dispose() => _phases?.Dispose();
}
