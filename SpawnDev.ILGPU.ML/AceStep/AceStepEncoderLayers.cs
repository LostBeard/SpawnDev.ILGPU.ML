using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;
using BV = ILGPU.Runtime.ArrayView1D<ILGPU.BFloat16, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.AceStep;

/// <summary>
/// ACE-Step 1.5's encoder layer (<c>AceStepEncoderLayer</c>: the lyric and timbre encoders and the audio-code detokenizer), run in place on h [L, 2048]:
/// h += o(attn(rope(qk_norm(q, k, v of RMS(h))))); h += down(silu(gate(RMS'(h))) up(RMS'(h))). Bidirectional GQA 16 / 8 x 128 with an optional window and an optional
/// group mask (independent sequences packed back to back), RoPE from the caller's per-row tables. The scratch is sized once for <c>maxTokens</c> rows.
/// </summary>
internal sealed class AceStepEncoderLayers : IDisposable
{
    public const int Dim = 2048, Heads = 16, KvHeads = 8, HeadDim = 128, Mlp = 6144;
    private const float Eps = 1e-6f;

    private readonly GpuWeights _w;
    private readonly DiTKernels _k;
    private readonly SkinTokensKernels _u;
    private readonly AceStepKernels _a;
    private readonly int _maxTokens;
    private readonly List<IDisposable> _buffers = new();
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _n, _q, _kk, _v, _att, _o, _g, _up, _gu, _scores;

    public DiTKernels Kernels => _k;

    public AceStepEncoderLayers(Accelerator acc, GpuWeights w, int maxTokens)
    {
        _w = w; _maxTokens = maxTokens;
        _k = new DiTKernels(acc) { PreciseActivations = true }; _u = new SkinTokensKernels(acc); _a = new AceStepKernels(acc);
        MemoryBuffer1D<float, Stride1D.Dense> A(long n) { var b = acc.Allocate1D<float>(Math.Max(1, n)); _buffers.Add(b); return b; }
        long N = maxTokens, kv = KvHeads * HeadDim;
        _n = A(N * Dim); _q = A(N * Dim); _kk = A(N * kv); _v = A(N * kv); _att = A(N * Dim); _o = A(N * Dim);
        _g = A(N * Mlp); _up = A(N * Mlp); _gu = A(N * Mlp); _scores = A((long)Heads * N * N);
    }

    /// <summary>Per-row rotate_half tables for positions <paramref name="positions"/> (Qwen3RotaryEmbedding, theta 1e6, float32 arguments as the reference evaluates them).</summary>
    public static (float[] Cos, float[] Sin) RopeTables(IReadOnlyList<int> positions, float theta = 1e6f)
    {
        int half = HeadDim / 2, n = positions.Count;
        var c = new float[n * half]; var s = new float[n * half];
        for (int k = 0; k < half; k++)
        {
            float inv = 1f / MathF.Pow(theta, 2f * k / HeadDim);
            for (int r = 0; r < n; r++) { float ang = positions[r] * inv; c[r * half + k] = (float)Math.Cos(ang); s[r * half + k] = (float)Math.Sin(ang); }
        }
        return (c, s);
    }

    private BV B(string name) => _w.BF16(name);
    private void Lin(FV x, string name, FV o, int rows, int K, int N) => _k.Linear(x, B(name + ".weight"), null, o, rows, K, N);

    /// <summary>One layer (<paramref name="b"/> = its tensor prefix, ending in ".") on <paramref name="h"/> [L, 2048] in place.</summary>
    public void Run(string b, FV h, int L, FV cos, FV sin, int window = -1, int group = 0)
    {
        if (L > _maxTokens) throw new ArgumentException($"{L} rows: the layers were sized for {_maxTokens}");
        int D = Dim, kv = KvHeads * HeadDim;
        _u.RmsNorm(h, _n.View, B(b + "input_layernorm.weight"), L, D, Eps);
        Lin(_n.View, b + "self_attn.q_proj", _q.View, L, D, D);
        Lin(_n.View, b + "self_attn.k_proj", _kk.View, L, D, kv);
        Lin(_n.View, b + "self_attn.v_proj", _v.View, L, D, kv);
        _u.RmsNormHeads(_q.View, B(b + "self_attn.q_norm.weight"), L, D, 0, Heads, HeadDim, Eps);
        _u.RmsNormHeads(_kk.View, B(b + "self_attn.k_norm.weight"), L, kv, 0, KvHeads, HeadDim, Eps);
        _u.RopeTables(_q.View, cos, sin, L, D, 0, Heads, HeadDim);
        _u.RopeTables(_kk.View, cos, sin, L, kv, 0, KvHeads, HeadDim);
        _a.Gqa(_q.View, D, _kk.View, kv, _v.View, kv, _scores.View, _att.View, L, L, Heads, HeadDim, Heads / KvHeads, window, group);
        Lin(_att.View, b + "self_attn.o_proj", _o.View, L, D, D);
        _k.GateResidual(h, _o.View, _o.View, -1, L * D, D);
        _u.RmsNorm(h, _n.View, B(b + "post_attention_layernorm.weight"), L, D, Eps);
        Lin(_n.View, b + "mlp.gate_proj", _g.View, L, D, Mlp);
        Lin(_n.View, b + "mlp.up_proj", _up.View, L, D, Mlp);
        _a.SiluMul(_g.View, _up.View, _gu.View, L * Mlp);
        Lin(_gu.View, b + "mlp.down_proj", _o.View, L, Mlp, D);
        _k.GateResidual(h, _o.View, _o.View, -1, L * D, D);
    }

    public void Dispose()
    {
        foreach (var b in _buffers) b.Dispose();
        _buffers.Clear();
    }
}
