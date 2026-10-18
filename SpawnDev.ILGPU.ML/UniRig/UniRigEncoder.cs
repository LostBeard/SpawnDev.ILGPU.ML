using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.UniRig;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// UniRig's shape encoder, the conditioning of both the skeleton stage (and half of the skin stage): a point cloud with normals becomes 1024 tokens of width 1024.
/// Re-implemented from the checkpoint's tensor names and shapes (tools/unirig-harness/reference/torch_encoder.py is the readable spec and matches the reference's
/// latents to relRMS 1.1e-6 in float32):
/// <code>
/// x     = input_proj([fourier(xyz) | normal])                         per point, 54 -> 512
/// q     = x[queryRows]                                                the 1024 query tokens (farthest-point-sampled points)
/// q     = q + c_proj(attention(c_q(ln_1(q)), c_kv(ln_2(x))))          cross-attention over ALL points, 8 heads of 64
/// q     = q + mlp(ln_3(q))                                            c_fc 512 -> 2048, GELU (erf), c_proj 2048 -> 512
/// 16 x  { q += c_proj(attention(c_qkv(ln_1(q)))) ;  q += mlp(ln_2(q)) }
/// out   = output_proj(ln_post(q))                                     512 -> 1024
/// </code>
/// Weights are bf16 with fp32 activations and accumulation (<see cref="DiTKernels"/>); the fused projections were regrouped to [q | k | v] blocks by
/// export_weights.py, which is what <see cref="DiTKernels.Attention"/> reads.
/// </summary>
public sealed class UniRigEncoder : IDisposable
{
    public const int Heads = 8, Width = 512, SelfLayers = 16, QueryTokens = 1024, OutWidth = 1024, HeadDim = Width / Heads;

    private readonly Accelerator _acc;
    private readonly GpuWeights _w;
    private readonly DiTKernels _k;
    private readonly UniRigKernels _u;
    private readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _ws = new();
    private int _n = -1;
    private MemoryBuffer1D<float, Stride1D.Dense> _feat = null!, _x = null!, _kn = null!, _kv = null!, _q = null!, _h = null!, _qkv = null!, _attn = null!, _proj = null!,
        _mlp = null!, _scores = null!, _dummy = null!, _out = null!;
    private MemoryBuffer1D<int, Stride1D.Dense>? _rows;

    /// <param name="weights">The skeleton checkpoint exported by export_weights.py (the encoder's tensors and output_proj are used).</param>
    public UniRigEncoder(Accelerator accelerator, GpuWeights weights)
    {
        _acc = accelerator; _w = weights;
        _k = new DiTKernels(accelerator); _u = new UniRigKernels(accelerator);
    }

    private MemoryBuffer1D<float, Stride1D.Dense> Alloc(long n) { var b = _acc.Allocate1D<float>(Math.Max(1, n)); _ws.Add(b); return b; }

    private void EnsureWorkspace(int n)
    {
        if (n == _n) return;
        foreach (var b in _ws) b.Dispose();
        _ws.Clear();
        _n = n;
        int C = Width, Nq = QueryTokens;
        _feat = Alloc((long)n * UniRigKernels.FourierWidth); _x = Alloc((long)n * C); _kn = Alloc((long)n * C); _kv = Alloc((long)n * 2 * C);
        _q = Alloc((long)Nq * C); _h = Alloc((long)Nq * C); _qkv = Alloc((long)Nq * 3 * C); _attn = Alloc((long)Nq * C); _proj = Alloc((long)Nq * C);
        _mlp = Alloc((long)Nq * 4 * C); _dummy = Alloc(1); _out = Alloc((long)Nq * OutWidth);
        // the reference attention materialises H*Nq*Nk scores; the flash kernels (CUDA tensor cores, portable tiled) need none
        _scores = Alloc(_k.UsesFlashAttention(HeadDim) ? 1 : (long)Heads * Nq * n);
        _rows ??= _acc.Allocate1D<int>(Nq);
    }

    private void LayerNorm(FV x, FV o, string name, int rows)
        => _k.LayerNormAffine(x, o, _w.BF16(name + ".weight"), _w.BF16(name + ".bias"), rows, Width, 1e-5f);

    private void Mlp(FV x, FV hidden, FV o, string name, int rows)
    {
        _k.Linear(x, _w.BF16(name + ".c_fc.weight"), _w.BF16(name + ".c_fc.bias"), hidden, rows, Width, 4 * Width, DiTKernels.Act.GeluErf);
        _k.Linear(hidden, _w.BF16(name + ".c_proj.weight"), _w.BF16(name + ".c_proj.bias"), o, rows, 4 * Width, Width);
    }

    /// <summary>
    /// <paramref name="pts"/> and <paramref name="nrm"/> are [n,3] (the mesh scaled into [-1,1], unit normals); <paramref name="queryRows"/> are the 1024 indices of the
    /// points that become query tokens (the reference's farthest-point-sampled subset of 4096 randomly pre-sampled points). Returns [1024, 1024] on the device;
    /// the buffer belongs to this encoder and is overwritten by the next call.
    /// </summary>
    public MemoryBuffer1D<float, Stride1D.Dense> Encode(FV pts, FV nrm, int n, int[] queryRows)
    {
        if (queryRows.Length != QueryTokens) throw new ArgumentException($"expected {QueryTokens} query rows", nameof(queryRows));
        EnsureWorkspace(n);
        int C = Width, Nq = QueryTokens, H = Heads, D = HeadDim;
        _rows!.CopyFromCPU(queryRows);

        _u.FourierEmbed(pts, nrm, _feat.View, n);
        _k.Linear(_feat.View, _w.BF16("mesh_encoder.encoder.input_proj.weight"), _w.BF16("mesh_encoder.encoder.input_proj.bias"), _x.View, n, UniRigKernels.FourierWidth, C);
        _u.GatherRows(_x.View, _rows.View, _q.View, Nq, C);

        // cross-attention: 1024 queries over every point
        const string ca = "mesh_encoder.encoder.cross_attn.";
        LayerNorm(_q.View, _h.View, ca + "ln_1", Nq);
        _k.Linear(_h.View, _w.BF16(ca + "attn.c_q.weight"), null, _attn.View, Nq, C, C);                  // queries (borrow _attn as scratch)
        LayerNorm(_x.View, _kn.View, ca + "ln_2", n);
        _k.Linear(_kn.View, _w.BF16(ca + "attn.c_kv.weight"), null, _kv.View, n, C, 2 * C);
        _k.Attention(_attn.View, C, 0, _kv.View, 2 * C, 0, C, _scores.View, _proj.View, Nq, n, H, D);
        _k.Linear(_proj.View, _w.BF16(ca + "attn.c_proj.weight"), _w.BF16(ca + "attn.c_proj.bias"), _h.View, Nq, C, C);
        _k.GateResidual(_q.View, _h.View, _dummy.View, -1, Nq * C, C);
        LayerNorm(_q.View, _h.View, ca + "ln_3", Nq);
        Mlp(_h.View, _mlp.View, _proj.View, ca + "mlp", Nq);
        _k.GateResidual(_q.View, _proj.View, _dummy.View, -1, Nq * C, C);

        for (int i = 0; i < SelfLayers; i++)
        {
            string b = $"mesh_encoder.encoder.self_attn.resblocks.{i}.";
            LayerNorm(_q.View, _h.View, b + "ln_1", Nq);
            _k.Linear(_h.View, _w.BF16(b + "attn.c_qkv.weight"), null, _qkv.View, Nq, C, 3 * C);
            _k.Attention(_qkv.View, 3 * C, 0, _qkv.View, 3 * C, C, 2 * C, _scores.View, _attn.View, Nq, Nq, H, D);
            _k.Linear(_attn.View, _w.BF16(b + "attn.c_proj.weight"), _w.BF16(b + "attn.c_proj.bias"), _proj.View, Nq, C, C);
            _k.GateResidual(_q.View, _proj.View, _dummy.View, -1, Nq * C, C);
            LayerNorm(_q.View, _h.View, b + "ln_2", Nq);
            Mlp(_h.View, _mlp.View, _proj.View, b + "mlp", Nq);
            _k.GateResidual(_q.View, _proj.View, _dummy.View, -1, Nq * C, C);
        }

        LayerNorm(_q.View, _h.View, "mesh_encoder.encoder.ln_post", Nq);
        _k.Linear(_h.View, _w.BF16("output_proj.weight"), _w.BF16("output_proj.bias"), _out.View, Nq, C, OutWidth);
        return _out;
    }

    public void Dispose()
    {
        foreach (var b in _ws) b.Dispose();
        _ws.Clear(); _rows?.Dispose(); _rows = null; _n = -1;
    }
}
