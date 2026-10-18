using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.SkinTokens;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>
/// The skin-weight VAE of SkinTokens (an FSQ conditional VAE built from Tripo2 transformer blocks), decode side only:
/// <code>
/// cond   = cond_quant(norm_out(selfx2(crossattn(proj_in(embed(384 sampled points)), proj_in(embed(all N points))))))         [384, 512]   per mesh
/// z      = project_out(FSQ digits of the 4 skin tokens of a bone)                                                             [4, 512]     per bone
/// kv     = 10 x selfattn-block(post_quant([z | cond]))                                                                        [388, 768]
/// weight = sigmoid(proj_out(norm_out(ff(crossattn(proj_query(embed(point, normal)), kv)))))                                   per bone, per point
/// </code>
/// A block is pre-LayerNorm (eps 1e-5): <c>h += attn(ln(h)); [h += cross(ln2(h), ln_cross(enc));] h += ff(ln3(h))</c> with a GELU(erf) feed-forward of 3072. The
/// attention weights are stored head-regrouped by export_weights.py (the Tripo2 processor interleaves q, k, v per head). bf16 weights, fp32 activations.
/// The bone-independent half of the last block (the query embedding, its layer norm and q projection for all N points) is computed once per mesh.
/// </summary>
public sealed class SkinTokensVae : IDisposable
{
    public const int Width = 768, Heads = 12, HeadDim = 64, Ffn = 3072, Latent = 512, CondTokens = 384, TokensPerBone = 4, SelfBlocks = 10, ChunkRows = 6000;
    public const float Eps = 1e-5f;

    private readonly Accelerator _acc;
    private readonly GpuWeights _w;
    private readonly DiTKernels _k;
    private readonly SkinTokensKernels _u;
    private readonly List<IDisposable> _owned = new();
    private readonly bool _flash;
    private MemoryBuffer1D<float, Stride1D.Dense>? _cond, _condPost, _q0, _qn;
    private int _n = -1;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _dummy;

    /// <summary>Diagnostics: when set, called with (name, a view of the stage's output, rows, cols) at the stages tools/unirig-harness/reference/golden_skintokens.py records. Costs nothing when null.</summary>
    public Action<string, FV, int, int>? Probe { get; set; }
    private void P(string name, FV v, int rows, int cols) { if (Probe != null) { _acc.Synchronize(); Probe(name, v, rows, cols); } }

    public SkinTokensVae(Accelerator accelerator, GpuWeights weights)
    {
        _acc = accelerator; _w = weights;
        _k = new DiTKernels(accelerator); _u = new SkinTokensKernels(accelerator);
        _flash = _k.UsesFlashAttention(HeadDim);
        _dummy = Alloc(1);
    }

    private MemoryBuffer1D<float, Stride1D.Dense> Alloc(long n) { var b = _acc.Allocate1D<float>(Math.Max(1, n)); _owned.Add(b); return b; }
    private MemoryBuffer1D<float, Stride1D.Dense> Scores(long n) => Alloc(_flash ? 1 : n);

    private void Ln(FV x, FV o, string name, int rows) => _k.LayerNormAffine(x, o, _w.BF16(name + ".weight"), _w.BF16(name + ".bias"), rows, Width, Eps);

    /// <summary>h += ff(ln3(h)) for [rows, 768].</summary>
    private void FeedForward(FV h, string p, int rows, FV hn, FV ffHidden, FV proj)
    {
        Ln(h, hn, p + "norm3", rows);
        _k.Linear(hn, _w.BF16(p + "ff.net.0.proj.weight"), _w.BF16(p + "ff.net.0.proj.bias"), ffHidden, rows, Width, Ffn, DiTKernels.Act.GeluErf);
        _k.Linear(ffHidden, _w.BF16(p + "ff.net.2.weight"), _w.BF16(p + "ff.net.2.bias"), proj, rows, Ffn, Width);
        _k.GateResidual(h, proj, _dummy.View, -1, rows * Width, Width);
    }

    /// <summary>A self-attention block over [T, 768] in place.</summary>
    private void SelfBlock(FV h, string p, int T)
    {
        using var hn = _acc.Allocate1D<float>((long)T * Width); using var qkv = _acc.Allocate1D<float>((long)T * 3 * Width); using var attn = _acc.Allocate1D<float>((long)T * Width);
        using var proj = _acc.Allocate1D<float>((long)T * Width); using var ff = _acc.Allocate1D<float>((long)T * Ffn);
        var scores = _flash ? null : _acc.Allocate1D<float>((long)Heads * T * T);
        Ln(h, hn.View, p + "norm1", T);
        _k.Linear(hn.View, _w.BF16(p + "attn1.qkv.weight"), null, qkv.View, T, Width, 3 * Width);
        _k.Attention(qkv.View, 3 * Width, 0, qkv.View, 3 * Width, Width, 2 * Width, scores?.View ?? _dummy.View, attn.View, T, T, Heads, HeadDim);
        _k.Linear(attn.View, _w.BF16(p + "attn1.to_out.0.weight"), _w.BF16(p + "attn1.to_out.0.bias"), proj.View, T, Width, Width);
        _k.GateResidual(h, proj.View, _dummy.View, -1, T * Width, Width);
        FeedForward(h, p, T, hn.View, ff.View, proj.View);
        _acc.Synchronize(); scores?.Dispose();
    }

    /// <summary>
    /// The conditioning of a mesh: 384 query points (the farthest-point-sampled subset of a random pre-sample, <paramref name="queryRows"/> index the cloud) attend over
    /// all n points. <paramref name="points"/> and <paramref name="normals"/> are [n,3] on the device. Returns [384, 512]; the buffer belongs to this object.
    /// </summary>
    public MemoryBuffer1D<float, Stride1D.Dense> EncodeCond(FV points, FV normals, int n, int[] queryRows)
    {
        if (queryRows.Length != CondTokens) throw new ArgumentException($"expected {CondTokens} query rows", nameof(queryRows));
        const string e = "vae.model.cond_encoder.";
        using var rows = _acc.Allocate1D<int>(CondTokens); rows.CopyFromCPU(queryRows);
        using var featKv = _acc.Allocate1D<float>((long)n * SkinTokensKernels.PointWidth); using var featQ = _acc.Allocate1D<float>((long)CondTokens * SkinTokensKernels.PointWidth);
        using var ptsQ = _acc.Allocate1D<float>(CondTokens * 3); using var nrmQ = _acc.Allocate1D<float>(CondTokens * 3);
        _u.PointEmbed(points, normals, featKv.View, n);
        _u.GatherRows(points, rows.View, ptsQ.View, CondTokens, 3); _u.GatherRows(normals, rows.View, nrmQ.View, CondTokens, 3);
        _u.PointEmbed(ptsQ.View, nrmQ.View, featQ.View, CondTokens);
        using var enc = _acc.Allocate1D<float>((long)n * Width); using var encN = _acc.Allocate1D<float>((long)n * Width); using var kv = _acc.Allocate1D<float>((long)n * 2 * Width);
        using var h = _acc.Allocate1D<float>((long)CondTokens * Width); using var hn = _acc.Allocate1D<float>((long)CondTokens * Width); using var q = _acc.Allocate1D<float>((long)CondTokens * Width);
        using var attn = _acc.Allocate1D<float>((long)CondTokens * Width); using var proj = _acc.Allocate1D<float>((long)CondTokens * Width); using var ff = _acc.Allocate1D<float>((long)CondTokens * Ffn);
        _k.Linear(featKv.View, _w.BF16(e + "proj_in.weight"), _w.BF16(e + "proj_in.bias"), enc.View, n, SkinTokensKernels.PointWidth, Width);
        _k.Linear(featQ.View, _w.BF16(e + "proj_in.weight"), _w.BF16(e + "proj_in.bias"), h.View, CondTokens, SkinTokensKernels.PointWidth, Width);
        P("cond.proj_in.q", h.View, CondTokens, Width); P("cond.proj_in.kv", enc.View, Math.Min(n, 3616), Width);

        // block 0: cross-attention of the 384 queries over every point
        string p = e + "blocks.0.";
        Ln(h.View, hn.View, p + "norm2", CondTokens);
        _k.Linear(hn.View, _w.BF16(p + "attn2.to_q.weight"), null, q.View, CondTokens, Width, Width);
        Ln(enc.View, encN.View, p + "attn2.norm_cross", n);
        _k.Linear(encN.View, _w.BF16(p + "attn2.kv.weight"), null, kv.View, n, Width, 2 * Width);
        using (var scores = _flash ? null : _acc.Allocate1D<float>((long)Heads * CondTokens * n))
            _k.Attention(q.View, Width, 0, kv.View, 2 * Width, 0, Width, scores?.View ?? _dummy.View, attn.View, CondTokens, n, Heads, HeadDim);
        _k.Linear(attn.View, _w.BF16(p + "attn2.to_out.0.weight"), _w.BF16(p + "attn2.to_out.0.bias"), proj.View, CondTokens, Width, Width);
        _k.GateResidual(h.View, proj.View, _dummy.View, -1, CondTokens * Width, Width);
        FeedForward(h.View, p, CondTokens, hn.View, ff.View, proj.View);
        _acc.Synchronize();
        P("cond.block0", h.View, CondTokens, Width);
        for (int b = 1; b <= 2; b++) { SelfBlock(h.View, e + $"blocks.{b}.", CondTokens); P($"cond.block{b}", h.View, CondTokens, Width); }
        Ln(h.View, hn.View, e + "norm_out", CondTokens);
        P("cond.norm_out", hn.View, CondTokens, Width);
        _cond ??= Alloc((long)CondTokens * Latent);
        _k.Linear(hn.View, _w.BF16("vae.model.cond_quant.weight"), _w.BF16("vae.model.cond_quant.bias"), _cond.View, CondTokens, Width, Latent);
        _acc.Synchronize();
        ProjectCond();
        return _cond;
    }

    /// <summary>Use [384, 512] cond latents computed elsewhere (a golden, or a cache) instead of running <see cref="EncodeCond"/>.</summary>
    public void SetCond(FV latents)
    {
        _cond ??= Alloc((long)CondTokens * Latent);
        _cond.View.CopyFrom(latents.SubView(0, (long)CondTokens * Latent));
        ProjectCond();
    }

    // the decoder reads post_quant([z | cond]); post_quant is row-wise, so the cond half is projected once
    private void ProjectCond()
    {
        _condPost ??= Alloc((long)CondTokens * Width);
        _k.Linear(_cond!.View, _w.BF16("vae.model.post_quant.weight"), _w.BF16("vae.model.post_quant.bias"), _condPost.View, CondTokens, Latent, Width);
        _acc.Synchronize();
    }

    /// <summary>
    /// Per-mesh preparation of the skin decoder: the query embedding (proj_query of the point/normal embedding) and its layer norm + q projection for all n points,
    /// which do not depend on the bone. <paramref name="points"/> and <paramref name="normals"/> are the n sampled points the weights are predicted at.
    /// </summary>
    public void BeginSkin(FV points, FV normals, int n)
    {
        const string d = "vae.model.decoder.", p = d + "blocks.10.";
        _n = n;
        _q0?.Dispose(); _qn?.Dispose(); _owned.Remove(_q0!); _owned.Remove(_qn!);
        _q0 = Alloc((long)n * Width); _qn = Alloc((long)n * Width);
        using var feat = _acc.Allocate1D<float>((long)ChunkRows * SkinTokensKernels.PointWidth); using var hn = _acc.Allocate1D<float>((long)ChunkRows * Width);
        for (int r0 = 0; r0 < n; r0 += ChunkRows)
        {
            int m = Math.Min(ChunkRows, n - r0);
            _u.PointEmbed(points.SubView((long)r0 * 3, (long)m * 3), normals.SubView((long)r0 * 3, (long)m * 3), feat.View, m);
            var q0 = _q0.View.SubView((long)r0 * Width, (long)m * Width);
            _k.Linear(feat.View, _w.BF16(d + "proj_query.weight"), _w.BF16(d + "proj_query.bias"), q0, m, SkinTokensKernels.PointWidth, Width);
            if (r0 == 0) P("dec.proj_query", q0, Math.Min(m, 4000), Width);
            Ln(q0, hn.View, p + "norm2", m);
            _k.Linear(hn.View, _w.BF16(p + "attn2.to_q.weight"), null, _qn.View.SubView((long)r0 * Width, (long)m * Width), m, Width, Width);
            _acc.Synchronize();
        }
    }

    /// <summary>The FSQ latents [4, 512] and the decoder state after the 10 self-attention blocks [388, 768] of one bone (the keys / values the query block reads).</summary>
    public (float[] Z, float[] State) BoneState(int[] tokens)
    {
        using var h = BoneStateBuffer(tokens, out var z);
        using (z) { _acc.Synchronize(); return (z.GetAsArray1D(), h.GetAsArray1D()); }
    }

    private MemoryBuffer1D<float, Stride1D.Dense> BoneStateBuffer(int[] tokens, out MemoryBuffer1D<float, Stride1D.Dense> z)
    {
        if (_condPost == null) throw new InvalidOperationException("EncodeCond or SetCond first");
        if (tokens.Length != TokensPerBone) throw new ArgumentException($"expected {TokensPerBone} tokens", nameof(tokens));
        const string d = "vae.model.decoder.";
        int T = TokensPerBone + CondTokens;
        using var ids = _acc.Allocate1D<int>(TokensPerBone); ids.CopyFromCPU(tokens);
        z = _acc.Allocate1D<float>(TokensPerBone * Latent); var h = _acc.Allocate1D<float>((long)T * Width);
        _u.FsqCodes(ids.View, _w.BF16("vae.model.FSQ.project_out.weight"), _w.BF16("vae.model.FSQ.project_out.bias"), z.View, TokensPerBone, Latent);
        _k.Linear(z.View, _w.BF16("vae.model.post_quant.weight"), _w.BF16("vae.model.post_quant.bias"), h.View.SubView(0, TokensPerBone * Width), TokensPerBone, Latent, Width);
        h.View.SubView((long)TokensPerBone * Width, (long)CondTokens * Width).CopyFrom(_condPost.View);
        for (int b = 0; b < SelfBlocks; b++) SelfBlock(h.View, d + $"blocks.{b}.", T);
        return h;
    }

    /// <summary>
    /// Decode one bone: <paramref name="tokens"/> are its 4 skin token ids minus the skeleton vocabulary (FSQ indices 0..32767). Writes sigmoid weights for every point
    /// of <see cref="BeginSkin"/> into <paramref name="skin"/> [n, columns] at column <paramref name="column"/>.
    /// </summary>
    public void DecodeBone(int[] tokens, FV skin, int column, int columns)
    {
        if (_q0 == null || _qn == null) throw new InvalidOperationException("BeginSkin first");
        const string d = "vae.model.decoder.", p = d + "blocks.10.";
        int T = TokensPerBone + CondTokens;
        using var h = BoneStateBuffer(tokens, out var z); z.Dispose();

        // the last block: the n query rows attend over the 388 state rows
        using var encN = _acc.Allocate1D<float>((long)T * Width); using var kv = _acc.Allocate1D<float>((long)T * 2 * Width);
        Ln(h.View, encN.View, p + "attn2.norm_cross", T);
        _k.Linear(encN.View, _w.BF16(p + "attn2.kv.weight"), null, kv.View, T, Width, 2 * Width);
        using var hc = _acc.Allocate1D<float>((long)ChunkRows * Width); using var hn = _acc.Allocate1D<float>((long)ChunkRows * Width);
        using var attn = _acc.Allocate1D<float>((long)ChunkRows * Width); using var proj = _acc.Allocate1D<float>((long)ChunkRows * Width); using var ff = _acc.Allocate1D<float>((long)ChunkRows * Ffn);
        using var scores = _flash ? null : _acc.Allocate1D<float>((long)Heads * ChunkRows * T);
        for (int r0 = 0; r0 < _n; r0 += ChunkRows)
        {
            int m = Math.Min(ChunkRows, _n - r0);
            _k.Attention(_qn.View.SubView((long)r0 * Width, (long)m * Width), Width, 0, kv.View, 2 * Width, 0, Width, scores?.View ?? _dummy.View, attn.View, m, T, Heads, HeadDim);
            _k.Linear(attn.View, _w.BF16(p + "attn2.to_out.0.weight"), _w.BF16(p + "attn2.to_out.0.bias"), proj.View, m, Width, Width);
            hc.View.SubView(0, (long)m * Width).CopyFrom(_q0.View.SubView((long)r0 * Width, (long)m * Width));
            _k.GateResidual(hc.View, proj.View, _dummy.View, -1, m * Width, Width);
            FeedForward(hc.View, p, m, hn.View, ff.View, proj.View);
            P("dec.block10", hc.View, Math.Min(m, 4000), Width);
            Ln(hc.View, hn.View, d + "norm_out", m);
            P("dec.norm_out", hn.View, Math.Min(m, 4000), Width);
            _u.SkinHead(hn.View, _w.BF16(d + "proj_out.weight"), _w.BF16(d + "proj_out.bias"), skin.SubView((long)r0 * columns, (long)m * columns), m, Width, column, columns);
            _acc.Synchronize();
        }
    }

    public void Dispose()
    {
        foreach (var o in _owned) o.Dispose();
        _owned.Clear();
    }
}
