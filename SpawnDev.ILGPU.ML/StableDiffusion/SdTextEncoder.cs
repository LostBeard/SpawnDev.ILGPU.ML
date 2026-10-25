using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.StableDiffusion;

using FV = ArrayView1D<float, Stride1D.Dense>;
using Buf = MemoryBuffer1D<float, Stride1D.Dense>;

/// <summary>The text conditioning: the context the UNet's cross-attention reads ([Rows, Width] row-major) and, for SDXL, the pooled embedding of the second tower.</summary>
public sealed record SdTextEmbedding(float[] Context, int Rows, int Width, float[]? Pooled);

/// <summary>
/// One CLIP text transformer (pre-norm, causal self-attention, MLP) in the loader's normalised naming: <c>tok</c>, <c>pos</c>, <c>layers.N.{ln1,qkv,out,ln2,fc1,fc2}</c>,
/// <c>final_ln</c>, optional <c>proj</c>. OpenAI CLIP (SD 1.x, SDXL's first tower) uses quick GELU; OpenCLIP (SD 2.x's ViT-H, SDXL's bigG) the exact GELU.
/// </summary>
internal sealed class SdClipTower
{
    private readonly Accelerator _acc;
    private readonly SdWeights _w;
    private readonly SdKernels _k;
    public readonly string P;
    public readonly int Width, Layers, Heads, MaxTokens;
    public readonly bool QuickGelu;

    public SdClipTower(Accelerator acc, SdWeights w, SdKernels k, string prefix, bool quickGelu)
    {
        _acc = acc; _w = w; _k = k; P = prefix; QuickGelu = quickGelu;
        var tok = w.Shape(prefix + "tok");
        Width = tok[1];
        MaxTokens = w.Shape(prefix + "pos")[0];
        while (w.Has($"{prefix}layers.{Layers}.qkv.weight")) Layers++;
        Heads = Width / 64;
    }

    public bool HasProjection => _w.Has(P + "proj.weight");
    public int ProjectionWidth => _w.Shape(P + "proj.weight")[0];

    /// <summary>
    /// Run layers [<paramref name="fromLayer"/>, <paramref name="toLayer"/>) on <paramref name="x"/> [T, C] (already embedded) in place; n, a [T, C], qkv [T, 3C], hid [T, 4C] are scratch.
    /// </summary>
    public void Run(FV x, int T, int fromLayer, int toLayer, FV n, FV qkv, FV a, FV hid)
    {
        int C = Width;
        for (int i = fromLayer; i < toLayer; i++)
        {
            string p = $"{P}layers.{i}.";
            _k.Sparse.LayerNorm(x, n, _w[p + "ln1.weight"], _w[p + "ln1.bias"], T, C, 1e-5f);
            _k.Linear(n, _w[p + "qkv.weight"], _w[p + "qkv.bias"], qkv, T, C, 3 * C);
            _k.CausalAttention(qkv, a, T, Heads, C / Heads);
            _k.Linear(a, _w[p + "out.weight"], _w[p + "out.bias"], x, T, C, C, 0, x);
            _k.Sparse.LayerNorm(x, n, _w[p + "ln2.weight"], _w[p + "ln2.bias"], T, C, 1e-5f);
            int inner = _w.Shape(p + "fc1.weight")[0];
            _k.Linear(n, _w[p + "fc1.weight"], _w[p + "fc1.bias"], hid, T, C, inner);
            if (QuickGelu) _k.QuickGeluInPlace(hid, (long)T * inner); else _k.GeluInPlace(hid, (long)T * inner);
            _k.Linear(hid, _w[p + "fc2.weight"], _w[p + "fc2.bias"], x, T, inner, C, 0, x);
        }
    }
}

/// <summary>
/// The Stable Diffusion text encoder(s) from the single-file checkpoint: CLIP ViT-L (SD 1.x), OpenCLIP ViT-H (SD 2.x), or both CLIP ViT-L and OpenCLIP ViT-bigG (SDXL).
/// Input is token ids (the CLIP BPE ids with start and end tokens, padded to 77; <see cref="Preprocessing.BPETokenizer.EncodeCLIP"/> makes them), output the context the
/// UNet's cross-attention reads. The defaults follow ComfyUI: SD 1.x takes the last layer, SD 2.x the penultimate, both through the final layer norm; SDXL takes the penultimate
/// layer of each tower WITHOUT the final norm, concatenated (768 + 1280 = 2048 wide), and the pooled embedding = the second tower's final-norm output at the end token through
/// its projection. <c>layer</c> overrides the layer (-1 the last, -2 the penultimate: ComfyUI's "CLIP Set Last Layer").
/// </summary>
public sealed partial class SdTextEncoder : IDisposable
{
    private bool _ownsWeights;            // loaded alone (LoadAsync / LoadSdxlAsync): Dispose frees the weights too
    private readonly Accelerator _acc;
    private readonly SdWeights _w;
    private readonly SdKernels _k;
    private readonly SdClipTower[] _towers;
    private readonly bool _xl, _finalNorm;
    private readonly int _defaultLayer;
    private readonly Dictionary<int, Buf> _slots = new();
    private readonly List<IDisposable> _retired = new();

    /// <summary>Context width the UNet expects (768, 1024 or 2048).</summary>
    public int Width => _towers.Sum(t => t.Width);
    /// <summary>The token count of one prompt chunk (77).</summary>
    public int MaxTokens => _towers[0].MaxTokens;
    public bool IsXL => _xl;

    internal SdTextEncoder(Accelerator acc, SdWeights weights, SdFamily family)
    {
        _acc = acc; _w = weights; _k = new SdKernels(acc);
        switch (family)
        {
            case SdFamily.SD1: _towers = new[] { new SdClipTower(acc, weights, _k, "l.", quickGelu: true) }; _finalNorm = true; _defaultLayer = -1; break;
            case SdFamily.SD2:
                _towers = new[] { new SdClipTower(acc, weights, _k, "h.", quickGelu: false) }; _finalNorm = true;
                _defaultLayer = _towers[0].Layers == 23 ? -1 : -2;        // the penultimate of ViT-H's 24; a diffusers SD 2.x text_encoder already dropped the last layer (23 left)
                break;
            default: _towers = new[] { new SdClipTower(acc, weights, _k, "l.", quickGelu: true), new SdClipTower(acc, weights, _k, "g.", quickGelu: false) }; _xl = true; _finalNorm = false; _defaultLayer = -2; break;
        }
    }

    private FV Slot(int slot, long n)
    {
        if (!_slots.TryGetValue(slot, out var b) || b.Length < n)
        {
            if (b != null) _retired.Add(b);
            b = _acc.Allocate1D<float>(Math.Max(n, 64));
            _slots[slot] = b;
        }
        return b.View.SubView(0, n);
    }

    /// <summary>
    /// Encode one prompt chunk. <paramref name="ids"/> are the token ids of the first (or only) tower ([77]: start token, BPE ids, end token, padding);
    /// <paramref name="ids2"/> those of SDXL's second tower (its tokenizer pads with 0 instead of the end token; null = the same ids). Returns the context [77, Width] and,
    /// for SDXL, the pooled embedding [1280].
    /// </summary>
    public async Task<SdTextEmbedding> EncodeAsync(int[] ids, int[]? ids2 = null, int? layer = null, CancellationToken cancellationToken = default)
    {
        int T = ids.Length, W = Width;
        var ctx = Slot(6, (long)T * W);
        FV? pooled = _xl ? Slot(7, _towers[1].ProjectionWidth) : null;
        await EncodeAsync(ids, ids2, layer, ctx, pooled, cancellationToken);
        await _acc.SynchronizeAsync();
        var context = await ctx.CopyToHostAsync();
        float[]? p = pooled.HasValue ? await pooled.Value.CopyToHostAsync() : null;
        return new SdTextEmbedding(context, T, W, p);
    }

    /// <summary>
    /// Encode one prompt chunk on the device: <paramref name="context"/> receives [ids.Length, Width] row-major (the towers side by side for SDXL), <paramref name="pooled"/>
    /// (SDXL only; [1280]) the pooled embedding. Same arguments as the host overload.
    /// </summary>
    public async Task EncodeAsync(int[] ids, int[]? ids2, int? layer, FV context, FV? pooled, CancellationToken cancellationToken = default)
    {
        int T = ids.Length;
        if (T < 1 || T > MaxTokens) throw new ArgumentException($"a prompt chunk is 1 to {MaxTokens} tokens (pad to {MaxTokens} as CLIP does)");
        int lay = layer ?? _defaultLayer;
        if (lay > -1) throw new ArgumentException("layer counts from the end: -1 = the last layer, -2 = the penultimate");
        int W = Width;
        if (context.Length < (long)T * W) throw new ArgumentException($"the context needs [{T}, {W}]");
        int col = 0;
        for (int ti = 0; ti < _towers.Length; ti++)
        {
            var tw = _towers[ti];
            var tid = ti == 1 && ids2 != null ? ids2 : ids;
            if (tid.Length != T) throw new ArgumentException("both towers take the same number of tokens");
            int C = tw.Width, stop = tw.Layers + 1 + lay;            // layers to run for the context (-1: all)
            if (stop < 1) throw new ArgumentException($"layer {lay} is below the first of {tw.Layers} layers");
            var idv = _acc.Allocate1D(tid);
            try
            {
                var x = Slot(0, (long)T * C); var n = Slot(1, (long)T * C); var qkv = Slot(2, (long)T * 3 * C); var a = Slot(3, (long)T * C);
                var hid = Slot(4, (long)T * tw.Width * 4);
                _k.Embed(idv.View, _w[tw.P + "tok"], _w[tw.P + "pos"], x, T, C);
                tw.Run(x, T, 0, stop, n, qkv, a, hid);
                var outv = x;
                if (_finalNorm) { _k.Sparse.LayerNorm(x, n, _w[tw.P + "final_ln.weight"], _w[tw.P + "final_ln.bias"], T, C, 1e-5f); outv = n; }
                _k.CopyColumns(outv, context, T, C, W, col);
                col += C;
                if (_xl && ti == 1 && tw.HasProjection && pooled.HasValue)
                {
                    // the pooled embedding: the rest of the layers, the final norm, the end token's row (the highest id: OpenCLIP's text.argmax), the projection
                    tw.Run(x, T, stop, tw.Layers, n, qkv, a, hid);
                    _k.Sparse.LayerNorm(x, n, _w[tw.P + "final_ln.weight"], _w[tw.P + "final_ln.bias"], T, C, 1e-5f);
                    int eot = 0; for (int t = 1; t < T; t++) if (tid[t] > tid[eot]) eot = t;
                    _k.Sparse.Linear(n.SubView((long)eot * C, C), _w[tw.P + "proj.weight"], null, pooled.Value, 1, C, tw.ProjectionWidth);
                }
                await _acc.SynchronizeAsync();      // the ids buffer goes now, and the next tower reuses the scratch
            }
            finally { idv.Dispose(); }
            cancellationToken.ThrowIfCancellationRequested();
        }
    }

    public void Dispose()
    {
        foreach (var b in _slots.Values) b.Dispose();
        _slots.Clear();
        foreach (var r in _retired) r.Dispose();
        _retired.Clear();
        if (_ownsWeights) _w.Dispose();
    }
}
