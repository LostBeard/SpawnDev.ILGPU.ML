using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.QwenImage;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// Qwen2.5-VL-7B's vision tower: the picture side of Qwen-Image Edit's text encoder (ComfyUI's TextEncodeQwenImageEditPlus), ported from Hugging Face transformers'
/// <c>Qwen2_5_VisionTransformerPretrainedModel</c> (Apache-2.0) and loaded from the <c>visual.*</c> tensors of ComfyUI's <c>qwen_2.5_vl_7b_fp8_scaled.safetensors</c>
/// (<see cref="Flux2Weights.LoadVisionAsync"/>: 8-bit matrices stay 8-bit and run exact, the scale on the input).
/// <code>
/// x = patch_embed(14 x 14 patches, both temporal slots = the picture)                    [n, 1176] -> [n, 1280], tokens in 2 x 2 merge blocks
/// x = x[window order]                                                                    merge blocks grouped window by window (112 px = 4 x 4 merged tokens)
/// 32 x { x += proj(attn(rope2d(qkv(rms1(x)))));  x += down(silu(gate(rms2(x))) * up(rms2(x))) }     16 heads of 80; blocks 7, 15, 23, 31 over the whole picture, the rest per window
/// y = mlp2(gelu(mlp0(rms_q(x) viewed as [n / 4, 5120])))                                 -> [n / 4, 3584]
/// y = y[original order]
/// </code>
/// The 2D rope rotates pairs 0-19 by the patch row and 20-39 by the patch column (frequencies 1 / 10000^(i / 40), i = 0, 2, .., 38), rotate_half layout.
/// Pictures are encoded together: full attention is per picture, windows never cross pictures. See Plans/qwen-image-edit-2509-2026-11-15.md.
/// </summary>
public sealed class QwenVisionEncoder : IDisposable
{
    public const int Patch = 14, Merge = 2, Width = 1280, Heads = 16, HeadDim = 80, Inter = 3420, OutDim = 3584, Depth = 32, WindowPx = 112, PatchIn = 3 * 2 * Patch * Patch;
    public const float Eps = 1e-6f, RopeTheta = 10000f;
    /// <summary>The blocks that attend over the whole picture (the rest attend within windows).</summary>
    public static readonly int[] FullAttentionBlocks = { 7, 15, 23, 31 };
    /// <summary>The processor's normalisation (OpenAI CLIP mean / std), applied to RGB in 0..1.</summary>
    public static readonly float[] Mean = { 0.48145466f, 0.4578275f, 0.40821073f }, Std = { 0.26862954f, 0.26130258f, 0.27577711f };

    private readonly Accelerator _acc;
    private readonly Flux2Weights _w;
    private readonly DiTKernels _k;
    private readonly SkinTokensKernels _u;
    private readonly Flux2Kernels _f;
    private readonly int _maxTokens;
    private readonly List<IDisposable> _owned = new();
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _in, _x, _h, _hn, _qkv, _attn, _proj, _gateUp, _act, _merged, _y, _out, _scores, _xs, _cos, _sin, _mean, _std;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _perm, _unperm;
    private readonly float[] _invFreq;

    /// <summary>Intermediate checkpoints for tests: name ("embed", "block{n}" 1-based, "merger"), device view, rows, columns. Rows are in window order except "embed" and the output.</summary>
    public Action<string, FV, int, int>? Probe;

    /// <summary>The merged tokens of the last <see cref="Encode"/>, [sum of n / 4, 3584] in picture order, each picture's rows in the processor's order.</summary>
    public FV Output => _out.View;

    /// <summary>
    /// CUDA: fp32-class activations through the tensor cores (<see cref="DiTKernels.PreciseActivations"/>). ON: ComfyUI runs the encoder in fp32 ("manual cast"). Other backends are fp32 anyway.
    /// </summary>
    public bool PreciseActivations { get => _k.PreciseActivations; set => _k.PreciseActivations = value; }

    /// <param name="maxTokens">the most patches (14 x 14) one <see cref="Encode"/> call holds over all its pictures (sizes every buffer)</param>
    public QwenVisionEncoder(Accelerator accelerator, Flux2Weights weights, int maxTokens)
    {
        if (maxTokens <= 0 || maxTokens % (Merge * Merge) != 0) throw new ArgumentOutOfRangeException(nameof(maxTokens), "a positive multiple of 4");
        _acc = accelerator; _w = weights; _maxTokens = maxTokens;
        foreach (var (name, shape) in new[] { ("visual.patch_embed.proj.weight", new[] { Width, 3, 2, Patch, Patch }), ("visual.blocks.0.attn.qkv.weight", new[] { 3 * Width, Width }),
            ("visual.blocks.0.mlp.gate_proj.weight", new[] { Inter, Width }), ("visual.merger.mlp.0.weight", new[] { 4 * Width, 4 * Width }), ("visual.merger.mlp.2.weight", new[] { OutDim, 4 * Width }) })
            if (!weights.Has(name) || !weights.Shape(name).SequenceEqual(shape)) throw new InvalidDataException($"{name}: expected [{string.Join(", ", shape)}] (Qwen2.5-VL-7B's vision tower)");
        if (weights.Has($"visual.blocks.{Depth}.norm1.weight") || !weights.Has($"visual.blocks.{Depth - 1}.norm1.weight")) throw new InvalidDataException($"expected {Depth} vision blocks");
        _k = new DiTKernels(accelerator) { PreciseActivations = true }; _u = new SkinTokensKernels(accelerator); _f = new Flux2Kernels(accelerator);
        long N = maxTokens;
        _in = Alloc(N * PatchIn); _x = Alloc(N * Width); _h = Alloc(N * Width); _hn = Alloc(N * Width); _qkv = Alloc(N * 3 * Width); _attn = Alloc(N * Width); _proj = Alloc(N * Width);
        _gateUp = Alloc(N * 2 * Inter); _act = Alloc(N * Inter); _merged = Alloc(N / 4 * 4 * Width); _y = Alloc(N / 4 * OutDim); _out = Alloc(N / 4 * OutDim);
        _scores = Alloc(_k.UsesFlashAttention(HeadDim) ? 1 : (long)Heads * N * N);
        _xs = Alloc(N * Inter);                                                                 // the widest 8-bit (scaled) input: down_proj's [n, 3420] (the merger's [n / 4, 5120] is n * 1280)
        var inv = new float[HeadDim / 2];
        for (int i = 0; i < HeadDim / 4; i++) inv[i] = inv[i + HeadDim / 4] = 1f / MathF.Pow(RopeTheta, 2f * i / (HeadDim / 2));   // 20 frequencies over dim 40, for h then for w
        _invFreq = inv; _cos = Alloc(N * (HeadDim / 2)); _sin = Alloc(N * (HeadDim / 2));
        _mean = _acc.Allocate1D(Mean); _owned.Add(_mean); _std = _acc.Allocate1D(Std); _owned.Add(_std);
        _perm = AllocI(N); _unperm = AllocI(N / 4);
    }

    private MemoryBuffer1D<float, Stride1D.Dense> Alloc(long n) { var b = _acc.Allocate1D<float>(Math.Max(1, n)); _owned.Add(b); return b; }
    private MemoryBuffer1D<int, Stride1D.Dense> AllocI(long n) { var b = _acc.Allocate1D<int>(Math.Max(1, n)); _owned.Add(b); return b; }

    /// <summary>A picture of <paramref name="height"/> x <paramref name="width"/> pixels, both multiples of 28, gives (height / 14) x (width / 14) patches and a quarter as many tokens.</summary>
    public static (int Rows, int Cols) Grid(int height, int width)
    {
        if (height <= 0 || width <= 0 || height % (Patch * Merge) != 0 || width % (Patch * Merge) != 0) throw new ArgumentException($"picture sides must be positive multiples of {Patch * Merge} (got {width} x {height})");
        return (height / Patch, width / Patch);
    }

    // ─────────────────────────────── host-side layout: positions, window order, segments ───────────────────────────────

    /// <summary>
    /// The token layout of pictures with patch grids <paramref name="grids"/> (rows, cols): each token's (h, w) patch position in the processor's block-major order, the window
    /// permutation of the merge units (transformers' <c>get_vision_window_index</c>), the window segments and the per-picture segments, both in tokens of the permuted order.
    /// </summary>
    public static (int[] Pos, int[] WindowIndex, int[] Windows, int[] Pictures) Layout(IReadOnlyList<(int Rows, int Cols)> grids)
    {
        int total = 0; foreach (var (r, c) in grids) total += r * c;
        var pos = new int[total * 2];
        var windowIndex = new List<int>(total / 4);
        var windows = new List<int> { 0 }; var pictures = new List<int> { 0 };
        int ws = WindowPx / Merge / Patch, tok = 0, unitBase = 0;                                // window = 4 x 4 merge units
        foreach (var (gh, gw) in grids)
        {
            for (int bh = 0; bh < gh / 2; bh++) for (int bw = 0; bw < gw / 2; bw++) for (int mh = 0; mh < 2; mh++) for (int mw = 0; mw < 2; mw++)
            { pos[tok * 2] = bh * 2 + mh; pos[tok * 2 + 1] = bw * 2 + mw; tok++; }
            int lh = gh / 2, lw = gw / 2, nwh = (lh + ws - 1) / ws, nww = (lw + ws - 1) / ws;
            for (int wy = 0; wy < nwh; wy++) for (int wx = 0; wx < nww; wx++)
            {
                int count = 0;
                for (int y = wy * ws; y < Math.Min(lh, wy * ws + ws); y++) for (int x = wx * ws; x < Math.Min(lw, wx * ws + ws); x++) { windowIndex.Add(unitBase + y * lw + x); count++; }
                if (count > 0) windows.Add(windows[^1] + count * 4);
            }
            unitBase += lh * lw;
            pictures.Add(pictures[^1] + gh * gw);
        }
        return (pos, windowIndex.ToArray(), windows.ToArray(), pictures.ToArray());
    }

    // ─────────────────────────────── kernels ───────────────────────────────

    private Action<Index1D, FV, FV, FV, FV, int, int>? _patchify;
    // one picture, RGB in 0..1 as [height, width, 3]; patch rows of the processor: token (bh, bw, mh, mw) block-major, feature (c, t, py, px), both t the same pixel
    private static void PatchifyImpl(Index1D i, FV rgb, FV mean, FV std, FV o, int width, int gw)
    {
        int f = i % PatchIn, tok = i / PatchIn;
        int c = f / (2 * Patch * Patch), p = f % (Patch * Patch), py = p / Patch, px = p % Patch;
        int mw = tok % 2, mh = (tok / 2) % 2, blk = tok / 4, bw = blk % (gw / 2), bh = blk / (gw / 2);
        int y = (bh * 2 + mh) * Patch + py, x = (bw * 2 + mw) * Patch + px;
        o[i] = (rgb[((long)y * width + x) * 3 + c] - mean[c]) / std[c];
    }

    /// <summary>
    /// Encode pictures together. Each is RGB in 0..1, [height, width, 3] row-major on the device, already resized to sides that are multiples of 28 (ComfyUI's size; see the plan).
    /// Returns <see cref="Output"/>: [sum of tokens / 4, 3584], picture after picture.
    /// </summary>
    public FV Encode(IReadOnlyList<(FV Rgb, int Height, int Width)> pictures)
    {
        var grids = pictures.Select(p => Grid(p.Height, p.Width)).ToList();
        long off = 0;
        _patchify ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, int, int>(PatchifyImpl);
        for (int p = 0; p < pictures.Count; p++)
        {
            int n = grids[p].Rows * grids[p].Cols;
            if (off + n > _maxTokens) throw new ArgumentException($"{off + n} patches: more than this encoder's {_maxTokens}");
            if (pictures[p].Rgb.Length < (long)pictures[p].Height * pictures[p].Width * 3) throw new ArgumentException($"picture {p}: fewer than height x width x 3 values");
            _patchify(n * PatchIn, pictures[p].Rgb, _mean.View, _std.View, _in.View.SubView(off * PatchIn, (long)n * PatchIn), pictures[p].Width, grids[p].Cols);
            off += n;
        }
        return EncodePatches(_in.View, grids);
    }

    // nn.Linear over n rows; an 8-bit scaled matrix runs exact (its unscaled decode, the input times its scale)
    private void Lin(FV x, string name, FV o, int M, int K, int N, DiTKernels.Act act = DiTKernels.Act.None)
    {
        string weight = name + ".weight";
        float s = _w.ActivationScale(weight);
        if (s != 1f) { var xs = _xs.View.SubView(0, (long)M * K); _f.ScaleInto(x, xs, s, M * K); x = xs; }
        _k.Linear(x, _w.MatrixExact(weight), _w.Has(name + ".bias") ? _w.Vector(name + ".bias") : null, o, M, K, N, act);
    }

    /// <summary>
    /// Encode pictures from their processor rows: <paramref name="patches"/> is [sum of rows x cols, 1176] in the processor's layout (Hugging Face's <c>pixel_values</c>), with
    /// <paramref name="grids"/> each picture's patch rows and columns (<c>image_grid_thw</c> without the 1).
    /// </summary>
    public FV EncodePatches(FV patches, IReadOnlyList<(int Rows, int Cols)> grids)
    {
        foreach (var (r, c) in grids) if (r <= 0 || c <= 0 || r % 2 != 0 || c % 2 != 0) throw new ArgumentException("patch grids: positive even rows and columns");
        var (pos, windowIndex, windows, pics) = Layout(grids);
        int n = pics[^1], m = n / 4;
        if (n > _maxTokens) throw new ArgumentException($"{n} patches: more than this encoder's {_maxTokens}");
        if (patches.Length < (long)n * PatchIn) throw new ArgumentException($"patches: fewer than {n} x {PatchIn} values");
        // token-level permutation (merge units of 4 consecutive tokens move together), the positions moved with them, and the inverse for the merged rows
        var perm = new int[n]; var permPos = new int[n * 2]; var unperm = new int[m];
        for (int r = 0; r < n; r++) { int src = windowIndex[r / 4] * 4 + r % 4; perm[r] = src; permPos[r * 2] = pos[src * 2]; permPos[r * 2 + 1] = pos[src * 2 + 1]; }
        for (int u = 0; u < m; u++) unperm[windowIndex[u]] = u;
        var (rc, rs) = SkinTokensKernels.AxesRopeTables(permPos, n, 2, _invFreq, HeadDim / 4, HeadDim / 2);   // host cos / sin (GPU sin / cos are inexact at large angles)
        _cos.View.SubView(0, rc.Length).CopyFromCPU(rc); _sin.View.SubView(0, rs.Length).CopyFromCPU(rs);
        _perm.View.SubView(0, n).CopyFromCPU(perm); _unperm.View.SubView(0, m).CopyFromCPU(unperm);

        Lin(patches, "visual.patch_embed.proj", _h.View, n, PatchIn, Width);                    // Conv3d with kernel = stride: a linear map of each flattened patch (no bias)
        Probe?.Invoke("embed", _h.View, n, Width);
        _u.GatherRows(_h.View, _perm.View, _x.View, n, Width);

        int maxSeg = 0;
        for (int s = 1; s < pics.Length; s++) maxSeg = Math.Max(maxSeg, pics[s] - pics[s - 1]);
        if (!_k.UsesFlashAttention(HeadDim) && _scores.Length < (long)Heads * maxSeg * maxSeg) throw new InvalidOperationException("scores buffer too small");
        for (int l = 0; l < Depth; l++)
        {
            string b = $"visual.blocks.{l}.";
            int[] segs = Array.IndexOf(FullAttentionBlocks, l) >= 0 ? pics : windows;
            _u.RmsNorm(_x.View, _hn.View, _w.Vector(b + "norm1.weight"), n, Width, Eps);
            Lin(_hn.View, b + "attn.qkv", _qkv.View, n, Width, 3 * Width);
            _u.RopeTables(_qkv.View, _cos.View, _sin.View, n, 3 * Width, 0, Heads, HeadDim);
            _u.RopeTables(_qkv.View, _cos.View, _sin.View, n, 3 * Width, Width, Heads, HeadDim);
            for (int s = 1; s < segs.Length; s++)
            {
                int a = segs[s - 1], len = segs[s] - a;
                var q = _qkv.View.SubView((long)a * 3 * Width, (long)len * 3 * Width);
                _k.Attention(q, 3 * Width, 0, q, 3 * Width, Width, 2 * Width, _scores.View, _attn.View.SubView((long)a * Width, (long)len * Width), len, len, Heads, HeadDim);
            }
            Lin(_attn.View, b + "attn.proj", _proj.View, n, Width, Width);
            _k.GateResidual(_x.View, _proj.View, _proj.View, -1, n * Width, Width);
            _u.RmsNorm(_x.View, _hn.View, _w.Vector(b + "norm2.weight"), n, Width, Eps);
            Lin(_hn.View, b + "mlp.gate_proj", _act.View, n, Width, Inter);
            _f.CopyCols(_act.View, _gateUp.View, n, Inter, Inter, 2 * Inter, 0);
            Lin(_hn.View, b + "mlp.up_proj", _act.View, n, Width, Inter);
            _f.CopyCols(_act.View, _gateUp.View, n, Inter, Inter, 2 * Inter, Inter);
            _u.Swiglu(_gateUp.View, _act.View, n, Inter);
            Lin(_act.View, b + "mlp.down_proj", _proj.View, n, Inter, Width);
            _k.GateResidual(_x.View, _proj.View, _proj.View, -1, n * Width, Width);
            Probe?.Invoke($"block{l + 1}", _x.View, n, Width);
        }
        // merger: RMS norm per token, then each merge unit's 4 consecutive tokens as one row of 5120
        _u.RmsNorm(_x.View, _merged.View, _w.Vector("visual.merger.ln_q.weight"), n, Width, Eps);
        Lin(_merged.View, "visual.merger.mlp.0", _hn.View, m, 4 * Width, 4 * Width, DiTKernels.Act.GeluErf);
        Lin(_hn.View, "visual.merger.mlp.2", _y.View, m, 4 * Width, OutDim);
        Probe?.Invoke("merger", _y.View, m, OutDim);
        _u.GatherRows(_y.View, _unperm.View, _out.View, m, OutDim);
        return _out.View.SubView(0, (long)m * OutDim);
    }

    public void Dispose()
    {
        foreach (var o in _owned) o.Dispose();
        _owned.Clear();
    }
}
