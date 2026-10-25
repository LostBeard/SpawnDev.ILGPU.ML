using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.StableDiffusion;

using FV = ArrayView1D<float, Stride1D.Dense>;
using HV = ArrayView1D<global::ILGPU.Half, Stride1D.Dense>;
using Buf = MemoryBuffer1D<float, Stride1D.Dense>;

/// <summary>
/// SDXL's extra conditioning: the pooled text embedding of the second text encoder and the six size numbers (original size, crop top-left, target size; ComfyUI's width / height
/// fields, in pixels).
/// </summary>
public sealed record SdxlConditioning(float[] Pooled, int OriginalHeight, int OriginalWidth, int CropTop, int CropLeft, int TargetHeight, int TargetWidth);

/// <summary>
/// The Stable Diffusion UNet (the LDM <c>UNetModel</c>, read from a single-file checkpoint in its original key names): SD 1.x, SD 2.x and SDXL are one code path, the layout
/// (levels, resnets per level, transformer depth, heads, linear or 1x1-conv projections, SDXL's label embedding) is read from the tensors themselves.
/// One call is one noise prediction at one timestep for batch 1 (the sampler, CFG and noise stay with the caller): <see cref="ForwardAsync(float[], int, int, double, float[], int, SdxlConditioning?, CancellationToken)"/>.
/// The latent may have any size whose sides are multiples of 8 (three halvings). Activations are channel-last fp32 [pixels, C], weights fp16, convolutions the gathered-row
/// implicit GEMM (tensor cores on CUDA), attention the portable flash kernels. The prediction is whatever the model was trained for (epsilon for SD 1.x / SDXL base, v for the
/// SD 2.x 768-v models: <see cref="SdCheckpoint.Prediction"/>).
/// </summary>
public sealed class SdUNet : IDisposable
{
    private readonly Accelerator _acc;
    private readonly SdWeights _w;
    private readonly SdKernels _k;
    private readonly DiTKernels _dk;
    private readonly Dictionary<(int, int, int, int, int, int, int, int), MemoryBuffer1D<int, Stride1D.Dense>> _tables = new();
    private readonly Dictionary<int, Buf> _slots = new();
    private readonly List<IDisposable> _retired = new();
    private readonly List<Module[]> _input = new(), _output = new();
    private readonly Module[] _middle;

    public int ModelChannels { get; }
    public int InChannels { get; }
    public int OutChannels { get; }
    /// <summary>The text context width (768 SD 1.x, 1024 SD 2.x, 2048 SDXL).</summary>
    public int ContextDim { get; }
    /// <summary>True for SDXL (the pooled text + size conditioning, <see cref="SdxlConditioning"/>).</summary>
    public bool IsXL { get; }
    private readonly int _timeDim, _admDim;
    private readonly bool _fixedHeads;      // SD 1.x: 8 heads at every width; SD 2.x / SDXL: 64-wide heads

    /// <summary>Test hook: (stage name, activation view [pixels, C] channel-last, height, width, channels); valid only during the call.</summary>
    public Action<string, FV, int, int, int>? Tap { get; set; }

    private enum Kind { Res, Attn, Down, Up }
    private sealed record Module(Kind Kind, string Prefix, int Cin, int Cout, int Depth);

    public SdUNet(Accelerator accelerator, SdWeights weights)
    {
        _acc = accelerator; _w = weights; _k = new SdKernels(accelerator); _dk = new DiTKernels(accelerator);
        var convIn = weights.Shape("input_blocks.0.0.weight");
        ModelChannels = convIn[0]; InChannels = convIn[1];
        OutChannels = weights.Shape("out.2.weight")[0];
        _timeDim = weights.Shape("time_embed.0.weight")[0];
        IsXL = weights.Has("label_emb.0.0.weight");
        _admDim = IsXL ? weights.Shape("label_emb.0.0.weight")[1] : 0;
        int c = ModelChannels;
        for (int i = 1; weights.Has($"input_blocks.{i}.0.in_layers.0.weight") || weights.Has($"input_blocks.{i}.0.op.weight"); i++)
        {
            var mods = ParseBlock($"input_blocks.{i}.", ref c);
            _input.Add(mods);
        }
        _middle = ParseBlock("middle_block.", ref c);
        for (int i = 0; weights.Has($"output_blocks.{i}.0.in_layers.0.weight"); i++)
            _output.Add(ParseBlock($"output_blocks.{i}.", ref c));
        int ctx = 0;
        foreach (var m in _input.Concat(new[] { _middle }).Concat(_output).SelectMany(b => b))
            if (m.Kind == Kind.Attn) { ctx = weights.Shape(m.Prefix + "transformer_blocks.0.attn2.kv.weight")[1]; break; }
        ContextDim = ctx;
        _fixedHeads = ctx == 768;
    }

    private Module[] ParseBlock(string p, ref int c)
    {
        var mods = new List<Module>();
        for (int j = 0; ; j++)
        {
            string q = $"{p}{j}.";
            if (_w.Has(q + "in_layers.0.weight"))
            {
                int cin = _w.Shape(q + "in_layers.0.weight")[0], cout = _w.Shape(q + "in_layers.2.weight")[0];
                mods.Add(new Module(Kind.Res, q, cin, cout, 0)); c = cout;
            }
            else if (_w.Has(q + "norm.weight") && _w.Has(q + "proj_in.weight"))
            {
                int depth = 0; while (_w.Has($"{q}transformer_blocks.{depth}.norm1.weight")) depth++;
                mods.Add(new Module(Kind.Attn, q, c, c, depth));
            }
            else if (_w.Has(q + "op.weight")) mods.Add(new Module(Kind.Down, q + "op.", c, c, 0));
            else if (_w.Has(q + "conv.weight")) mods.Add(new Module(Kind.Up, q + "conv.", c, c, 0));
            else break;
        }
        return mods.ToArray();
    }

    private int Heads(int C) => _fixedHeads ? 8 : C / 64;

    // ---------------------------------------------------------------------------------------------------
    // workspace: role slots that only grow (a replaced buffer is kept until Dispose: a browser may still have a dispatch on it queued)
    // ---------------------------------------------------------------------------------------------------
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
    private const int SlotH0 = 0, SlotH1 = 1, SlotNorm = 2, SlotT = 3, SlotWide = 4, SlotMid = 5, SlotQ = 6, SlotKv = 7, SlotEmb = 8, SlotProj = 10, SlotIn = 11, SlotOut = 12,
        SlotTFreq = 13, SlotTHid = 14, SlotY = 15, SlotYHid = 16, SlotCtx = 17, SlotScores = 18, SlotHostIn = 19, SlotHostOut = 20,
        SlotSkip = 100;   // + block index: one buffer per skip connection

    private ArrayView1D<int, Stride1D.Dense> Table(int inH, int inW, int outH, int outW, int k, int stride, int pad, int up)
    {
        var key = (inH, inW, outH, outW, k, stride, pad, up);
        if (_tables.TryGetValue(key, out var t)) return t.View;
        t = _acc.Allocate1D<int>((long)outH * outW * 27);
        _k.BuildTable(t.View, inH, inW, outH, outW, k, stride, pad, up);
        _tables[key] = t;
        return t.View;
    }

    // ---------------------------------------------------------------------------------------------------
    // the modules
    // ---------------------------------------------------------------------------------------------------
    private void ResBlock(Module m, FV x, FV dst, FV embAct, int H, int W)
    {
        int rows = H * W, cin = m.Cin, cout = m.Cout;
        string p = m.Prefix;
        var nIn = Slot(SlotNorm, (long)rows * Math.Max(cin, cout)).SubView(0, (long)rows * cin);
        _k.GroupNorm(x, nIn, _w[p + "in_layers.0.weight"], _w[p + "in_layers.0.bias"], rows, cin, 32, 1e-5f, silu: true);
        var c1 = Slot(SlotT, (long)rows * cout);
        _k.Conv(nIn, Table(H, W, H, W, 3, 1, 1, 1), _w[p + "in_layers.2.weight"], _w[p + "in_layers.2.bias"], c1, rows, cin, cout, 9);
        var proj = Slot(SlotProj, cout);
        _k.Linear(embAct, _w[p + "emb_layers.1.weight"], _w[p + "emb_layers.1.bias"], proj, 1, _timeDim, cout);
        _k.AddRowVector(c1, proj, rows, cout);
        var nOut = Slot(SlotNorm, (long)rows * Math.Max(cin, cout)).SubView(0, (long)rows * cout);
        _k.GroupNorm(c1, nOut, _w[p + "out_layers.0.weight"], _w[p + "out_layers.0.bias"], rows, cout, 32, 1e-5f, silu: true);
        _k.Conv(nOut, Table(H, W, H, W, 3, 1, 1, 1), _w[p + "out_layers.3.weight"], _w[p + "out_layers.3.bias"], dst, rows, cout, cout, 9);
        if (_w.Has(p + "skip_connection.weight")) _k.Linear(x, _w[p + "skip_connection.weight"], _w[p + "skip_connection.bias"], dst, rows, cin, cout, 0, dst);
        else _k.AddInPlace(dst, x, (long)rows * cout);
    }

    private void Attention(FV q, int qStride, int qOff, FV kv, int kvStride, int kOff, int vOff, FV o, int Nq, int Nk, int C)
    {
        int H = Heads(C), D = C / H;
        FV scores = _dk.UsesFlashAttention(D) ? Slot(SlotScores, 1) : Slot(SlotScores, (long)H * Nq * Nk);
        _dk.Attention(q, qStride, qOff, kv, kvStride, kOff, vOff, scores, o, Nq, Nk, H, D);
    }

    private void Transformer(Module m, FV x, FV dst, FV ctx, int ctxRows, int H, int W)
    {
        int rows = H * W, C = m.Cin;
        string p = m.Prefix;
        var n = Slot(SlotNorm, (long)rows * C);
        _k.GroupNorm(x, n, _w[p + "norm.weight"], _w[p + "norm.bias"], rows, C, 32, 1e-6f, silu: false);
        var h = Slot(SlotT, (long)rows * C);
        _k.Linear(n, _w[p + "proj_in.weight"], _w[p + "proj_in.bias"], h, rows, C, C);
        for (int j = 0; j < m.Depth; j++)
        {
            string b = $"{p}transformer_blocks.{j}.";
            // self attention
            _k.Sparse.LayerNorm(h, n, _w[b + "norm1.weight"], _w[b + "norm1.bias"], rows, C, 1e-5f);
            var qkv = Slot(SlotWide, (long)rows * 8 * C).SubView(0, (long)rows * 3 * C);
            _k.Linear(n, _w[b + "attn1.qkv.weight"], null, qkv, rows, C, 3 * C);
            var a = Slot(SlotMid, (long)rows * 4 * C).SubView(0, (long)rows * C);
            Attention(qkv, 3 * C, 0, qkv, 3 * C, C, 2 * C, a, rows, rows, C);
            _k.Linear(a, _w[b + "attn1.to_out.0.weight"], _w[b + "attn1.to_out.0.bias"], h, rows, C, C, 0, h);
            // cross attention to the text
            _k.Sparse.LayerNorm(h, n, _w[b + "norm2.weight"], _w[b + "norm2.bias"], rows, C, 1e-5f);
            var q = Slot(SlotQ, (long)rows * C);
            _k.Linear(n, _w[b + "attn2.to_q.weight"], null, q, rows, C, C);
            var kv = Slot(SlotKv, (long)ctxRows * 2 * C);
            _k.Linear(ctx, _w[b + "attn2.kv.weight"], null, kv, ctxRows, ContextDim, 2 * C);
            Attention(q, C, 0, kv, 2 * C, 0, C, a, rows, ctxRows, C);
            _k.Linear(a, _w[b + "attn2.to_out.0.weight"], _w[b + "attn2.to_out.0.bias"], h, rows, C, C, 0, h);
            // feed forward (GEGLU)
            _k.Sparse.LayerNorm(h, n, _w[b + "norm3.weight"], _w[b + "norm3.bias"], rows, C, 1e-5f);
            int inner = _w.Shape(b + "ff.net.2.weight")[1];
            var f = Slot(SlotWide, (long)rows * 8 * C).SubView(0, (long)rows * 2 * inner);
            _k.Linear(n, _w[b + "ff.net.0.proj.weight"], _w[b + "ff.net.0.proj.bias"], f, rows, C, 2 * inner);
            var g = Slot(SlotMid, (long)rows * 4 * C).SubView(0, (long)rows * inner);
            _k.Geglu(f, g, rows, inner);
            _k.Linear(g, _w[b + "ff.net.2.weight"], _w[b + "ff.net.2.bias"], h, rows, inner, C, 0, h);
        }
        _k.Linear(h, _w[p + "proj_out.weight"], _w[p + "proj_out.bias"], dst, rows, C, C, 0, x);
    }

    private void Down(Module m, FV x, FV dst, int H, int W)
    {
        int oh = (H + 1) / 2, ow = (W + 1) / 2;
        _k.Conv(x, Table(H, W, oh, ow, 3, 2, 1, 1), _w[m.Prefix + "weight"], _w[m.Prefix + "bias"], dst, oh * ow, m.Cin, m.Cout, 9);
    }

    private void Up(Module m, FV x, FV dst, int H, int W)
    {
        _k.Conv(x, Table(H, W, 2 * H, 2 * W, 3, 1, 1, 2), _w[m.Prefix + "weight"], _w[m.Prefix + "bias"], dst, 4 * H * W, m.Cin, m.Cout, 9);
    }

    /// <summary>
    /// Run one block's modules from <paramref name="x"/>; the last writes into <paramref name="last"/>, or (null) into whichever ping-pong slot its input is not in.
    /// Returns the output view and size.
    /// </summary>
    private (FV View, int H, int W, int C) RunBlock(Module[] mods, FV x, FV? last, FV embAct, FV ctx, int ctxRows, int H, int W, int c, long maxElems)
    {
        var cur = x;
        for (int i = 0; i < mods.Length; i++)
        {
            var m = mods[i];
            bool isLast = i == mods.Length - 1;
            int oh = m.Kind == Kind.Down ? (H + 1) / 2 : m.Kind == Kind.Up ? 2 * H : H;
            int ow = m.Kind == Kind.Down ? (W + 1) / 2 : m.Kind == Kind.Up ? 2 * W : W;
            long n = (long)oh * ow * m.Cout;
            FV dst = isLast && last.HasValue ? last.Value.SubView(0, n) : PickFree(cur, maxElems).SubView(0, n);
            switch (m.Kind)
            {
                case Kind.Res: ResBlock(m, cur, dst, embAct, H, W); break;
                case Kind.Attn: Transformer(m, cur, dst, ctx, ctxRows, H, W); break;
                case Kind.Down: Down(m, cur, dst, H, W); break;
                case Kind.Up: Up(m, cur, dst, H, W); break;
            }
            cur = dst; H = oh; W = ow; c = m.Cout;
        }
        return (cur, H, W, c);
    }

    private FV PickFree(FV cur, long maxElems)
    {
        var a = Slot(SlotH0, maxElems); var b = Slot(SlotH1, maxElems);
        IContiguousArrayView cv = cur.BaseView, av = a.BaseView;
        return cv.Buffer == av.Buffer ? b : a;
    }

    // ---------------------------------------------------------------------------------------------------
    // forward
    // ---------------------------------------------------------------------------------------------------

    /// <summary>
    /// One prediction on the device. <paramref name="latent"/> is [inChannels, h, w] channel-first (the scaled latent the sampler works on, the model input as diffusers gets it),
    /// <paramref name="context"/> [contextRows, ContextDim] row-major (the text encoder output; any number of rows: ComfyUI concatenates 77-token chunks for long prompts),
    /// <paramref name="y"/> SDXL's label vector [adm] (<see cref="SdxlLabel"/>) or null, <paramref name="output"/> [outChannels, h, w].
    /// </summary>
    public async Task ForwardAsync(FV latent, int h, int w, double timestep, FV context, int contextRows, FV? y, FV output, CancellationToken cancellationToken = default)
    {
        if (h % 8 != 0 || w % 8 != 0) throw new ArgumentException("the latent sides must be multiples of 8");
        if (IsXL != y.HasValue) throw new ArgumentException(IsXL ? "an SDXL UNet needs the label vector (pooled text + size ids)" : "only an SDXL UNet takes a label vector");
        int mc = ModelChannels, rows0 = h * w;
        // the largest activation: a level's pixels x the widest concatenated input of that level (output blocks read h | skip)
        long maxElems = (long)rows0 * mc * 2;
        {
            int H = h, W = w, c = mc;
            foreach (var blk in _input.Concat(new[] { _middle }).Concat(_output))
                foreach (var m in blk)
                {
                    maxElems = Math.Max(maxElems, (long)H * W * (m.Cin + mc));
                    if (m.Kind == Kind.Down) { H = (H + 1) / 2; W = (W + 1) / 2; }
                    if (m.Kind == Kind.Up) { H *= 2; W *= 2; }
                    maxElems = Math.Max(maxElems, (long)H * W * m.Cout * 2);
                }
        }

        // time (and SDXL label) embedding -> SiLU, shared by every resnet
        var tFreq = Slot(SlotTFreq, mc);
        tFreq.CopyFromCPU(SdKernels.TimestepEmbedding(timestep, mc));
        var tHid = Slot(SlotTHid, _timeDim); var emb = Slot(SlotEmb, _timeDim);
        _k.Linear(tFreq, _w["time_embed.0.weight"], _w["time_embed.0.bias"], tHid, 1, mc, _timeDim, 1);
        _k.Linear(tHid, _w["time_embed.2.weight"], _w["time_embed.2.bias"], emb, 1, _timeDim, _timeDim);
        if (IsXL)
        {
            var yh = Slot(SlotYHid, _timeDim);
            _k.Linear(y!.Value, _w["label_emb.0.0.weight"], _w["label_emb.0.0.bias"], yh, 1, _admDim, _timeDim, 1);
            _k.Linear(yh, _w["label_emb.0.2.weight"], _w["label_emb.0.2.bias"], emb, 1, _timeDim, _timeDim, 0, emb);
        }
        _k.Sparse.SiluInPlace(emb, _timeDim);                 // every resnet's emb_layers = SiLU, Linear
        var embAct = emb;

        // conv_in on the channel-last latent
        var xin = Slot(SlotIn, (long)rows0 * InChannels);
        _dk.Transpose(latent, xin, InChannels, rows0);
        var skips = new List<(FV View, int H, int W, int C)>();
        var s0 = Slot(SlotSkip, (long)rows0 * mc);
        _k.Conv(xin, Table(h, w, h, w, 3, 1, 1, 1), _w["input_blocks.0.0.weight"], _w["input_blocks.0.0.bias"], s0, rows0, InChannels, mc, 9);
        skips.Add((s0, h, w, mc));
        Tap?.Invoke("conv_in", s0, h, w, mc);

        int Hc = h, Wc = w, Cc = mc;
        var cur = s0;
        int level = 0;
        for (int i = 0; i < _input.Count; i++)
        {
            var mods = _input[i];
            int oh = Hc, ow = Wc, oc = mods[^1].Cout;
            foreach (var m in mods) { if (m.Kind == Kind.Down) { oh = (oh + 1) / 2; ow = (ow + 1) / 2; } }
            var dst = Slot(SlotSkip + 1 + i, (long)oh * ow * oc);
            (cur, Hc, Wc, Cc) = RunBlock(mods, cur, dst, embAct, context, contextRows, Hc, Wc, Cc, maxElems);
            skips.Add((cur, Hc, Wc, Cc));
            bool levelEnd = mods.Any(m => m.Kind == Kind.Down) || i == _input.Count - 1;
            if (levelEnd) { Tap?.Invoke($"down_{level}", cur, Hc, Wc, Cc); level++; }
            if (_acc.IsBrowser()) await _acc.SynchronizeAsync();
            cancellationToken.ThrowIfCancellationRequested();
        }

        (cur, Hc, Wc, Cc) = RunBlock(_middle, cur, null, embAct, context, contextRows, Hc, Wc, Cc, maxElems);
        Tap?.Invoke("mid", cur, Hc, Wc, Cc);
        if (_acc.IsBrowser()) await _acc.SynchronizeAsync();

        level = 0;
        for (int i = 0; i < _output.Count; i++)
        {
            var (sv, sh, sw, sc) = skips[^1];
            skips.RemoveAt(skips.Count - 1);
            if (sh != Hc || sw != Wc) throw new InvalidOperationException($"skip {sh}x{sw} does not meet {Hc}x{Wc} (latent sides must be multiples of 8)");
            var cat = PickFree(cur, maxElems).SubView(0, (long)Hc * Wc * (Cc + sc));
            _k.ConcatChannels(cur, sv, cat, Hc * Wc, Cc, sc);
            var mods = _output[i];
            var first = mods[0];
            if (first.Cin != Cc + sc) throw new InvalidOperationException($"output block {i}: expects {first.Cin} channels, got {Cc + sc}");
            (cur, Hc, Wc, Cc) = RunBlock(mods, cat, null, embAct, context, contextRows, Hc, Wc, Cc + sc, maxElems);
            bool levelEnd = mods.Any(m => m.Kind == Kind.Up) || i == _output.Count - 1;
            if (levelEnd) { Tap?.Invoke($"up_{level}", cur, Hc, Wc, Cc); level++; }
            if (_acc.IsBrowser()) await _acc.SynchronizeAsync();
            cancellationToken.ThrowIfCancellationRequested();
        }

        // out: GroupNorm, SiLU, conv to the latent channels, back to channel-first
        var n = Slot(SlotNorm, (long)rows0 * Cc);
        _k.GroupNorm(cur, n, _w["out.0.weight"], _w["out.0.bias"], rows0, Cc, 32, 1e-5f, silu: true);
        var o = Slot(SlotOut, (long)rows0 * OutChannels);
        _k.Conv(n, Table(h, w, h, w, 3, 1, 1, 1), _w["out.2.weight"], _w["out.2.bias"], o, rows0, Cc, OutChannels, 9);
        _dk.Transpose(o, output, rows0, OutChannels);
    }

    /// <summary>
    /// One prediction from host arrays: <paramref name="latent"/> [inChannels, h, w], <paramref name="context"/> [contextRows, ContextDim]; returns [outChannels, h, w].
    /// SDXL needs <paramref name="xl"/>.
    /// </summary>
    public async Task<float[]> ForwardAsync(float[] latent, int h, int w, double timestep, float[] context, int contextRows, SdxlConditioning? xl = null, CancellationToken cancellationToken = default)
    {
        if (latent.Length != (long)InChannels * h * w) throw new ArgumentException($"the latent must be [{InChannels}, {h}, {w}]");
        if (context.Length != (long)contextRows * ContextDim) throw new ArgumentException($"the context must be [{contextRows}, {ContextDim}]");
        var lat = Slot(SlotHostIn, latent.Length); lat.CopyFromCPU(latent);
        var ctx = Slot(SlotCtx, context.Length); ctx.CopyFromCPU(context);
        FV? y = null;
        if (IsXL)
        {
            if (xl == null) throw new ArgumentException("an SDXL UNet needs the SDXL conditioning (pooled text embedding and size ids)");
            var yv = Slot(SlotY, _admDim); yv.CopyFromCPU(SdxlLabel(xl, _admDim)); y = yv;
        }
        var o = Slot(SlotHostOut, (long)OutChannels * h * w);
        await ForwardAsync(lat, h, w, timestep, ctx, contextRows, y, o, cancellationToken);
        await _acc.SynchronizeAsync();
        return await o.CopyToHostAsync();
    }

    /// <summary>SDXL's label vector: the pooled text embedding, then the timestep embedding (256 wide each) of the six size numbers.</summary>
    public static float[] SdxlLabel(SdxlConditioning c, int admDim)
    {
        var ids = new double[] { c.OriginalHeight, c.OriginalWidth, c.CropTop, c.CropLeft, c.TargetHeight, c.TargetWidth };
        int per = (admDim - c.Pooled.Length) / ids.Length;
        if (per * ids.Length + c.Pooled.Length != admDim) throw new ArgumentException($"the pooled embedding ({c.Pooled.Length}) does not fit the label width {admDim}");
        var y = new float[admDim];
        c.Pooled.CopyTo(y, 0);
        for (int i = 0; i < ids.Length; i++) SdKernels.TimestepEmbedding(ids[i], per).CopyTo(y, c.Pooled.Length + i * per);
        return y;
    }

    public void Dispose()
    {
        foreach (var b in _slots.Values) b.Dispose();
        _slots.Clear();
        foreach (var r in _retired) r.Dispose();
        _retired.Clear();
        foreach (var t in _tables.Values) t.Dispose();
        _tables.Clear();
    }
}
