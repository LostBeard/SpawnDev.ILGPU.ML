using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.StableDiffusion;

using FV = ArrayView1D<float, Stride1D.Dense>;
using Buf = MemoryBuffer1D<float, Stride1D.Dense>;

/// <summary>
/// One ControlNet's output for one UNet call: a residual per skip connection plus one for the middle block (13 for SD 1.x), channel-last on the device, and the strength
/// it is applied with. Valid until the ControlNet runs again.
/// </summary>
public sealed class SdControlResiduals
{
    private readonly FV[] _views;
    internal SdControlResiduals(FV[] views, float strength) { _views = views; Strength = strength; }
    public float Strength { get; set; }
    public int Count => _views.Length;
    public FV this[int i] => _views[i];
}

/// <summary>
/// A ControlNet (lllyasviel's ControlNet v1.0 / v1.1 for SD 1.x in the single-file layout ComfyUI uses: <c>control_model.*</c>, e.g. <c>control_v11p_sd15_canny_fp16.safetensors</c>):
/// the UNet's encoder half (input blocks and middle block) fed the latent plus an encoded hint picture, whose block outputs go through zero-initialised 1x1 convolutions into
/// residuals for the UNet's skip connections and middle block. <see cref="RunAsync(FV, int, int, double, FV, int, FV, FV?, CancellationToken)"/> computes them,
/// <see cref="SdUNet.ForwardAsync(FV, int, int, double, FV, int, FV?, FV, IReadOnlyList{SdControlResiduals}?, CancellationToken)"/> adds them. The hint is the control picture
/// [3, 8h, 8w] in [0, 1] (a canny edge map, a depth map, ...), as ComfyUI passes it.
/// </summary>
public sealed class SdControlNet : IDisposable
{
    private readonly Accelerator _acc;
    private readonly SdWeights _w;
    private readonly SdUNet _enc;
    private readonly SdKernels _k;
    private readonly DiTKernels _dk;
    private readonly Dictionary<(int, int, int, int, int), MemoryBuffer1D<int, Stride1D.Dense>> _tables = new();
    private readonly List<Buf> _outs = new();
    private readonly Dictionary<int, Buf> _slots = new();
    private readonly List<IDisposable> _retired = new();
    private readonly int _hintBlocks;

    /// <summary>The number of residuals one call produces (skip connections + middle block).</summary>
    public int ResidualCount { get; }
    /// <summary>The text context width it expects (768 for SD 1.x).</summary>
    public int ContextDim => _enc.ContextDim;

    private SdControlNet(Accelerator acc, SdWeights w)
    {
        _acc = acc; _w = w; _enc = new SdUNet(acc, w); _k = new SdKernels(acc); _dk = new DiTKernels(acc);
        int zeros = 0; while (w.Has($"zero_convs.{zeros}.0.weight")) zeros++;
        ResidualCount = zeros + 1;
        while (w.Has($"input_hint_block.{2 * _hintBlocks}.weight")) _hintBlocks++;
    }

    /// <summary>Load a ControlNet file from a model source.</summary>
    public static async Task<SdControlNet> LoadAsync(Accelerator accelerator, IModelSource source, string repo, string file, CancellationToken cancellationToken = default)
    {
        await using var s = await source.OpenAsync(repo, file, cancellationToken);
        return await LoadAsync(accelerator, s, $"{repo}/{file}", cancellationToken);
    }

    /// <summary>Load a ControlNet file from a stream (read once, front to back).</summary>
    public static async Task<SdControlNet> LoadAsync(Accelerator accelerator, Stream stream, string name, CancellationToken cancellationToken = default)
    {
        var w = await SdComponentFile.LoadAsync(accelerator, stream, name, tensors =>
        {
            if (tensors.Any(t => t.Name.StartsWith("controlnet_cond_embedding.", StringComparison.Ordinal)))
                throw new NotSupportedException($"'{name}': a diffusers-format ControlNet; this loader reads the single-file (control_model.*) layout ComfyUI uses");
            if (tensors.Any(t => t.Name.Contains("lora", StringComparison.OrdinalIgnoreCase)))
                throw new NotSupportedException($"'{name}': a Control-LoRA; only full ControlNets are supported");
            return n =>
            {
                var s = SdComponentFile.StripPrefix(n, "control_model.");
                return s.StartsWith("input_blocks.", StringComparison.Ordinal) || s.StartsWith("middle_block.", StringComparison.Ordinal) || s.StartsWith("time_embed.", StringComparison.Ordinal)
                    || s.StartsWith("label_emb.", StringComparison.Ordinal) || s.StartsWith("input_hint_block.", StringComparison.Ordinal) || s.StartsWith("zero_convs.", StringComparison.Ordinal)
                    || s.StartsWith("middle_block_out.", StringComparison.Ordinal) ? (s, SdLayout.ConvGather) : null;
            };
        }, null, null, cancellationToken);
        try
        {
            if (!w.Has("input_hint_block.0.weight") || !w.Has("middle_block_out.0.weight")) throw new InvalidDataException($"'{name}': not a ControlNet (no input_hint_block / middle_block_out)");
            await SdNames.FuseUNetAsync(accelerator, w);
            return new SdControlNet(accelerator, w);
        }
        catch { w.Dispose(); throw; }
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

    private ArrayView1D<int, Stride1D.Dense> Table(int inH, int inW, int outH, int outW, int stride)
    {
        var key = (inH, inW, outH, outW, stride);
        if (_tables.TryGetValue(key, out var t)) return t.View;
        t = _acc.Allocate1D<int>((long)outH * outW * 27);
        _k.BuildTable(t.View, inH, inW, outH, outW, 3, stride, 1, 1);
        _tables[key] = t;
        return t.View;
    }

    /// <summary>
    /// The residuals for one UNet call: the same latent, timestep and text context the UNet gets (<paramref name="latent"/> [4, h, w] channel-first, the scaled latent),
    /// <paramref name="hint"/> the control picture [3, 8h, 8w] channel-first in [0, 1]. The returned set (strength 1, change it with <see cref="SdControlResiduals.Strength"/>)
    /// lives in this ControlNet's buffers until it runs again.
    /// </summary>
    public async Task<SdControlResiduals> RunAsync(FV latent, int h, int w, double timestep, FV context, int contextRows, FV hint, FV? y = null, CancellationToken cancellationToken = default)
    {
        int H = 8 * h, W = 8 * w, rows = H * W;
        // the hint encoder: 3x3 convolutions with SiLU between them, three of stride 2 (8x down), the last (zero-initialised) to the UNet's width
        var x = Slot(0, 3L * rows);
        _dk.Transpose(hint, x, 3, rows);
        int c = 3, ch = H, cw = W;
        var cur = x;
        for (int i = 0; i < _hintBlocks; i++)
        {
            string p = $"input_hint_block.{2 * i}.";
            int cout = _w.Shape(p + "weight")[0];
            bool down = cout != c && i > 0 && i < _hintBlocks - 1;      // the channel-changing convolutions between the first and the last (16->32, 32->96, 96->256) have stride 2
            int oh = down ? (ch + 1) / 2 : ch, ow = down ? (cw + 1) / 2 : cw;
            var o = Slot(1 + (i & 1), (long)oh * ow * cout);
            _k.Conv(cur, Table(ch, cw, oh, ow, down ? 2 : 1), _w[p + "weight"], _w[p + "bias"], o, oh * ow, c, cout, 9);
            if (i < _hintBlocks - 1) _k.Sparse.SiluInPlace(o, (int)((long)oh * ow * cout));
            cur = o; c = cout; ch = oh; cw = ow;
        }
        if (ch != h || cw != w || c != _enc.ModelChannels) throw new InvalidOperationException($"the hint encoder gave {c} x {ch} x {cw}, the UNet needs {_enc.ModelChannels} x {h} x {w}");
        var guided = cur;
        // the encoder half with the hint added after conv_in; every block output through its zero convolution
        while (_outs.Count < ResidualCount) _outs.Add(_acc.Allocate1D<float>(64));
        var views = new FV[ResidualCount];
        await _enc.RunAsync(latent, h, w, timestep, context, contextRows, y, default, null, guided, (i, view, bh, bw, bc) =>
        {
            long n = (long)bh * bw * bc;
            if (_outs[i].Length < n) { _retired.Add(_outs[i]); _outs[i] = _acc.Allocate1D<float>(n); }
            string p = i < ResidualCount - 1 ? $"zero_convs.{i}.0." : "middle_block_out.0.";
            _k.Linear(view, _w[p + "weight"], _w[p + "bias"], _outs[i].View.SubView(0, n), bh * bw, bc, bc);
            views[i] = _outs[i].View.SubView(0, n);
        }, cancellationToken);
        return new SdControlResiduals(views, 1f);
    }

    /// <summary>The residuals from host arrays: <paramref name="latent"/> [4, h, w], <paramref name="context"/> [rows, 768], <paramref name="hint"/> [3, 8h, 8w] in [0, 1].</summary>
    public async Task<SdControlResiduals> RunAsync(float[] latent, int h, int w, double timestep, float[] context, int contextRows, float[] hint, CancellationToken cancellationToken = default)
    {
        var lat = Slot(10, latent.Length); lat.CopyFromCPU(latent);
        var ctx = Slot(11, context.Length); ctx.CopyFromCPU(context);
        var hv = Slot(12, hint.Length); hv.CopyFromCPU(hint);
        return await RunAsync(lat, h, w, timestep, ctx, contextRows, hv, null, cancellationToken);
    }

    public void Dispose()
    {
        _enc.Dispose();
        foreach (var b in _outs) b.Dispose();
        _outs.Clear();
        foreach (var b in _slots.Values) b.Dispose();
        _slots.Clear();
        foreach (var r in _retired) r.Dispose();
        _retired.Clear();
        foreach (var t in _tables.Values) t.Dispose();
        _tables.Clear();
        _w.Dispose();
    }
}
