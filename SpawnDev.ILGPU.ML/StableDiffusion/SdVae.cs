using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.StableDiffusion;

using FV = ArrayView1D<float, Stride1D.Dense>;
using Buf = MemoryBuffer1D<float, Stride1D.Dense>;

/// <summary>
/// The Stable Diffusion VAE (LDM <c>AutoencoderKL</c>, read from the single-file checkpoint's <c>first_stage_model.*</c> tensors; SD 1.x, SD 2.x and SDXL share the layout):
/// <see cref="DecodeAsync(float[], int, int, CancellationToken)"/> turns a latent [4, h, w] into a picture [3, 8h, 8w] in [-1, 1] (not clamped), <see cref="EncodeAsync(float[], int, int, CancellationToken)"/>
/// turns a picture into the posterior MEAN [4, H/8, W/8]. Both work on the UNSCALED latent: the caller multiplies by <see cref="SdCheckpoint.LatentScale"/> (0.18215 SD 1.x / 2.x,
/// 0.13025 SDXL) after encoding and divides by it before decoding, as ComfyUI does. Activations are channel-last fp32 and every convolution keeps them fp32 (the portable tiled
/// GEMM, fp16 weights): SDXL's VAE overflows fp16 activations. The mid-block attention (one head over every latent pixel) is two chunked GEMMs.
/// </summary>
public sealed class SdVae : IDisposable
{
    private readonly Accelerator _acc;
    private readonly SdWeights _w;
    private readonly SdKernels _k;
    private readonly DiTKernels _dk;
    private readonly Dictionary<(int, int, int, int, int, int, int, int), MemoryBuffer1D<int, Stride1D.Dense>> _tables = new();
    private readonly Dictionary<int, Buf> _slots = new();
    private readonly Dictionary<int, MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>> _hslots = new();
    private readonly List<IDisposable> _retired = new();

    /// <summary>Latent channels (4).</summary>
    public int LatentChannels { get; }
    /// <summary>The number of resolution levels (4: an 8x reduction).</summary>
    public int Levels { get; }
    public bool HasDecoder { get; }
    public bool HasEncoder { get; }

    /// <summary>Test hook: (stage name, activation view [pixels, C] channel-last, height, width, channels); valid only during the call.</summary>
    public Action<string, FV, int, int, int>? Tap { get; set; }

    public SdVae(Accelerator accelerator, SdWeights weights)
    {
        _acc = accelerator; _w = weights; _k = new SdKernels(accelerator); _dk = new DiTKernels(accelerator);
        HasDecoder = weights.Has("decoder.conv_in.weight");
        HasEncoder = weights.Has("encoder.conv_in.weight");
        LatentChannels = HasDecoder ? weights.Shape("decoder.conv_in.weight")[1] : weights.Shape("quant_conv.weight")[0] / 2;
        int levels = 0;
        while (weights.Has($"decoder.up.{levels}.block.0.conv1.weight") || weights.Has($"encoder.down.{levels}.block.0.conv1.weight")) levels++;
        Levels = levels;
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
    private ArrayView1D<global::ILGPU.Half, Stride1D.Dense> HSlot(int slot, long n)
    {
        if (!_hslots.TryGetValue(slot, out var b) || b.Length < n)
        {
            if (b != null) _retired.Add(b);
            b = _acc.Allocate1D<global::ILGPU.Half>(Math.Max(n, 64));
            _hslots[slot] = b;
        }
        return b.View.SubView(0, n);
    }
    private const int SlotA = 0, SlotB = 1, SlotNorm = 2, SlotT = 3, SlotShort = 4, SlotQkv = 5, SlotQ = 6, SlotScores = 7, SlotAttn = 8, SlotIn = 9, SlotOut = 10, SlotHostIn = 11, SlotHostOut = 12;

    private ArrayView1D<int, Stride1D.Dense> Table(int inH, int inW, int outH, int outW, int k, int stride, int pad, int up)
    {
        var key = (inH, inW, outH, outW, k, stride, pad, up);
        if (_tables.TryGetValue(key, out var t)) return t.View;
        t = _acc.Allocate1D<int>((long)outH * outW * 27);
        _k.BuildTable(t.View, inH, inW, outH, outW, k, stride, pad, up);
        _tables[key] = t;
        return t.View;
    }

    private void Conv3(string name, FV x, FV y, int H, int W, int cin, int cout)
        => _k.Conv(x, Table(H, W, H, W, 3, 1, 1, 1), _w[name + ".weight"], _w[name + ".bias"], y, H * W, cin, cout, 9, exact: true);

    private void Conv1(string name, FV x, FV y, int rows, int cin, int cout, int H, int W)
        => _k.Conv(x, Table(H, W, H, W, 1, 1, 0, 1), _w[name + ".weight"], _w[name + ".bias"], y, rows, cin, cout, 1, exact: true);

    private FV Other(FV cur)
    {
        var a = _slots.GetValueOrDefault(SlotA); IContiguousArrayView cv = cur.BaseView;
        return a != null && cv.Buffer == ((IContiguousArrayView)a.View.BaseView).Buffer ? _slots[SlotB].View : _slots[SlotA].View;
    }

    /// <summary>ResnetBlock (no time embedding): x [rows, cin] -> dst [rows, cout]; dst must not alias x.</summary>
    private void Resnet(string p, FV x, FV dst, int H, int W, int cin, int cout)
    {
        int rows = H * W;
        var n = Slot(SlotNorm, (long)rows * Math.Max(cin, cout));
        var nIn = n.SubView(0, (long)rows * cin);
        _k.GroupNorm(x, nIn, _w[p + "norm1.weight"], _w[p + "norm1.bias"], rows, cin, 32, 1e-6f, silu: true);
        var c1 = Slot(SlotT, (long)rows * cout);
        Conv3(p + "conv1", nIn, c1, H, W, cin, cout);
        var nOut = n.SubView(0, (long)rows * cout);
        _k.GroupNorm(c1, nOut, _w[p + "norm2.weight"], _w[p + "norm2.bias"], rows, cout, 32, 1e-6f, silu: true);
        Conv3(p + "conv2", nOut, dst, H, W, cout, cout);
        if (_w.Has(p + "nin_shortcut.weight"))
        {
            var s = Slot(SlotShort, (long)rows * cout);
            Conv1(p + "nin_shortcut", x, s, rows, cin, cout, H, W);
            _k.AddInPlace(dst, s, (long)rows * cout);
        }
        else _k.AddInPlace(dst, x, (long)rows * cout);
    }

    /// <summary>AttnBlock: one head of width C over every pixel, x [rows, C] -> dst [rows, C].</summary>
    private void Attn(string p, FV x, FV dst, int H, int W, int C)
    {
        int rows = H * W;
        var n = Slot(SlotNorm, (long)rows * C);
        _k.GroupNorm(x, n, _w[p + "norm.weight"], _w[p + "norm.bias"], rows, C, 32, 1e-6f, silu: false);
        var qkv = Slot(SlotQkv, (long)rows * 3 * C);
        Conv1(p + "qkv", n, qkv, rows, C, 3 * C, H, W);
        // the score chunk: at most ~2^26 floats (256 MB) of scores per GEMM pair
        int chunk = (int)Math.Max(64, Math.Min(rows, (1L << 26) / rows));
        var a = Slot(SlotAttn, (long)rows * C);
        _k.SingleHeadAttention(qkv, a, rows, C, HSlot(0, (long)rows * C), HSlot(1, (long)rows * C), Slot(SlotQ, (long)chunk * C), Slot(SlotScores, (long)chunk * rows), chunk);
        Conv1(p + "proj_out", a, dst, rows, C, C, H, W);
        _k.AddInPlace(dst, x, (long)rows * C);
    }

    private FV Mid(string p, FV cur, int H, int W, int C, long maxElems)
    {
        var d1 = Other(cur).SubView(0, (long)H * W * C);
        Resnet(p + "block_1.", cur, d1, H, W, C, C);
        var d2 = Other(d1).SubView(0, (long)H * W * C);
        Attn(p + "attn_1.", d1, d2, H, W, C);
        var d3 = Other(d2).SubView(0, (long)H * W * C);
        Resnet(p + "block_2.", d2, d3, H, W, C, C);
        return d3;
    }

    // ---------------------------------------------------------------------------------------------------
    // decode
    // ---------------------------------------------------------------------------------------------------

    /// <summary>
    /// Decode on the device: <paramref name="latent"/> [4, h, w] channel-first, UNSCALED (the sampler's latent divided by <see cref="SdCheckpoint.LatentScale"/>);
    /// <paramref name="image"/> receives [3, 8h, 8w] channel-first in [-1, 1] (not clamped).
    /// </summary>
    public async Task DecodeAsync(FV latent, int h, int w, FV image, CancellationToken cancellationToken = default)
    {
        if (!HasDecoder) throw new InvalidOperationException("this checkpoint's VAE has no decoder");
        int L = LatentChannels;
        int topC = _w.Shape("decoder.conv_in.weight")[0];
        // the largest activation: the full-size picture at the widest channel count of the last levels (the upsampled input of a level)
        long maxElems = 0;
        {
            int sh = h, sw = w;
            for (int i = Levels - 1; i >= 0; i--)
            {
                int cin = _w.Shape($"decoder.up.{i}.block.0.conv1.weight")[1], cout = _w.Shape($"decoder.up.{i}.block.0.conv1.weight")[0];
                maxElems = Math.Max(maxElems, (long)sh * sw * Math.Max(cin, cout));
                if (i > 0) { sh *= 2; sw *= 2; maxElems = Math.Max(maxElems, (long)sh * sw * cout); }
            }
            maxElems = Math.Max(maxElems, (long)h * w * topC);
        }
        Slot(SlotA, maxElems); Slot(SlotB, maxElems);
        int rows = h * w;
        var zin = Slot(SlotIn, (long)rows * L);
        _dk.Transpose(latent, zin, L, rows);
        var z = Slot(SlotT, (long)rows * L);
        Conv1("post_quant_conv", zin, z, rows, L, L, h, w);
        Tap?.Invoke("post_quant_conv", z, h, w, L);
        var cur = Slot(SlotA, (long)rows * topC);
        Conv3("decoder.conv_in", z, cur, h, w, L, topC);
        int H = h, W = w, C = topC;
        cur = Mid("decoder.mid.", cur, H, W, C, maxElems);
        Tap?.Invoke("dec_mid", cur, H, W, C);
        if (_acc.IsBrowser()) await _acc.SynchronizeAsync();
        int k = 0;
        for (int i = Levels - 1; i >= 0; i--, k++)
        {
            for (int j = 0; _w.Has($"decoder.up.{i}.block.{j}.conv1.weight"); j++)
            {
                string p = $"decoder.up.{i}.block.{j}.";
                int cout = _w.Shape(p + "conv1.weight")[0];
                var dst = Other(cur).SubView(0, (long)H * W * cout);
                Resnet(p, cur, dst, H, W, C, cout);
                cur = dst; C = cout;
                if (_acc.IsBrowser()) await _acc.SynchronizeAsync();
            }
            if (_w.Has($"decoder.up.{i}.upsample.conv.weight"))
            {
                var dst = Other(cur).SubView(0, 4L * H * W * C);
                _k.Conv(cur, Table(H, W, 2 * H, 2 * W, 3, 1, 1, 2), _w[$"decoder.up.{i}.upsample.conv.weight"], _w[$"decoder.up.{i}.upsample.conv.bias"], dst, 4 * H * W, C, C, 9, exact: true);
                cur = dst; H *= 2; W *= 2;
            }
            Tap?.Invoke($"dec_up_{k}", cur, H, W, C);
            if (_acc.IsBrowser()) await _acc.SynchronizeAsync();
            cancellationToken.ThrowIfCancellationRequested();
        }
        var n = Slot(SlotNorm, (long)H * W * C);
        _k.GroupNorm(cur, n, _w["decoder.norm_out.weight"], _w["decoder.norm_out.bias"], H * W, C, 32, 1e-6f, silu: true);
        int outC = _w.Shape("decoder.conv_out.weight")[0];
        var o = Slot(SlotOut, (long)H * W * outC);
        Conv3("decoder.conv_out", n, o, H, W, C, outC);
        _dk.Transpose(o, image, H * W, outC);
    }

    /// <summary>Decode from a host array: <paramref name="latent"/> [4, h, w] UNSCALED; returns [3, 8h, 8w] in [-1, 1] (not clamped).</summary>
    public async Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken = default)
    {
        if (latent.Length != (long)LatentChannels * h * w) throw new ArgumentException($"the latent must be [{LatentChannels}, {h}, {w}]");
        var zin = Slot(SlotHostIn, latent.Length); zin.CopyFromCPU(latent);
        int f = 1 << (Levels - 1);
        var img = Slot(SlotHostOut, 3L * h * f * w * f);
        await DecodeAsync(zin, h, w, img, cancellationToken);
        await _acc.SynchronizeAsync();
        return await img.CopyToHostAsync();
    }

    // ---------------------------------------------------------------------------------------------------
    // encode
    // ---------------------------------------------------------------------------------------------------

    /// <summary>
    /// Encode on the device: <paramref name="image"/> [3, H, W] channel-first in [-1, 1] (H and W multiples of 8); <paramref name="mean"/> receives the posterior mean
    /// [4, H/8, W/8] channel-first, UNSCALED (multiply by <see cref="SdCheckpoint.LatentScale"/> for the sampler).
    /// </summary>
    public async Task EncodeAsync(FV image, int H, int W, FV mean, CancellationToken cancellationToken = default)
    {
        if (!HasEncoder) throw new InvalidOperationException("this checkpoint's VAE has no encoder");
        int f = 1 << (Levels - 1);
        if (H % f != 0 || W % f != 0) throw new ArgumentException($"the picture sides must be multiples of {f}");
        int inC = _w.Shape("encoder.conv_in.weight")[1], c0 = _w.Shape("encoder.conv_in.weight")[0];
        long maxElems = 0;
        for (int i = 0, h2 = H, w2 = W; i < Levels; i++)
        {
            int cout = _w.Shape($"encoder.down.{i}.block.0.conv1.weight")[0];
            maxElems = Math.Max(maxElems, (long)h2 * w2 * Math.Max(cout, i == 0 ? c0 : _w.Shape($"encoder.down.{i}.block.0.conv1.weight")[1]));
            if (i < Levels - 1) { h2 /= 2; w2 /= 2; }
        }
        Slot(SlotA, maxElems); Slot(SlotB, maxElems);
        int rows = H * W;
        var xin = Slot(SlotIn, (long)rows * inC);
        _dk.Transpose(image, xin, inC, rows);
        var cur = Slot(SlotA, (long)rows * c0);
        Conv3("encoder.conv_in", xin, cur, H, W, inC, c0);
        int Hc = H, Wc = W, C = c0;
        for (int i = 0; i < Levels; i++)
        {
            for (int j = 0; _w.Has($"encoder.down.{i}.block.{j}.conv1.weight"); j++)
            {
                string p = $"encoder.down.{i}.block.{j}.";
                int cout = _w.Shape(p + "conv1.weight")[0];
                var dst = Other(cur).SubView(0, (long)Hc * Wc * cout);
                Resnet(p, cur, dst, Hc, Wc, C, cout);
                cur = dst; C = cout;
                if (_acc.IsBrowser()) await _acc.SynchronizeAsync();
            }
            if (_w.Has($"encoder.down.{i}.downsample.conv.weight"))
            {
                // pad (0, 1, 0, 1) then a 3x3 convolution of stride 2 with no padding
                int oh = Hc / 2, ow = Wc / 2;
                var dst = Other(cur).SubView(0, (long)oh * ow * C);
                _k.Conv(cur, Table(Hc, Wc, oh, ow, 3, 2, 0, 1), _w[$"encoder.down.{i}.downsample.conv.weight"], _w[$"encoder.down.{i}.downsample.conv.bias"], dst, oh * ow, C, C, 9, exact: true);
                cur = dst; Hc = oh; Wc = ow;
            }
            cancellationToken.ThrowIfCancellationRequested();
        }
        cur = Mid("encoder.mid.", cur, Hc, Wc, C, maxElems);
        var n = Slot(SlotNorm, (long)Hc * Wc * C);
        _k.GroupNorm(cur, n, _w["encoder.norm_out.weight"], _w["encoder.norm_out.bias"], Hc * Wc, C, 32, 1e-6f, silu: true);
        int momC = _w.Shape("encoder.conv_out.weight")[0];
        var mom = Slot(SlotT, (long)Hc * Wc * momC);
        Conv3("encoder.conv_out", n, mom, Hc, Wc, C, momC);
        var q = Slot(SlotOut, (long)Hc * Wc * momC);
        Conv1("quant_conv", mom, q, Hc * Wc, momC, momC, Hc, Wc);
        // the mean is the first half of the moments' channels: channel-last [rows, 2L] -> channel-first [2L, rows], whose first L rows are the mean
        var t = Slot(SlotQkv, (long)Hc * Wc * momC);
        _dk.Transpose(q, t, Hc * Wc, momC);
        mean.SubView(0, (long)LatentChannels * Hc * Wc).CopyFrom(t.SubView(0, (long)LatentChannels * Hc * Wc));
    }

    /// <summary>Encode from a host array: <paramref name="image"/> [3, H, W] in [-1, 1]; returns the UNSCALED posterior mean [4, H/8, W/8].</summary>
    public async Task<float[]> EncodeAsync(float[] image, int H, int W, CancellationToken cancellationToken = default)
    {
        if (image.Length != 3L * H * W) throw new ArgumentException($"the picture must be [3, {H}, {W}]");
        var xin = Slot(SlotHostIn, image.Length); xin.CopyFromCPU(image);
        int f = 1 << (Levels - 1);
        var m = Slot(SlotHostOut, (long)LatentChannels * (H / f) * (W / f));
        await EncodeAsync(xin, H, W, m, cancellationToken);
        await _acc.SynchronizeAsync();
        return await m.CopyToHostAsync();
    }

    public void Dispose()
    {
        foreach (var b in _slots.Values) b.Dispose();
        _slots.Clear();
        foreach (var b in _hslots.Values) b.Dispose();
        _hslots.Clear();
        foreach (var r in _retired) r.Dispose();
        _retired.Clear();
        foreach (var t in _tables.Values) t.Dispose();
        _tables.Clear();
    }
}
