using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.StableDiffusion;

using FV = ArrayView1D<float, Stride1D.Dense>;
using Buf = MemoryBuffer1D<float, Stride1D.Dense>;

/// <summary>
/// TAESD / TAESDXL (madebyollin's tiny autoencoder: <c>taesd_decoder.safetensors</c>, <c>taesd_encoder.safetensors</c>, and the diffusers <c>AutoencoderTiny</c> file),
/// ComfyUI's live-preview decoder and its VAELoader "taesd" / "taesdxl" entries. A drop-in for <see cref="SdVae"/>: the same UNSCALED latents and [-1, 1] pictures
/// (the tiny model itself works on the scaled latent: it multiplies by the family's latent scale on the way in and divides on the way out). 64-channel residual
/// blocks of three 3x3 convolutions, nearest x2 upsampling; about 1 M parameters, a fraction of the full VAE's time.
/// </summary>
public sealed class SdTaesd : IDisposable
{
    private readonly Accelerator _acc;
    private readonly SdWeights? _dec, _enc;
    private readonly SdKernels _k;
    private readonly DiTKernels _dk;
    private readonly Dictionary<(int, int, int, int, int, int, int, int), MemoryBuffer1D<int, Stride1D.Dense>> _tables = new();
    private readonly Dictionary<int, Buf> _slots = new();
    private readonly List<IDisposable> _retired = new();
    private readonly MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense> _zeroBias;

    /// <summary>The latent scale the caller's latents are divided by (0.18215 SD 1.x / 2.x, 0.13025 SDXL).</summary>
    public float LatentScale { get; }
    public bool HasDecoder => _dec != null;
    public bool HasEncoder => _enc != null;

    private SdTaesd(Accelerator acc, SdWeights? dec, SdWeights? enc, float latentScale)
    {
        _acc = acc; _dec = dec; _enc = enc; LatentScale = latentScale;
        _k = new SdKernels(acc); _dk = new DiTKernels(acc);
        _zeroBias = acc.Allocate1D<global::ILGPU.Half>(64);
        _zeroBias.MemSetToZero();
    }

    /// <summary>
    /// Load the decoder (and optionally the encoder) from a model source. <paramref name="latentScale"/> is the family's: 0.18215 for TAESD (SD 1.x / 2.x),
    /// 0.13025 for TAESDXL. A file may also hold both halves (the diffusers AutoencoderTiny file): pass it as the decoder and leave the encoder null.
    /// </summary>
    public static async Task<SdTaesd> LoadAsync(Accelerator accelerator, IModelSource source, string repo, string decoderFile, string? encoderFile, float latentScale,
        CancellationToken cancellationToken = default)
    {
        await using var d = await source.OpenAsync(repo, decoderFile, cancellationToken);
        if (encoderFile == null) return await LoadAsync(accelerator, d, $"{repo}/{decoderFile}", null, null, latentScale, cancellationToken);
        await using var e = await source.OpenAsync(repo, encoderFile, cancellationToken);
        return await LoadAsync(accelerator, d, $"{repo}/{decoderFile}", e, $"{repo}/{encoderFile}", latentScale, cancellationToken);
    }

    /// <summary>Load from streams (the encoder stream may be null: decode only).</summary>
    public static async Task<SdTaesd> LoadAsync(Accelerator accelerator, Stream decoder, string decoderName, Stream? encoder, string? encoderName, float latentScale,
        CancellationToken cancellationToken = default)
    {
        var dec = new SdWeights(); SdWeights? enc = null;
        try
        {
            // names: the original files hold the nn.Sequential indices ("3.conv.0.weight"), ComfyUI prefixes them ("taesd_decoder."), diffusers AutoencoderTiny
            // puts both halves in one file under "decoder.layers." / "encoder.layers."
            bool combined = false;
            await SdComponentFile.LoadAsync(accelerator, decoder, decoderName, tensors =>
            {
                combined = tensors.Any(t => t.Name.StartsWith("encoder.layers.", StringComparison.Ordinal));
                return n => Route(n, isDecoder: true, combinedFile: combined);
            }, dec, null, cancellationToken);
            if (combined)
            {
                enc = new SdWeights();
                foreach (var name in dec.Names.Where(n => n.StartsWith("e:", StringComparison.Ordinal)).ToList())
                    dec.MoveTo(name, enc, name[2..]);
            }
            else if (encoder != null)
            {
                enc = new SdWeights();
                await SdComponentFile.LoadAsync(accelerator, encoder, encoderName ?? "encoder", _ => n => Route(n, isDecoder: false, combinedFile: false), enc, null, cancellationToken);
            }
            if (!dec.Has("1.weight")) throw new InvalidDataException($"'{decoderName}': not a TAESD decoder (no layer 1 convolution)");
            return new SdTaesd(accelerator, dec, enc, latentScale);
        }
        catch { dec.Dispose(); enc?.Dispose(); throw; }
    }

    private static (string, SdLayout)? Route(string n, bool isDecoder, bool combinedFile)
    {
        if (combinedFile)
        {
            if (n.StartsWith("decoder.layers.", StringComparison.Ordinal))
            {
                // diffusers' DecoderTiny applies the tanh clamp in forward() instead of as layer 0: its layer i is the original layer i + 1
                var rest = n["decoder.layers.".Length..];
                int dot = rest.IndexOf('.');
                return ((int.Parse(rest[..dot]) + 1) + rest[dot..], SdLayout.ConvGather);
            }
            if (n.StartsWith("encoder.layers.", StringComparison.Ordinal)) return ("e:" + n["encoder.layers.".Length..], SdLayout.ConvGather);
            return null;
        }
        var s = SdComponentFile.StripPrefix(n, isDecoder ? "taesd_decoder." : "taesd_encoder.", isDecoder ? "decoder.layers." : "encoder.layers.", isDecoder ? "decoder." : "encoder.");
        return char.IsDigit(s[0]) ? (s, SdLayout.ConvGather) : null;
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

    private ArrayView1D<int, Stride1D.Dense> Table(int inH, int inW, int outH, int outW, int stride, int up)
    {
        var key = (inH, inW, outH, outW, 3, stride, 1, up);
        if (_tables.TryGetValue(key, out var t)) return t.View;
        t = _acc.Allocate1D<int>((long)outH * outW * 27);
        _k.BuildTable(t.View, inH, inW, outH, outW, 3, stride, 1, up);
        _tables[key] = t;
        return t.View;
    }

    private void Conv(SdWeights w, string p, FV x, FV y, int inH, int inW, int outH, int outW, int cin, int cout, int stride = 1, int up = 1)
        => _k.Conv(x, Table(inH, inW, outH, outW, stride, up), w[p + "weight"], w.Opt(p + "bias") ?? _zeroBias.View.SubView(0, cout), y, outH * outW, cin, cout, 9);

    /// <summary>Block: relu(conv(relu(conv(relu(conv(x))))) + x), 64 channels; x is kept, dst receives the output.</summary>
    private void Block(SdWeights w, int i, FV x, FV dst, int H, int W)
    {
        int n = H * W * 64;
        var a = Slot(2, n); var b = Slot(3, n);
        Conv(w, $"{i}.conv.0.", x, a, H, W, H, W, 64, 64); _k.ReluInPlace(a, n);
        Conv(w, $"{i}.conv.2.", a, b, H, W, H, W, 64, 64); _k.ReluInPlace(b, n);
        Conv(w, $"{i}.conv.4.", b, dst, H, W, H, W, 64, 64);
        _k.AddReluInPlace(dst, x, n);
    }

    private FV Other(FV cur)
    {
        IContiguousArrayView cv = cur.BaseView;
        return cv.Buffer == ((IContiguousArrayView)_slots[0].View.BaseView).Buffer ? _slots[1].View : _slots[0].View;
    }

    /// <summary>Decode on the device: <paramref name="latent"/> [4, h, w] UNSCALED (as <see cref="SdVae"/>), <paramref name="image"/> [3, 8h, 8w] in [-1, 1] (not clamped).</summary>
    public async Task DecodeAsync(FV latent, int h, int w, FV image, CancellationToken cancellationToken = default)
    {
        var d = _dec ?? throw new InvalidOperationException("no TAESD decoder loaded");
        int L = d.Shape("1.weight")[1];
        long most = 64L * 64 * h * w;
        Slot(0, most); Slot(1, most); Slot(2, most); Slot(3, most);
        int rows = h * w;
        var zin = Slot(4, (long)rows * L);
        _dk.Transpose(latent, zin, L, rows);
        _k.TaesdIn(zin, (long)rows * L, LatentScale);                 // tanh(x * scale / 3) * 3: the scaled latent, clamped
        var cur = Slot(0, (long)rows * 64);
        Conv(d, "1.", zin, cur, h, w, h, w, L, 64); _k.ReluInPlace(cur, (long)rows * 64);
        int H = h, W = w;
        // (blocks, upsample + conv) x 3, then a block and the output convolution
        int[][] stages = { new[] { 3, 4, 5 }, new[] { 8, 9, 10 }, new[] { 13, 14, 15 } };
        int[] ups = { 7, 12, 17 };
        for (int s = 0; s < 3; s++)
        {
            foreach (int bi in stages[s]) { var nx = Other(cur).SubView(0, (long)H * W * 64); Block(d, bi, cur, nx, H, W); cur = nx; }
            var upd = Other(cur).SubView(0, 4L * H * W * 64);
            Conv(d, $"{ups[s]}.", cur, upd, H, W, 2 * H, 2 * W, 64, 64, up: 2);
            cur = upd; H *= 2; W *= 2;
            if (_acc.IsBrowser()) await _acc.SynchronizeAsync();
            cancellationToken.ThrowIfCancellationRequested();
        }
        { var nx = Other(cur).SubView(0, (long)H * W * 64); Block(d, 18, cur, nx, H, W); cur = nx; }
        var o = Slot(5, (long)H * W * 3);
        Conv(d, "19.", cur, o, H, W, H, W, 64, 3);
        _k.Affine(o, (long)H * W * 3, 2f, -1f);                     // [0, 1] -> [-1, 1]
        _dk.Transpose(o, image, H * W, 3);
    }

    /// <summary>Decode from a host array: [4, h, w] UNSCALED -> [3, 8h, 8w] in [-1, 1].</summary>
    public async Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken = default)
    {
        var zin = Slot(10, latent.Length); zin.CopyFromCPU(latent);
        var img = Slot(11, 3L * 64 * h * w);
        await DecodeAsync(zin, h, w, img, cancellationToken);
        await _acc.SynchronizeAsync();
        return await img.CopyToHostAsync();
    }

    /// <summary>Encode on the device: <paramref name="image"/> [3, H, W] in [-1, 1] (multiples of 8), <paramref name="latent"/> receives [4, H/8, W/8] UNSCALED.</summary>
    public async Task EncodeAsync(FV image, int H, int W, FV latent, CancellationToken cancellationToken = default)
    {
        var e = _enc ?? throw new InvalidOperationException("no TAESD encoder loaded");
        if (H % 8 != 0 || W % 8 != 0) throw new ArgumentException("the picture sides must be multiples of 8");
        int L = e.Shape("14.weight")[0];
        long most = 64L * H * W;
        Slot(0, most); Slot(1, most); Slot(2, most); Slot(3, most);
        int rows = H * W;
        var xin = Slot(4, 3L * rows);
        _dk.Transpose(image, xin, 3, rows);
        _k.Affine(xin, 3L * rows, 0.5f, 0.5f);                      // [-1, 1] -> [0, 1]
        var cur = Slot(0, (long)rows * 64);
        Conv(e, "0.", xin, cur, H, W, H, W, 3, 64);
        int h = H, w = W;
        { var nx = Other(cur).SubView(0, (long)h * w * 64); Block(e, 1, cur, nx, h, w); cur = nx; }
        int[] downs = { 2, 6, 10 };
        int[][] stages = { new[] { 3, 4, 5 }, new[] { 7, 8, 9 }, new[] { 11, 12, 13 } };
        for (int s = 0; s < 3; s++)
        {
            int oh = h / 2, ow = w / 2;
            var dn = Other(cur).SubView(0, (long)oh * ow * 64);
            Conv(e, $"{downs[s]}.", cur, dn, h, w, oh, ow, 64, 64, stride: 2);
            cur = dn; h = oh; w = ow;
            foreach (int bi in stages[s]) { var nx = Other(cur).SubView(0, (long)h * w * 64); Block(e, bi, cur, nx, h, w); cur = nx; }
            if (_acc.IsBrowser()) await _acc.SynchronizeAsync();
            cancellationToken.ThrowIfCancellationRequested();
        }
        var o = Slot(5, (long)h * w * L);
        Conv(e, "14.", cur, o, h, w, h, w, 64, L);
        _k.Affine(o, (long)h * w * L, 1f / LatentScale, 0f);         // the scaled latent -> unscaled
        _dk.Transpose(o, latent, h * w, L);
    }

    /// <summary>Encode from a host array: [3, H, W] in [-1, 1] -> [4, H/8, W/8] UNSCALED.</summary>
    public async Task<float[]> EncodeAsync(float[] image, int H, int W, CancellationToken cancellationToken = default)
    {
        var xin = Slot(10, image.Length); xin.CopyFromCPU(image);
        var lat = Slot(11, 4L * (H / 8) * (W / 8));
        await EncodeAsync(xin, H, W, lat, cancellationToken);
        await _acc.SynchronizeAsync();
        return await lat.CopyToHostAsync();
    }

    public void Dispose()
    {
        foreach (var b in _slots.Values) b.Dispose();
        _slots.Clear();
        foreach (var r in _retired) r.Dispose();
        _retired.Clear();
        foreach (var t in _tables.Values) t.Dispose();
        _tables.Clear();
        _zeroBias.Dispose();
        _dec?.Dispose(); _enc?.Dispose();
    }
}
