using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Wan;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;
using Buf = MemoryBuffer1D<float, Stride1D.Dense>;

/// <summary>
/// The Wan 2.2 VAE (<c>Wan2_2_VAE</c>): <see cref="EncodeImageAsync"/> for the image conditioning and the decoder (<c>Wan2_2_VAE.decode</c>): the latent [48, Fz, h, w] becomes video [3, 4 Fz - 3, 16 h, 16 w] in [-1, 1].
/// The reference decodes ONE latent frame per chunk through causal 3-D convolutions that keep the last two input frames of every convolution as a cache (so a chunk sees
/// its history without recomputing it); the first chunk decodes to one video frame, every later one to four. This port keeps the same chunking and caches, with
/// channel-last fp32 activations [voxels, C], fp16 weights, and every convolution as a gathered-row implicit GEMM (<see cref="SparseKernels.ConvTaps"/>, tensor cores on CUDA):
/// the neighbour tables of the dense grids (a 3x3x3 causal kernel, a (3,1,1) temporal one, and nearest x2 + 3x3) are built on the device and shared by all convolutions of a shape.
/// Weights: the export of tools/wan-harness/reference/export_vae_f16.py.
/// </summary>
public sealed class WanVae : IDisposable
{
    private readonly Accelerator _acc;
    private readonly GpuWeights _w;
    private readonly SparseKernels _sk;
    private readonly WanKernels _wk;
    private readonly DiTKernels _dk;
    private readonly List<Buf> _live = new();
    private readonly Dictionary<(int Kind, int T, int H, int W, int Cnt), MemoryBuffer1D<int, Stride1D.Dense>> _tables = new();
    private readonly Dictionary<string, Cache> _caches = new();
    private Buf? _scores;
    private bool _cacheEnabled = true;     // the encoder sees one frame: nothing to cache for a later chunk

    /// <summary>The last (up to) two input frames of a causal convolution, channel-last [frames * H * W, C]; <see cref="Rep"/> = the "Rep" marker of upsample3d's first call.</summary>
    private sealed class Cache { public Buf? Buf; public int Frames; public bool Rep; }

    /// <summary>Test hook: (stage name, activation view, frames, height, width, channels), called per chunk; valid only during the call.</summary>
    public Action<string, FV, int, int, int, int>? Tap { get; set; }

    public WanVae(Accelerator accelerator, GpuWeights weights)
    {
        _acc = accelerator; _w = weights; _sk = new SparseKernels(accelerator); _wk = new WanKernels(accelerator); _dk = new DiTKernels(accelerator);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------
    // buffers: temporaries are queued and freed together (after a fence: on WebGPU a buffer must outlive the dispatches that read it)
    // -----------------------------------------------------------------------------------------------------------------------------------
    private Buf Tmp(long n) { var b = _acc.Allocate1D<float>(n); _live.Add(b); return b; }
    private void Drop(Buf b) { /* freed at the next Flush */ _drop.Add(b); }
    private readonly List<Buf> _drop = new();
    private void Flush()
    {
        if (_drop.Count == 0) return;
        _acc.Fence();
        foreach (var b in _drop) { _live.Remove(b); b.Dispose(); }
        _drop.Clear();
    }

    private MemoryBuffer1D<int, Stride1D.Dense> Table(int kind, int T, int H, int W, int cnt)
    {
        var key = (kind, T, H, W, kind >= 2 ? 0 : cnt);
        if (_tables.TryGetValue(key, out var t)) return t;
        long rows = kind == 2 ? 4L * T * H * W : kind == 3 ? (long)T * (H / 2) * (W / 2) : (long)T * H * W;
        t = _acc.Allocate1D<int>(rows * 27);
        switch (kind)
        {
            case 0: _wk.CausalTable(t.View, T, H, W, cnt); break;          // 3x3x3 causal, output [T, H, W]
            case 1: _wk.TimeTable(t.View, T, H, W, cnt); break;            // (3,1,1) causal, output [T, H, W]
            case 2: _wk.UpTable(t.View, T, H, W); break;                   // nearest x2 + 3x3, input [T, H, W], output [T, 2H, 2W]
            case 3: _wk.DownTable(t.View, T, H, W); break;                 // zero pad right / bottom, 3x3 stride 2, input [T, H, W], output [T, H/2, W/2]
        }
        _tables[key] = t;
        return t;
    }

    private Buf Concat(Cache? c, int cnt, Buf x, int T, int hw, int cin)
    {
        if (cnt == 0 || c?.Buf == null) return x;
        var cat = Tmp((long)(cnt + T) * hw * cin);
        cat.View.SubView(0, (long)cnt * hw * cin).CopyFrom(c.Buf.View.SubView(0, (long)cnt * hw * cin));
        cat.View.SubView((long)cnt * hw * cin, (long)T * hw * cin).CopyFrom(x.View.SubView(0, (long)T * hw * cin));
        return cat;
    }

    private void StoreCache(string name, Buf xcat, int frames, int hw, int cin)
    {
        // the last min(2, frames) frames of the concatenated input
        int keep = Math.Min(2, frames);
        if (!_caches.TryGetValue(name, out var c)) _caches[name] = c = new Cache();
        c.Buf ??= _acc.Allocate1D<float>(2L * hw * cin);
        c.Buf.View.SubView(0, (long)keep * hw * cin).CopyFrom(xcat.View.SubView((long)(frames - keep) * hw * cin, (long)keep * hw * cin));
        c.Frames = keep; c.Rep = false;
    }

    /// <summary>CausalConv3d 3x3x3 with the feature cache: x [T*H*W, cin] -> [T*H*W, cout].</summary>
    private Buf ConvCausal(string name, Buf x, int T, int H, int W, int cin, int cout)
    {
        int hw = H * W;
        _caches.TryGetValue(name, out var c);
        int cnt = c?.Frames ?? 0;
        var xcat = Concat(c, cnt, x, T, hw, cin);
        if (_cacheEnabled) StoreCache(name, xcat, cnt + T, hw, cin);
        var y = Tmp((long)T * hw * cout);
        _sk.ConvTaps(xcat.View, Table(0, T, H, W, cnt).View, _w.F16(name + ".weight"), _w.F16(name + ".bias"), y.View, T * hw, cin, cout, 27);
        if (!ReferenceEquals(xcat, x)) Drop(xcat);
        return y;
    }

    private Buf Pointwise(string name, Buf x, int M, int cin, int cout)
    {
        var y = Tmp((long)M * cout);
        _sk.LinearF16(x.View, _w.F16(name + ".weight"), _w.F16(name + ".bias"), y.View, M, cin, cout);
        return y;
    }

    /// <summary>RMS_norm over the channels of every voxel (a copy: the input stays), then SiLU into a new buffer.</summary>
    private Buf NormSilu(Buf x, string gamma, int rows, int C)
    {
        var y = Tmp((long)rows * C);
        y.View.CopyFrom(x.View.SubView(0, (long)rows * C));
        _wk.ChannelRmsNorm(y.View, _w.F32(gamma), rows, C);
        var s = Tmp((long)rows * C);
        _dk.Silu(y.View, s.View, rows * C);
        Drop(y);
        return s;
    }

    private Buf ResBlock(string p, Buf x, int T, int H, int W, int cin, int cout)
    {
        int rows = T * H * W;
        var h = cin != cout ? Pointwise(p + "shortcut", x, rows, cin, cout) : x;
        var a = NormSilu(x, p + "residual.0.gamma", rows, cin);
        var c1 = ConvCausal(p + "residual.2", a, T, H, W, cin, cout); Drop(a);
        var b = NormSilu(c1, p + "residual.3.gamma", rows, cout); Drop(c1);
        var c2 = ConvCausal(p + "residual.6", b, T, H, W, cout, cout); Drop(b);
        _dk.GateResidual(c2.View, h.View, default, -1, rows * cout, cout);          // c2 += h (the residual)
        if (!ReferenceEquals(h, x)) Drop(h);
        Flush();
        return c2;
    }

    private Buf Attention(string p, Buf x, int T, int H, int W, int C)
    {
        int hw = H * W, rows = T * hw;
        var y = Tmp((long)rows * C);
        y.View.CopyFrom(x.View.SubView(0, (long)rows * C));
        _wk.ChannelRmsNorm(y.View, _w.F32(p + "norm.gamma"), rows, C);
        var qkv = Pointwise(p + "to_qkv", y, rows, C, 3 * C); Drop(y);
        _scores ??= _acc.Allocate1D<float>((long)hw * hw);
        if (_scores.Length < (long)hw * hw) { _scores.Dispose(); _scores = _acc.Allocate1D<float>((long)hw * hw); }
        var o = Tmp((long)rows * C);
        for (int f = 0; f < T; f++)
            _wk.SpatialAttention(qkv.View.SubView((long)f * hw * 3 * C, (long)hw * 3 * C), _scores.View, o.View.SubView((long)f * hw * C, (long)hw * C), hw, C);
        Drop(qkv);
        var proj = Pointwise(p + "proj", o, rows, C, C); Drop(o);
        _dk.GateResidual(proj.View, x.View, default, -1, rows * C, C);              // proj += x (identity)
        Flush();
        return proj;
    }

    /// <summary>Resample "upsample2d" / "upsample3d": the optional causal temporal doubling, then nearest x2 and a 3x3 convolution. Returns the new frame count.</summary>
    private Buf Upsample(string p, Buf x, ref int T, int H, int W, int C, bool temporal)
    {
        int hw = H * W;
        var cur = x; int T2 = T;
        if (temporal)
        {
            string name = p + "time_conv";
            _caches.TryGetValue(name, out var c);
            if (c == null) _caches[name] = new Cache { Rep = true };                    // the first chunk: no temporal convolution, no doubling
            else
            {
                int cnt = c.Rep ? 0 : c.Frames;
                // the new cache: the last two frames of the input; one frame is completed with the previous cache's last frame, or with zeros after "Rep"
                var cat = Concat(c, cnt, x, T, hw, C);
                var conv = Tmp((long)T * hw * 2 * C);
                _sk.ConvTaps(cat.View, Table(1, T, H, W, cnt).View, _w.F16(name + ".weight"), _w.F16(name + ".bias"), conv.View, T * hw, C, 2 * C, 3);
                if (T >= 2) StoreCache(name, x, T, hw, C);
                else
                {
                    var pair = Tmp(2L * hw * C);
                    if (c.Rep || c.Frames == 0) pair.View.SubView(0, (long)hw * C).MemSetToZero();
                    else pair.View.SubView(0, (long)hw * C).CopyFrom(c.Buf!.View.SubView((long)(c.Frames - 1) * hw * C, (long)hw * C));
                    pair.View.SubView((long)hw * C, (long)hw * C).CopyFrom(x.View.SubView(0, (long)hw * C));
                    StoreCache(name, pair, 2, hw, C);
                    Drop(pair);
                }
                if (!ReferenceEquals(cat, x)) Drop(cat);
                cur = Tmp(2L * T * hw * C);
                _wk.TimeDouble(conv.View, cur.View, T, hw, C);
                Drop(conv);
                T2 = 2 * T;
            }
        }
        var y = Tmp((long)T2 * 4 * hw * C);
        _sk.ConvTaps(cur.View, Table(2, T2, H, W, 0).View, _w.F16(p + "resample.1.weight"), _w.F16(p + "resample.1.bias"), y.View, T2 * 4 * hw, C, C, 9);
        if (!ReferenceEquals(cur, x)) Drop(cur);
        T = T2;
        return y;
    }

    private Buf UpBlock(int i, Buf x, ref int T, ref int H, ref int W, int cin, int cout, bool temporal, bool up, bool first)
    {
        string p = $"decoder.upsamples.{i}.";
        int T0 = T, H0 = H, W0 = W;
        var cur = x;
        int c = cin;
        for (int j = 0; j < 3; j++)
        {
            var next = ResBlock($"{p}upsamples.{j}.", cur, T, H, W, c, cout);
            if (!ReferenceEquals(cur, x)) Drop(cur);
            cur = next; c = cout;
        }
        if (up)
        {
            var next = Upsample($"{p}upsamples.3.", cur, ref T, H, W, cout, temporal);
            Drop(cur); cur = next; H *= 2; W *= 2;
            int ft = temporal ? 2 : 1;
            var sc = Tmp((long)(T0 * ft - (first ? ft - 1 : 0)) * H * W * cout);
            _wk.DupUp(x.View, sc.View, T0, H0, W0, cin, cout, ft, 2, first ? ft - 1 : 0);
            _dk.GateResidual(cur.View, sc.View, default, -1, (int)cur.Length, cout);
            Drop(sc);
        }
        Flush();
        return cur;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------
    // encoder (image conditioning): one frame, the first chunk of the reference's chunked encode
    // -----------------------------------------------------------------------------------------------------------------------------------

    /// <summary>Resample "downsample2d" / "downsample3d" for a single frame: pad right and bottom, 3x3 convolution of stride 2 (the temporal convolution of downsample3d is skipped on the first chunk).</summary>
    private Buf Downsample(string p, Buf x, int T, int H, int W, int C)
    {
        var y = Tmp((long)T * (H / 2) * (W / 2) * C);
        _sk.ConvTaps(x.View, Table(3, T, H, W, 0).View, _w.F16(p + "resample.1.weight"), _w.F16(p + "resample.1.bias"), y.View, T * (H / 2) * (W / 2), C, C, 9);
        return y;
    }

    private Buf DownBlock(int i, Buf x, ref int T, ref int H, ref int W, int cin, int cout, bool temporal, bool down)
    {
        string p = $"encoder.downsamples.{i}.";
        int T0 = T, H0 = H, W0 = W;
        var cur = x; int c = cin;
        for (int j = 0; j < 2; j++)
        {
            var next = ResBlock($"{p}downsamples.{j}.", cur, T, H, W, c, cout);
            if (!ReferenceEquals(cur, x)) Drop(cur);
            cur = next; c = cout;
        }
        if (down) { var next = Downsample($"{p}downsamples.2.", cur, T, H, W, cout); Drop(cur); cur = next; H /= 2; W /= 2; }
        int ft = temporal ? 2 : 1, fs = down ? 2 : 1;
        var sc = Tmp((long)((T0 + (ft - T0 % ft) % ft) / ft) * (H0 / fs) * (W0 / fs) * cout);
        _wk.AvgDown(x.View, sc.View, T0, H0, W0, cin, cout, ft, fs);
        if (temporal) T = (T0 + (ft - T0 % ft) % ft) / ft;
        _dk.GateResidual(cur.View, sc.View, default, -1, (int)cur.Length, cout);
        Drop(sc);
        Flush();
        return cur;
    }

    /// <summary>
    /// Encode ONE picture (<c>Wan2_2_VAE.encode</c> of a single frame, the image-to-video conditioning): <paramref name="image"/> is [3, 1, H, W] channel-first in [-1, 1] (H and W multiples of 16);
    /// returns the normalised latent mean [48, 1, H/16, W/16], channel-first, ready to replace the first latent frame.
    /// </summary>
    public async Task<float[]> EncodeImageAsync(float[] image, int H, int W)
    {
        if (H % 16 != 0 || W % 16 != 0) throw new ArgumentException("the picture size must be a multiple of 16");
        _caches.Clear(); _cacheEnabled = false;
        try
        {
            using var img = _acc.Allocate1D(image);
            int h2 = H / 2, w2 = W / 2;
            var x0 = Tmp((long)h2 * w2 * 12);
            _wk.PatchifyImage(img.View, x0.View, 1, H, W);
            int T = 1, Hc = h2, Wc = w2;
            var cur = ConvCausal("encoder.conv1", x0, T, Hc, Wc, 12, 160); Drop(x0);
            Tap?.Invoke("enc_conv1", cur.View, T, Hc, Wc, 160);
            var stages = new[] { (160, 160, false, true), (160, 320, true, true), (320, 640, true, true), (640, 640, false, false) };
            for (int s = 0; s < 4; s++)
            {
                var next = DownBlock(s, cur, ref T, ref Hc, ref Wc, stages[s].Item1, stages[s].Item2, stages[s].Item3, stages[s].Item4);
                Drop(cur); cur = next;
                Tap?.Invoke($"enc_down{s}", cur.View, T, Hc, Wc, stages[s].Item2);
                Flush(); await _acc.SynchronizeAsync();                    // a flush is not a wait: bound the queue to one block (browser WebGPU loses the device when seconds of work are queued)
            }
            var m0 = ResBlock("encoder.middle.0.", cur, T, Hc, Wc, 640, 640); Drop(cur);
            var m1 = Attention("encoder.middle.1.", m0, T, Hc, Wc, 640); Drop(m0);
            var m2 = ResBlock("encoder.middle.2.", m1, T, Hc, Wc, 640, 640); Drop(m1);
            Tap?.Invoke("enc_middle", m2.View, T, Hc, Wc, 640);
            var hn = NormSilu(m2, "encoder.head.0.gamma", T * Hc * Wc, 640); Drop(m2);
            var hc = ConvCausal("encoder.head.2", hn, T, Hc, Wc, 640, 96); Drop(hn);
            Tap?.Invoke("enc_head", hc.View, T, Hc, Wc, 96);
            var moments = Pointwise("conv1", hc, T * Hc * Wc, 96, 96); Drop(hc);        // WanVAE_.conv1: 1x1, then the first 48 channels are mu
            var host = await moments.View.CopyToHostAsync();
            Drop(moments); Flush();
            var mean = await _w.F32("latents_mean").CopyToHostAsync(); var std = await _w.F32("latents_std").CopyToHostAsync();
            int hw = Hc * Wc;
            var z = new float[48L * hw];                                                 // (mu - mean) / std, channel-first [48, 1, h, w]
            for (int c = 0; c < 48; c++) for (int p = 0; p < hw; p++) z[(long)c * hw + p] = (host[(long)p * 96 + c] - mean[c]) / std[c];
            return z;
        }
        finally { _cacheEnabled = true; }
    }

    /// <summary>
    /// Decode <paramref name="z"/> ([48, Fz, h, w], channel-first, the normalised latent the sampler produced) to video [3, 4 Fz - 3, 16 h, 16 w], clamped to [-1, 1].
    /// </summary>
    public async Task<float[]> DecodeAsync(float[] z, int Fz, int h, int w)
    {
        const int Z = 48;
        int hw = h * w;
        var mean = await _w.F32("latents_mean").CopyToHostAsync(); var std = await _w.F32("latents_std").CopyToHostAsync();
        // z * std + mean (the reference: z / (1 / std) + mean), channel-first -> channel-last
        var zc = new float[(long)Fz * hw * Z];
        for (int c = 0; c < Z; c++)
            for (int f = 0; f < Fz; f++)
                for (int p = 0; p < hw; p++)
                    zc[((long)f * hw + p) * Z + c] = z[((long)c * Fz + f) * hw + p] * std[c] + mean[c];
        _caches.Clear();
        var zIn = _acc.Allocate1D(zc);
        var zBuf = Pointwise("conv2", zIn, Fz * hw, Z, Z);
        Tap?.Invoke("dec_conv2", zBuf.View, Fz, h, w, Z);
        int H8 = h * 16, W8 = w * 16, totalFrames = 4 * Fz - 3;
        var head = _acc.Allocate1D<float>((long)totalFrames * (h * 8) * (w * 8) * 12);
        int writtenFrames = 0;
        for (int i = 0; i < Fz; i++)
        {
            int T = 1, H = h, W = w;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var xi = Tmp((long)hw * Z);
            xi.View.CopyFrom(zBuf.View.SubView((long)i * hw * Z, (long)hw * Z));
            var y = ConvCausal("decoder.conv1", xi, T, H, W, Z, 1024); Drop(xi);
            Tap?.Invoke("dec_conv1", y.View, T, H, W, 1024);
            var m0 = ResBlock("decoder.middle.0.", y, T, H, W, 1024, 1024); Drop(y);
            var m1 = Attention("decoder.middle.1.", m0, T, H, W, 1024); Drop(m0);
            var m2 = ResBlock("decoder.middle.2.", m1, T, H, W, 1024, 1024); Drop(m1);
            Tap?.Invoke("dec_middle", m2.View, T, H, W, 1024);
            if (Trace) { await _acc.SynchronizeAsync(); Console.WriteLine($"[Wan] vae frame {i} middle {sw.ElapsedMilliseconds} ms"); sw.Restart(); }
            var stages = new[] { (1024, 1024, true, true), (1024, 1024, true, true), (1024, 512, false, true), (512, 256, false, false) };
            var cur = m2;
            for (int s = 0; s < 4; s++)
            {
                var next = UpBlock(s, cur, ref T, ref H, ref W, stages[s].Item1, stages[s].Item2, stages[s].Item3, stages[s].Item4, i == 0);
                Drop(cur); cur = next;
                Tap?.Invoke($"dec_up{s}", cur.View, T, H, W, stages[s].Item2);
                Flush(); await _acc.SynchronizeAsync();
                if (Trace) { Console.WriteLine($"[Wan] vae frame {i} up{s} {sw.ElapsedMilliseconds} ms ({T}x{H}x{W}, {stages[s].Item2} ch)"); sw.Restart(); }
            }
            var hn = NormSilu(cur, "decoder.head.0.gamma", T * H * W, 256); Drop(cur);
            var hc = ConvCausal("decoder.head.2", hn, T, H, W, 256, 12); Drop(hn);
            Tap?.Invoke("dec_head", hc.View, T, H, W, 12);
            head.View.SubView((long)writtenFrames * H * W * 12, (long)T * H * W * 12).CopyFrom(hc.View.SubView(0, (long)T * H * W * 12));
            writtenFrames += T;
            Drop(hc);
            Flush(); await _acc.SynchronizeAsync();
        }
        if (writtenFrames != totalFrames) throw new InvalidOperationException($"decoded {writtenFrames} frames, expected {totalFrames}");
        var video = _acc.Allocate1D<float>(3L * totalFrames * H8 * W8);
        _wk.UnpatchifyVideo(head.View, video.View, totalFrames, h * 8, w * 8);
        var result = await video.View.CopyToHostAsync();
        video.Dispose(); head.Dispose(); zIn.Dispose(); Drop(zBuf); Flush();
        foreach (var c in _caches.Values) c.Buf?.Dispose();
        _caches.Clear();
        return result;
    }

    /// <summary>Log the time of every decoder stage (each is awaited to completion, so the times are the device time): <c>[Wan] vae ...</c> lines on the console.</summary>
    public bool Trace { get; set; }

    public void Dispose()
    {
        foreach (var b in _live) b.Dispose();
        _live.Clear(); _drop.Clear();
        foreach (var t in _tables.Values) t.Dispose();
        _tables.Clear();
        foreach (var c in _caches.Values) c.Buf?.Dispose();
        _caches.Clear();
        _scores?.Dispose();
    }
}
