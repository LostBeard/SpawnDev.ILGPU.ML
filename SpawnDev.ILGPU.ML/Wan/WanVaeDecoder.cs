using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Wan;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;
using Buf = MemoryBuffer1D<float, Stride1D.Dense>;

/// <summary>
/// The Wan 2.2 VAE decoder (<c>Wan2_2_VAE.decode</c>): the latent [48, Fz, h, w] becomes video [3, 4 Fz - 3, 16 h, 16 w] in [-1, 1].
/// The reference decodes ONE latent frame per chunk through causal 3-D convolutions that keep the last two input frames of every convolution as a cache (so a chunk sees
/// its history without recomputing it); the first chunk decodes to one video frame, every later one to four. This port keeps the same chunking and caches, with
/// channel-last fp32 activations [voxels, C], fp16 weights, and every convolution as a gathered-row implicit GEMM (<see cref="SparseKernels.ConvTaps"/>, tensor cores on CUDA):
/// the neighbour tables of the dense grids (a 3x3x3 causal kernel, a (3,1,1) temporal one, and nearest x2 + 3x3) are built on the device and shared by all convolutions of a shape.
/// Weights: the export of tools/wan-harness/reference/export_vae_f16.py.
/// </summary>
public sealed class WanVaeDecoder : IDisposable
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

    /// <summary>The last (up to) two input frames of a causal convolution, channel-last [frames * H * W, C]; <see cref="Rep"/> = the "Rep" marker of upsample3d's first call.</summary>
    private sealed class Cache { public Buf? Buf; public int Frames; public bool Rep; }

    /// <summary>Test hook: (stage name, activation view, frames, height, width, channels), called per chunk; valid only during the call.</summary>
    public Action<string, FV, int, int, int, int>? Tap { get; set; }

    public WanVaeDecoder(Accelerator accelerator, GpuWeights weights)
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
        var key = (kind, T, H, W, kind == 2 ? 0 : cnt);
        if (_tables.TryGetValue(key, out var t)) return t;
        t = _acc.Allocate1D<int>((long)T * H * W * (kind == 2 ? 4 : 1) * 27);
        switch (kind)
        {
            case 0: _wk.CausalTable(t.View, T, H, W, cnt); break;          // 3x3x3 causal, output [T, H, W]
            case 1: _wk.TimeTable(t.View, T, H, W, cnt); break;            // (3,1,1) causal, output [T, H, W]
            case 2: _wk.UpTable(t.View, T, H, W); break;                   // nearest x2 + 3x3, input [T, H, W], output [T, 2H, 2W]
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
        StoreCache(name, xcat, cnt + T, hw, cin);
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
            var xi = Tmp((long)hw * Z);
            xi.View.CopyFrom(zBuf.View.SubView((long)i * hw * Z, (long)hw * Z));
            var y = ConvCausal("decoder.conv1", xi, T, H, W, Z, 1024); Drop(xi);
            Tap?.Invoke("dec_conv1", y.View, T, H, W, 1024);
            var m0 = ResBlock("decoder.middle.0.", y, T, H, W, 1024, 1024); Drop(y);
            var m1 = Attention("decoder.middle.1.", m0, T, H, W, 1024); Drop(m0);
            var m2 = ResBlock("decoder.middle.2.", m1, T, H, W, 1024, 1024); Drop(m1);
            Tap?.Invoke("dec_middle", m2.View, T, H, W, 1024);
            var stages = new[] { (1024, 1024, true, true), (1024, 1024, true, true), (1024, 512, false, true), (512, 256, false, false) };
            var cur = m2;
            for (int s = 0; s < 4; s++)
            {
                var next = UpBlock(s, cur, ref T, ref H, ref W, stages[s].Item1, stages[s].Item2, stages[s].Item3, stages[s].Item4, i == 0);
                Drop(cur); cur = next;
                Tap?.Invoke($"dec_up{s}", cur.View, T, H, W, stages[s].Item2);
                Flush();
            }
            var hn = NormSilu(cur, "decoder.head.0.gamma", T * H * W, 256); Drop(cur);
            var hc = ConvCausal("decoder.head.2", hn, T, H, W, 256, 12); Drop(hn);
            Tap?.Invoke("dec_head", hc.View, T, H, W, 12);
            head.View.SubView((long)writtenFrames * H * W * 12, (long)T * H * W * 12).CopyFrom(hc.View.SubView(0, (long)T * H * W * 12));
            writtenFrames += T;
            Drop(hc);
            Flush();
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
