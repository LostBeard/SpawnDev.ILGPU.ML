using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;
using HV = ArrayView1D<global::ILGPU.Half, Stride1D.Dense>;

/// <summary>
/// TRELLIS-image-large's sparse-structure decoder (trellis2/models/sparse_structure_vae.py SparseStructureDecoder): the dense
/// latent [8, 16, 16, 16] -> occupancy logits [1, 64, 64, 64]. A dense 3x3x3 convolution with zero padding is exactly a submanifold
/// convolution on a FULL voxel grid, so the same gathered-row implicit GEMM runs it (neighbour table of the dense grid).
///   input conv (fp32) -> 2 middle ResBlocks (512) -> 2 ResBlocks (512) -> Upsample(512 -> 128) -> 2 ResBlocks (128)
///   -> Upsample(128 -> 32) -> 2 ResBlocks (32) -> channel LayerNorm, SiLU, conv (fp32) -> 1 channel.
/// ResBlock = LN, SiLU, conv, LN, SiLU, conv, + skip. Upsample = conv to 8x channels + pixel_shuffle_3d. LayerNorm eps 1e-5.
/// Activations are token-major [cells, channels]; the latent arrives channel-first like the reference's tensor.
/// </summary>
public sealed class SparseStructureDecoder : IDisposable
{
    private readonly Accelerator _acc;
    private readonly SparseKernels _sk;
    private readonly DiTKernels _dk;
    private readonly GpuWeights _w;
    private readonly int[] _channels;
    private readonly int _latentCh, _resBlocks, _middleBlocks;
    private readonly Dictionary<string, MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>> _h16 = new();
    private readonly Dictionary<string, MemoryBuffer1D<float, Stride1D.Dense>> _h32 = new();

    public SparseStructureDecoder(Accelerator accelerator, SparseKernels sparse, DiTKernels dit, GpuWeights weights)
    {
        _acc = accelerator; _sk = sparse; _dk = dit; _w = weights;
        var a = weights.Args;
        _channels = a.GetProperty("channels").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        _latentCh = a.GetProperty("latent_channels").GetInt32();
        _resBlocks = a.GetProperty("num_res_blocks").GetInt32();
        _middleBlocks = a.GetProperty("num_res_blocks_middle").GetInt32();
        if (a.TryGetProperty("norm_type", out var nt) && nt.GetString() != "layer") throw new NotSupportedException("only layer norm");
    }

    /// <summary>Permute a torch conv weight [Co][Ci][27] to the gather layout [Co][27][Ci] once, keeping its stored precision.</summary>
    private (HV? h, FV? f) Conv(string name)
    {
        var shape = _w.Shape(name);
        int co = shape[0], ci = shape[1];
        if (_w.IsF16(name))
        {
            if (!_h16.TryGetValue(name, out var b))
            {
                b = _acc.Allocate1D<global::ILGPU.Half>((long)co * 27 * ci);
                _sk.PermuteConvWeight(_w.F16(name), b.View, ci, co);
                _h16[name] = b;
            }
            return (b.View, null);
        }
        if (!_h32.TryGetValue(name, out var f))
        {
            f = _acc.Allocate1D<float>((long)co * 27 * ci);
            _sk.PermuteConvWeight(_w.F32(name), f.View, ci, co);
            _h32[name] = f;
        }
        return (null, f.View);
    }

    private HV? B16(string name) => _w.IsF16(name) ? _w.F16(name) : null;

    private void ConvLayer(FV x, IV nbr, string prefix, string biasName, FV o, int n, int cin, int cout)
    {
        var (h, f) = Conv(prefix);
        if (h.HasValue) _sk.Conv3(x, nbr, h.Value, _w.F16(biasName), o, n, cin, cout);
        else _sk.Conv3F32(x, nbr, f!.Value, _w.F32(biasName), o, n, cin, cout);
    }

    private void Norm(FV x, FV o, string prefix, int n, int c)
    {
        if (_w.IsF32(prefix + ".weight")) _sk.LayerNormF32(x, o, _w.F32(prefix + ".weight"), _w.F32(prefix + ".bias"), n, c, 1e-5f);
        else _sk.LayerNorm(x, o, _w.F16(prefix + ".weight"), _w.F16(prefix + ".bias"), n, c, 1e-5f);
    }

    private void ResBlock(ref MemoryBuffer1D<float, Stride1D.Dense> x, ref MemoryBuffer1D<float, Stride1D.Dense> t2, IV nbr, string p, int n, int c,
        MemoryBuffer1D<float, Stride1D.Dense> t1)
    {
        Norm(x.View, t1.View, p + "norm1", n, c);
        _sk.SiluInPlace(t1.View, n * c);
        ConvLayer(t1.View, nbr, p + "conv1.weight", p + "conv1.bias", t2.View, n, c, c);
        Norm(t2.View, t1.View, p + "norm2", n, c);
        _sk.SiluInPlace(t1.View, n * c);
        ConvLayer(t1.View, nbr, p + "conv2.weight", p + "conv2.bias", t2.View, n, c, c);
        // skip_connection = Identity: x + h
        _dk.GateResidual(t2.View, x.View, default, -1, n * c, c);
        (x, t2) = (t2, x);                  // the result is x; the old x becomes the scratch
    }

    /// <summary>
    /// <paramref name="latent"/> is [latentCh, res^3] channel-first. Returns the occupancy logits [(4 res)^3] (1 channel); caller disposes.
    /// </summary>
    public MemoryBuffer1D<float, Stride1D.Dense> Decode(FV latent, int res)
    {
        int g = res, n = g * g * g;
        MemoryBuffer1D<float, Stride1D.Dense> F(long k) => _acc.Allocate1D<float>(k);
        using var coords = _acc.Allocate1D<int>((long)n * 3);
        _dk.GridCoords(coords.View, g);
        var nbr = _acc.Allocate1D<int>((long)n * 27);
        _sk.BuildNeighbors(coords.View, n, g, nbr.View);

        int c = _channels[0];
        var tokIn = F((long)n * _latentCh);
        _dk.Transpose(latent, tokIn.View, _latentCh, n);                 // [C, N] -> [N, C]
        var x = F((long)n * c);
        ConvLayer(tokIn.View, nbr.View, "input_layer.weight", "input_layer.bias", x.View, n, _latentCh, c);
        tokIn.Dispose();
        var t1 = F((long)n * c); var t2 = F((long)n * c);

        for (int b = 0; b < _middleBlocks; b++) ResBlock(ref x, ref t2, nbr.View, $"middle_block.{b}.", n, c, t1);
        int blk = 0;
        for (int level = 0; level < _channels.Length; level++)
        {
            c = _channels[level];
            for (int r = 0; r < _resBlocks; r++, blk++) ResBlock(ref x, ref t2, nbr.View, $"blocks.{blk}.", n, c, t1);
            if (level < _channels.Length - 1)
            {
                int next = _channels[level + 1];
                var big = F((long)n * next * 8);
                ConvLayer(x.View, nbr.View, $"blocks.{blk}.conv.weight", $"blocks.{blk}.conv.bias", big.View, n, c, next * 8);
                blk++;
                int g2 = g * 2, n2 = n * 8;
                var up = F((long)n2 * next);
                _sk.PixelShuffle3d(big.View, up.View, g, next);
                big.Dispose(); x.Dispose(); t1.Dispose(); t2.Dispose(); nbr.Dispose();
                g = g2; n = n2; c = next; x = up;
                t1 = F((long)n * c); t2 = F((long)n * c);
                using var coords2 = _acc.Allocate1D<int>((long)n * 3);
                _dk.GridCoords(coords2.View, g);
                nbr = _acc.Allocate1D<int>((long)n * 27);
                _sk.BuildNeighbors(coords2.View, n, g, nbr.View);
            }
        }
        // out_layer: ChannelLayerNorm, SiLU, conv (32 -> 1)
        Norm(x.View, t1.View, "out_layer.0", n, c);
        _sk.SiluInPlace(t1.View, n * c);
        var logits = F(n);
        ConvLayer(t1.View, nbr.View, "out_layer.2.weight", "out_layer.2.bias", logits.View, n, c, 1);
        _acc.Synchronize();
        x.Dispose(); t1.Dispose(); t2.Dispose(); nbr.Dispose();
        return logits;
    }

    public void Dispose()
    {
        foreach (var b in _h16.Values) b.Dispose();
        foreach (var b in _h32.Values) b.Dispose();
        _h16.Clear(); _h32.Clear();
    }
}
