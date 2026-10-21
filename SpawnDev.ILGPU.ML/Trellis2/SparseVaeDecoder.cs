using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// The TRELLIS.2 structured-latent decoders (trellis2/models/sc_vaes/sparse_unet_vae.py), shape and texture, on the device.
///   from_latent Linear -> 5 levels of SparseConvNeXt blocks, a channel-to-spatial upsample (C2S) block between levels
///   -> plain LayerNorm -> output Linear.
/// The shape decoder predicts which of the 8 children of every voxel exist (<c>to_subdiv</c>, logit &gt; 0); the texture decoder
/// is given those subdivisions instead (<c>guide</c>), so both walk the same voxel sets. fp16 weights, fp32 activations;
/// the 3x3x3 convolutions run as gathered-row implicit GEMMs (see <see cref="SparseKernels"/>).
/// </summary>
public sealed class SparseVaeDecoder : IDisposable
{
    /// <summary>Test hook: (stage, features, coordinates, voxel count, channels); valid only during the call.</summary>
    public delegate void TapFn(string stage, FV feats, IV coords, int count, int channels);

    /// <summary>The result of <see cref="Decode"/>: features [Count, Channels] and coordinates [Count, 3] on the device.</summary>
    public sealed class Result : IDisposable
    {
        public MemoryBuffer1D<float, Stride1D.Dense> Feats = null!;
        public MemoryBuffer1D<int, Stride1D.Dense> Coords = null!;
        public int Count, Channels, Grid;
        /// <summary>Per-level subdivision logits [n_i, 8] (shape decoder with <c>keepSubdivisions</c>); guides the texture decoder.</summary>
        public List<(MemoryBuffer1D<float, Stride1D.Dense> Logits, int Count)> Subdivisions { get; } = new();
        public void Dispose()
        {
            Feats?.Dispose(); Coords?.Dispose();
            foreach (var s in Subdivisions) s.Logits.Dispose();
            Subdivisions.Clear();
        }
    }

    private readonly Accelerator _acc;
    private readonly SparseKernels _sk;
    private readonly DiTKernels _dk;
    private readonly GpuWeights _w;
    private readonly int[] _channels, _numBlocks;
    private readonly int _latentCh, _outCh;
    private readonly bool _predSubdiv;

    /// <summary>Floats of the upsample conv's output held at a time (the rest of the decoder's buffers are at most one voxel tensor each): 128 Mi floats = 512 MB.</summary>
    public static long ChunkFloats { get; set; } = long.TryParse(Environment.GetEnvironmentVariable("TRELLIS2_C2S_CHUNK_FLOATS"), out var cf) && cf > 0 ? cf : 128L << 20;   // env: tests and A/B runs

    public int LatentChannels => _latentCh;
    public int OutChannels => _outCh;
    public bool PredictsSubdivision => _predSubdiv;
    public TapFn? Tap { get; set; }

    public SparseVaeDecoder(Accelerator accelerator, SparseKernels sparse, DiTKernels dit, GpuWeights weights)
    {
        _acc = accelerator; _sk = sparse; _dk = dit; _w = weights;
        var a = weights.Args;
        _channels = a.GetProperty("model_channels").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        _numBlocks = a.GetProperty("num_blocks").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        _latentCh = a.GetProperty("latent_channels").GetInt32();
        _predSubdiv = !a.TryGetProperty("pred_subdiv", out var ps) || ps.GetBoolean();
        _outCh = weights.Shape("output_layer.weight")[0];
        foreach (var t in a.GetProperty("block_type").EnumerateArray())
            if (t.GetString() != "SparseConvNeXtBlock3d") throw new NotSupportedException($"block_type {t.GetString()}");
        foreach (var t in a.GetProperty("up_block_type").EnumerateArray())
            if (t.GetString() != "SparseResBlockC2S3d") throw new NotSupportedException($"up_block_type {t.GetString()}");
    }

    private MemoryBuffer1D<float, Stride1D.Dense> F(long n) => _acc.Allocate1D<float>(n);
    private MemoryBuffer1D<int, Stride1D.Dense> I(long n) => _acc.Allocate1D<int>(n);

    /// <summary>
    /// Decode <paramref name="n0"/> latent voxels (<paramref name="latent"/> [n0, latentCh], <paramref name="coords0"/> [n0, 3] on a
    /// <paramref name="grid0"/>^3 lattice). With <paramref name="guide"/> (the shape decoder's subdivisions, one entry per level) a
    /// texture decoder follows that structure; otherwise the subdivisions are predicted, and kept when <paramref name="keepSubdivisions"/>.
    /// </summary>
    public Result Decode(FV latent, IV coords0, int n0, int grid0, IReadOnlyList<(MemoryBuffer1D<float, Stride1D.Dense> Logits, int Count)>? guide = null, bool keepSubdivisions = false, bool forceGuide = false)
        => Task.Run(() => DecodeAsync(latent, coords0, n0, grid0, guide, keepSubdivisions, forceGuide)).GetAwaiter().GetResult();

    /// <summary><see cref="Decode"/>, browser-safe (the child counts are awaited readbacks). <paramref name="stopAfterUpsamples"/> &gt; 0 returns after that many channel-to-spatial upsamples with only <see cref="Result.Coords"/> set (a pred_subdiv decoder).</summary>
    public async Task<Result> DecodeAsync(FV latent, IV coords0, int n0, int grid0, IReadOnlyList<(MemoryBuffer1D<float, Stride1D.Dense> Logits, int Count)>? guide = null, bool keepSubdivisions = false, bool forceGuide = false, int stopAfterUpsamples = 0)
    {
        // forceGuide (tests): a pred_subdiv decoder still computes its own logits (tapped as "level{i}_subdiv") but takes the
        // STRUCTURE from the guide, so a low-precision run cannot diverge from the reference voxel sets by one flipped logit.
        if (guide != null && _predSubdiv && !forceGuide) throw new InvalidOperationException("this decoder predicts its own subdivisions; guide is for pred_subdiv=false decoders");
        if (guide == null && !_predSubdiv) throw new InvalidOperationException("this decoder needs the shape decoder's subdivisions (guide)");
        var result = new Result { Grid = grid0 };
        int n = n0, grid = grid0;
        var coords = I(n * 3); coords.View.CopyFrom(coords0.SubView(0, n * 3));
        var nbr = I((long)n * 27);
        _sk.BuildNeighbors(coords.View, n, grid, nbr.View);
        var x = F((long)n * _channels[0]);
        _sk.Linear(latent, _w.F16("from_latent.weight"), _w.F16("from_latent.bias"), x.View, n, _latentCh, _channels[0]);
        Tap?.Invoke("from_latent", x.View, coords.View, n, _channels[0]);

        int levels = _channels.Length;
        for (int i = 0; i < levels; i++)
        {
            int C = _channels[i];
            if (_numBlocks[i] > 0)
            {
                // The MLP's hidden tensor [n, 4C] is the largest of the block (3.7 GB at 1024: more than the 2 GiB a WebGPU buffer can be), so the
                // MLP runs over chunks of rows (a multiple of 64 so every chunk starts on a 256-byte boundary): hid holds one chunk.
                int mlpRows = (int)Math.Min(n, Math.Max(64L, ChunkFloats / (4L * C) / 64 * 64));
                var y = F((long)n * C); var z = F((long)n * C); var hid = F((long)mlpRows * 4 * C);
                for (int j = 0; j < _numBlocks[i]; j++)
                {
                    string p = $"blocks.{i}.{j}.";
                    _sk.Conv3(x.View, nbr.View, _w.F16(p + "conv.weight"), _w.F16(p + "conv.bias"), y.View, n, C, C);
                    _sk.LayerNorm(y.View, z.View, _w.F16(p + "norm.weight"), _w.F16(p + "norm.bias"), n, C, 1e-6f);
                    for (int r0 = 0; r0 < n; r0 += mlpRows)
                    {
                        int rows = Math.Min(mlpRows, n - r0);
                        _sk.Linear(z.View.SubView((long)r0 * C, (long)rows * C), _w.F16(p + "mlp.0.weight"), _w.F16(p + "mlp.0.bias"), hid.View, rows, C, 4 * C, act: 1);
                        _sk.Linear(hid.View, _w.F16(p + "mlp.2.weight"), _w.F16(p + "mlp.2.bias"), y.View.SubView((long)r0 * C, (long)rows * C), rows, 4 * C, C, res: x.View.SubView((long)r0 * C, (long)rows * C));
                    }
                    (x, y) = (y, x);                        // y now holds the old x, reused as scratch
                }
                y.Dispose(); z.Dispose(); hid.Dispose();
                Tap?.Invoke($"level{i}_convnext", x.View, coords.View, n, C);
            }
            if (i == levels - 1) break;

            // ---- channel-to-spatial upsample block (blocks.i.<numBlocks[i]>) ----
            string u = $"blocks.{i}.{_numBlocks[i]}.";
            int outC = _channels[i + 1], cc8 = C / 8;
            MemoryBuffer1D<float, Stride1D.Dense> subdiv; bool ownSubdiv;
            if (_predSubdiv)
            {
                subdiv = F((long)n * 8); ownSubdiv = true;
                _sk.Linear(x.View, _w.F16(u + "to_subdiv.weight"), _w.F16(u + "to_subdiv.bias"), subdiv.View, n, C, 8);
                if (forceGuide && guide != null)
                {
                    Tap?.Invoke($"level{i}_subdiv", subdiv.View, coords.View, n, 8);
                    if (guide.Count <= i || guide[i].Count != n) throw new InvalidOperationException($"guide level {i} mismatch");
                    subdiv.Dispose(); subdiv = guide[i].Logits; ownSubdiv = false;
                }
            }
            else
            {
                if (guide!.Count <= i || guide[i].Count != n) throw new InvalidOperationException($"guide level {i} has {(guide.Count > i ? guide[i].Count : -1)} voxels, decoder has {n}");
                subdiv = guide[i].Logits; ownSubdiv = false;
            }
            var xn = F((long)n * C);
            _sk.LayerNorm(x.View, xn.View, _w.F16(u + "norm1.weight"), _w.F16(u + "norm1.bias"), n, C, 1e-6f);
            _sk.SiluInPlace(xn.View, n * C);
            // The children are counted first, then the conv that gives each parent its 8 children's channels runs over CHUNKS of parents and each chunk is scattered straight into the children's tensor: the [n, 8 * outC]
            // conv output is never held whole (at 1024 it is 3.9 GB, more than the 2 GiB a WebGPU buffer can be, and it is the largest buffer of the decoder).
            using var offsets = I(n + 1);
            int m = await _sk.CountChildrenAsync(subdiv.View, n, offsets.View);
            var newCoords = I((long)m * 3); using var parent = I(m); using var slot = I(m);
            _sk.WriteChildren(subdiv.View, offsets.View, coords.View, n, newCoords.View, parent.View, slot.View);
            var h2 = F((long)m * outC);
            {
                long chunkRows = Math.Max(1, ChunkFloats / ((long)outC * 8));
                var offHost = await offsets.View.CopyToHostAsync();
                using var hc = F(Math.Min(chunkRows, n) * outC * 8);
                for (int r0 = 0; r0 < n; r0 += (int)chunkRows)
                {
                    int rows = (int)Math.Min(chunkRows, n - r0);
                    _sk.Conv3(xn.View, nbr.View.SubView((long)r0 * 27, (long)rows * 27), _w.F16(u + "conv1.weight"), _w.F16(u + "conv1.bias"), hc.View, rows, C, outC * 8);
                    int c0 = offHost[r0], c1 = offHost[r0 + rows];
                    if (c1 > c0) _sk.Channel2SpatialChunk(hc.View, parent.View.SubView(c0, c1 - c0), slot.View.SubView(c0, c1 - c0), h2.View.SubView((long)c0 * outC, (long)(c1 - c0) * outC), c1 - c0, outC, r0);
                }
                await _acc.SynchronizeAsync();                       // the chunk buffer and xn are reused / freed below: a real drain on the browser
            }
            xn.Dispose();
            using var xs = F((long)m * cc8);
            _sk.Channel2Spatial(x.View, parent.View, slot.View, xs.View, m, cc8);
            grid *= 2;
            var nbr2 = I((long)m * 27);
            _sk.BuildNeighbors(newCoords.View, m, grid, nbr2.View);
            var h3 = F((long)m * outC);
            _sk.LayerNorm(h2.View, h3.View, null, null, m, outC, 1e-6f);
            h2.Dispose();
            _sk.SiluInPlace(h3.View, m * outC);
            var h4 = F((long)m * outC);
            _sk.Conv3(h3.View, nbr2.View, _w.F16(u + "conv2.weight"), _w.F16(u + "conv2.bias"), h4.View, m, outC, outC);
            h3.Dispose();
            _sk.AddRepeat(h4.View, xs.View, m, outC, outC / cc8);

            if (!_predSubdiv || !ownSubdiv) { /* guided: the logits belong to the caller */ }
            else if (keepSubdivisions) result.Subdivisions.Add((subdiv, n));
            else subdiv.Dispose();
            x.Dispose(); coords.Dispose(); nbr.Dispose();
            x = h4; coords = newCoords; nbr = nbr2; n = m;
            Tap?.Invoke($"level{i}_up", x.View, coords.View, n, outC);
            if (stopAfterUpsamples > 0 && i + 1 == stopAfterUpsamples)
            {   // the reference decoder's upsample(x, times): only the coordinates after that many subdivisions, no output layer (the cascade pipelines' high-resolution token set)
                x.Dispose(); nbr.Dispose(); _acc.Fence();
                result.Coords = coords; result.Count = n; result.Grid = grid;
                return result;
            }
        }
        nbr.Dispose();

        // ---- output: F.layer_norm(h.feats) (eps 1e-5, no affine), output_layer ----
        int last = _channels[^1];
        var ln = F((long)n * last);
        _sk.LayerNorm(x.View, ln.View, null, null, n, last, 1e-5f);
        x.Dispose();
        var outF = F((long)n * _outCh);
        _sk.Linear(ln.View, _w.F16("output_layer.weight"), _w.F16("output_layer.bias"), outF.View, n, last, _outCh);
        ln.Dispose();
        _acc.Fence();
        result.Feats = outF; result.Coords = coords; result.Count = n; result.Channels = _outCh; result.Grid = grid;
        Tap?.Invoke("out", outF.View, coords.View, n, _outCh);
        return result;
    }

    public void Dispose() { }
}
