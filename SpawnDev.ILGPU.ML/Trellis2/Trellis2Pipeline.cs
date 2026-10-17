using System.Diagnostics;
using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// TRELLIS.2 image-to-3D, the 512 pipeline (pipeline_type "512" of trellis2_image_to_3d.py), on one GPU with ONE stage's weights
/// resident at a time: each stage loads its checkpoint, runs, and frees it, so peak weights are one 2.6 GB flow transformer.
///   conditioning features -> sparse-structure flow (12 steps, CFG) -> structure decoder -> occupancy (32^3)
///   -> shape SLat flow -> shape decoder (mesh) -> texture SLat flow (conditioned on the shape latent) -> texture decoder (PBR voxels)
/// The image features (DINOv3 tokens) arrive as a device tensor [tokens, 1024]; the negative condition is zeros.
/// Output stays on the device: the mesh, and per-voxel PBR attributes [N, 6] (base colour 3, metallic, roughness, alpha) at the
/// mesh's voxel coordinates, for the texture bake.
/// </summary>
public sealed class Trellis2Pipeline : IDisposable
{
    /// <summary>Everything the generation produces, on the device. Dispose frees it.</summary>
    public sealed class Output : IDisposable
    {
        public DualGridMesher.Mesh Mesh = null!;
        /// <summary>Texture-decoder voxels: attributes [Count, 6] = base colour rgb, metallic, roughness, alpha, all in [0,1].</summary>
        public MemoryBuffer1D<float, Stride1D.Dense> Attributes = null!;
        public MemoryBuffer1D<int, Stride1D.Dense> Coords = null!;
        public int VoxelCount, Grid;
        public void Dispose() { Mesh?.Dispose(); Attributes?.Dispose(); Coords?.Dispose(); }
    }

    /// <summary>Per-stage progress: (stage name, seconds spent in it).</summary>
    public Action<string, double>? StageDone { get; set; }

    private readonly Accelerator _acc;
    private readonly string _ckpts, _ssDecoderBase;
    private readonly DiTKernels _dk;
    private readonly SparseKernels _sk;
    private readonly FlowSampler _sampler;
    private readonly DualGridMesher _mesher;
    private readonly FlowSampler.Options _ssOpts, _shapeOpts, _texOpts;
    private readonly float[] _shapeMean, _shapeStd, _texMean, _texStd;

    /// <param name="modelDir">The TRELLIS.2-4B folder (pipeline.json, ckpts/).</param>
    /// <param name="ssDecoderBase">Path without extension of the TRELLIS-image-large ss_dec_conv3d_16l8_fp16 checkpoint.</param>
    public Trellis2Pipeline(Accelerator accelerator, string modelDir, string ssDecoderBase)
    {
        _acc = accelerator; _ckpts = Path.Combine(modelDir, "ckpts"); _ssDecoderBase = ssDecoderBase;
        _dk = new DiTKernels(accelerator); _sk = new SparseKernels(accelerator);
        _sampler = new FlowSampler(accelerator); _mesher = new DualGridMesher(accelerator, _sk);
        using var json = JsonDocument.Parse(File.ReadAllText(Path.Combine(modelDir, "pipeline.json")));
        var a = json.RootElement.GetProperty("args");
        FlowSampler.Options Sampler(string name)
        {
            var p = a.GetProperty(name).GetProperty("params");
            var iv = p.GetProperty("guidance_interval");
            return new FlowSampler.Options(p.GetProperty("steps").GetInt32(), (float)p.GetProperty("guidance_strength").GetDouble(),
                (float)p.GetProperty("guidance_rescale").GetDouble(), iv[0].GetDouble(), iv[1].GetDouble(), p.GetProperty("rescale_t").GetDouble(),
                (float)a.GetProperty(name).GetProperty("args").GetProperty("sigma_min").GetDouble());
        }
        _ssOpts = Sampler("sparse_structure_sampler"); _shapeOpts = Sampler("shape_slat_sampler"); _texOpts = Sampler("tex_slat_sampler");
        float[] Arr(string n, string k) => a.GetProperty(n).GetProperty(k).EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
        _shapeMean = Arr("shape_slat_normalization", "mean"); _shapeStd = Arr("shape_slat_normalization", "std");
        _texMean = Arr("tex_slat_normalization", "mean"); _texStd = Arr("tex_slat_normalization", "std");
    }

    // ---- small kernels ----------------------------------------------------------------------------------
    private Action<Index1D, FV, FV, FV, FV, int>? _affine;
    private static void AffineImpl(Index1D i, FV x, FV mean, FV std, FV o, int c) => o[i] = x[i] * std[i % c] + mean[i % c];
    private Action<Index1D, FV, FV, FV, int, int>? _concat;
    private static void ConcatImpl(Index1D idx, FV a, FV b, FV o, int ca, int cb)
    {
        int cc = ca + cb, n = idx / cc, c = idx % cc;
        o[idx] = c < ca ? a[n * ca + c] : b[n * cb + (c - ca)];
    }
    private Action<Index1D, FV, FV>? _halfShift;
    private static void HalfShiftImpl(Index1D i, FV x, FV o) => o[i] = x[i] * 0.5f + 0.5f;
    private Action<Index1D, FV>? _zero;
    private static void ZeroImpl(Index1D i, FV x) => x[i] = 0f;

    private MemoryBuffer1D<float, Stride1D.Dense> Upload(float[] data) => _acc.Allocate1D(data);

    private void Stage(string name, Stopwatch sw) { _acc.Synchronize(); StageDone?.Invoke(name, sw.Elapsed.TotalSeconds); sw.Restart(); }

    /// <summary>
    /// Generate from image features. <paramref name="cond"/> is [condTokens, 1024] on the device (DINOv3 tokens of the preprocessed
    /// image at 512 px = 1029 tokens). Deterministic for a given <paramref name="seed"/> (own counter-based Gaussian, not torch's).
    /// </summary>
    public Output Run512(FV cond, int condTokens, int seed = 42)
    {
        _affine ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, int>(AffineImpl);
        _concat ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int>(ConcatImpl);
        _halfShift ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(HalfShiftImpl);
        _zero ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV>(ZeroImpl);
        var sw = Stopwatch.StartNew();
        using var neg = _acc.Allocate1D<float>((long)condTokens * 1024);
        _zero((int)neg.Length, neg.View);

        // ---- 1. sparse structure ------------------------------------------------------------------------
        MemoryBuffer1D<int, Stride1D.Dense> coords32;
        int n0;
        {
            using var w = GpuWeights.Load(_acc, Path.Combine(_ckpts, "ss_flow_img_dit_1_3B_64_bf16"));
            using var flow = new SparseStructureFlow(_acc, _dk, w);
            int cells = flow.Resolution * flow.Resolution * flow.Resolution, count = flow.InChannels * cells;
            using var x = _acc.Allocate1D<float>(count);
            _sampler.Randn(x.View, count, (ulong)seed * 7919UL + 1);
            Stage("ss flow load", sw);
            _sampler.Sample((xv, t1000, useCond, v) => flow.Forward(xv, t1000, useCond ? cond : neg.View, condTokens, v), x.View, count, _ssOpts);
            Stage("ss flow sample", sw);
            w.Dispose();
            using var dw = GpuWeights.Load(_acc, _ssDecoderBase);
            using var dec = new SparseStructureDecoder(_acc, _sk, _dk, dw);
            using var logits = dec.Decode(x.View, flow.Resolution);
            coords32 = _sk.Occupancy(logits.View, flow.Resolution * 4, 32, out n0);
            Stage("ss decode", sw);
        }
        if (n0 == 0) throw new InvalidOperationException("the sparse structure is empty");

        // ---- 2. shape latent ---------------------------------------------------------------------------
        MemoryBuffer1D<float, Stride1D.Dense> shapeSlat;       // the sampler output (normalised space): conditions the texture flow
        {
            using var w = GpuWeights.Load(_acc, Path.Combine(_ckpts, "slat_flow_img2shape_dit_1_3B_512_bf16"));
            using var dit = new FlowDiT(_acc, _dk, w);
            dit.SetTokens(coords32.View, n0);
            int ch = dit.OutChannels, count = n0 * ch;
            shapeSlat = _acc.Allocate1D<float>(count);
            _sampler.Randn(shapeSlat.View, count, (ulong)seed * 7919UL + 2);
            Stage("shape flow load", sw);
            _sampler.Sample((xv, t1000, useCond, v) => dit.Forward(xv, t1000, useCond ? cond : neg.View, condTokens, v), shapeSlat.View, count, _shapeOpts);
            Stage("shape flow sample", sw);
        }

        // ---- 3. shape decoder -> mesh -------------------------------------------------------------------
        var result = new Output { Grid = 512 };
        SparseVaeDecoder.Result shapeDec;
        {
            using var sm = Upload(_shapeMean); using var ss = Upload(_shapeStd);
            using var lat = _acc.Allocate1D<float>(shapeSlat.Length);
            _affine((int)shapeSlat.Length, shapeSlat.View, sm.View, ss.View, lat.View, 32);
            using var w = GpuWeights.Load(_acc, Path.Combine(_ckpts, "shape_dec_next_dc_f16c32_fp16"));
            var dec = new SparseVaeDecoder(_acc, _sk, _dk, w);
            Stage("shape decoder load", sw);
            shapeDec = dec.Decode(lat.View, coords32.View, n0, 32, keepSubdivisions: true);
            result.Mesh = _mesher.Extract(shapeDec.Feats.View, shapeDec.Coords.View, shapeDec.Count, shapeDec.Grid);
            Stage("shape decode + mesh", sw);
        }

        // ---- 4. texture latent (conditioned on the shape latent) ------------------------------------------
        MemoryBuffer1D<float, Stride1D.Dense> texSlat;
        {
            using var w = GpuWeights.Load(_acc, Path.Combine(_ckpts, "slat_flow_imgshape2tex_dit_1_3B_512_bf16"));
            using var dit = new FlowDiT(_acc, _dk, w);
            dit.SetTokens(coords32.View, n0);
            int ch = dit.OutChannels, count = n0 * ch;
            texSlat = _acc.Allocate1D<float>(count);
            _sampler.Randn(texSlat.View, count, (ulong)seed * 7919UL + 3);
            using var modelIn = _acc.Allocate1D<float>((long)n0 * dit.InChannels);
            using var vOut = _acc.Allocate1D<float>(count);
            Stage("tex flow load", sw);
            // the model sees [x_t | shape slat] (sparse_cat([x, concat_cond])); only the first 32 channels are integrated
            _sampler.Sample((xv, t1000, useCond, v) =>
            {
                _concat(n0 * dit.InChannels, xv, shapeSlat.View, modelIn.View, ch, 32);
                dit.Forward(modelIn.View, t1000, useCond ? cond : neg.View, condTokens, v);
            }, texSlat.View, count, _texOpts);
            Stage("tex flow sample", sw);
        }
        shapeSlat.Dispose();

        // ---- 5. texture decoder -> PBR voxels -------------------------------------------------------------
        {
            using var tm = Upload(_texMean); using var ts = Upload(_texStd);
            using var lat = _acc.Allocate1D<float>(texSlat.Length);
            _affine((int)texSlat.Length, texSlat.View, tm.View, ts.View, lat.View, 32);
            texSlat.Dispose();
            using var w = GpuWeights.Load(_acc, Path.Combine(_ckpts, "tex_dec_next_dc_f16c32_fp16"));
            var dec = new SparseVaeDecoder(_acc, _sk, _dk, w);
            using var tex = dec.Decode(lat.View, coords32.View, n0, 32, guide: shapeDec.Subdivisions);
            result.Attributes = _acc.Allocate1D<float>(tex.Feats.Length);
            _halfShift((int)tex.Feats.Length, tex.Feats.View, result.Attributes.View);
            result.Coords = _acc.Allocate1D<int>((long)tex.Count * 3);
            result.Coords.View.CopyFrom(tex.Coords.View.SubView(0, (long)tex.Count * 3));
            result.VoxelCount = tex.Count;
            Stage("tex decode", sw);
        }
        shapeDec.Dispose();
        coords32.Dispose();
        return result;
    }

    public void Dispose() { }
}
