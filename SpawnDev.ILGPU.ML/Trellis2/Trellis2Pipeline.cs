using System.Diagnostics;
using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

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
        /// <summary>The latents this asset was generated from, on the host (a few MB): what <see cref="RetextureRegionAsync"/> edits. Null for an Output not made by a generation.</summary>
        public EditState? Edit;
        public void Dispose() { Mesh?.Dispose(); Attributes?.Dispose(); Coords?.Dispose(); }
    }

    /// <summary>The structured latents of a generated asset (sampler output, normalised space): the occupied tokens, the shape latent that conditions the texture flow, and the texture latent.</summary>
    public sealed class EditState
    {
        /// <summary>512 or 1024.</summary>
        public int Res;
        /// <summary>Tokens per axis (Res / 16).</summary>
        public int SlatGrid;
        public int Count;
        /// <summary>[Count, 3] token coordinates on the SlatGrid^3 lattice.</summary>
        public int[] Coords = Array.Empty<int>();
        /// <summary>[Count, 32]</summary>
        public float[] ShapeSlat = Array.Empty<float>();
        /// <summary>[Count, channels]</summary>
        public float[] TexSlat = Array.Empty<float>();
        /// <summary>The sparse-structure latent the occupied tokens were decoded from, [channels, SsRes^3] channel first (sampler output); what a SHAPE edit re-samples.</summary>
        public float[] SsLatent = Array.Empty<float>();
        /// <summary>Cells per axis of <see cref="SsLatent"/> (16).</summary>
        public int SsRes;
        /// <summary>World position (the volume is [-0.5, 0.5]^3) of the centre of token i.</summary>
        public (float X, float Y, float Z) TokenCentre(int i) => ((Coords[i * 3] + 0.5f) / SlatGrid - 0.5f, (Coords[i * 3 + 1] + 0.5f) / SlatGrid - 0.5f, (Coords[i * 3 + 2] + 0.5f) / SlatGrid - 0.5f);
    }

    /// <summary>Per-stage progress: (stage name, seconds spent in it).</summary>
    public Action<string, double>? StageDone { get; set; }

    private readonly Accelerator _acc;
    private readonly Trellis2Weights _weights;
    private readonly DiTKernels _dk;
    private readonly SparseKernels _sk;
    private readonly FlowSampler _sampler;
    private readonly DualGridMesher _mesher;
    private readonly FlowSampler.Options _ssOpts, _shapeOpts, _texOpts;
    private readonly float[] _shapeMean, _shapeStd, _texMean, _texStd;

    /// <param name="modelDir">The TRELLIS.2-4B folder (pipeline.json, ckpts/).</param>
    /// <param name="ssDecoderBase">Path without extension of the TRELLIS-image-large ss_dec_conv3d_16l8_fp16 checkpoint.</param>
    public Trellis2Pipeline(Accelerator accelerator, string modelDir, string ssDecoderBase)
        : this(accelerator, Trellis2Weights.FromFiles(new Trellis2Paths { ModelDir = modelDir, SparseStructureDecoder = ssDecoderBase }),
               File.ReadAllText(Path.Combine(modelDir, "pipeline.json"))) { }

    /// <summary>Create from any weight source (files, or the hub / OPFS in the browser): reads pipeline.json first.</summary>
    public static async Task<Trellis2Pipeline> CreateAsync(Accelerator accelerator, Trellis2Weights weights, CancellationToken ct = default)
        => new Trellis2Pipeline(accelerator, weights, await weights.ReadPipelineJsonAsync(ct));

    private Trellis2Pipeline(Accelerator accelerator, Trellis2Weights weights, string pipelineJson)
    {
        _acc = accelerator; _weights = weights;
        _dk = new DiTKernels(accelerator); _sk = new SparseKernels(accelerator);
        _sampler = new FlowSampler(accelerator); _mesher = new DualGridMesher(accelerator, _sk);
        using var json = JsonDocument.Parse(pipelineJson);
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

    /// <summary>Debug: mean, std and max |x| of the first <paramref name="n"/> values.</summary>
    private string Moments(FV v, int n)
    {
        _acc.Synchronize();
        var a = v.SubView(0, n).GetAsArray1D(); double s = 0, s2 = 0, mx = 0;
        foreach (var x in a) { s += x; s2 += (double)x * x; mx = Math.Max(mx, Math.Abs(x)); }
        double mean = s / a.Length; return $"mean {mean:F3} std {Math.Sqrt(s2 / a.Length - mean * mean):F3} max {mx:F2}";
    }

    /// <summary>Read ahead the first flow checkpoint (and optionally the feature extractor) while the caller prepares the image.</summary>
    public void PrefetchFirstStage(bool includeDino = false) => _weights.Prefetch(new[] { "ss_flow_img_dit_1_3B_64_bf16" }, includeDino);

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

    // a stage ends with a real (awaited) drain on the browser too: a flush alone lets the GPU time of one stage surface in the next one that
    // waits (MEASURED: the browser's "ss decode" read 55.8 s with the sparse-structure flow's work in it)
    private async Task StageAsync(string name, Stopwatch sw)
    {
        if (_acc.IsBrowser()) await _acc.WaitAsync(); else _acc.Fence();
        StageDone?.Invoke(name, sw.Elapsed.TotalSeconds); sw.Restart();
    }

    /// <summary>
    /// Generate from image features, 512 pipeline. <paramref name="cond"/> is [condTokens, 1024] on the device (DINOv3 tokens of the preprocessed
    /// image at 512 px = 1029 tokens). Deterministic for a given <paramref name="seed"/> (own counter-based Gaussian, not torch's).
    /// </summary>
    public Output Run512(FV cond, int condTokens, int seed = 42) => Task.Run(() => Run512Async(cond, condTokens, seed)).GetAwaiter().GetResult();
    /// <summary><see cref="Run512"/>, browser-safe.</summary>
    public Task<Output> Run512Async(FV cond, int condTokens, int seed = 42, CancellationToken ct = default) => RunCoreAsync(512, cond, condTokens, cond, condTokens, seed, ct);

    /// <summary>
    /// The 1024 pipeline of TRELLIS.2: the sparse structure at 64^3 from the 512 px features, then the shape and texture flows of the 1024 checkpoints
    /// on the features of the image at 1024 px (<paramref name="cond1024"/>, 4096 + 5 tokens), decoders up to 1024^3 voxels. About 4x the voxels and
    /// tokens of <see cref="Run512"/> (the whole run is minutes, not seconds) and a mesh of several million triangles.
    /// </summary>
    public Output Run1024(FV cond512, int tokens512, FV cond1024, int tokens1024, int seed = 42) => Task.Run(() => Run1024Async(cond512, tokens512, cond1024, tokens1024, seed)).GetAwaiter().GetResult();
    /// <summary><see cref="Run1024"/>, browser-safe.</summary>
    public Task<Output> Run1024Async(FV cond512, int tokens512, FV cond1024, int tokens1024, int seed = 42, CancellationToken ct = default)
        => RunCoreAsync(1024, cond512, tokens512, cond1024, tokens1024, seed, ct);

    private async Task<Output> RunCoreAsync(int res, FV ssCond, int ssTokens, FV cond, int condTokens, int seed, CancellationToken ct)
    {
        int slatGrid = res / 16;                            // the sparse tensors live on a res/16 grid (32^3 for 512, 64^3 for 1024)
        string suffix = res == 512 ? "512" : "1024";
        _affine ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, int>(AffineImpl);
        _concat ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int>(ConcatImpl);
        _halfShift ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(HalfShiftImpl);
        _zero ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV>(ZeroImpl);
        var sw = Stopwatch.StartNew();
        using var neg = _acc.Allocate1D<float>((long)condTokens * 1024);
        _zero((int)neg.Length, neg.View);
        using var ssNeg = _acc.Allocate1D<float>((long)ssTokens * 1024);
        _zero((int)ssNeg.Length, ssNeg.View);

        // ---- 1. sparse structure ------------------------------------------------------------------------
        var (coords32, n0, ssLatent, ssRes) = await SampleStructureAsync(ssCond, ssNeg.View, ssTokens, slatGrid, seed, suffix, sw, ct);

        // ---- 2. shape latent ---------------------------------------------------------------------------
        MemoryBuffer1D<float, Stride1D.Dense> shapeSlat;       // the sampler output (normalised space): conditions the texture flow
        {
            using var w = await _weights.LoadCheckpointAsync(_acc, "slat_flow_img2shape_dit_1_3B_" + suffix + "_bf16", ct);
            await StageAsync("shape flow weights", sw);
            using var dit = new FlowDiT(_acc, _dk, w);
            dit.SetTokens(coords32.View, n0);
            int ch = dit.OutChannels, count = n0 * ch;
            shapeSlat = _acc.Allocate1D<float>(count);
            _sampler.Randn(shapeSlat.View, count, (ulong)seed * 7919UL + 2);
            await StageAsync("shape flow load", sw);
            _weights.Prefetch(new[] { "slat_flow_imgshape2tex_dit_1_3B_" + suffix + "_bf16", "tex_dec_next_dc_f16c32_fp16" });
            await _sampler.SampleAsync((xv, t1000, useCond, v) => dit.Forward(xv, t1000, useCond ? cond : neg.View, condTokens, v), shapeSlat.View, count, _shapeOpts);
            await StageAsync($"shape flow sample (host {sw.Elapsed.TotalSeconds:F1} s)", sw);
            if (Environment.GetEnvironmentVariable("TRELLIS2_DEBUG_SLAT") == "1") Console.WriteLine($"  [slat] shape: {Moments(shapeSlat.View, count)}");
        }

        // ---- 3. shape decoder -> mesh -------------------------------------------------------------------
        var result = new Output { Grid = res };
        SparseVaeDecoder.Result shapeDec;
        {
            using var sm = Upload(_shapeMean); using var ss = Upload(_shapeStd);
            using var lat = _acc.Allocate1D<float>(shapeSlat.Length);
            _affine((int)shapeSlat.Length, shapeSlat.View, sm.View, ss.View, lat.View, 32);
            using var w = await _weights.LoadCheckpointAsync(_acc, "shape_dec_next_dc_f16c32_fp16", ct);
            var dec = new SparseVaeDecoder(_acc, _sk, _dk, w);
            await StageAsync("shape decoder load", sw);
            shapeDec = await dec.DecodeAsync(lat.View, coords32.View, n0, slatGrid, keepSubdivisions: true);
            result.Mesh = await _mesher.ExtractAsync(shapeDec.Feats.View, shapeDec.Coords.View, shapeDec.Count, shapeDec.Grid);
            await StageAsync($"shape decode + mesh ({shapeDec.Count} voxels, {result.Mesh.FaceCount} triangles)", sw);
        }

        // ---- 4. texture latent (conditioned on the shape latent) ------------------------------------------
        MemoryBuffer1D<float, Stride1D.Dense> texSlat;
        {
            using var w = await _weights.LoadCheckpointAsync(_acc, "slat_flow_imgshape2tex_dit_1_3B_" + suffix + "_bf16", ct);
            using var dit = new FlowDiT(_acc, _dk, w);
            dit.SetTokens(coords32.View, n0);
            int ch = dit.OutChannels, count = n0 * ch;
            texSlat = _acc.Allocate1D<float>(count);
            _sampler.Randn(texSlat.View, count, (ulong)seed * 7919UL + 3);
            using var modelIn = _acc.Allocate1D<float>((long)n0 * dit.InChannels);
            using var vOut = _acc.Allocate1D<float>(count);
            await StageAsync("tex flow load", sw);
            string dumpTex = Environment.GetEnvironmentVariable("TRELLIS2_DUMP_TEX") ?? "";
            if (dumpTex.Length > 0)
            {   // debug: everything the texture stage consumes, to rerun it in the reference (reference/golden_tex_stage.py)
                Directory.CreateDirectory(dumpTex); _acc.Synchronize();
                void DumpF(string name, FV v, long n) { var a = v.SubView(0, n).GetAsArray1D(); var b = new byte[a.Length * 4]; Buffer.BlockCopy(a, 0, b, 0, b.Length); File.WriteAllBytes(Path.Combine(dumpTex, name), b); }
                var cc = coords32.View.SubView(0, (long)n0 * 3).GetAsArray1D(); var cb = new byte[cc.Length * 4]; Buffer.BlockCopy(cc, 0, cb, 0, cb.Length);
                File.WriteAllBytes(Path.Combine(dumpTex, "coords.i32"), cb);
                DumpF("noise.f32", texSlat.View, count); DumpF("shape_slat.f32", shapeSlat.View, (long)n0 * 32); DumpF("cond.f32", cond, (long)condTokens * 1024);
            }
            // the model sees [x_t | shape slat] (sparse_cat([x, concat_cond])); only the first 32 channels are integrated
            await _sampler.SampleAsync((xv, t1000, useCond, v) =>
            {
                _concat(n0 * dit.InChannels, xv, shapeSlat.View, modelIn.View, ch, 32);
                dit.Forward(modelIn.View, t1000, useCond ? cond : neg.View, condTokens, v);
            }, texSlat.View, count, _texOpts);
            await StageAsync($"tex flow sample (host {sw.Elapsed.TotalSeconds:F1} s)", sw);
            if (Environment.GetEnvironmentVariable("TRELLIS2_DEBUG_SLAT") == "1") Console.WriteLine($"  [slat] tex: {Moments(texSlat.View, count)}");
            { var dt = Environment.GetEnvironmentVariable("TRELLIS2_DUMP_TEX") ?? ""; if (dt.Length > 0) { _acc.Synchronize(); var a = texSlat.View.SubView(0, count).GetAsArray1D(); var b = new byte[a.Length * 4]; Buffer.BlockCopy(a, 0, b, 0, b.Length); File.WriteAllBytes(Path.Combine(dt, "tex_slat.f32"), b); } }
        }
        var edit = new EditState { Res = res, SlatGrid = slatGrid, Count = n0, SsLatent = ssLatent, SsRes = ssRes };
        edit.Coords = await coords32.View.SubView(0, (long)n0 * 3).CopyToHostAsync();
        edit.ShapeSlat = await shapeSlat.View.SubView(0, (long)n0 * 32).CopyToHostAsync();
        result.Edit = edit;
        shapeSlat.Dispose();

        // ---- 5. texture decoder -> PBR voxels -------------------------------------------------------------
        {
            using var tm = Upload(_texMean); using var ts = Upload(_texStd);
            using var lat = _acc.Allocate1D<float>(texSlat.Length);
            _affine((int)texSlat.Length, texSlat.View, tm.View, ts.View, lat.View, 32);
            edit.TexSlat = await texSlat.View.SubView(0, texSlat.Length).CopyToHostAsync();
            texSlat.Dispose();
            using var w = await _weights.LoadCheckpointAsync(_acc, "tex_dec_next_dc_f16c32_fp16", ct);
            var dec = new SparseVaeDecoder(_acc, _sk, _dk, w);
            using var tex = await dec.DecodeAsync(lat.View, coords32.View, n0, slatGrid, guide: shapeDec.Subdivisions);
            result.Attributes = _acc.Allocate1D<float>(tex.Feats.Length);
            _halfShift((int)tex.Feats.Length, tex.Feats.View, result.Attributes.View);
            result.Coords = _acc.Allocate1D<int>((long)tex.Count * 3);
            result.Coords.View.CopyFrom(tex.Coords.View.SubView(0, (long)tex.Count * 3));
            result.VoxelCount = tex.Count;
            await StageAsync("tex decode", sw);
        }
        shapeDec.Dispose();
        coords32.Dispose();
        return result;
    }

    /// <summary>Stage 1 of every pipeline: the sparse structure flow (64^3 latent) and its decoder, then the occupied cells on the <paramref name="slatGrid"/>^3 token grid. <paramref name="suffix"/> names the shape/texture checkpoints prefetched next.</summary>
    private async Task<(MemoryBuffer1D<int, Stride1D.Dense> Coords, int Count, float[] SsLatent, int SsRes)> SampleStructureAsync(FV ssCond, FV ssNegView, int ssTokens, int slatGrid, int seed, string suffix, Stopwatch sw, CancellationToken ct)
    {
        MemoryBuffer1D<int, Stride1D.Dense> coords32;
        int n0;
        float[] ssLatent = Array.Empty<float>(); int ssRes = 0;
        {
            using var w = await _weights.LoadCheckpointAsync(_acc, "ss_flow_img_dit_1_3B_64_bf16", ct);
            await StageAsync("ss flow weights", sw);
            using var flow = new SparseStructureFlow(_acc, _dk, w);
            int cells = flow.Resolution * flow.Resolution * flow.Resolution, count = flow.InChannels * cells;
            using var x = _acc.Allocate1D<float>(count);
            _sampler.Randn(x.View, count, (ulong)seed * 7919UL + 1);
            string dumpDir = Environment.GetEnvironmentVariable("TRELLIS2_DUMP_SS") ?? "";
            if (dumpDir.Length > 0)
            {   // debug: noise, condition and the sampled latent, to rerun the same stage in the reference
                Directory.CreateDirectory(dumpDir);
                void Dump(string name, FV v, int n) { var a = v.SubView(0, n).GetAsArray1D(); var b = new byte[a.Length * 4]; Buffer.BlockCopy(a, 0, b, 0, b.Length); File.WriteAllBytes(Path.Combine(dumpDir, name), b); }
                Dump("noise.f32", x.View, count); Dump("cond.f32", ssCond, ssTokens * 1024);
            }
            await StageAsync("ss flow load", sw);
            _weights.Prefetch(new[] { "slat_flow_img2shape_dit_1_3B_" + suffix + "_bf16", "shape_dec_next_dc_f16c32_fp16" }, includeSsDecoder: true);
            await _sampler.SampleAsync((xv, t1000, useCond, v) => flow.Forward(xv, t1000, useCond ? ssCond : ssNegView, ssTokens, v), x.View, count, _ssOpts);
            await StageAsync($"ss flow sample (host {sw.Elapsed.TotalSeconds:F1} s)", sw);
            if (dumpDir.Length > 0) { var a = x.View.SubView(0, count).GetAsArray1D(); var b = new byte[a.Length * 4]; Buffer.BlockCopy(a, 0, b, 0, b.Length); File.WriteAllBytes(Path.Combine(dumpDir, "zs.f32"), b); }
            ssLatent = await x.View.SubView(0, count).CopyToHostAsync(); ssRes = flow.Resolution;
            w.Dispose();
            using var dw = await _weights.LoadSsDecoderAsync(_acc, ct);
            using var dec = new SparseStructureDecoder(_acc, _sk, _dk, dw);
            using var logits = dec.Decode(x.View, flow.Resolution);
            (coords32, n0) = await _sk.OccupancyAsync(logits.View, flow.Resolution * 4, slatGrid);
            await StageAsync($"ss decode ({n0} cells of {slatGrid}^3)", sw);
        }
        if (n0 == 0) throw new InvalidOperationException("the sparse structure is empty");
        if (Environment.GetEnvironmentVariable("TRELLIS2_DEBUG_SS") == "1")
        {
            // debug: the occupied 32^3 cells as ASCII layers along z (the sparse structure BEFORE the shape stage)
            var c = coords32.View.SubView(0, (long)n0 * 3).GetAsArray1D();
            var grid = new bool[slatGrid, slatGrid, slatGrid];
            for (int i = 0; i < n0; i++) grid[c[i * 3], c[i * 3 + 1], c[i * 3 + 2]] = true;
            for (int z = 0; z < slatGrid; z++)
            {
                int cnt = 0; for (int x = 0; x < slatGrid; x++) for (int y = 0; y < slatGrid; y++) if (grid[x, y, z]) cnt++;
                Console.WriteLine($"z={z,2} cells={cnt}");
                if (cnt > 0 && z % 3 == 0)
                    for (int y = 0; y < slatGrid; y++) { var sb = new System.Text.StringBuilder(); for (int x = 0; x < slatGrid; x++) sb.Append(grid[x, y, z] ? '#' : '.'); Console.WriteLine("   " + sb); }
            }
        }
        return (coords32, n0, ssLatent, ssRes);
    }

    /// <summary>
    /// The reference's token cap (<c>max_num_tokens</c> 49152), lowered where one buffer cannot hold the finest decoder level: the decoder's last level is a
    /// [voxels, 64] tensor with about 360 voxels per token (measured 361 at 1024, 345 at 1536), and a WebGPU buffer is at most <c>maxBufferSize</c> (2 GiB on this
    /// adapter: 20,971 tokens at 400 voxels per token with a margin). The resolution then steps down 128 at a time, as the reference does for its own cap.
    /// </summary>
    private int DefaultMaxTokens()
        => _acc.AcceleratorType == global::ILGPU.Runtime.AcceleratorType.WebGPU ? (int)Math.Min(49152L, _acc.MemorySize / (400L * 64 * 4)) : 49152;

    /// <summary>
    /// The CASCADE pipelines of TRELLIS.2 (<c>1024_cascade</c>, <c>1536_cascade</c>; the reference's default is <c>1024_cascade</c>): the sparse structure at 32^3 and the shape latent from the 512 checkpoint on the 512 px
    /// features, then the shape decoder's own upsampling turns the coarse tokens into the high-resolution token set (their cells quantised to the <paramref name="targetRes"/> / 16 grid, duplicates removed, and the
    /// resolution lowered by 128 until fewer than <paramref name="maxTokens"/> tokens remain, never below 1024), and the 1024 checkpoints' shape and texture flows run on those tokens with the 1024 px features
    /// (<paramref name="cond1024"/>); the decoders then go up to <c>Output.Grid</c> = the resolution that was reached (1536 for 1536_cascade unless the token cap lowered it). Ported from
    /// <c>Trellis2ImageTo3DPipeline.sample_shape_slat_cascade</c>. The result carries no <see cref="Output.Edit"/> state (region editing is for the 512 and 1024 pipelines).
    /// </summary>
    public async Task<Output> RunCascadeAsync(int targetRes, FV cond512, int tokens512, FV cond1024, int tokens1024, int seed = 42, int maxTokens = 0, CancellationToken ct = default)
    {
        if (maxTokens <= 0) maxTokens = DefaultMaxTokens();
        if (targetRes != 1024 && targetRes != 1536) throw new ArgumentException("A cascade targets 1024 or 1536");
        _affine ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, int>(AffineImpl);
        _concat ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int>(ConcatImpl);
        _halfShift ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(HalfShiftImpl);
        _zero ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV>(ZeroImpl);
        // TRELLIS2_CASCADE_CACHE=<dir>: the sampled latents (the minutes-long part) are kept there and reused, so the decoders, mesh and export can be worked on without re-sampling (development aid)
        string cacheDir = Environment.GetEnvironmentVariable("TRELLIS2_CASCADE_CACHE") ?? "";
        string stem = cacheDir.Length > 0 ? Path.Combine(cacheDir, $"cascade_{targetRes}_s{seed}") : "";
        CascadeLatents lat;
        if (stem.Length > 0 && File.Exists(stem + ".json")) lat = LoadCascade(stem);
        else
        {
            lat = await SampleCascadeAsync(targetRes, cond512, tokens512, cond1024, tokens1024, seed, maxTokens, ct);
            if (stem.Length > 0) { Directory.CreateDirectory(cacheDir); SaveCascade(stem, lat); }
        }
        return await DecodeCascadeAsync(lat, ct);
    }

    /// <summary>What the cascade's samplers produce: the token cells on the (HrRes / 16)^3 grid and the shape and texture latents (normalised space) on them.</summary>
    private sealed record CascadeLatents(int HrRes, int Tokens, int[] Coords, float[] ShapeSlat, float[] TexSlat);

    private static void SaveCascade(string stem, CascadeLatents l)
    {
        static void Raw(string p, Array a) { var b = new byte[Buffer.ByteLength(a)]; Buffer.BlockCopy(a, 0, b, 0, b.Length); File.WriteAllBytes(p, b); }
        Raw(stem + ".coords.i32", l.Coords); Raw(stem + ".shape.f32", l.ShapeSlat); Raw(stem + ".tex.f32", l.TexSlat);
        File.WriteAllText(stem + ".json", $"{{\"hrRes\":{l.HrRes},\"tokens\":{l.Tokens}}}");
    }

    private static CascadeLatents LoadCascade(string stem)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(stem + ".json"));
        int hrRes = doc.RootElement.GetProperty("hrRes").GetInt32(), n = doc.RootElement.GetProperty("tokens").GetInt32();
        static T[] Read<T>(string p) where T : struct { var b = File.ReadAllBytes(p); var a = new T[b.Length / System.Runtime.InteropServices.Marshal.SizeOf<T>()]; Buffer.BlockCopy(b, 0, a, 0, b.Length); return a; }
        return new CascadeLatents(hrRes, n, Read<int>(stem + ".coords.i32"), Read<float>(stem + ".shape.f32"), Read<float>(stem + ".tex.f32"));
    }

    private async Task<CascadeLatents> SampleCascadeAsync(int targetRes, FV cond512, int tokens512, FV cond1024, int tokens1024, int seed, int maxTokens, CancellationToken ct)
    {
        var sw = Stopwatch.StartNew();
        using var neg512 = _acc.Allocate1D<float>((long)tokens512 * 1024); _zero!((int)neg512.Length, neg512.View);
        using var neg1024 = _acc.Allocate1D<float>((long)tokens1024 * 1024); _zero((int)neg1024.Length, neg1024.View);

        // ---- 1. sparse structure at 32^3 (the same stage as the 512 pipeline) ----------------------------
        var (coords32, n0, _, _) = await SampleStructureAsync(cond512, neg512.View, tokens512, 32, seed, "512", sw, ct);

        // ---- 2. low-resolution shape latent: the 512 checkpoint on the 512 px features ----------------------
        MemoryBuffer1D<float, Stride1D.Dense> lrSlat;
        {
            using var w = await _weights.LoadCheckpointAsync(_acc, "slat_flow_img2shape_dit_1_3B_512_bf16", ct);
            await StageAsync("shape flow (512) weights", sw);
            using var dit = new FlowDiT(_acc, _dk, w);
            dit.SetTokens(coords32.View, n0);
            int count = n0 * dit.OutChannels;
            lrSlat = _acc.Allocate1D<float>(count);
            _sampler.Randn(lrSlat.View, count, (ulong)seed * 7919UL + 2);
            _weights.Prefetch(new[] { "shape_dec_next_dc_f16c32_fp16", "slat_flow_img2shape_dit_1_3B_1024_bf16" });
            await _sampler.SampleAsync((xv, t1000, useCond, v) => dit.Forward(xv, t1000, useCond ? cond512 : neg512.View, tokens512, v), lrSlat.View, count, _shapeOpts);
            await StageAsync($"shape flow (512) sample (host {sw.Elapsed.TotalSeconds:F1} s)", sw);
        }

        // ---- 3. the decoder's own upsampling: the high-resolution token set ---------------------------------
        int[] hrHost; int hrCount;
        {
            using var sm = Upload(_shapeMean); using var ss = Upload(_shapeStd);
            using var lat = _acc.Allocate1D<float>(lrSlat.Length);
            _affine!((int)lrSlat.Length, lrSlat.View, sm.View, ss.View, lat.View, 32);
            using var w = await _weights.LoadCheckpointAsync(_acc, "shape_dec_next_dc_f16c32_fp16", ct);
            var dec = new SparseVaeDecoder(_acc, _sk, _dk, w);
            using var up = await dec.DecodeAsync(lat.View, coords32.View, n0, 32, stopAfterUpsamples: 4);     // 4 subdivisions: the 512^3 voxels of the coarse shape
            hrCount = up.Count;
            hrHost = await up.Coords.View.SubView(0, (long)hrCount * 3).CopyToHostAsync();
            await StageAsync($"upsample ({hrCount} voxels at {up.Grid}^3)", sw);
        }
        lrSlat.Dispose(); coords32.Dispose();
        int hrRes = targetRes; int[] tokens; int nTok;
        while (true)
        {
            (tokens, nTok) = QuantizeUnique(hrHost, hrCount, 512, hrRes);
            if (nTok < maxTokens || hrRes == 1024) break;
            hrRes -= 128;
        }
        if (hrRes != targetRes) Console.WriteLine($"Due to the limited number of tokens, the resolution is reduced to {hrRes}.");
        using var hrCoords = _acc.Allocate1D<int>((long)nTok * 3);
        hrCoords.View.CopyFromCPU(tokens);
        StageDone?.Invoke($"high-resolution tokens ({nTok} at {hrRes / 16}^3, target {hrRes})", sw.Elapsed.TotalSeconds); sw.Restart();

        // ---- 4. high-resolution shape latent: the 1024 checkpoint on the 1024 px features -------------------
        using var shapeSlat = _acc.Allocate1D<float>(nTok * 32L);
        {
            using var w = await _weights.LoadCheckpointAsync(_acc, "slat_flow_img2shape_dit_1_3B_1024_bf16", ct);
            await StageAsync("shape flow (1024) weights", sw);
            using var dit = new FlowDiT(_acc, _dk, w);
            dit.SetTokens(hrCoords.View, nTok);
            int count = nTok * dit.OutChannels;
            _sampler.Randn(shapeSlat.View, count, (ulong)seed * 7919UL + 4);
            _weights.Prefetch(new[] { "slat_flow_imgshape2tex_dit_1_3B_1024_bf16", "tex_dec_next_dc_f16c32_fp16" });
            await _sampler.SampleAsync((xv, t1000, useCond, v) => dit.Forward(xv, t1000, useCond ? cond1024 : neg1024.View, tokens1024, v), shapeSlat.View, count, _shapeOpts, ct: ct);
            await StageAsync($"shape flow (1024) sample (host {sw.Elapsed.TotalSeconds:F1} s)", sw);
        }

        // ---- 5. texture latent (1024 checkpoint, conditioned on the shape latent) ----------------------------------
        using var texSlat = _acc.Allocate1D<float>(nTok * 32L);
        {
            using var w = await _weights.LoadCheckpointAsync(_acc, "slat_flow_imgshape2tex_dit_1_3B_1024_bf16", ct);
            using var dit = new FlowDiT(_acc, _dk, w);
            dit.SetTokens(hrCoords.View, nTok);
            int ch = dit.OutChannels, count = nTok * ch;
            _sampler.Randn(texSlat.View, count, (ulong)seed * 7919UL + 3);
            using var modelIn = _acc.Allocate1D<float>((long)nTok * dit.InChannels);
            await StageAsync("tex flow load", sw);
            await _sampler.SampleAsync((xv, t1000, useCond, v) =>
            {
                _concat!(nTok * dit.InChannels, xv, shapeSlat.View, modelIn.View, ch, 32);
                dit.Forward(modelIn.View, t1000, useCond ? cond1024 : neg1024.View, tokens1024, v);
            }, texSlat.View, count, _texOpts, ct: ct);
            await StageAsync($"tex flow sample (host {sw.Elapsed.TotalSeconds:F1} s)", sw);
        }
        return new CascadeLatents(hrRes, nTok, tokens, await shapeSlat.View.CopyToHostAsync(), await texSlat.View.CopyToHostAsync());
    }

    /// <summary>The cascade's decoders: the shape decoder to the mesh (at <c>HrRes</c>^3 voxels), the texture decoder, guided by the shape decoder's subdivisions, to the per-voxel PBR attributes.</summary>
    private async Task<Output> DecodeCascadeAsync(CascadeLatents l, CancellationToken ct)
    {
        var sw = Stopwatch.StartNew();
        int nTok = l.Tokens, grid = l.HrRes / 16;
        using var hrCoords = _acc.Allocate1D<int>((long)nTok * 3);
        hrCoords.View.CopyFromCPU(l.Coords);
        var result = new Output { Grid = l.HrRes };

        // ---- shape decoder -> mesh ------------------------------------------------------------------------
        SparseVaeDecoder.Result shapeDec;
        {
            using var shapeSlat = _acc.Allocate1D<float>(l.ShapeSlat);
            using var sm = Upload(_shapeMean); using var ss = Upload(_shapeStd);
            using var lat = _acc.Allocate1D<float>(shapeSlat.Length);
            _affine!((int)shapeSlat.Length, shapeSlat.View, sm.View, ss.View, lat.View, 32);
            using var w = await _weights.LoadCheckpointAsync(_acc, "shape_dec_next_dc_f16c32_fp16", ct);
            var dec = new SparseVaeDecoder(_acc, _sk, _dk, w);
            await StageAsync("shape decoder load", sw);
            shapeDec = await dec.DecodeAsync(lat.View, hrCoords.View, nTok, grid, keepSubdivisions: true);
            result.Mesh = await _mesher.ExtractAsync(shapeDec.Feats.View, shapeDec.Coords.View, shapeDec.Count, shapeDec.Grid);
            await StageAsync($"shape decode + mesh ({shapeDec.Count} voxels, {result.Mesh.FaceCount} triangles)", sw);
        }

        // ---- texture decoder -> PBR voxels -----------------------------------------------------------------
        {
            using var texSlat = _acc.Allocate1D<float>(l.TexSlat);
            using var tm = Upload(_texMean); using var ts = Upload(_texStd);
            using var lat = _acc.Allocate1D<float>(texSlat.Length);
            _affine((int)texSlat.Length, texSlat.View, tm.View, ts.View, lat.View, 32);
            using var w = await _weights.LoadCheckpointAsync(_acc, "tex_dec_next_dc_f16c32_fp16", ct);
            var dec = new SparseVaeDecoder(_acc, _sk, _dk, w);
            using var tex = await dec.DecodeAsync(lat.View, hrCoords.View, nTok, grid, guide: shapeDec.Subdivisions);
            result.Attributes = _acc.Allocate1D<float>(tex.Feats.Length);
            _halfShift!((int)tex.Feats.Length, tex.Feats.View, result.Attributes.View);
            result.Coords = _acc.Allocate1D<int>((long)tex.Count * 3);
            result.Coords.View.CopyFrom(tex.Coords.View.SubView(0, (long)tex.Count * 3));
            result.VoxelCount = tex.Count;
            await StageAsync("tex decode", sw);
        }
        shapeDec.Dispose();
        return result;
    }

    /// <summary>
    /// The reference's <c>quant_coords = ((hr_coords + 0.5) / lr_resolution * (hr_resolution // 16)).int()</c> then <c>unique(dim=0)</c> (rows sorted ascending by x, y, z): the high-resolution token cells, in float32 as torch does it.
    /// </summary>
    internal static (int[] Coords, int Count) QuantizeUnique(int[] coords, int n, int lrRes, int hrRes)
    {
        float scale = hrRes / 16;
        var keys = new long[n];
        for (int i = 0; i < n; i++)
        {
            long x = (int)((coords[i * 3] + 0.5f) / lrRes * scale), y = (int)((coords[i * 3 + 1] + 0.5f) / lrRes * scale), z = (int)((coords[i * 3 + 2] + 0.5f) / lrRes * scale);
            keys[i] = (x << 40) | (y << 20) | z;
        }
        Array.Sort(keys);
        var outCoords = new int[n * 3]; int m = 0; long prev = -1;
        foreach (var k in keys)
        {
            if (k == prev) continue;
            prev = k;
            outCoords[m * 3] = (int)(k >> 40); outCoords[m * 3 + 1] = (int)((k >> 20) & 0xFFFFF); outCoords[m * 3 + 2] = (int)(k & 0xFFFFF); m++;
        }
        return (outCoords.AsSpan(0, m * 3).ToArray(), m);
    }

    // ---- region editing (training-free, RePaint-style) ---------------------------------------------------------------
    private Action<Index1D, FV, FV, FV, IV, int, float, float>? _repaint;
    private static void RepaintImpl(Index1D i, FV x, FV known, FV eps, IV keep, int ch, float a, float b)
    {
        if (keep[i / ch] != 0) x[i] = a * known[i] + b * eps[i];
    }

    /// <summary>
    /// Re-texture a REGION of a generated asset from a new picture, keeping everything else. The texture flow is run again under the new image (conditioned on the same shape latent, so the
    /// geometry does not change), and after every Euler step the tokens OUTSIDE the region are put back to the original texture latent noised to the current time
    /// (x_t = (1 - t) x0 + (sigma + (1 - sigma) t) eps, the flow's own path): RePaint for flow matching, no training. The region is where <paramref name="inRegion"/> says so for the
    /// token's centre in world coordinates (the volume is [-0.5, 0.5]^3). The mesh is extracted again from the kept shape latent (identical), the texture is decoded from the edited latent.
    /// </summary>
    /// <param name="source">a generation's output (it carries its latents, <see cref="Output.Edit"/>); not disposed</param>
    /// <param name="cond">DINOv3 features of the new picture, [condTokens, 1024]: the 512 px features for a 512 asset, the 1024 px ones for a 1024 asset</param>
    public async Task<Output> RetextureRegionAsync(Output source, FV cond, int condTokens, Func<float, float, float, bool> inRegion, int seed = 1, CancellationToken ct = default)
    {
        var st = source.Edit ?? throw new InvalidOperationException("This output carries no latents to edit (it was not made by a generation).");
        int n0 = st.Count, res = st.Res, slatGrid = st.SlatGrid;
        string suffix = res == 512 ? "512" : "1024";
        _affine ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, int>(AffineImpl);
        _concat ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int>(ConcatImpl);
        _halfShift ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(HalfShiftImpl);
        _zero ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV>(ZeroImpl);
        _repaint ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, IV, int, float, float>(RepaintImpl);
        var keepHost = new int[n0]; int regenerated = 0;
        for (int i = 0; i < n0; i++)
        {
            var (cx, cy, cz) = st.TokenCentre(i);
            bool inside = inRegion(cx, cy, cz);
            keepHost[i] = inside ? 0 : 1;
            if (inside) regenerated++;
        }
        if (regenerated == 0) throw new ArgumentException("The region holds no tokens of this asset.", nameof(inRegion));
        var sw = Stopwatch.StartNew();
        using var neg = _acc.Allocate1D<float>((long)condTokens * 1024);
        _zero((int)neg.Length, neg.View);
        using var coords32 = _acc.Allocate1D(st.Coords);
        using var shapeSlat = _acc.Allocate1D(st.ShapeSlat);
        using var keep = _acc.Allocate1D(keepHost);
        var result = new Output { Grid = res };

        // the shape decoder again: the mesh (identical) and the subdivision guide the texture decoder needs
        SparseVaeDecoder.Result shapeDec;
        {
            using var sm = Upload(_shapeMean); using var ss = Upload(_shapeStd);
            using var lat = _acc.Allocate1D<float>(shapeSlat.Length);
            _affine((int)shapeSlat.Length, shapeSlat.View, sm.View, ss.View, lat.View, 32);
            using var w = await _weights.LoadCheckpointAsync(_acc, "shape_dec_next_dc_f16c32_fp16", ct);
            var dec = new SparseVaeDecoder(_acc, _sk, _dk, w);
            shapeDec = await dec.DecodeAsync(lat.View, coords32.View, n0, slatGrid, keepSubdivisions: true);
            result.Mesh = await _mesher.ExtractAsync(shapeDec.Feats.View, shapeDec.Coords.View, shapeDec.Count, shapeDec.Grid);
            await StageAsync($"edit: shape decode + mesh ({result.Mesh.FaceCount} triangles)", sw);
        }

        // the texture flow under the new picture, the kept tokens pinned to the original latent
        MemoryBuffer1D<float, Stride1D.Dense> texSlat;
        {
            using var w = await _weights.LoadCheckpointAsync(_acc, "slat_flow_imgshape2tex_dit_1_3B_" + suffix + "_bf16", ct);
            using var dit = new FlowDiT(_acc, _dk, w);
            dit.SetTokens(coords32.View, n0);
            int ch = dit.OutChannels, count = n0 * ch;
            if (st.TexSlat.Length != count) throw new InvalidOperationException("The stored texture latent does not match the checkpoint.");
            using var known = _acc.Allocate1D(st.TexSlat);
            using var eps = _acc.Allocate1D<float>(count);
            _sampler.Randn(eps.View, count, (ulong)seed * 7919UL + 3);
            texSlat = _acc.Allocate1D<float>(count);
            texSlat.View.CopyFrom(eps.View);
            using var modelIn = _acc.Allocate1D<float>((long)n0 * dit.InChannels);
            float sigma = _texOpts.SigmaMin;
            await StageAsync("edit: tex flow load", sw);
            await _sampler.SampleAsync((xv, t1000, useCond, v) =>
            {
                _concat(n0 * dit.InChannels, xv, shapeSlat.View, modelIn.View, ch, 32);
                dit.Forward(modelIn.View, t1000, useCond ? cond : neg.View, condTokens, v);
            }, texSlat.View, count, _texOpts, null, (step, tPrev) => _repaint(count, texSlat.View, known.View, eps.View, keep.View, ch, (float)(1.0 - tPrev), (float)(sigma + (1.0 - sigma) * tPrev)));
            await StageAsync($"edit: tex flow sample ({regenerated} of {n0} tokens regenerated)", sw);
        }

        var edit = new EditState { Res = res, SlatGrid = slatGrid, Count = n0, Coords = st.Coords, ShapeSlat = st.ShapeSlat, SsLatent = st.SsLatent, SsRes = st.SsRes };
        {
            using var tm = Upload(_texMean); using var ts = Upload(_texStd);
            using var lat = _acc.Allocate1D<float>(texSlat.Length);
            _affine((int)texSlat.Length, texSlat.View, tm.View, ts.View, lat.View, 32);
            edit.TexSlat = await texSlat.View.SubView(0, texSlat.Length).CopyToHostAsync();
            texSlat.Dispose();
            using var w = await _weights.LoadCheckpointAsync(_acc, "tex_dec_next_dc_f16c32_fp16", ct);
            var dec = new SparseVaeDecoder(_acc, _sk, _dk, w);
            using var tex = await dec.DecodeAsync(lat.View, coords32.View, n0, slatGrid, guide: shapeDec.Subdivisions);
            result.Attributes = _acc.Allocate1D<float>(tex.Feats.Length);
            _halfShift((int)tex.Feats.Length, tex.Feats.View, result.Attributes.View);
            result.Coords = _acc.Allocate1D<int>((long)tex.Count * 3);
            result.Coords.View.CopyFrom(tex.Coords.View.SubView(0, (long)tex.Count * 3));
            result.VoxelCount = tex.Count;
            await StageAsync("edit: tex decode", sw);
        }
        shapeDec.Dispose();
        result.Edit = edit;
        return result;
    }

    private Action<Index1D, FV, FV, FV, IV, int, float, float>? _repaintCells;
    private static void RepaintCellsImpl(Index1D i, FV x, FV known, FV eps, IV keepCell, int cells, float a, float b)
    {
        if (keepCell[i % cells] != 0) x[i] = a * known[i] + b * eps[i];          // channel-first latent: element i belongs to cell i % cells
    }

    /// <summary>
    /// Edit the SHAPE inside a region from a new picture (training-free, RePaint for flow matching, like <see cref="RetextureRegionAsync"/>, one stage earlier): the sparse-structure flow is
    /// re-sampled under the new picture with the latent cells outside the region pinned, which decides which tokens exist (geometry can be ADDED or REMOVED inside the region); the shape flow runs on the
    /// new token set with the tokens that already existed and lie outside the region pinned to their old shape latent; the texture flow likewise; then the mesh and the texture are decoded.
    /// Everything outside the region stays what it was. The region is a predicate on world coordinates (the volume is [-0.5, 0.5]^3, generator frame: x right, y depth, z up).
    /// </summary>
    /// <param name="ssCond">DINOv3 features of the new picture at 512 px, [ssTokens, 1024] (the structure flow always takes the 512 px features)</param>
    /// <param name="cond">the features the shape and texture flows take: the same 512 px ones for a 512 asset, the 1024 px ones for a 1024 asset</param>
    public async Task<Output> EditShapeRegionAsync(Output source, FV ssCond, int ssTokens, FV cond, int condTokens, Func<float, float, float, bool> inRegion, int seed = 1, CancellationToken ct = default)
    {
        var st = source.Edit ?? throw new InvalidOperationException("This output carries no latents to edit (it was not made by a generation).");
        if (st.SsLatent.Length == 0) throw new InvalidOperationException("This output has no structure latent: it was made before shape editing existed.");
        int res = st.Res, G = st.SlatGrid;
        string suffix = res == 512 ? "512" : "1024";
        _affine ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, int>(AffineImpl);
        _concat ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int>(ConcatImpl);
        _halfShift ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(HalfShiftImpl);
        _zero ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV>(ZeroImpl);
        _repaint ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, IV, int, float, float>(RepaintImpl);
        _repaintCells ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, IV, int, float, float>(RepaintCellsImpl);
        var sw = Stopwatch.StartNew();
        using var neg = _acc.Allocate1D<float>((long)condTokens * 1024); _zero((int)neg.Length, neg.View);
        using var ssNeg = _acc.Allocate1D<float>((long)ssTokens * 1024); _zero((int)ssNeg.Length, ssNeg.View);
        float World(int c, int g) => (c + 0.5f) / g - 0.5f;

        // ---- 1. the structure: which tokens exist ------------------------------------------------------------------------------------------
        int[] newCoords; int n1; float[] newSs;
        {
            using var w = await _weights.LoadCheckpointAsync(_acc, "ss_flow_img_dit_1_3B_64_bf16", ct);
            using var flow = new SparseStructureFlow(_acc, _dk, w);
            int R = flow.Resolution, cells = R * R * R, count = flow.InChannels * cells;
            if (st.SsLatent.Length != count) throw new InvalidOperationException("The stored structure latent does not match the checkpoint.");
            var keepCellHost = new int[cells]; int regenCells = 0;
            for (int n = 0; n < cells; n++)
            {
                int cx = n / (R * R), cy = n / R % R, cz = n % R;
                bool inside = inRegion(World(cx, R), World(cy, R), World(cz, R));
                keepCellHost[n] = inside ? 0 : 1; if (inside) regenCells++;
            }
            if (regenCells == 0) throw new ArgumentException("The region holds no cells of this asset.", nameof(inRegion));
            using var keepCell = _acc.Allocate1D(keepCellHost);
            using var known = _acc.Allocate1D(st.SsLatent);
            using var eps = _acc.Allocate1D<float>(count);
            _sampler.Randn(eps.View, count, (ulong)seed * 7919UL + 1);
            using var x = _acc.Allocate1D<float>(count);
            x.View.CopyFrom(eps.View);
            float sigma = _ssOpts.SigmaMin;
            await StageAsync("edit shape: structure flow load", sw);
            await _sampler.SampleAsync((xv, t1000, useCond, v) => flow.Forward(xv, t1000, useCond ? ssCond : ssNeg.View, ssTokens, v), x.View, count, _ssOpts, null,
                (step, tPrev) => _repaintCells(count, x.View, known.View, eps.View, keepCell.View, cells, (float)(1.0 - tPrev), (float)(sigma + (1.0 - sigma) * tPrev)));
            newSs = await x.View.SubView(0, count).CopyToHostAsync();
            await StageAsync($"edit shape: structure flow sample ({regenCells} of {cells} cells regenerated)", sw);
            w.Dispose();
            using var dw = await _weights.LoadSsDecoderAsync(_acc, ct);
            using var dec = new SparseStructureDecoder(_acc, _sk, _dk, dw);
            using var logits = dec.Decode(x.View, R);
            var (c32, occupied) = await _sk.OccupancyAsync(logits.View, R * 4, G);
            n1 = occupied;
            if (n1 == 0) { c32.Dispose(); throw new InvalidOperationException("The edited structure is empty."); }
            newCoords = await c32.View.SubView(0, (long)n1 * 3).CopyToHostAsync();
            c32.Dispose();
            await StageAsync($"edit shape: structure decode ({n1} tokens, was {st.Count})", sw);
        }

        // which tokens keep their old latents: existed before AND lie outside the region
        var oldIndex = new Dictionary<int, int>(st.Count);
        for (int i = 0; i < st.Count; i++) oldIndex[(st.Coords[i * 3] * G + st.Coords[i * 3 + 1]) * G + st.Coords[i * 3 + 2]] = i;
        int texCh = st.TexSlat.Length / st.Count;
        var keepTok = new int[n1]; var knownShape = new float[(long)n1 * 32]; var knownTex = new float[(long)n1 * texCh];
        int kept = 0;
        for (int i = 0; i < n1; i++)
        {
            int cx = newCoords[i * 3], cy = newCoords[i * 3 + 1], cz = newCoords[i * 3 + 2];
            if (oldIndex.TryGetValue((cx * G + cy) * G + cz, out int o) && !inRegion(World(cx, G), World(cy, G), World(cz, G)))
            {
                keepTok[i] = 1; kept++;
                Array.Copy(st.ShapeSlat, (long)o * 32, knownShape, (long)i * 32, 32);
                Array.Copy(st.TexSlat, (long)o * texCh, knownTex, (long)i * texCh, texCh);
            }
        }
        using var coords32 = _acc.Allocate1D(newCoords);
        using var keep = _acc.Allocate1D(keepTok);
        var result = new Output { Grid = res };

        // ---- 2. the shape latent on the new token set --------------------------------------------------------------------------------------
        MemoryBuffer1D<float, Stride1D.Dense> shapeSlat;
        {
            using var w = await _weights.LoadCheckpointAsync(_acc, "slat_flow_img2shape_dit_1_3B_" + suffix + "_bf16", ct);
            using var dit = new FlowDiT(_acc, _dk, w);
            dit.SetTokens(coords32.View, n1);
            int ch = dit.OutChannels, count = n1 * ch;
            using var known = _acc.Allocate1D(knownShape);
            using var eps = _acc.Allocate1D<float>(count);
            _sampler.Randn(eps.View, count, (ulong)seed * 7919UL + 2);
            shapeSlat = _acc.Allocate1D<float>(count);
            shapeSlat.View.CopyFrom(eps.View);
            float sigma = _shapeOpts.SigmaMin;
            await StageAsync("edit shape: shape flow load", sw);
            await _sampler.SampleAsync((xv, t1000, useCond, v) => dit.Forward(xv, t1000, useCond ? cond : neg.View, condTokens, v), shapeSlat.View, count, _shapeOpts, null,
                (step, tPrev) => _repaint(count, shapeSlat.View, known.View, eps.View, keep.View, ch, (float)(1.0 - tPrev), (float)(sigma + (1.0 - sigma) * tPrev)));
            await StageAsync($"edit shape: shape flow sample ({n1 - kept} of {n1} tokens regenerated)", sw);
        }
        var edit = new EditState { Res = res, SlatGrid = G, Count = n1, Coords = newCoords, SsLatent = newSs, SsRes = st.SsRes };
        edit.ShapeSlat = await shapeSlat.View.SubView(0, (long)n1 * 32).CopyToHostAsync();

        // ---- 3. the shape decoded: the mesh and the subdivision guide ----------------------------------------------------------------------
        SparseVaeDecoder.Result shapeDec;
        {
            using var sm = Upload(_shapeMean); using var ss = Upload(_shapeStd);
            using var lat = _acc.Allocate1D<float>(shapeSlat.Length);
            _affine((int)shapeSlat.Length, shapeSlat.View, sm.View, ss.View, lat.View, 32);
            using var w = await _weights.LoadCheckpointAsync(_acc, "shape_dec_next_dc_f16c32_fp16", ct);
            var dec = new SparseVaeDecoder(_acc, _sk, _dk, w);
            shapeDec = await dec.DecodeAsync(lat.View, coords32.View, n1, G, keepSubdivisions: true);
            result.Mesh = await _mesher.ExtractAsync(shapeDec.Feats.View, shapeDec.Coords.View, shapeDec.Count, shapeDec.Grid);
            await StageAsync($"edit shape: shape decode + mesh ({result.Mesh.FaceCount} triangles)", sw);
        }

        // ---- 4. the texture latent: kept tokens pinned, the rest under the new picture and the new shape -----------------------------------
        MemoryBuffer1D<float, Stride1D.Dense> texSlat;
        {
            using var w = await _weights.LoadCheckpointAsync(_acc, "slat_flow_imgshape2tex_dit_1_3B_" + suffix + "_bf16", ct);
            using var dit = new FlowDiT(_acc, _dk, w);
            dit.SetTokens(coords32.View, n1);
            int ch = dit.OutChannels, count = n1 * ch;
            if (ch != texCh) throw new InvalidOperationException("The stored texture latent does not match the checkpoint.");
            using var known = _acc.Allocate1D(knownTex);
            using var eps = _acc.Allocate1D<float>(count);
            _sampler.Randn(eps.View, count, (ulong)seed * 7919UL + 3);
            texSlat = _acc.Allocate1D<float>(count);
            texSlat.View.CopyFrom(eps.View);
            using var modelIn = _acc.Allocate1D<float>((long)n1 * dit.InChannels);
            float sigma = _texOpts.SigmaMin;
            await StageAsync("edit shape: tex flow load", sw);
            await _sampler.SampleAsync((xv, t1000, useCond, v) =>
            {
                _concat(n1 * dit.InChannels, xv, shapeSlat.View, modelIn.View, ch, 32);
                dit.Forward(modelIn.View, t1000, useCond ? cond : neg.View, condTokens, v);
            }, texSlat.View, count, _texOpts, null, (step, tPrev) => _repaint(count, texSlat.View, known.View, eps.View, keep.View, ch, (float)(1.0 - tPrev), (float)(sigma + (1.0 - sigma) * tPrev)));
            await StageAsync("edit shape: tex flow sample", sw);
        }
        shapeSlat.Dispose();
        {
            using var tm = Upload(_texMean); using var ts = Upload(_texStd);
            using var lat = _acc.Allocate1D<float>(texSlat.Length);
            _affine((int)texSlat.Length, texSlat.View, tm.View, ts.View, lat.View, 32);
            edit.TexSlat = await texSlat.View.SubView(0, texSlat.Length).CopyToHostAsync();
            texSlat.Dispose();
            using var w = await _weights.LoadCheckpointAsync(_acc, "tex_dec_next_dc_f16c32_fp16", ct);
            var dec = new SparseVaeDecoder(_acc, _sk, _dk, w);
            using var tex = await dec.DecodeAsync(lat.View, coords32.View, n1, G, guide: shapeDec.Subdivisions);
            result.Attributes = _acc.Allocate1D<float>(tex.Feats.Length);
            _halfShift((int)tex.Feats.Length, tex.Feats.View, result.Attributes.View);
            result.Coords = _acc.Allocate1D<int>((long)tex.Count * 3);
            result.Coords.View.CopyFrom(tex.Coords.View.SubView(0, (long)tex.Count * 3));
            result.VoxelCount = tex.Count;
            await StageAsync("edit shape: tex decode", sw);
        }
        shapeDec.Dispose();
        result.Edit = edit;
        return result;
    }

    /// <summary>
    /// Turn a generation into a GLB: read the mesh back (the topology work that follows is sequential and ends in a file),
    /// weld away unreferenced vertices, compute normals, unwrap a UV atlas, bake the PBR attributes into two textures on the device,
    /// and write the binary glTF (positions, normals, UVs, indices, base colour + metallic-roughness PNGs).
    /// </summary>
    public byte[] ToGlb(Output output, int textureSize = 2048, int dilation = 8, int maxFaces = 150_000)
        => Task.Run(() => ToGlbAsync(output, textureSize, dilation, maxFaces)).GetAwaiter().GetResult();

    /// <summary><see cref="ToGlb"/>, browser-safe (the mesh and textures are awaited readbacks).</summary>
    public async Task<byte[]> ToGlbAsync(Output output, int textureSize = 2048, int dilation = 8, int maxFaces = 150_000)
    {
        var sw = Stopwatch.StartNew();
        if (Remesh) return await RemeshGlbAsync(output, textureSize, dilation, maxFaces, sw);
        if (maxFaces > 0 && PreCluster && output.Mesh.FaceCount > maxFaces * 2 && Environment.GetEnvironmentVariable("TRELLIS2_NOCLUSTER") != "1")
        {
            // coarsen on the device first (O(n)); the host's edge collapse then finishes from a fraction of the faces
            int before = output.Mesh.FaceCount;
            _clusterer ??= new MeshClusterer(_acc, _sk);
            var (cp, cf) = await _clusterer.ClusterAsync(output.Mesh.Vertices.View, output.Mesh.VertexCount,
                output.Mesh.Faces.View.SubView(0, (long)before * 3), before, output.Grid, ClusterOptions);
            StageDone?.Invoke($"cluster {before} -> {cf.Length / 3} faces", sw.Elapsed.TotalSeconds); sw.Restart();
            return await FinishGlbAsync(output, cp, cf, textureSize, dilation, maxFaces, sw);
        }
        var verts = await output.Mesh.Vertices.View.CopyToHostAsync();
        var faces0 = await output.Mesh.Faces.View.SubView(0, (long)output.Mesh.FaceCount * 3).CopyToHostAsync();
        // keep only referenced vertices
        var remap = new int[verts.Length / 3]; Array.Fill(remap, -1);
        int used = 0;
        foreach (int v in faces0) if (remap[v] < 0) remap[v] = used++;
        var positions = new float[used * 3];
        for (int v = 0; v < remap.Length; v++)
            if (remap[v] >= 0) { positions[remap[v] * 3] = verts[v * 3]; positions[remap[v] * 3 + 1] = verts[v * 3 + 1]; positions[remap[v] * 3 + 2] = verts[v * 3 + 2]; }
        var faces = new int[faces0.Length];
        for (int i = 0; i < faces.Length; i++) faces[i] = remap[faces0[i]];
        return await FinishGlbAsync(output, positions, faces, textureSize, dilation, maxFaces, sw);
    }

    /// <summary>
    /// Rebuild the extracted mesh as a closed shell before simplifying, unwrapping and baking (<see cref="NarrowBandRemesher"/>: the reference's <c>to_glb(remesh=True)</c>, which its own example uses),
    /// and read every texel's colour at the nearest point of the ORIGINAL mesh (<see cref="TextureBaker"/> with a <see cref="MeshDistanceGrid"/>). Pinholes and cracks close, thin plates become one
    /// watertight outward-facing skin and the dark inner layer of a plate no longer bleeds into the baked colour. Default on (<see cref="Trellis2Options.Remesh"/>).
    /// </summary>
    public bool Remesh { get; set; } = true;
    /// <summary>Offset of the remeshed shell from the original surface, in voxels (reference 1).</summary>
    public float RemeshBand { get; set; } = 1f;

    /// <summary>Run the remesh in device kernels (<see cref="DeviceRemesher"/>: the mesh never leaves the GPU, no host repair first) instead of the host topology (<see cref="NarrowBandRemesher"/>). Default on; env TRELLIS2_HOSTREMESH=1 forces the host path (A/B).</summary>
    public bool DeviceRemesh { get; set; } = Environment.GetEnvironmentVariable("TRELLIS2_HOSTREMESH") != "1";
    private DeviceRemesher? _deviceRemesher;
    /// <summary>Simplify on the device (<see cref="GpuMeshSimplifier"/>) instead of the host edge collapse. Default on; env TRELLIS2_HOSTSIMPLIFY=1 forces the host path (A/B).</summary>
    public bool DeviceSimplify { get; set; } = Environment.GetEnvironmentVariable("TRELLIS2_HOSTSIMPLIFY") != "1";
    private GpuMeshSimplifier? _gpuSimplifier;
    /// <summary>Align the winding, compute the normals and unwrap the UV atlas on the device (<see cref="DeviceUvAtlas"/>) and bake from its buffers, so the mesh does not leave the GPU until the file is written. Default on; env TRELLIS2_HOSTATLAS=1 forces the host path (A/B). Used after the device remesh, unless the host repair of the simplified mesh is on.</summary>
    public bool DeviceAtlas { get; set; } = Environment.GetEnvironmentVariable("TRELLIS2_HOSTATLAS") != "1";
    private DeviceUvAtlas? _deviceAtlas;
    /// <summary>Run the host <see cref="MeshRepair"/> on the device-simplified mesh (splits non-manifold edges, closes the holes that leaves, re-winds). Off by default: the remeshed, simplified surface is already closed and consistently wound, and the repair opens pinholes along the edges it splits. Env TRELLIS2_REPAIR_SIMPLIFIED=1 turns it on (A/B).</summary>
    public bool RepairAfterDeviceSimplify { get; set; } = Environment.GetEnvironmentVariable("TRELLIS2_REPAIR_SIMPLIFIED") == "1";

    private async Task<byte[]> RemeshGlbAsync(Output output, int textureSize, int dilation, int maxFaces, Stopwatch sw)
    {
        if (DeviceRemesh)
        {
            _deviceRemesher ??= new DeviceRemesher(_acc, _sk);
            using var rmd = await _deviceRemesher.RemeshAsync(output.Mesh.Vertices.View, output.Mesh.VertexCount, output.Mesh.Faces.View, output.Mesh.FaceCount, output.Grid, RemeshBand);
            StageDone?.Invoke($"remesh on the device: {rmd.ActiveVoxels} band voxels, {rmd.GridVertices} corners -> {rmd.FaceCount} faces", sw.Elapsed.TotalSeconds); sw.Restart();
            DeviceTriMesh? simplified = null;
            if (DeviceSimplify && maxFaces > 0 && rmd.FaceCount > maxFaces)
            {
                _gpuSimplifier ??= new GpuMeshSimplifier(_acc, _sk);
                int before = rmd.FaceCount;
                // a hair under the budget: the repair that follows closes holes and adds a few hundred faces, which must not push the mesh back over it
                simplified = await _gpuSimplifier.SimplifyAsync(rmd.Positions.View, rmd.VertexCount, rmd.Faces.View, rmd.FaceCount, (int)(maxFaces * 0.995));
                StageDone?.Invoke($"simplify on the device {before} -> {simplified.FaceCount} faces in {_gpuSimplifier.LastRounds.Count} rounds", sw.Elapsed.TotalSeconds); sw.Restart();
            }
            using var simplifiedOwner = simplified;
            var vp = simplified?.Positions ?? rmd.Positions; var vf = simplified?.Faces ?? rmd.Faces;
            int vn = simplified?.VertexCount ?? rmd.VertexCount, fn = simplified?.FaceCount ?? rmd.FaceCount;
            if (DeviceAtlas && !(simplified != null && RepairMesh && RepairAfterDeviceSimplify))
            {
                if (MeshStage != null)
                {
                    var hp = vn == 0 ? Array.Empty<float>() : await vp.View.SubView(0, (long)vn * 3).CopyToHostAsync();
                    var hf = fn == 0 ? Array.Empty<int>() : await vf.View.SubView(0, (long)fn * 3).CopyToHostAsync();
                    MeshStage("remeshed", hp, hf);
                }
                return await FinishGlbOnDeviceAsync(output, vp.View, vn, vf.View, fn, textureSize, dilation, sw, rmd.Source);
            }
            // the stages after it (host repair, host atlas) need arrays
            var dpos = vn == 0 ? Array.Empty<float>() : await vp.View.SubView(0, (long)vn * 3).CopyToHostAsync();
            var dfaces = fn == 0 ? Array.Empty<int>() : await vf.View.SubView(0, (long)fn * 3).CopyToHostAsync();
            MeshStage?.Invoke("remeshed", dpos, dfaces);
            if (simplified != null && RepairMesh && RepairAfterDeviceSimplify) { (dpos, dfaces) = Repair("(simplified)", dpos, dfaces, sw); MeshStage?.Invoke("simplified", dpos, dfaces); }
            return await FinishGlbAsync(output, dpos, dfaces, textureSize, dilation, maxFaces, sw, rmd.Source, remeshed: true);
        }
        var verts = await output.Mesh.Vertices.View.CopyToHostAsync();
        var faces0 = await output.Mesh.Faces.View.SubView(0, (long)output.Mesh.FaceCount * 3).CopyToHostAsync();
        var remap = new int[verts.Length / 3]; Array.Fill(remap, -1);
        int used = 0;
        foreach (int v in faces0) if (remap[v] < 0) remap[v] = used++;
        var positions = new float[used * 3];
        for (int v = 0; v < remap.Length; v++)
            if (remap[v] >= 0) { positions[remap[v] * 3] = verts[v * 3]; positions[remap[v] * 3 + 1] = verts[v * 3 + 1]; positions[remap[v] * 3 + 2] = verts[v * 3 + 2]; }
        var faces = new int[faces0.Length];
        for (int i = 0; i < faces.Length; i++) faces[i] = remap[faces0[i]];
        StageDone?.Invoke($"mesh read ({faces.Length / 3} faces)", sw.Elapsed.TotalSeconds); sw.Restart();
        MeshStage?.Invoke("extracted", positions, faces);
        if (RepairMesh) { (positions, faces) = Repair("(dual grid)", positions, faces, sw); MeshStage?.Invoke("repaired", positions, faces); }
        var rm = await NarrowBandRemesher.RemeshAsync(_acc, positions, faces, output.Grid, RemeshBand);
        StageDone?.Invoke($"remesh: {rm.ActiveVoxels} band voxels, {rm.GridVertices} corners -> {rm.Faces.Length / 3} faces", sw.Elapsed.TotalSeconds); sw.Restart();
        MeshStage?.Invoke("remeshed", rm.Positions, rm.Faces);
        using (rm) return await FinishGlbAsync(output, rm.Positions, rm.Faces, textureSize, dilation, maxFaces, sw, rm.Source, remeshed: true);
    }

    /// <summary>Coarsen the dual-grid mesh on the device (<see cref="MeshClusterer"/>) before the host edge collapse. Default on.</summary>
    public bool PreCluster { get; set; } = true;
    /// <summary>Thresholds of the device pre-pass (null = defaults).</summary>
    public MeshClusterer.Options? ClusterOptions { get; set; }
    private MeshClusterer? _clusterer;

    /// <summary>
    /// Diagnostic: the mesh (positions xyz, triangle indices) as it leaves each stage of <see cref="ToGlbAsync"/>:
    /// "extracted" (what the topology work received: the dual-grid mesh, or its device pre-clustering), "simplified", "atlas" (UV seams split).
    /// Null costs nothing. Used to find which stage makes a mesh non-manifold or open (harness TRELLIS2_MESHSTAGES).
    /// </summary>
    public Action<string, float[], int[]>? MeshStage { get; set; }

    /// <summary>Run <see cref="MeshRepair"/> before and after the edge collapse (default on; <see cref="Trellis2Options.RepairMesh"/>).</summary>
    public bool RepairMesh { get; set; } = true;

    private (float[], int[]) Repair(string stage, float[] positions, int[] faces, Stopwatch sw)
    {
        var (p, f, r) = MeshRepair.Repair(positions, faces);
        StageDone?.Invoke($"mesh repair {stage}: open {r.Before.BoundaryEdges}->{r.After.BoundaryEdges}, non-manifold {r.Before.NonManifoldEdges}->{r.After.NonManifoldEdges}, " +
            $"wrong-way edges {r.Before.OrientationConflicts}->{r.After.OrientationConflicts}, {r.HolesFilled} holes closed, {r.FragmentComponents} fragments removed", sw.Elapsed.TotalSeconds);
        sw.Restart();
        return (p, f);
    }

    /// <summary>
    /// The end of the export with the mesh still in device buffers: winding and normals, the UV atlas, the bake, and only then the arrays the file writer needs (the finished vertices, UVs, normals, faces and two textures).
    /// </summary>
    private async Task<byte[]> FinishGlbOnDeviceAsync(Output output, ArrayView1D<float, Stride1D.Dense> positions, int vertexCount, ArrayView1D<int, Stride1D.Dense> faces, int faceCount, int textureSize, int dilation, Stopwatch sw, MeshDistanceGrid? project)
    {
        _deviceAtlas ??= new DeviceUvAtlas(_acc, _sk);
        using var atlas = await _deviceAtlas.UnwrapAsync(positions, vertexCount, faces, faceCount, new UvAtlas.Options { Resolution = textureSize }, alignWinding: RepairMesh);
        StageDone?.Invoke($"winding, normals and uv atlas on the device ({atlas.ChartCount} charts, {atlas.Utilization:P0}; {atlas.Flipped} triangles flipped; first kernels loaded in {_deviceAtlas.LoadSeconds:F1} s)", sw.Elapsed.TotalSeconds); sw.Restart();
        if (MeshStage != null)
            MeshStage("atlas", await atlas.Positions.View.SubView(0, (long)atlas.VertexCount * 3).CopyToHostAsync(), await atlas.Faces.View.SubView(0, (long)atlas.FaceCount * 3).CopyToHostAsync());

        var tex = await new TextureBaker(_acc, _sk).BakeAsync(atlas.Positions.View, atlas.Uvs.View, atlas.Faces.View, atlas.FaceCount, output.Coords.View, output.Attributes.View, output.VoxelCount, output.Grid, textureSize, dilation, project);
        StageDone?.Invoke("texture bake", sw.Elapsed.TotalSeconds); sw.Restart();

        var outPositions = await atlas.Positions.View.SubView(0, (long)atlas.VertexCount * 3).CopyToHostAsync();
        var outUvs = await atlas.Uvs.View.SubView(0, (long)atlas.VertexCount * 2).CopyToHostAsync();
        var outNormals = await atlas.Normals.View.SubView(0, (long)atlas.VertexCount * 3).CopyToHostAsync();
        var outFaces = await atlas.Faces.View.SubView(0, (long)atlas.FaceCount * 3).CopyToHostAsync();
        var bc = PngEncoder.EncodeRgba8(tex.BaseColorAlpha, textureSize, textureSize);
        var mr = PngEncoder.EncodeRgba8(tex.MetallicRoughness, textureSize, textureSize);
        var glb = PbrGlbWriter.Write(outPositions, outNormals, outUvs, outFaces, bc, mr);
        StageDone?.Invoke("png + glb", sw.Elapsed.TotalSeconds);
        return glb;
    }

    private async Task<byte[]> FinishGlbAsync(Output output, float[] positions, int[] faces, int textureSize, int dilation, int maxFaces, Stopwatch sw, MeshDistanceGrid? project = null, bool remeshed = false)
    {
        if (!remeshed) MeshStage?.Invoke("extracted", positions, faces);
        if (RepairMesh && !remeshed) { (positions, faces) = Repair("(dual grid)", positions, faces, sw); MeshStage?.Invoke("repaired", positions, faces); }
        if (maxFaces > 0 && faces.Length / 3 > maxFaces)
        {
            int before = faces.Length / 3;
            (positions, faces) = MeshSimplifier.Simplify(positions, faces, maxFaces);
            StageDone?.Invoke($"simplify {before} -> {faces.Length / 3} faces", sw.Elapsed.TotalSeconds); sw.Restart();
            // the edge collapse of a repaired mesh still leaves a few pinches and one-face gaps: repair again, on 150k faces it is a fraction of a second
            if (RepairMesh) (positions, faces) = Repair("(simplified)", positions, faces, sw);
            MeshStage?.Invoke("simplified", positions, faces);
        }
        if (RepairMesh)
        {
            int realigned = MeshRepair.AlignWindingToNormals(positions, faces);
            StageDone?.Invoke($"winding aligned to the normals: {realigned} triangles flipped", sw.Elapsed.TotalSeconds); sw.Restart();
        }
        var normals = PbrGlbWriter.VertexNormals(positions, faces);
        StageDone?.Invoke("mesh read + normals", sw.Elapsed.TotalSeconds); sw.Restart();

        var atlas = UvAtlas.Unwrap(positions, faces, new UvAtlas.Options { Resolution = textureSize });
        StageDone?.Invoke($"uv atlas ({atlas.ChartCount} charts, {atlas.Utilization:P0})", sw.Elapsed.TotalSeconds); sw.Restart();
        MeshStage?.Invoke("atlas", atlas.Positions, atlas.Faces);

        TextureBaker.Textures tex;
        using (var pos = _acc.Allocate1D(atlas.Positions))
        using (var uv = _acc.Allocate1D(atlas.Uvs))
        using (var fb = _acc.Allocate1D(atlas.Faces))
            tex = await new TextureBaker(_acc, _sk).BakeAsync(pos.View, uv.View, fb.View, atlas.Faces.Length / 3, output.Coords.View, output.Attributes.View, output.VoxelCount, output.Grid, textureSize, dilation, project);
        StageDone?.Invoke("texture bake", sw.Elapsed.TotalSeconds); sw.Restart();

        var splitNormals = new float[atlas.Positions.Length];
        for (int i = 0; i < atlas.SourceVertex.Length; i++)
        {
            int s = atlas.SourceVertex[i];
            splitNormals[i * 3] = normals[s * 3]; splitNormals[i * 3 + 1] = normals[s * 3 + 1]; splitNormals[i * 3 + 2] = normals[s * 3 + 2];
        }
        var bc = PngEncoder.EncodeRgba8(tex.BaseColorAlpha, textureSize, textureSize);
        var mr = PngEncoder.EncodeRgba8(tex.MetallicRoughness, textureSize, textureSize);
        var glb = PbrGlbWriter.Write(atlas.Positions, splitNormals, atlas.Uvs, atlas.Faces, bc, mr);
        StageDone?.Invoke("png + glb", sw.Elapsed.TotalSeconds);
        return glb;
    }

    public void Dispose() { }
}
