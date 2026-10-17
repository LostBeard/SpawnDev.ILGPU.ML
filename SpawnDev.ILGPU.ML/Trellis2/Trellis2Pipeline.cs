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
        public void Dispose() { Mesh?.Dispose(); Attributes?.Dispose(); Coords?.Dispose(); }
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
        MemoryBuffer1D<int, Stride1D.Dense> coords32;
        int n0;
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
            _sampler.Sample((xv, t1000, useCond, v) => flow.Forward(xv, t1000, useCond ? ssCond : ssNeg.View, ssTokens, v), x.View, count, _ssOpts);
            await StageAsync("ss flow sample", sw);
            if (dumpDir.Length > 0) { var a = x.View.SubView(0, count).GetAsArray1D(); var b = new byte[a.Length * 4]; Buffer.BlockCopy(a, 0, b, 0, b.Length); File.WriteAllBytes(Path.Combine(dumpDir, "zs.f32"), b); }
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
            _sampler.Sample((xv, t1000, useCond, v) => dit.Forward(xv, t1000, useCond ? cond : neg.View, condTokens, v), shapeSlat.View, count, _shapeOpts);
            await StageAsync("shape flow sample", sw);
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
            _sampler.Sample((xv, t1000, useCond, v) =>
            {
                _concat(n0 * dit.InChannels, xv, shapeSlat.View, modelIn.View, ch, 32);
                dit.Forward(modelIn.View, t1000, useCond ? cond : neg.View, condTokens, v);
            }, texSlat.View, count, _texOpts);
            await StageAsync("tex flow sample", sw);
            if (Environment.GetEnvironmentVariable("TRELLIS2_DEBUG_SLAT") == "1") Console.WriteLine($"  [slat] tex: {Moments(texSlat.View, count)}");
            { var dt = Environment.GetEnvironmentVariable("TRELLIS2_DUMP_TEX") ?? ""; if (dt.Length > 0) { _acc.Synchronize(); var a = texSlat.View.SubView(0, count).GetAsArray1D(); var b = new byte[a.Length * 4]; Buffer.BlockCopy(a, 0, b, 0, b.Length); File.WriteAllBytes(Path.Combine(dt, "tex_slat.f32"), b); } }
        }
        shapeSlat.Dispose();

        // ---- 5. texture decoder -> PBR voxels -------------------------------------------------------------
        {
            using var tm = Upload(_texMean); using var ts = Upload(_texStd);
            using var lat = _acc.Allocate1D<float>(texSlat.Length);
            _affine((int)texSlat.Length, texSlat.View, tm.View, ts.View, lat.View, 32);
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

    /// <summary>Coarsen the dual-grid mesh on the device (<see cref="MeshClusterer"/>) before the host edge collapse. Default on.</summary>
    public bool PreCluster { get; set; } = true;
    /// <summary>Thresholds of the device pre-pass (null = defaults).</summary>
    public MeshClusterer.Options? ClusterOptions { get; set; }
    private MeshClusterer? _clusterer;

    private async Task<byte[]> FinishGlbAsync(Output output, float[] positions, int[] faces, int textureSize, int dilation, int maxFaces, Stopwatch sw)
    {
        if (maxFaces > 0 && faces.Length / 3 > maxFaces)
        {
            int before = faces.Length / 3;
            (positions, faces) = MeshSimplifier.Simplify(positions, faces, maxFaces);
            StageDone?.Invoke($"simplify {before} -> {faces.Length / 3} faces", sw.Elapsed.TotalSeconds); sw.Restart();
        }
        var normals = PbrGlbWriter.VertexNormals(positions, faces);
        StageDone?.Invoke("mesh read + normals", sw.Elapsed.TotalSeconds); sw.Restart();

        var atlas = UvAtlas.Unwrap(positions, faces, new UvAtlas.Options { Resolution = textureSize });
        StageDone?.Invoke($"uv atlas ({atlas.ChartCount} charts, {atlas.Utilization:P0})", sw.Elapsed.TotalSeconds); sw.Restart();

        TextureBaker.Textures tex;
        using (var pos = _acc.Allocate1D(atlas.Positions))
        using (var uv = _acc.Allocate1D(atlas.Uvs))
        using (var fb = _acc.Allocate1D(atlas.Faces))
            tex = await new TextureBaker(_acc, _sk).BakeAsync(pos.View, uv.View, fb.View, atlas.Faces.Length / 3, output.Coords.View, output.Attributes.View, output.VoxelCount, output.Grid, textureSize, dilation);
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
