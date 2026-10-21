using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.Preprocessing;
using TypedArray = SpawnDev.SpawnJS.JSObjects.TypedArray;

namespace SpawnDev.ILGPU.ML.Trellis2;

using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>Where the checkpoints live.</summary>
public sealed class Trellis2Paths
{
    /// <summary>The microsoft/TRELLIS.2-4B folder (pipeline.json, ckpts/).</summary>
    public string ModelDir { get; init; } = "";
    /// <summary>Path WITHOUT extension of microsoft/TRELLIS-image-large ckpts/ss_dec_conv3d_16l8_fp16 (.json + .safetensors).</summary>
    public string SparseStructureDecoder { get; init; } = "";
    /// <summary>The DINOv3 ViT-L/16 weights (a .safetensors file in the Hugging Face / timm-converted layout the extractor accepts).</summary>
    public string DinoWeights { get; init; } = "";
}

public sealed class Trellis2Options
{
    /// <summary>Random seed of the sampler noise (own counter-based Gaussian, deterministic).</summary>
    public int Seed { get; init; } = 42;
    /// <summary>Side of the baked textures in texels.</summary>
    public int TextureSize { get; init; } = 2048;
    /// <summary>
    /// 512 (default, ~40 s on an RTX 4070, 3 GB) or 1024 (the 1024 pipeline of TRELLIS.2: four times the voxels, finer detail; minutes, several GB, needs the two
    /// 1024 flow checkpoints next to the 512 ones). The sparse structure comes from the 512 px features either way. The 1024 path matches the reference
    /// sampler on identical inputs (texture latent relRMS 0.9%) but its texture latent is less spread than the 512 one (std 0.68 vs 0.85), so colours come
    /// out paler: use 512 for colour fidelity, 1024 for fine geometry.
    /// </summary>
    public int Resolution { get; init; } = 512;
    /// <summary>
    /// The reference's CASCADE pipelines (<c>1024_cascade</c>, its default, and <c>1536_cascade</c>): the coarse shape from the 512 model, its decoder's upsampling as the high-resolution token set, then the 1024 models on it, decoded at
    /// <see cref="Resolution"/> (1024 or 1536; the token cap can lower 1536 to 1408, 1280 or 1024). Without it, 1024 is the plain <c>1024</c> pipeline (the structure at 64^3). 1536 needs the cascade.
    /// </summary>
    public bool Cascade { get; init; }
    /// <summary>The mesh is simplified to about this many triangles before the UV unwrap (0 keeps the full dual-grid mesh, 1-2.5M triangles). Measured on the official crown: 120k looks the same as 300k; at 40-50k thin sheets shred and texels go missing.</summary>
    public int MaxFaces { get; init; } = 150_000;
    /// <summary>
    /// Clean the mesh up (<see cref="MeshRepair"/>) before and after it is simplified, as the reference's <c>to_glb</c> does: no non-manifold edges, one consistent
    /// winding, small holes closed, floating fragments removed. The dual-grid mesh straight from the decoder is open and non-manifold (crown: 4.7% open edges,
    /// 227k non-manifold edges, 730k orientation conflicts). Costs a few seconds of host time on a multi-million face mesh.
    /// </summary>
    public bool RepairMesh { get; init; } = true;
    /// <summary>Rebuild the mesh as a closed shell by narrow-band dual contouring and bake colour from the original surface, as the reference's to_glb(remesh=True) does (<see cref="NarrowBandRemesher"/>). Default on.</summary>
    public bool Remesh { get; init; } = true;
}

/// <summary>
/// Image to 3D, the public entry point: an RGBA image whose alpha is the foreground mask -> PBR GLB (positions, normals, UVs, base colour and
/// metallic-roughness textures). All neural compute and the texture bake run on the device through one stage's weights at a time; only
/// the finished mesh crosses to the host (a terminal sink: the topology work that ends in the file).
/// Pixels are packed RGBA ints, R in the low byte (the library convention; <see cref="RgbaUpload"/>).
/// </summary>
public sealed class Trellis2ImageTo3D : IDisposable
{
    private readonly Accelerator _acc;
    private readonly Trellis2Weights _weights;
    private readonly DiTKernels _dk;
    private readonly SparseKernels _sk;
    private Task? _precompile;

    /// <summary>Per-stage progress: (stage, seconds).</summary>
    public Action<string, double>? StageDone { get; set; }

    /// <summary>Checkpoints from local files (desktop).</summary>
    public Trellis2ImageTo3D(Accelerator accelerator, Trellis2Paths paths) : this(accelerator, Trellis2Weights.FromFiles(paths)) { }

    /// <summary>Checkpoints from any source: <see cref="Trellis2Weights.FromModelSource"/> streams them from the hub (OPFS cache in the browser).</summary>
    public Trellis2ImageTo3D(Accelerator accelerator, Trellis2Weights weights)
    {
        _acc = accelerator; _weights = weights;
        _dk = new DiTKernels(accelerator); _sk = new SparseKernels(accelerator);
    }

    /// <summary>Generate from a device image. <paramref name="rgba"/> [width*height] packed RGBA. Desktop only (blocks); browsers use <see cref="GenerateGlbAsync(IV, int, int, Trellis2Options?, CancellationToken)"/>.</summary>
    public byte[] GenerateGlb(IV rgba, int width, int height, Trellis2Options? options = null)
        => Task.Run(() => GenerateGlbAsync(rgba, width, height, options)).GetAwaiter().GetResult();

    /// <summary>Generate from a device image, browser-safe. <paramref name="rgba"/> [width*height] packed RGBA (alpha = foreground).</summary>
    public async Task<byte[]> GenerateGlbAsync(IV rgba, int width, int height, Trellis2Options? options = null, CancellationToken ct = default)
    {
        options ??= new Trellis2Options();
        var (output, pipe) = await GenerateAsync(rgba, width, height, options, ct);
        pipe.RepairMesh = options.RepairMesh;
        pipe.Remesh = options.Remesh;
        using (output) return await pipe.ToGlbAsync(output, options.TextureSize, maxFaces: options.MaxFaces);
    }

    /// <summary>
    /// Re-texture a region of a generated asset from a new picture (see <see cref="Trellis2Pipeline.RetextureRegionAsync"/>): the picture is prepared and its DINOv3 features
    /// extracted like for a generation, the flow runs again with the tokens outside the region pinned to the asset's own texture latent. Returns the edited output (the caller disposes it;
    /// <paramref name="source"/> is left alone) and the pipeline to build a GLB from it with.
    /// </summary>
    public async Task<(Trellis2Pipeline.Output Output, Trellis2Pipeline Pipeline)> EditRegionAsync(Trellis2Pipeline.Output source, IV rgba, int width, int height,
        Func<float, float, float, bool> inRegion, int seed = 1, CancellationToken ct = default)
    {
        var st = source.Edit ?? throw new InvalidOperationException("This output carries no latents to edit.");
        var sw = Stopwatch.StartNew();
        _precompile ??= Task.WhenAll(_dk.PrecompileAsync(), _sk.PrecompileAsync());
        var pipeline = await Trellis2Pipeline.CreateAsync(_acc, _weights, ct);
        pipeline.StageDone = StageDone;
        using var image = await new Trellis2ImagePrep(_acc).PrepareAsync(rgba, width, height, st.Res);
        int tokens = 1 + 4 + (st.Res / 16) * (st.Res / 16);
        using var cond = _acc.Allocate1D<float>((long)tokens * 1024);
        {
            using var w = await _weights.LoadDinoAsync(_acc, ct);
            await _precompile;
            using var dino = new DinoV3Extractor(_acc, _sk, _dk, w);
            dino.Extract(image.View, st.Res, cond.View);
        }
        StageDone?.Invoke("edit: dinov3 features", sw.Elapsed.TotalSeconds);
        var output = await pipeline.RetextureRegionAsync(source, cond.View, tokens, inRegion, seed, ct);
        return (output, pipeline);
    }

    /// <summary>
    /// Edit the SHAPE inside a region of a generated asset from a new picture (see <see cref="Trellis2Pipeline.EditShapeRegionAsync"/>): geometry can be added or removed inside the region,
    /// everything outside keeps its shape and texture. Returns the edited output (the caller disposes it; <paramref name="source"/> is left alone) and the pipeline to build a GLB from it with.
    /// </summary>
    public async Task<(Trellis2Pipeline.Output Output, Trellis2Pipeline Pipeline)> EditShapeRegionAsync(Trellis2Pipeline.Output source, IV rgba, int width, int height,
        Func<float, float, float, bool> inRegion, int seed = 1, CancellationToken ct = default)
    {
        var st = source.Edit ?? throw new InvalidOperationException("This output carries no latents to edit.");
        var sw = Stopwatch.StartNew();
        _precompile ??= Task.WhenAll(_dk.PrecompileAsync(), _sk.PrecompileAsync());
        var pipeline = await Trellis2Pipeline.CreateAsync(_acc, _weights, ct);
        pipeline.StageDone = StageDone;
        const int tokens512 = 1 + 4 + 32 * 32;
        int tokens = 1 + 4 + (st.Res / 16) * (st.Res / 16);
        using var ssCond = _acc.Allocate1D<float>((long)tokens512 * 1024);
        MemoryBuffer1D<float, Stride1D.Dense>? cond1024 = null;
        try
        {
            using var w = await _weights.LoadDinoAsync(_acc, ct);
            await _precompile;
            using var dino = new DinoV3Extractor(_acc, _sk, _dk, w);
            using (var image = await new Trellis2ImagePrep(_acc).PrepareAsync(rgba, width, height, 512)) dino.Extract(image.View, 512, ssCond.View);
            if (st.Res == 1024)
            {
                using var image1024 = await new Trellis2ImagePrep(_acc).PrepareAsync(rgba, width, height, 1024);
                cond1024 = _acc.Allocate1D<float>((long)tokens * 1024);
                dino.Extract(image1024.View, 1024, cond1024.View);
            }
            StageDone?.Invoke("edit shape: dinov3 features", sw.Elapsed.TotalSeconds);
            var output = await pipeline.EditShapeRegionAsync(source, ssCond.View, tokens512, cond1024?.View ?? ssCond.View, tokens, inRegion, seed, ct);
            return (output, pipeline);
        }
        finally { cond1024?.Dispose(); }
    }

    /// <summary>Desktop wrapper of <see cref="GenerateAsync"/> (blocks).</summary>
    public Trellis2Pipeline.Output Generate(IV rgba, int width, int height, Trellis2Options? options, out Trellis2Pipeline pipeline)
    {
        var (o, p) = Task.Run(() => GenerateAsync(rgba, width, height, options)).GetAwaiter().GetResult();
        pipeline = p;
        return o;
    }

    /// <summary>
    /// Run the networks and return the device-resident result (mesh + per-voxel PBR attributes) without building the GLB; the caller disposes it.
    /// <c>Pipeline</c> is the instance to call <see cref="Trellis2Pipeline.ToGlbAsync"/> on.
    /// </summary>
    public async Task<(Trellis2Pipeline.Output Output, Trellis2Pipeline Pipeline)> GenerateAsync(IV rgba, int width, int height, Trellis2Options? options, CancellationToken ct = default)
    {
        options ??= new Trellis2Options();
        if (options.Resolution != 512 && options.Resolution != 1024 && options.Resolution != 1536) throw new ArgumentException("Resolution must be 512, 1024 or 1536");
        if (options.Resolution == 1536 && !options.Cascade) throw new ArgumentException("Resolution 1536 is the 1536_cascade pipeline: set Cascade");
        if (options.Cascade && options.Resolution == 512) throw new ArgumentException("A cascade runs at 1024 or 1536");
        var sw = Stopwatch.StartNew();
        // WebGPU: the large kernels' pipelines build in the background, in parallel, while the image is prepared and DINOv3 loads
        _precompile ??= Task.WhenAll(_dk.PrecompileAsync(), _sk.PrecompileAsync());
        var pipeline = await Trellis2Pipeline.CreateAsync(_acc, _weights, ct);
        pipeline.StageDone = StageDone;
        pipeline.PrefetchFirstStage(includeDino: true);      // the reads overlap the image preparation and the feature extraction
        using var image = await new Trellis2ImagePrep(_acc).PrepareAsync(rgba, width, height, 512);
        StageDone?.Invoke("image prep", sw.Elapsed.TotalSeconds); sw.Restart();

        const int tokens = 1 + 4 + 32 * 32;
        using var cond = _acc.Allocate1D<float>((long)tokens * 1024);
        MemoryBuffer1D<float, Stride1D.Dense>? cond1024 = null;
        const int tokens1024 = 1 + 4 + 64 * 64;
        try
        {
            {   // the feature extractor's weights are released before the flows load
                using var w = await _weights.LoadDinoAsync(_acc, ct);
                await _precompile;                           // a launch while its pipeline is still building would compile it a second time
                using var dino = new DinoV3Extractor(_acc, _sk, _dk, w);
                dino.Extract(image.View, 512, cond.View);
                if (options.Resolution >= 1024)
                {
                    // the same preprocessed image resampled to 1024 px for the shape / texture flows
                    using var image1024 = await new Trellis2ImagePrep(_acc).PrepareAsync(rgba, width, height, 1024);
                    cond1024 = _acc.Allocate1D<float>((long)tokens1024 * 1024);
                    dino.Extract(image1024.View, 1024, cond1024.View);
                    var dumpCond = Environment.GetEnvironmentVariable("TRELLIS2_DUMP_COND") ?? "";
                    if (dumpCond.Length > 0)
                    {   // debug: the normalised 1024 px image and its features, to compare against timm on the same tensor (reference/golden_dino_dump.py)
                        Directory.CreateDirectory(dumpCond);
                        async Task Dump(string name, ArrayView1D<float, Stride1D.Dense> v) { var a = await v.CopyToHostAsync(); var b = new byte[a.Length * 4]; Buffer.BlockCopy(a, 0, b, 0, b.Length); File.WriteAllBytes(Path.Combine(dumpCond, name), b); }
                        await Dump("image1024.f32", image1024.View); await Dump("cond1024.f32", cond1024.View);
                    }
                }
            }
            StageDone?.Invoke("dinov3 features", sw.Elapsed.TotalSeconds);

            var output = options.Resolution == 512
                ? await pipeline.Run512Async(cond.View, tokens, options.Seed, ct)
                : options.Cascade
                ? await pipeline.RunCascadeAsync(options.Resolution, cond.View, tokens, cond1024!.View, tokens1024, options.Seed, ct: ct)
                : (Environment.GetEnvironmentVariable("TRELLIS2_EXP_COND512") == "1"
                    ? await pipeline.Run1024Async(cond.View, tokens, cond.View, tokens, options.Seed, ct)
                    : await pipeline.Run1024Async(cond.View, tokens, cond1024!.View, tokens1024, options.Seed, ct));
            return (output, pipeline);
        }
        finally { cond1024?.Dispose(); }
    }

    /// <summary>Managed pixels (desktop / oracle path).</summary>
    public byte[] GenerateGlb(int[] rgbaPixels, int width, int height, Trellis2Options? options = null)
    {
        using var buf = RgbaUpload.FromManaged(_acc, rgbaPixels, width, height);
        return GenerateGlb(buf.View, width, height, options);
    }

    /// <summary>Managed pixels, browser-safe.</summary>
    public async Task<byte[]> GenerateGlbAsync(int[] rgbaPixels, int width, int height, Trellis2Options? options = null, CancellationToken ct = default)
    {
        using var buf = RgbaUpload.FromManaged(_acc, rgbaPixels, width, height);
        return await GenerateGlbAsync(buf.View, width, height, options, ct);
    }

    /// <summary>Browser path: RGBA bytes still in a JS typed array; they never enter the .NET heap.</summary>
    public async Task<byte[]> GenerateGlbAsync(TypedArray rgbaBytes, int width, int height, Trellis2Options? options = null, CancellationToken ct = default)
    {
        using var buf = RgbaUpload.FromTypedArray(_acc, rgbaBytes, width, height);
        return await GenerateGlbAsync(buf.View, width, height, options, ct);
    }

    public void Dispose() { }
}
