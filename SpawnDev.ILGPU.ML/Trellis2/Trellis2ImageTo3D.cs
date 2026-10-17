using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
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
    /// 1024 flow checkpoints next to the 512 ones). The sparse structure comes from the 512 px features either way.
    /// </summary>
    public int Resolution { get; init; } = 512;
    /// <summary>The mesh is simplified to about this many triangles before the UV unwrap (0 keeps the full dual-grid mesh, 1-2.5M triangles). Measured on the official crown: 120k looks the same as 300k; at 40-50k thin sheets shred and texels go missing.</summary>
    public int MaxFaces { get; init; } = 150_000;
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
    private readonly Trellis2Paths _paths;
    private readonly DiTKernels _dk;
    private readonly SparseKernels _sk;

    /// <summary>Per-stage progress: (stage, seconds).</summary>
    public Action<string, double>? StageDone { get; set; }

    public Trellis2ImageTo3D(Accelerator accelerator, Trellis2Paths paths)
    {
        _acc = accelerator; _paths = paths;
        _dk = new DiTKernels(accelerator); _sk = new SparseKernels(accelerator);
    }

    /// <summary>Generate from a device image. <paramref name="rgba"/> [width*height] packed RGBA.</summary>
    public byte[] GenerateGlb(IV rgba, int width, int height, Trellis2Options? options = null)
    {
        options ??= new Trellis2Options();
        using var output = Generate(rgba, width, height, options, out var pipe);
        return pipe.ToGlb(output, options.TextureSize, maxFaces: options.MaxFaces);
    }

    /// <summary>
    /// Run the networks and return the device-resident result (mesh + per-voxel PBR attributes) without building the GLB; the caller disposes it.
    /// <paramref name="pipeline"/> is the pipeline instance to call <see cref="Trellis2Pipeline.ToGlb"/> on.
    /// </summary>
    public Trellis2Pipeline.Output Generate(IV rgba, int width, int height, Trellis2Options? options, out Trellis2Pipeline pipeline)
    {
        options ??= new Trellis2Options();
        var sw = Stopwatch.StartNew();
        pipeline = new Trellis2Pipeline(_acc, _paths.ModelDir, _paths.SparseStructureDecoder) { StageDone = StageDone };
        pipeline.PrefetchWeights(_paths.DinoWeights);       // the disk reads overlap the image preparation and the feature extraction
        pipeline.PrefetchWeights();                          // then the first flow checkpoint
        using var image = new Trellis2ImagePrep(_acc).Prepare(rgba, width, height, 512);
        StageDone?.Invoke("image prep", sw.Elapsed.TotalSeconds); sw.Restart();

        if (options.Resolution != 512 && options.Resolution != 1024) throw new ArgumentException("Resolution must be 512 or 1024");
        const int tokens = 1 + 4 + 32 * 32;
        using var cond = _acc.Allocate1D<float>((long)tokens * 1024);
        MemoryBuffer1D<float, Stride1D.Dense>? cond1024 = null;
        const int tokens1024 = 1 + 4 + 64 * 64;
        try
        {
            {   // the feature extractor's weights are released before the flows load
                using var w = GpuWeights.LoadFile(_acc, _paths.DinoWeights);
                using var dino = new DinoV3Extractor(_acc, _sk, _dk, w);
                dino.Extract(image.View, 512, cond.View);
                if (options.Resolution == 1024)
                {
                    // the same preprocessed image resampled to 1024 px for the shape / texture flows
                    using var image1024 = new Trellis2ImagePrep(_acc).Prepare(rgba, width, height, 1024);
                    cond1024 = _acc.Allocate1D<float>((long)tokens1024 * 1024);
                    dino.Extract(image1024.View, 1024, cond1024.View);
                    var dumpCond = Environment.GetEnvironmentVariable("TRELLIS2_DUMP_COND") ?? "";
                    if (dumpCond.Length > 0)
                    {   // debug: the normalised 1024 px image and its features, to compare against timm on the same tensor (reference/golden_dino_timm.py)
                        Directory.CreateDirectory(dumpCond); _acc.Synchronize();
                        void Dump(string name, ArrayView1D<float, Stride1D.Dense> v) { var a = v.GetAsArray1D(); var b = new byte[a.Length * 4]; Buffer.BlockCopy(a, 0, b, 0, b.Length); File.WriteAllBytes(Path.Combine(dumpCond, name), b); }
                        Dump("image1024.f32", image1024.View); Dump("cond1024.f32", cond1024.View);
                    }
                }
            }
            StageDone?.Invoke("dinov3 features", sw.Elapsed.TotalSeconds);

            return options.Resolution == 512
                ? pipeline.Run512(cond.View, tokens, options.Seed)
                : pipeline.Run1024(cond.View, tokens, cond1024!.View, tokens1024, options.Seed);
        }
        finally { cond1024?.Dispose(); }
    }

    /// <summary>Managed pixels (desktop / oracle path).</summary>
    public byte[] GenerateGlb(int[] rgbaPixels, int width, int height, Trellis2Options? options = null)
    {
        using var buf = RgbaUpload.FromManaged(_acc, rgbaPixels, width, height);
        return GenerateGlb(buf.View, width, height, options);
    }

    /// <summary>Browser path: RGBA bytes still in a JS typed array; they never enter the .NET heap.</summary>
    public byte[] GenerateGlb(TypedArray rgbaBytes, int width, int height, Trellis2Options? options = null)
    {
        using var buf = RgbaUpload.FromTypedArray(_acc, rgbaBytes, width, height);
        return GenerateGlb(buf.View, width, height, options);
    }

    /// <summary>Asynchronous wrapper for UI callers (desktop backends run the work on a pool thread).</summary>
    public Task<byte[]> GenerateGlbAsync(int[] rgbaPixels, int width, int height, Trellis2Options? options = null)
        => Task.Run(() => GenerateGlb(rgbaPixels, width, height, options));

    public void Dispose() { }
}
