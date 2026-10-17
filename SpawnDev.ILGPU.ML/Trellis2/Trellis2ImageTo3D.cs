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
        return pipe.ToGlb(output, options.TextureSize);
    }

    /// <summary>
    /// Run the networks and return the device-resident result (mesh + per-voxel PBR attributes) without building the GLB; the caller disposes it.
    /// <paramref name="pipeline"/> is the pipeline instance to call <see cref="Trellis2Pipeline.ToGlb"/> on.
    /// </summary>
    public Trellis2Pipeline.Output Generate(IV rgba, int width, int height, Trellis2Options? options, out Trellis2Pipeline pipeline)
    {
        options ??= new Trellis2Options();
        var sw = Stopwatch.StartNew();
        using var image = new Trellis2ImagePrep(_acc).Prepare(rgba, width, height, 512);
        StageDone?.Invoke("image prep", sw.Elapsed.TotalSeconds); sw.Restart();

        const int tokens = 1 + 4 + 32 * 32;
        using var cond = _acc.Allocate1D<float>((long)tokens * 1024);
        {
            using var w = GpuWeights.LoadFile(_acc, _paths.DinoWeights);
            using var dino = new DinoV3Extractor(_acc, _sk, _dk, w);
            dino.Extract(image.View, 512, cond.View);
        }
        StageDone?.Invoke("dinov3 features", sw.Elapsed.TotalSeconds);

        pipeline = new Trellis2Pipeline(_acc, _paths.ModelDir, _paths.SparseStructureDecoder) { StageDone = StageDone };
        return pipeline.Run512(cond.View, tokens, options.Seed);
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
