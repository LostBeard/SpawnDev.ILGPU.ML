using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;

namespace SpawnDev.ILGPU.ML.Trellis2;

/// <summary>
/// Text to 3D: <see cref="ImageGenerationPipeline"/> (SD Turbo, the model the ML and SpawnDev.AI demos use) draws the object on a plain background,
/// <see cref="ForegroundMask"/> cuts it out, <see cref="Trellis2ImageTo3D"/> lifts the cut-out to a PBR GLB. The image model stays loaded between calls
/// by default (about 2.5 GB beside the ~3 GB the 3D stages need at their peak, well inside a 12 GB card); set <see cref="KeepImageModelLoaded"/> false
/// to release it after every image.
/// </summary>
public sealed class Trellis2TextTo3D : IDisposable
{
    private readonly Accelerator _acc;
    private readonly Trellis2ImageTo3D _to3D;
    private readonly IModelSource _imageModels;
    private readonly string _maskOnnx;
    private ImageGenerationPipeline? _image;

    /// <summary>Appended to every prompt: asks for one whole object on a plain background, which is what the image-to-3D model was trained on.</summary>
    public string PromptSuffix { get; set; } = ", a single object, centered, the whole object in frame with an empty margin around it, isolated on a plain white background, soft studio lighting, high detail";
    /// <summary>
    /// Image seeds to try when the cut-out touches the frame (the drawing cropped the object, and the 3D model would rebuild a truncated one).
    /// The image whose cut-out touches the border least is kept.
    /// </summary>
    public int MaxImageAttempts { get; set; } = 4;
    /// <summary>The image seed that was used for the last result (after retries).</summary>
    public int LastImageSeed { get; private set; }
    /// <summary>Keep the image model resident between calls (default true).</summary>
    public bool KeepImageModelLoaded { get; set; } = true;
    /// <summary>Per-stage progress: (stage, seconds).</summary>
    public Action<string, double>? StageDone { get => _to3D.StageDone; set => _to3D.StageDone = value; }
    /// <summary>The last generated image (RGBA bytes, before the cut-out), for previews.</summary>
    public ImageGenerationResult? LastImage { get; private set; }

    /// <param name="imageModels">Where the SD Turbo ONNX files come from (desktop: <see cref="HttpClientModelSource"/> with a <see cref="FileModelStore"/>).</param>
    /// <param name="foregroundMaskOnnx">The salient-object model for <see cref="ForegroundMask"/> (IS-Net general use).</param>
    public Trellis2TextTo3D(Accelerator accelerator, Trellis2Paths paths, IModelSource imageModels, string foregroundMaskOnnx)
    {
        _acc = accelerator; _imageModels = imageModels; _maskOnnx = foregroundMaskOnnx;
        _to3D = new Trellis2ImageTo3D(accelerator, paths);
    }

    /// <summary>Prompt -> GLB. <paramref name="imageSeed"/> picks the drawing; <paramref name="options"/> the 3D stages (seed, resolution, faces, texture).</summary>
    public async Task<byte[]> GenerateGlbAsync(string prompt, Trellis2Options? options = null, int imageSeed = 42)
    {
        var sw = Stopwatch.StartNew();
        if (_image == null)
        {
            _image = await ImageGenerationPipeline.CreateAsync(_acc, _imageModels, ModelHub.KnownModels.SDTurbo);
            if (!_image.IsReady) throw new InvalidOperationException("the SD Turbo pipeline did not load");
            StageDone?.Invoke("image model load", sw.Elapsed.TotalSeconds); sw.Restart();
        }
        MemoryBuffer1D<int, Stride1D.Dense>? cut = null; ImageGenerationResult? img = null; double bestTouch = double.MaxValue;
        using (var mask = new ForegroundMask(_acc, _maskOnnx))
        {
            for (int attempt = 0; attempt < Math.Max(1, MaxImageAttempts); attempt++)
            {
                var candidate = await _image.RunAsync(new ImageGenerationInput { Prompt = prompt + PromptSuffix, Seed = imageSeed + attempt });
                var px = new int[candidate.Width * candidate.Height];
                Buffer.BlockCopy(candidate.ImageRGBA, 0, px, 0, px.Length * 4);
                for (int i = 0; i < px.Length; i++) px[i] |= unchecked((int)0xFF000000);   // opaque: the mask decides the cut-out
                using var rgba = RgbaUpload.FromManaged(_acc, px, candidate.Width, candidate.Height);
                var c = mask.Apply(rgba.View, candidate.Width, candidate.Height);
                double touch = await BorderTouch(c, candidate.Width, candidate.Height);
                StageDone?.Invoke($"image seed {imageSeed + attempt} (cut-out on the border {touch:P1})", sw.Elapsed.TotalSeconds); sw.Restart();
                if (touch < bestTouch) { cut?.Dispose(); cut = c; img = candidate; bestTouch = touch; LastImageSeed = imageSeed + attempt; }
                else c.Dispose();
                if (touch < 0.005) break;
            }
        }
        LastImage = img;
        if (!KeepImageModelLoaded) { await _acc.SynchronizeAsync(); _image.Dispose(); _image = null; }
        using (cut!)
            return _to3D.GenerateGlb(cut!.View, img!.Width, img.Height, options);
    }

    /// <summary>Fraction of the outermost 2-pixel ring of the image that the cut-out covers (alpha above half).</summary>
    private static async Task<double> BorderTouch(MemoryBuffer1D<int, Stride1D.Dense> cut, int w, int h)
    {
        var px = await cut.CopyToHostAsync<int>(0, (long)w * h);
        int ring = 0, hit = 0;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                if (x >= 2 && y >= 2 && x < w - 2 && y < h - 2) continue;
                ring++;
                if (((px[y * w + x] >> 24) & 255) > 127) hit++;
            }
        return ring == 0 ? 0 : (double)hit / ring;
    }

    public void Dispose()
    {
        _image?.Dispose(); _image = null;
        _to3D.Dispose();
    }
}
