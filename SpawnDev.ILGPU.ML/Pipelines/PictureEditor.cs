using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Flux3;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// "Change this picture": FLUX reference-image editing in one call. The picture goes in as the reference image and the instruction says what to change
/// ("give her a red scarf", "arms further from the body", "paint the whole chest bright red"); the picture is kept, the change is drawn as a new picture of the same size.
/// FLUX 3 Image Klein uses the instruction as its caption, FLUX.2 klein as its prompt (Qwen3 does not rewrite it: the instruction is already the whole request). Both pipelines
/// load their VAE encoder (140 MB) on the first edit. Used by the Text to 3D, Image to 3D and Image Generation demos.
/// </summary>
public static class PictureEditor
{
    /// <summary>One edit. Pass the loaded pipeline of the model that should edit (<paramref name="flux3"/> wins when both are given).</summary>
    /// <param name="rgba">The picture, 4 bytes per pixel (R, G, B, A), <paramref name="width"/> x <paramref name="height"/>.</param>
    /// <param name="progress">(stage, step, steps) as the run moves along.</param>
    public static async Task<(int Width, int Height, byte[] Rgba)> EditAsync(Flux3Pipeline? flux3, Flux2Pipeline? flux2, string instruction, int width, int height, byte[] rgba, int seed,
        Action<string, int, int>? progress = null)
    {
        if (string.IsNullOrWhiteSpace(instruction)) throw new ArgumentException("Say what to change.", nameof(instruction));
        if (flux3 == null && flux2 == null) throw new InvalidOperationException("Editing needs FLUX 3 Image Klein or FLUX.2 klein (SD Turbo has no reference images).");
        if (rgba.Length != width * height * 4) throw new ArgumentException($"the picture is {rgba.Length} bytes, {width} x {height} x 4 = {width * height * 4}", nameof(rgba));
        var refs = new[] { new Flux2ReferenceImage(width, height, rgba) };
        if (flux3 != null)
        {
            var image = await flux3.RunAsync(new Flux3Request { Caption = instruction.Trim(), Seed = seed, Width = width, Height = height, References = refs, Progress = progress });
            return (image.Width, image.Height, image.Rgba);
        }
        var img2 = await flux2!.RunAsync(new Flux2Request { Prompt = instruction.Trim(), Seed = seed, Size = Math.Max(width, height), Width = width, Height = height, References = refs, Steps = 4, Progress = progress });
        return (img2.Width, img2.Height, img2.Rgba);
    }
}
