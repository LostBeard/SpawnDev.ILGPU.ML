using SpawnDev.ILGPU.ML.Assets;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// <see cref="ImageGenerator"/> (the pipeline of the browser demo's Image Generation page behind one call, used by the desktop examples) and the image-to-3D entry of <see cref="AssetGenerator"/>.
/// The model runs themselves are covered where the models are (the Flux2 / Flux3 / ImageGeneration tests); these are the decisions this layer makes without a model: the picture size a reference
/// image sets, which requests it refuses before it loads anything, and which pictures count as already cut out.
/// </summary>
public abstract partial class MLTestBase
{
    private static Flux2ReferenceImage SolidReference(int w, int h) => new(w, h, new byte[w * h * 4]);

    /// <summary>The exception <paramref name="work"/> throws (of type <typeparamref name="T"/>), or null when it does not throw one.</summary>
    private static async Task<T?> Ig_Throws<T>(Func<Task> work) where T : Exception
    {
        try { await work(); return null; }
        catch (T ex) { return ex; }
    }

    /// <summary>The picture size is a square of the chosen side, or with reference images the FIRST reference's proportions with that longer side, in multiples of 16 (the page's own formula).</summary>
    [TestMethod(Timeout = 30000)]
    public async Task ImageGenerator_PictureSizeFollowsTheFirstReference() => await RunTest(async accelerator =>
    {
        Ur_Expect(ImageGenerator.PictureSize(768, null) == (768, 768), "no references: a square");
        Ur_Expect(ImageGenerator.PictureSize(512, Array.Empty<Flux2ReferenceImage>()) == (512, 512), "an empty reference list is no references");
        var wide = ImageGenerator.PictureSize(768, new[] { SolidReference(1600, 900), SolidReference(100, 100) });
        Ur_Expect(wide == (768, 432), $"a 16:9 first reference gives 768 x 432, got {wide}");
        var tall = ImageGenerator.PictureSize(768, new[] { SolidReference(900, 1600) });
        Ur_Expect(tall == (432, 768), $"a 9:16 first reference gives 432 x 768, got {tall}");
        var thin = ImageGenerator.PictureSize(512, new[] { SolidReference(4000, 20) });
        Ur_Expect(thin.Width == 512 && thin.Height == 64, $"a very wide reference keeps at least 64 px, got {thin}");
        Ur_Expect(wide.Width % 16 == 0 && wide.Height % 16 == 0 && tall.Width % 16 == 0 && tall.Height % 16 == 0, "multiples of 16");
        await Task.CompletedTask;
    });

    /// <summary>A request that cannot work is refused before any model is loaded or downloaded: an empty prompt, and reference images for SD-Turbo (it cannot take them).</summary>
    [TestMethod(Timeout = 30000)]
    public async Task ImageGenerator_RefusesWhatCannotWorkBeforeLoadingAnything() => await RunTest(async accelerator =>
    {
        using var generator = new ImageGenerator(accelerator, new RefusingModelSource());
        bool loadedAnything = false;
        generator.Status = _ => loadedAnything = true;

        var empty = await Ig_Throws<ArgumentException>(() => generator.GenerateAsync(new ImageGenerationRequest { Prompt = "  ", Engine = ImageEngine.Flux2 }));
        Ur_Expect(empty != null, "an empty prompt is refused");
        var sd = await Ig_Throws<ArgumentException>(() => generator.GenerateAsync(new ImageGenerationRequest { Prompt = "a cat", Engine = ImageEngine.SdTurbo, References = new[] { SolidReference(64, 64) } }));
        Ur_Expect(sd != null, "SD-Turbo with reference images is refused");
        Ur_Expect(!loadedAnything, "nothing was loaded for a refused request");
    });

    /// <summary>A picture is "already cut out" when any pixel is not fully opaque: an opaque photo is cut out by RMBG first, a picture with a transparent background goes to the 3D stage as it is.</summary>
    [TestMethod(Timeout = 30000)]
    public async Task AssetGenerator_HasTransparencyMeansAnyPixelBelowFullAlpha() => await RunTest(async accelerator =>
    {
        var opaque = new byte[16 * 16 * 4]; for (int i = 3; i < opaque.Length; i += 4) opaque[i] = 255;
        Ur_Expect(!AssetGenerator.HasTransparency(opaque), "all alpha 255: opaque");
        var oneHole = (byte[])opaque.Clone(); oneHole[^1] = 254;
        Ur_Expect(AssetGenerator.HasTransparency(oneHole), "the last pixel's alpha 254: transparent");
        var firstHole = (byte[])opaque.Clone(); firstHole[3] = 0;
        Ur_Expect(AssetGenerator.HasTransparency(firstHole), "the first pixel's alpha 0: transparent");
        await Task.CompletedTask;
    });

    /// <summary>A source that fails the test if a model is asked for.</summary>
    private sealed class RefusingModelSource : IModelSource
    {
        public Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default) => throw new InvalidOperationException($"a model was requested: {repoId}/{filePath}");
        public Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken = default) => throw new InvalidOperationException($"a model was requested: {repoId}/{filePath}");
    }
}
