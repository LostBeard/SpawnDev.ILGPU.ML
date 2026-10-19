using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Assets;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// <see cref="PromptMatchScorer"/>: CLIP ViT-B/32 ranks a picture's own description above other descriptions. The cosine is checked on its own; the model part runs on CUDA with the
/// demo's cat and portrait samples (CLIP weights are downloaded to the model cache on the first run: HeavyModel).
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task PromptMatchScorer_Cosine_IsScaleFreeAndBounded() => await RunPureTest(async () =>
    {
        var a = new[] { 1f, 2f, 3f };
        Te_Expect(Math.Abs(PromptMatchScorer.Cosine(a, new[] { 2f, 4f, 6f }) - 1f) < 1e-6f, "parallel vectors: 1 whatever their length");
        Te_Expect(Math.Abs(PromptMatchScorer.Cosine(new[] { 1f, 0f }, new[] { 0f, 5f })) < 1e-6f, "orthogonal vectors: 0");
        Te_Expect(Math.Abs(PromptMatchScorer.Cosine(a, new[] { -1f, -2f, -3f }) + 1f) < 1e-6f, "opposite vectors: -1");
        Te_Expect(PromptMatchScorer.Cosine(new[] { 0f, 0f }, new[] { 1f, 1f }) == 0f, "a zero vector scores 0, not NaN");
        await Task.CompletedTask;
    });

    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task PromptMatchScorer_RanksAPicturesOwnDescriptionFirst() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.Cuda) throw new UnsupportedTestException("the scorer test runs on CUDA");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available for this backend");
        async Task<AssetPicture> Sample(string name)
        {
            var bin = await http.GetByteArrayAsync("samples/" + name);
            int w = BitConverter.ToInt32(bin, 0), h = BitConverter.ToInt32(bin, 4);
            var rgba = new byte[w * h * 4]; Buffer.BlockCopy(bin, 8, rgba, 0, rgba.Length);
            return new AssetPicture(0, w, h, rgba);
        }
        var cat = await Sample("cat_rgba.bin"); var portrait = await Sample("portrait_rgba.bin");
        using var web = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        var cache = Environment.GetEnvironmentVariable("SPAWNDEV_MODEL_CACHE");
        var store = string.IsNullOrEmpty(cache) ? FileModelStore.Default() : new FileModelStore(cache);
        using var scorer = await PromptMatchScorer.CreateAsync(accelerator, new HttpClientModelSource(web, store));
        var words = new[] { "a photo of a cat", "a portrait photo of a person", "a wooden barrel", "a landscape with mountains" };
        var (catScores, catBest) = await scorer.RankAsync(cat, words);
        var (personScores, personBest) = await scorer.RankAsync(portrait, words);
        Te_Expect(catBest == 0, $"the cat is closest to 'a photo of a cat' ({string.Join(", ", catScores.Select(x => x.ToString("F3")))})");
        Te_Expect(personBest == 1, $"the portrait is closest to 'a portrait photo of a person' ({string.Join(", ", personScores.Select(x => x.ToString("F3")))})");
        Te_Expect(catScores[0] - catScores.Skip(1).Max() > 0.03f && personScores[1] - personScores.Where((_, i) => i != 1).Max() > 0.03f, "with a clear margin");
        Te_Expect(catScores[0] > 0.2f && catScores[0] < 0.5f, $"a matched pair scores like CLIP's usual range ({catScores[0]:F3})");
        Console.WriteLine($"[PromptMatch] cat {string.Join(", ", catScores.Select(x => x.ToString("F3")))}; person {string.Join(", ", personScores.Select(x => x.ToString("F3")))}");
        await Task.CompletedTask;
    });
}
