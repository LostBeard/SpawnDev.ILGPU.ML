using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;

namespace SpawnDev.ILGPU.ML.Assets;

/// <summary>
/// How well a picture matches the words it was drawn from, with CLIP ViT-B/32 (local, 600 MB, a fraction of a second per picture): the cosine similarity of the picture's embedding and the text's.
/// Used to choose the best of several drawn pictures and to notice a picture that is not of the thing asked for (a blob, the wrong object) before the minutes-long 3D stage is spent on it.
/// CLIP ranks well and is not a fine judge: it separates "a barrel" from "a chair", not "bellows" from "bell". Typical matched pairs score 0.25-0.35, unrelated ones 0.10-0.20.
/// </summary>
public sealed class PromptMatchScorer : IDisposable
{
    private readonly ZeroShotClassificationPipeline _clip;

    private PromptMatchScorer(ZeroShotClassificationPipeline clip) => _clip = clip;

    /// <summary>Load CLIP ViT-B/32 (vision and text towers and the tokenizer) from <paramref name="source"/>; cached after the first download.</summary>
    public static async Task<PromptMatchScorer> CreateAsync(Accelerator accelerator, IModelSource source)
    {
        await using var vs = await source.OpenAsync(ModelHub.KnownModels.CLIPVitB32, ModelHub.KnownFiles.OnnxVisionModel);
        var vision = await InferenceSession.CreateFromStreamAsync(accelerator, vs);
        await using var ts = await source.OpenAsync(ModelHub.KnownModels.CLIPVitB32, ModelHub.KnownFiles.OnnxTextModel);
        var text = await InferenceSession.CreateFromStreamAsync(accelerator, ts);
        await using var tok = await source.OpenAsync(ModelHub.KnownModels.CLIPVitB32, "tokenizer.json");
        using var reader = new StreamReader(tok);
        var clip = new ZeroShotClassificationPipeline(vision, text, accelerator);
        clip.LoadTokenizer(await reader.ReadToEndAsync());
        return new PromptMatchScorer(clip);
    }

    /// <summary>Cosine similarity (-1..1) of the picture (RGBA bytes, <paramref name="width"/> x <paramref name="height"/>) and the description.</summary>
    public async Task<float> ScoreAsync(byte[] rgba, int width, int height, string description)
    {
        var px = new int[width * height];
        Buffer.BlockCopy(rgba, 0, px, 0, px.Length * 4);
        for (int i = 0; i < px.Length; i++) px[i] |= unchecked((int)0xFF000000);       // a drawing is opaque; alpha must not darken the embedding
        var image = await _clip.EncodeImageAsync(px, width, height);
        var text = await _clip.EncodeTextAsync(description);
        return Cosine(image, text);
    }

    /// <summary>Score a drawn picture against the words (the user's prompt, not the prompt with its studio-lighting suffix).</summary>
    public Task<float> ScoreAsync(AssetPicture picture, string description) => ScoreAsync(picture.Rgba, picture.Width, picture.Height, description);

    /// <summary>Scores of one picture against several descriptions, and which one is closest: the check that a picture is of ITS object and not another object of the scene.</summary>
    public async Task<(float[] Scores, int Best)> RankAsync(AssetPicture picture, IReadOnlyList<string> descriptions)
    {
        var px = new int[picture.Width * picture.Height];
        Buffer.BlockCopy(picture.Rgba, 0, px, 0, px.Length * 4);
        for (int i = 0; i < px.Length; i++) px[i] |= unchecked((int)0xFF000000);
        var image = await _clip.EncodeImageAsync(px, picture.Width, picture.Height);
        var scores = new float[descriptions.Count]; int best = 0;
        for (int i = 0; i < scores.Length; i++)
        {
            scores[i] = Cosine(image, await _clip.EncodeTextAsync(descriptions[i]));
            if (scores[i] > scores[best]) best = i;
        }
        return (scores, best);
    }

    public static float Cosine(float[] a, float[] b)
    {
        if (a.Length != b.Length) throw new ArgumentException("The embeddings differ in length.");
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return na <= 0 || nb <= 0 ? 0f : (float)(dot / Math.Sqrt(na * nb));
    }

    /// <summary>Frees both CLIP towers (the pipeline owns the sessions).</summary>
    public void Dispose() => _clip.Dispose();
}
