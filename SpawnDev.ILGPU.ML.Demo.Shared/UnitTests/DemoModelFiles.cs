using SpawnDev.ILGPU.ML.Hub;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// The demo model files that are too large for GitHub (over its 100 MB file limit, or kept out of git with them): the
/// <c>wwwroot/models/</c> folders listed in <c>.gitignore</c> (gpt2, distilbert-sst2, whisper-tiny, whisper-base.en,
/// depth-anything-v2-small). They are NOT in the repo, so a fresh clone, the GitHub Pages demo and every PMT lane on another
/// machine would get a 404 for <c>models/gpt2/model.onnx</c>. They live on the SpawnDev models host instead, byte-identical to
/// the files the tests and <c>wwwroot/references/</c> were generated against (gpt2, distilbert-sst2 and whisper-tiny are our
/// own ONNX exports, not on Hugging Face at any revision), as one repo with the same folder layout.
/// </summary>
/// <remarks>
/// 🔴 Never reference one of these folders by a relative <c>models/...</c> path again: use <see cref="Url"/>. A relative path
/// passes on a PC that happens to hold the ignored files and fails everywhere else (GitHub issue #2: tests failing on a fresh
/// machine). Small models that ARE committed (yolov8n, squeezenet, mobilenetv2, ...) keep their relative paths. A new large
/// model: upload it with <c>models.spawndev.com\upload-folder.bat</c> into this repo (or mirror its Hugging Face repo) first.
/// </remarks>
public static class DemoModelFiles
{
    /// <summary>The models-host repo holding the demo's large model folders.</summary>
    public const string RepoId = "LostBeard/spawndev-ilgpu-ml-demo-models";

    /// <summary>
    /// The URL of <paramref name="file"/> (the path the file had under the demo's <c>wwwroot/models/</c>, e.g.
    /// <c>gpt2/model.onnx</c>) on the models host.
    /// </summary>
    public static string Url(string file) => ModelsHost.GetUrl(RepoId, file);

    /// <summary>
    /// The URL of a test reference file that is too large for git (the path it had under the demo's <c>wwwroot/test-refs/</c>, e.g.
    /// <c>dav3/manifest.json</c>), stored in the same repo under <c>test-refs/</c>. Same rule as <see cref="Url"/>: a gate whose reference
    /// is gitignored degrades to a skip or a failure on every machine but the one that generated it.
    /// </summary>
    public static string TestRefUrl(string path) => ModelsHost.GetUrl(RepoId, "test-refs/" + path);
}
