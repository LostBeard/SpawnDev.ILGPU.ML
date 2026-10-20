using SpawnDev.ILGPU.ML.Flux3;
using SpawnDev.ILGPU.ML.Hub;

namespace SpawnDev.ILGPU.ML.Demo.Services;

/// <summary>
/// The browser demo's door to <see cref="Flux3CaptionWriter"/> (the library class, shared with the desktop examples): the full scene caption FLUX 3 Image Klein draws from, written by Qwen3 4B
/// from the demo's <see cref="HubModelSource"/>. Used by the Image Generation and Text to 3D pages.
/// </summary>
public static class CaptionWriter
{
    public const string Repo = Flux3CaptionWriter.Repo, File = Flux3CaptionWriter.File;
    public const string Label = Flux3CaptionWriter.Label, SizeText = Flux3CaptionWriter.SizeText;
    public const int CaptionTextIntervalMs = Flux3CaptionWriter.CaptionTextIntervalMs;

    /// <inheritdoc cref="Flux3CaptionWriter.WriteAsync"/>
    public static Task<string> WriteAsync(global::ILGPU.Runtime.Accelerator accelerator, HubModelSource source, string request, int width, int height, Action<string> status,
        Action<double>? progress = null, Action<string>? captionText = null)
        => Flux3CaptionWriter.WriteAsync(accelerator, (repo, file, download) => source.OpenAsync(repo, file, download), request, width, height, status, progress, captionText);
}
