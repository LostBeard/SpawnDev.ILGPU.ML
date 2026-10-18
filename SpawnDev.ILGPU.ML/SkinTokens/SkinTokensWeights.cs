using System.Text.Json;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.SkinTokens;

/// <summary>
/// Where the SkinTokens weights come from. The checkpoint is the published VAST-AI/SkinTokens grpo_1400 model (MIT) converted by
/// tools/unirig-harness/reference/export_weights.py skintokens into one 1.15 GB bf16 safetensors file; host it in any repo the hub can serve and pass that repo here.
/// </summary>
public static class SkinTokensWeights
{
    /// <summary>The file name export_weights.py writes.</summary>
    public const string FileName = "skintokens.safetensors";

    /// <summary>The file a run reads, as (repo, path): for storage planning (<see cref="RangeStreamModelSource.PlanCacheAsync"/>).</summary>
    public static IReadOnlyList<(string Repo, string Path)> HubWeightFiles(string repo) => new[] { (repo, FileName) };

    /// <summary>Stream the weights from a model source (browser OPFS / hub, or HttpClient + a file store on desktop) onto the device.</summary>
    public static async Task<GpuWeights> LoadAsync(Accelerator accelerator, IModelSource source, string repo, CancellationToken cancellationToken = default)
    {
        await using var stream = await source.OpenAsync(repo, FileName, cancellationToken);
        using var empty = JsonDocument.Parse("{}");
        return await GpuWeights.LoadAsync(accelerator, stream, empty.RootElement, $"{repo}/{FileName}", cancellationToken: cancellationToken);
    }

    /// <summary>The same from any readable stream (a test, a local server).</summary>
    public static async Task<GpuWeights> LoadAsync(Accelerator accelerator, Stream stream, CancellationToken cancellationToken = default)
    {
        using var empty = JsonDocument.Parse("{}");
        return await GpuWeights.LoadAsync(accelerator, stream, empty.RootElement, FileName, cancellationToken: cancellationToken);
    }
}
