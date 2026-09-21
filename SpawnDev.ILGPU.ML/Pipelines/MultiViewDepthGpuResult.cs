using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// GPU-resident multi-view depth result from a joint DAv3 (or similar) forward.
/// Caller owns all buffers and must <see cref="Dispose"/>.
/// </summary>
public sealed class MultiViewDepthGpuResult : IDisposable
{
    /// <summary>Per-view depth maps at the requested output resolution (GPU-resident).</summary>
    public required IReadOnlyList<(MemoryBuffer1D<float, Stride1D.Dense> RawDepth, float MinDepth, float MaxDepth, int Width, int Height)> Views { get; set; }

    /// <summary>
    /// Optional per-view confidence maps (same W×H as depth). Null when the model has no confidence output.
    /// </summary>
    public IReadOnlyList<MemoryBuffer1D<float, Stride1D.Dense>>? ConfidenceMaps { get; set; }

    /// <summary>
    /// Optional per-view 3×4 extrinsics [R|t] as length-12 row-major float arrays (one per view).
    /// Null when the model did not emit usable extrinsics.
    /// </summary>
    public float[][]? Extrinsics { get; init; }

    /// <summary>
    /// Optional per-view 3×3 intrinsics as length-9 row-major float arrays (one per view).
    /// </summary>
    public float[][]? Intrinsics { get; init; }

    public int ViewCount => Views.Count;

    /// <summary>
    /// Relinquish ownership of depth view buffers so the caller can dispose them separately.
    /// Confidence maps (if any) are still disposed by <see cref="Dispose"/> unless
    /// <see cref="DetachConfidenceMaps"/> is also called.
    /// </summary>
    public void DetachDepthViews()
    {
        Views = Array.Empty<(MemoryBuffer1D<float, Stride1D.Dense>, float, float, int, int)>();
    }

    /// <summary>
    /// Relinquish ownership of confidence map buffers so the caller can dispose them separately.
    /// </summary>
    public void DetachConfidenceMaps()
    {
        ConfidenceMaps = null;
    }

    public void Dispose()
    {
        foreach (var (raw, _, _, _, _) in Views)
            raw.Dispose();
        if (ConfidenceMaps != null)
        {
            foreach (var c in ConfidenceMaps)
                c.Dispose();
        }
        Views = Array.Empty<(MemoryBuffer1D<float, Stride1D.Dense>, float, float, int, int)>();
        ConfidenceMaps = null;
        GC.SuppressFinalize(this);
    }
}
