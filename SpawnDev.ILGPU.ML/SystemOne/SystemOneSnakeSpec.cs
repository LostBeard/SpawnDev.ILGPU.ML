namespace SpawnDev.ILGPU.ML.SystemOne;

/// <summary>
/// Shared dimensions for the Classic Snake System One head.
/// The demo encoder (<c>SnakeStateEncoder</c>) must match <see cref="StateDim"/>.
/// </summary>
public static class SystemOneSnakeSpec
{
    /// <summary>
    /// Compact feature layout (40 floats):
    /// 0-1 food Δ; 2-5 dir one-hot; 6-9 danger@1; 10-13 danger@2; 14 length;
    /// 15-18 free-run; 19-22 food-in-dir; 23-26 flood area; 27-30 can-reach-tail;
    /// 31 food-path-safe; 32 head flood; 33-34 tail Δ; 35-38 quadrant body dens; 39 hunger [0,1].
    /// </summary>
    public const int StateDim = 40;

    public const int NumActions = 4;
    public const int Hidden = 128;
    public const int TrainMaxBatch = 64;

    /// <summary>
    /// Bump when encoder/teacher semantics change so demo localStorage caches invalidate.
    /// </summary>
    public const int WeightsCacheVersion = 2;

    public static readonly string[] ActionKeys = ["up", "down", "left", "right"];
}
