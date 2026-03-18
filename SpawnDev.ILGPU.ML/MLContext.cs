using ILGPU;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// Factory for creating ILGPU contexts pre-configured for ML workloads.
/// Always enables Algorithms (required for MathF intrinsics on CUDA/OpenCL).
/// </summary>
public static class MLContext
{
    /// <summary>
    /// Create an ILGPU context builder with Algorithms enabled.
    /// Use this instead of Context.Create() for ML workloads.
    /// </summary>
    public static Context.Builder Create()
        => Context.Create().EnableAlgorithms();

    /// <summary>
    /// Create a fully initialized context with Algorithms enabled.
    /// </summary>
    public static Context CreateContext()
        => Create().ToContext();
}
