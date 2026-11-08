using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;

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
    {
        var b = Context.Create().AllAccelerators().EnableAlgorithms();
        // SpawnDev.ILGPU 5.4.18 made float math strict by default (CUDA / OpenCL no longer fuse a * b + c on their own, so kernels round as the C# does).
        // ML's hot loops (`sum += a * b` in the GEMMs, GEMVs, attention, norms) were written for the fused form; they keep it here until they are
        // converted to float.MultiplyAddEstimate (one fma by construction). MEASURED by SpawnDev.ILGPU: a CUDA 4096x4096 GEMV +23% strict. Env
        // ML_FP_CONTRACT=0 builds strict (the A/B). The plain CPU builder below stays strict: it is the exact reference.
        if (Environment.GetEnvironmentVariable("ML_FP_CONTRACT") != "0") b = b.EnableFloatingPointContraction();
        return b;
    }

    /// <summary>
    /// Create a fully initialized context with Algorithms enabled.
    /// </summary>
    public static Context CreateContext()
        => Create().ToContext();

    /// <summary>
    /// Create a context builder with Algorithms enabled and only the plain (interpreting) CPU accelerator: the slow, exact reference.
    /// <see cref="Create"/> registers Velocity as the CPU backend on 64-bit desktop processes, so the plain CPU accelerator needs this.
    /// </summary>
    public static Context.Builder CreatePlainCpu()
        => Context.Create().DefaultCPU().EnableAlgorithms();

    /// <summary>Create a fully initialized context with only the plain CPU accelerator (see <see cref="CreatePlainCpu"/>).</summary>
    public static Context CreatePlainCpuContext()
        => CreatePlainCpu().ToContext();

    /// <summary>
    /// Create the accelerator of the CPU backend that does the work in <paramref name="context"/>: the Velocity SIMD accelerator where the context has one
    /// (64-bit desktop processes), the plain CPU accelerator otherwise.
    /// </summary>
    public static Accelerator CreateCpuAccelerator(Context context)
    {
        var device = context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Velocity)
            ?? context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.CPU)
            ?? throw new InvalidOperationException("The context has no CPU or Velocity device.");
        return device.CreateAccelerator(context);
    }
}
