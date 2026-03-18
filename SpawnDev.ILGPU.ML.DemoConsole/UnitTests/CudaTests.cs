using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.DemoConsole.UnitTests;

/// <summary>
/// Runs ML kernel tests on the CUDA backend (NVIDIA GPU).
/// EnableAlgorithms() required for MathF intrinsics (Exp, Sqrt, etc.).
/// </summary>
public class CudaTests : MLTestBase
{
    protected override string BackendName => "CUDA";

    protected override Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
    {
        var context = Context.Create()
            .EnableAlgorithms()
            .ToContext();
        var cudaDevices = context.GetCudaDevices();
        if (cudaDevices.Count == 0)
            throw new UnsupportedTestException("No CUDA devices found");
        var accelerator = cudaDevices[0].CreateCudaAccelerator(context);
        return Task.FromResult((context, (Accelerator)accelerator));
    }
}
