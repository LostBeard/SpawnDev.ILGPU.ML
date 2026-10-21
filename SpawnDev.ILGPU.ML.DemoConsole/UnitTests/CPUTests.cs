using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

namespace SpawnDev.ILGPU.ML.DemoConsole.UnitTests;

/// <summary>
/// Runs ML kernel tests on the CPU backend.
/// CPU backend uses simple (non-tiled) MatMul.
/// </summary>
public class CPUTests : MLTestBase
{
    protected override string BackendName => "CPU";

    protected override Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
    {
        var context = MLContext.CreatePlainCpuContext();      // MLContext.CreateContext() registers Velocity as the CPU backend (VelocityTests); this lane is the plain interpreting CPU accelerator
        var accelerator = context.CreateCPUAccelerator(0);
        return Task.FromResult((context, (Accelerator)accelerator));
    }
}
