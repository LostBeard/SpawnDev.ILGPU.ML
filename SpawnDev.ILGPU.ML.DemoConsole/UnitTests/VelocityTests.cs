using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.DemoConsole.UnitTests;

/// <summary>
/// Runs ML kernel tests on the Velocity backend: the SIMD CPU accelerator that <see cref="MLContext.CreateContext"/> registers as the CPU
/// backend on 64-bit desktop processes (the plain interpreting CPU accelerator is <see cref="CPUTests"/>).
/// </summary>
public class VelocityTests : MLTestBase
{
    protected override string BackendName => "Velocity";

    protected override Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
    {
        var context = MLContext.CreateContext();
        var device = context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Velocity)
            ?? throw new UnsupportedTestException("No Velocity device (32-bit process?)");
        return Task.FromResult((context, device.CreateAccelerator(context)));
    }
}
