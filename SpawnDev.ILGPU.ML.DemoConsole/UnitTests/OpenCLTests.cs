using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.DemoConsole.UnitTests;

/// <summary>
/// Runs ML kernel tests on the OpenCL backend (AMD/Intel/NVIDIA GPU).
/// Uses tiled MatMul with shared memory.
/// </summary>
public class OpenCLTests : MLTestBase
{
    protected override string BackendName => "OpenCL";

    protected override Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
    {
        var context = Context.CreateDefault();
        var clDevices = context.GetCLDevices();
        if (clDevices.Count == 0)
            throw new UnsupportedTestException("No OpenCL devices found");
        var accelerator = clDevices[0].CreateCLAccelerator(context);
        return Task.FromResult((context, (Accelerator)accelerator));
    }
}
