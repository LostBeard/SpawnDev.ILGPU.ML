using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Backend;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.UnitTests;

/// <summary>
/// Runs ML kernel tests on the WebGPU backend.
/// WebGPU uses the TILED MatMul (shared memory + barriers).
/// If tests pass on CPU but fail here, the tiled MatMul WGSL codegen is the bug.
///
/// Test methods are declared here directly because UnitTestsView discovers
/// via DeclaredOnly reflection — inherited methods from the abstract base aren't found.
/// Each method delegates to the shared implementation in MLTestBase.
/// </summary>
public class WebGPUTests : MLTestBase
{
    protected override string BackendName => "WebGPU";

    protected override async Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
    {
        var builder = MLContext.Create();
        await builder.WebGPU();
        var context = builder.ToContext();
        var devices = context.GetWebGPUDevices();
        if (devices.Count == 0)
            throw new UnsupportedTestException("No WebGPU devices found");
        var accelerator = await devices[0].CreateAcceleratorAsync(context);
        return (context, accelerator);
    }

}
