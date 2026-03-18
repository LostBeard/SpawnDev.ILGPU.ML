using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;
using SpawnDev.ILGPU.WebGL;
using SpawnDev.ILGPU.WebGL.Backend;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.UnitTests
{
    /// <summary>
    /// WebGL2 backend tests. Inherits all shared tests from BackendTestBase
    /// and overrides unsupported features (shared memory, barriers, broadcasts,
    /// subgroups — WebGL2 vertex shaders are single-threaded with no workgroups).
    /// </summary>
    public class WebGLTests : MLTestBase
    {
        protected override string BackendName => "WebGL";

        protected override async Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
        {
            var builder = MLContext.Create();
            await builder.WebGL();
            var context = builder.ToContext();
            var devices = context.GetWebGLDevices();
            if (devices.Count == 0)
                throw new UnsupportedTestException("No WebGL2 devices found");
            var accelerator = devices[0].CreateAccelerator(context);
            return (context, accelerator);
        }

        //protected override async Task<(Context context, Accelerator accelerator)> CreateEmulatedAcceleratorAsync()
        //{
        //    var builder = MLContext.Create();
        //    await builder.WebGL();
        //    var context = builder.ToContext();
        //    var devices = context.GetWebGLDevices();
        //    if (devices.Count == 0)
        //        throw new UnsupportedTestException("No WebGL2 devices found");
        //    var accelerator = devices[0].CreateAccelerator(context, new WebGLBackendOptions
        //    {
        //        F64Emulation = F64EmulationMode.Dekker,
        //    });
        //    return (context, accelerator);
        //}

    }
}
