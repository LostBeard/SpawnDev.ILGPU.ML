using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;
using SpawnDev.ILGPU.Wasm;
using SpawnDev.ILGPU.Wasm.Backend;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.UnitTests
{
    /// <summary>
    /// Wasm backend tests. Inherits all shared tests from BackendTestBase.
    /// </summary>
    public class WasmTests : MLTestBase
    {
        private readonly System.Net.Http.HttpClient _http;

        public WasmTests(System.Net.Http.HttpClient http)
        {
            _http = http;
        }

        protected override string BackendName => "Wasm";

        protected override System.Net.Http.HttpClient? GetHttpClient() => _http;

        protected override async Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
        {
            var builder = MLContext.Create()
                .EnableAlgorithms()
                .EnableWasmAlgorithms()
                .Wasm();
            var context = builder.ToContext();
            WasmBackend.VerboseLogging = false;
            var accelerator = await context.CreateWasmAcceleratorAsync();
            return (context, accelerator);
        }

    }
}
