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
            // 32768 pages = 2 GiB SharedArrayBuffer ceiling. DA3-Small graph executor's
            // working set crosses the default 16384 (1 GiB) at op 93 and OOMs on grow().
            // 4.9.4-rc.2 threads MaxLinearMemoryPages through both the host WebAssembly.Memory
            // creation AND the kernel module's import.maximum declaration (rc.1 only updated
            // the host side and instantiation rejected at any cap > 16384 due to import-vs-host
            // max mismatch per WebAssembly spec). Bump to 65536 (4 GiB hard ceiling for
            // SharedArrayBuffer per browser tab) if 2 GiB still hits.
            var accelerator = await context.CreateWasmAcceleratorAsync(new WasmBackendOptions
            {
                MaxLinearMemoryPages = 32768,
            });
            return (context, accelerator);
        }

    }
}
