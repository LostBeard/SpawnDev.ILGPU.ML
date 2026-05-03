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
            // NOTE: 4.9.4-rc.1 MaxLinearMemoryPages>16384 currently breaks ALL Wasm
            // dispatches with "memory import has a larger maximum size 32768 than the
            // module's declared maximum 16384" - the kernel WASM module's declared max
            // doesn't track the option. Reported to Geordi 2026-05-03. Reverting to
            // default (16384/1 GiB) until the kernel-codegen side of the fix lands;
            // DA3-Small op 93 OOM stays open for now.
            var accelerator = await context.CreateWasmAcceleratorAsync();
            return (context, accelerator);
        }

    }
}
