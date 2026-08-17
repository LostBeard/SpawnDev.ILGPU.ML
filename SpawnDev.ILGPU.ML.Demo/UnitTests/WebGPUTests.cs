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
    private readonly System.Net.Http.HttpClient _http;
    private readonly SpawnDev.WebTorrent.WebTorrentClient _webTorrent;
    private readonly SpawnDev.AsyncFileSystem.IAsyncFS _asyncFs;

    public WebGPUTests(System.Net.Http.HttpClient http, SpawnDev.WebTorrent.WebTorrentClient webTorrent, SpawnDev.AsyncFileSystem.IAsyncFS asyncFs)
    {
        _http = http;
        _webTorrent = webTorrent;
        _asyncFs = asyncFs;
    }

    protected override string BackendName => "WebGPU";

    protected override System.Net.Http.HttpClient? GetHttpClient() => _http;

    // The DEMO's OPFS-backed client + filesystem (Program.cs) — so a download measurement exercises the real demo path.
    protected override SpawnDev.WebTorrent.WebTorrentClient? GetWebTorrentClient() => _webTorrent;
    protected override SpawnDev.AsyncFileSystem.IAsyncFS? GetAsyncFS() => _asyncFs;

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

    /// <summary>
    /// DIAGNOSTIC/REGRESSION: the WebGPU device must be created with the ADAPTER'S maxStorageBufferBindingSize,
    /// not the 128 MiB spec default. Models with a &gt;128 MiB storage binding (DA3 depth, the transformer
    /// Gather in TextGen - and SpawnDev.AI's chat model) fail at dispatch with
    /// "Binding size (N) is larger than the maximum storage buffer binding size (134217728)" when the device
    /// falls back to the default. WebGPUNativeAccelerator.CreateAsync requests requiredLimits from the adapter;
    /// this proves the request actually took effect on the live device. No model download - just device probe.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task WebGPU_DeviceStorageBufferLimit_MatchesAdapter() => await RunTest(async accelerator =>
    {
        var prevVerbose = WebGPUBackend.VerboseLogging;
        WebGPUBackend.VerboseLogging = true;
        try
        {
            var acc = ((WebGPUAccelerator)accelerator).NativeAccelerator;

            long? adapterMax = null;
            try { using var al = acc.Device?.Adapter?.Limits; adapterMax = al?.MaxStorageBufferBindingSize; } catch { }

            long? deviceMax = null;
            try { using var dl = acc.NativeDevice?.Limits; deviceMax = dl?.MaxStorageBufferBindingSize; } catch { }

            Console.WriteLine($"[LIMITPROBE] adapter.maxStorageBufferBindingSize={adapterMax}  device.maxStorageBufferBindingSize={deviceMax}  (spec default=134217728)");

            if (deviceMax is null)
                throw new Exception("could not read device.limits.maxStorageBufferBindingSize");

            // The device must have been granted the adapter's higher limit. The bug (2026-08-16 GH-Pages run):
            // device stuck at the 134217728 default while the adapter advertised ~2 GiB.
            if (adapterMax is > 134217728 && deviceMax <= 134217728)
                throw new Exception(
                    $"DEVICE DID NOT GET ADAPTER LIMIT: adapter allows {adapterMax} but device was created with only " +
                    $"{deviceMax} - a >128 MiB storage binding will fail at dispatch. requiredLimits was not applied.");

            Console.WriteLine($"[LIMITPROBE] PASS - device granted {deviceMax} storage-buffer binding");
        }
        finally { WebGPUBackend.VerboseLogging = prevVerbose; }
    });

}
