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

}
