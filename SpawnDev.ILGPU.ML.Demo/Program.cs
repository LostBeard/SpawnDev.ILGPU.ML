using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using SpawnDev.AsyncFileSystem;
using SpawnDev.AsyncFileSystem.BrowserWASM;
using SpawnDev.BlazorJS;
using SpawnDev.ILGPU.ML.Demo;
using SpawnDev.ILGPU.ML.Demo.UnitTests;
using SpawnDev.UnitTesting;
using SpawnDev.WebTorrent;
using System.Reflection;

// Print build timestamp so we can verify we're running the right build via browser console
Console.WriteLine($"[SpawnDev.ILGPU.ML.Demo] Build: {BuildTimestamp.Value}");

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.Services.AddBlazorJSRuntime();

// Cross-platform persistent file system (OPFS in browser, native on desktop)
builder.Services.AddSingleton<IAsyncFS, AsyncFSFileSystemDirectoryHandle>();

// WebTorrent client for P2P model delivery (direct stream access, no service worker needed)
builder.Services.AddSingleton<WebTorrentClient>();

// Register test types as singletons for UnitTestsView discovery
// DumpFolder test runs FIRST — verifies results can be written
builder.Services.AddSingleton<DumpFolderTests>();
// HuggingFace CDN tests run next — no GPU needed, validates API + downloads
builder.Services.AddSingleton<HuggingFaceTests>();
builder.Services.AddSingleton<WebGPUTests>();
builder.Services.AddSingleton<WasmTests>();
builder.Services.AddSingleton<WebGLTests>();
builder.Services.AddSingleton<DefaultTests>();

builder.Services.AddSingleton(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });
builder.Services.AddSingleton<SpawnDev.ILGPU.Services.ShaderDebugService>();

builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

await builder.Build().BlazorJSRunAsync();
