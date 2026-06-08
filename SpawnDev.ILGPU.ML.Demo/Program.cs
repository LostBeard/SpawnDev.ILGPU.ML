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

// WebTorrent client for P2P model delivery (direct stream access, no service worker needed).
// Wire the OPFS filesystem into the client so torrent pieces persist to OPFS AND the zero-copy browser
// download path fires (each piece's bytes stay in JS: fetch -> SubtleCrypto -> OPFS, no .NET byte[] hop).
// Without AsyncFileSystem the client falls back to an in-memory store + the byte[] download path, so the
// model download never uses zero-copy.
builder.Services.AddSingleton<WebTorrentClient>(sp =>
    new WebTorrentClient(new WebTorrentClientOptions { AsyncFileSystem = sp.GetRequiredService<IAsyncFS>() }));

// Shared OPFS model cache (browser). One instance so every demo + the cache-management page see the same
// cached models. ModelCache reads/writes the persistent OPFS "ilgpu-ml-models" dir, so even multiple
// instances would share storage — but a singleton gives the management UI a single source to query/purge.
builder.Services.AddSingleton<SpawnDev.ILGPU.ML.Hub.ModelCache>();

// Register test types as singletons for UnitTestsView discovery
// DumpFolder test runs FIRST — verifies results can be written
builder.Services.AddSingleton<DumpFolderTests>();
// HuggingFace CDN tests run next — no GPU needed, validates API + downloads
builder.Services.AddSingleton<HuggingFaceTests>();
// Model Inspector tests — pure CPU parsing, no GPU. Registered standalone so they run ONCE in the
// browser runtime, not once per backend lane (inspection never touches an accelerator).
builder.Services.AddSingleton<ModelInspectorTests>();
builder.Services.AddSingleton<WebGPUTests>();
builder.Services.AddSingleton<WasmTests>();
builder.Services.AddSingleton<WebGLTests>();
builder.Services.AddSingleton<DefaultTests>();

builder.Services.AddSingleton(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });
builder.Services.AddSingleton<SpawnDev.ILGPU.Services.ShaderDebugService>();

builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

await builder.Build().BlazorJSRunAsync();
