using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using SpawnDev.AsyncFileSystem;
using SpawnDev.AsyncFileSystem.BrowserWASM;
using SpawnDev.SpawnJS;
using SpawnDev.ILGPU.ML.Demo;
using SpawnDev.ILGPU.ML.Demo.UnitTests;
using SpawnDev.UnitTesting;
using SpawnDev.WebTorrent;
using System.Reflection;

// Print build timestamp so we can verify we're running the right build via browser console
Console.WriteLine($"[SpawnDev.ILGPU.ML.Demo] Build: {BuildTimestamp.Value}");

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.Services.AddSpawnJSRuntime(out var JS);
JS.Verbose = false;

// allow firing the gc collection from JS (for debugging purposes)
JS.Set("_gcCollect", () => GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true));

// Cross-platform persistent file system (OPFS in browser, native on desktop)
builder.Services.AddSingleton<IAsyncFS, AsyncFSFileSystemDirectoryHandle>();

// WebTorrent client for P2P model delivery. Persist torrents + pieces to OPFS (the IAsyncFS singleton =
// navigator.getDirectory()) so a downloaded model SURVIVES page reloads — the client restores its torrents
// on startup instead of re-downloading — and so the /cache page can list/cancel/remove/seed them. The
// AsyncFS exposes JS-typed writes (TypedArray/Blob/ArrayBuffer) so piece bytes stay JS-side (no .NET byte[]
// hop), and the loader streams them straight to the GPU via CopyFromStreamAsync's zero-copy IJSReadStream.
builder.Services.AddSingleton<WebTorrentClient>(sp =>
{
    var client = new WebTorrentClient(new WebTorrentClientOptions { AsyncFileSystem = sp.GetRequiredService<IAsyncFS>() });
    // Restore persisted torrents from OPFS so a page reload REUSES the downloaded pieces instead of
    // re-downloading (the ML demo never did this — unlike the WebTorrent demo — so every refresh re-pulled
    // the model). Fire-and-forget; runs when the client is first resolved (after the OPFS FS has started).
    _ = client.RestoreFromStorageAsync();
    return client;
});

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

await builder.Build().SpawnJSRunAsync();
