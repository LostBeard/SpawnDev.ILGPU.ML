using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using SpawnDev.BlazorJS;
using SpawnDev.ILGPU.ML.Demo;
using SpawnDev.ILGPU.ML.Demo.UnitTests;
using SpawnDev.UnitTesting;
using System.Reflection;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.Services.AddBlazorJSRuntime();

// Register test types as singletons for UnitTestsView discovery
builder.Services.AddSingleton<WebGPUTests>();
builder.Services.AddSingleton<WasmTests>();
builder.Services.AddSingleton<WebGLTests>();
builder.Services.AddSingleton<DefaultTests>();

builder.Services.AddScoped(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });

builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

await builder.Build().BlazorJSRunAsync();
