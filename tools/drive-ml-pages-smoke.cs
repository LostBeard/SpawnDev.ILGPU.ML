#:package Microsoft.Playwright@1.49.0
#:property JsonSerializerIsReflectionEnabledByDefault=true
// Smoke every Demo route: Blazor mounts, no pageerror, capture honesty banners + page title.
// Does NOT run inference (that's drive-ml-pages.cs for gated verified routes).
//
//   node tools/serve-demo.mjs <publish/wwwroot> 5000
//   dotnet run tools/drive-ml-pages-smoke.cs -- http://127.0.0.1:5000
using System.Text.Json;
using Microsoft.Playwright;

var url = args.Length > 0 && args[0].StartsWith("http") ? args[0].TrimEnd('/') : "http://127.0.0.1:5000";
var outPath = args.FirstOrDefault(a => a.EndsWith(".json"))
    ?? Path.Combine(Directory.GetCurrentDirectory(), "_mldump", "demo-page-smoke.json");

string[] routes =
[
    "/", "/getting-started", "/pipelines", "/ai-chat",
    "/classify", "/depth", "/detect", "/style", "/remove-bg", "/pose", "/super-res", "/face", "/clip",
    "/sentiment", "/embeddings", "/whisper", "/tts", "/text-gen",
    "/generate", "/image-to-3d", "/depth-voxel",
    "/explain", "/train", "/snake",
    "/benchmark", "/models", "/cache", "/inspector", "/tests",
];

Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);
var profileDir = Path.Combine(Path.GetTempPath(), "spawndev-ml-smoke-profile");
Directory.CreateDirectory(profileDir);

using var pw = await Playwright.CreateAsync();
await using var ctx = await pw.Chromium.LaunchPersistentContextAsync(profileDir, new()
{
    Headless = false,
    Channel = "chrome",
});

var results = new List<object>();
int ok = 0, fail = 0;

foreach (var route in routes)
{
    Console.WriteLine($"--- {route}");
    var page = await ctx.NewPageAsync();
    var consoleErrs = new List<string>();
    var pageErrors = new List<string>();
    page.Console += (_, m) =>
    {
        if (m.Type is "error" or "warning")
            consoleErrs.Add($"[{m.Type}] {m.Text}");
    };
    page.PageError += (_, e) => pageErrors.Add(e);

    string status = "FAIL";
    string title = "";
    string banner = "";
    string mountProbe = "";
    string detail = "";

    try
    {
        await page.GotoAsync(url + route, new() { WaitUntil = WaitUntilState.DOMContentLoaded, Timeout = 60_000 });

        // Wait for Blazor MainLayout — .navbar-brand is on every routed page.
        // Do NOT use Locator.Or here: when BOTH brand and h1 exist, WaitFor throws
        // "strict mode violation ... resolved to 2 elements" even though the page mounted.
        var brand = page.Locator(".navbar-brand").First;
        await brand.WaitForAsync(new() { Timeout = 120_000 });

        title = (await page.TitleAsync()).Trim();
        mountProbe = (await brand.InnerTextAsync()).Trim();

        var coming = page.Locator(".demo-status-banner").First;
        if (await coming.CountAsync() > 0)
            banner = (await coming.InnerTextAsync()).Replace('\n', ' ').Trim();
        else
        {
            var tag = page.Locator(".demo-status-tag, .coming-soon-title").First;
            if (await tag.CountAsync() > 0)
                banner = (await tag.InnerTextAsync()).Replace('\n', ' ').Trim();
        }

        // Fatal if the WASM runtime crashed (pageerror) after mount attempt.
        if (pageErrors.Count > 0)
        {
            status = "FAIL";
            detail = "pageerror: " + pageErrors[0];
            fail++;
        }
        else if (string.IsNullOrWhiteSpace(mountProbe))
        {
            status = "FAIL";
            detail = "empty mount probe";
            fail++;
        }
        else
        {
            status = "OK";
            detail = banner.Length > 0 ? "banner=" + banner : "mounted";
            ok++;
        }
    }
    catch (Exception ex)
    {
        status = "FAIL";
        detail = ex.GetType().Name + ": " + ex.Message.Split('\n')[0];
        fail++;
    }

    Console.WriteLine($"    {status}  title={title}");
    if (!string.IsNullOrEmpty(banner)) Console.WriteLine($"           banner={banner}");
    if (status != "OK") Console.WriteLine($"           {detail}");
    foreach (var e in pageErrors.Take(2)) Console.WriteLine($"           pageerror: {e}");
    foreach (var e in consoleErrs.Where(x => x.Contains("[error]", StringComparison.OrdinalIgnoreCase)).Take(3))
        Console.WriteLine($"           {e}");

    results.Add(new
    {
        route,
        status,
        title,
        mountProbe,
        banner,
        detail,
        pageErrors,
        consoleErrors = consoleErrs.Take(20).ToList(),
    });

    await page.CloseAsync();
}

var summary = new { when = DateTime.UtcNow.ToString("o"), url, ok, fail, total = routes.Length, results };
await File.WriteAllTextAsync(outPath, JsonSerializer.Serialize(summary, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine();
Console.WriteLine($"{ok}/{routes.Length} routes mounted ({fail} failed). Wrote {outPath}");
return fail;
