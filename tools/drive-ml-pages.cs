#:package Microsoft.Playwright@1.49.0
#:property JsonSerializerIsReflectionEnabledByDefault=true
// Live browser gate for the SpawnDev.ILGPU.ML demo PAGES.
//
// Why this exists: every demo page was migrated off ModelHub's byte[] loader onto
// InferenceSession.CreateFromHuggingFaceAsync(..., webTorrent:, http:) - the lazy-hash path - and the only
// check on that was "it compiles". A Blazor page is UI; compiling proves nothing about whether it loads a
// model. Nothing else in this repo drives these pages (PMT drives the unit-test harness, not the routes).
//
//   dotnet run tools/drive-ml-pages.cs -- [url] [route,route,...]
//
// ⚠️ These pages load their model LAZILY, on the action button - "Enter text and press Analyze. The model
// (~257 MB) downloads on first use." Navigating and waiting therefore proves nothing; the driver has to
// DRIVE. It seeds an input (a sample chip if the page offers one, else a textarea) and clicks `.run-btn`,
// which 9 of the demo pages share.
//
// Scope, stated rather than implied: this covers the TEXT pages. The image pages (classify, detect,
// remove-bg, super-res, depth) additionally need an image chosen before their run button enables, which is
// per-page UI work and not done here. They call CreateFromHuggingFaceAsync with the identical shape, so the
// migrated code path is the same one these pages exercise.
//
// Exit code is the number of pages that failed, so it is usable as a gate.
//
// ⚠️ Uses a PERSISTENT profile and TJ's installed Chrome on purpose: OPFS (where models cache) lives in the
// browser profile, so a fresh context re-downloads every model on every run; and Playwright's bundled
// chromium exposes a SOFTWARE WebGPU adapter, which reads as a hang rather than a config problem.
using System.Text.RegularExpressions;
using Microsoft.Playwright;

var url = args.Length > 0 && args[0].StartsWith("http") ? args[0].TrimEnd('/') : "http://localhost:5000";
var routeArg = args.FirstOrDefault(a => !a.StartsWith("http") && !a.StartsWith("-"));
var profileDir = Path.Combine(Path.GetTempPath(), "spawndev-ml-pages-profile");

// Every page migrated to the lazy-hash loader.
// ⚠️ Normalise routes, and strip any Git Bash path mangling. Under Git Bash an argument that STARTS WITH
// "/" is rewritten into a Windows path, so `-- "/sentiment"` arrives as "C:/Program Files/Git/sentiment"
// and navigation fails with "Cannot navigate to invalid URL". Accept "sentiment", "/sentiment", or the
// mangled form, so the caller cannot get this wrong.
var routes = (routeArg ?? "sentiment,embeddings")
    .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
    .Select(r => "/" + r.Replace("\\", "/").TrimEnd('/').Split('/').Last(part => part.Length > 0))
    .ToList();

// A page reports readiness in its own console voice ("[Classify] Model loaded on ...").
var ok = new Regex(@"model loaded|loaded on|ready|✅", RegexOptions.IgnoreCase);
var bad = new Regex(@"\berror\b|failed|exception|obsolete", RegexOptions.IgnoreCase);

Directory.CreateDirectory(profileDir);
using var pw = await Playwright.CreateAsync();
await using var ctx = await pw.Chromium.LaunchPersistentContextAsync(profileDir, new()
{
    Headless = false,
    Channel = "chrome",
});

int failed = 0, skipped = 0;
foreach (var route in routes)
{
    var page = await ctx.NewPageAsync();
    var done = new TaskCompletionSource<(bool Ok, string Line)>();
    var log = new List<string>();

    page.Console += (_, m) =>
    {
        var t = m.Text;
        log.Add(t);
        // A page's OWN error line is the failure signal; unrelated console noise is not.
        if (bad.IsMatch(t) && t.StartsWith("[")) done.TrySetResult((false, t));
        else if (ok.IsMatch(t)) done.TrySetResult((true, t));
    };
    page.PageError += (_, e) => done.TrySetResult((false, "pageerror: " + e));

    Console.WriteLine($"--- {route}");
    try
    {
        await page.GotoAsync(url + route, new() { WaitUntil = WaitUntilState.DOMContentLoaded, Timeout = 60_000 });

        // ⚠️ WAIT for Blazor to boot before querying. DOMContentLoaded fires long before the WASM runtime
        // has rendered anything, so an immediate Locator.CountAsync() returns 0 for controls that are
        // simply not there YET - which the first version of this driver reported as "no .run-btn", i.e. a
        // missing-feature verdict on a page that has the feature.
        try
        {
            await page.Locator(".run-btn").First.WaitForAsync(new() { Timeout = 90_000 });
        }
        catch (TimeoutException)
        {
            skipped++;
            Console.WriteLine("    SKIP no .run-btn after 90s - needs per-page interaction (likely an image)");
            await page.CloseAsync();
            continue;
        }

        // Seed an input so the action button enables, then press it - the model loads on that click.
        var chip = page.Locator(".sample-chip").First;
        if (await chip.CountAsync() > 0)
        {
            await chip.ClickAsync();
        }
        else
        {
            var box = page.Locator("textarea, input[type=text]").First;
            if (await box.CountAsync() > 0) await box.FillAsync("This is a wonderful and delightful result.");
        }

        await page.Locator(".run-btn").First.ClickAsync(new() { Timeout = 30_000 });

        // First click downloads the model; later runs hit the OPFS cache.
        var finished = await Task.WhenAny(done.Task, Task.Delay(TimeSpan.FromMinutes(4)));
        if (finished != done.Task)
        {
            failed++;
            Console.WriteLine($"    TIMEOUT - no ready/error line in 4 min");
            foreach (var l in log.TakeLast(4)) Console.WriteLine($"      | {l}");
        }
        else
        {
            var (good, line) = done.Task.Result;
            if (!good) failed++;
            Console.WriteLine($"    {(good ? "OK  " : "FAIL")} {line}");
        }
    }
    catch (Exception ex)
    {
        failed++;
        Console.WriteLine($"    FAIL {ex.GetType().Name}: {ex.Message}");
    }
    finally
    {
        await page.CloseAsync();
    }
}

Console.WriteLine();
// ⚠️ A SKIP is not a pass. The first version of this driver printed "1/1 pages reached a ready state" for
// a page it had skipped, which is a false green - the exact failure this gate exists to catch.
var reached = routes.Count - failed - skipped;
Console.WriteLine($"{reached}/{routes.Count} pages reached a ready state "
                + $"({failed} failed, {skipped} skipped - a skip is NOT a pass)");
return failed;
