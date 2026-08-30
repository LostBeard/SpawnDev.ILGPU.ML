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
// ⚠️ SCOPE IS NOT "every page". Docs/DEMO_AND_MODEL_STATUS.md is the source of truth for which demos were
// VERIFIED before this migration. A PARTIAL/WIP page failing here says nothing about the migration - it was
// already unfinished - so gating on one manufactures a regression that does not exist. Only add a route
// below once you have read its status row.
//
// ⚠️ WAIT ON THE RENDERED RESULT, NOT ON A CONSOLE LINE. Two real defects in the first version of this
// driver, both from keying on the MECHANISM instead of the MEANING:
//   1. It filled only the FIRST text input. /embeddings needs BOTH sentences - `ComputeSimilarity` opens
//      with `if (IsNullOrWhiteSpace(_sentenceA) || IsNullOrWhiteSpace(_sentenceB)) return;` - so the click
//      landed and the handler returned instantly. The page sat there doing nothing, correctly.
//   2. It waited for a /model loaded|ready/ console line. EmbeddingsPage logs ONLY on error (3 of 3
//      WriteLines are catch blocks), so that wait could never succeed and a healthy page would have been
//      reported as a 4-minute TIMEOUT - a false FAILURE, the mirror of a false pass.
// A page's logging is incidental; its rendered result is the fact. `.sentiment-verdict` showing POSITIVE
// means the model was fetched, loaded, tokenized, and run. That is what this gate asserts on.
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

// route -> (every input that must be seeded, the element that proves a real result rendered)
// Both routes below are ✅ VERIFIED in Docs/DEMO_AND_MODEL_STATUS.md and load weights via the migrated path.
var pages = new Dictionary<string, (string[] Fill, string Result)>
{
    ["/sentiment"]  = (new[] { ".sentiment-textarea" }, ".sentiment-verdict"),
    ["/embeddings"] = (new[] { ".sim-input" },          ".sim-score-value"),
};

// ⚠️ Normalise routes, and strip any Git Bash path mangling. Under Git Bash an argument that STARTS WITH
// "/" is rewritten into a Windows path, so `-- "/sentiment"` arrives as "C:/Program Files/Git/sentiment".
var routes = (routeArg ?? string.Join(',', pages.Keys))
    .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
    .Select(r => "/" + r.Replace("\\", "/").TrimEnd('/').Split('/').Last(part => part.Length > 0))
    .ToList();

var bad = new Regex(@"\berror\b|failed|exception", RegexOptions.IgnoreCase);

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
    Console.WriteLine($"--- {route}");
    if (!pages.TryGetValue(route, out var spec))
    {
        skipped++;
        Console.WriteLine("    SKIP not in the gate table - add its inputs + result selector first");
        continue;
    }

    var page = await ctx.NewPageAsync();
    var oops = new TaskCompletionSource<string>();
    var log = new List<string>();
    // A page's OWN error line ("[Embeddings] Error: ...") fails fast instead of burning the full timeout.
    page.Console += (_, m) =>
    {
        log.Add(m.Text);
        if (m.Text.StartsWith("[") && bad.IsMatch(m.Text)) oops.TrySetResult(m.Text);
    };
    page.PageError += (_, e) => oops.TrySetResult("pageerror: " + e);

    try
    {
        await page.GotoAsync(url + route, new() { WaitUntil = WaitUntilState.DOMContentLoaded, Timeout = 60_000 });

        // ⚠️ WAIT for Blazor to boot before querying. DOMContentLoaded fires long before the WASM runtime
        // renders, so an immediate CountAsync() returns 0 for controls that are simply not there YET.
        await page.Locator(".run-btn").First.WaitForAsync(new() { Timeout = 90_000 });

        // Seed EVERY required input - a page that validates its inputs will silently no-op otherwise.
        foreach (var sel in spec.Fill)
        {
            var boxes = page.Locator(sel);
            var n = await boxes.CountAsync();
            if (n == 0) throw new Exception($"input '{sel}' not found - the page markup changed");
            for (int i = 0; i < n; i++)
                await boxes.Nth(i).FillAsync(i == 0
                    ? "The cat sat quietly on the warm mat."
                    : "A kitten was resting on a soft rug.");
        }

        await page.Locator(".run-btn").First.ClickAsync(new() { Timeout = 30_000 });

        // The RESULT element is the assertion: it renders only after weights load and inference runs.
        // First click downloads the model; later runs hit the OPFS cache.
        var result = page.Locator(spec.Result).First;
        var ready = result.WaitForAsync(new() { Timeout = 240_000 });
        var finished = await Task.WhenAny(ready, oops.Task);

        if (finished == oops.Task)
        {
            failed++;
            Console.WriteLine($"    FAIL {oops.Task.Result}");
        }
        else
        {
            await ready;   // observe a Playwright timeout as a failure, not as success
            var text = (await result.InnerTextAsync()).Replace('\n', ' ').Trim();
            if (string.IsNullOrWhiteSpace(text))
            {
                failed++;
                Console.WriteLine($"    FAIL '{spec.Result}' rendered EMPTY - no real result");
            }
            else
            {
                Console.WriteLine($"    OK   {spec.Result} => {text}");
            }
        }
    }
    catch (Exception ex)
    {
        failed++;
        Console.WriteLine($"    FAIL {ex.GetType().Name}: {ex.Message.Split('\n')[0]}");
        foreach (var l in log.TakeLast(4)) Console.WriteLine($"      | {l}");
    }
    finally
    {
        await page.CloseAsync();
    }
}

Console.WriteLine();
// ⚠️ A SKIP is not a pass. The first version of this driver printed "1/1 pages reached a ready state" for
// a page it had skipped, which is a false green - the exact failure this gate exists to catch.
var passed = routes.Count - failed - skipped;
Console.WriteLine($"{passed}/{routes.Count} pages produced a real result "
                + $"({failed} failed, {skipped} skipped - a skip is NOT a pass)");
return failed;
