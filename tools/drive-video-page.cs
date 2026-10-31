#:package Microsoft.Playwright@1.49.0
#:property JsonSerializerIsReflectionEnabledByDefault=true
// Live browser check of the /video studio (SpawnDev.ILGPU.ML.Demo/Pages/VideoPage.razor): drives the page the way a visitor does and asserts on what renders.
//
//   dotnet run tools/drive-video-page.cs -- [demo url] [model server] [out dir]
//   e.g.  node tools/serve-demo.mjs <published wwwroot> 5071
//         node tools/unirig-harness/serve-files.mjs <folder laid out as org/repo/resolve/main/file> 8769
//         dotnet run tools/drive-video-page.cs -- http://localhost:5071 http://127.0.0.1:8769 <out dir>
//
// Steps: the "Animate the kitten" example (LTX image to video), the live preview must appear while it denoises (a screenshot of it is saved), then the clip;
// "Extend this clip" with a new prompt must append frames to it; the "Keyframes: fly into the house" example must produce a clip; "A story in three shots" must
// produce 97 + 2 x 80 frames. Screenshots go to the out dir.
// Uses TJ's installed Chrome with a persistent profile (real WebGPU adapter; the models cache in OPFS across runs), like drive-ml-pages.cs.
// Exit code = number of failed steps.
using System.Text.RegularExpressions;
using Microsoft.Playwright;

var url = (args.Length > 0 ? args[0] : "http://localhost:5071").TrimEnd('/');
var modelServer = args.Length > 1 ? args[1] : "";
var outDir = args.Length > 2 ? args[2] : Path.Combine(Path.GetTempPath(), "video-page-shots");
Directory.CreateDirectory(outDir);
var profileDir = Path.Combine(Path.GetTempPath(), "spawndev-ml-pages-profile");
Directory.CreateDirectory(profileDir);

using var pw = await Playwright.CreateAsync();
await using var ctx = await pw.Chromium.LaunchPersistentContextAsync(profileDir, new() { Headless = false, Channel = "chrome", ViewportSize = new() { Width = 1280, Height = 1100 } });
var page = ctx.Pages.Count > 0 ? ctx.Pages[0] : await ctx.NewPageAsync();
var log = new List<string>();
page.Console += (_, m) => { if (m.Text.Contains("[Video]") || m.Type == "error") { log.Add($"{m.Type}: {m.Text}"); Console.WriteLine($"    console {m.Type}: {m.Text[..Math.Min(200, m.Text.Length)]}"); } };
int failed = 0;
var sw = System.Diagnostics.Stopwatch.StartNew();

async Task<string> InfoAsync() => (await page.Locator(".vid-info-item").InnerTextAsync()).Trim();
async Task<bool> WaitResultAsync(string step, int minutes)
{
    // started = the Generate button changes ("Loading..." / "Generating..."); finished = it is back and the result panel is shown (or an error is).
    // Without the first wait the previous clip's finished state reads as this one's.
    var deadline = DateTime.UtcNow.AddMinutes(minutes);
    var startBy = DateTime.UtcNow.AddSeconds(60);
    while ((await page.Locator(".vid-btn").InnerTextAsync()).Trim() == "Generate")
    {
        if (DateTime.UtcNow > startBy) { Console.WriteLine("    FAIL: the run never started"); return false; }
        await page.WaitForTimeoutAsync(200);
    }
    bool sawPreview = false;
    var runClock = System.Diagnostics.Stopwatch.StartNew();
    string lastStage = "";
    while (DateTime.UtcNow < deadline)
    {
        // the stage line, timed: where a clip's time goes (text, conditions, loading the transformer, steps, decode)
        if (await page.Locator(".vid-progress-text").CountAsync() > 0)
        {
            var st = (await page.Locator(".vid-progress-text").InnerTextAsync()).Trim();
            var key = Regex.Replace(st, @"step \d+ of \d+", "steps");
            if (key != lastStage) { lastStage = key; Console.WriteLine($"      {runClock.Elapsed.TotalSeconds,5:F1} s  {st}"); }
        }
        if (!sawPreview && await page.Locator(".vid-live").IsVisibleAsync())
        {
            var label = await page.Locator(".vid-live-label").InnerTextAsync();
            if (Regex.IsMatch(label, @"step [2-9]"))
            {
                sawPreview = true;
                await page.WaitForTimeoutAsync(1500);
                await page.Locator(".vid-live").ScreenshotAsync(new() { Path = Path.Combine(outDir, $"{step}-preview.png") });
                Console.WriteLine($"    live preview: '{label}' (screenshot {step}-preview.png) at {sw.Elapsed.TotalSeconds:F0} s");
            }
        }
        if (await page.Locator(".vid-error").CountAsync() > 0) { Console.WriteLine($"    ERROR shown: {await page.Locator(".vid-error").InnerTextAsync()}"); return false; }
        var btn = (await page.Locator(".vid-btn").InnerTextAsync()).Trim();
        if (btn == "Generate" && await page.Locator(".vid-result").IsVisibleAsync()) break;
        await page.WaitForTimeoutAsync(1000);
    }
    if (!await page.Locator(".vid-result").IsVisibleAsync()) { Console.WriteLine("    TIMEOUT: no result"); return false; }
    await page.WaitForTimeoutAsync(2500);
    await page.Locator(".vid-result").ScreenshotAsync(new() { Path = Path.Combine(outDir, $"{step}-result.png") });
    Console.WriteLine($"    result: {await InfoAsync()} (preview seen: {sawPreview}) at {sw.Elapsed.TotalSeconds:F0} s");
    return sawPreview;
}

Console.WriteLine($"--- {url}/video");
await page.GotoAsync($"{url}/video", new() { Timeout = 120000 });
await page.Locator(".vid-prompt-input").First.WaitForAsync(new() { Timeout = 120000 });
if (modelServer.Length > 0)
{
    await page.Locator("details.vid-advanced summary").ClickAsync();
    await page.Locator("details.vid-advanced input.vid-prompt-input").Last.FillAsync(modelServer);
    await page.Locator("details.vid-advanced input.vid-prompt-input").Last.BlurAsync();
}

if (Environment.GetEnvironmentVariable("VIDEO_DRIVE_KEEP") == "1")
{
    // "Keep loaded" (LTX): the checkbox only exists once LTX is the model; the first example switches to it, so pick LTX here first
    await page.Locator("select.vid-select").First.SelectOptionAsync("ltx");
    await page.Locator("input.vid-keep").CheckAsync();
    Console.WriteLine("    Keep loaded: on");
}
Console.WriteLine("--- step 1: example 'Animate the kitten' (LTX image to video, live preview)");
await page.GetByRole(AriaRole.Button, new() { Name = "Animate the kitten" }).ClickAsync();
if (!await WaitResultAsync("1-kitten", 30)) failed++;
if (Environment.GetEnvironmentVariable("VIDEO_DRIVE_TIMING") == "1")
{
    // the same example again: the prompt's encoding is cached now, so this run shows what the rest of a clip costs
    Console.WriteLine("--- timing: the kitten again (cached prompt)");
    await page.GetByRole(AriaRole.Button, new() { Name = "Animate the kitten" }).ClickAsync();
    if (!await WaitResultAsync("1b-kitten", 30)) failed++;
    Console.WriteLine($"=== timing run done: {failed} failed");
    return failed;
}
var info1 = await InfoAsync();
int frames1 = int.TryParse(Regex.Match(info1, @"^(\d+) frames").Groups[1].Value, out var f1) ? f1 : 0;

Console.WriteLine("--- step 2: Extend this clip");
await page.GetByRole(AriaRole.Button, new() { Name = "Extend this clip" }).ClickAsync();
await page.Locator("textarea.vid-prompt-input").FillAsync("The kitten stands up, stretches and walks toward the camera, soft window light.");
await page.Locator(".vid-btn").ClickAsync();
if (!await WaitResultAsync("2-extend", 30)) failed++;
var info2 = await InfoAsync();
int frames2 = int.TryParse(Regex.Match(info2, @"^(\d+) frames").Groups[1].Value, out var f2) ? f2 : 0;
if (!(frames2 > frames1)) { Console.WriteLine($"    FAIL: extending did not add frames ({frames1} -> {frames2})"); failed++; }
else Console.WriteLine($"    extended {frames1} -> {frames2} frames");

Console.WriteLine("--- step 3: example 'Keyframes: fly into the house' (two keyframes)");
await page.GetByRole(AriaRole.Button, new() { Name = "Keyframes: fly into the house" }).ClickAsync();
if (!await WaitResultAsync("3-keyframes", 30)) failed++;

Console.WriteLine("--- step 4: example 'A story in three shots' (each shot extends the clip)");
await page.GetByRole(AriaRole.Button, new() { Name = "A story in three shots" }).ClickAsync();
if (!await WaitResultAsync("4-story", 45)) failed++;
var info4 = await InfoAsync();
int frames4 = int.TryParse(Regex.Match(info4, @"^(\d+) frames").Groups[1].Value, out var f4) ? f4 : 0;
if (frames4 != 97 + 2 * 80) { Console.WriteLine($"    FAIL: a three-shot story should be 257 frames, got {frames4}"); failed++; }

File.WriteAllLines(Path.Combine(outDir, "console.log"), log);
Console.WriteLine($"=== {failed} failed step(s) in {sw.Elapsed.TotalSeconds:F0} s; screenshots in {outDir}");
return failed;
