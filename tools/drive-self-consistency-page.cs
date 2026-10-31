#:package Microsoft.Playwright@1.49.0
#:property JsonSerializerIsReflectionEnabledByDefault=true
// Live browser check of /self-consistency (SpawnDev.ILGPU.ML.Demo/Pages/SelfConsistencyPage.razor): picks a model, asks one of the example problems N times and
// asserts on what renders: N answer cards, a vote with at least one group, the throughput line. A screenshot of the finished page goes to the out dir.
//
//   dotnet run tools/drive-self-consistency-page.cs -- [demo url] [model index: 0 = Qwen2.5-Math 1.5B, 1 = Qwen2.5 0.5B] [example index] [answers] [out dir]
//   e.g.  node tools/serve-demo.mjs <published wwwroot> 5071
//         dotnet run tools/drive-self-consistency-page.cs -- http://localhost:5071 1 1 8
//
// Uses TJ's installed Chrome with a persistent profile (real WebGPU adapter; models cache in OPFS across runs), like drive-video-page.cs. Exit code = failed checks.
using System.Text.RegularExpressions;
using Microsoft.Playwright;

var url = (args.Length > 0 ? args[0] : "http://localhost:5071").TrimEnd('/');
int model = args.Length > 1 ? int.Parse(args[1]) : 1, example = args.Length > 2 ? int.Parse(args[2]) : 1, answers = args.Length > 3 ? int.Parse(args[3]) : 8;
var outDir = args.Length > 4 ? args[4] : Path.Combine(Path.GetTempPath(), "self-consistency-shots");
Directory.CreateDirectory(outDir);
var profileDir = Path.Combine(Path.GetTempPath(), "spawndev-ml-pages-profile");
Directory.CreateDirectory(profileDir);

using var pw = await Playwright.CreateAsync();
await using var ctx = await pw.Chromium.LaunchPersistentContextAsync(profileDir, new() { Headless = false, Channel = "chrome", ViewportSize = new() { Width = 1280, Height = 1200 } });
var page = ctx.Pages.Count > 0 ? ctx.Pages[0] : await ctx.NewPageAsync();
page.Console += (_, m) => { if (m.Text.Contains("[SelfConsistency]") || m.Type == "error") Console.WriteLine($"    console {m.Type}: {m.Text[..Math.Min(300, m.Text.Length)]}"); };
int failed = 0;
void Check(bool ok, string what) { Console.WriteLine($"  {(ok ? "ok  " : "FAIL")} {what}"); if (!ok) failed++; }
var sw = System.Diagnostics.Stopwatch.StartNew();

await page.GotoAsync(url + "/self-consistency", new() { WaitUntil = WaitUntilState.NetworkIdle, Timeout = 120000 });
await page.Locator(".models li").Nth(model).ClickAsync(new() { Timeout = 120000 });
Console.WriteLine($"  model {model} picked; loading...");
// ready = the problem box is there (or an error panel)
await page.Locator(".problem textarea, .error").First.WaitForAsync(new() { Timeout = 30 * 60000 });
if (await page.Locator(".error").CountAsync() > 0) { Check(false, "load: " + (await page.Locator(".error pre").InnerTextAsync())); return failed; }
Console.WriteLine($"  loaded in {sw.Elapsed.TotalSeconds:F0} s");
await page.Locator(".examples .chip").Nth(example).ClickAsync();
await page.Locator(".controls select").SelectOptionAsync(answers.ToString());
await page.Locator(".controls input").Nth(2).FillAsync("512");                 // max tokens
var t0 = sw.Elapsed;
await page.Locator(".controls button.primary").ClickAsync();
await page.Locator("button.primary.stop").WaitForAsync(new() { Timeout = 60000 });
// a live screenshot while the answers stream
await page.WaitForTimeoutAsync(15000);
await page.ScreenshotAsync(new() { Path = Path.Combine(outDir, "streaming.png"), FullPage = true });
await page.Locator("button.primary.stop").WaitForAsync(new() { State = WaitForSelectorState.Detached, Timeout = 30 * 60000 });
Console.WriteLine($"  answered in {(sw.Elapsed - t0).TotalSeconds:F0} s");
await page.ScreenshotAsync(new() { Path = Path.Combine(outDir, "done.png"), FullPage = true });

int cards = await page.Locator(".grid .card").CountAsync();
Check(cards == answers, $"{cards} answer cards (want {answers})");
var groups = await page.Locator(".vote .group").AllInnerTextsAsync();
Check(groups.Count > 0, $"vote groups: {string.Join(" | ", groups.Select(g => Regex.Replace(g, @"\s+", " ")))}");
var stats = await page.Locator(".vote .stats").InnerTextAsync();
var m = Regex.Match(stats, @"([\d.]+) tok/s");
Check(m.Success && double.Parse(m.Groups[1].Value) > 0, "throughput: " + stats);
var boxes = await page.Locator(".card-head .box").AllInnerTextsAsync();
Console.WriteLine("  boxed: " + string.Join(", ", boxes));
Console.WriteLine($"  screenshots in {outDir}; {failed} failed");
return failed;
