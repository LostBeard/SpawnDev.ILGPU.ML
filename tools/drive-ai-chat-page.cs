#:package Microsoft.Playwright@1.49.0
#:property JsonSerializerIsReflectionEnabledByDefault=true
// Live browser check of /ai-chat (SpawnDev.ILGPU.ML.Demo/Pages/AiChatPage.razor): picks a model, sends a question and then a follow-up (the second turn reuses
// the first turn's cache rows), and asserts on what renders: two assistant answers, the engine named in the model bar, the page's [AiChat-Perf] console line
// (TTFT and tok/s) for each turn. A screenshot of the finished conversation goes to the out dir.
//
//   dotnet run tools/drive-ai-chat-page.cs -- [demo url] [model index: 0 = Qwen2.5 0.5B, 1 = SmolLM2 360M, 2 = Gemma 3 270M] [out dir]
//   e.g.  node tools/serve-demo.mjs <published wwwroot> 5071
//         dotnet run tools/drive-ai-chat-page.cs -- http://localhost:5071 0
//
// Uses TJ's installed Chrome with a persistent profile (real WebGPU adapter; models cache in OPFS across runs), like drive-self-consistency-page.cs.
// Exit code = failed checks.
using Microsoft.Playwright;

var url = (args.Length > 0 ? args[0] : "http://localhost:5071").TrimEnd('/');
int model = args.Length > 1 ? int.Parse(args[1]) : 0;
var outDir = args.Length > 2 ? args[2] : Path.Combine(Path.GetTempPath(), "ai-chat-shots");
Directory.CreateDirectory(outDir);
var profileDir = Path.Combine(Path.GetTempPath(), "spawndev-ml-pages-profile");
Directory.CreateDirectory(profileDir);

using var pw = await Playwright.CreateAsync();
await using var ctx = await pw.Chromium.LaunchPersistentContextAsync(profileDir, new() { Headless = false, Channel = "chrome", ViewportSize = new() { Width = 1280, Height = 1200 } });
var page = ctx.Pages.Count > 0 ? ctx.Pages[0] : await ctx.NewPageAsync();
var perf = new List<string>();
page.Console += (_, m) =>
{
    if (m.Text.Contains("[AiChat-Perf]")) perf.Add(m.Text);
    if (m.Text.Contains("[AiChat") || m.Type == "error") Console.WriteLine($"    console {m.Type}: {m.Text[..Math.Min(300, m.Text.Length)]}");
};
int failed = 0;
void Check(bool ok, string what) { Console.WriteLine($"  {(ok ? "ok  " : "FAIL")} {what}"); if (!ok) failed++; }
var sw = System.Diagnostics.Stopwatch.StartNew();

await page.GotoAsync(url + "/ai-chat", new() { WaitUntil = WaitUntilState.NetworkIdle, Timeout = 120000 });
await page.Locator(".models li").Nth(model).ClickAsync(new() { Timeout = 120000 });
Console.WriteLine($"  model {model} picked; loading...");
await page.Locator(".composer textarea, .error").First.WaitForAsync(new() { Timeout = 30 * 60000 });
if (await page.Locator(".error").CountAsync() > 0) { Check(false, "load: " + await page.Locator(".error").InnerTextAsync()); return failed; }
Console.WriteLine($"  loaded in {sw.Elapsed.TotalSeconds:F0} s");
var bar = await page.Locator(".model-tag").InnerTextAsync();
Console.WriteLine($"  model bar: {bar}");
Check(model == 2 ? bar.Contains("Generator") : bar.Contains("BatchDecoder"), $"engine shown in the bar ({(model == 2 ? "Generator: gemma3" : "BatchDecoder")})");

async Task Ask(string text, int turn)
{
    int perfBefore = perf.Count;
    await page.Locator(".composer textarea").FillAsync(text);
    await page.Locator(".composer button.primary").ClickAsync();
    // done = the turn's assistant message is in the transcript and the composer's Send button is back
    await page.Locator(".transcript .msg.assistant").Nth(turn).WaitForAsync(new() { Timeout = 10 * 60000 });
    await page.Locator(".composer button.primary:not(.stop)").WaitForAsync(new() { Timeout = 10 * 60000 });
    var answer = await page.Locator(".transcript .msg.assistant .text").Nth(turn).InnerTextAsync();
    Console.WriteLine($"  turn {turn + 1}: {answer.Replace('\n', ' ')[..Math.Min(160, answer.Length)]}");
    Check(answer.Trim().Length > 10, $"turn {turn + 1} answered");
    await Task.Delay(500);
    Check(perf.Count > perfBefore, $"turn {turn + 1} perf line: {(perf.Count > perfBefore ? perf[^1] : "none")}");
}
await Ask("Write a short paragraph about why the sky is blue.", 0);
await Ask("Now explain it to a five year old in two sentences.", 1);
await page.ScreenshotAsync(new() { Path = Path.Combine(outDir, $"ai-chat-model{model}.png"), FullPage = true });
Console.WriteLine($"  screenshot: {Path.Combine(outDir, $"ai-chat-model{model}.png")}; {failed} failed; {sw.Elapsed.TotalSeconds:F0} s");
return failed;
