#:package Microsoft.Playwright@1.49.0
#:property JsonSerializerIsReflectionEnabledByDefault=true
// Live browser gate for MICROPHONE CAPTURE on the /whisper demo page.
//
//   dotnet run tools/drive-mic-capture.cs -- [url]
//
// Why this exists: MediaStreamCapture.OnAudioReady was DECLARED and never raised - the class advertised
// microphone support while its only capture call was GetUserMedia(video: true, audio: false), and
// /whisper's Start Recording button was `=> Task.CompletedTask`. Both compiled; neither worked. The
// MLTestBase.MicrophoneCaptureTests suite covers the CONVERSION math (AudioData -> mono float32, green on
// WebGPU/WebGL/Wasm), but it constructs AudioData by hand. It cannot prove that a real getUserMedia track
// actually reaches OnAudioReady. This does.
//
// ⚠️ THE ASSERTION IS THE ELAPSED COUNTER, and that is deliberate. The page derives its recording time from
// the COUNT OF SAMPLES RECEIVED, not from a wall clock:
//     lock (_micSamples) { _micSamples.AddRange(chunk); seconds = _micSamples.Count / (double)rate; }
// so the button reading "Stop Recording (2.4s)" is proof that ~2.4 seconds of audio arrived through
// getUserMedia -> MediaStreamTrackProcessor -> AudioData -> OnAudioReady. If no audio flowed it reads 0.0s
// no matter how long the page sits there. A timer-driven counter would have proven nothing.
//
// Chrome supplies a synthetic microphone under --use-fake-device-for-media-capture, and
// --use-fake-ui-for-media-stream auto-grants permission, so this needs no human and no real mic.
//
// Scope: this gates CAPTURE. It does not download Whisper or assert transcription text - a synthetic tone
// has no words in it. Transcription is covered by the pipeline tests.
using System.Text.RegularExpressions;
using Microsoft.Playwright;

var url = (args.FirstOrDefault(a => a.StartsWith("http")) ?? "http://localhost:5000").TrimEnd('/');
var profileDir = Path.Combine(Path.GetTempPath(), "spawndev-ml-mic-profile");
Directory.CreateDirectory(profileDir);

using var pw = await Playwright.CreateAsync();
await using var ctx = await pw.Chromium.LaunchPersistentContextAsync(profileDir, new()
{
    Headless = false,
    Channel = "chrome",
    Args = new[]
    {
        "--use-fake-device-for-media-capture",   // synthetic mic - a tone, but real audio frames
        "--use-fake-ui-for-media-stream",        // auto-grant, so no permission prompt blocks us
    },
});

var page = await ctx.NewPageAsync();
var log = new List<string>();
page.Console += (_, m) => log.Add(m.Text);

int failed = 0;
Console.WriteLine($"--- {url}/whisper (microphone capture)");
try
{
    await page.GotoAsync($"{url}/whisper", new() { WaitUntil = WaitUntilState.DOMContentLoaded, Timeout = 60_000 });

    // Wait for Blazor to render before querying anything.
    var micMode = page.Locator(".mode-btn", new() { HasTextString = "Microphone" });
    await micMode.WaitForAsync(new() { Timeout = 90_000 });
    await micMode.ClickAsync();

    var micBtn = page.Locator(".mic-btn");
    await micBtn.WaitForAsync(new() { Timeout = 15_000 });
    await micBtn.ClickAsync();

    // The button flips to the recording state only after StartMicrophoneAsync returns true.
    await page.Locator(".mic-btn.recording").WaitForAsync(new() { Timeout = 30_000 });

    // Let real audio frames accumulate.
    await Task.Delay(4000);

    var text = (await page.Locator(".mic-btn.recording").InnerTextAsync()).Trim();
    var m = Regex.Match(text, @"\(([0-9]+(?:\.[0-9]+)?)s\)");
    if (!m.Success)
    {
        failed++;
        Console.WriteLine($"    FAIL could not read an elapsed time from the button: '{text}'");
    }
    else
    {
        var seconds = double.Parse(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
        // The counter IS the sample count, so anything above zero means audio really arrived.
        if (seconds < 0.5)
        {
            failed++;
            Console.WriteLine($"    FAIL only {seconds:F1}s of audio after 4s of recording - "
                            + "OnAudioReady is not delivering samples");
        }
        else
        {
            Console.WriteLine($"    OK   {seconds:F1}s of real microphone audio reached OnAudioReady "
                            + $"({seconds * 16000:N0} samples at 16 kHz)");
        }
    }

    // Stop cleanly so the page releases the track.
    try { await page.Locator(".mic-btn.recording").ClickAsync(new() { Timeout = 5000 }); } catch { }
}
catch (Exception ex)
{
    failed++;
    Console.WriteLine($"    FAIL {ex.GetType().Name}: {ex.Message.Split('\n')[0]}");
    foreach (var l in log.TakeLast(6)) Console.WriteLine($"      | {l}");
}
finally
{
    await page.CloseAsync();
}

Console.WriteLine();
Console.WriteLine(failed == 0
    ? "microphone capture VERIFIED end to end in a real browser"
    : $"microphone capture FAILED ({failed})");
return failed;
