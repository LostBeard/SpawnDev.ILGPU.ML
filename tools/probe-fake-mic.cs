#:package Microsoft.Playwright@1.49.0
#:property JsonSerializerIsReflectionEnabledByDefault=true
// Does Chrome's FAKE microphone actually produce audio in this environment?
//
//   dotnet run tools/probe-fake-mic.cs -- [url] [--file-audio]
//
// Why: /whisper's microphone gate reported "9.0s of real microphone audio reached OnAudioReady" and the
// transcript came back "[BLANK_AUDIO]" with rms=0.0000. Frames were arriving; audio was not. Our own
// AudioData -> mono float32 conversion is covered by MLTestBase.MicrophoneCaptureTests (green on WebGPU,
// WebGL and Wasm) and those tests build the AudioData by hand, so they cannot tell us what a REAL
// getUserMedia track carries.
//
// This probe bypasses our C# entirely: plain browser JS opens the microphone, runs it through an
// AnalyserNode, and reports RMS per 250 ms. It answers one question and nothing else -
// IS CHROME'S FAKE DEVICE EMITTING SOUND? - so a silent result indicts the harness and a loud one indicts
// our capture path. Guessing between those two costs far more than writing this.
//
// It needs a secure context for getUserMedia, so it runs against the app's own origin (localhost counts).
using Microsoft.Playwright;

var url = (args.FirstOrDefault(a => a.StartsWith("http")) ?? "http://localhost:5000").TrimEnd('/');
var fileAudio = args.Any(a => a is "--file-audio" or "-f");
// Chrome runs captured audio through its audio processing module (echo cancellation, noise
// suppression, auto gain) BY DEFAULT. With a synthetic device there is no real playback reference,
// and the APM can suppress the whole signal to digital silence. --raw asks for the unprocessed track.
var raw = args.Any(a => a is "--raw" or "-r");
var wav = Path.GetFullPath(Path.Combine("tools", "zipvoice-harness", "fixtures", "reference",
                                        "librivox-public-domain.wav"));

var chromeArgs = new List<string>
{
    "--use-fake-device-for-media-capture",
    "--use-fake-ui-for-media-stream",
};
if (fileAudio)
{
    if (!File.Exists(wav)) { Console.WriteLine($"fixture missing: {wav}"); return 1; }
    chromeArgs.Add($"--use-file-for-fake-audio-capture={wav}");
}

Console.WriteLine(fileAudio ? $"--- fake device FED FROM {Path.GetFileName(wav)}" : "--- fake device DEFAULT");

using var pw = await Playwright.CreateAsync();
await using var ctx = await pw.Chromium.LaunchPersistentContextAsync(
    Path.Combine(Path.GetTempPath(), "spawndev-mic-probe-profile"),
    new() { Headless = false, Channel = "chrome", Args = chromeArgs.ToArray() });

var page = await ctx.NewPageAsync();
await page.GotoAsync($"{url}/mic-probe.html", new() { WaitUntil = WaitUntilState.Load, Timeout = 60_000 });

// Pure JS. Nothing of ours is involved past this point.
var constraints = raw
    ? "{ echoCancellation: false, noiseSuppression: false, autoGainControl: false }"
    : "true";
Console.WriteLine($"    audio constraints: {constraints}");
var js = @"async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: AUDIO_CONSTRAINTS });
        const track = stream.getAudioTracks()[0];
        const settings = track.getSettings ? track.getSettings() : {};
        const ac = new AudioContext();
        const src = ac.createMediaStreamSource(stream);
        const an = ac.createAnalyser();
        an.fftSize = 2048;
        src.connect(an);
        const buf = new Float32Array(an.fftSize);
        const out = [];
        for (let i = 0; i < 24; i++) {
            await new Promise(r => setTimeout(r, 250));
            an.getFloatTimeDomainData(buf);
            let sq = 0;
            for (let k = 0; k < buf.length; k++) sq += buf[k] * buf[k];
            out.push(Math.sqrt(sq / buf.length).toFixed(4));
        }
        track.stop();
        return JSON.stringify({ rate: ac.sampleRate, settings, rms: out });
    } catch (e) {
        return JSON.stringify({ error: String(e) });
    }
}";
var result = await page.EvaluateAsync<string>(js.Replace("AUDIO_CONSTRAINTS", constraints));

Console.WriteLine(result);
await page.CloseAsync();

// Any non-zero reading anywhere means the device is producing sound.
var loud = System.Text.RegularExpressions.Regex.Matches(result, @"""(0\.\d{4})""")
    .Select(m => double.Parse(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture))
    .DefaultIfEmpty(0).Max();
Console.WriteLine();
Console.WriteLine(loud > 0.001
    ? $"CHROME IS PRODUCING AUDIO (peak rms {loud:F4}) - so silence downstream is OUR capture path"
    : "CHROME IS PRODUCING SILENCE - the fake device is the problem, not our capture path");
return 0;
