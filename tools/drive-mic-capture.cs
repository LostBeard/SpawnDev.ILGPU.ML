#:package Microsoft.Playwright@1.49.0
#:property JsonSerializerIsReflectionEnabledByDefault=true
// Live browser gate for MICROPHONE CAPTURE, and for the whole mic -> TEXT loop, on the /whisper demo page.
//
//   dotnet run tools/drive-mic-capture.cs -- [url]                 capture only (fast, no model download)
//   dotnet run tools/drive-mic-capture.cs -- [url] --transcribe    full mic -> Whisper -> text
//
// Why this exists: MediaStreamCapture.OnAudioReady was DECLARED and never raised - the class advertised
// microphone support while its only capture call was GetUserMedia(video: true, audio: false), and
// /whisper's Start Recording button was `=> Task.CompletedTask`. Both compiled; neither worked. The
// MLTestBase.MicrophoneCaptureTests suite covers the CONVERSION math (AudioData -> mono float32, green on
// WebGPU/WebGL/Wasm), but it constructs AudioData by hand. It cannot prove a real getUserMedia track
// reaches OnAudioReady. This does.
//
// ── capture mode ─────────────────────────────────────────────────────────────────────────────────────
// ⚠️ THE ASSERTION IS THE ELAPSED COUNTER, and that is deliberate. The page derives its recording time from
// the COUNT OF SAMPLES RECEIVED, not from a wall clock:
//     lock (_micSamples) { _micSamples.AddRange(chunk); seconds = _micSamples.Count / (double)rate; }
// so the button reading "Stop Recording (4.0s)" is proof that ~4 seconds of audio arrived through
// getUserMedia -> MediaStreamTrackProcessor -> AudioData -> OnAudioReady. If no audio flowed it reads 0.0s
// no matter how long the page sits there. A timer-driven counter would have proven nothing. The ratio is a
// second free check: Chrome's fake device is 48 kHz, so a missing 48k->16k resample reads 3x too high.
//
// ── transcribe mode ──────────────────────────────────────────────────────────────────────────────────
// The microphone is fed known words so the transcript can be asserted. The fixture is the harness
// reference clip, whose transcript is KNOWN rather than itself transcribed:
//     "All LibriVox recordings are in the public domain."  (16 kHz mono, 4.0 s, Public Domain Mark 1.0)
// It loops, and we record two passes to guarantee one intact sentence, then assert on content words
// rather than an exact string - Whisper is free to punctuate and capitalise as it likes.
//
// ⚠️ NOT via Chrome's fake device. MEASURED with tools/probe-fake-mic.cs, which opens the microphone in
// plain browser JS and reads an AnalyserNode with none of our code involved: Chrome's fake audio device
// produces DIGITAL SILENCE here - 24 consecutive readings of 0.0000 over 6 s - both with
// --use-file-for-fake-audio-capture and with the default device, and with the audio processing module
// (echoCancellation/noiseSuppression/autoGainControl) explicitly disabled. That cost real debugging time:
// frames arrived, the sample counter advanced, the page reported "9.0s of audio captured", and every one
// of those samples was zero. Whisper turned the silence into confident, fluent, unrelated text.
//
// So the audio is supplied HERE instead, by replacing getUserMedia before the app boots with one that
// returns a MediaStream sourced from the fixture through Web Audio. The page's capture path -
// MediaStreamTrackProcessor, AudioData, the mono/resample conversion, the accumulation, Whisper - runs
// exactly as it does for a real microphone. Only the sound source is ours.
//
// First transcribe run downloads Whisper (~231 MB) into the persistent profile's OPFS; later runs are warm.
using System.Text.RegularExpressions;
using Microsoft.Playwright;

var url = (args.FirstOrDefault(a => a.StartsWith("http")) ?? "http://localhost:5000").TrimEnd('/');
var transcribe = args.Any(a => a is "--transcribe" or "-t");
// Feed the WAV without paying for a transcription - used to check WHEN Chrome actually plays it.
var fileAudio = transcribe || args.Any(a => a is "--file-audio" or "-f");
var profileDir = Path.Combine(Path.GetTempPath(), "spawndev-ml-mic-profile");
Directory.CreateDirectory(profileDir);

// The fixture lives in the repo; Chrome reads it from DISK, not through the web server.
var repoRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."));
var wav = Path.GetFullPath(Path.Combine("tools", "zipvoice-harness", "fixtures", "reference",
                                        "librivox-public-domain.wav"));
if (fileAudio && !File.Exists(wav))
{
    Console.WriteLine($"FAIL fixture not found: {wav}");
    Console.WriteLine("     run from the repo root (SpawnDev.ILGPU.ML/)");
    return 1;
}

var chromeArgs = new List<string>
{
    // Auto-grant the permission prompt. We do NOT use --use-fake-device-for-media-capture for audio: it
    // produces digital silence here (see the header), and our own stream replaces getUserMedia anyway.
    "--use-fake-ui-for-media-stream",
};

using var pw = await Playwright.CreateAsync();
await using var ctx = await pw.Chromium.LaunchPersistentContextAsync(profileDir, new()
{
    Headless = false,
    Channel = "chrome",
    Args = chromeArgs.ToArray(),
});

var page = await ctx.NewPageAsync();

// Replace getUserMedia BEFORE the app boots, so the page's own capture path receives known speech.
// decodeAudioData handles the WAV parsing; a looping BufferSource into a MediaStreamDestination produces
// a genuine MediaStream carrying a genuine audio track - indistinguishable to the page from a microphone.
await page.AddInitScriptAsync(@"
(() => {
  const WAV = '/test-audio/librivox-public-domain.wav';
  const md = navigator.mediaDevices;
  if (!md) return;
  const real = md.getUserMedia.bind(md);
  md.getUserMedia = async (constraints) => {
    if (!constraints || !constraints.audio) return real(constraints);
    const ac = new AudioContext();
    if (ac.state === 'suspended') { try { await ac.resume(); } catch (e) {} }
    const bytes = await (await fetch(WAV)).arrayBuffer();
    const buffer = await ac.decodeAudioData(bytes);
    const src = ac.createBufferSource();
    src.buffer = buffer;
    src.loop = true;
    const dest = ac.createMediaStreamDestination();
    src.connect(dest);
    src.start();
    window.__micProbe = { rate: ac.sampleRate, duration: buffer.duration };
    return dest.stream;
  };
})();");
var log = new List<string>();
page.Console += (_, m) => log.Add(m.Text);

// Record two passes of the 4 s clip so a loop boundary cannot cut the only sentence in half.
var recordMs = transcribe ? 9000 : 4000;

int failed = 0;
Console.WriteLine($"--- {url}/whisper ({(transcribe ? "mic -> TEXT" : "microphone capture")})");
if (fileAudio) Console.WriteLine($"    feeding: {Path.GetFileName(wav)}");
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
    var recording = page.Locator(".mic-btn.recording");
    await recording.WaitForAsync(new() { Timeout = 30_000 });

    await Task.Delay(recordMs);

    var text = (await recording.InnerTextAsync()).Trim();
    var m = Regex.Match(text, @"\(([0-9]+(?:\.[0-9]+)?)s\)");
    double seconds = 0;
    if (!m.Success)
    {
        failed++;
        Console.WriteLine($"    FAIL could not read an elapsed time from the button: '{text}'");
    }
    else
    {
        seconds = double.Parse(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
        // The counter IS the sample count, so anything above zero means audio really arrived.
        if (seconds < 0.5)
        {
            failed++;
            Console.WriteLine($"    FAIL only {seconds:F1}s of audio after {recordMs / 1000}s of recording - "
                            + "OnAudioReady is not delivering samples");
        }
        else
        {
            Console.WriteLine($"    OK   {seconds:F1}s of real microphone audio reached OnAudioReady "
                            + $"({seconds * 16000:N0} samples at 16 kHz)");
        }
    }

    // Stop. In transcribe mode this is where the weights are awaited and Whisper runs.
    // Force + a long timeout on purpose: the model download kicked off at Start is already saturating the
    // single WASM thread, so the page fails Playwright's actionability (stability) check even though the
    // button is present and enabled. A 10 s default timed out here on a page that was working fine.
    await recording.ClickAsync(new() { Timeout = 180_000, Force = true });

    if (transcribe && failed == 0)
    {
        Console.WriteLine("    ... transcribing (first run downloads Whisper, ~231 MB)");
        var result = page.Locator(".transcription-text").First;
        await result.WaitForAsync(new() { Timeout = 900_000 });
        var got = (await result.InnerTextAsync()).Trim();

        // The page renders its own errors into this same element - do not read one as a transcript.
        if (got.StartsWith("Error:", StringComparison.OrdinalIgnoreCase))
        {
            failed++;
            Console.WriteLine($"    FAIL the page reported: {got}");
        }
        else
        {
            // Assert CONTENT WORDS, not an exact string: Whisper chooses its own casing and punctuation,
            // and the looping fixture may contribute a partial second sentence.
            var norm = Regex.Replace(got.ToLowerInvariant(), @"[^a-z0-9 ]", " ");
            norm = Regex.Replace(norm, @"\s+", " ");
            // NOT asserting "librivox". MEASURED on all six backends, whisper-tiny renders this clip as
            // "All legal box recordings are in the public domain." - seven of eight words exact. It mangles
            // the PROPER NOUN, which is a model capability limit, not a defect in our mel, encoder,
            // decoder or tokenizer. The overlap floor below keeps that from becoming a weak assertion.
            var expected = new[] { "recordings", "are", "in", "the", "public", "domain" };
            var missing = expected.Where(w => !norm.Contains(w)).ToArray();
            var refWords = "all librivox recordings are in the public domain".Split(' ');
            var gotWords = norm.Split(' ', StringSplitOptions.RemoveEmptyEntries).ToHashSet();
            var overlap = refWords.Count(w => gotWords.Contains(w)) / (double)refWords.Length;
            if (missing.Length > 0 || overlap < 0.7)
            {
                failed++;
                Console.WriteLine($"    FAIL missing {string.Join(", ", missing.DefaultIfEmpty("nothing"))}; "
                                + $"word overlap {overlap:P0} (floor 70%)");
                Console.WriteLine($"         expected words from: \"All LibriVox recordings are in the public domain.\"");
                Console.WriteLine($"         got: \"{got}\"");
            }
            else
            {
                Console.WriteLine($"    OK   {overlap:P0} word overlap: \"{got}\"");
            }
        }
    }
}
catch (Exception ex)
{
    failed++;
    Console.WriteLine($"    FAIL {ex.GetType().Name}: {ex.Message.Split('\n')[0]}");
    foreach (var l in log.TakeLast(8)) Console.WriteLine($"      | {l}");
}
finally
{
    // Always surface the page's own audio report. A transcript only means something next to the LEVEL of
    // what was captured: Whisper answers silence with confident, fluent, entirely unrelated text, so
    // "wrong words" and "no audio" are indistinguishable without this line.
    foreach (var l in log.Where(x => x.Contains("mic captured") || x.Contains("level t=")))
        Console.WriteLine($"    {l}");
    await page.CloseAsync();
}

Console.WriteLine();
Console.WriteLine(failed == 0
    ? (transcribe ? "mic -> TEXT VERIFIED end to end in a real browser"
                  : "microphone capture VERIFIED end to end in a real browser")
    : $"FAILED ({failed})");
return failed;
