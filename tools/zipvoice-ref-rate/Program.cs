// DOES SILENCE IN THE REFERENCE CLIP MAKE THE CLONE SPEAK SLOWLY? Measure it.
//
//   dotnet run --project tools/zipvoice-ref-rate -c Release              # encoder only, seconds
//   dotnet run --project tools/zipvoice-ref-rate -c Release -- --render  # + full synthesis, writes wavs
//
// WHY THIS EXISTS. The hands-free demo cloned the Captain's voice and spoke back at roughly a third of
// natural speed - "something out of a sci-fi movie", not a voice. The console arithmetic pointed at the
// speaking RATE, but arithmetic over two console lines is a hypothesis, not a mechanism.
//
// The mechanism is in text_encoder.onnx and it is not subtle. Walking the forward cone of the
// `prompt_features_len` input:
//
//     Cast -> Div(by len(prompt_tokens)) -> Mul(by len(prompt_tokens)+len(tokens)) -> Div(by speed) -> Ceil
//
// So the model measures FRAMES PER PROMPT TOKEN from the reference clip and multiplies it by the total
// token count. That ratio IS the whole duration prediction. It is also, deliberately, how ZipVoice clones
// delivery - copying the rate is the feature, not a bug.
//
// The defect is that `prompt_features_len` counts every mel frame of the reference, and silence has mel
// frames too. A reference whose span is half silence declares a rate half of what the speaker actually
// used, and every generated syllable is stretched to match. Nothing downstream can notice: the tensor
// shapes are right, the audio is speech, and the only symptom is that it sounds wrong.
//
// This measures the effect rather than arguing about it. One encoder run per variant is enough for the
// arithmetic - NumFrames IS the prediction - and --render adds the ear.
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.CPU;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;

bool render = args.Contains("--render");
// Rendering every variant is minutes of ODE for rows whose arithmetic already agrees. --only <substring>
// renders just the ones an ear needs, which in practice is the defect and the fix side by side.
var only = Array.IndexOf(args, "--only") is int oi && oi >= 0 && oi + 1 < args.Length ? args[oi + 1] : null;
var outDir = Path.Combine(Path.GetTempPath(), "zipvoice-ref-rate");

var repoRoot = @"D:\users\tj\Projects\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML";
var www = Path.Combine(repoRoot, "SpawnDev.ILGPU.ML.Demo", "wwwroot");
var vadModelPath = Path.Combine(www, "references", "vad", "silero_vad.onnx");

// The int8 package ships no vocoder, so the fp32 directory is the one with all three graphs.
var modelRoot = Environment.GetEnvironmentVariable("ZIPVOICE_MODELS")
    ?? @"D:\users\tj\Projects\SpawnDev.Reachy\SpawnDev.Reachy\models";
var fp32Dir = Path.Combine(modelRoot, "sherpa-onnx-zipvoice-distill-zh-en-emilia");

// The reference clip and its EXACT transcript. Anything in the audio and missing from the text bleeds
// into the start of the generated line, so this pairing is part of the fixture.
var refWav = Environment.GetEnvironmentVariable("ZIPVOICE_PROMPT_WAV")
    ?? Path.Combine(www, "test-audio", "librivox-public-domain.wav");
var refText = Environment.GetEnvironmentVariable("ZIPVOICE_PROMPT_TEXT")
    ?? "All LibriVox recordings are in the public domain.";

// A line long enough that a rate error is unmistakable, and fixed so every variant is comparable.
var line = Environment.GetEnvironmentVariable("ZIPVOICE_LINE")
    ?? "Paint the sockets in the wall dull green.";

foreach (var (what, path) in new[] { ("vad model", vadModelPath), ("reference wav", refWav) })
    if (!File.Exists(path)) { Console.WriteLine($"no {what} at {path}"); return 2; }
foreach (var f in new[] { "text_encoder.onnx", "fm_decoder.onnx", "vocos_24khz.onnx", "tokens.txt" })
    if (!File.Exists(Path.Combine(fp32Dir, f))) { Console.WriteLine($"no {f} in {fp32Dir}"); return 2; }

var refBytes = File.ReadAllBytes(refWav);
var refRaw = WavDecoder.DecodeWavFile(refBytes) ?? throw new Exception($"could not decode {refWav}");
int refRate = WavRate(refBytes);

Console.WriteLine($"reference : {Path.GetFileName(refWav)}  {refRaw.Length} samples @ {refRate} Hz "
                + $"= {refRaw.Length / (double)refRate:F2}s");
Console.WriteLine($"transcript: \"{refText}\"");
Console.WriteLine($"line      : \"{line}\"");
Console.WriteLine();

// -- accelerator --------------------------------------------------------------------------------------
var mlBuilder = MLContext.Create();
await mlBuilder.AllAcceleratorsAsync();
using var mlCtx = mlBuilder.ToContext();
var want = Environment.GetEnvironmentVariable("ZIPVOICE_BACKEND")?.Trim().ToLowerInvariant();
using var accel = want switch
{
    "cuda" => mlCtx.GetCudaDevices()[0].CreateCudaAccelerator(mlCtx),
    "opencl" => mlCtx.GetCLDevices()[0].CreateCLAccelerator(mlCtx),
    "cpu" => mlCtx.GetCPUDevices()[0].CreateCPUAccelerator(mlCtx),
    _ => await mlCtx.CreatePreferredAcceleratorAsync() ?? throw new InvalidOperationException("no accelerator"),
};
Console.WriteLine($"device    : {accel.AcceleratorType} {accel.Name}\n");

// -- the graphs ---------------------------------------------------------------------------------------
using var graphs = IlgpuZipVoiceGraphs.Create(
    accel,
    File.ReadAllBytes(Path.Combine(fp32Dir, "text_encoder.onnx")),
    File.ReadAllBytes(Path.Combine(fp32Dir, "fm_decoder.onnx")),
    File.ReadAllBytes(Path.Combine(fp32Dir, "vocos_24khz.onnx")));
using var pipeline = new ZipVoicePipeline(graphs);
var config = pipeline.Config;

var tokenizer = ZipVoiceTokenizer.CreateFromTokens(File.ReadAllText(Path.Combine(fp32Dir, "tokens.txt")));
var tokens = tokenizer.Encode(line);
var promptTokens = tokenizer.Encode(refText);
Console.WriteLine($"tokens    : line {tokens.Length}, reference transcript {promptTokens.Length}\n");

// -- the trim under test ------------------------------------------------------------------------------
// Silero, not an energy gate: the reference comes off a microphone in a room, and an energy threshold
// tuned to this room is a threshold that fails in the next one. We already own the right instrument.
var vadBytes = File.ReadAllBytes(vadModelPath);

async Task<List<(int Start, int End)>> SegmentsAsync(float[] samples, int rate)
{
    var at16k = rate == SileroVad.SampleRate
        ? samples
        : AudioPreprocessor.Resample(samples, rate, SileroVad.SampleRate);

    using var vad = SileroVad.Create(accel, vadBytes);
    // Tight options on purpose: this is not endpointing a live turn, it is locating speech inside a clip
    // we already hold whole. Nothing here needs a grace period.
    var options = new VadOptions
    {
        Threshold = 0.5f,
        MinSilenceDuration = TimeSpan.FromMilliseconds(200),
        MinSpeechDuration = TimeSpan.FromMilliseconds(100),
        MaxSpeechDuration = TimeSpan.FromHours(1),
        SpeechPad = TimeSpan.FromMilliseconds(100),
    };
    using var detector = new VoiceActivityDetector(vad, options);
    var raw = new List<(long Start, long End)>();
    detector.OnSegment += seg => raw.Add((seg.StartSample, seg.StartSample + seg.Samples.Length));
    await detector.AcceptWaveformAsync(at16k);
    await detector.FlushAsync();

    double scale = rate / (double)SileroVad.SampleRate;
    var mapped = new List<(int, int)>();
    foreach (var (s0, e0) in raw)
    {
        int s = (int)Math.Clamp(s0 * scale, 0, samples.Length);
        int e = (int)Math.Clamp(e0 * scale, s, samples.Length);
        if (e > s) mapped.Add((s, e));
    }
    return mapped;
}

// Leading and trailing dead air only: first segment start to last segment end.
async Task<float[]> TrimAsync(float[] samples, int rate)
{
    var segs = await SegmentsAsync(samples, rate);
    if (segs.Count == 0) return samples;             // heard nothing: do not cut what we cannot explain
    int s = segs[0].Start, e = segs[^1].End;
    var trimmed = new float[e - s];
    Array.Copy(samples, s, trimmed, 0, trimmed.Length);
    return trimmed;
}

// Leading, trailing AND internal dead air, with internal gaps CAPPED rather than removed.
//
// ⚠️ Capped, not deleted, and the distinction is the whole design. A pause carries rhythm, and the model
// clones rhythm - splicing every gap to zero would clone a speaker who never breathes. But a two-second
// think in the middle of a two-word reference is not rhythm, it is dead air being counted as speech
// (which is precisely the shape of "hello ... hello"). Capping keeps the pause and drops the dead air.
async Task<float[]> GapCapAsync(float[] samples, int rate, double capSeconds = 0.20)
{
    var segs = await SegmentsAsync(samples, rate);
    if (segs.Count == 0) return samples;
    int cap = (int)(capSeconds * rate);

    var kept = new List<float>(samples.Length);
    for (int i = 0; i < segs.Count; i++)
    {
        var (s, e) = segs[i];
        if (i > 0)
        {
            int gap = Math.Min(s - segs[i - 1].End, cap);
            for (int g = 0; g < gap; g++) kept.Add(samples[segs[i - 1].End + g]);
        }
        for (int k = s; k < e; k++) kept.Add(samples[k]);
    }
    return kept.ToArray();
}

// A reference with dead air in the MIDDLE - "hello ... hello", the shape the Captain actually spoke.
// Spliced at the quietest 10 ms frame so the insert does not land inside a word.
static float[] InsertInternalSilence(float[] samples, int rate, double seconds)
{
    int win = Math.Max(1, rate / 100);
    int frames = samples.Length / win;
    int quietest = frames / 2;
    double best = double.MaxValue;
    // Only look in the middle half: a split at the very edge is a leading/trailing pad, not an internal one.
    for (int f = frames / 4; f < frames * 3 / 4; f++)
    {
        double sum = 0;
        for (int i = f * win; i < (f + 1) * win; i++) sum += (double)samples[i] * samples[i];
        if (sum < best) { best = sum; quietest = f; }
    }
    int at = quietest * win;
    var outp = new float[samples.Length + (int)(seconds * rate)];
    Array.Copy(samples, 0, outp, 0, at);
    Array.Copy(samples, at, outp, at + (int)(seconds * rate), samples.Length - at);
    return outp;
}

static float[] Pad(float[] samples, int rate, double lead, double trail)
{
    int l = (int)(lead * rate), t = (int)(trail * rate);
    var padded = new float[l + samples.Length + t];
    Array.Copy(samples, 0, padded, l, samples.Length);
    return padded;
}

// The candidate LIBRARY default. Silero is the better instrument but it is a 643 KB model, and a TTS
// pipeline that cannot clone correctly without a second model download is a pipeline every consumer will
// get wrong. This gate is self-contained and RELATIVE to the clip's own loudness, so it does not carry a
// threshold tuned to one room into the next one.
static float[] EnergyTrim(float[] samples, int rate, double dbBelowPeak = 35, double hangoverMs = 60)
{
    if (samples.Length == 0) return samples;
    int win = Math.Max(1, rate / 100);                 // 10 ms frames
    int frames = samples.Length / win;
    if (frames < 3) return samples;

    var rms = new double[frames];
    for (int f = 0; f < frames; f++)
    {
        double sum = 0;
        for (int i = f * win; i < (f + 1) * win; i++) sum += (double)samples[i] * samples[i];
        rms[f] = Math.Sqrt(sum / win);
    }

    // The loudest frame, not the loudest SAMPLE: one click should not set the reference level.
    double peak = 0;
    for (int f = 0; f < frames; f++) if (rms[f] > peak) peak = rms[f];
    if (peak <= 0) return samples;
    double gate = peak * Math.Pow(10, -dbBelowPeak / 20.0);

    int first = -1, last = -1;
    for (int f = 0; f < frames; f++) if (rms[f] >= gate) { if (first < 0) first = f; last = f; }
    if (first < 0) return samples;

    int hang = (int)(hangoverMs / 10);                 // frames
    int s = Math.Max(0, first - hang) * win;
    int e = Math.Min(frames, last + 1 + hang) * win;
    var trimmed = new float[e - s];
    Array.Copy(samples, s, trimmed, 0, trimmed.Length);
    return trimmed;
}

// -- the variants -------------------------------------------------------------------------------------
var pad1 = Pad(refRaw, refRate, 1.0, 1.0);
var pad2 = Pad(refRaw, refRate, 2.0, 2.0);
var inner = InsertInternalSilence(refRaw, refRate, 2.0);

// The variants below hand in audio that has ALREADY been trimmed (or deliberately not), so the pipeline
// must not trim it again - otherwise every "untrimmed" row would silently be a trimmed one and the table
// would prove nothing.
pipeline.TrimReferenceSilence = false;
float[] Gate(float[] a) => ZipVoiceFeatures.TrimReferenceSilence(a, refRate);

var variants = new List<(string Name, float[] Audio)>
{
    ("raw", refRaw),
    ("raw, silero", await TrimAsync(refRaw, refRate)),
    ("raw, energy", EnergyTrim(refRaw, refRate)),

    ("+1.0s both", pad1),
    ("+1.0s both, silero", await TrimAsync(pad1, refRate)),
    ("+1.0s both, energy", EnergyTrim(pad1, refRate)),

    ("+2.0s both", pad2),
    ("+2.0s both, silero", await TrimAsync(pad2, refRate)),
    ("+2.0s both, energy", EnergyTrim(pad2, refRate)),

    // The "hello ... hello" shape: dead air in the MIDDLE, which an end-trim cannot touch.
    ("+2.0s INSIDE", inner),
    ("+2.0s INSIDE, silero", await TrimAsync(inner, refRate)),
    ("+2.0s INSIDE, energy", EnergyTrim(inner, refRate)),
    ("+2.0s INSIDE, gapcap", await GapCapAsync(inner, refRate)),
    ("raw, gapcap", await GapCapAsync(refRaw, refRate)),

    // The SHIPPED gate, called through the library so this measures production code and not a copy of it.
    ("raw, GATE", Gate(refRaw)),
    ("+2.0s both, GATE", Gate(pad2)),
    ("+2.0s INSIDE, GATE", Gate(inner)),
    ("+2s both +2s IN, GATE", Gate(InsertInternalSilence(pad2, refRate, 2.0))),
};

Console.WriteLine($"{"variant",-22} {"ref s",7} {"promptF",8} {"f/token",8} {"totalF",7} "
                + $"{"genF",6} {"gen s",7} {"tok/s",7}");
Console.WriteLine(new string('-', 82));

if (render) Directory.CreateDirectory(outDir);
var rendered = new List<(string Name, string Wav, double Seconds, double RefSeconds)>();

foreach (var (name, audio) in variants)
{
    // Exactly what the pipeline does: the deliberate 0.25 s tail pad, then the mel.
    var padded = new float[audio.Length + (int)(pipeline.ReferenceTailSilenceSeconds * refRate)];
    Array.Copy(audio, padded, audio.Length);
    var promptFeatures = ZipVoiceFeatures.ComputePromptFeatures(padded, refRate, config, out int promptFrames);

    var encoding = await graphs.RunEncoderAsync(tokens, promptTokens, promptFrames, config.Speed);
    int genFrames = encoding.NumFrames - promptFrames;
    double genSeconds = genFrames * config.HopLength / (double)config.SampleRate;

    Console.WriteLine($"{name,-22} {audio.Length / (double)refRate,7:F2} {promptFrames,8} "
                    + $"{promptFrames / (double)promptTokens.Length,8:F3} {encoding.NumFrames,7} "
                    + $"{genFrames,6} {genSeconds,7:F2} {tokens.Length / genSeconds,7:F2}");

    if (render && (only == null || name.Contains(only, StringComparison.OrdinalIgnoreCase)))
    {
        var result = await pipeline.SpeakAsync(line, refText, audio, refRate, tokenizer);
        var wav = Path.Combine(outDir,
            name.Replace(' ', '_').Replace(",", "").Replace('.', '-') + ".wav");
        File.WriteAllBytes(wav, EncodeWav(result.Audio, result.SampleRate));
        rendered.Add((name, wav, result.DurationSeconds, audio.Length / (double)refRate));
        Console.WriteLine($"{"",-22} rendered {result.DurationSeconds:F2}s -> {wav}");
    }
}

if (rendered.Count > 0)
{
    // ⚠️ ONE file, audio embedded as data URIs. A folder of wavs is not a comparison anybody can make -
    // the difference here is a RATE, and hearing a rate difference needs the two clips one click apart.
    var page = Path.Combine(outDir, "listen.html");
    File.WriteAllText(page, BuildPage(line, refText, rendered));
    Console.WriteLine();
    Console.WriteLine($"listen   : {page}");
}

Console.WriteLine();
Console.WriteLine("READ IT THIS WAY: `f/token` is the rate the model copies. If padding the reference with");
Console.WriteLine("silence moves it, then silence in a reference clip IS a speaking-rate error, and trimming");
Console.WriteLine("is not a cosmetic tidy - it is the difference between a clone and a slur.");
return 0;

// -- local helpers ------------------------------------------------------------------------------------
static string BuildPage(string line, string refText,
                        List<(string Name, string Wav, double Seconds, double RefSeconds)> rows)
{
    var sb = new System.Text.StringBuilder();
    sb.Append("<!doctype html><meta charset=\"utf-8\"><title>ZipVoice reference silence</title>");
    sb.Append("<style>body{font:15px/1.5 system-ui,sans-serif;max-width:52rem;margin:2rem auto;padding:0 1rem}");
    sb.Append("h1{font-size:1.3rem}code{background:#eee;padding:.1em .3em;border-radius:3px}");
    sb.Append("tr>*{text-align:left;padding:.5rem .75rem;border-bottom:1px solid #ddd;vertical-align:middle}");
    sb.Append("table{border-collapse:collapse;width:100%}audio{width:17rem}</style>");
    sb.Append("<h1>ZipVoice clones the reference clip&rsquo;s speaking rate</h1>");
    sb.Append("<p>The encoder derives frames-per-token from the reference and multiplies it by the total ");
    sb.Append("token count, so silence in the reference is indistinguishable from slow speech. Every row ");
    sb.Append("below speaks the <em>same line</em> from the <em>same voice</em> &mdash; only the amount of ");
    sb.Append("dead air in the reference differs.</p>");
    sb.Append($"<p>Line: <code>{System.Net.WebUtility.HtmlEncode(line)}</code><br>");
    sb.Append($"Reference transcript: <code>{System.Net.WebUtility.HtmlEncode(refText)}</code></p>");
    sb.Append("<table><tr><th>reference</th><th>ref length</th><th>spoken</th><th>listen</th></tr>");
    foreach (var (name, wav, seconds, refSeconds) in rows)
    {
        var b64 = Convert.ToBase64String(File.ReadAllBytes(wav));
        sb.Append($"<tr><td>{System.Net.WebUtility.HtmlEncode(name)}</td>");
        sb.Append($"<td>{refSeconds:F2}s</td><td><b>{seconds:F2}s</b></td>");
        sb.Append($"<td><audio controls preload=\"none\" src=\"data:audio/wav;base64,{b64}\"></audio></td></tr>");
    }
    sb.Append("</table>");
    return sb.ToString();
}

static int WavRate(byte[] wav)
{
    // Walk the chunk list rather than assuming byte 24: a WAV with a LIST chunk before `fmt ` is legal.
    for (int i = 12; i + 8 <= wav.Length;)
    {
        var id = System.Text.Encoding.ASCII.GetString(wav, i, 4);
        int size = BitConverter.ToInt32(wav, i + 4);
        if (id == "fmt ") return BitConverter.ToInt32(wav, i + 12);
        i += 8 + size + (size & 1);
    }
    throw new Exception("no fmt chunk");
}

static byte[] EncodeWav(float[] samples, int rate)
{
    var ms = new MemoryStream();
    var w = new BinaryWriter(ms);
    int dataBytes = samples.Length * 2;
    w.Write(System.Text.Encoding.ASCII.GetBytes("RIFF")); w.Write(36 + dataBytes);
    w.Write(System.Text.Encoding.ASCII.GetBytes("WAVEfmt ")); w.Write(16);
    w.Write((short)1); w.Write((short)1); w.Write(rate); w.Write(rate * 2);
    w.Write((short)2); w.Write((short)16);
    w.Write(System.Text.Encoding.ASCII.GetBytes("data")); w.Write(dataBytes);
    foreach (var s in samples) w.Write((short)(Math.Clamp(s, -1f, 1f) * 32767));
    w.Flush();
    return ms.ToArray();
}
