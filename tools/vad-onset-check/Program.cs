// HOW MUCH LEAD-IN DOES WHISPER NEED? Sweep VadOptions.SpeechPad and count the words it costs.
//
//   dotnet run --project tools/vad-onset-check -c Release [-- <file.wav>]
//
// VAD_BACKEND=cuda|opencl|cpu pins the backend instead of taking the preferred one.
//
// WHY THIS EXISTS. The hands-free demo transcribed "Hello. What is a chicken?" as "Oh, what is it
// chicken?". `Hello` -> `Oh` is the FRONT of the utterance being cut: an /h/ is low-energy and Silero does
// not cross its threshold until the vowel, so a segment that starts where the probability crosses starts
// after the word did. The detector's total reach-back is SpeechPad plus one frame - 30 ms + 32 ms with the
// shipped default, against the 100-300 ms that ASR front-ends normally want.
//
// That is a story, and a story is not a measurement. This is the measurement: transcribe the whole file
// once for a ground truth that no endpointing touched, then endpoint the SAME file at a range of pads,
// transcribe what the detector handed over, and report the word error each pad costs. The right pad is the
// smallest one that stops costing words - a number, not a preference.
//
// ⚠️ Whisper's window is a fixed 30 s and the mel is computed over the PADDED window regardless, so the
// baseline is taken over the same first 30 s the spans are drawn from. Comparing a 46 s file's spans to a
// 30 s baseline would score the missing 16 s as deletions and drown the effect being measured.
using System.Diagnostics;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;

var repoRoot = @"D:\users\tj\Projects\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML";
var www = Path.Combine(repoRoot, "SpawnDev.ILGPU.ML.Demo", "wwwroot");
var vadModelPath = Path.Combine(www, "references", "vad", "silero_vad.onnx");
var whisperDir = Path.Combine(www, "models", "whisper-tiny");

// Harvard sentences, phonetically balanced, from the Open Speech Repository - the same recording the
// whisper harness uses, so a transcript here is comparable to one there.
var wavPath = args.Length > 0 && !args[0].StartsWith("--")
    ? args[0]
    : @"C:\Users\TJ\Downloads\OSR_us_000_0030_8k.wav";

foreach (var (what, path) in new[] { ("vad model", vadModelPath), ("wav", wavPath) })
    if (!File.Exists(path)) { Console.WriteLine($"no {what} at {path}"); return 2; }
if (!Directory.Exists(whisperDir)) { Console.WriteLine($"no whisper model dir at {whisperDir}"); return 2; }

var (raw, wavRate, channels) = ReadWav(File.ReadAllBytes(wavPath));
var audio = wavRate == SileroVad.SampleRate
    ? raw
    : AudioPreprocessor.Resample(raw, wavRate, SileroVad.SampleRate);

// Both sides of the comparison see exactly the same audio - see the note at the top.
int window = SileroVad.SampleRate * 30;
if (audio.Length > window) audio = audio[..window];

Console.WriteLine($"wav      : {Path.GetFileName(wavPath)}  {wavRate} Hz, {channels} ch");
Console.WriteLine($"audio    : {audio.Length} samples @ 16 kHz = {audio.Length / 16000.0:F1}s "
                + "(clipped to Whisper's 30 s window)");

// -- accelerator ---------------------------------------------------------------------------------------
var mlBuilder = MLContext.Create();
await mlBuilder.AllAcceleratorsAsync();
using var mlCtx = mlBuilder.ToContext();
var want = Environment.GetEnvironmentVariable("VAD_BACKEND")?.Trim().ToLowerInvariant();
using var accel = want switch
{
    "cuda" => mlCtx.GetCudaDevices()[0].CreateCudaAccelerator(mlCtx),
    "opencl" => mlCtx.GetCLDevices()[0].CreateCLAccelerator(mlCtx),
    "cpu" => mlCtx.GetCPUDevices()[0].CreateCPUAccelerator(mlCtx),
    _ => await mlCtx.CreatePreferredAcceleratorAsync() ?? throw new InvalidOperationException("no accelerator"),
};
Console.WriteLine($"device   : {accel.AcceleratorType} {accel.Name}");

// -- whisper -------------------------------------------------------------------------------------------
using var encoder = InferenceSession.CreateFromFile(
    accel, File.ReadAllBytes(Path.Combine(whisperDir, "encoder_model.onnx")));
using var decoder = InferenceSession.CreateFromFile(
    accel, File.ReadAllBytes(Path.Combine(whisperDir, "decoder_model.onnx")));
InferenceSession? withPast = null;
var withPastPath = Path.Combine(whisperDir, "decoder_with_past_model.onnx");
if (File.Exists(withPastPath))
    withPast = InferenceSession.CreateFromFile(accel, File.ReadAllBytes(withPastPath));

var asr = new SpeechRecognitionPipeline(encoder, decoder, accel, withPast);
asr.LoadTokenizer(File.ReadAllText(Path.Combine(whisperDir, "tokenizer.json")));

// -- ground truth: the whole clip, no endpointing anywhere near it -------------------------------------
var sw = Stopwatch.StartNew();
var baseline = (await asr.TranscribeAsync(audio, SileroVad.SampleRate)).Text.Trim();
Console.WriteLine($"baseline : {sw.Elapsed.TotalSeconds:F1}s, {WordCount(baseline)} words");
Console.WriteLine($"           \"{Shorten(baseline)}\"");
Console.WriteLine();

var vadBytes = File.ReadAllBytes(vadModelPath);

Console.WriteLine($"{"SpeechPad",10} {"spans",6} {"span s",8} {"words",6} {"WER",7}   transcript");
Console.WriteLine(new string('-', 100));

// 30 ms is the shipped default and the suspect; the rest brackets what ASR front-ends normally ask for.
foreach (var padMs in new[] { 30, 60, 100, 150, 200, 300, 500 })
{
    using var vad = SileroVad.Create(accel, vadBytes);
    var options = new VadOptions { SpeechPad = TimeSpan.FromMilliseconds(padMs) };
    using var detector = new VoiceActivityDetector(vad, options);

    var spans = new List<(long Start, int Length)>();
    detector.OnSegment += seg => spans.Add((seg.StartSample, seg.Samples.Length));
    await detector.AcceptWaveformAsync(audio);
    await detector.FlushAsync();

    // Transcribe what the detector HANDED OVER, sliced out of the caller's own buffer - which is exactly
    // what the demo does with the spans it gets back. Transcribing the detector's copy instead would test
    // a path no consumer uses.
    var parts = new List<string>();
    double spanSeconds = 0;
    foreach (var (start, length) in spans)
    {
        int from = (int)Math.Clamp(start, 0, audio.Length);
        int to = (int)Math.Clamp(start + (long)length, from, audio.Length);
        if (to - from < SileroVad.SampleRate / 4) continue;      // under 250 ms: nothing to transcribe
        spanSeconds += (to - from) / (double)SileroVad.SampleRate;
        var text = (await asr.TranscribeAsync(audio[from..to], SileroVad.SampleRate)).Text.Trim();
        if (text.Length > 0) parts.Add(text);
    }

    var joined = string.Join(" ", parts);
    double wer = SpokenTextCheck.WordErrorRate(baseline, joined);
    Console.WriteLine($"{padMs + " ms",10} {spans.Count,6} {spanSeconds,8:F1} {WordCount(joined),6} "
                    + $"{wer,7:F3}   \"{Shorten(joined)}\"");
}

Console.WriteLine();
Console.WriteLine("READ IT THIS WAY: WER here is the endpointer's cost, not Whisper's - the baseline is the");
Console.WriteLine("SAME recogniser on the SAME audio with nothing cut. The right SpeechPad is the smallest");
Console.WriteLine("one whose WER stops improving; anything beyond that is silence bought for nothing, and");
Console.WriteLine("silence in a span is not free - it is also what makes a cloned voice speak slowly.");
return 0;

static int WordCount(string s) =>
    s.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Length;

static string Shorten(string s, int max = 46) => s.Length <= max ? s : s[..max] + "...";

static (float[] Samples, int Rate, int Channels) ReadWav(byte[] wav)
{
    int rate = 0, channels = 1, bits = 16, dataAt = -1, dataLen = 0;
    for (int i = 12; i + 8 <= wav.Length;)
    {
        var id = System.Text.Encoding.ASCII.GetString(wav, i, 4);
        int size = BitConverter.ToInt32(wav, i + 4);
        if (id == "fmt ")
        {
            channels = BitConverter.ToInt16(wav, i + 10);
            rate = BitConverter.ToInt32(wav, i + 12);
            bits = BitConverter.ToInt16(wav, i + 22);
        }
        else if (id == "data") { dataAt = i + 8; dataLen = size; }
        i += 8 + size + (size & 1);
    }
    if (dataAt < 0 || rate == 0) throw new Exception("not a PCM wav this tool understands");
    if (bits != 16) throw new Exception($"{bits}-bit wav; this tool reads 16-bit PCM");

    int frames = dataLen / 2 / Math.Max(1, channels);
    var samples = new float[frames];
    for (int f = 0; f < frames; f++)
    {
        // Downmix to mono: the detector and the recogniser are both mono, and taking channel 0 would
        // silently halve a stereo recording that has the speaker panned.
        double sum = 0;
        for (int c = 0; c < channels; c++)
            sum += BitConverter.ToInt16(wav, dataAt + (f * channels + c) * 2) / 32768.0;
        samples[f] = (float)(sum / channels);
    }
    return (samples, rate, channels);
}
