// Silero VAD segment boundaries from sherpa-onnx - the independent implementation.
//
//   dotnet run --project tools/vad-oracle -c Release -- <model.onnx> <audio.wav> [out.json]
//
// WHY: our VoiceActivityDetector is a PORT of the endpointing RoseEars runs on the robot, and the whole
// point of a port is that it behaves the same. Grading it against a reference I transcribed myself from
// the same upstream source would only prove that I read that source the same way twice. sherpa-onnx is a
// separate C++ implementation, it is what RoseEars actually calls, and its segments have been in front of
// a real user - so agreement with it is evidence about behaviour rather than about my reading.
//
// Parameters mirror RoseEars exactly (SpawnDev.Reachy.Rose/RoseEars.cs): threshold 0.5, min-silence 0.5s,
// min-speech 0.25s, max-speech 20s, 512-sample window at 16 kHz. Those were tuned for a ten year old
// talking to a robot, not copied from a dictation default.
using System.Globalization;
using System.Text.Json;
using SherpaOnnx;

if (args.Length < 2)
{
    Console.WriteLine("usage: vad-oracle <silero_vad.onnx> <audio.wav> [out.json]");
    return 2;
}

var modelPath = args[0];
var wavPath = args[1];
var outPath = args.Length > 2 ? args[2] : null;

if (!File.Exists(modelPath)) { Console.WriteLine($"no model at {modelPath}"); return 2; }
if (!File.Exists(wavPath)) { Console.WriteLine($"no audio at {wavPath}"); return 2; }

const int SampleRate = 16000;
const int Window = 512;

var (samples, rate) = ReadWav(wavPath);
if (rate != SampleRate)
{
    // Resampling here would put OUR resampler inside the oracle, which defeats the point.
    Console.WriteLine($"audio is {rate} Hz; this oracle takes 16 kHz mono so that no resampling of ours "
                    + "sits between the fixture and the reference.");
    return 2;
}

var config = new VadModelConfig
{
    SampleRate = SampleRate,
    NumThreads = 1,
    Provider = "cpu",
};
config.SileroVad.Model = modelPath;
config.SileroVad.Threshold = 0.5f;
config.SileroVad.MinSilenceDuration = 0.5f;
config.SileroVad.MinSpeechDuration = 0.25f;
config.SileroVad.MaxSpeechDuration = 20.0f;
config.SileroVad.WindowSize = Window;

using var vad = new VoiceActivityDetector(config, bufferSizeInSeconds: 60.0f);

var segments = new List<object>();
var frame = new float[Window];
int framed = 0;

void Drain()
{
    while (!vad.IsEmpty())
    {
        var seg = vad.Front();
        vad.Pop();
        segments.Add(new
        {
            start_sample = seg.Start,
            end_sample = seg.Start + seg.Samples.Length,
            start_seconds = Math.Round(seg.Start / (double)SampleRate, 4),
            duration_seconds = Math.Round(seg.Samples.Length / (double)SampleRate, 4),
            samples = seg.Samples.Length,
        });
    }
}

foreach (var s in samples)
{
    frame[framed++] = s;
    if (framed < Window) continue;
    vad.AcceptWaveform(frame);
    framed = 0;
    Drain();
}

if (framed > 0)
{
    Array.Clear(frame, framed, Window - framed);
    vad.AcceptWaveform(frame);
    Drain();
}

vad.Flush();
Drain();

Console.WriteLine($"audio    : {samples.Length} samples, {samples.Length / (double)SampleRate:F2}s");
Console.WriteLine($"segments : {segments.Count}");
foreach (dynamic seg in segments)
    Console.WriteLine($"  {seg.start_seconds,8:F3}s  +{seg.duration_seconds,7:F3}s  "
                    + $"[{seg.start_sample}..{seg.end_sample})");

if (segments.Count == 0)
{
    // Silence in, silence out is a legitimate answer for some clips - but as a REFERENCE it is worthless,
    // because an engine that detects nothing at all would match it perfectly.
    Console.WriteLine("REFUSING to write a reference with zero segments: it cannot distinguish a working "
                    + "detector from one that never fires.");
    return 1;
}

if (outPath != null)
{
    var doc = new
    {
        source = "sherpa-onnx 1.13.4 (org.k2fsa.sherpa.onnx)",
        model = Path.GetFileName(modelPath),
        audio = Path.GetFileName(wavPath),
        sample_rate = SampleRate,
        window = Window,
        threshold = 0.5,
        min_silence_seconds = 0.5,
        min_speech_seconds = 0.25,
        max_speech_seconds = 20.0,
        total_samples = samples.Length,
        segments,
    };
    File.WriteAllText(outPath, JsonSerializer.Serialize(doc, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"wrote    : {outPath}");
}

return 0;

// Decoded the same way SpawnDev.ILGPU.ML's WavDecoder does - int16 little-endian over 32768 - so that a
// difference between the two implementations cannot come from the audio itself.
static (float[] Samples, int Rate) ReadWav(string path)
{
    var data = File.ReadAllBytes(path);
    int pos = 12;
    int channels = 1, rate = 0, bits = 16;
    while (pos < data.Length - 8)
    {
        var id = System.Text.Encoding.ASCII.GetString(data, pos, 4);
        int size = BitConverter.ToInt32(data, pos + 4);
        pos += 8;
        if (id == "fmt ")
        {
            channels = BitConverter.ToInt16(data, pos + 2);
            rate = BitConverter.ToInt32(data, pos + 4);
            bits = BitConverter.ToInt16(data, pos + 14);
        }
        else if (id == "data")
        {
            if (bits != 16) throw new NotSupportedException($"{bits}-bit wav; this oracle reads 16-bit PCM");
            int n = size / 2;
            var outp = new float[n];
            for (int i = 0; i < n; i++)
                outp[i] = (short)(data[pos + i * 2] | (data[pos + i * 2 + 1] << 8)) / 32768f;
            if (channels == 2)
            {
                var mono = new float[n / 2];
                for (int i = 0; i < mono.Length; i++) mono[i] = (outp[i * 2] + outp[i * 2 + 1]) / 2f;
                outp = mono;
            }
            return (outp, rate);
        }
        pos += size;
    }
    throw new InvalidDataException($"no data chunk in {path}");
}
