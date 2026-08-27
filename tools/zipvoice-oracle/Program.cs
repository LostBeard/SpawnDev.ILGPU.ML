// Ground truth for the ZipVoice port, taken from sherpa-onnx - the independent implementation.
//
//   dotnet run --project tools/zipvoice-oracle -c Release -- "text to speak" [out.wav] 2> tokens.log
//
// WHY THIS EXISTS: two things about ZipVoice cannot be checked by our own code, because our own code is
// the thing under test.
//
//   1. TOKEN IDS. English text reaches the model through espeak-ng grapheme-to-phoneme - the shipped
//      lexicon is Chinese-only, so there is no table to read the answer out of. sherpa runs the real
//      espeak, and with debug on it prints the id sequence it fed the encoder. That printout is the
//      target our own phonemizer has to reproduce exactly.
//   2. WHAT A CORRECT CLONE SOUNDS LIKE. Flow matching starts from fresh noise every call, so no two
//      renders are sample-identical even from one implementation. A reference rendering is what makes
//      "ours sounds wrong" a statement about our code rather than about the noise draw.
//
// The token dump goes to STDERR because sherpa's logging is native and writes to fd 2 - redirect it
// with 2> rather than expecting it on stdout.
using SherpaOnnx;

var modelRoot = Environment.GetEnvironmentVariable("ZIPVOICE_MODELS")
    ?? @"D:\users\tj\Projects\SpawnDev.Reachy\SpawnDev.Reachy\models";
var int8Dir = Path.Combine(modelRoot, "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia");
var fp32Dir = Path.Combine(modelRoot, "sherpa-onnx-zipvoice-distill-zh-en-emilia");

var text = args.Length > 0 ? args[0] : "The quick brown fox jumps over the lazy dog.";
var outPath = args.Length > 1 ? args[1] : Path.Combine(Path.GetTempPath(), "zipvoice-oracle.wav");

// The reference clip and its EXACT transcript. ZipVoice clones delivery as well as timbre, and it bleeds
// any word that is in the audio but missing from the transcript into whatever it generates - so this
// pairing is part of the fixture, not a detail.
var promptWav = Environment.GetEnvironmentVariable("ZIPVOICE_PROMPT_WAV")
    ?? Path.Combine(fp32Dir, "prompt.wav");
var promptText = Environment.GetEnvironmentVariable("ZIPVOICE_PROMPT_TEXT")
    ?? "Some call me nature, others call me mother nature.";

if (!Directory.Exists(int8Dir)) { Console.WriteLine($"no model dir at {int8Dir}"); return 2; }
if (!File.Exists(promptWav)) { Console.WriteLine($"no prompt wav at {promptWav}"); return 2; }

var config = new OfflineTtsConfig();
config.Model.ZipVoice.Tokens = Path.Combine(int8Dir, "tokens.txt");
config.Model.ZipVoice.Encoder = Path.Combine(int8Dir, "encoder.int8.onnx");
config.Model.ZipVoice.Decoder = Path.Combine(int8Dir, "decoder.int8.onnx");
config.Model.ZipVoice.Vocoder = ResolveVocoder();
config.Model.ZipVoice.DataDir = Path.Combine(int8Dir, "espeak-ng-data");
config.Model.ZipVoice.Lexicon = Path.Combine(int8Dir, "lexicon.txt");
config.Model.NumThreads = Math.Max(1, Environment.ProcessorCount / 2);
config.Model.Provider = "cpu";
// Debug is the entire point of this tool: it makes the frontend print the token ids per word and the
// assembled sentence, which is the only place that information is observable.
config.Model.Debug = 1;

using var tts = new OfflineTts(config);

var (promptSamples, promptRate) = ReadWav(File.ReadAllBytes(promptWav));

Console.WriteLine($"model    : {int8Dir}");
Console.WriteLine($"prompt   : {Path.GetFileName(promptWav)} {promptRate} Hz, {promptSamples.Length} samples");
Console.WriteLine($"promptTxt: {promptText}");
Console.WriteLine($"text     : {text}");
Console.WriteLine();
Console.WriteLine("--- sherpa debug output follows on STDERR; token ids are the '(id)' values ---");
Console.Out.Flush();

var generation = new OfflineTtsGenerationConfig
{
    ReferenceAudio = promptSamples,
    ReferenceSampleRate = promptRate,
    ReferenceText = promptText,
    NumSteps = 4,
    Speed = 1.0f,
    Sid = 0,
};

// MEASURED, so nobody tries it again: aborting generation from the progress callback (returning 0) does
// NOT let this tool skip synthesis when only the token ids are wanted. ZipVoice does not stream - the
// callback fires once, after the audio already exists - so the "abort early" trick saves nothing here:
// 2076ms without it against 2192ms with it, on the same sentence.
var started = System.Diagnostics.Stopwatch.StartNew();
var audio = tts.GenerateWithConfig(text, generation, null!);
started.Stop();

Console.WriteLine();
Console.WriteLine($"generated: {audio.Samples.Length} samples @ {audio.SampleRate} Hz " +
                  $"({audio.Samples.Length / (double)audio.SampleRate:F2}s) in {started.ElapsedMilliseconds}ms");

File.WriteAllBytes(outPath, WriteWav(audio.Samples, audio.SampleRate));
Console.WriteLine($"wrote    : {outPath}");
return 0;

string ResolveVocoder()
{
    // The vocoder ships separately from the quantized package, so it may live in either folder.
    var inPackage = Path.Combine(int8Dir, "vocos_24khz.onnx");
    if (File.Exists(inPackage)) return inPackage;
    return Path.Combine(fp32Dir, "vocos_24khz.onnx");
}

static (float[] Samples, int SampleRate) ReadWav(byte[] data)
{
    if (data.Length < 44) throw new InvalidDataException("too short to be a WAV");
    int pos = 12, channels = 1, bits = 16, rate = 16000;
    while (pos + 8 <= data.Length)
    {
        var id = System.Text.Encoding.ASCII.GetString(data, pos, 4);
        int size = BitConverter.ToInt32(data, pos + 4);
        int body = pos + 8;
        if (id == "fmt ")
        {
            channels = BitConverter.ToInt16(data, body + 2);
            rate = BitConverter.ToInt32(data, body + 4);
            bits = BitConverter.ToInt16(data, body + 14);
        }
        else if (id == "data")
        {
            if (bits != 16) throw new NotSupportedException($"{bits}-bit WAV not supported (16-bit PCM only)");
            int count = Math.Min(size, data.Length - body) / 2;
            int frames = count / Math.Max(1, channels);
            var samples = new float[frames];
            for (int f = 0; f < frames; f++)
            {
                int sum = 0;
                for (int c = 0; c < channels; c++)
                    sum += BitConverter.ToInt16(data, body + (f * channels + c) * 2);
                samples[f] = sum / (float)channels / 32768f;
            }
            return (samples, rate);
        }
        pos = body + size + (size & 1);
    }
    throw new InvalidDataException("no data chunk");
}

static byte[] WriteWav(float[] samples, int sampleRate)
{
    using var stream = new MemoryStream();
    using var writer = new BinaryWriter(stream);
    int dataBytes = samples.Length * 2;
    writer.Write("RIFF"u8.ToArray());
    writer.Write(36 + dataBytes);
    writer.Write("WAVE"u8.ToArray());
    writer.Write("fmt "u8.ToArray());
    writer.Write(16);
    writer.Write((short)1);
    writer.Write((short)1);
    writer.Write(sampleRate);
    writer.Write(sampleRate * 2);
    writer.Write((short)2);
    writer.Write((short)16);
    writer.Write("data"u8.ToArray());
    writer.Write(dataBytes);
    foreach (var sample in samples)
        writer.Write((short)Math.Clamp((int)MathF.Round(sample * 32767f), short.MinValue, short.MaxValue));
    writer.Flush();
    return stream.ToArray();
}
