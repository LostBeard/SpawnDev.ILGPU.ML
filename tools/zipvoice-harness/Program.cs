// ZipVoice (zero-shot voice cloning TTS) desktop gate.
//
//   dotnet run --project tools/zipvoice-harness -c Release -- roundtrip [wav]
//
// WHY THIS EXISTS: ZipVoice is four separable pieces - mel features, a text encoder, a flow-matching
// decoder, and a vocoder whose last stage is an inverse STFT we write ourselves. Debugging them
// together, in a browser, from a robot's speaker, would make any one of them able to explain a bad
// result. Each command here isolates one piece and gives it a pass/fail an eye cannot argue with.
//
// ROUNDTRIP is the first gate and needs no tokenizer at all: take a real recording, compute the mel
// exactly as the model expects it, hand that mel to the REAL vocos vocoder, resynthesise, and compare.
// It is a fixed-point test - only a mel built to the right convention survives the loop unchanged, so a
// wrong window, mel scale, normalisation or magnitude-vs-power choice shows up as drift rather than as
// audio nobody can grade. The inverse STFT is on trial in the same pass, since it is what turns the
// vocoder's magnitude and phase back into sound.
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using ZipVoiceHarness;

var modelDir = Environment.GetEnvironmentVariable("ZIPVOICE_MODEL_DIR")
    ?? @"D:\users\tj\Projects\SpawnDev.Reachy\SpawnDev.Reachy\models\sherpa-onnx-zipvoice-distill-zh-en-emilia";

var command = args.Length > 0 ? args[0].ToLowerInvariant() : "roundtrip";

return command switch
{
    "roundtrip" => RoundTrip(args.Length > 1 ? args[1] : Path.Combine(modelDir, "prompt.wav")),
    "synth" => Synth(args.Length > 1 ? args[1] : "fixtures/paint-the-sockets.json",
                     args.Length > 2 ? args[2] : null),
    "compare" => CompareEngines(args.Length > 1 ? args[1] : "fixtures/paint-the-sockets.json"),
    "sensitivity" => RunSensitivity(args.Length > 1 ? args[1] : "fixtures/loaded-classes.json",
                                    args.Length > 2 ? args[2] : null),
    _ => Usage(command),
};

int Usage(string bad)
{
    Console.WriteLine($"unknown command '{bad}'. commands: roundtrip [wav] | synth [fixture.json] [out.wav] "
                    + "| compare [fixture.json] | sensitivity [fixture.json] [outDir]");
    return 2;
}

// How much phonemizer error this model actually tolerates - see Sensitivity.cs and
// Plans/mit-phonemizer-2026-08-27.md. Damages the ground-truth tokens the way a CMUdict-based
// frontend will and grades the resulting audio, so the frontend's precision target is measured
// rather than assumed.
int RunSensitivity(string fixturePath, string? outDir)
{
    if (!Directory.Exists(modelDir)) { Console.WriteLine($"no model dir at {modelDir}"); return 2; }

    // A directory means "every fixture in it". Replication across sentences is what separates a
    // measurement from an anecdote, so the many-fixture case is the normal one.
    var candidates = new[]
    {
        Path.IsPathRooted(fixturePath) ? fixturePath : Path.Combine(AppContext.BaseDirectory, fixturePath),
        Path.Combine(Environment.CurrentDirectory, fixturePath),
    };
    var resolved = candidates.FirstOrDefault(p => File.Exists(p) || Directory.Exists(p));
    if (resolved == null) { Console.WriteLine($"no fixture or fixture dir at {fixturePath}"); return 2; }

    var paths = Directory.Exists(resolved)
        ? Directory.GetFiles(resolved, "*.json").OrderBy(p => p).ToArray()
        : new[] { resolved };
    if (paths.Length == 0) { Console.WriteLine($"no *.json fixtures in {resolved}"); return 2; }

    var fixtures = new List<(string Path, ZipVoiceFixture Fixture)>();
    foreach (var p in paths)
    {
        var fixture = ZipVoiceFixture.Load(p);
        var promptWav = Sensitivity.ResolvePromptWav(modelDir, fixture);
        if (!File.Exists(promptWav)) { Console.WriteLine($"no prompt wav at {promptWav} (for {Path.GetFileName(p)})"); return 2; }
        fixtures.Add((p, fixture));
    }

    outDir ??= Path.Combine(Path.GetTempPath(), "zipvoice-sensitivity");
    return Sensitivity.RunAsync(modelDir, fixtures, outDir, ReadWav, WriteWav)
                      .GetAwaiter().GetResult();
}

// Full cloning path on the reference engine: ground-truth tokens in, cloned speech out.
// Runs the SHIPPING orchestration (mel features, flow-matching loop, inverse STFT) with onnxruntime
// executing the graphs, so a bad result here is the algorithm and never the engine.
int Synth(string fixturePath, string? outPath)
{
    if (!Directory.Exists(modelDir)) { Console.WriteLine($"no model dir at {modelDir}"); return 2; }

    var resolved = Path.IsPathRooted(fixturePath)
        ? fixturePath
        : Path.Combine(AppContext.BaseDirectory, fixturePath);
    if (!File.Exists(resolved)) resolved = Path.Combine(Environment.CurrentDirectory, fixturePath);
    if (!File.Exists(resolved)) { Console.WriteLine($"no fixture at {fixturePath}"); return 2; }

    var fixture = ZipVoiceFixture.Load(resolved);
    var promptWav = Path.IsPathRooted(fixture.PromptWav)
        ? fixture.PromptWav
        : Path.Combine(modelDir, fixture.PromptWav);
    if (!File.Exists(promptWav)) { Console.WriteLine($"no prompt wav at {promptWav}"); return 2; }

    var (reference, referenceRate, _) = ReadWav(File.ReadAllBytes(promptWav));

    Console.WriteLine($"text     : {fixture.Text}");
    Console.WriteLine($"promptTxt: {fixture.PromptText}");
    Console.WriteLine($"prompt   : {Path.GetFileName(promptWav)} {referenceRate} Hz, " +
                      $"{reference.Length / (double)referenceRate:F2}s");
    Console.WriteLine($"tokens   : {fixture.Tokens.Length} text, {fixture.PromptTokens.Length} prompt");

    var config = new ZipVoiceConfig();
    // The oracle runs the quantized graphs, so being able to select them is what makes a frame-count
    // or duration difference attributable to precision rather than to our code.
    bool int8 = Environment.GetEnvironmentVariable("ZIPVOICE_INT8") == "1";
    var graphDir = int8
        ? modelDir.Replace("zipvoice-distill-zh-en", "zipvoice-distill-int8-zh-en")
        : modelDir;
    using var graphs = new OrtZipVoiceGraphs(graphDir, int8);
    Console.WriteLine($"graphs   : {(int8 ? "int8" : "fp32")}");
    using var pipeline = new ZipVoicePipeline(graphs, config)
    {
        // Fixed so two runs of this gate are comparable; production leaves it null and re-rolls.
        NoiseSeed = 1234,
        // The reference implementation does NOT pad, so comparisons against it turn this off.
        ReferenceTailSilenceSeconds =
            Environment.GetEnvironmentVariable("ZIPVOICE_NO_PAD") == "1" ? 0f : 0.25f,
    };
    Console.WriteLine($"tailPad  : {pipeline.ReferenceTailSilenceSeconds}s");

    var result = pipeline.SynthesizeAsync(fixture.Tokens, fixture.PromptTokens, reference, referenceRate).GetAwaiter().GetResult();

    Console.WriteLine($"timing   : encoder {result.EncoderMs:F0}ms, decoder {result.DecoderMs:F0}ms " +
                      $"({config.NumSteps} steps), vocoder {result.VocoderMs:F0}ms, total {result.TotalMs:F0}ms");
    Console.WriteLine($"audio    : {result.Audio.Length} samples @ {result.SampleRate} Hz " +
                      $"({result.DurationSeconds:F2}s), rms {Rms(result.Audio):F4}), " +
                      $"{result.Audio.Length / config.HopLength + 1} generated frames");

    outPath ??= Path.Combine(Path.GetTempPath(), "zipvoice-ours.wav");
    File.WriteAllBytes(outPath, WriteWav(result.Audio, result.SampleRate));
    Console.WriteLine($"wrote    : {outPath}");

    // Silence and noise are the two ways this fails without throwing, and both are visible in the level.
    bool pass = result.DurationSeconds > 0.5 && Rms(result.Audio) > 0.005 && Rms(result.Audio) < 0.5;
    Console.WriteLine(pass ? "RESULT   : PASS (level plausible - grade the audio by transcribing it)" : "RESULT   : FAIL");
    return pass ? 0 : 1;
}

// Every graph on BOTH engines with identical inputs, stage by stage.
// The orchestration is shared and the inputs are pinned, so any difference reported here is our engine.
int CompareEngines(string fixturePath)
{
    if (!Directory.Exists(modelDir)) { Console.WriteLine($"no model dir at {modelDir}"); return 2; }

    var resolved = ResolveFixture(fixturePath);
    if (resolved == null) { Console.WriteLine($"no fixture at {fixturePath}"); return 2; }

    var fixture = ZipVoiceFixture.Load(resolved);
    var promptWav = Path.IsPathRooted(fixture.PromptWav) ? fixture.PromptWav : Path.Combine(modelDir, fixture.PromptWav);
    if (!File.Exists(promptWav)) { Console.WriteLine($"no prompt wav at {promptWav}"); return 2; }

    var (reference, referenceRate, _) = ReadWav(File.ReadAllBytes(promptWav));
    return Compare.RunAsync(modelDir, fixture, reference, referenceRate).GetAwaiter().GetResult();
}

string? ResolveFixture(string fixturePath)
{
    if (Path.IsPathRooted(fixturePath)) return File.Exists(fixturePath) ? fixturePath : null;
    var candidate = Path.Combine(AppContext.BaseDirectory, fixturePath);
    if (File.Exists(candidate)) return candidate;
    candidate = Path.Combine(Environment.CurrentDirectory, fixturePath);
    return File.Exists(candidate) ? candidate : null;
}

// Reference audio -> our mel -> the real vocoder -> our inverse STFT -> audio.
// A correct mel is a fixed point of that loop; a wrong one drifts.
int RoundTrip(string wavPath)
{
    if (!Directory.Exists(modelDir)) { Console.WriteLine($"no model dir at {modelDir}"); return 2; }
    if (!File.Exists(wavPath)) { Console.WriteLine($"no wav at {wavPath}"); return 2; }

    // The vocoder consumes UNSCALED log-mels. The model's 0.1 feature scale belongs to the encoder and
    // decoder, not to it, so it is left out here rather than applied and undone.
    var config = new ZipVoiceConfig { FeatScale = 1.0f };

    var (samples, sampleRate, channels) = ReadWav(File.ReadAllBytes(wavPath));
    Console.WriteLine($"wav      : {Path.GetFileName(wavPath)}  {sampleRate} Hz, {channels} ch, " +
                      $"{samples.Length} samples ({samples.Length / (double)sampleRate:F2}s)");

    var mel = ZipVoiceFeatures.ComputeMel(samples, sampleRate, config, out int frames);
    Console.WriteLine($"mel      : {frames} frames x {config.NumMels} mels, " +
                      $"range [{mel.Min():F3}, {mel.Max():F3}], mean {mel.Average():F3}");

    // Vocos reads [channels, frames]; our features are [frames, channels].
    var melChannelsFirst = new float[config.NumMels * frames];
    for (int f = 0; f < frames; f++)
        for (int c = 0; c < config.NumMels; c++)
            melChannelsFirst[c * frames + f] = mel[f * config.NumMels + c];

    // The oracle runs the quantized graphs, so being able to select them is what makes a frame-count
    // or duration difference attributable to precision rather than to our code.
    bool int8 = Environment.GetEnvironmentVariable("ZIPVOICE_INT8") == "1";
    var graphDir = int8
        ? modelDir.Replace("zipvoice-distill-zh-en", "zipvoice-distill-int8-zh-en")
        : modelDir;
    using var graphs = new OrtZipVoiceGraphs(graphDir, int8);
    Console.WriteLine($"graphs   : {(int8 ? "int8" : "fp32")}");
    var started = System.Diagnostics.Stopwatch.StartNew();
    var spectrum = graphs.RunVocoderAsync(melChannelsFirst, config.NumMels, frames).GetAwaiter().GetResult();
    var audio = ZipVoicePipeline.Vocode(spectrum, config);
    started.Stop();

    Console.WriteLine($"vocoder  : {spectrum.Bins} bins x {spectrum.Frames} frames -> " +
                      $"{audio.Length} samples ({audio.Length / (double)config.SampleRate:F2}s) in {started.ElapsedMilliseconds}ms");

    // The vocoder PREDICTS phase rather than preserving it, so the resynthesised waveform is not
    // sample-aligned with the original and comparing waveforms directly would fail on correct output.
    // The mel is the thing that must survive.
    var melAgain = ZipVoiceFeatures.ComputeMel(audio, config.SampleRate, config, out int framesAgain);
    int common = Math.Min(frames, framesAgain);

    double sumAbs = 0, maxAbs = 0;
    for (int f = 0; f < common; f++)
    {
        for (int c = 0; c < config.NumMels; c++)
        {
            double diff = Math.Abs(mel[f * config.NumMels + c] - melAgain[f * config.NumMels + c]);
            sumAbs += diff;
            if (diff > maxAbs) maxAbs = diff;
        }
    }
    double meanAbs = sumAbs / (common * config.NumMels);

    double rmsIn = Rms(samples), rmsOut = Rms(audio);
    Console.WriteLine($"levels   : rms in {rmsIn:F4}, out {rmsOut:F4}  (ratio {rmsOut / Math.Max(rmsIn, 1e-9):F3})");
    Console.WriteLine($"mel drift: mean |d| {meanAbs:F4} nats, max |d| {maxAbs:F3}, frames {frames} -> {framesAgain}");

    var outPath = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(wavPath))!,
                               Path.GetFileNameWithoutExtension(wavPath) + ".roundtrip.wav");
    File.WriteAllBytes(outPath, WriteWav(audio, config.SampleRate));
    Console.WriteLine($"wrote    : {outPath}");

    // A log-mel is in nats, so 0.25 is roughly a 28% error in magnitude on an average bin - loose
    // enough to allow the vocoder's own reconstruction error, tight enough that a wrong mel convention
    // (which moves whole bands, not single bins) cannot pass.
    bool pass = meanAbs < 0.25 && framesAgain >= frames - 1 && rmsOut > rmsIn * 0.5 && rmsOut < rmsIn * 2.0;
    Console.WriteLine(pass ? "RESULT   : PASS" : "RESULT   : FAIL");
    return pass ? 0 : 1;
}

static double Rms(float[] x)
{
    if (x.Length == 0) return 0;
    double sum = 0;
    foreach (var v in x) sum += (double)v * v;
    return Math.Sqrt(sum / x.Length);
}

static (float[] Samples, int SampleRate, int Channels) ReadWav(byte[] data)
{
    if (data.Length < 44) throw new InvalidDataException("too short to be a WAV");
    if (data[0] != 'R' || data[1] != 'I' || data[2] != 'F' || data[3] != 'F') throw new InvalidDataException("not RIFF");
    if (data[8] != 'W' || data[9] != 'A' || data[10] != 'V' || data[11] != 'E') throw new InvalidDataException("not WAVE");

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
            var outp = new float[frames];
            for (int f = 0; f < frames; f++)
            {
                int sum = 0;
                for (int c = 0; c < channels; c++)
                    sum += BitConverter.ToInt16(data, body + (f * channels + c) * 2);
                outp[f] = sum / (float)channels / 32768f;
            }
            return (outp, rate, channels);
        }
        pos = body + size + (size & 1);   // chunks are word-aligned
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
    writer.Write((short)1);              // PCM
    writer.Write((short)1);              // mono
    writer.Write(sampleRate);
    writer.Write(sampleRate * 2);        // byte rate
    writer.Write((short)2);              // block align
    writer.Write((short)16);             // bits per sample
    writer.Write("data"u8.ToArray());
    writer.Write(dataBytes);

    foreach (var sample in samples)
        writer.Write((short)Math.Clamp((int)MathF.Round(sample * 32767f), short.MinValue, short.MaxValue));

    writer.Flush();
    return stream.ToArray();
}
