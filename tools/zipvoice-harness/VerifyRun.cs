// Does listening to your own output actually rescue a garbled render?
//
//   dotnet run --project tools/zipvoice-harness -c Release -- verify [fixtureDirOrFile] [outDir]
//
// WHY THIS EXISTS: ZipVoice produces garbage on some noise draws. Measured: one sentence rendered at four
// seeds gave three clean results and one that transcribed as "Loner's call, Nanawa, Nenfer", and the
// reference implementation does the same - at one seed a clean sentence came back as "I'm not sure if I'm
// going to be here". It is a property of the model, not of the phonemes it is given.
//
// SpeakVerifiedAsync answers it by using the recogniser already present in this stack: speak the line,
// listen to it, and re-roll the noise when the words that come back are not the words asked for. This
// runs that on REAL models with REAL audio - no substitute graphs, no stand-in recogniser - and reports
// how often the first attempt was already fine, how often a re-roll rescued it, and how often nothing did.
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;

namespace ZipVoiceHarness;

public static class VerifyRun
{
    public static async Task<int> RunAsync(
        string modelDir,
        IReadOnlyList<(string Path, ZipVoiceFixture Fixture)> fixtures,
        string outDir,
        Func<byte[], (float[] Samples, int SampleRate, int Channels)> readWav,
        Func<float[], int, byte[]> writeWav)
    {
        Directory.CreateDirectory(outDir);

        // ---- The recogniser, which is the whole point ------------------------------------------------
        var whisperDir = Environment.GetEnvironmentVariable("WHISPER_MODEL_DIR")
            ?? Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..",
                                             "SpawnDev.ILGPU.ML.Demo", "wwwroot", "models", "whisper-base.en"));
        if (!Directory.Exists(whisperDir)) { Console.WriteLine($"no whisper model at {whisperDir}"); return 2; }

        var builder = MLContext.Create();
        await builder.AllAcceleratorsAsync();
        var accelerator = await builder.ToContext().CreatePreferredAcceleratorAsync();
        if (accelerator == null) { Console.WriteLine("no accelerator"); return 3; }

        var encoder = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(Path.Combine(whisperDir, "encoder_model.onnx")));
        var decoder = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(Path.Combine(whisperDir, "decoder_model.onnx")));
        var stt = new SpeechRecognitionPipeline(encoder, decoder, accelerator);
        stt.LoadTokenizer(File.ReadAllText(Path.Combine(whisperDir, "tokenizer.json")));
        Console.WriteLine($"recogniser: {Path.GetFileName(whisperDir)} on {accelerator.AcceleratorType}");

        // ---- The synthesiser, wired exactly as a consumer would --------------------------------------
        bool int8 = Environment.GetEnvironmentVariable("ZIPVOICE_INT8") == "1";
        var graphDir = int8 ? modelDir.Replace("zipvoice-distill-zh-en", "zipvoice-distill-int8-zh-en") : modelDir;
        using var graphs = new OrtZipVoiceGraphs(graphDir, int8);
        var tokenizer = ZipVoiceTokenizer.CreateDefault(modelDir);

        double tolerance = double.TryParse(Environment.GetEnvironmentVariable("VERIFY_TOLERANCE"), out var t) ? t : 0.2;
        int attempts = int.TryParse(Environment.GetEnvironmentVariable("VERIFY_ATTEMPTS"), out var a) ? a : 3;
        int startSeed = int.TryParse(Environment.GetEnvironmentVariable("VERIFY_SEED"), out var sd) ? sd : 1234;
        Console.WriteLine($"policy    : accept at {tolerance:P0} word error, up to {attempts} attempts, from seed {startSeed}");
        Console.WriteLine();

        int firstTry = 0, rescued = 0, failed = 0;
        foreach (var (path, fixture) in fixtures)
        {
            var promptWav = Sensitivity.ResolvePromptWav(modelDir, fixture);
            var (reference, rate, _) = readWav(File.ReadAllBytes(promptWav));

            // Seed pinned so this run is reproducible; SpeakVerifiedAsync advances it per attempt, which
            // is the whole mechanism - re-rolling the SAME noise would reproduce the same garbage.
            using var pipeline = new ZipVoicePipeline(graphs) { NoiseSeed = startSeed };

            // One un-verified render first, so the comparison is like for like: this is what a caller
            // gets today without the check.
            var plain = await pipeline.SynthesizeAsync(tokenizer.Encode(fixture.Text),
                                                       tokenizer.Encode(fixture.PromptText), reference, rate);
            var plainHeard = (await stt.TranscribeAsync(plain.Audio, plain.SampleRate)).Text?.Trim() ?? "";
            double plainError = SpokenTextCheck.WordErrorRate(fixture.Text, plainHeard);

            var verified = await pipeline.SpeakVerifiedAsync(
                fixture.Text, fixture.PromptText, reference, rate, tokenizer,
                async (audio, sampleRate) => (await stt.TranscribeAsync(audio, sampleRate)).Text?.Trim() ?? "",
                tolerance, attempts);

            var name = Path.GetFileNameWithoutExtension(path);
            File.WriteAllBytes(Path.Combine(outDir, name + "__verified.wav"),
                               writeWav(verified.Speech.Audio, verified.Speech.SampleRate));

            string verdict;
            if (plainError <= tolerance) { firstTry++; verdict = "already fine"; }
            else if (verified.Passed) { rescued++; verdict = "RESCUED"; }
            else { failed++; verdict = "still bad"; }

            Console.WriteLine($"{name[..Math.Min(46, name.Length)],-46} unverified {plainError,4:P0} -> "
                            + $"verified {verified.WordErrorRate,4:P0}  {verdict}");
            if (plainError > tolerance)
            {
                Console.WriteLine($"    wanted : {fixture.Text}");
                Console.WriteLine($"    without: {plainHeard}");
                Console.WriteLine($"    with   : {verified.Transcript}");
            }
        }

        Console.WriteLine();
        Console.WriteLine($"RESULT   : {firstTry} already fine, {rescued} RESCUED by re-rolling, {failed} still bad");
        Console.WriteLine("Verification costs a synthesis and a transcription per retry, and only pays on the");
        Console.WriteLine("renders that needed it - which is why it is opt-in rather than the default path.");
        return 0;
    }
}
