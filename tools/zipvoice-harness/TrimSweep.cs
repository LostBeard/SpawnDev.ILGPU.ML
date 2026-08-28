// Where should the generated audio be cut away from the reference clip the model regenerates ahead of it?
//
//   dotnet run --project tools/zipvoice-harness -c Release -- trimsweep [fixtureDirOrFile] [outDir]
//
// WHY THIS EXISTS: nearly every render opens with a few words of the voice being cloned - "Others call me
// Mother Nature" before the sentence actually asked for. It is the loudest remaining artifact in anything
// this stack would say out loud. It is NOT this port: the reference implementation does it too and worse.
// Both cut the generated mel at the reference's own frame count, so if the model runs past that boundary,
// both inherit the spill.
//
// The question is empirical - how much further should the cut go - and it has a trap. Cutting too little
// leaves the preamble; cutting too much eats the first word of the sentence, which is far worse. So this
// sweeps the trim and scores each setting two ways:
//
//   STRICT word error charges for the leading words, so it falls as the preamble is removed.
//   INFIX word error ignores them, so it RISES the moment the sentence itself starts being eaten.
//
// The right trim is where strict has stopped falling and infix has not yet started to rise. Reading one
// number alone would happily recommend cutting the sentence in half.
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;

namespace ZipVoiceHarness;

public static class TrimSweep
{
    public static async Task<int> RunAsync(
        string modelDir,
        IReadOnlyList<(string Path, ZipVoiceFixture Fixture)> fixtures,
        string outDir,
        Func<byte[], (float[] Samples, int SampleRate, int Channels)> readWav,
        Func<float[], int, byte[]> writeWav)
    {
        Directory.CreateDirectory(outDir);

        var trims = (Environment.GetEnvironmentVariable("TRIM_FRAMES") ?? "0,20,40,60,80,100")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(int.Parse).ToArray();
        var seeds = (Environment.GetEnvironmentVariable("SENSITIVITY_SEEDS") ?? "1234,7")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(int.Parse).ToArray();

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

        bool int8 = Environment.GetEnvironmentVariable("ZIPVOICE_INT8") == "1";
        var graphDir = int8 ? modelDir.Replace("zipvoice-distill-zh-en", "zipvoice-distill-int8-zh-en") : modelDir;
        using var graphs = new OrtZipVoiceGraphs(graphDir, int8);
        var config = new ZipVoiceConfig();
        double msPerFrame = 1000.0 * config.HopLength / config.SampleRate;

        Console.WriteLine($"trims    : {string.Join(", ", trims)} frames ({msPerFrame:F1}ms each)");
        Console.WriteLine($"seeds    : {string.Join(", ", seeds)}   fixtures: {fixtures.Count}");
        Console.WriteLine();

        var strict = new Dictionary<int, List<double>>();
        var infix = new Dictionary<int, List<double>>();
        foreach (var trim in trims) { strict[trim] = new List<double>(); infix[trim] = new List<double>(); }

        foreach (var (path, fixture) in fixtures)
        {
            var promptWav = Sensitivity.ResolvePromptWav(modelDir, fixture);
            var (reference, rate, _) = readWav(File.ReadAllBytes(promptWav));

            foreach (var seed in seeds)
                foreach (var trim in trims)
                {
                    using var pipeline = new ZipVoicePipeline(graphs, config)
                    {
                        NoiseSeed = seed,
                        TrimGeneratedStartFrames = trim,
                    };
                    var result = await pipeline.SynthesizeAsync(fixture.Tokens, fixture.PromptTokens, reference, rate);
                    var heard = (await stt.TranscribeAsync(result.Audio, result.SampleRate)).Text?.Trim() ?? "";

                    strict[trim].Add(SpokenTextCheck.WordErrorRateStrict(fixture.Text, heard));
                    infix[trim].Add(SpokenTextCheck.WordErrorRate(fixture.Text, heard));

                    if (trim == trims[0] || trim == trims[^1])
                        File.WriteAllBytes(
                            Path.Combine(outDir, $"{Path.GetFileNameWithoutExtension(path)}__trim{trim}__s{seed}.wav"),
                            writeWav(result.Audio, result.SampleRate));
                }
            Console.WriteLine($"  done {Path.GetFileNameWithoutExtension(path)}");
        }

        Console.WriteLine();
        Console.WriteLine($"{"trim",6} {"ms",7}  {"strict",8}  {"infix",8}   reading");
        Console.WriteLine(new string('-', 62));
        double bestStrict = double.MaxValue;
        int recommended = trims[0];
        foreach (var trim in trims)
        {
            double st = strict[trim].Average(), ix = infix[trim].Average();
            // The sentence is intact while infix stays near its best; strict falling means preamble going.
            string reading = ix > infix[trims[0]].Average() + 0.05 ? "EATING THE SENTENCE" : "sentence intact";
            Console.WriteLine($"{trim,6} {trim * msPerFrame,6:F0}  {st,7:P1}  {ix,7:P1}   {reading}");
            if (reading == "sentence intact" && st < bestStrict) { bestStrict = st; recommended = trim; }
        }

        Console.WriteLine();
        Console.WriteLine($"BEST     : {recommended} frames ({recommended * msPerFrame:F0}ms) - lowest strict error among");
        Console.WriteLine("           the settings that leave the sentence intact. Read BOTH columns: strict alone");
        Console.WriteLine("           would happily recommend cutting the sentence in half.");
        return 0;
    }
}
