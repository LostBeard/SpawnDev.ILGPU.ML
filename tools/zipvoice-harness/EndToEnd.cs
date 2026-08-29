// The honest test: does OUR phonemizer produce speech as good as the reference frontend's?
//
//   dotnet run --project tools/zipvoice-harness -c Release -- endtoend [fixtureDir] [outDir]
//
// WHY THIS EXISTS: every measurement of the phonemizer so far compares SYMBOLS against espeak's symbols.
// That is a proxy. It cannot tell a difference that matters from one that does not, it cannot hear a
// stutter, and it silently assumes espeak is the target rather than the audio being right.
//
// This closes the loop. The same sentence is spoken twice from the same reference voice and the same
// noise seed, once from the reference frontend's token ids and once from ours, and both are transcribed.
// Because the two renders differ ONLY in the phonemes they were given, any difference in what a listener
// recovers is attributable to the phonemizer and to nothing else.
//
// Reading it: our WER should be no worse than the reference's. Beating it is not the goal and would
// mostly be noise; being close is the claim, and being much worse is a defect with a name attached.
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.Phonemizer;
using System.Text.Json;

namespace ZipVoiceHarness;

public static class EndToEnd
{
    private sealed record Row(string Fixture, string Text, int Seed, string Source,
                              string Wav, string Transcript, double InfixWer, int UnknownWords);

    public static async Task<int> RunAsync(
        string modelDir,
        IReadOnlyList<(string Path, ZipVoiceFixture Fixture)> fixtures,
        string outDir,
        Func<byte[], (float[] Samples, int SampleRate, int Channels)> readWav,
        Func<float[], int, byte[]> writeWav)
    {
        var tokensPath = Path.Combine(modelDir, "tokens.txt");
        if (!File.Exists(tokensPath)) { Console.WriteLine($"no tokens.txt at {tokensPath}"); return 2; }

        var symToId = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var raw in File.ReadAllLines(tokensPath))
        {
            var line = raw.TrimEnd('\r', '\n');
            int cut = line.LastIndexOf('\t');
            if (cut < 0 || !long.TryParse(line[(cut + 1)..], out var id)) continue;
            symToId.TryAdd(line[..cut], id);
        }

        // ---- Our phonemizer, configured exactly as a consumer would ---------------------------------
        var dictPath = Environment.GetEnvironmentVariable("CMUDICT")
            ?? @"D:\users\tj\Projects\_ref\cmudict\cmudict.dict";
        var ltsPath = Environment.GetEnvironmentVariable("LTS_MODEL")
            ?? Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..",
                                             "SpawnDev.Phonemizer", "lts-model.txt"));
        // The SHIPPED path by default - the data embedded in the assembly, exactly what a consumer gets -
        // so this gate measures the product rather than a development configuration of it. The env vars
        // exist to test a candidate dictionary or a freshly trained model before it is embedded.
        EnglishPhonemizer phonemizer;
        if (Environment.GetEnvironmentVariable("CMUDICT") is { Length: > 0 } && File.Exists(dictPath))
        {
            phonemizer = new EnglishPhonemizer(PronunciationDictionary.Load(dictPath));
            if (File.Exists(ltsPath)) phonemizer.LetterToSound = LetterToSound.Load(ltsPath);
            Console.WriteLine($"phonemizer: files ({dictPath})");
        }
        else
        {
            phonemizer = EmbeddedData.CreatePhonemizer();
            Console.WriteLine("phonemizer: embedded in the assembly (the shipped path)");
        }

        // Switches for ISOLATING a rule's effect on the AUDIO. Symbol agreement with the reference is a
        // proxy; this gate is the thing it is a proxy for, and the two have already disagreed once - a
        // change that halved symbol disagreement left the audio no better.
        if (Environment.GetEnvironmentVariable("PHONEMIZER_NO_OVERRIDES") == "1")
        {
            phonemizer.UseReferenceOverrides = false;
            Console.WriteLine("rules    : reference overrides OFF");
        }
        if (Environment.GetEnvironmentVariable("PHONEMIZER_NO_HOMOGRAPHS") == "1")
        {
            phonemizer.ResolveHomographs = false;
            Console.WriteLine("rules    : homograph resolution OFF");
        }

        var seeds = (Environment.GetEnvironmentVariable("SENSITIVITY_SEEDS") ?? "1234")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(int.Parse).ToArray();

        Directory.CreateDirectory(outDir);
        var resultsPath = Path.Combine(outDir, "endtoend.json");
        var rows = File.Exists(resultsPath)
            ? JsonSerializer.Deserialize<List<Row>>(File.ReadAllText(resultsPath)) ?? new List<Row>()
            : new List<Row>();

        Console.WriteLine($"fixtures : {fixtures.Count}, seeds {string.Join(",", seeds)}");
        Console.WriteLine($"dict     : {dictPath}");
        Console.WriteLine($"outDir   : {outDir}");

        // ---- Turn each sentence into OUR token ids ---------------------------------------------------
        var jobs = new List<(string FixName, ZipVoiceFixture Fix, int Seed, string Source, long[] Tokens, string Wav, int Unknown)>();
        int untranslatable = 0, oovFixtures = 0;
        foreach (var (path, fixture) in fixtures)
        {
            var fixName = Path.GetFileNameWithoutExtension(path);
            var symbols = phonemizer.ToSymbols(fixture.Text);
            int unknownWords = phonemizer.LastUnknownWords.Count;
            if (unknownWords > 0) oovFixtures++;

            var ours = new List<long>(symbols.Count);
            bool translatable = true;
            foreach (var symbol in symbols)
            {
                if (symToId.TryGetValue(symbol, out var id)) { ours.Add(id); continue; }
                // A symbol the model has no token for cannot be spoken at all. Loud, not skipped.
                Console.WriteLine($"  {fixName}: no token for symbol '{symbol}' - this sentence cannot be rendered");
                translatable = false;
                break;
            }
            if (!translatable) { untranslatable++; continue; }

            foreach (var seed in seeds)
            {
                jobs.Add((fixName, fixture, seed, "reference", fixture.Tokens,
                          Path.Combine(outDir, $"{fixName}__reference__s{seed}.wav"), 0));
                var tag = Environment.GetEnvironmentVariable("PHONEMIZER_TAG") is { Length: > 0 } t ? "-" + t : "";
                jobs.Add((fixName, fixture, seed, "ours" + tag, ours.ToArray(),
                          Path.Combine(outDir, $"{fixName}__ours{tag}__s{seed}.wav"), unknownWords));
            }
        }
        if (untranslatable > 0) Console.WriteLine($"WARNING  : {untranslatable} sentence(s) produced a symbol the model has no token for");

        // ---- Is this run able to see letter-to-sound at all? -----------------------------------------
        // The default fixture sets contain ZERO words outside CMUdict, so the guessing path never runs and
        // this gate is structurally blind to it: the numbers come back identical whether that model is
        // improved or deleted. That reads as "no regression" and means "no measurement".
        //
        // So the run SAYS which it is. On an out-of-vocabulary set it also refuses to proceed if the
        // dictionary has quietly grown to cover the words - a gate that has stopped testing the thing it
        // was built to test must fail, not pass. Same rule as the sensitivity harness's positive control.
        bool oovRun = fixtures.Count > 0
            && string.Equals(new DirectoryInfo(Path.GetDirectoryName(fixtures[0].Path)!).Name, "oov",
                             StringComparison.OrdinalIgnoreCase);
        Console.WriteLine($"coverage : {oovFixtures}/{fixtures.Count} sentences contain a word CMUdict lacks");
        if (oovFixtures == 0)
            Console.WriteLine("           => letter-to-sound NEVER RUNS here. This gate cannot see it, and any "
                            + "number below is silent about it.");
        if (oovRun && oovFixtures < fixtures.Count)
        {
            Console.WriteLine($"VOID     : this is the out-of-vocabulary set, and {fixtures.Count - oovFixtures} "
                            + "sentence(s) no longer contain an unknown word - the set has stopped measuring "
                            + "what it exists to measure. Fix the sentences or the guard, do not read the result.");
            return 3;
        }

        Console.WriteLine($"planned  : {jobs.Count} renders");

        // ---- Render ----------------------------------------------------------------------------------
        var toRender = jobs.Where(j => !File.Exists(j.Wav)).ToList();
        if (toRender.Count > 0)
        {
            bool int8 = Environment.GetEnvironmentVariable("ZIPVOICE_INT8") == "1";
            var graphDir = int8 ? modelDir.Replace("zipvoice-distill-zh-en", "zipvoice-distill-int8-zh-en") : modelDir;
            using var graphs = new OrtZipVoiceGraphs(graphDir, int8);
            var config = new ZipVoiceConfig();
            var promptCache = new Dictionary<string, (float[] Audio, int Rate)>();
            int done = 0;

            foreach (var job in toRender)
            {
                var promptWav = Sensitivity.ResolvePromptWav(modelDir, job.Fix);
                if (!promptCache.TryGetValue(promptWav, out var prompt))
                {
                    var (samples, rate, _) = readWav(File.ReadAllBytes(promptWav));
                    promptCache[promptWav] = prompt = (samples, rate);
                }
                using var pipeline = new ZipVoicePipeline(graphs, config) { NoiseSeed = job.Seed };
                var result = await pipeline.SynthesizeAsync(job.Tokens, job.Fix.PromptTokens, prompt.Audio, prompt.Rate);
                File.WriteAllBytes(job.Wav, writeWav(result.Audio, result.SampleRate));
                if (++done % 10 == 0 || done == toRender.Count)
                    Console.WriteLine($"  rendered {done}/{toRender.Count}");
            }
        }

        // ---- Grade -----------------------------------------------------------------------------------
        var toGrade = jobs.Where(j => File.Exists(j.Wav)
                                   && !rows.Any(r => r.Fixture == j.FixName && r.Seed == j.Seed && r.Source == j.Source))
                          .ToList();
        if (toGrade.Count > 0)
        {
            var whisperDir = Environment.GetEnvironmentVariable("WHISPER_MODEL_DIR")
                ?? Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..",
                                                 "SpawnDev.ILGPU.ML.Demo", "wwwroot", "models", "whisper-base.en"));
            if (!Directory.Exists(whisperDir)) { Console.WriteLine($"no whisper model at {whisperDir}"); return 2; }

            var builder = MLContext.Create();
            await builder.AllAcceleratorsAsync();
            var accelerator = await builder.ToContext().CreatePreferredAcceleratorAsync();
            if (accelerator == null) { Console.WriteLine("no accelerator"); return 3; }
            Console.WriteLine($"grader   : {Path.GetFileName(whisperDir)} on {accelerator.AcceleratorType}");

            var encoder = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(Path.Combine(whisperDir, "encoder_model.onnx")));
            var decoder = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(Path.Combine(whisperDir, "decoder_model.onnx")));
            var stt = new SpeechRecognitionPipeline(encoder, decoder, accelerator);
            stt.LoadTokenizer(File.ReadAllText(Path.Combine(whisperDir, "tokenizer.json")));

            foreach (var job in toGrade)
            {
                var (samples, rate, _) = readWav(File.ReadAllBytes(job.Wav));
                var transcript = (await stt.TranscribeAsync(samples, rate)).Text?.Trim() ?? "";
                rows.Add(new Row(job.FixName, job.Fix.Text, job.Seed, job.Source, job.Wav, transcript,
                                 Sensitivity.InfixWerOf(job.Fix.Text, transcript), job.Unknown));
                File.WriteAllText(resultsPath, JsonSerializer.Serialize(rows, new JsonSerializerOptions { WriteIndented = true }));
            }
            Console.WriteLine($"graded   : {toGrade.Count} new, {rows.Count} total");
        }

        return Report(rows, outDir);
    }

    /// <summary>
    /// Compare two of OUR phonemizer configurations against EACH OTHER, on the same sentence and the same
    /// noise seed.
    /// </summary>
    /// <remarks>
    /// The comparison above measures a variant against the REFERENCE, which answers "are we as good as
    /// espeak". It does NOT answer "did my change do anything", and the two are easy to confuse: both
    /// variants can sit the same distance from the reference while differing on the sentences that matter,
    /// or - the case this was written for - be genuinely identical while a run-to-run difference looks
    /// like progress.
    ///
    /// Rendering a second variant with PHONEMIZER_TAG puts both in the same results file, so this fires
    /// automatically. It exists because a single noise seed once showed a letter-to-sound change as 9.2%
    /// against 10.3% - and at three seeds the same two models came out at 9.69% and 9.68%, differing on 4
    /// renders out of 60. Flow matching starts from fresh noise every call, so one render is a sample, not
    /// a measurement, and a variant comparison needs the pairing spelled out or it reports noise as a win.
    /// </remarks>
    private static void ReportVariants(List<Row> rows)
    {
        var variants = rows.Select(r => r.Source).Where(s => s.StartsWith("ours")).Distinct().OrderBy(s => s).ToList();
        if (variants.Count < 2) return;

        Console.WriteLine();
        Console.WriteLine($"VARIANTS, paired against each other on the same sentence and seed");
        for (int a = 0; a < variants.Count; a++)
            for (int b = a + 1; b < variants.Count; b++)
            {
                var left = rows.Where(r => r.Source == variants[a]).ToDictionary(r => (r.Fixture, r.Seed));
                var right = rows.Where(r => r.Source == variants[b]).ToDictionary(r => (r.Fixture, r.Seed));
                var keys = left.Keys.Where(right.ContainsKey).ToList();
                if (keys.Count == 0) continue;

                double dl = keys.Average(k => left[k].InfixWer), dr = keys.Average(k => right[k].InfixWer);
                int lb = keys.Count(k => left[k].InfixWer < right[k].InfixWer - 1e-9);
                int rb = keys.Count(k => right[k].InfixWer < left[k].InfixWer - 1e-9);
                Console.WriteLine($"  {variants[a]} {dl:P2} vs {variants[b]} {dr:P2} over {keys.Count} pairs "
                                + $"({dl - dr:+0.00%;-0.00%;0.00%})");
                Console.WriteLine($"    {variants[a]} better on {lb}, {variants[b]} better on {rb}, "
                                + $"IDENTICAL on {keys.Count - lb - rb}");

                // What this comparison could have SEEN. Without it, "0.00%" is unreadable: it means the
                // same thing whether the change did nothing or the set is too small to notice.
                var (_, sd, detectable) = PairedPower(keys.Select(k => left[k].InfixWer - right[k].InfixWer).ToList());
                Console.WriteLine($"    resolution: this set can detect a difference of about "
                                + $"{detectable:P2} at 95% confidence (paired sd {sd:P2} over {keys.Count} renders)");
                if (Math.Abs(dl - dr) < detectable)
                    Console.WriteLine($"    => the observed {Math.Abs(dl - dr):P2} is BELOW that, so this set cannot "
                                    + "tell these two apart. Not evidence they are the same - evidence the set is "
                                    + "too small to say. Add sentences or seeds (resolution improves as sqrt(n)).");
                foreach (var k in keys.Where(k => Math.Abs(left[k].InfixWer - right[k].InfixWer) > 1e-9)
                                      .OrderBy(k => left[k].InfixWer - right[k].InfixWer))
                {
                    Console.WriteLine($"    {k.Fixture} s{k.Seed}  {variants[a]} {left[k].InfixWer:P0} vs "
                                    + $"{variants[b]} {right[k].InfixWer:P0}");
                    Console.WriteLine($"      want: {left[k].Text}");
                    Console.WriteLine($"      {variants[a],-9}: {left[k].Transcript}");
                    Console.WriteLine($"      {variants[b],-9}: {right[k].Transcript}");
                }
            }
    }

    /// <summary>
    /// What difference this comparison could actually have DETECTED, given how much the paired
    /// differences scatter. Reported next to every result so "no difference" can be read honestly.
    /// </summary>
    /// <remarks>
    /// A gate that cannot resolve the effect you are looking for reports "no change" for a real
    /// improvement and for a no-op alike, and the two are indistinguishable from the output. That already
    /// bit this project once: a letter-to-sound change worth +8.9 points of symbol accuracy moved this
    /// gate 0.00%, which is a genuine finding ONLY if the gate could have seen a smaller move.
    ///
    /// Paired differences cancel sentence difficulty and the noise draw, so the spread here is the
    /// residual - and the 95% detectable difference is ~1.96 * sd / sqrt(n). Widening the set (more
    /// sentences, more seeds) shrinks it as sqrt(n), which is the honest way to buy resolution.
    /// </remarks>
    private static (double Mean, double Sd, double Detectable) PairedPower(IReadOnlyList<double> diffs)
    {
        int n = diffs.Count;
        if (n < 2) return (0, 0, double.PositiveInfinity);
        double mean = diffs.Average();
        double sd = Math.Sqrt(diffs.Sum(d => (d - mean) * (d - mean)) / (n - 1));
        return (mean, sd, 1.96 * sd / Math.Sqrt(n));
    }

    private static int Report(List<Row> rows, string outDir)
    {
        var reference = rows.Where(r => r.Source == "reference").ToDictionary(r => (r.Fixture, r.Seed));
        var wanted = "ours" + (Environment.GetEnvironmentVariable("PHONEMIZER_TAG") is { Length: > 0 } t ? "-" + t : "");
        var ours = rows.Where(r => r.Source == wanted).ToList();
        var paired = ours.Where(o => reference.ContainsKey((o.Fixture, o.Seed)))
                         .Select(o => (Ours: o, Reference: reference[(o.Fixture, o.Seed)])).ToList();
        if (paired.Count == 0) { Console.WriteLine("nothing paired to compare"); return 1; }

        double oursMean = paired.Average(p => p.Ours.InfixWer);
        double refMean = paired.Average(p => p.Reference.InfixWer);
        int worse = paired.Count(p => p.Ours.InfixWer > p.Reference.InfixWer + 0.05);
        int better = paired.Count(p => p.Reference.InfixWer > p.Ours.InfixWer + 0.05);
        int same = paired.Count - worse - better;

        ReportVariants(rows);

        Console.WriteLine();
        Console.WriteLine($"PAIRED, {paired.Count} sentence renders, same voice and same noise seed for both");
        Console.WriteLine($"  reference frontend : {refMean:P1} mean word error");
        Console.WriteLine($"  OUR phonemizer     : {oursMean:P1}");
        Console.WriteLine($"  difference         : {oursMean - refMean:+0.0%;-0.0%;0.0%}");

        var (_, refSd, refDetectable) = PairedPower(paired.Select(p => p.Ours.InfixWer - p.Reference.InfixWer).ToList());
        Console.WriteLine($"  resolution         : detects ~{refDetectable:P2} at 95% confidence "
                        + $"(paired sd {refSd:P2} over {paired.Count} renders)"
                        + (Math.Abs(oursMean - refMean) < refDetectable
                            ? " - the difference above is BELOW that, i.e. not resolvable by this set"
                            : ""));
        Console.WriteLine();
        Console.WriteLine($"  ours clearly worse : {worse}/{paired.Count}");
        Console.WriteLine($"  indistinguishable  : {same}/{paired.Count}");
        Console.WriteLine($"  ours better        : {better}/{paired.Count}");

        var worstFirst = paired.Where(p => p.Ours.InfixWer > p.Reference.InfixWer + 0.05)
                               .OrderByDescending(p => p.Ours.InfixWer - p.Reference.InfixWer).Take(8).ToList();
        if (worstFirst.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Where ours lost, worst first - these are the sentences to look at:");
            foreach (var p in worstFirst)
            {
                Console.WriteLine($"  {p.Ours.Fixture} s{p.Ours.Seed}  ours {p.Ours.InfixWer:P0} vs {p.Reference.InfixWer:P0}");
                Console.WriteLine($"    wanted: {p.Ours.Text}");
                Console.WriteLine($"    ours  : {p.Ours.Transcript}");
                Console.WriteLine($"    ref   : {p.Reference.Transcript}");
            }
        }

        Console.WriteLine();
        Console.WriteLine($"rows in {Path.Combine(outDir, "endtoend.json")}. WER hears wrong WORDS only - a clean");
        Console.WriteLine("row still deserves a listen, which is what tools/zipvoice-listen is for.");
        return 0;
    }
}
