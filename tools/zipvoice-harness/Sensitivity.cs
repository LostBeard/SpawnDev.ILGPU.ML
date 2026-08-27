// Phase 1 of Plans/mit-phonemizer-2026-08-27.md - HOW WRONG IS THE MODEL ALLOWED TO BE?
//
//   dotnet run --project tools/zipvoice-harness -c Release -- sensitivity [fixtureDirOrFile] [outDir]
//
// WHY THIS EXISTS: we are replacing espeak-ng (GPL) with an MIT frontend built on CMUdict. CMUdict is
// PHONEMIC (ARPAbet); the token stream this model was TRAINED on is espeak's narrower, allophonic
// output - flaps, reduced vowels, r-coloured schwa, explicit stress. So a dictionary-based frontend
// will differ from espeak in a small number of predictable ways no matter how carefully it is written.
//
// The question that decides how much precision the frontend needs is not "how close can we get" but
// "which differences does the MODEL actually care about". That is measurable before a line of the
// frontend exists: take the oracle's own correct token ids, damage them in exactly the ways CMUdict
// will, and grade what comes out.
//
// WHAT MAKES THE ANSWER TRUSTWORTHY, rather than one anecdote:
//   * PAIRED. Every perturbation is compared against the control rendered from the SAME sentence with
//     the SAME noise seed, so sentence difficulty and the noise draw cancel instead of being confounds.
//   * REPLICATED. Many sentences and several seeds. Flow matching starts from fresh noise, so a single
//     render is a sample, not a measurement.
//   * POSITIVE CONTROL. "wrong-vowel-last-word" deliberately mispronounces one word. If that row does
//     NOT show damage, the grader cannot see damage at all and every clean row is meaningless. A
//     perturbation study without it can only ever confirm what it hoped.
//   * NO-OP ROWS DROPPED. A sentence with no flap in it cannot say anything about flaps; those rows
//     have zero token edits and are excluded from the aggregate rather than diluting it toward zero.
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using System.Text.Json;

namespace ZipVoiceHarness;

public static class Sensitivity
{
    // A perturbation is named for the CMUdict-shaped error it stands in for, so the result table reads
    // as advice about the frontend rather than as a list of symbol edits.
    private sealed record Variant(string Name, string Why, Func<long[], long[]> Apply);

    private sealed record Row(
        string Fixture, int Seed, string Variant, int Edits,
        string Wav, string Transcript, double PlainWer, double InfixWer)
    {
        /// <summary>Acoustic distance from the control rendered with the SAME sentence and seed.</summary>
        /// <remarks>
        /// Nullable and defaulted so a results.json written before this existed still loads. Filled in by a
        /// pass after grading, because it needs the paired control audio.
        /// </remarks>
        public double? MelDistance { get; set; }

        /// <summary>Distance between two CONTROL renders of one sentence at different seeds.</summary>
        /// <remarks>
        /// The null model. Flow matching starts from fresh noise, so two identical requests already differ
        /// acoustically; without knowing by how much, a perturbation distance means nothing.
        /// </remarks>
        public double? SeedBaseline { get; set; }
    }

    public static async Task<int> RunAsync(
        string modelDir,
        IReadOnlyList<(string Path, ZipVoiceFixture Fixture)> fixtures,
        string outDir,
        Func<byte[], (float[] Samples, int SampleRate, int Channels)> readWav,
        Func<float[], int, byte[]> writeWav)
    {
        var tokensPath = Path.Combine(modelDir, "tokens.txt");
        if (!File.Exists(tokensPath)) { Console.WriteLine($"no tokens.txt at {tokensPath}"); return 2; }
        var (idToSym, symToId) = LoadTokens(tokensPath);

        long Id(string sym) => symToId.TryGetValue(sym, out var id)
            ? id
            : throw new InvalidOperationException($"symbol '{sym}' is not in {tokensPath}");

        // espeak's own symbols, looked up rather than hard-coded, so a different model's table still works.
        long flap = Id("ɾ");        // alveolar tap, what espeak writes for flapped t/d
        long t = Id("t"), d = Id("d");
        long barredI = Id("ᵻ"), smallCapI = Id("ɪ");
        long turnedA = Id("ɐ"), schwa = Id("ə");
        long rSchwa = Id("ɚ"), turnedR = Id("ɹ");
        long primary = Id("ˈ"), secondary = Id("ˌ"), length = Id("ː");
        long closeI = Id("i"), openO = Id("ɔ"), openA = Id("ɑ");
        long space = Id(" "), plainA = Id("a"), glottal = Id("ʔ"), syllabic = Id("̩");

        // Every vowel in the espeak inventory this model uses. Needed to move or replace a stress-bearing
        // vowel, because stress attaches to the vowel that follows the mark.
        var vowels = new HashSet<long>("aeiouæɐɑɒɔəɚɘɛɜɞɤɨɪɯʉʊʌʏɵɶøœᵻ"
            .Select(c => symToId.TryGetValue(c.ToString(), out var v) ? v : -1).Where(v => v >= 0));

        var variants = new List<Variant>
        {
            new("control", "unmodified ground truth - proves the rig, not the perturbation",
                s => (long[])s.Clone()),

            new("flap-to-t", "CMUdict has no flap: water/better/city come back with a plain T",
                s => s.Select(x => x == flap ? t : x).ToArray()),

            new("flap-to-d", "the other plausible un-flapping, since espeak flaps both T and D",
                s => s.Select(x => x == flap ? d : x).ToArray()),

            new("barred-i-to-small-i", "espeak's reduced high vowel in roses; ARPAbet has only IH",
                s => s.Select(x => x == barredI ? smallCapI : x).ToArray()),

            new("turned-a-to-schwa", "espeak's open reduced vowel in about; ARPAbet has only AH0",
                s => s.Select(x => x == turnedA ? schwa : x).ToArray()),

            new("r-schwa-split", "ARPAbet ER becomes schwa plus R rather than one r-coloured vowel",
                s => s.SelectMany(x => x == rSchwa ? new[] { schwa, turnedR } : new[] { x }).ToArray()),

            new("no-secondary-stress", "CMUdict marks stress 0/1/2 but a naive mapping drops the 2s",
                s => s.Where(x => x != secondary).ToArray()),

            new("no-stress-at-all", "worst case: a mapping that emits no stress marks whatsoever",
                s => s.Where(x => x != primary && x != secondary).ToArray()),

            new("stress-moved-later", "stress on the wrong syllable, the classic dictionary failure",
                s => MoveStressLater(s, primary, vowels)),

            new("no-length-marks", "the length mark has no ARPAbet counterpart and is easy to omit",
                s => s.Where(x => x != length).ToArray()),

            // ---- Classes found by tools/zipvoice-g2p-probe in REAL CMUdict output ----------------------
            // The rows above were my PREDICTION of what CMUdict gets wrong. The probe aligned real
            // CMUdict-derived sequences against real espeak ids and found these, which the prediction
            // missed. An untested error class is an unmeasured risk, so they are measured here.

            new("stress-added-function-words", "PROBE, 19.4% of real differences: CMUdict gives every "
                + "word its citation-form stress, but espeak leaves function words (at, for, in, and, a, "
                + "is, it, of, than) unstressed in connected speech",
                s => StressEveryUnstressedWord(s, primary, secondary, vowels, space)),

            new("barred-i-to-schwa", "PROBE, 6.5%: espeak writes the barred-i reduced vowel in packages, "
                + "collected and boxes where CMUdict has AH0, which maps to schwa",
                s => s.Select(x => x == barredI ? schwa : x).ToArray()),

            new("article-a-to-schwa", "PROBE, 6.5%: espeak emits a bare 'a' for the article; CMUdict's "
                + "AH0 maps to schwa",
                s => ReplaceWholeWord(s, space, new[] { plainA }, new[] { schwa })),

            new("glottal-and-syllabic-to-plain", "PROBE: espeak writes kitten with a glottal stop and a "
                + "syllabic n; a dictionary mapping produces a plain t and a schwa",
                s => GlottalAndSyllabicToPlain(s, glottal, syllabic, t, schwa)),

            new("open-o-to-open-a", "PROBE: the two dictionaries disagree about which vowel some words "
                + "take - espeak says 'on' has the open-o, CMUdict says the open-a",
                s => s.Select(x => x == openO ? openA : x).ToArray()),

            // POSITIVE CONTROL. Not a CMUdict error - a deliberate mispronunciation of one word, so that
            // "no damage" rows can be believed. If this scores clean, the grader is blind and the run is
            // void; the summary says so rather than reporting a comfortable result.
            new("wrong-vowel-last-word", "POSITIVE CONTROL: one word deliberately mispronounced",
                s => WrongVowelInLastStressedWord(s, primary, vowels, closeI, openO)),
        };

        var only = Environment.GetEnvironmentVariable("SENSITIVITY_ONLY");
        if (!string.IsNullOrWhiteSpace(only))
            variants = variants.Where(v => v.Name == "control" || v.Name == only).ToList();

        var seeds = (Environment.GetEnvironmentVariable("SENSITIVITY_SEEDS") ?? "1234")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(int.Parse).ToArray();

        Directory.CreateDirectory(outDir);
        var resultsPath = Path.Combine(outDir, "results.json");

        // Resumable: a run of this size does not fit one sitting, and re-rendering audio that already
        // exists buys nothing when synthesis is deterministic (pinned seed, pinned inputs).
        var rows = File.Exists(resultsPath)
            ? JsonSerializer.Deserialize<List<Row>>(File.ReadAllText(resultsPath)) ?? new List<Row>()
            : new List<Row>();
        bool regrade = Environment.GetEnvironmentVariable("SENSITIVITY_REGRADE") == "1";
        if (regrade) rows.Clear();

        Console.WriteLine($"fixtures : {fixtures.Count}");
        Console.WriteLine($"seeds    : {string.Join(", ", seeds)}");
        Console.WriteLine($"variants : {variants.Count}");
        Console.WriteLine($"planned  : {fixtures.Count * seeds.Length * variants.Count} renders, "
                        + $"{rows.Count} already in {Path.GetFileName(resultsPath)}");
        Console.WriteLine($"outDir   : {outDir}");
        Console.WriteLine();

        bool int8 = Environment.GetEnvironmentVariable("ZIPVOICE_INT8") == "1";
        var graphDir = int8
            ? modelDir.Replace("zipvoice-distill-zh-en", "zipvoice-distill-int8-zh-en")
            : modelDir;
        var config = new ZipVoiceConfig();

        float tailPad = 0.25f;
        if (Environment.GetEnvironmentVariable("ZIPVOICE_NO_PAD") == "1") tailPad = 0f;
        if (float.TryParse(Environment.GetEnvironmentVariable("ZIPVOICE_TAIL_PAD"), out var padOverride))
            tailPad = padOverride;

        // ---- Render ---------------------------------------------------------------------------------
        var jobs = new List<(string FixName, ZipVoiceFixture Fix, int Seed, Variant V, long[] Tokens, string Wav)>();
        foreach (var (path, fixture) in fixtures)
        {
            var fixName = Path.GetFileNameWithoutExtension(path);
            foreach (var seed in seeds)
                foreach (var variant in variants)
                    jobs.Add((fixName, fixture, seed, variant, variant.Apply(fixture.Tokens),
                              Path.Combine(outDir, $"{fixName}__{variant.Name}__s{seed}.wav")));
        }

        var toRender = jobs.Where(j => !File.Exists(j.Wav)).ToList();
        if (toRender.Count > 0)
        {
            using var graphs = new OrtZipVoiceGraphs(graphDir, int8);
            var promptCache = new Dictionary<string, (float[] Audio, int Rate)>();
            int done = 0;
            var swAll = System.Diagnostics.Stopwatch.StartNew();
            foreach (var job in toRender)
            {
                var promptWav = ResolvePromptWav(modelDir, job.Fix);
                if (!promptCache.TryGetValue(promptWav, out var prompt))
                {
                    var (samples, rate, _) = readWav(File.ReadAllBytes(promptWav));
                    prompt = (samples, rate);
                    promptCache[promptWav] = prompt;
                }

                using var pipeline = new ZipVoicePipeline(graphs, config)
                {
                    NoiseSeed = job.Seed,
                    ReferenceTailSilenceSeconds = tailPad,
                };
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var result = await pipeline.SynthesizeAsync(job.Tokens, job.Fix.PromptTokens, prompt.Audio, prompt.Rate);
                File.WriteAllBytes(job.Wav, writeWav(result.Audio, result.SampleRate));
                done++;
                Console.WriteLine($"[{done}/{toRender.Count}] {job.FixName} {job.V.Name} s{job.Seed} "
                                + $"{TokenEdits(job.Fix.Tokens, job.Tokens)} edits, "
                                + $"{result.DurationSeconds:F1}s audio in {sw.Elapsed.TotalSeconds:F0}s "
                                + $"(elapsed {swAll.Elapsed.TotalMinutes:F1}m)");
            }
        }

        if (Environment.GetEnvironmentVariable("SENSITIVITY_NO_GRADE") == "1")
        {
            Console.WriteLine($"\ngrading skipped (SENSITIVITY_NO_GRADE=1). Audio is in {outDir}");
            return 0;
        }

        // ---- Grade ----------------------------------------------------------------------------------
        // Whisper is the grader because the question is what a LISTENER recovers from the audio, which is
        // the only definition of "the model rendered it correctly" that does not beg the question.
        var toGrade = jobs.Where(j => File.Exists(j.Wav)
                                   && !rows.Any(r => r.Fixture == j.FixName && r.Seed == j.Seed && r.Variant == j.V.Name))
                          .ToList();
        if (toGrade.Count > 0)
        {
            var whisperDir = Environment.GetEnvironmentVariable("WHISPER_MODEL_DIR")
                ?? Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..",
                                "SpawnDev.ILGPU.ML.Demo", "wwwroot", "models", "whisper-tiny");
            whisperDir = Path.GetFullPath(whisperDir);
            if (!Directory.Exists(whisperDir))
            {
                Console.WriteLine($"no whisper model at {whisperDir} - set WHISPER_MODEL_DIR.");
                return 2;
            }

            Console.WriteLine($"\ngrader   : whisper at {whisperDir}");
            var builder = MLContext.Create();
            await builder.AllAcceleratorsAsync();
            var mlContext = builder.ToContext();
            var accelerator = await mlContext.CreatePreferredAcceleratorAsync();
            if (accelerator == null) { Console.WriteLine("no accelerator for the grader"); return 3; }
            Console.WriteLine($"device   : {accelerator.AcceleratorType} {accelerator.Name}");

            var encoder = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(Path.Combine(whisperDir, "encoder_model.onnx")));
            var decoder = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(Path.Combine(whisperDir, "decoder_model.onnx")));
            InferenceSession? withPast = null;
            var withPastPath = Path.Combine(whisperDir, "decoder_with_past_model.onnx");
            if (File.Exists(withPastPath))
                withPast = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(withPastPath));
            var stt = new SpeechRecognitionPipeline(encoder, decoder, accelerator, withPast);
            // Without the tokenizer the pipeline reports raw token ids ("[407] [385] ...") rather than
            // words, which scores as garbage against any reference and would condemn good audio.
            var tokenizerJson = Path.Combine(whisperDir, "tokenizer.json");
            if (!File.Exists(tokenizerJson)) { Console.WriteLine($"no tokenizer.json at {tokenizerJson}"); return 2; }
            stt.LoadTokenizer(File.ReadAllText(tokenizerJson));

            int graded = 0;
            foreach (var job in toGrade)
            {
                var (samples, rate, _) = readWav(File.ReadAllBytes(job.Wav));
                var transcript = (await stt.TranscribeAsync(samples, rate)).Text?.Trim() ?? "";
                var truthWords = Words(job.Fix.Text);
                var hypWords = Words(transcript);
                rows.Add(new Row(job.FixName, job.Seed, job.V.Name, TokenEdits(job.Fix.Tokens, job.Tokens),
                                 job.Wav, transcript, Wer(truthWords, hypWords), InfixWer(truthWords, hypWords)));
                graded++;
                // Written every time: a run this long must never lose finished work to an interruption.
                File.WriteAllText(resultsPath, JsonSerializer.Serialize(rows, new JsonSerializerOptions { WriteIndented = true }));
            }
            Console.WriteLine($"graded   : {graded} new, {rows.Count} total");
        }

        // ---- Acoustic distance ----------------------------------------------------------------------
        // Whisper can repair a mispronunciation back into the expected word, so WER alone cannot tell
        // "no damage" from "damage the language model papered over". This measures the audio itself.
        var byKey = rows.ToDictionary(r => (r.Fixture, r.Seed, r.Variant));
        var needDistance = rows.Where(r => r.MelDistance == null && r.Variant != "control"
                                        && byKey.ContainsKey((r.Fixture, r.Seed, "control"))).ToList();
        var needBaseline = rows.Where(r => r.Variant == "control" && r.SeedBaseline == null).ToList();
        if (needDistance.Count > 0 || needBaseline.Count > 0)
        {
            Console.WriteLine($"\nacoustic : {needDistance.Count} distances, {needBaseline.Count} seed baselines");
            var audio = new Dictionary<string, (float[] Samples, int Rate)>();
            (float[] Samples, int Rate) Audio(string wav)
            {
                if (!audio.TryGetValue(wav, out var got))
                {
                    var (samples, rate, _) = readWav(File.ReadAllBytes(wav));
                    audio[wav] = got = (samples, rate);
                }
                return got;
            }

            foreach (var r in needDistance)
            {
                var c = byKey[(r.Fixture, r.Seed, "control")];
                if (!File.Exists(r.Wav) || !File.Exists(c.Wav)) continue;
                var x = Audio(r.Wav); var y = Audio(c.Wav);
                r.MelDistance = AcousticDistance.Between(x.Samples, x.Rate, y.Samples, y.Rate);
            }

            // The null model: the same tokens rendered from a DIFFERENT noise draw.
            foreach (var group in needBaseline.GroupBy(r => r.Fixture))
            {
                var members = group.OrderBy(r => r.Seed).ToList();
                if (members.Count < 2) continue;
                for (int i = 0; i < members.Count; i++)
                {
                    var other = members[(i + 1) % members.Count];
                    if (!File.Exists(members[i].Wav) || !File.Exists(other.Wav)) continue;
                    var x = Audio(members[i].Wav); var y = Audio(other.Wav);
                    members[i].SeedBaseline = AcousticDistance.Between(x.Samples, x.Rate, y.Samples, y.Rate);
                }
            }
            File.WriteAllText(resultsPath, JsonSerializer.Serialize(rows, new JsonSerializerOptions { WriteIndented = true }));
        }

        return Report(rows, variants, outDir);
    }

    // ---- Reporting ------------------------------------------------------------------------------------
    private static int Report(List<Row> rows, List<Variant> variants, string outDir)
    {
        Console.WriteLine();
        var controls = rows.Where(r => r.Variant == "control").ToDictionary(r => (r.Fixture, r.Seed), r => r);
        if (controls.Count == 0) { Console.WriteLine("no control rows - nothing can be read from this."); return 1; }

        Console.WriteLine("CONTROLS (undamaged ground truth - the floor every other row is read against)");
        foreach (var c in controls.Values.OrderBy(c => c.Fixture).ThenBy(c => c.Seed))
            Console.WriteLine($"  {c.Fixture,-46} s{c.Seed,-8} {c.InfixWer,6:P0}");
        var badControls = controls.Values.Where(c => c.InfixWer > 0.15).ToList();
        Console.WriteLine($"  mean {controls.Values.Average(c => c.InfixWer):P1} over {controls.Count} renders"
                        + (badControls.Count > 0 ? $", {badControls.Count} above 15%" : ""));

        // Paired: each perturbation minus the control of the SAME sentence and seed.
        Console.WriteLine();
        Console.WriteLine("PAIRED DAMAGE (perturbation minus its own control; no-op rows excluded)");
        // The acoustic null model: how far apart two renders of the SAME tokens land purely because the
        // noise draw differed. A perturbation that moves the audio less than this moved nothing.
        var baselines = controls.Values.Where(c => c.SeedBaseline is > 0).Select(c => c.SeedBaseline.Value).ToList();
        double noiseFloor = baselines.Count > 0 ? baselines.Average() : double.NaN;

        Console.WriteLine($"  {"variant",-30} {"n",3} {"WER+",7} {"worst",7} {"hurt",7} {"sound",7}");
        Console.WriteLine("  " + new string('-', 76));

        var summary = new List<(string Name, int N, double Mean, double Worst, int Hurt, double Sound)>();
        foreach (var variant in variants.Where(v => v.Name != "control"))
        {
            var paired = new List<double>();
            var sounds = new List<double>();
            foreach (var r in rows.Where(r => r.Variant == variant.Name))
            {
                if (r.Edits == 0) continue;                                   // nothing was perturbed here
                if (!controls.TryGetValue((r.Fixture, r.Seed), out var c)) continue;
                paired.Add(r.InfixWer - c.InfixWer);
                if (r.MelDistance is > 0) sounds.Add(r.MelDistance.Value);
            }
            if (paired.Count == 0)
            {
                Console.WriteLine($"  {variant.Name,-30} {0,3}       -       -       -       -  never applied");
                continue;
            }
            double mean = paired.Average(), worst = paired.Max();
            double sound = sounds.Count > 0 && noiseFloor > 0 ? sounds.Average() / noiseFloor : double.NaN;
            int hurt = paired.Count(p => p > 0.05);
            summary.Add((variant.Name, paired.Count, mean, worst, hurt, sound));
            Console.WriteLine($"  {variant.Name,-30} {paired.Count,3} {mean,6:P1} {worst,6:P1} {hurt,4}/{paired.Count,-2} "
                            + (double.IsNaN(sound) ? "      -" : $"{sound,6:F2}x"));
        }
        Console.WriteLine();
        Console.WriteLine("  WER+ is infix WER above the paired control. 'sound' is acoustic distance from that same");
        Console.WriteLine("  control, as a multiple of the noise floor: two renders of the SAME tokens at different");
        Console.WriteLine($"  seeds sit {noiseFloor:F2} apart, which is 1.00x. Near 1x means the perturbation changed");
        Console.WriteLine("  nothing audible; well above 1x with a low WER means it sounds different but stays");
        Console.WriteLine("  intelligible - which WER alone cannot see, and which Whisper tends to hide.");

        // ---- Is the result readable at all? ---------------------------------------------------------
        Console.WriteLine();
        var positive = summary.FirstOrDefault(s => s.Name == "wrong-vowel-last-word");
        if (positive.Name == null)
            Console.WriteLine("NOTE     : no positive control in this run, so clean rows cannot be fully trusted.");
        else if (positive.Mean <= 0.05)
        {
            Console.WriteLine($"RESULT   : VOID - the positive control (a deliberate mispronunciation) scored "
                            + $"{positive.Mean:P1}, so the grader cannot see damage that IS there. Every clean row "
                            + "in this table is unreadable. Fix the grader before drawing any conclusion.");
            return 1;
        }
        else
            Console.WriteLine($"positive control damaged the transcript by {positive.Mean:P1}, so the grader can see "
                            + "damage when it exists and a clean row means something.");

        if (badControls.Count > 0)
            Console.WriteLine($"WARNING  : {badControls.Count} control render(s) above 15% infix WER "
                            + $"({string.Join(", ", badControls.Select(c => $"{c.Fixture} s{c.Seed} {c.InfixWer:P0}"))}). "
                            + "Those sentences are hard for the grader; their rows carry less weight.");

        var real = summary.Where(s => s.Name != "wrong-vowel-last-word").ToList();
        var damaging = real.Where(s => s.Mean > 0.05).OrderByDescending(s => s.Mean).ToList();
        var clean = real.Where(s => s.Mean <= 0.05).OrderBy(s => s.Mean).ToList();

        Console.WriteLine();
        Console.WriteLine("The frontend MUST get these right:");
        foreach (var s in damaging) Console.WriteLine($"  {s.Name,-30} {s.Mean,6:P1} mean WER+, {s.Hurt}/{s.N} hurt, sound {s.Sound:F2}x");
        if (damaging.Count == 0) Console.WriteLine("  (none)");
        Console.WriteLine("The model does not care about these - do not spend effort on them:");
        foreach (var s in clean) Console.WriteLine($"  {s.Name,-30} {s.Mean,6:P1} mean WER+, {s.Hurt}/{s.N} hurt, sound {s.Sound:F2}x");
        if (clean.Count == 0) Console.WriteLine("  (none)");

        Console.WriteLine();
        Console.WriteLine($"rows in {Path.Combine(outDir, "results.json")}; audio beside it. WER is coarse - it hears a "
                        + "wrong word, not an accent, so a 0% row still deserves a listen.");
        return 0;
    }

    // ---- Perturbations --------------------------------------------------------------------------------

    // Move every primary stress mark from the vowel it marks to the NEXT vowel, which is what a
    // dictionary-driven frontend does when it picks the wrong syllable.
    private static long[] MoveStressLater(long[] seq, long primary, HashSet<long> vowels)
    {
        var outSeq = new List<long>(seq.Length);
        for (int i = 0; i < seq.Length; i++)
        {
            if (seq[i] != primary) { outSeq.Add(seq[i]); continue; }
            int j = i + 1;
            while (j < seq.Length && !vowels.Contains(seq[j])) { outSeq.Add(seq[j]); j++; }
            if (j < seq.Length) { outSeq.Add(seq[j]); j++; }          // the originally stressed vowel
            while (j < seq.Length && !vowels.Contains(seq[j])) { outSeq.Add(seq[j]); j++; }
            if (j < seq.Length) { outSeq.Add(primary); outSeq.Add(seq[j]); } // stress the next one
            i = j;
        }
        return outSeq.ToArray();
    }

    // POSITIVE CONTROL: swap the stressed vowel of the last stressed word for a distant one, turning that
    // word into a different word. Small enough to be comparable with the other rows, unambiguous enough
    // that a working grader must notice it.
    private static long[] WrongVowelInLastStressedWord(long[] seq, long primary, HashSet<long> vowels, long closeI, long openO)
    {
        var outSeq = (long[])seq.Clone();
        for (int i = seq.Length - 2; i >= 0; i--)
        {
            if (seq[i] != primary) continue;
            int j = i + 1;
            while (j < seq.Length && !vowels.Contains(seq[j])) j++;
            if (j >= seq.Length) continue;
            outSeq[j] = seq[j] == closeI ? openO : closeI;
            break;
        }
        return outSeq;
    }

    // Give a primary stress mark to every word that has none, which is what a dictionary lookup does to
    // function words: CMUdict stores citation forms, where "for" and "at" carry stress that they lose in
    // running speech.
    private static long[] StressEveryUnstressedWord(long[] seq, long primary, long secondary, HashSet<long> vowels, long space)
    {
        var outSeq = new List<long>(seq.Length + 8);
        foreach (var word in SplitOn(seq, space))
        {
            if (word.Any(x => x == primary || x == secondary) || !word.Any(vowels.Contains))
            {
                AppendWord(outSeq, word, space);
                continue;
            }
            bool placed = false;
            foreach (var sym in word)
            {
                if (!placed && vowels.Contains(sym)) { outSeq.Add(primary); placed = true; }
                outSeq.Add(sym);
            }
            outSeq.Add(space);
        }
        TrimTrailing(outSeq, space);
        return outSeq.ToArray();
    }

    // Replace whole words that consist of exactly one symbol sequence, leaving that sequence inside longer
    // words alone - the article "a" must not match the "a" inside "about".
    private static long[] ReplaceWholeWord(long[] seq, long space, long[] match, long[] replacement)
    {
        var outSeq = new List<long>(seq.Length);
        foreach (var word in SplitOn(seq, space))
        {
            AppendWord(outSeq, word.SequenceEqual(match) ? replacement : word, space);
        }
        TrimTrailing(outSeq, space);
        return outSeq.ToArray();
    }

    // A glottal stop becomes a plain t, and a syllabic consonant becomes schwa plus that consonant.
    private static long[] GlottalAndSyllabicToPlain(long[] seq, long glottal, long syllabic, long t, long schwa)
    {
        var outSeq = new List<long>(seq.Length + 4);
        for (int i = 0; i < seq.Length; i++)
        {
            if (seq[i] == glottal) { outSeq.Add(t); continue; }
            if (i + 1 < seq.Length && seq[i + 1] == syllabic)
            {
                outSeq.Add(schwa);      // the vowel a syllabic consonant stands in for
                outSeq.Add(seq[i]);
                i++;                    // consume the syllabic mark
                continue;
            }
            outSeq.Add(seq[i]);
        }
        return outSeq.ToArray();
    }

    private static List<long[]> SplitOn(long[] seq, long sep)
    {
        var words = new List<long[]>();
        var cur = new List<long>();
        foreach (var x in seq)
        {
            if (x == sep) { if (cur.Count > 0) { words.Add(cur.ToArray()); cur.Clear(); } }
            else cur.Add(x);
        }
        if (cur.Count > 0) words.Add(cur.ToArray());
        return words;
    }

    private static void AppendWord(List<long> outSeq, IEnumerable<long> word, long space)
    {
        outSeq.AddRange(word);
        outSeq.Add(space);
    }

    private static void TrimTrailing(List<long> outSeq, long space)
    {
        while (outSeq.Count > 0 && outSeq[^1] == space) outSeq.RemoveAt(outSeq.Count - 1);
    }

    // ---- Helpers --------------------------------------------------------------------------------------

    public static string ResolvePromptWav(string modelDir, ZipVoiceFixture fixture)
        => Path.IsPathRooted(fixture.PromptWav) ? fixture.PromptWav : Path.Combine(modelDir, fixture.PromptWav);

    private static (Dictionary<long, string> IdToSym, Dictionary<string, long> SymToId) LoadTokens(string path)
    {
        var idToSym = new Dictionary<long, string>();
        var symToId = new Dictionary<string, long>();
        foreach (var raw in File.ReadAllLines(path))
        {
            var line = raw.TrimEnd('\r', '\n');
            if (line.Length == 0) continue;
            // The symbol itself can be a space, so split on the LAST tab.
            int cut = line.LastIndexOf('\t');
            if (cut < 0 || !long.TryParse(line[(cut + 1)..], out var id)) continue;
            idToSym[id] = line[..cut];
            symToId.TryAdd(line[..cut], id);
        }
        return (idToSym, symToId);
    }

    public static string Render(long[] tokens, Dictionary<long, string> idToSym)
        => string.Concat(tokens.Select(t => idToSym.TryGetValue(t, out var s) ? s : "?"));

    private static string[] Words(string text)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var c in text.ToLowerInvariant())
            sb.Append(char.IsLetterOrDigit(c) || c == '\'' ? c : ' ');
        return sb.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }

    private static double Wer(string[] truth, string[] hyp)
        => truth.Length == 0 ? (hyp.Length == 0 ? 0 : 1) : Edits(truth, hyp) / (double)truth.Length;

    /// <summary>
    /// WER that does not charge for words before or after the sentence we asked for.
    /// </summary>
    /// <remarks>
    /// ZipVoice regenerates the reference clip's own speech in front of the line it is asked to speak, and
    /// the cut at the prompt boundary does not land cleanly, so the transcript begins with a few words of
    /// the reference. That is NOT our pipeline: sherpa-onnx, the independent implementation, does it too
    /// and worse - graded here, sherpa's own output opens "others call me mother nature" before reaching
    /// the sentence. Charging for those words puts a floor under every row and buries the effect being
    /// measured. Free skips at the head and tail remove the artifact while still charging full price for
    /// every substitution, deletion and insertion INSIDE the sentence, which is where a mispronunciation
    /// shows up. The plain WER is recorded alongside it so nothing is hidden.
    /// </remarks>
    private static double InfixWer(string[] truth, string[] hyp)
    {
        if (truth.Length == 0) return hyp.Length == 0 ? 0 : 1;
        var prev = new int[hyp.Length + 1];        // row 0 all zeros: starting anywhere is free
        var cur = new int[hyp.Length + 1];
        for (int i = 1; i <= truth.Length; i++)
        {
            cur[0] = i;
            for (int j = 1; j <= hyp.Length; j++)
                cur[j] = Math.Min(Math.Min(prev[j] + 1, cur[j - 1] + 1),
                                  prev[j - 1] + (truth[i - 1] == hyp[j - 1] ? 0 : 1));
            (prev, cur) = (cur, prev);
        }
        return prev.Min() / (double)truth.Length;  // ending anywhere is free
    }

    private static int TokenEdits(long[] a, long[] b)
        => Edits(a.Select(x => x.ToString()).ToArray(), b.Select(x => x.ToString()).ToArray());

    // Plain Levenshtein. Small inputs, so the full table is easier to trust than a banded one.
    private static int Edits(string[] a, string[] b)
    {
        var prev = new int[b.Length + 1];
        var cur = new int[b.Length + 1];
        for (int j = 0; j <= b.Length; j++) prev[j] = j;
        for (int i = 1; i <= a.Length; i++)
        {
            cur[0] = i;
            for (int j = 1; j <= b.Length; j++)
                cur[j] = Math.Min(Math.Min(prev[j] + 1, cur[j - 1] + 1),
                                  prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
            (prev, cur) = (cur, prev);
        }
        return prev[b.Length];
    }
}
