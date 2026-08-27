// Phase 1 of Plans/mit-phonemizer-2026-08-27.md - HOW WRONG IS THE MODEL ALLOWED TO BE?
//
//   dotnet run --project tools/zipvoice-harness -c Release -- sensitivity [fixture.json] [outDir]
//
// WHY THIS EXISTS: we are replacing espeak-ng (GPL) with an MIT frontend built on CMUdict. CMUdict is
// PHONEMIC (ARPAbet); the token stream this model was TRAINED on is espeak's narrower, allophonic
// output - flaps, reduced vowels, r-coloured schwa, explicit stress. So a dictionary-based frontend
// will differ from espeak in a small number of predictable ways no matter how carefully it is written.
//
// The question that decides how much precision the frontend needs is not "how close can we get" but
// "which differences does the MODEL actually care about". That is measurable right now, before a line
// of the frontend exists: take the oracle's own correct token ids, damage them in exactly the ways
// CMUdict will, and listen to what comes out.
//
// The experiment is controlled because the only thing that varies between runs is the token sequence:
// same graphs, same reference clip, same prompt tokens, and NoiseSeed pinned so flow matching starts
// from identical noise every time. The "control" variant is the undamaged sequence - if the control
// does not transcribe cleanly, the rig is what is being measured and every other row is meaningless.
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;

namespace ZipVoiceHarness;

public static class Sensitivity
{
    // A perturbation is named for the CMUdict-shaped error it stands in for, so the result table reads
    // as advice about the frontend rather than as a list of symbol edits.
    private sealed record Variant(string Name, string Why, Func<long[], long[]> Apply);

    public static async Task<int> RunAsync(
        string modelDir,
        ZipVoiceFixture fixture,
        string promptWav,
        string outDir,
        Func<byte[], (float[] Samples, int SampleRate, int Channels)> readWav,
        Func<float[], int, byte[]> writeWav)
    {
        var tokensPath = Path.Combine(modelDir, "tokens.txt");
        if (!File.Exists(tokensPath)) { Console.WriteLine($"no tokens.txt at {tokensPath}"); return 2; }
        var (idToSym, symToId) = LoadTokens(tokensPath);

        long Id(string sym)
        {
            if (!symToId.TryGetValue(sym, out var id))
                throw new InvalidOperationException($"symbol '{sym}' is not in {tokensPath}");
            return id;
        }

        // espeak's own symbols, looked up rather than hard-coded, so a different model's table still works.
        long flap = Id("ɾ");        // alveolar tap, what espeak writes for flapped t/d
        long t = Id("t"), d = Id("d");
        long barredI = Id("ᵻ"), smallCapI = Id("ɪ");
        long turnedA = Id("ɐ"), schwa = Id("ə");
        long rSchwa = Id("ɚ"), turnedR = Id("ɹ");
        long primary = Id("ˈ"), secondary = Id("ˌ"), length = Id("ː");

        // Every vowel in the espeak inventory this model uses. Needed to move a stress mark, because
        // stress attaches to the vowel that follows it.
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
        };

        // One variant at a time, for probing the rig itself. The control always renders: without it there
        // is nothing to read the other row against.
        var only = Environment.GetEnvironmentVariable("SENSITIVITY_ONLY");
        if (!string.IsNullOrWhiteSpace(only))
            variants = variants.Where(v => v.Name == "control" || v.Name == only).ToList();

        Directory.CreateDirectory(outDir);
        var (reference, referenceRate, _) = readWav(File.ReadAllBytes(promptWav));

        Console.WriteLine($"text     : {fixture.Text}");
        Console.WriteLine($"prompt   : {Path.GetFileName(promptWav)} {referenceRate} Hz");
        Console.WriteLine($"truth    : {Render(fixture.Tokens, idToSym)}");
        Console.WriteLine($"outDir   : {outDir}");
        Console.WriteLine();

        bool int8 = Environment.GetEnvironmentVariable("ZIPVOICE_INT8") == "1";
        var graphDir = int8
            ? modelDir.Replace("zipvoice-distill-zh-en", "zipvoice-distill-int8-zh-en")
            : modelDir;
        var config = new ZipVoiceConfig();
        using var graphs = new OrtZipVoiceGraphs(graphDir, int8);

        // The reference clip ends mid-breath, so without trailing silence the model finishes the prompt's
        // last word at the start of the line it generates - which lands in the transcript as inserted words
        // and puts a floor under the control's WER. ZIPVOICE_TAIL_PAD exposes it so the floor can be
        // measured against the pad rather than assumed away.
        float tailPad = 0.25f;
        if (Environment.GetEnvironmentVariable("ZIPVOICE_NO_PAD") == "1") tailPad = 0f;
        if (float.TryParse(Environment.GetEnvironmentVariable("ZIPVOICE_TAIL_PAD"), out var padOverride))
            tailPad = padOverride;

        // One pipeline for every variant: pinned noise, one reference clip, one set of graphs, so the
        // token sequence is the only thing that differs between rows.
        using var pipeline = new ZipVoicePipeline(graphs, config)
        {
            NoiseSeed = 1234,
            ReferenceTailSilenceSeconds = tailPad,
        };
        Console.WriteLine($"tailPad  : {tailPad}s");

        // Synthesis is deterministic here (pinned seed, pinned inputs), so re-rendering audio that already
        // exists buys nothing when the thing being iterated on is the grader.
        bool reuse = Environment.GetEnvironmentVariable("SENSITIVITY_REUSE_WAVS") == "1";

        var rendered = new List<(Variant V, long[] Tokens, string Wav)>();
        foreach (var variant in variants)
        {
            var tokens = variant.Apply(fixture.Tokens);
            int edits = TokenEdits(fixture.Tokens, tokens);
            var wav = Path.Combine(outDir, $"{variant.Name}.wav");
            if (reuse && File.Exists(wav))
            {
                Console.WriteLine($"{variant.Name,-20} {edits,3} token edits, reused  {Path.GetFileName(wav)}");
                Console.WriteLine($"{"",-20} {Render(tokens, idToSym)}");
                rendered.Add((variant, tokens, wav));
                continue;
            }
            var result = await pipeline.SynthesizeAsync(tokens, fixture.PromptTokens, reference, referenceRate);
            File.WriteAllBytes(wav, writeWav(result.Audio, result.SampleRate));

            Console.WriteLine($"{variant.Name,-20} {edits,3} token edits, {result.DurationSeconds,5:F2}s  {Path.GetFileName(wav)}");
            Console.WriteLine($"{"",-20} {Render(tokens, idToSym)}");
            rendered.Add((variant, tokens, wav));
        }

        if (Environment.GetEnvironmentVariable("SENSITIVITY_NO_GRADE") == "1")
        {
            Console.WriteLine();
            Console.WriteLine("grading skipped (SENSITIVITY_NO_GRADE=1). Listen to the wavs in " + outDir);
            return 0;
        }

        // ---- Grade the audio -----------------------------------------------------------------------
        // Whisper is the grader because the question is what a LISTENER recovers from the audio, which is
        // the only definition of "the model rendered it correctly" that does not beg the question. It is
        // loaded once and reused, so the transcripts differ only by the audio handed to it.
        var whisperDir = Environment.GetEnvironmentVariable("WHISPER_MODEL_DIR")
            ?? Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..",
                            "SpawnDev.ILGPU.ML.Demo", "wwwroot", "models", "whisper-tiny");
        whisperDir = Path.GetFullPath(whisperDir);
        if (!Directory.Exists(whisperDir))
        {
            Console.WriteLine($"no whisper model at {whisperDir} - set WHISPER_MODEL_DIR, or "
                            + "grade by listening to the wavs above.");
            return 0;
        }

        Console.WriteLine();
        Console.WriteLine($"grader   : whisper at {whisperDir}");
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
        // Without the tokenizer the pipeline reports raw token ids ("[407] [385] ...") rather than words,
        // which scores as garbage against any reference text and would condemn perfectly good audio.
        var tokenizerJson = Path.Combine(whisperDir, "tokenizer.json");
        if (!File.Exists(tokenizerJson)) { Console.WriteLine($"no tokenizer.json at {tokenizerJson}"); return 2; }
        stt.LoadTokenizer(File.ReadAllText(tokenizerJson));

        Console.WriteLine();
        Console.WriteLine($"{"variant",-20} {"edits",5} {"WER",7} {"infix",7}  transcript");
        Console.WriteLine(new string('-', 110));
        var truthWords = Words(fixture.Text);
        var rows = new List<(string Name, int Edits, double Wer, string Text, string Why)>();
        foreach (var (variant, tokens, wav) in rendered)
        {
            var (samples, rate, _) = readWav(File.ReadAllBytes(wav));
            var transcript = (await stt.TranscribeAsync(samples, rate)).Text?.Trim() ?? "";
            var hypWords = Words(transcript);
            double plain = Wer(truthWords, hypWords);
            double wer = InfixWer(truthWords, hypWords);
            int edits = TokenEdits(fixture.Tokens, tokens);
            Console.WriteLine($"{variant.Name,-20} {edits,5} {plain,6:P0} {wer,6:P0}  \"{transcript}\"");
            rows.Add((variant.Name, edits, wer, transcript, variant.Why));
        }

        // ---- Read the result out loud ---------------------------------------------------------------
        var control = rows.FirstOrDefault(r => r.Name == "control");
        Console.WriteLine();
        if (control.Name == null || control.Wer > 0.15)
        {
            Console.WriteLine($"RESULT   : INVALID - the control transcribed at {control.Wer:P0} infix WER. The rig, the "
                            + "grader, or the fixture is what is being measured here, not the perturbations.");
            return 1;
        }
        Console.WriteLine($"control transcribed at {control.Wer:P0} infix WER, so the rig is sound and the rows below are real.");
        var damaging = rows.Where(r => r.Name != "control" && r.Wer > control.Wer + 0.10).ToList();
        if (damaging.Count == 0)
            Console.WriteLine("NO perturbation damaged the transcript. The model is tolerant of every error class a "
                            + "CMUdict frontend will make; spend the effort on coverage and normalization instead.");
        else
        {
            Console.WriteLine("These error classes DAMAGE the output and the frontend must get them right:");
            foreach (var r in damaging.OrderByDescending(r => r.Wer))
                Console.WriteLine($"  {r.Name,-20} {r.Wer,6:P0}  {r.Why}");
        }
        Console.WriteLine();
        Console.WriteLine("WER is a coarse grader: it cannot hear an accent, only a wrong word. Listen to the wavs "
                        + "in " + outDir + " before treating a 0% row as proof of no damage.");
        return 0;
    }

    // Move every primary stress mark from the vowel it marks to the NEXT vowel, which is what a
    // dictionary-driven frontend does when it picks the wrong syllable.
    private static long[] MoveStressLater(long[] seq, long primary, HashSet<long> vowels)
    {
        var outSeq = new List<long>(seq.Length);
        for (int i = 0; i < seq.Length; i++)
        {
            if (seq[i] != primary) { outSeq.Add(seq[i]); continue; }
            // Skip the vowel this mark currently attaches to, then insert before the following one.
            int j = i + 1;
            while (j < seq.Length && !vowels.Contains(seq[j])) { outSeq.Add(seq[j]); j++; }
            if (j < seq.Length) { outSeq.Add(seq[j]); j++; }          // the originally stressed vowel
            while (j < seq.Length && !vowels.Contains(seq[j])) { outSeq.Add(seq[j]); j++; }
            if (j < seq.Length) { outSeq.Add(primary); outSeq.Add(seq[j]); } // stress the next one
            i = j;
        }
        return outSeq.ToArray();
    }

    private static (Dictionary<long, string> IdToSym, Dictionary<string, long> SymToId) LoadTokens(string path)
    {
        var idToSym = new Dictionary<long, string>();
        var symToId = new Dictionary<string, long>();
        foreach (var raw in File.ReadAllLines(path))
        {
            var line = raw.TrimEnd('\r', '\n');
            if (line.Length == 0) continue;
            // The symbol itself can be a space or a tab-adjacent character, so split on the LAST tab.
            int cut = line.LastIndexOf('\t');
            if (cut < 0 || !long.TryParse(line[(cut + 1)..], out var id)) continue;
            var sym = line[..cut];
            idToSym[id] = sym;
            symToId.TryAdd(sym, id);
        }
        return (idToSym, symToId);
    }

    private static string Render(long[] tokens, Dictionary<long, string> idToSym)
        => string.Concat(tokens.Select(t => idToSym.TryGetValue(t, out var s) ? s : "?"));

    private static string[] Words(string text)
        => text.ToLowerInvariant()
               .Select(c => char.IsLetterOrDigit(c) || c == '\'' ? c : ' ').ToArray()
               .Aggregate(new System.Text.StringBuilder(), (sb, c) => sb.Append(c)).ToString()
               .Split(' ', StringSplitOptions.RemoveEmptyEntries);

    private static double Wer(string[] truth, string[] hyp)
        => truth.Length == 0 ? (hyp.Length == 0 ? 0 : 1) : Edits(truth, hyp) / (double)truth.Length;

    /// <summary>
    /// WER that does not charge for words before or after the sentence we asked for.
    /// </summary>
    /// <remarks>
    /// ZipVoice regenerates the reference clip's own speech in front of the line it is asked to speak, and
    /// the cut at the prompt boundary does not always land cleanly, so the transcript begins with a few
    /// words of the reference. That is NOT our pipeline: sherpa-onnx, the independent implementation, does
    /// it too, and worse - graded here it opens with "others call me mother nature" before reaching the
    /// sentence. Charging for those words puts a floor under every row and buries the effect being
    /// measured. Free skips at the head and tail of the hypothesis remove the artifact while still charging
    /// full price for every substitution, deletion and insertion INSIDE the sentence, which is where a
    /// mispronunciation shows up. The plain WER is reported alongside it so nothing is hidden.
    /// </remarks>
    private static double InfixWer(string[] truth, string[] hyp)
    {
        if (truth.Length == 0) return hyp.Length == 0 ? 0 : 1;
        // Row 0 all zeros: starting anywhere in the hypothesis is free.
        var prev = new int[hyp.Length + 1];
        var cur = new int[hyp.Length + 1];
        for (int i = 1; i <= truth.Length; i++)
        {
            cur[0] = i;
            for (int j = 1; j <= hyp.Length; j++)
                cur[j] = Math.Min(Math.Min(prev[j] + 1, cur[j - 1] + 1),
                                  prev[j - 1] + (truth[i - 1] == hyp[j - 1] ? 0 : 1));
            (prev, cur) = (cur, prev);
        }
        // Ending anywhere in the hypothesis is free.
        return prev.Min() / (double)truth.Length;
    }

    private static int TokenEdits(long[] a, long[] b)
        => Edits(a.Select(x => x.ToString()).ToArray(), b.Select(x => x.ToString()).ToArray());

    // Plain Levenshtein. Small inputs, so the full table is fine and is easier to trust than a banded one.
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
