// Learns letter-to-sound from CMUdict, so the phonemizer can pronounce words no dictionary contains.
//
//   dotnet run --project tools/lts-train -c Release -- [--dict path] [--out model.txt] [--holdout 5000]
//                                                     [--windows 3,2,1,0] [--min-count 1] [--analyze]
//
// WHY THIS EXISTS: CMUdict covered every word in 120 Harvard sentences, but it does not contain
// "Aubriella" - and that is the name that has to work. Names, coinages and brands are the real
// out-of-vocabulary case, and no dictionary will ever have them all.
//
// WHY IT IS TRAINED RATHER THAN HAND-WRITTEN: English spelling rules are famously long, and a
// hand-written set encodes one person's recollection of them. CMUdict is 126,052 words of evidence about
// how English is actually pronounced, it is BSD-2-Clause, and anything derived from it is ours to ship
// with its notice attached. So the rules are learned from the data and then MEASURED on words held out of
// training, which a hand-written set can never honestly claim.
//
// THE DESIGN, AND THE ONES THAT LOST. All measured on the same held-out 5,000 words:
//
//   stress digits baked into the sound emissions ............. 39.5% of words exactly right
//   a word-level stress model (word ending + syllable count).. 37.2%
//   sounds and stress as SEPARATE models over letter context.. the number printed at the end
//
// The middle one was the intuitive design - stress genuinely is a property of the word, not of a letter -
// and it lost anyway. Letter context carries more of the signal than a word ending does. Keeping stress
// out of the SOUND emissions is worth it regardless: it lifted stress-blind accuracy from 50.7% to 52.1%
// by making that model less sparse.
//
// WHAT --analyze MEASURED, AND WHAT IT OVERTURNED (2026-08-28). Two beliefs this tool used to encode:
//
//   "a context seen once is noise, drop it"  -  WRONG, and it was costing 2.8 points. Per-letter accuracy
//   by the width that answered says context WIDTH dominates evidence: the THINNEST wide context (support
//   2) is right 83.3% of the time, while the BEST-SUPPORTED narrow one (support 100+) manages 77.6%.
//   Dropping a thin wide rule does not fall back to something better, it falls back to something worse.
//   Pruning is therefore off by default. --min-count is kept so the claim stays falsifiable.
//
//   "the model in the file is the model that was measured"  -  it was NOT. Accuracy was computed on the
//   unpruned tables and then a pruned file was written, so the header described a model nobody ever ran.
//   Measurement now happens on exactly the rules that get written, and a check below fails the build of
//   the model if the two ever disagree again.
//
// The same run says where the remaining loss is: 88% of substitutions are on VOWELS (4,049 against 536 on
// consonants), and the top confusions are overwhelmingly the same vowel with the wrong stress digit
// (EH0/EH1, IH0/IH1, AH1/AH0) or a reduction (IH0/AH0). 26.6% of wrong words are wrong by a SINGLE phone.
// That is a model class that is sound and needs sharpening, not replacing - hence wider context rather
// than a different kind of model.
//
// The method is deliberately plain - alignment by Viterbi EM, then context models with backoff - so the
// output is inspectable text rather than weights nobody can read, and it runs anywhere with no
// dependency. It is not the state of the art; it is the honest baseline a better model has to beat.
using System.Text;

var dictPath = @"D:\users\tj\Projects\_ref\cmudict\cmudict.dict";
var outPath = Path.Combine(AppContext.BaseDirectory, "lts-model.txt");
int holdout = 5000, iterations = 6, minCount = 1;

// The widths the shipped model is built from. Chosen by measuring, not by taste - each width was trained
// and scored on the same held-out 5,000 words, and this is where the curve flattens:
//
//   +/-2   43.7%   191 KB gzipped        +/-5   49.8%   482 KB     <- shipped
//   +/-3   47.1%   354 KB                +/-6   50.1%   500 KB     <- +0.3 for another 20 KB
//   +/-4   48.8%   437 KB
//
// Going wider is what a letter-to-sound model of this class has left to give: the errors are single
// phones in nearly-right words, not a wrong idea about the word.
int[] windows = [5, 4, 3, 2, 1, 0];
bool analyze = false;
for (int i = 0; i < args.Length; i++)
{
    if (args[i] == "--dict" && i + 1 < args.Length) dictPath = args[++i];
    if (args[i] == "--out" && i + 1 < args.Length) outPath = args[++i];
    if (args[i] == "--holdout" && i + 1 < args.Length) holdout = int.Parse(args[++i]);
    if (args[i] == "--iterations" && i + 1 < args.Length) iterations = int.Parse(args[++i]);
    if (args[i] == "--min-count" && i + 1 < args.Length) minCount = int.Parse(args[++i]);
    if (args[i] == "--windows" && i + 1 < args.Length)
        windows = args[++i].Split(',').Select(int.Parse).OrderByDescending(w => w).ToArray();
    if (args[i] == "--analyze") analyze = true;
}
if (!File.Exists(dictPath)) { Console.WriteLine($"no cmudict at {dictPath}"); return 2; }
if (windows.Length == 0 || windows[^1] != 0)
{
    // The narrowest width is the backstop that guarantees every letter has some answer at all.
    Console.WriteLine("--windows must include 0");
    return 2;
}

// ---- Load --------------------------------------------------------------------------------------------
// Only plain alphabetic headwords: the dictionary also holds entries with punctuation in them, which
// teach a letter-to-sound model nothing about letters.
var entries = new List<(string Word, string[] Phones)>();
foreach (var line in File.ReadLines(dictPath))
{
    var body = line.Split('#')[0].Trim();
    if (body.Length == 0) continue;
    var parts = body.Split(' ', StringSplitOptions.RemoveEmptyEntries);
    if (parts.Length < 2 || parts[0].EndsWith(')')) continue;
    var word = parts[0].ToLowerInvariant();
    if (!word.All(c => c is >= 'a' and <= 'z')) continue;
    if (word.Length > 20 || parts.Length - 1 > word.Length * 2) continue;
    entries.Add((word, parts[1..]));
}
Console.WriteLine($"entries  : {entries.Count} alphabetic headwords");

// Held out BEFORE training touches anything. Fixed seed so the number is reproducible.
var rng = new Random(20260827);
var shuffled = entries.OrderBy(_ => rng.Next()).ToList();
var test = shuffled.Take(holdout).ToList();
var train = shuffled.Skip(holdout).ToList();
Console.WriteLine($"split    : {train.Count} train, {test.Count} held out");
Console.WriteLine($"widths   : {string.Join(", ", windows.Select(w => "+/-" + w))}, "
                + $"keeping contexts seen {minCount}+ times");

// Sounds are learned WITHOUT stress digits: stress has its own model, and mixing them made this one
// needlessly sparse.
var soundTrain = train.Select(e => (e.Word, Phones: e.Phones.Select(Bare).ToArray())).ToList();

// ---- Align -------------------------------------------------------------------------------------------
// Each letter emits zero, one or two phones. Viterbi EM: align with the current estimates, recount,
// repeat. Uniform initialisation suffices because length does most of the work - a six-letter word with
// five phones has few plausible alignments.
var counts = new Dictionary<(char Letter, string Emission), double>();
var letterTotals = new Dictionary<char, double>();

foreach (var (word, phones) in soundTrain)
    foreach (var letter in word)
        foreach (var phone in phones)
            Bump(letter, phone, 1.0 / (word.Length * phones.Length));
foreach (var letter in "abcdefghijklmnopqrstuvwxyz") Bump(letter, "", 0.05);

for (int iteration = 1; iteration <= iterations; iteration++)
{
    var next = new Dictionary<(char, string), double>();
    double aligned = 0;
    foreach (var (word, phones) in soundTrain)
    {
        var path = Align(word, phones);
        if (path == null) continue;
        aligned++;
        for (int i = 0; i < word.Length; i++)
        {
            var key = (word[i], path[i]);
            next[key] = next.GetValueOrDefault(key) + 1;
        }
    }
    counts = next;
    letterTotals = counts.GroupBy(k => k.Key.Item1).ToDictionary(g => g.Key, g => g.Sum(x => x.Value));
    Console.WriteLine($"align {iteration}  : {aligned:N0} words aligned, {counts.Count} letter-emission pairs");
}

// ---- Context models ----------------------------------------------------------------------------------
// What a letter emits given its neighbours, at several widths, tried widest first. That backoff is why
// "gh" can be silent in "night" and voiced in "ghost" without anyone writing either rule.
//
// Every width lives in ONE keyspace, because a key states its own width by its shape: +/-1 writes
// "c[a]t" and +/-2 writes "^c[a]t$", which can never collide. The runtime therefore needs no schema and
// no version field - it reads the widths out of the model file it was handed.
var sound = windows.ToDictionary(w => w, _ => new Dictionary<string, Dictionary<string, int>>());
var stress = windows.ToDictionary(w => w, _ => new Dictionary<string, Dictionary<string, int>>());

foreach (var (word, phones) in train)
{
    var barePhones = phones.Select(Bare).ToArray();
    var path = Align(word, barePhones);
    if (path == null) continue;

    int consumed = 0;
    for (int i = 0; i < word.Length; i++)
    {
        foreach (var w in windows) Record(sound[w], Key(word, i, w), path[i]);

        // The alignment consumes phones in order, so walking it beside the ORIGINAL phones recovers
        // which stress digit belongs to which letter.
        int count = path[i].Length == 0 ? 0 : path[i].Split(' ').Length;
        if (count > 0)
        {
            var digits = string.Concat(phones.Skip(consumed).Take(count).Select(StressDigit));
            if (digits.Any(char.IsDigit))
                foreach (var w in windows) Record(stress[w], Key(word, i, w), digits);
        }
        consumed += count;
    }
}
Console.WriteLine($"sounds   : {string.Join(", ", windows.Select(w => $"{sound[w].Count} at +/-{w}"))}");
Console.WriteLine($"stress   : {string.Join(", ", windows.Select(w => $"{stress[w].Count} at +/-{w}"))}");

// ---- Prune, BEFORE measuring ---------------------------------------------------------------------------
// Off by default, because measuring it said it COSTS 2.8 points - see the header. It stays available so
// that claim can be re-tested rather than believed. The narrowest width is never pruned: it is the
// backstop that guarantees every letter has some answer.
if (minCount > 1)
{
    foreach (var w in windows.Where(w => w != 0)) { Prune(sound[w], minCount); Prune(stress[w], minCount); }
    Console.WriteLine($"pruned   : {string.Join(", ", windows.Select(w => $"{sound[w].Count} at +/-{w}"))} sound contexts survive");
}

// ---- Flatten to the rules that actually ship -------------------------------------------------------------
// Done BEFORE measuring, deliberately. Only rules that CHANGE something are written: a wide rule whose
// answer is what the narrower widths would have said anyway is redundant, because the runtime reaches the
// same phone by backing off. That is what pays for the wider context - +/-5 learns 2.5 million rules and
// ships 136 thousand of them.
//
// It is not a neutral transformation, and that is worth stating plainly. The stress tie-break keeps the
// primary whose rule was most SPECIFIC, so dropping a redundant +/-5 rule changes which width answers and
// can change the word. Measuring the counted tables and shipping the compressed ones would therefore
// report an accuracy the library does not deliver - the exact defect found in this file's own header
// today. So the compressed rules ARE the model from here on, and everything below scores them.
//
// The surviving width is also the better signal for that tie-break: a stress rule that survives at +/-5
// is one the narrower contexts DISAGREED with, which is precisely what "this stress is context-determined"
// means.
var flatSounds = Compress(sound);
var flatStress = Compress(stress);
Console.WriteLine($"rules    : {flatSounds.Count} sound + {flatStress.Count} stress kept of "
                + $"{windows.Sum(w => sound[w].Count + stress[w].Count)} learned");

// ---- Measure on words training never saw ---------------------------------------------------------------
int exact = 0, phoneErrors = 0, phoneTotal = 0, exactIgnoringStress = 0, stressOnly = 0;
foreach (var (word, phones) in test)
{
    var predicted = Predict(word);
    var truth = phones.ToArray();
    phoneTotal += truth.Length;
    phoneErrors += Distance(predicted, truth);
    bool right = predicted.SequenceEqual(truth);
    if (right) exact++;

    if (predicted.Select(Bare).SequenceEqual(truth.Select(Bare)))
    {
        exactIgnoringStress++;
        if (!right) stressOnly++;
    }
}

Console.WriteLine();
Console.WriteLine($"HELD OUT : {exact}/{test.Count} words exactly right ({exact / (double)test.Count:P1})");
Console.WriteLine($"           phoneme error rate {phoneErrors / (double)phoneTotal:P1}");
Console.WriteLine($"           {exactIgnoringStress / (double)test.Count:P1} exactly right if stress is IGNORED");
Console.WriteLine($"           {stressOnly / (double)test.Count:P1} had every sound right and the WRONG STRESS");

// ---- Where the errors actually are (--analyze) ----------------------------------------------------------
// The next model class should be chosen from evidence about how THIS one fails, not from a list of things
// wrong with it in principle. Three questions, each with a decision hanging on it:
//   1. How far wrong is a wrong word? Words that miss by one phone say the model class is sound and needs
//      sharpening; words that miss by four say it is not.
//   2. Does a wide context that fired on thin evidence beat the narrow one it displaced? Lookup takes the
//      widest context that MATCHES, never the best-supported one. If thin contexts do worse than the width
//      below them, hard backoff is itself the defect. (Measured: they do BETTER. See the header.)
//   3. What does it confuse - and are the errors vowels or consonants? They fail for different reasons.
if (analyze)
{
    var sizes = new int[7];
    var confusion = new Dictionary<string, int>();
    var byContext = new Dictionary<(int Width, string Support), (int Right, int Total)>();
    int vowelSubs = 0, consonantSubs = 0, tooMany = 0, tooFew = 0;

    foreach (var (word, phones) in test)
    {
        var truth = phones.ToArray();
        var predicted = Predict(word);
        sizes[Math.Min(Distance(predicted, truth), 6)]++;

        foreach (var (got, want) in EditPairs(predicted, truth))
        {
            var label = $"{got ?? "."} -> {want ?? "."}";
            confusion[label] = confusion.GetValueOrDefault(label) + 1;
            if (want is null) tooMany++;
            else if (got is null) tooFew++;
            else if (IsVowel(want)) vowelSubs++;
            else consonantSubs++;
        }

        // Position level: what the lookup emitted, against what the alignment says that letter should emit.
        var truthPath = Align(word, truth.Select(Bare).ToArray());
        if (truthPath == null) continue;
        for (int i = 0; i < word.Length; i++)
        {
            var (emission, width, support) = Traced(word, i);
            var bucket = support switch
            {
                0 => "none", <= 2 => "2", <= 5 => "3-5", <= 20 => "6-20", <= 100 => "21-100", _ => "100+",
            };
            var cur = byContext.GetValueOrDefault((width, bucket));
            byContext[(width, bucket)] = (cur.Right + (emission == truthPath[i] ? 1 : 0), cur.Total + 1);
        }
    }

    Console.WriteLine();
    Console.WriteLine("ANALYSIS : how far wrong a wrong word is, in phones");
    for (int d = 0; d <= 6; d++)
        Console.WriteLine($"           {(d == 6 ? "6+" : d.ToString()),-3} {sizes[d],6}  {sizes[d] / (double)test.Count,7:P1}");

    Console.WriteLine();
    Console.WriteLine("           per-LETTER accuracy, by which context width answered and its evidence");
    Console.WriteLine("           width   support    letters    right");
    foreach (var kv in byContext.OrderByDescending(k => k.Key.Width).ThenBy(k => k.Key.Support, StringComparer.Ordinal))
        Console.WriteLine($"           {(kv.Key.Width < 0 ? "none" : "+/-" + kv.Key.Width),-7} {kv.Key.Support,-9} "
                        + $"{kv.Value.Total,8}  {kv.Value.Right / (double)kv.Value.Total,7:P1}");

    // ---- The stress tie-break, on the words it can actually change ----------------------------------
    // The model marks more than one primary on a minority of words; on every other word the policy is
    // irrelevant. Averaged over all 5,000 the difference reads as a rounding error either way, which is
    // not an answer - so score the two policies against each other on exactly the words where they
    // disagree, and report how often each is RIGHT there.
    int contested = 0, differ = 0, firstRight = 0, widestRight = 0, bothWrong = 0;
    foreach (var (word, phones) in test)
    {
        var truth = phones.ToArray();
        var raw = Raw(word, i => LookupFlat(flatSounds, word, i).Emission,
                            i => LookupFlat(flatStress, word, i));
        if (raw.Output.Count(p => p.EndsWith('1')) < 2) continue;
        contested++;

        var byFirst = new List<string>(raw.Output);
        var byWidest = new List<string>(raw.Output);
        SettleStress(byFirst, raw.Evidence, keepFirstPrimary: true);
        SettleStress(byWidest, raw.Evidence, keepFirstPrimary: false);
        if (byFirst.SequenceEqual(byWidest)) continue;

        differ++;
        bool f = byFirst.SequenceEqual(truth), w = byWidest.SequenceEqual(truth);
        if (f) firstRight++;
        if (w) widestRight++;
        if (!f && !w) bothWrong++;
    }

    Console.WriteLine();
    Console.WriteLine($"           stress tie-break: {contested} held-out words had more than one primary "
                    + $"marked, and the two policies disagree on {differ} of them");
    if (differ > 0)
        Console.WriteLine($"           of those {differ}: earliest-syllable right {firstRight}, "
                        + $"most-specific-rule right {widestRight}, neither {bothWrong}");

    Console.WriteLine();
    Console.WriteLine($"           substitutions: {vowelSubs} on vowels, {consonantSubs} on consonants; "
                    + $"{tooMany} phones too many, {tooFew} too few");
    Console.WriteLine("           most common (predicted -> truth):");
    foreach (var kv in confusion.OrderByDescending(k => k.Value).Take(24))
        Console.WriteLine($"           {kv.Value,6}  {kv.Key}");
}

// ---- Does decomposing beat guessing? -------------------------------------------------------------------
// Held-out words that are a known stem plus an ending should be DERIVED, never guessed. This measures
// that claim rather than assuming it: the dictionary handed to the decomposer contains only TRAINING
// words, so a held-out word is as unknown to it as it would be at runtime.
var trainDictionary = SpawnDev.Phonemizer.PronunciationDictionary.Parse(
    train.Select(e => e.Word + " " + string.Join(' ', e.Phones)));

int fired = 0, firedRight = 0, guessedRightOnSame = 0, combined = 0;
foreach (var (word, phones) in test)
{
    var truth = phones.ToArray();
    if (SpawnDev.Phonemizer.WordDecomposer.TryDecompose(word, trainDictionary, out var derived))
    {
        fired++;
        if (derived.SequenceEqual(truth)) firedRight++;
        if (Predict(word).SequenceEqual(truth)) guessedRightOnSame++;
        if (derived.SequenceEqual(truth)) combined++;
    }
    else if (Predict(word).SequenceEqual(truth)) combined++;
}

Console.WriteLine();
Console.WriteLine($"DECOMPOSE: fires on {fired}/{test.Count} held-out words ({fired / (double)test.Count:P1})");
if (fired > 0)
{
    Console.WriteLine($"           right {firedRight}/{fired} of those ({firedRight / (double)fired:P1})");
    Console.WriteLine($"           letter-to-sound alone was right {guessedRightOnSame}/{fired} of the same words "
                    + $"({guessedRightOnSame / (double)fired:P1})");
}
Console.WriteLine($"           decompose-then-guess overall: {combined / (double)test.Count:P1} against "
                + $"{exact / (double)test.Count:P1} for guessing alone");

// ---- Write the model -----------------------------------------------------------------------------------
// Plain text, so it can be read, diffed and corrected by hand. The rules were flattened above, before
// anything was measured, so what is written here is exactly what was scored.
//
// How much the compression is worth knowing about: it is reported rather than assumed away. The counted
// tables and the shipped rules give different answers on some words, because dropping a redundant wide
// rule changes which width answers and the stress tie-break reads that. Neither is "correct" a priori -
// what matters is that the number printed above describes the file written below, and it does.
int differs = test.Count(t => !Predict(t.Word).SequenceEqual(PredictCounted(t.Word)));
Console.WriteLine($"           ({differs} of {test.Count} held-out words are answered differently by the "
                + "uncompressed tables; the figures above are the SHIPPED rules)");

var sb = new StringBuilder();
sb.AppendLine("# Letter-to-sound model, learned from CMUdict (BSD-2-Clause) by tools/lts-train.");
sb.AppendLine("# Sound rules:  context TAB phones    (an emission of - means the letter is silent)");
sb.AppendLine("# Stress rules: *context TAB digits   (one digit per phone that context emits)");
sb.AppendLine("# A context states its own width by its shape: [a] is the letter alone, c[a]t is one letter");
sb.AppendLine("# either side, ^c[a]t$ is two. Lookup takes the WIDEST context present, so the widths in");
sb.AppendLine("# this file need no declaring and can change without a code change.");
sb.AppendLine("# Only rules that differ from what a narrower context would answer are kept.");
sb.AppendLine($"# Held out: {exact / (double)test.Count:P1} of words exactly right, "
            + $"{phoneErrors / (double)phoneTotal:P1} phoneme error rate, on {test.Count} words never trained on.");

foreach (var rule in flatSounds.OrderBy(kv => kv.Key.Length).ThenBy(kv => kv.Key, StringComparer.Ordinal))
    sb.Append(rule.Key).Append('\t').AppendLine(rule.Value.Length == 0 ? "-" : rule.Value);
foreach (var rule in flatStress.OrderBy(kv => kv.Key.Length).ThenBy(kv => kv.Key, StringComparer.Ordinal))
    sb.Append('*').Append(rule.Key).Append('\t').AppendLine(rule.Value);

File.WriteAllText(outPath, sb.ToString());
Console.WriteLine();
Console.WriteLine($"wrote    : {outPath} ({new FileInfo(outPath).Length / 1024} KB, "
                + $"{flatSounds.Count} sound + {flatStress.Count} stress rules, "
                + $"{flatSounds.Count + flatStress.Count} of "
                + $"{windows.Sum(w => sound[w].Count + stress[w].Count)} kept)");
return 0;

// --------------------------------------------------------------------------------------------------------

void Bump(char letter, string emission, double weight)
{
    counts[(letter, emission)] = counts.GetValueOrDefault((letter, emission)) + weight;
    letterTotals[letter] = letterTotals.GetValueOrDefault(letter) + weight;
}

double Score(char letter, string emission)
{
    var c = counts.GetValueOrDefault((letter, emission));
    var t = letterTotals.GetValueOrDefault(letter, 1);
    return Math.Log((c + 0.01) / (t + 1));
}

// Best assignment of phones to letters: each letter takes 0, 1 or 2 phones, in order.
string[]? Align(string word, string[] phones)
{
    int L = word.Length, P = phones.Length;
    if (P > L * 2) return null;
    var best = new double[L + 1, P + 1];
    var back = new int[L + 1, P + 1];
    for (int i = 0; i <= L; i++) for (int j = 0; j <= P; j++) best[i, j] = double.NegativeInfinity;
    best[0, 0] = 0;

    for (int i = 0; i < L; i++)
        for (int j = 0; j <= P; j++)
        {
            if (double.IsNegativeInfinity(best[i, j])) continue;
            Try(i + 1, j, best[i, j] + Score(word[i], ""), 0);
            if (j < P) Try(i + 1, j + 1, best[i, j] + Score(word[i], phones[j]), 1);
            if (j + 1 < P) Try(i + 1, j + 2, best[i, j] + Score(word[i], phones[j] + " " + phones[j + 1]), 2);

            void Try(int ni, int nj, double score, int move)
            {
                if (score <= best[ni, nj]) return;
                best[ni, nj] = score; back[ni, nj] = move;
            }
        }

    if (double.IsNegativeInfinity(best[L, P])) return null;
    var path = new string[L];
    for (int i = L, j = P; i > 0; i--)
    {
        int move = back[i, j];
        path[i - 1] = move switch
        {
            0 => "",
            1 => phones[j - 1],
            _ => phones[j - 2] + " " + phones[j - 1],
        };
        j -= move;
    }
    return path;
}

// THE model: the flat rules that get written, read exactly the way the runtime reads them. Everything is
// scored through this, so training cannot measure one model and ship another.
string[] Predict(string word) => Assemble(word,
    i => LookupFlat(flatSounds, word, i).Emission,
    i => LookupFlat(flatStress, word, i));

// The uncompressed tables, kept only to report what the compression costs or gains.
string[] PredictCounted(string word) => Assemble(word,
    i => Lookup(sound, word, i).Emission,
    i => Lookup(stress, word, i));

(string Emission, int Width) Lookup(Dictionary<int, Dictionary<string, Dictionary<string, int>>> tables,
                                    string word, int i)
{
    foreach (var w in windows)
        if (tables[w].TryGetValue(Key(word, i, w), out var inner)) return (Best(inner), w);
    return ("", -1);
}

(string Emission, int Width) LookupFlat(Dictionary<string, string> rules, string word, int i)
{
    foreach (var w in windows)
        if (rules.TryGetValue(Key(word, i, w), out var emission)) return (emission, w);
    return ("", -1);
}

// The shared assembly step, so the two predictors can never drift apart. Mirrors LetterToSound.Predict,
// including the stress tie-break - if these two disagree the accuracy printed above is not the accuracy
// the library delivers.
string[] Assemble(string word, Func<int, string> soundAt, Func<int, (string Digits, int Width)> stressAt,
                  bool keepFirstPrimary = true)
{
    var (output, evidence) = Raw(word, soundAt, stressAt);
    SettleStress(output, evidence, keepFirstPrimary);

    // Every English word has a stressed syllable. If the model marked none, stress the first, which is
    // where English puts it more often than anywhere else.
    if (output.Any(IsVowel) && !output.Any(x => x.EndsWith('1')))
    {
        int first = output.FindIndex(IsVowel);
        output[first] = Bare(output[first]) + "1";
    }
    return output.ToArray();
}

// The phones and the specificity of the stress rule behind each, BEFORE any tie-break is applied - so the
// tie-break policy can be measured on the words it can actually change rather than averaged over the
// 96% it cannot touch.
(List<string> Output, List<int> Evidence) Raw(string word, Func<int, string> soundAt,
                                              Func<int, (string Digits, int Width)> stressAt)
{
    var output = new List<string>();
    var evidence = new List<int>();
    for (int i = 0; i < word.Length; i++)
    {
        var emission = soundAt(i);
        if (emission.Length == 0) continue;

        var phones = emission.Split(' ');
        var (digits, width) = stressAt(i);
        for (int k = 0; k < phones.Length; k++)
        {
            output.Add(IsVowel(phones[k])
                ? phones[k] + (k < digits.Length && char.IsDigit(digits[k]) ? digits[k] : '0')
                : phones[k]);
            evidence.Add(width);
        }
    }
    return (output, evidence);
}

// Exactly ONE primary stress per word. The per-letter model can mark several, which is not English - and
// stress is the thing the downstream model punishes hardest, so emitting two is worse than picking the
// wrong one.
//
// WHICH one to keep is a real decision, and the two candidates are measured against each other in
// --analyze on the words where they differ:
//   keepFirstPrimary  - the earliest syllable, on the reasoning that English favours early stress.
//   otherwise         - the one whose context rule was most SPECIFIC. Position is not evidence, and
//                       keeping the first is what stressed "aubriella" on its first syllable while this
//                       same model gets "briella" right.
// Ties fall to the earlier syllable either way.
static void SettleStress(List<string> output, List<int> evidence, bool keepFirstPrimary)
{
    int keep = -1;
    for (int i = 0; i < output.Count; i++)
    {
        if (!output[i].EndsWith('1')) continue;
        if (keep < 0) { keep = i; if (keepFirstPrimary) break; continue; }
        if (!keepFirstPrimary && evidence[i] > evidence[keep]) keep = i;
    }
    for (int i = 0; i < output.Count; i++)
        if (i != keep && output[i].EndsWith('1')) output[i] = Bare(output[i]) + "0";
}

// Flatten the per-width tables into the one keyspace the model file uses, keeping a rule only when it says
// something the narrower widths would not have said. A key states its own width by its shape, so the next
// narrower key is this one with a character trimmed from each end - no re-derivation from the word needed.
// Widths are walked NARROWEST FIRST, so a wide rule is compared against the rules that will really be
// there to catch it.
Dictionary<string, string> Compress(Dictionary<int, Dictionary<string, Dictionary<string, int>>> tables)
{
    var flat = new Dictionary<string, string>(StringComparer.Ordinal);
    foreach (var w in windows.OrderBy(w => w))
        foreach (var (key, inner) in tables[w])
        {
            var emission = Best(inner);
            if (w > 0 && Backoff(key) == emission) continue;
            flat[key] = emission;
        }
    return flat;

    // What the widths below this key would answer, walking down until one of them has a rule. Null when
    // none does, which never equals an emission and so keeps the rule.
    string? Backoff(string key)
    {
        for (var k = key[1..^1]; ; k = k[1..^1])
        {
            if (flat.TryGetValue(k, out var answer)) return answer;
            if (k.Length <= 3) return null;
        }
    }
}

static string Best(Dictionary<string, int> inner) => inner.OrderByDescending(v => v.Value).First().Key;

// Drop every context whose best emission rests on too little evidence. Measured to COST accuracy - see the
// header - and off by default, but kept so the claim can be re-tested.
static void Prune(Dictionary<string, Dictionary<string, int>> table, int minCount)
{
    foreach (var key in table.Where(kv => kv.Value.Values.Max() < minCount).Select(kv => kv.Key).ToList())
        table.Remove(key);
}

// The same lookup Predict does, but reporting WHICH width answered and on how much evidence - so the
// backoff itself can be measured rather than assumed correct.
(string Emission, int Width, int Support) Traced(string word, int i)
{
    foreach (var w in windows)
        if (sound[w].TryGetValue(Key(word, i, w), out var inner))
        {
            var best = inner.OrderByDescending(v => v.Value).First();
            return (best.Key, w, best.Value);
        }
    return ("", -1, 0);
}

// A vowel phone, in ARPAbet, is one of these fifteen. Everything else is a consonant.
static bool IsVowel(string phone)
{
    var b = Bare(phone);
    return b is "AA" or "AE" or "AH" or "AO" or "AW" or "AY" or "EH" or "ER"
             or "EY" or "IH" or "IY" or "OW" or "OY" or "UH" or "UW";
}

static string Bare(string phone) => phone.Length > 0 && char.IsDigit(phone[^1]) ? phone[..^1] : phone;

static string StressDigit(string phone) => phone.Length > 0 && char.IsDigit(phone[^1]) ? phone[^1].ToString() : "-";

// The context key. The runtime builds this identically, or nothing ever matches.
static string Key(string word, int i, int window)
{
    var sb = new StringBuilder(window * 2 + 3);
    for (int k = i - window; k <= i + window; k++)
    {
        if (k == i) sb.Append('[');
        sb.Append(k < 0 ? '^' : k >= word.Length ? '$' : word[k]);
        if (k == i) sb.Append(']');
    }
    return sb.ToString();
}

static void Record(Dictionary<string, Dictionary<string, int>> table, string key, string emission)
{
    if (!table.TryGetValue(key, out var inner)) table[key] = inner = new Dictionary<string, int>();
    inner[emission] = inner.GetValueOrDefault(emission) + 1;
}

static int Distance(string[] a, string[] b)
{
    var prev = new int[b.Length + 1];
    var cur = new int[b.Length + 1];
    for (int j = 0; j <= b.Length; j++) prev[j] = j;
    for (int i = 1; i <= a.Length; i++)
    {
        cur[0] = i;
        for (int j = 1; j <= b.Length; j++)
            cur[j] = Math.Min(Math.Min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
        (prev, cur) = (cur, prev);
    }
    return prev[b.Length];
}

// The edit script behind the distance, so an error can be NAMED rather than only counted. A null on the
// predicted side is a phone the model failed to produce; a null on the truth side is one it invented.
static List<(string? Predicted, string? Truth)> EditPairs(string[] a, string[] b)
{
    var d = new int[a.Length + 1, b.Length + 1];
    for (int i = 0; i <= a.Length; i++) d[i, 0] = i;
    for (int j = 0; j <= b.Length; j++) d[0, j] = j;
    for (int i = 1; i <= a.Length; i++)
        for (int j = 1; j <= b.Length; j++)
            d[i, j] = Math.Min(Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1),
                               d[i - 1, j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));

    var pairs = new List<(string?, string?)>();
    int x = a.Length, y = b.Length;
    while (x > 0 || y > 0)
    {
        if (x > 0 && y > 0 && d[x, y] == d[x - 1, y - 1] + (a[x - 1] == b[y - 1] ? 0 : 1))
        {
            if (a[x - 1] != b[y - 1]) pairs.Add((a[x - 1], b[y - 1]));
            x--; y--;
        }
        else if (x > 0 && d[x, y] == d[x - 1, y] + 1) pairs.Add((a[--x], null));
        else pairs.Add((null, b[--y]));
    }
    return pairs;
}
