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
bool analyze = false, analogyProbe = false, truncationProbe = false;
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
    if (args[i] == "--analogy") analogyProbe = true;
    if (args[i] == "--truncation") truncationProbe = true;
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

// ---- A vowel letter must never be silent as a LAST RESORT ------------------------------------------------
// The single-letter rules are the backstop, and for "e" the majority answer is SILENT - correctly, since
// word-final silent e is everywhere in English. But that backstop only answers when NO context rule matched
// at any width, which means the model has never seen this spelling and has no idea. Answering "no sound at
// all" there is the worst available guess: a run of unmatched letters emits nothing and the word comes back
// with fewer syllables than it has vowels. "nevaeh" became "N AH1 V" - not a mispronunciation, a different
// and shorter word - because a, e and h each fell to a silent backstop in a row.
//
// So each vowel letter also carries its best NON-SILENT single-letter emission, used only when the backstop
// itself is what answered. Six extra rules; the file does not otherwise change, because every context rule
// that legitimately says "silent" still says it.
var vowelFallback = new Dictionary<char, string>();
foreach (var vowel in "aeiouy")
{
    if (!sound[0].TryGetValue($"[{vowel}]", out var inner)) continue;
    var loudest = inner.Where(kv => kv.Key.Length > 0).OrderByDescending(kv => kv.Value).ToList();
    if (loudest.Count > 0) vowelFallback[vowel] = loudest[0].Key;
}
Console.WriteLine($"fallback : {string.Join(", ", vowelFallback.Select(kv => $"{kv.Key}->{kv.Value}"))}");
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

    // ---- Catastrophic failures, which word-accuracy cannot see --------------------------------------
    // "49.8% exactly right" treats a word that is one stress digit off and a word that lost half its
    // syllables as the same failure. They are not: "nevaeh" comes back as "N AH1 V", which is not a
    // mispronunciation but a different, shorter word. Those are the failures a listener notices, so they
    // get counted separately.
    int truncated = 0, voiceless = 0;
    foreach (var (word, phones) in test)
    {
        var predicted = Predict(word);
        if (phones.Length >= 4 && predicted.Length * 2 <= phones.Length) truncated++;
        if (phones.Any(IsVowel) && !predicted.Any(IsVowel)) voiceless++;
    }
    Console.WriteLine();
    Console.WriteLine($"           CATASTROPHIC: {truncated} words lost half their phones or more, "
                    + $"{voiceless} came back with no vowel at all");

    // ---- Can the truncation be repaired without costing more than it saves? ANSWER: NO -------------
    //
    // "nevaeh" comes back "N AH1 V" and "huawei" "HH W AO1": a run of letters at the END produced
    // nothing at all. Four triggers were measured before and only the no-vowel guard paid - but every
    // one of them paired with a WHOLE-WORD repair, which re-spells letters that were already right.
    // Scope looked like the untried axis, so this sweeps the trigger (how long a trailing silent run)
    // AGAINST the repair scope (the whole word, or only the silent tail).
    //
    // ⛔ MEASURED 2026-08-29 AND REJECTED. Scope makes NO DIFFERENCE - tail-only scores identically to
    // whole-word at every run length (49.7% vs 49.7%, 49.8% vs 49.8%). The reasoning that a narrower
    // repair would avoid collateral damage was simply wrong, and the sweep says so in one line.
    //
    // And the prize is smaller than it looks: the baseline truncates only 2 words in 5,000 held out.
    // Every policy trades 0.1-0.2 points of exact accuracy to turn that 2 into a 1. Do not ship any of
    // them, and do not re-propose repair scope. The kept sweep is here so the claim stays falsifiable.
    //
    // What DOES answer this class of word is PronunciationDictionary.Define: a name you know should be
    // told, not guessed. Truncation bites hardest on proper nouns, which is exactly where a definition
    // is available and a rule never will be.
    if (truncationProbe)
    {
        Console.WriteLine();
        Console.WriteLine("  TRUNCATION REPAIR SWEEP - trailing silent run, whole-word vs tail-only repair");
        Console.WriteLine($"  {"policy",-44} {"exact",7} {"trunc",7} {"novowel",8}");

        foreach (var (label, minRun, tailOnly, needVowel) in new (string, int, bool, bool)[]
        {
            ("baseline (no-vowel guard only)", 99, false, false),
            ("run>=2, repair WHOLE word", 2, false, true),
            ("run>=2, repair TAIL only", 2, true, true),
            ("run>=3, repair WHOLE word", 3, false, true),
            ("run>=3, repair TAIL only", 3, true, true),
            ("run>=2, TAIL only, any letters", 2, true, false),
            ("run>=1, repair TAIL only", 1, true, true),
        })
        {
            int right = 0, trunc = 0, noVowel = 0;
            foreach (var (word, phones) in test)
            {
                var got = PredictTail(word, minRun, tailOnly, needVowel);
                if (got.SequenceEqual(phones)) right++;
                if (phones.Length >= 4 && got.Length * 2 <= phones.Length) trunc++;
                if (phones.Any(IsVowel) && !got.Any(IsVowel)) noVowel++;
            }
            Console.WriteLine($"  {label,-44} {right / (double)test.Count,7:P1} {trunc,7} {noVowel,8}");
        }
        Console.WriteLine("  A policy only ships if it cuts truncation WITHOUT losing exact-match accuracy.");
    }

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

// ---- Would pronunciation by ANALOGY help? (--analogy) ---------------------------------------------------
// The idea: an unknown word that RHYMES with a dictionary word should borrow that word's ending outright
// rather than guess it letter by letter. "aubriella" is not in CMUdict, but "gabriella" is, and they share
// seven letters - so the ending, INCLUDING WHERE THE STRESS FALLS, is already known and observed rather
// than predicted.
//
// This probe measures the ceiling BEFORE any of it is built into the library. It answers three things the
// design depends on and none of which I want to assume:
//   1. COVERAGE - how often does a held-out word even share a long-enough ending with a training word?
//   2. Is analogy RIGHT when it fires, and is it right more often than guessing the same word?
//   3. Which suffix length and how much support - long and rare, or short and well-attested?
//
// The suffix table is built from TRAINING words only, so a held-out word is as unknown to it as it would
// be at runtime. Phones carry their stress digits, because borrowing the stress is the whole point.
if (analogyProbe)
{
    const int minLen = 3, maxLen = 8;
    var analogy = new Dictionary<string, Dictionary<string, int>>(StringComparer.Ordinal);
    foreach (var (word, phones) in train)
    {
        var path = Align(word, phones.Select(Bare).ToArray());
        if (path == null) continue;

        var per = new int[word.Length];
        for (int i = 0; i < word.Length; i++) per[i] = path[i].Length == 0 ? 0 : path[i].Split(' ').Length;
        if (per.Sum() != phones.Length) continue;      // the alignment must account for every phone

        for (int k = minLen; k <= maxLen && word.Length > k; k++)
        {
            int consumed = per.Take(word.Length - k).Sum();
            if (consumed >= phones.Length) continue;   // the ending carries no phones at all
            Record(analogy, word[^k..], string.Join(' ', phones.Skip(consumed)));
        }
    }
    Console.WriteLine();
    Console.WriteLine($"ANALOGY  : {analogy.Count} distinct endings of {minLen}-{maxLen} letters, from training words only");

    // Firing on nearly every word is the failure mode to watch for: a 3-letter ending matches SOMETHING
    // almost always, which is not a rhyme, and it overrides a correct guess as often as it fixes a wrong
    // one. The question is therefore not "does analogy work" but "how long and how well-attested does the
    // shared ending have to be before borrowing it beats guessing".
    Console.WriteLine("  shortest  support  fires on         right when it fires   guessing on those   OVERALL");
    (double Best, int Len, int Sup) best = (exact / (double)test.Count, 0, 0);
    foreach (var shortest in new[] { 4, 5, 6, 7, 8 })
        foreach (var support in new[] { 1, 2, 4, 8 })
        {
            int hits = 0, right = 0, guessRightOnSame = 0, combinedRight = 0;
            foreach (var (word, phones) in test)
            {
                var truth = phones.ToArray();
                var byAnalogy = PredictByAnalogy(word, analogy, support, shortest, maxLen);
                if (byAnalogy != null)
                {
                    hits++;
                    if (byAnalogy.SequenceEqual(truth)) { right++; combinedRight++; }
                    if (Predict(word).SequenceEqual(truth)) guessRightOnSame++;
                }
                else if (Predict(word).SequenceEqual(truth)) combinedRight++;
            }
            double overall = combinedRight / (double)test.Count;
            if (overall > best.Best) best = (overall, shortest, support);
            Console.WriteLine($"  {shortest,-9} {support,-8} {hits,5} ({hits / (double)test.Count,5:P1})   "
                            + $"{right,5} ({right / (double)Math.Max(hits, 1),5:P1})        "
                            + $"{guessRightOnSame,5} ({guessRightOnSame / (double)Math.Max(hits, 1),5:P1})     "
                            + $"{overall:P1}{(overall > exact / (double)test.Count ? "  <-- beats guessing" : "")}");
        }
    // A CHEAPER VARIANT worth measuring before any of this ships. Borrowing whole phones replaces work the
    // letter-to-sound model already does well - its errors are single phones in nearly-right words. What it
    // is genuinely bad at is STRESS (7.5% of words have every sound right and the wrong stress, and stress
    // is what the downstream model punishes hardest). So: keep the guessed phones, and borrow only the
    // stress DIGITS from the rhyming word. Applied only when the borrowed ending has the same number of
    // phones the guess produced for those letters, since otherwise the digits do not line up.
    Console.WriteLine();
    Console.WriteLine("  STRESS-ONLY borrowing (keep the guessed phones, take only the ending's stress):");
    foreach (var shortest in new[] { 4, 5, 6 })
        foreach (var support in new[] { 1, 2, 4 })
        {
            int hits = 0, combinedRight = 0;
            foreach (var (word, phones) in test)
            {
                var truth = phones.ToArray();
                var restressed = RestressByAnalogy(word, analogy, support, shortest, maxLen);
                if (restressed != null) hits++;
                if ((restressed ?? Predict(word)).SequenceEqual(truth)) combinedRight++;
            }
            double overall = combinedRight / (double)test.Count;
            Console.WriteLine($"  {shortest,-9} {support,-8} {hits,5} ({hits / (double)test.Count,5:P1}) "
                            + $"restressed                                {overall:P1}"
                            + (overall > exact / (double)test.Count ? "  <-- beats guessing" : ""));
        }

    Console.WriteLine();
    Console.WriteLine($"  guessing alone: {exact / (double)test.Count:P1}. Best analogy setting: "
                    + (best.Len == 0 ? "NONE - analogy does not beat guessing at any setting tried."
                                     : $"shortest {best.Len}, support {best.Sup} -> {best.Best:P1}"));

    // The word this whole component exists for, and its dictionary family.
    Console.WriteLine();
    // The spot check runs at the setting the sweep chose, not at the loosest one - a name it mangles at
    // "any 3-letter ending" says nothing about the configuration that would actually ship.
    Console.WriteLine($"  (at the best setting: shortest {Math.Max(best.Len, minLen)}, support {Math.Max(best.Sup, 2)})");
    foreach (var probe in new[] { "aubriella", "briella", "nevaeh", "kayleigh", "makayla", "elowen", "jaxon",
                                  "anthropic", "blazor", "ryleigh", "tuvok" })
    {
        var byAnalogy = PredictByAnalogy(probe, analogy, Math.Max(best.Sup, 2), Math.Max(best.Len, minLen), maxLen);
        Console.WriteLine($"  {probe,-12} analogy: {(byAnalogy == null ? "(no ending matched)" : string.Join(' ', byAnalogy)),-34}"
                        + $"  guessing: {string.Join(' ', Predict(probe))}");
    }
}

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
sb.AppendLine("# Last-resort rules: !context TAB phones - used ONLY when the single-letter backstop answered");
sb.AppendLine("# and said the letter is silent. A vowel that no context rule recognises must still make a");
sb.AppendLine("# sound, or a run of unknown letters vanishes and the word loses syllables.");
sb.AppendLine($"# Held out: {exact / (double)test.Count:P1} of words exactly right, "
            + $"{phoneErrors / (double)phoneTotal:P1} phoneme error rate, on {test.Count} words never trained on.");

foreach (var rule in flatSounds.OrderBy(kv => kv.Key.Length).ThenBy(kv => kv.Key, StringComparer.Ordinal))
    sb.Append(rule.Key).Append('\t').AppendLine(rule.Value.Length == 0 ? "-" : rule.Value);
foreach (var rule in flatStress.OrderBy(kv => kv.Key.Length).ThenBy(kv => kv.Key, StringComparer.Ordinal))
    sb.Append('*').Append(rule.Key).Append('\t').AppendLine(rule.Value);
foreach (var rule in vowelFallback.OrderBy(kv => kv.Key))
    sb.Append('!').Append('[').Append(rule.Key).Append(']').Append('\t').AppendLine(rule.Value);

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
string[] Predict(string word)
{
    var spoken = Assemble(word, i => SoundAt(word, i, repair: false),
                                i => LookupFlat(flatStress, word, i));

    // A last-resort repair for a word that came back UNSAYABLE: spelled with vowels, pronounced with none.
    // That is not a mispronunciation, it is the failure that once turned "Aubriella" into the consonants
    // "bɹl", and it is unambiguous - no English word is all consonants.
    //
    // FOUR triggers were measured on the same held-out 5,000 words, and only this one pays:
    //
    //   trigger                             exactly right   words with no vowel at all
    //   none (leave it broken)                    49.8%            11
    //   whenever the backstop answered            30.3%             6
    //   fewer vowels than vowel groups            46.7%             1
    //   three-plus silent letters in a row        49.5%            11
    //   NO VOWEL AT ALL (this one)                49.9%             1
    //
    // The obvious one - fire whenever the single-letter backstop answered - is the WORST, and the reason
    // matters: under the redundancy compression, resolving at width 0 is the NORMAL case, not a sign the
    // model has no idea, because wide rules that agreed with it were dropped precisely BECAUSE they agreed.
    // The compressed file cannot tell "no evidence" from "the evidence agreed", so the trigger has to be
    // the OUTCOME rather than how the answer was reached. Counting vowel GROUPS looks principled and
    // over-fires: English really does spell more of them than it says ("business" has u, i, e against two
    // vowel sounds), so it cost 3.1 points to fix ten words.
    if (!spoken.Any(IsVowel) && word.Any(c => "aeiouy".Contains(c)))
        spoken = Assemble(word, i => SoundAt(word, i, repair: true),
                                i => LookupFlat(flatStress, word, i));
    return spoken;
}

// A candidate truncation repair: the shipped no-vowel guard first, then - if the word ENDS in a run of
// letters that produced nothing - the loudest-sound fallback applied either to the whole word or only to
// that trailing run. Only vowel letters have a fallback, so a run of silent consonants cannot be repaired
// and needVowel says whether to bother trying.
string[] PredictTail(string word, int minRun, bool tailOnly, bool needVowel)
{
    var spoken = Assemble(word, i => SoundAt(word, i, repair: false),
                                i => LookupFlat(flatStress, word, i));

    if (!spoken.Any(IsVowel) && word.Any(c => "aeiouy".Contains(c)))
        return Assemble(word, i => SoundAt(word, i, repair: true),
                              i => LookupFlat(flatStress, word, i));

    int run = 0;
    for (int i = word.Length - 1; i >= 0; i--)
    {
        if (LookupFlat(flatSounds, word, i).Emission.Length != 0) break;
        run++;
    }
    if (run < minRun || run == 0) return spoken;

    int start = word.Length - run;
    if (needVowel && !word[start..].Any(c => "aeiouy".Contains(c))) return spoken;

    return Assemble(word, i => SoundAt(word, i, repair: !tailOnly || i >= start),
                          i => LookupFlat(flatStress, word, i));
}

// The sound for one letter. In repair mode a vowel letter that came back silent is given its best
// non-silent single-letter emission instead.
string SoundAt(string word, int i, bool repair)
{
    var emission = LookupFlat(flatSounds, word, i).Emission;
    if (repair && emission.Length == 0 && vowelFallback.TryGetValue(word[i], out var loud)) return loud;
    return emission;
}

// Pronunciation by analogy: take the LONGEST ending this word shares with a dictionary word and use that
// word's phones for it, spelling out only what comes before by context rules. Null when no ending matches,
// which is the caller's signal to fall back to guessing the whole word.
//
// Stress is the reason this is worth doing, and it needs a rule. A borrowed ending carries stress OBSERVED
// in a real word, which is far better evidence than a per-letter model - so when the ending brings a
// primary, any primary the prefix predicted is demoted to SECONDARY rather than dropped. That is what the
// dictionary itself does with this shape: gabriella is G AA2 B R IY0 EH1 L AA2, isabella IH2 Z AH0 B EH1 L
// AH0 - secondary on the opening syllable, primary on the borrowed ending. When the ending brings no
// primary ("napster" -> S T ER0), the prefix keeps its own.
string[]? PredictByAnalogy(string word, Dictionary<string, Dictionary<string, int>> analogy,
                           int minSupport, int minLen, int maxLen)
{
    for (int k = Math.Min(maxLen, word.Length - 1); k >= minLen; k--)
    {
        if (!analogy.TryGetValue(word[^k..], out var inner)) continue;
        var best = inner.OrderByDescending(v => v.Value).First();
        if (best.Value < minSupport) continue;

        // The prefix is spelled out in the context of the WHOLE word - the letters of the ending are still
        // its right-hand context, which is what makes "aubr" in "aubriella" behave like "gabr" in
        // "gabriella" rather than like a word ending in "aubr".
        var output = new List<string>();
        for (int i = 0; i < word.Length - k; i++)
        {
            var emission = LookupFlat(flatSounds, word, i).Emission;
            if (emission.Length == 0) continue;
            var digits = LookupFlat(flatStress, word, i).Emission;
            var phones = emission.Split(' ');
            for (int p = 0; p < phones.Length; p++)
                output.Add(IsVowel(phones[p])
                    ? phones[p] + (p < digits.Length && char.IsDigit(digits[p]) ? digits[p] : '0')
                    : phones[p]);
        }

        var borrowed = best.Key.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        bool borrowedCarriesPrimary = borrowed.Any(p => p.EndsWith('1'));
        if (borrowedCarriesPrimary)
            for (int i = 0; i < output.Count; i++)
                if (output[i].EndsWith('1')) output[i] = Bare(output[i]) + "2";

        output.AddRange(borrowed);

        // Exactly one primary, same contract as everywhere else.
        bool seen = false;
        for (int i = 0; i < output.Count; i++)
        {
            if (!output[i].EndsWith('1')) continue;
            if (!seen) { seen = true; continue; }
            output[i] = Bare(output[i]) + "0";
        }
        if (output.Any(IsVowel) && !output.Any(x => x.EndsWith('1')))
        {
            int first = output.FindIndex(IsVowel);
            output[first] = Bare(output[first]) + "1";
        }
        return output.ToArray();
    }
    return null;
}

// Keep the guessed phones; take only the STRESS DIGITS from the rhyming word. Null when no ending matches
// or when the borrowed ending covers a different number of phones than the guess produced for those same
// letters - in that case the digits cannot be lined up, and inventing an alignment would be worse than
// leaving the guess alone.
string[]? RestressByAnalogy(string word, Dictionary<string, Dictionary<string, int>> analogy,
                            int minSupport, int minLen, int maxLen)
{
    for (int k = Math.Min(maxLen, word.Length - 1); k >= minLen; k--)
    {
        if (!analogy.TryGetValue(word[^k..], out var inner)) continue;
        var pick = inner.OrderByDescending(v => v.Value).First();
        if (pick.Value < minSupport) continue;

        var guess = Predict(word);
        var borrowed = pick.Key.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        // How many phones did the guess produce for the ending's letters?
        int prefixPhones = 0;
        for (int i = 0; i < word.Length - k; i++)
        {
            var emission = LookupFlat(flatSounds, word, i).Emission;
            if (emission.Length > 0) prefixPhones += emission.Split(' ').Length;
        }
        if (guess.Length - prefixPhones != borrowed.Length) continue;

        var output = guess.ToArray();
        for (int i = 0; i < borrowed.Length; i++)
        {
            int at = prefixPhones + i;
            if (!IsVowel(output[at]) || !IsVowel(borrowed[i])) continue;
            output[at] = Bare(output[at]) + (char.IsDigit(borrowed[i][^1]) ? borrowed[i][^1] : '0');
        }

        // The borrowed ending decides where the primary is when it carries one.
        if (borrowed.Any(p => p.EndsWith('1')))
            for (int i = 0; i < prefixPhones; i++)
                if (output[i].EndsWith('1')) output[i] = Bare(output[i]) + "2";

        bool seen = false;
        for (int i = 0; i < output.Length; i++)
        {
            if (!output[i].EndsWith('1')) continue;
            if (!seen) { seen = true; continue; }
            output[i] = Bare(output[i]) + "0";
        }
        if (output.Any(IsVowel) && !output.Any(x => x.EndsWith('1')))
        {
            int first = Array.FindIndex(output, IsVowel);
            output[first] = Bare(output[first]) + "1";
        }
        return output;
    }
    return null;
}

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
