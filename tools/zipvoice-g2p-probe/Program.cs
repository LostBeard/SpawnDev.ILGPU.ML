// Measures the REAL gap between our MIT phonemizer and the token ids the model was trained on.
//
//   dotnet run --project tools/zipvoice-g2p-probe -c Release -- [fixtureDir] [--dict path] [--no-flap] [--no-destress]
//
// WHY THIS EXISTS: the sensitivity experiment damages the reference tokens in ways I PREDICTED a
// dictionary frontend would get wrong. A prediction can be incomplete, and an error class nobody thought
// of is exactly what surfaces late and expensively. This closes the loop: run the REAL phonemizer, align
// its output against the ids espeak actually produced, and CLASSIFY every difference. Anything reported
// as UNTESTED-* has no perturbation measuring it, and is therefore an unmeasured risk.
//
// It also doubles as the phonemizer's own gate. Every rule added to SpawnDev.Phonemizer should move the
// total difference count, and the switches below make each rule's contribution separately measurable.
//
// CMUdict is BSD-2-Clause and is read from disk, not vendored, until packaging decides its shipped form.
using SpawnDev.Phonemizer;
using System.Text.Json;

var fixtureDir = args.Length > 0 && !args[0].StartsWith("--")
    ? args[0]
    : Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..",
                                    "zipvoice-harness", "fixtures", "phase1"));
var dictPath = @"D:\users\tj\Projects\_ref\cmudict\cmudict.dict";
var modelDir = Environment.GetEnvironmentVariable("ZIPVOICE_MODEL_DIR")
    ?? @"D:\users\tj\Projects\SpawnDev.Reachy\SpawnDev.Reachy\models\sherpa-onnx-zipvoice-distill-zh-en-emilia";
bool flapping = true, destress = true;
string? wordList = null;
for (int i = 0; i < args.Length; i++)
{
    if (args[i] == "--dict" && i + 1 < args.Length) dictPath = args[++i];
    if (args[i] == "--no-flap") flapping = false;
    if (args[i] == "--no-destress") destress = false;
    if (args[i] == "--words" && i + 1 < args.Length) wordList = args[++i];
}

if (!File.Exists(dictPath)) { Console.WriteLine($"no cmudict at {dictPath} (pass --dict)"); return 2; }

// ---- Pronounce words given on the command line (--words "aubriella tuvok ...") ------------------------
// WHY THIS MODE EXISTS: the headline number below is STRUCTURALLY BLIND to letter-to-sound. Every one of
// the 940 words in the fixtures is in CMUdict, so the guessing path never runs, and the probe would
// report an unchanged 4.0% whether that model were improved, broken, or deleted outright. An unmeasured
// component with a confident-looking number over it is the worst of both.
//
// So this mode takes words directly and says WHICH of the three paths answered. "The dictionary had it",
// "it was derived from a word the dictionary had" and "we guessed it from the spelling" deserve very
// different amounts of trust, and a bare string of phonemes hides which one you got.
if (wordList != null)
{
    var dict = PronunciationDictionary.Load(dictPath);
    var lts = EmbeddedData.LoadLetterToSound();
    var speaker = new EnglishPhonemizer(dict)
    {
        LetterToSound = lts,
        Flapping = flapping,
        DestressFunctionWords = destress,
    };
    Console.WriteLine($"cmudict  : {dict.Count} headwords");
    Console.WriteLine($"lts      : {lts.Count} context rules (the EMBEDDED model, so this tests what ships)");
    Console.WriteLine();
    Console.WriteLine($"{"word",-16} {"answered by",-16} {"ARPAbet",-40} IPA");
    Console.WriteLine(new string('-', 100));
    foreach (var word in wordList.Split(' ', StringSplitOptions.RemoveEmptyEntries))
    {
        string source, arpabet;
        if (dict.TryLookup(word, out var known))
        {
            source = "dictionary";
            arpabet = string.Join(' ', known);
        }
        else if (WordDecomposer.TryDecompose(word, dict, out var derived))
        {
            source = "decomposed";
            arpabet = string.Join(' ', derived);
        }
        else
        {
            source = "letter-to-sound";
            arpabet = string.Join(' ', lts.Predict(word));
        }
        Console.WriteLine($"{word,-16} {source,-16} {arpabet,-40} {speaker.ToIpa(word)}");
    }
    return 0;
}
if (!Directory.Exists(fixtureDir)) { Console.WriteLine($"no fixtures at {fixtureDir}"); return 2; }

// ---- The model's symbol table -----------------------------------------------------------------------
// The library owns tokens.txt parsing now, including the detail that a symbol can be whitespace.
var vocabulary = PhonemeVocabulary.Load(Path.Combine(modelDir, "tokens.txt"));

var dictionary = PronunciationDictionary.Load(dictPath);
var phonemizer = new EnglishPhonemizer(dictionary) { Flapping = flapping, DestressFunctionWords = destress };
Console.WriteLine($"cmudict  : {dictionary.Count} headwords");
Console.WriteLine($"rules    : flapping={flapping}, destress-function-words={destress}");

// ---- Walk the fixtures -------------------------------------------------------------------------------
var classCounts = new Dictionary<string, int>();
var classExamples = new Dictionary<string, List<string>>();
int totalOracle = 0, totalDiffs = 0, oov = 0, oovTotal = 0;

foreach (var path in Directory.GetFiles(fixtureDir, "*.json").OrderBy(p => p))
{
    using var doc = JsonDocument.Parse(File.ReadAllText(path));
    var text = doc.RootElement.GetProperty("text").GetString()!;
    var oracle = doc.RootElement.GetProperty("tokens").EnumerateArray()
        .Select(e => vocabulary.TryGetSymbol(e.GetInt64(), out var s) ? s : "?").ToArray();

    var ours = phonemizer.ToSymbols(text).ToArray();
    oovTotal += text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
    oov += phonemizer.LastUnknownWords.Count;

    // Aligned WORD BY WORD: a difference is only actionable if you know which word produced it, and a
    // single sentence-wide alignment happily pairs symbols from different words when lengths drift.
    var oracleWords = SplitWords(oracle);
    var ourWords = SplitWords(ours);
    var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
    int sentenceDiffs = 0;

    if (oracleWords.Count != ourWords.Count)
        Console.WriteLine($"    NOTE: {oracleWords.Count} reference words vs {ourWords.Count} ours - "
                        + "falling back to a whole-sentence alignment, so word labels are approximate.");

    var pairs = oracleWords.Count == ourWords.Count
        ? Enumerable.Range(0, oracleWords.Count)
                    .Select(i => (O: oracleWords[i], U: ourWords[i], W: i < words.Length ? words[i] : "?")).ToList()
        : new List<(string[] O, string[] U, string W)> { (oracle, ours, "(sentence)") };

    foreach (var (o, u, w) in pairs)
    {
        foreach (var d in Align(o, u))
        {
            sentenceDiffs++;
            var cls = Classify(d.Expected, d.Got);
            classCounts[cls] = classCounts.GetValueOrDefault(cls) + 1;
            if (!classExamples.TryGetValue(cls, out var ex)) classExamples[cls] = ex = new List<string>();
            var label = $"{Show(d.Expected)}->{Show(d.Got)} in \"{w.Trim('.', ',', ';', ':', '!', '?')}\"";
            if (!ex.Contains(label) && ex.Count < 6) ex.Add(label);
        }
    }
    totalOracle += oracle.Length;
    totalDiffs += sentenceDiffs;

    Console.WriteLine();
    Console.WriteLine($"--- {Path.GetFileNameWithoutExtension(path)}");
    Console.WriteLine($"    espeak : {string.Concat(oracle)}");
    Console.WriteLine($"    ours   : {string.Concat(ours)}");
    Console.WriteLine($"    {sentenceDiffs} differences over {oracle.Length} reference symbols "
                    + $"({sentenceDiffs / (double)oracle.Length:P0})");
}

// ---- The answer --------------------------------------------------------------------------------------
Console.WriteLine();
Console.WriteLine("=====================================================================================");
Console.WriteLine($"TOTAL: {totalDiffs} differences over {totalOracle} reference symbols "
                + $"({totalDiffs / (double)totalOracle:P1}); {oov}/{oovTotal} words not in the dictionary");
Console.WriteLine();
Console.WriteLine($"{"difference class",-40} {"count",6}  {"share",7}  examples");
Console.WriteLine(new string('-', 108));
foreach (var kv in classCounts.OrderByDescending(k => k.Value))
    Console.WriteLine($"{kv.Key,-40} {kv.Value,6}  {kv.Value / (double)Math.Max(1, totalDiffs),6:P1}  "
                    + string.Join("  ", classExamples[kv.Key].Take(3)));

Console.WriteLine();
Console.WriteLine("Classes named UNTESTED-* have no row in the sensitivity table and are unmeasured. Each one");
Console.WriteLine("needs a perturbation adding before the phonemizer's accuracy can be trusted.");

// ---- Can the model SPEAK everything this frontend emits? ---------------------------------------------
// Accuracy is about picking the right symbol. This is the cruder question underneath it: does a token
// exist for every symbol at all? A symbol with no token cannot be rendered - it is dropped or the render
// is refused - so a frontend change that starts emitting a new character breaks speech outright, and it
// would not move any number above. Cheap to check, and worth checking on every run.
Console.WriteLine();
var emitted = new SortedDictionary<string, string>(StringComparer.Ordinal);
var coverageSentences = 0;
foreach (var path in Directory.GetFiles(fixtureDir, "*.json").OrderBy(p => p))
{
    using var doc = JsonDocument.Parse(File.ReadAllText(path));
    var text = doc.RootElement.GetProperty("text").GetString()!;
    coverageSentences++;
    foreach (var symbol in phonemizer.ToSymbols(text))
        if (!vocabulary.TryGetId(symbol, out _)) emitted.TryAdd(symbol, text);
}

// The fixtures are read-aloud sentences and exercise a narrow inventory, so a broad sample of real
// dictionary words is swept too - they reach far more of the phone set than any sentence list does.
var coverageRng = new Random(1234);
var headwords = File.ReadLines(dictPath)
    .Where(l => l.Length > 0 && l[0] != ';')
    .Select(l => l.Split(' ', 2)[0])
    .Where(w => !w.Contains('('))
    .ToArray();
for (var i = 0; i < 8000 && headwords.Length > 0; i++)
{
    var word = headwords[coverageRng.Next(headwords.Length)];
    foreach (var symbol in phonemizer.ToSymbols(word))
        if (!vocabulary.TryGetId(symbol, out _)) emitted.TryAdd(symbol, word);
}

if (emitted.Count == 0)
{
    Console.WriteLine($"COVERAGE: every symbol emitted over {coverageSentences} sentences and 8,000 "
                    + "dictionary words has a token in this model's vocabulary.");
}
else
{
    Console.WriteLine($"COVERAGE FAILED: {emitted.Count} symbol(s) the model has no token for - these cannot "
                    + "be spoken at all:");
    foreach (var (symbol, source) in emitted)
        Console.WriteLine($"  '{symbol}' (U+{(int)symbol[0]:X4}) first seen in: {source}");
    return 1;
}

return 0;

// ------------------------------------------------------------------------------------------------------

// Classify one difference into a bucket named for the DECISION it forces on the frontend, and for what
// the sensitivity experiment already established about it.
static string Classify(string expected, string got)
{
    if (expected == "ɾ" && (got == "t" || got == "d")) return "flap (MEASURED: harmless)";
    if (expected == "ᵻ" && got == "ɪ") return "reduced barred-i vs small-i (MEASURED: harmless)";
    if (expected == "ᵻ" && got == "ə") return "reduced barred-i vs schwa (MEASURED: harmless)";
    if (expected == "ɐ" && (got == "ə" || got == "ʌ")) return "reduced turned-a (MEASURED: harmless)";
    if (expected == "ɚ" || got == "ɚ") return "r-coloured vowel (MEASURED: harmless on words)";
    if (expected == "ː" || got == "ː") return "length mark (MEASURED: moves the sound)";
    if (expected.Length == 0 && got is "ˈ" or "ˌ") return "stress ADDED (MEASURED: 18.2%, WORST REAL CLASS)";
    if (got.Length == 0 && expected is "ˈ" or "ˌ") return "stress DROPPED (MEASURED: damaging)";
    if (expected is "ˈ" or "ˌ" && got is "ˈ" or "ˌ") return "stress primary/secondary swapped (MEASURED: damaging)";
    if (expected is "ˈ" or "ˌ" || got is "ˈ" or "ˌ") return "stress vs a phoneme (MEASURED: damaging)";
    if (expected == "ʔ" || got == "ʔ") return "glottal stop (MEASURED: n=3, inconclusive)";
    if (expected == "̩" || got == "̩") return "syllabic consonant (MEASURED with the glottal stop)";
    if (expected == "a" && got == "ə") return "the article a (MEASURED: harmless)";
    if (expected == " " || got == " ") return "UNTESTED-word-boundary";
    return $"UNTESTED-other ({Show(expected)} vs {Show(got)})";
}

static string Show(string s) => s.Length == 0 ? "∅" : s == " " ? "␣" : s;

// Levenshtein alignment, so a difference is reported against the symbol it corresponds to rather than
// against whatever happens to sit at the same index.
static List<(string Expected, string Got)> Align(string[] a, string[] b)
{
    int n = a.Length, m = b.Length;
    var d = new int[n + 1, m + 1];
    for (int i = 0; i <= n; i++) d[i, 0] = i;
    for (int j = 0; j <= m; j++) d[0, j] = j;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
            d[i, j] = Math.Min(Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1),
                               d[i - 1, j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
    var diffs = new List<(string, string)>();
    int x = n, y = m;
    while (x > 0 || y > 0)
    {
        if (x > 0 && y > 0 && a[x - 1] == b[y - 1]) { x--; y--; continue; }
        if (x > 0 && y > 0 && d[x, y] == d[x - 1, y - 1] + 1) { diffs.Add((a[x - 1], b[y - 1])); x--; y--; }
        else if (x > 0 && d[x, y] == d[x - 1, y] + 1) { diffs.Add((a[x - 1], "")); x--; }
        else { diffs.Add(("", b[y - 1])); y--; }
    }
    diffs.Reverse();
    return diffs.Select(t => (Expected: t.Item1, Got: t.Item2)).ToList();
}

// Split a symbol stream into words on the space symbol, dropping the spaces themselves.
static List<string[]> SplitWords(string[] syms)
{
    var words = new List<string[]>();
    var cur = new List<string>();
    foreach (var s in syms)
    {
        if (s == " ") { if (cur.Count > 0) { words.Add(cur.ToArray()); cur.Clear(); } }
        else cur.Add(s);
    }
    if (cur.Count > 0) words.Add(cur.ToArray());
    return words;
}
