// Measures the REAL gap between CMUdict and the token ids this model was trained on.
//
//   dotnet run --project tools/zipvoice-g2p-probe -c Release -- [fixtureDir] [--dict path] [--flap]
//
// WHY THIS EXISTS: the sensitivity experiment damages the oracle's tokens in ways I PREDICTED a CMUdict
// frontend would get wrong. That prediction could be incomplete, and an error class nobody thought of is
// exactly the kind of thing that surfaces late and expensively. This closes the loop: build the sequence
// CMUdict actually produces, align it against the ids espeak actually produced, and CLASSIFY every
// difference. Classes that show up here and are absent from the sensitivity table have never been tested.
//
// It is deliberately a PROBE and not the phonemizer. The mapping below is a first cut whose only job is
// to make the differences enumerable; freezing a design before the measurement is what this whole plan
// exists to avoid. No flapping, no reduced-vowel context rules, no homograph handling - so the classes it
// reports are an UPPER bound on what a naive mapping gets wrong, which is the number worth knowing.
//
// CMUdict is BSD-2-Clause. It is read from disk here, not vendored, until Phase 7 decides its shipped form
// and adds the required copyright notice to THIRD-PARTY-NOTICES.md.
using System.Text;
using System.Text.Json;

var fixtureDir = args.Length > 0 && !args[0].StartsWith("--")
    ? args[0]
    : Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..",
                                    "zipvoice-harness", "fixtures", "phase1"));
var dictPath = @"D:\users\tj\Projects\_ref\cmudict\cmudict.dict";
var modelDir = Environment.GetEnvironmentVariable("ZIPVOICE_MODEL_DIR")
    ?? @"D:\users\tj\Projects\SpawnDev.Reachy\SpawnDev.Reachy\models\sherpa-onnx-zipvoice-distill-zh-en-emilia";
bool applyFlap = false;
for (int i = 0; i < args.Length; i++)
{
    if (args[i] == "--dict" && i + 1 < args.Length) dictPath = args[++i];
    if (args[i] == "--flap") applyFlap = true;
}

if (!File.Exists(dictPath)) { Console.WriteLine($"no cmudict at {dictPath} (pass --dict)"); return 2; }
if (!Directory.Exists(fixtureDir)) { Console.WriteLine($"no fixtures at {fixtureDir}"); return 2; }

// ---- The model's symbol table -----------------------------------------------------------------------
var idToSym = new Dictionary<long, string>();
var symToId = new Dictionary<string, long>();
foreach (var raw in File.ReadAllLines(Path.Combine(modelDir, "tokens.txt")))
{
    var line = raw.TrimEnd('\r', '\n');
    if (line.Length == 0) continue;
    int cut = line.LastIndexOf('\t');
    if (cut < 0 || !long.TryParse(line[(cut + 1)..], out var id)) continue;
    idToSym[id] = line[..cut];
    symToId.TryAdd(line[..cut], id);
}

// ---- CMUdict ----------------------------------------------------------------------------------------
// Format: "word  P1 P2 P3", with "word(2)" for alternate pronunciations. The first listed pronunciation
// is taken; CHOOSING among alternates is Phase 5's problem and pretending otherwise here would hide it.
var dict = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);
int alternates = 0;
foreach (var line in File.ReadLines(dictPath))
{
    var body = line.Split('#')[0].Trim();
    if (body.Length == 0) continue;
    var parts = body.Split(' ', StringSplitOptions.RemoveEmptyEntries);
    if (parts.Length < 2) continue;
    var word = parts[0];
    if (word.EndsWith(')')) { alternates++; continue; }        // word(2), word(3), ...
    dict.TryAdd(word, parts[1..]);
}
Console.WriteLine($"cmudict  : {dict.Count} headwords ({alternates} alternate pronunciations skipped)");

// ---- ARPAbet to the espeak symbols this model uses ---------------------------------------------------
// Correspondences read off real espeak output for real words (about, better, water, roses, understand,
// garden, served, canoe), not from memory.
var consonants = new Dictionary<string, string[]>
{
    ["B"] = ["b"], ["CH"] = ["t", "ʃ"], ["D"] = ["d"], ["DH"] = ["ð"], ["F"] = ["f"], ["G"] = ["ɡ"],
    ["HH"] = ["h"], ["JH"] = ["d", "ʒ"], ["K"] = ["k"], ["L"] = ["l"], ["M"] = ["m"], ["N"] = ["n"],
    ["NG"] = ["ŋ"], ["P"] = ["p"], ["R"] = ["ɹ"], ["S"] = ["s"], ["SH"] = ["ʃ"], ["T"] = ["t"],
    ["TH"] = ["θ"], ["V"] = ["v"], ["W"] = ["w"], ["Y"] = ["j"], ["Z"] = ["z"], ["ZH"] = ["ʒ"],
};
// Vowels, by base symbol. Stressed and unstressed forms differ for AH and ER, which is why they are split.
string[] Vowel(string v, int stress) => (v, stress) switch
{
    ("AA", _) => ["ɑ", "ː"],
    ("AE", _) => ["æ"],
    ("AH", 0) => ["ə"],
    ("AH", _) => ["ʌ"],
    ("AO", _) => ["ɔ", "ː"],
    ("AW", _) => ["a", "ʊ"],
    ("AY", _) => ["a", "ɪ"],
    ("EH", _) => ["ɛ"],
    ("ER", 0) => ["ɚ"],
    ("ER", _) => ["ɜ", "ː"],
    ("EY", _) => ["e", "ɪ"],
    ("IH", _) => ["ɪ"],
    ("IY", 0) => ["i"],
    ("IY", _) => ["i", "ː"],
    ("OW", _) => ["o", "ʊ"],
    ("OY", _) => ["ɔ", "ɪ"],
    ("UH", _) => ["ʊ"],
    ("UW", _) => ["u", "ː"],
    _ => [],
};

// ---- Walk the fixtures -------------------------------------------------------------------------------
var classCounts = new Dictionary<string, int>();
var classExamples = new Dictionary<string, List<string>>();
int totalOracle = 0, totalOurs = 0, totalDiffs = 0, oov = 0, oovTotal = 0;

foreach (var path in Directory.GetFiles(fixtureDir, "*.json").OrderBy(p => p))
{
    using var doc = JsonDocument.Parse(File.ReadAllText(path));
    var text = doc.RootElement.GetProperty("text").GetString()!;
    var oracleIds = doc.RootElement.GetProperty("tokens").EnumerateArray().Select(e => e.GetInt64()).ToArray();
    var oracle = oracleIds.Select(id => idToSym.TryGetValue(id, out var s) ? s : "?").ToArray();

    var ours = new List<string>();
    foreach (var rawWord in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
    {
        var word = new string(rawWord.Where(c => char.IsLetter(c) || c == '\'').ToArray());
        var trailing = rawWord.Where(c => ".,;:!?".Contains(c)).ToArray();
        oovTotal++;
        if (word.Length > 0 && dict.TryGetValue(word, out var arpa))
            ours.AddRange(MapWord(arpa));
        else if (word.Length > 0) { oov++; ours.Add("«" + word + "»"); }
        foreach (var c in trailing) { ours.Add(" "); ours.Add(c.ToString()); }
        if (trailing.Length == 0) ours.Add(" ");
    }
    if (ours.Count > 0 && ours[^1] == " ") ours.RemoveAt(ours.Count - 1);

    // Aligned WORD BY WORD rather than across the whole sentence: a difference is only actionable if you
    // know which word produced it, and a single sentence-wide alignment happily pairs symbols from
    // different words when the lengths drift.
    var oracleWords = SplitWords(oracle);
    var ourWords = SplitWords(ours.ToArray());
    var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
    int sentenceDiffs = 0;
    if (oracleWords.Count != ourWords.Count)
        Console.WriteLine($"    NOTE: {oracleWords.Count} espeak words vs {ourWords.Count} ours - "
                        + "falling back to a whole-sentence alignment, so word labels are approximate.");

    var pairs = oracleWords.Count == ourWords.Count
        ? Enumerable.Range(0, oracleWords.Count).Select(i => (O: oracleWords[i], U: ourWords[i],
              W: i < words.Length ? words[i] : "?")).ToList()
        : new List<(string[] O, string[] U, string W)> { (oracle, ours.ToArray(), "(sentence)") };

    foreach (var (o, u, w) in pairs)
    {
        var wordDiffs = Align(o, u);
        sentenceDiffs += wordDiffs.Count;
        foreach (var d in wordDiffs)
        {
            var cls = Classify(d.Expected, d.Got);
            classCounts[cls] = classCounts.GetValueOrDefault(cls) + 1;
            if (!classExamples.TryGetValue(cls, out var ex)) classExamples[cls] = ex = new List<string>();
            var label = $"{Show(d.Expected)}->{Show(d.Got)} in \"{w.Trim('.', ',', ';', ':', '!', '?')}\"";
            if (!ex.Contains(label) && ex.Count < 6) ex.Add(label);
        }
    }
    totalOracle += oracle.Length; totalOurs += ours.Count; totalDiffs += sentenceDiffs;

    Console.WriteLine();
    Console.WriteLine($"--- {Path.GetFileNameWithoutExtension(path)}");
    Console.WriteLine($"    espeak : {string.Concat(oracle)}");
    Console.WriteLine($"    ours   : {string.Concat(ours)}");
    Console.WriteLine($"    {sentenceDiffs} differences over {oracle.Length} espeak symbols "
                    + $"({sentenceDiffs / (double)oracle.Length:P0})");
}

// ---- The answer --------------------------------------------------------------------------------------
Console.WriteLine();
Console.WriteLine("=====================================================================================");
Console.WriteLine($"TOTAL: {totalDiffs} differences over {totalOracle} espeak symbols "
                + $"({totalDiffs / (double)totalOracle:P1}); {oov}/{oovTotal} words not in CMUdict");
Console.WriteLine($"flapping applied: {applyFlap}");
Console.WriteLine();
Console.WriteLine($"{"difference class",-34} {"count",6}  {"share",7}  examples");
Console.WriteLine(new string('-', 100));
foreach (var kv in classCounts.OrderByDescending(k => k.Value))
    Console.WriteLine($"{kv.Key,-34} {kv.Value,6}  {kv.Value / (double)Math.Max(1, totalDiffs),6:P1}  "
                    + string.Join("  ", classExamples[kv.Key].Take(3)));

Console.WriteLine();
Console.WriteLine("Classes named UNTESTED-* have no row in the sensitivity table and are unmeasured. Every");
Console.WriteLine("one of them needs a perturbation adding before the phonemizer's accuracy can be trusted.");
return 0;

// ------------------------------------------------------------------------------------------------------

string[] MapWord(string[] arpa)
{
    var outSyms = new List<string>();
    foreach (var raw in arpa)
    {
        var stressDigit = raw.Length > 0 && char.IsDigit(raw[^1]) ? raw[^1] - '0' : -1;
        var bare = stressDigit >= 0 ? raw[..^1] : raw;
        if (stressDigit >= 0)
        {
            // espeak writes the stress mark immediately before the VOWEL, not before the syllable onset:
            // better is bˈɛɾɚ, understand is ˌʌndɚstˈænd. Verified against real oracle output.
            if (stressDigit == 1) outSyms.Add("ˈ");
            else if (stressDigit == 2) outSyms.Add("ˌ");
            outSyms.AddRange(Vowel(bare, stressDigit));
        }
        else if (consonants.TryGetValue(bare, out var c)) outSyms.AddRange(c);
        else outSyms.Add("«" + raw + "»");
    }
    if (applyFlap) Flap(outSyms);
    return outSyms.ToArray();
}

// espeak taps a T or D that sits between vowels when what follows is unstressed: water, better, city.
void Flap(List<string> syms)
{
    var vowels = new HashSet<string>("aeiouæɐɑɒɔəɚɘɛɜɞɤɨɪɯʉʊʌʏɵɶøœᵻ".Select(c => c.ToString()));
    for (int i = 1; i < syms.Count - 1; i++)
    {
        if (syms[i] != "t" && syms[i] != "d") continue;
        var before = syms[i - 1];
        if (!vowels.Contains(before) && before != "ː" && before != "ɹ") continue;
        // Look ahead past a length mark to the next real symbol.
        int j = i + 1;
        while (j < syms.Count && syms[j] == "ː") j++;
        if (j >= syms.Count || !vowels.Contains(syms[j])) continue;
        if (j > 0 && (syms[j - 1] == "ˈ" || syms[j - 1] == "ˌ")) continue;   // following vowel is stressed
        syms[i] = "ɾ";
    }
}

// Classify one difference into a bucket named for the DECISION it forces on the frontend.
static string Classify(string expected, string got)
{
    if (got.StartsWith('«')) return "UNTESTED-out-of-vocabulary";
    if (expected == "ɾ" && (got == "t" || got == "d")) return "flap (TESTED: harmless)";
    if (expected == "ᵻ" && got == "ɪ") return "reduced-vowel barred-i (TESTED: harmless)";
    if (expected == "ɐ" && (got == "ə" || got == "ʌ")) return "reduced-vowel turned-a (TESTED: harmless)";
    if (expected == "ɚ" || got == "ɚ") return "r-coloured vowel (TESTED: harmless)";
    if (expected == "ː" || got == "ː") return "length mark (TESTED: DAMAGING)";
    // Direction matters. Dropping a mark, inventing one, and swapping primary for secondary are three
    // different defects with three different fixes, and lumping them hides the biggest one.
    if (expected.Length == 0 && got is "ˈ" or "ˌ") return "UNTESTED-stress-ADDED (citation-form function word)";
    if (got.Length == 0 && expected is "ˈ" or "ˌ") return "stress DROPPED (TESTED: DAMAGING)";
    if (expected is "ˈ" or "ˌ" && got is "ˈ" or "ˌ") return "stress primary/secondary swapped (TESTED: DAMAGING)";
    if (expected is "ˈ" or "ˌ" || got is "ˈ" or "ˌ") return "stress vs a phoneme (TESTED: DAMAGING)";
    if (expected == "ʔ" || got == "ʔ") return "UNTESTED-glottal-stop";
    if (expected == "̩" || got == "̩") return "UNTESTED-syllabic-consonant";
    if (expected == " " || got == " ") return "UNTESTED-word-boundary";
    if (expected == "-" || got == "-") return "UNTESTED-missing-symbol";
    return $"UNTESTED-other ({Show(expected)} vs {Show(got)})";
}

static string Show(string s) => s.Length == 0 ? "∅" : s == " " ? "␣" : s;

// Levenshtein alignment, so a difference is reported against the symbol it actually corresponds to
// rather than against whatever happens to sit at the same index.
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
