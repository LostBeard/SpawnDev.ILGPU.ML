// Learns letter-to-sound from CMUdict, so the phonemizer can pronounce words no dictionary contains.
//
//   dotnet run --project tools/lts-train -c Release -- [--dict path] [--out model.txt] [--holdout 5000]
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
// THE DESIGN, AND THE ONE THAT LOST. Both were measured on the same held-out 5,000 words:
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
// The method is deliberately plain - alignment by Viterbi EM, then context models with backoff - so the
// output is inspectable text rather than weights nobody can read, and it runs anywhere with no
// dependency. It is not the state of the art; it is the honest baseline a better model has to beat.
using System.Text;

var dictPath = @"D:\users\tj\Projects\_ref\cmudict\cmudict.dict";
var outPath = Path.Combine(AppContext.BaseDirectory, "lts-model.txt");
int holdout = 5000, iterations = 6;
for (int i = 0; i < args.Length; i++)
{
    if (args[i] == "--dict" && i + 1 < args.Length) dictPath = args[++i];
    if (args[i] == "--out" && i + 1 < args.Length) outPath = args[++i];
    if (args[i] == "--holdout" && i + 1 < args.Length) holdout = int.Parse(args[++i]);
    if (args[i] == "--iterations" && i + 1 < args.Length) iterations = int.Parse(args[++i]);
}
if (!File.Exists(dictPath)) { Console.WriteLine($"no cmudict at {dictPath}"); return 2; }

// ---- Load --------------------------------------------------------------------------------------------
// Only plain alphabetic headwords: the dictionary also holds entries like "h.'s", which teach a
// letter-to-sound model nothing about letters.
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
// What a letter emits given its neighbours: two either side, then one, then the letter alone. That
// backoff is why "gh" can be silent in "night" and voiced in "ghost" without anyone writing either rule.
var wide = new Dictionary<string, Dictionary<string, int>>();
var narrow = new Dictionary<string, Dictionary<string, int>>();
var bare = new Dictionary<string, Dictionary<string, int>>();

// Stress over the SAME contexts, as a separate model.
var stressWide = new Dictionary<string, Dictionary<string, int>>();
var stressNarrow = new Dictionary<string, Dictionary<string, int>>();
var stressBare = new Dictionary<string, Dictionary<string, int>>();

foreach (var (word, phones) in train)
{
    var barePhones = phones.Select(Bare).ToArray();
    var path = Align(word, barePhones);
    if (path == null) continue;

    int consumed = 0;
    for (int i = 0; i < word.Length; i++)
    {
        Record(wide, Key(word, i, 2), path[i]);
        Record(narrow, Key(word, i, 1), path[i]);
        Record(bare, Key(word, i, 0), path[i]);

        // The alignment consumes phones in order, so walking it beside the ORIGINAL phones recovers
        // which stress digit belongs to which letter.
        int count = path[i].Length == 0 ? 0 : path[i].Split(' ').Length;
        if (count > 0)
        {
            var digits = string.Concat(phones.Skip(consumed).Take(count).Select(StressDigit));
            if (digits.Any(char.IsDigit))
            {
                Record(stressWide, Key(word, i, 2), digits);
                Record(stressNarrow, Key(word, i, 1), digits);
                Record(stressBare, Key(word, i, 0), digits);
            }
        }
        consumed += count;
    }
}
Console.WriteLine($"sounds   : {wide.Count} wide, {narrow.Count} narrow, {bare.Count} bare contexts");
Console.WriteLine($"stress   : {stressWide.Count} wide, {stressNarrow.Count} narrow contexts");

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
Console.WriteLine();
Console.WriteLine("The gap between those two is the cost of the stress model alone. A word-level model keyed");
Console.WriteLine("on the word ENDING and syllable count was tried and measured WORSE - 37.2% against 39.5%.");

// ---- Write the model -----------------------------------------------------------------------------------
// Plain text, so it can be read, diffed and corrected by hand.
var sb = new StringBuilder();
sb.AppendLine("# Letter-to-sound model, learned from CMUdict (BSD-2-Clause) by tools/lts-train.");
sb.AppendLine("# Sound rules:  context TAB phones    (an emission of - means the letter is silent)");
sb.AppendLine("# Stress rules: *context TAB digits   (one digit per phone that context emits)");
sb.AppendLine($"# Held out: {exact / (double)test.Count:P1} of words exactly right, "
            + $"{phoneErrors / (double)phoneTotal:P1} phoneme error rate, on {test.Count} words never trained on.");

foreach (var table in new[] { wide, narrow, bare })
    foreach (var kv in table.OrderBy(k => k.Key, StringComparer.Ordinal))
    {
        var best = kv.Value.OrderByDescending(v => v.Value).First();
        if (best.Value < 2 && table != bare) continue;          // a context seen once is noise
        sb.Append(kv.Key).Append('\t').AppendLine(best.Key.Length == 0 ? "-" : best.Key);
    }

foreach (var table in new[] { stressWide, stressNarrow, stressBare })
    foreach (var kv in table.OrderBy(k => k.Key, StringComparer.Ordinal))
    {
        var best = kv.Value.OrderByDescending(v => v.Value).First();
        if (best.Value < 2 && table != stressBare) continue;
        sb.Append('*').Append(kv.Key).Append('\t').AppendLine(best.Key);
    }

File.WriteAllText(outPath, sb.ToString());
Console.WriteLine($"wrote    : {outPath} ({new FileInfo(outPath).Length / 1024} KB)");
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

string[] Predict(string word)
{
    var output = new List<string>();
    for (int i = 0; i < word.Length; i++)
    {
        var emission = Lookup(wide, Key(word, i, 2)) ?? Lookup(narrow, Key(word, i, 1))
                    ?? Lookup(bare, Key(word, i, 0)) ?? "";
        if (emission.Length == 0) continue;

        var phones = emission.Split(' ');
        var digits = Lookup(stressWide, Key(word, i, 2)) ?? Lookup(stressNarrow, Key(word, i, 1))
                  ?? Lookup(stressBare, Key(word, i, 0)) ?? "";
        for (int k = 0; k < phones.Length; k++)
            output.Add(IsVowel(phones[k])
                ? phones[k] + (k < digits.Length && char.IsDigit(digits[k]) ? digits[k] : '0')
                : phones[k]);
    }

    // Exactly ONE primary stress per word. The per-letter model can mark several, which is not English -
    // and stress is the thing the downstream model punishes hardest, so emitting two is worse than
    // guessing which. The first is kept because English favours earlier stress.
    bool seenPrimary = false;
    for (int i = 0; i < output.Count; i++)
    {
        if (!output[i].EndsWith('1')) continue;
        if (!seenPrimary) { seenPrimary = true; continue; }
        output[i] = Bare(output[i]) + "0";
    }

    // Every English word has a stressed syllable. If the model marked none, stress the first, which is
    // where English puts it more often than anywhere else.
    if (output.Any(IsVowel) && !output.Any(x => x.EndsWith('1')))
    {
        int first = output.FindIndex(IsVowel);
        output[first] = Bare(output[first]) + "1";
    }
    return output.ToArray();

    static string? Lookup(Dictionary<string, Dictionary<string, int>> table, string key)
        => table.TryGetValue(key, out var inner) ? inner.OrderByDescending(v => v.Value).First().Key : null;
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
