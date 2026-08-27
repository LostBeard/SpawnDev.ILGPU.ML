// Builds a self-contained listening page from a sensitivity run, so a human can grade the audio without
// knowing what a flap or a reduced vowel is.
//
//   dotnet run --project tools/zipvoice-listen -c Release -- <resultsDir> [out.html] [--fixture <name>] [--seed N]
//
// WHY THIS EXISTS: WER is a coarse grader. It hears a wrong WORD; it cannot hear a wrong ACCENT, an odd
// rhythm, or a word that is technically recognisable but sounds mechanical. Those are exactly the failures
// a phonemizer produces, and the only instrument for them is a person's ear.
//
// But "here are ten wav files, tell me if they sound wrong" is not a question anyone can answer. Each clip
// needs to say what was done to it and what the difference would sound like IF it were audible. That turns
// listening from a vague impression into a specific yes or no, which is a judgement a non-specialist can
// make confidently and which is worth trusting.
//
// The wavs are embedded as data URIs so the page is one file that works anywhere, including published as
// an artifact, with no server and no missing-file problems.
using System.Globalization;
using System.Text;
using System.Text.Json;

var resultsDir = args.Length > 0 && !args[0].StartsWith("--") ? args[0] : Path.Combine(Path.GetTempPath(), "zv-phase1");
var outPath = args.Length > 1 && !args[1].StartsWith("--") ? args[1] : Path.Combine(resultsDir, "listen.html");
string? wantFixture = null;
int? wantSeed = null;
for (int i = 0; i < args.Length; i++)
{
    if (args[i] == "--fixture" && i + 1 < args.Length) wantFixture = args[++i];
    if (args[i] == "--seed" && i + 1 < args.Length) wantSeed = int.Parse(args[++i]);
}

var resultsPath = Path.Combine(resultsDir, "results.json");
if (!File.Exists(resultsPath)) { Console.WriteLine($"no results.json in {resultsDir}"); return 2; }

using var doc = JsonDocument.Parse(File.ReadAllText(resultsPath));
var rows = doc.RootElement.EnumerateArray().Select(e => new Row(
    e.GetProperty("Fixture").GetString()!,
    e.GetProperty("Seed").GetInt32(),
    e.GetProperty("Variant").GetString()!,
    e.GetProperty("Edits").GetInt32(),
    e.GetProperty("Wav").GetString()!,
    e.GetProperty("Transcript").GetString() ?? "",
    e.GetProperty("InfixWer").GetDouble(),
    e.TryGetProperty("MelDistance", out var md) && md.ValueKind == JsonValueKind.Number ? md.GetDouble() : null,
    e.TryGetProperty("SeedBaseline", out var sb) && sb.ValueKind == JsonValueKind.Number ? sb.GetDouble() : null
)).ToList();

// Default to the sentence that exercises the most error classes, which is the one worth an ear.
wantFixture ??= rows.GroupBy(r => r.Fixture).OrderByDescending(g => g.Count(r => r.Edits > 0)).First().Key;
wantSeed ??= rows.Where(r => r.Fixture == wantFixture).Select(r => r.Seed).First();

var picked = rows.Where(r => r.Fixture == wantFixture && r.Seed == wantSeed)
                 .Where(r => r.Variant == "control" || r.Edits > 0).ToList();
if (picked.Count == 0) { Console.WriteLine($"nothing to show for {wantFixture} seed {wantSeed}"); return 2; }

var control = picked.FirstOrDefault(r => r.Variant == "control");
double noiseFloor = rows.Where(r => r.Variant == "control" && r.SeedBaseline is > 0)
                        .Select(r => r.SeedBaseline!.Value).DefaultIfEmpty(double.NaN).Average();

// What a listener should actually attend to, per variant. Written for someone who has never met the IPA.
var guide = new Dictionary<string, (string Change, string Listen)>
{
    ["control"] = ("Nothing. This is the reference rendering, and every other clip is a copy of it with one thing altered.",
        "Play this first and get used to it. Note that it opens with a few words of the voice-clone reference clip before the sentence starts. That happens in the reference implementation too, it is not what we are testing, and it is why the automatic scoring ignores anything before the sentence begins."),
    ["flap-to-t"] = ("The quick D-like tap that American English uses in the middle of water and better was replaced by a hard T.",
        "Does it sound like a crisp British wa-TER, or still natural and American? A hard T here is not broken English, just a different accent."),
    ["flap-to-d"] = ("The same tap was replaced by a hard D instead.",
        "Listen for wader and bedder. Slightly off is expected; unintelligible is not."),
    ["barred-i-to-small-i"] = ("The faint vowel ending roses and waited was replaced with a fuller ih.",
        "Compare the last syllable of those words against the reference. This should be very close to inaudible."),
    ["barred-i-to-schwa"] = ("The same faint vowel was replaced with an uh instead, which is what the dictionary gives us for packages and collected.",
        "Again the final syllables. If you cannot tell this from the reference, that is the finding."),
    ["turned-a-to-schwa"] = ("The unstressed vowel starting about was swapped for a plain uh.",
        "The first syllable of about. Almost certainly indistinguishable."),
    ["article-a-to-schwa"] = ("The standalone word a was given the dictionary's uh instead of the bare a the reference uses.",
        "The article on its own. Does it disappear into the following word, or still sound like a word?"),
    ["r-schwa-split"] = ("The single r-coloured vowel ending mother and better was split into two sounds, uh followed by r.",
        "Do those word endings sound stretched or clumsy, moth-uh-r rather than mother?"),
    ["no-secondary-stress"] = ("The lighter of the two stresses in a long word such as understand was removed.",
        "The rhythm of the long words. Does any of them sound flattened or hurried?"),
    ["no-stress-at-all"] = ("Every stress mark was removed, so no syllable is marked for emphasis anywhere.",
        "This is the one to judge on rhythm. Does it sound robotic or evenly paced, like a machine reading a list? Are the words still recognisable underneath?"),
    ["stress-moved-later"] = ("Every stress was moved onto the following syllable, so words are emphasised in the wrong place.",
        "Expect obvious damage: a-BOUT becoming ab-OUT. In testing this collapsed into a completely different sentence. If you hear something that is not the sentence at all, that is the result, not a mistake."),
    ["stress-added-function-words"] = ("Small words the reference leaves unstressed - the, a, at, in, and - were each given a stress, which is exactly what a dictionary lookup does to them.",
        "The most important clip on this page. Listen to the little words. Does the sentence sound evenly hammered out, each word given equal weight, instead of flowing? This is the single biggest difference between the dictionary and the reference, and it sits on the thing the model cares most about."),
    ["glottal-and-syllabic-to-plain"] = ("The reference says kitten the American way, with a catch in the throat instead of a T. This clip uses a plain T and a vowel: kit-ten.",
        "The word kitten only. Both are real pronunciations, so the question is whether it still sounds like the same speaker."),
    ["open-o-to-open-a"] = ("One vowel was shifted, the way two dictionaries disagree about the word on.",
        "Listen for words like on and often sounding closer to ahn."),
    ["wrong-vowel-last-word"] = ("A DELIBERATE MISPRONUNCIATION: one vowel in one word swapped for a completely different one. This is not a mistake our phonemizer would make.",
        "You SHOULD clearly hear one word come out wrong. This clip exists to prove the test can detect damage at all - if you cannot hear anything wrong here, then none of the clean results above mean anything."),
};

var order = new[] { "control", "wrong-vowel-last-word", "stress-added-function-words", "stress-moved-later",
                    "no-stress-at-all", "no-length-marks", "no-secondary-stress", "glottal-and-syllabic-to-plain",
                    "r-schwa-split", "open-o-to-open-a", "flap-to-t", "flap-to-d",
                    "article-a-to-schwa", "barred-i-to-small-i", "barred-i-to-schwa", "turned-a-to-schwa" };
picked = picked.OrderBy(r => Array.IndexOf(order, r.Variant) is var i && i >= 0 ? i : 99).ToList();

var sentence = SentenceOf(wantFixture);
var sb = new StringBuilder();
sb.AppendLine("<title>Phonemizer Listening Test</title>");
sb.AppendLine("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=Charis+SIL:ital,wght@0,400;0,700;1,400&display=swap">
<style>
  /* Light is the complete palette; the two blocks below only redefine tokens. Cool neutrals with a slight
     blue bias, a deep teal for signal, and a burnt red reserved for the one clip that is deliberately
     broken - severity is never carried by the accent. */
  :root {
    --bg:#eceff2; --card:#fff; --ink:#141a20; --muted:#5b6773; --line:#d3dae1;
    --accent:#0e6f6a; --accent-soft:#d7eceb; --flag:#9c3a22; --flag-soft:#f6e0d9; --shadow:0 1px 2px rgba(20,26,32,.06);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg:#0e1216; --card:#161c22; --ink:#e6ecf2; --muted:#94a2b0; --line:#252e37;
      --accent:#5ac8bf; --accent-soft:#123330; --flag:#e2886c; --flag-soft:#33201a; --shadow:none;
    }
  }
  :root[data-theme="dark"] {
    --bg:#0e1216; --card:#161c22; --ink:#e6ecf2; --muted:#94a2b0; --line:#252e37;
    --accent:#5ac8bf; --accent-soft:#123330; --flag:#e2886c; --flag-soft:#33201a; --shadow:none;
  }

  * { box-sizing:border-box; }
  body {
    margin:0; padding:3rem 1.25rem 6rem; background:var(--bg); color:var(--ink);
    font-family:Archivo,"Segoe UI",system-ui,sans-serif; font-size:16px; line-height:1.62;
    -webkit-font-smoothing:antialiased;
  }
  .wrap { max-width:54rem; margin:0 auto; display:flex; flex-direction:column; gap:1.5rem; }

  .eyebrow { font-size:.72rem; font-weight:600; letter-spacing:.13em; text-transform:uppercase; color:var(--muted); }
  h1 { font-size:clamp(1.75rem,4vw,2.4rem); font-weight:700; letter-spacing:-.02em; margin:.35rem 0 0; text-wrap:balance; }
  .lede { color:var(--muted); max-width:42rem; margin:.6rem 0 0; }

  .sentence {
    background:var(--card); border:1px solid var(--line); border-radius:.6rem; padding:1.4rem 1.6rem;
    box-shadow:var(--shadow);
  }
  .sentence p {
    font-family:"Charis SIL",Charis,Georgia,serif; font-size:1.35rem; line-height:1.45; margin:.4rem 0 0;
  }

  .legend { background:var(--card); border:1px solid var(--line); border-radius:.6rem; padding:1.3rem 1.6rem; box-shadow:var(--shadow); }
  .legend dl { display:grid; grid-template-columns:auto 1fr; gap:.55rem 1.1rem; margin:.7rem 0 0; }
  .legend dt { font-weight:600; white-space:nowrap; }
  .legend dd { margin:0; color:var(--muted); }

  .clip { background:var(--card); border:1px solid var(--line); border-radius:.6rem; padding:1.3rem 1.6rem 1.5rem; box-shadow:var(--shadow); }
  .clip.flagged { border-color:var(--flag); }
  .clip h2 { font-size:1.12rem; font-weight:600; letter-spacing:-.01em; margin:.3rem 0 0; }
  .chips { display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; }
  .chip {
    font-size:.7rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase;
    border-radius:2rem; padding:.16rem .6rem; background:var(--accent-soft); color:var(--accent);
  }
  .chip.flag { background:var(--flag-soft); color:var(--flag); }
  .chip.quiet { background:transparent; color:var(--muted); border:1px solid var(--line); }

  .field { margin-top:1rem; }
  .field .eyebrow { display:block; margin-bottom:.15rem; }
  .field p { margin:0; }
  audio { width:100%; margin-top:1.15rem; }
  .heard { font-family:"Charis SIL",Charis,Georgia,serif; font-style:italic; color:var(--muted); }
  .metrics { display:flex; flex-wrap:wrap; gap:1.6rem; margin-top:1.15rem; padding-top:1rem; border-top:1px solid var(--line); }
  .metric .eyebrow { display:block; }
  .metric b { font-size:1.15rem; font-weight:600; font-variant-numeric:tabular-nums; }
  .metric span { color:var(--muted); font-size:.85rem; }

  footer { color:var(--muted); font-size:.9rem; border-top:1px solid var(--line); padding-top:1.3rem; }
  a { color:var(--accent); }
  :focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
</style>
""");

sb.AppendLine("<div class=\"wrap\">");
sb.AppendLine("<header>");
sb.AppendLine("  <p class=\"eyebrow\">MIT phonemizer &middot; phase 1</p>");
sb.AppendLine("  <h1>Does the model notice?</h1>");
sb.AppendLine("  <p class=\"lede\">Every clip below speaks the same sentence, in the same cloned voice, from the "
            + "same random starting noise. The only thing that differs is how the words were spelled out in "
            + "phonemes. If a change is inaudible, our phonemizer does not need to get it right.</p>");
sb.AppendLine("</header>");

sb.AppendLine("<section class=\"sentence\">");
sb.AppendLine("  <p class=\"eyebrow\">The sentence being spoken</p>");
sb.AppendLine($"  <p>{Escape(sentence)}</p>");
sb.AppendLine("</section>");

sb.AppendLine("<section class=\"legend\">");
sb.AppendLine("  <p class=\"eyebrow\">How to read the two numbers</p>");
sb.AppendLine("  <dl>");
sb.AppendLine("    <dt>Words lost</dt><dd>How much worse an automatic transcriber did on this clip than on the reference. "
            + "It catches broken <em>words</em>, but it is a language model and will quietly repair a mispronunciation "
            + "back into the word it expected, so it under-reports.</dd>");
sb.AppendLine("    <dt>Sound change</dt><dd>How far the audio itself moved from the reference, as a multiple of the "
            + "noise floor. Rendering the very same phonemes twice with a different random draw already gives 1.0&times;, "
            + "so 1&times; means nothing happened. This number has no idea what the words are, which is exactly why it "
            + "is here.</dd>");
sb.AppendLine("  </dl>");
sb.AppendLine("</section>");

foreach (var row in picked)
{
    var g = guide.TryGetValue(row.Variant, out var got) ? got : (Change: "(no description)", Listen: "(no guidance)");
    bool isPc = row.Variant == "wrong-vowel-last-word";
    double? werDelta = control != null && row.Variant != "control" ? row.InfixWer - control.InfixWer : null;
    double? sound = row.MelDistance is > 0 && noiseFloor > 0 ? row.MelDistance / noiseFloor : null;

    sb.AppendLine($"<article class=\"clip{(isPc ? " flagged" : "")}\">");
    sb.AppendLine("  <div class=\"chips\">");
    if (row.Variant == "control") sb.AppendLine("    <span class=\"chip\">reference</span>");
    if (isPc) sb.AppendLine("    <span class=\"chip flag\">detector check</span>");
    if (row.Variant == "stress-added-function-words") sb.AppendLine("    <span class=\"chip\">biggest real difference</span>");
    sb.AppendLine($"    <span class=\"chip quiet\">{row.Edits} phoneme{(row.Edits == 1 ? "" : "s")} changed</span>");
    sb.AppendLine("  </div>");
    sb.AppendLine($"  <h2>{Escape(Pretty(row.Variant))}</h2>");
    sb.AppendLine($"  <div class=\"field\"><span class=\"eyebrow\">What changed</span><p>{Escape(g.Change)}</p></div>");
    sb.AppendLine($"  <div class=\"field\"><span class=\"eyebrow\">What to listen for</span><p>{Escape(g.Listen)}</p></div>");
    if (File.Exists(row.Wav))
        sb.AppendLine($"  <audio controls preload=\"none\" src=\"data:audio/wav;base64,{Convert.ToBase64String(File.ReadAllBytes(row.Wav))}\"></audio>");
    else
        sb.AppendLine($"  <p class=\"field\">(audio missing: {Escape(row.Wav)})</p>");
    sb.AppendLine($"  <div class=\"field\"><span class=\"eyebrow\">What the automatic transcriber heard</span>"
                + $"<p class=\"heard\">{Escape(row.Transcript)}</p></div>");
    sb.AppendLine("  <div class=\"metrics\">");
    sb.AppendLine("    <div class=\"metric\"><span class=\"eyebrow\">Words lost</span><b>"
                + (werDelta == null ? "&mdash;" : (werDelta.Value <= 0 ? "none" : werDelta.Value.ToString("P0", CultureInfo.InvariantCulture)))
                + "</b> <span>" + (row.Variant == "control" ? "this is the baseline" : "vs the reference") + "</span></div>");
    sb.AppendLine("    <div class=\"metric\"><span class=\"eyebrow\">Sound change</span><b>"
                + (sound == null ? "&mdash;" : sound.Value.ToString("0.00", CultureInfo.InvariantCulture) + "×")
                + "</b> <span>" + (sound == null ? "not measured" : sound.Value < 1.15 ? "at the noise floor" : "above the noise floor") + "</span></div>");
    sb.AppendLine("  </div>");
    sb.AppendLine("</article>");
}

sb.AppendLine("<footer>Generated by <code>tools/zipvoice-listen</code> from a <code>zipvoice-harness sensitivity</code> "
            + "run. The automatic transcriber only hears wrong words; your ear is the only instrument for a wrong "
            + "accent or a broken rhythm, which is why this page exists.</footer>");
sb.AppendLine("</div>");

File.WriteAllText(outPath, sb.ToString());
Console.WriteLine($"fixture  : {wantFixture} seed {wantSeed}");
Console.WriteLine($"clips    : {picked.Count}");
Console.WriteLine($"noise    : {noiseFloor:F2}");
Console.WriteLine($"wrote    : {outPath} ({new FileInfo(outPath).Length / 1024.0 / 1024:F1} MB)");
return 0;

static string Pretty(string variant) => variant.Replace('-', ' ');

static string SentenceOf(string fixtureName)
{
    foreach (var dir in new[]
    {
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "zipvoice-harness", "fixtures", "phase1"),
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "zipvoice-harness", "fixtures"),
    })
    {
        var p = Path.GetFullPath(Path.Combine(dir, fixtureName + ".json"));
        if (!File.Exists(p)) continue;
        using var d = JsonDocument.Parse(File.ReadAllText(p));
        return d.RootElement.GetProperty("text").GetString() ?? fixtureName;
    }
    return fixtureName;
}

static string Escape(string s) => s.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;");

internal sealed record Row(
    string Fixture, int Seed, string Variant, int Edits, string Wav, string Transcript,
    double InfixWer, double? MelDistance, double? SeedBaseline);
