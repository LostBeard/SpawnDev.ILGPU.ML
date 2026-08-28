// Ground-truth fixture generator: English sentences in, ZipVoice token ids out.
//
//   dotnet run --project tools/zipvoice-fixture -c Release -- --file sentences.txt
//   dotnet run --project tools/zipvoice-fixture -c Release -- "About the roses." "Paint the wall."
//
// WHY THIS EXISTS: the ids a sentence becomes cannot be read out of any table we ship. lexicon.txt is
// Chinese-only, so English goes through espeak-ng inside sherpa-onnx, and the ONLY place the result is
// observable is sherpa's debug log. Capturing that by hand does not scale past one sentence, and every
// measurement of the phonemizer needs many - so this drives the oracle, parses what it printed, and
// writes fixtures the harness can consume.
//
// It is a CHILD PROCESS rather than a library call on purpose: sherpa logs from native code straight to
// fd 2, which Console.SetError cannot intercept. Redirecting the child's stderr catches all of it.
//
// The prompt is a fixed clip, so its ids are a fixed PREFIX of every capture. Rather than assume how many
// debug lines the prompt occupies, the prefix is matched against a known-good fixture and the run FAILS
// if it does not line up - a silently mis-split capture would poison every measurement built on it.
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

var sentences = new List<string>();
string outDir = Path.Combine(AppContext.BaseDirectory, "fixtures");
string promptFrom = "";
string? oracleExe = null;
string? promptWavOverride = null, promptTextOverride = null;

for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--file":
            if (++i >= args.Length) return Fail("--file needs a path");
            foreach (var line in File.ReadAllLines(args[i]))
            {
                var s = line.Trim();
                if (s.Length > 0 && !s.StartsWith('#')) sentences.Add(s);
            }
            break;
        case "--out": if (++i >= args.Length) return Fail("--out needs a path"); outDir = args[i]; break;
        case "--prompt-from": if (++i >= args.Length) return Fail("--prompt-from needs a path"); promptFrom = args[i]; break;
        case "--oracle": if (++i >= args.Length) return Fail("--oracle needs a path"); oracleExe = args[i]; break;
        case "--prompt-wav": if (++i >= args.Length) return Fail("--prompt-wav needs a path"); promptWavOverride = args[i]; break;
        case "--prompt-text": if (++i >= args.Length) return Fail("--prompt-text needs text"); promptTextOverride = args[i]; break;
        default: sentences.Add(args[i]); break;
    }
}

if (sentences.Count == 0)
{
    Console.WriteLine("usage: zipvoice-fixture [--file sentences.txt] [--out dir] [--prompt-from fixture.json] [--oracle exe] \"sentence\" ...");
    return 2;
}

// The repo copy is the source of truth for both the prompt ids and the prompt's transcript, so a fixture
// generated today pairs with the same reference clip as the ones generated before it.
if (promptWavOverride == null || promptTextOverride == null)
{
    if (promptFrom.Length == 0)
        promptFrom = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "fixtures", "loaded-classes.json"));
    if (!File.Exists(promptFrom)) return Fail($"no reference fixture at {promptFrom} (pass --prompt-from)");
}

oracleExe ??= Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..",
                                            "zipvoice-oracle", "bin", "Release", "net10.0", "ZipVoiceOracle.exe"));
if (!File.Exists(oracleExe))
    return Fail($"no oracle at {oracleExe}\nbuild it first: dotnet build tools/zipvoice-oracle -c Release");


long[] promptTokens;
string promptText, promptWav;

if (promptWavOverride != null && promptTextOverride != null)
{
    // A DIFFERENT reference voice. Its token ids are not in any fixture yet, so they are DERIVED: run
    // two different sentences and take the common prefix, which is exactly the prompt and nothing else.
    // Counting debug lines instead would be wrong - the frontend splits on commas as well as full stops,
    // so how many lines a prompt occupies depends on its punctuation.
    promptWav = promptWavOverride;
    promptText = promptTextOverride;
    if (!File.Exists(promptWav)) return Fail($"no prompt wav at {promptWav}");

    Console.WriteLine($"prompt   : deriving ids for {Path.GetFileName(promptWav)} from two probe runs");
    var probeA = RunOracle(oracleExe, "The birch canoe slid on the smooth planks.", out _, promptWav, promptText);
    var probeB = RunOracle(oracleExe, "Rice is often served in round bowls.", out _, promptWav, promptText);
    var flatA = probeA.SelectMany(x => x).ToArray();
    var flatB = probeB.SelectMany(x => x).ToArray();
    int shared = 0;
    while (shared < Math.Min(flatA.Length, flatB.Length) && flatA[shared] == flatB[shared]) shared++;
    if (shared == 0) return Fail("the two probe runs share no prefix - the oracle did not print token ids");
    promptTokens = flatA[..shared];
}
else
{
    using var refDoc = JsonDocument.Parse(File.ReadAllText(promptFrom));
    promptTokens = refDoc.RootElement.GetProperty("promptTokens").EnumerateArray().Select(e => e.GetInt64()).ToArray();
    promptText = refDoc.RootElement.GetProperty("promptText").GetString() ?? "";
    promptWav = refDoc.RootElement.TryGetProperty("promptWav", out var pw) ? pw.GetString() ?? "prompt.wav" : "prompt.wav";
}

Directory.CreateDirectory(outDir);
Console.WriteLine($"oracle   : {oracleExe}");
Console.WriteLine($"prompt   : {promptTokens.Length} ids, {Path.GetFileName(promptWav)}");
Console.WriteLine($"outDir   : {outDir}");
Console.WriteLine();

int failures = 0;
foreach (var sentence in sentences)
{
    var slug = Slug(sentence);
    var path = Path.Combine(outDir, slug + ".json");
    Console.WriteLine($"--- {sentence}");

    var captured = RunOracle(oracleExe, sentence, out var stderrText, promptWavOverride, promptTextOverride);
    if (captured.Count == 0)
    {
        Console.WriteLine("    FAILED: the oracle printed no token ids. Its stderr:");
        Console.WriteLine("    " + string.Join("\n    ", stderrText.Split('\n').TakeLast(8)));
        failures++;
        continue;
    }

    // Peel the prompt off the front by matching ids, not by counting lines.
    var flat = captured.SelectMany(x => x).ToArray();
    if (flat.Length <= promptTokens.Length || !flat.Take(promptTokens.Length).SequenceEqual(promptTokens))
    {
        Console.WriteLine($"    FAILED: the capture does not start with the known prompt ids, so the split between "
                        + $"prompt and text is unknown. Captured {flat.Length} ids in {captured.Count} lines.");
        failures++;
        continue;
    }
    var textTokens = flat.Skip(promptTokens.Length).ToArray();

    var fixture = new
    {
        _comment = new[]
        {
            "Ground-truth ZipVoice token ids captured from sherpa-onnx with model debug on - the ids espeak-ng",
            "produced for this sentence. Observed, not derived, which is what makes them a valid target for",
            "our own phonemizer to reproduce.",
            "",
            "Generated by tools/zipvoice-fixture. Regenerate with:",
            $"  dotnet run --project tools/zipvoice-fixture -c Release -- \"{sentence.Replace("\"", "\\\"")}\"",
        },
        text = sentence,
        promptText,
        // Stored PORTABLY, not as the path that happened to be typed on the command line. The harness
        // resolves a relative promptWav against a fixed set of roots (model dir, the harness's own
        // fixtures dir, cwd), so a fixture carrying "tools/zipvoice-harness/fixtures/reference/x.wav"
        // resolves nowhere and every render dies on a missing prompt. Keep the path from the
        // "reference/" segment on, which is the convention every fixture in the tree already uses.
        promptWav = PortablePromptPath(promptWav),
        tokens = textTokens,
        promptTokens,
    };
    File.WriteAllText(path, JsonSerializer.Serialize(fixture, new JsonSerializerOptions
    {
        WriteIndented = true,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    }));
    Console.WriteLine($"    {textTokens.Length} text ids -> {Path.GetFileName(path)}");
}

Console.WriteLine();
Console.WriteLine(failures == 0
    ? $"RESULT   : {sentences.Count} fixture(s) written to {outDir}"
    : $"RESULT   : {failures} of {sentences.Count} FAILED");
return failures == 0 ? 0 : 1;

static int Fail(string message) { Console.WriteLine(message); return 2; }

// Runs the oracle for one sentence and returns every "new sentence: [...]" id list it printed, in order.
static List<long[]> RunOracle(string exe, string sentence, out string stderrText,
                              string? promptWav = null, string? promptText = null)
{
    var wav = Path.Combine(Path.GetTempPath(), "zipvoice-fixture-scratch.wav");
    var psi = new ProcessStartInfo(exe) { RedirectStandardError = true, RedirectStandardOutput = true };
    psi.ArgumentList.Add(sentence);
    psi.ArgumentList.Add(wav);
    // The ids are printed by the frontend, which is UTF-8; without this the IPA arrives mangled and the
    // parse below silently misses lines.
    psi.StandardErrorEncoding = Encoding.UTF8;
    psi.StandardOutputEncoding = Encoding.UTF8;
    if (promptWav != null) psi.Environment["ZIPVOICE_PROMPT_WAV"] = promptWav;
    if (promptText != null) psi.Environment["ZIPVOICE_PROMPT_TEXT"] = promptText;

    using var proc = Process.Start(psi) ?? throw new InvalidOperationException($"could not start {exe}");
    var err = new StringBuilder();
    proc.ErrorDataReceived += (_, e) => { if (e.Data != null) err.AppendLine(e.Data); };
    proc.BeginErrorReadLine();
    proc.StandardOutput.ReadToEnd();
    proc.WaitForExit();

    stderrText = err.ToString();
    var results = new List<long[]>();
    foreach (Match m in Regex.Matches(stderrText, @"new sentence:\s*\[([0-9,\s]*)\]"))
    {
        var body = m.Groups[1].Value.Trim();
        if (body.Length == 0) continue;
        results.Add(body.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                        .Select(long.Parse).ToArray());
    }
    return results;
}

static string Slug(string sentence)
{
    var sb = new StringBuilder();
    foreach (var c in sentence.ToLowerInvariant())
    {
        if (char.IsLetterOrDigit(c)) sb.Append(c);
        else if (sb.Length > 0 && sb[^1] != '-') sb.Append('-');
    }
    var slug = sb.ToString().Trim('-');
    return slug.Length <= 48 ? slug : slug[..48].TrimEnd('-');
}

// A fixture must name its prompt clip the way the harness can find it again. The harness resolves a
// relative promptWav against the model dir, its own fixtures dir and the working directory - never
// against the fixture's own location - so anything more specific than the "reference/..." convention
// resolves nowhere and every render fails on a missing prompt.
static string PortablePromptPath(string path)
{
    var normalized = path.Replace(Path.DirectorySeparatorChar, '/').Replace(Path.AltDirectorySeparatorChar, '/');
    int cut = normalized.LastIndexOf("reference/", StringComparison.OrdinalIgnoreCase);
    return cut >= 0 ? normalized[cut..] : Path.GetFileName(normalized);
}
