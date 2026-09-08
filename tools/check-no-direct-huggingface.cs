// Nothing in this repo may request huggingface.co directly. This proves it.
//
//   dotnet run tools/check-no-direct-huggingface.cs
//   dotnet run tools/check-no-direct-huggingface.cs -- <repo root>
//
// Exit code = number of violations, so it is usable as a gate.
//
// 🔴 WHY. hub.spawndev.com exists to be the only thing that talks to HuggingFace. It caches, it answers
// with CORS headers a browser accepts, it serves lazy-hash torrents for random-access streaming, and it
// keeps every client and every test out of HF's rate limiter. A hand-built huggingface.co URL throws all
// of that away, and it does so silently until the day HF says no.
//
// MEASURED 2026-09-08: a sweep row failed with
//   HttpRequestException: Response status code does not indicate success: 429 (Too Many Requests)
// purely because it built its own URL. It was one of EIGHTY-FIVE such literals across fourteen files, plus
// three in ModelHub and one in the demo's /models page - which took a repo id the user typed, fetched the
// whole model from HF and put it on the WASM heap to read its graph.
//
// The right call is HuggingFaceClient.GetDownloadUrl (hub-routed) for KB-scale files, and
// HubModelStream.OpenAsync for weights - a lazy-hash torrent whose bytes stay JS-side.
//
// ⚠️ This checks SOURCE, so it catches the shape before it can fail in a sweep. It is deliberately dumb:
// any occurrence of the origin string outside the allowlist is a violation, because "it is only a comment"
// is how the next one gets in.

using System.Text.RegularExpressions;

var root = args.FirstOrDefault(a => !a.StartsWith("-")) ?? ".";
root = Path.GetFullPath(root);
if (!Directory.Exists(root)) { Console.Error.WriteLine($"no such folder: {root}"); return 2; }

// Files allowed to name the origin, and why. Keep this list SHORT and justified.
var allowed = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
{
    ["HuggingFaceClient.cs"] = "defines HuggingFaceOrigin and the hub routing itself",
    ["ModelHub.cs"] = "the [Obsolete] HuggingFaceBaseUrl and its explanation",
    ["SafeTensorsParser.cs"] = "a link to the safetensors format docs",
    ["PipelinesPage.razor"] = "a docs link in the UI",
    ["ModelsPage.razor"] = "links to a model's HF page, and the note explaining the fix",
    ["GettingStartedPage.razor"] = "sample text",
    ["check-no-direct-huggingface.cs"] = "this file",
};

var skipDirs = new[] { $"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}",
                       $"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}",
                       $"{Path.DirectorySeparatorChar}.git{Path.DirectorySeparatorChar}",
                       $"{Path.DirectorySeparatorChar}node_modules{Path.DirectorySeparatorChar}" };

var violations = new List<string>();
int scanned = 0;

foreach (var file in Directory.EnumerateFiles(root, "*.*", SearchOption.AllDirectories))
{
    var ext = Path.GetExtension(file);
    if (ext is not (".cs" or ".razor")) continue;
    if (skipDirs.Any(d => file.Contains(d, StringComparison.OrdinalIgnoreCase))) continue;
    scanned++;

    var name = Path.GetFileName(file);
    if (allowed.ContainsKey(name)) continue;

    var lines = File.ReadAllLines(file);
    for (int i = 0; i < lines.Length; i++)
    {
        if (!lines[i].Contains("huggingface.co", StringComparison.OrdinalIgnoreCase)) continue;
        // A link to the HF docs site is not a model request.
        if (Regex.IsMatch(lines[i], @"huggingface\.co/docs", RegexOptions.IgnoreCase)) continue;
        var rel = Path.GetRelativePath(root, file);
        violations.Add($"{rel}:{i + 1}: {lines[i].Trim()}");
    }
}

Console.WriteLine($"scanned {scanned} source file(s) under {root}");
if (violations.Count == 0)
{
    Console.WriteLine("no direct huggingface.co requests - everything goes through the hub");
    return 0;
}

Console.WriteLine();
Console.WriteLine($"{violations.Count} DIRECT HUGGINGFACE REFERENCE(S) - these bypass the hub's cache, its");
Console.WriteLine("CORS headers and its rate-limit shielding:");
foreach (var v in violations) Console.WriteLine($"  {v}");
Console.WriteLine();
Console.WriteLine("  For a KB-scale file:  HuggingFaceClient.GetDownloadUrl(repoId, path)");
Console.WriteLine("  For model WEIGHTS:    new HubModelStream(webTorrentClient, http).OpenAsync(repoId, path)");
Console.WriteLine("                        then InferenceSession.CreateFromOnnxStreamAsync(...)");
return violations.Count;
