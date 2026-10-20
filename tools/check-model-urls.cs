// Every model URL in this repo goes through ModelsHost / HuggingFaceClient.GetDownloadUrl. This proves it.
//
//   dotnet run tools/check-model-urls.cs
//   dotnet run tools/check-model-urls.cs -- <repo root>
//
// Exit code = number of violations, so it is usable as a gate.
//
// 🔴 WHY. Models come from https://models.spawndev.com (Cloudflare R2: static files, free egress, CORS open, Range
// answered from the first byte), with Hugging Face as the automatic fallback: the two share one path layout, so
// ModelsHost.GetFallbackUrl maps one to the other and every transport retries the twin by itself. That only works
// if every model request starts from a URL ModelsHost built. A hand-written host name anywhere else throws the
// fallback, the base-URL override and the layout rules away, and does it silently until the day that host fails.
// The previous rule (2026-09, "only the SpawnDev hub may talk to Hugging Face") is gone with the hub's model
// serving: hub.spawndev.com is the WebTorrent tracker only since 2026-10-20.
//
// Three rules, checked on CODE lines only (a comment cannot make a request):
//   1. huggingface.co        only where ModelsHost / HuggingFaceClient define the fallback origin, plus docs/links.
//   2. models.spawndev.com   only in ModelsHost.cs, which defines it.
//   3. hub.spawndev.com / :44365   only in the WebTorrent adapter (HubModelStream's default hub address).
//
// ⚠️ This checks SOURCE, so it catches the shape before it fails in a sweep. It is deliberately dumb: any occurrence
// outside the allowlist is a violation, because "it is only a test URL" is how the next one gets in.

using System.Text.RegularExpressions;

var root = args.FirstOrDefault(a => !a.StartsWith("-")) ?? ".";
root = Path.GetFullPath(root);
if (!Directory.Exists(root)) { Console.Error.WriteLine($"no such folder: {root}"); return 2; }

// (needle, files allowed to contain it and why). Keep these lists SHORT and justified.
var rules = new (string Needle, string What, Dictionary<string, string> Allowed)[]
{
    ("huggingface.co", "Hugging Face is the FALLBACK, reached only through ModelsHost.GetFallbackUrl", new(StringComparer.OrdinalIgnoreCase)
    {
        ["ModelsHost.cs"] = "defines HuggingFaceBaseUrl and the models-host to Hugging Face mapping",
        ["HuggingFaceClient.cs"] = "the Hugging Face API client (BaseUrl, auth) and HuggingFaceOrigin",
        ["ModelHub.cs"] = "the [Obsolete] HuggingFaceBaseUrl and its explanation",
        ["SafeTensorsParser.cs"] = "a link to the safetensors format docs",
        ["PipelinesPage.razor"] = "a docs link in the UI",
        ["ModelsPage.razor"] = "links to a model's HF page",
        ["GettingStartedPage.razor"] = "sample text",
        ["MLTestBase.KokoroFrontEndTests.cs"] = "asserts a voice repo id is NOT a URL",
        ["check-model-urls.cs"] = "this file",
    }),
    ("models.spawndev.com", "the models host is named in exactly one place", new(StringComparer.OrdinalIgnoreCase)
    {
        ["ModelsHost.cs"] = "defines DefaultBaseUrl",
        ["check-model-urls.cs"] = "this file",
    }),
    ("hub.spawndev.com", "the hub no longer serves models; it is the WebTorrent tracker", new(StringComparer.OrdinalIgnoreCase)
    {
        ["HubModelStream.cs"] = "the WebTorrent adapter's default hub address (torrent delivery needs a hub you run)",
        ["check-model-urls.cs"] = "this file",
    }),
    (":44365", "the hub's port: the hub no longer serves models", new(StringComparer.OrdinalIgnoreCase)
    {
        ["HubModelStream.cs"] = "the WebTorrent adapter's default hub address",
        ["check-model-urls.cs"] = "this file",
    }),
};

var skipDirs = new[] { $"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}",
                       $"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}",
                       $"{Path.DirectorySeparatorChar}.git{Path.DirectorySeparatorChar}",
                       $"{Path.DirectorySeparatorChar}node_modules{Path.DirectorySeparatorChar}",
                       $"{Path.DirectorySeparatorChar}publish-demo-audit{Path.DirectorySeparatorChar}",
                       $"{Path.DirectorySeparatorChar}wwwroot{Path.DirectorySeparatorChar}" };

var violations = new List<string>();
int scanned = 0;

foreach (var file in Directory.EnumerateFiles(root, "*.*", SearchOption.AllDirectories))
{
    var ext = Path.GetExtension(file);
    if (ext is not (".cs" or ".razor")) continue;
    if (skipDirs.Any(d => file.Contains(d, StringComparison.OrdinalIgnoreCase))) continue;
    scanned++;

    var name = Path.GetFileName(file);
    var lines = File.ReadAllLines(file);
    for (int i = 0; i < lines.Length; i++)
    {
        // ⚠️ A COMMENT cannot make a request. The rule's own documentation was once the only thing the old gate ever
        // found, and a gate that reports the same findings every run is a gate nobody reads. Code only: a line whose
        // first non-space characters are // or * or /* is prose.
        var trimmed = lines[i].TrimStart();
        if (trimmed.StartsWith("//") || trimmed.StartsWith("*") || trimmed.StartsWith("/*")) continue;

        foreach (var (needle, what, allowed) in rules)
        {
            if (!lines[i].Contains(needle, StringComparison.OrdinalIgnoreCase)) continue;
            if (allowed.ContainsKey(name)) continue;
            // A link to the HF docs site is not a model request.
            if (needle == "huggingface.co" && Regex.IsMatch(lines[i], @"huggingface\.co/docs", RegexOptions.IgnoreCase)) continue;
            var rel = Path.GetRelativePath(root, file);
            violations.Add($"{rel}:{i + 1}: [{needle}] {lines[i].Trim()}   <- {what}");
        }
    }
}

Console.WriteLine($"scanned {scanned} source file(s) under {root}");
if (violations.Count == 0)
{
    Console.WriteLine("every model URL is built by ModelsHost / HuggingFaceClient.GetDownloadUrl");
    return 0;
}

Console.WriteLine();
Console.WriteLine($"{violations.Count} HAND-WRITTEN MODEL URL(S) - these bypass the models-host base URL, the Hugging Face fallback and the layout rules:");
foreach (var v in violations) Console.WriteLine($"  {v}");
Console.WriteLine();
Console.WriteLine("  A file in a Hugging Face repo:   HuggingFaceClient.GetDownloadUrl(repoId, path)");
Console.WriteLine("  An Ollama layer:                 ModelsHost.GetOllamaUrl(model, tag, layer)");
Console.WriteLine("  A member of a release archive:   ModelsHost.GetArchiveMemberUrl(archiveUrl, member)");
Console.WriteLine("  Model WEIGHTS in a browser:      HubModelSource / RangeStreamModelSource (bytes stay JS-side), not a byte[]");
return violations.Count;
