#:property JsonSerializerIsReflectionEnabledByDefault=true
// Is every operator we ADVERTISE actually implemented, and does it answer honestly when it cannot?
//
//   dotnet run tools/audit-operator-support.cs
//
// WHY: `If` was listed in OperatorRegistry.BuiltinOpTypes and had a complete-looking Execute. Neither
// branch ever ran - its subgraph was destroyed by a JSON round trip - so ZipVoice's text encoder computed
// every relative-position bias from a scalar and diverged from onnxruntime by 18.6% of peak while
// producing self-consistent shapes all the way to the output. It cost hours to localise.
//
// The list is a CLAIM. This checks it three ways:
//
//   1. LISTED BUT NOT IMPLEMENTED - in BuiltinOpTypes with no `OpType => "X"` anywhere.
//   2. SILENT FALLBACK - the implementation has a branch that returns, copies its input through, or fills
//      zeros when it cannot get what it needs. That is the shape that hurts: a plausible tensor of the
//      right shape, no exception, and a wrong answer that surfaces hundreds of nodes later.
//   3. SHAPE FROM INPUTS[0] - control-flow ops (If/Loop/Scan) whose output shape is taken from an input.
//      For If, inputs[0] is the CONDITION, so the output buffer gets one element.
//
// A finding is not automatically a bug: a fallback can be correct (an optional input that really is
// absent), and some ops legitimately pass data through. This prints WHERE to look, with the line, so the
// judgement stays with a person. It is a map, not a verdict.
using System.Text.RegularExpressions;

var root = args.FirstOrDefault(a => !a.StartsWith("-"))
           ?? Path.Combine(Directory.GetCurrentDirectory(), "SpawnDev.ILGPU.ML");
if (!Directory.Exists(root)) { Console.WriteLine($"no such directory: {root}"); return 2; }

var sources = Directory.GetFiles(root, "*.cs", SearchOption.AllDirectories)
    .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}")
             && !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}"))
    .ToArray();

// ── the advertised list ───────────────────────────────────────────────────────
var registry = sources.FirstOrDefault(f => f.EndsWith("OperatorRegistry.cs"));
if (registry == null) { Console.WriteLine("OperatorRegistry.cs not found"); return 2; }
var registryText = File.ReadAllText(registry);
var listStart = registryText.IndexOf("BuiltinOpTypes", StringComparison.Ordinal);
if (listStart < 0) { Console.WriteLine("BuiltinOpTypes not found"); return 2; }
var listEnd = registryText.IndexOf("};", listStart, StringComparison.Ordinal);
var listBody = registryText[listStart..(listEnd < 0 ? registryText.Length : listEnd)];
var advertised = Regex.Matches(listBody, "\"([A-Za-z0-9_]+)\"")
    .Select(m => m.Groups[1].Value).Distinct().OrderBy(x => x, StringComparer.Ordinal).ToList();

// ── what is actually implemented ──────────────────────────────────────────────
var implemented = new Dictionary<string, string>(StringComparer.Ordinal);
foreach (var f in sources)
    foreach (Match m in Regex.Matches(File.ReadAllText(f), @"OpType\s*=>\s*""([A-Za-z0-9_]+)"""))
        implemented.TryAdd(m.Groups[1].Value, Path.GetFileName(f));

Console.WriteLine($"advertised in BuiltinOpTypes : {advertised.Count}");
Console.WriteLine($"with an OpType implementation: {implemented.Count}");

var missing = advertised.Where(o => !implemented.ContainsKey(o)).ToList();
Console.WriteLine();
Console.WriteLine($"=== 1. LISTED BUT NOT IMPLEMENTED ({missing.Count}) ===");
Console.WriteLine("    Resolving one of these throws, so a model using it fails loudly - noisy, not silent.");
foreach (var chunk in missing.Chunk(8)) Console.WriteLine("    " + string.Join(", ", chunk));

// ── silent fallbacks, per operator class ──────────────────────────────────────
// Walk each file once, tracking which operator class we are inside, and flag the shapes that return a
// plausible wrong answer instead of failing.
var suspects = new List<(string Op, string File, int Line, string Kind, string Text)>();
var opStart = new Regex(@"OpType\s*=>\s*""([A-Za-z0-9_]+)""");
var nullGuard = new Regex(@"==\s*null\)\s*(\{)?\s*$|==\s*null\s*\|\|");
// ⚠⚠ Must match a return on the SAME line as the guard: `if (w == null || r == null) return;`
// is how RNN/LSTM/GRU silently produce nothing, and an anchored ^ regex missed all three.
var silentReturn = new Regex(@"(^|\)\s*)(return;|return\s+new\[\]|\{\s*return;)");
var passThrough = new Regex(@"CopyFrom\(|\.Scale\(.*,\s*1f\)|ElementWise\.Fill\(");
var shapeFromInput = new Regex(@"InferOutputShapes\([^)]*\)\s*=>\s*new\[\]\s*\{\s*i(nputs)?\.Length\s*>\s*0\s*\?\s*i(nputs)?\[0\]");

foreach (var f in sources)
{
    var lines = File.ReadAllLines(f);
    string current = "";
    for (int i = 0; i < lines.Length; i++)
    {
        var m = opStart.Match(lines[i]);
        if (m.Success) current = m.Groups[1].Value;
        if (current.Length == 0) continue;

        if (shapeFromInput.IsMatch(lines[i]))
            suspects.Add((current, Path.GetFileName(f), i + 1, "shape from inputs[0]", lines[i].Trim()));

        // A null guard whose body only returns / copies / fills is the silent-wrong-answer shape.
        if (nullGuard.IsMatch(lines[i]))
        {
            var window = string.Join(" ", lines.Skip(i).Take(4));
            if (silentReturn.IsMatch(lines[Math.Min(i + 1, lines.Length - 1)]) || silentReturn.IsMatch(lines[i])
                || passThrough.IsMatch(window))
            {
                var kind = passThrough.IsMatch(window) ? "falls back to input/zeros" : "returns without computing";
                suspects.Add((current, Path.GetFileName(f), i + 1, kind, lines[i].Trim()));
            }
        }
    }
}

Console.WriteLine();
Console.WriteLine($"=== 2/3. SILENT FALLBACKS AND SHAPE-FROM-INPUT ({suspects.Count} sites, {suspects.Select(s => s.Op).Distinct().Count()} operators) ===");
Console.WriteLine("    Each is a place the op can produce a plausible tensor instead of an answer.");
foreach (var g in suspects.GroupBy(s => s.Op).OrderBy(g => g.Key, StringComparer.Ordinal))
{
    Console.WriteLine($"  {g.Key}");
    foreach (var s in g.Take(4))
        Console.WriteLine($"      {s.File}:{s.Line}  [{s.Kind}]  {Trim(s.Text, 90)}");
    if (g.Count() > 4) Console.WriteLine($"      ... {g.Count() - 4} more site(s)");
}

Console.WriteLine();
Console.WriteLine("Reading order: control flow (If/Loop/Scan) and anything a real model uses at runtime.");
Console.WriteLine("A fallback is only acceptable when the thing it falls back to is CORRECT - otherwise the");
Console.WriteLine("op should throw and name what it needed.");
return 0;

static string Trim(string s, int n) => s.Length <= n ? s : s[..n] + "...";
