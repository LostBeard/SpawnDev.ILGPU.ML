// Reads a PMT gate log and prints WHY it failed, by CAUSE rather than by count.
//
// 🔴 WHY THIS EXISTS. The 2026-09-15 overnight sweep reported `Failed: 348, Passed: 4802`. That number is
// almost pure noise: 345 of the 348 were one D3D12 device loss. `InstanceNorm_StyleMosaicShape_MatchesCpu`
// hit the 30 s cap with DXGI_ERROR_DEVICE_HUNG, the OS removed the device, and every WebGPU test after it
// failed at requestDevice with DEVICE_REMOVED - including two DefaultTests, because
// CreatePreferredAcceleratorAsync picks WebGPU. Exactly ONE of the 348 was an unrelated defect.
//
// Read naively, 348 says "a broad regression landed". Read correctly, it says "one kernel hung the GPU and
// one WebGL test threw". Those lead to completely different mornings. The tell is distributional and a
// human should not have to spot it: a huge tally in ONE lane with the other lanes clean is a device loss,
// because a dead device fails only the lane that shares it.
//
// So: never report a PMT failure count without also reporting the distinct causes behind it.
//
//   dotnet run tools/gate-summary.cs -- <path-to-gate.log>
//
// ⚠️ This is a LOG PARSER and touches no GPU, so the global Rule 9 carve-out (no file-based `dotnet run`
//    for ILGPU work - Context.Create needs Reflection.Emit, which file-based apps disable) does not apply.

using System.Text.RegularExpressions;

if (args.Length < 1)
{
    Console.WriteLine("usage: dotnet run tools/gate-summary.cs -- <path-to-gate.log>");
    return 2;
}

var path = args[0];
if (!File.Exists(path))
{
    Console.WriteLine($"gate-summary: no such log: {path}");
    return 2;
}

// ⚠️ NOT File.ReadAllLines. The gate scripts invoke this with the log still open for append by the very
// cmd.exe that is running us (`>> "%LOG%"`), so an exclusive open throws IOException and the summary - the
// whole point of the tool - is replaced by a stack trace in the log. Open share-everything instead.
string[] lines;
using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
using (var sr = new StreamReader(fs))
{
    var all = new List<string>();
    string? ln;
    while ((ln = sr.ReadLine()) != null) all.Add(ln);
    lines = all.ToArray();
}

// "  Failed SomeLane.SomeTest [30 s]" - the lane prefix is what makes the distribution readable.
var failedRx = new Regex(@"^\s*Failed\s+([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)", RegexOptions.Compiled);
// Signatures of a device that is already gone - these failures are WRECKAGE, not findings.
string[] downstream =
{
    "DXGI_ERROR_DEVICE_REMOVED",
    "Device failed at creation",
    "Failed to execute 'requestDevice'",
};
// Signatures of the device DYING - the first of these is the actual event.
string[] deviceLoss = { "DXGI_ERROR_DEVICE_HUNG", "DEVICE_REMOVED_REASON", "Device removed reason" };

var byLane = new Dictionary<string, int>();
var failures = new List<(int Line, string Lane, string Test)>();
for (int i = 0; i < lines.Length; i++)
{
    var m = failedRx.Match(lines[i]);
    if (!m.Success) continue;
    var lane = m.Groups[1].Value;
    var test = m.Groups[2].Value;
    failures.Add((i, lane, test));
    byLane[lane] = byLane.GetValueOrDefault(lane) + 1;
}

// Where the device actually died. Everything BEFORE this line is a finding no matter what it says.
var lossAt = -1;
for (int i = 0; i < lines.Length && lossAt < 0; i++)
    foreach (var sig in deviceLoss)
        if (lines[i].Contains(sig, StringComparison.Ordinal)) { lossAt = i; break; }

// ⚠️ THE BOUNDARY IS THE WHOLE DIFFICULTY, and the first version of this got it wrong in the direction
// that LOOKS right: scanning from each "Failed" line to the NEXT one swept in the following test's console
// block, every one of which says "Device failed at creation" once the device is gone. It reported 348 of
// 348 downstream and 0 findings on the very log this tool was written for - hiding BOTH the cause and the
// one real defect. Validated against a log whose answer was already known by hand, which is the only reason
// it was caught.
//
// A failure's OWN record is "Failed ..." / "Error Message:" / ... / "Stack Trace:". Stop at "Stack Trace:"
// and nothing from a neighbouring test can be read as evidence about this one.
int Classify(int idx)
{
    int start = failures[idx].Line;
    int hardEnd = idx + 1 < failures.Count ? failures[idx + 1].Line : lines.Length;
    for (int i = start; i < hardEnd; i++)
    {
        if (i > start && lines[i].Contains("Stack Trace:", StringComparison.Ordinal)) break;
        foreach (var sig in downstream)
            if (lines[i].Contains(sig, StringComparison.Ordinal)) return 1;
    }
    return 0;
}

int downstreamCount = 0;
var realFailures = new List<(int Line, string Lane, string Test)>();
for (int i = 0; i < failures.Count; i++)
{
    // A failure that happened BEFORE the device died cannot be wreckage from it - it is a candidate for
    // being the CAUSE. This is what keeps the culprit (a 30 s timeout whose own block reports the hang)
    // out of the pile of victims.
    bool afterLoss = lossAt >= 0 && failures[i].Line > lossAt;
    if (afterLoss && Classify(i) == 1) downstreamCount++;
    else realFailures.Add(failures[i]);
}

var summaryLine = lines.LastOrDefault(l => l.StartsWith("Failed!") || l.StartsWith("Passed!")) ?? "(no PMT summary line)";

Console.WriteLine();
Console.WriteLine("──────────────────────────────────────────────────────────────────────────");
Console.WriteLine($"  gate-summary  {Path.GetFileName(path)}");
Console.WriteLine("──────────────────────────────────────────────────────────────────────────");
Console.WriteLine($"  PMT says : {summaryLine.Trim()}");
Console.WriteLine($"  Failed lines found: {failures.Count}");
Console.WriteLine();

if (failures.Count == 0)
{
    Console.WriteLine("  No failures. Nothing to attribute.");
    Console.WriteLine("──────────────────────────────────────────────────────────────────────────");
    return 0;
}

Console.WriteLine("  Failures by lane (a big tally in ONE lane with the rest clean = device loss):");
foreach (var kv in byLane.OrderByDescending(k => k.Value))
    Console.WriteLine($"    {kv.Key,-20} {kv.Value,5}");
Console.WriteLine();

// The device-loss event itself was located above; it is usually attached to the FIRST failure, which is
// the cause, and never to the hundreds that follow it.
if (downstreamCount > 0)
{
    Console.WriteLine($"  🔴 {downstreamCount} of {failures.Count} failures are DOWNSTREAM of a lost device");
    Console.WriteLine("     (their error names DEVICE_REMOVED / 'Device failed at creation' / requestDevice).");
    Console.WriteLine("     These are wreckage. Do not read them as defects, and do not count them.");
    if (lossAt >= 0)
        Console.WriteLine($"     Device loss first logged at line {lossAt + 1}: {lines[lossAt].Trim()}");
    Console.WriteLine();
}

Console.WriteLine($"  ⭐ {realFailures.Count} DISTINCT failure(s) to investigate — the first is usually the cause:");
foreach (var f in realFailures.Take(40))
    Console.WriteLine($"     line {f.Line + 1,6}  {f.Lane}.{f.Test}");
if (realFailures.Count > 40)
    Console.WriteLine($"     ... and {realFailures.Count - 40} more");

Console.WriteLine("──────────────────────────────────────────────────────────────────────────");
return 0;
