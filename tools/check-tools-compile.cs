// Compile every tool in this folder. Nothing else does.
//
//   dotnet run tools/check-tools-compile.cs
//   dotnet run tools/check-tools-compile.cs -- <folder>
//
// Exit code = number of tools that failed to build, so this is usable as a gate.
//
// 🔴 WHY THIS EXISTS. A standalone .cs in tools/ belongs to no solution, so no build, no sweep and no CI
// step ever compiles it - and a gate that does not compile is INVISIBLE. It does not fail; it is simply
// never run, while its file header goes on claiming coverage.
//
// MEASURED 2026-09-08, the day this was written, by compiling the two tools folders for the first time:
//   - SpawnDev.AI/tools/drive-hands-free.cs had not built since 2026-09-04 (602882a). It is the ONLY gate
//     that presses the real hands-free button, written expressly to catch "it listened but never stopped
//     listening" and "after it finished talking it did not start listening again". Four days uncovered.
//   - SpawnDev.ILGPU.ML/tools/probe-if-foldability.cs referenced InferenceSession.Graph and .ConstantData
//     after both left the public surface. Undated, and equally silent.
//
// Two of twenty-four tools, and both were found by trying to RUN one of them. That is not a search
// strategy. This is.
//
// ⚠️ A tool that BUILDS is not a tool that WORKS - this gate is the floor, not the ceiling. It catches the
// class of failure that hides completely, which is the class worth automating.

var dir = args.FirstOrDefault(a => !a.StartsWith("-"))
          ?? Path.GetDirectoryName(Path.GetFullPath("tools/check-tools-compile.cs"))!;
if (!Directory.Exists(dir)) { Console.Error.WriteLine($"no such folder: {dir}"); return 2; }

var self = Path.GetFullPath("tools/check-tools-compile.cs");
var files = Directory.GetFiles(dir, "*.cs")
    .Where(f => !string.Equals(Path.GetFullPath(f), self, StringComparison.OrdinalIgnoreCase))
    .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
    .ToList();

Console.WriteLine($"compiling {files.Count} tool(s) in {dir}");
var failed = new List<(string File, string Error)>();

foreach (var f in files)
{
    var name = Path.GetFileName(f);
    Console.Write($"  {name,-40} ");
    var psi = new System.Diagnostics.ProcessStartInfo("dotnet")
    {
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false,
    };
    psi.ArgumentList.Add("build");
    psi.ArgumentList.Add(f);
    psi.ArgumentList.Add("-c");
    psi.ArgumentList.Add("Release");

    using var p = System.Diagnostics.Process.Start(psi)!;
    var stdout = await p.StandardOutput.ReadToEndAsync();
    var stderr = await p.StandardError.ReadToEndAsync();
    await p.WaitForExitAsync();

    if (p.ExitCode == 0) { Console.WriteLine("OK"); continue; }

    // The FIRST error line is the one to act on - the rest are usually cascade. A broken string literal in
    // particular reports a dozen bogus follow-on errors and one real cause.
    var first = (stdout + "\n" + stderr)
        .Split('\n')
        .Select(l => l.Trim())
        .FirstOrDefault(l => l.Contains(": error ")) ?? $"exit {p.ExitCode}";
    Console.WriteLine("*** BUILD FAILED ***");
    Console.WriteLine($"      {first}");
    failed.Add((name, first));
}

Console.WriteLine();
if (failed.Count == 0)
{
    Console.WriteLine($"all {files.Count} tools compile");
    return 0;
}
Console.WriteLine($"{failed.Count} of {files.Count} tools DO NOT COMPILE - each is a gate that cannot run:");
foreach (var (file, error) in failed) Console.WriteLine($"  {file}: {error}");
return failed.Count;
