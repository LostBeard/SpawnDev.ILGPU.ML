using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Xml.Linq;

namespace PlaywrightMultiTest;

/// <summary>
/// Discovers C# projects that reference Playwright and have test methods.
/// </summary>
public static class ProjectDiscovery
{
    private static readonly string[] PlaywrightPackageIds = ["Microsoft.Playwright", "Microsoft.Playwright.NUnit"];
    private static readonly string[] TestSdkPackageIds = ["Microsoft.NET.Test.Sdk", "NUnit", "NUnit3TestAdapter", "xunit", "MSTest.TestFramework"];

    /// <summary>
    /// Gets the workspace root (one directory up from current directory).
    /// </summary>
    public static List<ProjectDetails> GetWorkspaceRoot()
    {
        var current = Directory.GetCurrentDirectory();
        // likely running in a bin/ subfolder. 
        // backtrack through the path heirarchy until we find at least 1 test configurtation or don't find any.
        while (current != null)
        {
            var files = Directory.GetFiles(current, "*.csproj", SearchOption.AllDirectories);
            var testableFiles = files.Select(o => new ProjectDetails(o)).Where(o => o.IsTestProject).ToList();
            if (testableFiles.Count > 0)
            {
                return testableFiles;
            }
            current = Path.GetDirectoryName(current);
        }
        return new List<ProjectDetails>();
    }

    /// <summary>
    /// Discovers all runnable Playwright test projects under the workspace root.
    /// </summary>
    /// <param name="workspaceRoot">Root directory to scan (typically one level up from PlaywrightTestRunner)</param>
    /// <returns>List of project file paths that reference Playwright and have test infrastructure</returns>
    public static IReadOnlyList<string> DiscoverProjects(string workspaceRoot)
    {
        var result = new List<string>();
        foreach (var csproj in Directory.GetFiles(workspaceRoot, "*.csproj", SearchOption.AllDirectories))
        {
            if (IsPlaywrightTestProject(csproj))
                result.Add(csproj);
        }
        return result;
    }

    /// <summary>
    /// Checks if a project references Playwright and has test infrastructure.
    /// </summary>
    public static bool IsPlaywrightTestProject(string csprojPath)
    {
        if (!File.Exists(csprojPath))
            return false;

        try
        {
            var doc = XDocument.Load(csprojPath);
            var hasPlaywright = doc.Descendants("PackageReference")
                .Any(r => PlaywrightPackageIds.Contains(r.Attribute("Include")?.Value ?? "", StringComparer.OrdinalIgnoreCase));
            if (!hasPlaywright)
                return false;

            var hasTestSdk = doc.Descendants("PackageReference")
                .Any(r => TestSdkPackageIds.Contains(r.Attribute("Include")?.Value ?? "", StringComparer.OrdinalIgnoreCase));
            return hasTestSdk;
        }
        catch
        {
            return false;
        }


        //var playWrightJsonFile = Path.Combine(Path.GetDirectoryName(csprojPath)!, TestRunnerConfig.Filename);
        //return File.Exists(playWrightJsonFile);

        //try
        //{
        //    var doc = XDocument.Load(csprojPath);
        //    var hasPlaywright = doc.Descendants("PackageReference")
        //        .Any(r => PlaywrightPackageIds.Contains(r.Attribute("Include")?.Value ?? "", StringComparer.OrdinalIgnoreCase));
        //    if (!hasPlaywright)
        //        return false;

        //    var hasTestSdk = doc.Descendants("PackageReference")
        //        .Any(r => TestSdkPackageIds.Contains(r.Attribute("Include")?.Value ?? "", StringComparer.OrdinalIgnoreCase));
        //    return hasTestSdk;
        //}
        //catch
        //{
        //    return false;
        //}
    }

    /// <summary>
    /// Runs dotnet test --list-tests for a project and returns the test names.
    /// </summary>
    public static async Task<IReadOnlyList<string>> ListTestsAsync(string projectPath, string? filter, CancellationToken ct = default)
    {
        var args = $"test \"{projectPath}\" --list-tests";
        if (!string.IsNullOrEmpty(filter))
            args += $" --filter \"{filter}\"";

        var startInfo = new ProcessStartInfo("dotnet", args)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        var output = new List<string>();
        using var process = new Process();
        process.StartInfo = startInfo;
        process.EnableRaisingEvents = true;

        process.OutputDataReceived += (_, e) =>
        {
            if (!string.IsNullOrWhiteSpace(e.Data))
                output.Add(e.Data!.Trim());
        };
        process.ErrorDataReceived += (_, e) => { }; // drain stderr to prevent buffer deadlock

        var exitTcs = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        process.Exited += (_, _) => exitTcs.TrySetResult(true);
        using var reg = ct.Register(() => exitTcs.TrySetResult(false));

        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        var exited = await exitTcs.Task;
        if (exited)
            process.WaitForExit(5000); // timed wait — no-arg WaitForExit can hang on child processes
        else
            try { process.Kill(entireProcessTree: true); } catch { }

        return output;
    }

    /// <summary>
    /// Gets the project type (Blazor WASM, Blazor Server, etc.) from a csproj file.
    /// </summary>
    public static ProjectType GetAppProjectType(string csprojPath)
    {
        if (!File.Exists(csprojPath))
            return ProjectType.Other;
        try
        {
            var doc = XDocument.Load(csprojPath);
            // if OutputType == Exe then it's a console app, otherwise we don't know.
            var outputType = doc.Descendants("OutputType").FirstOrDefault()?.Value ?? "";
            var sdk = doc.Root?.Attribute("Sdk")?.Value ?? "";
            if (sdk == "Microsoft.NET.Sdk")
            {
                if (outputType.Equals("Exe", StringComparison.OrdinalIgnoreCase))
                    return ProjectType.Exe;
            }
            var sdkId = sdk switch
            {
                "Microsoft.NET.Sdk.BlazorWebAssembly" => ProjectType.BlazorWasm,
                "Microsoft.NET.Sdk.Web" => ProjectType.BlazorServer,
                _ => ProjectType.Other
            };
            return sdkId;
        }
        catch
        {
            return ProjectType.Other;
        }
    }

    /// <summary>
    /// Gets the target framework (e.g. net10.0) from a csproj file.
    /// </summary>
    public static string? GetTargetFramework(string csprojPath)
    {
        if (!File.Exists(csprojPath))
            return null;
        try
        {
            var doc = XDocument.Load(csprojPath);
            return doc.Descendants("TargetFramework").FirstOrDefault()?.Value;
        }
        catch
        {
            return null;
        }
    }
}
public class ProjectDetails
{
    public string Name { get; }
    public string Directory { get; }
    public string CsprojPath { get; }
    public ProjectType AppProjectType { get; }
    public string? TargetFramework { get; }
    public XDocument ProjectDocument { get; }
    public bool IsTestProject { get; }
    public string OutputType { get; }
    public List<Reference> References { get; } = new List<Reference>();
    public string PublishPath { get; }
    public string PublishExe { get; }
    public string PublishDll { get; }
    public string? ExistingPublishBinary
    {
        get
        {
            if (File.Exists(PublishExe)) return PublishExe;
            if (File.Exists(PublishDll)) return PublishDll;
            return null;
        }
    }
    public string WwwRoot{ get; }
    public ProjectDetails(string csprojPath)
    {
        Name = Path.GetFileNameWithoutExtension(csprojPath);
        CsprojPath = csprojPath;
        Directory = Path.GetDirectoryName(csprojPath) ?? "";
        AppProjectType = ProjectDiscovery.GetAppProjectType(csprojPath);
        TargetFramework = ProjectDiscovery.GetTargetFramework(csprojPath);
        ProjectDocument = XDocument.Load(csprojPath);
        OutputType = ProjectDocument.Descendants("OutputType").FirstOrDefault()?.Value ?? "";
        var testable = ProjectDocument.Descendants("PlaywrightMultiTest").FirstOrDefault()?.Value;
        IsTestProject = testable != null;

        // enumerate references using both PackageReference and ProjectReference
        References.AddRange(ProjectDocument.Descendants("PackageReference").Select(r => new PackageReference
        {
            Include = r.Attribute("Include")?.Value ?? "",
            Version = r.Attribute("Version")?.Value ?? ""
        }));
        References.AddRange(ProjectDocument.Descendants("ProjectReference").Select(r => new Projectreference
        {
            Include = r.Attribute("Include")?.Value ?? ""
        }));
        PublishPath = Path.GetFullPath(Path.Combine(Directory, $"bin/Release/{TargetFramework}/publish"));

        WwwRoot = Path.Combine(PublishPath, "wwwroot");
        PublishExe = Path.Combine(PublishPath, $"{Name}.exe");
        PublishDll = Path.Combine(PublishPath, $"{Name}.dll");

        if (IsTestProject)
        {
            var art = true;
        }
        var nmt = true;
    }
}
public class Reference
{
    public string Include { get; set; }
}
public class PackageReference : Reference
{
    public string Version { get; set; }
}
public class Projectreference : Reference
{

}
