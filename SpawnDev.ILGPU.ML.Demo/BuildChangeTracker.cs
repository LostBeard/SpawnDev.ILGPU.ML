using System.Reflection;
using System.Text.Json;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;

namespace SpawnDev.ILGPU.ML.Demo;

/// <summary>
/// Tracks build changes across browser sessions using localStorage.
/// Prints what changed since the last run to the browser console.
/// </summary>
internal static class BuildChangeTracker
{
    private const string StorageKey = "spawndev_ilgpu_ml_build_state";

    public static void PrintDiff(BlazorJSRuntime js)
    {
        try
        {
            var current = GetCurrentBuildState();
            var previousJson = ReadFromStorage(js);
            var previous = previousJson != null
                ? JsonSerializer.Deserialize<BuildState>(previousJson)
                : null;

            if (previous == null)
            {
                Console.WriteLine($"[Build] First run on this browser — no previous build data");
            }
            else if (previous.BuildTimestamp == current.BuildTimestamp)
            {
                Console.WriteLine($"[Build] Same build as last run ({current.BuildTimestamp})");
            }
            else
            {
                Console.WriteLine($"[Build] Changed since last run (was: {previous.BuildTimestamp})");
                foreach (var (name, currentVer) in current.AssemblyVersions)
                {
                    if (previous.AssemblyVersions.TryGetValue(name, out var prevVer))
                    {
                        if (prevVer != currentVer)
                            Console.WriteLine($"[Build]   {name}: {prevVer} -> {currentVer}");
                    }
                    else
                    {
                        Console.WriteLine($"[Build]   {name}: NEW ({currentVer})");
                    }
                }
            }

            // Save current state for next run
            WriteToStorage(js, JsonSerializer.Serialize(current));
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Build] Change tracking error: {ex.Message}");
        }
    }

    private static BuildState GetCurrentBuildState()
    {
        var state = new BuildState
        {
            BuildTimestamp = BuildTimestamp.Value,
            AssemblyVersions = new Dictionary<string, string>()
        };

        // Track key assemblies
        var assemblies = new[]
        {
            typeof(BuildTimestamp).Assembly,                          // Demo
            typeof(SpawnDev.ILGPU.ML.InferenceSession).Assembly,     // ILGPU.ML
        };

        foreach (var asm in assemblies)
        {
            var name = asm.GetName().Name ?? "unknown";
            var infoVer = asm.GetCustomAttributes<AssemblyInformationalVersionAttribute>()
                .FirstOrDefault()?.InformationalVersion;
            state.AssemblyVersions[name] = infoVer ?? asm.GetName().Version?.ToString() ?? "?";
        }

        return state;
    }

    private static string? ReadFromStorage(BlazorJSRuntime js)
    {
        try
        {
            using var window = js.Get<Window>("window");
            using var storage = window.LocalStorage;
            return storage.GetItem(StorageKey);
        }
        catch { return null; }
    }

    private static void WriteToStorage(BlazorJSRuntime js, string json)
    {
        try
        {
            using var window = js.Get<Window>("window");
            using var storage = window.LocalStorage;
            storage.SetItem(StorageKey, json);
        }
        catch { }
    }

    private class BuildState
    {
        public string BuildTimestamp { get; set; } = "";
        public Dictionary<string, string> AssemblyVersions { get; set; } = new();
    }
}
