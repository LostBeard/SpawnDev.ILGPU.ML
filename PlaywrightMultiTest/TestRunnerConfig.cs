using System.Text.Json.Serialization;

namespace PlaywrightMultiTest;

/// <summary>
/// Configuration for a Playwright test project that tests an app (e.g. Blazor WASM).
/// Loaded from playwright-test.json in the test project directory.
/// </summary>
public class TestRunnerConfig
{
    public const string Filename = "playwright-test.json";
    /// <summary>
    /// Name of the app project to publish and serve (e.g. "SpawnDev.ILGPU.Demo").
    /// When set, the orchestrator will publish this project and serve it before running tests.
    /// </summary>
    [JsonPropertyName("appProject")]
    public string? AppProject { get; set; }

    /// <summary>
    /// Relative paths to the Blazor pages that host the unit test UI (e.g. ["tests"]).
    /// </summary>
    [JsonPropertyName("testPages")]
    public string[] TestPages { get; set; } = ["tests"];

    /// <summary>
    /// Loads config from playwright-test.json in the given directory, if it exists.
    /// </summary>
    public static TestRunnerConfig? LoadFrom(string fileOrDir)
    {
        if (Directory.Exists(fileOrDir))
        {
            var path = Path.Combine(fileOrDir, Filename);
            if (!File.Exists(path))
                return null;

            var json = File.ReadAllText(path);
            return System.Text.Json.JsonSerializer.Deserialize<TestRunnerConfig>(json);
        }
        else if (File.Exists(fileOrDir))
        {
            var json = File.ReadAllText(fileOrDir);
            return System.Text.Json.JsonSerializer.Deserialize<TestRunnerConfig>(json);
        }
        return null;
    }
}
