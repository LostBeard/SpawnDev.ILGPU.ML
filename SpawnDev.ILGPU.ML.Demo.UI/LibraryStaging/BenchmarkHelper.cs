namespace SpawnDev.ILGPU.ML.Demo.UI.Services;

/// <summary>
/// Backend benchmark helper for the "Race Mode" feature.
/// Runs the same inference on all available backends and compares timing.
/// Generates shareable results text.
/// </summary>
public class BenchmarkHelper
{
    /// <summary>
    /// Format race results as shareable text for clipboard.
    /// </summary>
    public static string FormatResults(string modelName, List<BenchmarkResult> results)
    {
        var lines = new List<string>
        {
            $"SpawnDev.ILGPU.ML — Backend Showdown ({modelName})",
            $"Date: {DateTime.UtcNow:yyyy-MM-dd HH:mm} UTC",
            ""
        };

        var sorted = results.OrderBy(r => r.InferenceMs).ToList();
        int rank = 1;
        foreach (var r in sorted)
        {
            string medal = rank switch { 1 => "1st", 2 => "2nd", 3 => "3rd", _ => $"{rank}th" };
            string speedup = rank > 1 ? $" ({sorted[0].InferenceMs / r.InferenceMs:F1}x slower)" : " (fastest)";
            lines.Add($"  {medal}: {r.BackendName} — {r.InferenceMs:F1}ms{speedup}");
            if (!string.IsNullOrEmpty(r.DeviceName))
                lines.Add($"        Device: {r.DeviceName}");
            rank++;
        }

        lines.Add("");
        lines.Add("Powered by SpawnDev.ILGPU — GPU Compute Everywhere");
        lines.Add("https://github.com/LostBeard/SpawnDev.ILGPU.ML");

        return string.Join("\n", lines);
    }

    /// <summary>
    /// Format results as a compact one-liner for social media.
    /// </summary>
    public static string FormatOneLiner(string modelName, List<BenchmarkResult> results)
    {
        var sorted = results.OrderBy(r => r.InferenceMs).ToList();
        var parts = sorted.Select(r => $"{r.BackendName}: {r.InferenceMs:F0}ms");
        return $"SpawnDev.ILGPU.ML {modelName} — {string.Join(" | ", parts)} — all from C# in the browser!";
    }

    /// <summary>
    /// Calculate speedup ratios relative to the fastest backend.
    /// </summary>
    public static List<BenchmarkResult> WithSpeedups(List<BenchmarkResult> results)
    {
        if (results.Count == 0) return results;

        var fastest = results.Min(r => r.InferenceMs);
        foreach (var r in results)
        {
            r.SpeedupVsFastest = r.InferenceMs / fastest;
            r.RelativeWidth = fastest / r.InferenceMs; // For bar chart (1.0 = full width)
        }

        return results.OrderBy(r => r.InferenceMs).ToList();
    }
}

public class BenchmarkResult
{
    public string BackendName { get; set; } = "";
    public string? DeviceName { get; set; }
    public double InferenceMs { get; set; }
    public double ModelLoadMs { get; set; }
    public string? TopPrediction { get; set; }
    public float TopConfidence { get; set; }

    /// <summary>How many times slower than the fastest backend (1.0 = fastest).</summary>
    public double SpeedupVsFastest { get; set; } = 1.0;

    /// <summary>Relative width for bar chart (1.0 = full width for fastest, smaller for slower).</summary>
    public double RelativeWidth { get; set; } = 1.0;
}
