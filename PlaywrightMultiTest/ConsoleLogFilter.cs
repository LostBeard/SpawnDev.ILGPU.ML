/// <summary>
/// PMT_CONSOLE_LOG: which console lines of a test reach the run log (browser lanes: console.log; desktop lanes: the subprocess's stdout).
/// Unset = none; "1" or "*" = all; otherwise a COMMA-SEPARATED list of substrings, a line kept when it contains ANY of them (case-insensitive),
/// the same list form as PMT_FILTER.
/// </summary>
/// <remarks>
/// 2026-11-27: PMT_CONSOLE_LOG="first opens,VideoProgress" kept nothing. It was one substring, matched whole, while PMT_FILTER next to it
/// took a comma list; the run printed no line and that read as "the test printed nothing".
/// </remarks>
public static class ConsoleLogFilter
{
    private static readonly string? Raw = Environment.GetEnvironmentVariable("PMT_CONSOLE_LOG");
    private static readonly string[] Terms = string.IsNullOrEmpty(Raw) ? Array.Empty<string>()
        : Raw.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);

    /// <summary>True when any line is to be kept (the variable is set).</summary>
    public static bool Enabled => !string.IsNullOrEmpty(Raw);

    /// <summary>True when every line is kept ("1" or "*").</summary>
    public static bool All => Raw is "1" or "*";

    /// <summary>Whether <paramref name="line"/> is kept.</summary>
    public static bool Matches(string? line)
    {
        if (!Enabled || line == null) return false;
        if (All) return true;
        foreach (var t in Terms) if (line.Contains(t, StringComparison.OrdinalIgnoreCase)) return true;
        return false;
    }
}
