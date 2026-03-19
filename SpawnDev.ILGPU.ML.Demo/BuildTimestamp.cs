namespace SpawnDev.ILGPU.ML.Demo;

/// <summary>
/// Build timestamp printed to browser console on app startup.
/// Uses the assembly's last write time as a reliable build indicator.
/// </summary>
internal static class BuildTimestamp
{
    public static readonly string Value = GetBuildTimestamp();

    private static string GetBuildTimestamp()
    {
        try
        {
            // In Blazor WASM, assembly location isn't available, but we can use
            // the assembly's build date from the informational version + current UTC
            var assembly = typeof(BuildTimestamp).Assembly;
            var version = assembly.GetName().Version;
            var infoVersion = assembly
                .GetCustomAttributes(typeof(System.Reflection.AssemblyInformationalVersionAttribute), false)
                .OfType<System.Reflection.AssemblyInformationalVersionAttribute>()
                .FirstOrDefault()?.InformationalVersion;

            // Check for our injected BuildTimestamp metadata
            var metadata = assembly
                .GetCustomAttributes(typeof(System.Reflection.AssemblyMetadataAttribute), false)
                .OfType<System.Reflection.AssemblyMetadataAttribute>();
            foreach (var meta in metadata)
            {
                if (meta.Key == "BuildTimestamp")
                    return meta.Value ?? "unknown";
            }

            return infoVersion ?? version?.ToString() ?? "unknown";
        }
        catch
        {
            return "unknown";
        }
    }
}
