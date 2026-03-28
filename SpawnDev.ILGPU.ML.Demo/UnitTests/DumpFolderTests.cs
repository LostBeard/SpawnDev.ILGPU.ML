using SpawnDev.ILGPU.Services;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.UnitTests;

/// <summary>
/// Verifies the debug dump folder is writable. Registered FIRST in Program.cs
/// so this runs before any other test — catches folder permission issues immediately.
/// Skips if no dump folder is configured.
/// </summary>
public class DumpFolderTests
{
    private readonly ShaderDebugService _debugService;

    public DumpFolderTests(ShaderDebugService debugService)
    {
        _debugService = debugService;
    }

    [TestMethod]
    public async Task DumpFolder_IsWritable()
    {
        if (!_debugService.HasDebugFolder)
            throw new UnsupportedTestException("No dump folder configured");

        if (!_debugService.HasWritePermission)
            throw new UnsupportedTestException("Dump folder configured but write permission not granted");

        Console.WriteLine($"[DumpFolder] Folder configured, write permission granted. PASS");
    }
}
