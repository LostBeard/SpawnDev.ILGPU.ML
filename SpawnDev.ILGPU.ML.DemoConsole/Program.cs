using SpawnDev.ILGPU.ML.DemoConsole;
using SpawnDev.UnitTesting;
using System.Reflection;
using System.Text.Json;

// Auto-flush stdout so PlaywrightMultiTest sees output immediately
Console.SetOut(new StreamWriter(Console.OpenStandardOutput()) { AutoFlush = true });

// Investigation diagnostic (NOT a PMT-substitute test runner): CPU-vs-CUDA per-node
// bisection for the CPU-backend style-transfer correctness bug.
if (args.Length > 0 && args[0] == "STYLEBISECT")
{
    await StyleBisect.Run(args);
    return 0;
}

// Investigation diagnostic (NOT a PMT test): tight-loop repro for the intermittent CPU-backend
// non-determinism in GGUFDecodeKVCache. Discriminates which path (full-recompute shared kernels
// vs decode-specific) is non-deterministic. Usage: KVRACE [iters] [CPU|Cuda|OpenCL]
if (args.Length > 0 && args[0] == "KVRACE")
{
    int iters = args.Length > 1 && int.TryParse(args[1], out var n) ? n : 200;
    string backend = args.Length > 2 ? args[2] : "CPU";
    SpawnDev.ILGPU.ML.Demo.Shared.UnitTests.MLTestBase harness = backend switch
    {
        "Cuda" => new SpawnDev.ILGPU.ML.DemoConsole.UnitTests.CudaTests(),
        "OpenCL" => new SpawnDev.ILGPU.ML.DemoConsole.UnitTests.OpenCLTests(),
        _ => new SpawnDev.ILGPU.ML.DemoConsole.UnitTests.CPUTests(),
    };
    await harness.DiagnoseKVDecodeRace(iters);
    return 0;
}

// Investigation diagnostic (NOT a PMT test): isolates the CPU shared-memory tree reduction (the GEMV
// mechanism) and stresses it for determinism — the CPU analog of the Wasm stale-read visibility bug.
// Usage: SHMEMRACE [reps] [CPU|Cuda|OpenCL]
if (args.Length > 0 && args[0] == "SHMEMRACE")
{
    int reps = args.Length > 1 && int.TryParse(args[1], out var n) ? n : 2000;
    string backend = args.Length > 2 ? args[2] : "CPU";
    SpawnDev.ILGPU.ML.Demo.Shared.UnitTests.MLTestBase harness = backend switch
    {
        "Cuda" => new SpawnDev.ILGPU.ML.DemoConsole.UnitTests.CudaTests(),
        "OpenCL" => new SpawnDev.ILGPU.ML.DemoConsole.UnitTests.OpenCLTests(),
        _ => new SpawnDev.ILGPU.ML.DemoConsole.UnitTests.CPUTests(),
    };
    await harness.DiagnoseSharedMemReduction(reps);
    return 0;
}

// Investigation diagnostic (NOT a PMT test): runs the REAL committed
// GGUFDecodeKVCache_IncrementalMatchesFullRecompute test method directly on the chosen backend in a
// plain console (null SynchronizationContext) with wall-clock timing + PID print. If it blocks here
// too, the CPU hang is a genuine deadlock/block, NOT an NUnit-sync-context artifact. Capture the hung
// managed stacks externally: dotnet-stack report -p <PID>. Usage: KVTEST [CPU|Cuda|OpenCL]
if (args.Length > 0 && args[0] == "KVTEST")
{
    string backend = args.Length > 1 ? args[1] : "CPU";
    SpawnDev.ILGPU.ML.Demo.Shared.UnitTests.MLTestBase harness = backend switch
    {
        "Cuda" => new SpawnDev.ILGPU.ML.DemoConsole.UnitTests.CudaTests(),
        "OpenCL" => new SpawnDev.ILGPU.ML.DemoConsole.UnitTests.OpenCLTests(),
        _ => new SpawnDev.ILGPU.ML.DemoConsole.UnitTests.CPUTests(),
    };
    Console.WriteLine($"[KVTEST:{backend}] PID={Environment.ProcessId} running GGUFDecodeKVCache_IncrementalMatchesFullRecompute (real committed test)...");
    var sw = System.Diagnostics.Stopwatch.StartNew();
    await harness.GGUFDecodeKVCache_IncrementalMatchesFullRecompute();
    Console.WriteLine($"[KVTEST:{backend}] COMPLETED OK in {sw.ElapsedMilliseconds} ms");
    return 0;
}

// Catch ILGPU assertion failures (CPU backend bounds checks) that would
// otherwise show "unknown hard error" dialogs and kill the process.
// Write a proper TEST: JSON line so PlaywrightMultiTest captures the error.
AppDomain.CurrentDomain.UnhandledException += (_, e) =>
{
    var errMsg = e.ExceptionObject?.ToString() ?? "Unknown fatal error";
    if (errMsg.Length > 500) errMsg = errMsg[..500];
    var testName = args.Length > 0 ? args[0] : "Unknown";
    var parts = testName.Split('.');
    var json = JsonSerializer.Serialize(new
    {
        TestName = testName,
        TestTypeName = parts.Length > 0 ? parts[0] : testName,
        TestMethodName = parts.Length > 1 ? parts[1] : testName,
        ResultText = "Error",
        Result = 1,
        State = 2,
        Duration = 0,
        Error = errMsg,
        StackTrace = ""
    });
    Console.WriteLine($"TEST: {json}");
    Console.Out.Flush();
    Environment.Exit(2);
};

try
{
    await ConsoleRunner.Run(args);
}
catch (Exception ex)
{
    Console.Error.WriteLine(ex);
    return 1;
}
return 0;
