using SpawnDev.ILGPU.ML.DemoConsole;
using SpawnDev.UnitTesting;
using System.Reflection;
using System.Text.Json;

// Auto-flush stdout so PlaywrightMultiTest sees output immediately
Console.SetOut(new StreamWriter(Console.OpenStandardOutput()) { AutoFlush = true });

// Investigation diagnostic (NOT a PMT test): can this engine run a given ONNX model at all?
//
// 🔴 WHY IT EXISTS. "Should we replace ZipVoice with something faster" is not answerable by opinion. The
// engine either has the operators a candidate model needs or it does not, and that is a fact readable
// from the file in a second - as opposed to discovering it partway through a port. Reports the ops used,
// which are unsupported, and the node count, which is what actually costs time in a browser (MEASURED
// ~1 ms per node of host orchestration, so node count IS the speed estimate).
//
//   dotnet run --project SpawnDev.ILGPU.ML.DemoConsole -- OPCHECK <path-to.onnx>
if (args.Length > 1 && args[0] == "OPCHECK")
{
    var path = args[1];
    if (!File.Exists(path)) { Console.WriteLine($"OPCHECK: no such file: {path}"); return 1; }
    var bytes = await File.ReadAllBytesAsync(path);
    Console.WriteLine($"OPCHECK {path} ({bytes.Length / 1048576.0:F1} MB)");
    var inspection = SpawnDev.ILGPU.ML.Onnx.ModelInspectorHelper.Inspect(bytes);
    var compat = SpawnDev.ILGPU.ML.Onnx.ModelInspectorHelper.CheckCompatibility(bytes);
    Console.WriteLine($"  graph      : {inspection.GraphName}");
    Console.WriteLine($"  producer   : {inspection.ProducerName} {inspection.ProducerVersion}");
    Console.WriteLine($"  opset      : {inspection.OpsetVersion}");
    Console.WriteLine($"  NODES      : {inspection.NodeCount}   <- ~1 ms each of host time in a browser");
    Console.WriteLine($"  ops used   : {compat.TotalOpsUsed}");
    Console.WriteLine($"  supported  : {compat.SupportedOps.Length} ({compat.CompatibilityPercent:F1}%)");
    Console.WriteLine(compat.UnsupportedOps.Length == 0
        ? "  UNSUPPORTED: none - this engine can run every operator in the graph"
        : $"  UNSUPPORTED: {string.Join(", ", compat.UnsupportedOps)}");
    // The signature is what a port actually hinges on: operator coverage says the graph can RUN, the
    // inputs say what has to be built to feed it (a phonemizer, a style vector, a tokenizer).
    foreach (var t in inspection.Inputs)
        Console.WriteLine($"  IN   {t.Name} : {t.DataType} [{string.Join(",", t.Shape)}]");
    foreach (var t in inspection.Outputs)
        Console.WriteLine($"  OUT  {t.Name} : {t.DataType} [{string.Join(",", t.Shape)}]");
    return 0;
}

// Investigation diagnostic (NOT a PMT test): does SpawnDev.Phonemizer's IPA actually land on Kokoro's
// vocabulary?
//
// 🔴 THE QUESTION THAT DECIDES THE PORT. Operator coverage says the graph can run and the hub says the
// weights arrive; neither says the FRONT END fits. Kokoro has 115 phoneme tokens and our phonemizer emits
// IPA of its own devising - if the two alphabets disagree, every disagreement is a sound the model cannot
// say, and the result is audio that is subtly wrong rather than audio that fails. A drop rate is the
// cheapest possible answer and it needs no GPU, no model and no network.
//
//   dotnet run --project SpawnDev.ILGPU.ML.DemoConsole -- KOKOROFIT ["some sentence"]
if (args.Length > 0 && args[0] == "KOKOROFIT")
{
    var phonemizer = SpawnDev.Phonemizer.EmbeddedData.CreatePhonemizer();
    var vocab = SpawnDev.ILGPU.ML.Pipelines.KokoroTokenizer.Vocabulary;
    Console.WriteLine($"KOKOROFIT: vocabulary has {vocab.Count} symbols");

    string[] samples = args.Length > 1
        ? new[] { args[1] }
        : new[]
        {
            "The capital of France is Paris.",
            "She waited for 2 more minutes, then left without saying anything.",
            "Hello! How are you today? I hope it's going well.",
            "Roughly seventy-three percent of the measurements agreed.",
            "A quick brown fox jumps over the lazy dog.",
        };

    var missing = new SortedDictionary<string, int>(StringComparer.Ordinal);
    int totalSymbols = 0, totalDropped = 0;
    foreach (var text in samples)
    {
        var symbols = phonemizer.ToSymbols(text);
        var (tokens, dropped) = SpawnDev.ILGPU.ML.Pipelines.KokoroTokenizer.Encode(symbols);
        totalSymbols += symbols.Count;
        totalDropped += dropped;
        foreach (var s in symbols)
            if (!vocab.TryGetId(s, out _))
                missing[s] = missing.TryGetValue(s, out var n) ? n + 1 : 1;
        Console.WriteLine($"  \"{text}\"");
        Console.WriteLine($"    {symbols.Count} symbols -> {tokens.Length} tokens (incl. 2 pad), "
                        + $"{dropped} dropped");
    }
    var fit = totalSymbols == 0 ? 100 : (totalSymbols - totalDropped) * 100.0 / totalSymbols;
    Console.WriteLine($"KOKOROFIT: {totalSymbols - totalDropped}/{totalSymbols} symbols encodable ({fit:F1}%)");
    Console.WriteLine(missing.Count == 0
        ? "KOKOROFIT: every phoneme our frontend emits has a Kokoro token."
        : "KOKOROFIT: UNMAPPED -> " + string.Join(", ", missing.Select(kv => $"'{kv.Key}' x{kv.Value}")));
    return 0;
}

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
