using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using SpawnDev.ILGPU.ML;
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

// Investigation diagnostic (NOT a PMT test): run a one-node model built by tools/op-oracle.mjs through
// OUR engine and diff against onnxruntime's answer for the identical inputs.
//
// 🔴 THE GENERAL FORM OF THE ONLY QUESTION THAT MATTERS while porting a model: does this operator, as WE
// implement it, agree with the spec? Our own unit tests compare against a CPU reference we wrote, which
// shares any misreading. This compares against the reference implementation, end to end through our
// shape inference, our compiler and our kernel - the whole path a real graph takes.
//
//   dotnet run --project SpawnDev.ILGPU.ML.DemoConsole -- ONNXDIFF <oracle-dir> [CPU|CUDA]
if (args.Length > 1 && args[0] == "ONNXDIFF")
{
    var odir = args[1];
    var obackend = args.Length > 2 ? args[2].ToUpperInvariant() : "CPU";
    static float[] LoadF32(string p)
    {
        var b = File.ReadAllBytes(p);
        var f = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, f, 0, b.Length);
        return f;
    }
    using var ometa = JsonDocument.Parse(await File.ReadAllTextAsync(Path.Combine(odir, "meta.json")));
    var oroot = ometa.RootElement;
    var oexpected = LoadF32(Path.Combine(odir, "expected.f32"));

    ILGPU.Context? octx = null;
    ILGPU.Runtime.Accelerator? oacc = null;
    try
    {
        if (obackend == "CUDA")
        {
            octx = MLContext.Create().ToContext();
            var od = octx.GetCudaDevices();
            if (od.Count == 0) { octx.Dispose(); octx = null; } else oacc = od[0].CreateCudaAccelerator(octx);
        }
        if (oacc == null) { octx ??= MLContext.CreateContext(); oacc = octx.CreateCPUAccelerator(0); }

        // ⚠️ CreateFromOnnx, not CreateFromFile: the format sniffer mis-detects a MINIMAL one-node model as
        // CoreML and dies in the wrong parser ("Unknown wire type: 3"). The file is known-ONNX here, so
        // there is nothing to detect - and a sniffer that is wrong on small files is its own bug, not
        // something this diagnostic should route around silently.
        using var osession = InferenceSession.CreateFromOnnx(oacc,
            await File.ReadAllBytesAsync(Path.Combine(odir, "model.onnx")));
        var oinputs = new Dictionary<string, SpawnDev.ILGPU.ML.Tensors.Tensor>();
        var obufs = new List<IDisposable>();
        foreach (var f in oroot.GetProperty("feeds").EnumerateArray())
        {
            var fname = f.GetProperty("name").GetString()!;
            var fdims = f.GetProperty("dims").EnumerateArray().Select(d => d.GetInt32()).ToArray();
            var fdata = LoadF32(Path.Combine(odir, $"{fname}.f32"));
            var fbuf = oacc.Allocate1D<float>(fdata.Length);
            fbuf.View.CopyFromCPU(fdata);
            obufs.Add(fbuf);
            oinputs[fname] = new SpawnDev.ILGPU.ML.Tensors.Tensor(fbuf.View, fdims);
        }
        var oout = await osession.RunAsync(oinputs);
        var otensor = oout[osession.OutputNames[0]];
        var ogot = new float[otensor.ElementCount];
        otensor.Data.SubView(0, ogot.Length).CopyToCPU(ogot);
        foreach (var d in obufs) d.Dispose();

        var oexpDims = string.Join(",", oroot.GetProperty("outputDims").EnumerateArray().Select(d => d.GetInt32()));
        Console.WriteLine($"ONNXDIFF {oroot.GetProperty("op").GetString()} on {oacc.AcceleratorType}: "
                        + $"ours [{string.Join(",", otensor.Shape)}] ({ogot.Length}) vs ORT [{oexpDims}] ({oexpected.Length})");
        if (ogot.Length != oexpected.Length)
        {
            Console.WriteLine("ONNXDIFF MISMATCH - different element counts, so this is a SHAPE bug, not arithmetic");
            return 1;
        }
        double omax = 0; int oworst = -1;
        for (var i = 0; i < ogot.Length; i++)
        {
            var d = Math.Abs(ogot[i] - oexpected[i]);
            if (d > omax) { omax = d; oworst = i; }
        }
        Console.WriteLine($"ONNXDIFF ours first 8: {string.Join(" ", ogot.Take(8).Select(v => v.ToString("0.#####")))}");
        Console.WriteLine($"ONNXDIFF ORT  first 8: {string.Join(" ", oexpected.Take(8).Select(v => v.ToString("0.#####")))}");
        Console.WriteLine($"ONNXDIFF max|diff| {omax:E3} at index {oworst}");
        Console.WriteLine(omax < 1e-4 ? "ONNXDIFF MATCH" : "ONNXDIFF MISMATCH");
        return omax < 1e-4 ? 0 : 1;
    }
    finally { oacc?.Dispose(); octx?.Dispose(); }
}

// Investigation diagnostic (NOT a PMT test): run OUR ConvTranspose1D on the exact inputs
// tools/op-oracle.mjs gave onnxruntime, and diff.
//
// 🔴 WHY IT IS NOT ENOUGH TO HAVE UNIT TESTS. MLTestBase.ConvTranspose1DTests compares this kernel against
// a CPU reference I wrote from the same reading of the spec - so a misread weight layout would satisfy
// both and look green. ORT is the independent authority.
//
//   dotnet run --project SpawnDev.ILGPU.ML.DemoConsole -- CT1DDIFF <oracle-dir> <inC> <inL> <outC> <kL> <stride>
if (args.Length > 6 && args[0] == "CT1DDIFF")
{
    var dir = args[1];
    int dInC = int.Parse(args[2]), dInL = int.Parse(args[3]);
    int dOutC = int.Parse(args[4]), dKL = int.Parse(args[5]), dStride = int.Parse(args[6]);
    static float[] LoadF32(string p)
    {
        var b = File.ReadAllBytes(p);
        var f = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, f, 0, b.Length);
        return f;
    }
    var dx = LoadF32(Path.Combine(dir, "x.f32"));
    var dw = LoadF32(Path.Combine(dir, "w.f32"));
    var dExp = LoadF32(Path.Combine(dir, "expected.f32"));

    using var dctx = MLContext.CreateContext();
    using var dacc = dctx.CreateCPUAccelerator(0);
    using var dxBuf = dacc.Allocate1D<float>(dx.Length); dxBuf.View.CopyFromCPU(dx);
    using var dwBuf = dacc.Allocate1D<float>(dw.Length); dwBuf.View.CopyFromCPU(dw);
    using var dbBuf = dacc.Allocate1D<float>(1);
    using var doBuf = dacc.Allocate1D<float>(dExp.Length);
    using var dk = new SpawnDev.ILGPU.ML.Kernels.ConvTranspose1DKernel(dacc);
    dk.Forward(dxBuf.View, dwBuf.View, dbBuf.View.SubView(0, 0), doBuf.View,
        1, dInC, dInL, dOutC, dKL, dStride, 0, 0, 1, 1);
    dacc.Synchronize();
    var dGot = new float[dExp.Length]; doBuf.View.CopyToCPU(dGot);

    double dMax = 0;
    for (var i = 0; i < Math.Min(dGot.Length, dExp.Length); i++)
        dMax = Math.Max(dMax, Math.Abs(dGot[i] - dExp[i]));
    Console.WriteLine($"CT1DDIFF ours {dGot.Length} vs ORT {dExp.Length}, max|diff| {dMax:E3}");
    Console.WriteLine($"CT1DDIFF ours first 8: {string.Join(" ", dGot.Take(8).Select(v => v.ToString("0.#####")))}");
    Console.WriteLine($"CT1DDIFF ORT  first 8: {string.Join(" ", dExp.Take(8).Select(v => v.ToString("0.#####")))}");
    Console.WriteLine(dMax < 1e-4 ? "CT1DDIFF MATCH" : "CT1DDIFF MISMATCH");
    return dMax < 1e-4 ? 0 : 1;
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

// Investigation diagnostic (NOT a PMT test): actually SPEAK with Kokoro, and time it.
//
// 🔴 EVERYTHING ELSE ABOUT THIS PORT IS STATIC ANALYSIS. Operator coverage, node count, phoneme fit and
// the style-table layout are all facts read off files. None of them proves the thing produces speech, and
// none of them produces a realtime factor - the number that actually decides whether a voice can stream.
// This runs the graph end to end and writes a WAV, so the claim is audible rather than argued.
//
// ⚠️ DESKTOP, on CUDA or CPU. The browser pays ~1 ms per node of HOST orchestration that a desktop run
// does not, so the RTF here is a floor, not the number a user gets. It is still the right first
// measurement: if it were slow HERE the port would be dead.
//
//   dotnet run --project SpawnDev.ILGPU.ML.DemoConsole -- KOKOROSPEAK ["text"] [voice] [CUDA|CPU]
if (args.Length > 0 && args[0] == "KOKOROSPEAK")
{
    var text = args.Length > 1 ? args[1] : "The capital of France is Paris.";
    var voiceName = args.Length > 2 ? args[2] : "af_heart";
    var backend = args.Length > 3 ? args[3].ToUpperInvariant() : "CUDA";
    const string HubBase = "https://hub.spawndev.com:44365/hf/onnx-community/Kokoro-82M-v1.0-ONNX";

    using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
    Console.WriteLine($"KOKOROSPEAK: fetching the model through the hub ({backend})");
    var modelBytes = await http.GetByteArrayAsync($"{HubBase}/onnx/model.onnx");
    var voiceBytes = await http.GetByteArrayAsync($"{HubBase}/voices/{voiceName}.bin");
    Console.WriteLine($"  model {modelBytes.Length / 1048576.0:F1} MB, voice {voiceBytes.Length} bytes");

    var pack = SpawnDev.ILGPU.ML.Pipelines.KokoroVoicePack.FromBytes(voiceName, voiceBytes);
    var phonemes = SpawnDev.Phonemizer.EmbeddedData.CreatePhonemizer().ToSymbols(text);
    Console.WriteLine($"  \"{text}\" -> {phonemes.Count} phonemes");
    // The exact token ids, so an external reference can be fed the IDENTICAL input. Comparing against a
    // reference that tokenized the text itself would compare two frontends as well as two engines.
    var (traceTokens, _) = SpawnDev.ILGPU.ML.Pipelines.KokoroTokenizer.Encode(phonemes);
    Console.WriteLine($"  TOKENS {string.Join(",", traceTokens)}");

    // KOKORO_TRACE=1 prints every node with its compile-time output shapes, which is how a shape bug deep
    // in a 2,323-node graph is localised: the failing node names its operands, and the trace says where
    // those operands' shapes were decided.
    if (Environment.GetEnvironmentVariable("KOKORO_TRACE") == "1")
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.VerboseLogging = true;

    // KOKORO_DUMP=<substring> prints the VALUES of every tensor whose name matches. Shapes localised the
    // structural bugs; the remaining ones are arithmetic, and no amount of shape tracing finds a duration
    // that came out as 1 when it should be 5.
    // KOKORO_STATS=1 prints max|value| for every node, in graph order. Finding the first node whose
    // magnitude leaves the plausible range localises an arithmetic bug the way the shape trace localised
    // the structural ones - a waveform with peak 2226 was produced by something, and it started somewhere.
    var stats = Environment.GetEnvironmentVariable("KOKORO_STATS") == "1";
    if (stats)
    {
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedOutputs = new Dictionary<string, float[]>();
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CaptureMaxElements = 256;
    }

    // KOKORO_CMP=<dir> diffs our value for every tensor tools/onnx-intermediates.mjs dumped there. Shapes
    // agreeing everywhere while the audio is wrong means the fault is arithmetic, and arithmetic is only
    // findable by comparing NUMBERS at the same point in the graph.
    var cmpDir = Environment.GetEnvironmentVariable("KOKORO_CMP");
    if (!string.IsNullOrEmpty(cmpDir))
    {
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedOutputs = new Dictionary<string, float[]>();
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CaptureMaxElements = 1 << 20;
    }

    // KOKORO_SAVE=<dir> writes every captured tensor to <dir>/<sanitised name>.f32. Comparing statistics
    // against a reference answers "how far apart are we"; only having OUR ACTUAL BYTES on disk answers
    // "is the rest of the graph right GIVEN our input" - the reference engine can be re-run with our
    // tensor spliced in, and whatever still differs after that is a defect and not accumulated rounding.
    var saveDir = Environment.GetEnvironmentVariable("KOKORO_SAVE");
    if (!string.IsNullOrEmpty(saveDir))
    {
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedOutputs ??= new Dictionary<string, float[]>();
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CaptureMaxElements = 1 << 22;
    }

    var dumpFilter = Environment.GetEnvironmentVariable("KOKORO_DUMP");
    if (!string.IsNullOrEmpty(dumpFilter))
    {
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedOutputs = new Dictionary<string, float[]>();
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CaptureMaxElements = 4096;
    }

    // KOKORO_STAGES=1 buckets every compiled node by the stage its NAME puts it in, with node counts,
    // output ELEMENT counts and per-node time.
    //
    // 🔴 WHY. The browser cost of this engine is per-node host orchestration - MEASURED at ~1.5 ms per
    // node in a worker - so a node that computes 182 floats costs the same to dispatch as one that
    // computes 54,600. That is the whole case for running part of a graph somewhere without dispatch
    // overhead (Wasm/SIMD) and part of it on the GPU: it only pays if the cheap-to-compute nodes are also
    // the MANY nodes. This says whether that is true for a given model, which is not something to assume.
    var stageBuckets = Environment.GetEnvironmentVariable("KOKORO_STAGES") == "1";
    if (stageBuckets)
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs = new Dictionary<string, double>();

    // KOKORO_SYNC_INTERVAL=<n> overrides GraphExecutor.SyncIntervalNodes for this run. Each periodic
    // drain is an async GPU round trip - MEASURED at ~78 ms in a browser worker against ~1 ms on CUDA -
    // and at the default 64 a 1,885-node graph pays ~29 of them. The byte cap
    // (MaxPendingReleaseBytes) bounds peak memory independently, so this trades only latency.
    if (int.TryParse(Environment.GetEnvironmentVariable("KOKORO_SYNC_INTERVAL"), out var ksi) && ksi > 0)
    {
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.SyncIntervalNodes = ksi;
        Console.WriteLine($"  SyncIntervalNodes = {ksi}");
    }

    ILGPU.Context? kctx = null;
    ILGPU.Runtime.Accelerator? kacc = null;
    try
    {
        if (backend == "CUDA")
        {
            kctx = MLContext.Create().ToContext();
            var devs = kctx.GetCudaDevices();
            if (devs.Count == 0) { Console.WriteLine("  no CUDA device; falling back to CPU"); kctx.Dispose(); kctx = null; }
            else kacc = devs[0].CreateCudaAccelerator(kctx);
        }
        if (kacc == null)
        {
            kctx ??= MLContext.CreateContext();
            kacc = kctx.CreateCPUAccelerator(0);
        }
        Console.WriteLine($"  accelerator: {kacc.AcceleratorType} {kacc.Name}");

        // ⚠️ The STREAM loader, which is the path a browser takes - not Create(byte[]). A console that
        // exercises the desktop-only overload leaves the browser's loader covered by nothing, and the two
        // reach different code in the session reader. Same bytes either way, so the numbers below are a
        // check on both.
        using var modelStream = new MemoryStream(modelBytes, writable: false);
        using var pipeline = await SpawnDev.ILGPU.ML.Pipelines.KokoroPipeline
            .CreateFromStreamAsync(kacc, modelStream);
        // KOKORO_CAPTURE=1 records the dispatch plan and replays it. ⚠️ Read the STATUS afterwards:
        // "requested" is not "live" - SessionGraphCapture falls through to a direct forward silently
        // whenever it cannot record, which is a speed difference that looks like nothing at all.
        pipeline.EnableGraphCapture = Environment.GetEnvironmentVariable("KOKORO_CAPTURE") == "1";
        // ⚠️ The COMPILED node count, printed every run. This engine costs ~1 ms per real node in a
        // browser, so the count IS the speed estimate - and a fusion pass that silently stops matching
        // (an exporter changes one attribute and the pattern is gone) shows up here and nowhere else.
        // ⚠️ "declined" is a legitimate outcome and is NOT a failure - it costs the dispatches and host
        // readbacks the tail would have saved, and the waveform is identical either way. But it has to be
        // VISIBLE, because a silent decline looks exactly like a working optimisation that stopped paying.
        Console.WriteLine($"  istft tail: {pipeline.TailStatus}");
        Console.WriteLine($"  compiled nodes: {pipeline.Session.NodeCount} "
                        + $"(fused-instancenorm {SpawnDev.ILGPU.ML.Graph.GraphOptimizer.LastInstanceNormFused}, "
                        + $"fused-atan2 {SpawnDev.ILGPU.ML.Graph.GraphOptimizer.LastAtan2Fused})");
        // ⚠️ TWICE. The first call compiles kernels, which is a one-off this engine pays per process and
        // which would otherwise be reported as the model's speed. The second is the steady state.
        for (var pass = 1; pass <= 2; pass++)
        {
            var audio = await pipeline.SpeakAsync(phonemes, pack);
            Console.WriteLine($"  pass {pass}: {audio.Samples.Length} samples = {audio.Seconds:F2}s of audio "
                            + $"in {audio.InferenceMs:F0} ms  ->  RTF {audio.RealtimeFactor:F2}x "
                            + $"({audio.Tokens} tokens, {audio.DroppedPhonemes} dropped)");
            if (pass == 2)
            {
                // ⚠️ The executor's own split, and the NAMES of every mid-graph readback. MEASURED in
                // the browser worker: drains + readbacks are 59% of a warm synthesis, while the residual
                // alone already matches a whole page-context run - so the round trips ARE the gap. The
                // names are a property of the GRAPH, not the backend, so CUDA names the same culprits in
                // seconds instead of minutes.
                var ex = SpawnDev.ILGPU.ML.Graph.GraphExecutor.CumulativeTotalMs;
                var rbN = SpawnDev.ILGPU.ML.Graph.GraphExecutor.CumulativeReadbackCount;
                var rbMs = SpawnDev.ILGPU.ML.Graph.GraphExecutor.CumulativeReadbackMs;
                var drN = SpawnDev.ILGPU.ML.Graph.GraphExecutor.CumulativeSyncDrainCount;
                var drMs = SpawnDev.ILGPU.ML.Graph.GraphExecutor.CumulativeSyncDrainMs;
                Console.WriteLine($"  executor {ex:F0}ms | readbacks {rbN} ({rbMs:F0}ms) | "
                                + $"drains {drN} ({drMs:F0}ms) | residual {ex - rbMs - drMs:F0}ms");
                var rbNames = SpawnDev.ILGPU.ML.Graph.GraphExecutor.LastRunReadbackNames;
                if (rbNames is { Count: > 0 })
                    Console.WriteLine($"  READBACK NODES ({rbNames.Count}): {string.Join(", ", rbNames)}");

                if (stageBuckets
                    && SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs is { } timings)
                {
                    // The stage a node belongs to, from its ONNX name. Coarse on purpose: the question is
                    // "encoder half or generator half", because that is where a placement split would cut.
                    static string Stage(string key)
                    {
                        var name = key[(key.IndexOf('_') + 1)..];
                        name = name[(name.IndexOf('_') + 1)..];
                        if (name.Contains("/generator/")) return "decoder/generator";
                        if (name.StartsWith("/decoder")) return "decoder (pre-generator)";
                        if (name.Contains("/bert")) return "encoder/bert";
                        if (name.Contains("/predictor")) return "encoder/predictor";
                        if (name.StartsWith("/encoder")) return "encoder (other)";
                        return "other";
                    }
                    var byStage = timings.GroupBy(kv => Stage(kv.Key))
                        .Select(g => (Stage: g.Key, Nodes: g.Count(), Ms: g.Sum(x => x.Value)))
                        .OrderByDescending(g => g.Nodes).ToList();
                    var totalNodes = byStage.Sum(g => g.Nodes);
                    var totalMs = byStage.Sum(g => g.Ms);
                    Console.WriteLine($"  STAGES ({totalNodes} timed nodes, {totalMs:F0} ms on this backend):");
                    foreach (var g in byStage)
                        Console.WriteLine($"    {g.Stage,-26} {g.Nodes,5} nodes ({g.Nodes * 100.0 / totalNodes,4:F1}%)"
                                        + $"  {g.Ms,8:F1} ms ({g.Ms * 100.0 / Math.Max(totalMs, 1e-9),4:F1}%)"
                                        + $"  {g.Ms / g.Nodes,6:F3} ms/node");
                    // ⚠️ The browser's cost is ~1.5 ms PER NODE regardless of size, so the node-count
                    // column - not this backend's millisecond column - is what predicts it.
                    Console.WriteLine($"    -> at the browser's ~1.5 ms/node, the node split alone implies "
                                    + string.Join(", ", byStage.Select(g => $"{g.Stage}={g.Nodes * 1.5:F0}ms")));

                    // ⭐ WHICH OPS SURVIVE TO DISPATCH - the list fusion and dispatch-elide have to shorten.
                    // The RAW graph's op census (tools/onnx-opcensus.mjs) counts what the exporter wrote;
                    // this counts what actually costs a crossing, which is a different and much shorter
                    // list. Dispatching 100 Unsqueezes to reshape a 2-element shape vector is 100 browser
                    // crossings for arithmetic the shape interpreter already did on the host.
                    var byOp = timings.GroupBy(kv =>
                        {
                            var k = kv.Key[(kv.Key.IndexOf('_') + 1)..];
                            return k[..k.IndexOf('_')];
                        })
                        .Select(g => (Op: g.Key, Nodes: g.Count(), Ms: g.Sum(x => x.Value)))
                        .OrderByDescending(g => g.Nodes).ToList();
                    Console.WriteLine($"  DISPATCHED OPS ({byOp.Count} distinct):");
                    foreach (var g in byOp.Take(20))
                        Console.WriteLine($"    {g.Op,-22} {g.Nodes,5} nodes  {g.Ms,8:F1} ms  "
                                        + $"{g.Ms / g.Nodes,6:F3} ms/node");
                    var tail20 = byOp.Skip(20).Sum(g => g.Nodes);
                    if (tail20 > 0) Console.WriteLine($"    {"(" + (byOp.Count - 20) + " more)",-22} {tail20,5} nodes");
                }

                Console.WriteLine($"  capture: requested={pipeline.EnableGraphCapture} "
                                + $"status={pipeline.CaptureStatus}");

                var peak = 0f;
                foreach (var s in audio.Samples) { var a = Math.Abs(s); if (a > peak) peak = a; }
                Console.WriteLine($"  peak amplitude {peak:F3}"
                                + (peak < 0.01f ? "  <- SILENCE: the graph ran and produced nothing audible" : ""));
                var wav = Path.GetFullPath($"kokoro-{voiceName}.wav");
                await File.WriteAllBytesAsync(wav, WavBytes(audio.Samples, audio.SampleRate));
                Console.WriteLine($"  wrote {wav}");

                // KOKORO_REF=<file.f32> compares against the onnxruntime reference sample by sample.
                // ⚠️ CORRELATION AS WELL AS ERROR. "max abs diff 2226" says the numbers differ and nothing
                // about HOW - a waveform that is the reference times a constant is a scale bug in one
                // place, while an uncorrelated one is a different computation entirely, and the two need
                // completely different searches.
                var refPath = Environment.GetEnvironmentVariable("KOKORO_REF");
                if (!string.IsNullOrEmpty(refPath) && File.Exists(refPath))
                {
                    var refBytes = await File.ReadAllBytesAsync(refPath);
                    var reference = new float[refBytes.Length / 4];
                    Buffer.BlockCopy(refBytes, 0, reference, 0, refBytes.Length);
                    Console.WriteLine($"  REF {reference.Length} samples vs ours {audio.Samples.Length}");
                    int n = Math.Min(reference.Length, audio.Samples.Length);
                    double maxAbs = 0, sumRR = 0, sumOO = 0, sumRO = 0;
                    int firstBad = -1;
                    for (int i = 0; i < n; i++)
                    {
                        double r = reference[i], o = audio.Samples[i];
                        var d = Math.Abs(r - o);
                        if (d > maxAbs) maxAbs = d;
                        if (firstBad < 0 && d > 1e-3) firstBad = i;
                        sumRR += r * r; sumOO += o * o; sumRO += r * o;
                    }
                    // The middle of the signal separately: an iSTFT's first and last frames divide by a
                    // window envelope that is near zero there, so an edge artefact is a different (and far
                    // smaller) bug than a wrong waveform. Correlation over the interior says which.
                    int lo = n / 10, hi = n - n / 10;
                    double mRR = 0, mOO = 0, mRO = 0, mMax = 0;
                    for (int i = lo; i < hi; i++)
                    {
                        double r = reference[i], o = audio.Samples[i];
                        mRR += r * r; mOO += o * o; mRO += r * o;
                        var d = Math.Abs(r - o); if (d > mMax) mMax = d;
                    }
                    Console.WriteLine($"  REF interior[{lo}..{hi}) max|diff| {mMax:F4}, correlation "
                                    + $"{(mRR > 0 && mOO > 0 ? mRO / Math.Sqrt(mRR * mOO) : 0):F4}");
                    // Correlation per tenth of the signal. A single wrong operator gives a flat profile;
                    // accumulating PHASE drift (the sine source integrates F0 with a CumSum, so any tiny
                    // difference grows without bound) decays monotonically. The two need different fixes,
                    // and one line of output tells them apart.
                    var profile = new List<string>();
                    for (int seg = 0; seg < 10; seg++)
                    {
                        int a = n * seg / 10, b = n * (seg + 1) / 10;
                        double rr = 0, oo = 0, ro = 0;
                        for (int i = a; i < b; i++)
                        {
                            double r = reference[i], o = audio.Samples[i];
                            rr += r * r; oo += o * o; ro += r * o;
                        }
                        profile.Add((rr > 0 && oo > 0 ? ro / Math.Sqrt(rr * oo) : 0).ToString("0.00"));
                    }
                    Console.WriteLine($"  REF correlation by tenth: {string.Join(" ", profile)}");
                    Console.WriteLine($"  REF first 8 ours: {string.Join(" ", audio.Samples.Take(8).Select(v => v.ToString("0.####")))}");
                    Console.WriteLine($"  REF first 8 ref : {string.Join(" ", reference.Take(8).Select(v => v.ToString("0.####")))}");
                    Console.WriteLine($"  REF mid 8 ours  : {string.Join(" ", audio.Samples.Skip(n / 2).Take(8).Select(v => v.ToString("0.####")))}");
                    Console.WriteLine($"  REF mid 8 ref   : {string.Join(" ", reference.Skip(n / 2).Take(8).Select(v => v.ToString("0.####")))}");
                    var corr = sumRR > 0 && sumOO > 0 ? sumRO / Math.Sqrt(sumRR * sumOO) : 0;
                    var scale = sumRR > 0 ? sumRO / sumRR : 0;
                    Console.WriteLine($"  REF max|diff| {maxAbs:F4}, correlation {corr:F4}, "
                                    + $"best-fit ours≈{(scale != 0 ? 1 / scale : 0):F1}x reference, "
                                    + $"first diff>1e-3 at sample {firstBad}");
                }
            }
        }
    }
    finally
    {
        if (!string.IsNullOrEmpty(cmpDir)
            && SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedOutputs is { } mine)
        {
            // Our capture keys are "<idx>_<Op>_<tensor name>"; the reference files are the tensor name with
            // the path characters replaced. Match on the sanitised tail so the two line up.
            static string Safe(string s) => new string(s.Select(c =>
                char.IsLetterOrDigit(c) || c is '.' or '_' or '-' ? c : '_').ToArray());
            var byName = new Dictionary<string, float[]>(StringComparer.Ordinal);
            foreach (var kv in mine)
            {
                var tail = kv.Key[(kv.Key.IndexOf('_') + 1)..];
                tail = tail[(tail.IndexOf('_') + 1)..];
                byName[Safe(tail)] = kv.Value;
            }
            foreach (var file in Directory.GetFiles(cmpDir, "*.f32").OrderBy(f => f))
            {
                var key = Path.GetFileNameWithoutExtension(file);
                if (!byName.TryGetValue(key, out var ours)) { Console.WriteLine($"  CMP {key}: not captured"); continue; }
                var rb = await File.ReadAllBytesAsync(file);
                var theirs = new float[rb.Length / 4];
                Buffer.BlockCopy(rb, 0, theirs, 0, rb.Length);
                var cn = Math.Min(ours.Length, theirs.Length);
                double cmax = 0, crr = 0, coo = 0, cro = 0, cdd = 0;
                for (var i = 0; i < cn; i++)
                {
                    double a = ours[i], b2 = theirs[i];
                    var dd = a - b2;
                    cmax = Math.Max(cmax, Math.Abs(dd));
                    crr += b2 * b2; coo += a * a; cro += a * b2; cdd += dd * dd;
                }
                var ccorr = crr > 0 && coo > 0 ? cro / Math.Sqrt(crr * coo) : 1;
                // ⚠️ RELATIVE RMS, not max|diff|. A max is a TAIL statistic over tens of thousands of
                // elements - it cannot separate "float32 noise amplified by an ill-conditioned normalise"
                // from "this operator is wrong". ||ours-ref|| / ||ref|| can: ~1e-6 is float32 working
                // correctly, ~1e-3 is a defect, and the FIRST node where it leaves 1e-6 is the defect.
                var rel = crr > 0 ? Math.Sqrt(cdd / crr) : 0;
                Console.WriteLine($"  CMP {key}: n={cn} (ours {ours.Length}, ref {theirs.Length}) "
                                + $"relRMS={rel:E2} max|diff|={cmax:E2} corr={ccorr:F5}");
                // The three worst elements, by INDEX and by VALUE. A summary statistic says a tensor is
                // wrong; only the values say HOW - "ours 1.5708 vs ref -1.5708" is a sign, "ours NaN" is a
                // domain, "ours 0" is a missing write, and each needs a different search.
                if (rel > 1e-4 && cn > 0)
                {
                    var worst = Enumerable.Range(0, cn)
                        .OrderByDescending(i => Math.Abs(ours[i] - theirs[i])).Take(3);
                    foreach (var i in worst)
                        Console.WriteLine($"      [{i}] ours={ours[i]:G7} ref={theirs[i]:G7}");
                }
            }
        }
        if (!string.IsNullOrEmpty(saveDir)
            && SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedOutputs is { } toSave)
        {
            Directory.CreateDirectory(saveDir);
            static string SafeName(string s2) => new string(s2.Select(c =>
                char.IsLetterOrDigit(c) || c is '.' or '_' or '-' ? c : '_').ToArray());
            var written = 0;
            foreach (var kv in toSave)
            {
                // Our capture keys are "<idx>_<Op>_<tensor name>"; strip both prefixes so the file is named
                // by the ONNX tensor, which is what the reference tooling addresses tensors by.
                var tail = kv.Key[(kv.Key.IndexOf('_') + 1)..];
                tail = tail[(tail.IndexOf('_') + 1)..];
                var bytes = new byte[kv.Value.Length * 4];
                Buffer.BlockCopy(kv.Value, 0, bytes, 0, bytes.Length);
                await File.WriteAllBytesAsync(Path.Combine(saveDir, SafeName(tail) + ".f32"), bytes);
                written++;
            }
            Console.WriteLine($"  KOKORO_SAVE wrote {written} tensors to {Path.GetFullPath(saveDir)}");
        }
        if (stats && SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedOutputs is { } captured)
        {
            foreach (var kv in captured.OrderBy(k => k.Key, StringComparer.Ordinal))
            {
                float peak = 0; var nan = false;
                foreach (var v in kv.Value)
                {
                    if (float.IsNaN(v) || float.IsInfinity(v)) { nan = true; break; }
                    var a = Math.Abs(v); if (a > peak) peak = a;
                }
                Console.WriteLine($"  STAT {kv.Key} max|v|={(nan ? "NaN/Inf" : peak.ToString("0.####"))}");
            }
        }
        if (!string.IsNullOrEmpty(dumpFilter)
            && SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedOutputs is { } dumped)
        {
            foreach (var kv in dumped.Where(k => k.Key.Contains(dumpFilter, StringComparison.OrdinalIgnoreCase))
                                     .OrderBy(k => k.Key))
            {
                var vals = kv.Value.Take(48).Select(v => v.ToString("0.###"));
                Console.WriteLine($"  DUMP {kv.Key} [{kv.Value.Length}] = {string.Join(" ", vals)}"
                                + (kv.Value.Length > 48 ? " ..." : ""));
            }
        }
        kacc?.Dispose(); kctx?.Dispose();
    }
    return 0;

    // Minimal 16-bit PCM WAV, so the result can be LISTENED TO. An amplitude check proves the graph
    // produced signal; only a human ear proves it produced the right words.
    static byte[] WavBytes(float[] samples, int rate)
    {
        var data = new byte[samples.Length * 2];
        for (var i = 0; i < samples.Length; i++)
        {
            var v = (short)Math.Clamp(samples[i] * short.MaxValue, short.MinValue, short.MaxValue);
            data[i * 2] = (byte)(v & 0xFF);
            data[i * 2 + 1] = (byte)((v >> 8) & 0xFF);
        }
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write("RIFF"u8.ToArray()); w.Write(36 + data.Length); w.Write("WAVE"u8.ToArray());
        w.Write("fmt "u8.ToArray()); w.Write(16); w.Write((short)1); w.Write((short)1);
        w.Write(rate); w.Write(rate * 2); w.Write((short)2); w.Write((short)16);
        w.Write("data"u8.ToArray()); w.Write(data.Length); w.Write(data);
        w.Flush();
        return ms.ToArray();
    }
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
