// Parakeet TDT bring-up and regression gate on the desktop (CUDA preferred), against the onnxruntime oracle.
//
//   node tools/parakeet-oracle.mjs <modelDir> <wav> <oracleDir>          # the reference, once per clip
//   dotnet run --project tools/parakeet-harness -c Release -- <modelDir> <wav> [oracleDir] [--half] [--int8] [--runs N]
//
// Reports: which bytes were loaded, mel and encoder parity against the oracle's dumps (max abs error relative to
// the oracle's own scale), the transcript next to the oracle's, and warm timings over N runs. A transcript that
// "looks right" is not a pass: the stages are compared, so a divergence is located, not guessed.
using System.Diagnostics;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Tensors;

if (args.Contains("--sync")) SpawnDev.ILGPU.ML.Graph.GraphExecutor.PerOpSync = true;   // device sync after every node: an async fault is reported at ITS node
if (args.Contains("--no-reg-attn")) SpawnDev.ILGPU.ML.Kernels.FusedAttentionKernel.EnableRegisterAttention = false;   // A/B switch
var positional = args.Where(a => !a.StartsWith("--")).ToArray();
if (positional.Length < 2) { Console.Error.WriteLine("usage: <modelDir> <wav> [oracleDir] [--half] [--int8] [--runs N]"); return 2; }
var modelDir = positional[0];
var wavPath = positional[1];
var oracleDir = positional.Length > 2 ? positional[2] : null;
var half = args.Contains("--half");
var int8 = args.Contains("--int8");
var runsArg = Array.IndexOf(args, "--runs");
var runs = runsArg >= 0 ? int.Parse(args[runsArg + 1]) : 5;

var samples = WavDecoder.DecodeWavFile(File.ReadAllBytes(wavPath)) ?? throw new InvalidDataException("not a decodable WAV");
Console.WriteLine($"wav      : {Path.GetFileName(wavPath)} {samples.Length / 16000.0:F2} s (16 kHz mono assumed)");

var builder = MLContext.Create();
await builder.AllAcceleratorsAsync();
var context = builder.ToContext();
var accelerator = await context.CreatePreferredAcceleratorAsync() ?? throw new Exception("no accelerator");
Console.WriteLine($"device   : {accelerator.AcceleratorType} {accelerator.Name}");

async Task<InferenceSession> Load(string file, bool lowPrecisionWeights)
{
    var path = Path.Combine(modelDir, file);
    var data = path + ".data";
    var graph = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.Asynchronous | FileOptions.RandomAccess);
    var external = File.Exists(data) ? new FileStream(data, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.Asynchronous | FileOptions.RandomAccess) : null;
    var sw = Stopwatch.StartNew();
    var s = await InferenceSession.CreateFromStreamAsync(accelerator, graph, externalDataStream: external,
        weightStorage: lowPrecisionWeights ? WeightStorage.Half : WeightStorage.Source);
    Console.WriteLine($"loaded   : {file} {new FileInfo(path).Length / 1e6:F1} MB (+{(external == null ? 0 : new FileInfo(data).Length) / 1e6:F1} MB data) in {sw.ElapsedMilliseconds} ms, {s.NodeCount} nodes");
    return s;
}

// --probe-dir <dir> <names,...>: the cuts made by tools/onnx-bisect-audio.mjs, in graph order. Each is run through
// OUR engine on the same WAV and compared with onnxruntime's value; the first that disagrees is where to look.
var probeAt = Array.IndexOf(args, "--probe-dir");
if (probeAt >= 0)
{
    var dir = args[probeAt + 1];
    var namesArg = args[probeAt + 2];
    if (namesArg.StartsWith('@')) namesArg = File.ReadAllText(namesArg[1..]);   // "@path": the cuts.txt onnx-bisect-audio wrote
    var names = namesArg.Split(',', StringSplitOptions.RemoveEmptyEntries);
    string? firstBad = null;
    foreach (var name in names)
    {
        var file = name.Replace("/", "__");
        var cutPath = Path.Combine(dir, file + ".onnx");
        if (!File.Exists(cutPath)) { Console.WriteLine($"{name,-24} (no cut)"); continue; }
        // --no-opt: the same cut with ALL graph optimisation (fusions) off - the A/B that changes only that.
        using var cut = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(cutPath), enableOptimization: !args.Contains("--no-opt"));
        using var pcm = accelerator.Allocate1D(samples);
        using var len = accelerator.Allocate1D(new[] { (float)samples.Length });
        var feeds = new Dictionary<string, Tensor>();
        // Speaker encoders take a fixed 6 s window: repeat-fill short audio, centre-crop long (as SpeakerEmbeddingPipeline).
        var window = new float[96000];
        if (samples.Length >= 96000) Array.Copy(samples, (samples.Length - 96000) / 2, window, 0, 96000);
        else for (int i = 0; i < 96000; i++) window[i] = samples[i % samples.Length];
        using var audioWindow = accelerator.Allocate1D(window);
        foreach (var input in cut.InputNames)
            feeds[input] = input == "audio" ? new Tensor(audioWindow.View, new[] { 1, 96000 })
                : input == "waveforms" ? new Tensor(pcm.View, new[] { 1, samples.Length }) : new Tensor(len.View, new[] { 1 });
        var outs = await cut.RunAsync(feeds);
        if (!outs.TryGetValue(cut.OutputNames[0], out var t))
        {
            // The engine fused this tensor away (it was an internal edge of a fused op): say so and carry on.
            Console.WriteLine($"{name,-24} (not produced: fused into another op; outputs were [{string.Join(", ", outs.Keys)}])");
            continue;
        }
        var got = new float[t.Data.Length];
        t.Data.BaseView.CopyToCPU(got);
        var want = new float[File.ReadAllBytes(Path.Combine(dir, file + ".f32")).Length / 4];
        Buffer.BlockCopy(File.ReadAllBytes(Path.Combine(dir, file + ".f32")), 0, want, 0, want.Length * 4);
        int n = Math.Min(got.Length, want.Length);
        double worst = 0, peak = 0; int at = -1;
        for (int i = 0; i < n; i++) { peak = Math.Max(peak, Math.Abs(want[i])); var d = Math.Abs(got[i] - want[i]); if (d > worst) { worst = d; at = i; } }
        var rel = worst / Math.Max(peak, 1e-30);
        var ok = got.Length == want.Length && rel < 1e-3;
        if (!ok) firstBad ??= name;
        Console.WriteLine($"{name,-24} shape [{string.Join(",", t.Shape)}] n {got.Length}/{want.Length}  max|err| {worst:G3} = {rel:P4} of peak {peak:G4} (at {at})  {(ok ? "ok" : "DIVERGES")}");
    }
    Console.WriteLine(firstBad == null ? "all cuts agree" : $"FIRST DIVERGENCE: {firstBad}");
    return firstBad == null ? 0 : 1;
}

var encoderFile = int8 ? "encoder-model.int8.onnx" : "encoder-model.onnx";
var pre = args.Contains("--pre-bytes")
    ? InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(Path.Combine(modelDir, "nemo128.onnx")))
    : await Load("nemo128.onnx", false);
if (args.Contains("--pre-only"))
{
    // Stage isolation: only the log-mel graph, with the engine's own per-node shape log.
    InferenceSession.VerboseLogging = Environment.GetEnvironmentVariable("VERBOSE") == "1";
    using var pcm = accelerator.Allocate1D(samples);
    using var len = accelerator.Allocate1D(new[] { (float)samples.Length });
    var outs = await pre.RunAsync(new Dictionary<string, Tensor>
    {
        [pre.InputNames[0]] = new Tensor(pcm.View, new[] { 1, samples.Length }),
        [pre.InputNames[1]] = new Tensor(len.View, new[] { 1 }),
    });
    foreach (var (name, t) in outs)
    {
        var host = new float[t.Data.Length];
        t.Data.BaseView.CopyToCPU(host);
        Console.WriteLine($"pre out  : {name} shape [{string.Join(",", t.Shape)}] first {string.Join(" ", host.Take(4).Select(v => v.ToString("G5")))}");
        if (oracleDir != null && name == pre.OutputNames[0])
        {
            var want = new float[File.ReadAllBytes(Path.Combine(oracleDir, "features.f32")).Length / 4];
            Buffer.BlockCopy(File.ReadAllBytes(Path.Combine(oracleDir, "features.f32")), 0, want, 0, want.Length * 4);
            Console.WriteLine($"oracle   : features first {string.Join(" ", want.Take(4).Select(v => v.ToString("G5")))} ({want.Length} values, ours {host.Length})");
        }
    }
    return 0;
}
var enc = await Load(encoderFile, half);
var dec = await Load("decoder_joint-model.onnx", false);
Console.WriteLine($"inputs   : pre[{string.Join(",", pre.InputNames)}] enc[{string.Join(",", enc.InputNames)}] dec[{string.Join(",", dec.InputNames)}]");
if (args.Contains("--hold"))
{
    // Load only: park here so the loaded footprint can be read from nvidia-smi before any transcription grows the pools.
    await accelerator.SynchronizeAsync();
    Console.WriteLine("held: sessions loaded, nothing run");
    await Task.Delay(9000);
    return 0;
}
var vocab = ParakeetTdtPipeline.ParseVocabulary(File.ReadAllText(Path.Combine(modelDir, "vocab.txt")));
var pipeline = new ParakeetTdtPipeline(pre, enc, dec, accelerator, vocab);

float[]? gotFeatures = null, gotEncoded = null; int[] featShape = [], encShape = [];
pipeline.OnStage += (stage, t) =>
{
    var host = new float[t.Data.Length];
    t.Data.BaseView.CopyToCPU(host);
    if (stage == "features") { gotFeatures = host; featShape = t.Shape; } else { gotEncoded = host; encShape = t.Shape; }
};

static float[] ReadF32(string path) { var b = File.ReadAllBytes(path); var f = new float[b.Length / 4]; Buffer.BlockCopy(b, 0, f, 0, b.Length); return f; }
static (double rel, double max, int at) Diff(float[] got, float[] want)
{
    int n = Math.Min(got.Length, want.Length);
    double worst = 0, scale = 0; int at = -1;
    for (int i = 0; i < n; i++) { scale = Math.Max(scale, Math.Abs(want[i])); var d = Math.Abs(got[i] - want[i]); if (d > worst) { worst = d; at = i; } }
    return (worst / Math.Max(scale, 1e-9), worst, at);
}

var result = await pipeline.TranscribeAsync(samples);   // first run: kernels compile
Console.WriteLine($"first    : {result.InferenceTimeMs:F0} ms (cold: compile + shapes)");
Console.WriteLine($"text     : {result.Text}");

var failures = 0;
if (oracleDir != null)
{
    var want = File.ReadAllText(Path.Combine(oracleDir, "text.txt")).Trim();
    Console.WriteLine($"oracle   : {want}");
    if (gotFeatures != null)
    {
        var (rel, max, at) = Diff(gotFeatures, ReadF32(Path.Combine(oracleDir, "features.f32")));
        Console.WriteLine($"mel      : shape [{string.Join(",", featShape)}] max|err| {max:G3} = {rel:P3} of the oracle's peak (worst at {at})");
        if (rel > 0.01) { Console.WriteLine("FAIL     : mel diverges from the oracle"); failures++; }
    }
    if (gotEncoded != null)
    {
        var (rel, max, at) = Diff(gotEncoded, ReadF32(Path.Combine(oracleDir, "encoder.f32")));
        Console.WriteLine($"encoder  : shape [{string.Join(",", encShape)}] max|err| {max:G3} = {rel:P3} of the oracle's peak (worst at {at})");
        if (rel > (half || int8 ? 0.15 : 0.02)) { Console.WriteLine("FAIL     : encoder output diverges from the oracle"); failures++; }
    }
    if (!string.Equals(result.Text, want, StringComparison.Ordinal)) { Console.WriteLine("FAIL     : transcript differs from the oracle"); failures++; }
}

var warm = new List<double>();
for (int i = 0; i < runs; i++)
{
    var r = await pipeline.TranscribeAsync(samples);
    warm.Add(r.InferenceTimeMs);
    Console.WriteLine($"run {i + 1}    : {r.InferenceTimeMs:F0} ms  (mel {r.MelTimeMs:F0}, encoder {r.EncoderMs:F0}, decode {r.DecodeStepsMs:F0} over {r.DecodeSteps} steps)");
}
if (warm.Count > 0) { warm.Sort(); Console.WriteLine($"warm     : median {warm[warm.Count / 2]:F0} ms, min {warm[0]:F0} ms for {samples.Length / 16000.0:F2} s of audio"); }
Console.WriteLine(failures == 0 ? "EXIT 0" : $"FAILED: {failures}");
return failures == 0 ? 0 : 1;
