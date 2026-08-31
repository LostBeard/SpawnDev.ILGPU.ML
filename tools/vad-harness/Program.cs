// Silero VAD on our engine: how fast, and where does it put the boundaries?
//
//   dotnet run --project tools/vad-harness -c Release -- bench    [frames]
//   dotnet run --project tools/vad-harness -c Release -- segments
//   dotnet run --project tools/vad-harness -c Release -- capture           (repro the CUDA capture crash)
//
// VAD_BACKEND=cuda|opencl|cpu forces a backend instead of taking the preferred one.
//
// WHY: PMT is the gate, but it buffers per-test console output, so a passing run tells you the boundaries
// were inside tolerance and not what they actually were - and its durations mix model time with harness
// setup. This prints the numbers.
//
// The rate matters as much as the answer here. A frame is 512 samples at 16 kHz = 32 ms of audio, so a
// detector needs to finish a frame in WELL under 32 ms or it cannot keep up with a live microphone. That
// is not a nice-to-have for a VAD: it is the whole job.
using System.Diagnostics;
using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;

var command = args.Length > 0 ? args[0] : "bench";

var repoRoot = FindRepoRoot();
var www = Path.Combine(repoRoot, "SpawnDev.ILGPU.ML.Demo", "wwwroot");
var modelPath = Path.Combine(www, "references", "vad", "silero_vad.onnx");
if (!File.Exists(modelPath))
{
    Console.WriteLine($"no model at {modelPath} - run tools/gen_silero_vad_reference.py first");
    return 2;
}

var mlBuilder = MLContext.Create();
await mlBuilder.AllAcceleratorsAsync();
using var mlCtx = mlBuilder.ToContext();

// The preferred accelerator is OpenCL on this box, so a CUDA-specific defect is unreachable without a
// way to ASK for CUDA. That is not a hypothetical: capture/replay access-violates on CUDA for this graph.
var want = Environment.GetEnvironmentVariable("VAD_BACKEND")?.Trim().ToLowerInvariant();
using var accel = want switch
{
    // Device-first, the way CudaTests/CPUTests in the demo console do it.
    "cuda" => mlCtx.GetCudaDevices()[0].CreateCudaAccelerator(mlCtx),
    "opencl" => mlCtx.GetCLDevices()[0].CreateCLAccelerator(mlCtx),
    "cpu" => mlCtx.GetCPUDevices()[0].CreateCPUAccelerator(mlCtx),
    _ => await mlCtx.CreatePreferredAcceleratorAsync()
         ?? throw new InvalidOperationException("no accelerator"),
};
Console.WriteLine($"device   : {accel.AcceleratorType} {accel.Name}");

var modelBytes = File.ReadAllBytes(modelPath);

return command switch
{
    "bench" => await Bench(args.Length > 1 ? int.Parse(args[1]) : 200),
    "segments" => await Segments(),
    "capture" => await Capture(),
    _ => Usage(),
};

int Usage()
{
    Console.WriteLine("commands: bench [frames] | segments | capture");
    Console.WriteLine("env:      VAD_BACKEND=cuda|opencl|cpu");
    return 2;
}

/// Forces graph capture ON for whatever backend was selected, with the per-node capture trace armed.
///
/// WHY: capture/replay access-violates on CUDA for this graph (exit=-1073741819). An access violation is
/// not a catchable exception - the process dies and the managed stack shows only async plumbing - so the
/// only way to localise it is a trace that is FLUSHED BEFORE each node's work. GraphExecutor.CaptureTraceFile
/// does exactly that, so after the crash the file's LAST LINE names the node that was executing.
async Task<int> Capture()
{
    var tracePath = Path.Combine(Path.GetTempPath(), "vad-capture-trace.txt");
    if (File.Exists(tracePath)) File.Delete(tracePath);
    SpawnDev.ILGPU.ML.Graph.GraphExecutor.CaptureTraceFile = tracePath;
    Console.WriteLine($"trace    : {tracePath}");
    Console.WriteLine($"capture  : FORCED ON for {accel.AcceleratorType}");
    Console.Out.Flush();

    var wav = Path.Combine(www, "test-audio", "librivox-public-domain.wav");
    var samples = WavDecoder.DecodeWavFile(File.ReadAllBytes(wav))!;

    using var vad = SileroVad.Create(accel, modelBytes, enableGraphCapture: true);
    var frame = new float[SileroVad.WindowSize];

    for (int f = 0; f < 8; f++)
    {
        Array.Copy(samples, f * SileroVad.WindowSize, frame, 0, frame.Length);
        Console.WriteLine($"frame {f}: calling ProcessFrameAsync (captured={vad.IsCaptured}) ...");
        Console.Out.Flush();
        float p = await vad.ProcessFrameAsync(frame);
        Console.WriteLine($"frame {f}: prob {p:F6}  captured={vad.IsCaptured} dispatches={vad.DispatchCount}");
        Console.Out.Flush();
    }

    SpawnDev.ILGPU.ML.Graph.GraphExecutor.CaptureTraceFile = null;
    var lines = File.Exists(tracePath) ? File.ReadAllLines(tracePath) : Array.Empty<string>();
    Console.WriteLine($"\nSURVIVED. trace has {lines.Length} lines; last 5:");
    foreach (var l in lines.TakeLast(5)) Console.WriteLine($"  {l}");
    return 0;
}

async Task<int> Bench(int frames)
{
    var wav = Path.Combine(www, "test-audio", "librivox-public-domain.wav");
    var samples = WavDecoder.DecodeWavFile(File.ReadAllBytes(wav))!;

    using var vad = SileroVad.Create(accel, modelBytes);
    var frame = new float[SileroVad.WindowSize];

    // Warm up: the first frame pays for kernel compilation and buffer-pool growth, and folding that into
    // the average would flatter or slander the steady state depending only on how many frames were run.
    Array.Copy(samples, 0, frame, 0, frame.Length);
    for (int i = 0; i < 3; i++) await vad.ProcessFrameAsync(frame);

    var times = new List<double>(frames);
    var sw = new Stopwatch();
    for (int f = 0; f < frames; f++)
    {
        int at = (f * SileroVad.WindowSize) % (samples.Length - SileroVad.WindowSize);
        Array.Copy(samples, at, frame, 0, frame.Length);
        sw.Restart();
        await vad.ProcessFrameAsync(frame);
        sw.Stop();
        times.Add(sw.Elapsed.TotalMilliseconds);
    }

    times.Sort();
    double total = times.Sum();
    double mean = total / times.Count;
    double p50 = times[times.Count / 2];
    double p99 = times[(int)(times.Count * 0.99)];
    const double FrameMs = SileroVad.WindowSize * 1000.0 / SileroVad.SampleRate;

    Console.WriteLine($"frames   : {frames}");
    Console.WriteLine($"per frame: mean {mean:F2}ms  p50 {p50:F2}ms  p99 {p99:F2}ms  min {times[0]:F2}ms");
    Console.WriteLine($"audio    : {FrameMs:F1}ms per frame");
    Console.WriteLine($"realtime : {FrameMs / mean:F2}x   ({(mean <= FrameMs ? "keeps up" : "TOO SLOW for a live microphone")})");
    return mean <= FrameMs ? 0 : 1;
}

async Task<int> Segments()
{
    var wav = Path.Combine(www, "references", "vad", "vad_three_utterances.wav");
    var refPath = Path.Combine(www, "references", "vad", "vad_three_utterances_segments.json");
    if (!File.Exists(wav) || !File.Exists(refPath))
    {
        Console.WriteLine("fixture missing - run tools/gen_vad_segment_fixture.py then tools/vad-oracle");
        return 2;
    }

    var samples = WavDecoder.DecodeWavFile(File.ReadAllBytes(wav))!;
    using var doc = JsonDocument.Parse(File.ReadAllText(refPath));
    var expected = doc.RootElement.GetProperty("segments").EnumerateArray()
        .Select(s => (Start: s.GetProperty("start_sample").GetInt64(),
                      End: s.GetProperty("end_sample").GetInt64()))
        .ToArray();

    using var vad = SileroVad.Create(accel, modelBytes);
    using var detector = new VoiceActivityDetector(vad, new VadOptions());
    var got = new List<(long Start, long End)>();
    detector.OnSegment += s => got.Add((s.StartSample, s.StartSample + s.Samples.Length));

    const int rtpChunk = 320;   // what RTP delivers - deliberately not a multiple of the model's 512
    for (int i = 0; i < samples.Length; i += rtpChunk)
        await detector.AcceptWaveformAsync(samples, i, Math.Min(rtpChunk, samples.Length - i));
    await detector.FlushAsync();

    Console.WriteLine($"ours     : {got.Count} segment(s)");
    Console.WriteLine($"sherpa   : {expected.Length} segment(s)");
    Console.WriteLine();
    Console.WriteLine($"{"#",2}  {"ours start",12} {"sherpa",10} {"d ms",8}   {"ours end",12} {"sherpa",10} {"d ms",8}");
    double worst = 0;
    for (int i = 0; i < Math.Max(got.Count, expected.Length); i++)
    {
        if (i >= got.Count || i >= expected.Length) { Console.WriteLine($"{i,2}  (only one side has this segment)"); continue; }
        double os = got[i].Start / 16000.0, es = expected[i].Start / 16000.0;
        double oe = got[i].End / 16000.0, ee = expected[i].End / 16000.0;
        double ds = Math.Abs(os - es) * 1000, de = Math.Abs(oe - ee) * 1000;
        worst = Math.Max(worst, Math.Max(ds, de));
        Console.WriteLine($"{i,2}  {os,11:F3}s {es,9:F3}s {ds,7:F0}   {oe,11:F3}s {ee,9:F3}s {de,7:F0}");
    }
    Console.WriteLine();
    Console.WriteLine($"worst boundary difference vs sherpa-onnx: {worst:F0} ms");
    return got.Count == expected.Length ? 0 : 1;
}

static string FindRepoRoot()
{
    var dir = AppContext.BaseDirectory;
    while (dir != null && !File.Exists(Path.Combine(dir, "SpawnDev.ILGPU.ML.slnx")))
        dir = Path.GetDirectoryName(dir);
    return dir ?? throw new InvalidOperationException("could not locate the repo root");
}
