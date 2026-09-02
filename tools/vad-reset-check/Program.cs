// DOES Reset() ACTUALLY RESET? A/B with graph capture on and off.
//
// WHY: the hands-free demo's speech probability decayed monotonically across turns - 0.999, 0.978,
// 0.760, 0.504 - on input whose measured level did not change. The demo calls ResetStreamAsync between
// turns, which reaches SileroVad.Reset(), which zeroes h/c with a HOST-SIDE CopyFromCPU. Capture/replay
// records GPU DISPATCHES only, so a host-side write can be invisible to a replayed plan. If that is what
// is happening, pass 2 after a Reset will NOT reproduce pass 1 with capture on, and WILL with it off.
//
// Deterministic: same file, same frames, no microphone, no browser.
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.CPU;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;

var repoRoot = @"D:\users\tj\Projects\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML";
var www = Path.Combine(repoRoot, "SpawnDev.ILGPU.ML.Demo", "wwwroot");
var modelPath = Path.Combine(www, "references", "vad", "silero_vad.onnx");
var wavPath = Path.Combine(www, "test-audio", "librivox-public-domain.wav");
if (!File.Exists(modelPath)) { Console.WriteLine($"no model at {modelPath}"); return 2; }
if (!File.Exists(wavPath)) { Console.WriteLine($"no wav at {wavPath}"); return 2; }

var mlBuilder = MLContext.Create();
await mlBuilder.AllAcceleratorsAsync();
using var mlCtx = mlBuilder.ToContext();
var want = Environment.GetEnvironmentVariable("VAD_BACKEND")?.Trim().ToLowerInvariant();
using var accel = want switch
{
    "cuda" => mlCtx.GetCudaDevices()[0].CreateCudaAccelerator(mlCtx),
    "opencl" => mlCtx.GetCLDevices()[0].CreateCLAccelerator(mlCtx),
    "cpu" => mlCtx.GetCPUDevices()[0].CreateCPUAccelerator(mlCtx),
    _ => await mlCtx.CreatePreferredAcceleratorAsync() ?? throw new InvalidOperationException("no accelerator"),
};
Console.WriteLine($"device   : {accel.AcceleratorType} {accel.Name}");

var modelBytes = File.ReadAllBytes(modelPath);
var samples = WavDecoder.DecodeWavFile(File.ReadAllBytes(wavPath))!;
int frames = Math.Min(60, samples.Length / SileroVad.WindowSize);
Console.WriteLine($"frames   : {frames} ({frames * 32} ms of audio)\n");

await RunPass(true);
await RunPass(false);
return 0;

async Task RunPass(bool capture)
{
    using var vad = SileroVad.Create(accel, modelBytes, enableGraphCapture: capture);
    var frame = new float[SileroVad.WindowSize];

    var a = new float[frames];
    for (int f = 0; f < frames; f++)
    {
        Array.Copy(samples, f * SileroVad.WindowSize, frame, 0, frame.Length);
        a[f] = await vad.ProcessFrameAsync(frame);
    }

    // The whole point: clear the recurrent state and replay the IDENTICAL audio.
    vad.Reset();

    var b = new float[frames];
    for (int f = 0; f < frames; f++)
    {
        Array.Copy(samples, f * SileroVad.WindowSize, frame, 0, frame.Length);
        b[f] = await vad.ProcessFrameAsync(frame);
    }

    double maxDiff = 0; int firstDiff = -1;
    for (int f = 0; f < frames; f++)
    {
        var d = Math.Abs(a[f] - b[f]);
        if (d > maxDiff) maxDiff = d;
        if (d > 1e-6 && firstDiff < 0) firstDiff = f;
    }
    double maxA = 0, maxB = 0;
    for (int f = 0; f < frames; f++) { if (a[f] > maxA) maxA = a[f]; if (b[f] > maxB) maxB = b[f]; }

    Console.WriteLine($"capture={capture,-5} captured={vad.IsCaptured,-5} "
                    + $"pass1 max p={maxA:F4}  pass2-after-Reset max p={maxB:F4}");
    Console.WriteLine($"    max |pass1 - pass2| = {maxDiff:E3}   first differing frame = "
                    + (firstDiff < 0 ? "none - Reset RESTORED the stream exactly" : firstDiff.ToString()));
    Console.WriteLine($"    first 8 pass1: {string.Join(" ", a.Take(8).Select(x => x.ToString("F4")))}");
    Console.WriteLine($"    first 8 pass2: {string.Join(" ", b.Take(8).Select(x => x.ToString("F4")))}\n");
}
