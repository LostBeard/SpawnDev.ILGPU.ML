// Training milestone M3 probe (Plans/training-autograd-lora-2026-11-04.md): LoRA training on a LARGE GGUF (llama3.1:8b Q4_K_M) where no
// float64 reference fits in RAM. Checks, on the engine alone:
//   1. gradient correctness by a directional derivative: (L(theta + eps v) - L(theta - eps v)) / 2 eps against grad . v, for a random v over
//      every adapter parameter, with B nonzero so every gradient path is exercised;
//   2. that training moves the loss (a few AdamW steps on one sequence);
//   3. step time and the VRAM the process takes (nvidia-smi before / after).
//
// Run: dotnet run --project tools/TrainLoraProbe -c Release -- <model.gguf> [--seq 128] [--steps 5] [--rank 8] [--targets attn_q,attn_v] [--opencl]
using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Training;

if (args.Length < 1) { Console.WriteLine("usage: TrainLoraProbe <model.gguf> [--seq N] [--steps N] [--rank N] [--targets a,b] [--opencl]"); return 2; }
string Arg(string name, string def) { int i = Array.IndexOf(args, name); return i >= 0 && i + 1 < args.Length ? args[i + 1] : def; }
int seq = int.Parse(Arg("--seq", "128")), steps = int.Parse(Arg("--steps", "5")), rank = int.Parse(Arg("--rank", "8"));
var targets = Arg("--targets", "attn_q,attn_v").Split(',');

static long VramMiB()
{
    try
    {
        var p = Process.Start(new ProcessStartInfo("nvidia-smi", "--query-gpu=memory.used --format=csv,noheader,nounits") { RedirectStandardOutput = true, UseShellExecute = false })!;
        var s = p.StandardOutput.ReadToEnd(); p.WaitForExit();
        return long.Parse(s.Split('\n')[0].Trim());
    }
    catch { return -1; }
}

using var context = MLContext.Create().ToContext();
Device device = args.Contains("--opencl") ? context.GetCLDevices()[0]
    : context.GetCudaDevices().Count > 0 ? context.GetCudaDevices()[0] : context.GetPreferredDevice(false);
using var accelerator = device.CreateAccelerator(context);
long vram0 = VramMiB();
Console.WriteLine($"accelerator: {accelerator.Name} ({accelerator.AcceleratorType}); VRAM in use before load {vram0} MiB");

var sw = Stopwatch.StartNew();
var options = new LoraOptions { Targets = targets, Rank = rank, Alpha = 2 * rank, LearningRate = 1e-3f, Seed = 7, CheckpointLayers = args.Contains("--checkpoint"), LossOnLabeledPositionsOnly = args.Contains("--labeled-only") };
using var trainer = await GgufLoraTrainer.CreateFromFileAsync(accelerator, args[0], options);
Console.WriteLine($"loaded + compiled in {sw.Elapsed.TotalSeconds:F1} s; {trainer.Adapters.Count} adapters, rank {rank}, checkpoint {options.CheckpointLayers}; VRAM {VramMiB()} MiB");

// B nonzero (PEFT starts it at zero, which makes every A gradient exactly zero and the check vacuous for A).
var rng = new Random(11);
var start = await trainer.GetAdaptersAsync();
foreach (var k in start.Keys.Where(k => k.EndsWith("lora_B.weight")).ToList())
    start[k] = start[k].Select(_ => (float)((rng.NextDouble() * 2 - 1) * 0.02)).ToArray();
await trainer.SetAdaptersAsync(start);

var ids = Enumerable.Range(0, seq).Select(_ => rng.Next(100, trainer.VocabSize - 100)).ToArray();
var labels = ids.Select((_, t) => t >= seq / 2 && t < seq - 1 ? ids[t + 1] : -100).ToArray();

sw.Restart();
float loss0 = await trainer.StepAsync(ids, labels, update: false);
double firstMs = sw.Elapsed.TotalMilliseconds;
long vramStep = VramMiB();
var grads = await trainer.GetGradientsAsync();
Console.WriteLine($"seq {seq}: loss {loss0:F5}; first forward+backward {firstMs:F0} ms; VRAM after a step {vramStep} MiB (+{vramStep - vram0} for this process)");

// ── directional derivative ──
// v = normalize(g/|g| + r) with r a random unit vector: a random direction alone makes grad . v ~ |g|/sqrt(params), below the float32
// resolution of a loss near 10 (MEASURED on smollm2: grad . v = 1e-4, the FD noise-dominated). eps is chosen so the loss moves by ~0.05 and
// ~0.02: big enough to resolve, small enough that the O(eps^2) curvature term stays small (two eps values show which dominates).
double gnorm = Math.Sqrt(grads.Values.Sum(a => a.Sum(x => (double)x * x)));
var v = grads.ToDictionary(kv => kv.Key, kv => kv.Value.Select(g => (float)(g / gnorm)).ToArray());
var r = start.ToDictionary(kv => kv.Key, kv => kv.Value.Select(_ => rng.NextDouble() * 2 - 1).ToArray());
double rnorm = Math.Sqrt(r.Values.Sum(a => a.Sum(x => x * x)));
foreach (var k in v.Keys.ToList()) v[k] = v[k].Zip(r[k], (a, b) => (float)(a + b / rnorm)).ToArray();
double vnorm = Math.Sqrt(v.Values.Sum(a => a.Sum(x => (double)x * x)));
foreach (var a in v.Values) for (int i = 0; i < a.Length; i++) a[i] = (float)(a[i] / vnorm);
double gv = grads.Sum(kv => kv.Value.Zip(v[kv.Key], (g, d) => (double)g * d).Sum());
var fd = new List<string>();
foreach (double move in new[] { 0.05, 0.02 })
{
    double eps = move / Math.Abs(gv);
    await trainer.SetAdaptersAsync(start.ToDictionary(kv => kv.Key, kv => kv.Value.Zip(v[kv.Key], (p, d) => (float)(p + eps * d)).ToArray()));
    float lp = await trainer.StepAsync(ids, labels, update: false);
    await trainer.SetAdaptersAsync(start.ToDictionary(kv => kv.Key, kv => kv.Value.Zip(v[kv.Key], (p, d) => (float)(p - eps * d)).ToArray()));
    float lm = await trainer.StepAsync(ids, labels, update: false);
    double d = (lp - lm) / (2 * eps);
    fd.Add($"eps {eps:G2} (loss moves ~{move}): FD {d:G6} (rel diff {Math.Abs(d - gv) / Math.Abs(gv):E2})");
}
await trainer.SetAdaptersAsync(start);
Console.WriteLine($"directional derivative: |grad| {gnorm:G4}, grad . v = {gv:G6}; " + string.Join("; ", fd));

// ── training moves the loss ──
var times = new List<double>();
var losses = new List<float>();
for (int s = 0; s < steps; s++)
{
    sw.Restart();
    losses.Add(await trainer.StepAsync(ids, labels));
    times.Add(sw.Elapsed.TotalMilliseconds);
}
Console.WriteLine($"{steps} AdamW steps (lr {options.LearningRate}): loss " + string.Join(" -> ", losses.Select(l => l.ToString("F4")))
    + $"; median step {times.OrderBy(t => t).ElementAt(times.Count / 2):F0} ms; VRAM {VramMiB()} MiB");

// ── the adapters at inference: GgufBatchDecoder + LoraRuntime (unmerged) against the training graph's forward, same adapters ──
if (options.LossOnLabeledPositionsOnly) { Console.WriteLine("decoder check skipped (--labeled-only forms no full logits)"); return 0; }
float lossNow = await trainer.StepAsync(ids, labels, update: false);
var trainLogits = await trainer.GetLogitsAsync(seq - 1);
var adapters = await trainer.GetAdaptersAsync();
await using (var fs = new FileStream(args[0], FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true))
{
    if (!SpawnDev.ILGPU.ML.Pipelines.GgufBatchDecoder.Supports(await SpawnDev.ILGPU.ML.GGUF.GGUFParser.ParseHeaderAsync(fs), out var why))
        Console.WriteLine($"batch decoder: not supported for this model ({why}); skipped");
    else
    {
        fs.Seek(0, SeekOrigin.Begin);
        using var dec = await SpawnDev.ILGPU.ML.Pipelines.GgufBatchDecoder.CreateAsync(accelerator, fs, Math.Max(256, seq + 8), 1, null);
        await dec.PrefillAsync(ids);
        var baseLogits = await dec.ReadLogitsAsync();
        using var lora = new LoraRuntime(accelerator, adapters, rank, options.Alpha);
        dec.Lora = lora;
        await dec.PrefillAsync(ids);
        var decLogits = await dec.ReadLogitsAsync();
        double scale = trainLogits.Max(Math.Abs), diff = 0, moved = 0;
        for (int i = 0; i < trainLogits.Length; i++) { diff = Math.Max(diff, Math.Abs(decLogits[i] - trainLogits[i])); moved = Math.Max(moved, Math.Abs(baseLogits[i] - trainLogits[i])); }
        int am(float[] a) { int b = 0; for (int i = 1; i < a.Length; i++) if (a[i] > a[b]) b = i; return b; }
        Console.WriteLine($"batch decoder + LoraRuntime vs training forward (last position): max |diff| {diff:E2} (max |logit| {scale:F2}); "
            + $"base decoder differs by {moved:E2} (the adapters' effect); argmax {am(decLogits)} vs {am(trainLogits)}");
        if (diff > 2e-3 * Math.Max(1, scale) || moved < 10 * diff) { Console.WriteLine("FAIL: the decoder's adapters do not reproduce the training forward"); return 1; }
    }
}
return 0;
