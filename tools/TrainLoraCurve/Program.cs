// Training milestone M2 (Plans/training-autograd-lora-2026-11-04.md): reproduce a float64 torch LoRA fine-tune of a REAL GGUF
// (tools/gen_lora_curve_reference.py) step by step with GgufLoraTrainer, from the same initial adapters, sequences and AdamW settings.
// Gate: the loss at EVERY step matches. The final adapters are reported, not gated: Adam normalises near-zero gradient components to
// +-lr, so float32-vs-float64 noise in them grows into adapter differences that do not change the loss.
//
// Run: dotnet run --project tools/TrainLoraCurve -c Release -- <model.gguf> <reference.json> [--opencl|--cpu]
using System.Diagnostics;
using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Training;

if (args.Length < 2) { Console.WriteLine("usage: TrainLoraCurve <model.gguf> <reference.json> [--opencl|--cpu]"); return 2; }
using var refDoc = JsonDocument.Parse(File.ReadAllText(args[1]));
var root = refDoc.RootElement;
var cfg = root.GetProperty("config");

using var context = MLContext.Create().ToContext();
Device device = args.Contains("--cpu") ? context.GetPreferredDevice(preferCPU: true)
    : args.Contains("--opencl") ? context.GetCLDevices()[0]
    : context.GetCudaDevices().Count > 0 ? context.GetCudaDevices()[0] : context.GetPreferredDevice(false);
using var accelerator = device.CreateAccelerator(context);
Console.WriteLine($"accelerator: {accelerator.Name} ({accelerator.AcceleratorType})");

float[] Arr(JsonElement e) => e.EnumerateArray().Select(v => v.GetSingle()).ToArray();
var options = new LoraOptions
{
    Targets = cfg.GetProperty("targets").EnumerateArray().Select(t => t.GetString()!).ToArray(),
    Rank = cfg.GetProperty("rank").GetInt32(), Alpha = cfg.GetProperty("alpha").GetSingle(),
    LearningRate = cfg.GetProperty("lr").GetSingle(), Beta1 = cfg.GetProperty("beta1").GetSingle(), Beta2 = cfg.GetProperty("beta2").GetSingle(),
    Epsilon = cfg.GetProperty("eps").GetSingle(), WeightDecay = cfg.GetProperty("weight_decay").GetSingle(),
    Initial = root.GetProperty("initial").EnumerateObject().ToDictionary(p => p.Name, p => Arr(p.Value)),
};
var sw = Stopwatch.StartNew();
using var trainer = GgufLoraTrainer.Create(accelerator, File.ReadAllBytes(args[0]), options);
Console.WriteLine($"loaded + compiled in {sw.Elapsed.TotalSeconds:F1} s; {trainer.Adapters.Count} adapters, rank {options.Rank}");

var seqs = root.GetProperty("sequences").EnumerateArray().ToArray();
var want = root.GetProperty("losses").EnumerateArray().Select(v => v.GetDouble()).ToArray();
double worstRel = 0;
for (int step = 0; step < want.Length; step++)
{
    var s = seqs[step % seqs.Length];
    var ids = s.GetProperty("input_ids").EnumerateArray().Select(v => v.GetInt32()).ToArray();
    var labels = s.GetProperty("labels").EnumerateArray().Select(v => v.GetInt32()).ToArray();
    sw.Restart();
    float loss = await trainer.StepAsync(ids, labels);
    double rel = Math.Abs(loss - want[step]) / Math.Max(1e-6, Math.Abs(want[step]));
    worstRel = Math.Max(worstRel, rel);
    Console.WriteLine($"step {step + 1,2}: loss {loss,10:F6}  torch {want[step],10:F6}  rel diff {rel:E2}  ({sw.Elapsed.TotalMilliseconds:F0} ms)");
}

var final = await trainer.GetAdaptersAsync();
double num = 0, den = 0, dnum = 0;
var init = options.Initial!;
foreach (var p in root.GetProperty("final").EnumerateObject())
{
    var w = Arr(p.Value); var g = final[p.Name]; var i0 = init[p.Name];
    for (int i = 0; i < w.Length; i++) { num += (g[i] - w[i]) * (g[i] - w[i]); den += (w[i] - i0[i]) * (w[i] - i0[i]); dnum += w[i] * w[i]; }
}
Console.WriteLine($"final adapters: |ours - torch| / |torch's change from initial| = {Math.Sqrt(num / Math.Max(1e-30, den)):E2} (reported, not gated)");
Console.WriteLine($"worst per-step loss relative diff {worstRel:E2}; loss {want[0]:F4} -> {want[^1]:F4} over {want.Length} steps");
bool ok = worstRel < 1e-3;
Console.WriteLine(ok ? "PASS" : "FAIL: loss curve differs from torch");
return ok ? 0 : 1;
