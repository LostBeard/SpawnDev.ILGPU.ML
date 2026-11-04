// Decode speed with and without a LoRA adapter (Plans/training-autograd-lora-2026-11-04.md, M4: "decode not slower"): a random rank-r adapter on
// the given projections of every layer, built from the GGUF header (no training), then greedy generation of N tokens through GgufGenerator
// (Auto engine) base vs adapter; reports prefill (time to first token) and decode tokens/s, median of a few runs.
//
// Run: dotnet run --project tools/LoraDecodeBench -c Release -- <model.gguf> [--rank 16] [--targets attn_q,attn_k,attn_v,attn_output]
//        [--tokens 128] [--prompt 512] [--runs 3] [--ungrouped] [--opencl]
using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Training;

if (args.Length < 1) { Console.WriteLine("usage: LoraDecodeBench <model.gguf> [options]"); return 2; }
string Arg(string name, string def) { int i = Array.IndexOf(args, name); return i >= 0 && i + 1 < args.Length ? args[i + 1] : def; }
int rank = int.Parse(Arg("--rank", "16")), tokens = int.Parse(Arg("--tokens", "128")), promptLen = int.Parse(Arg("--prompt", "512")), runs = int.Parse(Arg("--runs", "3"));
var targets = Arg("--targets", "attn_q,attn_k,attn_v,attn_output").Split(',');
if (args.Contains("--ungrouped")) LoraRuntime.GroupedDown = false;

using var context = MLContext.Create().ToContext();
Device device = args.Contains("--opencl") ? context.GetCLDevices()[0]
    : context.GetCudaDevices().Count > 0 ? context.GetCudaDevices()[0] : context.GetPreferredDevice(false);
using var accelerator = device.CreateAccelerator(context);

GGUFModel header;
await using (var hs = File.OpenRead(args[0])) header = await GGUFParser.ParseHeaderAsync(hs);
var peft = new Dictionary<string, string> { ["attn_q"] = "self_attn.q_proj", ["attn_k"] = "self_attn.k_proj", ["attn_v"] = "self_attn.v_proj",
    ["attn_output"] = "self_attn.o_proj", ["ffn_gate"] = "mlp.gate_proj", ["ffn_up"] = "mlp.up_proj", ["ffn_down"] = "mlp.down_proj" };
var rng = new Random(1);
var tensors = new Dictionary<string, float[]>();
for (int l = 0; l < (int)header.BlockCount; l++)
    foreach (var t in targets)
    {
        var info = header.Tensors.First(x => x.Name == $"blk.{l}.{t}.weight");
        int fin = (int)info.Dimensions[0], fout = (int)info.Dimensions[1];
        string b = $"base_model.model.model.layers.{l}.{peft[t]}";
        tensors[b + ".lora_A.weight"] = Enumerable.Range(0, rank * fin).Select(_ => (float)((rng.NextDouble() * 2 - 1) / Math.Sqrt(fin))).ToArray();
        tensors[b + ".lora_B.weight"] = Enumerable.Range(0, fout * rank).Select(_ => (float)((rng.NextDouble() * 2 - 1) * 0.01)).ToArray();
    }
var set = new LoraAdapterSet(tensors, rank, 2f * rank);
var prompt = Enumerable.Range(0, promptLen).Select(_ => rng.Next(1000, (int)Math.Min(30000, header.VocabSize))).ToArray();
var cfg = new SpawnDev.ILGPU.ML.Preprocessing.GenerationConfig { Strategy = "greedy", MaxNewTokens = tokens };

foreach (var withLora in new[] { false, true })
{
    await using var fs = new FileStream(args[0], FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
    using var gen = await GgufGenerator.CreateAsync(accelerator, fs, promptLen + tokens + 16, lora: withLora ? set : null);
    GgufGenerator.EnablePrefixCache = false;   // (static) every run prefills the whole prompt
    var ttft = new List<double>(); var tps = new List<double>();
    for (int r = 0; r <= runs; r++)
    {
        var (ids, firstMs) = await gen.GenerateFirstTokenIdsAsync(prompt);   // warm-up on r = 0
        var sw = Stopwatch.StartNew();
        var res = await gen.GenerateAsync(prompt, cfg);
        double ms = sw.Elapsed.TotalMilliseconds;
        if (r == 0) continue;
        ttft.Add(firstMs);
        tps.Add(res.GeneratedTokens / Math.Max(1e-3, (ms - firstMs) / 1000.0));
    }
    double Med(List<double> v) => v.OrderBy(x => x).ElementAt(v.Count / 2);
    Console.WriteLine($"{accelerator.AcceleratorType} {gen.Engine} {(withLora ? $"LoRA r{rank} on {string.Join(",", targets)} ({tensors.Count / 2} adapters{(LoraRuntime.GroupedDown ? "" : ", ungrouped")})" : "base")}: "
        + $"prompt {promptLen} -> time to first token {Med(ttft):F0} ms; decode ~{Med(tps):F1} tok/s ({tokens} tokens)");
}
return 0;
