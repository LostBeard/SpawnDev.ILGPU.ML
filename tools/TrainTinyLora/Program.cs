// Cross-language check of the training path's OUTPUT FILES (Plans/training-autograd-lora-2026-11-04.md, M1.4): run one LoRA step on the
// tiny llama fixture with GgufLoraTrainer, write PEFT's adapter_model.safetensors + adapter_config.json, and let Python's own `safetensors`
// read them (tools/check_tiny_lora_peft_files.py) against the torch reference's post-AdamW adapters.
//
// Run: dotnet run --project tools/TrainTinyLora -c Release -- <out dir> [--cpu]
using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Training;

string fixtureDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "SpawnDev.ILGPU.ML.Demo", "wwwroot", "test-train"));
string outDir = args.FirstOrDefault(a => !a.StartsWith("--")) ?? Path.Combine(Path.GetTempPath(), "tiny-lora-out");
Directory.CreateDirectory(outDir);
var gguf = File.ReadAllBytes(Path.Combine(fixtureDir, "tiny-llama-train.gguf"));
using var refDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(fixtureDir, "tiny-llama-train-ref.json")));
var root = refDoc.RootElement;
var cfg = root.GetProperty("config");

using var context = MLContext.Create().ToContext();
Device device;
if (args.Contains("--cpu")) device = context.GetPreferredDevice(preferCPU: true);
else
{
    var cuda = context.GetCudaDevices();
    var ocl = context.GetCLDevices();
    device = cuda.Count > 0 ? (Device)cuda[0] : ocl.Count > 0 ? (Device)ocl[0] : context.GetPreferredDevice(false);
}
using var accelerator = device.CreateAccelerator(context);
Console.WriteLine($"accelerator: {accelerator.Name} ({accelerator.AcceleratorType})");

var tokens = root.GetProperty("tokens").EnumerateArray().Select(t => t.GetInt32()).ToArray();
int lossFrom = root.GetProperty("loss_from").GetInt32();
var labels = tokens.Select((_, t) => t >= lossFrom && t < tokens.Length - 1 ? tokens[t + 1] : -100).ToArray();
var initial = root.GetProperty("lora").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetProperty("value").EnumerateArray().Select(v => v.GetSingle()).ToArray());
var options = new LoraOptions
{
    Targets = new[] { "attn_q", "attn_v" },
    Rank = cfg.GetProperty("rank").GetInt32(), Alpha = cfg.GetProperty("alpha").GetSingle(),
    LearningRate = cfg.GetProperty("lr").GetSingle(), Beta1 = cfg.GetProperty("beta1").GetSingle(), Beta2 = cfg.GetProperty("beta2").GetSingle(),
    Epsilon = cfg.GetProperty("adam_eps").GetSingle(), WeightDecay = cfg.GetProperty("weight_decay").GetSingle(),
    Initial = initial,
};
using var trainer = GgufLoraTrainer.Create(accelerator, gguf, options);
float loss = await trainer.StepAsync(tokens, labels);
var (st, configJson) = await trainer.SaveAdaptersAsync("tiny-llama-train");
File.WriteAllBytes(Path.Combine(outDir, "adapter_model.safetensors"), st);
File.WriteAllText(Path.Combine(outDir, "adapter_config.json"), configJson);
Console.WriteLine($"loss {loss:F6}; wrote {st.Length} B to {outDir}");
return 0;
