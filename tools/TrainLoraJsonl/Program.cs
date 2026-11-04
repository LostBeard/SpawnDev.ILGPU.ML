// LoRA fine-tuning of a GGUF chat model on JSONL training pairs (Plans/training-autograd-lora-2026-11-04.md, M4): Seven's Vertex data,
// format 3+ - one JSON object per line with `prompt_ids` (exactly what the engine prefilled) and `target_ids` (what the model generated, the
// end-of-turn token excluded), plus `split` ("train" | "heldout"/"dev"). The end-of-turn id (--eot, or the line's `end_of_turn_ids`) is
// appended as the last label so the adapter learns to stop. Lines without ids are counted and skipped: re-tokenizing text is how a LoRA
// silently trains on sequences inference never produces.
//
// Run: dotnet run --project tools/TrainLoraJsonl -c Release -- <model.gguf> <data.jsonl>[,more.jsonl] --out <dir>
//        [--eot 128009] [--epochs 3] [--rank 16] [--alpha 32] [--lr 2e-4] [--accum 8] [--targets attn_q,attn_k,attn_v,attn_output]
//        [--max-len 4096] [--checkpoint] [--seed 1] [--opencl] [--eval-only] [--engine auto|generator|batch]
// --eval-only: skip training; evaluate the adapters already in --out. Every run writes <out>/generation_eval.jsonl: per dev line the
// prompt, the target and the generated ids of the base model and of the adapters (a fixture for SpawnDev.AI's adapter test).
using System.Diagnostics;
using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Training;

if (args.Length < 2) { Console.WriteLine("usage: TrainLoraJsonl <model.gguf> <data.jsonl[,more]> --out <dir> [options]"); return 2; }
string Arg(string name, string def) { int i = Array.IndexOf(args, name); return i >= 0 && i + 1 < args.Length ? args[i + 1] : def; }
string outDir = Arg("--out", "lora-out");
int epochs = int.Parse(Arg("--epochs", "3")), rank = int.Parse(Arg("--rank", "16")), accum = int.Parse(Arg("--accum", "8"));
int maxLen = int.Parse(Arg("--max-len", "4096")), seed = int.Parse(Arg("--seed", "1"));
float alpha = float.Parse(Arg("--alpha", (2 * rank).ToString())), lr = float.Parse(Arg("--lr", "2e-4"));
int eotArg = int.Parse(Arg("--eot", "-1"));
var targets = Arg("--targets", "attn_q,attn_k,attn_v,attn_output").Split(',');
Directory.CreateDirectory(outDir);

// ── data ──
var train = new List<(string Id, int[] Ids, int[] Labels)>();
var dev = new List<(string Id, int[] Ids, int[] Labels)>();
int skippedNoIds = 0, skippedLong = 0, lines = 0;
// Vertex lines carry `expected_tool` (null = the answer calls no tool): scored as tool choice besides exact ids
var expectedTool = new Dictionary<string, string?>();
bool hasToolField = false;
foreach (var path in args[1].Split(','))
    foreach (var line in File.ReadLines(path))
    {
        if (string.IsNullOrWhiteSpace(line)) continue;
        lines++;
        using var doc = JsonDocument.Parse(line);
        var r = doc.RootElement;
        int[]? Ids(string key) => r.TryGetProperty(key, out var e) && e.ValueKind == JsonValueKind.Array ? e.EnumerateArray().Select(v => v.GetInt32()).ToArray() : null;
        var p = Ids("prompt_ids"); var t = Ids("target_ids");
        if (p == null || t == null || p.Length == 0 || t.Length == 0) { skippedNoIds++; continue; }
        int eot = eotArg >= 0 ? eotArg : (Ids("end_of_turn_ids") is { Length: > 0 } e0 ? e0[0] : throw new InvalidOperationException("no --eot and the line has no end_of_turn_ids"));
        // input = prompt + target; label at position i = the token after it; the prompt predicts nothing (-100) except its last position,
        // which predicts the first target token; the last target position predicts the end of turn.
        var ids = p.Concat(t).ToArray();
        if (ids.Length > maxLen) { skippedLong++; continue; }
        var labels = new int[ids.Length];
        for (int i = 0; i < ids.Length; i++) labels[i] = i < p.Length - 1 ? -100 : i < ids.Length - 1 ? ids[i + 1] : eot;
        string split = r.TryGetProperty("split", out var s) ? s.GetString() ?? "train" : "train";
        string id = r.TryGetProperty("id", out var ide) ? ide.ToString() : lines.ToString();
        if (r.TryGetProperty("expected_tool", out var et)) { hasToolField = true; expectedTool[id] = et.ValueKind == JsonValueKind.String ? et.GetString() : null; }
        (split == "train" ? train : dev).Add((id, ids, labels));
    }
Console.WriteLine($"{lines} lines: {train.Count} train, {dev.Count} dev; skipped {skippedNoIds} without ids, {skippedLong} longer than {maxLen}");
if (train.Count == 0) { Console.WriteLine("nothing to train on"); return 1; }

// ── model ──
using var context = MLContext.Create().ToContext();
Device device = args.Contains("--opencl") ? context.GetCLDevices()[0]
    : context.GetCudaDevices().Count > 0 ? context.GetCudaDevices()[0] : context.GetPreferredDevice(false);
using var accelerator = device.CreateAccelerator(context);
var options = new LoraOptions { Targets = targets, Rank = rank, Alpha = alpha, LearningRate = lr, Seed = seed, CheckpointLayers = args.Contains("--checkpoint"), LossOnLabeledPositionsOnly = !args.Contains("--full-logits") };
var sw = Stopwatch.StartNew();
bool evalOnly = args.Contains("--eval-only");
var log = new List<object>();
if (!evalOnly)
{
using var trainer = await GgufLoraTrainer.CreateFromFileAsync(accelerator, args[0], options);
Console.WriteLine($"{accelerator.Name} ({accelerator.AcceleratorType}): loaded in {sw.Elapsed.TotalSeconds:F1} s; {trainer.Adapters.Count} adapters rank {rank} alpha {alpha} on {string.Join(",", targets)}; lr {lr}, accum {accum}, checkpoint {options.CheckpointLayers}");

async Task<double> DevLossAsync()
{
    if (dev.Count == 0) return double.NaN;
    double sum = 0;
    foreach (var (_, ids, labels) in dev) sum += await trainer.StepAsync(ids, labels, update: false);
    return sum / dev.Count;
}

double best = await DevLossAsync();
Console.WriteLine($"epoch 0: dev loss {best:F4}");
log.Add(new { epoch = 0, dev_loss = best });
var rng = new Random(seed);
for (int ep = 1; ep <= epochs; ep++)
{
    var order = train.OrderBy(_ => rng.Next()).ToList();
    double trainSum = 0;
    sw.Restart();
    for (int i = 0; i < order.Count; i++)
    {
        trainSum += await trainer.AccumulateAsync(order[i].Ids, order[i].Labels);
        if (trainer.AccumulatedSequences == accum || i == order.Count - 1) await trainer.ApplyAsync();
    }
    double devLoss = await DevLossAsync();
    Console.WriteLine($"epoch {ep}: train loss {trainSum / order.Count:F4}, dev loss {devLoss:F4}, {trainer.Step} AdamW steps, {sw.Elapsed.TotalSeconds:F0} s");
    log.Add(new { epoch = ep, train_loss = trainSum / order.Count, dev_loss = devLoss, steps = trainer.Step, seconds = sw.Elapsed.TotalSeconds });
    if (dev.Count == 0 || devLoss < best)
    {
        best = devLoss;
        var (st, cfgJson) = await trainer.SaveAdaptersAsync(Path.GetFileName(args[0]));
        File.WriteAllBytes(Path.Combine(outDir, "adapter_model.safetensors"), st);
        File.WriteAllText(Path.Combine(outDir, "adapter_config.json"), cfgJson);
        Console.WriteLine($"  saved adapters (best dev loss so far) to {outDir}");
    }
}
}

// ── generation on the dev split, base vs the saved adapters, through the inference engine (GgufGenerator, Auto engine) ──
// exact match = the greedy continuation's ids equal target_ids and the model then emits the end of turn
if (dev.Count > 0 && File.Exists(Path.Combine(outDir, "adapter_model.safetensors")))
{
    var set = LoraAdapterSet.Load(File.ReadAllBytes(Path.Combine(outDir, "adapter_model.safetensors")), File.ReadAllText(Path.Combine(outDir, "adapter_config.json")));
    var engine = Arg("--engine", "auto") switch
    {
        "generator" => SpawnDev.ILGPU.ML.Pipelines.GgufEngine.Generator,
        "batch" => SpawnDev.ILGPU.ML.Pipelines.GgufEngine.BatchDecoder,
        _ => SpawnDev.ILGPU.ML.Pipelines.GgufEngine.Auto,
    };
    var perLine = dev.Select(_ => new Dictionary<string, int[]>()).ToList();
    var cfg = new SpawnDev.ILGPU.ML.Preprocessing.GenerationConfig { Strategy = "greedy" };
    foreach (bool withLora in new[] { false, true })
    {
        // a generator per pass, the adapters given at load (GgufGenerator.CreateAsync(lora:)): the path that works on either engine
        await using var fs = new FileStream(args[0], FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
        using var gen = await SpawnDev.ILGPU.ML.Pipelines.GgufGenerator.CreateAsync(accelerator, fs, Math.Min(maxLen + 64, 8192), engine: engine,
            lora: withLora ? set : null);
        int exact = 0, toolOk = 0;
        for (int di = 0; di < dev.Count; di++)
        {
            var (id, ids, labels) = dev[di];
            int p = Array.FindIndex(labels, l => l != -100) + 1;              // prompt length
            var target = ids[p..];
            int eot = labels[^1];
            cfg.MaxNewTokens = target.Length + 4;
            var res = await gen.GenerateAsync(ids[..p], cfg, stopTokenIds: new[] { eot });
            perLine[di][withLora ? "adapter_ids" : "base_ids"] = res.GeneratedTokenIds;
            if (hasToolField && expectedTool.TryGetValue(id, out var want))
                toolOk += want != null ? (res.Text.Contains($"\"name\": \"{want}\"") ? 1 : 0) : (res.Text.Contains("<tool_call>") ? 0 : 1);
            if (res.GeneratedTokenIds.SequenceEqual(target) && res.Stop is SpawnDev.ILGPU.ML.Pipelines.StopReason.StopToken or SpawnDev.ILGPU.ML.Pipelines.StopReason.Eos) exact++;
        }
        Console.WriteLine($"generation on dev ({gen.Engine}): {(withLora ? "with adapters" : "base")}: {exact}/{dev.Count} exact (target ids, then end of turn)"
            + (hasToolField ? $"; tool choice {toolOk}/{dev.Count} (the expected tool named, or no tool call when none is expected)" : ""));
        log.Add(new { generation = withLora ? "adapters" : "base", exact, tool_choice = toolOk, of = dev.Count });
    }
    using (var w = new StreamWriter(Path.Combine(outDir, engine == SpawnDev.ILGPU.ML.Pipelines.GgufEngine.Auto ? "generation_eval.jsonl" : $"generation_eval.{Arg("--engine", "auto")}.jsonl")))
        for (int di = 0; di < dev.Count; di++)
        {
            var (id, ids, labels) = dev[di];
            int p = Array.FindIndex(labels, l => l != -100) + 1;
            w.WriteLine(JsonSerializer.Serialize(new { id, prompt_ids = ids[..p], target_ids = ids[p..], end_of_turn_id = labels[^1],
                base_ids = perLine[di]["base_ids"], adapter_ids = perLine[di]["adapter_ids"] }));
        }
}
File.WriteAllText(Path.Combine(outDir, "train_log.json"), JsonSerializer.Serialize(log, new JsonSerializerOptions { WriteIndented = true }));
return 0;
