// LoRA fine-tuning of a GGUF chat model on JSONL training pairs (Plans/training-autograd-lora-2026-11-04.md, M4): Seven's Vertex data,
// format 3+ - one JSON object per line with `prompt_ids` (exactly what the engine prefilled) and `target_ids` (what the model generated, the
// end-of-turn token excluded), plus `split` ("train" | "heldout"/"dev"). The end-of-turn id (--eot, or the line's `end_of_turn_ids`) is
// appended as the last label so the adapter learns to stop. Lines without ids are counted and skipped: re-tokenizing text is how a LoRA
// silently trains on sequences inference never produces.
//
// Run: dotnet run --project tools/TrainLoraJsonl -c Release -- <model.gguf> <data.jsonl>[,more.jsonl] --out <dir>
//        [--eot 128009] [--epochs 3] [--rank 16] [--alpha 32] [--lr 2e-4] [--accum 8] [--targets attn_q,attn_k,attn_v,attn_output]
//        [--max-len 4096] [--checkpoint] [--seed 1] [--opencl] [--eval-only] [--engine auto|generator|batch] [--no-dev-loss]
//        [--gpu-share 1] (fraction of GPU time to use; 0.5 = idle as long as each step runs; <out>/gpu-share.txt overrides it live)
//        [--progress 10] (a progress line every N AdamW steps) [--save-every 25] (save the adapters every N AdamW steps; 0 = per epoch) [--exclude-source synthetic-v2-hinted[,more]] [--eval-max-new 160]
// --dry-run: print the data summary and exit (no model load).
// --exclude-source: TRAIN lines whose `source` matches are left out (the A/B for teacher-hinted lines; dev lines are never dropped).
// Dev lines without target ids but with `expected_tool` (Seven's v2 `target_usable: false`) are kept for tool choice only; they take
// no part in exact match or dev loss and generate up to --eval-max-new tokens. Scores are reported per data file and per expected tool.
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
var train = new List<Line>();
var dev = new List<Line>();
int skippedNoIds = 0, skippedLong = 0, skippedSource = 0, lines = 0;
var excludeSources = new HashSet<string>(Arg("--exclude-source", "").Split(',', StringSplitOptions.RemoveEmptyEntries));
int evalMaxNew = int.Parse(Arg("--eval-max-new", "160"));
var trainSources = new Dictionary<string, int>();
foreach (var path in args[1].Split(','))
    foreach (var line in File.ReadLines(path))
    {
        if (string.IsNullOrWhiteSpace(line)) continue;
        lines++;
        using var doc = JsonDocument.Parse(line);
        var r = doc.RootElement;
        int[]? Ids(string key) => r.TryGetProperty(key, out var e) && e.ValueKind == JsonValueKind.Array ? e.EnumerateArray().Select(v => v.GetInt32()).ToArray() : null;
        string? Str(string key) => r.TryGetProperty(key, out var e) && e.ValueKind == JsonValueKind.String ? e.GetString() : null;
        var p = Ids("prompt_ids"); var t = Ids("target_ids");
        string split = Str("split") ?? "train";
        string id = r.TryGetProperty("id", out var ide) ? ide.ToString() : lines.ToString();
        // Vertex lines carry `expected_tool` (null = the answer calls no tool): scored as tool choice besides exact ids
        bool hasTool = r.TryGetProperty("expected_tool", out var et);
        string? tool = hasTool && et.ValueKind == JsonValueKind.String ? et.GetString() : null;
        string source = Str("source") ?? "";
        string group = Path.GetFileNameWithoutExtension(path);
        bool hasTarget = t is { Length: > 0 };
        // a dev line without a target still scores tool choice when it says which tool is expected
        if (p == null || p.Length == 0 || (!hasTarget && !(split != "train" && hasTool))) { skippedNoIds++; continue; }
        int eot = eotArg >= 0 ? eotArg : (Ids("end_of_turn_ids") is { Length: > 0 } e0 ? e0[0] : throw new InvalidOperationException("no --eot and the line has no end_of_turn_ids"));
        if (split == "train" && excludeSources.Contains(source)) { skippedSource++; continue; }
        // input = prompt + target; label at position i = the token after it; the prompt predicts nothing (-100) except its last position,
        // which predicts the first target token; the last target position predicts the end of turn.
        var ids = hasTarget ? p.Concat(t!).ToArray() : p;
        if (ids.Length > maxLen) { skippedLong++; continue; }
        var labels = new int[ids.Length];
        for (int i = 0; i < ids.Length; i++) labels[i] = !hasTarget || i < p.Length - 1 ? -100 : i < ids.Length - 1 ? ids[i + 1] : eot;
        var item = new Line(id, ids, labels, p.Length, eot, hasTarget, group, source, hasTool, tool);
        if (split == "train") { train.Add(item); trainSources[group + "/" + source] = trainSources.GetValueOrDefault(group + "/" + source) + 1; }
        else dev.Add(item);
    }
Console.WriteLine($"{lines} lines: {train.Count} train, {dev.Count} dev ({dev.Count(d => !d.HasTarget)} of them tool choice only); skipped {skippedNoIds} without ids, "
    + $"{skippedLong} longer than {maxLen}, {skippedSource} train lines of an excluded source");
Console.WriteLine("  train by file/source: " + string.Join(", ", trainSources.OrderBy(k => k.Key).Select(k => $"{k.Key} {k.Value}")));
if (train.Count == 0) { Console.WriteLine("nothing to train on"); return 1; }
if (args.Contains("--dry-run")) return 0;   // the data summary only: no model, no GPU

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

// --no-dev-loss: a dev loss runs the training graph's backward too (~3x a forward); long runs skip it, save every epoch and rely on
// the final generation eval
bool noDevLoss = args.Contains("--no-dev-loss");
async Task<double> DevLossAsync()
{
    if (dev.Count == 0 || noDevLoss) return double.NaN;
    double sum = 0;
    int n = 0;
    foreach (var d in dev.Where(d => d.HasTarget)) { sum += await trainer.StepAsync(d.Ids, d.Labels, update: false); n++; }
    return n == 0 ? double.NaN : sum / n;
}

double best = await DevLossAsync();
Console.WriteLine($"epoch 0: dev loss {best:F4}");
log.Add(new { epoch = 0, dev_loss = best });
int progressEvery = Math.Max(1, int.Parse(Arg("--progress", "10")));
int saveEvery = int.Parse(Arg("--save-every", "25"));
double shareArg = double.Parse(Arg("--gpu-share", "1"), System.Globalization.CultureInfo.InvariantCulture);
string shareFile = Path.Combine(outDir, "gpu-share.txt");
double GpuShare()
{
    double v = shareArg;
    try { if (File.Exists(shareFile) && double.TryParse(File.ReadAllText(shareFile).Trim(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var f)) v = f; }
    catch (IOException) { }
    return Math.Clamp(v, 0.05, 1.0);
}   // AdamW steps between checkpoint saves; 0 = only at the end of an epoch
var rng = new Random(seed);
for (int ep = 1; ep <= epochs; ep++)
{
    var order = train.OrderBy(_ => rng.Next()).ToList();
    double trainSum = 0;
    sw.Restart();
    for (int i = 0; i < order.Count; i++)
    {
        var stepSw = Stopwatch.StartNew();
        trainSum += await trainer.AccumulateAsync(order[i].Ids, order[i].Labels);   // returns the loss: the readback waits for the GPU
        // --gpu-share: leave the GPU idle for the rest of each period so the PC stays usable (TJ 2026-11-05). Re-read every sequence from
        // <out>/gpu-share.txt when it exists, so a running job can be slowed down or sped up without a restart.
        double share = GpuShare();
        if (share < 1.0) await Task.Delay(TimeSpan.FromMilliseconds(stepSw.Elapsed.TotalMilliseconds * (1.0 / share - 1.0)));
        if (trainer.AccumulatedSequences == accum || i == order.Count - 1)
        {
            await trainer.ApplyAsync();
            // progress per --progress AdamW steps (default 10): long runs otherwise print nothing until the epoch ends
            if (trainer.Step % progressEvery == 0 || i == order.Count - 1)
            {
                double el = sw.Elapsed.TotalSeconds, eta = el / (i + 1) * (order.Count - i - 1);
                Console.WriteLine($"  epoch {ep}: {i + 1}/{order.Count} sequences, step {trainer.Step}, mean train loss {trainSum / (i + 1):F4}, {el:F0} s, ~{eta / 60:F0} min left in the epoch, GPU share {GpuShare():0.##}");
            }
            // --save-every: a long run leaves usable adapters behind as it goes (a killed run otherwise saves nothing)
            if (saveEvery > 0 && trainer.Step % saveEvery == 0 && i < order.Count - 1)
            {
                var (cst, ccfg) = await trainer.SaveAdaptersAsync(Path.GetFileName(args[0]));
                File.WriteAllBytes(Path.Combine(outDir, "adapter_model.safetensors"), cst);
                File.WriteAllText(Path.Combine(outDir, "adapter_config.json"), ccfg);
                File.WriteAllText(Path.Combine(outDir, "checkpoint.json"), JsonSerializer.Serialize(new { epoch = ep, sequences = i + 1, of = order.Count, step = trainer.Step, mean_train_loss = trainSum / (i + 1) }));
                Console.WriteLine($"  checkpoint: adapters at step {trainer.Step} saved to {outDir}");
            }
        }
    }
    double devLoss = await DevLossAsync();
    Console.WriteLine($"epoch {ep}: train loss {trainSum / order.Count:F4}, dev loss {devLoss:F4}, {trainer.Step} AdamW steps, {sw.Elapsed.TotalSeconds:F0} s");
    log.Add(new { epoch = ep, train_loss = trainSum / order.Count, dev_loss = devLoss, steps = trainer.Step, seconds = sw.Elapsed.TotalSeconds });
    if (dev.Count == 0 || noDevLoss || devLoss < best)
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
    var perLine = dev.Select(_ => new Dictionary<string, object>()).ToList();
    var cfg = new SpawnDev.ILGPU.ML.Preprocessing.GenerationConfig { Strategy = "greedy" };
    foreach (bool withLora in new[] { false, true })
    {
        // a generator per pass, the adapters given at load (GgufGenerator.CreateAsync(lora:)): the path that works on either engine
        await using var fs = new FileStream(args[0], FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
        using var gen = await SpawnDev.ILGPU.ML.Pipelines.GgufGenerator.CreateAsync(accelerator, fs, Math.Min(maxLen + evalMaxNew + 64, 8192), engine: engine,
            lora: withLora ? set : null);
        string tag = withLora ? "adapter" : "base";
        // per group (data file): exact over lines with a target, tool choice over lines with expected_tool; per expected tool likewise
        var byGroup = new SortedDictionary<string, int[]>();   // exact, of, toolOk, toolOf
        var byTool = new SortedDictionary<string, int[]>();    // toolOk, toolOf
        for (int di = 0; di < dev.Count; di++)
        {
            var d = dev[di];
            var target = d.Ids[d.PromptLen..];
            cfg.MaxNewTokens = d.HasTarget ? target.Length + 4 : evalMaxNew;
            var res = await gen.GenerateAsync(d.Ids[..d.PromptLen], cfg, stopTokenIds: new[] { d.Eot });
            var g = byGroup.TryGetValue(d.Group, out var gv) ? gv : byGroup[d.Group] = new int[4];
            bool isExact = d.HasTarget && res.GeneratedTokenIds.SequenceEqual(target)
                && res.Stop is SpawnDev.ILGPU.ML.Pipelines.StopReason.StopToken or SpawnDev.ILGPU.ML.Pipelines.StopReason.Eos;
            if (d.HasTarget) { g[1]++; if (isExact) g[0]++; }
            bool? toolOk = null;
            if (d.HasTool)
            {
                toolOk = d.Tool != null ? res.Text.Contains($"\"name\": \"{d.Tool}\"") : !res.Text.Contains("<tool_call>");
                g[3]++; if (toolOk == true) g[2]++;
                var tk = d.Tool ?? "(no tool)";
                var tv = byTool.TryGetValue(tk, out var x) ? x : byTool[tk] = new int[2];
                tv[1]++; if (toolOk == true) tv[0]++;
            }
            perLine[di][tag + "_ids"] = res.GeneratedTokenIds;
            perLine[di][tag + "_text"] = res.Text;
            perLine[di][tag + "_exact"] = isExact;
            if (toolOk != null) perLine[di][tag + "_tool_ok"] = toolOk.Value;
        }
        foreach (var (grp, v) in byGroup)
        {
            Console.WriteLine($"generation on dev ({gen.Engine}) {grp}: {tag}: exact {v[0]}/{v[1]} (target ids, then end of turn)"
                + (v[3] > 0 ? $"; tool choice {v[2]}/{v[3]} (the expected tool named, or no tool call when none is expected)" : ""));
            log.Add(new { generation = tag, group = grp, exact = v[0], exact_of = v[1], tool_choice = v[2], tool_of = v[3] });
        }
        if (byTool.Count > 0)
        {
            Console.WriteLine($"  {tag} tool choice per expected tool: " + string.Join(", ", byTool.Select(k => $"{k.Key} {k.Value[0]}/{k.Value[1]}")));
            log.Add(new { generation = tag, per_tool = byTool.ToDictionary(k => k.Key, k => new { ok = k.Value[0], of = k.Value[1] }) });
        }
    }
    using (var w = new StreamWriter(Path.Combine(outDir, engine == SpawnDev.ILGPU.ML.Pipelines.GgufEngine.Auto ? "generation_eval.jsonl" : $"generation_eval.{Arg("--engine", "auto")}.jsonl")))
        for (int di = 0; di < dev.Count; di++)
        {
            var d = dev[di];
            var row = new Dictionary<string, object?> { ["id"] = d.Id, ["group"] = d.Group, ["expected_tool"] = d.Tool, ["prompt_ids"] = d.Ids[..d.PromptLen],
                ["target_ids"] = d.HasTarget ? d.Ids[d.PromptLen..] : null, ["end_of_turn_id"] = d.Eot };
            foreach (var (k, v) in perLine[di]) row[k] = v;
            w.WriteLine(JsonSerializer.Serialize(row));
        }
}
File.WriteAllText(Path.Combine(outDir, "train_log.json"), JsonSerializer.Serialize(log, new JsonSerializerOptions { WriteIndented = true, NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals }));
return 0;

// one data line: Ids = prompt (+ target), Labels per position, PromptLen, the end-of-turn id, and what it is scored on
record Line(string Id, int[] Ids, int[] Labels, int PromptLen, int Eot, bool HasTarget, string Group, string Source, bool HasTool, string? Tool);
