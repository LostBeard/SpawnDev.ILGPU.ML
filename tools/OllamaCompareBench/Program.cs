// Ollama (llama.cpp) vs SpawnDev.ILGPU.ML on the SAME GGUF file, prompt, settings and GPU: prefill and decode tokens/s.
//
// Our side runs GgufGenerator.CreateAsync (Auto engine = GgufBatchDecoder for qwen2 / qwen3 / llama: what SpawnDev.AI and /ai-chat use).
// The model file is Ollama's own blob (resolved from its manifest), so both engines read identical weights.
// Each run's prompt starts with a run number, so neither engine reuses a cached prefix: every run prefills the whole prompt.
// Ollama's model is unloaded (keep_alive 0) before our engine runs, so the two never share the GPU.
//
// Ollama: decode = eval_count / eval_duration, prefill = prompt_eval_count / prompt_eval_duration (its own timers).
// Ours: prefill = prompt tokens / time to the first streamed text, decode = (generated - 1) / (time after the first text).
//
// Run: dotnet run --project tools/OllamaCompareBench -c Release -- [--models qwen3:1.7b-q8_0,qwen3:1.7b] [--tokens 256] [--runs 3]
//        [--filler 1800] [--ctx 4096] [--opencl] [--ollama http://127.0.0.1:11434] [--only ollama|ours]
using System.Diagnostics;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Nodes;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;

string Arg(string name, string def) { int i = Array.IndexOf(args, name); return i >= 0 && i + 1 < args.Length ? args[i + 1] : def; }
var models = Arg("--models", "qwen3:1.7b-q8_0,qwen3:1.7b").Split(',');
int tokens = int.Parse(Arg("--tokens", "256")), runs = int.Parse(Arg("--runs", "3")), filler = int.Parse(Arg("--filler", "1800")), ctx = int.Parse(Arg("--ctx", "4096"));
string ollamaUrl = Arg("--ollama", "http://127.0.0.1:11434"), only = Arg("--only", "");
string root = Environment.GetEnvironmentVariable("OLLAMA_MODELS") is { Length: > 0 } env ? env
    : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".ollama", "models");

string? Blob(string model)
{
    var parts = model.Split(':');
    var manifest = Path.Combine(root, "manifests", "registry.ollama.ai", "library", parts[0], parts.Length > 1 ? parts[1] : "latest");
    if (!File.Exists(manifest)) return null;
    using var doc = JsonDocument.Parse(File.ReadAllText(manifest));
    foreach (var layer in doc.RootElement.GetProperty("layers").EnumerateArray())
        if (layer.GetProperty("mediaType").GetString()!.EndsWith(".model"))
            return Path.Combine(root, "blobs", layer.GetProperty("digest").GetString()!.Replace(':', '-'));
    return null;
}

string fillerText = string.Join(" ", Enumerable.Range(0, filler).Select(i => new[] { "the", "keeper", "watched", "storm", "light", "over", "grey", "water" }[i % 8])) + "\n\n";
string Content(bool longCtx, int run) => $"Run {run}. " + (longCtx ? fillerText : "") + "Write a long story about a lighthouse keeper and a storm.";

// arms: greedy, and a chat-style sampler with the same settings on both engines (top-p 0.9, T 0.7, penalty 1.1 over 64; our engine takes top-p OR top-k, so Ollama's top-k is off: 0)
var arms = new (string Name, double Temp, int TopK, double TopP, double Penalty)[] { ("greedy", 0, 1, 1, 1), ("sampled", 0.7, 0, 0.9, 1.1) };
double Med(IEnumerable<double> v) { var a = v.OrderBy(x => x).ToArray(); return a.Length == 0 ? double.NaN : a[a.Length / 2]; }
var results = new List<string>();
string lastCapture = "";

using var http = new HttpClient { BaseAddress = new Uri(ollamaUrl), Timeout = TimeSpan.FromMinutes(10) };
using var context = MLContext.Create().ToContext();
Device device = args.Contains("--opencl") ? context.GetCLDevices()[0]
    : context.GetCudaDevices().Count > 0 ? context.GetCudaDevices()[0] : context.GetPreferredDevice(false);
string gpu = device.Name;

foreach (var model in models)
{
    var path = Blob(model);
    if (path == null || !File.Exists(path)) { Console.WriteLine($"{model}: not in the Ollama cache ({root})"); continue; }
    GGUFModel gguf;
    await using (var hs = File.OpenRead(path)) gguf = await GGUFParser.ParseHeaderAsync(hs);
    var tok = SentencePieceTokenizer.FromGGUF(gguf)!;
    string quant = gguf.Tensors.FirstOrDefault(t => t.Name == "blk.0.ffn_down.weight")?.Type.ToString() ?? "?";
    Console.WriteLine($"== {model} ({new FileInfo(path).Length / 1048576.0:F0} MiB, ffn_down {quant}) on {gpu}");

    foreach (bool longCtx in new[] { false, true })
        foreach (var arm in arms)
        {
            string label = $"{model} | {(longCtx ? "long" : "short")} prompt | {arm.Name}";
            // ---------- Ollama ----------
            var oPre = new List<double>(); var oDec = new List<double>(); int oPrompt = 0, oGen = 0;
            if (only != "ours")
            {
                for (int r = 0; r <= runs; r++)
                {
                    var body = new JsonObject
                    {
                        ["model"] = model, ["stream"] = false, ["think"] = false, ["keep_alive"] = "10m",
                        ["messages"] = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = Content(longCtx, r) }),
                        ["options"] = new JsonObject
                        {
                            ["num_predict"] = tokens, ["num_ctx"] = ctx, ["seed"] = 1, ["temperature"] = arm.Temp, ["top_k"] = arm.TopK,
                            ["top_p"] = arm.TopP, ["repeat_penalty"] = arm.Penalty, ["repeat_last_n"] = 64,
                        },
                    };
                    var resp = await http.PostAsJsonAsync("/api/chat", body);
                    resp.EnsureSuccessStatusCode();
                    var j = JsonNode.Parse(await resp.Content.ReadAsStringAsync())!;
                    if (r == 0) continue;   // load + warm
                    oPrompt = (int)j["prompt_eval_count"]!; oGen = (int)j["eval_count"]!;
                    oPre.Add(oPrompt / ((long)j["prompt_eval_duration"]! / 1e9));
                    oDec.Add(oGen / ((long)j["eval_duration"]! / 1e9));
                }
                // unload: the runner frees its VRAM before our engine runs
                (await http.PostAsJsonAsync("/api/generate", new JsonObject { ["model"] = model, ["keep_alive"] = 0 })).EnsureSuccessStatusCode();
                await Task.Delay(3000);
            }
            // ---------- ours ----------
            var sPre = new List<double>(); var sDec = new List<double>(); int sPrompt = 0, sGen = 0; string engine = "";
            if (only != "ollama")
            {
                using var accelerator = device.CreateAccelerator(context);
                await using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
                using var gen = await GgufGenerator.CreateAsync(accelerator, fs, ctx, gguf);
                GgufGenerator.EnablePrefixCache = false;
                engine = gen.Engine.ToString();
                if (args.Contains("--no-cuda-graph") && gen.BatchDecoder != null) gen.BatchDecoder.CaptureCuda = false;
                var cfg = arm.Name == "greedy"
                    ? new GenerationConfig { Strategy = "greedy", MaxNewTokens = tokens }
                    : new GenerationConfig { Strategy = "top_p", Temperature = (float)arm.Temp, TopP = (float)arm.TopP, RepetitionPenalty = (float)arm.Penalty, RepeatLastN = 64, Seed = 1, MaxNewTokens = tokens };
                for (int r = 0; r <= runs; r++)
                {
                    var (ids, stops) = ChatTemplates.BuildChatPrompt(gguf, tok, new List<(string, string)> { ("user", Content(longCtx, r)) }, thinking: false);
                    var sw = Stopwatch.StartNew(); double firstMs = -1;
                    var res = await gen.GenerateAsync(ids, cfg, stopTokenIds: stops, onDelta: _ => { if (firstMs < 0) firstMs = sw.Elapsed.TotalMilliseconds; return Task.CompletedTask; });
                    double ms = sw.Elapsed.TotalMilliseconds;
                    if (r == 0 || firstMs < 0 || res.GeneratedTokens < 2) continue;
                    sPrompt = ids.Length; sGen = res.GeneratedTokens;
                    sPre.Add(ids.Length / (firstMs / 1000.0));
                    sDec.Add((res.GeneratedTokens - 1) / ((ms - firstMs) / 1000.0));
                }
                var bd = gen.BatchDecoder; lastCapture = bd == null ? "-" : $"{bd.CaptureStats.Captures} captured, {bd.CaptureStats.Replays} replayed, cuda {bd.CaptureCuda}{(bd.LastCudaCaptureRefusal is string w ? ", refused: " + w : "")}";
            }
            if (only != "ollama") engine += $" (captures {lastCapture})";
            string line = $"{label} | prompt {oPrompt}/{sPrompt} tok (ollama/ours), generated {oGen}/{sGen} | prefill ollama {Med(oPre):F0} vs ours {Med(sPre):F0} tok/s | decode ollama {Med(oDec):F1} vs ours {Med(sDec):F1} tok/s ({Med(sDec) / Med(oDec):F2}x) | ours {engine}";
            Console.WriteLine(line);
            results.Add(line);
        }
}
Console.WriteLine();
// --only ours must not need a running Ollama: the version is asked only when Ollama took part (it threw "connection refused" here before 2026-11-08)
string ollamaVersion = only == "ours" ? "not run" : (await http.GetFromJsonAsync<JsonObject>("/api/version"))?["version"]?.ToString() ?? "?";
Console.WriteLine($"SUMMARY ({gpu}, {runs} timed runs each, medians; Ollama {ollamaVersion})");
foreach (var l in results) Console.WriteLine(l);
return 0;
