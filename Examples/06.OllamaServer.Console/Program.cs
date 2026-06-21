// Example 06 — Ollama-compatible inference server (drop-in for Ollama, backed by SpawnDev.ILGPU.ML).
// Lets prebuilt agentic frontends (Claude CLI, Pi, Codex, OpenCode, …) use our native-GPU GGUF inference.
//
// v1 IN PROGRESS. This entry point currently exposes `--list`, which reads Ollama's model cache
// (zero-copy) and prints every model it can serve. The Kestrel host + OpenAI/Ollama/Anthropic
// endpoint families land next (see Plans/ollama-server-example-v1-design.md).
//
//   dotnet run --project Examples/06.OllamaServer.Console -- --list

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using OllamaServer.Console;
using OllamaServer.Console.Api;

// Enable the multi-row dequant-GEMM (M>1/prefill): A/B-verified bit-identical tokens with ~7.6x faster prefill
// on Q4_K models (qwen/gemma). The library keeps it off by default pending the full 6-backend sweep; this
// testbed opts in so long-prompt / tool-calling requests are fast.
SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.EnableMultiRowGemm = true;

// --chat <model> "<prompt>" : the server's core generation flow as a CLI — resolve a cached model,
// build its chat prompt (format auto-detected), generate with stop tokens, stream the answer. Proves
// the whole chain (Ollama cache → load → chat template → generator) before the HTTP host goes on top.
if (args.Length >= 1 && args[0] == "--chat")
{
    var store = new OllamaModelStore();
    var name = args.Length >= 2 ? args[1] : "qwen2.5-coder:latest";
    var userPrompt = args.Length >= 3 ? args[2] : "Write a one-line Python function that returns the square of a number.";
    var model = store.Resolve(name);
    if (model == null) { Console.WriteLine($"Not found in cache: {name}"); return; }

    using var ctx = MLContext.Create().ToContext();
    var cuda = ctx.GetCudaDevices();
    var device = cuda.Count > 0 ? (Device)cuda[0] : ctx.GetPreferredDevice(preferCPU: false);
    using var accelerator = device.CreateAccelerator(ctx);

    await using var hs = File.OpenRead(model.GgufPath);
    var gm = await GGUFParser.ParseHeaderAsync(hs);
    var fmt = ChatTemplates.DetectChatFormat(gm);
    Console.WriteLine($"Accelerator: {accelerator.Name}\nModel: {model.Name}  arch={gm.Architecture}  chat-format={fmt}");

    using var session = await InferenceSession.CreateFromGGUFFileAsync(accelerator, model.GgufPath);
    var tok = SentencePieceTokenizer.FromGGUF(gm)!;
    var messages = new List<(string Role, string Content)> { ("user", userPrompt) };
    var (promptIds, stopIds) = ChatTemplates.BuildChatPrompt(gm, tok, messages);
    Console.WriteLine($"Prompt: {promptIds.Length} tokens, stop ids=[{string.Join(",", stopIds)}]\n\nResponse: ");

    using var gen = new GgufGenerator(session, accelerator, gm, maxSeqLen: promptIds.Length + 256 + 8);
    var res = await gen.GenerateAsync(promptIds,
        config: new GenerationConfig { MaxNewTokens = 256 },
        stopTokenIds: stopIds,
        onDelta: d => { Console.Write(d); return Task.CompletedTask; });
    Console.WriteLine($"\n\n[stop={res.Stop}, gen={res.GeneratedTokens} tokens]");
    return;
}

// --template <model> : dump a cached model's GGUF chat_template (Jinja2) + key metadata. Used to design
// the chat-template engine against the REAL templates, not guesses.
if (args.Length >= 1 && args[0] == "--template")
{
    var store2 = new OllamaModelStore();
    var name = args.Length >= 2 ? args[1] : "qwen2.5-coder:latest";
    var model = store2.Resolve(name);
    if (model == null) { Console.WriteLine($"Not found: {name}"); return; }
    await using var hs = File.OpenRead(model.GgufPath);
    var gm = await SpawnDev.ILGPU.ML.GGUF.GGUFParser.ParseHeaderAsync(hs);
    Console.WriteLine($"== {model.Name} ==  arch={gm.Architecture} ctx={gm.ContextLength} vocab={gm.VocabSize}");
    var tmpl = gm.GetMetadataString("tokenizer.chat_template");
    var bos = gm.GetMetadataString("tokenizer.ggml.bos_token_id");
    var eos = gm.GetMetadataString("tokenizer.ggml.eos_token_id");
    var tkModel = gm.GetMetadataString("tokenizer.ggml.model");
    var tkPre = gm.GetMetadataString("tokenizer.ggml.pre");
    Console.WriteLine($"tokenizer.model={tkModel}  pre={tkPre}  bos_id={bos} eos_id={eos}");
    Console.WriteLine($"chat_template ({(tmpl?.Length ?? 0)} chars):\n----\n{tmpl ?? "(none in GGUF metadata)"}\n----");
    return;
}

if (args.Contains("--list"))
{
    var store = new OllamaModelStore();
    Console.WriteLine($"Ollama cache: {OllamaModelStore.DefaultRoot()}  (exists={store.CacheExists})");
    if (!store.CacheExists)
    {
        Console.WriteLine("No Ollama cache found. Install Ollama and pull a model, or set OLLAMA_MODELS.");
        return;
    }

    var models = store.List();
    Console.WriteLine($"\n{models.Count} model(s) servable (zero-copy from the cache):\n");
    foreach (var m in models)
    {
        var extras = new List<string>();
        if (m.MmprojPath != null) extras.Add("vision/mmproj");
        if (m.HasOllamaTemplate) extras.Add("tmpl");
        if (m.HasSystem) extras.Add("sys");
        string flags = extras.Count > 0 ? "  [" + string.Join(", ", extras) + "]" : "";
        Console.WriteLine($"  {m.Name,-52} {m.GgufSize / 1_000_000_000.0,5:F2} GB{flags}");
        if (m.ParamsJson != null)
            Console.WriteLine($"      params: {m.ParamsJson.Replace("\n", " ").Trim()}");
    }
    return;
}

// Default (no subcommand, or "serve"): start the Ollama-compatible HTTP server on :11434.
{
    var store = new OllamaModelStore();
    if (!store.CacheExists) { Console.WriteLine($"No Ollama cache at {OllamaModelStore.DefaultRoot()}. Set OLLAMA_MODELS or pull a model."); return; }

    using var serverCtx = MLContext.Create().ToContext();
    var cudaDevs = serverCtx.GetCudaDevices();
    var dev = cudaDevs.Count > 0 ? (Device)cudaDevs[0] : serverCtx.GetPreferredDevice(preferCPU: false);
    using var serverAccel = dev.CreateAccelerator(serverCtx);
    await using var registry = new ModelRegistry(store, serverAccel);

    int port = int.TryParse(Environment.GetEnvironmentVariable("OLLAMA_PORT"), out var p) ? p : 11434;
    var builder = WebApplication.CreateBuilder();
    builder.Logging.ClearProviders();
    builder.WebHost.UseUrls($"http://localhost:{port}");
    var app = builder.Build();
    app.MapOllamaApi(registry);

    Console.WriteLine($"SpawnDev.ILGPU.ML — Ollama-compatible server");
    Console.WriteLine($"  http://localhost:{port}   accelerator: {serverAccel.Name}");
    Console.WriteLine($"  {store.List().Count} models from {OllamaModelStore.DefaultRoot()}");
    Console.WriteLine($"  Point a client at it: OpenAI base_url=http://localhost:{port}/v1 · Ollama OLLAMA_HOST=http://localhost:{port} · Claude CLI ANTHROPIC_BASE_URL=http://localhost:{port}");
    await app.RunAsync();
}
