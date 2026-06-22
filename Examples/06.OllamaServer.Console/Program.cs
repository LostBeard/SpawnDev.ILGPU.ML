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
using Microsoft.AspNetCore.Http;
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

// Enable the grouped-per-query fused attention (the prefill attention win): each Q·K score is computed ONCE
// in shared memory instead of once per output dim, so FusedAttention drops ~14.5x at long prompts
// (qwen2.5-coder @1081 tok: 17.8s -> 1.2s; prefill 24.6s -> 8.1s). A/B-verified bit-identical tokens on
// qwen2.5-coder + gemma4 and grouped==per-element across all 6 backends (PMT). Library-default off (browser-GPU
// falls back; huge-context SKV needs flash attention) — this testbed opts in.
SpawnDev.ILGPU.ML.Kernels.FusedAttentionKernel.EnableGroupedAttention = true;

// Last-position-only logits: at prefill the LM head computes logits for only the last token (the one being
// sampled) instead of all prompt positions — turning the vocab projection (qwen's single biggest prefill node)
// from M=seq into an M=1 GEMV. Waste elimination that SCALES with prompt length (big for agentic 16k+ prompts).
// Token-identical (the generator only reads the last position); library-default off. Correct for generation only.
SpawnDev.ILGPU.ML.GGUF.GGUFGraphBuilder.EnableLastPositionLogits = true;

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
    // 16K context — agentic frontends (Claude CLI) send a large system prompt + tool defs; 8K can overflow.
    // Capped to the model's own context length in the registry. Override via OLLAMA_NUM_CTX.
    int maxCtx = int.TryParse(Environment.GetEnvironmentVariable("OLLAMA_NUM_CTX"), out var nc) ? nc : 16384;
    await using var registry = new ModelRegistry(store, serverAccel, maxSeqLen: maxCtx);

    int port = int.TryParse(Environment.GetEnvironmentVariable("OLLAMA_PORT"), out var p) ? p : 11434;
    var builder = WebApplication.CreateBuilder();
    builder.Logging.ClearProviders();
    builder.WebHost.UseUrls($"http://localhost:{port}");
    var app = builder.Build();

    // Diagnostic + hardening: log every request (method/path/body) and any unhandled exception to a file, and
    // turn a crash into a clean error instead of a dropped connection. Captures real Claude CLI traffic so we can
    // see exactly what it sends and where it breaks. Log: %TEMP%\claude-cli-requests.log.
    var reqLog = Path.Combine(Path.GetTempPath(), "claude-cli-requests.log");
    Console.WriteLine($"  request log: {reqLog}");
    app.Use(async (ctx, next) =>
    {
        ctx.Request.EnableBuffering();
        string body = "";
        if ((ctx.Request.ContentLength ?? 0) > 0)
        {
            using var rd = new StreamReader(ctx.Request.Body, leaveOpen: true);
            body = await rd.ReadToEndAsync();
            ctx.Request.Body.Position = 0;
        }
        string bodyShort = body.Length > 4000 ? body[..4000] + "…" : body;
        try
        {
            await next();
            File.AppendAllText(reqLog, $"[{DateTime.Now:HH:mm:ss}] {ctx.Request.Method} {ctx.Request.Path} -> {ctx.Response.StatusCode}\n  body: {bodyShort}\n\n");
        }
        catch (Exception ex)
        {
            File.AppendAllText(reqLog, $"[{DateTime.Now:HH:mm:ss}] {ctx.Request.Method} {ctx.Request.Path} -> EXCEPTION\n  body: {bodyShort}\n  ex: {ex}\n\n");
            if (!ctx.Response.HasStarted)
            {
                ctx.Response.StatusCode = 500;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync("{\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":" + System.Text.Json.JsonSerializer.Serialize(ex.Message) + "}}");
            }
        }
    });

    app.MapOllamaApi(registry);

    // Pre-load a model BEFORE the HTTP server starts listening (set OLLAMA_PRELOAD=<model>, e.g. by
    // run-claude-cli.bat). Multi-GB models take seconds to load onto the GPU, and loading happens UNDER the
    // single generation gate — so a lazy first-request load makes an agentic client's CONCURRENT startup burst
    // (Claude CLI fires a title request + the main message + a warmup at once) pile up behind the load and get
    // canceled (RequestAborted → OperationCanceledException, the failure seen in the request log). Pre-loading
    // here means /api/version only comes up once the model is resident, so a client that waits for readiness
    // never races the load. Lazy load still works for ad-hoc clients (just with that cold-start caveat).
    var preloadModel = Environment.GetEnvironmentVariable("OLLAMA_PRELOAD");
    if (!string.IsNullOrWhiteSpace(preloadModel))
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        Console.WriteLine($"  Pre-loading '{preloadModel}' onto the GPU (so the first client request doesn't wait for a load)...");
        try
        {
            using var lease = await registry.AcquireAsync(preloadModel);
            Console.WriteLine($"  Model resident in {sw.Elapsed.TotalSeconds:F1}s — ready for clients.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  Pre-load of '{preloadModel}' failed ({ex.GetType().Name}: {ex.Message}); it will load on the first request instead.");
        }
    }

    Console.WriteLine($"SpawnDev.ILGPU.ML — Ollama-compatible server");
    Console.WriteLine($"  http://localhost:{port}   accelerator: {serverAccel.Name}");
    Console.WriteLine($"  {store.List().Count} models from {OllamaModelStore.DefaultRoot()}");
    Console.WriteLine($"  Point a client at it: OpenAI base_url=http://localhost:{port}/v1 · Ollama OLLAMA_HOST=http://localhost:{port} · Claude CLI ANTHROPIC_BASE_URL=http://localhost:{port}");
    await app.RunAsync();
}
