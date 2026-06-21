using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Routing;
using SpawnDev.ILGPU.ML.Preprocessing;

namespace OllamaServer.Console.Api;

/// <summary>
/// Maps the Ollama-compatible HTTP surface: OpenAI-compat (/v1/chat/completions SSE, /v1/models), Ollama
/// native (/api/chat NDJSON, /api/tags, /api/version), and Anthropic Messages (/v1/messages SSE +
/// count_tokens) for Claude CLI. All three drive the one <see cref="ModelRegistry"/>. v1 = text only.
/// </summary>
public static class ServerEndpoints
{
    private static readonly JsonSerializerOptions J = new(JsonSerializerDefaults.Web);

    public static void MapOllamaApi(this IEndpointRouteBuilder app, ModelRegistry registry)
    {
        // ── Liveness / version ────────────────────────────────────────────────────────────────────
        app.MapGet("/", () => Results.Text("Ollama is running (SpawnDev.ILGPU.ML)"));
        app.MapGet("/api/version", () => Results.Json(new { version = "0.1.0-spawndev" }, J));

        // ── Model listing ─────────────────────────────────────────────────────────────────────────
        app.MapGet("/api/tags", () =>
        {
            var models = registry.Store.List().Select(m => new
            {
                name = m.Name,
                model = m.Name,
                size = m.GgufSize,
                details = new { family = "gguf", parameter_size = "", quantization_level = "" },
            });
            return Results.Json(new { models }, J);
        });
        app.MapGet("/v1/models", () =>
        {
            var data = registry.Store.List().Select(m => new
            { id = m.Name, @object = "model", created = 0, owned_by = "spawndev-ilgpu-ml" });
            return Results.Json(new { @object = "list", data }, J);
        });

        // ── OpenAI: POST /v1/chat/completions ───────────────────────────────────────────────────────
        app.MapPost("/v1/chat/completions", async (HttpContext ctx) =>
        {
            var req = await ReadJsonAsync(ctx);
            string model = GetString(req, "model") ?? "";
            var messages = ParseMessages(req, "messages");
            bool stream = GetBool(req, "stream") ?? false;
            var cfg = ReadOpenAiConfig(req);
            var stops = GetStringArray(req, "stop");
            string id = "chatcmpl-" + Guid.NewGuid().ToString("N")[..24];
            long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

            if (!stream)
            {
                var (text, res) = await GenerateOnce(registry, model, messages, cfg, stops, ctx.RequestAborted);
                return Results.Json(new
                {
                    id, @object = "chat.completion", created, model,
                    choices = new[] { new { index = 0, message = new { role = "assistant", content = text }, finish_reason = FinishReason(res) } },
                    usage = new { prompt_tokens = res?.PromptTokens ?? 0, completion_tokens = res?.GeneratedTokens ?? 0, total_tokens = (res?.PromptTokens ?? 0) + (res?.GeneratedTokens ?? 0) },
                }, J);
            }

            await StartSse(ctx);
            bool first = true;
            var res2 = await GenerateStreaming(registry, model, messages, cfg, stops, ctx.RequestAborted, async delta =>
            {
                var choice = first
                    ? new { index = 0, delta = (object)new { role = "assistant", content = delta }, finish_reason = (string?)null }
                    : new { index = 0, delta = (object)new { content = delta }, finish_reason = (string?)null };
                first = false;
                await WriteSse(ctx, new { id, @object = "chat.completion.chunk", created, model, choices = new[] { choice } });
            });
            await WriteSse(ctx, new { id, @object = "chat.completion.chunk", created, model, choices = new[] { new { index = 0, delta = new { }, finish_reason = FinishReason(res2) } } });
            await ctx.Response.WriteAsync("data: [DONE]\n\n");
            return Results.Empty;
        });

        // ── Ollama: POST /api/chat (NDJSON) ──────────────────────────────────────────────────────────
        app.MapPost("/api/chat", async (HttpContext ctx) =>
        {
            var req = await ReadJsonAsync(ctx);
            string model = GetString(req, "model") ?? "";
            var messages = ParseMessages(req, "messages");
            bool stream = GetBool(req, "stream") ?? true; // Ollama defaults to streaming
            var cfg = ReadOllamaOptions(req);
            string created = DateTimeOffset.UtcNow.ToString("o");

            if (!stream)
            {
                var (text, res) = await GenerateOnce(registry, model, messages, cfg, null, ctx.RequestAborted);
                return Results.Json(new { model, created_at = created, message = new { role = "assistant", content = text }, done = true, done_reason = OllamaDone(res) }, J);
            }

            ctx.Response.ContentType = "application/x-ndjson";
            var res2 = await GenerateStreaming(registry, model, messages, cfg, null, ctx.RequestAborted, async delta =>
            {
                await WriteNdjson(ctx, new { model, created_at = created, message = new { role = "assistant", content = delta }, done = false });
            });
            await WriteNdjson(ctx, new { model, created_at = created, message = new { role = "assistant", content = "" }, done = true, done_reason = OllamaDone(res2) });
            return Results.Empty;
        });
        app.MapPost("/api/generate", async (HttpContext ctx) =>
        {
            var req = await ReadJsonAsync(ctx);
            string model = GetString(req, "model") ?? "";
            string prompt = GetString(req, "prompt") ?? "";
            bool stream = GetBool(req, "stream") ?? true;
            var cfg = ReadOllamaOptions(req);
            var messages = new List<(string, string)> { ("user", prompt) };
            string created = DateTimeOffset.UtcNow.ToString("o");
            if (!stream)
            {
                var (text, res) = await GenerateOnce(registry, model, messages, cfg, null, ctx.RequestAborted);
                return Results.Json(new { model, created_at = created, response = text, done = true, done_reason = OllamaDone(res) }, J);
            }
            ctx.Response.ContentType = "application/x-ndjson";
            var res2 = await GenerateStreaming(registry, model, messages, cfg, null, ctx.RequestAborted, async delta =>
            {
                await WriteNdjson(ctx, new { model, created_at = created, response = delta, done = false });
            });
            await WriteNdjson(ctx, new { model, created_at = created, response = "", done = true, done_reason = OllamaDone(res2) });
            return Results.Empty;
        });

        // ── Anthropic Messages: POST /v1/messages (SSE) + count_tokens — for Claude CLI ────────────────
        app.MapPost("/v1/messages/count_tokens", async (HttpContext ctx) =>
        {
            var req = await ReadJsonAsync(ctx);
            string model = GetString(req, "model") ?? registry.Store.List().FirstOrDefault()?.Name ?? "";
            var messages = ParseMessages(req, "messages");
            int n = await CountTokens(registry, model, messages, ctx.RequestAborted);
            return Results.Json(new { input_tokens = n }, J);
        });
        app.MapPost("/v1/messages", async (HttpContext ctx) =>
        {
            var req = await ReadJsonAsync(ctx);
            string model = GetString(req, "model") ?? "";
            var messages = ParseMessages(req, "messages", anthropicSystem: GetString(req, "system"));
            bool stream = GetBool(req, "stream") ?? false;
            var cfg = ReadAnthropicConfig(req);
            var stops = GetStringArray(req, "stop_sequences");
            string id = "msg_" + Guid.NewGuid().ToString("N")[..24];

            if (!stream)
            {
                var (text, res) = await GenerateOnce(registry, model, messages, cfg, stops, ctx.RequestAborted);
                return Results.Json(new
                {
                    id, type = "message", role = "assistant", model,
                    content = new[] { new { type = "text", text } },
                    stop_reason = AnthropicStop(res), stop_sequence = (string?)null,
                    usage = new { input_tokens = res?.PromptTokens ?? 0, output_tokens = res?.GeneratedTokens ?? 0 },
                }, J);
            }

            await StartSse(ctx);
            await WriteSseEvent(ctx, "message_start", new { type = "message_start", message = new { id, type = "message", role = "assistant", model, content = Array.Empty<object>(), stop_reason = (string?)null, stop_sequence = (string?)null, usage = new { input_tokens = 0, output_tokens = 1 } } });
            await WriteSseEvent(ctx, "content_block_start", new { type = "content_block_start", index = 0, content_block = new { type = "text", text = "" } });
            var res2 = await GenerateStreaming(registry, model, messages, cfg, stops, ctx.RequestAborted, async delta =>
            {
                await WriteSseEvent(ctx, "content_block_delta", new { type = "content_block_delta", index = 0, delta = new { type = "text_delta", text = delta } });
            });
            await WriteSseEvent(ctx, "content_block_stop", new { type = "content_block_stop", index = 0 });
            await WriteSseEvent(ctx, "message_delta", new { type = "message_delta", delta = new { stop_reason = AnthropicStop(res2), stop_sequence = (string?)null }, usage = new { output_tokens = res2?.GeneratedTokens ?? 0 } });
            await WriteSseEvent(ctx, "message_stop", new { type = "message_stop" });
            return Results.Empty;
        });
    }

    // ── Generation bridge ────────────────────────────────────────────────────────────────────────
    private static async Task<(string Text, SpawnDev.ILGPU.ML.Pipelines.GenerationResult? Res)> GenerateOnce(
        ModelRegistry registry, string model, List<(string, string)> messages, GenerationConfig cfg,
        string[]? stops, CancellationToken ct)
    {
        using var lease = await registry.AcquireAsync(model, ct);
        var lm = lease.Model;
        var (promptIds, stopIds) = ChatTemplates.BuildChatPrompt(lm.Gguf, lm.Tokenizer, messages);
        var res = await lm.Generator.GenerateAsync(promptIds, cfg, stops, stopIds, onDelta: null, ct);
        return (res.Text, res);
    }

    private static async Task<SpawnDev.ILGPU.ML.Pipelines.GenerationResult?> GenerateStreaming(
        ModelRegistry registry, string model, List<(string, string)> messages, GenerationConfig cfg,
        string[]? stops, CancellationToken ct, Func<string, Task> onDelta)
    {
        using var lease = await registry.AcquireAsync(model, ct);
        var lm = lease.Model;
        var (promptIds, stopIds) = ChatTemplates.BuildChatPrompt(lm.Gguf, lm.Tokenizer, messages);
        return await lm.Generator.GenerateAsync(promptIds, cfg, stops, stopIds, onDelta, ct);
    }

    private static async Task<int> CountTokens(ModelRegistry registry, string model, List<(string, string)> messages, CancellationToken ct)
    {
        using var lease = await registry.AcquireAsync(model, ct);
        var lm = lease.Model;
        var (promptIds, _) = ChatTemplates.BuildChatPrompt(lm.Gguf, lm.Tokenizer, messages);
        return promptIds.Length;
    }

    // ── SSE / NDJSON writers ───────────────────────────────────────────────────────────────────────
    private static async Task StartSse(HttpContext ctx)
    {
        ctx.Response.ContentType = "text/event-stream";
        ctx.Response.Headers.CacheControl = "no-cache";
        await ctx.Response.Body.FlushAsync();
    }
    private static async Task WriteSse(HttpContext ctx, object payload)
    {
        await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(payload, J)}\n\n");
        await ctx.Response.Body.FlushAsync();
    }
    private static async Task WriteSseEvent(HttpContext ctx, string evt, object payload)
    {
        await ctx.Response.WriteAsync($"event: {evt}\ndata: {JsonSerializer.Serialize(payload, J)}\n\n");
        await ctx.Response.Body.FlushAsync();
    }
    private static async Task WriteNdjson(HttpContext ctx, object payload)
    {
        await ctx.Response.WriteAsync(JsonSerializer.Serialize(payload, J) + "\n");
        await ctx.Response.Body.FlushAsync();
    }

    // ── Request parsing ────────────────────────────────────────────────────────────────────────────
    private static async Task<JsonElement> ReadJsonAsync(HttpContext ctx)
    {
        using var doc = await JsonDocument.ParseAsync(ctx.Request.Body, default, ctx.RequestAborted);
        return doc.RootElement.Clone();
    }
    private static string? GetString(JsonElement e, string name)
        => e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
    private static bool? GetBool(JsonElement e, string name)
        => e.TryGetProperty(name, out var v) && (v.ValueKind == JsonValueKind.True || v.ValueKind == JsonValueKind.False) ? v.GetBoolean() : null;
    private static int? GetInt(JsonElement e, string name)
        => e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : null;
    private static float? GetFloat(JsonElement e, string name)
        => e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetSingle() : null;
    private static string[]? GetStringArray(JsonElement e, string name)
    {
        if (!e.TryGetProperty(name, out var v)) return null;
        if (v.ValueKind == JsonValueKind.String) return new[] { v.GetString()! };
        if (v.ValueKind == JsonValueKind.Array) return v.EnumerateArray().Where(x => x.ValueKind == JsonValueKind.String).Select(x => x.GetString()!).ToArray();
        return null;
    }

    // messages[] → (role, content). Anthropic content can be an array of blocks; we take text blocks.
    private static List<(string, string)> ParseMessages(JsonElement req, string name, string? anthropicSystem = null)
    {
        var list = new List<(string, string)>();
        if (!string.IsNullOrEmpty(anthropicSystem)) list.Add(("system", anthropicSystem));
        if (req.TryGetProperty(name, out var msgs) && msgs.ValueKind == JsonValueKind.Array)
        {
            foreach (var m in msgs.EnumerateArray())
            {
                string role = m.TryGetProperty("role", out var r) ? r.GetString() ?? "user" : "user";
                string content = m.TryGetProperty("content", out var c) ? ExtractContent(c) : "";
                list.Add((role, content));
            }
        }
        return list;
    }
    private static string ExtractContent(JsonElement c)
    {
        if (c.ValueKind == JsonValueKind.String) return c.GetString() ?? "";
        if (c.ValueKind == JsonValueKind.Array) // OpenAI/Anthropic content blocks
        {
            var sb = new StringBuilder();
            foreach (var block in c.EnumerateArray())
                if (block.TryGetProperty("text", out var t) && t.ValueKind == JsonValueKind.String) sb.Append(t.GetString());
            return sb.ToString();
        }
        return "";
    }

    private static GenerationConfig ReadOpenAiConfig(JsonElement req)
    {
        var cfg = new GenerationConfig { MaxNewTokens = GetInt(req, "max_tokens") ?? GetInt(req, "max_completion_tokens") ?? 512 };
        ApplySampling(cfg, GetFloat(req, "temperature"), GetFloat(req, "top_p"), null, GetInt(req, "seed"));
        return cfg;
    }
    private static GenerationConfig ReadAnthropicConfig(JsonElement req)
    {
        var cfg = new GenerationConfig { MaxNewTokens = GetInt(req, "max_tokens") ?? 512 };
        ApplySampling(cfg, GetFloat(req, "temperature"), GetFloat(req, "top_p"), GetInt(req, "top_k"), null);
        return cfg;
    }
    private static GenerationConfig ReadOllamaOptions(JsonElement req)
    {
        var cfg = new GenerationConfig { MaxNewTokens = 512 };
        if (req.TryGetProperty("options", out var o) && o.ValueKind == JsonValueKind.Object)
        {
            cfg.MaxNewTokens = GetInt(o, "num_predict") ?? 512;
            ApplySampling(cfg, GetFloat(o, "temperature"), GetFloat(o, "top_p"), GetInt(o, "top_k"), GetInt(o, "seed"));
        }
        return cfg;
    }
    private static void ApplySampling(GenerationConfig cfg, float? temp, float? topP, int? topK, int? seed)
    {
        if (seed is int s) cfg.Seed = s;
        // temp<=0 → greedy (argmax). Else pick top_p (default) or top_k if explicitly set.
        if (temp is float t && t > 0)
        {
            cfg.Temperature = t;
            if (topK is int k && k > 0) { cfg.Strategy = "top_k"; cfg.TopK = k; }
            else { cfg.Strategy = "top_p"; cfg.TopP = topP ?? 1.0f; }
        }
        else cfg.Strategy = "greedy";
    }

    private static string FinishReason(SpawnDev.ILGPU.ML.Pipelines.GenerationResult? r)
        => r?.Stop == SpawnDev.ILGPU.ML.Pipelines.StopReason.Length ? "length" : "stop";
    private static string OllamaDone(SpawnDev.ILGPU.ML.Pipelines.GenerationResult? r)
        => r?.Stop == SpawnDev.ILGPU.ML.Pipelines.StopReason.Length ? "length" : "stop";
    private static string AnthropicStop(SpawnDev.ILGPU.ML.Pipelines.GenerationResult? r)
        => r?.Stop == SpawnDev.ILGPU.ML.Pipelines.StopReason.Length ? "max_tokens" : "end_turn";
}
