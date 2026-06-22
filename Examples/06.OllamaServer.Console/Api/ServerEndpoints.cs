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

    // Protocol-shaped error responses (so a client probing an unknown model gets a clean 404 it understands,
    // not an opaque 500). The server stays up either way.
    private static IResult OpenAiError(string msg, int code) =>
        Results.Json(new { error = new { message = msg, type = "invalid_request_error", code = "model_not_found" } }, J, statusCode: code);
    private static IResult OllamaError(string msg, int code) => Results.Json(new { error = msg }, J, statusCode: code);
    private static IResult AnthropicError(string msg, int code) =>
        Results.Json(new { type = "error", error = new { type = "not_found_error", message = msg } }, J, statusCode: code);

    public static void MapOllamaApi(this IEndpointRouteBuilder app, ModelRegistry registry)
    {
        // ── Liveness / version ────────────────────────────────────────────────────────────────────
        app.MapGet("/", () => Results.Text("Ollama is running (SpawnDev.ILGPU.ML)"));
        app.MapMethods("/", new[] { "HEAD" }, () => Results.Ok()); // Claude CLI / clients ping HEAD for liveness
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
            if (registry.Store.Resolve(model) == null) return OpenAiError($"model '{model}' not found in the Ollama cache", StatusCodes.Status404NotFound);
            var messages = ParseMessages(req, "messages");
            bool stream = GetBool(req, "stream") ?? false;
            var cfg = ReadOpenAiConfig(req);
            var stops = GetStringArray(req, "stop");
            var tools = GetTools(req);
            string id = "chatcmpl-" + Guid.NewGuid().ToString("N")[..24];
            long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

            // Tools present → non-streaming path (the tool_call arrives at the end; we generate, then parse).
            if (tools != null || !stream)
            {
                var (text, res) = await GenerateOnce(registry, model, messages, cfg, stops, ctx.RequestAborted, tools);
                var toolCalls = tools != null ? ChatTemplates.ParseToolCalls(text) : new List<ChatTemplates.ParsedToolCall>();
                object message = toolCalls.Count > 0
                    ? new { role = "assistant", content = (string?)null, tool_calls = toolCalls.Select((tc, ix) => new { id = $"call_{ix}_{Guid.NewGuid().ToString("N")[..8]}", type = "function", function = new { name = tc.Name, arguments = tc.ArgumentsJson } }).ToArray() }
                    : new { role = "assistant", content = (string?)text };
                string finish = toolCalls.Count > 0 ? "tool_calls" : FinishReason(res);
                return Results.Json(new
                {
                    id, @object = "chat.completion", created, model,
                    choices = new[] { new { index = 0, message, finish_reason = finish } },
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
            if (registry.Store.Resolve(model) == null) return OllamaError($"model '{model}' not found", StatusCodes.Status404NotFound);
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
            if (registry.Store.Resolve(model) == null) return OllamaError($"model '{model}' not found", StatusCodes.Status404NotFound);
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
            if (registry.Store.Resolve(model) == null) return AnthropicError($"model '{model}' not found in the Ollama cache", StatusCodes.Status404NotFound);
            var messages = ParseMessages(req, "messages", anthropicSystem: GetString(req, "system"));
            bool stream = GetBool(req, "stream") ?? false;
            var cfg = ReadAnthropicConfig(req);
            var stops = GetStringArray(req, "stop_sequences");
            var tools = GetTools(req);
            string id = "msg_" + Guid.NewGuid().ToString("N")[..24];

            // Tools present (Claude CLI ALWAYS sends its toolset). Non-stream: buffer + format tool_use blocks.
            // Stream: we MUST emit the SSE and text deltas AS the model generates — Claude CLI times out (API
            // error) if it gets no data for the whole generation, and a huge agentic prompt prefill + up to
            // MaxOutputTokens is minutes. We stream text live, holding back only a partial "<tool_call>" suffix so
            // the tool markup never leaks as visible text, and emit tool_use blocks at the end. Streaming also
            // makes a client disconnect ABORT generation (the SSE write throws → generation stops → GPU frees),
            // which the old buffered path didn't (it ran to completion after Claude gave up, pegging VRAM).
            if (tools != null)
            {
                if (!stream)
                {
                    var (genText, gres) = await GenerateOnce(registry, model, messages, cfg, stops, ctx.RequestAborted, tools);
                    var calls0 = ChatTemplates.ParseToolCalls(genText);
                    bool hasTool0 = calls0.Count > 0;
                    string preamble0 = StripToolCalls(genText).Trim();
                    var content = new List<object>();
                    if (!hasTool0) content.Add(new { type = "text", text = genText });
                    else if (preamble0.Length > 0) content.Add(new { type = "text", text = preamble0 });
                    foreach (var tc in calls0)
                        content.Add(new { type = "tool_use", id = "toolu_" + Guid.NewGuid().ToString("N")[..20], name = tc.Name, input = ParseJsonOrEmpty(tc.ArgumentsJson) });
                    return Results.Json(new { id, type = "message", role = "assistant", model,
                        content, stop_reason = hasTool0 ? "tool_use" : AnthropicStop(gres), stop_sequence = (string?)null,
                        usage = new { input_tokens = gres?.PromptTokens ?? 0, output_tokens = gres?.GeneratedTokens ?? 0 } }, J);
                }

                await StartSse(ctx);
                await WriteSseEvent(ctx, "message_start", new { type = "message_start", message = new { id, type = "message", role = "assistant", model, content = Array.Empty<object>(), stop_reason = (string?)null, stop_sequence = (string?)null, usage = new { input_tokens = 0, output_tokens = 1 } } });
                await WriteSseEvent(ctx, "content_block_start", new { type = "content_block_start", index = 0, content_block = new { type = "text", text = "" } });

                const string TC = "<tool_call>";
                var sb = new StringBuilder();
                int emitted = 0;     // chars of sb already sent as text_delta
                bool stopText = false; // once a tool call begins, the rest is tool territory (don't stream as text)
                async Task FlushText(int upTo)
                {
                    if (upTo > emitted)
                    {
                        await WriteSseEvent(ctx, "content_block_delta", new { type = "content_block_delta", index = 0, delta = new { type = "text_delta", text = sb.ToString(emitted, upTo - emitted) } });
                        emitted = upTo;
                    }
                }
                var toolRes = await GenerateStreaming(registry, model, messages, cfg, stops, ctx.RequestAborted, async delta =>
                {
                    sb.Append(delta);
                    if (stopText) return;
                    var s = sb.ToString();
                    int tc = s.IndexOf(TC, Math.Max(0, emitted - TC.Length), StringComparison.Ordinal);
                    if (tc >= 0) { await FlushText(tc); stopText = true; return; } // emit preamble up to the tool call, then stop
                    // No full "<tool_call>" yet — hold back the longest suffix that could be its partial opening tag.
                    int hold = 0, maxH = Math.Min(TC.Length - 1, s.Length - emitted);
                    for (int h = maxH; h > 0; h--) if (s.AsSpan(s.Length - h).SequenceEqual(TC.AsSpan(0, h))) { hold = h; break; }
                    await FlushText(s.Length - hold);
                });
                var calls = ChatTemplates.ParseToolCalls(sb.ToString());
                if (calls.Count == 0) await FlushText(sb.Length); // no tool call → flush the held-back tail
                await WriteSseEvent(ctx, "content_block_stop", new { type = "content_block_stop", index = 0 });
                int bi = 1;
                foreach (var tc in calls)
                {
                    await WriteSseEvent(ctx, "content_block_start", new { type = "content_block_start", index = bi, content_block = new { type = "tool_use", id = "toolu_" + Guid.NewGuid().ToString("N")[..20], name = tc.Name, input = new { } } });
                    await WriteSseEvent(ctx, "content_block_delta", new { type = "content_block_delta", index = bi, delta = new { type = "input_json_delta", partial_json = tc.ArgumentsJson } });
                    await WriteSseEvent(ctx, "content_block_stop", new { type = "content_block_stop", index = bi });
                    bi++;
                }
                await WriteSseEvent(ctx, "message_delta", new { type = "message_delta", delta = new { stop_reason = calls.Count > 0 ? "tool_use" : AnthropicStop(toolRes), stop_sequence = (string?)null }, usage = new { output_tokens = toolRes?.GeneratedTokens ?? 0 } });
                await WriteSseEvent(ctx, "message_stop", new { type = "message_stop" });
                return Results.Empty;
            }

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
        string[]? stops, CancellationToken ct, IReadOnlyList<string>? toolsJson = null)
    {
        using var lease = await registry.AcquireAsync(model, ct);
        var lm = lease.Model;
        var (promptIds, stopIds) = ChatTemplates.BuildChatPrompt(lm.Gguf, lm.Tokenizer, messages, toolsJson: toolsJson);
        var res = await lm.Generator.GenerateAsync(promptIds, cfg, stops, stopIds, onDelta: null, ct);
        return (res.Text, res);
    }

    // Extract tool definitions from a request as raw JSON strings (each is forwarded into the prompt verbatim).
    private static List<string>? GetTools(JsonElement req)
    {
        if (!req.TryGetProperty("tools", out var t) || t.ValueKind != JsonValueKind.Array || t.GetArrayLength() == 0) return null;
        var list = new List<string>();
        foreach (var tool in t.EnumerateArray()) list.Add(tool.GetRawText());
        return list;
    }

    // Remove the <tool_call>…</tool_call> blocks from generated text, leaving any natural-language preamble.
    private static string StripToolCalls(string text)
    {
        const string open = "<tool_call>", close = "</tool_call>";
        var sb = new StringBuilder();
        int i = 0;
        while (true)
        {
            int s = text.IndexOf(open, i, StringComparison.Ordinal);
            if (s < 0) { sb.Append(text, i, text.Length - i); break; }
            sb.Append(text, i, s - i);
            int e = text.IndexOf(close, s, StringComparison.Ordinal);
            if (e < 0) break;
            i = e + close.Length;
        }
        return sb.ToString();
    }

    // Parse a tool-arguments JSON string into a JsonElement (the Anthropic tool_use `input` is an object, not a
    // string). Detached via Clone so it survives the JsonDocument dispose. Falls back to {} on malformed input.
    private static JsonElement ParseJsonOrEmpty(string json)
    {
        try { using var d = JsonDocument.Parse(json); return d.RootElement.Clone(); }
        catch { using var d = JsonDocument.Parse("{}"); return d.RootElement.Clone(); }
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

                // Tool-calling round-trip (ChatML): an assistant turn may carry tool_calls instead of/with text —
                // render them back as <tool_call> blocks so the model sees its own prior call. A role:"tool"
                // result is delivered (qwen/ChatML) as a user turn wrapped in <tool_response>.
                if (role == "assistant" && m.TryGetProperty("tool_calls", out var tcs) && tcs.ValueKind == JsonValueKind.Array)
                {
                    var sb = new StringBuilder(content);
                    foreach (var tc in tcs.EnumerateArray())
                    {
                        if (!tc.TryGetProperty("function", out var fn)) continue;
                        string fname = fn.TryGetProperty("name", out var nn) ? nn.GetString() ?? "" : "";
                        string fargs = fn.TryGetProperty("arguments", out var aa)
                            ? (aa.ValueKind == JsonValueKind.String ? aa.GetString() ?? "{}" : aa.GetRawText())
                            : "{}";
                        sb.Append($"\n<tool_call>\n{{\"name\": \"{fname}\", \"arguments\": {fargs}}}\n</tool_call>");
                    }
                    content = sb.ToString();
                }
                if (role == "tool")
                {
                    content = $"<tool_response>\n{content}\n</tool_response>";
                    role = "user";
                }
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
            {
                string btype = block.TryGetProperty("type", out var bt) ? bt.GetString() ?? "" : "";
                if (btype == "tool_use") // Anthropic: assistant's prior tool call → render as <tool_call>
                {
                    string nm = block.TryGetProperty("name", out var n) ? n.GetString() ?? "" : "";
                    string inp = block.TryGetProperty("input", out var ip) ? ip.GetRawText() : "{}";
                    sb.Append($"\n<tool_call>\n{{\"name\": \"{nm}\", \"arguments\": {inp}}}\n</tool_call>");
                }
                else if (btype == "tool_result") // Anthropic: tool result (in a user turn) → <tool_response>
                {
                    string rc = block.TryGetProperty("content", out var cc)
                        ? (cc.ValueKind == JsonValueKind.String ? cc.GetString() ?? "" : ExtractContent(cc)) : "";
                    sb.Append($"<tool_response>\n{rc}\n</tool_response>");
                }
                else if (block.TryGetProperty("text", out var t) && t.ValueKind == JsonValueKind.String)
                    sb.Append(t.GetString());
            }
            return sb.ToString();
        }
        return "";
    }

    // Cap requested output tokens — agentic clients ask for huge values (Claude CLI: 32000) that a small local
    // model would either ramble into or that wouldn't fit the context.
    private const int MaxOutputTokens = 4096;

    private static GenerationConfig ReadOpenAiConfig(JsonElement req)
    {
        var cfg = new GenerationConfig { MaxNewTokens = Math.Min(GetInt(req, "max_tokens") ?? GetInt(req, "max_completion_tokens") ?? 512, MaxOutputTokens) };
        ApplySampling(cfg, GetFloat(req, "temperature"), GetFloat(req, "top_p"), null, GetInt(req, "seed"));
        return cfg;
    }
    private static GenerationConfig ReadAnthropicConfig(JsonElement req)
    {
        var cfg = new GenerationConfig { MaxNewTokens = Math.Min(GetInt(req, "max_tokens") ?? 512, MaxOutputTokens) };
        ApplySampling(cfg, GetFloat(req, "temperature"), GetFloat(req, "top_p"), GetInt(req, "top_k"), null);
        return cfg;
    }
    private static GenerationConfig ReadOllamaOptions(JsonElement req)
    {
        var cfg = new GenerationConfig { MaxNewTokens = 512 };
        if (req.TryGetProperty("options", out var o) && o.ValueKind == JsonValueKind.Object)
        {
            cfg.MaxNewTokens = Math.Min(GetInt(o, "num_predict") ?? 512, MaxOutputTokens);
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
