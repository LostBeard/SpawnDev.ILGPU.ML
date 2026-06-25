namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Chat message templates for instruct-tuned LLMs.
/// Each model family uses a different prompt format — using the wrong format
/// produces garbage output even from a correct model.
/// </summary>
public static class ChatTemplates
{
    /// <summary>
    /// Format a conversation for Phi-3 Mini Instruct.
    /// Format: &lt;|user|&gt;\n{msg}&lt;|end|&gt;\n&lt;|assistant|&gt;\n
    /// </summary>
    public static string FormatPhi3(string systemPrompt, IEnumerable<(string Role, string Content)> messages)
    {
        var sb = new System.Text.StringBuilder();

        if (!string.IsNullOrWhiteSpace(systemPrompt))
        {
            sb.AppendLine("<|system|>");
            sb.AppendLine(systemPrompt);
            sb.AppendLine("<|end|>");
        }

        foreach (var (role, content) in messages)
        {
            sb.AppendLine(role == "user" ? "<|user|>" : "<|assistant|>");
            sb.AppendLine(content);
            sb.AppendLine("<|end|>");
        }

        sb.AppendLine("<|assistant|>");
        return sb.ToString();
    }

    /// <summary>
    /// Format a conversation for ChatML (used by many models).
    /// Format: &lt;|im_start|&gt;role\ncontent&lt;|im_end|&gt;
    /// </summary>
    public static string FormatChatML(string systemPrompt, IEnumerable<(string Role, string Content)> messages)
    {
        var sb = new System.Text.StringBuilder();

        if (!string.IsNullOrWhiteSpace(systemPrompt))
        {
            sb.AppendLine("<|im_start|>system");
            sb.AppendLine(systemPrompt);
            sb.AppendLine("<|im_end|>");
        }

        foreach (var (role, content) in messages)
        {
            sb.AppendLine($"<|im_start|>{role}");
            sb.AppendLine(content);
            sb.AppendLine("<|im_end|>");
        }

        sb.AppendLine("<|im_start|>assistant");
        return sb.ToString();
    }

    /// <summary>
    /// Format a conversation for LLaMA 2/3 Chat.
    /// Format: [INST] &lt;&lt;SYS&gt;&gt;\n{system}\n&lt;&lt;/SYS&gt;&gt;\n\n{user} [/INST]
    /// </summary>
    public static string FormatLlama(string systemPrompt, IEnumerable<(string Role, string Content)> messages)
    {
        var sb = new System.Text.StringBuilder();
        bool firstUser = true;

        foreach (var (role, content) in messages)
        {
            if (role == "user")
            {
                sb.Append("[INST] ");
                if (firstUser && !string.IsNullOrWhiteSpace(systemPrompt))
                {
                    sb.AppendLine($"<<SYS>>\n{systemPrompt}\n<</SYS>>\n");
                    firstUser = false;
                }
                sb.Append(content);
                sb.Append(" [/INST] ");
            }
            else
            {
                sb.Append(content);
                sb.Append(" ");
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Simple prompt format for base models (GPT-2, DistilGPT-2).
    /// No special tokens — just concatenate the conversation.
    /// </summary>
    public static string FormatSimple(string systemPrompt, string userMessage, string characterName)
    {
        if (!string.IsNullOrWhiteSpace(systemPrompt))
            return $"{systemPrompt}\nUser: {userMessage}\n{characterName}:";
        return $"User: {userMessage}\n{characterName}:";
    }

    /// <summary>
    /// Build a gemma4 chat prompt as TOKEN IDS (not a string). gemma4's turn structure uses CONTROL
    /// tokens — <c>&lt;bos&gt;</c>, turn-open <c>&lt;|turn&gt;</c>, turn-close <c>&lt;turn|&gt;</c>, and the
    /// thinking toggle <c>&lt;|think|&gt;</c> — which must be emitted as SINGLE vocab ids, not as their
    /// literal text run through sub-word tokenization. From the model's <c>tokenizer.chat_template</c>:
    /// <code>
    /// &lt;bos&gt; &lt;|turn&gt;system\n &lt;|think|&gt;\n &lt;turn|&gt;\n  &lt;|turn&gt;user\n{prompt}&lt;turn|&gt;\n  &lt;|turn&gt;model\n
    /// </code>
    /// The <c>&lt;|think|&gt;</c> at the top of the first system turn arms the thinking model; pass
    /// <paramref name="thinking"/>=false to omit it. NOT gemma2/3 (those use <c>&lt;start_of_turn&gt;</c>).
    /// Returns the prompt ids ready to feed to the decode loop; generation appends from there and stops
    /// on the turn-close token (<c>&lt;turn|&gt;</c>).
    /// </summary>
    /// <param name="tok">A gemma4 SentencePiece tokenizer (control-token ids resolved via TryGetId).</param>
    /// <param name="systemPrompt">Optional system text placed in the system turn (after the think toggle).</param>
    /// <param name="userMessage">The user prompt.</param>
    /// <param name="thinking">Emit the <c>&lt;|think|&gt;</c> toggle (default true — gemma4 is a thinking model).</param>
    public static int[] BuildGemma4PromptTokens(SentencePieceTokenizer tok, string? systemPrompt, string userMessage, bool thinking = true)
    {
        int Id(string s) => tok.TryGetId(s, out var v) ? v
            : throw new InvalidOperationException($"gemma4 control token '{s}' not found in the tokenizer vocab — is this actually a gemma4 model?");
        int bos = tok.TryGetId("<bos>", out var b) ? b : -1;
        int turnO = Id("<|turn>"), turnC = Id("<turn|>");

        var ids = new List<int>();
        if (bos >= 0) ids.Add(bos);
        // System turn (carries the thinking toggle + optional system text).
        ids.Add(turnO); ids.AddRange(tok.Encode("system\n"));
        if (thinking) ids.Add(Id("<|think|>"));
        if (!string.IsNullOrWhiteSpace(systemPrompt)) ids.AddRange(tok.Encode(systemPrompt));
        ids.AddRange(tok.Encode("\n")); ids.Add(turnC); ids.AddRange(tok.Encode("\n"));
        // User turn.
        ids.Add(turnO); ids.AddRange(tok.Encode("user\n" + userMessage)); ids.Add(turnC); ids.AddRange(tok.Encode("\n"));
        // Generation prompt — model turn left open for the assistant to complete.
        ids.Add(turnO); ids.AddRange(tok.Encode("model\n"));
        return ids.ToArray();
    }

    /// <summary>The gemma4 turn-close control token id (<c>&lt;turn|&gt;</c>) — the stop token for a
    /// gemma4 generation loop. -1 if absent.</summary>
    public static int Gemma4TurnCloseId(SentencePieceTokenizer tok) => tok.TryGetId("<turn|>", out var v) ? v : -1;

    // ── General multi-turn chat-prompt building (token-level, control tokens as single ids) ──────────
    // Maps an OpenAI/Ollama/Anthropic messages[] list to the model's prompt format. We DETECT the format
    // from the model's own tokenizer.chat_template (ChatML / Llama3 / gemma4), and emit the structural
    // markers (<|im_start|>, <|eot_id|>, …) as SINGLE vocab ids when present (else as encoded text), so
    // greedy sub-word matching can't fracture them. (v1: text only — tool_call branches of the templates
    // are phase 2. Full Jinja2-from-GGUF rendering is the tracked generalization beyond these families.)

    /// <summary>The chat prompt format detected for a model.</summary>
    public enum ChatFormat
    {
        /// <summary>Unknown / unsupported — caller should fall back (we use ChatML best-effort).</summary>
        Unknown,
        /// <summary>ChatML: <c>&lt;|im_start|&gt;role\ncontent&lt;|im_end|&gt;\n</c> (qwen, many others).</summary>
        ChatML,
        /// <summary>Llama 3: <c>&lt;|start_header_id|&gt;role&lt;|end_header_id|&gt;\n\ncontent&lt;|eot_id|&gt;</c>.</summary>
        Llama3,
        /// <summary>gemma4 turn format (control tokens <c>&lt;|turn&gt;</c> / <c>&lt;turn|&gt;</c>).</summary>
        Gemma4,
    }

    /// <summary>Detect the chat format from the model's GGUF <c>tokenizer.chat_template</c> + architecture.</summary>
    public static ChatFormat DetectChatFormat(GGUF.GGUFModel model)
    {
        var t = model.GetMetadataString("tokenizer.chat_template") ?? "";
        if ((model.Architecture ?? "").StartsWith("gemma4") || t.Contains("<|turn>") || t.Contains("<turn|>"))
            return ChatFormat.Gemma4;
        if (t.Contains("<|im_start|>")) return ChatFormat.ChatML;
        if (t.Contains("<|start_header_id|>")) return ChatFormat.Llama3;
        return ChatFormat.Unknown;
    }

    private static void EmitMarker(List<int> ids, SentencePieceTokenizer tok, string marker)
    {
        if (tok.TryGetId(marker, out var id)) ids.Add(id);   // single special-token id (the correct path)
        else ids.AddRange(tok.Encode(marker));               // fallback: encode the literal text
    }

    /// <summary>Build ChatML prompt token ids for a multi-turn conversation.</summary>
    public static int[] BuildChatMLPromptTokens(SentencePieceTokenizer tok,
        IReadOnlyList<(string Role, string Content)> messages, bool addGenerationPrompt = true)
    {
        var ids = new List<int>();
        foreach (var (role, content) in messages)
        {
            EmitMarker(ids, tok, "<|im_start|>");
            ids.AddRange(tok.Encode($"{role}\n{content}"));
            EmitMarker(ids, tok, "<|im_end|>");
            ids.AddRange(tok.Encode("\n"));
        }
        if (addGenerationPrompt) { EmitMarker(ids, tok, "<|im_start|>"); ids.AddRange(tok.Encode("assistant\n")); }
        return ids.ToArray();
    }

    /// <summary>Build Llama 3 prompt token ids for a multi-turn conversation.</summary>
    public static int[] BuildLlama3PromptTokens(SentencePieceTokenizer tok,
        IReadOnlyList<(string Role, string Content)> messages, bool addGenerationPrompt = true)
    {
        var ids = new List<int>();
        EmitMarker(ids, tok, "<|begin_of_text|>");
        foreach (var (role, content) in messages)
        {
            EmitMarker(ids, tok, "<|start_header_id|>");
            ids.AddRange(tok.Encode(role));
            EmitMarker(ids, tok, "<|end_header_id|>");
            ids.AddRange(tok.Encode("\n\n" + content));
            EmitMarker(ids, tok, "<|eot_id|>");
        }
        if (addGenerationPrompt)
        {
            EmitMarker(ids, tok, "<|start_header_id|>");
            ids.AddRange(tok.Encode("assistant"));
            EmitMarker(ids, tok, "<|end_header_id|>");
            ids.AddRange(tok.Encode("\n\n"));
        }
        return ids.ToArray();
    }

    /// <summary>Build gemma4 prompt token ids for a multi-turn conversation (per-turn
    /// <c>&lt;|turn&gt;role\ncontent&lt;turn|&gt;\n</c>; the thinking toggle arms the first turn).</summary>
    public static int[] BuildGemma4MultiTurnPromptTokens(SentencePieceTokenizer tok,
        IReadOnlyList<(string Role, string Content)> messages, bool thinking = true, bool addGenerationPrompt = true,
        IReadOnlyList<string>? toolsJson = null)
    {
        int Id(string s) => tok.TryGetId(s, out var v) ? v : -1;
        int bos = Id("<bos>"), turnO = Id("<|turn>"), turnC = Id("<turn|>"), think = Id("<|think|>");
        var ids = new List<int>();
        if (bos >= 0) ids.Add(bos);
        bool armed = false;
        int startIdx = 0;

        // Native gemma4 tool injection. gemma4 does NOT use the ChatML <tools>/<tool_call> convention — it
        // expects tool signatures inside a leading system turn as <|tool>declaration:NAME{…}<tool|> blocks
        // (its own DSL, with <|"|> as the string-quote control token), and it EMITS calls as
        // <|tool_call>call:NAME{…}<tool_call|>. We build that system turn up front; the thinking toggle is
        // armed here so the message loop doesn't double-emit it. (No tools → identical to prior behavior.)
        if (toolsJson != null && toolsJson.Count > 0)
        {
            if (turnO >= 0) ids.Add(turnO);
            ids.AddRange(tok.Encode("system\n"));
            if (thinking && think >= 0) { ids.Add(think); ids.AddRange(tok.Encode("\n")); armed = true; }
            if (messages.Count > 0 && (messages[0].Role == "system" || messages[0].Role == "developer"))
            {
                ids.AddRange(tok.Encode(messages[0].Content.Trim()));
                startIdx = 1;
            }
            foreach (var t in toolsJson)
            {
                EmitMarker(ids, tok, "<|tool>");
                EmitGemma4ToolDecl(ids, tok, t);
                EmitMarker(ids, tok, "<tool|>");
            }
            if (turnC >= 0) ids.Add(turnC);
            ids.AddRange(tok.Encode("\n"));
        }

        for (int mi = startIdx; mi < messages.Count; mi++)
        {
            var (role, content) = messages[mi];
            if (turnO >= 0) ids.Add(turnO);
            ids.AddRange(tok.Encode($"{role}\n"));
            if (!armed && thinking && think >= 0) { ids.Add(think); armed = true; }
            ids.AddRange(tok.Encode(content));
            ids.AddRange(tok.Encode("\n"));
            if (turnC >= 0) ids.Add(turnC);
            ids.AddRange(tok.Encode("\n"));
        }
        if (addGenerationPrompt) { if (turnO >= 0) ids.Add(turnO); ids.AddRange(tok.Encode("model\n")); }
        return ids.ToArray();
    }

    /// <summary>Emit one tool's gemma4 <c>declaration:NAME{…}</c> DSL (a port of the model's
    /// <c>format_function_declaration</c> Jinja macro). <paramref name="toolJson"/> is the full
    /// <c>{"type":"function","function":{name,description,parameters}}</c> object the client sent. String
    /// literals are wrapped in the <c>&lt;|"|&gt;</c> quote token. Covers the common scalar-parameter shape
    /// (string/number/boolean/integer); nested object/array params fall back to their bare type name.</summary>
    private static void EmitGemma4ToolDecl(List<int> ids, SentencePieceTokenizer tok, string toolJson)
    {
        void Txt(string s) { if (!string.IsNullOrEmpty(s)) ids.AddRange(tok.Encode(s)); }
        void Quote(string s) { EmitMarker(ids, tok, "<|\"|>"); if (!string.IsNullOrEmpty(s)) ids.AddRange(tok.Encode(s)); EmitMarker(ids, tok, "<|\"|>"); }

        System.Text.Json.JsonElement fn;
        System.Text.Json.JsonDocument? doc = null;
        try
        {
            doc = System.Text.Json.JsonDocument.Parse(toolJson);
            var root = doc.RootElement;
            fn = root.TryGetProperty("function", out var f) ? f : root;
        }
        catch { doc?.Dispose(); return; }

        using (doc)
        {
            string name = fn.TryGetProperty("name", out var n) ? n.GetString() ?? "" : "";
            string desc = fn.TryGetProperty("description", out var d) ? d.GetString() ?? "" : "";
            Txt("declaration:" + name + "{description:");
            Quote(desc);
            if (fn.TryGetProperty("parameters", out var p) && p.ValueKind == System.Text.Json.JsonValueKind.Object)
            {
                Txt(",parameters:{");
                if (p.TryGetProperty("properties", out var props) && props.ValueKind == System.Text.Json.JsonValueKind.Object)
                {
                    var keys = new List<string>();
                    foreach (var pr in props.EnumerateObject()) keys.Add(pr.Name);
                    if (keys.Count > 0)
                    {
                        keys.Sort(StringComparer.Ordinal);   // gemma renders properties in dictsort order
                        Txt("properties:{");
                        for (int i = 0; i < keys.Count; i++)
                        {
                            var pv = props.GetProperty(keys[i]);
                            Txt(keys[i] + ":{");
                            bool comma = false;
                            if (pv.TryGetProperty("description", out var pd) && pd.ValueKind == System.Text.Json.JsonValueKind.String)
                            { Txt("description:"); Quote(pd.GetString() ?? ""); comma = true; }
                            if (comma) Txt(",");
                            string pt = pv.TryGetProperty("type", out var ptv) && ptv.ValueKind == System.Text.Json.JsonValueKind.String ? ptv.GetString() ?? "string" : "string";
                            Txt("type:"); Quote(pt.ToUpperInvariant()); Txt("}");
                            if (i < keys.Count - 1) Txt(",");
                        }
                        Txt("},");
                    }
                }
                if (p.TryGetProperty("required", out var req) && req.ValueKind == System.Text.Json.JsonValueKind.Array && req.GetArrayLength() > 0)
                {
                    Txt("required:[");
                    int ri = 0, rc = req.GetArrayLength();
                    foreach (var r in req.EnumerateArray()) { Quote(r.GetString() ?? ""); if (++ri < rc) Txt(","); }
                    Txt("],");
                }
                string ptype = p.TryGetProperty("type", out var ptt) && ptt.ValueKind == System.Text.Json.JsonValueKind.String ? ptt.GetString() ?? "object" : "object";
                Txt("type:"); Quote(ptype.ToUpperInvariant()); Txt("}");
            }
            Txt("}");
        }
    }

    /// <summary>
    /// Build the prompt token ids for a multi-turn conversation in the model's own format, plus the stop
    /// token id(s) that end an assistant turn for that format. Dispatches on <see cref="DetectChatFormat"/>;
    /// unknown formats fall back to ChatML (the most common). The returned stop ids should be added to the
    /// generator's stop set alongside EOS.
    /// </summary>
    public static (int[] PromptIds, int[] StopTokenIds) BuildChatPrompt(GGUF.GGUFModel model,
        SentencePieceTokenizer tok, IReadOnlyList<(string Role, string Content)> messages, bool thinking = true,
        IReadOnlyList<string>? toolsJson = null)
    {
        var format = DetectChatFormat(model);
        bool hasTools = toolsJson != null && toolsJson.Count > 0;

        // gemma4 injects tools at the TOKEN level in its own declaration DSL (see
        // BuildGemma4MultiTurnPromptTokens) — NOT the ChatML system-message convention — so route it before
        // the BuildToolSystem injection below.
        if (format == ChatFormat.Gemma4)
            return (BuildGemma4MultiTurnPromptTokens(tok, messages, thinking, toolsJson: hasTools ? toolsJson : null),
                    Ids(tok, "<turn|>"));

        // ChatML / Llama3: advertise the tools in the system message (qwen <tools>/<tool_call> convention).
        // The generated text is later scanned by ParseToolCalls.
        if (hasTools)
        {
            var aug = new List<(string Role, string Content)>(messages);
            string? sys = aug.Count > 0 && aug[0].Role == "system" ? aug[0].Content : null;
            string toolSys = BuildToolSystem(format, sys, toolsJson!);
            if (aug.Count > 0 && aug[0].Role == "system") aug[0] = ("system", toolSys);
            else aug.Insert(0, ("system", toolSys));
            messages = aug;
        }

        switch (format)
        {
            case ChatFormat.Llama3:
                return (BuildLlama3PromptTokens(tok, messages),
                        Ids(tok, "<|eot_id|>", "<|eom_id|>"));
            case ChatFormat.ChatML:
            default:
                return (BuildChatMLPromptTokens(tok, messages),
                        Ids(tok, "<|im_end|>"));
        }

        static int[] Ids(SentencePieceTokenizer tok, params string[] markers)
        {
            var list = new List<int>();
            foreach (var m in markers) if (tok.TryGetId(m, out var id)) list.Add(id);
            return list.ToArray();
        }
    }

    // ── Tool / function-calling (v2) ─────────────────────────────────────────────────────────────────
    /// <summary>A tool call parsed from a model's generated text. <see cref="ArgumentsJson"/> is the raw JSON
    /// of the arguments (an object), ready to forward as the OpenAI <c>arguments</c> string or the Anthropic
    /// <c>input</c> object.</summary>
    public readonly record struct ParsedToolCall(string Name, string ArgumentsJson);

    /// <summary>Build the system-message content that advertises the available <paramref name="toolsJson"/>
    /// (each already a serialized tool definition) in the model's tool-calling format. ChatML (qwen) uses the
    /// <c>&lt;tools&gt;</c>/<c>&lt;tool_call&gt;</c> convention; other formats reuse it as a reasonable default
    /// until their own tool format is wired.</summary>
    public static string BuildToolSystem(ChatFormat format, string? systemContent, IReadOnlyList<string> toolsJson)
    {
        // Mirror the model's OWN native tool template (qwen2.5 chat_template) as closely as possible — the exact
        // wording drives elicitation. The critical clause is "with NO other text. Do not include any backticks":
        // without it qwen2.5-coder tends to EXPLAIN the call in prose (and even write the function as code)
        // instead of emitting the tool-call JSON. Each tool entry is forwarded verbatim — the client already
        // sends the full {"type":"function","function":{…}} object the template expects.
        var sb = new System.Text.StringBuilder();
        if (!string.IsNullOrWhiteSpace(systemContent)) { sb.Append(systemContent); sb.Append('\n'); }
        sb.Append("\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n");
        sb.Append("You are provided with function signatures within <tools></tools>:\n<tools>");
        foreach (var t in toolsJson) { sb.Append('\n'); sb.Append(t); }
        sb.Append("\n</tools>\n\nFor each function call, return a json object with function name and arguments ");
        sb.Append("within <tool_call></tool_call> with NO other text. Do not include any backticks or ```json.\n");
        sb.Append("<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
        return sb.ToString();
    }

    /// <summary>Extract tool calls from generated text. Handles the <c>&lt;tool_call&gt;{json}&lt;/tool_call&gt;</c>
    /// convention (ChatML/qwen, and the default used by <see cref="BuildToolSystem"/>). Malformed blocks are
    /// skipped. Returns an empty list when the model produced plain text.</summary>
    public static List<ParsedToolCall> ParseToolCalls(string text)
    {
        var calls = new List<ParsedToolCall>();

        // gemma4 native emission: <|tool_call>call:NAME{key:<|"|>val<|"|>,…}<tool_call|> — its own DSL, not
        // JSON or the ChatML <tool_call> convention. Handle it first when present.
        if (text.Contains("<|tool_call>call:", StringComparison.Ordinal))
        {
            ParseGemma4ToolCalls(text, calls);
            if (calls.Count > 0) return calls;
        }

        const string open = "<tool_call>", close = "</tool_call>";
        int i = 0;
        while (true)
        {
            int s = text.IndexOf(open, i, StringComparison.Ordinal);
            if (s < 0) break;
            int e = text.IndexOf(close, s + open.Length, StringComparison.Ordinal);
            if (e < 0) break;
            TryAddToolCall(text.Substring(s + open.Length, e - s - open.Length).Trim(), calls);
            i = e + close.Length;
        }

        // Fallback: the model emitted the call(s) as BARE JSON with no <tool_call> wrapper. qwen2.5-coder does
        // this frequently even under its OWN native template (verified against Ollama: tool call lands in
        // message.content, message.tool_calls is null). ParseToolCalls is only invoked in a tool-calling
        // context (callers pass tools), so scanning for {"name","arguments"} objects here is safe and lets us
        // return STRUCTURED tool_calls where Ollama leaks raw text. Strip markdown fences, then take each
        // balanced top-level JSON object that looks like a call.
        if (calls.Count == 0)
            foreach (var obj in ExtractTopLevelJsonObjects(StripCodeFences(text)))
                TryAddToolCall(obj, calls);

        return calls;
    }

    /// <summary>Parse one JSON object as a <c>{"name","arguments"}</c> tool call and append it if valid.
    /// Requires a non-empty string <c>name</c>; accepts <c>arguments</c> or <c>parameters</c> as the args object
    /// (defaults to <c>{}</c>). Malformed/non-matching JSON is silently skipped.</summary>
    private static void TryAddToolCall(string json, List<ParsedToolCall> calls)
    {
        try
        {
            using var doc = System.Text.Json.JsonDocument.Parse(json);
            var root = doc.RootElement;
            if (root.ValueKind != System.Text.Json.JsonValueKind.Object) return;
            string name = root.TryGetProperty("name", out var n) && n.ValueKind == System.Text.Json.JsonValueKind.String
                ? (n.GetString() ?? "") : "";
            if (string.IsNullOrEmpty(name)) return;
            string args = root.TryGetProperty("arguments", out var a) ? a.GetRawText()
                        : root.TryGetProperty("parameters", out var p) ? p.GetRawText() : "{}";
            calls.Add(new ParsedToolCall(name, args));
        }
        catch { /* skip a malformed block rather than fail the whole response */ }
    }

    /// <summary>Remove markdown code fences (<c>```json … ```</c> / <c>``` … ```</c>), keeping inner content, so a
    /// fenced tool-call JSON is still parseable. No-op when the text has no fences.</summary>
    private static string StripCodeFences(string text)
    {
        if (text.IndexOf("```", StringComparison.Ordinal) < 0) return text;
        var sb = new System.Text.StringBuilder(text.Length);
        int i = 0;
        while (i < text.Length)
        {
            int f = text.IndexOf("```", i, StringComparison.Ordinal);
            if (f < 0) { sb.Append(text, i, text.Length - i); break; }
            sb.Append(text, i, f - i);
            int nl = text.IndexOf('\n', f + 3);                 // skip an optional language tag on the fence line
            int afterFence = nl < 0 ? f + 3 : nl + 1;
            int closeFence = text.IndexOf("```", afterFence, StringComparison.Ordinal);
            if (closeFence < 0) { sb.Append(text, afterFence, text.Length - afterFence); break; }
            sb.Append(text, afterFence, closeFence - afterFence);
            i = closeFence + 3;
        }
        return sb.ToString();
    }

    /// <summary>Yield each balanced top-level <c>{…}</c> run in <paramref name="text"/>. String-aware: braces and
    /// quotes inside JSON string literals are ignored, so embedded prose/braces don't break extraction.</summary>
    private static IEnumerable<string> ExtractTopLevelJsonObjects(string text)
    {
        int depth = 0, start = -1; bool inStr = false, esc = false;
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (inStr)
            {
                if (esc) esc = false;
                else if (c == '\\') esc = true;
                else if (c == '"') inStr = false;
                continue;
            }
            if (c == '"') inStr = true;
            else if (c == '{') { if (depth++ == 0) start = i; }
            else if (c == '}' && depth > 0 && --depth == 0 && start >= 0)
            {
                yield return text.Substring(start, i - start + 1);
                start = -1;
            }
        }
    }

    // ── gemma4 native tool-call parsing ──────────────────────────────────────────────────────────────
    private const string Gemma4Quote = "<|\"|>";   // gemma4's string-quote control token

    /// <summary>Extract gemma4 tool calls from generated text. Each is
    /// <c>&lt;|tool_call&gt;call:NAME{key:&lt;|"|&gt;val&lt;|"|&gt;,key:bare,…}&lt;tool_call|&gt;</c>; the args
    /// DSL is converted to a JSON object string so the result matches the ChatML path's shape.</summary>
    private static void ParseGemma4ToolCalls(string text, List<ParsedToolCall> calls)
    {
        const string open = "<|tool_call>call:";
        int i = 0;
        while (true)
        {
            int s = text.IndexOf(open, i, StringComparison.Ordinal);
            if (s < 0) break;
            int ns = s + open.Length;
            int brace = text.IndexOf('{', ns);
            if (brace < 0) break;
            string name = text.Substring(ns, brace - ns).Trim();
            var (json, next) = ParseGemmaObject(text, brace);
            if (json != null && name.Length > 0) calls.Add(new ParsedToolCall(name, json));
            i = next > s ? next : ns;
        }
    }

    /// <summary>Parse a gemma DSL object starting at <c>text[pos]=='{'</c> into a JSON object string.
    /// Returns the JSON and the index just past the matching <c>}</c>.</summary>
    private static (string?, int) ParseGemmaObject(string t, int pos)
    {
        if (pos >= t.Length || t[pos] != '{') return (null, pos);
        var sb = new System.Text.StringBuilder("{");
        int i = pos + 1;
        bool first = true;
        while (i < t.Length && t[i] != '}')
        {
            while (i < t.Length && (char.IsWhiteSpace(t[i]) || t[i] == ',')) i++;
            if (i >= t.Length || t[i] == '}') break;
            int colon = t.IndexOf(':', i);
            if (colon < 0) break;
            string key = t.Substring(i, colon - i).Trim();
            i = colon + 1;
            var (vjson, vnext) = ParseGemmaValue(t, i);
            i = vnext;
            if (key.Length == 0) continue;
            if (!first) sb.Append(',');
            first = false;
            sb.Append(System.Text.Json.JsonSerializer.Serialize(key)).Append(':').Append(vjson);
        }
        if (i < t.Length && t[i] == '}') i++;
        sb.Append('}');
        return (sb.ToString(), i);
    }

    /// <summary>Parse a single gemma DSL value (object / array / <c>&lt;|"|&gt;</c>-quoted string / bare
    /// number|bool|null) into a JSON value string. Returns the JSON and the index past the value.</summary>
    private static (string, int) ParseGemmaValue(string t, int i)
    {
        while (i < t.Length && char.IsWhiteSpace(t[i])) i++;
        if (i < t.Length && t[i] == '{') { var (o, n) = ParseGemmaObject(t, i); return (o ?? "{}", n); }
        if (i < t.Length && t[i] == '[')
        {
            var sb = new System.Text.StringBuilder("[");
            i++; bool first = true;
            while (i < t.Length && t[i] != ']')
            {
                while (i < t.Length && (char.IsWhiteSpace(t[i]) || t[i] == ',')) i++;
                if (i >= t.Length || t[i] == ']') break;
                var (v, n) = ParseGemmaValue(t, i);
                if (!first) sb.Append(','); first = false;
                sb.Append(v); i = n;
            }
            if (i < t.Length && t[i] == ']') i++;
            sb.Append(']');
            return (sb.ToString(), i);
        }
        int q = Gemma4Quote.Length;
        if (i + q <= t.Length && string.CompareOrdinal(t, i, Gemma4Quote, 0, q) == 0)
        {
            int start = i + q;
            int end = t.IndexOf(Gemma4Quote, start, StringComparison.Ordinal);
            if (end < 0) end = t.Length;
            string raw = t.Substring(start, end - start);
            return (System.Text.Json.JsonSerializer.Serialize(raw), end < t.Length ? end + q : end);
        }
        // bareword: number / true / false / null (else fall back to a JSON string)
        int j = i;
        while (j < t.Length && t[j] != ',' && t[j] != '}' && t[j] != ']') j++;
        string w = t.Substring(i, j - i).Trim();
        if (w == "true" || w == "false" || w == "null") return (w, j);
        if (double.TryParse(w, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out _)) return (w, j);
        return (System.Text.Json.JsonSerializer.Serialize(w), j);
    }
}
