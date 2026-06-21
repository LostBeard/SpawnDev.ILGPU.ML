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
        IReadOnlyList<(string Role, string Content)> messages, bool thinking = true, bool addGenerationPrompt = true)
    {
        int Id(string s) => tok.TryGetId(s, out var v) ? v : -1;
        int bos = Id("<bos>"), turnO = Id("<|turn>"), turnC = Id("<turn|>"), think = Id("<|think|>");
        var ids = new List<int>();
        if (bos >= 0) ids.Add(bos);
        bool armed = false;
        foreach (var (role, content) in messages)
        {
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
        // Tool-calling (v2): inject the tool signatures into the system message per format, so the model knows
        // it may emit <tool_call>…</tool_call>. The generated text is later scanned by ParseToolCalls.
        if (toolsJson != null && toolsJson.Count > 0)
        {
            var aug = new List<(string Role, string Content)>(messages);
            string? sys = aug.Count > 0 && aug[0].Role == "system" ? aug[0].Content : null;
            string toolSys = BuildToolSystem(DetectChatFormat(model), sys, toolsJson);
            if (aug.Count > 0 && aug[0].Role == "system") aug[0] = ("system", toolSys);
            else aug.Insert(0, ("system", toolSys));
            messages = aug;
        }

        switch (DetectChatFormat(model))
        {
            case ChatFormat.Gemma4:
                return (BuildGemma4MultiTurnPromptTokens(tok, messages, thinking),
                        Ids(tok, "<turn|>"));
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
        var sb = new System.Text.StringBuilder();
        sb.Append(string.IsNullOrWhiteSpace(systemContent) ? "You are a helpful assistant." : systemContent);
        sb.Append("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n");
        sb.Append("You are provided with function signatures within <tools></tools> XML tags:\n<tools>");
        foreach (var t in toolsJson) { sb.Append('\n'); sb.Append(t); }
        sb.Append("\n</tools>\n\nFor each function call, return a json object with function name and arguments ");
        sb.Append("within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, ");
        sb.Append("\"arguments\": <args-json-object>}\n</tool_call>");
        return sb.ToString();
    }

    /// <summary>Extract tool calls from generated text. Handles the <c>&lt;tool_call&gt;{json}&lt;/tool_call&gt;</c>
    /// convention (ChatML/qwen, and the default used by <see cref="BuildToolSystem"/>). Malformed blocks are
    /// skipped. Returns an empty list when the model produced plain text.</summary>
    public static List<ParsedToolCall> ParseToolCalls(string text)
    {
        var calls = new List<ParsedToolCall>();
        const string open = "<tool_call>", close = "</tool_call>";
        int i = 0;
        while (true)
        {
            int s = text.IndexOf(open, i, StringComparison.Ordinal);
            if (s < 0) break;
            int e = text.IndexOf(close, s + open.Length, StringComparison.Ordinal);
            if (e < 0) break;
            string inner = text.Substring(s + open.Length, e - s - open.Length).Trim();
            try
            {
                using var doc = System.Text.Json.JsonDocument.Parse(inner);
                var root = doc.RootElement;
                string name = root.TryGetProperty("name", out var n) ? n.GetString() ?? "" : "";
                string args = root.TryGetProperty("arguments", out var a) ? a.GetRawText() : "{}";
                if (!string.IsNullOrEmpty(name)) calls.Add(new ParsedToolCall(name, args));
            }
            catch { /* skip a malformed block rather than fail the whole response */ }
            i = e + close.Length;
        }
        return calls;
    }
}
