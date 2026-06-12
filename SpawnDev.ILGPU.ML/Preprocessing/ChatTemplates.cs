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
}
