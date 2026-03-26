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
}
