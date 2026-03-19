using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Auto-loads tokenizers from HuggingFace model repositories.
/// Supports the standard HuggingFace tokenizer file formats:
/// - tokenizer.json (new format — used by most models)
/// - vocab.json + merges.txt (legacy BPE format — GPT-2, CLIP)
/// - sentencepiece (not yet supported)
///
/// Usage:
/// <code>
/// var tokenizer = await TokenizerLoader.LoadAsync(httpClient, "models/distilbert-sst2");
/// var tokens = tokenizer.Encode("Hello world");
/// </code>
/// </summary>
public static class TokenizerLoader
{
    /// <summary>
    /// Load a tokenizer from a model directory (local or HTTP).
    /// Tries tokenizer.json first, then vocab.json + merges.txt.
    /// </summary>
    public static async Task<LoadedTokenizer> LoadAsync(HttpClient http, string basePath)
    {
        basePath = basePath.TrimEnd('/');

        // Try tokenizer.json (HuggingFace fast tokenizer format)
        try
        {
            var tokenizerJson = await http.GetStringAsync($"{basePath}/tokenizer.json");
            return ParseTokenizerJson(tokenizerJson);
        }
        catch { }

        // Fall back to vocab.json + merges.txt (legacy BPE)
        try
        {
            var vocabJson = await http.GetStringAsync($"{basePath}/vocab.json");
            var mergesText = await http.GetStringAsync($"{basePath}/merges.txt");
            return ParseVocabAndMerges(vocabJson, mergesText);
        }
        catch { }

        // Try tokenizer_config.json for special tokens info
        try
        {
            var configJson = await http.GetStringAsync($"{basePath}/tokenizer_config.json");
            // If we get here, there's a tokenizer config but no vocab — might be sentencepiece
            throw new NotSupportedException("SentencePiece tokenizers are not yet supported. Only BPE (vocab.json + merges.txt) and HuggingFace fast tokenizer (tokenizer.json) formats are supported.");
        }
        catch (NotSupportedException) { throw; }
        catch { }

        throw new FileNotFoundException($"No tokenizer files found at {basePath}. Expected tokenizer.json or vocab.json + merges.txt");
    }

    /// <summary>
    /// Parse HuggingFace fast tokenizer format (tokenizer.json).
    /// This is a complex JSON structure but we only need the model vocabulary and merges.
    /// </summary>
    private static LoadedTokenizer ParseTokenizerJson(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        // Extract vocabulary from model.vocab
        var vocab = new Dictionary<string, int>();
        if (root.TryGetProperty("model", out var model))
        {
            if (model.TryGetProperty("vocab", out var vocabObj))
            {
                foreach (var prop in vocabObj.EnumerateObject())
                {
                    vocab[prop.Name] = prop.Value.GetInt32();
                }
            }
        }

        // Extract merges from model.merges
        var merges = new List<string>();
        if (root.TryGetProperty("model", out var model2))
        {
            if (model2.TryGetProperty("merges", out var mergesArr))
            {
                foreach (var merge in mergesArr.EnumerateArray())
                {
                    merges.Add(merge.GetString()!);
                }
            }
        }

        // Extract special tokens
        var specialTokens = new Dictionary<string, int>();
        if (root.TryGetProperty("added_tokens", out var addedTokens))
        {
            foreach (var token in addedTokens.EnumerateArray())
            {
                if (token.TryGetProperty("content", out var content) &&
                    token.TryGetProperty("id", out var id))
                {
                    specialTokens[content.GetString()!] = id.GetInt32();
                    vocab[content.GetString()!] = id.GetInt32();
                }
            }
        }

        // Extract config
        int? padTokenId = null;
        int? eosTokenId = null;
        int? bosTokenId = null;

        if (root.TryGetProperty("post_processor", out var postProc))
        {
            // Try to find special token IDs from post_processor config
            if (postProc.TryGetProperty("special_tokens", out var specToks))
            {
                foreach (var prop in specToks.EnumerateObject())
                {
                    if (prop.Value.TryGetProperty("ids", out var ids))
                    {
                        var idArr = ids.EnumerateArray().Select(x => x.GetInt32()).ToArray();
                        if (idArr.Length > 0)
                        {
                            var name = prop.Name.ToLowerInvariant();
                            if (name.Contains("eos") || name.Contains("sep") || name.Contains("end"))
                                eosTokenId = idArr[0];
                            else if (name.Contains("bos") || name.Contains("cls") || name.Contains("start"))
                                bosTokenId = idArr[0];
                        }
                    }
                }
            }
        }

        // Look for pad token in added_tokens
        foreach (var (token, id) in specialTokens)
        {
            var lower = token.ToLowerInvariant();
            if (lower.Contains("pad")) padTokenId = id;
            if (lower.Contains("eos") || lower.Contains("</s>") || lower == "[sep]") eosTokenId ??= id;
            if (lower.Contains("bos") || lower.Contains("<s>") || lower == "[cls]") bosTokenId ??= id;
        }

        var bpe = new BPETokenizer(vocab, merges.ToArray());

        return new LoadedTokenizer
        {
            Tokenizer = bpe,
            VocabSize = vocab.Count,
            PadTokenId = padTokenId ?? 0,
            EosTokenId = eosTokenId ?? -1,
            BosTokenId = bosTokenId ?? -1,
            SpecialTokens = specialTokens,
        };
    }

    /// <summary>
    /// Parse legacy vocab.json + merges.txt (GPT-2, CLIP style).
    /// </summary>
    private static LoadedTokenizer ParseVocabAndMerges(string vocabJson, string mergesText)
    {
        var bpe = BPETokenizer.Load(vocabJson, mergesText);

        // Try to identify special tokens from vocab
        var specialTokens = new Dictionary<string, int>();
        // Common special tokens
        string[] specialNames = { "<|endoftext|>", "<|startoftext|>", "<|padding|>", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]", "<s>", "</s>", "<pad>" };

        // Parse vocab to find special tokens
        using var doc = JsonDocument.Parse(vocabJson);
        foreach (var name in specialNames)
        {
            if (doc.RootElement.TryGetProperty(name, out var val))
            {
                specialTokens[name] = val.GetInt32();
            }
        }

        int? padTokenId = specialTokens.GetValueOrDefault("<pad>", specialTokens.GetValueOrDefault("[PAD]", 0));
        int? eosTokenId = specialTokens.GetValueOrDefault("<|endoftext|>", specialTokens.GetValueOrDefault("</s>", specialTokens.GetValueOrDefault("[SEP]", -1)));
        int? bosTokenId = specialTokens.GetValueOrDefault("<|startoftext|>", specialTokens.GetValueOrDefault("<s>", specialTokens.GetValueOrDefault("[CLS]", -1)));

        return new LoadedTokenizer
        {
            Tokenizer = bpe,
            VocabSize = bpe.VocabSize,
            PadTokenId = padTokenId ?? 0,
            EosTokenId = eosTokenId ?? -1,
            BosTokenId = bosTokenId ?? -1,
            SpecialTokens = specialTokens,
        };
    }
}

/// <summary>
/// A loaded tokenizer with all metadata needed for inference.
/// </summary>
public class LoadedTokenizer
{
    /// <summary>The BPE tokenizer for encoding/decoding.</summary>
    public BPETokenizer Tokenizer { get; init; } = null!;

    /// <summary>Total vocabulary size.</summary>
    public int VocabSize { get; init; }

    /// <summary>Padding token ID.</summary>
    public int PadTokenId { get; init; }

    /// <summary>End-of-sequence token ID (-1 if not set).</summary>
    public int EosTokenId { get; init; }

    /// <summary>Beginning-of-sequence token ID (-1 if not set).</summary>
    public int BosTokenId { get; init; }

    /// <summary>Special tokens map (name → ID).</summary>
    public Dictionary<string, int> SpecialTokens { get; init; } = new();

    /// <summary>Encode text to token IDs.</summary>
    public int[] Encode(string text) => Tokenizer.Encode(text);

    /// <summary>Decode token IDs to text.</summary>
    public string Decode(int[] tokenIds) => Tokenizer.Decode(tokenIds);

    /// <summary>
    /// Encode with padding and attention mask for model input.
    /// </summary>
    public (int[] InputIds, int[] AttentionMask) EncodeForModel(string text, int maxLength = 128)
    {
        var tokens = new List<int>();

        // Add BOS if the model uses it
        if (BosTokenId >= 0) tokens.Add(BosTokenId);

        tokens.AddRange(Tokenizer.Encode(text));

        // Add EOS if the model uses it
        if (EosTokenId >= 0) tokens.Add(EosTokenId);

        // Truncate
        if (tokens.Count > maxLength)
            tokens = tokens.Take(maxLength).ToList();

        var inputIds = TextPreprocessor.PadOrTruncate(tokens.ToArray(), maxLength, PadTokenId);
        var attentionMask = TextPreprocessor.CreateAttentionMask(inputIds, PadTokenId);

        return (inputIds, attentionMask);
    }

    /// <summary>
    /// Encode for CLIP (start token + text + end token, padded to maxLength).
    /// </summary>
    public int[] EncodeForCLIP(string text, int maxLength = 77)
    {
        return Tokenizer.EncodeCLIP(text, maxLength);
    }
}
