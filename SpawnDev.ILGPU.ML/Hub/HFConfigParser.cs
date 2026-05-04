using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// Parses HuggingFace config.json to extract model architecture parameters.
/// Used by SafeTensors and PyTorch loaders to build the computation graph
/// when only weights + config are available (no ONNX graph).
///
/// config.json fields vary by model type but common patterns:
/// - model_type: "llama", "gpt2", "bert", "vit", etc.
/// - hidden_size / d_model: embedding dimension
/// - num_hidden_layers / n_layer: number of transformer blocks
/// - num_attention_heads / n_head: attention heads
/// - intermediate_size / n_inner: FFN hidden dimension
/// - vocab_size: vocabulary size
/// - max_position_embeddings: max sequence length
/// </summary>
public class HFModelConfig
{
    public string ModelType { get; set; } = "";
    public int HiddenSize { get; set; }
    public int NumHiddenLayers { get; set; }
    public int NumAttentionHeads { get; set; }
    public int NumKeyValueHeads { get; set; }
    public int IntermediateSize { get; set; }
    public int VocabSize { get; set; }
    public int MaxPositionEmbeddings { get; set; }
    public string HiddenAct { get; set; } = "gelu";
    public float RmsNormEps { get; set; } = 1e-6f;
    public float LayerNormEps { get; set; } = 1e-5f;
    public bool TieWordEmbeddings { get; set; } = false;
    public string ArchitectureFamily { get; set; } = ""; // decoder, encoder, encoder-decoder, vision

    /// <summary>Parse config.json content into a structured config object.</summary>
    public static HFModelConfig Parse(string json)
    {
        var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        var config = new HFModelConfig();

        config.ModelType = GetString(root, "model_type", "");
        config.HiddenSize = GetInt(root, "hidden_size", GetInt(root, "d_model", GetInt(root, "n_embd", 768)));
        config.NumHiddenLayers = GetInt(root, "num_hidden_layers", GetInt(root, "n_layer", GetInt(root, "num_layers", 12)));
        config.NumAttentionHeads = GetInt(root, "num_attention_heads", GetInt(root, "n_head", GetInt(root, "num_heads", 12)));
        config.NumKeyValueHeads = GetInt(root, "num_key_value_heads", config.NumAttentionHeads);
        config.IntermediateSize = GetInt(root, "intermediate_size", GetInt(root, "n_inner", config.HiddenSize * 4));
        config.VocabSize = GetInt(root, "vocab_size", 32000);
        config.MaxPositionEmbeddings = GetInt(root, "max_position_embeddings", GetInt(root, "n_positions", 2048));
        config.HiddenAct = GetString(root, "hidden_act", GetString(root, "activation_function", "gelu"));
        config.RmsNormEps = GetFloat(root, "rms_norm_eps", 1e-6f);
        config.LayerNormEps = GetFloat(root, "layer_norm_eps", GetFloat(root, "layer_norm_epsilon", 1e-5f));
        config.TieWordEmbeddings = GetBool(root, "tie_word_embeddings", false);

        // Determine architecture family from model_type
        config.ArchitectureFamily = config.ModelType.ToLowerInvariant() switch
        {
            "llama" or "mistral" or "phi" or "phi3" or "qwen2" or "gemma" or "gemma2"
                or "starcoder2" or "codellama" or "falcon" => "decoder",
            "gpt2" or "gpt_neo" or "gpt_neox" or "opt" or "bloom" => "decoder",
            "bert" or "distilbert" or "roberta" or "albert" or "electra" or "deberta" => "encoder",
            "t5" or "bart" or "mbart" or "pegasus" or "marian" => "encoder-decoder",
            "vit" or "deit" or "beit" or "swin" or "convnext" => "vision",
            "whisper" => "encoder-decoder",
            "clip" => "vision",
            _ => "decoder" // default assumption
        };

        return config;
    }

    /// <summary>
    /// Determine tensor name prefix pattern for this architecture.
    /// Different HF models use different naming conventions for weight tensors.
    /// </summary>
    public string GetLayerPrefix(int layer) => ModelType.ToLowerInvariant() switch
    {
        "llama" or "mistral" or "qwen2" or "gemma" or "gemma2" or "phi" or "phi3"
            => $"model.layers.{layer}",
        "gpt2" => $"transformer.h.{layer}",
        "gpt_neo" or "gpt_neox" => $"gpt_neox.layers.{layer}",
        "bert" or "distilbert" or "roberta" => $"bert.encoder.layer.{layer}",
        "t5" => $"encoder.block.{layer}",
        "vit" or "deit" => $"vit.encoder.layer.{layer}",
        _ => $"model.layers.{layer}"
    };

    public string GetEmbeddingName() => ModelType.ToLowerInvariant() switch
    {
        "llama" or "mistral" or "qwen2" or "gemma" or "phi" => "model.embed_tokens.weight",
        "gpt2" => "transformer.wte.weight",
        "bert" or "distilbert" => "embeddings.word_embeddings.weight",
        _ => "model.embed_tokens.weight"
    };

    public string GetFinalNormName() => ModelType.ToLowerInvariant() switch
    {
        "llama" or "mistral" or "qwen2" or "gemma" or "phi" => "model.norm.weight",
        "gpt2" => "transformer.ln_f.weight",
        "bert" => "bert.encoder.layer_norm.weight",
        _ => "model.norm.weight"
    };

    public string GetLMHeadName() => ModelType.ToLowerInvariant() switch
    {
        "llama" or "mistral" or "qwen2" or "gemma" => "lm_head.weight",
        "gpt2" => "lm_head.weight",
        _ => "lm_head.weight"
    };

    public bool UsesRMSNorm => ModelType.ToLowerInvariant() is
        "llama" or "mistral" or "qwen2" or "gemma" or "gemma2";

    public bool UsesSiLU => HiddenAct is "silu" or "swiglu";
    public bool UsesGELU => HiddenAct is "gelu" or "gelu_new" or "gelu_fast";

    private static int GetInt(JsonElement e, string key, int def)
    {
        if (e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number) return v.GetInt32();
        return def;
    }
    private static float GetFloat(JsonElement e, string key, float def)
    {
        if (e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number) return v.GetSingle();
        return def;
    }
    private static string GetString(JsonElement e, string key, string def)
    {
        if (e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String) return v.GetString() ?? def;
        return def;
    }
    private static bool GetBool(JsonElement e, string key, bool def)
    {
        if (e.TryGetProperty(key, out var v))
        {
            if (v.ValueKind == JsonValueKind.True) return true;
            if (v.ValueKind == JsonValueKind.False) return false;
        }
        return def;
    }
}
