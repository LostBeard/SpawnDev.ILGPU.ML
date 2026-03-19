using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Auto-detect model configuration from HuggingFace model files.
/// Reads config.json and preprocessor_config.json to determine:
/// - Input/output tensor names and shapes
/// - Preprocessing requirements (normalization, resize, crop)
/// - Model architecture and task type
///
/// This enables "just point at a model and run" without manual configuration.
/// </summary>
public static class ModelAutoConfig
{
    /// <summary>
    /// Load and auto-detect model configuration from a model directory.
    /// Tries config.json, preprocessor_config.json, and tokenizer_config.json.
    /// </summary>
    public static async Task<AutoDetectedConfig> DetectAsync(HttpClient http, string basePath)
    {
        basePath = basePath.TrimEnd('/');
        var config = new AutoDetectedConfig();

        // Try config.json (model architecture)
        try
        {
            var json = await http.GetStringAsync($"{basePath}/config.json");
            ParseModelConfig(json, config);
        }
        catch { }

        // Try preprocessor_config.json (image preprocessing)
        try
        {
            var json = await http.GetStringAsync($"{basePath}/preprocessor_config.json");
            ParsePreprocessorConfig(json, config);
        }
        catch { }

        // Try tokenizer_config.json (text preprocessing)
        try
        {
            var json = await http.GetStringAsync($"{basePath}/tokenizer_config.json");
            ParseTokenizerConfig(json, config);
        }
        catch { }

        // Infer task from architecture
        config.InferredTask = InferTaskFromArchitecture(config.ModelType, config.Architectures);

        return config;
    }

    private static void ParseModelConfig(string json, AutoDetectedConfig config)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        if (root.TryGetProperty("model_type", out var mt))
            config.ModelType = mt.GetString() ?? "";

        if (root.TryGetProperty("architectures", out var archs))
            config.Architectures = archs.EnumerateArray().Select(a => a.GetString() ?? "").ToArray();

        // Vision model properties
        if (root.TryGetProperty("image_size", out var imgSize))
            config.ImageSize = imgSize.GetInt32();
        if (root.TryGetProperty("patch_size", out var patchSize))
            config.PatchSize = patchSize.GetInt32();
        if (root.TryGetProperty("num_channels", out var numCh))
            config.NumChannels = numCh.GetInt32();
        if (root.TryGetProperty("num_labels", out var numLabels))
            config.NumLabels = numLabels.GetInt32();

        // Transformer properties
        if (root.TryGetProperty("hidden_size", out var hs))
            config.HiddenSize = hs.GetInt32();
        if (root.TryGetProperty("num_hidden_layers", out var nhl))
            config.NumLayers = nhl.GetInt32();
        if (root.TryGetProperty("num_attention_heads", out var nah))
            config.NumHeads = nah.GetInt32();
        if (root.TryGetProperty("intermediate_size", out var ims))
            config.IntermediateSize = ims.GetInt32();
        if (root.TryGetProperty("vocab_size", out var vs))
            config.VocabSize = vs.GetInt32();
        if (root.TryGetProperty("max_position_embeddings", out var mpe))
            config.MaxPositionEmbeddings = mpe.GetInt32();

        // Generation config
        if (root.TryGetProperty("eos_token_id", out var eos))
        {
            if (eos.ValueKind == JsonValueKind.Number) config.EosTokenId = eos.GetInt32();
            else if (eos.ValueKind == JsonValueKind.Array) config.EosTokenId = eos[0].GetInt32();
        }
        if (root.TryGetProperty("bos_token_id", out var bos))
            config.BosTokenId = bos.GetInt32();
        if (root.TryGetProperty("pad_token_id", out var pad))
            config.PadTokenId = pad.GetInt32();

        // Label mapping
        if (root.TryGetProperty("id2label", out var id2label))
        {
            config.Labels = new Dictionary<int, string>();
            foreach (var prop in id2label.EnumerateObject())
            {
                if (int.TryParse(prop.Name, out int id))
                    config.Labels[id] = prop.Value.GetString() ?? "";
            }
        }
    }

    private static void ParsePreprocessorConfig(string json, AutoDetectedConfig config)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        if (root.TryGetProperty("image_mean", out var mean))
            config.ImageMean = mean.EnumerateArray().Select(v => v.GetSingle()).ToArray();
        if (root.TryGetProperty("image_std", out var std))
            config.ImageStd = std.EnumerateArray().Select(v => v.GetSingle()).ToArray();

        if (root.TryGetProperty("size", out var size))
        {
            if (size.TryGetProperty("height", out var h) && size.TryGetProperty("width", out var w))
            {
                config.ImageHeight = h.GetInt32();
                config.ImageWidth = w.GetInt32();
            }
            else if (size.TryGetProperty("shortest_edge", out var se))
            {
                config.ImageSize = se.GetInt32();
            }
        }

        if (root.TryGetProperty("do_resize", out var doResize))
            config.DoResize = doResize.GetBoolean();
        if (root.TryGetProperty("do_center_crop", out var doCrop))
            config.DoCenterCrop = doCrop.GetBoolean();
        if (root.TryGetProperty("do_normalize", out var doNorm))
            config.DoNormalize = doNorm.GetBoolean();
        if (root.TryGetProperty("do_rescale", out var doRescale))
            config.DoRescale = doRescale.GetBoolean();

        if (root.TryGetProperty("crop_size", out var cropSize))
        {
            if (cropSize.TryGetProperty("height", out var ch) && cropSize.TryGetProperty("width", out var cw))
            {
                config.CropHeight = ch.GetInt32();
                config.CropWidth = cw.GetInt32();
            }
        }

        if (root.TryGetProperty("feature_extractor_type", out var fet))
            config.FeatureExtractorType = fet.GetString() ?? "";

        // Audio-specific
        if (root.TryGetProperty("sampling_rate", out var sr))
            config.SamplingRate = sr.GetInt32();
        if (root.TryGetProperty("feature_size", out var fs))
            config.FeatureSize = fs.GetInt32();
    }

    private static void ParseTokenizerConfig(string json, AutoDetectedConfig config)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        if (root.TryGetProperty("model_max_length", out var mml))
        {
            if (mml.ValueKind == JsonValueKind.Number)
                config.ModelMaxLength = mml.GetInt32();
        }
        if (root.TryGetProperty("tokenizer_class", out var tc))
            config.TokenizerClass = tc.GetString() ?? "";

        config.HasTokenizer = true;
    }

    /// <summary>
    /// Infer the pipeline task from model architecture.
    /// </summary>
    private static string InferTaskFromArchitecture(string modelType, string[] architectures)
    {
        var allNames = string.Join(" ", architectures).ToLowerInvariant() + " " + modelType.ToLowerInvariant();

        if (allNames.Contains("forsequenceclassification") || allNames.Contains("forsst"))
            return "text-classification";
        if (allNames.Contains("fortokenclassification"))
            return "token-classification";
        if (allNames.Contains("forquestionanswering"))
            return "question-answering";
        if (allNames.Contains("formaskedlm"))
            return "fill-mask";
        if (allNames.Contains("forcausallm") || allNames.Contains("forgenerationwithpast"))
            return "text-generation";
        if (allNames.Contains("forconditionalgeneration") && allNames.Contains("whisper"))
            return "automatic-speech-recognition";
        if (allNames.Contains("forconditionalgeneration"))
            return "text2text-generation";
        if (allNames.Contains("forimageclassification"))
            return "image-classification";
        if (allNames.Contains("forobjectdetection"))
            return "object-detection";
        if (allNames.Contains("forsemanticsegmentation") || allNames.Contains("forpanopticsegmentation"))
            return "image-segmentation";
        if (allNames.Contains("fordepthestimation") || allNames.Contains("dpt"))
            return "depth-estimation";
        if (allNames.Contains("foraudioclassification"))
            return "audio-classification";
        if (allNames.Contains("clip") || allNames.Contains("siglip"))
            return "zero-shot-image-classification";
        if (allNames.Contains("dino") || allNames.Contains("vit") && !allNames.Contains("classification"))
            return "image-feature-extraction";

        // Model type fallbacks
        return modelType.ToLowerInvariant() switch
        {
            "whisper" => "automatic-speech-recognition",
            "gpt2" or "llama" or "qwen2" or "mistral" or "phi" => "text-generation",
            "bert" or "distilbert" or "roberta" => "text-classification",
            "t5" or "bart" or "mbart" => "text2text-generation",
            "vit" or "mobilenet_v2" or "convnext" or "squeezenet" => "image-classification",
            "detr" or "yolos" => "object-detection",
            "dpt" or "depth_anything" => "depth-estimation",
            "segformer" or "mask2former" => "image-segmentation",
            "clip" or "siglip" => "zero-shot-image-classification",
            "wav2vec2" or "hubert" => "audio-classification",
            _ => "unknown",
        };
    }

    /// <summary>
    /// Convert auto-detected config to a ModelConfig for preprocessing.
    /// </summary>
    public static ModelConfig ToModelConfig(AutoDetectedConfig auto)
    {
        int width = auto.CropWidth > 0 ? auto.CropWidth : (auto.ImageWidth > 0 ? auto.ImageWidth : (auto.ImageSize > 0 ? auto.ImageSize : 224));
        int height = auto.CropHeight > 0 ? auto.CropHeight : (auto.ImageHeight > 0 ? auto.ImageHeight : (auto.ImageSize > 0 ? auto.ImageSize : 224));

        return new ModelConfig
        {
            Name = auto.ModelType,
            InputWidth = width,
            InputHeight = height,
            NormalizeMean = auto.ImageMean,
            NormalizeStd = auto.ImageStd,
            ScaleTo01 = auto.DoRescale,
        };
    }
}

/// <summary>
/// Auto-detected model configuration from HuggingFace config files.
/// </summary>
public class AutoDetectedConfig
{
    // Model architecture
    public string ModelType { get; set; } = "";
    public string[] Architectures { get; set; } = Array.Empty<string>();
    public string InferredTask { get; set; } = "unknown";

    // Vision
    public int ImageSize { get; set; }
    public int ImageWidth { get; set; }
    public int ImageHeight { get; set; }
    public int PatchSize { get; set; }
    public int NumChannels { get; set; } = 3;
    public int NumLabels { get; set; }
    public float[]? ImageMean { get; set; }
    public float[]? ImageStd { get; set; }
    public bool DoResize { get; set; } = true;
    public bool DoCenterCrop { get; set; }
    public bool DoNormalize { get; set; } = true;
    public bool DoRescale { get; set; } = true;
    public int CropWidth { get; set; }
    public int CropHeight { get; set; }
    public string FeatureExtractorType { get; set; } = "";

    // Transformer
    public int HiddenSize { get; set; }
    public int NumLayers { get; set; }
    public int NumHeads { get; set; }
    public int IntermediateSize { get; set; }
    public int VocabSize { get; set; }
    public int MaxPositionEmbeddings { get; set; }

    // Tokens
    public int EosTokenId { get; set; } = -1;
    public int BosTokenId { get; set; } = -1;
    public int PadTokenId { get; set; }

    // Labels
    public Dictionary<int, string>? Labels { get; set; }

    // Tokenizer
    public bool HasTokenizer { get; set; }
    public string TokenizerClass { get; set; } = "";
    public int ModelMaxLength { get; set; }

    // Audio
    public int SamplingRate { get; set; }
    public int FeatureSize { get; set; }

    /// <summary>Build a KVCacheConfig from this model's architecture.</summary>
    public KVCacheConfig? ToKVCacheConfig()
    {
        if (NumLayers == 0 || NumHeads == 0 || HiddenSize == 0) return null;
        return new KVCacheConfig
        {
            NumLayers = NumLayers,
            NumHeads = NumHeads,
            HeadDim = HiddenSize / NumHeads,
            MaxSeqLen = MaxPositionEmbeddings > 0 ? MaxPositionEmbeddings : 2048,
        };
    }
}
