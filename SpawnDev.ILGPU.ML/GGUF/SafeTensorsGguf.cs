using System.Text.Json;

namespace SpawnDev.ILGPU.ML.GGUF;

/// <summary>
/// A Hugging Face causal language model in a safetensors file presented as a <see cref="GGUFModel"/> over the SAME bytes, so the GGUF text engines
/// (<see cref="Pipelines.GgufGenerator"/>, the batch decoder) run it without a conversion or a copy: the tensor descriptors point into the safetensors data
/// section (<see cref="GGUFModel.DataStartOffset"/> = its start), names and dimension order are rewritten to GGUF's, and the hyper-parameters and the tokenizer
/// are filled in from the tensor shapes and the model's <c>tokenizer.json</c>. Pass the same stream to the engine.
/// <para>The files this is for are ComfyUI's text-encoder files, which ComfyUI's Generate Text node also runs as language models (<c>qwen_3_4b.safetensors</c>:
/// Qwen3-4B, all 36 layers and the final norm, bf16, no head: the head is tied to the embedding, as the GGUF engines already handle).</para>
/// </summary>
public static class SafeTensorsGguf
{
    private static readonly (string Hf, string Gguf)[] LayerNames =
    {
        ("input_layernorm.weight", "attn_norm.weight"),
        ("post_attention_layernorm.weight", "ffn_norm.weight"),
        ("self_attn.q_proj.weight", "attn_q.weight"),
        ("self_attn.k_proj.weight", "attn_k.weight"),
        ("self_attn.v_proj.weight", "attn_v.weight"),
        ("self_attn.o_proj.weight", "attn_output.weight"),
        ("self_attn.q_norm.weight", "attn_q_norm.weight"),
        ("self_attn.k_norm.weight", "attn_k_norm.weight"),
        ("self_attn.q_proj.bias", "attn_q.bias"),
        ("self_attn.k_proj.bias", "attn_k.bias"),
        ("self_attn.v_proj.bias", "attn_v.bias"),
        ("mlp.gate_proj.weight", "ffn_gate.weight"),
        ("mlp.up_proj.weight", "ffn_up.weight"),
        ("mlp.down_proj.weight", "ffn_down.weight"),
    };

    /// <summary>
    /// Read the header of <paramref name="file"/> (a Hugging Face Qwen3 causal LM: <c>model.embed_tokens</c>, <c>model.layers.N.*</c>, <c>model.norm</c>, optional
    /// <c>lm_head</c>; any prefix in front of <c>model.</c> is allowed) and describe it as a GGUF <c>qwen3</c> model. The file holds no hyper-parameters that its shapes
    /// do not show, so the rotary base and the norm epsilon are Qwen3's (<paramref name="ropeTheta"/>, <paramref name="rmsEps"/>). <paramref name="tokenizerJson"/>:
    /// the model's <c>tokenizer.json</c> text (byte-level BPE); <paramref name="chatTemplate"/>: its chat template (tokenizer_config.json), which the chat helpers read
    /// to choose the turn format and the thinking switch.
    /// </summary>
    public static async Task<GGUFModel> FromHfQwen3Async(Stream file, string tokenizerJson, string? chatTemplate = null, string name = "qwen3",
        float ropeTheta = 1_000_000f, float rmsEps = 1e-6f, long contextLength = 40960, CancellationToken ct = default)
    {
        if (!file.CanSeek) throw new ArgumentException("the tensors are read by offset: a seekable stream is required", nameof(file));
        file.Seek(0, SeekOrigin.Begin);
        var lenBytes = new byte[8]; await file.ReadExactlyAsync(lenBytes, ct).ConfigureAwait(false);
        long headerSize = BitConverter.ToInt64(lenBytes);
        if (headerSize <= 0 || headerSize > 100_000_000) throw new InvalidDataException($"'{name}': invalid SafeTensors header size {headerSize}");
        var header = new byte[headerSize]; await file.ReadExactlyAsync(header, ct).ConfigureAwait(false);
        long dataStart = 8 + headerSize;
        var (tensors, _) = SafeTensors.SafeTensorsFileReader.ParseHeader(name, header, dataStart, file.Length - dataStart);

        const string Anchor = "model.embed_tokens.weight";
        var anchor = tensors.FirstOrDefault(t => t.Name == Anchor || t.Name.EndsWith("." + Anchor, StringComparison.Ordinal))
            ?? throw new InvalidDataException($"'{name}' is not a Hugging Face causal LM ({Anchor} not found)");
        string prefix = anchor.Name[..^Anchor.Length];

        var infos = new List<GGUFTensorInfo>();
        int layers = 0;
        foreach (var t in tensors)
        {
            if (!t.Name.StartsWith(prefix, StringComparison.Ordinal)) continue;
            string hf = t.Name[prefix.Length..], gguf;
            if (hf == "model.embed_tokens.weight") gguf = "token_embd.weight";
            else if (hf == "model.norm.weight") gguf = "output_norm.weight";
            else if (hf == "lm_head.weight") gguf = "output.weight";
            else if (hf.StartsWith("model.layers.", StringComparison.Ordinal))
            {
                int dot = hf.IndexOf('.', "model.layers.".Length);
                if (dot < 0 || !int.TryParse(hf.AsSpan("model.layers.".Length, dot - "model.layers.".Length), out int l)) continue;
                string rest = hf[(dot + 1)..];
                var map = LayerNames.FirstOrDefault(m => m.Hf == rest);
                if (map.Hf == null) throw new NotSupportedException($"'{name}': {hf} is not a Qwen3 tensor this reader knows");
                gguf = $"blk.{l}.{map.Gguf}";
                layers = Math.Max(layers, l + 1);
            }
            else continue;
            var type = t.DType switch
            {
                "BF16" => GGMLType.BF16,
                "F16" => GGMLType.F16,
                "F32" => GGMLType.F32,
                _ => throw new NotSupportedException($"'{name}': {hf} is {t.DType} (supported: BF16, F16, F32)"),
            };
            infos.Add(new GGUFTensorInfo
            {
                Name = gguf,
                Dimensions = t.Shape.Reverse().Select(d => (long)d).ToArray(),               // GGUF lists the fastest dimension first
                Type = type,
                DataOffset = (ulong)(t.DataOffset - dataStart),
            });
        }
        GGUFTensorInfo Need(string n) => infos.FirstOrDefault(i => i.Name == n) ?? throw new InvalidDataException($"'{name}': {n} missing");
        var embd = Need("token_embd.weight"); Need("output_norm.weight");
        if (layers == 0) throw new InvalidDataException($"'{name}': no layers");
        long hidden = embd.Dimensions[0], vocab = embd.Dimensions[1];
        long headDim = Need("blk.0.attn_q_norm.weight").Dimensions[0];
        long qRows = Need("blk.0.attn_q.weight").Dimensions[1], kvRows = Need("blk.0.attn_k.weight").Dimensions[1], ffn = Need("blk.0.ffn_gate.weight").Dimensions[1];

        var meta = new Dictionary<string, object>
        {
            ["general.architecture"] = "qwen3",
            ["general.name"] = name,
            ["qwen3.block_count"] = (uint)layers,
            ["qwen3.context_length"] = (uint)contextLength,
            ["qwen3.embedding_length"] = (uint)hidden,
            ["qwen3.feed_forward_length"] = (uint)ffn,
            ["qwen3.attention.head_count"] = (uint)(qRows / headDim),
            ["qwen3.attention.head_count_kv"] = (uint)(kvRows / headDim),
            ["qwen3.attention.key_length"] = (uint)headDim,
            ["qwen3.attention.value_length"] = (uint)headDim,
            ["qwen3.rope.freq_base"] = ropeTheta,
            ["qwen3.attention.layer_norm_rms_epsilon"] = rmsEps,
        };
        AddTokenizer(meta, tokenizerJson, (int)vocab);
        if (chatTemplate != null) meta["tokenizer.chat_template"] = chatTemplate;
        return new GGUFModel { Version = 3, Metadata = meta, Tensors = infos.ToArray(), DataStartOffset = dataStart };
    }

    /// <summary>
    /// GGUF's tokenizer keys from a byte-level BPE <c>tokenizer.json</c>: the vocabulary and added tokens by id (special added tokens are control tokens, the others
    /// user-defined: Qwen's &lt;think&gt; decodes as text), padded to the embedding's row count with unused slots, the merges as "a b", and Qwen's special ids.
    /// </summary>
    private static void AddTokenizer(Dictionary<string, object> meta, string tokenizerJson, int rows)
    {
        using var doc = JsonDocument.Parse(tokenizerJson);
        var root = doc.RootElement; var model = root.GetProperty("model");
        if (model.GetProperty("type").GetString() != "BPE") throw new NotSupportedException("tokenizer.json: only byte-level BPE vocabularies are read");
        var tokens = new string?[rows]; var types = new int[rows];
        foreach (var p in model.GetProperty("vocab").EnumerateObject())
        {
            int id = p.Value.GetInt32();
            if (id >= rows) throw new InvalidDataException($"tokenizer.json: id {id} past the embedding's {rows} rows");
            tokens[id] = p.Name; types[id] = Preprocessing.SentencePieceTokenizer.TypeNormal;
        }
        int? eos = null, pad = null;
        if (root.TryGetProperty("added_tokens", out var added))
            foreach (var a in added.EnumerateArray())
            {
                int id = a.GetProperty("id").GetInt32(); string content = a.GetProperty("content").GetString() ?? "";
                if (id >= rows) throw new InvalidDataException($"tokenizer.json: added id {id} past the embedding's {rows} rows");
                tokens[id] = content;
                types[id] = a.TryGetProperty("special", out var sp) && sp.GetBoolean() ? Preprocessing.SentencePieceTokenizer.TypeControl : Preprocessing.SentencePieceTokenizer.TypeUserDefined;
                if (content == "<|im_end|>") eos = id;
                if (content == "<|endoftext|>") pad = id;
            }
        for (int i = 0; i < rows; i++)
            if (tokens[i] == null) { tokens[i] = $"[PAD{i}]"; types[i] = Preprocessing.SentencePieceTokenizer.TypeUnused; }
        var merges = model.GetProperty("merges").EnumerateArray()
            .Select(m => m.ValueKind == JsonValueKind.Array ? m[0].GetString() + " " + m[1].GetString() : m.GetString() ?? "").ToArray();
        meta["tokenizer.ggml.model"] = "gpt2";
        meta["tokenizer.ggml.pre"] = "qwen2";
        meta["tokenizer.ggml.tokens"] = tokens.Select(t => t!).ToArray();
        meta["tokenizer.ggml.token_type"] = types;
        meta["tokenizer.ggml.merges"] = merges;
        if (eos is int e) meta["tokenizer.ggml.eos_token_id"] = (uint)e;
        if (pad is int pd) { meta["tokenizer.ggml.padding_token_id"] = (uint)pd; meta["tokenizer.ggml.bos_token_id"] = (uint)pd; }
        meta["tokenizer.ggml.add_bos_token"] = false;
    }
}
