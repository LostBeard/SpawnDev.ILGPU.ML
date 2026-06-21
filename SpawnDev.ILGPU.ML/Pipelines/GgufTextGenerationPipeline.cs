using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// First-class text generation for GGUF chat models (gemma4) on top of <see cref="InferenceSession"/>.
/// Assembles the proven pieces into one call: the gemma4 chat template
/// (<see cref="ChatTemplates.BuildGemma4PromptTokens"/>), the O(n) incremental KV-cache decode
/// (<see cref="InferenceSession.RunDecodeStepAsync"/> + <see cref="GGUFDecodeKVCache"/>), and the
/// shared <see cref="TextGenerationSampler"/> (greedy / top-k / top-p / repetition penalty via
/// <see cref="GenerationConfig"/>). Browser-portable (async readback, no sync GPU waits).
///
/// Usage:
/// <code>
/// var model = await GGUFParser.ParseHeaderAsync(stream);          // for tokenizer + attn geometry
/// using var session = await InferenceSession.CreateFromGGUFFileAsync(acc, path);
/// using var gen = new GgufTextGenerationPipeline(session, acc, model);
/// string answer = await gen.GenerateAsync("What is the capital of France?");
/// </code>
/// </summary>
public sealed class GgufTextGenerationPipeline : IDisposable
{
    private readonly InferenceSession _session;
    private readonly Accelerator _accelerator;
    private readonly SentencePieceTokenizer _tokenizer;
    private readonly GGUFDecodeKVCache _cache;
    private readonly GpuArgMax _argmax;
    private readonly int _turnCloseId;
    private readonly int _eosId;

    /// <summary>The model's tokenizer (SentencePiece, from the GGUF vocab).</summary>
    public SentencePieceTokenizer Tokenizer => _tokenizer;

    /// <summary>
    /// Build the pipeline. <paramref name="model"/> is a parsed GGUF header (tokenizer + per-layer
    /// attention geometry come from it); <paramref name="session"/> is the loaded inference session for
    /// the same model. Allocates the decode KV-cache (sized to <paramref name="maxSeqLen"/>) and enables
    /// incremental decode on the session.
    /// </summary>
    public GgufTextGenerationPipeline(InferenceSession session, Accelerator accelerator, GGUF.GGUFModel model, int maxSeqLen = 4096)
    {
        _session = session ?? throw new ArgumentNullException(nameof(session));
        _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        _tokenizer = SentencePieceTokenizer.FromGGUF(model)
            ?? throw new InvalidOperationException("GGUF model has no SentencePiece tokenizer metadata.");

        int nLayers = (int)model.BlockCount, nHeads = (int)model.AttentionHeadCount;
        int defNKV = (int)model.AttentionHeadCountKV; if (defNKV == 0) defNKV = nHeads;
        int defHd = nHeads > 0 ? (int)model.EmbeddingLength / nHeads : 0;
        var kvHeadsArr = new int[nLayers]; var hdArr = new int[nLayers];
        for (int L = 0; L < nLayers; L++)
        {
            var cfg = GGUFGraphBuilder.GetLayerAttnConfig(model, L, nHeads, defNKV, defHd);
            kvHeadsArr[L] = cfg.NKVHeads; hdArr[L] = cfg.HeadDim;
        }
        _cache = new GGUFDecodeKVCache(accelerator, kvHeadsArr, hdArr, maxSeqLen);
        _session.EnableGGUFDecode(_cache);
        // Fixed-shape decode loop: recycle the per-step output buffers (no OOM on long generations) and warm-cache
        // the proven-stable shape-derived readbacks (skips their GPU round-trips — the browser-readback win). The
        // cache auto-detects stability (probe→stable→finalize) and falls back to live readback for anything not
        // proven stable, and this loop consumes each step's logits (argmax) before the next step, satisfying the
        // output-recycling contract.
        _session.CacheShapeReadbacks = true;

        _turnCloseId = ChatTemplates.Gemma4TurnCloseId(_tokenizer);
        _eosId = _tokenizer.EosId;
        _argmax = new GpuArgMax(accelerator);
    }

    /// <summary>
    /// Generate a response to <paramref name="userPrompt"/> using the gemma4 chat template + incremental
    /// KV-cache decode. Returns the decoded assistant text (includes the model's &lt;|channel&gt;thought
    /// block, since gemma4 is a thinking model — split on the channel markers if you want only the final
    /// answer). <paramref name="config"/> selects greedy (default) / top-k / top-p sampling;
    /// <paramref name="onToken"/> streams (tokenCount, textSoFar) after each token.
    /// </summary>
    public async Task<string> GenerateAsync(string userPrompt, string? systemPrompt = null,
        int maxNewTokens = 128, GenerationConfig? config = null, Func<int, string, Task>? onToken = null)
    {
        _session.ResetGGUFDecode();
        var promptIds = ChatTemplates.BuildGemma4PromptTokens(_tokenizer, systemPrompt, userPrompt, thinking: true);
        var rng = config?.Seed is int seed ? new Random(seed) : Random.Shared;
        var generated = new List<int>();
        int[] stepIds = promptIds;  // prefill = whole prompt, then 1 token/step

        for (int step = 0; step < maxNewTokens; step++)
        {
            var idf = new float[stepIds.Length];
            for (int i = 0; i < stepIds.Length; i++) idf[i] = stepIds[i];
            using var inBuf = _accelerator.Allocate1D(idf);
            var outputs = await _session.RunDecodeStepAsync(new Dictionary<string, Tensor>
            { ["input_ids"] = new Tensor(inBuf.View, new[] { 1, stepIds.Length }, "input_ids") });

            var logitsT = outputs.TryGetValue("logits", out var l) ? l : outputs.Values.First();
            int vocab = logitsT.Shape[^1];
            int seqOut = logitsT.ElementCount / vocab;
            long lastOff = (long)(seqOut - 1) * vocab;
            var lastLogits = logitsT.Data.SubView(lastOff, vocab);

            bool sampling = config?.Strategy is "top_k" or "top_p";
            bool repPen = config?.RepetitionPenalty is float r && r != 1.0f && generated.Count > 0;
            int next;
            if (!sampling && !repPen)
            {
                // Greedy with no host-side logit edit → argmax ON the GPU, read back one int (not ~1 MB).
                next = await _argmax.ArgMaxAsync(lastLogits, vocab);
            }
            else
            {
                // Sampling / repetition-penalty need the full distribution on the host. Browser-portable
                // readback of the last position's logits (no sync CopyToCPU).
                using var read = _accelerator.Allocate1D<float>(vocab);
                await read.View.CopyFromAsync(lastLogits);
                await _accelerator.SynchronizeAsync();
                var logits = await read.CopyToHostAsync<float>(0, vocab);

                if (repPen)
                    TextGenerationSampler.ApplyRepetitionPenalty(logits, generated.ToArray(), config!.RepetitionPenalty);
                next = config?.Strategy switch
                {
                    "top_k" => TextGenerationSampler.TopK(logits, config.TopK, config.Temperature, rng),
                    "top_p" => TextGenerationSampler.TopP(logits, config.TopP, config.Temperature, rng),
                    _ => TextGenerationSampler.Greedy(logits),
                };
            }

            generated.Add(next);
            if (onToken != null) await onToken(generated.Count, _tokenizer.Decode(generated.ToArray()));
            if (next == _turnCloseId || next == _eosId) break;
            stepIds = new[] { next };
        }

        return _tokenizer.Decode(generated.ToArray());
    }

    /// <summary>Releases the decode KV-cache + argmax buffers. Does NOT dispose the session or accelerator (caller-owned).</summary>
    public void Dispose() { _cache.Dispose(); _argmax.Dispose(); }
}
