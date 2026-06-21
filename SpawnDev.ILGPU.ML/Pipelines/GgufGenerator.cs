using ILGPU;
using ILGPU.Runtime;
using System.Text;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>Why generation stopped.</summary>
public enum StopReason
{
    /// <summary>Hit the model's EOS token.</summary>
    Eos,
    /// <summary>Hit a caller-supplied stop token id.</summary>
    StopToken,
    /// <summary>Matched a caller-supplied stop string.</summary>
    StopString,
    /// <summary>Reached the max-new-tokens budget.</summary>
    Length,
    /// <summary>Cancelled.</summary>
    Cancelled,
}

/// <summary>The result of a generation: the decoded text and accounting.</summary>
public sealed record GenerationResult(string Text, int PromptTokens, int GeneratedTokens, StopReason Stop);

/// <summary>
/// General GGUF text generator on top of <see cref="InferenceSession"/>: the O(n) incremental KV-cache
/// decode (<see cref="InferenceSession.RunDecodeStepAsync"/> + <see cref="GGUFDecodeKVCache"/>), the shared
/// <see cref="TextGenerationSampler"/> (greedy / top-k / top-p / repetition penalty), the UTF-8-safe
/// incremental detokenizer (<see cref="SentencePieceStreamingDecoder"/>), and stop handling
/// (EOS + arbitrary stop token ids + arbitrary stop strings). Architecture-agnostic: the caller supplies
/// already-formatted prompt token ids (raw, a chat template, etc.), so this drives qwen / llama / gpt-oss /
/// deepseek / gemma4 alike. Browser-portable (async readback, no sync GPU waits).
/// </summary>
public sealed class GgufGenerator : IDisposable
{
    private readonly InferenceSession _session;
    private readonly Accelerator _accelerator;
    private readonly SentencePieceTokenizer _tokenizer;
    private readonly GGUFDecodeKVCache _cache;
    private readonly GpuArgMax _argmax;
    private readonly int _eosId;

    /// <summary>The model's tokenizer (SentencePiece, from the GGUF vocab).</summary>
    public SentencePieceTokenizer Tokenizer => _tokenizer;

    /// <summary>The model's EOS token id.</summary>
    public int EosId => _eosId;

    /// <summary>
    /// Build the generator. <paramref name="model"/> is the parsed GGUF header (tokenizer + per-layer
    /// attention geometry); <paramref name="session"/> is the loaded session for the same model. Allocates
    /// the decode KV-cache (sized to <paramref name="maxSeqLen"/>) and enables incremental decode.
    /// </summary>
    public GgufGenerator(InferenceSession session, Accelerator accelerator, GGUFModel model, int maxSeqLen = 4096)
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
        _session.CacheShapeReadbacks = true; // recycle fixed-shape decode buffers + warm-cache stable readbacks
        _eosId = _tokenizer.EosId;
        _argmax = new GpuArgMax(accelerator);
    }

    /// <summary>
    /// Generate from already-formatted <paramref name="promptIds"/>. Streams the incremental text delta via
    /// <paramref name="onDelta"/> (UTF-8-safe; partial stop strings are held back and never leak). Stops on
    /// EOS, any id in <paramref name="stopTokenIds"/>, any string in <paramref name="stopStrings"/>, or the
    /// <see cref="GenerationConfig.MaxNewTokens"/> budget. Returns the full decoded text (already trimmed of
    /// the matched stop string).
    /// </summary>
    public async Task<GenerationResult> GenerateAsync(
        int[] promptIds,
        GenerationConfig? config = null,
        IReadOnlyList<string>? stopStrings = null,
        IReadOnlyList<int>? stopTokenIds = null,
        Func<string, Task>? onDelta = null,
        CancellationToken ct = default)
    {
        _session.ResetGGUFDecode();
        int maxNew = config?.MaxNewTokens is int mn && mn > 0 ? mn : 128;
        var rng = config?.Seed is int seed ? new Random(seed) : Random.Shared;
        var detok = _tokenizer.CreateStreamingDecoder();
        var generated = new List<int>();
        var full = new StringBuilder();
        int emitted = 0;
        int maxStopLen = 0;
        if (stopStrings != null) foreach (var s in stopStrings) if (s.Length > maxStopLen) maxStopLen = s.Length;

        int[] stepIds = promptIds; // prefill = whole prompt, then 1 token/step
        StopReason stop = StopReason.Length;

        for (int step = 0; step < maxNew; step++)
        {
            if (ct.IsCancellationRequested) { stop = StopReason.Cancelled; break; }

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
                next = await _argmax.ArgMaxAsync(lastLogits, vocab); // greedy on the GPU, read back one int
            }
            else
            {
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

            // Stop tokens (EOS + caller list) end the turn WITHOUT contributing text.
            if (next == _eosId) { stop = StopReason.Eos; break; }
            if (stopTokenIds != null && stopTokenIds.Contains(next)) { stop = StopReason.StopToken; break; }

            generated.Add(next);
            full.Append(detok.Push(next));

            // Stop strings: if any appears, truncate at the FIRST match, emit up to it, stop.
            int cut = EarliestStopMatch(full, emitted, stopStrings);
            if (cut >= 0)
            {
                if (cut > emitted && onDelta != null) await onDelta(full.ToString(emitted, cut - emitted));
                full.Length = cut;
                emitted = cut;
                stop = StopReason.StopString;
                break;
            }

            // Stream everything that can't still be the prefix of a stop string (hold back the tail).
            int safe = full.Length - PartialStopSuffix(full, maxStopLen, stopStrings);
            if (safe > emitted && onDelta != null)
            {
                await onDelta(full.ToString(emitted, safe - emitted));
                emitted = safe;
            }

            stepIds = new[] { next }; // incremental decode: after the prefill, feed only the new token
        }

        // Flush the detokenizer + any held-back safe tail (only if we didn't stop on a stop string).
        if (stop != StopReason.StopString)
        {
            full.Append(detok.Finish());
            if (full.Length > emitted && onDelta != null)
                await onDelta(full.ToString(emitted, full.Length - emitted));
        }

        return new GenerationResult(full.ToString(), promptIds.Length, generated.Count, stop);
    }

    /// <summary>Index of the earliest stop-string match at or after <paramref name="from"/>, or -1.</summary>
    private static int EarliestStopMatch(StringBuilder sb, int from, IReadOnlyList<string>? stops)
    {
        if (stops == null || stops.Count == 0) return -1;
        string s = sb.ToString();
        int best = -1;
        foreach (var stopStr in stops)
        {
            if (string.IsNullOrEmpty(stopStr)) continue;
            // Search a little before `from` so a stop string straddling the last emit boundary is still found.
            int searchFrom = Math.Max(0, from - stopStr.Length + 1);
            int idx = s.IndexOf(stopStr, searchFrom, StringComparison.Ordinal);
            if (idx >= 0 && (best < 0 || idx < best)) best = idx;
        }
        return best;
    }

    /// <summary>
    /// Length of the suffix of <paramref name="sb"/> that is a (possibly full) prefix of some stop string —
    /// the tail that must be held back from streaming until we know it isn't the start of a stop match.
    /// </summary>
    private static int PartialStopSuffix(StringBuilder sb, int maxStopLen, IReadOnlyList<string>? stops)
    {
        if (stops == null || stops.Count == 0 || maxStopLen == 0) return 0;
        string s = sb.ToString();
        int maxCheck = Math.Min(maxStopLen, s.Length);
        for (int len = maxCheck; len > 0; len--)
        {
            var tail = s.AsSpan(s.Length - len);
            foreach (var stopStr in stops)
            {
                if (string.IsNullOrEmpty(stopStr) || stopStr.Length < len) continue;
                if (stopStr.AsSpan(0, len).SequenceEqual(tail)) return len; // tail is a prefix of this stop
            }
        }
        return 0;
    }

    /// <summary>Releases the decode KV-cache + argmax buffers. Does NOT dispose the session or accelerator (caller-owned).</summary>
    public void Dispose() { _cache.Dispose(); _argmax.Dispose(); }
}
