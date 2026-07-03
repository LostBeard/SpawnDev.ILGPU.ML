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

    /// <summary>The underlying decode-enabled <see cref="InferenceSession"/> (owned by the pipeline/creator).</summary>
    public InferenceSession Session => _session;
    private readonly Accelerator _accelerator;
    private readonly SentencePieceTokenizer _tokenizer;
    private readonly GGUFDecodeKVCache _cache;
    private readonly GpuArgMax _argmax;
    private readonly int _eosId;
    private readonly int _maxSeqLen;

    /// <summary>The full token sequence (prompt + generated) currently resident in the KV cache from the
    /// previous <see cref="GenerateAsync"/> call; null before the first call. Used to compute the reusable
    /// common prefix on the next request.</summary>
    private int[]? _cachedIds;

    /// <summary>Enable KV-prefix caching: on each request, reuse the cached K/V for the longest common token
    /// prefix shared with the previous request (same tokens at the same absolute positions → bit-identical
    /// K/V), prefilling only the new suffix. The #1 win for agentic clients (Claude CLI) that re-send a near-
    /// identical ~14.5K-token prompt every turn. Set false to force a full re-prefill every call (the original
    /// behavior). Default true.</summary>
    public static bool EnablePrefixCache { get; set; } = true;

    /// <summary>Minimum reusable-prefix length to bother with prefix-cache reuse. Below this, a full prefill
    /// is cheap enough that the reuse bookkeeping isn't worth it.</summary>
    private const int MinReusePrefix = 16;

    /// <summary>Last request's reused-prefix length P (0 = no reuse / full prefill). Diagnostic for tests.</summary>
    public int LastReusedPrefix { get; private set; }

    /// <summary>
    /// Opt-in WebGPU decode capture/replay: the first single-token decode step captures the decode
    /// graph as a dispatch plan (plus a one-step probe that discovers every KV-cursor-dependent
    /// param - see <see cref="WebGPUDecodeCapture"/>); every subsequent decode step is a patched
    /// single-interop-crossing replay. Measured: 686ms/tok -> ~21ms/tok on the same 4070 (the
    /// per-node dispatch orchestration collapses). No-op on non-WebGPU accelerators. Prefill and
    /// multi-token steps always run the direct path. Gate:
    /// GGUF_WebGPU_DecodeCapture_TokenIdentical (greedy decode token-identical to the direct path).
    /// </summary>
    public bool EnableWebGPUDecodeCapture { get; set; }
    private WebGPUDecodeCapture? _decodeCapture;
    /// <summary>Diagnostics: (ops, scalar/copy/slot patch counts) of the active decode capture, or null.</summary>
    public (int Ops, int Scalars, int Copies, int Slots)? DecodeCaptureInfo =>
        _decodeCapture is { } c ? (c.DispatchCount, c.PatchCounts.Scalars, c.PatchCounts.Copies, c.PatchCounts.Slots) : null;

    /// <summary>Per-phase ms of the most recent patched replay step (diagnostics), or null.</summary>
    /// <summary>Patch sub-phases of the most recent replay step (diagnostics), or null.</summary>
    public (double Input, double Scalars, double Slots, double Copies)? LastDecodeCapturePatchSplitMs =>
        _decodeCapture is { } cc ? cc.LastPatchSplitMs : null;

    public (double Patch, double Replay, double Sync)? LastDecodeCaptureStepMs =>
        _decodeCapture is { } c ? (c.LastPatchMs, c.LastReplayMs, c.LastSyncMs) : null;

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
        _maxSeqLen = maxSeqLen;
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
        int maxNew = config?.MaxNewTokens is int mn && mn > 0 ? mn : 128;

        // Fit prompt + generation into the KV cache. Agentic clients (Claude CLI) routinely send prompts far
        // larger than a small local model's context (they assume a 200K-context Claude) and ask for huge
        // max_tokens — so an over-long prompt is TAIL-truncated (keep the most recent tokens, which hold the
        // actual question) and maxNew is capped, rather than overflowing the cache and crashing.
        bool truncated = false;
        if (promptIds.Length + maxNew > _maxSeqLen - 1)
        {
            int reserve = Math.Clamp(maxNew, 64, Math.Max(64, _maxSeqLen / 8));
            int keep = _maxSeqLen - 1 - reserve;
            if (keep >= 1 && promptIds.Length > keep)
            {
                promptIds = promptIds[^keep..];
                truncated = true;
            }
            maxNew = Math.Min(maxNew, _maxSeqLen - 1 - promptIds.Length);
            if (maxNew < 1) maxNew = 1;
        }

        var rng = config?.Seed is int seed ? new Random(seed) : Random.Shared;
        var detok = _tokenizer.CreateStreamingDecoder();
        var generated = new List<int>();
        var full = new StringBuilder();
        int emitted = 0;
        int maxStopLen = 0;
        if (stopStrings != null) foreach (var s in stopStrings) if (s.Length > maxStopLen) maxStopLen = s.Length;

        // KV-prefix cache: reuse the bit-identical K/V of the longest common token prefix that this prompt
        // shares with the sequence already resident in the cache (same tokens, same absolute positions, so
        // RoPE positions match → token-identical to a full re-prefill). Prefill only the new suffix.
        // P is capped at promptIds.Length-1 (must prefill ≥1 token so there are logits to decode from) and at
        // _cachedIds.Length (can't reuse beyond what's cached). Below MinReusePrefix it's not worth it.
        // CORRECTNESS: reuse is valid ONLY when the prompt was NOT tail-truncated this turn — truncation shifts
        // every kept token to a lower absolute position, so the new prompt's position i no longer matches the
        // cached token at absolute position i (RoPE uses absolute position). A coincidental value match would
        // reuse K/V computed at the wrong position. On truncation, force a full prefill.
        int P = 0;
        if (EnablePrefixCache && !truncated && _cachedIds != null)
        {
            int maxP = Math.Min(_cachedIds.Length, promptIds.Length - 1);
            while (P < maxP && _cachedIds[P] == promptIds[P]) P++;
            if (P < MinReusePrefix) P = 0;
        }

        int[] stepIds; // prefill tokens for step 0 (whole prompt, or just the new suffix on reuse)
        if (P > 0)
        {
            // Reuse path: leave the cached K/V for tokens 0..P-1 in place, set the cursor to P, prefill P..end.
            _session.SetGGUFDecodePastLen(P);
            stepIds = promptIds[P..];
        }
        else
        {
            // No reuse: full prefill from position 0 (original behavior).
            _session.ResetGGUFDecode();
            stepIds = promptIds;
        }
        LastReusedPrefix = P;
        StopReason stop = StopReason.Length;

        for (int step = 0; step < maxNew; step++)
        {
            if (ct.IsCancellationRequested) { stop = StopReason.Cancelled; break; }

            IReadOnlyDictionary<string, Tensor>? outputs;
            MemoryBuffer1D<float, Stride1D.Dense>? inBuf = null;
            int? fastToken = null;        // greedy single-fence replay token (skips the logits section)
            float[]? fastLogits = null;   // sampled single-fence replay logits (host array, skips the GPU read)
            bool wantSampling = config?.Strategy is "top_k" or "top_p";
            bool wantRepPen = config?.RepetitionPenalty is float rpv && rpv != 1.0f;
            try
            {
                if (EnableWebGPUDecodeCapture && stepIds.Length == 1)
                {
                    // Capture on the first single-token step; patched replay ever after (valid at ANY
                    // pastLen - the patches are affine in the cursor - so it survives across turns and
                    // prefix-cache reuse; prefill/multi-token steps below stay on the direct path).
                    if (_decodeCapture == null)
                    {
                        _decodeCapture = await WebGPUDecodeCapture.TryCaptureAsync(_session, stepIds[0], _argmax);
                        if (_decodeCapture != null)
                            outputs = _decodeCapture.Outputs;   // the capture pass IS this step's forward
                        else
                        {
                            EnableWebGPUDecodeCapture = false;  // non-WebGPU: don't retry every step
                            outputs = await RunDirectAsync();
                        }
                    }
                    else if (!wantSampling && !(wantRepPen && generated.Count > 0))
                    {
                        // Greedy fast path: the plan's folded argmax + ONE fence per token.
                        fastToken = await _decodeCapture.PatchAndDecodeGreedyAsync(stepIds[0], _session.DecodePastLen);
                        outputs = null;
                    }
                    else
                    {
                        // Sampled fast path (the /ai-chat top_p + repetition-penalty config): patch +
                        // replay + direct host logits read - still ONE fence per token.
                        fastLogits = await _decodeCapture.PatchAndReadLogitsAsync(stepIds[0], _session.DecodePastLen);
                        outputs = null;
                    }
                }
                else
                    outputs = await RunDirectAsync();

                async Task<IReadOnlyDictionary<string, Tensor>> RunDirectAsync()
                {
                    var idf = new float[stepIds.Length];
                    for (int i = 0; i < stepIds.Length; i++) idf[i] = stepIds[i];
                    inBuf = _accelerator.Allocate1D(idf);
                    return await _session.RunDecodeStepAsync(new Dictionary<string, Tensor>
                    { ["input_ids"] = new Tensor(inBuf.View, new[] { 1, stepIds.Length }, "input_ids") });
                }

            int next;
            if (fastToken is int ft)
            {
                next = ft;   // greedy replay already produced the token (folded argmax, one fence)
            }
            else if (fastLogits != null)
            {
                // Sampled replay already produced host logits (one fence) - sample on the host.
                if (wantRepPen && generated.Count > 0)
                    TextGenerationSampler.ApplyRepetitionPenalty(fastLogits, generated.ToArray(), config!.RepetitionPenalty);
                next = config?.Strategy switch
                {
                    "top_k" => TextGenerationSampler.TopK(fastLogits, config.TopK, config.Temperature, rng),
                    "top_p" => TextGenerationSampler.TopP(fastLogits, config.TopP, config.Temperature, rng),
                    _ => TextGenerationSampler.Greedy(fastLogits),
                };
            }
            else
            {
            var logitsT = outputs!.TryGetValue("logits", out var l) ? l : outputs.Values.First();
            int vocab = logitsT.Shape[^1];
            int seqOut = logitsT.ElementCount / vocab;
            long lastOff = (long)(seqOut - 1) * vocab;
            var lastLogits = logitsT.Data.SubView(lastOff, vocab);

            bool sampling = wantSampling;
            bool repPen = wantRepPen && generated.Count > 0;
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
            }   // end !fastToken (logits path)

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
            finally { inBuf?.Dispose(); }
        }

        // Flush the detokenizer + any held-back safe tail (only if we didn't stop on a stop string).
        if (stop != StopReason.StopString)
        {
            full.Append(detok.Finish());
            if (full.Length > emitted && onDelta != null)
                await onDelta(full.ToString(emitted, full.Length - emitted));
        }

        // Record the full sequence now resident in the KV cache (prompt + generated) so the NEXT request can
        // reuse this turn's prompt+response as its prefix. The generated tokens were written into the cache at
        // their absolute positions during decode; only the tokens contributing to the cache count (a stop
        // token ends the turn WITHOUT being written, so it is excluded — `generated` already excludes it).
        // Cap to the cache's valid region (maxSeqLen-1, matching the prefill/decode bound used above).
        if (EnablePrefixCache)
        {
            int total = promptIds.Length + generated.Count;
            int cap = _maxSeqLen - 1;
            var resident = new int[Math.Min(total, cap)];
            int copyPrompt = Math.Min(promptIds.Length, resident.Length);
            Array.Copy(promptIds, 0, resident, 0, copyPrompt);
            for (int i = 0; copyPrompt + i < resident.Length && i < generated.Count; i++)
                resident[copyPrompt + i] = generated[i];
            _cachedIds = resident;
        }
        else _cachedIds = null;

        return new GenerationResult(full.ToString(), promptIds.Length, generated.Count, stop);
    }

    /// <summary>
    /// TEST/DIAGNOSTIC: greedy-decode exactly <see cref="GenerationConfig.MaxNewTokens"/> token ids from
    /// <paramref name="promptIds"/>, returning the RAW token ids (no EOS/stop suppression, no detokenization)
    /// and the prefill TTFT (wall time of the first step = prefill + first-token argmax). Drives the exact same
    /// KV-prefix-cache reuse path as <see cref="GenerateAsync"/> (so a reuse run is directly comparable to a
    /// fresh run for the token-identity gate) and updates <see cref="_cachedIds"/> the same way. Greedy only.
    /// </summary>
    public async Task<(int[] ids, double ttftMs)> GenerateFirstTokenIdsAsync(int[] promptIds, GenerationConfig? config = null)
    {
        int maxNew = config?.MaxNewTokens is int mn && mn > 0 ? mn : 10;

        bool truncated = false;
        if (promptIds.Length + maxNew > _maxSeqLen - 1)
        {
            int reserve = Math.Clamp(maxNew, 64, Math.Max(64, _maxSeqLen / 8));
            int keep = _maxSeqLen - 1 - reserve;
            if (keep >= 1 && promptIds.Length > keep) { promptIds = promptIds[^keep..]; truncated = true; }
            maxNew = Math.Min(maxNew, _maxSeqLen - 1 - promptIds.Length);
            if (maxNew < 1) maxNew = 1;
        }

        // Same prefix-cache reuse decision as GenerateAsync (see that method for the correctness rationale).
        int P = 0;
        if (EnablePrefixCache && !truncated && _cachedIds != null)
        {
            int maxP = Math.Min(_cachedIds.Length, promptIds.Length - 1);
            while (P < maxP && _cachedIds[P] == promptIds[P]) P++;
            if (P < MinReusePrefix) P = 0;
        }

        int[] stepIds;
        if (P > 0) { _session.SetGGUFDecodePastLen(P); stepIds = promptIds[P..]; }
        else { _session.ResetGGUFDecode(); stepIds = promptIds; }
        LastReusedPrefix = P;

        var generated = new List<int>();
        double ttftMs = 0;
        for (int step = 0; step < maxNew; step++)
        {
            var idf = new float[stepIds.Length];
            for (int i = 0; i < stepIds.Length; i++) idf[i] = stepIds[i];
            var sw = step == 0 ? System.Diagnostics.Stopwatch.StartNew() : null;
            using var inBuf = _accelerator.Allocate1D(idf);
            var outputs = await _session.RunDecodeStepAsync(new Dictionary<string, Tensor>
            { ["input_ids"] = new Tensor(inBuf.View, new[] { 1, stepIds.Length }, "input_ids") });

            var logitsT = outputs.TryGetValue("logits", out var l) ? l : outputs.Values.First();
            int vocab = logitsT.Shape[^1];
            int seqOut = logitsT.ElementCount / vocab;
            long lastOff = (long)(seqOut - 1) * vocab;
            int next = await _argmax.ArgMaxAsync(logitsT.Data.SubView(lastOff, vocab), vocab);
            if (sw != null) { await _accelerator.SynchronizeAsync(); sw.Stop(); ttftMs = sw.Elapsed.TotalMilliseconds; }

            generated.Add(next);
            stepIds = new[] { next };
        }

        if (EnablePrefixCache)
        {
            int total = promptIds.Length + generated.Count;
            int cap = _maxSeqLen - 1;
            var resident = new int[Math.Min(total, cap)];
            int copyPrompt = Math.Min(promptIds.Length, resident.Length);
            Array.Copy(promptIds, 0, resident, 0, copyPrompt);
            for (int i = 0; copyPrompt + i < resident.Length && i < generated.Count; i++)
                resident[copyPrompt + i] = generated[i];
            _cachedIds = resident;
        }
        else _cachedIds = null;

        return (generated.ToArray(), ttftMs);
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
    public void Dispose() { _decodeCapture?.Dispose(); _decodeCapture = null; _cache.Dispose(); _argmax.Dispose(); }
}
