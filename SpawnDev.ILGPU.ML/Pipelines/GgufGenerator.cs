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
public sealed record GenerationResult(string Text, int PromptTokens, int GeneratedTokens, StopReason Stop)
{
    /// <summary>
    /// The ids the model generated, in order: exactly the tokens <see cref="Text"/> was decoded from, as the model emitted them (which can be
    /// a non-canonical tokenization of the text, so re-encoding the text does not reproduce them). The terminating stop / EOS token is NOT
    /// included (neither engine keeps it); a stop STRING's tokens are, as the text before trimming held them. Training pairs use these
    /// (Plans/training-autograd-lora-2026-11-04.md).
    /// </summary>
    public int[] GeneratedTokenIds { get; init; } = Array.Empty<int>();
}

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
    // the batch decoder's engine (CreateAsync chose it): every public member routes to it; the session-path fields are then unset
    private readonly GgufBatchTextEngine? _batch;
    private InferenceSession? _ownedSession;   // CreateAsync's session (the generator engine), disposed with this generator
    private Stream? _ownedStream;

    /// <summary>The underlying decode-enabled <see cref="InferenceSession"/> (owned by the pipeline/creator). The generator engine only: a generator on the batch
    /// decoder (<see cref="Engine"/>) has no session.</summary>
    public InferenceSession Session => _batch == null ? _session
        : throw new InvalidOperationException("this generator decodes with the batch decoder (Engine = BatchDecoder), which has no InferenceSession: create it with engine: GgufEngine.Generator");

    /// <summary>The engine that decodes (<see cref="CreateAsync"/> resolves Auto; the session constructor is always <see cref="GgufEngine.Generator"/>).</summary>
    public GgufEngine Engine => _batch != null ? GgufEngine.BatchDecoder : GgufEngine.Generator;

    /// <summary>Why <see cref="CreateAsync"/> with <see cref="GgufEngine.Auto"/> chose the generator engine ("" when it chose the batch decoder or the caller chose).</summary>
    public string EngineReason { get; private set; } = "";

    /// <summary>The batch decoder when <see cref="Engine"/> is <see cref="GgufEngine.BatchDecoder"/>, else null.</summary>
    public GgufBatchDecoder? BatchDecoder => _batch?.Decoder;

    /// <summary>
    /// LoRA adapters applied unmerged at every step (<see cref="Training.LoraRuntime"/>, PEFT tensors; null = the base model). The batch decoder
    /// engine only (<see cref="Engine"/> = BatchDecoder, which Auto picks for llama / qwen on CUDA, OpenCL and WebGPU); setting it on the
    /// session engine throws. Changing it forgets the reusable prefix. The caller owns the runtime's lifetime.
    /// </summary>
    public Training.LoraRuntime? Lora
    {
        get => _batch?.Decoder.Lora;
        set
        {
            if (_batch == null) throw new NotSupportedException("LoRA adapters at inference need the batch decoder engine (Engine = BatchDecoder)");
            _batch.Decoder.Lora = value;
        }
    }

    /// <summary>The model's parsed GGUF header (the chat template, architecture and tokenizer metadata come from it).</summary>
    public GGUFModel Gguf { get; }

    /// <summary>
    /// Load a GGUF chat model from a seekable stream and pick the faster engine for it: <see cref="GgufBatchDecoder"/> (one sequence, prefix reuse) for qwen2,
    /// qwen3 and llama (llama 3 rope_freqs included) on CUDA, OpenCL and WebGPU (<see cref="GgufTextGenerationPipeline.ChooseEngine"/>; MEASURED 2026-11-01, Qwen2.5-0.5B,
    /// one chat answer: WebGPU 67.8 -> 260.8 tok/s, CUDA 104.6 -> 165.1, OpenCL 47.3 -> 184.0), the session engine otherwise (other architectures, WebGL, Wasm,
    /// the CPU backends). Same <see cref="GenerateAsync"/>: stop tokens and strings, streamed text, the prefix cache (<see cref="LastReusedPrefix"/>), cancel.
    /// <paramref name="model"/>: the header if the caller already parsed it (it is not parsed twice). The weights stream to the GPU (a browser JS stream goes
    /// JS -> GPU, never the .NET heap). Keep <paramref name="gguf"/> open until this generator is disposed: the session engine reads it on demand; the batch
    /// engine disposes it with itself. <paramref name="ownsStream"/>: dispose it with this generator on either engine.
    /// </summary>
    /// <remarks>Seeded sampling is reproducible on each engine, not across them (the batch decoder's device draws take their uniform from seed, sequence and position).</remarks>
    public static async Task<GgufGenerator> CreateAsync(Accelerator accelerator, Stream gguf, int maxSeqLen = 4096, GGUFModel? model = null,
        Action<string, int>? onProgress = null, GgufEngine engine = GgufEngine.Auto, bool ownsStream = false, CancellationToken ct = default)
    {
        if (!gguf.CanSeek) throw new ArgumentException("GgufGenerator.CreateAsync requires a seekable stream.", nameof(gguf));
        if (model == null)
        {
            gguf.Seek(0, SeekOrigin.Begin);
            onProgress?.Invoke("parse", 0);
            model = await GGUFParser.ParseHeaderAsync(gguf, ct).ConfigureAwait(false);
            onProgress?.Invoke("parse", 100);
        }
        string reason = "";
        if (engine != GgufEngine.Generator)
        {
            var chosen = GgufTextGenerationPipeline.ChooseEngine(accelerator, model, out reason);
            if (chosen == GgufEngine.BatchDecoder)
            {
                var dec = await GgufBatchDecoder.CreateAsync(accelerator, gguf, model, maxSeqLen, 1, onProgress, ct).ConfigureAwait(false);
                return new GgufGenerator(dec, accelerator, model);
            }
            if (engine == GgufEngine.BatchDecoder) throw new NotSupportedException("GgufEngine.BatchDecoder: " + reason);
        }
        gguf.Seek(0, SeekOrigin.Begin);
        var session = await InferenceSession.CreateFromGGUFStreamAsync(accelerator, gguf, onProgress, ct, parsedHeader: model).ConfigureAwait(false);
        return new GgufGenerator(session, accelerator, model, maxSeqLen) { _ownedSession = session, _ownedStream = ownsStream ? gguf : null, EngineReason = reason };
    }

    private GgufGenerator(GgufBatchDecoder decoder, Accelerator accelerator, GGUFModel model)
    {
        _session = null!; _cache = null!; _argmax = null!;              // the session engine's; never touched on this engine
        _accelerator = accelerator;
        _tokenizer = SentencePieceTokenizer.FromGGUF(model) ?? throw new InvalidOperationException("GGUF model has no SentencePiece tokenizer metadata.");
        _eosId = _tokenizer.EosId;
        _maxSeqLen = decoder.MaxSeq;
        Gguf = model;
        _batch = new GgufBatchTextEngine(decoder, _tokenizer);
    }
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

    /// <summary>
    /// A run is about to overwrite the K/V rows from <paramref name="p"/> on: until it finishes and records what it
    /// left behind, only the first <paramref name="p"/> rows are known to hold <see cref="_cachedIds"/>. Without this a
    /// run that was CANCELLED (Vertex interrupts a reply when someone speaks) or threw left the record describing
    /// rows it had overwritten, and the next prompt sharing that prefix decoded from another prompt's context:
    /// token soup after every interrupted reply (Vertex BrainTests, 2026-10-12).
    /// </summary>
    private void ForgetCachedRowsFrom(int p) => _cachedIds = p > 0 && _cachedIds != null ? _cachedIds[..p] : null;

    /// <summary>
    /// Records the token sequence the K/V cache now holds, so the next request can reuse it as its prefix: the prompt
    /// then the generated tokens, but only the first <paramref name="written"/> - the rows a step actually wrote. The
    /// old record assumed prompt + every generated token. Wrong twice: a generation cancelled BEFORE its prefill ran
    /// (a cooperative cancel at the loop top, gen=0) recorded a whole prompt whose rows were never written, and the
    /// next request "reused" them (Vertex 2026-10-12: reused 2,847 of 2,949 tokens of nothing -> token soup; also in
    /// 5.3.6); and the LAST sampled token, never fed back, was always counted as resident.
    /// </summary>
    private void RecordResident(int[] promptIds, IReadOnlyList<int> generated, int written)
    {
        if (!EnablePrefixCache) { _cachedIds = null; return; }
        int n = Math.Min(Math.Min(written, promptIds.Length + generated.Count), _maxSeqLen - 1);
        var resident = new int[n];
        for (int i = 0; i < n; i++)
            resident[i] = i < promptIds.Length ? promptIds[i] : generated[i - promptIds.Length];
        _cachedIds = resident;
    }

    /// <summary>Enable KV-prefix caching: on each request, reuse the cached K/V for the longest common token
    /// prefix shared with the previous request (same tokens at the same absolute positions → bit-identical
    /// K/V), prefilling only the new suffix. The #1 win for agentic clients (Claude CLI) that re-send a near-
    /// identical ~14.5K-token prompt every turn. Set false to force a full re-prefill every call (the original
    /// behavior). Default true.</summary>
    public static bool EnablePrefixCache { get; set; } = true;

    /// <summary>Minimum reusable-prefix length to bother with prefix-cache reuse. Below this, a full prefill
    /// is cheap enough that the reuse bookkeeping isn't worth it.</summary>
    private const int MinReusePrefix = 16;

    /// <summary>
    /// Veto KV-prefix reuse when the model carries per-step CONV STATE that does not describe cursor P.
    ///
    /// K/V is POSITION-ADDRESSED (row = absolute position, RoPE recomputed from it), so reusing the first P
    /// tokens and restarting at cursor P is bit-identical to a full prefill. `ShortConvStateCache` (LFM2 and
    /// any short-conv/SSM mixer) is NOT: it is a shift register holding the history for exactly ONE cursor -
    /// `StatePos`, where the last turn ended. Reusing at any other P feeds the conv layers a history that
    /// describes DIFFERENT tokens. It is not subtle: the same prompt asked twice returned a different answer
    /// on CUDA/WebGPU/CPU and outright multilingual garbage on WebGL (measured 2026-07-16). It fires on a new
    /// conversation without a reload (the system-prompt prefix still matches!), an edited system prompt, or a
    /// re-ask - and P == StatePos only in the plain append-only case, which is why single-turn testing missed it.
    ///
    /// Correct and cheap: fall back to a FULL prefill (P=0), which zero-pads and rebuilds the state from
    /// scratch. Only conv-state models pay it; K/V-only archs (qwen/gemma) keep full reuse. A future
    /// optimization could snapshot conv state per position, but correctness comes first.
    /// </summary>
    private int ConvStateAllowsReuse(int P)
    {
        if (P <= 0) return P;
        var conv = _session.ConvStateCache;
        if (conv == null || conv.NumLayers == 0) return P;   // no conv state (qwen/gemma): reuse is valid
        return P == conv.StatePos ? P : 0;
    }

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
    public bool EnableWebGPUDecodeCapture
    {
        get => _batch != null ? _batch.Decoder.CaptureWebGPU : _enableCapture;
        set { if (_batch != null) _batch.Decoder.CaptureWebGPU = value; else _enableCapture = value; }
    }
    private bool _enableCapture;
    private WebGPUDecodeCapture? _decodeCapture;
    /// <summary>Recycled input_ids upload buffer (capacity = maxSeqLen). Avoids per-token Allocate1D.</summary>
    private MemoryBuffer1D<float, Stride1D.Dense>? _inputIdsBuf;
    /// <summary>Recycled logits staging for sampling readback. Avoids per-token Allocate1D(vocab).</summary>
    private MemoryBuffer1D<float, Stride1D.Dense>? _logitsReadBuf;
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
        Gguf = model;
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
        _session.OutputsLiveUntilNextRun = true; // the prefill's full-prompt logits too, not only the decode steps'
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
        if (_batch != null)
        {
            _batch.EnablePrefixCache = EnablePrefixCache;
            var br = await _batch.GenerateAsync(promptIds, config, stopStrings, stopTokenIds, onDelta, ct).ConfigureAwait(false);
            LastReusedPrefix = _batch.LastReusedPrefix;
            return br;
        }
        int maxNew = config?.MaxNewTokens is int mn && mn > 0 ? mn : 128;

        // Fit prompt + generation into the KV cache (an over-long prompt keeps its most recent tokens: GenerationText.FitPrompt)
        bool truncated;
        (promptIds, maxNew, truncated) = GenerationText.FitPrompt(promptIds, maxNew, _maxSeqLen);

        var rng = config?.Seed is int seed ? new Random(seed) : Random.Shared;
        var generated = new List<int>();
        var text = new TextStream(_tokenizer, stopStrings, onDelta);

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
            P = ConvStateAllowsReuse(P);
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
        ForgetCachedRowsFrom(P);
        int written = P;   // K/V rows that hold real data: the reused prefix, plus each step once it has RUN
        StopReason stop = StopReason.Length;

        for (int step = 0; step < maxNew; step++)
        {
            if (ct.IsCancellationRequested) { stop = StopReason.Cancelled; break; }

            IReadOnlyDictionary<string, Tensor>? outputs;
            int? fastToken = null;        // greedy single-fence replay token (skips the logits section)
            float[]? fastLogits = null;   // sampled single-fence replay logits (host array, skips the GPU read)
            bool penaltyApplied = false;  // the device sampler already penalised the row these logits came from
            bool wantSampling = config?.Strategy is "top_k" or "top_p";
            bool wantRepPen = config?.RepetitionPenalty is float rpv && rpv != 1.0f;
            // a device draw this step: its settings go up NOW, before the forward is queued (on CUDA/OpenCL a later upload waits for the forward)
            double stagedU = UseDeviceSampling && (wantSampling || (wantRepPen && generated.Count > 0)) ? StageDeviceSample(config, generated, rng) : 0.0;
            {
                if (EnableWebGPUDecodeCapture && stepIds.Length == 1)
                {
                    // Capture on the first single-token step; patched replay ever after (valid at ANY
                    // pastLen - the patches are affine in the cursor - so it survives across turns and
                    // prefix-cache reuse; prefill/multi-token steps below stay on the direct path).
                    if (_decodeCapture == null)
                    {
                        // 🔴 NOTHING ABOUT BUILDING A CAPTURE MAY BE FATAL. A capture is an OPTIMISATION -
                        // replay a recorded dispatch plan instead of walking the graph - and the fallback
                        // right below is a complete, correct decode. So a capture that cannot be built is
                        // a performance outcome, never a failed generation.
                        //
                        // ⚠️ THE NARROW FIX WAS NOT ENOUGH. 5.2.17 made one guard inside TryCaptureAsync
                        // return null instead of throwing ("decode capture parity mismatch"), and the very
                        // next report was a DIFFERENT guard in the same method - "attention slot count
                        // mismatch 44/49" - still killing every chat turn. Guards are exactly the thing
                        // that gets added to over time, so catching them one by one at the throw site is a
                        // losing shape. The contract belongs here, where the fallback lives: whatever goes
                        // wrong, this optimisation is abandoned and the model still answers.
                        try
                        {
                            _decodeCapture = await WebGPUDecodeCapture.TryCaptureAsync(_session, stepIds[0], _argmax);
                        }
                        catch (Exception ex)
                        {
                            // Loud, once - EnableWebGPUDecodeCapture goes false below, so this cannot spam.
                            Console.WriteLine($"[capture] decode capture ABANDONED: {ex.GetType().Name}: "
                                + $"{ex.Message}. Decoding directly; this session will not retry capture.");
                            _decodeCapture = null;
                        }

                        if (_decodeCapture != null)
                            outputs = _decodeCapture.StepOutputs;   // this step's answer (recomputed after the capture's probe passes)
                        else
                        {
                            EnableWebGPUDecodeCapture = false;  // non-WebGPU, or uncapturable: don't retry
                            outputs = await RunDirectAsync();
                        }
                    }
                    else if (!wantSampling && !(wantRepPen && generated.Count > 0))
                    {
                        // Greedy fast path: the plan's folded argmax + ONE fence per token.
                        fastToken = await _decodeCapture.PatchAndDecodeGreedyAsync(stepIds[0], _session.DecodePastLen);
                        outputs = null;
                    }
                    else if (UseDeviceSampling)
                    {
                        // Sampled fast path (the /ai-chat top_p + repetition-penalty config): patch + replay, then the
                        // draw ON THE DEVICE (penalty, candidates, draw): the one-int read is the per-token fence, not
                        // the ~vocab*4-byte logits read
                        var lv = await _decodeCapture.PatchAndReplayKeepLogitsAsync(stepIds[0], _session.DecodePastLen);
                        var (tok, row) = await DeviceSampleAsync(lv, _vocabSize(lv), stagedU);
                        if (tok is int dt) fastToken = dt; else { fastLogits = row; penaltyApplied = true; }
                        outputs = null;
                    }
                    else
                    {
                        // Sampled fast path, host draw (DeviceSampling off): patch + replay + direct host logits read -
                        // still ONE fence per token.
                        fastLogits = await _decodeCapture.PatchAndReadLogitsAsync(stepIds[0], _session.DecodePastLen);
                        outputs = null;
                    }
                }
                else
                    outputs = await RunDirectAsync();

                async Task<IReadOnlyDictionary<string, Tensor>> RunDirectAsync()
                {
                    var idView = UploadInputIds(stepIds);
                    return await _session.RunDecodeStepAsync(new Dictionary<string, Tensor>
                    { ["input_ids"] = new Tensor(idView, new[] { 1, stepIds.Length }, "input_ids") });
                }

            written += stepIds.Length;   // this step ran: its tokens are now in the K/V cache

            int next;
            if (fastToken is int ft)
            {
                next = ft;   // greedy replay already produced the token (folded argmax, one fence)
            }
            else if (fastLogits != null)
            {
                // Sampled replay already produced host logits (one fence) - sample on the host (a device-sampler fallback:
                // the row is already penalised, and the draw uses the uniform the device was given).
                if (wantRepPen && generated.Count > 0 && !penaltyApplied)
                    TextGenerationSampler.ApplyRepetitionPenalty(fastLogits, RepeatWindow(generated, config!.RepeatLastN), config!.RepetitionPenalty);
                var r = penaltyApplied ? _fallbackRng : rng;
                next = config?.Strategy switch
                {
                    "top_k" => TextGenerationSampler.TopK(fastLogits, config.TopK, config.Temperature, r),
                    "top_p" => TextGenerationSampler.TopP(fastLogits, config.TopP, config.Temperature, r),
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
            else if (UseDeviceSampling)
            {
                // penalty, candidates and draw on the device: one int read (it used to copy the row, synchronise, then read the
                // whole vocabulary: two waits and ~vocab*4 bytes per token)
                var (tok, row) = await DeviceSampleAsync(lastLogits, vocab, stagedU);
                next = tok ?? (config?.Strategy switch
                {
                    "top_k" => TextGenerationSampler.TopK(row!, config.TopK, config.Temperature, _fallbackRng),
                    "top_p" => TextGenerationSampler.TopP(row!, config.TopP, config.Temperature, _fallbackRng),
                    _ => TextGenerationSampler.Greedy(row!),
                });
            }
            else
            {
                var read = EnsureLogitsReadBuf(vocab);
                await read.View.SubView(0, vocab).CopyFromAsync(lastLogits);
                await _accelerator.SynchronizeAsync();
                var logits = await read.CopyToHostAsync<float>(0, vocab);
                if (repPen)
                    TextGenerationSampler.ApplyRepetitionPenalty(logits, RepeatWindow(generated, config!.RepeatLastN), config!.RepetitionPenalty);
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
            // the text: a stop string cuts it at its first match and ends the turn; what cannot still start one is streamed
            if (!await text.PushAsync(next)) { stop = StopReason.StopString; break; }

            stepIds = new[] { next }; // incremental decode: after the prefill, feed only the new token
            }
            // _inputIdsBuf / _logitsReadBuf are instance-owned — disposed in Dispose(), never per-token
        }

        // Flush the detokenizer + any held-back safe tail (a no-op after a stop string).
        await text.FinishAsync();

        // Record what the KV cache now holds (only rows a step wrote: see RecordResident) so the NEXT request can
        // reuse this turn's prompt+response as its prefix.
        RecordResident(promptIds, generated, written);

        return new GenerationResult(text.Text, promptIds.Length, generated.Count, stop) { GeneratedTokenIds = generated.ToArray() };
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
        if (_batch != null) throw new NotSupportedException("GenerateFirstTokenIdsAsync drives the session engine's decode (create the generator with engine: GgufEngine.Generator)");
        int maxNew = config?.MaxNewTokens is int mn && mn > 0 ? mn : 10;

        bool truncated;
        (promptIds, maxNew, truncated) = GenerationText.FitPrompt(promptIds, maxNew, _maxSeqLen);

        // Same prefix-cache reuse decision as GenerateAsync (see that method for the correctness rationale).
        int P = 0;
        if (EnablePrefixCache && !truncated && _cachedIds != null)
        {
            int maxP = Math.Min(_cachedIds.Length, promptIds.Length - 1);
            while (P < maxP && _cachedIds[P] == promptIds[P]) P++;
            if (P < MinReusePrefix) P = 0;
            P = ConvStateAllowsReuse(P);
        }

        int[] stepIds;
        if (P > 0) { _session.SetGGUFDecodePastLen(P); stepIds = promptIds[P..]; }
        else { _session.ResetGGUFDecode(); stepIds = promptIds; }
        LastReusedPrefix = P;
        ForgetCachedRowsFrom(P);
        int written = P;

        var generated = new List<int>();
        double ttftMs = 0;
        for (int step = 0; step < maxNew; step++)
        {
            var idView = UploadInputIds(stepIds);
            var sw = step == 0 ? System.Diagnostics.Stopwatch.StartNew() : null;
            var outputs = await _session.RunDecodeStepAsync(new Dictionary<string, Tensor>
            { ["input_ids"] = new Tensor(idView, new[] { 1, stepIds.Length }, "input_ids") });
            written += stepIds.Length;

            var logitsT = outputs.TryGetValue("logits", out var l) ? l : outputs.Values.First();
            int vocab = logitsT.Shape[^1];
            int seqOut = logitsT.ElementCount / vocab;
            long lastOff = (long)(seqOut - 1) * vocab;
            int next = await _argmax.ArgMaxAsync(logitsT.Data.SubView(lastOff, vocab), vocab);
            if (sw != null) { await _accelerator.SynchronizeAsync(); sw.Stop(); ttftMs = sw.Elapsed.TotalMilliseconds; }

            generated.Add(next);
            stepIds = new[] { next };
        }

        RecordResident(promptIds, generated, written);

        return (generated.ToArray(), ttftMs);
    }

    /// <summary>The repetition-penalty window: the last <paramref name="repeatLastN"/> generated
    /// tokens (Ollama's repeat_last_n semantics), or the full history when n &lt;= 0. Penalizing the
    /// FULL history permanently punishes every common grammar token and decays long generations
    /// into word salad - the window keeps the penalty anti-loop, not anti-language.</summary>
    /// <summary>
    /// Sampled steps (and greedy steps with a repetition penalty) draw on the device (<see cref="DeviceSampler"/>: penalty, candidates and draw, one
    /// int read per token) instead of reading the vocabulary's logits to the host. Same strategies and semantics as <see cref="TextGenerationSampler"/>,
    /// the uniform from the generation's own <see cref="Random"/> (seeded runs stay reproducible); ties go to the lower id. Off: the host draws.
    /// Not on WebGL (the host draws there).
    /// </summary>
    public bool DeviceSampling
    {
        get => _batch != null ? _batch.Decoder.DeviceSampling : _deviceSampling;
        set { if (_batch != null) _batch.Decoder.DeviceSampling = value && GgufBatchDecoder.SupportsDeviceDraws(_accelerator); else _deviceSampling = value; }
    }
    private bool _deviceSampling = true;
    private bool UseDeviceSampling => DeviceSampling && DeviceSampler.Supports(_accelerator);
    private DeviceSampler? _deviceSampler;
    private readonly FixedUniform _fallbackRng = new();
    private static int _vocabSize(ArrayView1D<float, Stride1D.Dense> logits) => (int)logits.Length;

    // a Random whose next uniform is set by the caller: a device-sampler fallback samples the row on the host with the uniform the device was given
    private sealed class FixedUniform : Random
    {
        public double U;
        public override double NextDouble() => U;
        protected override double Sample() => U;
    }

    /// <summary>Tokens the device sampler drew since the generator was made (diagnostics: which path a generation took).</summary>
    public int DeviceSampledTokens { get; private set; }
    /// <summary>Steps where the device sampler's candidates could not answer and the host sampled the row (diagnostics).</summary>
    public int DeviceSamplerFallbacks { get; private set; }

    // the step's draw settings to the device (before its forward is queued); returns the uniform (one per sampled token, as the host sampler draws;
    // greedy draws none)
    private double StageDeviceSample(GenerationConfig? config, List<int> generated, Random rng)
    {
        var strategy = config?.Strategy switch { "top_k" => DeviceSampler.Strategy.TopK, "top_p" => DeviceSampler.Strategy.TopP, _ => DeviceSampler.Strategy.Greedy };
        double u = strategy == DeviceSampler.Strategy.Greedy ? 0.0 : rng.NextDouble();
        bool repPen = config?.RepetitionPenalty is float rp && rp != 1f && generated.Count > 0;
        _deviceSampler ??= new DeviceSampler(_accelerator);
        _deviceSampler.Stage(strategy, config?.Temperature ?? 1f, config?.TopP ?? 1f, config?.TopK ?? 0, u,
            repPen ? RepeatWindow(generated, config!.RepeatLastN) : Array.Empty<int>(), repPen ? config!.RepetitionPenalty : 1f);
        return u;
    }

    // the staged draw: the token, or (null, the penalised row) when the candidates cannot answer (the caller samples the row with _fallbackRng)
    private async Task<(int? Token, float[]? Row)> DeviceSampleAsync(ArrayView1D<float, Stride1D.Dense> logits, int vocab, double u)
    {
        int tok = await _deviceSampler!.SampleAsync(logits, vocab);
        if (tok >= 0) { DeviceSampledTokens++; return (tok, null); }
        DeviceSamplerFallbacks++;
        _fallbackRng.U = u;
        return (null, await logits.SubView(0, vocab).CopyToHostAsync());
    }

    private static int[] RepeatWindow(List<int> generated, int repeatLastN)
    {
        if (repeatLastN <= 0 || generated.Count <= repeatLastN) return generated.ToArray();
        var window = new int[repeatLastN];
        generated.CopyTo(generated.Count - repeatLastN, window, 0, repeatLastN);
        return window;
    }

    /// <summary>Upload <paramref name="stepIds"/> into the recycled instance buffer; returns a view of length N.
    /// Buffer is sized to <see cref="_maxSeqLen"/> once and never disposed mid-generation (WebGPU lifetime).</summary>
    private ArrayView1D<float, Stride1D.Dense> UploadInputIds(int[] stepIds)
    {
        int n = stepIds.Length;
        if (n > _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(stepIds), n, $"input_ids length {n} exceeds maxSeqLen {_maxSeqLen}");
        _inputIdsBuf ??= _accelerator.Allocate1D<float>(_maxSeqLen);
        var idf = new float[n];
        for (int i = 0; i < n; i++) idf[i] = stepIds[i];
        _inputIdsBuf.View.SubView(0, n).CopyFromCPU(idf);
        return _inputIdsBuf.View.SubView(0, n);
    }

    /// <summary>Recycled logits staging buffer for sampling readback (grows to vocab once).</summary>
    private MemoryBuffer1D<float, Stride1D.Dense> EnsureLogitsReadBuf(int vocab)
    {
        if (_logitsReadBuf == null || _logitsReadBuf.Length < vocab)
        {
            // Grow only between tokens (prior step fully drained). Retire old buffer to Dispose().
            var old = _logitsReadBuf;
            _logitsReadBuf = _accelerator.Allocate1D<float>(vocab);
            old?.Dispose();
        }
        return _logitsReadBuf;
    }

    /// <summary>Releases the decode KV-cache + argmax + recycled decode buffers, the session and stream <see cref="CreateAsync"/> made this generator own, or the
    /// batch decoder. Never a session passed to the constructor, nor the accelerator (caller-owned).</summary>
    public void Dispose()
    {
        if (_batch != null) { _batch.Dispose(); _ownedStream?.Dispose(); return; }
        _decodeCapture?.Dispose(); _decodeCapture = null;
        _inputIdsBuf?.Dispose(); _inputIdsBuf = null;
        _logitsReadBuf?.Dispose(); _logitsReadBuf = null;
        _cache.Dispose();
        _argmax.Dispose();
        _ownedSession?.Dispose(); _ownedSession = null;
        _ownedStream?.Dispose(); _ownedStream = null;
    }
}
