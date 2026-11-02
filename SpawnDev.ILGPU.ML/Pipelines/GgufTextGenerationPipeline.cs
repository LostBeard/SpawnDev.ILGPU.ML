using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Preprocessing;
using System.Text;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// First-class, ARCHITECTURE-AGNOSTIC text generation for GGUF chat models on top of
/// <see cref="InferenceSession"/> + <see cref="GgufGenerator"/>. One call turns a user prompt (or a
/// multi-turn conversation) into the model's answer: the chat template is auto-detected from the GGUF's
/// own <c>tokenizer.chat_template</c> + architecture (<see cref="ChatTemplates.DetectChatFormat"/> →
/// ChatML / Llama3 / gemma4) and applied for you (<see cref="ChatTemplates.BuildChatPrompt"/>), then the
/// O(n) incremental KV-cache decode + sampler run via the shared <see cref="GgufGenerator"/> (greedy /
/// top-k / top-p / repetition penalty, KV-prefix reuse, UTF-8-safe streaming, stop tokens + strings).
///
/// Modeled on the Transformers.js <c>pipeline('text-generation', model)</c> ergonomics: a single object you
/// create once and then call with EITHER a raw string OR a list of <c>(role, content)</c> chat messages;
/// the caller never tokenizes or hand-builds a template. Browser-portable (async readback, no sync GPU
/// waits). Works for qwen (ChatML), llama/smollm (ChatML/Llama3), gemma4, and any GGUF whose template the
/// detector recognizes (unknown → ChatML best-effort).
///
/// Usage (one-call factory — recommended):
/// <code>
/// using var pipe = await GgufTextGenerationPipeline.CreateFromFileAsync(accelerator, "model.gguf");
/// string answer = await pipe.GenerateAsync("What is the capital of France?");
/// // or multi-turn, with streaming:
/// var msgs = new (string, string)[] { ("system", "You are helpful."), ("user", "Hi!") };
/// await pipe.GenerateAsync(msgs, onToken: (n, soFar) => { Console.Write(soFar); return Task.CompletedTask; });
/// </code>
/// Or wrap an already-loaded session: <c>new GgufTextGenerationPipeline(session, acc, parsedModel)</c>.
/// </summary>
public sealed class GgufTextGenerationPipeline : IDisposable
{
    private readonly GgufGenerator _gen;            // owned: the session engine or the batch decoder (GgufGenerator.CreateAsync chose)
    private readonly GGUFModel _model;
    private readonly SentencePieceTokenizer _tokenizer;
    private readonly ChatTemplates.ChatFormat _format;

    /// <summary>The model's tokenizer (SentencePiece, from the GGUF vocab).</summary>
    public SentencePieceTokenizer Tokenizer => _tokenizer;

    /// <summary>The chat format auto-detected from the GGUF (ChatML / Llama3 / gemma4 / Unknown→ChatML).</summary>
    public ChatTemplates.ChatFormat ChatFormat => _format;

    /// <summary>The model architecture string from the GGUF metadata (e.g. "qwen2", "llama", "gemma3").</summary>
    public string Architecture => _model.Architecture;

    /// <summary>The generator underneath (its engine, prefix cache and diagnostics); owned by this pipeline.</summary>
    public GgufGenerator Generator => _gen;

    /// <summary>The engine that decodes: <see cref="GgufEngine.BatchDecoder"/> or <see cref="GgufEngine.Generator"/> (never Auto: the factory resolves it).</summary>
    public GgufEngine Engine => _gen.Engine;

    /// <summary>Why <see cref="GgufEngine.Auto"/> chose the generator engine for this model ("" when it chose the batch decoder or the caller chose).</summary>
    public string EngineReason => _gen.EngineReason;

    /// <summary>The underlying decode-enabled <see cref="InferenceSession"/>. Advanced consumers -
    /// capture/replay probes, the Ollama-server model runner - drive decode state directly through it;
    /// the pipeline still owns its lifetime (do not dispose). The generator engine only: a pipeline on the batch decoder has no session
    /// (create it with <see cref="GgufEngine.Generator"/> to get one).</summary>
    public InferenceSession Session => _gen.Session;

    /// <summary>The batch decoder, when <see cref="Engine"/> is <see cref="GgufEngine.BatchDecoder"/> (null otherwise): its decode timings, row reads and settings.</summary>
    public GgufBatchDecoder? BatchDecoder => _gen.BatchDecoder;

    /// <summary>WebGPU decode capture/replay - see <see cref="GgufGenerator.EnableWebGPUDecodeCapture"/> (686ms/tok -> ~21ms/tok measured; no-op on non-WebGPU
    /// backends). The batch decoder: <see cref="GgufBatchDecoder.CaptureWebGPU"/> (on by default on WebGPU).</summary>
    public bool EnableWebGPUDecodeCapture
    {
        get => _gen.EnableWebGPUDecodeCapture;
        set => _gen.EnableWebGPUDecodeCapture = value;
    }

    /// <summary>Sampled steps draw on the device (one int read per token) - see <see cref="GgufGenerator.DeviceSampling"/>. Default true where supported.</summary>
    public bool DeviceSampling
    {
        get => _gen.DeviceSampling;
        set => _gen.DeviceSampling = value;
    }

    /// <summary>The session engine: tokens drawn on the device so far, and steps the host sampled instead (see <see cref="GgufGenerator.DeviceSampledTokens"/>);
    /// (0, 0) on the batch decoder (its draws: <see cref="GgufBatchDecoder.RowReads"/>).</summary>
    public (int Device, int Fallbacks) DeviceSamplerCounts => (_gen.DeviceSampledTokens, _gen.DeviceSamplerFallbacks);

    /// <summary>Diagnostics for the active decode capture (ops + patch counts), or null (always on the batch decoder).</summary>
    public (int Ops, int Scalars, int Copies, int Slots)? DecodeCaptureInfo => _gen.DecodeCaptureInfo;

    /// <summary>Length of the KV-prefix the last request reused (0 = full prefill). Diagnostic - lets a test
    /// assert WHICH path a generation took (see Lfm2_PrefixCacheReuse_SamePromptTwice_IsIdentical).</summary>
    public int LastReusedPrefix => _gen.LastReusedPrefix;

    /// <summary>Per-phase ms of the most recent patched replay step (diagnostics), or null (always on the batch decoder: <see cref="GgufBatchDecoder.DecodeTimings"/>).</summary>
    public (double Patch, double Replay, double Sync)? LastDecodeCaptureStepMs => _gen.LastDecodeCaptureStepMs;

    /// <summary>Patch sub-phases of the most recent replay step (diagnostics), or null.</summary>
    public (double Input, double Scalars, double Slots, double Copies)? LastDecodeCapturePatchSplitMs => _gen.LastDecodeCapturePatchSplitMs;

    /// <summary>Default cap on generated tokens when neither the call nor the <see cref="GenerationConfig"/> sets one.</summary>
    public int MaxNewTokens { get; set; } = 256;

    /// <summary>
    /// Wrap an already-loaded <paramref name="session"/> for <paramref name="model"/> (the session engine). The caller OWNS the
    /// session (this pipeline does NOT dispose it). Allocates the decode KV-cache (sized to
    /// <paramref name="maxSeqLen"/>) and enables incremental decode via <see cref="GgufGenerator"/>.
    /// </summary>
    public GgufTextGenerationPipeline(InferenceSession session, Accelerator accelerator, GGUFModel model, int maxSeqLen = 4096)
        : this(new GgufGenerator(session, accelerator, model ?? throw new ArgumentNullException(nameof(model)), maxSeqLen)) { }

    private GgufTextGenerationPipeline(GgufGenerator generator)
    {
        _gen = generator;
        _model = generator.Gguf;
        _tokenizer = generator.Tokenizer;
        _format = ChatTemplates.DetectChatFormat(_model);
    }

    /// <summary>
    /// The engine <see cref="GgufEngine.Auto"/> picks for <paramref name="model"/> on <paramref name="accelerator"/>: the batch decoder when it loads the model
    /// (<see cref="GgufBatchDecoder.Supports"/>) and the backend is CUDA, OpenCL or WebGPU (MEASURED 2026-11-01, Qwen2.5-0.5B, one chat answer: WebGPU 260.8 tok/s
    /// against 67.8 on the session engine, CUDA 165.1 against 104.6, OpenCL 184.0 against 47.3); the session engine otherwise (other architectures, WebGL, Wasm,
    /// the CPU backends).
    /// </summary>
    public static GgufEngine ChooseEngine(Accelerator accelerator, GGUFModel model, out string reason)
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
        { reason = $"{accelerator.AcceleratorType}: the batch decoder is gated on CUDA, OpenCL and WebGPU"; return GgufEngine.Generator; }
        if (!GgufBatchDecoder.Supports(model, out var why)) { reason = "the batch decoder cannot load it: " + why; return GgufEngine.Generator; }
        reason = "";
        return GgufEngine.BatchDecoder;
    }

    /// <summary>
    /// One-call factory: load a GGUF model from a file and build a ready-to-use text-generation pipeline on the engine
    /// <paramref name="engine"/> picks (<see cref="GgufGenerator.CreateAsync"/>). The pipeline owns the file and the engine and disposes them on
    /// <see cref="Dispose"/>. Mirrors Transformers.js <c>pipeline('text-generation', path)</c>.
    /// </summary>
    public static async Task<GgufTextGenerationPipeline> CreateFromFileAsync(Accelerator accelerator, string ggufPath,
        int maxSeqLen = 4096, Action<string, int>? onProgress = null, CancellationToken ct = default, GgufEngine engine = GgufEngine.Auto)
    {
        var fs = new FileStream(ggufPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
        try { return new GgufTextGenerationPipeline(await GgufGenerator.CreateAsync(accelerator, fs, maxSeqLen, null, onProgress, engine, ownsStream: true, ct).ConfigureAwait(false)); }
        catch { fs.Dispose(); throw; }
    }

    /// <summary>
    /// One-call factory from a SEEKABLE .gguf stream (browser / hub / OPFS delivery) on the engine <paramref name="engine"/> picks
    /// (<see cref="GgufGenerator.CreateAsync"/>): the weights stream to the GPU without ever materializing the model as a byte[] (a browser JS stream goes
    /// JS -> GPU, never the .NET heap). The header is parsed once. Keep the stream open until the pipeline is disposed (the session engine reads it on
    /// demand; the batch engine disposes it with itself).
    /// </summary>
    public static async Task<GgufTextGenerationPipeline> CreateFromStreamAsync(Accelerator accelerator, Stream seekableGguf,
        int maxSeqLen = 4096, Action<string, int>? onProgress = null, CancellationToken ct = default, GgufEngine engine = GgufEngine.Auto)
    {
        if (!seekableGguf.CanSeek)
            throw new ArgumentException("CreateFromStreamAsync requires a seekable stream.", nameof(seekableGguf));
        return new GgufTextGenerationPipeline(await GgufGenerator.CreateAsync(accelerator, seekableGguf, maxSeqLen, null, onProgress, engine, ownsStream: false, ct).ConfigureAwait(false));
    }

    /// <summary>
    /// Generate a response to a multi-turn <paramref name="messages"/> conversation (each entry is a
    /// <c>(Role, Content)</c> pair; Role is conventionally "system" / "user" / "assistant"). The model's own
    /// chat template is applied automatically and the assistant turn is generated. <paramref name="onToken"/>
    /// streams <c>(tokenCount, textSoFar)</c> after each emitted token. Returns the decoded assistant text.
    /// </summary>
    public async Task<string> GenerateAsync(IReadOnlyList<(string Role, string Content)> messages,
        GenerationConfig? config = null, bool thinking = true, Func<int, string, Task>? onToken = null,
        CancellationToken ct = default)
    {
        var (promptIds, stopIds) = ChatTemplates.BuildChatPrompt(_model, _tokenizer, messages, thinking);
        return await GenerateFromIdsAsync(promptIds, stopIds, config, onToken, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Generate a response to a single <paramref name="userPrompt"/> (optionally with a
    /// <paramref name="systemPrompt"/>). Convenience over the messages overload for the common one-shot case.
    /// </summary>
    public Task<string> GenerateAsync(string userPrompt, string? systemPrompt = null, int maxNewTokens = 0,
        GenerationConfig? config = null, Func<int, string, Task>? onToken = null, CancellationToken ct = default)
    {
        var messages = string.IsNullOrEmpty(systemPrompt)
            ? new List<(string, string)> { ("user", userPrompt) }
            : new List<(string, string)> { ("system", systemPrompt!), ("user", userPrompt) };
        if (maxNewTokens > 0)
        {
            config ??= new GenerationConfig();
            config.MaxNewTokens = maxNewTokens;
        }
        return GenerateAsync(messages, config, onToken: onToken, ct: ct);
    }

    private async Task<string> GenerateFromIdsAsync(int[] promptIds, int[] stopIds, GenerationConfig? config,
        Func<int, string, Task>? onToken, CancellationToken ct)
    {
        // Default the token budget from the pipeline if the caller left it open, so a bare GenerateAsync("...")
        // doesn't run to the cache limit.
        if (config == null) config = new GenerationConfig { MaxNewTokens = MaxNewTokens };
        else if (config.MaxNewTokens is null or <= 0) config.MaxNewTokens = MaxNewTokens;

        var sb = onToken != null ? new StringBuilder() : null;
        int count = 0;
        Func<string, Task>? onDelta = onToken == null ? null : async delta =>
        {
            sb!.Append(delta);
            count++;
            await onToken(count, sb.ToString()).ConfigureAwait(false);
        };
        var res = await _gen.GenerateAsync(promptIds, config, stopTokenIds: stopIds, onDelta: onDelta, ct: ct).ConfigureAwait(false);
        LastStopReason = res.Stop;
        return res.Text;
    }

    /// <summary>Why the most recent <see cref="GenerateAsync(IReadOnlyList{ValueTuple{string,string}},GenerationConfig?,Func{int,string,Task}?,CancellationToken)"/>
    /// stopped - <see cref="StopReason.Length"/> means the answer was TRUNCATED at MaxNewTokens
    /// (small models often never emit EOS on open prompts; a UI should show a length-limit marker).</summary>
    public StopReason LastStopReason { get; private set; }

    /// <summary>Releases the decode KV-cache + argmax buffers, and the session ONLY if this pipeline created
    /// it (factory path). Never disposes a session/accelerator passed into the constructor (caller-owned).</summary>
    public void Dispose() => _gen.Dispose();
}

/// <summary>The decode engine of a <see cref="GgufTextGenerationPipeline"/>.</summary>
public enum GgufEngine
{
    /// <summary>The batch decoder when it loads the model on a backend it is gated on (<see cref="GgufTextGenerationPipeline.ChooseEngine"/>), else the generator.</summary>
    Auto,
    /// <summary><see cref="GgufGenerator"/> over an <see cref="InferenceSession"/>: every architecture the graph builder knows; exposes the session.</summary>
    Generator,
    /// <summary><see cref="GgufBatchDecoder"/> with one sequence (qwen2, qwen3, llama, llama 3 rope_freqs included): the faster engine; throws when it cannot load the model.</summary>
    BatchDecoder,
}
