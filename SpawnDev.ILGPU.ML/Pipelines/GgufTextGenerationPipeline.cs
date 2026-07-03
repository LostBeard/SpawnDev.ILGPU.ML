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
    private readonly GgufGenerator _gen;
    private readonly GGUFModel _model;
    private readonly SentencePieceTokenizer _tokenizer;
    private readonly ChatTemplates.ChatFormat _format;
    private readonly InferenceSession? _ownedSession; // non-null only when WE created the session (factory path)

    /// <summary>The model's tokenizer (SentencePiece, from the GGUF vocab).</summary>
    public SentencePieceTokenizer Tokenizer => _tokenizer;

    /// <summary>The chat format auto-detected from the GGUF (ChatML / Llama3 / gemma4 / Unknown→ChatML).</summary>
    public ChatTemplates.ChatFormat ChatFormat => _format;

    /// <summary>The model architecture string from the GGUF metadata (e.g. "qwen2", "llama", "gemma3").</summary>
    public string Architecture => _model.Architecture;

    /// <summary>The underlying decode-enabled <see cref="InferenceSession"/>. Advanced consumers -
    /// capture/replay probes, the Ollama-server model runner - drive decode state directly through it;
    /// the pipeline still owns its lifetime (do not dispose).</summary>
    public InferenceSession Session => _gen.Session;

    /// <summary>Opt-in WebGPU decode capture/replay - see <see cref="GgufGenerator.EnableWebGPUDecodeCapture"/>
    /// (686ms/tok -> ~21ms/tok measured; no-op on non-WebGPU backends).</summary>
    public bool EnableWebGPUDecodeCapture
    {
        get => _gen.EnableWebGPUDecodeCapture;
        set => _gen.EnableWebGPUDecodeCapture = value;
    }

    /// <summary>Diagnostics for the active decode capture (ops + patch counts), or null.</summary>
    public (int Ops, int Scalars, int Copies, int Slots)? DecodeCaptureInfo => _gen.DecodeCaptureInfo;

    /// <summary>Per-phase ms of the most recent patched replay step (diagnostics), or null.</summary>
    public (double Patch, double Replay, double Sync)? LastDecodeCaptureStepMs => _gen.LastDecodeCaptureStepMs;

    /// <summary>Patch sub-phases of the most recent replay step (diagnostics), or null.</summary>
    public (double Input, double Scalars, double Slots, double Copies)? LastDecodeCapturePatchSplitMs => _gen.LastDecodeCapturePatchSplitMs;

    /// <summary>Default cap on generated tokens when neither the call nor the <see cref="GenerationConfig"/> sets one.</summary>
    public int MaxNewTokens { get; set; } = 256;

    /// <summary>
    /// Wrap an already-loaded <paramref name="session"/> for <paramref name="model"/>. The caller OWNS the
    /// session (this pipeline does NOT dispose it). Allocates the decode KV-cache (sized to
    /// <paramref name="maxSeqLen"/>) and enables incremental decode via <see cref="GgufGenerator"/>.
    /// </summary>
    public GgufTextGenerationPipeline(InferenceSession session, Accelerator accelerator, GGUFModel model, int maxSeqLen = 4096)
        : this(session, accelerator, model, maxSeqLen, ownedSession: null) { }

    private GgufTextGenerationPipeline(InferenceSession session, Accelerator accelerator, GGUFModel model,
        int maxSeqLen, InferenceSession? ownedSession)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _tokenizer = SentencePieceTokenizer.FromGGUF(model)
            ?? throw new InvalidOperationException("GGUF model has no SentencePiece tokenizer metadata.");
        _format = ChatTemplates.DetectChatFormat(model);
        _gen = new GgufGenerator(session, accelerator, model, maxSeqLen);
        _ownedSession = ownedSession;
    }

    /// <summary>
    /// One-call factory: load a GGUF model from a file and build a ready-to-use text-generation pipeline
    /// (parses the header for the tokenizer + chat format, streams the weights to the GPU, allocates the
    /// decode cache). The returned pipeline OWNS the underlying session and disposes it on
    /// <see cref="Dispose"/>. Mirrors Transformers.js <c>pipeline('text-generation', path)</c>.
    /// </summary>
    public static async Task<GgufTextGenerationPipeline> CreateFromFileAsync(Accelerator accelerator, string ggufPath,
        int maxSeqLen = 4096, Action<string, int>? onProgress = null, CancellationToken ct = default)
    {
        var session = await InferenceSession.CreateFromGGUFFileAsync(accelerator, ggufPath, onProgress, ct).ConfigureAwait(false);
        GGUFModel model;
        await using (var fs = new FileStream(ggufPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, useAsync: true))
            model = await GGUFParser.ParseHeaderAsync(fs, ct).ConfigureAwait(false);
        return new GgufTextGenerationPipeline(session, accelerator, model, maxSeqLen, ownedSession: session);
    }

    /// <summary>
    /// One-call factory from a SEEKABLE .gguf stream (browser / hub torrent / OPFS delivery): streams the
    /// weights to the GPU without ever materializing the whole model as a byte[]. The stream must outlive
    /// this call and be seekable. The returned pipeline OWNS the session. (The session keeps the stream for
    /// on-demand small-tensor reads; we parse the header once up front for the tokenizer + chat format.)
    /// </summary>
    public static async Task<GgufTextGenerationPipeline> CreateFromStreamAsync(Accelerator accelerator, Stream seekableGguf,
        int maxSeqLen = 4096, Action<string, int>? onProgress = null, CancellationToken ct = default)
    {
        if (!seekableGguf.CanSeek)
            throw new ArgumentException("CreateFromStreamAsync requires a seekable stream.", nameof(seekableGguf));
        seekableGguf.Seek(0, SeekOrigin.Begin);
        var model = await GGUFParser.ParseHeaderAsync(seekableGguf, ct).ConfigureAwait(false);
        seekableGguf.Seek(0, SeekOrigin.Begin);
        var session = await InferenceSession.CreateFromGGUFStreamAsync(accelerator, seekableGguf, onProgress, ct).ConfigureAwait(false);
        return new GgufTextGenerationPipeline(session, accelerator, model, maxSeqLen, ownedSession: session);
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
        var res = await _gen.GenerateAsync(promptIds, config, stopTokenIds: stopIds,
            onDelta: onToken == null ? null : async delta =>
            {
                sb!.Append(delta);
                count++;
                await onToken(count, sb.ToString()).ConfigureAwait(false);
            },
            ct: ct).ConfigureAwait(false);
        LastStopReason = res.Stop;
        return res.Text;
    }

    /// <summary>Why the most recent <see cref="GenerateAsync(IReadOnlyList{ValueTuple{string,string}},GenerationConfig?,Func{int,string,Task}?,CancellationToken)"/>
    /// stopped - <see cref="StopReason.Length"/> means the answer was TRUNCATED at MaxNewTokens
    /// (small models often never emit EOS on open prompts; a UI should show a length-limit marker).</summary>
    public StopReason LastStopReason { get; private set; }

    /// <summary>Releases the decode KV-cache + argmax buffers, and the session ONLY if this pipeline created
    /// it (factory path). Never disposes a session/accelerator passed into the constructor (caller-owned).</summary>
    public void Dispose()
    {
        _gen.Dispose();
        _ownedSession?.Dispose();
    }
}
