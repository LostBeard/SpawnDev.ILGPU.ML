using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;

namespace OllamaServer.Console;

/// <summary>
/// A model loaded onto the accelerator: the session, the reusable generator, the tokenizer, and the
/// detected chat format. One per resident model.
/// </summary>
public sealed class LoadedModel : IDisposable
{
    public required OllamaModel Meta { get; init; }
    public required GGUFModel Gguf { get; init; }
    public required InferenceSession Session { get; init; }
    public required GgufGenerator Generator { get; init; }
    public required SentencePieceTokenizer Tokenizer { get; init; }
    public required ChatTemplates.ChatFormat Format { get; init; }

    public void Dispose() { Generator.Dispose(); Session.Dispose(); }
}

/// <summary>
/// Loads models from the Ollama cache on demand and serializes generation. <see cref="InferenceSession"/>
/// is single-decode-at-a-time (one mutable KV cursor, no locks), so all generation goes through a single
/// gate — which is also what real Ollama does on one GPU. v1 keeps ONE model resident and swaps when a
/// request asks for a different one (these models are GBs; one-at-a-time bounds GPU memory).
/// </summary>
public sealed class ModelRegistry : IAsyncDisposable
{
    private readonly OllamaModelStore _store;
    private readonly Accelerator _accelerator;
    private readonly int _maxSeqLen;
    private readonly SemaphoreSlim _gate = new(1, 1); // serialize decode (and model swaps)
    private LoadedModel? _resident;

    public ModelRegistry(OllamaModelStore store, Accelerator accelerator, int maxSeqLen = 8192)
    {
        _store = store;
        _accelerator = accelerator;
        _maxSeqLen = maxSeqLen;
    }

    /// <summary>The model store backing this registry (for listing / metadata endpoints).</summary>
    public OllamaModelStore Store => _store;

    /// <summary>
    /// A serialized lease on a loaded model. Hold it for the duration of ONE generation, then dispose to
    /// release the gate. The model is loaded (or swapped in) before the lease is returned.
    /// </summary>
    public sealed class Lease : IDisposable
    {
        private readonly SemaphoreSlim _gate;
        private bool _released;
        public LoadedModel Model { get; }
        internal Lease(LoadedModel model, SemaphoreSlim gate) { Model = model; _gate = gate; }
        public void Dispose() { if (!_released) { _released = true; _gate.Release(); } }
    }

    /// <summary>
    /// Acquire the generation gate and ensure <paramref name="modelName"/> is the resident model (loading
    /// or swapping as needed). Throws <see cref="FileNotFoundException"/> if the model isn't in the cache.
    /// </summary>
    public async Task<Lease> AcquireAsync(string modelName, CancellationToken ct = default)
    {
        await _gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            var meta = _store.Resolve(modelName)
                ?? throw new FileNotFoundException($"Model '{modelName}' is not in the Ollama cache.");

            if (_resident == null || !string.Equals(_resident.Meta.Name, meta.Name, StringComparison.OrdinalIgnoreCase))
            {
                _resident?.Dispose();
                _resident = null;
                _resident = await LoadAsync(meta, ct).ConfigureAwait(false);
            }
            return new Lease(_resident, _gate);
        }
        catch
        {
            _gate.Release(); // never strand the gate on a load failure
            throw;
        }
    }

    private async Task<LoadedModel> LoadAsync(OllamaModel meta, CancellationToken ct)
    {
        await using var hs = File.OpenRead(meta.GgufPath);
        var gguf = await GGUFParser.ParseHeaderAsync(hs, ct).ConfigureAwait(false);
        var tok = SentencePieceTokenizer.FromGGUF(gguf)
            ?? throw new InvalidOperationException($"'{meta.Name}' has no SentencePiece tokenizer metadata.");
        var session = await InferenceSession.CreateFromGGUFFileAsync(_accelerator, meta.GgufPath, ct: ct)
            .ConfigureAwait(false);
        int ctxCap = gguf.ContextLength > 0 ? Math.Min((int)gguf.ContextLength, _maxSeqLen) : _maxSeqLen;
        var gen = new GgufGenerator(session, _accelerator, gguf, maxSeqLen: ctxCap);
        return new LoadedModel
        {
            Meta = meta,
            Gguf = gguf,
            Session = session,
            Generator = gen,
            Tokenizer = tok,
            Format = ChatTemplates.DetectChatFormat(gguf),
        };
    }

    public async ValueTask DisposeAsync()
    {
        await _gate.WaitAsync().ConfigureAwait(false);
        try { _resident?.Dispose(); _resident = null; }
        finally { _gate.Release(); _gate.Dispose(); }
    }
}
