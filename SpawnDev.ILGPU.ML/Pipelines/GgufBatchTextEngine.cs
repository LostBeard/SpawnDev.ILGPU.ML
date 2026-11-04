using SpawnDev.ILGPU.ML.Preprocessing;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// One answer at a time on <see cref="GgufBatchDecoder"/> (one sequence), with what a chat needs from <see cref="GgufGenerator"/>: the prompt fitted into the
/// cache, the conversation's prefix reused across requests (only the new part of the next turn is prefilled), greedy / top-k / top-p with the repetition
/// penalty, stop tokens and stop strings, UTF-8-safe streamed text and the stop reason. <see cref="GgufTextGenerationPipeline"/> uses it for the architectures
/// the decoder loads (qwen2, qwen3, llama, llama 3 rope_freqs included) on CUDA, OpenCL and WebGPU.
/// </summary>
/// <remarks>
/// Seeded sampling is reproducible on this engine, not across engines: the decoder's device draws take their uniform from (seed, sequence, position), the
/// generator from a <see cref="Random"/>(seed). Top-k draws on the host (the device draw kernel is top-p only): the candidates are read each step, the row only
/// when k passes them.
/// </remarks>
internal sealed class GgufBatchTextEngine : IDisposable
{
    private readonly GgufBatchDecoder _dec;
    private readonly SentencePieceTokenizer _tokenizer;
    private readonly int _eosId;

    public GgufBatchTextEngine(GgufBatchDecoder decoder, SentencePieceTokenizer tokenizer)
    {
        _dec = decoder;
        _tokenizer = tokenizer;
        _eosId = tokenizer.EosId;
    }

    public GgufBatchDecoder Decoder => _dec;

    /// <summary>Reuse the K/V of the longest prefix a request shares with the previous request's prompt and answer (as <see cref="GgufGenerator.EnablePrefixCache"/>).</summary>
    public bool EnablePrefixCache { get; set; } = true;

    public int LastReusedPrefix => _dec.LastReusedPrefix;

    public async Task<GenerationResult> GenerateAsync(int[] promptIds, GenerationConfig? config = null, IReadOnlyList<string>? stopStrings = null,
        IReadOnlyList<int>? stopTokenIds = null, Func<string, Task>? onDelta = null, CancellationToken ct = default)
    {
        int maxNew = config?.MaxNewTokens is int mn && mn > 0 ? mn : 128;
        bool truncated;
        (promptIds, maxNew, truncated) = GenerationText.FitPrompt(promptIds, maxNew, _dec.MaxSeq);
        // the strategy: greedy, top-k (k best, then all of their mass), top-p (the nucleus)
        float T = config?.Temperature ?? 1f, topP = 1f;
        int topK = 0;
        switch (config?.Strategy)
        {
            case "top_k": topK = Math.Max(1, config.TopK); break;
            case "top_p": topP = config.TopP; break;
            default: T = 0f; break;
        }
        if (!(T > 0f)) { T = 0f; topK = 0; topP = 1f; }
        float penalty = config?.RepetitionPenalty ?? 1f;
        int window = config?.RepeatLastN ?? 64;
        int seed = config?.Seed ?? Random.Shared.Next();
        var stops = new HashSet<int>(stopTokenIds ?? Array.Empty<int>());
        if (_eosId >= 0) stops.Add(_eosId);
        var text = new TextStream(_tokenizer, stopStrings, onDelta);
        int produced = 0;
        var ids = new List<int>();
        StopReason stop;
        int? stopTokenId = null;
        try
        {
            await _dec.GenerateAsync(promptIds, 1, maxNew, T, topP, seed, stops, ct: ct, repetitionPenalty: penalty, repeatLastN: window,
                reusePrefix: EnablePrefixCache && !truncated, shouldContinue: (_, token) => { produced++; ids.Add(token); return text.PushAsync(token); }, topK: topK).ConfigureAwait(false);
            int stopToken = _dec.LastStopTokens.Length > 0 ? _dec.LastStopTokens[0] : -1;
            if (stopToken >= 0 && !text.StoppedOnString) stopTokenId = stopToken;
            stop = text.StoppedOnString ? StopReason.StopString
                : stopToken >= 0 ? (stopToken == _eosId ? StopReason.Eos : StopReason.StopToken)
                : StopReason.Length;
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            // as the generator: a cancelled request returns what it produced so far
            stop = StopReason.Cancelled;
        }
        await text.FinishAsync().ConfigureAwait(false);
        return new GenerationResult(text.Text, promptIds.Length, produced, stop) { GeneratedTokenIds = ids.ToArray(), StopTokenId = stopTokenId };
    }

    public void Dispose() => _dec.Dispose();
}
