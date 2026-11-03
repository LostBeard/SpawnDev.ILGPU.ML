using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>One label's score from <see cref="ClapAudioClassifier.ClassifyAsync(ArrayView1D{float, Stride1D.Dense}, int, IReadOnlyList{string})"/>.</summary>
/// <param name="Label">The label (the text prompt) as given.</param>
/// <param name="Logit">logit_scale x cosine(audio, text): CLAP's logits_per_audio.</param>
/// <param name="Probability">Softmax of the logits over the labels of the call.</param>
public readonly record struct ClapScore(string Label, float Logit, float Probability);

/// <summary>
/// Zero-shot audio classification with CLAP (LAION <c>clap-htsat-unfused</c>: an HTSAT audio tower and a RoBERTa text tower trained
/// contrastively): score a 48 kHz mono clip against any text labels ("a gunshot", "footsteps on gravel", ...) with no training.
/// </summary>
/// <remarks>
/// <para>
/// Pipeline: <see cref="ClapFeatureExtractor"/> (GPU: repeat-pad / centre 10 s, STFT, Slaney mel, dB) -> the audio tower -> an L2-normalised
/// 512-d embedding; each label -> <see cref="BPETokenizer.EncodeRoberta"/> -> the text tower -> an L2-normalised 512-d embedding, cached per
/// label; logits = <see cref="LogitScale"/> x the dot products; probabilities = their softmax. This is transformers' <c>ClapModel</c>
/// <c>logits_per_audio</c>, held to it by <c>Clap_Towers_MatchTransformers</c> (towers cosine &gt;= 0.9999, logits within 0.05).
/// </para>
/// <para>
/// Model: <c>Xenova/clap-htsat-unfused</c> ONNX (from the SpawnDev models host; base checkpoint laion/clap-htsat-unfused, Apache-2.0).
/// Its text_model.onnx has no attention_mask input, so every label runs alone at its own length (padding would attend the pads);
/// the text session recompiles once per new token count, then the embedding is cached.
/// </para>
/// <para>
/// 🔴 Word labels in CLAP's training template, <c>"This is a sound of " + label</c>. With bare labels this checkpoint scored three LibriSpeech /
/// LibriVox speech clips as "music" at 0.90-0.98 (transformers itself, not only this port); with the template, speech came first at 0.91 / 0.68 /
/// 0.41 (MEASURED 2026-11-03).
/// </para>
/// <para>
/// Checkpoints (both Apache-2.0, same feature extractor and tokenizer): <see cref="DefaultRepo"/> (HTSAT-tiny audio tower, 118 MB) and
/// <see cref="LargerGeneralRepo"/> (larger_clap_general, 282 MB audio tower, trained on more general audio). Neither wins everywhere: on a DayZ
/// recording (AssistZ lab run 2026-11-03, 1 s windows of an automatic-rifle burst) the larger one read 17 of 18 windows as "machine gun fire" where
/// the default read 6 as "footsteps", but it scored clean read speech (LibriSpeech / LibriVox) as "a dog barking" or "rain" under every label
/// wording tried, where the default with the template put speech first. Evaluate on your own audio.
/// </para>
/// </remarks>
public sealed class ClapAudioClassifier : IDisposable
{
    /// <summary>The default model repository on the models host: Xenova/clap-htsat-unfused.</summary>
    public const string DefaultRepo = "Xenova/clap-htsat-unfused";
    /// <summary>The larger general-audio checkpoint: Xenova/larger_clap_general (laion/larger_clap_general).</summary>
    public const string LargerGeneralRepo = "Xenova/larger_clap_general";
    /// <summary>
    /// laion/clap-htsat-unfused's <c>logit_scale_a.exp()</c> (transformers 5.20.2). The exported towers do not carry it; ClapModel scales its
    /// audio-to-text logits by it.
    /// </summary>
    public const float DefaultLogitScale = 18.661177f;
    /// <summary>laion/larger_clap_general's <c>logit_scale_a.exp()</c> (transformers 5.20.2).</summary>
    public const float LargerGeneralLogitScale = 38.664719f;

    /// <summary>The logit scale of a known checkpoint repository, or null when it is not one of the checkpoints this class has been gated on.</summary>
    public static float? KnownLogitScale(string repoId) => repoId switch
    {
        DefaultRepo => DefaultLogitScale,
        LargerGeneralRepo => LargerGeneralLogitScale,
        _ => null,
    };
    /// <summary>Embedding width of both towers.</summary>
    public const int EmbedDim = 512;
    /// <summary>The sample rate the audio must be at.</summary>
    public const int SampleRate = ClapFeatureExtractor.SampleRate;

    private readonly Accelerator _accelerator;
    private readonly InferenceSession _audio, _text;
    private readonly BPETokenizer _tokenizer;
    private readonly ClapFeatureExtractor _features;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _featureBuf;
    private readonly Dictionary<string, float[]> _textCache = new();
    private bool _disposed;

    /// <summary>Scale applied to the cosine similarities to form logits (and so the softmax temperature): the checkpoint's own.</summary>
    public float LogitScale { get; set; }

    /// <summary>Wrap already-created sessions and a tokenizer. The classifier owns and disposes them.</summary>
    /// <param name="logitScale">The checkpoint's <c>logit_scale_a.exp()</c> (<see cref="KnownLogitScale"/>); the towers do not carry it.</param>
    public ClapAudioClassifier(Accelerator accelerator, InferenceSession audioTower, InferenceSession textTower, BPETokenizer tokenizer,
        float logitScale = DefaultLogitScale)
    {
        LogitScale = logitScale;
        _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        _audio = audioTower ?? throw new ArgumentNullException(nameof(audioTower));
        _text = textTower ?? throw new ArgumentNullException(nameof(textTower));
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        // run in a loop by design: each run's outputs are given back at the next run instead of piling up
        _audio.OutputsLiveUntilNextRun = true;
        _text.OutputsLiveUntilNextRun = true;
        _features = new ClapFeatureExtractor(accelerator);
        _featureBuf = accelerator.Allocate1D<float>(ClapFeatureExtractor.Frames * ClapFeatureExtractor.Mels);
    }

    /// <summary>
    /// Download (once; the source caches) and load the towers and tokenizer from <paramref name="repoId"/>: <c>onnx/audio_model.onnx</c>
    /// (118 MB default, 282 MB larger), <c>onnx/text_model.onnx</c> (502 MB), <c>tokenizer.json</c>.
    /// </summary>
    /// <param name="source">A model source: <see cref="Hub.HubModelSource"/> in the browser (OPFS cache), <see cref="Hub.HttpClientModelSource"/> on desktop.</param>
    /// <param name="repoId"><see cref="DefaultRepo"/>, <see cref="LargerGeneralRepo"/>, or another CLAP export with the same file layout.</param>
    /// <param name="logitScale">The checkpoint's logit scale; required for a repository <see cref="KnownLogitScale"/> does not know.</param>
    public static async Task<ClapAudioClassifier> CreateFromHubAsync(Accelerator accelerator, Hub.IModelSource source,
        string repoId = DefaultRepo, Action<string, int>? onProgress = null, CancellationToken ct = default, float? logitScale = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        float scale = logitScale ?? KnownLogitScale(repoId)
            ?? throw new ArgumentException($"no known logit scale for '{repoId}': pass logitScale (the checkpoint's logit_scale_a.exp())", nameof(logitScale));
        onProgress?.Invoke("tokenizer", 0);
        var tokenizer = BPETokenizer.LoadFromTokenizerJson(System.Text.Encoding.UTF8.GetString(
            await source.FetchBytesAsync(repoId, "tokenizer.json", ct).ConfigureAwait(false)));
        onProgress?.Invoke("tokenizer", 100);
        InferenceSession? audio = null, text = null;
        try
        {
            audio = await LoadAsync("audio", "onnx/audio_model.onnx").ConfigureAwait(false);
            text = await LoadAsync("text", "onnx/text_model.onnx").ConfigureAwait(false);
            return new ClapAudioClassifier(accelerator, audio, text, tokenizer, scale);
        }
        catch
        {
            audio?.Dispose();
            text?.Dispose();
            throw;
        }

        async Task<InferenceSession> LoadAsync(string stage, string file)
        {
            onProgress?.Invoke(stage, 0);
            await using var stream = await source.OpenAsync(repoId, file, ct).ConfigureAwait(false);
            var session = await InferenceSession.CreateFromStreamAsync(accelerator, stream,
                onProgress: (s, p) => onProgress?.Invoke($"{stage}:{s}", p), ct: ct).ConfigureAwait(false);
            onProgress?.Invoke(stage, 100);
            return session;
        }
    }

    /// <summary>The L2-normalised text embedding of <paramref name="label"/>, computed once and cached.</summary>
    public async Task<float[]> EmbedTextAsync(string label)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(label);
        if (_textCache.TryGetValue(label, out var cached)) return cached;
        var ids = _tokenizer.EncodeRoberta(label);
        var idv = new float[ids.Length];
        for (int i = 0; i < ids.Length; i++) idv[i] = ids[i];
        using var idBuf = _accelerator.Allocate1D(idv);
        var outs = await _text.RunAsync(new Dictionary<string, Tensor> { [_text.InputNames[0]] = new Tensor(idBuf.View, new[] { 1, ids.Length }) })
            .ConfigureAwait(false);
        var e = Normalize(await outs[_text.OutputNames[0]].Data.SubView(0, EmbedDim).CopyToHostAsync().ConfigureAwait(false));
        _textCache[label] = e;
        return e;
    }

    /// <summary>
    /// The L2-normalised audio embedding of a 48 kHz mono clip already on the accelerator. Clips shorter than 10 s are repeat-padded and
    /// longer ones use their centre 10 s (transformers' rand_trunc / repeatpad, deterministic).
    /// </summary>
    public async Task<float[]> EmbedAudioAsync(ArrayView1D<float, Stride1D.Dense> pcm, int length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _features.Extract(pcm, length, _featureBuf.View);
        var shape = new[] { 1, 1, ClapFeatureExtractor.Frames, ClapFeatureExtractor.Mels };
        var outs = await _audio.RunAsync(new Dictionary<string, Tensor> { [_audio.InputNames[0]] = new Tensor(_featureBuf.View, shape) })
            .ConfigureAwait(false);
        return Normalize(await outs[_audio.OutputNames[0]].Data.SubView(0, EmbedDim).CopyToHostAsync().ConfigureAwait(false));
    }

    /// <summary>Score a 48 kHz mono clip on the accelerator against <paramref name="labels"/>; the scores are in label order.</summary>
    public async Task<ClapScore[]> ClassifyAsync(ArrayView1D<float, Stride1D.Dense> pcm, int length, IReadOnlyList<string> labels)
    {
        ArgumentNullException.ThrowIfNull(labels);
        if (labels.Count == 0) throw new ArgumentException("at least one label is required", nameof(labels));
        var audio = await EmbedAudioAsync(pcm, length).ConfigureAwait(false);
        var texts = new float[labels.Count][];
        for (int i = 0; i < labels.Count; i++) texts[i] = await EmbedTextAsync(labels[i]).ConfigureAwait(false);
        return Score(audio, texts, labels, LogitScale);
    }

    /// <summary>Score a 48 kHz mono clip held on the host (uploaded once) against <paramref name="labels"/>.</summary>
    public async Task<ClapScore[]> ClassifyAsync(float[] pcm, IReadOnlyList<string> labels)
    {
        ArgumentNullException.ThrowIfNull(pcm);
        // disposed only after ClassifyAsync's embedding readback has completed, which fences the feature kernel that reads it
        using var buf = _accelerator.Allocate1D(pcm);
        return await ClassifyAsync(buf.View, pcm.Length, labels).ConfigureAwait(false);
    }

    /// <summary>logits = scale x audio . text (both L2-normalised), probabilities = softmax(logits); in label order.</summary>
    public static ClapScore[] Score(float[] audioEmbed, IReadOnlyList<float[]> textEmbeds, IReadOnlyList<string> labels, float logitScale)
    {
        var logits = new double[textEmbeds.Count];
        double max = double.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            double d = 0;
            var t = textEmbeds[i];
            for (int k = 0; k < audioEmbed.Length; k++) d += (double)audioEmbed[k] * t[k];
            logits[i] = logitScale * d;
            max = Math.Max(max, logits[i]);
        }
        double sum = 0;
        for (int i = 0; i < logits.Length; i++) sum += Math.Exp(logits[i] - max);
        var scores = new ClapScore[logits.Length];
        for (int i = 0; i < logits.Length; i++)
            scores[i] = new ClapScore(labels[i], (float)logits[i], (float)(Math.Exp(logits[i] - max) / sum));
        return scores;
    }

    private static float[] Normalize(float[] v)
    {
        double n = 0;
        foreach (var x in v) n += (double)x * x;
        n = Math.Sqrt(n);
        if (n == 0) return v;
        for (int i = 0; i < v.Length; i++) v[i] = (float)(v[i] / n);
        return v;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _audio.Dispose();
        _text.Dispose();
        _features.Dispose();
        _featureBuf.Dispose();
    }
}
