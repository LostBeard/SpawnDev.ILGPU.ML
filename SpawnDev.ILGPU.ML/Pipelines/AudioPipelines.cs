using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.SpawnJS.JSObjects;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Automatic Speech Recognition using Whisper.
/// Encoder-decoder architecture with autoregressive token generation.
///
/// Usage:
///   var pipeline = new SpeechRecognitionPipeline(encoderSession, decoderSession, accelerator);
///   pipeline.LoadTokenizer(tokenizerJson);
///   var result = await pipeline.TranscribeAsync(audioSamples, sampleRate: 44100);
///   Console.WriteLine(result.Text);
/// </summary>
public class SpeechRecognitionPipeline : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly InferenceSession _encoderSession;
    private readonly InferenceSession _decoderSession;
    /// <summary>Optional `decoder_with_past_model.onnx` session; null = the O(n^2) full-recompute path.</summary>
    private readonly InferenceSession? _decoderWithPastSession;

    /// <summary>
    /// Capture-once/replay-many for the ENCODER, which is the ideal candidate in this pipeline.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠️ Whisper's encoder input is <c>[1, 80, 3000]</c> - a FIXED shape, because the audio is padded to
    /// 30 s before the mel. Unlike ZipVoice's decoder, whose shape is the utterance length and whose
    /// capture must therefore be rebuilt per utterance, one capture here serves every transcription for the
    /// life of the pipeline. It also runs exactly once per transcription, so nothing else in the graph gets
    /// to amortise its per-node host cost.
    /// </para>
    /// <para>
    /// ⚠️ The cost this targets is DISPATCH, not compute. MEASURED in the SpawnDev.AI demo, 2026-09-03,
    /// transcribing a 4.0 s utterance in 13,926 ms: 12 graph runs, executor 9,465 ms, of which
    /// <b>residual 7,726 ms was dispatch + CPU + alloc</b> against just 531 ms of readbacks. A recorded
    /// plan replaces per-node dispatch encoding with one crossing, which is exactly that term.
    /// </para>
    /// <para>
    /// The decoder deliberately gets no capture: <c>decoder_with_past</c>'s past-K/V grow by one position
    /// every step, so its shapes change on every single call and a recording would be invalid immediately.
    /// </para>
    /// <para>
    /// Capture is best-effort - ineligible backends and failed captures fall through to the direct forward,
    /// and <see cref="Graph.SessionGraphCapture.CaptureStatus"/> says which.
    /// </para>
    /// </remarks>
    private Graph.SessionGraphCapture? _encoderCapture;

    /// <summary>GPU log-mel (pad → centred STFT → Slaney mel → Whisper normalize). Lazy; session-owned.</summary>
    private WhisperMelPreprocessor? _melPreprocess;

    /// <summary>Enable capture/replay of the encoder. Off runs a plain forward.</summary>
    public bool EnableGraphCapture { get; set; } = true;

    /// <summary>WHY the encoder capture is or is not live.</summary>
    public string EncoderCaptureStatus => _encoderCapture?.CaptureStatus ?? "no capture constructed yet";

    /// <summary>Whether the encoder is actually replaying a recorded plan.</summary>
    public bool EncoderCaptured => _encoderCapture?.IsCaptured ?? false;
    private BPETokenizer? _tokenizer;

    // Whisper special tokens, as verified against the model's own tokenizer.json (Xenova/whisper-tiny).
    // TRANSCRIBE and NO_TIMESTAMPS were each one too high: 50360 is <|startoflm|> and 50364 is a timestamp
    // token, so the decoder was primed with a prompt Whisper never emits and returned nothing at all - for
    // clean, speech-level audio. The language block runs 50259..50357, which is what puts <|translate|> at
    // 50358 and <|transcribe|> at 50359; an off-by-one lands inside that block and looks plausible.
    // These are the MULTILINGUAL ids, used as defaults and then replaced by whatever the loaded
    // tokenizer actually says - see LoadTokenizer. Hard-coding them meant this pipeline could only ever
    // run multilingual Whisper: the English-only (.en) checkpoints have a BPE vocabulary one token
    // smaller, so every special id shifts down by one, and priming an .en decoder with the multilingual
    // set produced an EMPTY transcript with no error - indistinguishable from silent audio.
    private int SOT = 50258;           // <|startoftranscript|>
    private int LANG_EN = 50259;       // <|en|>
    private int TRANSCRIBE = 50359;    // <|transcribe|>
    private int NO_TIMESTAMPS = 50363; // <|notimestamps|>
    private int EOT = 50257;           // <|endoftext|>

    /// <summary>True when the loaded model is an English-only (.en) Whisper checkpoint.</summary>
    /// <remarks>
    /// English-only models were trained WITHOUT the language and task tokens, so their prompt is
    /// [startoftranscript, notimestamps] rather than the four-token multilingual prompt. Detected from
    /// the tokenizer rather than configured, because getting it wrong fails silently.
    /// </remarks>
    public bool IsEnglishOnlyModel { get; private set; }

    /// <summary>
    /// Raised for each greedy step with (stepIndex, tokenId) as decoding proceeds.
    /// </summary>
    /// <remarks>
    /// Decoding is the one stage whose failure is invisible from the outside: a wrong encoder output or a
    /// wrong prompt does not throw, it just makes the model emit end-of-text immediately and return an
    /// empty string, which is indistinguishable from silent audio. Watching the token stream separates
    /// "stopped instantly" from "looped on one token" from "produced words" without a debugger.
    /// </remarks>
    public event Action<int, int>? OnTokenGenerated;

    public bool IsReady => true;
    public string ModelName { get; init; } = "Whisper Tiny";

    /// <summary>
    /// Maximum tokens to GENERATE. Defaults to the model's real ceiling: <see cref="MaxTargetPositions"/>
    /// minus the 4-token prompt.
    /// </summary>
    /// <remarks>
    /// This used to default to 224, which silently truncated long transcripts - 30s of dense speech can run
    /// past that, and the loop simply stopped mid-sentence with no error. It is a generation bound, not a
    /// safety valve: the hard limit is <see cref="MaxTargetPositions"/> and it is enforced separately, so
    /// raising this cannot walk off the end of the positional embeddings.
    /// </remarks>
    public int MaxTokens { get; set; } = 444;

    /// <summary>
    /// The decoder's learned positional-embedding count (`max_target_positions`, 448 for every Whisper
    /// size). Generation stops once prompt + generated reaches it.
    /// </summary>
    /// <remarks>
    /// This is a CORRECTNESS limit, not a preference: position N is a lookup into an embedding table with
    /// exactly this many rows, so going past it reads off the end. Enforced in both decode paths.
    /// </remarks>
    public int MaxTargetPositions { get; set; } = 448;

    /// <summary>
    /// The language Whisper is prompted to transcribe, as a short code ("en", "fr", "de", ...).
    /// </summary>
    /// <remarks>
    /// This DRIVES the prompt's language token. It used to be decoration: the prompt hard-coded
    /// <c>&lt;|en|&gt;</c> and this property was only echoed back into
    /// <see cref="TranscriptionResult.Language"/>, so setting "fr" produced an English-prompted
    /// transcript that CLAIMED to be French. Resolution now fails loudly rather than silently falling
    /// back to English - a wrong prompt does not throw inside Whisper, it just returns confident text,
    /// which is the one failure mode this pipeline has no way to notice.
    /// </remarks>
    public string Language { get; set; } = "en";

    /// <summary>
    /// The token id for <see cref="Language"/>, resolved from the loaded tokenizer.
    /// </summary>
    /// <remarks>
    /// Accepts "en" or "&lt;|en|&gt;" and is case-insensitive. Throws rather than defaulting, because
    /// every silent failure in this file has cost a day: an unrecognised language that quietly became
    /// English would be indistinguishable from a correct run.
    /// </remarks>
    private int ResolveLanguageToken()
    {
        var lang = (Language ?? string.Empty).Trim().Trim('<', '>', '|').ToLowerInvariant();
        if (lang.Length == 0) lang = "en";
        if (lang == "en") return LANG_EN;
        if (_tokenizer == null)
            throw new InvalidOperationException(
                $"Language '{Language}' needs a tokenizer to resolve <|{lang}|>; call LoadTokenizer first. " +
                "Only \"en\" has a compiled-in default.");
        if (!_tokenizer.TryGetTokenId($"<|{lang}|>", out var id))
            throw new InvalidOperationException(
                $"The loaded tokenizer has no token <|{lang}|>, so language '{Language}' cannot be " +
                "requested of this model. Multilingual Whisper checkpoints list one token per language; " +
                "the English-only (.en) checkpoints have none.");
        return id;
    }

    /// <summary>True when a with-past decoder was supplied, so decoding is O(n) rather than O(n^2).</summary>
    public bool UsesKVCache => _decoderWithPastSession != null;

    /// <param name="decoderWithPastSession">
    /// Optional `decoder_with_past_model.onnx`. Without it every step re-feeds the WHOLE token sequence and
    /// recomputes every previous position's K/V - quadratic, and it also allocates full-sequence logits
    /// ([1, seq, 51865] ~ 13 MB) every step. With it, step 0 prefills on the plain decoder and every later
    /// step feeds ONE token plus the previous step's `present.*` tensors straight back as
    /// `past_key_values.*`, GPU-resident throughout (Rule 4: no readback, no host copy).
    /// </param>
    public SpeechRecognitionPipeline(
        InferenceSession encoderSession,
        InferenceSession decoderSession,
        Accelerator accelerator,
        InferenceSession? decoderWithPastSession = null)
    {
        _encoderSession = encoderSession;
        _decoderSession = decoderSession;
        _decoderWithPastSession = decoderWithPastSession;
        _accelerator = accelerator;
    }

    /// <summary>Load tokenizer from HuggingFace tokenizer.json.</summary>
    /// <remarks>
    /// Also resolves the decoder's special tokens FROM THAT TOKENIZER. The ids differ between the
    /// multilingual and English-only checkpoints, and a wrong prompt does not throw - the model simply
    /// emits end-of-text immediately and returns "". Reading them from the model's own tokenizer is the
    /// only way to be right for both without asking the caller to know which one they have.
    /// </remarks>
    public void LoadTokenizer(string tokenizerJson)
    {
        _tokenizer = BPETokenizer.LoadFromTokenizerJson(tokenizerJson);

        if (_tokenizer.TryGetTokenId("<|startoftranscript|>", out var sot)) SOT = sot;
        if (_tokenizer.TryGetTokenId("<|endoftext|>", out var eot)) EOT = eot;
        if (_tokenizer.TryGetTokenId("<|transcribe|>", out var transcribe)) TRANSCRIBE = transcribe;
        if (_tokenizer.TryGetTokenId("<|notimestamps|>", out var noTs)) NO_TIMESTAMPS = noTs;
        if (_tokenizer.TryGetTokenId("<|en|>", out var langEn)) LANG_EN = langEn;

        // The English-only checkpoints drop one entry from the byte-level BPE vocabulary, which lands
        // their end-of-text at 50256 instead of 50257. That one-token difference is the whole signal:
        // both tokenizers list the language tokens, but only the multilingual MODEL was trained to be
        // prompted with them.
        IsEnglishOnlyModel = EOT == 50256;
    }

    /// <summary>
    /// Transcribe audio samples to text.
    /// Handles resampling, GPU mel spectrogram, encoder, and autoregressive decoder.
    /// </summary>
    /// <remarks>
    /// ⚠️ IN A BROWSER, prefer <see cref="TranscribeAsync(Float32Array, int)"/> when samples are still
    /// a JS typed array — that path uploads via <see cref="MediaInterop.UploadToDevice{T}"/> with no
    /// managed-heap crossing for the waveform (at 16 kHz). Non-16 kHz still needs a host resample until
    /// GPU windowed-sinc lands.
    /// </remarks>
    public async Task<TranscriptionResult> TranscribeAsync(
        float[] audioSamples, int sampleRate = 16000)
    {
        ArgumentNullException.ThrowIfNull(audioSamples);
        if (sampleRate != AudioPreprocessor.WhisperSampleRate)
            audioSamples = AudioPreprocessor.Resample(audioSamples, sampleRate, AudioPreprocessor.WhisperSampleRate);

        using var pcmBuf = _accelerator.Allocate1D(audioSamples.Length == 0 ? new float[1] : audioSamples);
        return await TranscribeCoreFromPcmAsync(pcmBuf.View, audioSamples.Length).ConfigureAwait(false);
    }

    /// <summary>
    /// Browser path: accept PCM as a JS <see cref="Float32Array"/> and upload straight to the device.
    /// </summary>
    /// <remarks>
    /// At 16 kHz there is no managed-heap crossing for the waveform: <see cref="MediaInterop.UploadToDevice{T}"/>
    /// then <see cref="WhisperMelPreprocessor"/> keeps mel on-device into the encoder. Non-16 kHz still
    /// <c>ToArray</c>s once for the CPU windowed-sinc resampler (aliasing-safe; linear GPU resample is not
    /// a drop-in). Prefer this overload over converting in the caller.
    /// </remarks>
    public async Task<TranscriptionResult> TranscribeAsync(
        Float32Array audioSamples, int sampleRate = 16000)
    {
        ArgumentNullException.ThrowIfNull(audioSamples);
        if (sampleRate != AudioPreprocessor.WhisperSampleRate)
            return await TranscribeAsync(audioSamples.ToArray(), sampleRate).ConfigureAwait(false);

        int n = (int)audioSamples.Length;
        var pcmBuf = _accelerator.Allocate1D<float>(Math.Max(n, 1));
        try
        {
            if (n > 0)
                MediaInterop.UploadToDevice(audioSamples, pcmBuf);
            return await TranscribeCoreFromPcmAsync(pcmBuf.View, n).ConfigureAwait(false);
        }
        finally
        {
            // Mel dispatches may still reference pcm on WebGPU/Wasm until drained.
            await _accelerator.SynchronizeAsync().ConfigureAwait(false);
            pcmBuf.Dispose();
        }
    }

    /// <summary>
    /// Core path: 16 kHz PCM already on-device → GPU mel → encoder → decoder.
    /// </summary>
    private async Task<TranscriptionResult> TranscribeCoreFromPcmAsync(
        ArrayView1D<float, Stride1D.Dense> pcm,
        int pcmLength)
    {
        var sw = Stopwatch.StartNew();

        // GPU log-mel [80, 3000] — pad/trim + centred STFT + Slaney mel + Whisper normalize on-device.
        // Timed separately: fixed cost over the padded 30 s window (endpointing cannot shrink it).
        var melSw = Stopwatch.StartNew();
        _melPreprocess ??= new WhisperMelPreprocessor(_accelerator);
        using var melBuf = _accelerator.Allocate1D<float>(WhisperMelPreprocessor.MelLength);
        _melPreprocess.ComputeLogMelSpectrogram(pcm, pcmLength, melBuf.View);
        // Drain before the encoder capture may reorder work; also lets Float32Array finally dispose pcm.
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        melSw.Stop();
        var melMs = melSw.Elapsed.TotalMilliseconds;

        // Run encoder (capturable — fixed [1,80,3000] shape).
        var encSw = Stopwatch.StartNew();
        var melTensor = new Tensor(melBuf.View, new[] { 1, 80, 3000 });
        _encoderCapture ??= new Graph.SessionGraphCapture(_encoderSession, _accelerator);
        _encoderCapture.Enabled = EnableGraphCapture;
        var encoderOutputs = await _encoderCapture.RunAsync(new Dictionary<string, Tensor>
        {
            [_encoderSession.InputNames[0]] = melTensor
        });
        var encoderHidden = encoderOutputs[_encoderSession.OutputNames[0]];
        encSw.Stop();
        double encoderMs = encSw.Elapsed.TotalMilliseconds;

        // Autoregressive decoder (unchanged from here).
        var requestedLanguage = (Language ?? string.Empty).Trim().Trim('<', '>', '|').ToLowerInvariant();
        if (requestedLanguage.Length == 0) requestedLanguage = "en";
        if (IsEnglishOnlyModel && requestedLanguage != "en")
            throw new InvalidOperationException(
                $"Language '{Language}' was requested, but the loaded model is an English-only (.en) " +
                "Whisper checkpoint, which was trained without language tokens. Load a multilingual " +
                "checkpoint or set Language = \"en\".");

        var tokens = IsEnglishOnlyModel
            ? new List<int> { SOT, NO_TIMESTAMPS }
            : new List<int> { SOT, ResolveLanguageToken(), TRANSCRIBE, NO_TIMESTAMPS };
        int promptLength = tokens.Count;

        using var argmax = new GpuArgMax(_accelerator);

        double prefillMs = 0, stepsMs = 0, stepSetupMs = 0, stepRunMs = 0, stepArgmaxMs = 0;
        int decodeSteps = 0;
        if (_decoderWithPastSession != null)
        {
            (prefillMs, stepsMs, decodeSteps, stepSetupMs, stepRunMs, stepArgmaxMs) = await DecodeWithKVCacheAsync(tokens, encoderHidden, argmax);
        }
        else
        for (int step = 0; step < MaxTokens; step++)
        {
            var inputIds = tokens.Select(t => (float)t).ToArray();
            using var idsBuf = _accelerator.Allocate1D(inputIds);
            var idsTensor = new Tensor(idsBuf.View, new[] { 1, tokens.Count });

            var decoderInputs = new Dictionary<string, Tensor>
            {
                [_decoderSession.InputNames[0]] = idsTensor,
                [_decoderSession.InputNames[1]] = encoderHidden,
            };

            var decoderOutputs = await _decoderSession.RunAsync(decoderInputs);
            var logits = decoderOutputs[_decoderSession.OutputNames[0]];

            int vocabSize = logits.Shape.Length >= 3 ? logits.Shape[^1] : 51865;
            int lastPosOffset = (tokens.Count - 1) * vocabSize;

            int nextToken = await argmax.ArgMaxAsync(logits.Data.SubView(lastPosOffset, vocabSize), vocabSize);
            OnTokenGenerated?.Invoke(step, nextToken);

            if (nextToken == EOT) break;
            tokens.Add(nextToken);
            if (tokens.Count >= MaxTargetPositions) break;
        }


        sw.Stop();

        // 6. Decode tokens to text
        // Skip the prompt - whose LENGTH now varies by model family - and drop any special token that
        // slipped through, so a stray timestamp or task token cannot appear as literal text.
        var contentTokens = tokens.Skip(promptLength).Where(id => id < EOT).ToArray();
        string text = _tokenizer != null
            ? _tokenizer.Decode(contentTokens)
            : string.Join(" ", contentTokens.Select(t => $"[{t}]"));

        return new TranscriptionResult
        {
            Text = text.Trim(),
            // The language the decoder was actually PROMPTED with, normalized - not the raw property.
            // An .en checkpoint is prompted with no language token at all, which is English by
            // construction.
            Language = IsEnglishOnlyModel ? "en" : requestedLanguage,
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds,
            MelTimeMs = melMs,
            ModelTimeMs = sw.Elapsed.TotalMilliseconds - melMs,
            EncoderCaptureStatus = EncoderCaptureStatus,
            EncoderMs = encoderMs,
            PrefillMs = prefillMs,
            DecodeStepsMs = stepsMs,
            DecodeSteps = decodeSteps,
            DecodeSetupMs = stepSetupMs,
            DecodeGraphMs = stepRunMs,
            DecodeArgmaxMs = stepArgmaxMs,
            // From the SESSION, not from a model of it - see the remarks on these properties.
            EncoderNodeCount = _encoderSession.NodeCount,
            DecoderNodeCount = (_decoderWithPastSession ?? _decoderSession).NodeCount,
            DecoderOpHistogram = (_decoderWithPastSession ?? _decoderSession).CompiledOpHistogram,
        };
    }

    /// <summary>
    /// O(n) greedy decode: prefill the prompt once, then feed ONE token per step with the previous step's
    /// K/V handed straight back as `past_key_values.*`.
    /// </summary>
    /// <remarks>
    /// Whisper's exported decoder splits into two graphs. `decoder_model.onnx` takes the whole prompt plus
    /// `encoder_hidden_states` and emits `present.{L}.decoder.*` AND `present.{L}.encoder.*`.
    /// `decoder_with_past_model.onnx` takes ONE token plus `past_key_values.{L}.{decoder,encoder}.*` and
    /// emits only the DECODER `present.*` - the cross-attention K/V are a function of the encoder output
    /// alone, so they are computed once during prefill and then passed through unchanged forever. It takes
    /// no `encoder_hidden_states` at all, which is exactly why the encoder entries have to be carried
    /// across every step rather than recomputed.
    ///
    /// Every tensor here stays on the GPU: a step's outputs become the next step's inputs by reference, so
    /// there is no readback and no host copy (Rule 4). Only the winning token index crosses the boundary,
    /// via GpuArgMax, the same as the non-cached path.
    ///
    /// This is also a large MEMORY win, not just a speed one: the quadratic path materialises full-sequence
    /// logits ([1, seq, 51865] ~ 13 MB) on every single step, where this materialises one position (~0.2 MB).
    /// </remarks>
    private async Task<(double PrefillMs, double StepsMs, int Steps, double SetupMs, double RunMs, double ArgmaxMs)> DecodeWithKVCacheAsync(
        List<int> tokens, Tensor encoderHidden, GpuArgMax argmax)
    {
        var withPast = _decoderWithPastSession!;

        // ── Prefill: the whole prompt, once, on the plain decoder. ──
        var prefillSw = Stopwatch.StartNew();
        var promptIds = tokens.Select(t => (float)t).ToArray();
        using var promptBuf = _accelerator.Allocate1D(promptIds);
        var prefill = await _decoderSession.RunAsync(new Dictionary<string, Tensor>
        {
            [_decoderSession.InputNames[0]] = new Tensor(promptBuf.View, new[] { 1, tokens.Count }),
            [_decoderSession.InputNames[1]] = encoderHidden,
        });

        var logits = prefill[_decoderSession.OutputNames[0]];
        int vocabSize = logits.Shape.Length >= 3 ? logits.Shape[^1] : 51865;
        int nextToken = await argmax.ArgMaxAsync(
            logits.Data.SubView((tokens.Count - 1) * vocabSize, vocabSize), vocabSize);
        OnTokenGenerated?.Invoke(0, nextToken);
        prefillSw.Stop();
        double prefillMs = prefillSw.Elapsed.TotalMilliseconds;
        if (nextToken == EOT) return (prefillMs, 0, 0, 0, 0, 0);
        tokens.Add(nextToken);

        // present.X -> past_key_values.X, for both the decoder and the encoder families. Built from the
        // graph's own output names so a differently-sized model (more layers) needs no changes here.
        var past = new Dictionary<string, Tensor>();
        foreach (var name in _decoderSession.OutputNames)
            if (name.StartsWith("present.", StringComparison.Ordinal))
                past["past_key_values." + name.Substring("present.".Length)] = prefill[name];

        // ── Steps 1..n: one token at a time. ──
        var stepsSw = Stopwatch.StartNew();
        int stepsRun = 0;
        double setupMs = 0, runMs = 0, argmaxMs = 0;
        for (int step = 1; step < MaxTokens; step++)
        {
            stepsRun++;
            // ⚠️ Timed in three parts. A decode step measured at 1,131 ms (whisper-TINY, one token) and the
            // executor's cumulative counters can only say what all eleven graph runs did together - they
            // cannot separate the per-token host setup from the graph itself from the argmax round trip,
            // and those are three different fixes. Cutting before knowing which one is most of it is how a
            // day gets spent on the wrong one.
            var setupSw = Stopwatch.StartNew();
            var stepIds = new[] { (float)nextToken };
            using var idsBuf = _accelerator.Allocate1D(stepIds);
            var inputs = new Dictionary<string, Tensor>();
            foreach (var name in withPast.InputNames)
            {
                if (past.TryGetValue(name, out var t)) inputs[name] = t;
                else inputs[name] = new Tensor(idsBuf.View, new[] { 1, 1 });   // input_ids
            }
            setupSw.Stop();
            setupMs += setupSw.Elapsed.TotalMilliseconds;

            var runSw = Stopwatch.StartNew();
            var outputs = await withPast.RunAsync(inputs);
            runSw.Stop();
            runMs += runSw.Elapsed.TotalMilliseconds;
            var stepLogits = outputs[withPast.OutputNames[0]];

            // One position in, one position out - the winning logits start at offset 0.
            var amSw = Stopwatch.StartNew();
            nextToken = await argmax.ArgMaxAsync(stepLogits.Data.SubView(0, vocabSize), vocabSize);
            amSw.Stop();
            argmaxMs += amSw.Elapsed.TotalMilliseconds;
            OnTokenGenerated?.Invoke(step, nextToken);
            if (nextToken == EOT) break;
            tokens.Add(nextToken);
            // Hard stop at the positional-embedding count - past it the decoder indexes off the end.
            if (tokens.Count >= MaxTargetPositions) break;

            // Roll the DECODER entries forward. The encoder entries are deliberately left untouched: this
            // graph does not re-emit them, and they never change.
            foreach (var name in withPast.OutputNames)
                if (name.StartsWith("present.", StringComparison.Ordinal))
                    past["past_key_values." + name.Substring("present.".Length)] = outputs[name];
        }
        stepsSw.Stop();
        return (prefillMs, stepsSw.Elapsed.TotalMilliseconds, stepsRun, setupMs, runMs, argmaxMs);
    }

    public async Task<TranscriptionResult> RunAsync(float[] audioSamples) =>
        await TranscribeAsync(audioSamples);

    public void Dispose()
    {
        _encoderCapture?.Dispose();
        _melPreprocess?.Dispose();
        _melPreprocess = null;
        _encoderSession?.Dispose();
        _decoderSession?.Dispose();
        _decoderWithPastSession?.Dispose();
    }
}

/// <summary>
/// Simple WAV file decoder for non-browser usage.
/// </summary>
public static class WavDecoder
{
    public static float[]? DecodeWavFile(byte[] data)
    {
        if (data.Length < 44) return null;
        if (data[0] != 'R' || data[1] != 'I' || data[2] != 'F' || data[3] != 'F') return null;
        if (data[8] != 'W' || data[9] != 'A' || data[10] != 'V' || data[11] != 'E') return null;

        int pos = 12;
        int channels = 1;
        int bitsPerSample = 16;

        while (pos < data.Length - 8)
        {
            string chunkId = System.Text.Encoding.ASCII.GetString(data, pos, 4);
            int chunkSize = BitConverter.ToInt32(data, pos + 4);
            pos += 8;

            if (chunkId == "fmt ")
            {
                channels = BitConverter.ToInt16(data, pos + 2);
                bitsPerSample = BitConverter.ToInt16(data, pos + 14);
            }
            else if (chunkId == "data")
            {
                var samples = AudioPreprocessor.PcmBytesToFloat(data[pos..(pos + chunkSize)]);
                if (channels == 2)
                    samples = AudioPreprocessor.StereoToMono(samples);
                return samples;
            }

            pos += chunkSize;
        }

        return null;
    }
}
