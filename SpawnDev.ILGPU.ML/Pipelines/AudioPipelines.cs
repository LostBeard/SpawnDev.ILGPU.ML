using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
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
    private BPETokenizer? _tokenizer;

    // Whisper special tokens, as verified against the model's own tokenizer.json (Xenova/whisper-tiny).
    // TRANSCRIBE and NO_TIMESTAMPS were each one too high: 50360 is <|startoflm|> and 50364 is a timestamp
    // token, so the decoder was primed with a prompt Whisper never emits and returned nothing at all - for
    // clean, speech-level audio. The language block runs 50259..50357, which is what puts <|translate|> at
    // 50358 and <|transcribe|> at 50359; an off-by-one lands inside that block and looks plausible.
    private const int SOT = 50258;           // <|startoftranscript|>
    private const int LANG_EN = 50259;       // <|en|>
    private const int TRANSCRIBE = 50359;    // <|transcribe|>
    private const int NO_TIMESTAMPS = 50363; // <|notimestamps|>
    private const int EOT = 50257;           // <|endoftext|>

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
    public int MaxTokens { get; set; } = 224;
    public string Language { get; set; } = "en";

    public SpeechRecognitionPipeline(
        InferenceSession encoderSession,
        InferenceSession decoderSession,
        Accelerator accelerator)
    {
        _encoderSession = encoderSession;
        _decoderSession = decoderSession;
        _accelerator = accelerator;
    }

    /// <summary>Load tokenizer from HuggingFace tokenizer.json.</summary>
    public void LoadTokenizer(string tokenizerJson)
    {
        _tokenizer = BPETokenizer.LoadFromTokenizerJson(tokenizerJson);
    }

    /// <summary>
    /// Transcribe audio samples to text.
    /// Handles resampling, mel spectrogram, encoder, and autoregressive decoder.
    /// </summary>
    public async Task<TranscriptionResult> TranscribeAsync(
        float[] audioSamples, int sampleRate = 16000)
    {
        var sw = Stopwatch.StartNew();

        // 1. Resample to 16kHz
        if (sampleRate != AudioPreprocessor.WhisperSampleRate)
            audioSamples = AudioPreprocessor.Resample(audioSamples, sampleRate, AudioPreprocessor.WhisperSampleRate);

        // 2. Pad/trim to 30 seconds
        audioSamples = AudioPreprocessor.PadOrTrim(audioSamples, AudioPreprocessor.WhisperSampleRate * 30);

        // 3. Compute log-mel spectrogram [80, 3000]
        var mel = AudioPreprocessor.ComputeLogMelSpectrogram(audioSamples);

        // 4. Run encoder
        using var melBuf = _accelerator.Allocate1D(mel);
        var melTensor = new Tensor(melBuf.View, new[] { 1, 80, 3000 });
        var encoderOutputs = await _encoderSession.RunAsync(new Dictionary<string, Tensor>
        {
            [_encoderSession.InputNames[0]] = melTensor
        });
        var encoderHidden = encoderOutputs[_encoderSession.OutputNames[0]];

        // 5. Autoregressive decoder
        var tokens = new List<int> { SOT, LANG_EN, TRANSCRIBE, NO_TIMESTAMPS };

        // Greedy next-token selection stays GPU-side: read back one index per token, not the whole vocab.
        using var argmax = new GpuArgMax(_accelerator);
        for (int step = 0; step < MaxTokens; step++)
        {
            // Create input_ids tensor
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

            // Last-position logits — shape [1, seq_len, vocab_size].
            int vocabSize = logits.Shape.Length >= 3 ? logits.Shape[^1] : 51865;
            int lastPosOffset = (tokens.Count - 1) * vocabSize;

            // Greedy argmax ON THE GPU — read back ONLY the winning index, not the whole ~52K-float vocab every
            // token (Rule 4: no unnecessary copies). GpuArgMax tie-breaks lowest-index, identical to the old CPU
            // first-max-wins scan; its partial buffers are reused, so there is no per-token allocation.
            int nextToken = await argmax.ArgMaxAsync(logits.Data.SubView(lastPosOffset, vocabSize), vocabSize);
            OnTokenGenerated?.Invoke(step, nextToken);

            if (nextToken == EOT) break;
            tokens.Add(nextToken);
        }

        sw.Stop();

        // 6. Decode tokens to text
        var contentTokens = tokens.Skip(4).ToArray(); // skip SOT, LANG, TRANSCRIBE, NO_TIMESTAMPS
        string text = _tokenizer != null
            ? _tokenizer.Decode(contentTokens)
            : string.Join(" ", contentTokens.Select(t => $"[{t}]"));

        return new TranscriptionResult
        {
            Text = text.Trim(),
            Language = Language,
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds,
        };
    }

    public async Task<TranscriptionResult> RunAsync(float[] audioSamples) =>
        await TranscribeAsync(audioSamples);

    public void Dispose()
    {
        _encoderSession?.Dispose();
        _decoderSession?.Dispose();
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
