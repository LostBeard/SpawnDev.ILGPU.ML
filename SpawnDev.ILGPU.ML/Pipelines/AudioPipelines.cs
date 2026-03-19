using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Automatic Speech Recognition: audio bytes → transcribed text.
/// Models: Whisper (tiny, base, small, medium).
/// </summary>
public class SpeechRecognitionPipeline : IPipeline<float[], TranscriptionResult>
{
    private readonly Accelerator _accelerator;
    private InferenceSession? _encoderSession;
    private InferenceSession? _decoderSession;

    public bool IsReady => _encoderSession != null;
    public string ModelName { get; private set; } = "";
    public string BackendName => _accelerator.AcceleratorType.ToString();
    public string? Language { get; set; }

    private SpeechRecognitionPipeline(Accelerator accelerator) => _accelerator = accelerator;

    public static async Task<SpeechRecognitionPipeline> CreateAsync(
        Accelerator accelerator, HttpClient http, string? modelId, PipelineOptions options)
    {
        var pipe = new SpeechRecognitionPipeline(accelerator);
        var path = modelId ?? options.ModelPath ?? "models/whisper-tiny";
        pipe.ModelName = path;
        // Whisper has encoder + decoder
        // TODO: Load both model parts
        return pipe;
    }

    /// <summary>
    /// Transcribe audio samples (float[], mono, any sample rate).
    /// Automatically resamples to 16kHz and computes mel spectrogram.
    /// </summary>
    public async Task<TranscriptionResult> RunAsync(float[] audioSamples)
    {
        if (_encoderSession == null) throw new InvalidOperationException("Model not loaded");
        // Pipeline:
        // 1. Resample to 16kHz if needed
        // 2. Compute log-mel spectrogram (AudioPreprocessor.ComputeLogMelSpectrogram)
        // 3. Run encoder on mel features
        // 4. Autoregressive decoder with forced tokens (language, task, timestamps)
        // 5. Decode token IDs to text
        throw new NotImplementedException("Awaiting encoder-decoder + KV cache support");
    }

    /// <summary>
    /// Transcribe from raw audio file bytes (WAV, MP3, etc.).
    /// Requires AudioContext.DecodeAudioData (SpawnDev.BlazorJS) for browser decoding.
    /// </summary>
    public async Task<TranscriptionResult> TranscribeFileAsync(byte[] audioFileBytes, int sampleRate = 16000)
    {
        // For non-browser usage, decode WAV directly
        var samples = DecodeWavFile(audioFileBytes);
        if (samples == null)
            throw new NotSupportedException("Only WAV format supported for direct file decoding. Use AudioContext.DecodeAudioData for other formats.");

        samples = AudioPreprocessor.Resample(samples, sampleRate, AudioPreprocessor.WhisperSampleRate);
        return await RunAsync(samples);
    }

    private static float[]? DecodeWavFile(byte[] data)
    {
        // Simple WAV parser for PCM format
        if (data.Length < 44) return null;
        if (data[0] != 'R' || data[1] != 'I' || data[2] != 'F' || data[3] != 'F') return null;
        if (data[8] != 'W' || data[9] != 'A' || data[10] != 'V' || data[11] != 'E') return null;

        // Find data chunk
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
                int sampleCount = chunkSize / (bitsPerSample / 8);
                var samples = AudioPreprocessor.PcmBytesToFloat(data[pos..(pos + chunkSize)]);
                if (channels == 2)
                    samples = AudioPreprocessor.StereoToMono(samples);
                return samples;
            }

            pos += chunkSize;
        }

        return null;
    }

    public void Dispose()
    {
        _encoderSession?.Dispose();
        _decoderSession?.Dispose();
    }
}

/// <summary>
/// Audio Classification: audio → (label, score) predictions.
/// Models: Wav2Vec2, HuBERT, AST.
/// </summary>
public class AudioClassificationPipeline : IPipeline<float[], ClassificationResult>
{
    private readonly Accelerator _accelerator;
    private InferenceSession? _session;

    public bool IsReady => _session != null;
    public string ModelName { get; private set; } = "";
    public string BackendName => _accelerator.AcceleratorType.ToString();

    private AudioClassificationPipeline(Accelerator accelerator) => _accelerator = accelerator;

    public static async Task<AudioClassificationPipeline> CreateAsync(
        Accelerator accelerator, HttpClient http, string? modelId, PipelineOptions options)
    {
        var pipe = new AudioClassificationPipeline(accelerator);
        var path = modelId ?? options.ModelPath ?? "models/wav2vec2-ks";
        pipe.ModelName = path;
        pipe._session = await InferenceSession.CreateAsync(accelerator, http, path);
        return pipe;
    }

    public async Task<ClassificationResult> RunAsync(float[] audioSamples)
    {
        if (_session == null) throw new InvalidOperationException("Model not loaded");
        // Pipeline: resample to model rate → normalize → inference → softmax → labels
        throw new NotImplementedException("Awaiting InferenceSession integration");
    }

    public void Dispose() => _session?.Dispose();
}
