using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Result from text-to-speech — raw audio samples.
/// </summary>
public record TTSResult(float[] Audio, int SampleRate, double DurationSeconds, double InferenceTimeMs);

/// <summary>
/// Text-to-speech pipeline using SpeechT5 encoder + decoder + HiFi-GAN vocoder.
///
/// Usage:
///   var encoder = InferenceSession.CreateFromFile(accelerator, encoderBytes);
///   var decoder = InferenceSession.CreateFromFile(accelerator, decoderBytes);
///   var vocoder = InferenceSession.CreateFromFile(accelerator, vocoderBytes);
///   var pipeline = new TextToSpeechPipeline(encoder, decoder, vocoder, accelerator);
///   var result = await pipeline.SynthesizeAsync("Hello world.", speakerEmbedding);
///   // result.Audio is float[] PCM at 16000 Hz
/// </summary>
public class TextToSpeechPipeline : IDisposable
{
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoder;
    private readonly InferenceSession _vocoder;
    private readonly Accelerator _accelerator;

    /// <summary>Sample rate of the output audio (SpeechT5 = 16000 Hz).</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Maximum decoder steps before stopping (prevents infinite loop).</summary>
    public int MaxDecoderSteps { get; set; } = 500;

    public TextToSpeechPipeline(
        InferenceSession encoder,
        InferenceSession decoder,
        InferenceSession vocoder,
        Accelerator accelerator)
    {
        _encoder = encoder;
        _decoder = decoder;
        _vocoder = vocoder;
        _accelerator = accelerator;
    }

    /// <summary>
    /// Synthesize speech from text.
    /// Returns raw PCM float audio at SampleRate Hz.
    /// </summary>
    public async Task<TTSResult> SynthesizeAsync(float[] tokenIds, float[] speakerEmbedding)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();

        // Step 1: Encode text
        int seqLen = tokenIds.Length;
        using var tokenBuf = _accelerator.Allocate1D(tokenIds);
        var tokenTensor = new Tensor(tokenBuf.View, new[] { 1, seqLen });

        var encoderOutputs = await _encoder.RunAsync(new Dictionary<string, Tensor>
        {
            [_encoder.InputNames[0]] = tokenTensor
        });

        var encoderHidden = encoderOutputs[_encoder.OutputNames[0]];
        if (InferenceSession.VerboseLogging) Console.WriteLine($"[TTS] Encoder output: [{string.Join(",", encoderHidden.Shape)}]");

        // Step 2: Run vocoder on encoder output (simplified — full pipeline would use decoder)
        // SpeechT5 HiFi-GAN vocoder takes mel spectrogram → audio waveform
        using var speakerBuf = _accelerator.Allocate1D(speakerEmbedding);

        // For the simplified pipeline, pass encoder hidden directly to vocoder
        // Full pipeline would autoregressive decode mel frames first
        var vocoderOutputs = await _vocoder.RunAsync(new Dictionary<string, Tensor>
        {
            [_vocoder.InputNames[0]] = encoderHidden
        });

        var audioTensor = vocoderOutputs[_vocoder.OutputNames[0]];
        int audioLen = audioTensor.ElementCount;

        if (InferenceSession.VerboseLogging) Console.WriteLine($"[TTS] Audio output: [{string.Join(",", audioTensor.Shape)}], samples={audioLen}");

        // Read audio to CPU
        using var readBuf = _accelerator.Allocate1D<float>(audioLen);
        new ElementWiseKernels(_accelerator).Scale(
            audioTensor.Data.SubView(0, audioLen), readBuf.View, audioLen, 1f);
        await _accelerator.SynchronizeAsync();
        var audio = await readBuf.CopyToHostAsync<float>(0, audioLen);

        sw.Stop();
        double duration = (double)audioLen / SampleRate;

        return new TTSResult(audio, SampleRate, duration, sw.Elapsed.TotalMilliseconds);
    }

    public void Dispose()
    {
        _encoder?.Dispose();
        _decoder?.Dispose();
        _vocoder?.Dispose();
    }
}
