using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Has the speaker finished their turn? Smart Turn v3.2 (pipecat-ai, BSD-2): a Whisper-tiny encoder with a classifier head, run on
/// the LAST 8 seconds of audio, returning the probability that what was said is a complete turn. A voice agent that waits a fixed
/// half second of silence cuts people off mid-thought and answers slowly to everyone else; this lets it answer sooner when the sentence is
/// finished and wait when it is not.
/// </summary>
/// <remarks>
/// <para>
/// Input handling matches the model authors' <c>inference.py</c>: audio longer than 8 s keeps its last 8 s, shorter audio is zero-padded at
/// the START, the waveform is normalised to zero mean / unit variance, and the 8 s log-mel (<c>[80, 800]</c>) is computed on the device
/// (<see cref="WhisperMelPreprocessor"/>). The graph's single output is already a sigmoid probability although the export names it
/// "logits". The model was trained on spoken English plus 22 other languages; it judges the audio alone, not the words.
/// </para>
/// <para>
/// The decision threshold (the authors use 0.5) is the caller's; this class returns the probability. Bulk audio never reaches the host: only the
/// single float result is read back.
/// </para>
/// </remarks>
public sealed class SmartTurnPipeline : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly InferenceSession _session;
    private readonly WhisperMelPreprocessor _mel;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _features;

    /// <summary>Seconds of audio the model sees.</summary>
    public const int WindowSeconds = 8;

    /// <summary>Samples in the model's window at 16 kHz.</summary>
    public const int WindowSamples = WindowSeconds * 16000;

    /// <summary>Mel frames the model expects.</summary>
    public const int Frames = 800;

    /// <param name="session">The Smart Turn ONNX session created with input shape <c>input_features [1, 80, 800]</c>.</param>
    public SmartTurnPipeline(InferenceSession session, Accelerator accelerator)
    {
        _session = session;
        _accelerator = accelerator;
        _mel = new WhisperMelPreprocessor(accelerator, WindowSamples, alignToEnd: true, normalizeWaveform: true);
        if (_mel.Frames != Frames) throw new InvalidOperationException($"mel has {_mel.Frames} frames, the model takes {Frames}");
        _features = accelerator.Allocate1D<float>(_mel.MelElements);
        _session.OutputsLiveUntilNextRun = true;
    }

    /// <summary>The input shape to create the session with (the export's batch axis is dynamic).</summary>
    public static IReadOnlyDictionary<string, int[]> InputShapes { get; } = new Dictionary<string, int[]> { ["input_features"] = new[] { 1, WhisperMelPreprocessor.NMels, Frames } };

    /// <summary>P(the turn is complete) for 16 kHz mono PCM, the end of which is "now" (any other rate is resampled on the host first).</summary>
    public async Task<float> PredictAsync(float[] pcm, int sampleRate = 16000)
    {
        ArgumentNullException.ThrowIfNull(pcm);
        if (sampleRate != AudioPreprocessor.WhisperSampleRate)
            pcm = AudioPreprocessor.Resample(pcm, sampleRate, AudioPreprocessor.WhisperSampleRate);
        using var buf = _accelerator.Allocate1D(pcm.Length == 0 ? new float[1] : pcm);
        return await PredictAsync(buf.View, pcm.Length).ConfigureAwait(false);
    }

    /// <summary>Browser path: the PCM is a JS typed array and goes straight to the device.</summary>
    public async Task<float> PredictAsync(Float32Array pcm, int sampleRate = 16000)
    {
        ArgumentNullException.ThrowIfNull(pcm);
        if (sampleRate != AudioPreprocessor.WhisperSampleRate)
            return await PredictAsync(pcm.ToArray(), sampleRate).ConfigureAwait(false);
        int n = (int)pcm.Length;
        var buf = _accelerator.Allocate1D<float>(Math.Max(n, 1));
        try
        {
            if (n > 0) MediaInterop.UploadToDevice(pcm, buf);
            return await PredictAsync(buf.View, n).ConfigureAwait(false);
        }
        finally
        {
            await _accelerator.SynchronizeAsync().ConfigureAwait(false);
            buf.Dispose();
        }
    }

    /// <summary>Device path: 16 kHz mono PCM already on the accelerator (the entry the other two funnel into).</summary>
    public async Task<float> PredictAsync(ArrayView1D<float, Stride1D.Dense> pcm, int length)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        _mel.ComputeLogMelSpectrogram(pcm, length, _features.View);
        var outs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = new Tensor(_features.View, new[] { 1, WhisperMelPreprocessor.NMels, Frames }),
        }).ConfigureAwait(false);
        var probability = outs[_session.OutputNames[0]];
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        // Terminal sink: one float.
        using var host = _accelerator.Allocate1D<float>(1);
        host.View.CopyFrom(probability.Data.SubView(0, 1));
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        return (await host.CopyToHostAsync<float>(0, 1).ConfigureAwait(false))[0];
    }

    public void Dispose()
    {
        _features.Dispose();
        _mel.Dispose();
        _session.Dispose();
    }
}
