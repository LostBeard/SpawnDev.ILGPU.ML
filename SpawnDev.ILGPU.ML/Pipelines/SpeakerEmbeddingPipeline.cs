using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Who is speaking: a fixed-size voice embedding for a stretch of speech, from a ReDimNet2 speaker encoder
/// (<c>soniqo/ReDimNet2-B6-ONNX-FP32</c>, MIT; PalabraAI/redimnet2, 12.3M parameters, 0.29% EER on Vox1-O).
/// </summary>
/// <remarks>
/// <para>
/// The export takes raw 16 kHz audio, <c>[1, 96000]</c> (6 s), and returns an L2-normalised 192-d vector. Two
/// voices are compared by the dot product of their embeddings (cosine similarity). The window is fixed, so this
/// pipeline fits ANY clip to it on the device: audio shorter than the window is repeated to fill it (the model's
/// own recommendation), longer audio is centre-cropped. Under about two seconds of speech the embedding is much
/// less reliable - <see cref="MinReliableSeconds"/>.
/// </para>
/// <para>
/// An embedding says "these two clips sound like the same voice"; it is a label, not authentication, and it does not
/// separate overlapping speakers. Do not feed it mixed or overlapped speech as identity evidence.
/// </para>
/// <para>
/// Bulk audio never reaches the host: the waveform is uploaded once (or arrives as a device view), the window is built
/// by a kernel, and only the 192 floats of the result are read back - the terminal sink for this call.
/// </para>
/// </remarks>
public sealed class SpeakerEmbeddingPipeline : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly InferenceSession _session;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _window;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _fitKernel;

    /// <summary>Samples in the model's fixed input window (6 s at 16 kHz).</summary>
    public const int WindowSamples = 96000;

    /// <summary>Length of the embedding.</summary>
    public int EmbeddingSize { get; }

    /// <summary>Below this much speech the embedding is much less reliable (the model card's guidance).</summary>
    public const double MinReliableSeconds = 2.0;

    /// <param name="session">The ReDimNet2 ONNX session: one input <c>audio [1, 96000]</c>, one output <c>[1, N]</c>.</param>
    public SpeakerEmbeddingPipeline(InferenceSession session, Accelerator accelerator, int embeddingSize = 192)
    {
        _session = session;
        _accelerator = accelerator;
        EmbeddingSize = embeddingSize;
        _window = accelerator.Allocate1D<float>(WindowSamples);
        _session.OutputsLiveUntilNextRun = true;
    }

    /// <summary>The embedding of 16 kHz mono PCM (any other rate is resampled on the host first).</summary>
    public async Task<float[]> EmbedAsync(float[] pcm, int sampleRate = 16000)
    {
        ArgumentNullException.ThrowIfNull(pcm);
        if (sampleRate != AudioPreprocessor.WhisperSampleRate)
            pcm = AudioPreprocessor.Resample(pcm, sampleRate, AudioPreprocessor.WhisperSampleRate);
        if (pcm.Length == 0) throw new ArgumentException("no audio to embed", nameof(pcm));
        using var buf = _accelerator.Allocate1D(pcm);
        return await EmbedAsync(buf.View, pcm.Length).ConfigureAwait(false);
    }

    /// <summary>Browser path: the PCM is a JS typed array and goes straight to the device.</summary>
    public async Task<float[]> EmbedAsync(Float32Array pcm, int sampleRate = 16000)
    {
        ArgumentNullException.ThrowIfNull(pcm);
        if (sampleRate != AudioPreprocessor.WhisperSampleRate)
            return await EmbedAsync(pcm.ToArray(), sampleRate).ConfigureAwait(false);
        int n = (int)pcm.Length;
        if (n == 0) throw new ArgumentException("no audio to embed", nameof(pcm));
        var buf = _accelerator.Allocate1D<float>(n);
        try
        {
            MediaInterop.UploadToDevice(pcm, buf);
            return await EmbedAsync(buf.View, n).ConfigureAwait(false);
        }
        finally
        {
            await _accelerator.SynchronizeAsync().ConfigureAwait(false);
            buf.Dispose();
        }
    }

    /// <summary>Device path: 16 kHz mono PCM already on the accelerator (the entry the other two funnel into).</summary>
    public async Task<float[]> EmbedAsync(ArrayView1D<float, Stride1D.Dense> pcm, int length)
    {
        if (length <= 0) throw new ArgumentException("no audio to embed", nameof(length));
        _fitKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(FitToWindow);
        // Long audio: the middle 6 s. Short audio: repeated, so the window is full of the same speech.
        int offset = length > WindowSamples ? (length - WindowSamples) / 2 : 0;
        _fitKernel(WindowSamples, pcm, _window.View, length, offset);

        var outs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = new Tensor(_window.View, new[] { 1, WindowSamples }),
        }).ConfigureAwait(false);
        var embedding = outs[_session.OutputNames[0]];
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        // Terminal sink: EmbeddingSize floats.
        using var host = _accelerator.Allocate1D<float>(EmbeddingSize);
        host.View.CopyFrom(embedding.Data.SubView(0, EmbeddingSize));
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        return await host.CopyToHostAsync<float>(0, EmbeddingSize).ConfigureAwait(false);
    }

    /// <summary>Cosine similarity of two embeddings (both unit length from the model, so it is their dot product).</summary>
    public static float Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length) throw new ArgumentException("embeddings differ in length");
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return na == 0 || nb == 0 ? 0f : (float)(dot / Math.Sqrt(na * nb));
    }

    private static void FitToWindow(Index1D i, ArrayView1D<float, Stride1D.Dense> pcm,
        ArrayView1D<float, Stride1D.Dense> window, int length, int offset)
    {
        window[i] = length >= WindowSamples ? pcm[offset + i] : pcm[i % length];
    }

    public void Dispose()
    {
        _window.Dispose();
        _session.Dispose();
    }
}
