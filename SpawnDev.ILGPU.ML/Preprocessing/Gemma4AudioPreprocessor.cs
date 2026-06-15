namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Audio preprocessing for Gemma 4 12B "Unified" audio (projector type <c>gemma4ua</c>). The encoder-free
/// audio path takes the RAW waveform — there is NO STFT / mel filterbank (the <c>num_mel_bins</c> metadata
/// is a misnomer). Per llama.cpp <c>tools/mtmd/mtmd-audio.cpp</c>: 16 kHz mono, chunked into non-overlapping
/// 640-sample frames (= 40 ms, 25 tokens/sec), the last frame zero-padded. Each <c>[nFrames, 640]</c> block
/// feeds <c>mm.a.input_projection</c> (after a weightless RMSNorm) → <c>[nFrames, 3840]</c> embeddings.
/// </summary>
public static class Gemma4AudioPreprocessor
{
    public const int SampleRate = 16000;
    public const int FrameLen = 640;   // 40 ms @ 16 kHz, non-overlapping

    /// <summary>
    /// Chunk 16 kHz mono float samples (normalized to roughly [-1, 1]) into <c>[nFrames, 640]</c> frames,
    /// the final frame zero-padded. Returns the flat frame buffer + the frame (token) count
    /// = ceil(samples.Length / 640).
    /// </summary>
    public static (float[] frames, int nFrames) Frame(float[] samples)
    {
        if (samples.Length == 0) return (Array.Empty<float>(), 0);
        int nFrames = (samples.Length + FrameLen - 1) / FrameLen;   // ceil
        var frames = new float[(long)nFrames * FrameLen];           // zero-init provides the tail padding
        Array.Copy(samples, frames, samples.Length);
        return (frames, nFrames);
    }
}
