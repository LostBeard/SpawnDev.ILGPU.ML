namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// NeMo's log-mel features, computed frame by frame AS AUDIO ARRIVES: the input of the cache-aware streaming FastConformer
/// (<c>nvidia/nemotron-speech-streaming-en-0.6b</c>) and of NeMo's other 128-bin models.
/// </summary>
/// <remarks>
/// <para>
/// The recipe is NeMo's <c>FilterbankFeatures</c> exactly (and was checked against sherpa-onnx's transcripts, not against a guess):
/// pre-emphasis 0.97, a SYMMETRIC Hann window of 400 samples (<c>hann_window(400, periodic=False)</c> - NeMo's, unlike Whisper's) centred in a 512-point
/// FFT, frames centred on <c>f * 160</c> with ZERO padding outside the signal, power spectrum, 128 Slaney mel filters from 0 to 8 kHz, then
/// <c>log(x + 2^-24)</c> and NO normalisation (<c>normalize: NA</c> - a per-utterance statistic cannot exist while the speaker is still talking).
/// </para>
/// <para>
/// Frame <c>f</c> covers samples <c>[f*160 - 200, f*160 + 200)</c>, so it is ready 200 samples after its centre. Everything is double precision on the host:
/// a chunk is 160 ms of audio (a few thousand samples, already on the host when it comes from a microphone), the work is ~0.2 ms, and it leaves no trigonometric
/// error to account for in the comparison with the reference.
/// </para>
/// </remarks>
public sealed class NemoStreamingFeatures
{
    public const int SampleRate = 16000;
    public const int FftSize = 512;
    public const int WindowLength = 400;
    public const int Hop = 160;
    public const int Mels = 128;
    private const int Bins = FftSize / 2 + 1;
    private const int HalfWindow = WindowLength / 2;

    private readonly double _preemphasis;
    private readonly double[] _window = new double[WindowLength];
    private readonly float[,] _filters;
    private readonly double[] _cos = new double[FftSize / 2];
    private readonly double[] _sin = new double[FftSize / 2];
    private readonly double[] _re = new double[FftSize];
    private readonly double[] _im = new double[FftSize];

    // Pre-emphasised samples from global index _origin on; everything older than the next frame needs is dropped.
    private float[] _samples = new float[SampleRate];
    private int _count;
    private long _origin;
    private float _lastRaw;
    private bool _anyAudio;
    private long _total;

    /// <summary>Index of the next frame to be produced (0 at the start of a stream).</summary>
    public long NextFrame { get; private set; }

    public NemoStreamingFeatures(double preemphasis = 0.97)
    {
        _preemphasis = preemphasis;
        // torch.hann_window(400, periodic=False)
        for (int i = 0; i < WindowLength; i++) _window[i] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * i / (WindowLength - 1));
        _filters = AudioPreprocessor.GenerateMelFilterbankSlaney(Mels, Bins, SampleRate);
        for (int k = 0; k < FftSize / 2; k++)
        {
            _cos[k] = Math.Cos(2 * Math.PI * k / FftSize);
            _sin[k] = -Math.Sin(2 * Math.PI * k / FftSize);
        }
    }

    /// <summary>Forgets the stream: the next audio is the start of a new one.</summary>
    public void Reset()
    {
        _count = 0; _origin = 0; _lastRaw = 0; _anyAudio = false; _total = 0; NextFrame = 0;
    }

    /// <summary>Appends 16 kHz mono audio in [-1, 1].</summary>
    public void Accept(ReadOnlySpan<float> pcm)
    {
        if (pcm.IsEmpty) return;
        if (_count + pcm.Length > _samples.Length) Array.Resize(ref _samples, Math.Max(_samples.Length * 2, _count + pcm.Length));
        for (int i = 0; i < pcm.Length; i++)
        {
            // x'[0] = x[0]; x'[t] = x[t] - 0.97 x[t-1]
            _samples[_count++] = !_anyAudio ? pcm[i] : (float)(pcm[i] - _preemphasis * _lastRaw);
            _lastRaw = pcm[i];
            _anyAudio = true;
        }
        _total += pcm.Length;
    }

    /// <summary>Frames that can be produced from the audio so far (their whole window has arrived).</summary>
    public int FramesAvailable => _total < HalfWindow ? 0 : (int)Math.Max(0, (_total - HalfWindow) / Hop + 1 - NextFrame);

    /// <summary>Produces the next frame into <paramref name="mel"/> (128 values), or false when its window has not fully arrived yet.</summary>
    public bool TryNext(float[] mel)
    {
        if (mel.Length < Mels) throw new ArgumentException("needs 128 values", nameof(mel));
        long start = NextFrame * Hop - HalfWindow;                 // first sample of the window, in the stream
        if (start + WindowLength > _total) return false;

        int offset = (FftSize - WindowLength) / 2;                 // the 400-sample window sits in the middle of the 512-point frame
        Array.Clear(_re); Array.Clear(_im);
        for (int i = 0; i < WindowLength; i++)
        {
            long at = start + i;                                   // zero outside the signal (centre=True, constant padding)
            if (at < 0) continue;
            _re[offset + i] = _samples[at - _origin] * _window[i];
        }
        Fft();

        for (int m = 0; m < Mels; m++)
        {
            double sum = 0;
            for (int k = 0; k < Bins; k++)
            {
                float w = _filters[m, k];
                if (w != 0f) sum += w * (_re[k] * _re[k] + _im[k] * _im[k]);
            }
            mel[m] = (float)Math.Log(sum + 5.9604644775390625e-8);  // 2^-24
        }

        NextFrame++;
        // Samples older than the next frame's window are never needed again.
        long keepFrom = NextFrame * Hop - HalfWindow;
        if (keepFrom > _origin)
        {
            int drop = (int)Math.Min(keepFrom - _origin, _count);
            if (drop > 0)
            {
                Array.Copy(_samples, drop, _samples, 0, _count - drop);
                _count -= drop; _origin += drop;
            }
        }
        return true;
    }

    /// <summary>In-place radix-2 FFT of <see cref="_re"/> / <see cref="_im"/> (double precision).</summary>
    private void Fft()
    {
        int n = FftSize;
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) { (_re[i], _re[j]) = (_re[j], _re[i]); (_im[i], _im[j]) = (_im[j], _im[i]); }
        }
        for (int len = 2; len <= n; len <<= 1)
        {
            int step = n / len;
            for (int i = 0; i < n; i += len)
                for (int k = 0; k < len / 2; k++)
                {
                    double wr = _cos[k * step], wi = _sin[k * step];
                    int a = i + k, b = a + len / 2;
                    double tr = _re[b] * wr - _im[b] * wi, ti = _re[b] * wi + _im[b] * wr;
                    _re[b] = _re[a] - tr; _im[b] = _im[a] - ti;
                    _re[a] += tr; _im[a] += ti;
                }
        }
    }
}
