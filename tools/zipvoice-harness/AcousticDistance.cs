// A second opinion on "did this perturbation change anything", with no language model in the loop.
//
// WHY THIS EXISTS: Whisper is the grader for the sensitivity experiment, and Whisper is a strong language
// model. Handed a mispronounced word in a sentence whose meaning is obvious, it will happily transcribe
// the word the sentence made likely rather than the sounds actually present. Its bias therefore runs in
// exactly the wrong direction for us: it makes the model look MORE tolerant of phonemizer error than it
// is, and a 0% WER row could mean "no damage" or "damage Whisper papered over".
//
// This measures the audio directly. Same sentence, same voice, same noise seed, so the control and the
// perturbed render should be near-identical waveforms; whatever distance remains is what the perturbation
// did to the SOUND, whether or not it survived as a word. Read together the two numbers separate three
// cases that WER alone cannot:
//
//   low WER, low distance  -> the model genuinely ignored the change
//   low WER, HIGH distance -> it sounds different but stays intelligible: accent, rhythm, delivery
//   high WER               -> the words themselves broke
//
// Distance is mean per-frame Euclidean distance over Whisper's log-mel, DTW-aligned because a
// perturbation can change the duration, with a band so the alignment cannot wander arbitrarily far.
using SpawnDev.ILGPU.ML.Preprocessing;

namespace ZipVoiceHarness;

public static class AcousticDistance
{
    private const int Band = 200;   // frames; ~2s of slack is far more than any perturbation shifts timing

    /// <summary>Mean per-frame log-mel distance between two clips, DTW-aligned.</summary>
    public static double Between(float[] a, int rateA, float[] b, int rateB)
    {
        var melA = Mel(a, rateA);
        var melB = Mel(b, rateB);
        if (melA.Count == 0 || melB.Count == 0) return double.NaN;
        return Dtw(melA, melB);
    }

    // Whisper's mel is fixed-window: it pads to 30 seconds. Padding is silence and identical in both
    // clips, so leaving it in would drag every distance toward zero and hide exactly what we are looking
    // for. Only the frames the audio actually occupies are kept.
    private static List<float[]> Mel(float[] samples, int rate)
    {
        var probe = rate == AudioPreprocessor.WhisperSampleRate
            ? samples
            : AudioPreprocessor.Resample(samples, rate, AudioPreprocessor.WhisperSampleRate);
        var flat = AudioPreprocessor.ComputeLogMelSpectrogram(probe);
        int bins = AudioPreprocessor.WhisperMelBins;
        int totalFrames = flat.Length / bins;
        int realFrames = Math.Min(totalFrames, probe.Length / 160);   // 160 = Whisper's hop
        var frames = new List<float[]>(realFrames);
        for (int f = 0; f < realFrames; f++)
        {
            var frame = new float[bins];
            // ComputeLogMelSpectrogram returns [bins, frames] - bin-major, so stride by totalFrames.
            for (int c = 0; c < bins; c++) frame[c] = flat[c * totalFrames + f];
            frames.Add(frame);
        }
        return frames;
    }

    private static double Dtw(List<float[]> a, List<float[]> b)
    {
        int n = a.Count, m = b.Count;
        var prev = new double[m + 1];
        var cur = new double[m + 1];
        Array.Fill(prev, double.PositiveInfinity);
        prev[0] = 0;
        for (int i = 1; i <= n; i++)
        {
            Array.Fill(cur, double.PositiveInfinity);
            int lo = Math.Max(1, i - Band), hi = Math.Min(m, i + Band);
            for (int j = lo; j <= hi; j++)
            {
                double cost = Distance(a[i - 1], b[j - 1]);
                double best = Math.Min(Math.Min(prev[j], cur[j - 1]), prev[j - 1]);
                cur[j] = cost + best;
            }
            (prev, cur) = (cur, prev);
        }
        double total = prev[m];
        return double.IsInfinity(total) ? double.NaN : total / Math.Max(n, m);
    }

    private static double Distance(float[] x, float[] y)
    {
        double sum = 0;
        for (int i = 0; i < x.Length; i++) { double d = x[i] - y[i]; sum += d * d; }
        return Math.Sqrt(sum);
    }
}
