using SpawnDev.Phonemizer;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// A narration track for a story video: each line is spoken by Kokoro and placed at the time its shot starts, in one mono track as long as the video. A line longer than its
/// shot is spoken faster (up to <see cref="MaxSpeed"/>); one that still does not fit starts after the line before it ends, so lines never overlap; the track is cut at the video's end.
/// </summary>
public static class StoryNarrator
{
    /// <summary>The fastest a line is spoken to fit its shot (Kokoro's speed; 1 = natural).</summary>
    public const float MaxSpeed = 1.3f;
    /// <summary>Silence kept between two lines, and before the end of a shot, in seconds.</summary>
    public const double Gap = 0.15;
    /// <summary>How long after its shot starts a line begins, in seconds: the picture changes first.</summary>
    public const double Lead = 0.25;

    /// <summary>
    /// The narration: <paramref name="lines"/>[i] spoken from <paramref name="shotStarts"/>[i] + <see cref="Lead"/> seconds, in a track of <paramref name="totalSeconds"/> at Kokoro's
    /// 24 kHz. Empty lines are skipped.
    /// </summary>
    public static async Task<float[]> NarrateAsync(KokoroPipeline kokoro, KokoroVoicePack voice, EnglishPhonemizer phonemizer, IReadOnlyList<string> lines, IReadOnlyList<double> shotStarts,
        double totalSeconds, Action<int>? lineDone = null, CancellationToken ct = default)
    {
        if (lines.Count > shotStarts.Count) throw new ArgumentException($"{lines.Count} lines for {shotStarts.Count} shots");
        var clips = new List<float[]>(); var starts = new List<double>();
        for (int i = 0; i < lines.Count; i++)
        {
            if (string.IsNullOrWhiteSpace(lines[i])) continue;
            double start = shotStarts[i] + Lead, end = i + 1 < shotStarts.Count ? shotStarts[i + 1] : totalSeconds;
            var phonemes = phonemizer.ToSymbols(lines[i].Trim());
            var audio = await kokoro.SpeakAsync(phonemes, voice, 1f, ct);
            double slot = end - start - Gap;
            if (slot > 0 && audio.Seconds > slot)
                audio = await kokoro.SpeakAsync(phonemes, voice, (float)Math.Min(MaxSpeed, audio.Seconds / slot), ct);
            clips.Add(audio.Samples); starts.Add(start);
            lineDone?.Invoke(i);
        }
        return Mix(clips, starts, totalSeconds, KokoroPipeline.OutputSampleRate);
    }

    /// <summary>
    /// The clips in one track of <paramref name="totalSeconds"/>: clip i at <paramref name="startSeconds"/>[i], or <see cref="Gap"/> after the clip before it ends when that is later;
    /// cut at the end with a 50 ms fade, samples clamped to [-1, 1].
    /// </summary>
    public static float[] Mix(IReadOnlyList<float[]> clips, IReadOnlyList<double> startSeconds, double totalSeconds, int sampleRate)
    {
        var track = new float[(int)Math.Round(totalSeconds * sampleRate)];
        int freeFrom = 0;                                          // the first sample after the last placed clip and its gap
        for (int i = 0; i < clips.Count; i++)
        {
            int at = Math.Max((int)Math.Round(startSeconds[i] * sampleRate), freeFrom);
            var c = clips[i];
            for (int s = 0; s < c.Length && at + s < track.Length; s++) track[at + s] += c[s];
            freeFrom = at + c.Length + (int)Math.Round(Gap * sampleRate);
        }
        if (freeFrom - (int)Math.Round(Gap * sampleRate) > track.Length)           // the last line runs past the end: fade it out
        {
            int fade = Math.Min(track.Length, sampleRate / 20);
            for (int s = 0; s < fade; s++) track[track.Length - fade + s] *= 1f - (s + 1) / (float)fade;
        }
        for (int s = 0; s < track.Length; s++) track[s] = Math.Clamp(track[s], -1f, 1f);
        return track;
    }
}
