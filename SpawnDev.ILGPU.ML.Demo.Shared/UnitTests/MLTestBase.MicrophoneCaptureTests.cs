using System;
using System.Threading.Tasks;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Gate for microphone capture (<see cref="MediaStreamCapture.StartMicrophoneAsync"/>), which feeds
/// speech-to-text.
///
/// <para>
/// WHY THIS FILE EXISTS: <c>MediaStreamCapture.OnAudioReady</c> was DECLARED and never raised - the class
/// advertised microphone support while its only capture call was
/// <c>GetUserMedia(video: true, audio: false)</c>. Nothing caught that, because "the event exists" and "the
/// event fires" look identical to a compiler and to a reader. These tests assert the second thing.
/// </para>
///
/// <para>
/// A real microphone cannot be opened in an automated lane, so these drive the half that carries all of the
/// logic: <see cref="MediaInterop.FromAudioDataAsync"/>, which turns a WebCodecs <c>AudioData</c> frame into
/// the mono float32 a speech model consumes. Each test asserts an INVARIANT that a broken conversion would
/// violate - a DC level that must survive downmix and resample, an s16 scale factor, a frame count - rather
/// than merely that some array came back non-empty.
/// </para>
/// </summary>
public abstract partial class MLTestBase
{
    private const int MicRate = 16000;   // what Whisper expects

    /// <summary>AudioData is a browser API; the desktop lanes report Skip rather than a false pass.</summary>
    private static void RequireBrowserForAudio()
    {
        var js = SpawnJSRuntime.Instance;
        if (js == null || !js.IsBrowser)
            throw new UnsupportedTestException("AudioData/WebCodecs is browser-only (not a browser lane)");
    }

    private static AudioData MakeAudioData(string format, int rate, int frames, int channels, TypedArray data)
        => new AudioData(new AudioDataOptions
        {
            Format = format,
            SampleRate = rate,
            NumberOfFrames = frames,
            NumberOfChannels = channels,
            Timestamp = 0,
            Data = data,
        });

    /// <summary>
    /// Stereo f32-planar must downmix to the AVERAGE of the two channels. Constant channels make the
    /// expected value exact: L=+0.5, R=-0.1 must give 0.2 everywhere. Summing without the 1/channels
    /// scale (the easy bug) gives 0.4 and fails.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task Microphone_F32Planar_StereoDownmixesToChannelAverage()
    {
        RequireBrowserForAudio();
        const int frames = 480;
        var planes = new float[frames * 2];
        for (int i = 0; i < frames; i++) { planes[i] = 0.5f; planes[frames + i] = -0.1f; }

        using var data = new Float32Array(planes);
        using var audio = MakeAudioData("f32-planar", MicRate, frames, 2, data);
        var mono = await MediaInterop.FromAudioDataAsync(audio, MicRate);
        audio.Close();

        if (mono.Length != frames)
            throw new Exception($"expected {frames} mono samples, got {mono.Length}");
        const float expected = 0.2f;
        for (int i = 0; i < mono.Length; i++)
            if (Math.Abs(mono[i] - expected) > 1e-5f)
                throw new Exception($"sample {i}: expected {expected} (average of 0.5 and -0.1), got {mono[i]}");
    }

    /// <summary>
    /// Interleaved f32 must be de-interleaved by FRAME, not read as one flat run. A left ramp against a
    /// constant right channel makes a stride error obvious: the average has to keep rising monotonically.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task Microphone_F32Interleaved_DeinterleavesByFrame()
    {
        RequireBrowserForAudio();
        const int frames = 256;
        var interleaved = new float[frames * 2];
        for (int i = 0; i < frames; i++)
        {
            interleaved[i * 2] = i / (float)frames;   // left: ramp 0 -> ~1
            interleaved[i * 2 + 1] = 0f;              // right: silent
        }

        using var data = new Float32Array(interleaved);
        using var audio = MakeAudioData("f32", MicRate, frames, 2, data);
        var mono = await MediaInterop.FromAudioDataAsync(audio, MicRate);
        audio.Close();

        if (mono.Length != frames)
            throw new Exception($"expected {frames} mono samples, got {mono.Length}");
        for (int i = 0; i < frames; i++)
        {
            float expected = (i / (float)frames) * 0.5f;   // ramp averaged with silence
            if (Math.Abs(mono[i] - expected) > 1e-5f)
                throw new Exception($"sample {i}: expected {expected}, got {mono[i]} - interleave stride is wrong");
        }
    }

    /// <summary>
    /// s16 must be scaled to unit float by 1/32768. Full-scale negative is the value that catches a
    /// wrong divisor: -32768 has to land on exactly -1.0.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task Microphone_S16_ScalesToUnitFloat()
    {
        RequireBrowserForAudio();
        var samples = new short[] { 0, 16384, -16384, 32767, -32768 };

        using var data = new Int16Array(samples);
        using var audio = MakeAudioData("s16", MicRate, samples.Length, 1, data);
        var mono = await MediaInterop.FromAudioDataAsync(audio, MicRate);
        audio.Close();

        var expected = new[] { 0f, 0.5f, -0.5f, 32767f / 32768f, -1.0f };
        if (mono.Length != expected.Length)
            throw new Exception($"expected {expected.Length} samples, got {mono.Length}");
        for (int i = 0; i < expected.Length; i++)
            if (Math.Abs(mono[i] - expected[i]) > 1e-5f)
                throw new Exception($"sample {i}: expected {expected[i]}, got {mono[i]}");
    }

    /// <summary>
    /// 48 kHz mic hardware into a 16 kHz model is the production path, so the resample must actually run.
    /// A constant signal must survive it at the same level (any correct resampler preserves DC), and the
    /// frame count must drop by 3x. A no-op resample fails the count; a broken one fails the level.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task Microphone_48kTo16k_ResamplesAndPreservesLevel()
    {
        RequireBrowserForAudio();
        const int srcRate = 48000, frames = 4800;   // 100 ms
        var dc = new float[frames];
        for (int i = 0; i < frames; i++) dc[i] = 0.25f;

        using var data = new Float32Array(dc);
        using var audio = MakeAudioData("f32-planar", srcRate, frames, 1, data);
        var mono = await MediaInterop.FromAudioDataAsync(audio, MicRate);
        audio.Close();

        int expectedLen = frames * MicRate / srcRate;   // 1600
        if (Math.Abs(mono.Length - expectedLen) > 2)
            throw new Exception($"expected ~{expectedLen} samples after 48k->16k, got {mono.Length}");
        if (mono.Length == frames)
            throw new Exception("length unchanged - the resample did not run");

        // Skip the edges, where a resampler's window legitimately rolls off.
        for (int i = 4; i < mono.Length - 4; i++)
            if (Math.Abs(mono[i] - 0.25f) > 1e-3f)
                throw new Exception($"sample {i}: DC level {mono[i]} should have survived resampling as 0.25");
    }

    /// <summary>
    /// An unreadable format must THROW, never return silence. Silence is indistinguishable from a muted
    /// microphone, which is exactly how the missing implementation stayed invisible in the first place.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task Microphone_UnsupportedFormat_ThrowsInsteadOfReturningSilence()
    {
        RequireBrowserForAudio();
        const int frames = 128;
        using var data = new Float32Array(new float[frames]);
        using var audio = MakeAudioData("u8", MicRate, frames, 1, data);

        try
        {
            var mono = await MediaInterop.FromAudioDataAsync(audio, MicRate);
            throw new Exception($"expected NotSupportedException for format 'u8', got {mono.Length} samples");
        }
        catch (NotSupportedException)
        {
            // correct: loud failure
        }
        finally
        {
            try { audio.Close(); } catch { }
        }
    }
}
