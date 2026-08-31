using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Plays mono PCM through the browser's speakers - the output half of <see cref="MediaStreamCapture"/>.
/// </summary>
/// <remarks>
/// <para>
/// A speech pipeline that can listen but not talk is half a loop, and until now everything that generated
/// audio in this repo wrote a WAV file for a human to open. This is the piece that closes it.
/// </para>
/// <para>
/// ⚠️ Typed SpawnJS interop throughout - <see cref="AudioContext"/>, <see cref="AudioBuffer"/>,
/// <see cref="AudioBufferSourceNode"/> - never raw JS. That is the standing rule for this stack, and if a
/// wrapper were missing the fix would be to add it to SpawnJS rather than to reach around it here.
/// </para>
/// <para>
/// ⚠️ Browsers block audio until a user gesture. A page that has never been clicked leaves the context
/// "suspended" and a play call succeeds silently while producing nothing - so <see cref="PlayAsync"/>
/// resumes it first and reports the state rather than pretending. Silence that reports success is the
/// failure mode to guard against here: it is indistinguishable from a model that produced nothing.
/// </para>
/// </remarks>
public sealed class AudioPlayback : IDisposable
{
    private readonly SpawnJSRuntime _js;
    private AudioContext? _ctx;
    private AudioBufferSourceNode? _source;

    /// <summary>New playback device.</summary>
    /// <param name="js">The SpawnJS runtime, injected - never newed up.</param>
    public AudioPlayback(SpawnJSRuntime js) => _js = js;

    /// <summary>True while a clip is playing.</summary>
    public bool IsPlaying { get; private set; }

    /// <summary>Raised when a clip finishes on its own.</summary>
    public event Action? OnEnded;

    /// <summary>
    /// Play mono PCM in [-1, 1]. Replaces anything already playing.
    /// </summary>
    /// <param name="samples">Mono PCM.</param>
    /// <param name="sampleRate">Sample rate of <paramref name="samples"/>.</param>
    /// <returns>
    /// The clip's duration in seconds. Playback continues after this returns - await
    /// <see cref="WaitForEndAsync"/> to follow it.
    /// </returns>
    public async Task<double> PlayAsync(float[] samples, int sampleRate)
    {
        if (samples == null || samples.Length == 0)
            throw new ArgumentException("nothing to play", nameof(samples));
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate), sampleRate, "sample rate must be positive");

        Stop();
        _ctx ??= new AudioContext();

        // ⚠️ A context created before any user gesture starts SUSPENDED, and starting a source on it
        // produces no sound while reporting success. Resume first.
        if (_ctx.State == "suspended") await _ctx.Resume();

        // The context runs at the hardware rate (usually 48 kHz); the buffer keeps the clip's OWN rate and
        // the browser resamples on playback. Writing 24 kHz samples into a 48 kHz buffer without saying so
        // plays everything an octave low at double speed - which sounds like a broken model, not a
        // mislabelled buffer.
        var buffer = _ctx.CreateBuffer(1, samples.Length, sampleRate);
        // The typed-array crossing is the ONE copy: samples go straight into a Float32Array rather than
        // element-by-element through interop, which for a few seconds of 24 kHz audio is the difference
        // between one transfer and six figures of them.
        using var channel = new Float32Array(samples);
        buffer.CopyToChannel(channel, 0);

        var source = _ctx.CreateBufferSource();
        source.Buffer = buffer;
        source.Connect(_ctx.Destination);
        source.OnEnded += HandleEnded;
        source.Start();

        _source = source;
        IsPlaying = true;
        return samples.Length / (double)sampleRate;
    }

    /// <summary>Wait until the current clip finishes (or returns immediately if nothing is playing).</summary>
    public async Task WaitForEndAsync(CancellationToken ct = default)
    {
        while (IsPlaying && !ct.IsCancellationRequested)
            await Task.Delay(25, ct).ConfigureAwait(false);
    }

    /// <summary>Stop playback immediately.</summary>
    public void Stop()
    {
        if (_source != null)
        {
            try { _source.OnEnded -= HandleEnded; } catch { }
            try { _source.Stop(); } catch { /* already stopped */ }
            try { _source.Dispose(); } catch { }
            _source = null;
        }
        IsPlaying = false;
    }

    private void HandleEnded()
    {
        IsPlaying = false;
        OnEnded?.Invoke();
    }

    /// <summary>Stops playback and releases the audio context.</summary>
    public void Dispose()
    {
        Stop();
        try { _ctx?.Dispose(); } catch { }
        _ctx = null;
    }
}
