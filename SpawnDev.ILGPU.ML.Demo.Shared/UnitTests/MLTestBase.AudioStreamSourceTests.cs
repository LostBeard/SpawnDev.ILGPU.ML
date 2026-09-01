using System;
using System.Threading;
using System.Threading.Tasks;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Gate for <see cref="MediaStreamCapture.StartFromAudioStreamAsync"/> - capturing from a MediaStream the
/// CALLER supplies, instead of opening the microphone.
///
/// <para>
/// WHY THIS FILE EXISTS: this is the seam that moves the hands-free loop off this machine.
/// <c>StartMicrophoneAsync</c> calls <c>getUserMedia</c> itself, which is right for a browser demo and
/// useless for the robot - Gemineachy hears through a WebRTC track arriving from Rose and there is no
/// microphone on that side at all. Everything downstream (frame loop, native-rate handling, buffering, VAD,
/// recogniser) is shared, so the loop is written once and fed from either end. An untested seam is how that
/// claim quietly stops being true.
/// </para>
///
/// <para>
/// ⚠️ The stream here is REAL, not a mock: an oscillator wired into a
/// <c>MediaStreamAudioDestinationNode</c> produces a genuine <c>MediaStream</c> with a live audio track and
/// needs no microphone permission. So this exercises the actual WebCodecs
/// <c>MediaStreamTrackProcessor</c> path a WebRTC track would take. Chrome's fake audio device yields
/// digital SILENCE on this machine, so a test that merely counted frames could pass on nothing - this one
/// asserts the samples carry ENERGY.
/// </para>
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// A caller-supplied MediaStream delivers real audio frames through <c>OnAudioReady</c>.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task AudioStreamSource_CapturesFromCallerSuppliedStream()
    {
        RequireBrowserForAudio();

        var js = SpawnJSRuntime.Instance!;
        using var ctx = new AudioContext();
        using var dest = ctx.CreateMediaStreamDestination();
        using var osc = ctx.CreateOscillator();
        using var freq = osc.Frequency;
        freq.Value = 440f;
        osc.Connect(dest);
        osc.Start();

        using var stream = dest.Stream;
        using var capture = new MediaStreamCapture(js);

        int frames = 0;
        long samples = 0;
        double energy = 0;
        int reportedRate = 0;
        capture.OnAudioReady += (float[] pcm, int rate) =>
        {
            Interlocked.Increment(ref frames);
            reportedRate = rate;
            samples += pcm.Length;
            foreach (var v in pcm) energy += (double)v * v;
        };

        // ownsStream: false is the Gemineachy case - the caller keeps the stream (there, a live WebRTC
        // connection). Asserted below by using the stream again AFTER capture stops.
        var started = await capture.StartFromAudioStreamAsync(stream, targetSampleRate: 0,
            maxBufferedFrames: 3000, ownsStream: false);
        if (!started)
            throw new Exception("StartFromAudioStreamAsync returned false"
                              + (capture.LastAudioError != null ? $": {capture.LastAudioError.Message}" : ""));

        // A MediaStreamTrackProcessor emits ~10 ms frames, so half a second is many frames with margin.
        var deadline = DateTime.UtcNow.AddSeconds(5);
        while (frames < 5 && DateTime.UtcNow < deadline)
            await Task.Delay(50);

        capture.StopMicrophone();

        if (frames == 0)
            throw new Exception("no audio frames arrived from the caller-supplied stream - OnAudioReady "
                              + "never fired, which is the exact failure this class shipped with before "
                              + "(a declared event that was never raised)");
        if (reportedRate <= 0)
            throw new Exception($"reported sample rate {reportedRate} - the callback must report the rate "
                              + "it is handing over, since the recogniser resamples from it");

        var rms = Math.Sqrt(energy / Math.Max(1, samples));
        if (rms < 1e-4)
            throw new Exception($"{frames} frames / {samples} samples arrived but RMS is {rms:E2} - that is "
                              + "SILENCE with the right shape, which is what a fake audio device produces "
                              + "and what an unconnected graph produces");

        // ownsStream:false must NOT have stopped the caller's track. A stopped track reports "ended".
        using var tracks = stream.GetAudioTracks();
        foreach (var t in tracks.ToArray())
        {
            var state = t.ReadyState;
            t.Dispose();
            if (state == "ended")
                throw new Exception("capture stopped the CALLER's track (ownsStream:false) - doing this to a "
                                  + "live WebRTC stream would end the call in order to stop listening to it");
        }

        Console.WriteLine($"[AudioStreamSource] {frames} frames, {samples} samples @ {reportedRate}Hz, rms {rms:F4}");
    }
}
