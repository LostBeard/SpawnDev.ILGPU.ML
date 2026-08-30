using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// High-level capture pipeline for real-time webcam and microphone inference.
/// Combines MediaInterop (efficient pixel/audio extraction) with InferenceRateController
/// (FPS limiting, motion gating) to provide a zero-configuration capture loop.
///
/// Usage:
/// <code>
/// var capture = new MediaStreamCapture(js);
/// await capture.StartWebcamAsync(640, 480);
/// capture.OnFrameReady += (rgba, w, h) => { /* preprocess and run inference */ };
/// </code>
/// </summary>
public class MediaStreamCapture : IDisposable
{
    private readonly SpawnJSRuntime _js;
    private readonly MediaInterop _interop;
    private MediaStream? _stream;
    private HTMLVideoElement? _video;
    private HTMLCanvasElement? _hiddenCanvas;
    private CancellationTokenSource? _captureCts;
    private bool _isCapturing;
    private MediaStream? _audioStream;
    private MediaStreamTrackProcessor? _audioProcessor;
    private ReadableStreamDefaultReader? _audioReader;
    private CancellationTokenSource? _audioCts;
    private int _audioTargetRate = 16000;

    /// <summary>Current capture dimensions.</summary>
    public int Width { get; private set; }
    public int Height { get; private set; }

    /// <summary>Whether the capture is currently running.</summary>
    public bool IsCapturing => _isCapturing;

    /// <summary>
    /// Fired when a new video frame is captured.
    /// Parameters: (byte[] rgba, int width, int height)
    /// </summary>
    public event Action<byte[], int, int>? OnFrameReady;

    /// <summary>
    /// Fired for every chunk of captured microphone audio, as MONO float32 at the rate requested from
    /// <see cref="StartMicrophoneAsync"/> (16 kHz by default, which is what Whisper expects).
    /// Parameters: (float[] samples, int sampleRate)
    /// </summary>
    public event Action<float[], int>? OnAudioReady;

    /// <summary>
    /// Fired when audio capture STOPS because of an error - an unreadable sample format, most likely.
    /// Without this a failing capture is indistinguishable from a silent microphone.
    /// </summary>
    public event Action<Exception>? OnAudioError;

    /// <summary>The error that ended audio capture, if any. Cleared by a new StartMicrophoneAsync.</summary>
    public Exception? LastAudioError { get; private set; }

    /// <summary>Whether microphone capture is currently running.</summary>
    public bool IsCapturingAudio => _audioReader != null;

    /// <summary>Target capture FPS. Actual rate may be lower if inference is slow.</summary>
    public float TargetFps { get; set; } = 30;

    /// <summary>Skip frames with motion below this threshold. 0 = never skip.</summary>
    public float MotionThreshold { get; set; }

    public MediaStreamCapture(SpawnJSRuntime js)
    {
        _js = js;
        _interop = new MediaInterop(js);
    }

    /// <summary>
    /// Start capturing video from the user's webcam.
    /// Frames are delivered via OnFrameReady at TargetFps.
    /// </summary>
    public async Task<bool> StartWebcamAsync(int width = 640, int height = 480, bool facingUser = true)
    {
        if (_isCapturing) return false;

        try
        {
            using var navigator = _js.Get<Navigator>("navigator");
            using var mediaDevices = navigator.MediaDevices;
            _stream = await mediaDevices.GetUserMedia(video: true, audio: false);
            if (_stream == null) return false;

            Width = width;
            Height = height;

            // Create hidden video element to receive the stream
            _video = new HTMLVideoElement();
            _video.SrcObject = _stream;
            _video.Play();

            // Wait for video to start
            await Task.Delay(100);

            _isCapturing = true;
            _captureCts = new CancellationTokenSource();
            _ = CaptureLoop(_captureCts.Token);

            return true;
        }
        catch
        {
            Stop();
            return false;
        }
    }

    /// <summary>
    /// Start capturing from an existing MediaStream (e.g., screen share, remote stream).
    /// </summary>
    public void StartFromStream(MediaStream stream, int width, int height)
    {
        if (_isCapturing) return;

        _stream = stream;
        Width = width;
        Height = height;

        _video = new HTMLVideoElement();
        _video.SrcObject = _stream;
        _video.Play();

        _isCapturing = true;
        _captureCts = new CancellationTokenSource();
        _ = CaptureLoop(_captureCts.Token);
    }

    /// <summary>
    /// Start capturing microphone audio. Chunks arrive on <see cref="OnAudioReady"/> as mono float32
    /// resampled to <paramref name="targetSampleRate"/>, ready to hand straight to a speech model.
    /// </summary>
    /// <remarks>
    /// Uses <c>MediaStreamTrackProcessor</c> - the browser hands us decoded <c>AudioData</c> frames
    /// directly, so there is no <c>ScriptProcessorNode</c> on the audio thread and no polling loop.
    /// <para>
    /// Audio DOES cross into the .NET heap here, against the usual "bulk data stays in JS" rule. It is
    /// the justified exception: a chunk is one AudioData frame (order of 10 ms - a few hundred floats),
    /// speech models consume CPU-side float samples anyway, and the mel preprocessing that follows is
    /// CPU work. Video frames, orders of magnitude larger, keep using the JS-side path.
    /// </para>
    /// </remarks>
    /// <returns>True if the microphone opened and the read loop started.</returns>
    /// <param name="targetSampleRate">
    /// Rate to resample each chunk to, or <b>0 (the default) to deliver the device's NATIVE rate</b>.
    /// <para>
    /// ⚠️ Prefer 0. Chunks arrive about every 10 ms, so a non-zero value resamples each one
    /// INDEPENDENTLY, and a windowed kernel has no signal either side of a chunk boundary to work with -
    /// which stitches a discontinuity into the audio every 10 ms. Capturing native and converting the
    /// finished recording once is both higher quality and less work. <see cref="OnAudioReady"/> reports
    /// the rate it is handing you, and <c>SpeechRecognitionPipeline.TranscribeAsync</c> already resamples
    /// whatever rate you pass it.
    /// </para>
    /// </param>
    /// <param name="maxBufferedFrames">
    /// How many AudioData frames the browser may queue for us. The default queue is short, so anything
    /// that stalls the single WASM thread - a large model download, a long GPU compile - makes the browser
    /// DROP frames, and dropped frames do not announce themselves: the capture simply comes back short and
    /// the audio is silently chopped. MEASURED: capturing while a 231 MB download was in flight yielded
    /// 7.2 s of audio over 9 s of wall time (80%), against 100% with no download running. Queuing instead
    /// of dropping costs a little memory and keeps the recording intact.
    /// </param>
    public async Task<bool> StartMicrophoneAsync(int targetSampleRate = 0, int maxBufferedFrames = 3000)
    {
        if (_audioProcessor != null) return false;
        if (targetSampleRate < 0) throw new ArgumentOutOfRangeException(nameof(targetSampleRate));

        LastAudioError = null;
        _audioTargetRate = targetSampleRate;
        try
        {
            using var navigator = _js.Get<Navigator>("navigator");
            using var mediaDevices = navigator.MediaDevices;
            _audioStream = await mediaDevices.GetUserMedia(video: false, audio: true);
            if (_audioStream == null) return false;

            using var tracks = _audioStream.GetAudioTracks();
            var track = tracks.ToArray().FirstOrDefault();
            if (track == null) { StopMicrophone(); return false; }

            _audioProcessor = new MediaStreamTrackProcessor(new MediaStreamTrackProcessorOptions
            {
                Track = track,
                MaxBufferSize = maxBufferedFrames > 0 ? maxBufferedFrames : null,
            });
            using var readable = _audioProcessor.Readable;
            _audioReader = readable.GetReader();

            _audioCts = new CancellationTokenSource();
            _ = AudioLoop(_audioCts.Token);
            return true;
        }
        catch (Exception ex)
        {
            LastAudioError = ex;
            StopMicrophone();
            return false;
        }
    }

    /// <summary>Stop microphone capture and release the audio track.</summary>
    public void StopMicrophone()
    {
        _audioCts?.Cancel();
        _audioCts?.Dispose();
        _audioCts = null;

        _audioReader?.Dispose();
        _audioReader = null;
        _audioProcessor?.Dispose();
        _audioProcessor = null;

        if (_audioStream != null)
        {
            using var tracks = _audioStream.GetAudioTracks();
            tracks.ToArray().UsingEach(t => t.Stop());
            _audioStream.Dispose();
            _audioStream = null;
        }
    }

    private async Task AudioLoop(CancellationToken ct)
    {
        // Nothing may escape this method. An unhandled exception on a runtime callback EXITS the .NET
        // WASM runtime, taking the whole page with it - so a failure is reported through OnAudioError.
        try
        {
            while (!ct.IsCancellationRequested && _audioReader != null)
            {
                ReadableStreamReaderReadResponse res;
                try { res = await _audioReader.Read(); }
                catch { break; }
                if (res.Done) { res.Dispose(); break; }

                // The chunk of an audio MediaStreamTrackProcessor is an AudioData, not a byte view -
                // read it with the correct wrapper type rather than the reader's byte-typed Value.
                var audioData = res.JSRef!.Get<AudioData?>("value");
                res.Dispose();
                if (audioData is null) continue;

                try
                {
                    // 0 = native: hand over the frame's own rate and do not touch the samples.
                    int rate = _audioTargetRate > 0 ? _audioTargetRate : (int)audioData.SampleRate;
                    var samples = await MediaInterop.FromAudioDataAsync(audioData, rate);
                    if (samples.Length > 0) OnAudioReady?.Invoke(samples, rate);
                }
                finally
                {
                    try { audioData.Close(); } catch { }
                    audioData.Dispose();
                }
            }
        }
        catch (Exception ex)
        {
            // A format we cannot read would otherwise throw on EVERY frame and look like silence.
            LastAudioError = ex;
            try { OnAudioError?.Invoke(ex); } catch { }
        }
    }

    /// <summary>
    /// Stop capturing and release all resources.
    /// </summary>
    public void Stop()
    {
        StopMicrophone();
        _isCapturing = false;
        _captureCts?.Cancel();
        _captureCts?.Dispose();
        _captureCts = null;

        if (_stream != null)
        {
            using var tracks = _stream.GetTracks();
            tracks.ToArray().UsingEach(t => t.Stop());
            _stream.Dispose();
            _stream = null;
        }

        _video?.Dispose();
        _video = null;
        _hiddenCanvas?.Dispose();
        _hiddenCanvas = null;
    }

    /// <summary>
    /// Capture a single frame right now (outside the automatic loop).
    /// Returns RGBA pixel data.
    /// </summary>
    public byte[]? CaptureFrame()
    {
        if (_video == null) return null;
        return _interop.FromVideoElement(_video, Width, Height);
    }

    /// <summary>
    /// Capture a single frame and preprocess it for a specific model.
    /// Returns a float tensor ready for inference.
    /// </summary>
    public float[]? CaptureAndPreprocess(ModelConfig config)
    {
        if (_video == null) return null;
        return _interop.VideoToTensor(_video, config);
    }

    private async Task CaptureLoop(CancellationToken ct)
    {
        var rateController = new InferenceRateController(TargetFps, MotionThreshold);
        byte[]? prevFrame = null;

        while (!ct.IsCancellationRequested && _isCapturing)
        {
            try
            {
                if (_video == null) break;

                if (rateController.ShouldRunInference(prevFrame))
                {
                    var rgba = _interop.FromVideoElement(_video, Width, Height);
                    rateController.MarkInferenceRun(rgba);
                    prevFrame = rgba;

                    OnFrameReady?.Invoke(rgba, Width, Height);
                }

                // Yield to keep UI responsive
                await Task.Delay(1, ct);
            }
            catch (OperationCanceledException) { break; }
            catch { /* Frame capture failed, try next frame */ }
        }
    }

    public void Dispose()
    {
        Stop();
        _interop.Dispose();
    }
}
