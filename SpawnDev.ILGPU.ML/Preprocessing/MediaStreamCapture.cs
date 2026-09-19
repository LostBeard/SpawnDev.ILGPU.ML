using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using TypedArray = SpawnDev.SpawnJS.JSObjects.TypedArray;

namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// High-level capture pipeline for real-time webcam and microphone inference.
/// Combines MediaInterop (efficient pixel/audio extraction) with InferenceRateController
/// (FPS limiting, motion gating) to provide a zero-configuration capture loop.
///
/// Usage (browser — prefer the JS path):
/// <code>
/// var capture = new MediaStreamCapture(js);
/// await capture.StartWebcamAsync(640, 480);
/// capture.OnFrameReadyJs += (rgba, w, h) => { /* UploadToDevice / pipeline TypedArray overload */ };
/// </code>
/// Desktop/managed consumers may use <see cref="OnFrameReady"/> (<c>byte[]</c>); that path
/// crosses into the .NET heap via <c>ReadBytes</c> and should not be the browser default.
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
    /// <summary>True when this instance opened <see cref="_audioStream"/> and must stop its tracks.</summary>
    private bool _ownsAudioStream;
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
    /// Fired when a new video frame is captured as managed RGBA bytes.
    /// Parameters: (byte[] rgba, int width, int height).
    /// </summary>
    /// <remarks>
    /// ⚠️ IN A BROWSER, prefer <see cref="OnFrameReadyJs"/>. This event pulls the frame onto the .NET
    /// heap via <c>ReadBytes</c>. It is kept for desktop/managed consumers and for callers that truly
    /// need <c>byte[]</c> (file save, CPU codecs). Subscribing only to this event still works — the
    /// capture loop converts from the JS typed array when needed — but it is the slow path.
    /// </remarks>
    public event Action<byte[], int, int>? OnFrameReady;

    /// <summary>
    /// Fired when a new video frame is captured as a JS typed array (RGBA <see cref="Uint8ClampedArray"/>).
    /// Parameters: (TypedArray rgba, int width, int height).
    /// </summary>
    /// <remarks>
    /// The preferred browser path: pixels never enter the .NET managed heap. Hand the array to
    /// <see cref="MediaInterop.UploadToDevice{T}"/> or a pipeline's <c>TypedArray</c> overload.
    /// Ownership: the capture loop disposes the array after synchronous handlers return — upload or
    /// clone inside the handler if you need the pixels after the event returns.
    /// </remarks>
    public event Action<TypedArray, int, int>? OnFrameReadyJs;

    /// <summary>
    /// Fired for every chunk of captured microphone audio, as MONO float32 at the rate requested from
    /// <see cref="StartMicrophoneAsync"/> (16 kHz by default, which is what Whisper expects).
    /// Parameters: (float[] samples, int sampleRate)
    /// </summary>
    /// <remarks>
    /// ⚠️ IN A BROWSER, prefer <see cref="OnAudioReadyJs"/>. This event pulls each chunk onto the .NET
    /// heap. Kept for desktop/managed consumers and utterance ring buffers that still need <c>float[]</c>.
    /// </remarks>
    public event Action<float[], int>? OnAudioReady;

    /// <summary>
    /// Fired for every chunk of captured microphone audio as a JS <see cref="Float32Array"/> (mono float32).
    /// Parameters: (Float32Array samples, int sampleRate).
    /// </summary>
    /// <remarks>
    /// Preferred browser path when the chunk can stay in JS (f32 source, no per-chunk resample). Ownership:
    /// the capture loop disposes the array after synchronous handlers return — clone or transfer inside
    /// the handler if you need the samples after the event returns. When downmix/resample/s16 forces a
    /// host path, a new <see cref="Float32Array"/> is still delivered here so VAD/worker callers never
    /// need to convert themselves.
    /// </remarks>
    public event Action<Float32Array, int>? OnAudioReadyJs;

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
    /// Frames are delivered via <see cref="OnFrameReadyJs"/> (preferred) and/or <see cref="OnFrameReady"/>
    /// at <see cref="TargetFps"/>.
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
    /// Start capturing microphone audio. Chunks arrive on <see cref="OnAudioReadyJs"/> (preferred in
    /// browser) and/or <see cref="OnAudioReady"/> as mono float32 resampled to
    /// <paramref name="targetSampleRate"/>, ready to hand straight to a speech model.
    /// </summary>
    /// <remarks>
    /// Uses <c>MediaStreamTrackProcessor</c> - the browser hands us decoded <c>AudioData</c> frames
    /// directly, so there is no <c>ScriptProcessorNode</c> on the audio thread and no polling loop.
    /// Prefer <see cref="OnAudioReadyJs"/> so f32 frames never enter the .NET heap; subscribe to
    /// <see cref="OnAudioReady"/> only when you truly need managed PCM (utterance ring, WAV write).
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
        try
        {
            using var navigator = _js.Get<Navigator>("navigator");
            using var mediaDevices = navigator.MediaDevices;
            var stream = await mediaDevices.GetUserMedia(video: false, audio: true);
            if (stream == null) return false;
            // The mic stream is OURS, so stopping capture stops its tracks and releases the device.
            return await StartFromAudioStreamAsync(stream, targetSampleRate, maxBufferedFrames, ownsStream: true);
        }
        catch (Exception ex)
        {
            LastAudioError = ex;
            StopMicrophone();
            return false;
        }
    }

    /// <summary>
    /// Capture audio frames from an EXISTING <see cref="MediaStream"/> - a WebRTC remote track, a screen
    /// share, a synthetic test source - instead of opening the microphone.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is what lets the hands-free loop move off this machine. <see cref="StartMicrophoneAsync"/> calls
    /// <c>getUserMedia</c> itself, which is right for a browser demo and useless for a robot: Gemineachy
    /// hears through a WebRTC track arriving from Rose, and there is no microphone on this side of the link
    /// at all. Everything downstream - the frame loop, the native-rate handling, the buffering, the VAD and
    /// the recogniser - is identical, so the loop is written once and fed from either end.
    /// </para>
    /// <para>
    /// ⚠️ <paramref name="ownsStream"/> decides whether <see cref="StopMicrophone"/> STOPS the tracks. For a
    /// stream we opened (the mic) it must, or the device stays hot and the browser keeps showing the
    /// recording indicator. For a stream the CALLER owns - a live WebRTC connection carrying the
    /// conversation - it must not: stopping their track would kill the call to stop listening to it.
    /// </para>
    /// </remarks>
    /// <param name="stream">The stream to read audio frames from. Must carry at least one audio track.</param>
    /// <param name="targetSampleRate">As <see cref="StartMicrophoneAsync"/>; 0 keeps the native rate.</param>
    /// <param name="maxBufferedFrames">As <see cref="StartMicrophoneAsync"/>.</param>
    /// <param name="ownsStream">
    /// True to stop and dispose the stream's tracks on stop. Default FALSE - a caller-supplied stream is
    /// assumed to be owned by the caller, which is the safe default for a shared WebRTC connection.
    /// </param>
    public async Task<bool> StartFromAudioStreamAsync(MediaStream stream, int targetSampleRate = 0,
        int maxBufferedFrames = 3000, bool ownsStream = false)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (_audioProcessor != null) return false;
        if (targetSampleRate < 0) throw new ArgumentOutOfRangeException(nameof(targetSampleRate));

        LastAudioError = null;
        _audioTargetRate = targetSampleRate;
        _ownsAudioStream = ownsStream;
        try
        {
            _audioStream = stream;
            using var tracks = _audioStream.GetAudioTracks();
            var track = tracks.ToArray().FirstOrDefault();
            if (track == null)
            {
                // Report it rather than returning a bare false: "no audio track" and "the browser refused
                // the microphone" are different problems and the caller cannot tell them apart otherwise.
                LastAudioError = new InvalidOperationException(
                    "the MediaStream carries no audio track, so there is nothing to capture");
                StopMicrophone();
                return false;
            }

            _audioProcessor = new MediaStreamTrackProcessor(new MediaStreamTrackProcessorOptions
            {
                Track = track,
                MaxBufferSize = maxBufferedFrames > 0 ? maxBufferedFrames : null,
            });
            using var readable = _audioProcessor.Readable;
            _audioReader = readable.GetReader();

            _loggedFirstAudioFrame = false;
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
            // Only stop tracks we opened. A caller-supplied stream (a live WebRTC track carrying the
            // conversation) belongs to the caller - stopping it here would end their call.
            if (_ownsAudioStream)
            {
                using var tracks = _audioStream.GetAudioTracks();
                tracks.ToArray().UsingEach(t => t.Stop());
                _audioStream.Dispose();
            }
            _audioStream = null;
            _ownsAudioStream = false;
        }
    }

    private bool _loggedFirstAudioFrame;

    private async Task AudioLoop(CancellationToken ct)
    {
        Console.WriteLine("[capture] audio loop started");
        // Nothing may escape this method. An unhandled exception on a runtime callback EXITS the .NET
        // WASM runtime, taking the whole page with it - so a failure is reported through OnAudioError.
        try
        {
            while (!ct.IsCancellationRequested && _audioReader != null)
            {
                ReadableStreamReaderReadResponse res;
                try { res = await _audioReader.Read(); }
                catch (Exception ex)
                {
                    // A BARE `catch { break; }` USED TO BE HERE, and it hid the only fact worth having.
                    // When the reader throws - a track this browser will not attach a processor to, a
                    // reader already locked, a track that ended under us - the loop exited quietly and
                    // the caller saw a capture that had "started" and then delivered nothing at all,
                    // forever, with LastAudioError still null. That is indistinguishable from a silent
                    // room, and it cost a full hardware run to even locate. Report and then stop.
                    LastAudioError = ex;
                    try { OnAudioError?.Invoke(ex); } catch { }
                    break;
                }
                if (res.Done)
                {
                    // Not an error - the track ended - but the consumer still needs to know why the
                    // frames stopped, because from outside it looks the same as the loop dying.
                    res.Dispose();
                    LastAudioError = new InvalidOperationException(
                        "the audio track ended, so there are no more frames to read");
                    break;
                }

                // The chunk of an audio MediaStreamTrackProcessor is an AudioData, not a byte view -
                // read it with the correct wrapper type rather than the reader's byte-typed Value.
                var audioData = res.JSRef!.Get<AudioData?>("value");
                res.Dispose();
                if (audioData is null) continue;

                try
                {
                    // 0 = native: hand over the frame's own rate and do not touch the samples.
                    int rate = _audioTargetRate > 0 ? _audioTargetRate : (int)audioData.SampleRate;
                    bool wantJs = OnAudioReadyJs != null;
                    bool wantManaged = OnAudioReady != null;
                    if (!wantJs && !wantManaged) continue;

                    Float32Array? jsChunk = null;
                    float[]? managed = null;
                    try
                    {
                        // Prefer staying in JS: f32 mono planar at the delivery rate.
                        string fmt = audioData.Format ?? "f32-planar";
                        bool f32 = fmt.StartsWith("f32", StringComparison.Ordinal);
                        bool planar = fmt.EndsWith("-planar", StringComparison.Ordinal);
                        int channels = Math.Max(1, audioData.NumberOfChannels);
                        int srcRate = (int)audioData.SampleRate;
                        bool canStayJs = wantJs && f32 && planar && channels == 1 && srcRate == rate;

                        if (canStayJs)
                        {
                            jsChunk = await MediaInterop.FromAudioDataPlaneJSAsync(audioData, 0);
                            if (wantManaged) managed = jsChunk.ToArray();
                        }
                        else
                        {
                            managed = await MediaInterop.FromAudioDataAsync(audioData, rate);
                            if (wantJs && managed.Length > 0)
                            {
                                jsChunk = new Float32Array(managed.Length);
                                jsChunk.Set(managed);
                            }
                        }

                        if (!_loggedFirstAudioFrame)
                        {
                            _loggedFirstAudioFrame = true;
                            int n = managed?.Length ?? (jsChunk != null ? (int)jsChunk.Length : 0);
                            Console.WriteLine($"[capture] first audio frame: {n} samples @{rate} Hz "
                                + $"(source rate {srcRate} Hz, js={!canStayJs || wantJs})");
                        }

                        if (jsChunk != null && jsChunk.Length > 0) OnAudioReadyJs?.Invoke(jsChunk, rate);
                        if (managed != null && managed.Length > 0) OnAudioReady?.Invoke(managed, rate);
                    }
                    finally
                    {
                        try { jsChunk?.Dispose(); } catch { }
                    }
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
    /// Capture a single frame right now (outside the automatic loop) as managed RGBA bytes.
    /// ⚠️ IN A BROWSER, prefer <see cref="CaptureFrameJs"/> — this path uses <c>ReadBytes</c>.
    /// </summary>
    public byte[]? CaptureFrame()
    {
        if (_video == null) return null;
        return _interop.FromVideoElement(_video, Width, Height);
    }

    /// <summary>
    /// Capture a single frame as a JS typed array (RGBA), without crossing into the .NET heap.
    /// Caller disposes the returned array. Prefer this for anything headed to an accelerator.
    /// </summary>
    public TypedArray? CaptureFrameJs()
    {
        if (_video == null) return null;
        return _interop.FromVideoElementJS(_video, Width, Height);
    }

    /// <summary>
    /// Capture a single frame and preprocess it for a specific model.
    /// Returns a float tensor ready for inference.
    /// </summary>
    /// <remarks>
    /// Still managed: <see cref="MediaInterop.VideoToTensor"/> runs CPU resize/normalize. Frame extract
    /// inside that helper should migrate to <c>*JS</c> + GPU preprocess separately; this overload is
    /// unchanged for callers that already need a host float tensor.
    /// </remarks>
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

                // Time gate only here (null skips motion — we evaluate motion after we have the new frame).
                if (rateController.ShouldRunInference(null))
                {
                    using var rgbaJs = _interop.FromVideoElementJS(_video, Width, Height);

                    bool needManaged = OnFrameReady != null || MotionThreshold > 0;
                    byte[]? rgba = null;
                    if (needManaged)
                    {
                        rgba = rgbaJs.ReadBytes();
                        if (MotionThreshold > 0 && prevFrame != null)
                        {
                            float motion = VideoPreprocessor.ComputeMotionScore(prevFrame, rgba);
                            if (motion < MotionThreshold)
                            {
                                // Still count the attempt so FPS limiting stays honest, but do not deliver.
                                rateController.MarkInferenceRun(rgba);
                                prevFrame = rgba;
                                continue;
                            }
                        }
                    }

                    // Preferred browser delivery — no ReadBytes when only OnFrameReadyJs is subscribed
                    // and motion gating is off.
                    OnFrameReadyJs?.Invoke(rgbaJs, Width, Height);

                    if (rgba != null)
                    {
                        rateController.MarkInferenceRun(rgba);
                        prevFrame = rgba;
                        OnFrameReady?.Invoke(rgba, Width, Height);
                    }
                    else
                    {
                        rateController.MarkInferenceRun(null);
                    }
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
