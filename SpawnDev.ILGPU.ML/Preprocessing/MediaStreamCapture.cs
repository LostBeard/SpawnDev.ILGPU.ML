using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;

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
    private readonly BlazorJSRuntime _js;
    private readonly MediaInterop _interop;
    private MediaStream? _stream;
    private HTMLVideoElement? _video;
    private HTMLCanvasElement? _hiddenCanvas;
    private CancellationTokenSource? _captureCts;
    private bool _isCapturing;

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
    /// Fired when audio samples are captured (microphone mode).
    /// Parameters: (float[] samples, int sampleRate)
    /// </summary>
    public event Action<float[], int>? OnAudioReady;

    /// <summary>Target capture FPS. Actual rate may be lower if inference is slow.</summary>
    public float TargetFps { get; set; } = 30;

    /// <summary>Skip frames with motion below this threshold. 0 = never skip.</summary>
    public float MotionThreshold { get; set; }

    public MediaStreamCapture(BlazorJSRuntime js)
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
    /// Stop capturing and release all resources.
    /// </summary>
    public void Stop()
    {
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
