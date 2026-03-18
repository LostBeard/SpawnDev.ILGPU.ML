using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;

namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Efficient interop between browser media types and ML preprocessing.
/// Provides the fastest path from JavaScript image/video/audio sources
/// to float tensors ready for GPU inference — minimizing copies and allocations.
///
/// IMPORTANT: These methods use SpawnDev.BlazorJS typed wrappers.
/// Never go through data URLs (base64 encode/decode) for pixel data —
/// it's 3-4x slower than direct typed array access.
/// </summary>
public class MediaInterop
{
    private readonly BlazorJSRuntime _js;

    // Reusable offscreen canvas to avoid allocation per frame
    private HTMLCanvasElement? _scratchCanvas;
    private CanvasRenderingContext2D? _scratchCtx;
    private int _scratchWidth;
    private int _scratchHeight;

    public MediaInterop(BlazorJSRuntime js)
    {
        _js = js;
    }

    // ──────────────────────────────────────────────
    //  Image Sources → RGBA bytes (fast path)
    // ──────────────────────────────────────────────

    /// <summary>
    /// Extract RGBA pixels from an ImageData object.
    /// This is the fastest path — ImageData already holds raw pixels.
    /// Zero JavaScript overhead beyond the typed array read.
    /// </summary>
    public static byte[] FromImageData(ImageData imageData)
    {
        using var data = imageData.Data;
        return data.ReadBytes();
    }

    /// <summary>
    /// Extract RGBA pixels from an HTMLCanvasElement.
    /// Uses getImageData for direct pixel access (no encoding).
    /// </summary>
    public static byte[] FromCanvas(HTMLCanvasElement canvas)
    {
        using var ctx = canvas.Get2DContext();
        using var imageData = ctx.GetImageData(0, 0, canvas.Width, canvas.Height);
        return FromImageData(imageData);
    }

    /// <summary>
    /// Capture RGBA pixels from an HTMLVideoElement's current frame.
    /// Draws the current frame to an offscreen canvas, then extracts pixels.
    /// Reuses the canvas across calls to avoid allocation churn.
    /// </summary>
    public byte[] FromVideoElement(HTMLVideoElement video, int? targetWidth = null, int? targetHeight = null)
    {
        int w = targetWidth ?? video.VideoWidth;
        int h = targetHeight ?? video.VideoHeight;
        EnsureScratchCanvas(w, h);
        _scratchCtx!.DrawImage(video, 0, 0, w, h);
        using var imageData = _scratchCtx.GetImageData(0, 0, w, h);
        return FromImageData(imageData);
    }

    /// <summary>
    /// Extract RGBA pixels from a VideoFrame (WebCodecs API).
    /// Uses CopyTo for direct buffer access — the most efficient path for video.
    /// Falls back to canvas if CopyTo format isn't RGBA.
    /// </summary>
    public byte[] FromVideoFrame(VideoFrame frame, int? targetWidth = null, int? targetHeight = null)
    {
        int w = targetWidth ?? frame.DisplayWidth;
        int h = targetHeight ?? frame.DisplayHeight;

        // Try direct CopyTo (fastest if format is RGBA/BGRA)
        if (frame.Format == "RGBA" || frame.Format == "BGRA")
        {
            int size = frame.AllocationSize();
            var buffer = new byte[size];
            frame.CopyTo(buffer);

            if (frame.Format == "BGRA")
            {
                // Swap B and R channels in-place
                for (int i = 0; i < buffer.Length; i += 4)
                {
                    (buffer[i], buffer[i + 2]) = (buffer[i + 2], buffer[i]);
                }
            }

            // Resize if needed
            if (w != frame.CodedWidth || h != frame.CodedHeight)
            {
                return ImageOps.Resize(buffer, frame.CodedWidth, frame.CodedHeight, w, h);
            }

            return buffer;
        }

        // Fallback: draw to canvas for format conversion (handles I420, NV12, etc.)
        EnsureScratchCanvas(w, h);
        _scratchCtx!.DrawImage(frame, 0, 0, w, h);
        using var imageData = _scratchCtx.GetImageData(0, 0, w, h);
        return FromImageData(imageData);
    }

    /// <summary>
    /// Extract RGBA pixels from an ImageBitmap.
    /// ImageBitmap has no direct pixel read API — must go through canvas.
    /// </summary>
    public byte[] FromImageBitmap(ImageBitmap bitmap, int? targetWidth = null, int? targetHeight = null)
    {
        int w = targetWidth ?? (int)bitmap.Width;
        int h = targetHeight ?? (int)bitmap.Height;
        EnsureScratchCanvas(w, h);
        _scratchCtx!.DrawImage(bitmap, 0, 0, w, h);
        using var imageData = _scratchCtx.GetImageData(0, 0, w, h);
        return FromImageData(imageData);
    }

    /// <summary>
    /// Load and extract RGBA pixels from a Blob (e.g., uploaded File).
    /// Creates an ImageBitmap from the blob, then extracts pixels.
    /// </summary>
    public async Task<(byte[] Pixels, int Width, int Height)> FromBlobAsync(Blob blob, int? maxWidth = null, int? maxHeight = null)
    {
        using var window = _js.Get<Window>("window");
        using var bitmap = await window.CreateImageBitmap(blob);

        int w = (int)bitmap.Width;
        int h = (int)bitmap.Height;

        // Optionally limit size
        if (maxWidth.HasValue || maxHeight.HasValue)
        {
            float scale = Math.Min(
                (maxWidth ?? w) / (float)w,
                (maxHeight ?? h) / (float)h);
            if (scale < 1)
            {
                w = (int)(w * scale);
                h = (int)(h * scale);
            }
        }

        var pixels = FromImageBitmap(bitmap, w, h);
        return (pixels, w, h);
    }

    /// <summary>
    /// Load and extract RGBA pixels from an HTMLImageElement.
    /// </summary>
    public byte[] FromImageElement(HTMLImageElement image, int? targetWidth = null, int? targetHeight = null)
    {
        int w = targetWidth ?? image.NaturalWidth;
        int h = targetHeight ?? image.NaturalHeight;
        EnsureScratchCanvas(w, h);
        _scratchCtx!.DrawImage(image, 0, 0, w, h);
        using var imageData = _scratchCtx.GetImageData(0, 0, w, h);
        return FromImageData(imageData);
    }

    // ──────────────────────────────────────────────
    //  RGBA bytes → Display (fast path back)
    // ──────────────────────────────────────────────

    /// <summary>
    /// Write RGBA pixels to a canvas element for display.
    /// Uses putImageData for direct pixel write (no encoding overhead).
    /// </summary>
    public static void ToCanvas(byte[] rgba, int width, int height, HTMLCanvasElement canvas)
    {
        canvas.Width = width;
        canvas.Height = height;
        using var ctx = canvas.Get2DContext();
        using var imageData = ImageData.FromBytes(rgba, width, height);
        ctx.PutImageData(imageData, 0, 0);
    }

    /// <summary>
    /// Create an ImageData from RGBA bytes. Useful for canvas putImageData.
    /// </summary>
    public static ImageData ToImageData(byte[] rgba, int width, int height)
    {
        return ImageData.FromBytes(rgba, width, height);
    }

    /// <summary>
    /// Convert RGBA bytes to a data URL string.
    /// AVOID THIS FOR PERFORMANCE-CRITICAL PATHS — base64 encoding adds overhead.
    /// Use ToCanvas + canvas display instead. This is only for cases where
    /// a data URL is strictly required (e.g., img element src).
    /// </summary>
    public string ToDataUrl(byte[] rgba, int width, int height, string mimeType = "image/png")
    {
        EnsureScratchCanvas(width, height);
        using var imageData = ImageData.FromBytes(rgba, width, height);
        _scratchCtx!.PutImageData(imageData, 0, 0);
        return _scratchCanvas!.ToDataURL(mimeType);
    }

    // ──────────────────────────────────────────────
    //  Image Sources → ML Tensor (direct pipeline)
    // ──────────────────────────────────────────────

    /// <summary>
    /// Video element → preprocessed NCHW float tensor in one call.
    /// This is the optimal path for real-time webcam inference.
    /// Reuses internal canvas, extracts pixels, preprocesses — no intermediate allocations.
    /// </summary>
    public float[] VideoToTensor(HTMLVideoElement video, ModelConfig config)
    {
        var rgba = FromVideoElement(video, config.InputWidth, config.InputHeight);
        return config.Preprocess(rgba, config.InputWidth, config.InputHeight);
    }

    /// <summary>
    /// VideoFrame → preprocessed NCHW float tensor in one call.
    /// Optimal for WebCodecs-based video processing.
    /// </summary>
    public float[] VideoFrameToTensor(VideoFrame frame, ModelConfig config)
    {
        var rgba = FromVideoFrame(frame, config.InputWidth, config.InputHeight);
        return config.Preprocess(rgba, config.InputWidth, config.InputHeight);
    }

    /// <summary>
    /// Canvas → preprocessed NCHW float tensor in one call.
    /// </summary>
    public static float[] CanvasToTensor(HTMLCanvasElement canvas, ModelConfig config)
    {
        var rgba = FromCanvas(canvas);
        return config.Preprocess(rgba, canvas.Width, canvas.Height);
    }

    // ──────────────────────────────────────────────
    //  Audio Sources → Float samples
    // ──────────────────────────────────────────────

    /// <summary>
    /// Extract audio samples from an AudioBuffer as mono float array.
    /// If multi-channel, averages all channels to mono.
    /// Returns samples in [-1, 1] range at the buffer's sample rate.
    /// </summary>
    public static float[] FromAudioBuffer(AudioBuffer buffer)
    {
        int channels = buffer.NumberOfChannels;
        int length = (int)buffer.Length;

        if (channels == 1)
        {
            using var channelData = buffer.GetChannelData(0);
            return channelData.ToArray();
        }

        // Mix down to mono
        var mono = new float[length];
        for (int ch = 0; ch < channels; ch++)
        {
            using var channelData = buffer.GetChannelData(ch);
            var samples = channelData.ToArray();
            for (int i = 0; i < length; i++)
            {
                mono[i] += samples[i];
            }
        }

        float scale = 1f / channels;
        for (int i = 0; i < length; i++)
        {
            mono[i] *= scale;
        }

        return mono;
    }

    /// <summary>
    /// Extract audio samples from an AudioBuffer and resample to target rate.
    /// Convenience method for Whisper (16kHz) and other models.
    /// </summary>
    public static float[] FromAudioBuffer(AudioBuffer buffer, int targetSampleRate)
    {
        var mono = FromAudioBuffer(buffer);
        int srcRate = (int)buffer.SampleRate;
        return AudioPreprocessor.Resample(mono, srcRate, targetSampleRate);
    }

    // ──────────────────────────────────────────────
    //  Internal: scratch canvas management
    // ──────────────────────────────────────────────

    private void EnsureScratchCanvas(int width, int height)
    {
        if (_scratchCanvas != null && _scratchWidth == width && _scratchHeight == height)
            return;

        _scratchCtx?.Dispose();
        _scratchCanvas?.Dispose();

        _scratchCanvas = new HTMLCanvasElement();
        _scratchCanvas.Width = width;
        _scratchCanvas.Height = height;
        _scratchCtx = _scratchCanvas.Get2DContext();
        _scratchWidth = width;
        _scratchHeight = height;
    }

    /// <summary>
    /// Dispose the scratch canvas and context.
    /// Call this when the MediaInterop instance is no longer needed.
    /// </summary>
    public void Dispose()
    {
        _scratchCtx?.Dispose();
        _scratchCanvas?.Dispose();
        _scratchCtx = null;
        _scratchCanvas = null;
    }
}
