using ILGPU;
using ILGPU.Runtime;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Interop between browser media types and ML preprocessing.
///
/// 🔴 READ THIS BEFORE USING THE <c>byte[]</c> / <c>float[]</c> METHODS. They pull pixels onto the .NET
/// WASM managed heap. That is the right thing ONLY when the data genuinely has to enter .NET. It is the
/// WRONG thing when the destination is an accelerator — WebGPU, WebGL and Wasm all take JS typed arrays
/// directly, so a frame bound for inference should never touch managed memory at all.
///
/// This file used to claim the opposite. <see cref="FromImageData"/> was documented as *"the fastest path —
/// zero JavaScript overhead beyond the typed array read"*, but the typed array read IS the cost: it copies
/// the whole frame across the boundary. The chain a classified webcam frame actually took was
///
///     canvas → Uint8Array → ReadBytes() → byte[] → repack to int[] → Allocate1D → GPU
///
/// — three full-size copies of a 3.69 MB frame, single-threaded on the WASM heap, per frame. The library
/// one layer down already ships the answer and says so in its own summary:
/// <c>IBrowserMemoryBuffer.CopyFromJS(TypedArray, long)</c> — *"Copies data from a JS TypedArray directly
/// into the GPU buffer without crossing into .NET managed memory. This is the zero-copy path for browser
/// backends."* Nothing here was using it.
///
/// ⭐ So: use the <c>...JS</c> methods below for anything feeding an accelerator, and
/// <see cref="UploadToDevice{T}"/> to land it on the GPU. Keep the <c>byte[]</c> overloads for the cases
/// that really do need managed data (saving a file, a CPU-side codec, a unit-test oracle).
///
/// IMPORTANT: These methods use SpawnDev.SpawnJS typed wrappers.
/// Never go through data URLs (base64 encode/decode) for pixel data —
/// it's 3-4x slower than direct typed array access.
/// </summary>
public class MediaInterop
{
    private readonly SpawnJSRuntime _js;

    // Reusable offscreen canvas to avoid allocation per frame
    private HTMLCanvasElement? _scratchCanvas;
    private CanvasRenderingContext2D? _scratchCtx;
    private int _scratchWidth;
    private int _scratchHeight;

    public MediaInterop(SpawnJSRuntime js)
    {
        _js = js;
    }

    // ──────────────────────────────────────────────
    //  Image Sources → GPU, WITHOUT touching the .NET heap  ⭐ PREFER THESE FOR INFERENCE
    // ──────────────────────────────────────────────
    //
    // Each returns the pixels as a JS typed array that has never entered managed memory. Hand it to
    // UploadToDevice (or straight to IBrowserMemoryBuffer.CopyFromJS) and the frame goes JS → GPU with
    // no copy through .NET at all. The GPU then does the resize, the HWC→NCHW transpose and the
    // normalization via ImagePreprocessKernel — all work that ModelConfig.Preprocess was doing on the
    // single-threaded WASM CPU.
    //
    // ⚠️ OWNERSHIP: the returned array is a live JS handle and SpawnJS slot lifetime is MANUAL — nothing
    // collects it. Dispose it (a `using`) or you leak a slot per frame, which at 30 fps is not subtle.

    /// <summary>RGBA pixels of an <see cref="ImageData"/>, left in JS. The caller disposes the result.</summary>
    public static Uint8ClampedArray FromImageDataJS(ImageData imageData) => imageData.Data;

    /// <summary>RGBA pixels of a canvas, left in JS. The caller disposes the result.</summary>
    public static Uint8ClampedArray FromCanvasJS(HTMLCanvasElement canvas)
    {
        using var ctx = canvas.Get2DContext();
        using var imageData = ctx.GetImageData(0, 0, canvas.Width, canvas.Height);
        return FromImageDataJS(imageData);
    }

    /// <summary>Current video frame as RGBA, left in JS. The caller disposes the result.</summary>
    public Uint8ClampedArray FromVideoElementJS(HTMLVideoElement video, int? targetWidth = null, int? targetHeight = null)
    {
        int w = targetWidth ?? video.VideoWidth;
        int h = targetHeight ?? video.VideoHeight;
        EnsureScratchCanvas(w, h);
        _scratchCtx!.DrawImage(video, 0, 0, w, h);
        using var imageData = _scratchCtx.GetImageData(0, 0, w, h);
        return FromImageDataJS(imageData);
    }

    /// <summary>An <see cref="ImageBitmap"/> as RGBA, left in JS. The caller disposes the result.</summary>
    public Uint8ClampedArray FromImageBitmapJS(ImageBitmap bitmap, int? targetWidth = null, int? targetHeight = null)
    {
        int w = targetWidth ?? (int)bitmap.Width;
        int h = targetHeight ?? (int)bitmap.Height;
        EnsureScratchCanvas(w, h);
        _scratchCtx!.DrawImage(bitmap, 0, 0, w, h);
        using var imageData = _scratchCtx.GetImageData(0, 0, w, h);
        return FromImageDataJS(imageData);
    }

    /// <summary>An <see cref="HTMLImageElement"/> as RGBA, left in JS. The caller disposes the result.</summary>
    public Uint8ClampedArray FromImageElementJS(HTMLImageElement image, int? targetWidth = null, int? targetHeight = null)
    {
        int w = targetWidth ?? image.NaturalWidth;
        int h = targetHeight ?? image.NaturalHeight;
        EnsureScratchCanvas(w, h);
        _scratchCtx!.DrawImage(image, 0, 0, w, h);
        using var imageData = _scratchCtx.GetImageData(0, 0, w, h);
        return FromImageDataJS(imageData);
    }

    /// <summary>
    /// Uploads a JS typed array straight into a device buffer — the whole point of the <c>...JS</c> methods
    /// above. Works for any element type because the copy is bytes: RGBA pixels into an <c>int</c> buffer
    /// (4 bytes per pixel is exactly the RGBA layout, so no repack is needed or performed), PCM samples into
    /// a <c>float</c> buffer, and so on. <paramref name="destination"/> must be large enough to hold them.
    /// </summary>
    /// <remarks>
    /// ⚠️ ONE method for every element type ON PURPOSE. A per-type pair (UploadRgba.../UploadSamples...)
    /// would be two mechanisms doing one job, which is how a fix lands on one of them and not the other.
    /// </remarks>
    /// <exception cref="NotSupportedException">
    /// The buffer is not a browser buffer. On CPU/CUDA/OpenCL there is no JS heap to copy from, and a caller
    /// reaching here has a desktop accelerator with browser data — a wiring mistake worth failing loudly
    /// rather than silently falling back to a managed copy, which is the very thing this path exists to
    /// avoid.
    /// </exception>
    public static void UploadToDevice<T>(TypedArray source, MemoryBuffer1D<T, Stride1D.Dense> destination)
        where T : unmanaged
    {
        // ⚠️ It is the UNDERLYING MemoryBuffer that implements IBrowserMemoryBuffer, not the
        // MemoryBuffer1D<T,TStride> view wrapper around it. Testing the wrapper compiles, is always false,
        // and degrades silently to "unsupported" on the very backends this exists for. BufferPool's
        // zero-copy weight path already had this right (`buffer.Buffer is IBrowserMemoryBuffer`).
        if (destination.Buffer is not IBrowserMemoryBuffer browserBuffer)
            throw new NotSupportedException(
                $"UploadToDevice needs a browser memory buffer; got {destination.Buffer.GetType().Name} " +
                $"on {destination.Accelerator.AcceleratorType}. On a desktop accelerator use the managed path.");
        browserBuffer.CopyFromJS(source);
    }

    // ──────────────────────────────────────────────
    //  Image Sources → RGBA bytes IN MANAGED MEMORY
    // ──────────────────────────────────────────────
    //
    // ⚠️ These cross the .NET boundary. Correct when the data must enter .NET; WRONG when it is headed for
    // an accelerator — use the ...JS methods above for that. See the type summary.

    /// <summary>
    /// Copy RGBA pixels from an ImageData object INTO MANAGED MEMORY.
    /// ⚠️ <c>ReadBytes()</c> copies the whole frame onto the .NET WASM heap. For anything bound for a GPU
    /// use <see cref="FromImageDataJS"/> + <see cref="UploadToDevice{T}"/> instead.
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
    public async Task<byte[]> FromVideoFrameAsync(VideoFrame frame, int? targetWidth = null, int? targetHeight = null)
    {
        int w = targetWidth ?? frame.DisplayWidth;
        int h = targetHeight ?? frame.DisplayHeight;

        // Try direct CopyTo (fastest if format is RGBA/BGRA)
        if (frame.Format == "RGBA" || frame.Format == "BGRA")
        {
            int size = frame.AllocationSize();
            var buffer = new byte[size];
            await frame.CopyTo(buffer); // CopyTo is async (returns Promise per spec)

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
    public async Task<float[]> VideoFrameToTensorAsync(VideoFrame frame, ModelConfig config)
    {
        var rgba = await FromVideoFrameAsync(frame, config.InputWidth, config.InputHeight);
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
    //  Audio Sources → GPU, WITHOUT touching the .NET heap  ⭐ PREFER THESE
    // ──────────────────────────────────────────────
    //
    // 🔴 SIZE IS NOT AN EXCUSE. A 10 ms WebCodecs AudioData frame is ~480 samples - far past the
    // ~64-element metadata exemption - and the accelerators take JS typed arrays directly, so there is no
    // reason for a sample to enter managed memory on the way to the GPU. TJ, 2026-09-15: *"it always
    // matters when taking the less performant path for no reason."* The chain a VAD frame took was
    //
    //     AudioData -> Float32Array -> .ToArray() -> float[] -> (downmix/resample in .NET) -> CopyFromCPU -> GPU
    //
    // i.e. out of JS and back to the GPU, once per frame, at ~30 frames/second per turn.

    /// <summary>
    /// One plane of a WebCodecs <see cref="AudioData"/> as a JS <see cref="Float32Array"/> that never enters
    /// managed memory. Hand it to <see cref="UploadToDevice{T}"/>. The caller disposes the result.
    /// </summary>
    /// <remarks>
    /// f32 formats only, and deliberately so: an s16 source needs a /32768 scale per sample, and doing that
    /// in .NET would reintroduce exactly the crossing this method exists to remove. That conversion belongs
    /// on the GPU; until it is written, s16 callers keep <see cref="FromAudioDataAsync"/> and this method
    /// says why rather than silently handing back wrong values.
    /// </remarks>
    public static async Task<Float32Array> FromAudioDataPlaneJSAsync(AudioData audioData, int planeIndex = 0)
    {
        string fmt = audioData.Format ?? "f32-planar";
        if (!fmt.StartsWith("f32", StringComparison.Ordinal))
            throw new NotSupportedException(
                $"FromAudioDataPlaneJSAsync handles f32 formats; this frame is '{fmt}'. An s16 frame needs a "
              + "per-sample scale that belongs on the GPU - use FromAudioDataAsync until that kernel exists.");

        int frames = audioData.NumberOfFrames;
        int channels = Math.Max(1, audioData.NumberOfChannels);
        bool planar = fmt.EndsWith("-planar", StringComparison.Ordinal);
        int count = planar ? frames : frames * channels;

        var dest = new Float32Array(count);
        await audioData.CopyTo(dest, new AudioDataCopyToOptions { PlaneIndex = planeIndex });
        return dest;   // stays in JS
    }

    // ──────────────────────────────────────────────
    //  Audio Sources → Float samples IN MANAGED MEMORY
    // ──────────────────────────────────────────────
    //
    // ⚠️ These cross the .NET boundary. Correct when the samples must enter .NET (a CPU codec, a WAV write,
    // a test oracle); WRONG when they are headed for an accelerator - see above.

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

    /// <summary>
    /// Convert one WebCodecs <c>AudioData</c> frame to MONO float32 at <paramref name="targetSampleRate"/>.
    /// Handles planar and interleaved layouts in f32 or s16, downmixes to mono, and resamples.
    /// This is the extraction half of microphone capture - see <c>MediaStreamCapture.StartMicrophoneAsync</c>.
    /// </summary>
    public static async Task<float[]> FromAudioDataAsync(AudioData audioData, int targetSampleRate)
    {
        int frames = audioData.NumberOfFrames;
        int channels = Math.Max(1, audioData.NumberOfChannels);
        int rate = (int)audioData.SampleRate;
        string fmt = audioData.Format ?? "f32-planar";
        bool planar = fmt.EndsWith("-planar", StringComparison.Ordinal);
        if (frames <= 0) return System.Array.Empty<float>();

        var mono = new float[frames];
        if (planar)
        {
            // One plane per channel - sum them.
            for (int c = 0; c < channels; c++)
            {
                var plane = await CopyPlaneAsFloatsAsync(audioData, fmt, frames, c);
                for (int i = 0; i < frames && i < plane.Length; i++) mono[i] += plane[i];
            }
        }
        else
        {
            // One interleaved plane holds every channel.
            var all = await CopyPlaneAsFloatsAsync(audioData, fmt, frames * channels, 0);
            for (int i = 0; i < frames; i++)
            {
                float sum = 0f;
                for (int c = 0; c < channels; c++)
                {
                    int idx = i * channels + c;
                    if (idx < all.Length) sum += all[idx];
                }
                mono[i] = sum;
            }
        }

        if (channels > 1)
        {
            float scale = 1f / channels;
            for (int i = 0; i < frames; i++) mono[i] *= scale;
        }

        return rate == targetSampleRate ? mono : AudioPreprocessor.Resample(mono, rate, targetSampleRate);
    }

    private static async Task<float[]> CopyPlaneAsFloatsAsync(AudioData audioData, string fmt, int count, int planeIndex)
    {
        var options = new AudioDataCopyToOptions { PlaneIndex = planeIndex };
        if (fmt.StartsWith("f32", StringComparison.Ordinal))
        {
            using var dest = new Float32Array(count);
            await audioData.CopyTo(dest, options);
            return dest.ToArray();
        }
        if (fmt.StartsWith("s16", StringComparison.Ordinal))
        {
            using var dest = new Int16Array(count);
            await audioData.CopyTo(dest, options);
            var s = dest.ToArray();
            var f = new float[s.Length];
            for (int i = 0; i < s.Length; i++) f[i] = s[i] / 32768f;
            return f;
        }
        // Fail loudly rather than emit silence or garbage for a format we have not implemented.
        throw new NotSupportedException(
            $"AudioData format '{fmt}' is not supported yet. Expected an f32* or s16* format.");
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
