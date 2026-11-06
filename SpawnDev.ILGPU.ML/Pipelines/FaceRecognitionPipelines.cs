using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.ILGPU.ML.Pipelines;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>A face found in a picture: box, five landmarks and a confidence, all in the pixels of the picture it was found in.</summary>
/// <param name="X">Left edge of the box.</param>
/// <param name="Y">Top edge of the box.</param>
/// <param name="Width">Box width.</param>
/// <param name="Height">Box height.</param>
/// <param name="Score">Confidence in [0, 1].</param>
/// <param name="Landmarks">Ten floats, x then y for each of: the face's right eye (image left), left eye, nose tip, right mouth corner, left mouth corner.</param>
public sealed record FaceDetection(float X, float Y, float Width, float Height, float Score, float[] Landmarks);

/// <summary>Byte order of a 32-bit pixel, first byte in memory first.</summary>
public enum PixelOrder
{
    /// <summary>B, G, R, A (Windows bitmaps, OpenCV, most camera APIs).</summary>
    Bgra,
    /// <summary>R, G, B, A (canvas ImageData, VideoFrame "RGBA").</summary>
    Rgba,
}

/// <summary>
/// Finds faces in a picture: YuNet (<c>opencv/opencv_zoo</c> <c>face_detection_yunet</c>, MIT; a 75,000-parameter anchor-free detector that finds faces from about 10 px up
/// and gives five landmarks), the detector OpenCV's own <c>FaceDetectorYN</c> runs.
/// </summary>
/// <remarks>
/// <para>
/// The network runs at a FIXED size (default 640 x 640, the only size the published <c>2023mar</c> export supports; the dynamic-shape <c>2026may</c> export takes any multiple of 32, e.g. 640 x 384 for a 16:9 camera): the picture is shrunk to fit with its aspect kept and the rest padded with black, and the
/// boxes come back in the picture's own pixels. Create the session with <see cref="InputShapes"/> for the size used here. Input is BGR, 0..255, no normalisation (the
/// network has none).
/// </para>
/// <para>
/// Everything up to the faces runs on the accelerator: the picture (managed bytes, a JS typed array, or a view already on the device) is letterboxed into the network's
/// input by a kernel, and the anchors are decoded and thresholded there, so what crosses back is the count and the faces above the threshold (one read), not the
/// 8,400 anchors. Overlap suppression of those few runs on the host. One call at a time per detector (its device buffers are reused).
/// </para>
/// </remarks>
public sealed class YuNetFaceDetector : IDisposable
{
    internal static readonly int[] Strides = { 8, 16, 32 };
    private readonly Accelerator _accelerator;
    private readonly InferenceSession _session;
    private readonly FacePixels _pixels;
    private readonly FaceStages _stages;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _input;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _found;

    /// <summary>Width of the network input (a multiple of 32).</summary>
    public int InputWidth { get; }

    /// <summary>Height of the network input (a multiple of 32).</summary>
    public int InputHeight { get; }

    /// <summary>The shapes to create the YuNet session with for a given input size.</summary>
    public static Dictionary<string, int[]> InputShapes(int width = 640, int height = 640) => new() { ["input"] = new[] { 1, 3, height, width } };

    /// <param name="session">The YuNet session (<c>face_detection_yunet_2023mar.onnx</c>, fixed at 640 x 640, or the dynamic-shape <c>2026may</c> export at any multiple of 32), created with <see cref="InputShapes"/>.</param>
    public YuNetFaceDetector(InferenceSession session, Accelerator accelerator, int inputWidth = 640, int inputHeight = 640)
    {
        if (inputWidth % 32 != 0 || inputHeight % 32 != 0) throw new ArgumentException("the input size must be a multiple of 32");
        _session = session; _accelerator = accelerator; InputWidth = inputWidth; InputHeight = inputHeight;
        _session.OutputsLiveUntilNextRun = true;
        _pixels = new FacePixels(accelerator);
        _stages = new FaceStages(accelerator);
        _input = accelerator.Allocate1D<float>(3L * inputWidth * inputHeight);
        _found = FaceStages.AllocateFound(accelerator, inputWidth, inputHeight);
    }

    /// <summary>Faces in a 32-bit B, G, R, A picture (top row first, no padding), best first. Desktop / managed path: the bytes are uploaded once.</summary>
    /// <param name="scoreThreshold">Faces below this confidence are dropped (OpenCV's demo uses 0.7 to 0.9).</param>
    /// <param name="nmsThreshold">Overlap above which the weaker of two boxes is dropped.</param>
    public Task<IReadOnlyList<FaceDetection>> DetectAsync(byte[] bgra, int width, int height, float scoreThreshold = 0.7f, float nmsThreshold = 0.3f, int maxFaces = 20) =>
        DetectAsync(_pixels.FromManaged(bgra, width, height), width, height, PixelOrder.Bgra, scoreThreshold, nmsThreshold, maxFaces);

    /// <summary>Faces in a 32-bit picture held in a JS typed array (a camera frame, canvas ImageData): its bytes go straight to the device, never through .NET.</summary>
    public Task<IReadOnlyList<FaceDetection>> DetectAsync(TypedArray pixels, int width, int height, PixelOrder order, float scoreThreshold = 0.7f, float nmsThreshold = 0.3f, int maxFaces = 20) =>
        DetectAsync(_pixels.FromTypedArray(pixels, width, height), width, height, order, scoreThreshold, nmsThreshold, maxFaces);

    /// <summary>Faces in a picture already on the device: one packed 32-bit pixel per element (<paramref name="order"/>), top row first. Every overload ends here.</summary>
    public async Task<IReadOnlyList<FaceDetection>> DetectAsync(ArrayView1D<int, Stride1D.Dense> pixels, int width, int height, PixelOrder order, float scoreThreshold = 0.7f, float nmsThreshold = 0.3f, int maxFaces = 20)
    {
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException(nameof(width), "a picture has pixels");
        if (pixels.Length < (long)width * height) throw new ArgumentException($"{pixels.Length} pixels is not {width}x{height}");
        var (scale, padX, padY) = _stages.Letterbox(pixels, width, height, order, _input.View, InputWidth, InputHeight);
        var outs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = new Tensor(_input.View, new[] { 1, 3, InputHeight, InputWidth }),
        }).ConfigureAwait(false);
        // Output order: cls_8, cls_16, cls_32, obj_8.., bbox_8.., kps_8.. (twelve tensors).
        var o = new FV[12];
        for (var i = 0; i < 12; i++) o[i] = outs[_session.OutputNames[i]].Data;
        var raw = await _stages.DecodeAsync(o, InputWidth, InputHeight, _found, scale, padX, padY, scoreThreshold).ConfigureAwait(false);
        _session.ReturnOutputs(outs);   // pool-rented outputs: returned once their last reader is queued (unreturned, every call leaked one set; found 2026-11-06 with YOLO's per-frame leak)
        return Suppress(raw, nmsThreshold, maxFaces);
    }

    // best first (ties: the earlier anchor), then the weaker of two boxes overlapping more than nmsThreshold dropped, as OpenCV's FaceDetectorYN does
    internal static IReadOnlyList<FaceDetection> Suppress(List<(FaceDetection Face, int Anchor)> candidates, float nmsThreshold, int maxFaces)
    {
        var ordered = candidates.Select(c => c.Face).OrderByDescending(f => f.Score).ToList();
        var kept = new List<FaceDetection>();
        foreach (var f in ordered)
        {
            if (kept.Count >= maxFaces) break;
            if (kept.All(k => Iou(k, f) <= nmsThreshold)) kept.Add(f);
        }
        return kept;
    }

    // ---- host reference versions (the gates' oracles) ----

    /// <summary>Host reference of the letterbox kernel: shrinks the picture to fit the network input with its aspect ratio kept (bilinear, like OpenCV's INTER_LINEAR), centres it on black, and returns the planar B, G, R floats (0..255).</summary>
    internal static float[] Letterbox(byte[] bgra, int width, int height, int inW, int inH, out float scale, out float padX, out float padY)
    {
        scale = Math.Min((float)inW / width, (float)inH / height);
        var newW = Math.Max(1, (int)Math.Round(width * scale));
        var newH = Math.Max(1, (int)Math.Round(height * scale));
        padX = (inW - newW) / 2;
        padY = (inH - newH) / 2;
        var blob = new float[3 * inW * inH];
        var plane = inW * inH;
        var sx = (float)width / newW; var sy = (float)height / newH;
        for (var y = 0; y < newH; y++)
        {
            var fy = Math.Clamp((y + 0.5f) * sy - 0.5f, 0, height - 1);
            var y0 = (int)fy; var y1 = Math.Min(y0 + 1, height - 1); var wy = fy - y0;
            for (var x = 0; x < newW; x++)
            {
                var fx = Math.Clamp((x + 0.5f) * sx - 0.5f, 0, width - 1);
                var x0 = (int)fx; var x1 = Math.Min(x0 + 1, width - 1); var wx = fx - x0;
                var dst = ((int)padY + y) * inW + (int)padX + x;
                for (var c = 0; c < 3; c++)   // B, G, R are bytes 0, 1, 2 of a BGRA pixel
                {
                    var p00 = bgra[(y0 * width + x0) * 4 + c]; var p01 = bgra[(y0 * width + x1) * 4 + c];
                    var p10 = bgra[(y1 * width + x0) * 4 + c]; var p11 = bgra[(y1 * width + x1) * 4 + c];
                    blob[c * plane + dst] = (p00 * (1 - wx) + p01 * wx) * (1 - wy) + (p10 * (1 - wx) + p11 * wx) * wy;
                }
            }
        }
        return blob;
    }

    /// <summary>Host reference of the anchor decode: anchor decoding and non-maximum suppression exactly as OpenCV's FaceDetectorYN does them.</summary>
    internal static IReadOnlyList<FaceDetection> Decode(float[][] o, int inW, int inH, float scale, float padX, float padY, float scoreThreshold, float nmsThreshold, int maxFaces)
    {
        var candidates = new List<(FaceDetection, int)>();
        int anchor = 0;
        for (var s = 0; s < Strides.Length; s++)
        {
            var stride = Strides[s];
            var cols = inW / stride; var rows = inH / stride;
            var cls = o[s]; var obj = o[3 + s]; var bbox = o[6 + s]; var kps = o[9 + s];
            for (var r = 0; r < rows; r++)
                for (var c = 0; c < cols; c++, anchor++)
                {
                    var idx = r * cols + c;
                    var score = MathF.Sqrt(Math.Clamp(cls[idx], 0f, 1f) * Math.Clamp(obj[idx], 0f, 1f));
                    if (score < scoreThreshold) continue;
                    var cx = (c + bbox[idx * 4]) * stride; var cy = (r + bbox[idx * 4 + 1]) * stride;
                    var w = MathF.Exp(bbox[idx * 4 + 2]) * stride; var h = MathF.Exp(bbox[idx * 4 + 3]) * stride;
                    var landmarks = new float[10];
                    for (var n = 0; n < 5; n++)
                    {
                        landmarks[2 * n] = ((kps[idx * 10 + 2 * n] + c) * stride - padX) / scale;
                        landmarks[2 * n + 1] = ((kps[idx * 10 + 2 * n + 1] + r) * stride - padY) / scale;
                    }
                    candidates.Add((new FaceDetection((cx - w / 2 - padX) / scale, (cy - h / 2 - padY) / scale, w / scale, h / scale, score, landmarks), anchor));
                }
        }
        return Suppress(candidates, nmsThreshold, maxFaces);
    }

    private static float Iou(FaceDetection a, FaceDetection b)
    {
        var x1 = Math.Max(a.X, b.X); var y1 = Math.Max(a.Y, b.Y);
        var x2 = Math.Min(a.X + a.Width, b.X + b.Width); var y2 = Math.Min(a.Y + a.Height, b.Y + b.Height);
        var inter = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
        var union = a.Width * a.Height + b.Width * b.Height - inter;
        return union <= 0 ? 0 : inter / union;
    }

    public void Dispose()
    {
        _session.Dispose();
        _pixels.Dispose();
        _input.Dispose();
        _found.Dispose();
    }
}

/// <summary>
/// Who a face is: a 128-number embedding of an aligned face from SFace (<c>opencv/opencv_zoo</c> <c>face_recognition_sface</c>, Apache-2.0; a MobileFaceNet trained with the SFace
/// loss, the recogniser OpenCV's <c>FaceRecognizerSF</c> runs). Two faces are the same person when the cosine similarity of their embeddings is high.
/// </summary>
/// <remarks>
/// <para>
/// OpenCV's recommended cosine threshold is 0.363. MEASURED on six official portraits (two people photographed years apart, two single-photo distractors): the same person
/// scored 0.755 and 0.599, different people at most 0.318 - the household's own photos will differ (lighting, a camera across a room), so calibrate on them.
/// </para>
/// <para>
/// The alignment (OpenCV's <c>alignCrop</c>: the similarity transform from the five landmarks to the template, a bilinear sample, black outside) is a kernel that writes
/// the network's input; only the 128 numbers come back. One call at a time per recogniser (its device buffers are reused).
/// </para>
/// <para>
/// An embedding is a label made from how a face looks, not proof of who someone is. It is not private data to be shared: it stays on this machine, and a person can ask for it to be deleted.
/// </para>
/// </remarks>
public sealed class SFaceRecognizer : IDisposable
{
    /// <summary>Side of the aligned face the network takes.</summary>
    public const int AlignedSize = 112;

    /// <summary>Length of an embedding.</summary>
    public const int EmbeddingSize = 128;

    /// <summary>OpenCV's recommended cosine similarity for "same person".</summary>
    public const float OpenCvCosineThreshold = 0.363f;

    // The ArcFace 112 x 112 template the five landmarks are warped onto: left eye, right eye, nose tip, left mouth corner, right mouth corner (image order).
    private static readonly float[] Template =
    {
        38.2946f, 51.6963f,
        73.5318f, 51.5014f,
        56.0252f, 71.7366f,
        41.5493f, 92.3655f,
        70.7299f, 92.2041f,
    };

    private readonly Accelerator _accelerator;
    private readonly InferenceSession _session;
    private readonly FacePixels _pixels;
    private readonly FaceStages _stages;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _input;

    /// <param name="session">The SFace session: input <c>data [1, 3, 112, 112]</c>, output <c>fc1 [1, 128]</c>.</param>
    public SFaceRecognizer(InferenceSession session, Accelerator accelerator)
    {
        _session = session; _accelerator = accelerator;
        _session.OutputsLiveUntilNextRun = true;
        _pixels = new FacePixels(accelerator);
        _stages = new FaceStages(accelerator);
        _input = accelerator.Allocate1D<float>(3 * AlignedSize * AlignedSize);
    }

    /// <summary>The embedding of a face found in a B, G, R, A picture: the face is aligned to the standard template from its landmarks first. Desktop / managed path.</summary>
    public Task<float[]> EmbedAsync(byte[] bgra, int width, int height, FaceDetection face) =>
        EmbedAsync(_pixels.FromManaged(bgra, width, height), width, height, face, PixelOrder.Bgra);

    /// <summary>The embedding of a face in a 32-bit picture held in a JS typed array: its bytes go straight to the device, never through .NET.</summary>
    public Task<float[]> EmbedAsync(TypedArray pixels, int width, int height, FaceDetection face, PixelOrder order) =>
        EmbedAsync(_pixels.FromTypedArray(pixels, width, height), width, height, face, order);

    /// <summary>The embedding of a face in a picture already on the device (one packed 32-bit pixel per element). Every overload ends here.</summary>
    public async Task<float[]> EmbedAsync(ArrayView1D<int, Stride1D.Dense> pixels, int width, int height, FaceDetection face, PixelOrder order)
    {
        ArgumentNullException.ThrowIfNull(face);
        if (pixels.Length < (long)width * height) throw new ArgumentException($"{pixels.Length} pixels is not {width}x{height}");
        _stages.Align(pixels, width, height, face.Landmarks, order, _input.View);
        return await RunAsync().ConfigureAwait(false);
    }

    /// <summary>The embedding of an already aligned 112 x 112 face, 24-bit B, G, R (what <see cref="AlignCrop"/> returns, or OpenCV's <c>alignCrop</c>).</summary>
    public async Task<float[]> EmbedAlignedAsync(byte[] alignedBgr)
    {
        if (alignedBgr.Length != AlignedSize * AlignedSize * 3) throw new ArgumentException("an aligned face is 112 x 112 x 3 bytes");
        // 112 x 112 x 3 bytes is a whole number of 32-bit words: the kernel picks the bytes out of them
        _stages.AlignedBgr(_pixels.FromManagedWords(alignedBgr), _input.View);
        return await RunAsync().ConfigureAwait(false);
    }

    private async Task<float[]> RunAsync()
    {
        var outs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = new Tensor(_input.View, new[] { 1, 3, AlignedSize, AlignedSize }),
        }).ConfigureAwait(false);
        var t = outs[_session.OutputNames[0]];
        // the read is ordered after the network (no SynchronizeAsync: that would be a second round trip)
        var embedding = await t.Data.SubView(0, EmbeddingSize).CopyToHostAsync<float, Stride1D.Dense>().ConfigureAwait(false);
        _session.ReturnOutputs(outs);   // pool-rented outputs: returned once their last reader is queued (unreturned, every call leaked one set; found 2026-11-06 with YOLO's per-frame leak)
        return embedding;
    }

    // Least-squares similarity from the five landmarks to the template as one complex number z and the two means: dst = z * (src - srcMean) + dstMean.
    // The landmarks arrive as right eye (image left), left eye, nose, right mouth, left mouth: the template's order.
    internal static (double Zr, double Zi, double Sx, double Sy, double Dx, double Dy) Similarity(ReadOnlySpan<float> landmarks)
    {
        if (landmarks.Length < 10) throw new ArgumentException("five landmarks (ten numbers) are needed");
        double sx = 0, sy = 0, dx = 0, dy = 0;
        for (var i = 0; i < 5; i++) { sx += landmarks[2 * i]; sy += landmarks[2 * i + 1]; dx += Template[2 * i]; dy += Template[2 * i + 1]; }
        sx /= 5; sy /= 5; dx /= 5; dy /= 5;
        double re = 0, im = 0, norm = 0;
        for (var i = 0; i < 5; i++)
        {
            double ax = landmarks[2 * i] - sx, ay = landmarks[2 * i + 1] - sy;
            double bx = Template[2 * i] - dx, by = Template[2 * i + 1] - dy;
            re += ax * bx + ay * by;       // conj(a) * b
            im += ax * by - ay * bx;
            norm += ax * ax + ay * ay;
        }
        return (re / norm, im / norm, sx, sy, dx, dy);
    }

    /// <summary>
    /// Host reference of the alignment kernel: warps the face to the 112 x 112 template with the similarity transform (rotation, one scale, translation) that best maps its five
    /// landmarks onto the template's, sampling bilinearly and filling outside the picture with black, as OpenCV's <c>alignCrop</c> does. Returns 24-bit B, G, R.
    /// </summary>
    public static byte[] AlignCrop(byte[] bgra, int width, int height, ReadOnlySpan<float> landmarks)
    {
        var (zr, zi, sx, sy, dx, dy) = Similarity(landmarks);
        // Forward map: dst = z * (src - srcMean) + dstMean. Invert it to sample the source for each destination pixel.
        var zn = zr * zr + zi * zi;
        var crop = new byte[AlignedSize * AlignedSize * 3];
        for (var v = 0; v < AlignedSize; v++)
            for (var u = 0; u < AlignedSize; u++)
            {
                double px = u - dx, py = v - dy;
                var srcX = (px * zr + py * zi) / zn + sx;      // multiply by conj(z) / |z|^2
                var srcY = (py * zr - px * zi) / zn + sy;
                var o = (v * AlignedSize + u) * 3;
                Sample(bgra, width, height, srcX, srcY, crop, o);
            }
        return crop;
    }

    private static void Sample(byte[] bgra, int width, int height, double x, double y, byte[] dst, int o)
    {
        // Bilinear with the constant-black border OpenCV's warpAffine uses (a neighbour outside the picture counts as 0).
        var x0 = (int)Math.Floor(x); var y0 = (int)Math.Floor(y);
        var fx = x - x0; var fy = y - y0;
        for (var c = 0; c < 3; c++)
        {
            double P(int px, int py) => px < 0 || py < 0 || px >= width || py >= height ? 0 : bgra[(py * width + px) * 4 + c];
            var value = (P(x0, y0) * (1 - fx) + P(x0 + 1, y0) * fx) * (1 - fy) + (P(x0, y0 + 1) * (1 - fx) + P(x0 + 1, y0 + 1) * fx) * fy;
            dst[o + c] = (byte)Math.Clamp((int)Math.Round(value), 0, 255);
        }
    }

    /// <summary>Cosine similarity of two embeddings.</summary>
    public static float Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length) throw new ArgumentException("embeddings differ in length");
        double dot = 0, na = 0, nb = 0;
        for (var i = 0; i < a.Length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return na == 0 || nb == 0 ? 0f : (float)(dot / Math.Sqrt(na * nb));
    }

    public void Dispose()
    {
        _session.Dispose();
        _pixels.Dispose();
        _input.Dispose();
    }
}

/// <summary>
/// The face pipelines' picture on the device: one reused buffer of packed 32-bit pixels, filled from managed bytes (desktop) or from a JS typed array (the bytes go
/// device to device in the browser, never through .NET: <see cref="MediaInterop.UploadToDevice{T}"/>). It grows to the largest picture seen.
/// </summary>
internal sealed class FacePixels : IDisposable
{
    private readonly Accelerator _accelerator;
    private MemoryBuffer1D<int, Stride1D.Dense>? _buffer;

    public FacePixels(Accelerator accelerator) => _accelerator = accelerator;

    private MemoryBuffer1D<int, Stride1D.Dense> Ensure(long words)
    {
        // a replaced buffer is disposed only after a full drain: every call awaits its read before returning, so no queued kernel still reads it
        if (_buffer == null || _buffer.Length < words) { _buffer?.Dispose(); _buffer = _accelerator.Allocate1D<int>(Math.Max(words, 64)); }
        return _buffer;
    }

    public IV FromManaged(byte[] pixels, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(pixels);
        long n = (long)width * height;
        if (width <= 0 || height <= 0 || pixels.Length != n * 4) throw new ArgumentException($"{pixels.Length} bytes is not {width}x{height} at 4 bytes a pixel");
        return FromManagedWords(pixels);
    }

    // bytes whose count is a multiple of 4, as 32-bit words
    public IV FromManagedWords(byte[] bytes)
    {
        if (bytes.Length % 4 != 0) throw new ArgumentException("a whole number of 32-bit words");
        var words = MemoryMarshal.Cast<byte, int>(bytes.AsSpan());
        var view = Ensure(words.Length).View.SubView(0, words.Length);
        view.CopyFromCPU(ref MemoryMarshal.GetReference(words), words.Length);
        return view;
    }

    public IV FromTypedArray(TypedArray pixels, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(pixels);
        long n = (long)width * height;
        if (width <= 0 || height <= 0 || pixels.ByteLength != n * 4) throw new ArgumentException($"{pixels.ByteLength} bytes is not {width}x{height} at 4 bytes a pixel");
        var buffer = Ensure(n);
        MediaInterop.UploadToDevice(pixels, buffer);
        return buffer.View.SubView(0, n);
    }

    public void Dispose() { _buffer?.Dispose(); _buffer = null; }
}

/// <summary>
/// The face pipelines' device stages on one accelerator (their kernels, loaded on first use): the letterbox, the anchor decode, the alignment, the aligned-face
/// conversion. Internal so the kernel gates can hold each to its host reference without a model.
/// </summary>
internal sealed class FaceStages
{
    // faces read back with the count in the first read; more than this above the threshold take a second read
    public const int FirstRead = 256;
    public const int Fields = 16;      // per face: score, anchor index, x, y, w, h, ten landmark numbers
    private readonly Accelerator _accelerator;
    private Action<Index1D, IV, FV, int, int, int, int, int, int, int, float, float, int>? _letterbox;
    private Action<Index1D, FV, FV, FV, FV, IV, IV, int, int, int, float, float, float, float, int>? _decode;
    private Action<Index1D, FV, FV, FV, FV, IV, int, int, int, float, float, float>? _decodeDense;
    private Action<Index1D, IV, IV>? _countToFound;
    private MemoryBuffer1D<int, Stride1D.Dense>? _count;
    private Action<Index1D, IV, FV, int, int, float, float, float, float, float, float, float, int>? _align;
    private Action<Index1D, IV, FV>? _aligned;

    public FaceStages(Accelerator accelerator) { _accelerator = accelerator; Dense = accelerator.AcceleratorType == AcceleratorType.WebGL; }

    /// <summary>The decode writes every anchor instead of appending the ones above the threshold (WebGL, which cannot hand out slots); settable for the gate.</summary>
    public bool Dense { get; set; }

    public static int Anchors(int inW, int inH) => YuNetFaceDetector.Strides.Sum(s => inW / s * (inH / s));

    // the decode's output: the count, then room for every anchor (a threshold of 0 keeps them all)
    public static MemoryBuffer1D<int, Stride1D.Dense> AllocateFound(Accelerator accelerator, int inW, int inH) =>
        accelerator.Allocate1D<int>(1 + (long)Anchors(inW, inH) * Fields);

    // the picture letterboxed into `blob` (planar B, G, R floats, 0..255, black around) at inW x inH; returns the scale and padding that map boxes back
    public (float Scale, float PadX, float PadY) Letterbox(IV pixels, int width, int height, PixelOrder order, FV blob, int inW, int inH)
    {
        _letterbox ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, IV, FV, int, int, int, int, int, int, int, float, float, int>(FaceKernels.LetterboxImpl);
        var scale = Math.Min((float)inW / width, (float)inH / height);
        var newW = Math.Max(1, (int)Math.Round(width * scale));
        var newH = Math.Max(1, (int)Math.Round(height * scale));
        int padX = (inW - newW) / 2, padY = (inH - newH) / 2;
        _letterbox(inW * inH, pixels, blob, width, height, inW, newW, newH, padX, padY, (float)width / newW, (float)height / newH, FaceKernels.BlueShift(order));
        return (scale, padX, padY);
    }

    // every anchor above the threshold, decoded on the device and read back with the count (one read unless more than FirstRead pass); in anchor order.
    // o: the twelve outputs, cls_8, cls_16, cls_32, obj_8.., bbox_8.., kps_8..
    public async Task<List<(FaceDetection Face, int Anchor)>> DecodeAsync(FV[] o, int inW, int inH, MemoryBuffer1D<int, Stride1D.Dense> found, float scale, float padX, float padY, float scoreThreshold)
    {
        int anchors = Anchors(inW, inH);
        if (found.Length < 1 + (long)anchors * Fields) throw new ArgumentException("the found buffer is too small for this input size");
        // WebGL cannot hand out slots (its Atomic.Add returns 0) and captures one element per store site per thread: every field of every anchor is written
        // by its own thread at the anchor's own index, and filtered here (DecodeDenseImpl)
        bool dense = Dense;
        if (dense) _decodeDense ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, IV, int, int, int, float, float, float>(FaceKernels.DecodeDenseImpl);
        else
        {
            _decode ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, IV, IV, int, int, int, float, float, float, float, int>(FaceKernels.DecodeImpl);
            _countToFound ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(FaceKernels.CountToFoundImpl);
            _count ??= _accelerator.Allocate1D<int>(64);
            _count.View.SubView(0, 1).MemSetToZero();
        }
        int anchorBase = 0;
        for (var s = 0; s < YuNetFaceDetector.Strides.Length; s++)
        {
            int stride = YuNetFaceDetector.Strides[s], cols = inW / stride, n = cols * (inH / stride);
            if (dense) _decodeDense!(n * Fields, o[s], o[3 + s], o[6 + s], o[9 + s], found.View, cols, stride, anchorBase, scale, padX, padY);
            else _decode!(n, o[s], o[3 + s], o[6 + s], o[9 + s], found.View, _count!.View, cols, stride, anchorBase, scale, padX, padY, scoreThreshold, anchors);
            anchorBase += n;
        }
        // the count into found[0], so one read brings the count and the first faces
        if (!dense) _countToFound!(1, _count!.View, found.View);
        // the read is ordered after the kernels (no SynchronizeAsync: that would be a second round trip)
        int count;
        int[] rows;
        if (dense)
        {
            rows = await found.View.SubView(0, 1 + (long)anchors * Fields).CopyToHostAsync().ConfigureAwait(false);
            count = anchors;
        }
        else
        {
            rows = await found.View.SubView(0, 1 + (long)Math.Min(FirstRead, anchors) * Fields).CopyToHostAsync().ConfigureAwait(false);
            count = Math.Min(rows[0], anchors);
            if (count > FirstRead) rows = await found.View.SubView(0, 1 + (long)count * Fields).CopyToHostAsync().ConfigureAwait(false);
        }
        var list = new List<(FaceDetection, int)>(dense ? 16 : count);
        for (var k = 0; k < count; k++)
        {
            int b = 1 + k * Fields;
            float F(int j) => BitConverter.Int32BitsToSingle(rows[b + j]);
            if (dense && !(F(0) >= scoreThreshold)) continue;
            var lm = new float[10];
            for (var j = 0; j < 10; j++) lm[j] = F(6 + j);
            list.Add((new FaceDetection(F(2), F(3), F(4), F(5), F(0), lm), rows[b + 1]));
        }
        // the device appended them in any order: back to anchor order, so the stable sort by score after breaks ties as the host decode does
        list.Sort((a, c) => a.Item2.CompareTo(c.Item2));
        return list;
    }

    // the aligned face (OpenCV's alignCrop, rounded to bytes as its 8-bit output is) written into `blob` as the network's R, G, B planes (0..255)
    public void Align(IV pixels, int width, int height, ReadOnlySpan<float> landmarks, PixelOrder order, FV blob)
    {
        _align ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, IV, FV, int, int, float, float, float, float, float, float, float, int>(FaceKernels.AlignImpl);
        var (zr, zi, sx, sy, dx, dy) = SFaceRecognizer.Similarity(landmarks);
        _align(SFaceRecognizer.AlignedSize * SFaceRecognizer.AlignedSize, pixels, blob, width, height, (float)zr, (float)zi, (float)(zr * zr + zi * zi), (float)sx, (float)sy, (float)dx, (float)dy, FaceKernels.BlueShift(order));
    }

    // an aligned 24-bit B, G, R face (its bytes as 32-bit words) as the network's R, G, B planes
    public void AlignedBgr(IV words, FV blob)
    {
        _aligned ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, IV, FV>(FaceKernels.AlignedBgrImpl);
        _aligned(SFaceRecognizer.AlignedSize * SFaceRecognizer.AlignedSize, words, blob);
    }
}

/// <summary>The face pipelines' kernels.</summary>
internal static class FaceKernels
{
    // where blue sits in a packed pixel (byte 0 of BGRA, byte 2 of RGBA); red is the other end, green always byte 1
    public static int BlueShift(PixelOrder order) => order == PixelOrder.Bgra ? 0 : 16;

    private static float Channel(int p, int shift) => (p >> shift) & 255;

    // the letterbox: output pixel i of the inW-wide input; inside the shrunk picture a bilinear sample (OpenCV INTER_LINEAR's half-pixel centres), black around;
    // planar B, G, R. Arithmetic as YuNetFaceDetector.Letterbox (host)
    public static void LetterboxImpl(Index1D i, IV px, FV blob, int width, int height, int inW, int newW, int newH, int padX, int padY, float sx, float sy, int bShift)
    {
        int ii = i, plane = (int)(blob.Length / 3), x = ii % inW, y = ii / inW, xx = x - padX, yy = y - padY;
        float b = 0f, g = 0f, r = 0f;
        bool inside = (xx >= 0) & (xx < newW) & (yy >= 0) & (yy < newH);
        if (inside)
        {
            float fy = MathF.Min(MathF.Max((yy + 0.5f) * sy - 0.5f, 0f), height - 1);
            int y0 = (int)fy, y1 = Math.Min(y0 + 1, height - 1);
            float wy = fy - y0;
            float fx = MathF.Min(MathF.Max((xx + 0.5f) * sx - 0.5f, 0f), width - 1);
            int x0 = (int)fx, x1 = Math.Min(x0 + 1, width - 1);
            float wx = fx - x0;
            int p00 = px[y0 * width + x0], p01 = px[y0 * width + x1], p10 = px[y1 * width + x0], p11 = px[y1 * width + x1];
            int rShift = 16 - bShift;
            b = (Channel(p00, bShift) * (1 - wx) + Channel(p01, bShift) * wx) * (1 - wy) + (Channel(p10, bShift) * (1 - wx) + Channel(p11, bShift) * wx) * wy;
            g = (Channel(p00, 8) * (1 - wx) + Channel(p01, 8) * wx) * (1 - wy) + (Channel(p10, 8) * (1 - wx) + Channel(p11, 8) * wx) * wy;
            r = (Channel(p00, rShift) * (1 - wx) + Channel(p01, rShift) * wx) * (1 - wy) + (Channel(p10, rShift) * (1 - wx) + Channel(p11, rShift) * wx) * wy;
        }
        blob[ii] = b; blob[plane + ii] = g; blob[2 * plane + ii] = r;
    }

    // a pixel's channel at (x, y), 0 outside the picture (OpenCV warpAffine's constant-black border)
    private static float At(IV px, int width, int height, int x, int y, int shift)
    {
        bool outside = (x < 0) | (y < 0) | (x >= width) | (y >= height);
        return outside ? 0f : Channel(px[y * width + x], shift);
    }

    private static float Round255(float v) => MathF.Min(MathF.Max(MathF.Round(v), 0f), 255f);

    // the alignment: output pixel i of the 112 x 112 crop samples the picture where the inverse similarity puts it, bilinear, rounded to a byte as alignCrop's
    // 8-bit output is, written as the network's R, G, B planes. Arithmetic as SFaceRecognizer.AlignCrop (host; there in double)
    public static void AlignImpl(Index1D i, IV px, FV blob, int width, int height, float zr, float zi, float zn, float smx, float smy, float dmx, float dmy, int bShift)
    {
        const int N = SFaceRecognizer.AlignedSize;
        int ii = i, u = ii % N, v = ii / N, plane = N * N, rShift = 16 - bShift;
        float qx = u - dmx, qy = v - dmy;
        float srcX = (qx * zr + qy * zi) / zn + smx, srcY = (qy * zr - qx * zi) / zn + smy;
        float fx0 = MathF.Floor(srcX), fy0 = MathF.Floor(srcY);
        int x0 = (int)fx0, y0 = (int)fy0;
        float fx = srcX - fx0, fy = srcY - fy0;
        float r = (At(px, width, height, x0, y0, rShift) * (1 - fx) + At(px, width, height, x0 + 1, y0, rShift) * fx) * (1 - fy)
                + (At(px, width, height, x0, y0 + 1, rShift) * (1 - fx) + At(px, width, height, x0 + 1, y0 + 1, rShift) * fx) * fy;
        float g = (At(px, width, height, x0, y0, 8) * (1 - fx) + At(px, width, height, x0 + 1, y0, 8) * fx) * (1 - fy)
                + (At(px, width, height, x0, y0 + 1, 8) * (1 - fx) + At(px, width, height, x0 + 1, y0 + 1, 8) * fx) * fy;
        float b = (At(px, width, height, x0, y0, bShift) * (1 - fx) + At(px, width, height, x0 + 1, y0, bShift) * fx) * (1 - fy)
                + (At(px, width, height, x0, y0 + 1, bShift) * (1 - fx) + At(px, width, height, x0 + 1, y0 + 1, bShift) * fx) * fy;
        blob[ii] = Round255(r); blob[plane + ii] = Round255(g); blob[2 * plane + ii] = Round255(b);
    }

    // an aligned 24-bit B, G, R face (bytes packed in 32-bit words) as the network's R, G, B planes
    public static void AlignedBgrImpl(Index1D i, IV words, FV blob)
    {
        const int N = SFaceRecognizer.AlignedSize;
        int ii = i, plane = N * N;
        for (int c = 0; c < 3; c++)
        {
            int at = ii * 3 + c;
            float v = (words[at >> 2] >> ((at & 3) * 8)) & 255;
            blob[(2 - c) * plane + ii] = v;                 // byte 0 is blue: the last plane
        }
    }

    // one anchor of one stride: its score (sqrt of class x objectness, both clamped to [0, 1]); above the threshold, a slot from the counter (its own buffer:
    // found then takes only plain stores, and an atomic-typed buffer would make every one of them atomic) and its 16 words in found (found[1 + 16 slot ..]:
    // score, anchor index, x, y, w, h, ten landmark numbers, the floats as their bits) in the picture's pixels. Arithmetic as YuNetFaceDetector.Decode (host).
    // The stores are written out, not looped: one element per store site.
    public static void DecodeImpl(Index1D i, FV cls, FV obj, FV bbox, FV kps, IV found, IV count, int cols, int stride, int anchorBase, float scale, float padX, float padY, float threshold, int cap)
    {
        int idx = i, r = idx / cols, c = idx % cols;
        float score = MathF.Sqrt(MathF.Min(MathF.Max(cls[idx], 0f), 1f) * MathF.Min(MathF.Max(obj[idx], 0f), 1f));
        if (score < threshold) return;
        int slot = Atomic.Add(ref count[0], 1);
        if (slot >= cap) return;
        long o = 1 + (long)slot * 16;
        float cx = (c + bbox[idx * 4]) * stride, cy = (r + bbox[idx * 4 + 1]) * stride;
        float w = MathF.Exp(bbox[idx * 4 + 2]) * stride, h = MathF.Exp(bbox[idx * 4 + 3]) * stride;
        int k = idx * 10;
        found[o] = (int)Interop.FloatAsInt(score);
        found[o + 1] = anchorBase + idx;
        found[o + 2] = (int)Interop.FloatAsInt((cx - w / 2 - padX) / scale);
        found[o + 3] = (int)Interop.FloatAsInt((cy - h / 2 - padY) / scale);
        found[o + 4] = (int)Interop.FloatAsInt(w / scale);
        found[o + 5] = (int)Interop.FloatAsInt(h / scale);
        found[o + 6] = (int)Interop.FloatAsInt(((kps[k] + c) * stride - padX) / scale);
        found[o + 7] = (int)Interop.FloatAsInt(((kps[k + 1] + r) * stride - padY) / scale);
        found[o + 8] = (int)Interop.FloatAsInt(((kps[k + 2] + c) * stride - padX) / scale);
        found[o + 9] = (int)Interop.FloatAsInt(((kps[k + 3] + r) * stride - padY) / scale);
        found[o + 10] = (int)Interop.FloatAsInt(((kps[k + 4] + c) * stride - padX) / scale);
        found[o + 11] = (int)Interop.FloatAsInt(((kps[k + 5] + r) * stride - padY) / scale);
        found[o + 12] = (int)Interop.FloatAsInt(((kps[k + 6] + c) * stride - padX) / scale);
        found[o + 13] = (int)Interop.FloatAsInt(((kps[k + 7] + r) * stride - padY) / scale);
        found[o + 14] = (int)Interop.FloatAsInt(((kps[k + 8] + c) * stride - padX) / scale);
        found[o + 15] = (int)Interop.FloatAsInt(((kps[k + 9] + r) * stride - padY) / scale);
    }

    // the counter into found[0] (one thread), so the host reads the count and the faces in one read
    public static void CountToFoundImpl(Index1D i, IV count, IV found) => found[0] = count[0];

    // dense (WebGL: no slots from atomics, one element per store site per thread): thread i = (anchor, field) writes that field of every anchor of the stride at
    // the anchor's own index, whatever its score; the host keeps those at or above the threshold. The same arithmetic as DecodeImpl, one store.
    public static void DecodeDenseImpl(Index1D i, FV cls, FV obj, FV bbox, FV kps, IV found, int cols, int stride, int anchorBase, float scale, float padX, float padY)
    {
        int ii = i, idx = ii / 16, f = ii % 16, r = idx / cols, c = idx % cols;
        float score = MathF.Sqrt(MathF.Min(MathF.Max(cls[idx], 0f), 1f) * MathF.Min(MathF.Max(obj[idx], 0f), 1f));
        float cx = (c + bbox[idx * 4]) * stride, cy = (r + bbox[idx * 4 + 1]) * stride;
        float w = MathF.Exp(bbox[idx * 4 + 2]) * stride, h = MathF.Exp(bbox[idx * 4 + 3]) * stride;
        int lmIndex = Math.Max(f - 6, 0);                                  // fields 6.. are the landmarks: x of point n at 6 + 2n, y at 7 + 2n
        bool isY = (lmIndex & 1) == 1;
        float lm = ((kps[idx * 10 + lmIndex] + (isY ? r : c)) * stride - (isY ? padY : padX)) / scale;
        float v = f == 0 ? score : f == 2 ? (cx - w / 2 - padX) / scale : f == 3 ? (cy - h / 2 - padY) / scale : f == 4 ? w / scale : f == 5 ? h / scale : lm;
        int bits = f == 1 ? anchorBase + idx : (int)Interop.FloatAsInt(v);
        found[1 + (long)(anchorBase + idx) * 16 + f] = bits;
    }
}
