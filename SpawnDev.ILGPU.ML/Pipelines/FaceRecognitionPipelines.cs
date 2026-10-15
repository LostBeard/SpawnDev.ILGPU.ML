using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>A face found in a picture: box, five landmarks and a confidence, all in the pixels of the picture it was found in.</summary>
/// <param name="X">Left edge of the box.</param>
/// <param name="Y">Top edge of the box.</param>
/// <param name="Width">Box width.</param>
/// <param name="Height">Box height.</param>
/// <param name="Score">Confidence in [0, 1].</param>
/// <param name="Landmarks">Ten floats, x then y for each of: the face's right eye (image left), left eye, nose tip, right mouth corner, left mouth corner.</param>
public sealed record FaceDetection(float X, float Y, float Width, float Height, float Score, float[] Landmarks);

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
/// Pre- and post-processing are on the host: the picture arrives from a camera as bytes, and what leaves the network is 8,400 anchors times 16 numbers (about 540 KB), decoded
/// in well under a millisecond. The convolutions are on the accelerator.
/// </para>
/// </remarks>
public sealed class YuNetFaceDetector : IDisposable
{
    private static readonly int[] Strides = { 8, 16, 32 };
    private readonly Accelerator _accelerator;
    private readonly InferenceSession _session;

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
    }

    /// <summary>Faces in a 32-bit B, G, R, A picture (top row first, no padding), best first.</summary>
    /// <param name="scoreThreshold">Faces below this confidence are dropped (OpenCV's demo uses 0.7 to 0.9).</param>
    /// <param name="nmsThreshold">Overlap above which the weaker of two boxes is dropped.</param>
    public async Task<IReadOnlyList<FaceDetection>> DetectAsync(byte[] bgra, int width, int height, float scoreThreshold = 0.7f, float nmsThreshold = 0.3f, int maxFaces = 20)
    {
        ArgumentNullException.ThrowIfNull(bgra);
        if (bgra.Length != width * height * 4) throw new ArgumentException($"{bgra.Length} bytes is not {width}x{height} BGRA");

        var blob = Letterbox(bgra, width, height, InputWidth, InputHeight, out var scale, out var padX, out var padY);
        using var input = _accelerator.Allocate1D(blob);
        var outs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = new Tensor(input.View, new[] { 1, 3, InputHeight, InputWidth }),
        }).ConfigureAwait(false);
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);

        // Output order: cls_8, cls_16, cls_32, obj_8.., bbox_8.., kps_8.. (twelve tensors).
        var host = new float[12][];
        for (var i = 0; i < 12; i++)
        {
            var t = outs[_session.OutputNames[i]];
            host[i] = await t.Data.SubView(0, t.ElementCount).CopyToHostAsync<float, Stride1D.Dense>().ConfigureAwait(false);
        }
        return Decode(host, scale, padX, padY, scoreThreshold, nmsThreshold, maxFaces);
    }

    /// <summary>Shrinks the picture to fit the network input with its aspect ratio kept (bilinear, like OpenCV's INTER_LINEAR), centres it on black, and returns the planar B, G, R floats (0..255).</summary>
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

    /// <summary>Anchor decoding and non-maximum suppression exactly as OpenCV's FaceDetectorYN does them.</summary>
    internal IReadOnlyList<FaceDetection> Decode(float[][] o, float scale, float padX, float padY, float scoreThreshold, float nmsThreshold, int maxFaces)
    {
        var candidates = new List<FaceDetection>();
        for (var s = 0; s < Strides.Length; s++)
        {
            var stride = Strides[s];
            var cols = InputWidth / stride; var rows = InputHeight / stride;
            var cls = o[s]; var obj = o[3 + s]; var bbox = o[6 + s]; var kps = o[9 + s];
            for (var r = 0; r < rows; r++)
                for (var c = 0; c < cols; c++)
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
                    candidates.Add(new FaceDetection((cx - w / 2 - padX) / scale, (cy - h / 2 - padY) / scale, w / scale, h / scale, score, landmarks));
                }
        }

        var ordered = candidates.OrderByDescending(f => f.Score).ToList();
        var kept = new List<FaceDetection>();
        foreach (var f in ordered)
        {
            if (kept.Count >= maxFaces) break;
            if (kept.All(k => Iou(k, f) <= nmsThreshold)) kept.Add(f);
        }
        return kept;
    }

    private static float Iou(FaceDetection a, FaceDetection b)
    {
        var x1 = Math.Max(a.X, b.X); var y1 = Math.Max(a.Y, b.Y);
        var x2 = Math.Min(a.X + a.Width, b.X + b.Width); var y2 = Math.Min(a.Y + a.Height, b.Y + b.Height);
        var inter = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
        var union = a.Width * a.Height + b.Width * b.Height - inter;
        return union <= 0 ? 0 : inter / union;
    }

    public void Dispose() => _session.Dispose();
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

    /// <param name="session">The SFace session: input <c>data [1, 3, 112, 112]</c>, output <c>fc1 [1, 128]</c>.</param>
    public SFaceRecognizer(InferenceSession session, Accelerator accelerator)
    {
        _session = session; _accelerator = accelerator;
        _session.OutputsLiveUntilNextRun = true;
    }

    /// <summary>The embedding of a face found in a B, G, R, A picture: the face is aligned to the standard template from its landmarks first.</summary>
    public async Task<float[]> EmbedAsync(byte[] bgra, int width, int height, FaceDetection face)
    {
        var aligned = AlignCrop(bgra, width, height, face.Landmarks);
        return await EmbedAlignedAsync(aligned).ConfigureAwait(false);
    }

    /// <summary>The embedding of an already aligned 112 x 112 face, 24-bit B, G, R (what <see cref="AlignCrop"/> returns).</summary>
    public async Task<float[]> EmbedAlignedAsync(byte[] alignedBgr)
    {
        if (alignedBgr.Length != AlignedSize * AlignedSize * 3) throw new ArgumentException("an aligned face is 112 x 112 x 3 bytes");
        // The network wants RGB planes in 0..255; its first operators subtract 127.5 and scale.
        var plane = AlignedSize * AlignedSize;
        var blob = new float[3 * plane];
        for (var i = 0; i < plane; i++)
        {
            blob[i] = alignedBgr[i * 3 + 2];
            blob[plane + i] = alignedBgr[i * 3 + 1];
            blob[2 * plane + i] = alignedBgr[i * 3];
        }
        using var input = _accelerator.Allocate1D(blob);
        var outs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = new Tensor(input.View, new[] { 1, 3, AlignedSize, AlignedSize }),
        }).ConfigureAwait(false);
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        var t = outs[_session.OutputNames[0]];
        return await t.Data.SubView(0, EmbeddingSize).CopyToHostAsync<float, Stride1D.Dense>().ConfigureAwait(false);
    }

    /// <summary>
    /// Warps the face to the 112 x 112 template with the similarity transform (rotation, one scale, translation) that best maps its five landmarks onto the template's, sampling
    /// bilinearly and filling outside the picture with black, as OpenCV's <c>alignCrop</c> does. Returns 24-bit B, G, R.
    /// </summary>
    public static byte[] AlignCrop(byte[] bgra, int width, int height, ReadOnlySpan<float> landmarks)
    {
        if (landmarks.Length < 10) throw new ArgumentException("five landmarks (ten numbers) are needed");
        // The landmarks arrive as right eye (image left), left eye, nose, right mouth, left mouth: the template's order.
        // Least-squares similarity as one complex number z and a translation: dst = z * src + t.
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
        var zr = re / norm; var zi = im / norm;
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

    public void Dispose() => _session.Dispose();
}
