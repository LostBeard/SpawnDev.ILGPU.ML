using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;
using TypedArray = SpawnDev.SpawnJS.JSObjects.TypedArray;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>One image's DINOv2 embedding (<see cref="Dinov2EmbeddingPipeline"/>).</summary>
public sealed class Dinov2Embedding
{
    /// <summary>
    /// The retrieval descriptor, length 2 x <see cref="Dim"/>: L2norm(concat(L2norm(CLS), L2norm(mean of the patch tokens))). The class token plus
    /// average-pooled patch tokens is the DINOv2 paper's linear-evaluation feature; each part is normalised so neither dominates, and the whole is
    /// normalised so a dot product is the cosine similarity.
    /// </summary>
    public required float[] Pooled { get; init; }
    /// <summary>The model's last_hidden_state [1 + PatchesPerSide^2, Dim] (CLS first, then patches row by row), or null unless requested.</summary>
    public float[]? Tokens { get; init; }
    /// <summary>Hidden width: 384 (small), 768 (base).</summary>
    public int Dim { get; init; }
    /// <summary>Patch tokens per side: input size / 14 (16 at 224).</summary>
    public int PatchesPerSide { get; init; }
}

/// <summary>
/// Image embeddings from DINOv2 (facebook/dinov2-small / -base, Apache-2.0; the ONNX exports <see cref="SmallRepo"/> / <see cref="BaseRepo"/>):
/// an RGBA frame in, a pooled L2-normalised descriptor out (<see cref="Dinov2Embedding.Pooled"/>), the per-patch tokens on request. Built for place
/// retrieval (AssistZ): embed a frame, take the nearest entries of an index by dot product.
/// </summary>
/// <remarks>
/// Preprocessing follows the checkpoints' Hugging Face processor (BitImageProcessor): resize the SHORT edge to 256 with PIL's antialiased bicubic
/// filter, center-crop 224, /255, ImageNet mean and std. It runs on the GPU in one kernel (no intermediate image). PIL rounds its intermediate
/// image to 8 bits between the two axis passes and this kernel does not, so pixels differ slightly from the reference processor; the gate
/// measures what that does to the descriptor. Other input sizes: a multiple of 14, with the resize kept at the same 256 / 224 ratio.
/// </remarks>
public sealed class Dinov2EmbeddingPipeline : IDisposable
{
    /// <summary>DINOv2 ViT-S/14 (22M parameters, 384 wide), ONNX export.</summary>
    public const string SmallRepo = "Xenova/dinov2-small";
    /// <summary>DINOv2 ViT-B/14 (86M parameters, 768 wide), ONNX export.</summary>
    public const string BaseRepo = "Xenova/dinov2-base";
    /// <summary>The model file inside either repository.</summary>
    public const string ModelFile = "onnx/model.onnx";

    private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] Std = { 0.229f, 0.224f, 0.225f };

    private readonly Accelerator _acc;
    private readonly InferenceSession _session;
    private readonly bool _ownsSession;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _pixels;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _pooled;
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, PrepParams>? _prep;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _pool;
    private bool _disposed;

    /// <summary>Model input side (224 by default).</summary>
    public int InputSize { get; }
    /// <summary>Hidden width of the loaded checkpoint (384 small, 768 base), from the compiled graph's output shape; the pooled descriptor has 2 x Dim values.</summary>
    public int Dim { get; private set; }

    /// <summary>Wrap a session already loaded with input shape [1, 3, <paramref name="inputSize"/>, <paramref name="inputSize"/>].</summary>
    public Dinov2EmbeddingPipeline(Accelerator accelerator, InferenceSession session, int inputSize = 224, bool ownsSession = true)
    {
        if (inputSize % 14 != 0) throw new ArgumentException("the input size must be a multiple of the patch size 14", nameof(inputSize));
        _acc = accelerator; _session = session; _ownsSession = ownsSession; InputSize = inputSize;
        _pixels = accelerator.Allocate1D<float>(3L * inputSize * inputSize);
        _pooled = accelerator.Allocate1D<float>(2 * 1536);   // up to ViT-L width x 2
        // known at load (Riker 2026-11-04: Dim read 0 until the first embed)
        if (session.OutputShapes.TryGetValue(session.OutputNames[0], out var os) && os.Length > 0 && os[^1] > 0) Dim = os[^1];
    }

    /// <summary>Load <see cref="ModelFile"/> of <paramref name="repoId"/> from a model source (the models host / OPFS in the browser).</summary>
    public static async Task<Dinov2EmbeddingPipeline> CreateFromHubAsync(Accelerator accelerator, Hub.IModelSource source, string repoId = BaseRepo,
        int inputSize = 224, Action<string, int>? onProgress = null, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        await using var stream = await source.OpenAsync(repoId, ModelFile, ct).ConfigureAwait(false);
        var session = await InferenceSession.CreateFromStreamAsync(accelerator, stream, onProgress,
            inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 3, inputSize, inputSize } }, ct: ct).ConfigureAwait(false);
        return new Dinov2EmbeddingPipeline(accelerator, session, inputSize);
    }

    /// <summary>Embed packed RGBA pixels held on the host (uploaded once).</summary>
    public async Task<Dinov2Embedding> EmbedAsync(int[] rgbaPixels, int width, int height, bool includeTokens = false)
    {
        using var buf = RgbaUpload.FromManaged(_acc, rgbaPixels, width, height);
        return await EmbedAsync(buf.View, width, height, includeTokens).ConfigureAwait(false);
    }

    /// <summary>Browser path: RGBA still in a JS typed array; the pixels never enter the .NET heap.</summary>
    public async Task<Dinov2Embedding> EmbedAsync(TypedArray rgbaPixels, int width, int height, bool includeTokens = false)
    {
        using var buf = RgbaUpload.FromTypedArray(_acc, rgbaPixels, width, height);
        return await EmbedAsync(buf.View, width, height, includeTokens).ConfigureAwait(false);
    }

    /// <summary>Embed packed RGBA (R in the low byte) already on the accelerator.</summary>
    public Task<Dinov2Embedding> EmbedAsync(ArrayView1D<int, Stride1D.Dense> rgba, int width, int height, bool includeTokens = false)
    {
        Preprocess(rgba, width, height, _pixels.View);
        return EmbedPixelsAsync(_pixels.View, includeTokens);
    }

    /// <summary>
    /// The model input for an RGBA image: [3, InputSize, InputSize], short edge resized to InputSize x 256 / 224 (antialiased bicubic), center
    /// crop, ImageNet normalisation. Exposed so the preprocessing can be checked on its own.
    /// </summary>
    public void Preprocess(ArrayView1D<int, Stride1D.Dense> rgba, int width, int height, ArrayView1D<float, Stride1D.Dense> pixels)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (width <= 0 || height <= 0 || rgba.Length < (long)width * height) throw new ArgumentException("the RGBA view is smaller than width x height");
        // Hugging Face: shortest_edge = 256 for crop 224; other sizes keep the ratio
        int shortEdge = (int)Math.Round(InputSize * 256.0 / 224.0);
        int rw, rh;
        if (width <= height) { rw = shortEdge; rh = (int)((long)shortEdge * height / width); }
        else { rh = shortEdge; rw = (int)((long)shortEdge * width / height); }
        var p = new PrepParams
        {
            SrcW = width, SrcH = height, Size = InputSize,
            Left = (rw - InputSize) / 2, Top = (rh - InputSize) / 2,
            ScaleX = (float)width / rw, ScaleY = (float)height / rh,
            M0 = Mean[0], M1 = Mean[1], M2 = Mean[2], I0 = 1f / Std[0], I1 = 1f / Std[1], I2 = 1f / Std[2],
        };
        _prep ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, PrepParams>(PrepImpl);
        _prep(3 * InputSize * InputSize, rgba, pixels, p);
    }

    /// <summary>Embed an already preprocessed input [3, InputSize, InputSize] (see <see cref="Preprocess"/>).</summary>
    public async Task<Dinov2Embedding> EmbedPixelsAsync(ArrayView1D<float, Stride1D.Dense> pixels, bool includeTokens = false)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var outs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = new Tensor(pixels, new[] { 1, 3, InputSize, InputSize })
        }).ConfigureAwait(false);
        var t = outs[_session.OutputNames[0]];
        int side = InputSize / 14, n = 1 + side * side;
        int dim = t.Shape[^1];
        if (t.ElementCount != (long)n * dim) throw new InvalidOperationException($"unexpected output shape [{string.Join(",", t.Shape)}] for input {InputSize}");
        if (2 * dim > _pooled.Length) throw new NotSupportedException($"hidden width {dim} is wider than ViT-L");
        Dim = dim;
        _pool ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(PoolImpl);
        _pool(2 * dim, t.Data, _pooled.View, dim, n - 1);
        // the readback is the fence: tokens first (same queue), then the pooled parts
        float[]? tokens = includeTokens ? await t.Data.SubView(0, (long)n * dim).CopyToHostAsync().ConfigureAwait(false) : null;
        var raw = await _pooled.View.SubView(0, 2 * dim).CopyToHostAsync().ConfigureAwait(false);
        _session.ReturnOutputs(outs);   // pool-rented outputs: returned once consumed (unreturned, every call leaked one set; found 2026-11-06 with YOLO's per-frame leak)
        return new Dinov2Embedding { Pooled = PoolNormalize(raw, dim), Tokens = tokens, Dim = dim, PatchesPerSide = side };
    }

    /// <summary>L2norm(concat(L2norm(cls), L2norm(mean))) of the raw [cls | mean] pair (host math in double).</summary>
    public static float[] PoolNormalize(float[] clsThenMean, int dim)
    {
        double a = 0, b = 0;
        for (int i = 0; i < dim; i++) { a += (double)clsThenMean[i] * clsThenMean[i]; b += (double)clsThenMean[dim + i] * clsThenMean[dim + i]; }
        double sa = 1.0 / Math.Max(Math.Sqrt(a), 1e-12), sb = 1.0 / Math.Max(Math.Sqrt(b), 1e-12);
        // each half then has norm 1, so the concatenation has norm sqrt(2)
        double s = 1.0 / Math.Sqrt(2.0);
        var o = new float[2 * dim];
        for (int i = 0; i < dim; i++) { o[i] = (float)(clsThenMean[i] * sa * s); o[dim + i] = (float)(clsThenMean[dim + i] * sb * s); }
        return o;
    }

    /// <summary>Preprocess scalars.</summary>
    public struct PrepParams
    {
        /// <summary>Source size.</summary>
        public int SrcW, SrcH;
        /// <summary>Output side.</summary>
        public int Size;
        /// <summary>Crop origin in resized pixels.</summary>
        public int Left, Top;
        /// <summary>Source pixels per resized pixel, per axis.</summary>
        public float ScaleX, ScaleY;
        /// <summary>Mean per channel.</summary>
        public float M0, M1, M2;
        /// <summary>1 / std per channel.</summary>
        public float I0, I1, I2;
    }

    // PIL's bicubic kernel (a = -0.5)
    private static float Cubic(float x)
    {
        const float a = -0.5f;
        x = MathF.Abs(x);
        if (x < 1f) return ((a + 2f) * x - (a + 3f)) * x * x + 1f;
        if (x < 2f) return (((x - 5f) * x + 8f) * x - 4f) * a;
        return 0f;
    }

    // One thread per output value (c, y, x): PIL's antialiased bicubic resample (support 2 x max(scale, 1), weights normalised per axis) of the
    // resized pixel (Left + x, Top + y), as one 2-D pass. RGBA packed with R in the low byte.
    private static void PrepImpl(Index1D idx, ArrayView1D<int, Stride1D.Dense> rgba, ArrayView1D<float, Stride1D.Dense> o, PrepParams p)
    {
        int hw = p.Size * p.Size;
        int c = idx / hw, rem = idx - c * hw;
        int oy = rem / p.Size, ox = rem - oy * p.Size;
        float ssx = p.ScaleX > 1f ? p.ScaleX : 1f, ssy = p.ScaleY > 1f ? p.ScaleY : 1f;
        float cx = (p.Left + ox + 0.5f) * p.ScaleX, cy = (p.Top + oy + 0.5f) * p.ScaleY;
        int x0 = (int)(cx - 2f * ssx + 0.5f), x1 = (int)(cx + 2f * ssx + 0.5f);
        int y0 = (int)(cy - 2f * ssy + 0.5f), y1 = (int)(cy + 2f * ssy + 0.5f);
        if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
        if (x1 > p.SrcW) x1 = p.SrcW; if (y1 > p.SrcH) y1 = p.SrcH;
        int shift = c * 8;
        float acc = 0f, wySum = 0f, wxSum = 0f;
        for (int x = x0; x < x1; x++) wxSum += Cubic((x - cx + 0.5f) / ssx);
        for (int y = y0; y < y1; y++)
        {
            float wy = Cubic((y - cy + 0.5f) / ssy);
            wySum += wy;
            float row = 0f;
            for (int x = x0; x < x1; x++)
                row += Cubic((x - cx + 0.5f) / ssx) * ((rgba[y * p.SrcW + x] >> shift) & 0xFF);
            acc += wy * row;
        }
        float v = acc / (wySum * wxSum) / 255f;
        if (v < 0f) v = 0f; if (v > 1f) v = 1f;   // PIL clips each pass to [0, 255]
        float m = c == 0 ? p.M0 : c == 1 ? p.M1 : p.M2;
        float inv = c == 0 ? p.I0 : c == 1 ? p.I1 : p.I2;
        o[idx] = (v - m) * inv;
    }

    // o[0, dim) = CLS token; o[dim, 2 dim) = mean of the patch tokens (one store per thread)
    private static void PoolImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> tokens, ArrayView1D<float, Stride1D.Dense> o, int dim, int patches)
    {
        if (i < dim) { o[i] = tokens[i]; return; }
        int d = i - dim;
        float s = 0f;
        for (int k = 1; k <= patches; k++) s += tokens[k * dim + d];
        o[i] = s / patches;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _pixels.Dispose();
        _pooled.Dispose();
        if (_ownsSession) _session.Dispose();
    }
}
