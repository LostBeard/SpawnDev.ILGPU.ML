using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Tensors;

/// <summary>
/// Shape-tracked view into GPU memory. Does NOT own the underlying buffer —
/// the buffer's lifetime is managed by BufferPool or the caller.
///
/// All data is contiguous row-major. Reshape and slice are zero-copy
/// (they create new Tensor views over the same data).
/// </summary>
public class Tensor
{
    /// <summary>Optional name (for debugging and graph execution).</summary>
    public string? Name { get; init; }

    /// <summary>Shape dimensions (e.g., [1370, 384] for T×C). Settable for runtime Reshape.</summary>
    public int[] Shape { get; set; }

    /// <summary>Total number of elements (product of shape dimensions). Recomputed when Shape changes.</summary>
    public int ElementCount => TensorHelpers.ElementCount(Shape);

    /// <summary>GPU data view. Length == ElementCount.</summary>
    public ArrayView1D<float, Stride1D.Dense> Data { get; }

    /// <summary>Row-major strides computed from shape.</summary>
    public int[] Strides { get; }

    public Tensor(ArrayView1D<float, Stride1D.Dense> data, int[] shape, string? name = null)
    {
        int count = TensorHelpers.ElementCount(shape);
        if (data.Length < count)
            throw new ArgumentException($"Data length {data.Length} < shape element count {count}");
        Data = data.SubView(0, count);
        Shape = shape;
        Strides = TensorHelpers.ComputeStrides(shape);
        Name = name;
    }

    // ── f16-native weights ──
    // A WEIGHT loaded as fp16 is carried as a half-backed Tensor: IsHalf == true, the data lives in
    // HalfData (ILGPU.Half — half the GPU bytes), and NO float buffer is allocated (Data is empty). The
    // graph's tensor map + node inputs stay Tensor-typed (no churn to the executor or op handlers' shape
    // logic); weight-consuming op handlers branch on IsHalf and route to their half-weight kernel variant
    // (e.g. MatMulKernel.MatMulHalfWeight), reading HalfData + accumulating fp32. Activations are always
    // fp32 (IsHalf == false). The f16 spike (2026-06-05) proved ILGPU.Half storage + fp32 compute on all
    // 6 backends; a generic-math (System.Half) kernel does not compile (BitCast intrinsic).

    /// <summary>Storage element type. <see cref="TensorDataType.Float32"/> = activations / fp32 weights (data in
    /// <see cref="Data"/>); any low-precision value = a native-typed weight (data in the low-p view, <see cref="Data"/>
    /// empty). Replaces the old <c>bool IsHalf</c> — a bool could only say fp16-or-fp32; weights now stay native in
    /// any low-p type and convert to f32 only at the arithmetic, in-kernel, via <c>ILGPU.PrecisionConvert</c>.</summary>
    public TensorDataType DType { get; } = TensorDataType.Float32;

    /// <summary>True iff the data is fp16 (<c>ILGPU.Half</c>) in <see cref="HalfData"/>, not fp32 in <see cref="Data"/>.
    /// Back-compat alias for <c>DType == TensorDataType.Float16</c> — existing weight consumers that branch on this keep
    /// working unchanged (only fp16 weights are loaded low-p today; bf16/FP8 land as the op dispatch moves to DType).</summary>
    public bool IsHalf => DType == TensorDataType.Float16;

    // Low-precision weight storage: the native-typed GPU view (ILGPU.Half / BFloat16 / Float8E*) is boxed once
    // (a weight is created once) and read back via AsView<T>(). NO float buffer is allocated (Data is empty); the
    // op kernel reads the native type + converts to f32 in-register via PrecisionConvert (no f32 temp buffer).
    private readonly object? _lowPView;

    /// <summary>The native low-precision GPU view as <typeparamref name="T"/> — must match <see cref="DType"/>
    /// (Half↔Float16, BFloat16↔BFloat16, Float8E4M3/E5M2↔FP8). Length == ElementCount. Throws if this tensor is not
    /// stored as <typeparamref name="T"/> (e.g. an fp32 activation, or a dtype mismatch) — fails loud, never silent.</summary>
    public ArrayView1D<T, Stride1D.Dense> AsView<T>() where T : unmanaged
        => _lowPView is ArrayView1D<T, Stride1D.Dense> v
            ? v
            : throw new InvalidOperationException($"Tensor '{Name}' is {DType}; cannot view it as {typeof(T).Name}.");

    /// <summary>fp16 GPU data view — valid IFF <see cref="IsHalf"/>. Back-compat alias for <c>AsView&lt;Half&gt;()</c>.</summary>
    public ArrayView1D<global::ILGPU.Half, Stride1D.Dense> HalfData
        => _lowPView is ArrayView1D<global::ILGPU.Half, Stride1D.Dense> v ? v : default;

    /// <summary>Wrap a fp16 <see cref="HalfTensor"/> as a half-backed Tensor for the graph (carries shape
    /// + the ILGPU.Half view; NO float buffer). The executor map stays Tensor-typed; handlers check IsHalf.</summary>
    public static Tensor FromHalf(HalfTensor half) => new Tensor(half.Data, half.Shape, half.Name);

    /// <summary>Wrap a native low-precision GPU view (<c>ILGPU.Half</c> / <c>BFloat16</c> / <c>Float8E4M3</c> /
    /// <c>Float8E5M2</c>) as a low-p-backed Tensor (carries shape + the typed view; NO float buffer). The executor
    /// map stays Tensor-typed; weight handlers switch on <see cref="DType"/> and read via <see cref="AsView{T}"/>.</summary>
    public static Tensor FromLowP<T>(ArrayView1D<T, Stride1D.Dense> view, TensorDataType dtype, int[] shape, string? name = null)
        where T : unmanaged
    {
        int count = TensorHelpers.ElementCount(shape);
        if (view.Length < count)
            throw new ArgumentException($"Low-p data length {view.Length} < shape element count {count}");
        return new Tensor((object)view.SubView(0, count), dtype, shape, name);
    }

    /// <summary>
    /// A shape-only tensor with NO backing buffer - for graph entries whose real data
    /// lives elsewhere (GGUF-quantized weights: raw bytes in QuantizedWeights, consumed
    /// by fused dequant kernels that need this entry only for its SHAPE). Any accidental
    /// use of <see cref="Data"/> hits an empty view and fails loudly instead of computing
    /// on garbage. Never allocate a full F32 buffer just to carry a shape - for a
    /// 262k×3840 embedding table that is ~4GB of dead VRAM.
    /// </summary>
    public static Tensor ShapeOnly(int[] shape, string? name = null) => new Tensor(shape, name);

    private Tensor(int[] shape, string? name)
    {
        Data = default;
        Shape = shape;
        Strides = TensorHelpers.ComputeStrides(shape);
        Name = name;
    }

    private Tensor(ArrayView1D<global::ILGPU.Half, Stride1D.Dense> halfData, int[] shape, string? name)
    {
        int count = TensorHelpers.ElementCount(shape);
        if (halfData.Length < count)
            throw new ArgumentException($"Half data length {halfData.Length} < shape element count {count}");
        _lowPView = halfData.SubView(0, count);
        DType = TensorDataType.Float16;
        Shape = shape;
        Strides = TensorHelpers.ComputeStrides(shape);
        Name = name;
        // Data (float) intentionally left empty — a low-p-backed tensor has NO float buffer.
    }

    private Tensor(object lowPView, TensorDataType dtype, int[] shape, string? name)
    {
        _lowPView = lowPView;
        DType = dtype;
        Shape = shape;
        Strides = TensorHelpers.ComputeStrides(shape);
        Name = name;
        // Data (float) intentionally left empty — a low-p-backed tensor has NO float buffer.
    }

    /// <summary>Number of dimensions.</summary>
    public int Rank => Shape.Length;

    /// <summary>
    /// Kernel-passable snapshot of this tensor as a blittable <see cref="TensorView{T}"/>.
    /// Phase 2+ kernels accept <c>TensorView&lt;float&gt;</c> instead of unpacking the
    /// data view + scalar shape parameters at the call site. Constructing the view is
    /// cheap (no managed allocations beyond the inline struct fields).
    /// </summary>
    public TensorView<float> View => new TensorView<float>(Data, Shape);

    /// <summary>
    /// Stream this tensor's raw fp32 GPU bytes OUT to <paramref name="target"/> in bounded chunks (default
    /// 16 MiB) — the SAVE mirror of the streaming-load path (<c>BufferPool.AllocatePermanentFromStreamAsync</c> /
    /// <c>CopyFromStreamAsync</c>). On the browser backends, when <paramref name="target"/> implements
    /// <c>SpawnDev.SpawnJS.Toolbox.IJSWriteStream</c> (e.g. an OPFS <c>FileSystemHandleWritableStream</c>), each
    /// chunk goes GPU→JS <c>Uint8Array</c>→stream WITHOUT entering the .NET/WASM managed heap — so a large tensor
    /// (e.g. a 588&#160;MB scene buffer) exports with one 16&#160;MiB chunk resident, never the whole buffer (no
    /// OOM). Desktop backends stream via a managed chunk buffer. Writes <c>ElementCount * 4</c> bytes.
    /// <para>Does NOT own or close <paramref name="target"/>. For OPFS the disk commit is the async <c>close()</c>:
    /// <c>await using</c> the writable stream (or <c>await CloseAsync()</c>) or the file can be empty/short.</para>
    /// <code>
    ///   await using var writable = await fileHandle.GetWritableStream();
    ///   await tensor.CopyToStreamAsync(writable);   // one 16 MiB chunk at a time, GPU→OPFS, zero managed-heap copy
    /// </code>
    /// </summary>
    public System.Threading.Tasks.Task CopyToStreamAsync(
        System.IO.Stream target,
        int chunkSizeInBytes = 16 * 1024 * 1024,
        System.Threading.CancellationToken cancellationToken = default)
        => Data.CopyToStreamAsync(target, chunkSizeInBytes, cancellationToken);

    /// <summary>
    /// Zero-copy reshape. Validates element count matches.
    /// Use -1 for one inferred dimension.
    /// </summary>
    public Tensor Reshape(int[] newShape)
    {
        var resolved = TensorHelpers.InferShape(newShape, ElementCount);
        return new Tensor(Data, resolved, Name);
    }

    /// <summary>
    /// Slice along the first dimension: takes elements [start*innerSize, (start+count)*innerSize).
    /// Zero-copy SubView.
    /// </summary>
    public Tensor Slice(int start, int count)
    {
        if (Shape.Length == 0) throw new InvalidOperationException("Cannot slice a scalar tensor");
        int innerSize = ElementCount / Shape[0];
        var newShape = (int[])Shape.Clone();
        newShape[0] = count;
        return new Tensor(Data.SubView(start * innerSize, count * innerSize), newShape, Name);
    }

    /// <summary>
    /// Create a sub-tensor at an arbitrary offset with a new shape.
    /// Zero-copy SubView.
    /// </summary>
    public Tensor SubTensor(long offset, int elementCount, int[] shape)
    {
        return new Tensor(Data.SubView(offset, elementCount), shape);
    }

    public override string ToString()
    {
        var shapeStr = string.Join(", ", Shape);
        return Name != null ? $"Tensor(\"{Name}\", [{shapeStr}])" : $"Tensor([{shapeStr}])";
    }
}
