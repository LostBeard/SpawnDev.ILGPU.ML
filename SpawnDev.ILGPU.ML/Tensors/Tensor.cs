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

    /// <summary>True if the data is fp16 in <see cref="HalfData"/> (a weight), not fp32 in <see cref="Data"/>.</summary>
    public bool IsHalf { get; }

    /// <summary>fp16 GPU data view — valid IFF <see cref="IsHalf"/>. Length == ElementCount.</summary>
    public ArrayView1D<global::ILGPU.Half, Stride1D.Dense> HalfData { get; }

    /// <summary>Wrap a fp16 <see cref="HalfTensor"/> as a half-backed Tensor for the graph (carries shape
    /// + the ILGPU.Half view; NO float buffer). The executor map stays Tensor-typed; handlers check IsHalf.</summary>
    public static Tensor FromHalf(HalfTensor half) => new Tensor(half.Data, half.Shape, half.Name);

    private Tensor(ArrayView1D<global::ILGPU.Half, Stride1D.Dense> halfData, int[] shape, string? name)
    {
        int count = TensorHelpers.ElementCount(shape);
        if (halfData.Length < count)
            throw new ArgumentException($"Half data length {halfData.Length} < shape element count {count}");
        HalfData = halfData.SubView(0, count);
        IsHalf = true;
        Shape = shape;
        Strides = TensorHelpers.ComputeStrides(shape);
        Name = name;
        // Data (float) intentionally left empty — a half-backed tensor has NO float buffer.
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
