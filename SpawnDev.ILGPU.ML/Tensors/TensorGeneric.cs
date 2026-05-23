using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Tensors;

/// <summary>
/// Shape-tracked view into accelerator memory, generic over the element type.
/// This is the host-side counterpart to <see cref="TensorView{T}"/> — keep this class
/// alive for lifetime / reshape / slice / debug operations, and pass its
/// <see cref="View"/> property when dispatching kernels.
///
/// <para>
/// Modelled after Transformers.js / ONNX Runtime tensor semantics:
/// <c>Tensor&lt;T&gt;</c> ≈ Transformers.js <c>Tensor</c> (with <c>.dims</c>, <c>.data</c>,
/// <c>.type</c>). The element type encoding moves from a runtime string
/// (<c>"float32"</c>) to a compile-time generic parameter, so <c>Tensor&lt;float&gt;</c>
/// and <c>Tensor&lt;int&gt;</c> are distinct types that the compiler can keep apart.
/// </para>
///
/// <para>
/// Like the non-generic <see cref="Tensor"/>, this class does NOT own the underlying
/// <see cref="ArrayView1D{T, TStride}"/> — the caller (or a BufferPool / InferenceSession)
/// manages the backing buffer's lifetime. Reshape and Slice are zero-copy.
/// </para>
///
/// <para>
/// The non-generic <see cref="Tensor"/> stays as-is for backwards compatibility. Kernels
/// migrate from raw <c>ArrayView</c> + scalar shape parameters to <see cref="TensorView{T}"/>
/// one operator family at a time.
/// </para>
/// </summary>
public class Tensor<T> where T : unmanaged
{
    /// <summary>Optional name (for debugging and graph execution).</summary>
    public string? Name { get; init; }

    /// <summary>Shape dimensions (e.g., <c>[1, 3, 224, 224]</c> for an NCHW image tensor).
    /// Settable for runtime reshape, but the element count must match.</summary>
    public int[] Shape { get; set; }

    /// <summary>Total number of elements (product of shape dimensions). Recomputed when
    /// <see cref="Shape"/> changes.</summary>
    public int ElementCount => TensorHelpers.ElementCount(Shape);

    /// <summary>Accelerator data view. Length is &gt;= <see cref="ElementCount"/>.</summary>
    public ArrayView1D<T, Stride1D.Dense> Data { get; }

    /// <summary>Row-major strides computed from <see cref="Shape"/>.</summary>
    public int[] Strides { get; }

    /// <summary>Number of dimensions.</summary>
    public int Rank => Shape.Length;

    /// <summary>
    /// Kernel-passable snapshot. Construct fresh on each kernel dispatch — the struct
    /// is cheap (six 32-bit fields plus the ArrayView) and decouples kernel-side use
    /// from class-side mutation (e.g. Reshape).
    /// </summary>
    public TensorView<T> View => new TensorView<T>(Data, Shape);

    /// <summary>
    /// Wrap an existing accelerator data view with shape metadata. The view is sub-viewed
    /// to exactly the shape's element count.
    /// </summary>
    public Tensor(ArrayView1D<T, Stride1D.Dense> data, int[] shape, string? name = null)
    {
        int count = TensorHelpers.ElementCount(shape);
        if (data.Length < count)
            throw new ArgumentException(
                $"Data length {data.Length} < shape element count {count}");
        Data = data.SubView(0, count);
        Shape = shape;
        Strides = TensorHelpers.ComputeStrides(shape);
        Name = name;
    }

    /// <summary>
    /// Zero-copy reshape. Validates element count matches.
    /// Use -1 for one inferred dimension (e.g. <c>Reshape([batch, -1])</c> flattens trailing dims).
    /// </summary>
    public Tensor<T> Reshape(int[] newShape)
    {
        var resolved = TensorHelpers.InferShape(newShape, ElementCount);
        return new Tensor<T>(Data, resolved, Name);
    }

    /// <summary>
    /// Slice along the first dimension: takes elements
    /// <c>[start * innerSize, (start + count) * innerSize)</c>. Zero-copy via
    /// <see cref="ArrayView{T}.SubView"/>.
    /// </summary>
    public Tensor<T> Slice(int start, int count)
    {
        if (Shape.Length == 0)
            throw new InvalidOperationException("Cannot slice a scalar tensor");
        int innerSize = ElementCount / Shape[0];
        var newShape = (int[])Shape.Clone();
        newShape[0] = count;
        return new Tensor<T>(Data.SubView(start * innerSize, count * innerSize), newShape, Name);
    }

    /// <summary>
    /// Create a sub-tensor at an arbitrary offset with a new shape. Zero-copy via
    /// <see cref="ArrayView{T}.SubView"/>.
    /// </summary>
    public Tensor<T> SubTensor(long offset, int elementCount, int[] shape)
    {
        return new Tensor<T>(Data.SubView(offset, elementCount), shape);
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        var shapeStr = string.Join(", ", Shape);
        var typeStr = typeof(T).Name;
        return Name != null
            ? $"Tensor<{typeStr}>(\"{Name}\", [{shapeStr}])"
            : $"Tensor<{typeStr}>([{shapeStr}])";
    }
}
