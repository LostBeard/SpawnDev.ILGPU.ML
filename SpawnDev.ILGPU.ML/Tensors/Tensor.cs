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

    /// <summary>Number of dimensions.</summary>
    public int Rank => Shape.Length;

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
