using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Tensors;

/// <summary>
/// Shape-tracked view into GPU memory holding fp16 (<see cref="ILGPU.Half"/>) data — the f16-native
/// counterpart to <see cref="Tensor"/>. Used for WEIGHTS: a model's fp16 weights are stored as ILGPU.Half
/// (half the bytes of the fp32 <see cref="Tensor"/>), and weight-consuming kernels (e.g.
/// <c>MatMulKernel.MatMulHalfWeight</c>) read them as ILGPU.Half, upconvert per-element, and accumulate
/// fp32 (ORT-style mixed precision — no accuracy loss). Activations stay fp32 (<see cref="Tensor"/>).
///
/// Does NOT own the underlying buffer — lifetime is managed by <see cref="BufferPool"/> or the caller.
/// The f16 spike (2026-06-05) confirmed ILGPU.Half storage + fp32 compute works on all 6 backends; a
/// generic-math (System.Half + INumber) kernel does not (BitCast intrinsic unsupported), so weights are
/// carried as this dedicated half type rather than a generic Tensor&lt;T&gt;.
/// </summary>
public class HalfTensor
{
    /// <summary>Optional name (for debugging and graph execution).</summary>
    public string? Name { get; init; }

    /// <summary>Shape dimensions. Settable for runtime Reshape.</summary>
    public int[] Shape { get; set; }

    /// <summary>Total number of elements (product of shape dimensions).</summary>
    public int ElementCount => TensorHelpers.ElementCount(Shape);

    /// <summary>GPU fp16 data view. Length == ElementCount.</summary>
    public ArrayView1D<global::ILGPU.Half, Stride1D.Dense> Data { get; }

    /// <summary>Row-major strides computed from shape.</summary>
    public int[] Strides { get; }

    public HalfTensor(ArrayView1D<global::ILGPU.Half, Stride1D.Dense> data, int[] shape, string? name = null)
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

    /// <summary>Kernel-passable blittable snapshot as a <see cref="TensorView{T}"/> of ILGPU.Half.</summary>
    public TensorView<global::ILGPU.Half> View => new TensorView<global::ILGPU.Half>(Data, Shape);

    /// <summary>Zero-copy reshape. Validates element count matches. Use -1 for one inferred dimension.</summary>
    public HalfTensor Reshape(int[] newShape)
    {
        var resolved = TensorHelpers.InferShape(newShape, ElementCount);
        return new HalfTensor(Data, resolved, Name);
    }

    public override string ToString()
    {
        var shapeStr = string.Join(", ", Shape);
        return Name != null ? $"HalfTensor(\"{Name}\", [{shapeStr}])" : $"HalfTensor([{shapeStr}])";
    }
}
