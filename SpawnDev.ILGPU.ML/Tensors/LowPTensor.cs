using System.Numerics;
using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Tensors;

/// <summary>
/// Shape-tracked view into GPU memory holding weights in their NATIVE low-precision type
/// (<see cref="ILGPU.Half"/> / <see cref="ILGPU.BFloat16"/> / <c>Float8E4M3</c> / <c>Float8E5M2</c>) - the
/// generic counterpart to <see cref="Tensor"/> (fp32) and <see cref="HalfTensor"/> (fp16 only).
/// </summary>
/// <remarks>
/// The data is NEVER expanded in memory. Weight-consuming kernels read it in its native type and upconvert
/// per element in-register, accumulating fp32 (<c>MatMulKernel.MatMulLowPWeight&lt;T&gt;</c>) - so a bf16
/// weight occupies bf16 bytes on the GPU, not 2x that, and an FP8 weight 1 byte, not 4x. On a constrained
/// device that is the difference between a model fitting and not, and it is the reason the low-precision
/// format was chosen in the first place.
/// <para>
/// ⚠️ <see cref="HalfTensor"/>'s remarks say a generic tensor was impossible because "a generic-math
/// (System.Half + INumber) kernel does not [work] (BitCast intrinsic unsupported)". That was true when it was
/// written (the f16 spike, 2026-06-05) and is NOT true any more: <c>ILGPU.PrecisionConvert</c> shipped in
/// 4.13.0-local.9 with a <c>[ConvertIntrinisc]</c> lowering that is bit-exact on all six backends, and
/// <c>MatMulLowPWeight&lt;T&gt;</c> plus <c>F16_MatMulBFloat16Weight_MatchesFp32Reference</c> already prove
/// the generic path end to end. This type closes the remaining gap: the KERNELS were generic while the
/// LOADER was hard-coded to Half, so bf16/FP8 had no route into GPU memory in their native type.
/// </para>
/// <para>
/// Does NOT own the underlying buffer - lifetime belongs to <see cref="BufferPool"/> or the caller.
/// </para>
/// </remarks>
/// <typeparam name="T">Native element type (unmanaged, <see cref="INumber{TSelf}"/>).</typeparam>
public class LowPTensor<T> where T : unmanaged, INumber<T>
{
    /// <summary>Optional name (for debugging and graph execution).</summary>
    public string? Name { get; init; }

    /// <summary>Shape dimensions. Settable for runtime Reshape.</summary>
    public int[] Shape { get; set; }

    /// <summary>Total number of elements (product of shape dimensions).</summary>
    public int ElementCount => TensorHelpers.ElementCount(Shape);

    /// <summary>GPU data view in the NATIVE element type. Length == ElementCount.</summary>
    public ArrayView1D<T, Stride1D.Dense> Data { get; }

    /// <summary>Row-major strides computed from shape.</summary>
    public int[] Strides { get; }

    /// <summary>Creates a shape-tracked view over <paramref name="data"/>.</summary>
    /// <param name="data">GPU view holding at least the shape's element count.</param>
    /// <param name="shape">Row-major shape.</param>
    /// <param name="name">Optional name.</param>
    public LowPTensor(ArrayView1D<T, Stride1D.Dense> data, int[] shape, string? name = null)
    {
        int count = TensorHelpers.ElementCount(shape);
        if (data.Length < count)
            throw new ArgumentException($"Data length {data.Length} < shape element count {count}");
        Data = data.SubView(0, count);
        Shape = shape;
        Strides = TensorHelpers.ComputeStrides(shape);
        Name = name;
    }

    /// <inheritdoc/>
    public override string ToString() => $"{Name ?? "(unnamed)"} {typeof(T).Name}[{string.Join(",", Shape)}]";
}
