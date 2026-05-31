using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Tensors;

/// <summary>
/// Blittable, kernel-passable snapshot of a <see cref="Tensor{T}"/>. This is the
/// struct counterpart to <see cref="Tensor{T}"/> — the same split ILGPU itself uses
/// between <c>MemoryBuffer&lt;T&gt;</c> (class, lifetime-managing) and
/// <c>ArrayView&lt;T&gt;</c> (struct, kernel-passable).
///
/// <para>
/// Pass this directly as an ILGPU kernel parameter. Kernels read the data through
/// <see cref="Data"/> and use the inline <c>D0..D3</c> dimensions for index math,
/// avoiding the verbose pattern of passing every shape dimension as a separate scalar
/// kernel parameter.
/// </para>
///
/// <para>
/// <b>Rank limit:</b> up to 4 dimensions, covering 99% of ML cases (1D vectors,
/// 2D matrices/embeddings, 3D sequences <c>[B, T, C]</c>, 4D images <c>[N, C, H, W]</c>).
/// Higher ranks throw at construction — reshape to ≤4D first, or open an issue if a
/// real model needs more.
/// </para>
///
/// <para>
/// <b>Strides:</b> the view assumes contiguous row-major layout (the default for
/// <see cref="Tensor{T}"/>). Index methods compute strides inline from <c>D0..D3</c>.
/// Non-contiguous slices need a separate <c>StridedTensorView&lt;T&gt;</c> (future work).
/// </para>
///
/// <para>
/// <b>Lifetime:</b> a <see cref="TensorView{T}"/> borrows from a <see cref="Tensor{T}"/>
/// (or directly from an <see cref="ArrayView1D{T, TStride}"/>). It does NOT own the
/// underlying buffer. Hold a reference to the source <see cref="Tensor{T}"/> as long
/// as you intend to dispatch kernels against the view.
/// </para>
/// </summary>
public readonly struct TensorView<T> where T : unmanaged
{
    /// <summary>Underlying contiguous row-major data on the accelerator.</summary>
    public readonly ArrayView1D<T, Stride1D.Dense> Data;

    /// <summary>Outermost dimension. For an [N, C, H, W] tensor this is N. For lower-rank
    /// tensors the unused dimensions are 1.</summary>
    public readonly int D0;

    /// <summary>Second dimension (e.g., C in NCHW). 1 when rank &lt; 2.</summary>
    public readonly int D1;

    /// <summary>Third dimension (e.g., H in NCHW). 1 when rank &lt; 3.</summary>
    public readonly int D2;

    /// <summary>Fourth dimension (e.g., W in NCHW). 1 when rank &lt; 4.</summary>
    public readonly int D3;

    /// <summary>Actual number of dimensions in the source shape (1, 2, 3, or 4).</summary>
    public readonly int Rank;

    /// <summary>
    /// Construct from a contiguous row-major <see cref="ArrayView1D{T, TStride}"/> and a
    /// shape with rank between 1 and 4.
    /// </summary>
    public TensorView(ArrayView1D<T, Stride1D.Dense> data, int[] shape)
    {
        if (shape is null)
            throw new ArgumentNullException(nameof(shape));
        if (shape.Length < 1 || shape.Length > 4)
            throw new ArgumentException(
                $"TensorView supports rank 1-4. Reshape the source tensor down to ≤4D before constructing a TensorView. Got rank {shape.Length}.",
                nameof(shape));

        Data = data;
        Rank = shape.Length;
        D0 = shape[0];
        D1 = Rank > 1 ? shape[1] : 1;
        D2 = Rank > 2 ? shape[2] : 1;
        D3 = Rank > 3 ? shape[3] : 1;
    }

    /// <summary>Construct directly from inline dimensions (no managed-array allocation).</summary>
    public TensorView(ArrayView1D<T, Stride1D.Dense> data, int d0, int d1 = 1, int d2 = 1, int d3 = 1, int rank = 4)
    {
        Data = data;
        D0 = d0;
        D1 = d1;
        D2 = d2;
        D3 = d3;
        Rank = rank;
    }

    /// <summary>Total element count: <c>D0 * D1 * D2 * D3</c> (unused dims are 1).</summary>
    public int ElementCount => D0 * D1 * D2 * D3;

    // ─── Kernel-side accessors. Strides computed inline from D0..D3. ───
    //
    // The kernel author picks the indexer that matches their tensor's rank. Mismatched
    // ranks (e.g. Get2D on a 4D tensor) silently produce incorrect data — same trade-off
    // PyTorch's tensor[(i, j)] makes for raw indexing. Use the typed indexers below in
    // host code if rank-safety matters; kernels prioritize zero-overhead access.

    /// <summary>1D access. Equivalent to <c>Data[i]</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Get1D(int i) => Data[i];

    /// <summary>2D access in row-major <c>[D0, D1]</c> layout.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Get2D(int i0, int i1) => Data[i0 * D1 + i1];

    /// <summary>3D access in row-major <c>[D0, D1, D2]</c> layout.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Get3D(int i0, int i1, int i2) => Data[(i0 * D1 + i1) * D2 + i2];

    /// <summary>4D access in row-major <c>[D0, D1, D2, D3]</c> layout (e.g., NCHW).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Get4D(int i0, int i1, int i2, int i3) => Data[((i0 * D1 + i1) * D2 + i2) * D3 + i3];

    /// <summary>1D write. Equivalent to <c>Data[i] = v</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Set1D(int i, T v) => Data[i] = v;

    /// <summary>2D write in row-major <c>[D0, D1]</c> layout.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Set2D(int i0, int i1, T v) => Data[i0 * D1 + i1] = v;

    /// <summary>3D write in row-major <c>[D0, D1, D2]</c> layout.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Set3D(int i0, int i1, int i2, T v) => Data[(i0 * D1 + i1) * D2 + i2] = v;

    /// <summary>4D write in row-major <c>[D0, D1, D2, D3]</c> layout.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Set4D(int i0, int i1, int i2, int i3, T v) => Data[((i0 * D1 + i1) * D2 + i2) * D3 + i3] = v;
}
