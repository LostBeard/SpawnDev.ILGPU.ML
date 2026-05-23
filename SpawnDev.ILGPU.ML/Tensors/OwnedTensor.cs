using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Tensors;

/// <summary>
/// A <see cref="Tensor{T}"/> that <b>owns</b> its underlying accelerator buffer and
/// disposes it when the OwnedTensor is disposed. Use this for pipeline-output tensors
/// where the caller is expected to keep / discard them (Transformers.js
/// <c>session.run(...)</c> returned tensor semantics).
///
/// <para>
/// <see cref="Tensor{T}"/> is a non-owning view — multiple Tensor instances can refer to
/// the same underlying buffer for reshape / slice / sub-tensor operations.
/// <see cref="OwnedTensor{T}"/> is the lifecycle anchor: dispose it when you're done
/// and the buffer + every Tensor / TensorView pointed at it become invalid.
/// </para>
///
/// <para>
/// Composition over inheritance. The owned wrapper holds a single
/// <see cref="MemoryBuffer1D{T, TStride}"/> and exposes a non-owning <see cref="Tensor{T}"/>
/// through <see cref="AsTensor"/>. Implicit conversions to <c>Tensor&lt;T&gt;</c> and
/// <c>TensorView&lt;T&gt;</c> let owned tensors pass into any API that accepts those
/// types without manual <c>.AsTensor</c> / <c>.View</c> noise.
/// </para>
/// </summary>
public sealed class OwnedTensor<T> : IDisposable where T : unmanaged
{
    private readonly MemoryBuffer1D<T, Stride1D.Dense> _buffer;
    private bool _disposed;

    /// <summary>Optional name (for debugging / graph execution).</summary>
    public string? Name => AsTensor.Name;

    /// <summary>Shape dimensions.</summary>
    public int[] Shape => AsTensor.Shape;

    /// <summary>Total element count.</summary>
    public int ElementCount => AsTensor.ElementCount;

    /// <summary>Number of dimensions.</summary>
    public int Rank => AsTensor.Rank;

    /// <summary>Row-major strides.</summary>
    public int[] Strides => AsTensor.Strides;

    /// <summary>Non-owning <see cref="Tensor{T}"/> view over the owned buffer. Stays valid
    /// for the lifetime of this <see cref="OwnedTensor{T}"/>; do not use after dispose.</summary>
    public Tensor<T> AsTensor { get; }

    /// <summary>Kernel-passable <see cref="TensorView{T}"/> snapshot. Identical to
    /// <c>AsTensor.View</c>.</summary>
    public TensorView<T> View => AsTensor.View;

    /// <summary>Raw accelerator data view.</summary>
    public ArrayView1D<T, Stride1D.Dense> Data => AsTensor.Data;

    /// <summary>
    /// Wrap an existing accelerator buffer with shape metadata. The OwnedTensor takes
    /// ownership of the buffer and will dispose it when this OwnedTensor is disposed.
    /// </summary>
    public OwnedTensor(MemoryBuffer1D<T, Stride1D.Dense> buffer, int[] shape, string? name = null)
    {
        _buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
        AsTensor = new Tensor<T>(buffer.View, shape, name);
    }

    /// <summary>
    /// Allocate a fresh buffer on the accelerator and wrap it as an OwnedTensor.
    /// Convenience factory mirroring <c>torch.empty(shape)</c> semantics — contents
    /// are uninitialised until something writes to the buffer.
    /// </summary>
    public static OwnedTensor<T> Allocate(Accelerator accelerator, int[] shape, string? name = null)
    {
        int count = TensorHelpers.ElementCount(shape);
        var buffer = accelerator.Allocate1D<T>(count);
        return new OwnedTensor<T>(buffer, shape, name);
    }

    /// <summary>
    /// Allocate and copy host data in one step. Mirrors <c>torch.tensor(data)</c>.
    /// </summary>
    public static OwnedTensor<T> FromHost(Accelerator accelerator, T[] hostData, int[] shape, string? name = null)
    {
        int count = TensorHelpers.ElementCount(shape);
        if (hostData.Length != count)
            throw new ArgumentException($"Host data length {hostData.Length} != shape element count {count}");
        var buffer = accelerator.Allocate1D(hostData);
        return new OwnedTensor<T>(buffer, shape, name);
    }

    /// <summary>Implicit conversion to the non-owning view. Avoids <c>.AsTensor</c> noise.</summary>
    public static implicit operator Tensor<T>(OwnedTensor<T> owned) => owned.AsTensor;

    /// <summary>Implicit conversion to the kernel-passable struct. Avoids <c>.View</c> noise.</summary>
    public static implicit operator TensorView<T>(OwnedTensor<T> owned) => owned.View;

    /// <summary>Read the entire tensor back to host memory.</summary>
    public Task<T[]> ToHostAsync() => _buffer.CopyToHostAsync<T>(0, ElementCount);

    /// <inheritdoc/>
    public override string ToString() => $"Owned{AsTensor}";

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _buffer.Dispose();
    }
}
