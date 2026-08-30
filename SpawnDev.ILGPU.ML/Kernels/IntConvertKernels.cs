using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Converts NATIVE 8-bit integer weights to fp32 on the GPU, for the quantized (int8/uint8) storage path.
/// </summary>
/// <remarks>
/// The float-format low-precision types go through <see cref="PrecisionConvertKernels"/>, which is generic
/// over <c>INumber&lt;T&gt;</c> and lowers via <c>ILGPU.PrecisionConvert</c>. Integers cannot use that path:
/// <c>ConvertToSingle</c> is defined for the float encodings, and a generic-math widening does not lower to
/// a kernel on every backend. So these are two CONCRETE kernels using ordinary casts, which every backend
/// compiles - deliberately not generic, because a generic version that fails to lower on WebGL would be
/// worse than two explicit ones.
/// <para>
/// The point is storage: an int8 weight occupies a QUARTER of the fp32 bytes and stays that way in memory.
/// The widening happens here, at use, into the op's own output - never as a resident expanded copy.
/// </para>
/// </remarks>
public sealed class IntConvertKernels
{
    private readonly Accelerator _accelerator;
    private Action<Index1D, ArrayView1D<sbyte, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _i8;
    private Action<Index1D, ArrayView1D<byte, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _u8;

    /// <summary>Creates a new instance bound to <paramref name="accelerator"/>.</summary>
    /// <param name="accelerator">Accelerator the kernels are compiled for.</param>
    public IntConvertKernels(Accelerator accelerator) => _accelerator = accelerator;

    private static void Int8Impl(Index1D i, ArrayView1D<sbyte, Stride1D.Dense> src, ArrayView1D<float, Stride1D.Dense> dst)
        => dst[i] = src[i];

    private static void UInt8Impl(Index1D i, ArrayView1D<byte, Stride1D.Dense> src, ArrayView1D<float, Stride1D.Dense> dst)
        => dst[i] = src[i];

    /// <summary>Widen <paramref name="count"/> signed 8-bit values to fp32.</summary>
    /// <param name="src">Native int8 source view.</param>
    /// <param name="dst">fp32 destination view.</param>
    /// <param name="count">Element count.</param>
    public void Int8ToFloat(ArrayView1D<sbyte, Stride1D.Dense> src, ArrayView1D<float, Stride1D.Dense> dst, int count)
    {
        _i8 ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<sbyte, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(Int8Impl);
        _i8(count, src, dst);
    }

    /// <summary>Widen <paramref name="count"/> unsigned 8-bit values to fp32.</summary>
    /// <param name="src">Native uint8 source view.</param>
    /// <param name="dst">fp32 destination view.</param>
    /// <param name="count">Element count.</param>
    public void UInt8ToFloat(ArrayView1D<byte, Stride1D.Dense> src, ArrayView1D<float, Stride1D.Dense> dst, int count)
    {
        _u8 ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<byte, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(UInt8Impl);
        _u8(count, src, dst);
    }
}
