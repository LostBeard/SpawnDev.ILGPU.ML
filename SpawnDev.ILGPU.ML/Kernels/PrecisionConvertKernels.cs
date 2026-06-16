using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU element-wise precision conversions between fp32 activations and low-precision storage
/// (<see cref="ILGPU.Half"/> fp16 / <see cref="ILGPU.BFloat16"/> bf16). The boundary primitive for
/// mixed-precision activations (Plans/fp16-bf16-mixed-precision-activations-2026-06-16.md): the executor
/// keeps heavy intermediates in low precision (half the bytes) and converts at fp32 boundaries.
///
/// HAND-WRITTEN per type with explicit <c>(Half)</c>/<c>(float)</c>/<c>(BFloat16)</c> casts — the same cast
/// pattern <c>MatMulKernel.MatMulHalfWeight</c> uses, which transpiles + runs on all 6 backends. (A single
/// generic <c>INumber&lt;T&gt;</c> convert does NOT work uniformly yet — PTX bf16-codegen / Wasm value /
/// low-p scalar-param gaps, tracked to Geordi; see that DevComms.) One store per thread at its own index →
/// WebGL Transform-Feedback safe.
/// </summary>
public sealed class PrecisionConvertKernels : IDisposable
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<global::ILGPU.Half, Stride1D.Dense>>? _f2h;
    private Action<Index1D, ArrayView1D<global::ILGPU.Half, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _h2f;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<global::ILGPU.BFloat16, Stride1D.Dense>>? _f2b;
    private Action<Index1D, ArrayView1D<global::ILGPU.BFloat16, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _b2f;

    public PrecisionConvertKernels(Accelerator accelerator) => _accelerator = accelerator;

    private static void FloatToHalfImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> src, ArrayView1D<global::ILGPU.Half, Stride1D.Dense> dst)
        => dst[i] = (global::ILGPU.Half)src[i];
    private static void HalfToFloatImpl(Index1D i, ArrayView1D<global::ILGPU.Half, Stride1D.Dense> src, ArrayView1D<float, Stride1D.Dense> dst)
        => dst[i] = (float)src[i];
    private static void FloatToBFloat16Impl(Index1D i, ArrayView1D<float, Stride1D.Dense> src, ArrayView1D<global::ILGPU.BFloat16, Stride1D.Dense> dst)
        => dst[i] = (global::ILGPU.BFloat16)src[i];
    private static void BFloat16ToFloatImpl(Index1D i, ArrayView1D<global::ILGPU.BFloat16, Stride1D.Dense> src, ArrayView1D<float, Stride1D.Dense> dst)
        => dst[i] = (float)src[i];

    /// <summary>fp32 → fp16 (Half), element-wise. <paramref name="count"/> elements.</summary>
    public void FloatToHalf(ArrayView1D<float, Stride1D.Dense> src, ArrayView1D<global::ILGPU.Half, Stride1D.Dense> dst, int count)
    {
        _f2h ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<global::ILGPU.Half, Stride1D.Dense>>(FloatToHalfImpl);
        _f2h(count, src, dst);
    }

    /// <summary>fp16 (Half) → fp32, element-wise.</summary>
    public void HalfToFloat(ArrayView1D<global::ILGPU.Half, Stride1D.Dense> src, ArrayView1D<float, Stride1D.Dense> dst, int count)
    {
        _h2f ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<global::ILGPU.Half, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(HalfToFloatImpl);
        _h2f(count, src, dst);
    }

    /// <summary>fp32 → bf16 (BFloat16), element-wise.</summary>
    public void FloatToBFloat16(ArrayView1D<float, Stride1D.Dense> src, ArrayView1D<global::ILGPU.BFloat16, Stride1D.Dense> dst, int count)
    {
        _f2b ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<global::ILGPU.BFloat16, Stride1D.Dense>>(FloatToBFloat16Impl);
        _f2b(count, src, dst);
    }

    /// <summary>bf16 (BFloat16) → fp32, element-wise.</summary>
    public void BFloat16ToFloat(ArrayView1D<global::ILGPU.BFloat16, Stride1D.Dense> src, ArrayView1D<float, Stride1D.Dense> dst, int count)
    {
        _b2f ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<global::ILGPU.BFloat16, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(BFloat16ToFloatImpl);
        _b2f(count, src, dst);
    }

    public void Dispose() { }
}
