using System.Numerics;
using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU element-wise precision conversions between fp32 activations and ANY low-precision storage type
/// (<c>ILGPU.Half</c> fp16 / <c>ILGPU.BFloat16</c> bf16 / <c>ILGPU.Float8E4M3</c> / <c>ILGPU.Float8E5M2</c> /
/// future FP4·INT4). The boundary primitive for mixed-precision activations
/// (Plans/fp16-bf16-mixed-precision-activations-2026-06-16.md): the executor keeps heavy intermediates in low
/// precision (a fraction of the bytes) and converts at fp32 boundaries.
///
/// ONE generic <c>where T : unmanaged, INumber&lt;T&gt;</c> kernel per direction via Geordi's
/// <c>ILGPU.PrecisionConvert</c> (shipped 4.13.0-local.9, bit-exact on all 6 backends — CPU/CUDA/OpenCL/
/// WebGPU/WebGL/Wasm — see DevComms <c>geordi-to-tuvok-PrecisionConvert-shipped-local9</c>). It is tagged
/// <c>[ConvertIntrinisc]</c> and lowers to the SAME <c>ConvertValue</c> IR node the concrete <c>(T)(float)</c>
/// cast emits, resolved per instantiation — so a single generic kernel covers every low-precision type instead
/// of a hand-written pair each (the old per-type approach predated local.9). Replaces needless low-p→f32 temp
/// passes with read-low-p / accumulate-f32-in-registers / write-low-p, per the shared no-needless-conversion
/// rule. One store per thread at its own index → WebGL Transform-Feedback safe.
/// </summary>
public sealed class PrecisionConvertKernels : IDisposable
{
    private readonly Accelerator _accelerator;

    // One compiled kernel per concrete T (cached). object-typed because the delegate is T-specific.
    private readonly Dictionary<Type, object> _floatToLowP = new();
    private readonly Dictionary<Type, object> _lowPToFloat = new();

    public PrecisionConvertKernels(Accelerator accelerator) => _accelerator = accelerator;

    private static void FloatToLowPImpl<T>(Index1D i, ArrayView1D<float, Stride1D.Dense> src, ArrayView1D<T, Stride1D.Dense> dst)
        where T : unmanaged, INumber<T>
        => dst[i] = PrecisionConvert.ConvertFromSingle<T>(src[i]);

    private static void LowPToFloatImpl<T>(Index1D i, ArrayView1D<T, Stride1D.Dense> src, ArrayView1D<float, Stride1D.Dense> dst)
        where T : unmanaged, INumber<T>
        => dst[i] = PrecisionConvert.ConvertToSingle(src[i]);

    /// <summary>fp32 → any low-precision <typeparamref name="T"/> (Half/bf16/FP8/…), element-wise, <paramref name="count"/> elements.</summary>
    public void FloatToLowP<T>(ArrayView1D<float, Stride1D.Dense> src, ArrayView1D<T, Stride1D.Dense> dst, int count)
        where T : unmanaged, INumber<T>
    {
        if (!_floatToLowP.TryGetValue(typeof(T), out var k))
            _floatToLowP[typeof(T)] = k = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>>(FloatToLowPImpl<T>);
        ((Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>>)k)(count, src, dst);
    }

    /// <summary>Any low-precision <typeparamref name="T"/> → fp32, element-wise.</summary>
    public void LowPToFloat<T>(ArrayView1D<T, Stride1D.Dense> src, ArrayView1D<float, Stride1D.Dense> dst, int count)
        where T : unmanaged, INumber<T>
    {
        if (!_lowPToFloat.TryGetValue(typeof(T), out var k))
            _lowPToFloat[typeof(T)] = k = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(LowPToFloatImpl<T>);
        ((Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>)k)(count, src, dst);
    }

    // ── Typed convenience wrappers (callers unchanged — they forward to the generic kernel) ─────────────────
    /// <summary>fp32 → fp16 (Half), element-wise.</summary>
    public void FloatToHalf(ArrayView1D<float, Stride1D.Dense> src, ArrayView1D<global::ILGPU.Half, Stride1D.Dense> dst, int count)
        => FloatToLowP(src, dst, count);
    /// <summary>fp16 (Half) → fp32, element-wise.</summary>
    public void HalfToFloat(ArrayView1D<global::ILGPU.Half, Stride1D.Dense> src, ArrayView1D<float, Stride1D.Dense> dst, int count)
        => LowPToFloat(src, dst, count);
    /// <summary>fp32 → bf16 (BFloat16), element-wise.</summary>
    public void FloatToBFloat16(ArrayView1D<float, Stride1D.Dense> src, ArrayView1D<global::ILGPU.BFloat16, Stride1D.Dense> dst, int count)
        => FloatToLowP(src, dst, count);
    /// <summary>bf16 (BFloat16) → fp32, element-wise.</summary>
    public void BFloat16ToFloat(ArrayView1D<global::ILGPU.BFloat16, Stride1D.Dense> src, ArrayView1D<float, Stride1D.Dense> dst, int count)
        => LowPToFloat(src, dst, count);

    public void Dispose() { }
}
