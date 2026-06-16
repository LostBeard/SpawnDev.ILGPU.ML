using System.Collections.Concurrent;
using System.Numerics;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Approach-(i) precision-AWARE op kernels: a SINGLE generic <c>where T : INumber&lt;T&gt;</c> kernel per op
/// that reads its low-precision input DIRECTLY, computes in fp32 (precision), and writes low precision
/// DIRECTLY — NO fp32 temp buffers, NO convert-around-node (that was the workaround that didn't cut memory).
/// One kernel covers float / <see cref="ILGPU.Half"/> / <see cref="ILGPU.BFloat16"/> (and fp8 when its GPU
/// codegen lands), via <see cref="ILGPU.PrecisionConvert"/> (transpilable generic float↔T, ILGPU local.9).
/// This is the Rule-1/Rule-4 path (no workaround, half the activation bytes + bandwidth). The executor stores
/// intermediates as <c>T</c> and dispatches these for <c>T=Half</c> in F16 mode.
///
/// Plan: Plans/fp16-bf16-mixed-precision-activations-2026-06-16.md.
/// </summary>
public sealed class PrecisionAwareKernels : IDisposable
{
    private readonly Accelerator _accelerator;
    // Per-element-type kernel cache (one compiled kernel per T per op). Keyed by typeof(T).
    private readonly ConcurrentDictionary<Type, object> _siluCache = new();

    public PrecisionAwareKernels(Accelerator accelerator) => _accelerator = accelerator;

    // ── SiLU (x * sigmoid(x)) — read T, compute fp32, write T. The gemma/SD activation; tanh/exp force fp32. ──
    private static void SiLUImpl<T>(Index1D i, ArrayView1D<T, Stride1D.Dense> input, ArrayView1D<T, Stride1D.Dense> output)
        where T : unmanaged, INumber<T>
    {
        float x = PrecisionConvert.ConvertToSingle(input[i]);
        // SiLU = x / (1 + e^-x). Clamp the exponent tail for fp32 stability (matches the fp32 activation kernels).
        float s;
        if (x > 30f) s = x;
        else if (x < -30f) s = 0f;
        else s = x / (1f + XMath.Exp(-x));
        output[i] = PrecisionConvert.ConvertFromSingle<T>(s);
    }

    /// <summary>SiLU (x·sigmoid(x)) in place-of-precision T (float/Half/bf16). out[i] = silu(in[i]).
    /// in/out are <c>ArrayView1D&lt;T&gt;</c> — true low-precision I/O, no fp32 temp.</summary>
    public void SiLU<T>(ArrayView1D<T, Stride1D.Dense> input, ArrayView1D<T, Stride1D.Dense> output, int count)
        where T : unmanaged, INumber<T>
    {
        var k = (Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>>)
            _siluCache.GetOrAdd(typeof(T), _ => _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>>(SiLUImpl<T>));
        k(count, input, output);
    }

    public void Dispose() { }
}
