using ILGPU;
using ILGPU.Runtime;
using System.Runtime.CompilerServices;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// ONNX Concat operation as fused parallel kernels.
///
/// Replaces the per-(input, outer-block) Scale dispatch pattern in
/// ConcatOperator.Execute which on Wasm pays worker-pool round-trip overhead
/// per call and accumulates to seconds per Concat node in transformer attention
/// RoPE blocks (BREAK_AT=800 diagnostic 2026-05-05 showed 12+ Concat nodes at
/// 1500-2271ms each on Wasm). One kernel dispatch per Concat node instead of
/// N*outer calls.
///
/// ILGPU kernel signatures are fixed-arity, so we provide separate variants for
/// 2/3/4 inputs (covers virtually all production ONNX models - RoPE in particular
/// is 2-input). Concat with more than 4 inputs falls back to the existing
/// per-pair Scale dispatch path in ConcatOperator.Execute.
///
/// Per-thread: decomposes its linear output index into (outer, concat-axis,
/// inner) coordinates, finds which input owns the concat-axis position via
/// cumulative offsets, reads from that input.
/// </summary>
public class ConcatKernel : IDisposable
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int>? _kernel2;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int, int>? _kernel3;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int, int, int>? _kernel4;

    public ConcatKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>True if the fused kernel can handle this input count.</summary>
    public static bool CanHandle(int numInputs) => numInputs >= 2 && numInputs <= 4;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ConcatImpl2(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> input0,
        ArrayView1D<float, Stride1D.Dense> input1,
        int outer, int inner, int len0, int len1)
    {
        int totalConcat = len0 + len1;
        int o = idx / (totalConcat * inner);
        int rem = idx - o * totalConcat * inner;
        int c = rem / inner;
        int i = rem - c * inner;

        if (c < len0)
        {
            int srcIdx = o * len0 * inner + c * inner + i;
            output[idx] = (srcIdx >= 0 && (long)srcIdx < input0.Length) ? input0[srcIdx] : 0f;
        }
        else
        {
            int cInInput = c - len0;
            int srcIdx = o * len1 * inner + cInInput * inner + i;
            output[idx] = (srcIdx >= 0 && (long)srcIdx < input1.Length) ? input1[srcIdx] : 0f;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ConcatImpl3(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> input0,
        ArrayView1D<float, Stride1D.Dense> input1,
        ArrayView1D<float, Stride1D.Dense> input2,
        int outer, int inner, int len0, int len1, int len2)
    {
        int totalConcat = len0 + len1 + len2;
        int o = idx / (totalConcat * inner);
        int rem = idx - o * totalConcat * inner;
        int c = rem / inner;
        int i = rem - c * inner;

        if (c < len0)
        {
            int srcIdx = o * len0 * inner + c * inner + i;
            output[idx] = (srcIdx >= 0 && (long)srcIdx < input0.Length) ? input0[srcIdx] : 0f;
        }
        else if (c < len0 + len1)
        {
            int cIn = c - len0;
            int srcIdx = o * len1 * inner + cIn * inner + i;
            output[idx] = (srcIdx >= 0 && (long)srcIdx < input1.Length) ? input1[srcIdx] : 0f;
        }
        else
        {
            int cIn = c - len0 - len1;
            int srcIdx = o * len2 * inner + cIn * inner + i;
            output[idx] = (srcIdx >= 0 && (long)srcIdx < input2.Length) ? input2[srcIdx] : 0f;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ConcatImpl4(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> input0,
        ArrayView1D<float, Stride1D.Dense> input1,
        ArrayView1D<float, Stride1D.Dense> input2,
        ArrayView1D<float, Stride1D.Dense> input3,
        int outer, int inner, int len0, int len1, int len2, int len3)
    {
        int totalConcat = len0 + len1 + len2 + len3;
        int o = idx / (totalConcat * inner);
        int rem = idx - o * totalConcat * inner;
        int c = rem / inner;
        int i = rem - c * inner;

        if (c < len0)
        {
            int srcIdx = o * len0 * inner + c * inner + i;
            output[idx] = (srcIdx >= 0 && (long)srcIdx < input0.Length) ? input0[srcIdx] : 0f;
        }
        else if (c < len0 + len1)
        {
            int cIn = c - len0;
            int srcIdx = o * len1 * inner + cIn * inner + i;
            output[idx] = (srcIdx >= 0 && (long)srcIdx < input1.Length) ? input1[srcIdx] : 0f;
        }
        else if (c < len0 + len1 + len2)
        {
            int cIn = c - len0 - len1;
            int srcIdx = o * len2 * inner + cIn * inner + i;
            output[idx] = (srcIdx >= 0 && (long)srcIdx < input2.Length) ? input2[srcIdx] : 0f;
        }
        else
        {
            int cIn = c - len0 - len1 - len2;
            int srcIdx = o * len3 * inner + cIn * inner + i;
            output[idx] = (srcIdx >= 0 && (long)srcIdx < input3.Length) ? input3[srcIdx] : 0f;
        }
    }

    /// <summary>
    /// Fused 2-input Concat. lenN is the size of input N along the concat axis.
    /// outer = product of dims before concat axis; inner = product of dims after.
    /// </summary>
    public void Concat2(
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> input0,
        ArrayView1D<float, Stride1D.Dense> input1,
        int outer, int inner, int len0, int len1)
    {
        EnsureLoaded2();
        int total = outer * (len0 + len1) * inner;
        if (total <= 0) return;
        _kernel2!(total, output, input0, input1, outer, inner, len0, len1);
    }

    /// <summary>Fused 3-input Concat.</summary>
    public void Concat3(
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> input0,
        ArrayView1D<float, Stride1D.Dense> input1,
        ArrayView1D<float, Stride1D.Dense> input2,
        int outer, int inner, int len0, int len1, int len2)
    {
        EnsureLoaded3();
        int total = outer * (len0 + len1 + len2) * inner;
        if (total <= 0) return;
        _kernel3!(total, output, input0, input1, input2, outer, inner, len0, len1, len2);
    }

    /// <summary>Fused 4-input Concat.</summary>
    public void Concat4(
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> input0,
        ArrayView1D<float, Stride1D.Dense> input1,
        ArrayView1D<float, Stride1D.Dense> input2,
        ArrayView1D<float, Stride1D.Dense> input3,
        int outer, int inner, int len0, int len1, int len2, int len3)
    {
        EnsureLoaded4();
        int total = outer * (len0 + len1 + len2 + len3) * inner;
        if (total <= 0) return;
        _kernel4!(total, output, input0, input1, input2, input3, outer, inner, len0, len1, len2, len3);
    }

    private void EnsureLoaded2()
    {
        _kernel2 ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int, int>(ConcatImpl2);
    }

    private void EnsureLoaded3()
    {
        _kernel3 ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int, int>(ConcatImpl3);
    }

    private void EnsureLoaded4()
    {
        _kernel4 ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int, int, int, int>(ConcatImpl4);
    }

    public void Dispose() { /* No persistent buffers - kernel actions hold no state */ }
}
