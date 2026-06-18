using System;
using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// MoE (mixture-of-experts) specific GPU kernels. Currently the gpt-oss / OpenAI-MoE activation.
/// The routing (router matmul, top-k, softmax-over-selected, weighted combine) lives in
/// <c>MoEOperator</c>; the per-expert gate/up/down matmuls reuse <c>MatMulKernel</c> /
/// <c>FusedDequantMatMul</c> (so MXFP4 experts decode in-register, no f32 expansion).
/// </summary>
public class MoEKernels
{
    private readonly Accelerator _accelerator;
    public MoEKernels(Accelerator accelerator) { _accelerator = accelerator; }

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, float, float>? _swiGluOai;

    /// <summary>
    /// gpt-oss SwiGLU-OAI gated activation, EXACT port of ggml ggml_compute_forward_swiglu_oai_f32:
    /// <code>
    ///   xg = min(gate, limit)
    ///   yu = clamp(up, -limit, limit)
    ///   out = (xg / (1 + exp(alpha * -xg))) * (yu + 1)
    /// </code>
    /// gpt-oss uses alpha = 1.702, limit = 7.0. <paramref name="gate"/> and <paramref name="up"/> are the
    /// two halves of the expert hidden (each length <paramref name="count"/>); writes <paramref name="output"/>.
    /// </summary>
    public void SwiGluOai(
        ArrayView1D<float, Stride1D.Dense> gate,
        ArrayView1D<float, Stride1D.Dense> up,
        ArrayView1D<float, Stride1D.Dense> output,
        int count, float alpha, float limit)
    {
        _swiGluOai ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, float, float>(SwiGluOaiImpl);
        _swiGluOai(count, gate, up, output, alpha, limit);
    }

    private static void SwiGluOaiImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> gate,
        ArrayView1D<float, Stride1D.Dense> up,
        ArrayView1D<float, Stride1D.Dense> output,
        float alpha, float limit)
    {
        float xg = MathF.Min(gate[idx], limit);
        float yu = MathF.Max(-limit, MathF.Min(up[idx], limit));
        float glu = xg / (1f + MathF.Exp(alpha * (-xg)));
        output[idx] = glu * (yu + 1f);
    }
}
