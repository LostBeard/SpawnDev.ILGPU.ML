using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// Routes a NATIVE low-precision weight <see cref="Tensor"/> (carrying a real <see cref="TensorDataType"/>) to
/// the matching generic kernel overload, by its <see cref="Tensor.DType"/>. This is the dispatch layer of the
/// no-needless-lowp-&gt;f32-conversion work: a weight stays native in GPU memory and is converted to float
/// in-register at the arithmetic (inside the generic <c>...LowPWeight&lt;T&gt;</c> kernel via
/// <c>ILGPU.PrecisionConvert</c>) - never a managed f32 temp buffer.
///
/// Replaces the old <c>if (w.IsHalf)</c> bool branch, which could only express fp16-or-fp32 and so forced
/// every other low-p weight (bf16 / FP8 / FP4) to be unpacked to f32. The concrete <c>w.AsView&lt;T&gt;()</c>
/// per case keeps the kernel instantiation static (no reflection); all four low-p types implement
/// <c>INumber&lt;T&gt;</c> so the generic kernel accepts them.
/// </summary>
internal static class LowPWeightDispatch
{
    /// <summary>True when <paramref name="t"/> is a native low-precision weight (not fp32) - i.e. it must be
    /// routed through one of the <c>...LowPWeight</c> kernels rather than the fp32 path.</summary>
    public static bool IsLowP(Tensor t) => t.DType != TensorDataType.Float32;

    /// <summary>MatMul C[M,N] = A[M,K] (fp32) × B[K,N] (native low-p), fp32 accumulate.</summary>
    public static void MatMul(MatMulKernel mm,
        ArrayView1D<float, Stride1D.Dense> a, Tensor b, ArrayView1D<float, Stride1D.Dense> c,
        int M, int K, int N)
    {
        switch (b.DType)
        {
            case TensorDataType.Float16: mm.MatMulLowPWeight(a, b.AsView<global::ILGPU.Half>(), c, M, K, N); break;
            case TensorDataType.BFloat16: mm.MatMulLowPWeight(a, b.AsView<BFloat16>(), c, M, K, N); break;
            case TensorDataType.Float8E4M3: mm.MatMulLowPWeight(a, b.AsView<Float8E4M3>(), c, M, K, N); break;
            case TensorDataType.Float8E5M2: mm.MatMulLowPWeight(a, b.AsView<Float8E5M2>(), c, M, K, N); break;
            default: throw Unexpected(b);
        }
    }

    /// <summary>MatMul C[M,N] = A[M,K] (fp32) × B[N,K]^T (native low-p, transB layout), fp32 accumulate.</summary>
    public static void MatMulTransB(MatMulKernel mm,
        ArrayView1D<float, Stride1D.Dense> a, Tensor b, ArrayView1D<float, Stride1D.Dense> c,
        int M, int K, int N)
    {
        switch (b.DType)
        {
            case TensorDataType.Float16: mm.MatMulLowPWeightTransB(a, b.AsView<global::ILGPU.Half>(), c, M, K, N); break;
            case TensorDataType.BFloat16: mm.MatMulLowPWeightTransB(a, b.AsView<BFloat16>(), c, M, K, N); break;
            case TensorDataType.Float8E4M3: mm.MatMulLowPWeightTransB(a, b.AsView<Float8E4M3>(), c, M, K, N); break;
            case TensorDataType.Float8E5M2: mm.MatMulLowPWeightTransB(a, b.AsView<Float8E5M2>(), c, M, K, N); break;
            default: throw Unexpected(b);
        }
    }

    /// <summary>Batched MatMul C[b] = A[b] × B (native low-p, shared weight), fp32 accumulate.</summary>
    public static void BatchedMatMul(MatMulKernel mm,
        ArrayView1D<float, Stride1D.Dense> a, Tensor b, ArrayView1D<float, Stride1D.Dense> c,
        int batch, int M, int K, int N)
    {
        switch (b.DType)
        {
            case TensorDataType.Float16: mm.BatchedMatMulLowPWeight(a, b.AsView<global::ILGPU.Half>(), c, batch, M, K, N); break;
            case TensorDataType.BFloat16: mm.BatchedMatMulLowPWeight(a, b.AsView<BFloat16>(), c, batch, M, K, N); break;
            case TensorDataType.Float8E4M3: mm.BatchedMatMulLowPWeight(a, b.AsView<Float8E4M3>(), c, batch, M, K, N); break;
            case TensorDataType.Float8E5M2: mm.BatchedMatMulLowPWeight(a, b.AsView<Float8E5M2>(), c, batch, M, K, N); break;
            default: throw Unexpected(b);
        }
    }

    /// <summary>Conv2D NCHW (asymmetric ONNX pads) with a native low-p filter weight, fp32/fp64 accumulate.</summary>
    public static void Conv2DPadded(Conv2DKernel conv,
        ArrayView1D<float, Stride1D.Dense> input, Tensor weight,
        ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW, int outC, int kH, int kW,
        int stride, int padTop, int padLeft, int padBottom, int padRight,
        int dilationH = 1, int dilationW = 1)
    {
        switch (weight.DType)
        {
            case TensorDataType.Float16:
                conv.ForwardPaddedLowPWeight(input, weight.AsView<global::ILGPU.Half>(), bias, output, inC, inH, inW, outC, kH, kW, stride, padTop, padLeft, padBottom, padRight, dilationH, dilationW); break;
            case TensorDataType.BFloat16:
                conv.ForwardPaddedLowPWeight(input, weight.AsView<BFloat16>(), bias, output, inC, inH, inW, outC, kH, kW, stride, padTop, padLeft, padBottom, padRight, dilationH, dilationW); break;
            case TensorDataType.Float8E4M3:
                conv.ForwardPaddedLowPWeight(input, weight.AsView<Float8E4M3>(), bias, output, inC, inH, inW, outC, kH, kW, stride, padTop, padLeft, padBottom, padRight, dilationH, dilationW); break;
            case TensorDataType.Float8E5M2:
                conv.ForwardPaddedLowPWeight(input, weight.AsView<Float8E5M2>(), bias, output, inC, inH, inW, outC, kH, kW, stride, padTop, padLeft, padBottom, padRight, dilationH, dilationW); break;
            default: throw Unexpected(weight);
        }
    }

    /// <summary>Fused linear Output[M,N] = Activation(Input[M,K] (fp32) × W[K,N] (native low-p) + Bias[N]).
    /// W stays native (no f32 upcast); converted to float in-register inside the generic kernel. Same [K,N]
    /// weight layout as the fp32 FusedLinear (FuseLinearLayers excludes transB).</summary>
    public static void FusedLinear(FusedLinearKernel fl,
        ArrayView1D<float, Stride1D.Dense> input, Tensor w,
        ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> output,
        int M, int K, int N, FusedActivation activation)
    {
        switch (w.DType)
        {
            case TensorDataType.Float16: fl.ForwardLowP(input, w.AsView<global::ILGPU.Half>(), bias, output, M, K, N, activation); break;
            case TensorDataType.BFloat16: fl.ForwardLowP(input, w.AsView<BFloat16>(), bias, output, M, K, N, activation); break;
            case TensorDataType.Float8E4M3: fl.ForwardLowP(input, w.AsView<Float8E4M3>(), bias, output, M, K, N, activation); break;
            case TensorDataType.Float8E5M2: fl.ForwardLowP(input, w.AsView<Float8E5M2>(), bias, output, M, K, N, activation); break;
            default: throw Unexpected(w);
        }
    }

    private static InvalidOperationException Unexpected(Tensor t) => new(
        $"Low-p weight dispatch: tensor '{t.Name ?? "?"}' has DType {t.DType}, which is not a supported native " +
        "low-precision weight type (Float16 / BFloat16 / Float8E4M3 / Float8E5M2).");
}
