using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// One operator input in mixed-precision (F16-activation) execution: either a low-precision activation
/// (<see cref="HalfTensor"/>) the executor is holding fp16, or an fp32 <see cref="Tensor"/> (a weight,
/// bias, or constant that is NOT part of the activation working set). Operators pick per-input which they
/// need: e.g. Conv reads its activation input as <see cref="Half"/> and its weight/bias as <see cref="Float"/>.
/// </summary>
public readonly struct PrecisionAwareInput
{
    /// <summary>The low-precision activation, when this input is stored fp16. Null if fp32.</summary>
    public readonly HalfTensor? Half;
    /// <summary>The fp32 tensor (weight/bias/const), when this input is stored fp32. Null if fp16.</summary>
    public readonly Tensor? Float;

    public PrecisionAwareInput(HalfTensor half) { Half = half; Float = null; }
    public PrecisionAwareInput(Tensor f) { Half = null; Float = f; }

    /// <summary>True when this input is a low-precision activation.</summary>
    public bool IsHalf => Half != null;
    /// <summary>Shape of the underlying tensor (half or fp32).</summary>
    public int[] Shape => Half?.Shape ?? Float!.Shape;
    /// <summary>Element count of the underlying tensor.</summary>
    public int ElementCount => Half?.ElementCount ?? Float!.ElementCount;
}

/// <summary>
/// Approach-(i) opt-in: an operator that can execute reading its low-precision (fp16) activation inputs
/// DIRECTLY and writing a low-precision output DIRECTLY, with NO fp32 temp buffers (compute still accumulates
/// in fp32 inside the kernel). Implemented by the memory-dominant SD-VAE ops (Conv, InstanceNorm, Sigmoid,
/// Mul, Add, Relu). The executor calls <see cref="TryExecuteHalf"/> when <c>ActivationDtype == F16</c>; the
/// op returns <c>false</c> for any case it does not handle (broadcast, non-fp16 input, unsupported layout),
/// and the executor falls back to the fp32 convert-around-node path. This is what actually cuts the activation
/// working set — the convert-around-node path keeps an fp32 temp live next to the fp32 output, so it does not.
///
/// Plan: Plans/fp16-bf16-mixed-precision-activations-2026-06-16.md.
/// </summary>
public interface IPrecisionAwareOperator
{
    /// <summary>
    /// Try to run this op reading low-p inputs and writing the pre-rented low-p <paramref name="output"/>,
    /// no fp32 temp. Return <c>false</c> (without touching <paramref name="output"/>) to make the executor
    /// fall back to the fp32 convert-around-node path. <paramref name="ctx"/> carries only attributes/format
    /// (its Inputs/Outputs are not populated on this path — read tensors from <paramref name="inputs"/>).
    /// </summary>
    bool TryExecuteHalf(OnnxOpContext ctx, PrecisionAwareInput[] inputs, HalfTensor output, PrecisionAwareKernels pak);
}
