using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// What a CUDA device's compute capability allows ML's hand-written PTX (<c>CudaAsm.Emit</c>) to use. Every kernel that emits an instruction with a minimum
/// architecture is chosen through here, so a device below it takes the portable kernel instead of failing to load (ptxas: "Feature ... requires .target
/// sm_80 or higher", CUDA_ERROR_INVALID_PTX).
/// </summary>
/// <remarks>
/// Found on an RTX 2060 (Turing, sm_75; Seven, 2026-11-01): the tensor-core GEMM / conv / attention kernels (<c>Trellis2/MmaGemm</c>, <c>MmaAttention</c>:
/// <c>mma.sync.aligned.m16n8k16</c> with f16 / bf16 operands and <c>cvt.rn.f16x2.f32</c> / <c>cvt.rn.bf16x2.f32</c>, all sm_80) were chosen on any CUDA
/// device, so every ONNX Conv that reached them failed to load on every RTX 20 / GTX 16 card. The gates only ran on sm_86 / sm_89.
/// </remarks>
public static class CudaCaps
{
    /// <summary>The device's compute capability (major, minor), or null when <paramref name="accelerator"/> is not CUDA.</summary>
    public static (int Major, int Minor)? ComputeCapability(Accelerator accelerator) =>
        accelerator is CudaAccelerator cuda ? (cuda.Architecture.Major, cuda.Architecture.Minor) : null;

    /// <summary>True when compute capability (<paramref name="major"/>, <paramref name="minor"/>) is at least (<paramref name="needMajor"/>, <paramref name="needMinor"/>).</summary>
    public static bool AtLeast(int major, int minor, int needMajor, int needMinor) => major > needMajor || (major == needMajor && minor >= needMinor);

    private static bool AtLeast(Accelerator accelerator, int needMajor, int needMinor) =>
        ComputeCapability(accelerator) is { } cc && AtLeast(cc.Major, cc.Minor, needMajor, needMinor);

    /// <summary>
    /// The tensor-core kernels: <c>mma.sync.aligned.m16n8k16</c> with f16 / bf16 operands and the packed <c>cvt.rn.f16x2.f32</c> / <c>cvt.rn.bf16x2.f32</c>
    /// conversions need sm_80 (Ampere) or later.
    /// </summary>
    public static bool TensorCores(Accelerator accelerator) => AtLeast(accelerator, TensorCoreMajor, TensorCoreMinor);
    /// <summary>The minimum compute capability of <see cref="TensorCores"/>.</summary>
    public const int TensorCoreMajor = 8, TensorCoreMinor = 0;

    /// <summary>
    /// Turing's tensor cores: <c>mma.sync.aligned.m16n8k8</c> with f16 operands and fp32 accumulation exists from sm_75 (no m16n8k16, no bf16, no
    /// <c>cvt.rn.f16x2.f32</c>). <see cref="MmaQ4KGemm"/>'s KernelSm75 is built from those instructions only (RTX 20 / GTX 16; Data, Vertex's RTX 2060, 2026-11-02).
    /// </summary>
    public static bool TuringTensorCores(Accelerator accelerator) => AtLeast(accelerator, TuringTensorCoreMajor, TuringTensorCoreMinor);
    /// <summary>The minimum compute capability of <see cref="TuringTensorCores"/>.</summary>
    public const int TuringTensorCoreMajor = 7, TuringTensorCoreMinor = 5;

    /// <summary><c>dp4a</c> (the int8 dot-product GEMV) needs sm_61 (Pascal) or later.</summary>
    public static bool Dp4a(Accelerator accelerator) => AtLeast(accelerator, Dp4aMajor, Dp4aMinor);
    /// <summary>The minimum compute capability of <see cref="Dp4a"/>.</summary>
    public const int Dp4aMajor = 6, Dp4aMinor = 1;
}
