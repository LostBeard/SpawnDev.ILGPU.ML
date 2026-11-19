using System.ComponentModel;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// The attention geometry the generated backward kernels take (<see cref="MmaFlashAttnBwd"/>): the same fields, in the same order and types, as the public
/// <c>SpawnDev.ILGPU.ML.Training.BackwardKernels.AttnDims</c>, which sits in the assembly above this one (kernels here cannot name it). The host copies the fields at
/// the launch. Kernel_LayoutTwins_MatchTheirPublicStructs pins the two layouts equal (size and every field's offset), so a field added to one fails the test.
/// PUBLIC because a kernel parameter type must be: SpawnDev.ILGPU emits each kernel's launcher into a dynamic assembly, which cannot see internal types
/// ("TypeLoadException: Access is denied", the first form, 2026-11-19). An implementation detail, hidden from IntelliSense: use BackwardKernels.AttnDims.
/// </summary>
[EditorBrowsable(EditorBrowsableState.Never)]
public struct KernelAttnDims
{
    public int Heads, KvHeads, SeqQ, SeqKV, HeadDim, Causal, Window, KvOffset;
    public float Scale;
}
