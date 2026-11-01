using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// The CUDA kernels that emit PTX with a minimum architecture are chosen by the device's compute capability (<see cref="CudaCaps"/>): the tensor-core
    /// GEMM / conv / attention (mma m16n8k16 f16 / bf16, cvt f16x2 / bf16x2) from sm_80, dp4a from sm_61. An RTX 2060 (sm_75) took the tensor-core conv and
    /// failed to load it (CUDA_ERROR_INVALID_PTX, Seven 2026-11-01); this PC's card is sm_89, so the thresholds are held here and the device's own choice is
    /// checked against its capability; the sm_75 card itself is Vertex's GPU run on Aubs' PC.
    /// </summary>
    [TestMethod]
    public async Task CudaCaps_ChoosesKernelsByComputeCapability() => await RunTest(async accelerator =>
    {
        var cases = new (int Major, int Minor, bool Tensor, bool Dp4a)[]
        {
            (5, 2, false, false),   // Maxwell
            (6, 0, false, false),   // Pascal P100: no dp4a
            (6, 1, false, true),    // Pascal GTX 10
            (7, 0, false, true),    // Volta
            (7, 5, false, true),    // Turing: RTX 20, GTX 16 (the RTX 2060)
            (8, 0, true, true),     // Ampere A100
            (8, 6, true, true),     // Ampere RTX 30
            (8, 9, true, true),     // Ada RTX 40
            (9, 0, true, true),     // Hopper
            (12, 0, true, true),    // Blackwell RTX 50
        };
        var wrong = cases.Where(c => CudaCaps.AtLeast(c.Major, c.Minor, CudaCaps.TensorCoreMajor, CudaCaps.TensorCoreMinor) != c.Tensor
            || CudaCaps.AtLeast(c.Major, c.Minor, CudaCaps.Dp4aMajor, CudaCaps.Dp4aMinor) != c.Dp4a).ToList();
        var cc = CudaCaps.ComputeCapability(accelerator);
        bool deviceOk = cc is { } d
            ? CudaCaps.TensorCores(accelerator) == (d.Major >= 8) && CudaCaps.Dp4a(accelerator) == (d.Major > 6 || (d.Major == 6 && d.Minor >= 1))
            : !CudaCaps.TensorCores(accelerator) && !CudaCaps.Dp4a(accelerator);
        var report = $"{accelerator.AcceleratorType}: {cases.Length - wrong.Count}/{cases.Length} architectures choose as they must; this device {(cc is { } x ? $"sm_{x.Major}{x.Minor}" : "not CUDA")}: tensor cores {CudaCaps.TensorCores(accelerator)}, dp4a {CudaCaps.Dp4a(accelerator)}";
        Console.WriteLine("[CudaCaps] " + report);
        if (wrong.Count > 0 || !deviceOk) throw new Exception(report + (wrong.Count > 0 ? " WRONG: " + string.Join(", ", wrong.Select(w => $"sm_{w.Major}{w.Minor}")) : " device choice inconsistent"));
        await Task.CompletedTask;
        return report;
    });
}
