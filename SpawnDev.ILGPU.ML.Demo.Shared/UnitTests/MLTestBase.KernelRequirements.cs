using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// For a test of a kernel that needs atomics (counters, min/max, union-find) or writes more than one element per thread (scatter, accumulate into a shared buffer). WebGL has neither: the backend throws for
    /// <c>Atomic.Max</c> and <c>Atomic.CompareExchange</c> and its transform-feedback model drops or relocates the extra stores (measured: a remesh finds 0 band voxels instead of 14920, a tile blend returns NaN). That is a limit of the
    /// backend, documented in SpawnDev.ILGPU's feature matrix (<c>AcceleratorRequirements.RequiresAtomics</c> / <c>RequiresScatterStores</c>), so the test is skipped there with the reason rather than reported red.
    /// </summary>
    protected static void RequireAtomicsAndScatter(Accelerator accelerator)
    {
        if (accelerator.AcceleratorType == AcceleratorType.WebGL)
            throw new UnsupportedTestException("this kernel needs atomics or several stores per thread, which the WebGL backend cannot do (AcceleratorRequirements.RequiresAtomics / RequiresScatterStores)");
    }
}
