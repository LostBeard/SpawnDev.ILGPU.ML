using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Backend;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.UnitTests;

/// <summary>
/// Runs ML kernel tests on the WebGPU backend.
/// WebGPU uses the TILED MatMul (shared memory + barriers).
/// If tests pass on CPU but fail here, the tiled MatMul WGSL codegen is the bug.
///
/// Test methods are declared here directly because UnitTestsView discovers
/// via DeclaredOnly reflection — inherited methods from the abstract base aren't found.
/// Each method delegates to the shared implementation in MLTestBase.
/// </summary>
public class WebGPUTests : MLTestBase
{
    protected override string BackendName => "WebGPU";

    protected override async Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
    {
        var builder = Context.Create();
        await builder.WebGPU();
        var context = builder.ToContext();
        var devices = context.GetWebGPUDevices();
        if (devices.Count == 0)
            throw new UnsupportedTestException("No WebGPU devices found");
        var accelerator = await devices[0].CreateAcceleratorAsync(context);
        return (context, accelerator);
    }

    // ── MatMul tests ──
    [TestMethod] public new async Task MatMul_QkvDimensions() => await base.MatMul_QkvDimensions();
    [TestMethod] public new async Task MatMul_MlpFc2Dimensions() => await base.MatMul_MlpFc2Dimensions();
    [TestMethod] public new async Task MatMul_BatchedAttentionScores() => await base.MatMul_BatchedAttentionScores();
    [TestMethod] public new async Task MatMul_SmallNonAligned() => await base.MatMul_SmallNonAligned();

    // ── Kernel tests ──
    [TestMethod] public new async Task LayerNorm_Dav3Dimensions() => await base.LayerNorm_Dav3Dimensions();
    [TestMethod] public new async Task GELU_MatchesCpuErf() => await base.GELU_MatchesCpuErf();
    [TestMethod] public new async Task BroadcastMul_LayerScale() => await base.BroadcastMul_LayerScale();
    [TestMethod] public new async Task Softmax_AttentionDimensions() => await base.Softmax_AttentionDimensions();
    [TestMethod] public new async Task TransposeLastTwo_RoundTrip() => await base.TransposeLastTwo_RoundTrip();
}
