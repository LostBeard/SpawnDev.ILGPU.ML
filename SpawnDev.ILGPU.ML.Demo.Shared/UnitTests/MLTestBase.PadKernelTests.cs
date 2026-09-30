using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// PadKernel at style-mosaic's first node: [1,3,224,224] reflect-padded by 4 (H, W) to [1,3,232,232], against a CPU
    /// reference, for all three modes. 2026-09-30: this node hung the GPU on WebGPU (DXGI_ERROR_DEVICE_HUNG) and took the
    /// whole WebGPU lane down with it; no test ran the kernel directly.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task PadKernel_StyleMosaicShape_AllModesMatchCpu() => await RunTest(async accelerator =>
    {
        int[] shape = [1, 3, 224, 224];
        int[] pads = [0, 0, 4, 4, 0, 0, 4, 4];
        int[] outShape = [1, 3, 232, 232];
        int n = 3 * 224 * 224, no = 3 * 232 * 232;
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (i * 37 % 1009) * 0.5f;
        using var inBuf = accelerator.Allocate1D(x);
        var pad = new SpawnDev.ILGPU.ML.Kernels.PadKernel(accelerator);
        foreach (int mode in new[] { 2, 1, 0 })
        {
            using var outBuf = accelerator.Allocate1D<float>(no);
            pad.Forward(inBuf.View, outBuf.View, shape, pads, mode, -7f);
            await accelerator.SynchronizeAsync();
            var got = await outBuf.CopyToHostAsync<float>();
            int bad = 0; string first = "";
            for (int c = 0; c < 3; c++)
                for (int h = 0; h < 232; h++)
                    for (int w = 0; w < 232; w++)
                    {
                        int sh = h - 4, sw = w - 4;
                        float want;
                        if (mode == 0 && (sh < 0 || sh >= 224 || sw < 0 || sw >= 224)) want = -7f;
                        else
                        {
                            sh = Src(sh, 224, mode); sw = Src(sw, 224, mode);
                            want = x[(c * 224 + sh) * 224 + sw];
                        }
                        float g = got[(c * 232 + h) * 232 + w];
                        if (g != want && bad++ == 0) first = $"[{c},{h},{w}] = {g}, expected {want}";
                    }
            if (bad != 0) throw new Exception($"mode {mode}: {bad} of {no} wrong; first {first}");
        }
    });

    private static int Src(int s, int dim, int mode)
    {
        if (s >= 0 && s < dim) return s;
        if (mode == 1) return s < 0 ? 0 : dim - 1;
        if (s < 0) s = -s;
        if (s >= dim) s = 2 * (dim - 1) - s;
        return s;
    }
}
