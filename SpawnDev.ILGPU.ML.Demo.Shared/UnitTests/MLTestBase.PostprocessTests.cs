using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Tests for GPU postprocessing kernels: GrayscaleToRGBA, DepthToColormap, Normalize.
/// These kernels keep data on GPU for zero-copy rendering via CanvasRendererFactory.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task Postprocess_GrayscaleToRGBA_WhiteIsWhite() => await RunTest(async accelerator =>
    {
        var kernel = new Kernels.ImagePostprocessKernel(accelerator);

        // All white (1.0) → should produce (255, 255, 255, 255)
        var gray = new float[] { 1.0f, 0.0f, 0.5f };
        using var grayBuf = accelerator.Allocate1D(gray);
        using var rgbaBuf = accelerator.Allocate1D<int>(3);

        kernel.GrayscaleToRGBA(grayBuf.View, rgbaBuf.View, 3);
        await accelerator.SynchronizeAsync();
        var rgba = await rgbaBuf.CopyToHostAsync<int>(0, 3);

        // White: R=255, G=255, B=255, A=255
        int white = 255 | (255 << 8) | (255 << 16) | (0xFF << 24);
        if (rgba[0] != white)
            throw new Exception($"White pixel: expected {white:X8}, got {rgba[0]:X8}");

        // Black: R=0, G=0, B=0, A=255
        int black = 0 | (0 << 8) | (0 << 16) | (0xFF << 24);
        if (rgba[1] != black)
            throw new Exception($"Black pixel: expected {black:X8}, got {rgba[1]:X8}");

        // Gray (0.5): R~128, G~128, B~128, A=255
        int r = rgba[2] & 0xFF;
        if (r < 126 || r > 130)
            throw new Exception($"Gray pixel R={r}, expected ~128");

        Console.WriteLine("[Postprocess] GrayscaleToRGBA: white/black/gray correct");
    });

    [TestMethod]
    public async Task Postprocess_DepthToColormap_EndpointsCorrect() => await RunTest(async accelerator =>
    {
        var kernel = new Kernels.ImagePostprocessKernel(accelerator);

        // Depth values: min (0) and max (1)
        var depth = new float[] { 0.0f, 1.0f };
        using var depthBuf = accelerator.Allocate1D(depth);
        using var rgbaBuf = accelerator.Allocate1D<int>(2);

        kernel.DepthToColormap(depthBuf.View, rgbaBuf.View, 2, 0f, 1f);
        await accelerator.SynchronizeAsync();
        var rgba = await rgbaBuf.CopyToHostAsync<int>(0, 2);

        // t=0 should be dark purple (plasma start): ~(13, 8, 135)
        int r0 = rgba[0] & 0xFF, g0 = (rgba[0] >> 8) & 0xFF, b0 = (rgba[0] >> 16) & 0xFF;
        if (r0 > 20 || b0 < 120)
            throw new Exception($"Plasma t=0: R={r0},G={g0},B={b0} — expected dark purple");

        // t=1 should be yellow (plasma end): ~(240, 249, 33)
        int r1 = rgba[1] & 0xFF, g1 = (rgba[1] >> 8) & 0xFF, b1 = (rgba[1] >> 16) & 0xFF;
        if (r1 < 220 || g1 < 230 || b1 > 50)
            throw new Exception($"Plasma t=1: R={r1},G={g1},B={b1} — expected yellow");

        Console.WriteLine($"[Postprocess] DepthToColormap: t=0 ({r0},{g0},{b0}), t=1 ({r1},{g1},{b1}) correct");
    });

    [TestMethod]
    public async Task Postprocess_Normalize_RangeCorrect() => await RunTest(async accelerator =>
    {
        var kernel = new Kernels.ImagePostprocessKernel(accelerator);

        var input = new float[] { 10f, 20f, 30f, 40f, 50f };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(5);

        kernel.Normalize(inBuf.View, outBuf.View, 5, 10f, 50f);
        await accelerator.SynchronizeAsync();
        var output = await outBuf.CopyToHostAsync<float>(0, 5);

        // Should map 10→0, 20→0.25, 30→0.5, 40→0.75, 50→1.0
        float[] expected = { 0f, 0.25f, 0.5f, 0.75f, 1.0f };
        for (int i = 0; i < 5; i++)
        {
            if (MathF.Abs(output[i] - expected[i]) > 1e-5f)
                throw new Exception($"Normalize[{i}]={output[i]:F4}, expected {expected[i]:F4}");
        }

        Console.WriteLine("[Postprocess] Normalize: range [10,50]→[0,1] correct");
    });

    /// <summary>
    /// Phase 2 TensorView regression guard. The TensorView&lt;float&gt;+TensorView&lt;int&gt;
    /// overload of <c>DepthToColormapPalette</c> writes to a
    /// <c>MemoryBuffer2D&lt;int, Stride2D.DenseX&gt;.View.BaseView</c> — the path the
    /// depth demo uses for zero-copy canvas presentation. On 2026-05-24 the demo went
    /// flat-blue after migrating this kernel to TensorView while the kernel completed
    /// without throwing. We reverted in <c>2c7dd0f</c> as a workaround. This test
    /// drives BOTH overloads against MemoryBuffer2D-backed output and asserts the
    /// readback matches byte-for-byte. If the TensorView path produces different
    /// data on any backend, the kernel codegen has a real bug we need to root out
    /// (NO workarounds — fix the library).
    /// </summary>
    [TestMethod]
    public async Task Postprocess_DepthToColormapPalette_TensorView_Matches_Legacy() => await RunTest(async accelerator =>
    {
        var kernel = new Kernels.ImagePostprocessKernel(accelerator);

        // 4x4 depth ramp [0, 15] - exercises every plasma colour band.
        const int W = 4, H = 4;
        var depth = new float[W * H];
        for (int i = 0; i < depth.Length; i++) depth[i] = i;

        using var depthBuf = accelerator.Allocate1D(depth);
        using var legacyOut = accelerator.Allocate2DDenseX<int>(new Index2D(W, H));
        using var tvOut = accelerator.Allocate2DDenseX<int>(new Index2D(W, H));

        // Legacy overload: ArrayView1D + count.
        kernel.DepthToColormapPalette(depthBuf.View, legacyOut.View.BaseView,
            W * H, 0f, 15f, Kernels.ImagePostprocessKernel.PalettePlasma);

        // TensorView overload: wraps the same MemoryBuffer2D BaseView in TensorView<int>.
        var depthView = new Tensors.TensorView<float>(depthBuf.View, new[] { H, W });
        var rgbaView = new Tensors.TensorView<int>(tvOut.View.BaseView, new[] { H, W });
        kernel.DepthToColormapPalette(depthView, rgbaView, 0f, 15f,
            Kernels.ImagePostprocessKernel.PalettePlasma);

        // Stage both 2D outputs into 1D buffers (CopyToHostAsync is on MemoryBuffer1D).
        using var legacyStage = accelerator.Allocate1D<int>(W * H);
        using var tvStage = accelerator.Allocate1D<int>(W * H);
        legacyStage.View.CopyFrom(legacyOut.View.BaseView);
        tvStage.View.CopyFrom(tvOut.View.BaseView);
        await accelerator.SynchronizeAsync();

        var legacyPixels = await legacyStage.CopyToHostAsync<int>(0, W * H);
        var tvPixels = await tvStage.CopyToHostAsync<int>(0, W * H);

        int diffs = 0;
        var msg = new System.Text.StringBuilder();
        for (int i = 0; i < W * H; i++)
        {
            if (legacyPixels[i] != tvPixels[i])
            {
                diffs++;
                if (diffs <= 4)
                    msg.Append($"[{i}] legacy=0x{legacyPixels[i]:X8} tv=0x{tvPixels[i]:X8} ");
            }
        }
        if (diffs > 0)
            throw new Exception($"TensorView path diverges from legacy at {diffs}/{W * H} pixels. First mismatches: {msg}");

        // Also assert the values are non-trivial (alpha=0xFF in plasma kernel) - guards
        // against both kernels silently writing zeros.
        bool sawNonZero = false;
        foreach (var p in legacyPixels) if (p != 0) { sawNonZero = true; break; }
        if (!sawNonZero) throw new Exception("Legacy kernel produced all-zero output - kernel didn't run?");

        Console.WriteLine($"[Postprocess] TensorView matches legacy across {W * H} pixels. " +
            $"Sample: legacy[0]=0x{legacyPixels[0]:X8}, legacy[{W * H - 1}]=0x{legacyPixels[W * H - 1]:X8}");
    });

    [TestMethod]
    public async Task Postprocess_NCHWToRGBA_PacksCorrectly() => await RunTest(async accelerator =>
    {
        var kernel = new Kernels.ImagePostprocessKernel(accelerator);

        // 1x1 image, 3 channels: R=100, G=150, B=200
        var nchw = new float[] { 100f, 150f, 200f }; // [3, 1, 1]
        using var nchwBuf = accelerator.Allocate1D(nchw);
        using var rgbaBuf = accelerator.Allocate1D<int>(1);

        kernel.NCHWToRGBA(nchwBuf.View, rgbaBuf.View, 1, 1);
        await accelerator.SynchronizeAsync();
        var rgba = await rgbaBuf.CopyToHostAsync<int>(0, 1);

        int r = rgba[0] & 0xFF;
        int g = (rgba[0] >> 8) & 0xFF;
        int b = (rgba[0] >> 16) & 0xFF;
        int a = (rgba[0] >> 24) & 0xFF;

        if (r != 100 || g != 150 || b != 200 || a != 255)
            throw new Exception($"NCHWToRGBA: R={r},G={g},B={b},A={a} — expected 100,150,200,255");

        Console.WriteLine("[Postprocess] NCHWToRGBA: channels packed correctly");
    });
}
