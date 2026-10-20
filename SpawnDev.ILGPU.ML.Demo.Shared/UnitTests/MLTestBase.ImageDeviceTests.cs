using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// The picture work a page used to do in managed loops, now on the device: applying a segmentation mask to a picture (<see cref="BackgroundRemovalPipeline.ApplyMaskAsync"/>) and asking about alpha (<see cref="RgbaStats"/>),
/// each against the host code it replaced.
/// </summary>
public abstract partial class MLTestBase
{
    private static void Id_Expect(bool ok, string what) { if (!ok) throw new Exception(what); }

    // the host implementation the pipeline used before: sigmoid when the values look like logits, bilinear resize of the mask, alpha = round(mask * 255)
    private static int[] Id_HostApply(int[] rgba, float[] raw, int maskW, int maskH, int w, int h)
    {
        var mask = (float[])raw.Clone();
        if (mask.Any(v => v < -0.1f || v > 1.1f))
            for (int i = 0; i < mask.Length; i++) mask[i] = 1f / (1f + MathF.Exp(-mask[i]));
        var result = new int[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                float srcX = (x + 0.5f) * maskW / w - 0.5f, srcY = (y + 0.5f) * maskH / h - 0.5f;
                int x0 = Math.Clamp((int)srcX, 0, maskW - 1), y0 = Math.Clamp((int)srcY, 0, maskH - 1);
                int x1 = Math.Clamp(x0 + 1, 0, maskW - 1), y1 = Math.Clamp(y0 + 1, 0, maskH - 1);
                float fx = srcX - x0, fy = srcY - y0;
                float v = mask[y0 * maskW + x0] * (1 - fx) * (1 - fy) + mask[y0 * maskW + x1] * fx * (1 - fy) + mask[y1 * maskW + x0] * (1 - fx) * fy + mask[y1 * maskW + x1] * fx * fy;
                int a = Math.Clamp((int)(v * 255f + 0.5f), 0, 255);
                result[y * w + x] = (rgba[y * w + x] & 0x00FFFFFF) | (a << 24);
            }
        return result;
    }

    private async Task Id_CheckApply(Accelerator accelerator, bool logits, int w, int h, int maskW, int maskH)
    {
        RequireAtomicsAndScatter(accelerator);          // the mask's range check is an Atomic.Max
        var rnd = new Random(11);
        var rgba = new int[w * h];
        for (int i = 0; i < rgba.Length; i++) rgba[i] = rnd.Next() | (rnd.Next(256) << 24);
        var mask = new float[maskW * maskH];
        for (int y = 0; y < maskH; y++)
            for (int x = 0; x < maskW; x++)
            {
                double d = Math.Sqrt((x - maskW / 2.0) * (x - maskW / 2.0) + (y - maskH / 2.0) * (y - maskH / 2.0)) / (maskW / 2.0);
                mask[y * maskW + x] = logits ? (float)((0.7 - d) * 12 + rnd.NextDouble() * 0.2) : (float)Math.Clamp(1.2 - d * 1.2 + rnd.NextDouble() * 0.05, 0, 1);
            }
        var expect = Id_HostApply(rgba, mask, maskW, maskH, w, h);
        using var rb = accelerator.Allocate1D(rgba); using var mb = accelerator.Allocate1D(mask);
        using var got = await BackgroundRemovalPipeline.ApplyMaskAsync(accelerator, rb.View, mb.View, maskW, maskH, w, h);
        var gotPixels = await got.View.CopyToHostAsync();
        int colourWrong = 0, alphaOff = 0, alphaBig = 0;
        for (int i = 0; i < gotPixels.Length; i++)
        {
            if ((gotPixels[i] & 0x00FFFFFF) != (rgba[i] & 0x00FFFFFF)) colourWrong++;
            int da = Math.Abs(((gotPixels[i] >> 24) & 255) - ((expect[i] >> 24) & 255));
            if (da > 0) alphaOff++; if (da > 1) alphaBig++;
        }
        Id_Expect(colourWrong == 0, $"{(logits ? "logits" : "probabilities")} {maskW}x{maskH} -> {w}x{h}: {colourWrong} pixels lost their colour");
        // the GPU's exp differs from the host's in the last bits: an alpha may land one step either side of a rounding boundary, never more
        Id_Expect(alphaBig == 0 && alphaOff <= gotPixels.Length / 200, $"{(logits ? "logits" : "probabilities")} {maskW}x{maskH} -> {w}x{h}: {alphaOff} alphas differ from the host's ({alphaBig} by more than one)");
    }

    [TestMethod]
    public async Task ImageDevice_ApplyMask_Probabilities_MatchesTheHostCode() => await RunTest(async accelerator =>
    {
        await Id_CheckApply(accelerator, false, 200, 150, 64, 64);
        await Id_CheckApply(accelerator, false, 96, 96, 96, 96);
    });

    [TestMethod]
    public async Task ImageDevice_ApplyMask_Logits_MatchesTheHostCode() => await RunTest(async accelerator =>
    {
        await Id_CheckApply(accelerator, true, 257, 130, 128, 128);
        await Id_CheckApply(accelerator, true, 64, 64, 128, 128);
    });

    [TestMethod]
    public async Task ImageDevice_Compositor_MatchesTheHostLoops() => await RunTest(async accelerator =>
    {
        int w = 83, h = 61;
        var rnd = new Random(8);
        var cut = new int[w * h];
        for (int i = 0; i < cut.Length; i++) cut[i] = rnd.Next(1 << 24) | (rnd.Next(256) << 24);
        // host: the box blur and the composite the page ran (alpha from the 8-bit mask)
        var tmp = new int[w * h]; var blurred = new int[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int r = 0, g = 0, b = 0;
                for (int k = -2; k <= 2; k++) { int p = cut[y * w + Math.Clamp(x + k, 0, w - 1)]; r += p & 255; g += (p >> 8) & 255; b += (p >> 16) & 255; }
                tmp[y * w + x] = (r / 5) | ((g / 5) << 8) | ((b / 5) << 16);
            }
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int r = 0, g = 0, b = 0;
                for (int k = -2; k <= 2; k++) { int p = tmp[Math.Clamp(y + k, 0, h - 1) * w + x]; r += p & 255; g += (p >> 8) & 255; b += (p >> 16) & 255; }
                blurred[y * w + x] = (r / 5) | ((g / 5) << 8) | ((b / 5) << 16);
            }
        int Over(int p, int bg)
        {
            float a = ((p >> 24) & 255) / 255f;
            int r = (int)((p & 255) * a + (bg & 255) * (1 - a) + 0.5f), g = (int)(((p >> 8) & 255) * a + ((bg >> 8) & 255) * (1 - a) + 0.5f), b = (int)(((p >> 16) & 255) * a + ((bg >> 16) & 255) * (1 - a) + 0.5f);
            return (r & 255) | ((g & 255) << 8) | ((b & 255) << 16) | (255 << 24);
        }
        using var cb = accelerator.Allocate1D(cut);
        using var white = await ImageCompositor.OverColorAsync(accelerator, cb.View, w, h, 0xFFFFFF);
        using var blur = await ImageCompositor.OverBlurAsync(accelerator, cb.View, w, h);
        var gotWhite = await white.View.CopyToHostAsync(); var gotBlur = await blur.View.CopyToHostAsync();
        int badWhite = 0, badBlur = 0;
        for (int i = 0; i < cut.Length; i++)
        {
            if (gotWhite[i] != Over(cut[i], 0xFFFFFF)) badWhite++;
            if (gotBlur[i] != Over(cut[i], blurred[i])) badBlur++;
        }
        Id_Expect(badWhite <= cut.Length / 500, $"{badWhite} pixels differ from the host's white composite");
        Id_Expect(badBlur <= cut.Length / 500, $"{badBlur} pixels differ from the host's blurred composite");
    });

    [TestMethod]
    public async Task ImageDevice_RgbaStats_MatchTheHostLoops() => await RunTest(async accelerator =>
    {
        RequireAtomicsAndScatter(accelerator);          // the counters are atomics
        int w = 101, h = 77;
        var rnd = new Random(2);
        var px = new int[w * h];
        for (int i = 0; i < px.Length; i++) px[i] = (rnd.Next(100) < 30 ? 255 : rnd.Next(256)) << 24 | rnd.Next(1 << 24);
        // host: the loops the page ran
        int ring = 0, hit = 0;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                if (x >= 2 && y >= 2 && x < w - 2 && y < h - 2) continue;
                ring++;
                if (((px[y * w + x] >> 24) & 255) > 127) hit++;
            }
        using var buf = accelerator.Allocate1D(px);
        double touch = await RgbaStats.BorderTouchAsync(accelerator, buf.View, w, h);
        Id_Expect(Math.Abs(touch - (double)hit / ring) < 1e-9, $"border touch {touch:F6}, host {(double)hit / ring:F6}");
        Id_Expect(!await RgbaStats.IsFullyOpaqueAsync(accelerator, buf.View, w, h), "a picture with translucent pixels reported as opaque");
        for (int i = 0; i < px.Length; i++) px[i] |= unchecked((int)0xFF000000);
        using var opaque = accelerator.Allocate1D(px);
        Id_Expect(await RgbaStats.IsFullyOpaqueAsync(accelerator, opaque.View, w, h), "an opaque picture reported as translucent");
        Id_Expect(await RgbaStats.BorderTouchAsync(accelerator, opaque.View, w, h) == 1.0, "an opaque picture touches the whole frame");
    });
}
