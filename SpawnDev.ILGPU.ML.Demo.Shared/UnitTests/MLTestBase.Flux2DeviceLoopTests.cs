using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// The FLUX.2 steps that used to run in host loops, now kernels (<see cref="Flux2Kernels"/>): the Euler step, unpatchifying latents for the VAE, patchifying a reference, and the tiled decode's gather, blend and finish. Each against the host
/// code it replaced, on random data (no weights needed).
/// </summary>
public abstract partial class MLTestBase
{
    private static void F2d_Expect(bool ok, string what) { if (!ok) throw new Exception(what); }

    private static float[] F2d_Random(int n, int seed, float scale = 1f, float shift = 0f)
    {
        var r = new Random(seed); var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() * 2 - 1) * scale + shift;
        return a;
    }

    private static double F2d_MaxDiff(float[] a, float[] b)
    {
        F2d_Expect(a.Length == b.Length, $"length {a.Length} vs {b.Length}");
        double worst = 0; for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    [TestMethod]
    public async Task Flux2Device_EulerStep_AddsTheScaledPrediction() => await RunTest(async accelerator =>
    {
        int n = 5000, count = 4096;
        var lat = F2d_Random(n, 1); var pred = F2d_Random(n, 2); float dt = -0.37f;
        var expect = (float[])lat.Clone();
        for (int i = 0; i < count; i++) expect[i] += dt * pred[i];
        using var lb = accelerator.Allocate1D(lat); using var pb = accelerator.Allocate1D(pred);
        new Flux2Kernels(accelerator).Euler(lb.View, pb.View, dt, count);
        var got = await lb.View.CopyToHostAsync();
        F2d_Expect(F2d_MaxDiff(got, expect) < 1e-6, $"Euler step differs by {F2d_MaxDiff(got, expect):G3} (elements past the count must stay)");
    });

    [TestMethod]
    public async Task Flux2Device_ToVaeInputAndRefTokens_MatchTheHostLoops() => await RunTest(async accelerator =>
    {
        int rows = 6, cols = 10, outH = rows * 2, outW = cols * 2;
        var mean = F2d_Random(128, 3, 0.5f); var std = F2d_Random(128, 4, 0.3f, 1f);
        var packed = F2d_Random(rows * cols * 128, 5);
        var expect = new float[32 * outH * outW];
        for (int cell = 0; cell < rows * cols; cell++)
        {
            int y = cell / cols, x = cell % cols;
            for (int ch = 0; ch < 128; ch++)
            {
                int c = ch / 4, py = (ch / 2) % 2, px = ch % 2;
                expect[(c * outH + 2 * y + py) * outW + 2 * x + px] = packed[cell * 128 + ch] * std[ch] + mean[ch];
            }
        }
        using var pb = accelerator.Allocate1D(packed); using var mb = accelerator.Allocate1D(mean); using var sb = accelerator.Allocate1D(std);
        using var ob = accelerator.Allocate1D<float>(expect.Length);
        var k = new Flux2Kernels(accelerator);
        k.ToVaeInput(pb.View, mb.View, sb.View, ob.View, rows, cols);
        F2d_Expect(F2d_MaxDiff(await ob.View.CopyToHostAsync(), expect) < 1e-6, "unpatchified latents differ from the host's");

        // a reference: the encoder's [32, lh, lw] to tokens, the other way round
        int lh = outH, lw = outW;
        var lat = F2d_Random(32 * lh * lw, 6);
        var tokens = new float[rows * cols * 128];
        for (int c = 0; c < 32; c++)
            for (int y = 0; y < rows; y++)
                for (int x = 0; x < cols; x++)
                    for (int py = 0; py < 2; py++)
                        for (int px = 0; px < 2; px++)
                        {
                            int ch = c * 4 + py * 2 + px;
                            tokens[(y * cols + x) * 128 + ch] = (lat[(c * lh + 2 * y + py) * lw + 2 * x + px] - mean[ch]) / std[ch];
                        }
        using var lb = accelerator.Allocate1D(lat); using var tb = accelerator.Allocate1D<float>(tokens.Length);
        k.RefTokens(lb.View, mb.View, sb.View, tb.View, lh, lw);
        F2d_Expect(F2d_MaxDiff(await tb.View.CopyToHostAsync(), tokens) < 1e-5, "reference tokens differ from the host's");
    });

    [TestMethod]
    public async Task Flux2Device_TiledBlend_MatchesTheHostBlend() => await RunTest(async accelerator =>
    {
        // a picture of 10 x 12 latent cells in tiles of 8 with an overlap of 3: the old host code, with a stand-in for the VAE (a function of the tile and the pixel)
        int T = 8, O = 3, stride = T - O, tilePx = T * 8, ramp = O * 8, latH = 10, latW = 12, height = latH * 8, width = latW * 8;
        List<int> Starts(int len) { var l = new List<int>(); if (len <= T) { l.Add(0); return l; } for (int x = 0; x + T < len; x += stride) l.Add(x); l.Add(len - T); return l; }
        var ys = Starts(latH); var xs = Starts(latW);
        float Feather(int i, int n, bool first, bool last)
        {
            float w = 1f;
            if (!first && i < ramp) w = Math.Min(w, (i + 1f) / (ramp + 1f));
            if (!last && n - 1 - i < ramp) w = Math.Min(w, (n - i) / (ramp + 1f));
            return w;
        }
        var vaeInput = F2d_Random(32 * latH * latW, 7);
        var tiles = new List<float[]>();
        var sum = new float[3 * height * width]; var wsum = new float[height * width];
        int ti = 0;
        var gatheredOk = true;
        using var src = accelerator.Allocate1D(vaeInput); using var tileIn = accelerator.Allocate1D<float>(32 * T * T);
        using var sumB = accelerator.Allocate1D<float>(sum.Length); using var wsumB = accelerator.Allocate1D<float>(wsum.Length);
        sumB.MemSetToZero(); wsumB.MemSetToZero();
        var k = new Flux2Kernels(accelerator);
        var tileBufs = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
        foreach (int y0 in ys)
            foreach (int x0 in xs)
            {
                // the gather: replicate the last row / column past the edge
                var gathered = new float[32 * T * T];
                for (int c = 0; c < 32; c++)
                    for (int y = 0; y < T; y++)
                        for (int x = 0; x < T; x++)
                            gathered[(c * T + y) * T + x] = vaeInput[(c * latH + Math.Min(y0 + y, latH - 1)) * latW + Math.Min(x0 + x, latW - 1)];
                k.TileGather(src.View, tileIn.View, latH, latW, y0, x0, T);
                gatheredOk &= F2d_MaxDiff(await tileIn.View.CopyToHostAsync(), gathered) == 0;
                // the stand-in decoder output for this tile
                var tile = F2d_Random(3 * tilePx * tilePx, 100 + ti++);
                var tb = accelerator.Allocate1D(tile); tileBufs.Add(tb);
                int px0 = x0 * 8, py0 = y0 * 8;
                bool firstY = y0 == 0, lastY = latH <= T || y0 == latH - T, firstX = x0 == 0, lastX = latW <= T || x0 == latW - T;
                for (int y = 0; y < tilePx && py0 + y < height; y++)
                {
                    float wy = Feather(y, tilePx, firstY, lastY);
                    for (int x = 0; x < tilePx && px0 + x < width; x++)
                    {
                        float w = wy * Feather(x, tilePx, firstX, lastX);
                        int o = (py0 + y) * width + px0 + x;
                        wsum[o] += w;
                        for (int c = 0; c < 3; c++) sum[c * height * width + o] += w * tile[(c * tilePx + y) * tilePx + x];
                    }
                }
                int flags = (firstY ? 1 : 0) | (lastY ? 2 : 0) | (firstX ? 4 : 0) | (lastX ? 8 : 0);
                k.TileBlend(tb.View, sumB.View, wsumB.View, tilePx, width, height, px0, py0, ramp, flags);
            }
        F2d_Expect(gatheredOk, "a gathered tile differs from the host's");
        int pixels = height * width;
        var rgbaExpect = new int[pixels]; var planesExpect = new float[3 * pixels];
        for (int i = 0; i < pixels; i++)
        {
            float inv = 1f / wsum[i]; int pk = 0;
            for (int c = 0; c < 3; c++)
            {
                float v = sum[c * pixels + i] * inv; planesExpect[c * pixels + i] = v;
                pk |= Math.Clamp((int)((v + 1f) * 0.5f * 255f + 0.5f), 0, 255) << (8 * c);
            }
            rgbaExpect[i] = pk | (255 << 24);
        }
        using var rgbaB = accelerator.Allocate1D<int>(pixels); using var planesB = accelerator.Allocate1D<float>(3 * pixels);
        k.TileFinalize(sumB.View, wsumB.View, rgbaB.View, planesB.View, pixels, 1);
        var gotPlanes = await planesB.View.CopyToHostAsync(); var gotRgba = await rgbaB.View.CopyToHostAsync();
        foreach (var t in tileBufs) t.Dispose();
        double worst = F2d_MaxDiff(gotPlanes, planesExpect);
        int off = 0; for (int i = 0; i < pixels; i++) if (gotRgba[i] != rgbaExpect[i]) off++;
        F2d_Expect(worst < 2e-5, $"blended planes differ from the host's by up to {worst:G3}");
        F2d_Expect(off <= pixels / 500, $"{off} of {pixels} pixels differ from the host's (a rounding step at most)");
    });
}
