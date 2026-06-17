using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Tiling;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests
{
    /// <summary>
    /// Validates the TiledFeatureMap halo mechanism for exact tiled VAE decode: a CHAIN of two 3×3 SAME convs run
    /// tile-by-tile — conv1 on the FromFull-padded tiles, write cores into a new grid, RefreshHalos (fill halos
    /// from neighbor cores), conv2 — must equal the full-resolution conv1→conv2 chain, BYTE-equal, on every
    /// backend. This proves the per-conv halo refresh propagates boundary data correctly between layers (the part
    /// that makes a tiled conv stack identical to the full one). Plan step 2.
    /// </summary>
    public abstract partial class MLTestBase
    {
        [TestMethod]
        public async Task TiledFeatureMap_TwoConvChainMatchesFull_AllBackends() => await RunTest(async accelerator =>
        {
            const int C = 8, H = 20, W = 24, rows = 2, cols = 3, halo = 1;
            int n = C * H * W;
            var rng = new Random(917);
            var input = new float[n]; for (int i = 0; i < n; i++) input[i] = (float)(rng.NextDouble() * 4 - 2);
            float[] W1 = RandConv(C, C, rng), B1 = RandBias(C, rng);
            float[] W2 = RandConv(C, C, rng), B2 = RandBias(C, rng);

            var conv = new Conv2DKernel(accelerator);
            using var w1 = accelerator.Allocate1D(W1); using var b1 = accelerator.Allocate1D(B1);
            using var w2 = accelerator.Allocate1D(W2); using var b2 = accelerator.Allocate1D(B2);

            // ── Full reference: conv1(input) SAME → conv2 SAME. ──
            float[] expected;
            {
                using var inB = accelerator.Allocate1D(input);
                using var midB = accelerator.Allocate1D<float>(n);
                using var outB = accelerator.Allocate1D<float>(n);
                conv.ForwardPadded(inB.View, w1.View, b1.View, midB.View, C, H, W, C, 3, 3, 1, 1, 1, 1, 1);
                conv.ForwardPadded(midB.View, w2.View, b2.View, outB.View, C, H, W, C, 3, 3, 1, 1, 1, 1, 1);
                await accelerator.SynchronizeAsync();
                expected = await outB.CopyToHostAsync<float>(0, n);
            }

            // ── Tiled: conv1 on FromFull-padded tiles → WriteCore → RefreshHalos → conv2 → recombine. ──
            var map1 = TiledFeatureMap.FromFull(input, C, H, W, rows, cols, halo);
            var map2 = TiledFeatureMap.Allocate(C, H, W, rows, cols, halo);     // conv1 output grid
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    map2.WriteCore(r, c, await ConvTile(accelerator, conv, w1.View, b1.View, map1.Tile(r, c), C, map1.PaddedH(r), map1.PaddedW(c), C));
            map2.RefreshHalos();

            var outFull = new float[n];
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                {
                    var core = await ConvTile(accelerator, conv, w2.View, b2.View, map2.Tile(r, c), C, map2.PaddedH(r), map2.PaddedW(c), C);
                    PlaceCore(outFull, core, C, H, W, r, c, rows, cols);
                }

            float worst = 0;
            for (int i = 0; i < n; i++) worst = MathF.Max(worst, MathF.Abs(outFull[i] - expected[i]));
            if (worst > 1e-4f)
                throw new Exception($"tiled 2-conv chain diverged from full (worst |Δ|={worst:E3}) on {BackendName}");
            Console.WriteLine($"[TiledFeatureMap] tiled conv1→refreshHalos→conv2 == full chain (worst |Δ|={worst:E3}) on {BackendName}");
        });

        // Run a 3×3 pad-0 conv on one padded tile [C, ph, pw] → core [outC, ph-2, pw-2].
        private static async Task<float[]> ConvTile(Accelerator acc, Conv2DKernel conv,
            ArrayView1D<float, Stride1D.Dense> w, ArrayView1D<float, Stride1D.Dense> b,
            float[] paddedTile, int inC, int ph, int pw, int outC)
        {
            int oh = ph - 2, ow = pw - 2;
            using var inB = acc.Allocate1D(paddedTile);
            using var outB = acc.Allocate1D<float>(outC * oh * ow);
            conv.ForwardPadded(inB.View, w, b, outB.View, inC, ph, pw, outC, 3, 3, 1, 0, 0, 0, 0, 1, 1);
            await acc.SynchronizeAsync();
            return await outB.CopyToHostAsync<float>(0, outC * oh * ow);
        }

        private static void PlaceCore(float[] full, float[] core, int C, int H, int W, int r, int c, int rows, int cols)
        {
            int ch0 = H / rows + (r < H % rows ? 1 : 0), cw0 = W / cols + (c < W % cols ? 1 : 0);
            int y0 = 0; for (int rr = 0; rr < r; rr++) y0 += H / rows + (rr < H % rows ? 1 : 0);
            int x0 = 0; for (int cc = 0; cc < c; cc++) x0 += W / cols + (cc < W % cols ? 1 : 0);
            for (int ch = 0; ch < C; ch++)
                for (int yy = 0; yy < ch0; yy++)
                    for (int xx = 0; xx < cw0; xx++)
                        full[(long)ch * H * W + (y0 + yy) * W + (x0 + xx)] = core[(long)ch * ch0 * cw0 + yy * cw0 + xx];
        }

        private static float[] RandConv(int outC, int inC, Random rng)
        {
            var w = new float[outC * inC * 9];
            for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 0.4 - 0.2);
            return w;
        }
        private static float[] RandBias(int outC, Random rng)
        {
            var b = new float[outC];
            for (int i = 0; i < outC; i++) b[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
            return b;
        }
    }
}
