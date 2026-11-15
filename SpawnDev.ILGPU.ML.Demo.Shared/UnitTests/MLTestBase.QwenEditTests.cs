using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.QwenImage;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="SkinTokensKernels.RopeTables"/> with <see cref="SkinTokensKernels.AxesRopeTables"/> (Qwen2.5-VL's M-RoPE and its vision tower's 2D rope) against a double-precision
    /// evaluation of the reference's formula (float32 angle pos * inv_freq, accurate cos / sin), on random rows and positions up to 600 (a text prompt with pictures reaches a few
    /// hundred): 3 axes with sections [16, 24, 24] over head width 128, and 2 axes over head width 80 (pairs 0-19 by the row, 20-39 by the column). Bound 1e-6: fp32 rounding only.
    /// (GPU sin / cos evaluated in the kernel were 3.8e-5 off on WebGPU and WebGL at these angles, which is why the tables come from the host.) A text-only prompt's 3-axis tables
    /// must equal plain 1D tables bit for bit.
    /// </summary>
    [TestMethod]
    public async Task QwenEdit_RopeAxes_MatchesHostAndPlainRope() => await RunTest(async accelerator =>
    {
        var u = new SkinTokensKernels(accelerator);
        var rng = new Random(11);
        var notes = new List<string>(); bool ok = true;
        async Task Case(string name, int rows, int H, int D, int A, int b1, int b2, float theta, bool pairedTable)
        {
            int stride = 2 * H * D + 7, off = 3;                                                    // an odd stride and offset, as a fused q | k | v row has
            var buf = new float[rows * stride]; for (int i = 0; i < buf.Length; i++) buf[i] = (float)(rng.NextDouble() * 2 - 1);
            var inv = new float[D / 2];
            if (pairedTable) for (int i = 0; i < D / 4; i++) inv[i] = inv[i + D / 4] = 1f / MathF.Pow(theta, 2f * i / (D / 2));
            else for (int k = 0; k < D / 2; k++) inv[k] = 1f / MathF.Pow(theta, 2f * k / D);
            var pos = new int[rows * A]; for (int i = 0; i < pos.Length; i++) pos[i] = rng.Next(0, 600);
            var (cos, sin) = SkinTokensKernels.AxesRopeTables(pos, rows, A, inv, b1, b2);
            using var dBuf = accelerator.Allocate1D(buf); using var dCos = accelerator.Allocate1D(cos); using var dSin = accelerator.Allocate1D(sin);
            u.RopeTables(dBuf.View, dCos.View, dSin.View, rows, stride, off, H, D);
            var got = await dBuf.View.CopyToHostAsync();
            double worst = 0;
            for (int n = 0; n < rows; n++) for (int h = 0; h < H; h++) for (int k = 0; k < D / 2; k++)
            {
                int axis = k < b1 ? 0 : (k < b2 ? 1 : 2);
                float ang = pos[n * A + axis] * inv[k];                                             // the reference's float32 angle
                double c = Math.Cos(ang), s = Math.Sin(ang);
                long b = (long)n * stride + off + h * D;
                double x0 = buf[b + k], x1 = buf[b + D / 2 + k];
                worst = Math.Max(worst, Math.Max(Math.Abs(got[b + k] - (x0 * c - x1 * s)), Math.Abs(got[b + D / 2 + k] - (x1 * c + x0 * s))));
            }
            for (long i = 0; i < buf.Length; i++)                                                  // nothing outside the rotated columns moves
            {
                long col = i % stride;
                if ((col < off || col >= off + H * D) && got[i] != buf[i]) { worst = double.PositiveInfinity; break; }
            }
            ok &= worst < 1e-6;
            notes.Add($"{name}: worst {worst:E2}");
        }
        await Case("M-RoPE 3 axes, 28 heads of 128, sections 16 / 24 / 24", 37, 28, 128, 3, 16, 40, 1_000_000f, false);
        await Case("vision 2D rope, 16 heads of 80, pairs 0-19 h / 20-39 w", 41, 16, 80, 2, 20, 40, 10000f, true);
        // text-only: the 3-axis tables of a prompt without pictures equal plain 1D tables (position = row) bit for bit
        {
            int rows = 512, D = 128;
            var inv = new float[D / 2]; for (int k = 0; k < D / 2; k++) inv[k] = 1f / MathF.Pow(1_000_000f, 2f * k / D);
            var (c3, s3) = SkinTokensKernels.AxesRopeTables(Flux2TextEncoder.MRopePositions(rows, Array.Empty<(int, int, int)>()), rows, 3, inv, 16, 40);
            var (c1, s1) = SkinTokensKernels.AxesRopeTables(Enumerable.Range(0, rows).ToArray(), rows, 1, inv, D / 2, D / 2);
            int diff = 0; for (int i = 0; i < c3.Length; i++) if (c3[i] != c1[i] || s3[i] != s1[i]) diff++;
            ok &= diff == 0;
            notes.Add($"text-only M-RoPE tables vs 1D: {diff} differ");
        }
        var report = $"{accelerator.AcceleratorType}: " + string.Join("; ", notes);
        if (!ok) throw new Exception(report);
        return report;
    });

    /// <summary>
    /// The host-side layouts of Qwen-Image Edit: <see cref="Flux2TextEncoder.MRopePositions"/> on a hand-worked prompt (text, a 2 x 3 picture, text, a 3 x 2 picture: transformers'
    /// <c>get_rope_index</c> rule), and <see cref="QwenVisionEncoder.Layout"/> for two pictures (Lore's ComfyUI grids 24 x 32 and 22 x 36): the window order is a permutation of the
    /// merge units, each window lies inside one 4 x 4 unit tile of its own picture, windows and pictures partition the tokens, and the positions are the processor's block-major
    /// patch coordinates.
    /// </summary>
    [TestMethod]
    public async Task QwenEdit_Layouts_MatchTheReferenceRules() => await RunTest(async accelerator =>
    {
        await Task.CompletedTask;
        var notes = new List<string>();
        // M-RoPE: 2 text, picture 2 x 3 at 2, 1 text, picture 3 x 2 at 9, 2 text = 17 tokens
        var pos = Flux2TextEncoder.MRopePositions(17, new[] { (2, 2, 3), (9, 3, 2) });
        var want = new List<int[]> { new[] { 0, 0, 0 }, new[] { 1, 1, 1 } };
        for (int y = 0; y < 2; y++) for (int x = 0; x < 3; x++) want.Add(new[] { 2, 2 + y, 2 + x });
        want.Add(new[] { 5, 5, 5 });                                                                // p = 2 + max(2, 3)
        for (int y = 0; y < 3; y++) for (int x = 0; x < 2; x++) want.Add(new[] { 6, 6 + y, 6 + x });
        want.Add(new[] { 9, 9, 9 }); want.Add(new[] { 10, 10, 10 });                                // p = 6 + max(3, 2)
        for (int t = 0; t < 17; t++) for (int a = 0; a < 3; a++)
            if (pos[t * 3 + a] != want[t][a]) throw new Exception($"M-RoPE token {t} axis {a}: {pos[t * 3 + a]}, want {want[t][a]}");
        notes.Add("M-RoPE positions: 17/17 tokens as the rule gives");

        var grids = new[] { (24, 32), (22, 36) };
        var (vp, wi, windows, pics) = QwenVisionEncoder.Layout(grids);
        int n = 24 * 32 + 22 * 36, m = n / 4;
        if (pics.Length != 3 || pics[1] != 24 * 32 || pics[2] != n) throw new Exception($"picture segments {string.Join(",", pics)}");
        if (wi.Length != m || wi.Distinct().Count() != m || wi.Min() != 0 || wi.Max() != m - 1) throw new Exception("window index is not a permutation of the merge units");
        if (windows[0] != 0 || windows[^1] != n || windows.Zip(windows.Skip(1)).Any(p => p.Second <= p.First || (p.Second - p.First) % 4 != 0)) throw new Exception("window segments");
        // each window: units of one picture inside one 4 x 4 tile
        int units0 = 24 * 32 / 4;
        for (int s = 1; s < windows.Length; s++)
        {
            var units = Enumerable.Range(windows[s - 1] / 4, (windows[s] - windows[s - 1]) / 4).Select(i => wi[i]).ToList();
            bool second = units[0] >= units0;
            int lw = second ? 18 : 16, b = second ? units0 : 0;
            var tiles = units.Select(u => { if ((u >= units0) != second) return (-1, -1); int v = u - b; return (v / lw / 4, v % lw / 4); }).Distinct().ToList();
            if (tiles.Count != 1 || tiles[0].Item1 < 0) throw new Exception($"window {s} spans {tiles.Count} tiles or two pictures");
        }
        // windows: picture 1 12 x 16 units = 3 x 4 tiles of 16; picture 2 11 x 18 = 3 x 5 tiles, the last row and column partial
        if (windows.Length - 1 != 12 + 15) throw new Exception($"{windows.Length - 1} windows, want 27");
        // positions: token 0..3 of picture 1 are patches (0,0) (0,1) (1,0) (1,1); token 4 starts merge block (0, 1) = patch (0, 2)
        int[] p0 = { 0, 0, 0, 1, 1, 0, 1, 1, 0, 2 };
        for (int i = 0; i < p0.Length; i++) if (vp[i] != p0[i]) throw new Exception($"vision positions {string.Join(",", vp.Take(10))}");
        notes.Add($"vision layout: {m} units, {windows.Length - 1} windows inside 4 x 4 tiles, 2 pictures, block-major positions");
        return string.Join("; ", notes);
    });
}
