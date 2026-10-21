using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// <see cref="DeviceUvAtlas"/> (winding, normals and the UV atlas in device kernels) on meshes whose answer is known and against the host <see cref="UvAtlas"/> / <see cref="MeshRepair.AlignWindingToNormals"/>:
/// every face keeps its geometry through the seam duplication, every UV triangle is wound like its face and lies inside the atlas, no two triangles of the atlas overlap, the texel density is uniform, and the layout is
/// about as tight as the host's.
/// </summary>
public abstract partial class MLTestBase
{
    private static void Da_Expect(bool ok, string what) { if (!ok) throw new Exception(what); }

    private sealed class AtlasOut
    {
        public float[] P = Array.Empty<float>(), Uv = Array.Empty<float>(), N = Array.Empty<float>();
        public int[] F = Array.Empty<int>(), Src = Array.Empty<int>();
        public int Charts, Flipped; public double Utilization, Density;
        public double Ms;
    }

    private async Task<AtlasOut> Da_Run(Accelerator accelerator, float[] p, int[] f, int res, bool align, UvAtlas.Options? options = null)
    {
        RequireAtomicsAndScatter(accelerator);
        using var pb = accelerator.Allocate1D(p); using var fb = accelerator.Allocate1D(f);
        var atlas = new DeviceUvAtlas(accelerator, new SparseKernels(accelerator));
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var r = await atlas.UnwrapAsync(pb.View, p.Length / 3, fb.View, f.Length / 3, options ?? new UvAtlas.Options { Resolution = res }, align);
        foreach (var line in atlas.Log) Console.WriteLine($"[DeviceAtlas] {line}");
        var o = new AtlasOut { Ms = sw.Elapsed.TotalMilliseconds, Charts = r.ChartCount, Flipped = r.Flipped, Utilization = r.Utilization, Density = r.TexelsPerUnit };
        o.P = await r.Positions.View.SubView(0, (long)r.VertexCount * 3).CopyToHostAsync();
        o.Uv = await r.Uvs.View.SubView(0, (long)r.VertexCount * 2).CopyToHostAsync();
        o.N = await r.Normals.View.SubView(0, (long)r.VertexCount * 3).CopyToHostAsync();
        o.F = await r.Faces.View.SubView(0, (long)r.FaceCount * 3).CopyToHostAsync();
        o.Src = await r.SourceVertex.View.SubView(0, r.VertexCount).CopyToHostAsync();
        return o;
    }

    /// <summary>Checks that hold for every valid atlas; returns the number of texels two faces share (overlap) on a 1024 grid.</summary>
    private static void Da_CheckAtlas(AtlasOut a, float[] p, int[] f, string what, int res)
    {
        int nf = f.Length / 3;
        Da_Expect(a.F.Length == f.Length, $"{what}: {a.F.Length / 3} faces for {nf}");
        // geometry survives: each corner sits where the input corner did
        for (int i = 0; i < f.Length; i++)
        {
            int nv = a.F[i], ov = f[i];
            Da_Expect(a.Src[nv] == ov, $"{what}: corner {i} maps to source {a.Src[nv]}, input has {ov}");
            for (int k = 0; k < 3; k++) Da_Expect(a.P[nv * 3 + k] == p[ov * 3 + k], $"{what}: corner {i} moved");
        }
        // inside the atlas, wound like the face
        int flipped = 0; double worstLo = 1, worstHi = 0;
        for (int t = 0; t < nf; t++)
        {
            int a0 = a.F[t * 3], b0 = a.F[t * 3 + 1], c0 = a.F[t * 3 + 2];
            for (int k = 0; k < 3; k++)
            {
                int v = a.F[t * 3 + k];
                worstLo = Math.Min(worstLo, Math.Min(a.Uv[v * 2], a.Uv[v * 2 + 1])); worstHi = Math.Max(worstHi, Math.Max(a.Uv[v * 2], a.Uv[v * 2 + 1]));
            }
            double det = (a.Uv[b0 * 2] - a.Uv[a0 * 2]) * (a.Uv[c0 * 2 + 1] - a.Uv[a0 * 2 + 1]) - (a.Uv[c0 * 2] - a.Uv[a0 * 2]) * (a.Uv[b0 * 2 + 1] - a.Uv[a0 * 2 + 1]);
            if (det <= 0) flipped++;
        }
        Da_Expect(worstLo >= 0 && worstHi <= 1, $"{what}: uv outside the atlas ({worstLo:G4} .. {worstHi:G4})");
        Da_Expect(flipped == 0, $"{what}: {flipped} UV triangles are wound the other way");
        // no overlap: rasterise every triangle's texel centres on a grid and count a texel owned by two faces
        int G = 1024; var owner = new int[G * G]; Array.Fill(owner, -1); long covered = 0, shared = 0;
        for (int t = 0; t < nf; t++)
        {
            int a0 = a.F[t * 3], b0 = a.F[t * 3 + 1], c0 = a.F[t * 3 + 2];
            double ax = a.Uv[a0 * 2] * G, ay = a.Uv[a0 * 2 + 1] * G, bx = a.Uv[b0 * 2] * G, by = a.Uv[b0 * 2 + 1] * G, cx = a.Uv[c0 * 2] * G, cy = a.Uv[c0 * 2 + 1] * G;
            double det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
            if (Math.Abs(det) < 1e-12) continue;
            int x0 = Math.Max(0, (int)Math.Floor(Math.Min(ax, Math.Min(bx, cx)))), x1 = Math.Min(G - 1, (int)Math.Ceiling(Math.Max(ax, Math.Max(bx, cx))));
            int y0 = Math.Max(0, (int)Math.Floor(Math.Min(ay, Math.Min(by, cy)))), y1 = Math.Min(G - 1, (int)Math.Ceiling(Math.Max(ay, Math.Max(by, cy))));
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++)
                {
                    double px = x + 0.5, py = y + 0.5;
                    double w0 = ((bx - px) * (cy - py) - (cx - px) * (by - py)) / det, w1 = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) / det, w2 = 1 - w0 - w1;
                    if (w0 <= 1e-4 || w1 <= 1e-4 || w2 <= 1e-4) continue;
                    if (owner[y * G + x] >= 0 && owner[y * G + x] != t) shared++; else covered++;
                    owner[y * G + x] = t;
                }
        }
        Da_Expect(shared <= covered / 200, $"{what}: {shared} texels are covered by two triangles ({covered} covered)");
        // one texel density: the UV area of a face over its area, in texels, is the atlas density squared times the chart's growth (1 unless the chart was enlarged to the minimum size)
        var ratios = new List<double>();
        for (int t = 0; t < nf; t++)
        {
            int a0 = a.F[t * 3], b0 = a.F[t * 3 + 1], c0 = a.F[t * 3 + 2];
            double uvArea = 0.5 * Math.Abs((a.Uv[b0 * 2] - a.Uv[a0 * 2]) * (a.Uv[c0 * 2 + 1] - a.Uv[a0 * 2 + 1]) - (a.Uv[c0 * 2] - a.Uv[a0 * 2]) * (a.Uv[b0 * 2 + 1] - a.Uv[a0 * 2 + 1])) * res * (double)res;
            double ux = a.P[b0 * 3] - a.P[a0 * 3], uy = a.P[b0 * 3 + 1] - a.P[a0 * 3 + 1], uz = a.P[b0 * 3 + 2] - a.P[a0 * 3 + 2], vx = a.P[c0 * 3] - a.P[a0 * 3], vy = a.P[c0 * 3 + 1] - a.P[a0 * 3 + 1], vz = a.P[c0 * 3 + 2] - a.P[a0 * 3 + 2];
            double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
            double geo = 0.5 * Math.Sqrt(nx * nx + ny * ny + nz * nz);
            if (geo > 1e-12) ratios.Add(uvArea / geo / (a.Density * a.Density));
        }
        ratios.Sort();
        // a face is projected along its chart's direction: its UV area is its area times the cosine to that direction, so the ratio is at most the chart growth and at least cos(65 deg)
        Da_Expect(ratios[ratios.Count / 2] > 0.7 && ratios[ratios.Count / 2] < 1.1, $"{what}: median texel density ratio {ratios[ratios.Count / 2]:F3} (1 = every chart at the atlas density)");
    }

    /// <summary>
    /// Charts grown from filtered normals (the gold-visor atlas: 31,000 tiny charts for a smooth dome with voxel-scale bumps): on a bumpy sphere the same mesh needs far fewer charts, and the atlas
    /// is still valid (no UV triangle folded or overlapping, every corner where it was).
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task DeviceAtlas_BumpySphere_FilteredChartNormalsMakeFewerCharts() => await RunTest(async accelerator =>
    {
        var (p, f) = NoisySphere(0.02, 7);
        var plain = await Da_Run(accelerator, p, f, 1024, true);
        var smooth = await Da_Run(accelerator, p, f, 1024, true, new UvAtlas.Options { Resolution = 1024, ChartNormalSmoothingIterations = 3, ChartNormalSmoothingDegrees = 30 });
        Da_CheckAtlas(plain, p, f, "bumpy sphere, raw chart normals", 1024);
        Da_CheckAtlas(smooth, p, f, "bumpy sphere, filtered chart normals", 1024);
        Da_Expect(smooth.Charts * 2 < plain.Charts, $"filtered chart normals gave {smooth.Charts} charts against {plain.Charts} raw: not at least 2x fewer");
        return $"bumpy sphere (5120 faces): charts {plain.Charts} -> {smooth.Charts}, utilization {plain.Utilization:P0} -> {smooth.Utilization:P0}";
    });

    [TestMethod]
    public async Task DeviceAtlas_Sphere_ValidAndAsTightAsTheHost() => await RunTest(async accelerator =>
    {
        var (p, f) = Mr_Sphere(4);                       // 5120 faces
        var d = await Da_Run(accelerator, p, (int[])f.Clone(), 1024, false);
        Da_CheckAtlas(d, p, f, "sphere", 1024);
        var h = UvAtlas.Unwrap(p, f, new UvAtlas.Options { Resolution = 1024 });
        Console.WriteLine($"[DeviceAtlas] sphere: device {d.Charts} charts, {d.Utilization:P1} used, density {d.Density:F1}, {d.Ms:F0} ms on {accelerator.AcceleratorType}; host {h.ChartCount} charts, {h.Utilization:P1}, density {h.TexelsPerUnit:F1}");
        Da_Expect(d.Charts <= h.ChartCount * 3 + 8, $"charts {d.Charts} vs host {h.ChartCount}");
        Da_Expect(d.Density >= 0.6 * h.TexelsPerUnit, $"texel density {d.Density:F1} vs host {h.TexelsPerUnit:F1}");
    });

    [TestMethod]
    public async Task DeviceAtlas_Torus_ValidAndAsTightAsTheHost() => await RunTest(async accelerator =>
    {
        int nu = 96, nv = 48; double R = 1.0, r = 0.35;
        var p = new float[nu * nv * 3]; var f = new int[nu * nv * 6];
        for (int i = 0; i < nu; i++)
            for (int j = 0; j < nv; j++)
            {
                double a = 2 * Math.PI * i / nu, b = 2 * Math.PI * j / nv;
                int o = (i * nv + j) * 3;
                p[o] = (float)((R + r * Math.Cos(b)) * Math.Cos(a)); p[o + 1] = (float)((R + r * Math.Cos(b)) * Math.Sin(a)); p[o + 2] = (float)(r * Math.Sin(b));
                int i1 = (i + 1) % nu, j1 = (j + 1) % nv, q = (i * nv + j) * 6;
                int v00 = i * nv + j, v10 = i1 * nv + j, v11 = i1 * nv + j1, v01 = i * nv + j1;
                f[q] = v00; f[q + 1] = v10; f[q + 2] = v11; f[q + 3] = v00; f[q + 4] = v11; f[q + 5] = v01;
            }
        var d = await Da_Run(accelerator, p, (int[])f.Clone(), 1024, false);
        Da_CheckAtlas(d, p, f, "torus", 1024);
        var h = UvAtlas.Unwrap(p, f, new UvAtlas.Options { Resolution = 1024 });
        Console.WriteLine($"[DeviceAtlas] torus: device {d.Charts} charts, {d.Utilization:P1} used, density {d.Density:F1}, {d.Ms:F0} ms; host {h.ChartCount} charts, {h.Utilization:P1}, density {h.TexelsPerUnit:F1}");
        Da_Expect(d.Density >= 0.6 * h.TexelsPerUnit, $"texel density {d.Density:F1} vs host {h.TexelsPerUnit:F1}");
    });

    [TestMethod]
    public async Task DeviceAtlas_FoldedSheet_SplitsWhereTheProjectionWouldFold() => await RunTest(async accelerator =>
    {
        // a sheet bent into a roll more than a half turn: one plane cannot hold it without folding, so it must come out as several charts that do not overlap
        int nu = 60, nv = 20;
        var p = new float[nu * nv * 3]; var fl = new List<int>();
        for (int i = 0; i < nu; i++)
            for (int j = 0; j < nv; j++)
            {
                double a = 1.6 * Math.PI * i / (nu - 1), y = j / (double)(nv - 1);
                int o = (i * nv + j) * 3; p[o] = (float)Math.Cos(a); p[o + 1] = (float)y; p[o + 2] = (float)Math.Sin(a);
                if (i < nu - 1 && j < nv - 1) { int a0 = i * nv + j, b0 = (i + 1) * nv + j, c0 = (i + 1) * nv + j + 1, d0 = i * nv + j + 1; fl.AddRange(new[] { a0, c0, b0, a0, d0, c0 }); }
            }
        var f = fl.ToArray();
        var d = await Da_Run(accelerator, p, (int[])f.Clone(), 1024, false);
        Da_CheckAtlas(d, p, f, "roll", 1024);
        Da_Expect(d.Charts >= 3, $"a 290-degree roll came out as {d.Charts} charts");
    });

    [TestMethod]
    public async Task DeviceAtlas_FlippedFaces_AreRewoundAndNormalsMatchTheHost() => await RunTest(async accelerator =>
    {
        var (p, f0) = Mr_Sphere(4);
        var f = (int[])f0.Clone();
        var rnd = new Random(5); int flips = 0;
        for (int t = 0; t < f.Length / 3; t += 17) { (f[t * 3 + 1], f[t * 3 + 2]) = (f[t * 3 + 2], f[t * 3 + 1]); flips++; }
        var hostF = (int[])f.Clone();
        int hostFlipped = MeshRepair.AlignWindingToNormals(p, hostF);
        var d = await Da_Run(accelerator, p, (int[])f.Clone(), 1024, true);
        Console.WriteLine($"[DeviceAtlas] {flips} faces turned over: host re-winds {hostFlipped}, device {d.Flipped}");
        Da_Expect(d.Flipped == hostFlipped, $"device flipped {d.Flipped}, host {hostFlipped}");
        // the unwrapped faces carry the corrected winding: compare their corners' source vertices with the host's result
        for (int i = 0; i < hostF.Length; i++) Da_Expect(d.Src[d.F[i]] == hostF[i], $"corner {i}: source {d.Src[d.F[i]]}, host {hostF[i]}");
        var hn = PbrGlbWriter.VertexNormals(p, hostF);
        double worst = 0;
        for (int i = 0; i < d.Src.Length; i++)
            for (int k = 0; k < 3; k++) worst = Math.Max(worst, Math.Abs(d.N[i * 3 + k] - hn[d.Src[i] * 3 + k]));
        Da_Expect(worst < 1e-4, $"normals differ from the host by up to {worst:G3}");
    });
}
