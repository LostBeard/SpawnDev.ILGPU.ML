using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// The reference's <c>to_glb(remesh=True)</c>: <see cref="NarrowBandRemesher"/> (narrow-band dual contouring of the distance field of the generated mesh) and <see cref="MeshDistanceGrid"/>
/// (the device distance queries behind it and behind the texture bake's projection onto the original surface). Synthetic meshes with known geometry.
/// </summary>
public abstract partial class MLTestBase
{
    private static void Rm_Expect(bool ok, string what) { if (!ok) throw new Exception(what); }

    /// <summary>The distance from p to triangle (a, b, c) by dense sampling-free exact projection, in double precision (the oracle).</summary>
    private static double Rm_PointTriangle(double[] p, double[] a, double[] b, double[] c)
    {
        double[] Sub(double[] x, double[] y) => new[] { x[0] - y[0], x[1] - y[1], x[2] - y[2] };
        double Dot(double[] x, double[] y) => x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
        var ab = Sub(b, a); var ac = Sub(c, a); var ap = Sub(p, a);
        double d1 = Dot(ab, ap), d2 = Dot(ac, ap);
        double[] r;
        if (d1 <= 0 && d2 <= 0) r = a;
        else
        {
            var bp = Sub(p, b); double d3 = Dot(ab, bp), d4 = Dot(ac, bp);
            if (d3 >= 0 && d4 <= d3) r = b;
            else
            {
                double vc = d1 * d4 - d3 * d2;
                if (vc <= 0 && d1 >= 0 && d3 <= 0) { double v = d1 / (d1 - d3); r = new[] { a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2] }; }
                else
                {
                    var cp = Sub(p, c); double d5 = Dot(ab, cp), d6 = Dot(ac, cp);
                    if (d6 >= 0 && d5 <= d6) r = c;
                    else
                    {
                        double vb = d5 * d2 - d1 * d6;
                        if (vb <= 0 && d2 >= 0 && d6 <= 0) { double w = d2 / (d2 - d6); r = new[] { a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2] }; }
                        else
                        {
                            double va = d3 * d6 - d5 * d4;
                            if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) { double w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); r = new[] { b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2]) }; }
                            else { double den = 1 / (va + vb + vc); double v = vb * den, w = vc * den; r = new[] { a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w }; }
                        }
                    }
                }
            }
        }
        var d = Sub(p, r); return Math.Sqrt(Dot(d, d));
    }

    private static (float[] P, int[] F) Rm_ScaledSphere(int level, float radius)
    {
        var (p, f) = Mr_Sphere(level);
        var q = new float[p.Length];
        for (int i = 0; i < p.Length; i++) q[i] = p[i] * radius;
        return (q, f);
    }

    /// <summary>
    /// <see cref="MeshDistanceGrid.MaxSearchThreadsPerSubmit"/> (2026-11-19: an RX 590 in Chrome lost the WebGPU device in TRELLIS.2's device remesh, one
    /// nearest-triangle search dispatch over every band voxel): every search kernel runs in chunks of points, each its own command buffer in the browser. The
    /// points are independent, so each entry point (distances, band voxels on the host and on the device, device distances, projection, the bake's texel
    /// projection) must give BIT FOR BIT the values of one dispatch, here with the smallest chunk (64 points; band voxels 64 cells) on a subdivided sphere.
    /// </summary>
    [TestMethod]
    public async Task Remesh_DistanceGrid_ChunkedSearchesMatchOneDispatch() => await RunTest(async accelerator =>
    {
        const int R = 64;
        var (p, f) = Rm_ScaledSphere(4, 0.3f);
        var rng = new Random(11);
        int nPts = 5000;
        var pts = new float[nPts * 3];
        for (int i = 0; i < pts.Length; i++) pts[i] = (float)(rng.NextDouble() * (R - 4) + 2);
        for (int i = 0; i < nPts / 2; i++) { int v = rng.Next(p.Length / 3); for (int a = 0; a < 3; a++) pts[i * 3 + a] = p[v * 3 + a] * R + R / 2 + (float)(rng.NextDouble() * 4 - 2); }   // half of them near the surface
        int texels = 5000, nf = f.Length / 3;
        var idMap = new int[texels]; var bary = new float[texels * 2];
        for (int i = 0; i < texels; i++) { idMap[i] = i % 7 == 0 ? 0 : rng.Next(nf) + 1; float w0 = (float)rng.NextDouble(), w1 = (float)rng.NextDouble() * (1 - w0); bary[i * 2] = w0; bary[i * 2 + 1] = w1; }
        var sk = new SparseKernels(accelerator);
        int was = MeshDistanceGrid.MaxSearchThreadsPerSubmit;
        async Task<(float[] D, int[] B, int[] BD, float[] DD, float[] P, float[] T)> Run(int threads)
        {
            MeshDistanceGrid.MaxSearchThreadsPerSubmit = threads;
            using var grid = MeshDistanceGrid.Build(accelerator, p, f, R, R, R / 2f);
            var d = await grid.DistancesAsync(pts.Select(x => x).ToArray());
            var b = await grid.BandVoxelsAsync(1f, 0.87f);
            var (bdBuf, bdN) = await grid.BandVoxelsDeviceAsync(sk, 1f, 0.87f);
            int[] bd; using (bdBuf) bd = (await bdBuf.View.CopyToHostAsync()).Take(bdN * 3).ToArray();
            using var ptsDev = accelerator.Allocate1D(pts); using var dd = accelerator.Allocate1D<float>(nPts);
            grid.DistancesDevice(ptsDev.View, nPts, dd.View, 0.25f);
            var ddh = await dd.View.CopyToHostAsync();
            var pr = await grid.ProjectAsync(pts);
            using var idDev = accelerator.Allocate1D(idMap); using var baryDev = accelerator.Allocate1D(bary);
            using var fDev = accelerator.Allocate1D(f); using var pDev = accelerator.Allocate1D(p); using var outP = accelerator.Allocate1D<float>(texels * 3L);
            outP.MemSetToZero();
            grid.ProjectTexels(texels, idDev.View, baryDev.View, fDev.View, pDev.View, outP.View);
            var tp = await outP.View.CopyToHostAsync();
            return (d, b, bd, ddh, pr, tp);
        }
        try
        {
            var one = await Run(int.MaxValue / 2);
            var chunked = await Run(64);
            static int Diff<T>(T[] a, T[] b) where T : IEquatable<T> { int n = Math.Abs(a.Length - b.Length); for (int i = 0; i < Math.Min(a.Length, b.Length); i++) if (!a[i].Equals(b[i])) n++; return n; }
            var diffs = new (string Name, int N, int Of)[]
            {
                ("distances", Diff(one.D, chunked.D), one.D.Length), ("band voxels", Diff(one.B, chunked.B), one.B.Length), ("band voxels (device)", Diff(one.BD, chunked.BD), one.BD.Length),
                ("device distances", Diff(one.DD, chunked.DD), one.DD.Length), ("projection", Diff(one.P, chunked.P), one.P.Length), ("texel projection", Diff(one.T, chunked.T), one.T.Length),
            };
            Rm_Expect(one.B.Length > 300 && one.B.Length == one.BD.Length, $"the band must be real and both band paths agree ({one.B.Length / 3} / {one.BD.Length / 3} voxels)");
            foreach (var (name, n, of) in diffs) Rm_Expect(n == 0, $"{name}: chunked differs from one dispatch in {n} of {of} values");
            return $"{nf} triangles, {one.B.Length / 3} band voxels, {nPts} points, {texels} texels: every search chunked (64) == one dispatch, bit for bit";
        }
        finally { MeshDistanceGrid.MaxSearchThreadsPerSubmit = was; }
    });

    [TestMethod]
    public async Task Remesh_DistanceGrid_MatchesBruteForce() => await RunTest(async accelerator =>
    {
        // 150 small triangles scattered in a 64^3 lattice (world == grid units), 600 query points near them
        var rng = new Random(7);
        var p = new List<float>(); var f = new List<int>();
        for (int t = 0; t < 150; t++)
        {
            double cx = rng.Next(8, 56), cy = rng.Next(8, 56), cz = rng.Next(8, 56);
            for (int k = 0; k < 3; k++) { p.Add((float)(cx + rng.NextDouble() * 3 - 1.5)); p.Add((float)(cy + rng.NextDouble() * 3 - 1.5)); p.Add((float)(cz + rng.NextDouble() * 3 - 1.5)); f.Add(t * 3 + k); }
        }
        var pts = new float[600 * 3];
        for (int i = 0; i < pts.Length; i++) pts[i] = (float)(rng.NextDouble() * 60 + 2);
        for (int i = 0; i < 400; i++) { int t = rng.Next(150), k = rng.Next(3); for (int a = 0; a < 3; a++) pts[i * 3 + a] = p[(t * 3 + k) * 3 + a] + (float)(rng.NextDouble() * 4 - 2); }   // most of them near a triangle
        using var grid = MeshDistanceGrid.Build(accelerator, p.ToArray(), f.ToArray(), 64, 1f, 0f);
        var d = await grid.DistancesAsync(pts);
        double worst = 0; int near = 0;
        for (int i = 0; i < 600; i++)
        {
            double best = double.MaxValue;
            for (int t = 0; t < 150; t++)
            {
                double[] V(int k) => new double[] { p[(t * 3 + k) * 3], p[(t * 3 + k) * 3 + 1], p[(t * 3 + k) * 3 + 2] };
                best = Math.Min(best, Rm_PointTriangle(new double[] { pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2] }, V(0), V(1), V(2)));
            }
            double expect = Math.Min(best, MeshDistanceGrid.Reach);
            if (best < MeshDistanceGrid.Reach) near++;
            worst = Math.Max(worst, Math.Abs(d[i] - expect));
        }
        Rm_Expect(near > 200, $"the test must exercise real distances ({near} of 600 points within reach)");
        Rm_Expect(worst < 2e-3, $"device distance equals the exact one (worst error {worst:G3} voxel)");
    });

    [TestMethod]
    public async Task Remesh_NarrowBand_SphereBecomesAClosedOutwardShellOneBandOut() => await RunTest(async accelerator =>
    {
        const float radius = 0.3f; const int R = 64;
        var (p, f) = Rm_ScaledSphere(3, radius);
        using var r = await NarrowBandRemesher.RemeshAsync(accelerator, p, f, R, 1f, dropInnerSkins: false);
        var topo = MeshRepair.Measure(r.Positions, r.Faces);
        Rm_Expect(r.Faces.Length > 1000, $"a shell came out ({r.Faces.Length / 3} faces)");
        Rm_Expect(topo.BoundaryEdges == 0, $"closed: no open edges ({topo.BoundaryEdges})");
        Rm_Expect(topo.NonManifoldEdges == 0, $"manifold: no edge with more than two faces ({topo.NonManifoldEdges})");
        Rm_Expect(topo.OrientationConflicts == 0, $"one winding everywhere ({topo.OrientationConflicts} conflicts)");
        Rm_Expect(topo.Components == 2, $"the band of a closed surface has two skins, one outside it and one inside ({topo.Components} pieces)");
        // the band between the skins has positive volume: the outer skin faces out, the inner one faces in
        double vol = 0;
        for (int t = 0; t < r.Faces.Length / 3; t++)
        {
            int a = r.Faces[t * 3] * 3, b = r.Faces[t * 3 + 1] * 3, c = r.Faces[t * 3 + 2] * 3;
            vol += r.Positions[a] * (r.Positions[b + 1] * r.Positions[c + 2] - r.Positions[b + 2] * r.Positions[c + 1])
                 - r.Positions[a + 1] * (r.Positions[b] * r.Positions[c + 2] - r.Positions[b + 2] * r.Positions[c])
                 + r.Positions[a + 2] * (r.Positions[b] * r.Positions[c + 1] - r.Positions[b + 1] * r.Positions[c]);
        }
        Rm_Expect(vol > 0, $"the winding is consistent: the skins bound the band (signed volume {vol:G3})");
        // one band (1 voxel of the domain-scaled grid) outside and inside the input surface
        double voxel = (R + 3.0) / R / R, outer = 0, inner = 0; int no = 0, ni = 0;
        for (int i = 0; i < r.Positions.Length / 3; i++)
        {
            double rad = Math.Sqrt(r.Positions[i * 3] * r.Positions[i * 3] + r.Positions[i * 3 + 1] * r.Positions[i * 3 + 1] + r.Positions[i * 3 + 2] * r.Positions[i * 3 + 2]);
            if (rad > radius) { outer += rad; no++; } else { inner += rad; ni++; }
        }
        outer /= no; inner /= ni;
        Rm_Expect(Math.Abs(outer - (radius + voxel)) < 0.5 * voxel, $"the outer skin sits one band outside the sphere (mean radius {outer:F4}, expected {radius + voxel:F4})");
        Rm_Expect(Math.Abs(inner - (radius - voxel)) < 0.5 * voxel, $"the inner skin sits one band inside (mean radius {inner:F4}, expected {radius - voxel:F4})");
    });

    [TestMethod]
    public async Task Remesh_NarrowBand_DropsTheSkinInsideTheModel() => await RunTest(async accelerator =>
    {
        var (p, f) = Rm_ScaledSphere(3, 0.3f);
        using var both = await NarrowBandRemesher.RemeshAsync(accelerator, p, f, 64, 1f, dropInnerSkins: false);
        using var r = await NarrowBandRemesher.RemeshAsync(accelerator, p, f, 64, 1f);
        var topo = MeshRepair.Measure(r.Positions, r.Faces);
        Rm_Expect(r.Faces.Length < both.Faces.Length && r.Faces.Length > both.Faces.Length / 3, $"the inner skin is gone ({both.Faces.Length / 3} -> {r.Faces.Length / 3} faces)");
        Rm_Expect(topo.Components == 1 && topo.BoundaryEdges == 0 && topo.OrientationConflicts == 0, $"one closed outward skin remains ({topo.Components} pieces, {topo.BoundaryEdges} open, {topo.OrientationConflicts} conflicts)");
        for (int i = 0; i < r.Positions.Length / 3; i++)
        {
            double rad = Math.Sqrt(r.Positions[i * 3] * r.Positions[i * 3] + r.Positions[i * 3 + 1] * r.Positions[i * 3 + 1] + r.Positions[i * 3 + 2] * r.Positions[i * 3 + 2]);
            Rm_Expect(rad > 0.3, $"every remaining vertex lies outside the input surface (radius {rad:F4})");
        }
    });

    [TestMethod]
    public async Task Remesh_NarrowBand_ClosesAPinholeAndAThinGap() => await RunTest(async accelerator =>
    {
        // the sphere with one triangle missing (a hole about 2 voxels across): the input has an open edge loop, the shell must not
        var (p, f) = Rm_ScaledSphere(3, 0.3f);
        var open = new List<int>(f); open.RemoveRange(0, 3);
        var inTopo = MeshRepair.Measure(p, open.ToArray());
        Rm_Expect(inTopo.BoundaryEdges == 3, $"the input has the pinhole ({inTopo.BoundaryEdges} open edges)");
        using var r = await NarrowBandRemesher.RemeshAsync(accelerator, p, open.ToArray(), 64, 1f, dropInnerSkins: false);
        var topo = MeshRepair.Measure(r.Positions, r.Faces);
        Rm_Expect(topo.BoundaryEdges == 0 && topo.Components == 2, $"the skins are closed over the pinhole ({topo.BoundaryEdges} open edges, {topo.Components} pieces)");

        // a thin double sheet (two parallel quads 1 voxel apart): one watertight skin around both
        var pp = new List<float>(); var ff = new List<int>();
        void Quad(float z)
        {
            int b = pp.Count / 3;
            pp.AddRange(new[] { -0.2f, -0.2f, z, 0.2f, -0.2f, z, 0.2f, 0.2f, z, -0.2f, 0.2f, z });
            ff.AddRange(new[] { b, b + 1, b + 2, b, b + 2, b + 3 });
        }
        Quad(0f); Quad(1f / 64f);
        using var r2 = await NarrowBandRemesher.RemeshAsync(accelerator, pp.ToArray(), ff.ToArray(), 64, 1f);
        var t2 = MeshRepair.Measure(r2.Positions, r2.Faces);
        Rm_Expect(t2.BoundaryEdges == 0 && t2.OrientationConflicts == 0 && t2.Components == 1, $"the thin double sheet becomes one closed skin ({t2.BoundaryEdges} open, {t2.OrientationConflicts} conflicts, {t2.Components} pieces)");
    });

    [TestMethod]
    public async Task Remesh_ProjectsOntoTheOriginalSurface() => await RunTest(async accelerator =>
    {
        RequireAtomicsAndScatter(accelerator);          // the distance grid is built with atomics
        var (p, f) = Rm_ScaledSphere(3, 0.3f);
        using var r = await NarrowBandRemesher.RemeshAsync(accelerator, p, f, 64, 1f);
        // the remeshed vertices sit a band outside; moved onto the original they must land on the sphere
        var g = new float[r.Positions.Length];
        for (int i = 0; i < g.Length; i++) g[i] = r.Positions[i] * r.Source.Scale + r.Source.Offset;
        var q = await r.Source.ProjectAsync(g);
        double worst = 0; int n = g.Length / 3;
        for (int i = 0; i < n; i++)
        {
            double x = (q[i * 3] - r.Source.Offset) / r.Source.Scale, y = (q[i * 3 + 1] - r.Source.Offset) / r.Source.Scale, z = (q[i * 3 + 2] - r.Source.Offset) / r.Source.Scale;
            worst = Math.Max(worst, Math.Abs(Math.Sqrt(x * x + y * y + z * z) - 0.3));
        }
        Rm_Expect(worst < 0.002, $"every projected vertex lies on the original sphere (worst radius error {worst:G3})");
    });
}
