using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// <see cref="GpuMeshSimplifier"/> (parallel quadric edge collapse on the device) on meshes whose answer is known: it must reach the face budget, keep a closed surface closed (Euler characteristic, no open or non-manifold edges, one winding),
/// keep an open sheet's boundary where it was, not flip faces, and stay as close to the original surface as the host simplifier (<see cref="MeshSimplifier"/>) does.
/// </summary>
public abstract partial class MLTestBase
{
    private static void Gs_Expect(bool ok, string what) { if (!ok) throw new Exception(what); }

    private static int Gs_Euler(float[] p, int[] f)
    {
        var edges = new HashSet<long>(); var used = new HashSet<int>();
        for (int t = 0; t < f.Length; t += 3)
            for (int k = 0; k < 3; k++)
            {
                int a = f[t + k], b = f[t + (k + 1) % 3]; used.Add(a);
                edges.Add((long)Math.Min(a, b) * 4_000_000_007L + Math.Max(a, b));
            }
        return used.Count - edges.Count + f.Length / 3;
    }

    private static double Gs_Area(float[] p, int[] f)
    {
        double area = 0;
        for (int t = 0; t < f.Length; t += 3)
        {
            int a = f[t] * 3, b = f[t + 1] * 3, c = f[t + 2] * 3;
            double ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2], vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
            double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
            area += 0.5 * Math.Sqrt(nx * nx + ny * ny + nz * nz);
        }
        return area;
    }

    private async Task<(float[] P, int[] F, int Rounds)> Gs_Simplify(Accelerator accelerator, float[] p, int[] f, int target)
    {
        using var pb = accelerator.Allocate1D(p); using var fb = accelerator.Allocate1D(f);
        var simplifier = new GpuMeshSimplifier(accelerator, new SparseKernels(accelerator));
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var r = await simplifier.SimplifyAsync(pb.View, p.Length / 3, fb.View, f.Length / 3, target);
        var rp = await r.Positions.View.SubView(0, (long)r.VertexCount * 3).CopyToHostAsync();
        var rf = await r.Faces.View.SubView(0, (long)r.FaceCount * 3).CopyToHostAsync();
        Console.WriteLine($"[GpuSimplify] {f.Length / 3} -> {rf.Length / 3} faces (target {target}) in {simplifier.LastRounds.Count} rounds, {sw.Elapsed.TotalMilliseconds:F0} ms on {accelerator.AcceleratorType}");
        return (rp, rf, simplifier.LastRounds.Count);
    }

    [TestMethod]
    public async Task GpuSimplify_Sphere_ReachesTheBudgetAndStaysClosed() => await RunTest(async accelerator =>
    {
        var (p, f) = Mr_Sphere(4);                                    // 5120 faces
        int target = f.Length / 3 / 8;
        var (gp, gf, rounds) = await Gs_Simplify(accelerator, p, f, target);
        int got = gf.Length / 3;
        Gs_Expect(got <= target * 1.03 && got >= target * 0.9, $"faces {got} for a budget of {target}");
        var t = MeshRepair.Measure(gp, gf);
        Gs_Expect(t.BoundaryEdges == 0 && t.NonManifoldEdges == 0 && t.OrientationConflicts == 0 && t.Components == 1, $"stays one closed manifold ({t.BoundaryEdges} open, {t.NonManifoldEdges} non-manifold, {t.OrientationConflicts} conflicts, {t.Components} pieces)");
        Gs_Expect(Gs_Euler(gp, gf) == 2, $"Euler characteristic {Gs_Euler(gp, gf)} (a sphere has 2)");
        double worst = 0;
        for (int i = 0; i < gp.Length / 3; i++) worst = Math.Max(worst, Math.Abs(Math.Sqrt(gp[i * 3] * gp[i * 3] + gp[i * 3 + 1] * gp[i * 3 + 1] + gp[i * 3 + 2] * gp[i * 3 + 2]) - 1));
        Gs_Expect(worst < 0.03, $"vertices stay on the sphere (radius off by up to {worst:F4})");
        double vol = Mr_Volume(gp, gf), full = Mr_Volume(p, f);
        Gs_Expect(vol > 0 && Math.Abs(vol - full) < 0.04 * full, $"volume {vol:F4} of {full:F4}");
        // as close as the host's simplifier gets
        var (hp, hf) = MeshSimplifier.Simplify(p, f, target);
        double hv = Mr_Volume(hp, hf);
        Console.WriteLine($"[GpuSimplify] sphere: device volume {vol:F4}, host {hv:F4}, original {full:F4}; radius off by {worst:F4}; {rounds} rounds");
    });

    [TestMethod]
    public async Task GpuSimplify_Torus_KeepsItsHole() => await RunTest(async accelerator =>
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
        int target = f.Length / 3 / 6;
        var (gp, gf, _) = await Gs_Simplify(accelerator, p, f, target);
        Gs_Expect(gf.Length / 3 <= target * 1.03, $"faces {gf.Length / 3} for a budget of {target}");
        var t = MeshRepair.Measure(gp, gf);
        Gs_Expect(t.BoundaryEdges == 0 && t.NonManifoldEdges == 0, $"closed manifold ({t.BoundaryEdges} open, {t.NonManifoldEdges} non-manifold)");
        Gs_Expect(Gs_Euler(gp, gf) == 0, $"Euler characteristic {Gs_Euler(gp, gf)} (a torus has 0)");
    });

    [TestMethod]
    public async Task GpuSimplify_OpenSheet_KeepsItsBoundaryAndNeverFlips() => await RunTest(async accelerator =>
    {
        int n = 49;
        var p = new float[n * n * 3]; var f = new List<int>();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                float x = i / (float)(n - 1), y = j / (float)(n - 1);
                int o = (i * n + j) * 3; p[o] = x; p[o + 1] = y; p[o + 2] = 0.08f * MathF.Sin(6f * x) * MathF.Cos(5f * y);
                if (i < n - 1 && j < n - 1) { int a = i * n + j, b = (i + 1) * n + j, c = (i + 1) * n + j + 1, d = i * n + j + 1; f.AddRange(new[] { a, b, c, a, c, d }); }
            }
        var fa = f.ToArray(); int target = fa.Length / 3 / 10;
        double area0 = Gs_Area(p, fa);
        var (gp, gf, _) = await Gs_Simplify(accelerator, p, fa, target);
        Gs_Expect(gf.Length / 3 <= target * 1.05 && gf.Length / 3 >= target * 0.8, $"faces {gf.Length / 3} for a budget of {target}");
        // every open edge lies on the border of the square
        var edgeUse = new Dictionary<long, int>();
        for (int t = 0; t < gf.Length; t += 3)
            for (int k = 0; k < 3; k++) { int a = gf[t + k], b = gf[t + (k + 1) % 3]; long key = (long)Math.Min(a, b) * 4_000_000_007L + Math.Max(a, b); edgeUse[key] = edgeUse.GetValueOrDefault(key) + 1; }
        double Off(float[] q, int[] fq)
        {
            var use = new Dictionary<long, int>();
            for (int t = 0; t < fq.Length; t += 3)
                for (int k = 0; k < 3; k++) { int a = fq[t + k], b = fq[t + (k + 1) % 3]; long key = (long)Math.Min(a, b) * 4_000_000_007L + Math.Max(a, b); use[key] = use.GetValueOrDefault(key) + 1; }
            double worstOff = 0;
            foreach (var (key, uses) in use)
            {
                if (uses != 1) continue;
                foreach (int v in new[] { (int)(key / 4_000_000_007L), (int)(key % 4_000_000_007L) })
                    worstOff = Math.Max(worstOff, Math.Min(Math.Min(q[v * 3], 1 - q[v * 3]), Math.Min(q[v * 3 + 1], 1 - q[v * 3 + 1])));
            }
            return worstOff;
        }
        // the boundary penalty keeps the border where it was; the host's simplifier lets it drift by a little, and so does this (never by more than that)
        var (hp, hf) = MeshSimplifier.Simplify(p, fa, target);
        double offDevice = Off(gp, gf), offHost = Off(hp, hf);
        Console.WriteLine($"[GpuSimplify] open sheet: border vertices off the border by up to {offDevice:G3} (host {offHost:G3})");
        Gs_Expect(offDevice <= Math.Max(2 * offHost, 2e-3), $"boundary vertices left the border of the sheet by up to {offDevice:G3} (host {offHost:G3})");
        int flipped = 0;
        for (int t = 0; t < gf.Length; t += 3)
        {
            int a = gf[t] * 3, b = gf[t + 1] * 3, c = gf[t + 2] * 3;
            float nz = (gp[b] - gp[a]) * (gp[c + 1] - gp[a + 1]) - (gp[b + 1] - gp[a + 1]) * (gp[c] - gp[a]);
            if (nz <= 0) flipped++;
        }
        Gs_Expect(flipped == 0, $"{flipped} faces turned over (the sheet is a height field: every face must still face up in projection)");
        double area1 = Gs_Area(gp, gf);
        Gs_Expect(Math.Abs(area1 - area0) < 0.03 * area0, $"area {area1:F4} of {area0:F4}");
    });
}
