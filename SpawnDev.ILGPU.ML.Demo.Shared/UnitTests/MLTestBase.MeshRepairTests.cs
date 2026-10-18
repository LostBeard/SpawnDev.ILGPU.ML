using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.SpawnJS;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// <see cref="MeshRepair"/>: the clean-up the reference runs on the TRELLIS.2 dual-grid mesh before it simplifies and unwraps it. The generated
/// meshes are open, non-manifold and inconsistently wound (crown, 3.0M faces: 4.7% open edges, 227k non-manifold edges, 730k orientation conflicts).
/// Every defect class is injected into a closed sphere whose topology is known exactly, and the repaired mesh must be exactly the clean one.
/// </summary>
public abstract partial class MLTestBase
{
    private static (float[] P, int[] F) Mr_Sphere(int level)
    {
        double t = (1 + Math.Sqrt(5)) / 2;
        var v = new List<double[]>
        {
            new[]{-1,t,0}, new[]{1,t,0}, new[]{-1,-t,0}, new[]{1,-t,0}, new[]{0,-1,t}, new[]{0,1,t},
            new[]{0,-1,-t}, new[]{0,1,-t}, new[]{t,0,-1}, new[]{t,0,1}, new[]{-t,0,-1}, new[]{-t,0,1},
        };
        for (int i = 0; i < v.Count; i++) { var x = v[i]; double l = Math.Sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]); v[i] = new[] { x[0] / l, x[1] / l, x[2] / l }; }
        var f = new List<int[]>
        {
            new[]{0,11,5}, new[]{0,5,1}, new[]{0,1,7}, new[]{0,7,10}, new[]{0,10,11}, new[]{1,5,9}, new[]{5,11,4}, new[]{11,10,2}, new[]{10,7,6}, new[]{7,1,8},
            new[]{3,9,4}, new[]{3,4,2}, new[]{3,2,6}, new[]{3,6,8}, new[]{3,8,9}, new[]{4,9,5}, new[]{2,4,11}, new[]{6,2,10}, new[]{8,6,7}, new[]{9,8,1},
        };
        for (int s = 0; s < level; s++)
        {
            var cache = new Dictionary<long, int>(); var nf = new List<int[]>(f.Count * 4);
            int Mid(int a, int b)
            {
                long k = (long)Math.Min(a, b) * 1_000_000_007L + Math.Max(a, b);
                if (cache.TryGetValue(k, out int m)) return m;
                var x = new[] { v[a][0] + v[b][0], v[a][1] + v[b][1], v[a][2] + v[b][2] };
                double l = Math.Sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
                v.Add(new[] { x[0] / l, x[1] / l, x[2] / l }); return cache[k] = v.Count - 1;
            }
            foreach (var tri in f)
            {
                int a = tri[0], b = tri[1], c = tri[2], ab = Mid(a, b), bc = Mid(b, c), ca = Mid(c, a);
                nf.Add(new[] { a, ab, ca }); nf.Add(new[] { b, bc, ab }); nf.Add(new[] { c, ca, bc }); nf.Add(new[] { ab, bc, ca });
            }
            f = nf;
        }
        var p = new float[v.Count * 3]; for (int i = 0; i < v.Count; i++) for (int k = 0; k < 3; k++) p[i * 3 + k] = (float)v[i][k];
        var idx = new int[f.Count * 3]; for (int i = 0; i < f.Count; i++) for (int k = 0; k < 3; k++) idx[i * 3 + k] = f[i][k];
        return (p, idx);
    }

    private static double Mr_Volume(float[] p, int[] f)
    {
        double vol = 0;
        for (int t = 0; t < f.Length; t += 3)
        {
            int a = f[t], b = f[t + 1], c = f[t + 2];
            vol += p[a * 3] * (p[b * 3 + 1] * p[c * 3 + 2] - p[b * 3 + 2] * p[c * 3 + 1]) + p[a * 3 + 1] * (p[b * 3 + 2] * p[c * 3] - p[b * 3] * p[c * 3 + 2])
                 + p[a * 3 + 2] * (p[b * 3] * p[c * 3 + 1] - p[b * 3 + 1] * p[c * 3]);
        }
        return vol / 6;
    }

    private static void Mr_Expect(bool ok, string what) { if (!ok) throw new Exception(what); }
    private static void Mr_Clean(MeshRepair.Topology t, string what)
        => Mr_Expect(t.BoundaryEdges == 0 && t.NonManifoldEdges == 0 && t.OrientationConflicts == 0 && t.Components == 1, $"{what}: not a clean closed surface: {t}");

    /// <summary>A clean mesh must come out exactly as it went in (a repair that touches good meshes is worse than none).</summary>
    [TestMethod(Timeout = 120000)]
    public Task MeshRepair_LeavesACleanSphereAlone() => RunTest(async accelerator =>
    {
        var (p, f) = Mr_Sphere(4);
        var (rp, rf, r) = MeshRepair.Repair(p, f);
        Mr_Clean(r.Before, "input"); Mr_Clean(r.After, "output");
        Mr_Expect(rf.Length == f.Length && rp.Length == p.Length, $"size changed: {f.Length / 3} -> {rf.Length / 3} faces");
        Mr_Expect(r.DegenerateFaces == 0 && r.DuplicateFaces == 0 && r.VerticesSplit == 0 && r.FragmentFaces == 0 && r.HolesFilled == 0 && r.FacesFlipped == 0 && r.PiecesTurnedOutward == 0, $"a clean mesh was modified: {r}");
        Mr_Expect(Mr_Volume(rp, rf) > 0, "volume sign");
        await Task.CompletedTask;
    });

    /// <summary>Holes: a one-face hole is closed with a triangle, a six-face hole with a fan, and a hole longer than the limit is left open.</summary>
    [TestMethod(Timeout = 120000)]
    public Task MeshRepair_ClosesSmallHolesOnly() => RunTest(async accelerator =>
    {
        var (p, f) = Mr_Sphere(4);   // 5120 faces, edge ~0.07 on a unit sphere
        // one face missing: 3 open edges
        var one = f.Skip(3).ToArray();
        // a one-face hole is 3 edges of ~0.066 = 0.2 round; the default limit is 3% of the longest side (2.0) = 0.06, so it needs 0.5 here
        var (_, rf1, r1) = MeshRepair.Repair(p, one, new MeshRepair.Options { MaxHolePerimeter = 0.25 });
        Mr_Expect(r1.Before.BoundaryEdges == 3, $"setup: {r1.Before}");
        var (_, _, tooBig) = MeshRepair.Repair(p, one);
        Mr_Expect(tooBig.HolesFilled == 0 && tooBig.After.BoundaryEdges == 3, $"a 0.2 hole must stay open under the default 0.06 limit: {tooBig}");
        Mr_Clean(r1.After, "one-face hole"); Mr_Expect(r1.HolesFilled == 1 && rf1.Length == f.Length, $"one-face hole: {r1}");
        // all faces around vertex 0 missing (5 faces): a 5-edge loop, perimeter ~0.35, longest side 2 -> default limit 0.06 leaves it, 0.5 closes it
        var around = new List<int>(); int dropped = 0;
        for (int t = 0; t < f.Length; t += 3)
            if (f[t] == 0 || f[t + 1] == 0 || f[t + 2] == 0) dropped++; else { around.Add(f[t]); around.Add(f[t + 1]); around.Add(f[t + 2]); }
        var (rp2, rf2, left) = MeshRepair.Repair(p, around.ToArray());
        Mr_Expect(left.Before.BoundaryEdges >= 5 && left.After.BoundaryEdges == left.Before.BoundaryEdges && left.HolesFilled == 0, $"a hole over the limit must stay open: {left}");
        var (rp3, rf3, closed) = MeshRepair.Repair(p, around.ToArray(), new MeshRepair.Options { MaxHolePerimeter = 0.5 });
        Mr_Clean(closed.After, "fan-filled hole"); Mr_Expect(closed.HolesFilled == 1 && closed.HoleFacesAdded == closed.Before.BoundaryEdges, $"fan fill: {closed}");
        Mr_Expect(Mr_Volume(rp3, rf3) > 0, "filled sphere must keep positive volume");
        await Task.CompletedTask;
    });

    /// <summary>Winding: random faces flipped, and a whole sphere turned inside out, come back consistent and outward.</summary>
    [TestMethod(Timeout = 120000)]
    public Task MeshRepair_UnifiesWindingAndFacesOutward() => RunTest(async accelerator =>
    {
        var (p, f) = Mr_Sphere(4);
        var rng = new Random(7); var g = (int[])f.Clone(); int flips = 0;
        for (int t = 0; t < g.Length; t += 3) if (rng.Next(3) == 0) { (g[t + 1], g[t + 2]) = (g[t + 2], g[t + 1]); flips++; }
        var (rp, rf, r) = MeshRepair.Repair(p, g);
        Mr_Expect(r.Before.OrientationConflicts > 1000, $"setup: only {r.Before.OrientationConflicts} conflicts");
        Mr_Clean(r.After, "random flips"); Mr_Expect(Mr_Volume(rp, rf) > 0, "random flips: inside out after repair");
        var inv = new int[f.Length]; for (int t = 0; t < f.Length; t += 3) { inv[t] = f[t]; inv[t + 1] = f[t + 2]; inv[t + 2] = f[t + 1]; }
        Mr_Expect(Mr_Volume(p, inv) < 0, "setup: inverted sphere must have negative volume");
        var (ip, iF, ri) = MeshRepair.Repair(p, inv);
        Mr_Clean(ri.After, "inverted"); Mr_Expect(Mr_Volume(ip, iF) > 0 && ri.PiecesTurnedOutward == 1, $"inverted sphere not turned outward: {ri}");
        var (_, _, keep) = MeshRepair.Repair(p, inv, new MeshRepair.Options { OrientOutward = false });
        Mr_Expect(keep.PiecesTurnedOutward == 0, "OrientOutward=false must not turn pieces");
        await Task.CompletedTask;
    });

    /// <summary>Non-manifold: a book of three pages on one edge and a bow-tie vertex are split, every face kept, nothing left non-manifold.</summary>
    [TestMethod(Timeout = 120000)]
    public Task MeshRepair_SplitsNonManifoldEdgesAndPinchedVertices() => RunTest(async accelerator =>
    {
        // three triangles sharing edge 0-1 (apexes 2,3,4): 3 faces on one edge
        var p = new float[] { 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1 };
        var f = new[] { 0, 1, 2, 1, 0, 3, 0, 1, 4 };
        var (rp, rf, r) = MeshRepair.Repair(p, f, new MeshRepair.Options { MinComponentArea = 0 });
        Mr_Expect(r.Before.NonManifoldEdges == 1, $"setup: {r.Before}");
        Mr_Expect(r.After.NonManifoldEdges == 0 && rf.Length == f.Length, $"three-page book: {r}");
        // pages 2 (+y) and 3 (-y) are 180 degrees apart round the edge: one flat sheet, so they stay joined; page 4 (+z) is a fin and separates
        Mr_Expect(r.VerticesSplit > 0 && r.After.Components == 2, $"two opposite pages stay one sheet, the fin separates: {r}");
        // bow tie: two triangles meeting only at vertex 0
        var p2 = new float[] { 0, 0, 0, 1, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 0 };
        var f2 = new[] { 0, 1, 2, 0, 3, 4 };
        var (_, rf2, r2) = MeshRepair.Repair(p2, f2, new MeshRepair.Options { MinComponentArea = 0 });
        Mr_Expect(rf2.Length == 6 && r2.VerticesSplit == 1, $"bow tie must split vertex 0 once: {r2}");
        // two unit spheres touching at one vertex: a pinched closed surface -> two clean closed spheres
        var (sp, sf) = Mr_Sphere(3);
        int nv = sp.Length / 3; var two = new float[sp.Length * 2]; Array.Copy(sp, two, sp.Length);
        for (int i = 0; i < nv; i++) { two[sp.Length + i * 3] = sp[i * 3] + 2f; two[sp.Length + i * 3 + 1] = sp[i * 3 + 1]; two[sp.Length + i * 3 + 2] = sp[i * 3 + 2]; }
        int touch = -1, touch2 = -1; for (int i = 0; i < nv; i++) { if (sp[i * 3] > 0.9999f && Math.Abs(sp[i * 3 + 1]) < 1e-4f && Math.Abs(sp[i * 3 + 2]) < 1e-4f) touch = i; if (sp[i * 3] < -0.9999f && Math.Abs(sp[i * 3 + 1]) < 1e-4f && Math.Abs(sp[i * 3 + 2]) < 1e-4f) touch2 = i; }
        if (touch < 0 || touch2 < 0) throw new UnsupportedTestException("setup: no pole vertex on the x axis in this sphere");
        var tf = new int[sf.Length * 2]; Array.Copy(sf, tf, sf.Length);
        for (int i = 0; i < sf.Length; i++) tf[sf.Length + i] = sf[i] == touch2 ? touch : sf[i] + nv;   // sphere B's far pole IS sphere A's pole
        for (int i = 0; i < nv; i++) if (i == touch2) { two[sp.Length + i * 3] = sp[touch * 3]; two[sp.Length + i * 3 + 1] = sp[touch * 3 + 1]; two[sp.Length + i * 3 + 2] = sp[touch * 3 + 2]; }
        // sphere B must reach the pole: shift it so ITS touch2 vertex lands on A's touch vertex
        float dx = sp[touch * 3] - sp[touch2 * 3];
        for (int i = 0; i < nv; i++) two[sp.Length + i * 3] = sp[i * 3] + dx;
        var (tp, tfr, rt) = MeshRepair.Repair(two, tf);
        Mr_Expect(rt.After.NonManifoldEdges == 0 && rt.After.Components == 2 && rt.After.BoundaryEdges == 0 && rt.VerticesSplit >= 1, $"touching spheres: {rt}");
        await Task.CompletedTask;
    });

    private static MeshRepair.Topology Mr_WeldedTopology(float[] p, int[] f)
    {
        // what a loader that merges vertices by position (Blender, trimesh) would see
        double[] lo = { double.MaxValue, double.MaxValue, double.MaxValue }, hi = { double.MinValue, double.MinValue, double.MinValue };
        for (int i = 0; i < p.Length; i += 3) for (int k = 0; k < 3; k++) { lo[k] = Math.Min(lo[k], p[i + k]); hi[k] = Math.Max(hi[k], p[i + k]); }
        double diag = Math.Sqrt((hi[0] - lo[0]) * (hi[0] - lo[0]) + (hi[1] - lo[1]) * (hi[1] - lo[1]) + (hi[2] - lo[2]) * (hi[2] - lo[2])), q = 2e-6 * diag;
        var ids = new Dictionary<(long, long, long), int>(); var np = new List<float>(); var remap = new int[p.Length / 3];
        for (int v = 0; v < remap.Length; v++)
        {
            var key = ((long)Math.Round(p[v * 3] / q), (long)Math.Round(p[v * 3 + 1] / q), (long)Math.Round(p[v * 3 + 2] / q));
            if (!ids.TryGetValue(key, out int id)) { ids[key] = id = np.Count / 3; np.Add(p[v * 3]); np.Add(p[v * 3 + 1]); np.Add(p[v * 3 + 2]); }
            remap[v] = id;
        }
        return MeshRepair.Measure(np.ToArray(), f.Select(i => remap[i]).ToArray());
    }

    /// <summary>
    /// The repair must survive a loader that merges vertices by position. Split vertex copies are moved a hair toward their own fan; without that
    /// a weld glues them back together and every non-manifold edge returns (MEASURED on the crown: 175,233 non-manifold edges after a weld of a mesh
    /// whose own index topology had none).
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public Task MeshRepair_RepairSurvivesAWeldByPosition() => RunTest(async accelerator =>
    {
        var p = new float[] { 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0.5f, 0.5f, -1 };
        var f = new[] { 0, 1, 2, 1, 0, 3, 0, 1, 4, 0, 1, 5 };      // FOUR pages on edge 0-1: two sheets crossing along the edge
        var (rp, rf, r) = MeshRepair.Repair(p, f, new MeshRepair.Options { MinComponentArea = 0 });
        Mr_Expect(r.After.NonManifoldEdges == 0, $"setup: index topology must be manifold: {r}");
        var welded = Mr_WeldedTopology(rp, rf);
        Mr_Expect(welded.NonManifoldEdges == 0, $"a weld by position brought non-manifold edges back: {welded}");
        // the bow tie too: two triangles on one vertex
        var (bp, bf, _) = MeshRepair.Repair(new float[] { 0, 0, 0, 1, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 0 }, new[] { 0, 1, 2, 0, 3, 4 }, new MeshRepair.Options { MinComponentArea = 0 });
        Mr_Expect(Mr_WeldedTopology(bp, bf).Components == 2, "a welded bow tie must stay two pieces");
        await Task.CompletedTask;
    });

    /// <summary>Degenerate faces, exact duplicates and a floating speck go; a second real piece stays.</summary>
    [TestMethod(Timeout = 120000)]
    public Task MeshRepair_DropsDegenerateDuplicateAndFloatingFragments() => RunTest(async accelerator =>
    {
        var (p, f) = Mr_Sphere(3);
        int nv = p.Length / 3;
        // extras: a speck far from the sphere (area 5e-9 vs threshold 1e-5 * 4), a zero-area face, a repeated index face, a duplicate of face 0 (reversed)
        var pp = new List<float>(p) { 5f, 5f, 5f, 5.0001f, 5f, 5f, 5f, 5.0001f, 5f, 0.1f, 0.1f, 0.1f, 0.2f, 0.2f, 0.2f, 0.3f, 0.3f, 0.3f };
        var ff = new List<int>(f) { nv, nv + 1, nv + 2, nv + 3, nv + 4, nv + 5, 0, 0, 1, f[0], f[2], f[1] };
        var (rp, rf, r) = MeshRepair.Repair(pp.ToArray(), ff.ToArray());
        Mr_Expect(r.DegenerateFaces == 2 && r.DuplicateFaces == 1, $"degenerate/duplicate counts: {r}");
        Mr_Expect(r.FragmentComponents == 1 && r.FragmentFaces == 1, $"the speck must go: {r}");
        Mr_Clean(r.After, "sphere + junk"); Mr_Expect(rf.Length == f.Length && rp.Length == p.Length, $"sphere not restored exactly: {rf.Length / 3} faces, {rp.Length / 3} verts");
        // a second REAL piece (a unit-scale triangle) must stay: fragments are judged by area, not by being separate
        var big = new List<float>(p) { 3f, 0f, 0f, 4f, 0f, 0f, 3f, 1f, 0f };
        var bf = new List<int>(f) { nv, nv + 1, nv + 2 };
        var (_, rbf, rb) = MeshRepair.Repair(big.ToArray(), bf.ToArray());
        Mr_Expect(rb.FragmentFaces == 0 && rbf.Length == bf.Count, $"a real second piece was removed: {rb}");
        await Task.CompletedTask;
    });

    /// <summary>
    /// Production-scale damage on one mesh: ~330k faces with scattered holes, flipped faces, duplicates, specks and a pinched vertex. The repaired
    /// mesh must be a clean sphere again, within a bounded time (the browser runs this same code).
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public Task MeshRepair_ProductionScaleMixedDamage() => RunTest(async accelerator =>
    {
        bool browser = SpawnJSRuntime.Instance?.IsBrowser == true;
        var (p, f) = Mr_Sphere(browser ? 5 : 7);   // 20k faces in the browser lanes (interpreted Wasm), 327,680 on the desktop
        int nf = f.Length / 3; var rng = new Random(11);
        var g = new List<int>(f.Length + 3000);
        for (int t = 0; t < nf; t++)
        {
            int roll = rng.Next(1000);
            if (roll < 4) continue;                                            // 0.4% of faces missing: scattered 1-face holes
            int a = f[t * 3], b = f[t * 3 + 1], c = f[t * 3 + 2];
            if (roll < 54) (b, c) = (c, b);                                    // 5% flipped
            g.Add(a); g.Add(b); g.Add(c);
            if (roll >= 990) { g.Add(a); g.Add(c); g.Add(b); }                 // 1% duplicated (reversed)
        }
        // scattered holes of one face or a few adjacent faces (edges are 0.033 on the 20k-face sphere, 0.008 on the 327k one) -> allow 0.25 x 2.0 = 0.5 round
        var (rp, rf, r) = MeshRepair.Repair(p, g.ToArray(), new MeshRepair.Options { MaxHolePerimeter = 0.25 });
        Mr_Expect(r.Before.OrientationConflicts > nf / 20 && r.Before.BoundaryEdges > 0, $"setup: damage too small: {r.Before}");
        Mr_Clean(r.After, $"{nf}-face sphere with mixed damage");
        Mr_Expect(Mr_Volume(rp, rf) > 0, "outward after repair");
        Mr_Expect(Math.Abs(rf.Length / 3 - nf) <= nf / 100 + 10 || rf.Length / 3 >= nf, $"face count drifted: {nf} -> {rf.Length / 3}");
        Console.WriteLine($"[MeshRepair] {nf} faces damaged -> {rf.Length / 3}: {r}");
        await Task.CompletedTask;
    });
}
