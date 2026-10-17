using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.Trellis2;

namespace Trellis2Harness;

/// <summary>
/// Device mesh clustering probe: generate once, then (1) a host double-precision oracle of the level-1 cell errors (RMS distance of the
/// regularised quadric minimiser, normal cone) to check the device's merge decisions, (2) a sweep of thresholds -> face counts.
///   cluster &lt;raw prefix&gt;
/// </summary>
internal static class ClusterProbe
{
    public static int Run(string rawPrefix)
    {
        var size = File.ReadAllText(rawPrefix + ".size").Split(' ');
        int w = int.Parse(size[0]), h = int.Parse(size[1]);
        var bytes = File.ReadAllBytes(rawPrefix + ".u32");
        var px = new int[w * h]; Buffer.BlockCopy(bytes, 0, px, 0, bytes.Length);
        var (ctx, accel) = Gpu.Create();
        using var _c = ctx; using var _a = accel;
        using var rgba = accel.Allocate1D(px);
        using var gen = new Trellis2ImageTo3D(accel, Trellis2Weights.FromFiles(GenerateRun.Paths()));
        using var o = gen.Generate(rgba.View, w, h, new Trellis2Options { Seed = 42 }, out var pipe);
        int nv = o.Mesh.VertexCount, nf = o.Mesh.FaceCount, grid = o.Grid;
        Console.WriteLine($"mesh: {nv} vertices, {nf} faces, grid {grid}");
        accel.Synchronize();
        var verts = o.Mesh.Vertices.GetAsArray1D();
        var faces = o.Mesh.Faces.View.SubView(0, (long)nf * 3).GetAsArray1D();

        // ---- host oracle, level 1 --------------------------------------------------------------------------------------------------
        foreach (int level in new[] { 1, 2 })
        {
            var cell = new Dictionary<long, int>();
            var vc = new int[nv];
            int G = grid >> level;
            for (int v = 0; v < nv; v++)
            {
                long key = 0;
                for (int k = 0; k < 3; k++) { int q = Math.Clamp((int)Math.Floor((verts[v * 3 + k] + 0.5) * grid), 0, grid - 1) >> level; key = key * G + q; }
                if (!cell.TryGetValue(key, out int id)) { id = cell.Count; cell[key] = id; }
                vc[v] = id;
            }
            int m = cell.Count;
            var Q = new double[m * 10]; var N = new double[m * 3]; var A = new double[m]; var P = new double[m * 3]; var C = new int[m];
            var origin = new double[m * 3];
            foreach (var (key, id) in cell) { long k = key; int z = (int)(k % G); k /= G; int y = (int)(k % G); int x = (int)(k / G); origin[id * 3] = x << level; origin[id * 3 + 1] = y << level; origin[id * 3 + 2] = z << level; }
            for (int v = 0; v < nv; v++) { int c = vc[v]; for (int k = 0; k < 3; k++) P[c * 3 + k] += (verts[v * 3 + k] + 0.5) * grid - origin[c * 3 + k]; C[c]++; }
            for (int t = 0; t < nf; t++)
            {
                double[] p = new double[9];
                for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++) p[j * 3 + k] = (verts[faces[t * 3 + j] * 3 + k] + 0.5) * grid;
                double ux = p[3] - p[0], uy = p[4] - p[1], uz = p[5] - p[2], wx = p[6] - p[0], wy = p[7] - p[1], wz = p[8] - p[2];
                double nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx, len = Math.Sqrt(nx * nx + ny * ny + nz * nz);
                if (len <= 1e-12) continue;
                nx /= len; ny /= len; nz /= len; double wgt = len * 0.5;
                for (int j = 0; j < 3; j++)
                {
                    int c = vc[faces[t * 3 + j]];
                    double d = -(nx * (p[0] - origin[c * 3]) + ny * (p[1] - origin[c * 3 + 1]) + nz * (p[2] - origin[c * 3 + 2]));
                    int b = c * 10;
                    Q[b] += wgt * nx * nx; Q[b + 1] += wgt * nx * ny; Q[b + 2] += wgt * nx * nz; Q[b + 3] += wgt * nx * d;
                    Q[b + 4] += wgt * ny * ny; Q[b + 5] += wgt * ny * nz; Q[b + 6] += wgt * ny * d;
                    Q[b + 7] += wgt * nz * nz; Q[b + 8] += wgt * nz * d; Q[b + 9] += wgt * d * d;
                    N[c * 3] += wgt * nx; N[c * 3 + 1] += wgt * ny; N[c * 3 + 2] += wgt * nz; A[c] += wgt;
                }
            }
            var rms = new List<double>(); var cones = new List<double>();
            for (int c = 0; c < m; c++)
            {
                if (A[c] <= 0 || C[c] < 2) continue;
                double x = P[c * 3] / C[c], y = P[c * 3 + 1] / C[c], z = P[c * 3 + 2] / C[c];
                int b = c * 10; double lam = 1e-3 * A[c];
                double m00 = Q[b] + lam, m01 = Q[b + 1], m02 = Q[b + 2], m11 = Q[b + 4] + lam, m12 = Q[b + 5], m22 = Q[b + 7] + lam;
                double r0 = lam * x - Q[b + 3], r1 = lam * y - Q[b + 6], r2 = lam * z - Q[b + 8];
                double c00 = m11 * m22 - m12 * m12, c01 = m02 * m12 - m01 * m22, c02 = m01 * m12 - m02 * m11, det = m00 * c00 + m01 * c01 + m02 * c02;
                double c11 = m00 * m22 - m02 * m02, c12 = m01 * m02 - m00 * m12, c22 = m00 * m11 - m01 * m01;
                x = (c00 * r0 + c01 * r1 + c02 * r2) / det; y = (c01 * r0 + c11 * r1 + c12 * r2) / det; z = (c02 * r0 + c12 * r1 + c22 * r2) / det;
                double hi = (1 << level) + 0.5;
                x = Math.Clamp(x, -0.5, hi); y = Math.Clamp(y, -0.5, hi); z = Math.Clamp(z, -0.5, hi);
                double err = Q[b] * x * x + 2 * Q[b + 1] * x * y + 2 * Q[b + 2] * x * z + 2 * Q[b + 3] * x + Q[b + 4] * y * y + 2 * Q[b + 5] * y * z + 2 * Q[b + 6] * y + Q[b + 7] * z * z + 2 * Q[b + 8] * z + Q[b + 9];
                rms.Add(Math.Sqrt(Math.Max(err, 0) / A[c]));
                cones.Add(Math.Sqrt(N[c * 3] * N[c * 3] + N[c * 3 + 1] * N[c * 3 + 1] + N[c * 3 + 2] * N[c * 3 + 2]) / A[c]);
            }
            string Pct(List<double> l) { var s = l.OrderBy(v => v).ToArray(); return string.Join(" ", new[] { 0.1, 0.25, 0.5, 0.75, 0.9, 0.99 }.Select(q => $"p{q * 100:F0}={s[(int)(q * (s.Length - 1))]:F3}")); }
            int pass = 0; for (int i = 0; i < rms.Count; i++) if (rms[i] <= 0.3 && cones[i] >= 0.75) pass++;
            Console.WriteLine($"host L{level}: {m} cells ({rms.Count} multi-vertex), pass(0.3, 0.75) = {pass}");
            Console.WriteLine($"   rms  {Pct(rms)}");
            Console.WriteLine($"   cone {Pct(cones)}");
        }

        // ---- device sweep ------------------------------------------------------------------------------------------------------------
        var cl = new MeshClusterer(accel, new SparseKernels(accel));
        // TRELLIS2_SWEEP="err,cone,levels;..." ; TRELLIS2_SWEEP_GLB=dir writes a GLB per setting (cluster + edge collapse + atlas + bake)
        var spec = Environment.GetEnvironmentVariable("TRELLIS2_SWEEP") ?? "0.3,0.75,4;0.3,0,4;0.5,0,4;0.25,0.2,5;0.35,0,5;0.5,0,5";
        var glbDir = Environment.GetEnvironmentVariable("TRELLIS2_SWEEP_GLB");
        foreach (var item in spec.Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            if (item == "off" && glbDir != null)
            {   // the edge-collapse-only reference
                pipe.PreCluster = false;
                pipe.StageDone = (name, sec) => Console.WriteLine($"    {name,-40} {sec,7:F2}s");
                File.WriteAllBytes(Path.Combine(glbDir, "cl_off.glb"), pipe.ToGlb(o));
                pipe.PreCluster = true;
                continue;
            }
            var parts = item.Split(',');
            float err = float.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture), cone = float.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture);
            int lv = int.Parse(parts[2]);
            if (glbDir != null)
            {
                pipe.ClusterOptions = new MeshClusterer.Options { MaxError = err, MinCone = cone, MaxLevel = lv };
                pipe.StageDone = (name, sec) => Console.WriteLine($"    {name,-40} {sec,7:F2}s");
                var glb = pipe.ToGlb(o);
                var path = Path.Combine(glbDir, $"cl_{err}_{cone}_{lv}.glb");
                File.WriteAllBytes(path, glb);
                Console.WriteLine($"  wrote {path}");
            }
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var (p, f) = cl.ClusterAsync(o.Mesh.Vertices.View, nv, o.Mesh.Faces.View.SubView(0, (long)nf * 3), nf, grid, new MeshClusterer.Options { MaxError = err, MinCone = cone, MaxLevel = lv }).GetAwaiter().GetResult();
            Console.WriteLine($"device err {err} cone {cone} levels {lv}: {f.Length / 3} faces, {p.Length / 3} vertices, {sw.Elapsed.TotalSeconds:F2}s");
        }
        return 0;
    }
}
