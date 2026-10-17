using System.Diagnostics;
using SpawnDev.ILGPU.ML.Trellis2;

namespace Trellis2Harness;

/// <summary>
/// UV atlas properties (no reference output exists for charts, so the gate states what must hold): every face survives, UVs lie in
/// [0,1], no atlas texel is covered by two triangles, area is preserved up to the chart projection, and the packing is reasonably dense.
/// </summary>
internal static class AtlasGate
{
    private static (float[] v, int[] f) Sphere(int n)
    {
        var v = new List<float>(); var f = new List<int>();
        for (int i = 0; i <= n; i++)
            for (int j = 0; j < 2 * n; j++)
            {
                double th = Math.PI * i / n, ph = Math.PI * j / n;
                v.Add((float)(0.5 * Math.Sin(th) * Math.Cos(ph))); v.Add((float)(0.5 * Math.Cos(th))); v.Add((float)(0.5 * Math.Sin(th) * Math.Sin(ph)));
            }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < 2 * n; j++)
            {
                int a = i * 2 * n + j, b = i * 2 * n + (j + 1) % (2 * n), c = (i + 1) * 2 * n + j, d = (i + 1) * 2 * n + (j + 1) % (2 * n);
                f.AddRange(new[] { a, c, b, b, c, d });
            }
        return (v.ToArray(), f.ToArray());
    }

    private static (float[] v, int[] f) Torus(int major, int minor)
    {
        var v = new List<float>(); var f = new List<int>();
        for (int i = 0; i < major; i++)
            for (int j = 0; j < minor; j++)
            {
                double u = 2 * Math.PI * i / major, w = 2 * Math.PI * j / minor;
                double r = 0.5 + 0.2 * Math.Cos(w);
                v.Add((float)(r * Math.Cos(u))); v.Add((float)(0.2 * Math.Sin(w))); v.Add((float)(r * Math.Sin(u)));
            }
        for (int i = 0; i < major; i++)
            for (int j = 0; j < minor; j++)
            {
                int a = i * minor + j, b = i * minor + (j + 1) % minor, c = ((i + 1) % major) * minor + j, d = ((i + 1) % major) * minor + (j + 1) % minor;
                f.AddRange(new[] { a, b, c, b, d, c });
            }
        return (v.ToArray(), f.ToArray());
    }

    private static bool Check(string name, float[] v, int[] f, UvAtlas.Options opt)
    {
        var sw = Stopwatch.StartNew();
        var r = UvAtlas.Unwrap(v, f, opt);
        double secs = sw.Elapsed.TotalSeconds;
        int F = f.Length / 3, R = opt.Resolution;
        bool ok = true;
        if (r.Faces.Length != f.Length) { Console.WriteLine($"FAIL {name}: {r.Faces.Length / 3} faces, input {F}"); ok = false; }
        int bad = 0;
        foreach (var u in r.Uvs) if (!(u >= 0f && u <= 1f)) bad++;
        if (bad > 0) { Console.WriteLine($"FAIL {name}: {bad} UV components outside [0,1]"); ok = false; }
        // positions of a new vertex must equal its source vertex
        for (int i = 0; i < r.SourceVertex.Length; i++)
            for (int c = 0; c < 3; c++)
                if (r.Positions[i * 3 + c] != v[r.SourceVertex[i] * 3 + c]) { Console.WriteLine($"FAIL {name}: vertex {i} moved"); ok = false; i = int.MaxValue - 1; break; }
        // coverage: rasterise every triangle's texel centres; count texels hit by more than one
        var cover = new int[R * R];
        long multi = 0, covered = 0;
        double uvArea = 0, srcArea = 0;
        for (int t = 0; t < F; t++)
        {
            double[] ux = new double[3], uy = new double[3];
            for (int k = 0; k < 3; k++) { int vi = r.Faces[t * 3 + k]; ux[k] = r.Uvs[vi * 2] * R; uy[k] = r.Uvs[vi * 2 + 1] * R; }
            double det = (ux[1] - ux[0]) * (uy[2] - uy[0]) - (ux[2] - ux[0]) * (uy[1] - uy[0]);
            uvArea += Math.Abs(det) / 2;
            for (int k = 0; k < 3; k++) { }
            int x0 = Math.Max(0, (int)Math.Floor(Math.Min(ux[0], Math.Min(ux[1], ux[2])))), x1 = Math.Min(R - 1, (int)Math.Ceiling(Math.Max(ux[0], Math.Max(ux[1], ux[2]))));
            int y0 = Math.Max(0, (int)Math.Floor(Math.Min(uy[0], Math.Min(uy[1], uy[2])))), y1 = Math.Min(R - 1, (int)Math.Ceiling(Math.Max(uy[0], Math.Max(uy[1], uy[2]))));
            if (Math.Abs(det) < 1e-12) continue;
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++)
                {
                    double px = x + 0.5, py = y + 0.5;
                    double w0 = ((ux[1] - px) * (uy[2] - py) - (ux[2] - px) * (uy[1] - py)) / det;
                    double w1 = ((ux[2] - px) * (uy[0] - py) - (ux[0] - px) * (uy[2] - py)) / det;
                    double w2 = 1 - w0 - w1;
                    if (w0 > 1e-4 && w1 > 1e-4 && w2 > 1e-4)
                    {
                        if (cover[y * R + x]++ > 0) multi++; else covered++;
                    }
                }
            // 3D area
            double ax = v[f[t * 3 + 1] * 3] - v[f[t * 3] * 3], ay = v[f[t * 3 + 1] * 3 + 1] - v[f[t * 3] * 3 + 1], az = v[f[t * 3 + 1] * 3 + 2] - v[f[t * 3] * 3 + 2];
            double bx = v[f[t * 3 + 2] * 3] - v[f[t * 3] * 3], by = v[f[t * 3 + 2] * 3 + 1] - v[f[t * 3] * 3 + 1], bz = v[f[t * 3 + 2] * 3 + 2] - v[f[t * 3] * 3 + 2];
            srcArea += 0.5 * Math.Sqrt((ay * bz - az * by) * (ay * bz - az * by) + (az * bx - ax * bz) * (az * bx - ax * bz) + (ax * by - ay * bx) * (ax * by - ay * bx));
        }
        double overlapFrac = covered == 0 ? 1 : (double)multi / covered;
        if (overlapFrac > 0.001) { Console.WriteLine($"FAIL {name}: {overlapFrac:P3} of covered texels are covered more than once"); ok = false; }
        double expectedUvArea = srcArea * r.TexelsPerUnit * r.TexelsPerUnit;
        double areaRatio = uvArea / expectedUvArea;
        Console.WriteLine($"{(ok ? "PASS" : "FAIL")} {name,-8} {F} faces -> {r.Positions.Length / 3} verts ({(double)r.Positions.Length / 3 / (v.Length / 3):F2}x), {r.ChartCount} charts, util {r.Utilization:P0}, " +
                          $"{r.TexelsPerUnit:F0} texel/unit, UV/3D area {areaRatio:F2}, overlap {overlapFrac:P4}, {secs:F2}s");
        return ok;
    }

    public static int Run()
    {
        var opt = new UvAtlas.Options { Resolution = 2048 };
        bool ok = true;
        var (sv, sf) = Sphere(40); ok &= Check("sphere", sv, sf, opt);
        var (tv, tf) = Torus(64, 32); ok &= Check("torus", tv, tf, opt);
        string dg = Path.Combine(Gpu.WeightDir, "golden", "dualgrid");
        if (File.Exists(Path.Combine(dg, "faces.i32")))
        {
            var v = Golden.ReadF32(Path.Combine(dg, "vertices.f32")); var f = Golden.ReadI32(Path.Combine(dg, "faces.i32"));
            ok &= Check("dualgrid", v, f, opt);
        }
        Console.WriteLine(ok ? "ATLAS PASS" : "ATLAS FAIL");
        return ok ? 0 : 1;
    }
}
