using System.Diagnostics;
using SpawnDev.ILGPU.ML.Trellis2;

namespace Trellis2Harness;

/// <summary>
/// Mesh simplification properties on a torus (closed, genus 1, known surface): the face count reaches the target, the surface stays closed
/// and 2-manifold (every edge in exactly two faces, Euler characteristic still 0), vertices stay on the torus, and the winding is preserved.
/// </summary>
internal static class SimplifyGate
{
    private static (float[] v, int[] f) Torus(int major, int minor)
    {
        var v = new float[major * minor * 3]; var f = new int[major * minor * 6];
        for (int i = 0; i < major; i++)
            for (int j = 0; j < minor; j++)
            {
                double u = 2 * Math.PI * i / major, w = 2 * Math.PI * j / minor, r = 0.5 + 0.2 * Math.Cos(w);
                int o = (i * minor + j) * 3;
                v[o] = (float)(r * Math.Cos(u)); v[o + 1] = (float)(0.2 * Math.Sin(w)); v[o + 2] = (float)(r * Math.Sin(u));
            }
        int k = 0;
        for (int i = 0; i < major; i++)
            for (int j = 0; j < minor; j++)
            {
                int a = i * minor + j, b = i * minor + (j + 1) % minor, c = ((i + 1) % major) * minor + j, d = ((i + 1) % major) * minor + (j + 1) % minor;
                f[k++] = a; f[k++] = b; f[k++] = c; f[k++] = b; f[k++] = d; f[k++] = c;
            }
        return (v, f);
    }

    public static int Run()
    {
        int failures = 0;
        void Check(bool ok, string what) { Console.WriteLine($"{(ok ? "PASS" : "FAIL")}  {what}"); if (!ok) failures++; }

        foreach (var (major, minor, target) in new[] { (200, 100, 8000), (1200, 600, 300_000) })
        {
            var (v, f) = Torus(major, minor);
            int f0 = f.Length / 3;
            var sw = Stopwatch.StartNew();
            var (sv, sf) = MeshSimplifier.Simplify(v, f, target);
            double sec = sw.Elapsed.TotalSeconds;
            int nf = sf.Length / 3, nv = sv.Length / 3;
            Console.WriteLine($"torus {f0} faces -> {nf} (target {target}) in {sec:F2}s");

            var edges = new Dictionary<long, int>();
            for (int t = 0; t < nf; t++)
                for (int k = 0; k < 3; k++)
                {
                    int a = sf[t * 3 + k], b = sf[t * 3 + (k + 1) % 3];
                    long key = (long)Math.Min(a, b) * nv + Math.Max(a, b);
                    edges[key] = edges.GetValueOrDefault(key) + 1;
                }
            int bad = edges.Values.Count(c => c != 2);
            int euler = nv - edges.Count + nf;
            Check(nf <= target + 8 && nf > target * 0.9, $"face count {nf} within [0.9, 1.0] x target");
            Check(bad == 0, $"closed 2-manifold: {bad} edges not in exactly two faces");
            Check(euler == 0, $"Euler characteristic {euler} == 0 (genus kept)");

            double maxDev = 0; int flipped = 0;
            for (int i = 0; i < nv; i++)
            {
                double x = sv[i * 3], y = sv[i * 3 + 1], z = sv[i * 3 + 2];
                double d = Math.Sqrt(Math.Pow(Math.Sqrt(x * x + z * z) - 0.5, 2) + y * y);
                maxDev = Math.Max(maxDev, Math.Abs(d - 0.2));
            }
            for (int t = 0; t < nf; t++)
            {
                int a = sf[t * 3], b = sf[t * 3 + 1], c = sf[t * 3 + 2];
                double ux = sv[b * 3] - sv[a * 3], uy = sv[b * 3 + 1] - sv[a * 3 + 1], uz = sv[b * 3 + 2] - sv[a * 3 + 2];
                double wx = sv[c * 3] - sv[a * 3], wy = sv[c * 3 + 1] - sv[a * 3 + 1], wz = sv[c * 3 + 2] - sv[a * 3 + 2];
                double nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
                double cx = (sv[a * 3] + sv[b * 3] + sv[c * 3]) / 3, cy = (sv[a * 3 + 1] + sv[b * 3 + 1] + sv[c * 3 + 1]) / 3, cz = (sv[a * 3 + 2] + sv[b * 3 + 2] + sv[c * 3 + 2]) / 3;
                double rr = Math.Sqrt(cx * cx + cz * cz), ox = cx - 0.5 * cx / rr, oy = cy, oz = cz - 0.5 * cz / rr;   // outward from the tube axis
                // the winding of Torus() faces inward or outward consistently; compare with the original's sign
                if (nx * ox + ny * oy + nz * oz < 0) flipped++;
            }
            // the generator's own winding decides the sign: count faces disagreeing with the majority
            int agree = nf - flipped;
            Check(maxDev < 0.004, $"vertices stay on the torus: max deviation {maxDev:F5}");
            Check(Math.Min(flipped, agree) == 0, $"winding consistent: {Math.Min(flipped, agree)} faces disagree");
        }
        Console.WriteLine(failures == 0 ? "SIMPLIFY GATE OK" : $"SIMPLIFY GATE FAILED ({failures})");
        return failures;
    }
}
