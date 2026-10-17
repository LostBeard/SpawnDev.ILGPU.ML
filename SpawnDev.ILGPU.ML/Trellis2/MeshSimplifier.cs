namespace SpawnDev.ILGPU.ML.Trellis2;

/// <summary>
/// Quadric-error edge-collapse simplification (Garland and Heckbert 1997) with boundary penalty quadrics, a link condition that keeps
/// the surface from pinching, and a face-flip guard. Sequential topology work on the host: the mesh has already left the device
/// to become a file, and the UV unwrap that follows is sequential too. The dual-grid mesh of a 512 shape has ~1M+ faces, far more
/// than a texture atlas can chart; ~300k keeps the shape (the bake samples the volume by position, so nothing is lost to it).
/// </summary>
public static class MeshSimplifier
{
    private readonly record struct Edge(int A, int B, int VerA, int VerB);

    /// <summary>Reduce <paramref name="faces"/> to about <paramref name="targetFaces"/>; returns compacted positions and faces.</summary>
    public static (float[] Positions, int[] Faces) Simplify(float[] positions, int[] faces, int targetFaces)
    {
        int nv = positions.Length / 3, nf = faces.Length / 3;
        if (nf <= targetFaces) return (positions, faces);
        var prof = Environment.GetEnvironmentVariable("TRELLIS2_MESHPROFILE") == "1"; var psw = System.Diagnostics.Stopwatch.StartNew();

        var pos = new double[nv * 3];
        for (int i = 0; i < pos.Length; i++) pos[i] = positions[i];
        var f = (int[])faces.Clone();
        var alive = new bool[nf]; Array.Fill(alive, true);

        // vertex -> faces (CSR seed, grown on demand when collapses move faces)
        var vf = new int[nv][]; var vfn = new int[nv];
        {
            var cnt = new int[nv];
            for (int i = 0; i < f.Length; i++) cnt[f[i]]++;
            for (int v = 0; v < nv; v++) vf[v] = new int[Math.Max(cnt[v], 1) + 2];
            for (int t = 0; t < nf; t++)
                for (int k = 0; k < 3; k++) { int v = f[t * 3 + k]; vf[v][vfn[v]++] = t; }
        }

        // quadrics: area-weighted face planes, plus a heavy perpendicular plane on every boundary edge
        var Q = new double[nv * 10];
        for (int t = 0; t < nf; t++)
        {
            int a = f[t * 3], b = f[t * 3 + 1], c = f[t * 3 + 2];
            Normal(pos, a, b, c, out double nx, out double ny, out double nz, out double area2);
            if (area2 <= 0) continue;
            double w = area2 * 0.5, len = area2;
            double ux = nx / len, uy = ny / len, uz = nz / len;
            double d = -(ux * pos[a * 3] + uy * pos[a * 3 + 1] + uz * pos[a * 3 + 2]);
            AddPlane(Q, a, ux, uy, uz, d, w); AddPlane(Q, b, ux, uy, uz, d, w); AddPlane(Q, c, ux, uy, uz, d, w);
        }
        for (int t = 0; t < nf; t++)
            for (int k = 0; k < 3; k++)
            {
                int a = f[t * 3 + k], b = f[t * 3 + (k + 1) % 3];
                if (SharedFaces(vf, vfn, f, alive, a, b) != 1) continue;
                Normal(pos, f[t * 3], f[t * 3 + 1], f[t * 3 + 2], out double nx, out double ny, out double nz, out double a2);
                if (a2 <= 0) continue;
                double ex = pos[b * 3] - pos[a * 3], ey = pos[b * 3 + 1] - pos[a * 3 + 1], ez = pos[b * 3 + 2] - pos[a * 3 + 2];
                // plane containing the edge and the face normal
                double px = ey * nz - ez * ny, py = ez * nx - ex * nz, pz = ex * ny - ey * nx;
                double pl = Math.Sqrt(px * px + py * py + pz * pz);
                if (pl <= 0) continue;
                px /= pl; py /= pl; pz /= pl;
                double d = -(px * pos[a * 3] + py * pos[a * 3 + 1] + pz * pos[a * 3 + 2]);
                double el2 = ex * ex + ey * ey + ez * ez;
                AddPlane(Q, a, px, py, pz, d, 100.0 * el2); AddPlane(Q, b, px, py, pz, d, 100.0 * el2);
            }

        var qs = new double[10];                       // scratch for Cost (no allocation per edge)
        var ver = new int[nv];
        var dead = new bool[nv];
        var heap = new PriorityQueue<Edge, double>(nf * 2);
        for (int t = 0; t < nf; t++)
            for (int k = 0; k < 3; k++)
            {
                int a = f[t * 3 + k], b = f[t * 3 + (k + 1) % 3];
                if (a > b) (a, b) = (b, a);
                if (a == b || !IsFirstFace(vf, vfn, f, alive, a, b, t)) continue;
                heap.Enqueue(new Edge(a, b, 0, 0), Cost(Q, pos, a, b, qs, out _, out _, out _));
            }

        if (prof) { Console.WriteLine($"  [simplify] setup {psw.Elapsed.TotalSeconds:F2}s, {heap.Count} edges"); psw.Restart(); }
        int aliveFaces = nf; long popped = 0, rejected = 0, stale = 0;
        var nbr = new List<int>(32); var na = new List<int>(32); var nb = new List<int>(32);
        while (aliveFaces > targetFaces && heap.TryDequeue(out var e, out double cost))
        {
            popped++;
            if (dead[e.A] || dead[e.B] || ver[e.A] != e.VerA || ver[e.B] != e.VerB) { stale++; continue; }
            Cost(Q, pos, e.A, e.B, qs, out double px, out double py, out double pz);
            if (!CanCollapse(pos, f, vf, vfn, alive, e.A, e.B, px, py, pz, na, nb)) { rejected++; continue; }

            // collapse B into A
            int a = e.A, b = e.B;
            pos[a * 3] = px; pos[a * 3 + 1] = py; pos[a * 3 + 2] = pz;
            for (int k = 0; k < 10; k++) Q[a * 10 + k] += Q[b * 10 + k];
            for (int i = 0; i < vfn[b]; i++)
            {
                int t = vf[b][i];
                if (!alive[t] || !Has(f, t, b)) continue;
                if (Has(f, t, a)) { alive[t] = false; aliveFaces--; continue; }
                for (int k = 0; k < 3; k++) if (f[t * 3 + k] == b) f[t * 3 + k] = a;
                if (vfn[a] == vf[a].Length) Array.Resize(ref vf[a], vf[a].Length * 2);
                vf[a][vfn[a]++] = t;
            }
            dead[b] = true; vfn[b] = 0; vf[b] = Array.Empty<int>();
            ver[a]++;

            // refresh the edges around the merged vertex
            Neighbours(f, vf, vfn, alive, a, nbr);
            Compact(f, vf, vfn, alive, a);
            foreach (int n in nbr)                         // ver[a] was bumped: every queued edge touching a is stale, these replace them
            {
                int lo = Math.Min(a, n), hi = Math.Max(a, n);
                heap.Enqueue(new Edge(lo, hi, ver[lo], ver[hi]), Cost(Q, pos, lo, hi, qs, out _, out _, out _));
            }
        }

        if (prof) Console.WriteLine($"  [simplify] collapse loop {psw.Elapsed.TotalSeconds:F2}s: popped {popped}, stale {stale}, rejected {rejected}");
        // compact
        var map = new int[nv]; Array.Fill(map, -1);
        int used = 0; var outFaces = new int[aliveFaces * 3]; int fi = 0;
        for (int t = 0; t < nf; t++)
        {
            if (!alive[t]) continue;
            for (int k = 0; k < 3; k++) { int v = f[t * 3 + k]; if (map[v] < 0) map[v] = used++; outFaces[fi++] = map[v]; }
        }
        var outPos = new float[used * 3];
        for (int v = 0; v < nv; v++)
            if (map[v] >= 0) { outPos[map[v] * 3] = (float)pos[v * 3]; outPos[map[v] * 3 + 1] = (float)pos[v * 3 + 1]; outPos[map[v] * 3 + 2] = (float)pos[v * 3 + 2]; }
        return (outPos, outFaces);
    }

    private static bool Has(int[] f, int t, int v) => f[t * 3] == v || f[t * 3 + 1] == v || f[t * 3 + 2] == v;

    private static void Normal(double[] p, int a, int b, int c, out double nx, out double ny, out double nz, out double area2)
    {
        double ux = p[b * 3] - p[a * 3], uy = p[b * 3 + 1] - p[a * 3 + 1], uz = p[b * 3 + 2] - p[a * 3 + 2];
        double vx = p[c * 3] - p[a * 3], vy = p[c * 3 + 1] - p[a * 3 + 1], vz = p[c * 3 + 2] - p[a * 3 + 2];
        nx = uy * vz - uz * vy; ny = uz * vx - ux * vz; nz = ux * vy - uy * vx;
        area2 = Math.Sqrt(nx * nx + ny * ny + nz * nz);
    }

    /// <summary>Q += w * [a b c d]^T [a b c d], packed upper triangle (aa ab ac ad bb bc bd cc cd dd).</summary>
    private static void AddPlane(double[] Q, int v, double a, double b, double c, double d, double w)
    {
        int o = v * 10;
        Q[o] += w * a * a; Q[o + 1] += w * a * b; Q[o + 2] += w * a * c; Q[o + 3] += w * a * d;
        Q[o + 4] += w * b * b; Q[o + 5] += w * b * c; Q[o + 6] += w * b * d;
        Q[o + 7] += w * c * c; Q[o + 8] += w * c * d; Q[o + 9] += w * d * d;
    }

    private static double Err(double[] q, int o, double x, double y, double z)
        => q[o] * x * x + 2 * q[o + 1] * x * y + 2 * q[o + 2] * x * z + 2 * q[o + 3] * x
         + q[o + 4] * y * y + 2 * q[o + 5] * y * z + 2 * q[o + 6] * y
         + q[o + 7] * z * z + 2 * q[o + 8] * z + q[o + 9];

    /// <summary>Cost of collapsing a-b and the position to collapse to: the quadric minimiser when the system is well conditioned, else the best of the ends and midpoint.</summary>
    private static double Cost(double[] Q, double[] pos, int a, int b, double[] q, out double x, out double y, out double z)
    {
        for (int k = 0; k < 10; k++) q[k] = Q[a * 10 + k] + Q[b * 10 + k];
        double ax = pos[a * 3], ay = pos[a * 3 + 1], az = pos[a * 3 + 2], bx = pos[b * 3], by = pos[b * 3 + 1], bz = pos[b * 3 + 2];
        double best = Err(q, 0, ax, ay, az); x = ax; y = ay; z = az;
        double eb = Err(q, 0, bx, by, bz); if (eb < best) { best = eb; x = bx; y = by; z = bz; }
        double mx = (ax + bx) * 0.5, my = (ay + by) * 0.5, mz = (az + bz) * 0.5;
        double em = Err(q, 0, mx, my, mz); if (em < best) { best = em; x = mx; y = my; z = mz; }
        // minimiser: [q0 q1 q2; q1 q4 q5; q2 q5 q7] p = -[q3 q6 q8]
        double a00 = q[0], a01 = q[1], a02 = q[2], a11 = q[4], a12 = q[5], a22 = q[7];
        double det = a00 * (a11 * a22 - a12 * a12) - a01 * (a01 * a22 - a12 * a02) + a02 * (a01 * a12 - a11 * a02);
        double scale = Math.Abs(a00) + Math.Abs(a11) + Math.Abs(a22);
        if (scale > 0 && Math.Abs(det) > 1e-10 * scale * scale * scale)
        {
            double r0 = -q[3], r1 = -q[6], r2 = -q[8];
            double ox = (r0 * (a11 * a22 - a12 * a12) - a01 * (r1 * a22 - a12 * r2) + a02 * (r1 * a12 - a11 * r2)) / det;
            double oy = (a00 * (r1 * a22 - a12 * r2) - r0 * (a01 * a22 - a12 * a02) + a02 * (a01 * r2 - r1 * a02)) / det;
            double oz = (a00 * (a11 * r2 - r1 * a12) - a01 * (a01 * r2 - r1 * a02) + r0 * (a01 * a12 - a11 * a02)) / det;
            // stay near the edge: a minimiser far outside it is an ill-conditioned artefact
            double ex = bx - ax, ey = by - ay, ez = bz - az, el = Math.Sqrt(ex * ex + ey * ey + ez * ez);
            double dx = ox - mx, dy = oy - my, dz = oz - mz;
            if (dx * dx + dy * dy + dz * dz <= el * el * 4)
            {
                double eo = Err(q, 0, ox, oy, oz);
                if (eo < best) { best = eo; x = ox; y = oy; z = oz; }
            }
        }
        return Math.Max(best, 0) + 1e-12 * (Math.Abs(ax - bx) + Math.Abs(ay - by) + Math.Abs(az - bz));
    }

    private static int SharedFaces(int[][] vf, int[] vfn, int[] f, bool[] alive, int a, int b)
    {
        int n = 0;
        for (int i = 0; i < vfn[a]; i++) { int t = vf[a][i]; if (alive[t] && Has(f, t, a) && Has(f, t, b)) n++; }
        return n;
    }

    private static bool IsFirstFace(int[][] vf, int[] vfn, int[] f, bool[] alive, int a, int b, int t)
    {
        for (int i = 0; i < vfn[a]; i++) { int u = vf[a][i]; if (u < t && alive[u] && Has(f, u, a) && Has(f, u, b)) return false; }
        return true;
    }

    private static void Neighbours(int[] f, int[][] vf, int[] vfn, bool[] alive, int v, List<int> outList)
    {
        outList.Clear();
        for (int i = 0; i < vfn[v]; i++)
        {
            int t = vf[v][i];
            if (!alive[t] || !Has(f, t, v)) continue;
            for (int k = 0; k < 3; k++)
            {
                int u = f[t * 3 + k];
                if (u != v && !outList.Contains(u)) outList.Add(u);
            }
        }
    }

    private static void Compact(int[] f, int[][] vf, int[] vfn, bool[] alive, int v)
    {
        int w = 0;
        for (int i = 0; i < vfn[v]; i++) { int t = vf[v][i]; if (alive[t] && Has(f, t, v)) vf[v][w++] = t; }
        vfn[v] = w;
    }

    /// <summary>Link condition (the two ends share exactly the vertices opposite the shared faces) + no face flips or collapses to a sliver.</summary>
    private static bool CanCollapse(double[] pos, int[] f, int[][] vf, int[] vfn, bool[] alive, int a, int b, double px, double py, double pz, List<int> na, List<int> nb)
    {
        Neighbours(f, vf, vfn, alive, a, na); Neighbours(f, vf, vfn, alive, b, nb);
        int common = 0;
        foreach (int n in na) if (n != b && nb.Contains(n)) common++;
        int shared = SharedFaces(vf, vfn, f, alive, a, b);
        if (shared == 0 || common != shared) return false;

        for (int side = 0; side < 2; side++)
        {
            int v = side == 0 ? a : b;
            for (int i = 0; i < vfn[v]; i++)
            {
                int t = vf[v][i];
                if (!alive[t] || !Has(f, t, v) || Has(f, t, side == 0 ? b : a)) continue;
                int i0 = f[t * 3], i1 = f[t * 3 + 1], i2 = f[t * 3 + 2];
                Normal(pos, i0, i1, i2, out double ox, out double oy, out double oz, out double oa);
                double x0 = pos[i0 * 3], y0 = pos[i0 * 3 + 1], z0 = pos[i0 * 3 + 2];
                double x1 = pos[i1 * 3], y1 = pos[i1 * 3 + 1], z1 = pos[i1 * 3 + 2];
                double x2 = pos[i2 * 3], y2 = pos[i2 * 3 + 1], z2 = pos[i2 * 3 + 2];
                if (i0 == v) { x0 = px; y0 = py; z0 = pz; } else if (i1 == v) { x1 = px; y1 = py; z1 = pz; } else { x2 = px; y2 = py; z2 = pz; }
                double ux = x1 - x0, uy = y1 - y0, uz = z1 - z0, vx = x2 - x0, vy = y2 - y0, vz = z2 - z0;
                double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
                double na2 = Math.Sqrt(nx * nx + ny * ny + nz * nz);
                if (na2 <= 1e-14) return false;
                if (oa > 0 && (nx * ox + ny * oy + nz * oz) / (na2 * oa) < 0.2) return false;
            }
        }
        return true;
    }
}
