using System.Numerics;

namespace SpawnDev.ILGPU.ML.Trellis2;

/// <summary>
/// UV atlas for a triangle mesh: faces are grown into charts bounded by a normal-deviation cone, each chart is projected on the
/// plane of its average normal and rotated to its minimum bounding rectangle, a chart whose projection overlaps itself is split
/// and retried, and the rectangles are shelf-packed into a square atlas at the largest uniform texel density that fits. Seams
/// duplicate vertices (<see cref="Result.SourceVertex"/> maps new to original). The reference does this with CuMesh's chart
/// clustering followed by xatlas (both MIT); this is a managed implementation of the same job with no native dependency.
/// This is the managed side of the pipeline (sequential, branchy topology work on a mesh that is about to be written to a file);
/// the bulk per-texel work (rasterise, sample, dilate) stays on the device.
/// </summary>
public static class UvAtlas
{
    public sealed class Options
    {
        /// <summary>Atlas side in texels (the packer targets this).</summary>
        public int Resolution = 2048;
        /// <summary>
        /// Empty texels kept around every chart (bleeding / dilation room). 0 = automatic: the widest of 4, 3, 2, 1 that still leaves every
        /// triangle at least one texel of room on average (a mesh of 100k+ small charts cannot afford 4 texels around each).
        /// </summary>
        public int PaddingTexels = 0;
        /// <summary>A face joins a chart while its normal is within this angle of the chart's average normal.</summary>
        public double MaxNormalDeviationDegrees = 40;
        /// <summary>
        /// A chart is scaled up (uniformly) until its longer side spans this many texels. Below ~2 texels a triangle can miss every texel centre,
        /// and the dilation then fills it from a neighbouring chart (wrong colour).
        /// </summary>
        public int MinChartTexels = 3;
        /// <summary>How many times an overlapping chart may be split with a tighter cone.</summary>
        public int MaxSplitDepth = 8;
    }

    public sealed class Result
    {
        public float[] Positions = Array.Empty<float>();   // V' * 3
        public float[] Uvs = Array.Empty<float>();         // V' * 2, [0,1], v up
        public int[] Faces = Array.Empty<int>();           // F * 3 (same face order as the input)
        public int[] SourceVertex = Array.Empty<int>();    // V' -> input vertex
        public int ChartCount;
        public double Utilization;                          // chart area / atlas area at the chosen density
        public double TexelsPerUnit;                        // texel density of the atlas
    }

    private sealed class Chart
    {
        public List<int> Faces = new();
        public double[] Uv = Array.Empty<double>();         // per face-corner (3 per face): projected, rotated, origin at (0,0)
        public double Width, Height;
        public double X, Y;                                 // packed origin, texels
        public double Scale = 1;                            // extra uniform scale a small chart gets (texels per unit = density * Scale)
    }

    public static Result Unwrap(float[] positions, int[] faces, Options? options = null)
    {
        options ??= new Options();
        int F = faces.Length / 3;
        var normal = new Vector3[F];
        var area = new double[F];
        for (int f = 0; f < F; f++)
        {
            var a = P(positions, faces[f * 3]); var b = P(positions, faces[f * 3 + 1]); var c = P(positions, faces[f * 3 + 2]);
            var cross = Vector3.Cross(b - a, c - a);
            double len = cross.Length();
            area[f] = 0.5 * len;
            normal[f] = len > 1e-20 ? cross / (float)len : new Vector3(0, 0, 1);
        }
        var prof = Environment.GetEnvironmentVariable("TRELLIS2_MESHPROFILE") == "1"; var psw = System.Diagnostics.Stopwatch.StartNew();
        var adj = BuildAdjacency(faces, F);
        if (prof) { Console.WriteLine($"  [atlas] normals+adjacency {psw.Elapsed.TotalSeconds:F2}s"); psw.Restart(); }

        // ---- charts ----
        var charts = new List<Chart>();
        var all = new List<int>(F);
        for (int f = 0; f < F; f++) all.Add(f);
        long tg = 0, tp = 0, to = 0;
        var stack = new Stack<(List<int> faces, double deg, int depth)>();
        stack.Push((all, options.MaxNormalDeviationDegrees, 0));
        var assigned = new int[F]; Array.Fill(assigned, -1);
        var inSet = new int[F]; int stamp = 0;
        while (stack.Count > 0)
        {
            var (set, deg, depth) = stack.Pop();
            stamp++;
            foreach (int f in set) inSet[f] = stamp;
            long g0 = System.Diagnostics.Stopwatch.GetTimestamp();
            var grown = Grow(set, adj, normal, area, inSet, stamp, Math.Cos(deg * Math.PI / 180));
            tg += System.Diagnostics.Stopwatch.GetTimestamp() - g0;
            foreach (var faceList in grown)
            {
                long p0 = System.Diagnostics.Stopwatch.GetTimestamp();
                var chart = Project(faceList, positions, faces, normal, area);
                long p1 = System.Diagnostics.Stopwatch.GetTimestamp(); tp += p1 - p0;
                bool ov = faceList.Count > 1 && depth < options.MaxSplitDepth && Overlaps(chart);
                to += System.Diagnostics.Stopwatch.GetTimestamp() - p1;
                if (ov)
                {
                    stack.Push((faceList, Math.Max(deg * 0.6, 3.0), depth + 1));
                    continue;
                }
                charts.Add(chart);
            }
        }

        if (prof) { double fq = System.Diagnostics.Stopwatch.Frequency; Console.WriteLine($"  [atlas] charts {psw.Elapsed.TotalSeconds:F2}s ({charts.Count}): grow {tg / fq:F2}s project {tp / fq:F2}s overlap {to / fq:F2}s"); psw.Restart(); }
        // ---- pack ----
        double totalArea = 0;
        foreach (var f in area) totalArea += f;
        int pad = options.PaddingTexels;
        double density;
        var byHeight = charts.ToArray();                    // shelf order (tallest first) is the same for every padding: sort once
        Array.Sort(byHeight.Select(c => -c.Height).ToArray(), byHeight);
        if (pad > 0) density = Pack(byHeight, options.Resolution, pad, options.MinChartTexels);
        else
        {
            // The widest padding that costs at most 20% of the best texel density. (Stopping at the first padding that reached ~1 texel per
            // mean edge took 4 texels around each of 32k charts on a generated chest: 5% of the atlas held texture, the rest was padding.)
            int[] candidates = { 4, 3, 2, 1 };
            var dens = candidates.Select(c => Pack(byHeight, options.Resolution, c, options.MinChartTexels)).ToArray();
            double best = dens.Max();
            int pick = Array.FindIndex(dens, d => d >= 0.8 * best);
            pad = candidates[pick];
            density = Pack(byHeight, options.Resolution, pad, options.MinChartTexels);   // commit the chosen layout
        }
        double used = 0;
        foreach (var c in charts) used += c.Width * density * c.Scale * c.Height * density * c.Scale;

        if (prof) { Console.WriteLine($"  [atlas] pack {psw.Elapsed.TotalSeconds:F2}s"); psw.Restart(); }
        // ---- emit ----
        var outPos = new List<float>(positions.Length);
        var outUv = new List<float>(positions.Length / 3 * 2);
        var src = new List<int>(positions.Length / 3);
        var outFaces = new int[F * 3];
        var remap = new int[positions.Length / 3]; var remapStamp = new int[remap.Length]; int rs = 0;
        double R = options.Resolution;
        foreach (var chart in charts)
        {
            rs++;
            for (int i = 0; i < chart.Faces.Count; i++)
            {
                int f = chart.Faces[i];
                for (int k = 0; k < 3; k++)
                {
                    int v = faces[f * 3 + k];
                    if (remapStamp[v] != rs)
                    {
                        remapStamp[v] = rs;
                        remap[v] = src.Count;
                        src.Add(v);
                        outPos.Add(positions[v * 3]); outPos.Add(positions[v * 3 + 1]); outPos.Add(positions[v * 3 + 2]);
                        // corner uv is per face; vertices of one chart share the same planar map, so any corner of v gives its uv
                        outUv.Add((float)((chart.X + pad + chart.Uv[(i * 3 + k) * 2] * density * chart.Scale) / R));
                        outUv.Add((float)((chart.Y + pad + chart.Uv[(i * 3 + k) * 2 + 1] * density * chart.Scale) / R));
                    }
                    outFaces[f * 3 + k] = remap[v];
                }
            }
        }
        return new Result
        {
            Positions = outPos.ToArray(), Uvs = outUv.ToArray(), Faces = outFaces, SourceVertex = src.ToArray(),
            ChartCount = charts.Count, TexelsPerUnit = density, Utilization = used / (R * R),
        };
    }

    private static Vector3 P(float[] p, int i) => new(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);

    private static int[] BuildAdjacency(int[] faces, int F)
    {
        // Sort the (edge key, face-edge) pairs and pair neighbours in the sorted order: O(n log n) with no hash. (A Dictionary keyed by the
        // packed edge hashes a ^ b, which collides massively for nearby vertex indices: it never finished on a 5M-triangle mesh.)
        var adj = new int[F * 3]; Array.Fill(adj, -1);
        var keys = new ulong[F * 3]; var slot = new int[F * 3];
        for (int f = 0; f < F; f++)
            for (int e = 0; e < 3; e++)
            {
                int a = faces[f * 3 + e], b = faces[f * 3 + (e + 1) % 3];
                keys[f * 3 + e] = a < b ? ((ulong)(uint)a << 32) | (uint)b : ((ulong)(uint)b << 32) | (uint)a;
                slot[f * 3 + e] = f * 3 + e;
            }
        Array.Sort(keys, slot);
        for (int i = 0; i + 1 < keys.Length; i++)
        {
            if (keys[i] != keys[i + 1]) continue;
            int s0 = slot[i], s1 = slot[i + 1];
            if (adj[s0] < 0 && adj[s1] < 0) { adj[s0] = s1 / 3; adj[s1] = s0 / 3; i++; }       // a third face on the same edge is left unlinked
        }
        return adj;
    }

    /// <summary>Region growing inside <paramref name="set"/> (membership by stamp): a face joins while it stays inside the cone around the average normal.</summary>
    private static List<List<int>> Grow(List<int> set, int[] adj, Vector3[] normal, double[] area, int[] inSet, int stamp, double cosThreshold)
    {
        var charts = new List<List<int>>();
        var done = new HashSet<int>();
        var queue = new Queue<int>();
        foreach (int seed in set)
        {
            if (done.Contains(seed)) continue;
            var faces = new List<int> { seed };
            done.Add(seed);
            Vector3 sum = normal[seed] * (float)Math.Max(area[seed], 1e-12);
            queue.Clear(); queue.Enqueue(seed);
            while (queue.Count > 0)
            {
                int f = queue.Dequeue();
                for (int e = 0; e < 3; e++)
                {
                    int nb = adj[f * 3 + e];
                    if (nb < 0 || inSet[nb] != stamp || done.Contains(nb)) continue;
                    var avg = Vector3.Normalize(sum);
                    if (Vector3.Dot(normal[nb], avg) < cosThreshold) continue;
                    done.Add(nb); faces.Add(nb); queue.Enqueue(nb);
                    sum += normal[nb] * (float)Math.Max(area[nb], 1e-12);
                }
            }
            charts.Add(faces);
        }
        return charts;
    }

    /// <summary>Planar projection along the chart's average normal, rotated to the minimum-area bounding rectangle.</summary>
    private static Chart Project(List<int> faceList, float[] positions, int[] faces, Vector3[] normal, double[] area)
    {
        var chart = new Chart { Faces = faceList };
        Vector3 sum = Vector3.Zero;
        foreach (int f in faceList) sum += normal[f] * (float)Math.Max(area[f], 1e-12);
        var d = Vector3.Normalize(sum.LengthSquared() > 0 ? sum : normal[faceList[0]]);
        var helper = Math.Abs(d.X) < 0.9f ? Vector3.UnitX : Vector3.UnitY;
        var u0 = Vector3.Normalize(Vector3.Cross(helper, d));
        var v0 = Vector3.Cross(d, u0);
        int n = faceList.Count;
        var raw = new double[n * 6];
        for (int i = 0; i < n; i++)
            for (int k = 0; k < 3; k++)
            {
                var p = P(positions, faces[faceList[i] * 3 + k]);
                raw[(i * 3 + k) * 2] = Vector3.Dot(p, u0);
                raw[(i * 3 + k) * 2 + 1] = Vector3.Dot(p, v0);
            }
        // minimum bounding rectangle over a sweep of rotations (the rectangle repeats every 90 degrees)
        double bestArea = double.MaxValue, bestCos = 1, bestSin = 0;
        double minX = 0, minY = 0, w = 0, h = 0;
        for (int step = 0; step < 24; step++)
        {
            double ang = step * (Math.PI / 2) / 24, c = Math.Cos(ang), s = Math.Sin(ang);
            double x0 = double.MaxValue, y0 = double.MaxValue, x1 = double.MinValue, y1 = double.MinValue;
            for (int i = 0; i < n * 3; i++)
            {
                double x = raw[i * 2] * c - raw[i * 2 + 1] * s, y = raw[i * 2] * s + raw[i * 2 + 1] * c;
                if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
            }
            double a = (x1 - x0) * (y1 - y0);
            if (a < bestArea) { bestArea = a; bestCos = c; bestSin = s; minX = x0; minY = y0; w = x1 - x0; h = y1 - y0; }
        }
        chart.Uv = new double[n * 6];
        for (int i = 0; i < n * 3; i++)
        {
            chart.Uv[i * 2] = raw[i * 2] * bestCos - raw[i * 2 + 1] * bestSin - minX;
            chart.Uv[i * 2 + 1] = raw[i * 2] * bestSin + raw[i * 2 + 1] * bestCos - minY;
        }
        chart.Width = Math.Max(w, 1e-9); chart.Height = Math.Max(h, 1e-9);
        return chart;
    }

    /// <summary>True when two triangles of the chart cover a common cell centre of a coarse grid (a fold in the projection).</summary>
    private static bool Overlaps(Chart chart)
    {
        int n = chart.Faces.Count;
        // grid side scales with the chart: a 3-triangle chart needs 8 x 8 cells, not 96 x 96 (121k such charts took 4.8 s at a fixed 96)
        int G = Math.Clamp(4 * (int)Math.Sqrt(n) + 4, 8, 96);
        double sx = G / chart.Width, sy = G / chart.Height;
        var cover = new byte[G * G];
        int overlap = 0, covered = 0;
        for (int i = 0; i < n; i++)
        {
            double ax = chart.Uv[(i * 3) * 2], ay = chart.Uv[(i * 3) * 2 + 1];
            double bx = chart.Uv[(i * 3 + 1) * 2], by = chart.Uv[(i * 3 + 1) * 2 + 1];
            double cx = chart.Uv[(i * 3 + 2) * 2], cy = chart.Uv[(i * 3 + 2) * 2 + 1];
            double det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
            if (Math.Abs(det) < 1e-18) continue;
            if (det < 0) return true;                       // flipped triangle: the projection folds
            int gx0 = (int)Math.Floor(Math.Min(ax, Math.Min(bx, cx)) * sx), gx1 = (int)Math.Ceiling(Math.Max(ax, Math.Max(bx, cx)) * sx);
            int gy0 = (int)Math.Floor(Math.Min(ay, Math.Min(by, cy)) * sy), gy1 = (int)Math.Ceiling(Math.Max(ay, Math.Max(by, cy)) * sy);
            for (int gy = Math.Max(gy0, 0); gy < Math.Min(gy1, G); gy++)
                for (int gx = Math.Max(gx0, 0); gx < Math.Min(gx1, G); gx++)
                {
                    double px = (gx + 0.5) / sx, py = (gy + 0.5) / sy;
                    double w0 = ((bx - px) * (cy - py) - (cx - px) * (by - py)) / det;
                    double w1 = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) / det;
                    double w2 = 1 - w0 - w1;
                    const double eps = 1e-6;
                    if (w0 > eps && w1 > eps && w2 > eps)
                    {
                        if (cover[gy * G + gx] != 0) overlap++; else { cover[gy * G + gx] = 1; covered++; }
                    }
                }
        }
        return overlap > 0 && overlap > covered / 200;      // tolerate isolated numerical ties
    }

    /// <summary>
    /// Shelf-pack the chart rectangles into a resolution x resolution atlas at the largest uniform density (texels per unit) that fits.
    /// Writes <see cref="Chart.X"/>/<see cref="Chart.Y"/> (texels, rectangle including its padding) and returns the density.
    /// </summary>
    private static double Pack(Chart[] order, int resolution, int padding, int minTexels)
    {
        bool Fits(double density, bool commit)
        {
            double x = 0, y = 0, shelfH = 0;
            foreach (var c in order)
            {
                double k = Math.Max(1.0, minTexels / Math.Max(Math.Max(c.Width, c.Height) * density, 1e-12));
                double w = Math.Ceiling(c.Width * density * k) + 2 * padding, h = Math.Ceiling(c.Height * density * k) + 2 * padding;
                if (w > resolution || h > resolution) return false;
                if (x + w > resolution) { y += shelfH; x = 0; shelfH = 0; }
                if (y + h > resolution) return false;
                if (commit) { c.X = x; c.Y = y; c.Scale = k; }
                x += w; if (h > shelfH) shelfH = h;
            }
            return true;
        }
        double lo = 0, hi = 1;
        while (Fits(hi, false)) { lo = hi; hi *= 2; if (hi > 1e9) break; }
        // 0.1% is far below a texel at any density that fits (each Fits walks every chart: 40 fixed halvings cost seconds interpreted)
        for (int i = 0; i < 40 && (lo <= 0 || hi > lo * 1.001); i++)
        {
            double mid = (lo + hi) / 2;
            if (mid <= 0) break;
            if (Fits(mid, false)) lo = mid; else hi = mid;
        }
        if (lo <= 0) lo = 1e-6;
        Fits(lo, true);
        return lo;
    }
}
