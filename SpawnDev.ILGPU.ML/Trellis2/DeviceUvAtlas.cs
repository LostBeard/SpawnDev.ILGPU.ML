using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// The mesh tail of the TRELLIS.2 export on the device: winding aligned to the vertex normals, vertex normals, and the UV atlas (<see cref="UvAtlas"/>'s job: charts bounded by a normal cone, planar projection rotated to the minimum bounding
/// rectangle, shelf packing at the largest uniform texel density, seams duplicate vertices). The mesh never leaves the GPU; a few counts and the packing's density candidates cross to the host.
/// <list type="bullet">
/// <item>Charts grow in parallel from seeds (random-priority local minima within a few rings, re-seeded among what is left until every face is in a chart); a face joins the neighbouring chart whose average normal it is closest to, if it lies inside the cone.
/// All joins of a round see the averages as the round began, so the result does not depend on thread timing.</item>
/// <item>A chart whose faces lean more than 65 degrees from its final average normal (the projection would fold) dissolves, and its faces grow again with a tighter cone, like the host version's split on overlap.</item>
/// <item>The minimum bounding rectangle comes from 24 rotations, every corner folding its coordinates into per-chart bounds with atomic min/max.</item>
/// <item>Packing: charts bucketed by height (tallest first), then many candidate densities per padding are tried at once, each one a shelf pass over every chart on its own thread; three rounds narrow the bracket to 0.003%.</item>
/// <item>Seams: a vertex is duplicated once per distinct chart among its faces, found from the vertex-to-faces table, so no hash and no sort.</item>
/// </list>
/// </summary>
public sealed class DeviceUvAtlas
{
    private readonly Accelerator _acc;
    private readonly SparseKernels _sk;
    public DeviceUvAtlas(Accelerator accelerator, SparseKernels sparse) { _acc = accelerator; _sk = sparse; }

    /// <summary>The unwrapped mesh in device buffers: the vertices duplicated along seams, their UVs in [0,1] (v up), the faces in the input order, and the normals of the source vertices.</summary>
    public sealed class Result : IDisposable
    {
        public MemoryBuffer1D<float, Stride1D.Dense> Positions = null!;      // V' * 3
        public MemoryBuffer1D<float, Stride1D.Dense> Uvs = null!;            // V' * 2
        public MemoryBuffer1D<float, Stride1D.Dense> Normals = null!;        // V' * 3 (the unit normal of the vertex's source vertex, area weighted)
        public MemoryBuffer1D<int, Stride1D.Dense> Faces = null!;            // F * 3, same face order as the input
        public MemoryBuffer1D<int, Stride1D.Dense> SourceVertex = null!;     // V' -> input vertex
        public int VertexCount, FaceCount, ChartCount;
        public double Utilization, TexelsPerUnit;
        /// <summary>Triangles flipped by the winding alignment (the input faces were changed in place).</summary>
        public int Flipped;
        public void Dispose() { Positions?.Dispose(); Uvs?.Dispose(); Normals?.Dispose(); Faces?.Dispose(); SourceVertex?.Dispose(); }
    }

    /// <summary>What the last <see cref="UnwrapAsync"/> did, step by step (diagnostic).</summary>
    public List<string> Log { get; } = new();

    private const int PS = 16;          // floats per chart: u0 (3), v0 (3), cos, sin, minX, minY, w, h, scale, x, y, -
    private const int Angles = 24;
    private const int Buckets = 512;
    private const int Candidates = 64;
    private const int MaxChartChunk = 65536;
    private const int WeightScale = 1024;

    // ---- the vertex -> faces table -------------------------------------------------------------------------------------------------------------------
    /// <summary>Each vertex's list in ascending face order, so every later choice made from it is the same on every run.</summary>
    private static void SortListImpl(Index1D v, IV vstart, IV vfaces)
    {
        int s = vstart[v], e = vstart[v + 1];
        for (int i = s + 1; i < e; i++)
        {
            int x = vfaces[i], j = i - 1;
            while (j >= s && vfaces[j] > x) { vfaces[j + 1] = vfaces[j]; j--; }
            vfaces[j + 1] = x;
        }
    }

    private static bool FaceHas(IV faces, int f, int v) => faces[f * 3] == v || faces[f * 3 + 1] == v || faces[f * 3 + 2] == v;

    // ---- normals and winding -------------------------------------------------------------------------------------------------------------------------
    private static void VertexNormalsImpl(Index1D v, FV pos, IV faces, IV vstart, IV vfaces, FV vn)
    {
        float sx = 0f, sy = 0f, sz = 0f;
        for (int i = vstart[v]; i < vstart[v + 1]; i++)
        {
            int f = vfaces[i], a = faces[f * 3], b = faces[f * 3 + 1], c = faces[f * 3 + 2];
            float ax = pos[b * 3] - pos[a * 3], ay = pos[b * 3 + 1] - pos[a * 3 + 1], az = pos[b * 3 + 2] - pos[a * 3 + 2];
            float bx = pos[c * 3] - pos[a * 3], by = pos[c * 3 + 1] - pos[a * 3 + 1], bz = pos[c * 3 + 2] - pos[a * 3 + 2];
            sx += ay * bz - az * by; sy += az * bx - ax * bz; sz += ax * by - ay * bx;      // area-weighted (length = 2 * area)
        }
        float l = MathF.Sqrt(sx * sx + sy * sy + sz * sz);
        if (l < 1e-20f) { vn[v * 3] = 0f; vn[v * 3 + 1] = 0f; vn[v * 3 + 2] = 1f; return; }
        vn[v * 3] = sx / l; vn[v * 3 + 1] = sy / l; vn[v * 3 + 2] = sz / l;
    }

    /// <summary>A triangle whose own normal opposes the sum of its corners' normals by more than cos = -0.2 is wound the wrong way: swap two corners (as <see cref="MeshRepair.AlignWindingToNormals"/>).</summary>
    private static void AlignFlipImpl(Index1D t, FV pos, IV faces, FV vn, IV counter)
    {
        int a = faces[t * 3], b = faces[t * 3 + 1], c = faces[t * 3 + 2];
        float ax = pos[b * 3] - pos[a * 3], ay = pos[b * 3 + 1] - pos[a * 3 + 1], az = pos[b * 3 + 2] - pos[a * 3 + 2];
        float bx = pos[c * 3] - pos[a * 3], by = pos[c * 3 + 1] - pos[a * 3 + 1], bz = pos[c * 3 + 2] - pos[a * 3 + 2];
        float nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
        float sx = vn[a * 3] + vn[b * 3] + vn[c * 3], sy = vn[a * 3 + 1] + vn[b * 3 + 1] + vn[c * 3 + 1], sz = vn[a * 3 + 2] + vn[b * 3 + 2] + vn[c * 3 + 2];
        float ln = MathF.Sqrt(nx * nx + ny * ny + nz * nz), ls = MathF.Sqrt(sx * sx + sy * sy + sz * sz);
        if (ln < 1e-20f || ls < 1e-12f) return;
        if ((nx * sx + ny * sy + nz * sz) / (ln * ls) >= -0.2f) return;
        faces[t * 3 + 1] = c; faces[t * 3 + 2] = b;
        Atomic.Add(ref counter[0], 1);
    }

    // ---- face data -----------------------------------------------------------------------------------------------------------------------------------
    private static void FaceInfoImpl(Index1D f, FV pos, IV faces, FV fn, FV area, IV amaxBits)
    {
        int a = faces[f * 3], b = faces[f * 3 + 1], c = faces[f * 3 + 2];
        float ax = pos[b * 3] - pos[a * 3], ay = pos[b * 3 + 1] - pos[a * 3 + 1], az = pos[b * 3 + 2] - pos[a * 3 + 2];
        float bx = pos[c * 3] - pos[a * 3], by = pos[c * 3 + 1] - pos[a * 3 + 1], bz = pos[c * 3 + 2] - pos[a * 3 + 2];
        float nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
        float len = MathF.Sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 1e-20f) { fn[f * 3] = nx / len; fn[f * 3 + 1] = ny / len; fn[f * 3 + 2] = nz / len; }
        else { fn[f * 3] = 0f; fn[f * 3 + 1] = 0f; fn[f * 3 + 2] = 1f; }
        float ar = 0.5f * len;
        area[f] = ar;
        Atomic.Max(ref amaxBits[0], (int)Interop.FloatAsInt(ar));        // non-negative floats order like their bits
    }

    /// <summary>The area weight of a face in a chart's average normal, in fixed point (floats cannot be added atomically): 1/32 .. 1 of the largest face, so a chart of tiny faces still has a direction.</summary>
    private static void FaceWeightImpl(Index1D f, FV area, IV amaxBits, IV wq)
    {
        float amax = MathF.Max(Interop.IntAsFloat((uint)amaxBits[0]), 1e-30f);
        wq[f] = (int)(WeightScale * MathF.Max(area[f] / amax, 1f / 32f));
    }

    private static void AdjacencyImpl(Index1D f, IV faces, IV vstart, IV vfaces, IV adj)
    {
        for (int e = 0; e < 3; e++)
        {
            int a = faces[f * 3 + e], b = faces[f * 3 + (e + 1) % 3], nb = -1;
            for (int i = vstart[a]; i < vstart[a + 1]; i++)
            {
                int g = vfaces[i];
                if (g != f && FaceHas(faces, g, b)) { nb = g; break; }       // the lists are ascending: the lowest face on the edge
            }
            adj[f * 3 + e] = nb;
        }
    }

    // ---- seeding and growing -------------------------------------------------------------------------------------------------------------------------
    private static void PriorityImpl(Index1D f, IV prio, int idBits)
    {
        uint h = (uint)f * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        uint hashBits = (h & ((1u << (30 - idBits)) - 1u));
        prio[f] = (int)((hashBits << idBits) | (uint)f);                       // unique: the face index fills the low bits
    }

    private static void PropInitImpl(Index1D f, IV label, IV prio, IV val) { val[f] = label[f] < 0 ? prio[f] : int.MaxValue; }

    /// <summary>
    /// The smallest priority within one more ring, through unlabelled faces only. The loop has its condition inside it on purpose: as a loop inside an <c>if (label[f] &lt; 0)</c> it lost its result
    /// under SpawnDev.ILGPU 5.3.8's loop unrolling (fixed in the fork after 5.3.8, see its CHANGELOG), which made every face a seed.
    /// </summary>
    private static void PropStepImpl(Index1D f, IV adj, IV label, IV vin, IV vout)
    {
        int m = vin[f];
        bool free = label[f] < 0;
        for (int e = 0; e < 3; e++)
        {
            int g = adj[f * 3 + e];
            if (free && g >= 0)
            {
                int vg = vin[g];
                if (label[g] < 0 && vg < m) m = vg;
            }
        }
        vout[f] = m;
    }

    private static void SeedImpl(Index1D f, IV label, IV prio, IV val, FV fn, IV wq, IV sumN, IV counter)
    {
        if (label[f] >= 0 || val[f] != prio[f]) return;
        label[f] = f;
        int w = wq[f];
        sumN[f * 3] = (int)MathF.Floor(fn[f * 3] * w + 0.5f); sumN[f * 3 + 1] = (int)MathF.Floor(fn[f * 3 + 1] * w + 0.5f); sumN[f * 3 + 2] = (int)MathF.Floor(fn[f * 3 + 2] * w + 0.5f);
        Atomic.Add(ref counter[0], 1);
    }

    /// <summary>An unlabelled face joins the best neighbouring chart (smallest angle to the chart's average normal, then lowest id) when it is inside the cone. Reads <paramref name="label"/>, writes <paramref name="newLabel"/>.</summary>
    private static void GrowImpl(Index1D f, IV adj, IV label, IV newLabel, FV fn, IV sumN, IV frozen, FV cone)
    {
        if (label[f] >= 0) return;
        float nx = fn[f * 3], ny = fn[f * 3 + 1], nz = fn[f * 3 + 2];
        int best = -1; float bestDot = -2f;
        for (int e = 0; e < 3; e++)
        {
            int g = adj[f * 3 + e];
            if (g < 0) continue;
            int c = label[g];
            if (c < 0 || frozen[c] != 0) continue;
            float sx = sumN[c * 3], sy = sumN[c * 3 + 1], sz = sumN[c * 3 + 2];
            float l = MathF.Sqrt(sx * sx + sy * sy + sz * sz);
            if (l <= 0f) continue;
            float d = (nx * sx + ny * sy + nz * sz) / l;
            if (d >= cone[f] && (d > bestDot || (d == bestDot && c < best))) { best = c; bestDot = d; }
        }
        if (best >= 0) newLabel[f] = best;
    }

    private static void AccumImpl(Index1D f, IV label, IV newLabel, FV fn, IV wq, IV sumN, IV counter)
    {
        int c = newLabel[f];
        if (label[f] >= 0 || c < 0) return;
        int w = wq[f];
        Atomic.Add(ref sumN[c * 3], (int)MathF.Floor(fn[f * 3] * w + 0.5f));
        Atomic.Add(ref sumN[c * 3 + 1], (int)MathF.Floor(fn[f * 3 + 1] * w + 0.5f));
        Atomic.Add(ref sumN[c * 3 + 2], (int)MathF.Floor(fn[f * 3 + 2] * w + 0.5f));
        Atomic.Add(ref counter[0], 1);
    }

    private static void CountUnlabelledImpl(Index1D f, IV label, IV counter) { if (label[f] < 0) Atomic.Add(ref counter[0], 1); }

    private static void FillFloatImpl(Index1D i, FV v, float value) { v[i] = value; }
    private static void FillIntImpl(Index1D i, IV v, int value) { v[i] = value; }

    // ---- splitting charts whose projection would fold ------------------------------------------------------------------------------------------------
    private static void BadImpl(Index1D f, IV label, FV fn, IV sumN, IV bad, float cosBad)
    {
        int c = label[f];
        float sx = sumN[c * 3], sy = sumN[c * 3 + 1], sz = sumN[c * 3 + 2];
        float l = MathF.Sqrt(sx * sx + sy * sy + sz * sz);
        float d = l > 0f ? (fn[f * 3] * sx + fn[f * 3 + 1] * sy + fn[f * 3 + 2] * sz) / l : 1f;
        if (d < cosBad) Atomic.Max(ref bad[c], 1);
    }

    /// <summary>Faces of a bad chart become unlabelled with a tighter cone; every other chart is frozen so nothing joins it any more.</summary>
    private static void DissolveImpl(Index1D f, IV label, IV bad, IV frozen, FV cone, IV counter, float newCone)
    {
        int c = label[f];
        if (c < 0) return;
        if (bad[c] != 0) { label[f] = -1; cone[f] = newCone; frozen[c] = 0; Atomic.Add(ref counter[0], 1); }
        else frozen[c] = 1;
    }

    // ---- charts: ids, basis, bounding rectangle -------------------------------------------------------------------------------------------------------
    private static void RootFlagImpl(Index1D f, IV label, IV flag) { flag[f] = label[f] == f ? 1 : 0; }
    private static void ChartOfFaceImpl(Index1D f, IV label, IV cid, IV chartOf) { chartOf[f] = cid[label[f]]; }

    private static void BasisImpl(Index1D f, IV label, IV cid, IV sumN, FV fn, FV prm)
    {
        if (label[f] != f) return;
        int c = cid[f];
        float sx = sumN[f * 3], sy = sumN[f * 3 + 1], sz = sumN[f * 3 + 2];
        float l = MathF.Sqrt(sx * sx + sy * sy + sz * sz);
        float dx, dy, dz;
        if (l > 0f) { dx = sx / l; dy = sy / l; dz = sz / l; } else { dx = fn[f * 3]; dy = fn[f * 3 + 1]; dz = fn[f * 3 + 2]; }
        // helper axis: X unless the normal is close to it, then Y (as the host version)
        float hx = MathF.Abs(dx) < 0.9f ? 1f : 0f, hy = MathF.Abs(dx) < 0.9f ? 0f : 1f;
        float ux = hy * dz, uy = -hx * dz, uz = hx * dy - hy * dx;           // cross(helper, d), helper = (hx, hy, 0)
        float ul = MathF.Sqrt(ux * ux + uy * uy + uz * uz);
        ux /= ul; uy /= ul; uz /= ul;
        float vx = dy * uz - dz * uy, vy = dz * ux - dx * uz, vz = dx * uy - dy * ux;   // cross(d, u)
        int o = c * PS;
        prm[o] = ux; prm[o + 1] = uy; prm[o + 2] = uz; prm[o + 3] = vx; prm[o + 4] = vy; prm[o + 5] = vz;
    }

    private static int Ordered(float x) { int bits = (int)Interop.FloatAsInt(x); return bits >= 0 ? bits : bits ^ 0x7FFFFFFF; }
    private static float FromOrdered(int o) => Interop.IntAsFloat((uint)(o >= 0 ? o : o ^ 0x7FFFFFFF));

    private static void BoundsInitImpl(Index1D i, IV bounds) { bounds[i] = (i & 1) == 0 ? int.MaxValue : int.MinValue; }

    /// <summary>Every face corner of a chart in [c0, c0 + chunk) folds its rotated coordinates into the chart's bounds, for each of the 24 rotations.</summary>
    private static void BoundsImpl(Index1D idx, FV pos, IV faces, IV chartOf, FV prm, IV bounds, int c0, int chunk)
    {
        int f = idx / 3;
        int c = chartOf[f] - c0;
        if (c < 0 || c >= chunk) return;
        int v = faces[idx], o = (c + c0) * PS;
        float px = pos[v * 3], py = pos[v * 3 + 1], pz = pos[v * 3 + 2];
        float x = px * prm[o] + py * prm[o + 1] + pz * prm[o + 2], y = px * prm[o + 3] + py * prm[o + 4] + pz * prm[o + 5];
        for (int a = 0; a < Angles; a++)
        {
            float ang = a * (1.5707964f / Angles), cs = MathF.Cos(ang), sn = MathF.Sin(ang);
            float rx = x * cs - y * sn, ry = x * sn + y * cs;
            int b = (c * Angles + a) * 4;
            int ox = Ordered(rx), oy = Ordered(ry);
            Atomic.Min(ref bounds[b], ox); Atomic.Max(ref bounds[b + 1], ox);
            Atomic.Min(ref bounds[b + 2], oy); Atomic.Max(ref bounds[b + 3], oy);
        }
    }

    /// <summary>The rotation with the smallest bounding rectangle (the first of equals, as the host's strict comparison).</summary>
    private static void PickImpl(Index1D i, IV bounds, FV prm, IV hbits, int c0)
    {
        int c = c0 + i;
        float bestArea = 3.0e38f, bc = 1f, bs = 0f, mx = 0f, my = 0f, bw = 0f, bh = 0f;
        for (int a = 0; a < Angles; a++)
        {
            int b = (i * Angles + a) * 4;
            float x0 = FromOrdered(bounds[b]), x1 = FromOrdered(bounds[b + 1]), y0 = FromOrdered(bounds[b + 2]), y1 = FromOrdered(bounds[b + 3]);
            float ar = (x1 - x0) * (y1 - y0);
            if (ar < bestArea)
            {
                float ang = a * (1.5707964f / Angles);
                bestArea = ar; bc = MathF.Cos(ang); bs = MathF.Sin(ang); mx = x0; my = y0; bw = x1 - x0; bh = y1 - y0;
            }
        }
        bw = MathF.Max(bw, 1e-9f); bh = MathF.Max(bh, 1e-9f);
        int o = c * PS;
        prm[o + 6] = bc; prm[o + 7] = bs; prm[o + 8] = mx; prm[o + 9] = my; prm[o + 10] = bw; prm[o + 11] = bh;
        Atomic.Max(ref hbits[0], (int)Interop.FloatAsInt(bh));
    }

    // ---- packing -------------------------------------------------------------------------------------------------------------------------------------
    private static void BucketKeyImpl(Index1D c, FV prm, IV hbits, IV key)
    {
        float hmax = Interop.IntAsFloat((uint)hbits[0]), h = prm[c * PS + 11];
        int k = (int)(MathF.Log(hmax / h) * (16f * 1.4426950f));               // sixteen buckets per octave of height, tallest first
        key[c] = Math.Max(0, Math.Min(Buckets - 1, k));
    }

    private static void HistogramImpl(Index1D c, IV key, IV hist) { Atomic.Add(ref hist[key[c]], 1); }

    /// <summary>One thread per bucket walks the charts in index order, so the order inside a bucket is the same on every run.</summary>
    private static void PlaceImpl(Index1D b, IV key, IV bstart, IV order, int nC)
    {
        int run = bstart[b];
        for (int c = 0; c < nC; c++) if (key[c] == b) { order[run] = c; run++; }
    }

    private static void SumRectAreaImpl(Index1D _, FV prm, int nC, FV outv)
    {
        float s = 0f;
        for (int c = 0; c < nC; c++) s += prm[c * PS + 10] * prm[c * PS + 11];
        outv[0] = s;
    }

    private static bool Shelf(FV prm, IV order, int nC, float density, int pad, int res, int minTexels, bool commit, out float used)
    {
        float x = 0f, y = 0f, shelfH = 0f; used = 0f;
        for (int i = 0; i < nC; i++)
        {
            int c = order[i], o = c * PS;
            float w = prm[o + 10], h = prm[o + 11];
            float k = MathF.Max(1f, minTexels / MathF.Max(MathF.Max(w, h) * density, 1e-12f));
            float W = MathF.Ceiling(w * density * k) + 2 * pad, H = MathF.Ceiling(h * density * k) + 2 * pad;
            if (W > res || H > res) return false;
            if (x + W > res) { y += shelfH; x = 0f; shelfH = 0f; }
            if (y + H > res) return false;
            if (commit) { prm[o + 12] = k; prm[o + 13] = x; prm[o + 14] = y; used += w * density * k * h * density * k; }
            x += W; if (H > shelfH) shelfH = H;
        }
        return true;
    }

    private static void FitsImpl(Index1D cand, FV prm, IV order, FV dens, IV pads, IV fit, int nC, int res, int minTexels)
    {
        fit[cand] = Shelf(prm, order, nC, dens[cand], pads[cand], res, minTexels, false, out float _) ? 1 : 0;
    }

    private static void CommitImpl(Index1D _, FV prm, IV order, FV outv, int nC, float density, int pad, int res, int minTexels)
    {
        bool ok = Shelf(prm, order, nC, density, pad, res, minTexels, true, out float used);
        outv[0] = ok ? 1f : 0f; outv[1] = used;
    }

    // ---- seams ---------------------------------------------------------------------------------------------------------------------------------------
    /// <summary>The number of copies a vertex needs: one per distinct chart among its faces.</summary>
    private static void CopiesImpl(Index1D v, IV vstart, IV vfaces, IV chartOf, IV copies)
    {
        int n = 0;
        for (int i = vstart[v]; i < vstart[v + 1]; i++)
        {
            int c = chartOf[vfaces[i]]; bool first = true;
            for (int j = vstart[v]; j < i && first; j++) if (chartOf[vfaces[j]] == c) first = false;
            if (first) n++;
        }
        copies[v] = n;
    }

    /// <summary>Each corner's new vertex: the vertex's base plus the rank of its chart among the distinct charts in the vertex's list.</summary>
    private static void CornerIndexImpl(Index1D idx, IV faces, IV vstart, IV vfaces, IV chartOf, IV vbase, IV outFaces, IV outSrc)
    {
        int t = idx / 3, v = faces[idx], c = chartOf[t], rank = 0;
        for (int i = vstart[v]; i < vstart[v + 1]; i++)
        {
            int ci = chartOf[vfaces[i]]; bool first = true;
            for (int j = vstart[v]; j < i && first; j++) if (chartOf[vfaces[j]] == ci) first = false;
            if (!first) continue;
            if (ci == c) break;
            rank++;
        }
        int o = vbase[v] + rank;
        outFaces[idx] = o; outSrc[o] = v;
    }

    private static void CornerUvImpl(Index1D idx, FV pos, IV faces, IV outFaces, IV chartOf, FV prm, FV outPos, FV outUv, float density, int pad, float invRes)
    {
        int t = idx / 3, v = faces[idx], o = outFaces[idx], c = chartOf[t], b = c * PS;
        float px = pos[v * 3], py = pos[v * 3 + 1], pz = pos[v * 3 + 2];
        float x = px * prm[b] + py * prm[b + 1] + pz * prm[b + 2], y = px * prm[b + 3] + py * prm[b + 4] + pz * prm[b + 5];
        float cs = prm[b + 6], sn = prm[b + 7];
        float u = x * cs - y * sn - prm[b + 8], w = x * sn + y * cs - prm[b + 9];
        float s = density * prm[b + 12];
        outUv[o * 2] = (prm[b + 13] + pad + u * s) * invRes;
        outUv[o * 2 + 1] = (prm[b + 14] + pad + w * s) * invRes;
        outPos[o * 3] = px; outPos[o * 3 + 1] = py; outPos[o * 3 + 2] = pz;
    }

    private static void GatherNormalsImpl(Index1D i, IV src, FV vn, FV outN)
    {
        int s = src[i];
        outN[i * 3] = vn[s * 3]; outN[i * 3 + 1] = vn[s * 3 + 1]; outN[i * 3 + 2] = vn[s * 3 + 2];
    }

    // ---- the pipeline --------------------------------------------------------------------------------------------------------------------------------
    /// <summary>
    /// Aligns the winding to the vertex normals (when <paramref name="alignWinding"/>; <paramref name="faces"/> is changed in place) and unwraps the mesh. The inputs stay on the device and are not disposed.
    /// </summary>
    public async Task<Result> UnwrapAsync(FV positions, int vertexCount, IV faces, int faceCount, UvAtlas.Options? options = null, bool alignWinding = true)
    {
        options ??= new UvAtlas.Options();
        int nv = vertexCount, nf = faceCount, res = options.Resolution;
        var count = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(GpuMeshSimplifier.CountImpl);
        var fill = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV>(GpuMeshSimplifier.FillImpl);
        var sortList = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(SortListImpl);
        var vnormals = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, IV, FV>(VertexNormalsImpl);
        var alignFlip = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, FV, IV>(AlignFlipImpl);
        var faceInfo = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, FV, FV, IV>(FaceInfoImpl);
        var faceWeight = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV>(FaceWeightImpl);
        var adjacency = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV>(AdjacencyImpl);
        var priority = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, int>(PriorityImpl);
        var propInit = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV>(PropInitImpl);
        var propStep = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV>(PropStepImpl);
        var seed = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, FV, IV, IV, IV>(SeedImpl);
        var grow = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, FV, IV, IV, FV>(GrowImpl);
        var accum = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, FV, IV, IV, IV>(AccumImpl);
        var countUnlabelled = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(CountUnlabelledImpl);
        var fillFloat = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, float>(FillFloatImpl);
        var fillInt = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, int>(FillIntImpl);
        var badK = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, FV, IV, IV, float>(BadImpl);
        var dissolve = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, FV, IV, float>(DissolveImpl);

        // the vertex -> faces table
        using var vcount = _acc.Allocate1D<int>(Math.Max(1, nv)); using var vfill = _acc.Allocate1D<int>(Math.Max(1, nv));
        using var vstart = _acc.Allocate1D<int>(nv + 1L); using var vfaces = _acc.Allocate1D<int>(Math.Max(3L, (long)nf * 3));
        vcount.MemSetToZero(); vfill.MemSetToZero();
        count(nf, faces, vcount.View);
        await _sk.ExclusiveScanAsync(vcount.View, nv, vstart.View);
        fill(nf, faces, vstart.View, vfill.View, vfaces.View);
        sortList(nv, vstart.View, vfaces.View);

        // winding and normals
        using var vn = _acc.Allocate1D<float>(Math.Max(3L, (long)nv * 3));
        using var cnt = _acc.Allocate1D<int>(4);
        int flipped = 0;
        for (int pass = 0; pass < (alignWinding ? 3 : 0); pass++)
        {
            vnormals(nv, positions, faces, vstart.View, vfaces.View, vn.View);
            cnt.MemSetToZero();
            alignFlip(nf, positions, faces, vn.View, cnt.View);
            int got = (await cnt.View.CopyToHostAsync())[0];
            flipped += got;
            if (got == 0) break;
        }
        vnormals(nv, positions, faces, vstart.View, vfaces.View, vn.View);

        // face data
        using var fn = _acc.Allocate1D<float>(Math.Max(3L, (long)nf * 3)); using var area = _acc.Allocate1D<float>(Math.Max(1, nf));
        using var amax = _acc.Allocate1D<int>(1); using var wq = _acc.Allocate1D<int>(Math.Max(1, nf));
        amax.MemSetToZero();
        faceInfo(nf, positions, faces, fn.View, area.View, amax.View);
        faceWeight(nf, area.View, amax.View, wq.View);
        using var adj = _acc.Allocate1D<int>(Math.Max(3L, (long)nf * 3));
        adjacency(nf, faces, vstart.View, vfaces.View, adj.View);

        // charts
        var label = _acc.Allocate1D<int>(Math.Max(1, nf)); var newLabel = _acc.Allocate1D<int>(Math.Max(1, nf));
        using var sumN = _acc.Allocate1D<int>(Math.Max(3L, (long)nf * 3));
        using var frozen = _acc.Allocate1D<int>(Math.Max(1, nf)); using var bad = _acc.Allocate1D<int>(Math.Max(1, nf));
        using var cone = _acc.Allocate1D<float>(Math.Max(1, nf));
        using var prio = _acc.Allocate1D<int>(Math.Max(1, nf));
        var valA = _acc.Allocate1D<int>(Math.Max(1, nf)); var valB = _acc.Allocate1D<int>(Math.Max(1, nf));
        try
        {
            int idBits = 1; while ((1L << idBits) <= nf) idBits++;
            priority(nf, prio.View, idBits);
            fillInt(nf, label.View, -1); fillInt(nf, frozen.View, 0); sumN.MemSetToZero();
            double coneDeg = options.MaxNormalDeviationDegrees;
            fillFloat(nf, cone.View, (float)Math.Cos(coneDeg * Math.PI / 180));
            for (int depth = 0; ; depth++)
            {
                int left = nf;
                int[] rings = { 16, 8, 4, 2, 1 };
                for (int iter = 0; left > 0 && iter < 256; iter++)
                {
                    int R = iter < rings.Length ? rings[iter] : (iter < 64 ? 1 : 0);
                    propInit(nf, label.View, prio.View, valA.View);
                    for (int r = 0; r < R; r++) { propStep(nf, adj.View, label.View, valA.View, valB.View); (valA, valB) = (valB, valA); }
                    cnt.MemSetToZero();
                    seed(nf, label.View, prio.View, valA.View, fn.View, wq.View, sumN.View, cnt.View);
                    // grow until a batch of rounds labels nothing
                    int labelled = 0;
                    for (int batch = 0; batch < 100000; batch++)
                    {
                        cnt.MemSetToZero();
                        for (int round = 0; round < 4; round++)
                        {
                            newLabel.View.CopyFrom(label.View);
                            grow(nf, adj.View, label.View, newLabel.View, fn.View, sumN.View, frozen.View, cone.View);
                            accum(nf, label.View, newLabel.View, fn.View, wq.View, sumN.View, cnt.View);
                            (label, newLabel) = (newLabel, label);
                        }
                        labelled = (await cnt.View.CopyToHostAsync())[0];
                        if (labelled == 0) break;
                    }
                    cnt.MemSetToZero(); countUnlabelled(nf, label.View, cnt.View);
                    left = (await cnt.View.CopyToHostAsync())[0];
                    Log.Add($"depth {depth} iter {iter} rings {R}: unlabelled {left}");
                }
                if (left > 0) throw new InvalidOperationException($"UV atlas: {left} faces did not join any chart.");
                if (depth >= options.MaxSplitDepth) break;
                // charts whose faces lean too far from the chart's own direction fold in the projection: grow them again, tighter
                bad.MemSetToZero();
                badK(nf, label.View, fn.View, sumN.View, bad.View, (float)Math.Cos(65 * Math.PI / 180));
                cnt.MemSetToZero();
                coneDeg = Math.Max(coneDeg * 0.6, 3.0);
                dissolve(nf, label.View, bad.View, frozen.View, cone.View, cnt.View, (float)Math.Cos(coneDeg * Math.PI / 180));
                int dissolved = (await cnt.View.CopyToHostAsync())[0];
                Log.Add($"depth {depth}: {dissolved} faces dissolved, cone now {coneDeg:F1}");
                if (dissolved == 0) break;
            }

            // chart ids, basis, bounding rectangles
            var rootFlag = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(RootFlagImpl);
            var chartOfFace = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV>(ChartOfFaceImpl);
            var basis = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, FV, FV>(BasisImpl);
            var boundsInit = _acc.LoadAutoGroupedStreamKernel<Index1D, IV>(BoundsInitImpl);
            var bounds = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, FV, IV, int, int>(BoundsImpl);
            var pick = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, FV, IV, int>(PickImpl);
            using var flag = _acc.Allocate1D<int>(Math.Max(1, nf)); using var cid = _acc.Allocate1D<int>(nf + 1L);
            rootFlag(nf, label.View, flag.View);
            int nC = await _sk.ExclusiveScanAsync(flag.View, nf, cid.View);
            using var chartOf = _acc.Allocate1D<int>(Math.Max(1, nf));
            chartOfFace(nf, label.View, cid.View, chartOf.View);
            using var prm = _acc.Allocate1D<float>(Math.Max(PS, (long)nC * PS)); prm.MemSetToZero();
            basis(nf, label.View, cid.View, sumN.View, fn.View, prm.View);
            using var hbits = _acc.Allocate1D<int>(1); hbits.MemSetToZero();
            for (int c0 = 0; c0 < nC; c0 += MaxChartChunk)
            {
                int chunk = Math.Min(MaxChartChunk, nC - c0);
                using var b = _acc.Allocate1D<int>((long)chunk * Angles * 4);
                boundsInit(chunk * Angles * 4, b.View);
                bounds(nf * 3, positions, faces, chartOf.View, prm.View, b.View, c0, chunk);
                pick(chunk, b.View, prm.View, hbits.View, c0);
                await _acc.SynchronizeAsync();                       // b is released here: nothing queued still reads it
            }

            // order the charts tallest first
            var bucketKey = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV>(BucketKeyImpl);
            var histogram = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(HistogramImpl);
            var place = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, int>(PlaceImpl);
            using var key = _acc.Allocate1D<int>(Math.Max(1, nC)); using var hist = _acc.Allocate1D<int>(Buckets); hist.MemSetToZero();
            using var bstart = _acc.Allocate1D<int>(Buckets + 1L); using var order = _acc.Allocate1D<int>(Math.Max(1, nC));
            bucketKey(nC, prm.View, hbits.View, key.View);
            histogram(nC, key.View, hist.View);
            await _sk.ExclusiveScanAsync(hist.View, Buckets, bstart.View);
            place(Buckets, key.View, bstart.View, order.View, nC);

            // pack: the largest density that fits, for each padding; the widest padding that costs at most 20% of the best
            var sumRect = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, int, FV>(SumRectAreaImpl);
            var fits = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, FV, IV, IV, int, int, int>(FitsImpl);
            var commit = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, FV, int, float, int, int, int>(CommitImpl);
            using var rectSum = _acc.Allocate1D<float>(1);
            sumRect(1, prm.View, nC, rectSum.View);
            double rectArea = (await rectSum.View.CopyToHostAsync())[0];
            double dmax = Math.Sqrt((double)res * res / Math.Max(rectArea, 1e-30)) * 1.01;
            using var dens = _acc.Allocate1D<float>(4 * Candidates); using var padBuf = _acc.Allocate1D<int>(4 * Candidates); using var fit = _acc.Allocate1D<int>(4 * Candidates);
            double density = 0, utilization = 0; int pad = 0;
            for (int minTexels = options.MinChartTexels; density <= 0 && minTexels >= 1; minTexels--)
            {
                int[] pads = options.PaddingTexels > 0 ? new[] { options.PaddingTexels } : minTexels == 1 ? new[] { 0 } : new[] { 4, 3, 2, 1 };
                var lo = new double[pads.Length]; var hi = new double[pads.Length]; var best = new double[pads.Length];
                for (int p = 0; p < pads.Length; p++) { lo[p] = dmax * 1e-3; hi[p] = dmax; }
                for (int round = 0; round < 3; round++)
                {
                    var dv = new float[4 * Candidates]; var pv = new int[4 * Candidates];
                    for (int p = 0; p < pads.Length; p++)
                        for (int i = 0; i < Candidates; i++)
                        {
                            dv[p * Candidates + i] = (float)(lo[p] * Math.Pow(hi[p] / lo[p], i / (double)(Candidates - 1)));
                            pv[p * Candidates + i] = pads[p];
                        }
                    dens.View.CopyFromCPU(dv); padBuf.View.CopyFromCPU(pv);
                    fits(pads.Length * Candidates, prm.View, order.View, dens.View, padBuf.View, fit.View, nC, res, minTexels);
                    var ok = await fit.View.CopyToHostAsync();
                    for (int p = 0; p < pads.Length; p++)
                    {
                        int top = -1;
                        for (int i = 0; i < Candidates; i++) if (ok[p * Candidates + i] != 0) top = i;
                        if (top < 0) { best[p] = 0; hi[p] = lo[p]; continue; }
                        best[p] = dv[p * Candidates + top];
                        lo[p] = dv[p * Candidates + top];
                        hi[p] = top + 1 < Candidates ? dv[p * Candidates + top + 1] : lo[p] * 1.0001;
                    }
                }
                double bestAll = best.Max();
                if (bestAll <= 0) continue;
                int pick2 = Array.FindIndex(best, d => d > 0 && d >= 0.8 * bestAll);
                pad = pads[pick2]; density = best[pick2];
                // commit the chosen layout; a rounding difference between the search and the placement (the same arithmetic, but not the same dispatch) backs the density off a hair
                using var outv = _acc.Allocate1D<float>(2);
                bool placed = false; float used = 0;
                for (int tries = 0; tries < 16 && !placed; tries++)
                {
                    commit(1, prm.View, order.View, outv.View, nC, (float)density, pad, res, minTexels);
                    var r = await outv.View.CopyToHostAsync();
                    placed = r[0] != 0f; used = r[1];
                    if (!placed) density *= 0.999;
                }
                if (!placed) density = 0;
                else utilization = used / ((double)res * res);
            }
            if (density <= 0)
                throw new InvalidOperationException($"UV atlas: {nC} charts do not fit a {res} texture even at 1 texel each; simplify the mesh further or use a larger texture.");

            // seams: a vertex once per distinct chart among its faces
            var copiesK = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV>(CopiesImpl);
            var cornerIndex = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV, IV, IV, IV>(CornerIndexImpl);
            var cornerUv = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, IV, FV, FV, FV, float, int, float>(CornerUvImpl);
            var gatherNormals = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, FV, FV>(GatherNormalsImpl);
            using var copies = _acc.Allocate1D<int>(Math.Max(1, nv)); using var vbase = _acc.Allocate1D<int>(nv + 1L);
            copiesK(nv, vstart.View, vfaces.View, chartOf.View, copies.View);
            int nOut = await _sk.ExclusiveScanAsync(copies.View, nv, vbase.View);
            var result = new Result { VertexCount = nOut, FaceCount = nf, ChartCount = nC, TexelsPerUnit = density, Flipped = flipped, Utilization = utilization };
            result.Faces = _acc.Allocate1D<int>(Math.Max(3L, (long)nf * 3));
            result.SourceVertex = _acc.Allocate1D<int>(Math.Max(1, nOut));
            result.Positions = _acc.Allocate1D<float>(Math.Max(3L, (long)nOut * 3));
            result.Uvs = _acc.Allocate1D<float>(Math.Max(2L, (long)nOut * 2));
            result.Normals = _acc.Allocate1D<float>(Math.Max(3L, (long)nOut * 3));
            cornerIndex(nf * 3, faces, vstart.View, vfaces.View, chartOf.View, vbase.View, result.Faces.View, result.SourceVertex.View);
            cornerUv(nf * 3, positions, faces, result.Faces.View, chartOf.View, prm.View, result.Positions.View, result.Uvs.View, (float)density, pad, 1f / res);
            gatherNormals(nOut, result.SourceVertex.View, vn.View, result.Normals.View);
            await _acc.SynchronizeAsync();
            return result;
        }
        finally { label.Dispose(); newLabel.Dispose(); valA.Dispose(); valB.Dispose(); }
    }

}
