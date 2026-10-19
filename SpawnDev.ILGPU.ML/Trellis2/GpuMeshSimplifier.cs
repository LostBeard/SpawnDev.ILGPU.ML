using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>A triangle mesh in device buffers (xyz positions, triangle indices).</summary>
public sealed class DeviceTriMesh : IDisposable
{
    public MemoryBuffer1D<float, Stride1D.Dense> Positions = null!;
    public MemoryBuffer1D<int, Stride1D.Dense> Faces = null!;
    public int VertexCount, FaceCount;
    public void Dispose() { Positions?.Dispose(); Faces?.Dispose(); }
}

/// <summary>
/// Quadric-error edge-collapse simplification on the device (Garland and Heckbert 1997, as <see cref="MeshSimplifier"/>: boundary penalty quadrics, the link condition, a face-flip guard) in parallel rounds, like CuMesh's GPU simplify.
/// Each round every vertex finds its cheapest valid collapse; an edge is collapsed when it is the cheapest within its whole one-ring neighbourhood (both ends hold it as their best, and no cheaper candidate touches either end's ring), which makes the
/// collapses of a round touch disjoint sets of faces, so they run at once with no conflicts. Faces are renumbered and compacted each round; the loop ends at the target (the last round collapses only the cheapest edges that reach it).
/// Every quadric lives in the frame of its own vertex's original position, so float32 keeps the tiny error values (a 300k-triangle mesh has edges of a few thousandths of the model) without cancelling against the model's size.
/// Nothing but a handful of counts crosses to the host.
/// </summary>
public sealed class GpuMeshSimplifier
{
    private readonly Accelerator _acc;
    private readonly SparseKernels _sk;
    public GpuMeshSimplifier(Accelerator accelerator, SparseKernels sparse) { _acc = accelerator; _sk = sparse; }

    /// <summary>Rounds the last <see cref="SimplifyAsync"/> took and the faces after each (diagnostic).</summary>
    public List<(int Faces, int Collapses)> LastRounds { get; private set; } = new();

    private const float Inf = 3.0e38f;
    private const int SubPasses = 5;

    // ---- quadric math (all in the frame of vertex a's original position) -----------------------------------------------------------------------------
    private static float Err(float aa, float ab, float ac, float bb, float bc, float cc, float bx, float by, float bz, float c, float x, float y, float z)
        => aa * x * x + 2f * ab * x * y + 2f * ac * x * z + bb * y * y + 2f * bc * y * z + cc * z * z + 2f * (bx * x + by * y + bz * z) + c;

    /// <summary>
    /// Cost of collapsing the edge (a, b) with a &lt; b and the world position to collapse to: the quadric minimiser when the system is well conditioned and near the edge, else the best of the ends and the midpoint.
    /// The same arithmetic runs for both ends of the edge, so both see the same cost.
    /// </summary>
    private static float Evaluate(FV pos, FV org, FV Q, int a, int b, out float px, out float py, out float pz)
    {
        float tx = org[a * 3] - org[b * 3], ty = org[a * 3 + 1] - org[b * 3 + 1], tz = org[a * 3 + 2] - org[b * 3 + 2];
        int qb = b * 10, qa = a * 10;
        float baa = Q[qb], bab = Q[qb + 1], bac = Q[qb + 2], bbb = Q[qb + 3], bbc = Q[qb + 4], bcc = Q[qb + 5];
        float atx = baa * tx + bab * ty + bac * tz, aty = bab * tx + bbb * ty + bbc * tz, atz = bac * tx + bbc * ty + bcc * tz;
        float nbx = Q[qb + 6] + atx, nby = Q[qb + 7] + aty, nbz = Q[qb + 8] + atz;
        float nc = Q[qb + 9] + 2f * (Q[qb + 6] * tx + Q[qb + 7] * ty + Q[qb + 8] * tz) + (tx * atx + ty * aty + tz * atz);
        float aa = Q[qa] + baa, ab = Q[qa + 1] + bab, ac = Q[qa + 2] + bac, bb = Q[qa + 3] + bbb, bc = Q[qa + 4] + bbc, cc = Q[qa + 5] + bcc;
        float bx = Q[qa + 6] + nbx, by = Q[qa + 7] + nby, bz = Q[qa + 8] + nbz, c = Q[qa + 9] + nc;

        float ox = org[a * 3], oy = org[a * 3 + 1], oz = org[a * 3 + 2];
        float ax = pos[a * 3] - ox, ay = pos[a * 3 + 1] - oy, az = pos[a * 3 + 2] - oz;
        float ex = pos[b * 3] - ox, ey = pos[b * 3 + 1] - oy, ez = pos[b * 3 + 2] - oz;
        float best = Err(aa, ab, ac, bb, bc, cc, bx, by, bz, c, ax, ay, az); float x = ax, y = ay, z = az;
        float eb = Err(aa, ab, ac, bb, bc, cc, bx, by, bz, c, ex, ey, ez); if (eb < best) { best = eb; x = ex; y = ey; z = ez; }
        float mx = (ax + ex) * 0.5f, my = (ay + ey) * 0.5f, mz = (az + ez) * 0.5f;
        float em = Err(aa, ab, ac, bb, bc, cc, bx, by, bz, c, mx, my, mz); if (em < best) { best = em; x = mx; y = my; z = mz; }
        float det = aa * (bb * cc - bc * bc) - ab * (ab * cc - bc * ac) + ac * (ab * bc - bb * ac);
        float scale = MathF.Abs(aa) + MathF.Abs(bb) + MathF.Abs(cc);
        if (scale > 0f && MathF.Abs(det) > 1e-6f * scale * scale * scale)
        {
            float r0 = -bx, r1 = -by, r2 = -bz;
            float sx = (r0 * (bb * cc - bc * bc) - ab * (r1 * cc - bc * r2) + ac * (r1 * bc - bb * r2)) / det;
            float sy = (aa * (r1 * cc - bc * r2) - r0 * (ab * cc - bc * ac) + ac * (ab * r2 - r1 * ac)) / det;
            float sz = (aa * (bb * r2 - r1 * bc) - ab * (ab * r2 - r1 * ac) + r0 * (ab * bc - bb * ac)) / det;
            float dx = ex - ax, dy = ey - ay, dz = ez - az, el2 = dx * dx + dy * dy + dz * dz;
            float qx = sx - mx, qy = sy - my, qz = sz - mz;
            if (qx * qx + qy * qy + qz * qz <= el2 * 4f)
            {
                float eo = Err(aa, ab, ac, bb, bc, cc, bx, by, bz, c, sx, sy, sz);
                if (eo < best) { best = eo; x = sx; y = sy; z = sz; }
            }
        }
        px = ox + x; py = oy + y; pz = oz + z;
        return MathF.Max(best, 0f);
    }

    /// <summary>The merged quadric of (a, b) written into a's slot, in a's frame.</summary>
    private static void MergeInto(FV org, FV Q, int a, int b)
    {
        float tx = org[a * 3] - org[b * 3], ty = org[a * 3 + 1] - org[b * 3 + 1], tz = org[a * 3 + 2] - org[b * 3 + 2];
        int qb = b * 10, qa = a * 10;
        float baa = Q[qb], bab = Q[qb + 1], bac = Q[qb + 2], bbb = Q[qb + 3], bbc = Q[qb + 4], bcc = Q[qb + 5];
        float atx = baa * tx + bab * ty + bac * tz, aty = bab * tx + bbb * ty + bbc * tz, atz = bac * tx + bbc * ty + bcc * tz;
        Q[qa] += baa; Q[qa + 1] += bab; Q[qa + 2] += bac; Q[qa + 3] += bbb; Q[qa + 4] += bbc; Q[qa + 5] += bcc;
        Q[qa + 6] += Q[qb + 6] + atx; Q[qa + 7] += Q[qb + 7] + aty; Q[qa + 8] += Q[qb + 8] + atz;
        Q[qa + 9] += Q[qb + 9] + 2f * (Q[qb + 6] * tx + Q[qb + 7] * ty + Q[qb + 8] * tz) + (tx * atx + ty * aty + tz * atz);
    }

    private static bool FaceHas(IV faces, int f, int v) => faces[f * 3] == v || faces[f * 3 + 1] == v || faces[f * 3 + 2] == v;

    private static bool IsNeighbour(IV faces, IV vstart, IV vfaces, int v, int x)
    {
        for (int i = vstart[v]; i < vstart[v + 1]; i++) if (FaceHas(faces, vfaces[i], x)) return true;
        return false;
    }

    /// <summary>
    /// The link condition (the ends share exactly the vertices opposite the faces they share) and the face-flip / sliver guard for moving both ends to (px, py, pz).
    /// </summary>
    private static bool Valid(FV pos, IV faces, IV vstart, IV vfaces, int a, int b, float px, float py, float pz)
    {
        int shared = 0, common = 0;
        for (int i = vstart[a]; i < vstart[a + 1]; i++)
        {
            int f = vfaces[i];
            if (FaceHas(faces, f, b)) shared++;
            for (int k = 0; k < 3; k++)
            {
                int x = faces[f * 3 + k];
                if (x == a || x == b) continue;
                bool seen = false;
                for (int j = vstart[a]; j < i && !seen; j++) seen = FaceHas(faces, vfaces[j], x);
                if (!seen && IsNeighbour(faces, vstart, vfaces, b, x)) common++;
            }
        }
        if (shared == 0 || common != shared) return false;
        for (int side = 0; side < 2; side++)
        {
            int v = side == 0 ? a : b, other = side == 0 ? b : a;
            for (int i = vstart[v]; i < vstart[v + 1]; i++)
            {
                int f = vfaces[i];
                if (FaceHas(faces, f, other)) continue;
                int i0 = faces[f * 3], i1 = faces[f * 3 + 1], i2 = faces[f * 3 + 2];
                float x0 = pos[i0 * 3], y0 = pos[i0 * 3 + 1], z0 = pos[i0 * 3 + 2];
                float x1 = pos[i1 * 3], y1 = pos[i1 * 3 + 1], z1 = pos[i1 * 3 + 2];
                float x2 = pos[i2 * 3], y2 = pos[i2 * 3 + 1], z2 = pos[i2 * 3 + 2];
                float ox = (y1 - y0) * (z2 - z0) - (z1 - z0) * (y2 - y0), oy = (z1 - z0) * (x2 - x0) - (x1 - x0) * (z2 - z0), oz = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
                float oa = MathF.Sqrt(ox * ox + oy * oy + oz * oz);
                if (i0 == v) { x0 = px; y0 = py; z0 = pz; } else if (i1 == v) { x1 = px; y1 = py; z1 = pz; } else { x2 = px; y2 = py; z2 = pz; }
                float nx = (y1 - y0) * (z2 - z0) - (z1 - z0) * (y2 - y0), ny = (z1 - z0) * (x2 - x0) - (x1 - x0) * (z2 - z0), nz = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
                float na = MathF.Sqrt(nx * nx + ny * ny + nz * nz);
                if (na <= 1e-14f) return false;
                if (oa > 0f && (nx * ox + ny * oy + nz * oz) / (na * oa) < 0.2f) return false;
            }
        }
        return true;
    }

    // ---- the adjacency: vertex -> incident faces, rebuilt every round with atomics and a scan -------------------------------------------------------------
    private static void CountImpl(Index1D t, IV faces, IV count)
    {
        Atomic.Add(ref count[faces[t * 3]], 1); Atomic.Add(ref count[faces[t * 3 + 1]], 1); Atomic.Add(ref count[faces[t * 3 + 2]], 1);
    }

    private static void FillImpl(Index1D t, IV faces, IV vstart, IV fill, IV vfaces)
    {
        for (int k = 0; k < 3; k++)
        {
            int v = faces[t * 3 + k];
            vfaces[vstart[v] + Atomic.Add(ref fill[v], 1)] = t;
        }
    }

    // ---- quadrics (round 0): area-weighted face planes, and a heavy plane perpendicular to the face on every boundary edge ------------------------------------
    private static void AddPlane(FV Q, int v, float a, float b, float c, float d, float w)
    {
        int o = v * 10;
        Q[o] += w * a * a; Q[o + 1] += w * a * b; Q[o + 2] += w * a * c; Q[o + 3] += w * b * b; Q[o + 4] += w * b * c; Q[o + 5] += w * c * c;
        Q[o + 6] += w * a * d; Q[o + 7] += w * b * d; Q[o + 8] += w * c * d; Q[o + 9] += w * d * d;
    }

    private static void QuadricImpl(Index1D v, FV pos, FV org, FV Q, IV faces, IV vstart, IV vfaces)
    {
        float ox = org[v * 3], oy = org[v * 3 + 1], oz = org[v * 3 + 2];
        for (int i = vstart[v]; i < vstart[v + 1]; i++)
        {
            int f = vfaces[i];
            int ia = faces[f * 3], ib = faces[f * 3 + 1], ic = faces[f * 3 + 2];
            float ax = pos[ia * 3] - ox, ay = pos[ia * 3 + 1] - oy, az = pos[ia * 3 + 2] - oz;
            float bx = pos[ib * 3] - ox, by = pos[ib * 3 + 1] - oy, bz = pos[ib * 3 + 2] - oz;
            float cx = pos[ic * 3] - ox, cy = pos[ic * 3 + 1] - oy, cz = pos[ic * 3 + 2] - oz;
            float nx = (by - ay) * (cz - az) - (bz - az) * (cy - ay), ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az), nz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
            float len = MathF.Sqrt(nx * nx + ny * ny + nz * nz);
            if (len <= 0f) continue;
            float ux = nx / len, uy = ny / len, uz = nz / len;
            AddPlane(Q, v, ux, uy, uz, -(ux * ax + uy * ay + uz * az), 0.5f * len);
            // boundary edges of this face at v: an edge (v, x) no other face of v contains
            int x1, x2;
            if (ia == v) { x1 = ib; x2 = ic; } else if (ib == v) { x1 = ic; x2 = ia; } else { x1 = ia; x2 = ib; }
            for (int e = 0; e < 2; e++)
            {
                int x = e == 0 ? x1 : x2;
                int uses = 0;
                for (int j = vstart[v]; j < vstart[v + 1]; j++) if (FaceHas(faces, vfaces[j], x)) uses++;
                if (uses != 1) continue;
                float ex = pos[x * 3] - pos[v * 3], ey = pos[x * 3 + 1] - pos[v * 3 + 1], ez = pos[x * 3 + 2] - pos[v * 3 + 2];
                float px = ey * uz - ez * uy, py = ez * ux - ex * uz, pz = ex * uy - ey * ux;
                float pl = MathF.Sqrt(px * px + py * py + pz * pz);
                if (pl <= 0f) continue;
                px /= pl; py /= pl; pz /= pl;
                float vx = pos[v * 3] - ox, vy = pos[v * 3 + 1] - oy, vz = pos[v * 3 + 2] - oz;       // 0: org[v] is v's own position at the start
                AddPlane(Q, v, px, py, pz, -(px * vx + py * vy + pz * vz), 100f * (ex * ex + ey * ey + ez * ez));
            }
        }
    }

    // ---- one round -----------------------------------------------------------------------------------------------------------------------------------------
    // Equal costs (flat regions cost exactly nothing) are ordered by a hash of the edge, not by vertex number: a numbering that runs along the surface would make "cheapest in its neighbourhood" pick one edge per
    // long run and the loop would need a round per run; a random priority makes the selected set a random maximal independent set, which grows to most of the mesh in a few rounds.
    private static uint EdgeHash(int a, int b)
    {
        uint h = (uint)a * 2654435761u ^ ((uint)b + 0x9E3779B9u) * 2246822519u;
        h ^= h >> 15; h *= 2654435761u; h ^= h >> 13;
        return h;
    }

    private static bool KeyLess(float c1, int a1, int b1, float c2, int a2, int b2)
    {
        if (c1 != c2) return c1 < c2;
        uint h1 = EdgeHash(a1, b1), h2 = EdgeHash(a2, b2);
        if (h1 != h2) return h1 < h2;
        return a1 < a2 || (a1 == a2 && b1 < b2);
    }

    private static void BestEdgeImpl(Index1D v, FV pos, FV org, FV Q, IV faces, IV vstart, IV vfaces, FV bestCost, IV bestNbr)
    {
        float bc = Inf; int bu = -1, bestA = -1, bestB = -1;
        for (int i = vstart[v]; i < vstart[v + 1]; i++)
        {
            int f = vfaces[i];
            for (int k = 0; k < 3; k++)
            {
                int x = faces[f * 3 + k];
                if (x == v) continue;
                bool seen = false;
                for (int j = vstart[v]; j < i && !seen; j++) seen = FaceHas(faces, vfaces[j], x);
                if (seen) continue;
                int a = v < x ? v : x, b = v < x ? x : v;
                float cost = Evaluate(pos, org, Q, a, b, out float px, out float py, out float pz);
                if (bu >= 0 && !KeyLess(cost, a, b, bc, bestA, bestB)) continue;       // not better than the best so far: skip the (costly) validity test
                if (!Valid(pos, faces, vstart, vfaces, a, b, px, py, pz)) continue;
                bc = cost; bu = x; bestA = a; bestB = b;
            }
        }
        bestCost[v] = bc; bestNbr[v] = bu;
    }

    private static void RingMinImpl(Index1D v, IV faces, IV vstart, IV vfaces, FV bestCost, IV bestNbr, IV blocked, FV nmCost, IV nmA, IV nmB)
    {
        float c = Inf; int ka = -1, kb = -1;
        if (blocked[v] == 0)
        {
            // the vertex itself ...
            int w0 = bestNbr[v];
            if (w0 >= 0 && blocked[w0] == 0) { c = bestCost[v]; ka = v < w0 ? v : w0; kb = v < w0 ? w0 : v; }
            // ... and every free neighbour: the best key among their best edges (an edge into a blocked vertex does not count)
            for (int i = vstart[v]; i < vstart[v + 1]; i++)
                for (int k = 0; k < 3; k++)
                {
                    int u = faces[vfaces[i] * 3 + k];
                    if (u == v || blocked[u] != 0 || bestNbr[u] < 0 || blocked[bestNbr[u]] != 0) continue;
                    int w = bestNbr[u], a = u < w ? u : w, b = u < w ? w : u;
                    if (ka < 0 || KeyLess(bestCost[u], a, b, c, ka, kb)) { c = bestCost[u]; ka = a; kb = b; }
                }
        }
        nmCost[v] = c; nmA[v] = ka; nmB[v] = kb;
    }

    // adds to the round's selection (never clears it): free vertices whose best edge is mutual and the cheapest in both ends' free rings
    private static void SelectImpl(Index1D v, FV bestCost, IV bestNbr, FV nmCost, IV nmA, IV nmB, IV blocked, IV selected, IV counter)
    {
        if (blocked[v] != 0) return;
        int u = bestNbr[v];
        if (u < 0 || u < v || blocked[u] != 0 || bestNbr[u] != v) return;      // handled once, from the lower end
        float c = bestCost[v];
        if (nmCost[v] != c || nmA[v] != v || nmB[v] != u || nmCost[u] != c || nmA[u] != v || nmB[u] != u) return;
        selected[v] = u;
        Atomic.Add(ref counter[0], 1);
    }

    // everything touching a selected edge or its neighbours is off limits for the rest of the round
    private static void BlockImpl(Index1D v, IV selected, IV faces, IV vstart, IV vfaces, IV blocked)
    {
        int u = selected[v];
        if (u < 0) return;
        blocked[v] = 1; blocked[u] = 1;
        for (int side = 0; side < 2; side++)
        {
            int w = side == 0 ? v : u;
            for (int i = vstart[w]; i < vstart[w + 1]; i++)
                for (int k = 0; k < 3; k++) blocked[faces[vfaces[i] * 3 + k]] = 1;
        }
    }

    private static void ThresholdImpl(Index1D v, FV bestCost, IV selected, IV counter, float tau)
    {
        if (selected[v] < 0) return;
        if (bestCost[v] > tau) selected[v] = -1; else Atomic.Add(ref counter[0], 1);
    }

    private static void CountBelowImpl(Index1D v, FV bestCost, IV selected, IV counter, float tau)
    {
        if (selected[v] >= 0 && bestCost[v] <= tau) Atomic.Add(ref counter[0], 1);
    }

    private static void MaxCostImpl(Index1D v, FV bestCost, IV selected, IV maxBits)
    {
        if (selected[v] >= 0) Atomic.Max(ref maxBits[0], (int)Interop.FloatAsInt(bestCost[v]));     // costs are >= 0: float bit patterns order like the values
    }

    private static void IotaImpl(Index1D i, IV v) { v[i] = i; }
    private static void FillMinus1Impl(Index1D i, IV v) { v[i] = -1; }

    private static void CollapseImpl(Index1D v, FV pos, FV org, FV Q, IV selected, IV rep)
    {
        int u = selected[v];
        if (u < 0) return;
        Evaluate(pos, org, Q, v, u, out float px, out float py, out float pz);
        MergeInto(org, Q, v, u);
        pos[v * 3] = px; pos[v * 3 + 1] = py; pos[v * 3 + 2] = pz;
        rep[u] = v;
    }

    private static void RemapImpl(Index1D t, IV faces, IV rep, IV keep)
    {
        int a = rep[faces[t * 3]], b = rep[faces[t * 3 + 1]], c = rep[faces[t * 3 + 2]];
        faces[t * 3] = a; faces[t * 3 + 1] = b; faces[t * 3 + 2] = c;
        keep[t] = a != b && b != c && a != c ? 1 : 0;
    }

    private static void CompactImpl(Index1D t, IV keep, IV offsets, IV faces, IV outFaces)
    {
        if (keep[t] == 0) return;
        int o = offsets[t] * 3;
        outFaces[o] = faces[t * 3]; outFaces[o + 1] = faces[t * 3 + 1]; outFaces[o + 2] = faces[t * 3 + 2];
    }

    private static void MarkUsedImpl(Index1D t, IV faces, IV used) { used[faces[t * 3]] = 1; used[faces[t * 3 + 1]] = 1; used[faces[t * 3 + 2]] = 1; }

    private static void GatherPositionsImpl(Index1D v, IV used, IV offsets, FV pos, FV outPos)
    {
        if (used[v] == 0) return;
        int o = offsets[v] * 3;
        outPos[o] = pos[v * 3]; outPos[o + 1] = pos[v * 3 + 1]; outPos[o + 2] = pos[v * 3 + 2];
    }

    private static void RenumberImpl(Index1D i, IV offsets, IV faces) { faces[i] = offsets[faces[i]]; }

    /// <summary>
    /// Reduce the mesh to about <paramref name="targetFaces"/>. The inputs stay untouched; the result is a new mesh in device buffers (compacted positions and faces).
    /// </summary>
    public async Task<DeviceTriMesh> SimplifyAsync(FV positions, int vertexCount, IV faces, int faceCount, int targetFaces, int maxRounds = 400)
    {
        LastRounds = new List<(int, int)>();
        var count = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(CountImpl);
        var fill = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV>(FillImpl);
        var quadric = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, IV, IV, IV>(QuadricImpl);
        var best = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, IV, IV, IV, FV, IV>(BestEdgeImpl);
        var ring = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, FV, IV, IV, FV, IV, IV>(RingMinImpl);
        var select = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, FV, IV, IV, IV, IV, IV>(SelectImpl);
        var block = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV, IV>(BlockImpl);
        var threshold = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, float>(ThresholdImpl);
        var fillMinus1 = _acc.LoadAutoGroupedStreamKernel<Index1D, IV>(FillMinus1Impl);
        var countBelow = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, float>(CountBelowImpl);
        var maxCost = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV>(MaxCostImpl);
        var iota = _acc.LoadAutoGroupedStreamKernel<Index1D, IV>(IotaImpl);
        var collapse = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, IV, IV>(CollapseImpl);
        var remap = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV>(RemapImpl);
        var compact = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV>(CompactImpl);

        int nv = vertexCount, nf = faceCount;
        using var pos = _acc.Allocate1D<float>(Math.Max(3L, (long)nv * 3));
        using var org = _acc.Allocate1D<float>(Math.Max(3L, (long)nv * 3));
        pos.View.SubView(0, (long)nv * 3).CopyFrom(positions.SubView(0, (long)nv * 3));
        org.View.SubView(0, (long)nv * 3).CopyFrom(positions.SubView(0, (long)nv * 3));
        using var Q = _acc.Allocate1D<float>(Math.Max(10L, (long)nv * 10)); Q.MemSetToZero();
        var facesA = _acc.Allocate1D<int>(Math.Max(3L, (long)nf * 3)); facesA.View.SubView(0, (long)nf * 3).CopyFrom(faces.SubView(0, (long)nf * 3));
        var facesB = _acc.Allocate1D<int>(Math.Max(3L, (long)nf * 3));
        using var vcount = _acc.Allocate1D<int>(nv);
        using var vfill = _acc.Allocate1D<int>(nv);
        using var vstart = _acc.Allocate1D<int>(nv + 1L);
        using var vfaces = _acc.Allocate1D<int>(Math.Max(3L, (long)nf * 3));
        using var bestCost = _acc.Allocate1D<float>(nv); using var bestNbr = _acc.Allocate1D<int>(nv);
        using var nmCost = _acc.Allocate1D<float>(nv); using var nmA = _acc.Allocate1D<int>(nv); using var nmB = _acc.Allocate1D<int>(nv);
        using var selected = _acc.Allocate1D<int>(nv); using var rep = _acc.Allocate1D<int>(nv); using var blocked = _acc.Allocate1D<int>(nv);
        using var counter = _acc.Allocate1D<int>(1);
        using var keep = _acc.Allocate1D<int>(Math.Max(1L, nf)); using var koff = _acc.Allocate1D<int>(nf + 1L);
        var cur = facesA; var other = facesB;
        try
        {
            for (int round = 0; round < maxRounds && nf > targetFaces; round++)
            {
                vcount.MemSetToZero(); vfill.MemSetToZero();
                count(nf, cur.View, vcount.View);
                await _sk.ExclusiveScanAsync(vcount.View, nv, vstart.View);
                fill(nf, cur.View, vstart.View, vfill.View, vfaces.View);
                if (round == 0) quadric(nv, pos.View, org.View, Q.View, cur.View, vstart.View, vfaces.View);
                best(nv, pos.View, org.View, Q.View, cur.View, vstart.View, vfaces.View, bestCost.View, bestNbr.View);
                // a few matching passes over the same edge costs: each takes the cheapest edges that do not touch an earlier pass's, so a round collapses a large share of the mesh instead of a sparse local-minimum set
                fillMinus1(nv, selected.View); blocked.MemSetToZero();
                int selectedCount = 0;
                for (int pass = 0; pass < SubPasses; pass++)
                {
                    ring(nv, cur.View, vstart.View, vfaces.View, bestCost.View, bestNbr.View, blocked.View, nmCost.View, nmA.View, nmB.View);
                    counter.MemSetToZero();
                    select(nv, bestCost.View, bestNbr.View, nmCost.View, nmA.View, nmB.View, blocked.View, selected.View, counter.View);
                    int got = (await counter.View.CopyToHostAsync())[0];
                    if (got == 0) break;
                    selectedCount += got;
                    block(nv, selected.View, cur.View, vstart.View, vfaces.View, blocked.View);
                }
                if (selectedCount == 0) break;
                // faces left to remove, at about two per collapse; when a full round would overshoot, collapse only the cheapest edges that reach the target
                int toRemove = nf - targetFaces;
                if (selectedCount * 2 > toRemove)
                {
                    int want = Math.Max(1, toRemove / 2);
                    counter.MemSetToZero(); maxCost(nv, bestCost.View, selected.View, counter.View);
                    int hi = (await counter.View.CopyToHostAsync())[0], lo = 0;      // bisection over float bit patterns for the want-th smallest selected cost
                    while (lo < hi)
                    {
                        int mid = lo + (hi - lo) / 2;
                        counter.MemSetToZero();
                        countBelow(nv, bestCost.View, selected.View, counter.View, BitConverter.Int32BitsToSingle(mid));
                        int below = (await counter.View.CopyToHostAsync())[0];
                        if (below >= want) hi = mid; else lo = mid + 1;
                    }
                    float tau = BitConverter.Int32BitsToSingle(lo);
                    counter.MemSetToZero();
                    threshold(nv, bestCost.View, selected.View, counter.View, tau);
                    selectedCount = (await counter.View.CopyToHostAsync())[0];
                }
                iota(nv, rep.View);
                collapse(nv, pos.View, org.View, Q.View, selected.View, rep.View);
                remap(nf, cur.View, rep.View, keep.View);
                int kept = await _sk.ExclusiveScanAsync(keep.View, nf, koff.View);
                compact(nf, keep.View, koff.View, cur.View, other.View);
                (cur, other) = (other, cur);
                LastRounds.Add((kept, selectedCount));
                if (kept == nf) break;
                nf = kept;
            }

            // the vertices some face still uses
            var result = new DeviceTriMesh { FaceCount = nf };
            var markUsed = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(MarkUsedImpl);
            var gather = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, FV, FV>(GatherPositionsImpl);
            var renumber = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(RenumberImpl);
            using var used = _acc.Allocate1D<int>(nv); used.MemSetToZero();
            markUsed(nf, cur.View, used.View);
            using var uoff = _acc.Allocate1D<int>(nv + 1L);
            int nOut = await _sk.ExclusiveScanAsync(used.View, nv, uoff.View);
            result.Positions = _acc.Allocate1D<float>(Math.Max(3L, (long)nOut * 3));
            gather(nv, used.View, uoff.View, pos.View, result.Positions.View);
            renumber(nf * 3, uoff.View, cur.View);
            result.Faces = _acc.Allocate1D<int>(Math.Max(3L, (long)nf * 3));
            result.Faces.View.SubView(0, (long)nf * 3).CopyFrom(cur.View.SubView(0, (long)nf * 3));
            result.VertexCount = nOut;
            await _acc.SynchronizeAsync();
            return result;
        }
        finally { facesA.Dispose(); facesB.Dispose(); }
    }
}
