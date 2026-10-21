using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// <see cref="NarrowBandRemesher"/> with the mesh on the device from end to end: the narrow band of the shifted distance field is found, its corners deduplicated with an atomic hash, the dual vertices, quads and triangles built, and the pieces
/// that face into the model dropped (connected components by a lock-free union-find, signed volume by fixed-point atomics), all in kernels. The host topology of the original (hash tables, lists, union-find over millions of faces in C#) was seconds of
/// main-thread time and hundreds of MB of managed arrays in the browser. Nothing but a handful of counts that size the next buffers crosses to the host; the result stays in device buffers for the stages that follow.
/// Same algorithm and same quad / triangle rules as the host version (<see cref="NarrowBandRemesher"/>): the vertex set is identical, the triangle split is decided in float instead of double.
/// </summary>
public sealed class DeviceRemesher
{
    private readonly Accelerator _acc;
    private readonly SparseKernels _sk;
    public DeviceRemesher(Accelerator accelerator, SparseKernels sparse) { _acc = accelerator; _sk = sparse; }

    /// <summary>The remeshed surface in device buffers (world units), and the distance grid of the INPUT mesh for the texture bake. Dispose frees them all.</summary>
    public sealed class Mesh : IDisposable
    {
        public MemoryBuffer1D<float, Stride1D.Dense> Positions = null!;
        public MemoryBuffer1D<int, Stride1D.Dense> Faces = null!;
        public int VertexCount, FaceCount;
        public MeshDistanceGrid Source = null!;
        public int ActiveVoxels, GridVertices;
        public void Dispose() { Positions?.Dispose(); Faces?.Dispose(); Source?.Dispose(); }
    }

    // ---- corners -------------------------------------------------------------------------------------------------------------------------------
    private static void CornerInsertImpl(Index1D i, IV coords, IV keys, IV overflow, int v1, int mask, int shift)
    {
        int x = coords[i * 3], y = coords[i * 3 + 1], z = coords[i * 3 + 2];
        for (int o = 0; o < 8; o++)
        {
            int key = ((x + (o & 1)) * v1 + (y + ((o >> 1) & 1))) * v1 + (z + ((o >> 2) & 1));
            int slot = (int)(((uint)key * 2654435761u) >> shift);
            int probes = 0;
            while (true)
            {
                int old = Atomic.CompareExchange(ref keys[slot], -1, key);
                if (old == -1 || old == key) break;
                slot = (slot + 1) & mask;
                if (++probes > mask) { overflow[0] = 1; break; }       // the table is full: the caller rebuilds it bigger
            }
        }
    }

    private static void SlotFlagImpl(Index1D s, IV keys, IV flag) { flag[s] = keys[s] != -1 ? 1 : 0; }

    private static void CornerPointsImpl(Index1D s, IV keys, IV flag, IV offsets, FV pts, int v1)
    {
        if (flag[s] == 0) return;
        int key = keys[s], id = offsets[s];
        pts[id * 3] = key / (v1 * v1); pts[id * 3 + 1] = (key / v1) % v1; pts[id * 3 + 2] = key % v1;
    }

    private static void VoxelCornerIdImpl(Index1D idx, IV coords, IV keys, IV offsets, IV cornerId, int v1, int mask, int shift)
    {
        int i = idx >> 3, o = idx & 7;
        int key = ((coords[i * 3] + (o & 1)) * v1 + (coords[i * 3 + 1] + ((o >> 1) & 1))) * v1 + (coords[i * 3 + 2] + ((o >> 2) & 1));
        int slot = (int)(((uint)key * 2654435761u) >> shift);
        while (keys[slot] != key) slot = (slot + 1) & mask;
        cornerId[idx] = offsets[slot];
    }

    // ---- dual vertices and crossing edges --------------------------------------------------------------------------------------------------
    private static void DualImpl(Index1D i, IV coords, IV cornerId, FV dist, FV dual, IV dir)
    {
        float sx = 0f, sy = 0f, sz = 0f; int cnt = 0;
        int x = coords[i * 3], y = coords[i * 3 + 1], z = coords[i * 3 + 2];
        for (int axis = 0; axis < 3; axis++)
            for (int u = 0; u <= 1; u++)
                for (int v = 0; v <= 1; v++)
                {
                    int o1, o2;
                    if (axis == 0) { o1 = (u << 1) | (v << 2); o2 = o1 | 1; }
                    else if (axis == 1) { o1 = u | (v << 2); o2 = o1 | 2; }
                    else { o1 = u | (v << 1); o2 = o1 | 4; }
                    float v1 = dist[cornerId[i * 8 + o1]], v2 = dist[cornerId[i * 8 + o2]];
                    bool cross = (v1 < 0f && v2 >= 0f) || (v1 >= 0f && v2 < 0f);
                    if (cross)
                    {
                        float t = -v1 / (v2 - v1);
                        if (axis == 0) { sx += x + t; sy += y + u; sz += z + v; }
                        else if (axis == 1) { sx += x + u; sy += y + t; sz += z + v; }
                        else { sx += x + u; sy += y + v; sz += z + t; }
                        cnt++;
                    }
                    if (u == 1 && v == 1) dir[i * 3 + axis] = v1 < 0f && v2 >= 0f ? 1 : (v1 >= 0f && v2 < 0f ? -1 : 0);
                }
        if (cnt > 0) { dual[i * 3] = sx / cnt; dual[i * 3 + 1] = sy / cnt; dual[i * 3 + 2] = sz / cnt; }
        else { dual[i * 3] = x + 0.5f; dual[i * 3 + 1] = y + 0.5f; dual[i * 3 + 2] = z + 0.5f; }
    }

    // ---- quads -------------------------------------------------------------------------------------------------------------------------------------
    private static void QuadFlagImpl(Index1D idx, IV coords, IV keys, IV vals, IV dir, IV quad, IV flag, int resolution, int mask, int shift)
    {
        int i = idx / 3, axis = idx % 3;
        flag[idx] = 0;
        if (dir[idx] == 0) return;
        int x = coords[i * 3], y = coords[i * 3 + 1], z = coords[i * 3 + 2];
        int d1x = 0, d1y = 0, d1z = 0, d2x = 0, d2y = 0, d2z = 0, d3x = 0, d3y = 0, d3z = 0;
        if (axis == 0) { d1z = 1; d2y = 1; d2z = 1; d3y = 1; }
        else if (axis == 1) { d1x = 1; d2x = 1; d2z = 1; d3z = 1; }
        else { d1y = 1; d2x = 1; d2y = 1; d3x = 1; }
        int q1 = SparseKernels.Lookup(keys, vals, x + d1x, y + d1y, z + d1z, resolution, mask, shift);
        int q2 = SparseKernels.Lookup(keys, vals, x + d2x, y + d2y, z + d2z, resolution, mask, shift);
        int q3 = SparseKernels.Lookup(keys, vals, x + d3x, y + d3y, z + d3z, resolution, mask, shift);
        if (q1 < 0 || q2 < 0 || q3 < 0) return;
        quad[idx * 4] = i; quad[idx * 4 + 1] = q1; quad[idx * 4 + 2] = q2; quad[idx * 4 + 3] = q3;
        flag[idx] = 1;
    }

    private static void QuadCompactImpl(Index1D idx, IV flag, IV offsets, IV quad, IV dir, IV qc, IV qdir)
    {
        if (flag[idx] == 0) return;
        int o = offsets[idx];
        qc[o * 4] = quad[idx * 4]; qc[o * 4 + 1] = quad[idx * 4 + 1]; qc[o * 4 + 2] = quad[idx * 4 + 2]; qc[o * 4 + 3] = quad[idx * 4 + 3];
        qdir[o] = dir[idx];
    }

    private static void MarkQuadVerticesImpl(Index1D k, IV qc, IV used)
    {
        used[qc[k * 4]] = 1; used[qc[k * 4 + 1]] = 1; used[qc[k * 4 + 2]] = 1; used[qc[k * 4 + 3]] = 1;
    }

    private static void OutPositionsImpl(Index1D v, IV used, IV offsets, FV dual, FV outPos, int resolution, float scale)
    {
        if (used[v] == 0) return;
        int o = offsets[v] * 3;
        outPos[o] = (dual[v * 3] / resolution - 0.5f) * scale;
        outPos[o + 1] = (dual[v * 3 + 1] / resolution - 0.5f) * scale;
        outPos[o + 2] = (dual[v * 3 + 2] / resolution - 0.5f) * scale;
    }

    // the reference's four ways to cut a quad into two triangles, 2 bits per index: N1 = 0 1 2 0 2 3, P1 = 0 2 1 0 3 2, N2 = 0 1 3 3 1 2, P2 = 0 3 1 3 2 1
    private const int SplitN1 = 3620, SplitP1 = 2840, SplitN2 = 2548, SplitP2 = 1756;

    /// <summary>|n0 . n1| exactly as the reference computes it, on the first four vertices of the split.</summary>
    private static float Align(FV p, int q0, int q1, int q2, int q3, int split)
    {
        int ia = QuadVertex(q0, q1, q2, q3, split & 3) * 3, ib = QuadVertex(q0, q1, q2, q3, (split >> 2) & 3) * 3;
        int ic = QuadVertex(q0, q1, q2, q3, (split >> 4) & 3) * 3, id = QuadVertex(q0, q1, q2, q3, (split >> 6) & 3) * 3;
        float ux = p[ib] - p[ia], uy = p[ib + 1] - p[ia + 1], uz = p[ib + 2] - p[ia + 2];
        float vx = p[ic] - p[ia], vy = p[ic + 1] - p[ia + 1], vz = p[ic + 2] - p[ia + 2];
        float n0x = uy * vz - uz * vy, n0y = uz * vx - ux * vz, n0z = ux * vy - uy * vx;
        float wx = p[ic] - p[ib], wy = p[ic + 1] - p[ib + 1], wz = p[ic + 2] - p[ib + 2];
        float zx = p[id] - p[ib], zy = p[id + 1] - p[ib + 1], zz = p[id + 2] - p[ib + 2];
        float n1x = wy * zz - wz * zy, n1y = wz * zx - wx * zz, n1z = wx * zy - wy * zx;
        return MathF.Abs(n0x * n1x + n0y * n1y + n0z * n1z);
    }

    private static int QuadVertex(int q0, int q1, int q2, int q3, int which) => which == 0 ? q0 : which == 1 ? q1 : which == 2 ? q2 : q3;

    private static void TrianglesImpl(Index1D k, IV qc, IV qdir, IV map, FV outPos, IV tris)
    {
        int q0 = map[qc[k * 4]], q1 = map[qc[k * 4 + 1]], q2 = map[qc[k * 4 + 2]], q3 = map[qc[k * 4 + 3]];
        bool plus = qdir[k] == 1;
        int s1 = plus ? SplitP1 : SplitN1, s2 = plus ? SplitP2 : SplitN2;
        float a0 = Align(outPos, q0, q1, q2, q3, s1), a1 = Align(outPos, q0, q1, q2, q3, s2);
        int s = a0 > a1 ? s1 : s2;
        for (int c = 0; c < 6; c++) tris[k * 6 + c] = QuadVertex(q0, q1, q2, q3, (s >> (2 * c)) & 3);
    }

    // ---- components and inward pieces --------------------------------------------------------------------------------------------------------
    private static void IotaImpl(Index1D i, IV v) { v[i] = i; }

    // Every access to `parent` is an atomic operation: the buffer is atomic in the entry kernel (the CAS in Link), and the WGSL emitter does not turn a PLAIN load or store through a helper
    // function's pointer parameter into atomicLoad / atomicStore ("cannot assign 'atomic<i32>' to 'i32'"). Tracked in Plans/webgpu-atomic-buffer-helper-access-2026-10-19.md.
    private static int Find(IV parent, int x)
    {
        while (true)
        {
            int p = Atomic.Add(ref parent[x], 0);                    // atomic load
            if (p == x) return x;
            int gp = Atomic.Add(ref parent[p], 0);
            if (gp != p) Atomic.CompareExchange(ref parent[x], p, gp);   // path halving; a lost race only leaves a longer path
            x = gp;
        }
    }

    private static void Link(IV parent, int a, int b)
    {
        while (true)
        {
            a = Find(parent, a); b = Find(parent, b);
            if (a == b) return;
            if (a < b) { int t = a; a = b; b = t; }
            int old = Atomic.CompareExchange(ref parent[a], a, b);       // the larger root hangs under the smaller: no cycles
            if (old == a) return;
        }
    }

    private static void UnionImpl(Index1D t, IV tris, IV parent)
    {
        int a = tris[t * 3], b = tris[t * 3 + 1], c = tris[t * 3 + 2];
        Link(parent, a, b); Link(parent, b, c);
    }

    private static void RootImpl(Index1D v, IV parent, IV root) { root[v] = Find(parent, v); }

    private const float VolumeScale = 268435456f;      // 2^28: a face's signed volume term (<= ~1e-2) as a fixed-point int; the sums wrap harmlessly (modular), the total stays far below 2^31 / 2^28 = 8

    private static void VolumeImpl(Index1D t, IV tris, FV p, IV root, IV vol)
    {
        int a = tris[t * 3] * 3, b = tris[t * 3 + 1] * 3, c = tris[t * 3 + 2] * 3;
        float triple = p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1]) - p[a + 1] * (p[b] * p[c + 2] - p[b + 2] * p[c]) + p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c]);
        Atomic.Add(ref vol[root[tris[t * 3]]], (int)MathF.Round(triple / 6f * VolumeScale));
    }

    private const float AreaScale = 67108864f;         // 2^26: a face's area as a fixed-point int; a surface of up to ~30 square units stays below 2^31

    private static void AreaImpl(Index1D t, IV tris, FV p, IV root, IV area)
    {
        int a = tris[t * 3] * 3, b = tris[t * 3 + 1] * 3, c = tris[t * 3 + 2] * 3;
        float ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2], vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
        float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        Atomic.Add(ref area[root[tris[t * 3]]], (int)MathF.Round(0.5f * MathF.Sqrt(nx * nx + ny * ny + nz * nz) * AreaScale));
    }

    // a piece goes when it faces into the model (negative signed volume) or is a floating fragment (area below the threshold, as MeshRepair's RemoveFragments)
    private static void AnyDropImpl(Index1D v, IV root, IV vol, IV area, IV flag, int minArea) { if (root[v] == v && (vol[v] < 0 || area[v] < minArea)) flag[0] = 1; }

    private static void KeepFaceImpl(Index1D t, IV tris, IV root, IV vol, IV area, IV keep, int minArea) { int r = root[tris[t * 3]]; keep[t] = vol[r] >= 0 && area[r] >= minArea ? 1 : 0; }

    private static void CompactFacesImpl(Index1D t, IV keep, IV offsets, IV tris, IV outTris)
    {
        if (keep[t] == 0) return;
        int o = offsets[t] * 3;
        outTris[o] = tris[t * 3]; outTris[o + 1] = tris[t * 3 + 1]; outTris[o + 2] = tris[t * 3 + 2];
    }

    private static void MarkFaceVerticesImpl(Index1D t, IV tris, IV used) { used[tris[t * 3]] = 1; used[tris[t * 3 + 1]] = 1; used[tris[t * 3 + 2]] = 1; }

    private static void RemapPositionsImpl(Index1D v, IV used, IV offsets, FV pos, FV outPos)
    {
        if (used[v] == 0) return;
        int o = offsets[v] * 3;
        outPos[o] = pos[v * 3]; outPos[o + 1] = pos[v * 3 + 1]; outPos[o + 2] = pos[v * 3 + 2];
    }

    private static void RemapFacesImpl(Index1D i, IV offsets, IV tris) { tris[i] = offsets[tris[i]]; }

    private static void FillMinus1Impl(Index1D i, IV v) { v[i] = -1; }

    /// <summary>Pieces with less area than this fraction of the volume's side squared are dropped as floating fragments (<see cref="MeshRepair.Options.MinComponentArea"/>); 0 keeps every piece with a positive volume.</summary>
    public double MinComponentArea { get; set; } = 1e-5;

    /// <param name="worldPositions">the input mesh on the device, xyz in world units (the volume is [-0.5, 0.5]^3); vertices no face uses are fine</param>
    /// <param name="faces">its triangles</param>
    /// <param name="resolution">the voxel grid of the volume</param>
    /// <param name="band">shell offset in voxels (reference 1)</param>
    /// <param name="dropInnerSkins">drop connected pieces whose signed volume is negative (see <see cref="NarrowBandRemesher.RemeshAsync"/>)</param>
    public async Task<Mesh> RemeshAsync(FV worldPositions, int vertexCount, IV faces, int faceCount, int resolution, float band = 1f, bool dropInnerSkins = true)
    {
        double scale = (resolution + 3.0 * band) / resolution;
        float toGrid = (float)(resolution / scale), offset = resolution * 0.5f;
        var result = new Mesh { Positions = _acc.Allocate1D<float>(3), Faces = _acc.Allocate1D<int>(3) };
        result.Source = await MeshDistanceGrid.BuildAsync(_acc, _sk, worldPositions, vertexCount, faces, faceCount, resolution, toGrid, offset);

        var (coords, nv) = await result.Source.BandVoxelsDeviceAsync(_sk, band, 0.87f);
        using var coordsOwner = coords;
        result.ActiveVoxels = nv;
        if (nv == 0) return result;

        var cornerInsert = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, int, int, int>(CornerInsertImpl);
        var slotFlag = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(SlotFlagImpl);
        var cornerPoints = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, FV, int>(CornerPointsImpl);
        var voxelCornerId = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV, int, int, int>(VoxelCornerIdImpl);
        var dualK = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, FV, FV, IV>(DualImpl);
        var quadFlag = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV, IV, IV, int, int, int>(QuadFlagImpl);
        var quadCompact = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV, IV, IV>(QuadCompactImpl);
        var markQuad = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(MarkQuadVerticesImpl);
        var outPositions = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, FV, FV, int, float>(OutPositionsImpl);
        var trianglesK = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, FV, IV>(TrianglesImpl);
        var fillMinus1 = _acc.LoadAutoGroupedStreamKernel<Index1D, IV>(FillMinus1Impl);

        // ---- unique corners of the active voxels, and the distance at each ---------------------------------------------------------------------
        int v1 = resolution + 1;
        if ((long)v1 * v1 * v1 >= uint.MaxValue) throw new ArgumentOutOfRangeException(nameof(resolution));        // corner keys are 32-bit bit patterns (wrapping arithmetic), exact while v1^3 < 2^32 - 1: 1537^3 = 3.6e9 fits
        using var voxHash = _sk.BuildHash(coords.View, nv, resolution);
        MemoryBuffer1D<int, Stride1D.Dense> cornerKeys;
        int cap = 64, bits = 6;
        while (cap < 3L * nv) { cap <<= 1; bits++; }
        while (true)
        {
            cornerKeys = _acc.Allocate1D<int>(cap);
            fillMinus1(cap, cornerKeys.View);
            using var overflow = _acc.Allocate1D<int>(1); overflow.MemSetToZero();
            cornerInsert(nv, coords.View, cornerKeys.View, overflow.View, v1, cap - 1, 32 - bits);
            var ovf = await overflow.View.CopyToHostAsync();
            if (ovf[0] == 0) break;
            cornerKeys.Dispose(); cap <<= 1; bits++;                 // load factor too high: rebuild at twice the size
        }
        using var cornerKeysOwner = cornerKeys;
        using var slotFlags = _acc.Allocate1D<int>(cap);
        using var slotOffsets = _acc.Allocate1D<int>(cap + 1L);
        slotFlag(cap, cornerKeys.View, slotFlags.View);
        int nc = await _sk.ExclusiveScanAsync(slotFlags.View, cap, slotOffsets.View);
        result.GridVertices = nc;
        using var pts = _acc.Allocate1D<float>(Math.Max(1L, (long)nc * 3));
        cornerPoints(cap, cornerKeys.View, slotFlags.View, slotOffsets.View, pts.View, v1);
        using var cornerId = _acc.Allocate1D<int>((long)nv * 8);
        voxelCornerId(nv * 8, coords.View, cornerKeys.View, slotOffsets.View, cornerId.View, v1, cap - 1, 32 - bits);
        using var dist = _acc.Allocate1D<float>(Math.Max(1L, nc));
        result.Source.DistancesDevice(pts.View, nc, dist.View, band);           // < 0 inside the band, >= 0 outside

        // ---- dual vertices, the edges that cross, and the quads around them --------------------------------------------------------------------
        using var dual = _acc.Allocate1D<float>((long)nv * 3);
        using var dir = _acc.Allocate1D<int>((long)nv * 3);
        dualK(nv, coords.View, cornerId.View, dist.View, dual.View, dir.View);
        using var quad = _acc.Allocate1D<int>((long)nv * 12);
        using var qflag = _acc.Allocate1D<int>((long)nv * 3);
        quadFlag(nv * 3, coords.View, voxHash.Keys.View, voxHash.Vals.View, dir.View, quad.View, qflag.View, resolution, voxHash.Mask, voxHash.Shift);
        using var qoff = _acc.Allocate1D<int>((long)nv * 3 + 1);
        int nq = await _sk.ExclusiveScanAsync(qflag.View, nv * 3, qoff.View);
        if (nq == 0) return result;
        using var qc = _acc.Allocate1D<int>((long)nq * 4);
        using var qdir = _acc.Allocate1D<int>(nq);
        quadCompact(nv * 3, qflag.View, qoff.View, quad.View, dir.View, qc.View, qdir.View);

        // ---- the vertices some quad uses, and the triangles ------------------------------------------------------------------------------------
        using var used = _acc.Allocate1D<int>(nv); used.MemSetToZero();
        markQuad(nq, qc.View, used.View);
        using var uoff = _acc.Allocate1D<int>(nv + 1L);
        int nOut = await _sk.ExclusiveScanAsync(used.View, nv, uoff.View);
        using var outPos = _acc.Allocate1D<float>((long)nOut * 3);
        outPositions(nv, used.View, uoff.View, dual.View, outPos.View, resolution, (float)scale);
        using var tris = _acc.Allocate1D<int>((long)nq * 6);
        trianglesK(nq, qc.View, qdir.View, uoff.View, outPos.View, tris.View);
        int nf = nq * 2;

        // ---- drop the pieces that face into the model ------------------------------------------------------------------------------------------
        if (dropInnerSkins)
        {
            var iota = _acc.LoadAutoGroupedStreamKernel<Index1D, IV>(IotaImpl);
            var union = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(UnionImpl);
            var rootK = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(RootImpl);
            var volumeK = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, FV, IV, IV>(VolumeImpl);
            var areaK = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, FV, IV, IV>(AreaImpl);
            var anyDrop = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV, int>(AnyDropImpl);
            var keepK = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV, IV, int>(KeepFaceImpl);
            using var parent = _acc.Allocate1D<int>(nOut);
            iota(nOut, parent.View);
            union(nf, tris.View, parent.View);
            using var root = _acc.Allocate1D<int>(nOut);
            rootK(nOut, parent.View, root.View);
            using var vol = _acc.Allocate1D<int>(nOut); vol.MemSetToZero();
            volumeK(nf, tris.View, outPos.View, root.View, vol.View);
            using var area = _acc.Allocate1D<int>(nOut); area.MemSetToZero();
            areaK(nf, tris.View, outPos.View, root.View, area.View);
            int minAreaFixed = (int)Math.Min(int.MaxValue, MinComponentArea * scale * scale * AreaScale);
            using var neg = _acc.Allocate1D<int>(1); neg.MemSetToZero();
            anyDrop(nOut, root.View, vol.View, area.View, neg.View, minAreaFixed);
            if ((await neg.View.CopyToHostAsync())[0] != 0)
            {
                using var keep = _acc.Allocate1D<int>(nf);
                keepK(nf, tris.View, root.View, vol.View, area.View, keep.View, minAreaFixed);
                using var foff = _acc.Allocate1D<int>(nf + 1L);
                int nKept = await _sk.ExclusiveScanAsync(keep.View, nf, foff.View);
                using var kept = _acc.Allocate1D<int>(Math.Max(3L, (long)nKept * 3));
                var compactFaces = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV>(CompactFacesImpl);
                compactFaces(nf, keep.View, foff.View, tris.View, kept.View);
                // renumber the vertices the kept faces use
                var markFaces = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(MarkFaceVerticesImpl);
                var remapPositions = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, FV, FV>(RemapPositionsImpl);
                var remapFaces = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(RemapFacesImpl);
                using var used2 = _acc.Allocate1D<int>(nOut); used2.MemSetToZero();
                markFaces(nKept, kept.View, used2.View);
                using var uoff2 = _acc.Allocate1D<int>(nOut + 1L);
                int n2 = await _sk.ExclusiveScanAsync(used2.View, nOut, uoff2.View);
                result.Positions.Dispose(); result.Faces.Dispose();
                result.Positions = _acc.Allocate1D<float>(Math.Max(3L, (long)n2 * 3));
                remapPositions(nOut, used2.View, uoff2.View, outPos.View, result.Positions.View);
                remapFaces(nKept * 3, uoff2.View, kept.View);
                result.Faces = _acc.Allocate1D<int>(Math.Max(3L, (long)nKept * 3));
                result.Faces.View.SubView(0, (long)nKept * 3).CopyFrom(kept.View.SubView(0, (long)nKept * 3));
                result.VertexCount = n2; result.FaceCount = nKept;
                await _acc.SynchronizeAsync();
                return result;
            }
        }
        result.Positions.Dispose(); result.Faces.Dispose();
        result.Positions = _acc.Allocate1D<float>(Math.Max(3L, (long)nOut * 3));
        result.Positions.View.SubView(0, (long)nOut * 3).CopyFrom(outPos.View.SubView(0, (long)nOut * 3));
        result.Faces = _acc.Allocate1D<int>(Math.Max(3L, (long)nf * 3));
        result.Faces.View.SubView(0, (long)nf * 3).CopyFrom(tris.View.SubView(0, (long)nf * 3));
        result.VertexCount = nOut; result.FaceCount = nf;
        await _acc.SynchronizeAsync();
        return result;
    }
}
