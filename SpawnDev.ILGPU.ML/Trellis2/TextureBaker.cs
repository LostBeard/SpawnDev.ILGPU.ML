using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// Bakes the per-voxel PBR attributes into atlas textures on the device (the bake of o_voxel.postprocess.to_glb):
///   1. rasterise every triangle in UV space: texel centre inside the triangle -> (triangle id, barycentrics),
///   2. for each covered texel interpolate the 3D position and sample the sparse attribute volume trilinearly
///      (8 voxel centres, weights 1 - |c - p| per axis, absent voxels drop out and the rest are renormalised),
///   3. dilate the covered texels outwards so mip-mapping and bilinear filtering never read an empty texel,
///   4. pack 8-bit RGBA images: base colour + alpha, and glTF metallic-roughness (G = roughness, B = metallic).
/// Voxel (i,j,k) has its centre at (i+0.5, j+0.5, k+0.5) voxel units, volume origin -0.5 (the aabb of the pipeline).
/// </summary>
public sealed class TextureBaker
{
    private readonly Accelerator _acc;
    private readonly SparseKernels _sk;
    public TextureBaker(Accelerator accelerator, SparseKernels sparse) { _acc = accelerator; _sk = sparse; }

    /// <summary>The baked atlas: two RGBA8 images (row 0 first, little-endian packed R|G&lt;&lt;8|B&lt;&lt;16|A&lt;&lt;24) read to the host (a terminal file sink).</summary>
    public sealed class Textures
    {
        public int Size;
        public uint[] BaseColorAlpha = Array.Empty<uint>();
        public uint[] MetallicRoughness = Array.Empty<uint>();
        public long CoveredTexels;
    }

    private Action<Index1D, FV, IV, IV, FV, int, int>? _raster;
    private static void RasterImpl(Index1D tri, FV uv, IV faces, IV idMap, FV bary, int size, int numFaces)
    {
        int i0 = faces[tri * 3], i1 = faces[tri * 3 + 1], i2 = faces[tri * 3 + 2];
        float ax = uv[i0 * 2] * size, ay = uv[i0 * 2 + 1] * size;
        float bx = uv[i1 * 2] * size, by = uv[i1 * 2 + 1] * size;
        float cx = uv[i2 * 2] * size, cy = uv[i2 * 2 + 1] * size;
        float det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
        if (MathF.Abs(det) < 1e-12f) return;
        int x0 = (int)MathF.Max(0f, MathF.Floor(MathF.Min(ax, MathF.Min(bx, cx))));
        int x1 = (int)MathF.Min(size - 1f, MathF.Ceiling(MathF.Max(ax, MathF.Max(bx, cx))));
        int y0 = (int)MathF.Max(0f, MathF.Floor(MathF.Min(ay, MathF.Min(by, cy))));
        int y1 = (int)MathF.Min(size - 1f, MathF.Ceiling(MathF.Max(ay, MathF.Max(by, cy))));
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++)
            {
                float px = x + 0.5f, py = y + 0.5f;
                float w0 = ((bx - px) * (cy - py) - (cx - px) * (by - py)) / det;
                float w1 = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) / det;
                float w2 = 1f - w0 - w1;
                const float eps = -1e-5f;     // include edge texels so shared edges leave no crack
                if (w0 >= eps && w1 >= eps && w2 >= eps)
                {
                    int t = y * size + x;
                    idMap[t] = tri + 1;
                    bary[t * 2] = w0; bary[t * 2 + 1] = w1;
                }
            }
    }

    /// <summary>
    /// Second raster pass, for texels no triangle covered at the texel centre: a sliver or a tiny triangle can sit between texel centres and
    /// would leave a hole (black speck) in its own chart. A still-empty texel whose centre lies within 0.75 texel of a triangle takes that
    /// triangle, sampled at the nearest point of it (barycentrics clamped into the triangle). Runs after the centre pass (separate dispatch),
    /// so it never overrides a properly covered texel.
    /// </summary>
    private Action<Index1D, FV, IV, IV, FV, int, int>? _rasterNear;
    private static void RasterNearImpl(Index1D tri, FV uv, IV faces, IV idMap, FV bary, int size, int numFaces)
    {
        int i0 = faces[tri * 3], i1 = faces[tri * 3 + 1], i2 = faces[tri * 3 + 2];
        float ax = uv[i0 * 2] * size, ay = uv[i0 * 2 + 1] * size;
        float bx = uv[i1 * 2] * size, by = uv[i1 * 2 + 1] * size;
        float cx = uv[i2 * 2] * size, cy = uv[i2 * 2 + 1] * size;
        float det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
        if (MathF.Abs(det) < 1e-12f) return;
        int x0 = (int)MathF.Max(0f, MathF.Floor(MathF.Min(ax, MathF.Min(bx, cx)) - 0.75f));
        int x1 = (int)MathF.Min(size - 1f, MathF.Ceiling(MathF.Max(ax, MathF.Max(bx, cx)) + 0.75f));
        int y0 = (int)MathF.Max(0f, MathF.Floor(MathF.Min(ay, MathF.Min(by, cy)) - 0.75f));
        int y1 = (int)MathF.Min(size - 1f, MathF.Ceiling(MathF.Max(ay, MathF.Max(by, cy)) + 0.75f));
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++)
            {
                int t = y * size + x;
                if (idMap[t] != 0) continue;
                float px = x + 0.5f, py = y + 0.5f;
                float w0 = ((bx - px) * (cy - py) - (cx - px) * (by - py)) / det;
                float w1 = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) / det;
                float w2 = 1f - w0 - w1;
                w0 = MathF.Max(w0, 0f); w1 = MathF.Max(w1, 0f); w2 = MathF.Max(w2, 0f);
                float ws = w0 + w1 + w2; if (ws <= 1e-12f) continue;
                w0 /= ws; w1 /= ws; w2 /= ws;
                float qx = ax * w0 + bx * w1 + cx * w2 - px, qy = ay * w0 + by * w1 + cy * w2 - py;   // offset of the nearest point of the triangle from the texel centre
                if (qx * qx + qy * qy > 0.5625f) continue;
                idMap[t] = tri + 1;
                bary[t * 2] = w0; bary[t * 2 + 1] = w1;
            }
    }

    private Action<Index1D, IV, FV, IV, FV, IV, IV, FV, FV, int, int, int>? _bake;
    private static void BakeImpl(Index1D texel, IV idMap, FV bary, IV faces, FV positions, IV keys, IV vals, FV attrs, FV outAttrs,
        int grid, int mask, int shift)
    {
        int id = idMap[texel];
        if (id == 0) return;
        int tri = id - 1;
        float w0 = bary[texel * 2], w1 = bary[texel * 2 + 1], w2 = 1f - w0 - w1;
        int i0 = faces[tri * 3], i1 = faces[tri * 3 + 1], i2 = faces[tri * 3 + 2];
        // volume origin -0.5, voxel size 1/grid  ->  position in voxel units
        float px = (positions[i0 * 3] * w0 + positions[i1 * 3] * w1 + positions[i2 * 3] * w2 + 0.5f) * grid;
        float py = (positions[i0 * 3 + 1] * w0 + positions[i1 * 3 + 1] * w1 + positions[i2 * 3 + 1] * w2 + 0.5f) * grid;
        float pz = (positions[i0 * 3 + 2] * w0 + positions[i1 * 3 + 2] * w1 + positions[i2 * 3 + 2] * w2 + 0.5f) * grid;
        float a0 = 0f, a1 = 0f, a2 = 0f, a3 = 0f, a4 = 0f, a5 = 0f, wsum = 0f;
        for (int o = 0; o < 8; o++)
        {
            int nx = (int)(px + ((o & 4) != 0 ? 0.5f : -0.5f));
            int ny = (int)(py + ((o & 2) != 0 ? 0.5f : -0.5f));
            int nz = (int)(pz + ((o & 1) != 0 ? 0.5f : -0.5f));
            int v = SparseKernels.Lookup(keys, vals, nx, ny, nz, grid, mask, shift);
            if (v < 0) continue;
            float w = (1f - MathF.Abs(nx + 0.5f - px)) * (1f - MathF.Abs(ny + 0.5f - py)) * (1f - MathF.Abs(nz + 0.5f - pz));
            a0 += w * attrs[v * 6]; a1 += w * attrs[v * 6 + 1]; a2 += w * attrs[v * 6 + 2];
            a3 += w * attrs[v * 6 + 3]; a4 += w * attrs[v * 6 + 4]; a5 += w * attrs[v * 6 + 5];
            wsum += w;
        }
        if (wsum < 1e-6f)
        {
            // no voxel within a voxel of the sample (a simplified mesh can sit a few voxels off the surface): take the nearest voxel within 3 instead of black
            float best = 1e30f;
            int cx = (int)MathF.Floor(px), cy = (int)MathF.Floor(py), cz = (int)MathF.Floor(pz);
            for (int dz = -3; dz <= 3; dz++)
                for (int dy = -3; dy <= 3; dy++)
                    for (int dx = -3; dx <= 3; dx++)
                    {
                        int v = SparseKernels.Lookup(keys, vals, cx + dx, cy + dy, cz + dz, grid, mask, shift);
                        if (v < 0) continue;
                        float ddx = cx + dx + 0.5f - px, ddy = cy + dy + 0.5f - py, ddz = cz + dz + 0.5f - pz;
                        float d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                        if (d2 >= best) continue;
                        best = d2; a0 = attrs[v * 6]; a1 = attrs[v * 6 + 1]; a2 = attrs[v * 6 + 2]; a3 = attrs[v * 6 + 3]; a4 = attrs[v * 6 + 4]; a5 = attrs[v * 6 + 5]; wsum = 1f;
                    }
        }
        float inv = 1f / MathF.Max(wsum, 1e-12f);
        outAttrs[texel * 6] = a0 * inv; outAttrs[texel * 6 + 1] = a1 * inv; outAttrs[texel * 6 + 2] = a2 * inv;
        outAttrs[texel * 6 + 3] = a3 * inv; outAttrs[texel * 6 + 4] = a4 * inv; outAttrs[texel * 6 + 5] = a5 * inv;
    }

    // one dilation pass: an uncovered texel takes the mean of its covered 8-neighbours
    private Action<Index1D, FV, IV, FV, IV, int>? _dilate;
    private static void DilateImpl(Index1D texel, FV src, IV srcCovered, FV dst, IV dstCovered, int size)
    {
        int x = texel % size, y = texel / size;
        if (srcCovered[texel] != 0)
        {
            for (int c = 0; c < 6; c++) dst[texel * 6 + c] = src[texel * 6 + c];
            dstCovered[texel] = 1;
            return;
        }
        float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f, s4 = 0f, s5 = 0f; int n = 0;
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
            {
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
                int nt = ny * size + nx;
                if (srcCovered[nt] == 0) continue;
                s0 += src[nt * 6]; s1 += src[nt * 6 + 1]; s2 += src[nt * 6 + 2]; s3 += src[nt * 6 + 3]; s4 += src[nt * 6 + 4]; s5 += src[nt * 6 + 5];
                n++;
            }
        if (n == 0) { dstCovered[texel] = 0; return; }
        float inv = 1f / n;
        dst[texel * 6] = s0 * inv; dst[texel * 6 + 1] = s1 * inv; dst[texel * 6 + 2] = s2 * inv;
        dst[texel * 6 + 3] = s3 * inv; dst[texel * 6 + 4] = s4 * inv; dst[texel * 6 + 5] = s5 * inv;
        dstCovered[texel] = 1;
    }

    private Action<Index1D, IV, IV>? _coverFlag;
    private static void CoverFlagImpl(Index1D t, IV idMap, IV covered) => covered[t] = idMap[t] != 0 ? 1 : 0;

    private Action<Index1D, FV, IV, IV>? _pack;
    private static uint Q(float v) => (uint)(MathF.Max(0f, MathF.Min(1f, v)) * 255f + 0.5f);
    private static void PackImpl(Index1D t, FV attrs, IV baseColorOut, IV mrOut)
    {
        uint r = Q(attrs[t * 6]), g = Q(attrs[t * 6 + 1]), b = Q(attrs[t * 6 + 2]);
        uint metallic = Q(attrs[t * 6 + 3]), roughness = Q(attrs[t * 6 + 4]), alpha = Q(attrs[t * 6 + 5]);
        baseColorOut[t] = (int)(r | (g << 8) | (b << 16) | (alpha << 24));
        mrOut[t] = (int)((0u) | (roughness << 8) | (metallic << 16) | (255u << 24));      // glTF: G = roughness, B = metallic
    }

    private Action<Index1D, IV>? _zeroInt;
    private static void ZeroIntImpl(Index1D i, IV v) => v[i] = 0;
    private Action<Index1D, FV>? _zeroF;
    private static void ZeroFImpl(Index1D i, FV v) => v[i] = 0f;

    /// <summary>
    /// Bake. <paramref name="positions"/> [V,3], <paramref name="uvs"/> [V,2] in [0,1] (row = v * size), <paramref name="faces"/> [F,3] describe the
    /// unwrapped mesh; <paramref name="voxelCoords"/> [N,3] / <paramref name="voxelAttrs"/> [N,6] the attribute volume on a <paramref name="grid"/>^3 lattice.
    /// </summary>
    public Textures Bake(FV positions, FV uvs, IV faces, int faceCount, IV voxelCoords, FV voxelAttrs, int voxelCount, int grid, int size, int dilation = 8)
        => Task.Run(() => BakeAsync(positions, uvs, faces, faceCount, voxelCoords, voxelAttrs, voxelCount, grid, size, dilation)).GetAwaiter().GetResult();

    /// <summary><see cref="Bake"/>, browser-safe (the textures are awaited readbacks).</summary>
    public async Task<Textures> BakeAsync(FV positions, FV uvs, IV faces, int faceCount, IV voxelCoords, FV voxelAttrs, int voxelCount, int grid, int size, int dilation = 8)
    {
        _raster ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, FV, int, int>(RasterImpl);
        _rasterNear ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, FV, int, int>(RasterNearImpl);
        _bake ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, FV, IV, FV, IV, IV, FV, FV, int, int, int>(BakeImpl);
        _dilate ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, FV, IV, int>(DilateImpl);
        _coverFlag ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(CoverFlagImpl);
        _pack ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV>(PackImpl);
        _zeroInt ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV>(ZeroIntImpl);
        _zeroF ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV>(ZeroFImpl);
        int texels = size * size;
        using var idMap = _acc.Allocate1D<int>(texels);
        using var bary = _acc.Allocate1D<float>((long)texels * 2);
        _zeroInt(texels, idMap.View);
        _raster(faceCount, uvs, faces, idMap.View, bary.View, size, faceCount);   // (dispatches run in order: the near pass sees every centre-covered texel)
        _rasterNear(faceCount, uvs, faces, idMap.View, bary.View, size, faceCount);
        using var hash = _sk.BuildHash(voxelCoords, voxelCount, grid);
        var a = _acc.Allocate1D<float>((long)texels * 6);
        var b = _acc.Allocate1D<float>((long)texels * 6);
        _zeroF(texels * 6, a.View); _zeroF(texels * 6, b.View);
        _bake(texels, idMap.View, bary.View, faces, positions, hash.Keys.View, hash.Vals.View, voxelAttrs, a.View, grid, hash.Mask, hash.Shift);
        var covA = _acc.Allocate1D<int>(texels);
        var covB = _acc.Allocate1D<int>(texels);
        _coverFlag(texels, idMap.View, covA.View);
        var tex = new Textures { Size = size };
        for (int pass = 0; pass < dilation; pass++)
        {
            _dilate(texels, a.View, covA.View, b.View, covB.View, size);
            (a, b) = (b, a); (covA, covB) = (covB, covA);
        }
        using var bc = _acc.Allocate1D<int>(texels);
        using var mr = _acc.Allocate1D<int>(texels);
        _pack(texels, a.View, bc.View, mr.View);
        tex.BaseColorAlpha = ToUInt(await bc.View.CopyToHostAsync());
        tex.MetallicRoughness = ToUInt(await mr.View.CopyToHostAsync());
        // covered texels (before dilation) for reporting
        var ids = await idMap.View.CopyToHostAsync(); long covered = 0; foreach (var v in ids) if (v != 0) covered++;
        tex.CoveredTexels = covered;
        a.Dispose(); b.Dispose(); covA.Dispose(); covB.Dispose();
        return tex;
    }

    private static uint[] ToUInt(int[] src) { var r = new uint[src.Length]; Buffer.BlockCopy(src, 0, r, 0, src.Length * 4); return r; }
}
