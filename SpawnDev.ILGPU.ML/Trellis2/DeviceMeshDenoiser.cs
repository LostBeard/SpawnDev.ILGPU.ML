using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// Feature-preserving mesh denoising on the device: the voxel-scale bumps of a remeshed surface (about a fifth of a voxel in height) that a mirror-like material turns into visible facets are
/// smoothed away while sharp edges stay sharp. After Zheng et al. 2011, "Bilateral normal filtering for mesh denoising": the face normals are filtered bilaterally (each face averages the faces that
/// share a vertex with it, weighted by their area and by exp(-(1 - cos angle) / sigma^2) against its own normal, so a face across a sharp edge counts for nothing), then every vertex moves to where
/// its faces' planes, turned to the filtered normals, would put it: <c>p += 1/|F| * sum_f n_f * (n_f . (c_f - p))</c> with c_f the face centre. The two steps alternate. The reference does not do this;
/// <see cref="Trellis2Options.SmoothNormals"/> turns it (and the normal smoothing) off.
/// </summary>
public sealed class DeviceMeshDenoiser
{
    private readonly Accelerator _acc;
    private readonly SparseKernels _sk;
    public DeviceMeshDenoiser(Accelerator accelerator, SparseKernels sparse) { _acc = accelerator; _sk = sparse; }

    /// <summary>Alternations of (filter the normals, move the vertices).</summary>
    public int Rounds { get; set; } = 4;
    /// <summary>Bilateral passes over the face normals in each round.</summary>
    public int NormalPasses { get; set; } = 5;
    /// <summary>Vertex update passes in each round.</summary>
    public int VertexPasses { get; set; } = 12;
    /// <summary>The range weight of the normal filter (degrees).</summary>
    public double SigmaDegrees { get; set; } = 30;

    private static void VertexUpdateImpl(Index1D v, FV src, IV faces, IV vstart, IV vfaces, FV fn, FV dst)
    {
        float px = src[v * 3], py = src[v * 3 + 1], pz = src[v * 3 + 2], sx = 0f, sy = 0f, sz = 0f;
        int deg = vstart[v + 1] - vstart[v];
        for (int i = vstart[v]; i < vstart[v + 1]; i++)
        {
            int f = vfaces[i], a = faces[f * 3], b = faces[f * 3 + 1], c = faces[f * 3 + 2];
            float cx = (src[a * 3] + src[b * 3] + src[c * 3]) * (1f / 3f), cy = (src[a * 3 + 1] + src[b * 3 + 1] + src[c * 3 + 1]) * (1f / 3f), cz = (src[a * 3 + 2] + src[b * 3 + 2] + src[c * 3 + 2]) * (1f / 3f);
            float nx = fn[f * 3], ny = fn[f * 3 + 1], nz = fn[f * 3 + 2];
            float d = nx * (cx - px) + ny * (cy - py) + nz * (cz - pz);
            sx += nx * d; sy += ny * d; sz += nz * d;
        }
        float k = deg > 0 ? 1f / deg : 0f;
        dst[v * 3] = px + sx * k; dst[v * 3 + 1] = py + sy * k; dst[v * 3 + 2] = pz + sz * k;
    }

    /// <summary>Denoise <paramref name="positions"/> (xyz per vertex) in place.</summary>
    public async Task DenoiseAsync(FV positions, int vertexCount, IV faces, int faceCount)
    {
        if (vertexCount == 0 || faceCount == 0 || Rounds <= 0) return;
        int nv = vertexCount, nf = faceCount;
        var count = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(GpuMeshSimplifier.CountImpl);
        var fill = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV>(GpuMeshSimplifier.FillImpl);
        var sortList = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(DeviceUvAtlas.SortListImpl);
        var faceK = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, FV, FV>(DeviceUvAtlas.FaceNormalAreaImpl);
        var filter = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, FV, FV, FV, float>(DeviceUvAtlas.BilateralFaceImpl);
        var update = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, IV, FV, FV>(VertexUpdateImpl);

        // the vertex -> faces table
        using var vcount = _acc.Allocate1D<int>(Math.Max(1, nv)); using var vfill = _acc.Allocate1D<int>(Math.Max(1, nv));
        using var vstart = _acc.Allocate1D<int>(nv + 1L); using var vfaces = _acc.Allocate1D<int>(Math.Max(3L, (long)nf * 3));
        vcount.MemSetToZero(); vfill.MemSetToZero();
        count(nf, faces, vcount.View);
        await _sk.ExclusiveScanAsync(vcount.View, nv, vstart.View);
        fill(nf, faces, vstart.View, vfill.View, vfaces.View);
        sortList(nv, vstart.View, vfaces.View);

        using var fa = _acc.Allocate1D<float>(Math.Max(3L, (long)nf * 3)); using var fb = _acc.Allocate1D<float>(Math.Max(3L, (long)nf * 3));
        using var area = _acc.Allocate1D<float>(Math.Max(1, nf));
        using var pa = _acc.Allocate1D<float>(Math.Max(3L, (long)nv * 3)); using var pb = _acc.Allocate1D<float>(Math.Max(3L, (long)nv * 3));
        pa.View.CopyFrom(positions.SubView(0, (long)nv * 3));
        float sg = (float)(SigmaDegrees * Math.PI / 180), inv = 1f / (sg * sg);
        var cur = pa; var other = pb;
        for (int round = 0; round < Rounds; round++)
        {
            faceK(nf, cur.View, faces, fa.View, area.View);                                  // the normals of the current surface
            var src = fa; var dst = fb;
            for (int it = 0; it < NormalPasses; it++) { filter(nf, faces, vstart.View, vfaces.View, src.View, area.View, dst.View, inv); (src, dst) = (dst, src); }
            for (int it = 0; it < VertexPasses; it++) { update(nv, cur.View, faces, vstart.View, vfaces.View, src.View, other.View); (cur, other) = (other, cur); }
        }
        positions.SubView(0, (long)nv * 3).CopyFrom(cur.View.SubView(0, (long)nv * 3));
        await _acc.SynchronizeAsync();
    }
}
