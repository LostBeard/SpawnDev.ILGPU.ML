using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// Mesh extraction from the shape decoder's output (o_voxel.convert.flexible_dual_grid_to_mesh with the post-processing of
/// FlexiDualGridVaeDecoder.forward, inference branch), on the device. Every output voxel carries 7 channels:
/// 3 dual-vertex offsets (vertex = (1 + 2m) sigmoid(h) - m, m = 0.5, inside the voxel plus a margin), 3 edge-intersection flags
/// (h &gt; 0) and a split weight (softplus). An intersected edge along axis a owns the quad of the 4 voxels around it; when all four
/// exist the quad becomes two triangles, split along the diagonal with the larger weight product. Vertex k of the mesh is voxel k.
/// Triangle order = voxel-major, axis-minor (the reference's boolean-mask order), so results compare element for element.
/// </summary>
public sealed class DualGridMesher
{
    private readonly Accelerator _acc;
    private readonly SparseKernels _sk;
    public DualGridMesher(Accelerator accelerator, SparseKernels sparse) { _acc = accelerator; _sk = sparse; }

    /// <summary>Vertices [VertexCount, 3] and triangles [FaceCount, 3] on the device.</summary>
    public sealed class Mesh : IDisposable
    {
        public MemoryBuffer1D<float, Stride1D.Dense> Vertices = null!;
        public MemoryBuffer1D<int, Stride1D.Dense> Faces = null!;
        public int VertexCount, FaceCount;
        public void Dispose() { Vertices?.Dispose(); Faces?.Dispose(); }
    }

    private const float Margin = 0.5f;

    private Action<Index1D, FV, IV, FV, int>? _vertices;
    private static void VerticesImpl(Index1D i, FV h, IV coords, FV verts, int grid)
    {
        float voxel = 1f / grid;
        for (int c = 0; c < 3; c++)
        {
            float s = 1f / (1f + MathF.Exp(-h[i * 7 + c]));
            float v = (1f + 2f * Margin) * s - Margin;
            verts[i * 3 + c] = (coords[i * 3 + c] + v) * voxel + -0.5f;
        }
    }

    private Action<Index1D, FV, IV, IV, IV, IV, IV, int, int, int>? _quads;
    private static void QuadsImpl(Index1D idx, FV h, IV coords, IV keys, IV vals, IV quad, IV flag, int grid, int mask, int shift)
    {
        int i = idx / 3, a = idx % 3;
        flag[idx] = 0;
        if (h[i * 7 + 3 + a] <= 0f) return;
        int x = coords[i * 3], y = coords[i * 3 + 1], z = coords[i * 3 + 2];
        // the four voxels around the edge, in the reference's order (edge_neighbor_voxel_offset)
        int d1x = 0, d1y = 0, d1z = 0, d2x = 0, d2y = 0, d2z = 0, d3x = 0, d3y = 0, d3z = 0;
        if (a == 0) { d1z = 1; d2y = 1; d2z = 1; d3y = 1; }
        else if (a == 1) { d1x = 1; d2x = 1; d2z = 1; d3z = 1; }
        else { d1y = 1; d2x = 1; d2y = 1; d3x = 1; }
        int q0 = SparseKernels.Lookup(keys, vals, x, y, z, grid, mask, shift);
        int q1 = SparseKernels.Lookup(keys, vals, x + d1x, y + d1y, z + d1z, grid, mask, shift);
        int q2 = SparseKernels.Lookup(keys, vals, x + d2x, y + d2y, z + d2z, grid, mask, shift);
        int q3 = SparseKernels.Lookup(keys, vals, x + d3x, y + d3y, z + d3z, grid, mask, shift);
        if (q0 < 0 || q1 < 0 || q2 < 0 || q3 < 0) return;
        quad[idx * 4] = q0; quad[idx * 4 + 1] = q1; quad[idx * 4 + 2] = q2; quad[idx * 4 + 3] = q3;
        flag[idx] = 1;
    }

    private Action<Index1D, FV, IV, IV, IV, IV>? _tris;
    private static float Softplus(float x) => x > 20f ? x : MathF.Log(1f + MathF.Exp(x));
    private static void TrisImpl(Index1D idx, FV h, IV quad, IV flag, IV offsets, IV faces)
    {
        if (flag[idx] == 0) return;
        int q0 = quad[idx * 4], q1 = quad[idx * 4 + 1], q2 = quad[idx * 4 + 2], q3 = quad[idx * 4 + 3];
        float w02 = Softplus(h[q0 * 7 + 6]) * Softplus(h[q2 * 7 + 6]);
        float w13 = Softplus(h[q1 * 7 + 6]) * Softplus(h[q3 * 7 + 6]);
        int o = offsets[idx] * 6;
        if (w02 > w13) { faces[o] = q0; faces[o + 1] = q1; faces[o + 2] = q2; faces[o + 3] = q0; faces[o + 4] = q2; faces[o + 5] = q3; }
        else { faces[o] = q0; faces[o + 1] = q1; faces[o + 2] = q3; faces[o + 3] = q3; faces[o + 4] = q1; faces[o + 5] = q2; }
    }

    /// <summary>
    /// <paramref name="h"/> is the decoder output [n, 7]; <paramref name="coords"/> [n, 3] voxel coordinates on a
    /// <paramref name="grid"/>^3 lattice covering [-0.5, 0.5]^3. Caller disposes the returned mesh.
    /// </summary>
    public Mesh Extract(FV h, IV coords, int n, int grid) => Task.Run(() => ExtractAsync(h, coords, n, grid)).GetAwaiter().GetResult();

    /// <summary><see cref="Extract"/>, browser-safe (the quad count is an awaited readback).</summary>
    public async Task<Mesh> ExtractAsync(FV h, IV coords, int n, int grid)
    {
        _vertices ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, FV, int>(VerticesImpl);
        _quads ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, IV, IV, IV, int, int, int>(QuadsImpl);
        _tris ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, IV, IV>(TrisImpl);
        var mesh = new Mesh { VertexCount = n };
        mesh.Vertices = _acc.Allocate1D<float>((long)n * 3);
        _vertices(n, h, coords, mesh.Vertices.View, grid);
        using var hash = _sk.BuildHash(coords, n, grid);
        using var quad = _acc.Allocate1D<int>((long)n * 3 * 4);
        using var flag = _acc.Allocate1D<int>((long)n * 3);
        _quads(n * 3, h, coords, hash.Keys.View, hash.Vals.View, quad.View, flag.View, grid, hash.Mask, hash.Shift);
        using var offsets = _acc.Allocate1D<int>((long)n * 3 + 1);
        int quads = await _sk.ExclusiveScanAsync(flag.View, n * 3, offsets.View);
        mesh.FaceCount = quads * 2;
        mesh.Faces = _acc.Allocate1D<int>(Math.Max(1L, (long)quads * 6));
        _tris(n * 3, h, quad.View, flag.View, offsets.View, mesh.Faces.View);
        _acc.Fence();
        return mesh;
    }
}
