using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.UniRig;

/// <summary>
/// Auto-rigging of a generated character: a binary glTF in, the same file with a skeleton, a skin and per-vertex weights out.
/// <code>
/// mesh      = positions + triangles of the GLB, vertices that coincide (UV seams) welded for the analysis
/// skeleton  = UniRig skeleton stage on the device (perceiver encoder + OPT decoder + grammar)
/// weights   = UniRig's skinning prior (geodesic voxel binding), mapped back to every GLB vertex
/// output    = <see cref="RiggedGlb"/> (joint nodes, skin, JOINTS_0 / WEIGHTS_0); textures and material untouched
/// </code>
/// </summary>
public sealed class UniRigRigger : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly UniRigSkeletonGenerator _generator;

    /// <summary>Voxels per side of the interior grid the skinning prior measures through.</summary>
    public int VoxelGrid { get; set; } = 196;

    public sealed record Result(byte[] Glb, UniRigTokenizer.Skeleton Skeleton);

    /// <param name="skeletonWeights">The skeleton checkpoint exported by export_weights.py (encoder, output_proj and transformer tensors).</param>
    public UniRigRigger(Accelerator accelerator, GpuWeights skeletonWeights)
    {
        _accelerator = accelerator;
        _generator = new UniRigSkeletonGenerator(accelerator, skeletonWeights);
    }

    public async Task<Result> RigGlbAsync(byte[] glb, int seed = 0, int maxNewTokens = 1024)
    {
        var mesh = RiggedGlb.ReadMesh(glb);
        int V = mesh.Positions.Length / 3;

        // weld exact duplicates (a GLB repeats a vertex along every UV seam; the skeleton and the skin only care about the surface)
        var ids = new Dictionary<(int, int, int), int>(); var map = new int[V]; var welded = new List<float>();
        for (int v = 0; v < V; v++)
        {
            var key = (BitConverter.SingleToInt32Bits(mesh.Positions[v * 3]), BitConverter.SingleToInt32Bits(mesh.Positions[v * 3 + 1]), BitConverter.SingleToInt32Bits(mesh.Positions[v * 3 + 2]));
            if (!ids.TryGetValue(key, out int id)) { ids[key] = id = welded.Count / 3; welded.Add(mesh.Positions[v * 3]); welded.Add(mesh.Positions[v * 3 + 1]); welded.Add(mesh.Positions[v * 3 + 2]); }
            map[v] = id;
        }
        var faces = new List<int>();
        for (int t = 0; t < mesh.Indices.Length; t += 3)
        {
            int a = map[mesh.Indices[t]], b = map[mesh.Indices[t + 1]], c = map[mesh.Indices[t + 2]];
            if (a != b && b != c && c != a) { faces.Add(a); faces.Add(b); faces.Add(c); }
        }
        var wp = welded.ToArray(); var wf = faces.ToArray();

        var (skeleton, _) = await _generator.RigMeshAsync(_accelerator, wp, wf, seed, maxNewTokens);

        // the prior works in the mesh scaled into [-1,1] (centre of the bounding box, longest side = 2), like everything UniRig does
        int nw = wp.Length / 3, J = skeleton.Count;
        float[] lo = { float.MaxValue, float.MaxValue, float.MaxValue }, hi = { float.MinValue, float.MinValue, float.MinValue };
        for (int v = 0; v < nw; v++) for (int k = 0; k < 3; k++) { lo[k] = Math.Min(lo[k], wp[v * 3 + k]); hi[k] = Math.Max(hi[k], wp[v * 3 + k]); }
        float scale = Math.Max(hi[0] - lo[0], Math.Max(hi[1] - lo[1], hi[2] - lo[2])) / 2;
        var vn = new float[wp.Length]; var jn = new float[J * 3];
        for (int v = 0; v < nw; v++) for (int k = 0; k < 3; k++) vn[v * 3 + k] = (wp[v * 3 + k] - (lo[k] + hi[k]) / 2) / scale;
        for (int j = 0; j < J; j++) for (int k = 0; k < 3; k++) jn[j * 3 + k] = (skeleton.Joints[j * 3 + k] - (lo[k] + hi[k]) / 2) / scale;
        var voxels = UniRigVoxelSkin.Voxelize(vn, wf, VoxelGrid);
        var skin = UniRigVoxelSkin.Compute(vn, wf, jn, voxels);

        var weights = new float[V * J];
        for (int v = 0; v < V; v++) Array.Copy(skin, map[v] * J, weights, v * J, J);
        return new Result(RiggedGlb.Add(glb, skeleton, weights), skeleton);
    }

    public void Dispose() => _generator.Dispose();
}
