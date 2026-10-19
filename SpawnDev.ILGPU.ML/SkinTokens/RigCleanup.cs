namespace SpawnDev.ILGPU.ML.SkinTokens;

/// <summary>
/// Repairs the one defect learned skinning leaves on generated characters: <b>bridges</b>. A model whose hand rests on its hip is one connected surface, so the skin weights near the contact blend a hand
/// bone with a hip bone, and the moment the arm swings the surface between them stretches into a long dark sliver. Two fixes, both measured in <i>hops</i> (edges of the skeleton tree between two bones):
/// influences from bones <see cref="SnapFarInfluences"/> hops or more from the bone a vertex mostly follows are dropped (the vertex follows its own part only), and triangles whose three vertices mostly follow
/// bones that far apart are removed (<see cref="BridgeTriangles"/>): they join two parts that move independently, so there is nothing to keep. Ordinary joints stay untouched: a shoulder blends bones 1-2 hops
/// apart, the far ends of a limb never share a surface.
/// </summary>
public static class RigCleanup
{
    /// <summary>Hops between every pair of bones, [J * J]: the length of the path between them in the tree.</summary>
    public static int[] HopMatrix(int[] parents)
    {
        int J = parents.Length; var depth = new int[J];
        for (int j = 0; j < J; j++) depth[j] = parents[j] < 0 ? 0 : depth[parents[j]] + 1;       // parents precede children
        var hops = new int[J * J];
        for (int a = 0; a < J; a++)
            for (int b = a + 1; b < J; b++)
            {
                int x = a, y = b, h = 0;
                while (x != y)
                {
                    if (x < 0 || y < 0) { h = int.MaxValue / 2; break; }                         // separate trees
                    if (depth[x] >= depth[y]) x = parents[x]; else y = parents[y];
                    h++;
                }
                hops[a * J + b] = hops[b * J + a] = h;
            }
        return hops;
    }

    /// <summary>The bone each vertex follows most, from [V * J] weights.</summary>
    public static int[] DominantBones(float[] weights, int V, int J)
    {
        var dom = new int[V];
        for (int v = 0; v < V; v++)
        {
            int best = 0; float bw = float.MinValue;
            for (int j = 0; j < J; j++) if (weights[v * J + j] > bw) { bw = weights[v * J + j]; best = j; }
            dom[v] = best;
        }
        return dom;
    }

    /// <summary>
    /// Zero the weight a vertex puts on any bone <paramref name="farHops"/> or more from its dominant bone and renormalise the row. Returns how many vertices lost more than 2% of their weight this way. <paramref name="farHops"/> 0 does nothing.
    /// </summary>
    public static int SnapFarInfluences(float[] weights, int V, int J, int[] hops, int farHops)
    {
        if (farHops <= 0) return 0;
        var dom = DominantBones(weights, V, J);
        int changed = 0;
        for (int v = 0; v < V; v++)
        {
            float removed = 0, sum = 0;
            for (int j = 0; j < J; j++)
            {
                if (hops[dom[v] * J + j] >= farHops && weights[v * J + j] > 0) { removed += weights[v * J + j]; weights[v * J + j] = 0; }
                sum += weights[v * J + j];
            }
            if (removed <= 0) continue;
            if (removed > 0.02f) changed++;                                                  // the learned weights are dense (every bone has a sliver of weight everywhere): count only real blends
            float inv = 1f / Math.Max(sum, 1e-8f);
            for (int j = 0; j < J; j++) weights[v * J + j] *= inv;
        }
        return changed;
    }

    /// <summary>Which triangles (three indices each) join vertices whose dominant bones are more than <paramref name="farHops"/> apart. <paramref name="farHops"/> 0 finds none.</summary>
    public static bool[] BridgeTriangles(int[] indices, int[] dominant, int J, int[] hops, int farHops)
    {
        var bridge = new bool[indices.Length / 3];
        if (farHops <= 0) return bridge;
        for (int t = 0; t < bridge.Length; t++)
        {
            int a = dominant[indices[t * 3]], b = dominant[indices[t * 3 + 1]], c = dominant[indices[t * 3 + 2]];
            bridge[t] = Math.Max(hops[a * J + b], Math.Max(hops[b * J + c], hops[a * J + c])) > farHops;
        }
        return bridge;
    }
}
