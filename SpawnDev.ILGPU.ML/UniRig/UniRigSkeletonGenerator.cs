using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.UniRig;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>
/// UniRig's skeleton stage end to end on the device: point cloud with normals -> <see cref="UniRigEncoder"/> -> OPT <see cref="UniRigDecoder"/> generating skeleton
/// tokens under the <see cref="UniRigTokenizer"/> grammar -> joints and parents. Greedy decoding (the reference's shipped inference samples 15 beams at temperature 1.5,
/// which is random; greedy is deterministic and is what the goldens pin).
/// </summary>
public sealed class UniRigSkeletonGenerator : IDisposable
{
    private readonly UniRigEncoder _encoder;
    private readonly UniRigDecoder _decoder;

    /// <param name="weights">The skeleton checkpoint (export_weights.py) with the encoder's, output_proj's and the transformer's tensors loaded.</param>
    public UniRigSkeletonGenerator(Accelerator accelerator, GpuWeights weights)
    {
        _encoder = new UniRigEncoder(accelerator, weights);
        _decoder = new UniRigDecoder(accelerator, weights);
    }

    public sealed record Result(int[] Tokens, UniRigTokenizer.Skeleton Skeleton);

    /// <summary>
    /// The skeleton of a mesh, in the mesh's own space: <see cref="UniRigMeshSampler"/> turns it into the input, the device does the rest.
    /// <paramref name="positions"/> xyz and <paramref name="faces"/> triangle indices; the joints in the result are scaled back to the mesh.
    /// </summary>
    public async Task<(UniRigTokenizer.Skeleton Skeleton, int[] Tokens)> RigMeshAsync(Accelerator accelerator, float[] positions, int[] faces, int seed = 0, int maxNewTokens = 1024)
    {
        var s = UniRigMeshSampler.Sample(positions, faces, seed);
        using var pts = accelerator.Allocate1D(s.Points); using var nrm = accelerator.Allocate1D(s.Normals);
        var r = await GenerateAsync(pts.View, nrm.View, UniRigMeshSampler.NumSamples, s.QueryRows, maxNewTokens);
        var joints = new float[r.Skeleton.Joints.Length];
        s.ToMeshSpace(r.Skeleton.Joints, joints);
        return (new UniRigTokenizer.Skeleton(joints, r.Skeleton.Parents), r.Tokens);
    }

    /// <param name="pts">[n,3] mesh points scaled into [-1,1] (see <see cref="UniRigMeshSampler"/>).</param>
    /// <param name="nrm">[n,3] unit normals.</param>
    /// <param name="queryRows">The 1024 encoder query points (<see cref="UniRigMeshSampler.QueryRows"/>).</param>
    /// <param name="cls">Dataset class token: <see cref="UniRigTokenizer.ClsArticulationXl"/> (general characters and creatures), or ClsVroid for VRoid-style avatars.</param>
    public async Task<Result> GenerateAsync(FV pts, FV nrm, int n, int[] queryRows, int maxNewTokens = 1024, int cls = UniRigTokenizer.ClsArticulationXl)
    {
        var cond = _encoder.Encode(pts, nrm, n, queryRows);
        var tokens = new List<int> { UniRigTokenizer.Bos, cls };
        var logits = await _decoder.PrefillAsync(cond.View, UniRigEncoder.QueryTokens, tokens.ToArray());
        var state = UniRigTokenizer.StateAfter(tokens.ToArray());
        for (int step = 0; step < maxNewTokens; step++)
        {
            int best = -1; float bv = float.NegativeInfinity;
            for (int v = 0; v < UniRigTokenizer.VocabSize; v++)
                if (UniRigTokenizer.IsAllowed(state, v) && logits[v] > bv) { bv = logits[v]; best = v; }
            tokens.Add(best);
            if (best == UniRigTokenizer.Eos) break;
            state = UniRigTokenizer.Advance(state, best);
            logits = await _decoder.StepAsync(best);
        }
        if (tokens[^1] != UniRigTokenizer.Eos)                                             // cut off at the limit: drop the unfinished joint and close the walk
        {
            int keep = UniRigTokenizer.CompleteLength(tokens.ToArray());
            if (keep == 0) throw new InvalidOperationException($"no complete joint within {maxNewTokens} tokens");
            tokens.RemoveRange(keep, tokens.Count - keep);
            tokens.Add(UniRigTokenizer.Eos);
        }
        var ids = tokens.ToArray();
        return new Result(ids, UniRigTokenizer.Detokenize(ids));
    }

    public void Dispose() { _encoder.Dispose(); _decoder.Dispose(); }
}
