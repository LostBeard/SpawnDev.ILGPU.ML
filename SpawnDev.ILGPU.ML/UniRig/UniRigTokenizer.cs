namespace SpawnDev.ILGPU.ML.UniRig;

/// <summary>
/// The skeleton language of UniRig's autoregressive stage and the grammar that constrains it. A skeleton is a depth-first walk of the bone tree, one joint at a
/// time, each joint three coordinate tokens (the mesh is scaled into [-1,1], 256 bins per axis):
/// <code>
/// bos  cls  [part]  x y z   [branch] [part] x y z   [branch] [part] x y z  ...  eos
/// </code>
/// A joint is a child of the previous joint, unless a <c>branch</c> token precedes it: then six coordinates follow, the position of the parent joint and then the
/// new joint (the walk jumps back to a branch point). Tokens: 0..255 coordinates, 256 branch, 257 bos, 258 eos, 259 pad, 260 spring (no part), 261 body, 262 hand,
/// 263 no class, 264 vroid, 265 mixamo, 266 articulationxl; vocabulary 267. Verified against the reference tokenizer's allowed-token sets at every step of a real
/// generation and against its detokenized skeleton (tools/unirig-harness golden).
/// </summary>
public sealed class UniRigTokenizer
{
    public const int NumDiscrete = 256, Branch = 256, Bos = 257, Eos = 258, Pad = 259, Spring = 260, PartBody = 261, PartHand = 262,
        ClsNone = 263, ClsVroid = 264, ClsMixamo = 265, ClsArticulationXl = 266, VocabSize = 267;

    public enum State { ExpectBos, ExpectClsOrPartOrJoint, ExpectPartOrJoint, ExpectJoint2, ExpectJoint3, ExpectBranchOrPartOrJoint, ExpectJoint }

    private static bool IsCls(int id) => id >= ClsNone && id <= ClsArticulationXl;
    private static bool IsPart(int id) => id == Spring || id == PartBody || id == PartHand;

    /// <summary>The state after consuming <paramref name="id"/> in <paramref name="state"/>.</summary>
    public static State Advance(State state, int id)
    {
        switch (state)
        {
            case State.ExpectBos:
                if (id != Bos) throw new ArgumentException("ids do not start with bos");
                return State.ExpectClsOrPartOrJoint;
            case State.ExpectClsOrPartOrJoint:
                if (id < NumDiscrete) return State.ExpectJoint2;
                return IsCls(id) ? State.ExpectPartOrJoint : State.ExpectJoint;
            case State.ExpectPartOrJoint:
                return id < NumDiscrete ? State.ExpectJoint2 : State.ExpectPartOrJoint;
            case State.ExpectJoint2: return State.ExpectJoint3;
            case State.ExpectJoint3: return State.ExpectBranchOrPartOrJoint;
            case State.ExpectBranchOrPartOrJoint:
                return id == Branch ? State.ExpectJoint : id < NumDiscrete ? State.ExpectJoint2 : State.ExpectJoint;
            case State.ExpectJoint: return State.ExpectJoint2;
            default: throw new InvalidOperationException(state.ToString());
        }
    }

    /// <summary>The state after a whole id sequence (which must start with bos).</summary>
    public static State StateAfter(ReadOnlySpan<int> ids)
    {
        var s = State.ExpectBos;
        foreach (int id in ids) s = Advance(s, id);
        return s;
    }

    /// <summary>Length of the longest prefix of <paramref name="ids"/> that ends right after a complete joint (so a cut-off walk can be closed with eos); 0 if none.</summary>
    public static int CompleteLength(ReadOnlySpan<int> ids)
    {
        var s = State.ExpectBos; int best = 0;
        for (int i = 0; i < ids.Length; i++)
        {
            s = Advance(s, ids[i]);
            if (s == State.ExpectBranchOrPartOrJoint) best = i + 1;
        }
        return best;
    }

    /// <summary>Whether <paramref name="id"/> may come next in <paramref name="state"/>.</summary>
    public static bool IsAllowed(State state, int id)
    {
        bool joint = id < NumDiscrete;
        switch (state)
        {
            case State.ExpectBos: return id == Bos;
            case State.ExpectClsOrPartOrJoint: return joint || IsCls(id) || IsPart(id);
            case State.ExpectPartOrJoint: return joint || IsPart(id) || id == Eos;
            case State.ExpectJoint2:
            case State.ExpectJoint3:
            case State.ExpectJoint: return joint;
            case State.ExpectBranchOrPartOrJoint: return joint || IsPart(id) || id == Branch || id == Eos;
            default: return false;
        }
    }

    /// <summary>Every token allowed next in <paramref name="state"/>, ascending.</summary>
    public static List<int> Allowed(State state)
    {
        var l = new List<int>();
        for (int id = 0; id < VocabSize; id++) if (IsAllowed(state, id)) l.Add(id);
        return l;
    }

    /// <summary>A decoded skeleton in the mesh's normalised space ([-1,1]).</summary>
    /// <param name="Joints">J joint positions, xyz.</param>
    /// <param name="Parents">J parent indices, -1 for the root.</param>
    public sealed record Skeleton(float[] Joints, int[] Parents)
    {
        public int Count => Parents.Length;
    }

    /// <summary>Bin centre of coordinate token <paramref name="t"/> in [-1,1].</summary>
    public static float Undiscretize(int t) => ((t + 0.5f) / NumDiscrete) * 2f - 1f;

    /// <summary>
    /// Ids as generated: bos first, eos last (trailing pad ignored). The parent of a joint is the joint before it in the walk, or after a branch token the EARLIER
    /// joint nearest to the parent position that was spelled out (the reference's make_skeleton: nearest, latest first).
    /// </summary>
    public static Skeleton Detokenize(ReadOnlySpan<int> ids)
    {
        int end = ids.Length;
        while (end > 0 && ids[end - 1] == Pad) end--;
        if (ids.Length == 0 || ids[0] != Bos) throw new ArgumentException("first token is not bos");
        if (end < 2 || ids[end - 1] != Eos) throw new ArgumentException("last token is not eos");
        var joints = new List<float>(); var parentPos = new List<float>(); var parents = new List<int>();
        int i = 1, count = 0; bool branch = false; int last = -1;
        while (i < end - 1)
        {
            int id = ids[i];
            if (id < NumDiscrete)
            {
                float[] p;
                if (branch)
                {
                    p = new[] { Undiscretize(ids[i]), Undiscretize(ids[i + 1]), Undiscretize(ids[i + 2]) };
                    i += 3;
                }
                else p = Array.Empty<float>();
                float x = Undiscretize(ids[i]), y = Undiscretize(ids[i + 1]), z = Undiscretize(ids[i + 2]);
                i += 3;
                int parent;
                if (count == 0) parent = -1;
                else if (!branch) parent = last;
                else
                {
                    parent = -1; float best = float.MaxValue;
                    for (int j = count - 1; j >= 0; j--)                       // latest first; only a strictly nearer joint replaces the choice
                    {
                        float dx = joints[j * 3] - p[0], dy = joints[j * 3 + 1] - p[1], dz = joints[j * 3 + 2] - p[2], d = dx * dx + dy * dy + dz * dz;
                        if (d < best) { best = d; parent = j; }
                    }
                }
                joints.Add(x); joints.Add(y); joints.Add(z); parents.Add(parent);
                last = count++; branch = false;
            }
            else if (id == Branch) { branch = true; i++; }
            else if (IsPart(id) || IsCls(id)) i++;
            else throw new ArgumentException($"unexpected token {id} at {i}");
        }
        return new Skeleton(joints.ToArray(), parents.ToArray());
    }
}
