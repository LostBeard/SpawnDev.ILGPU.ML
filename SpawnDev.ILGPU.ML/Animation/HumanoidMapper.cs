namespace SpawnDev.ILGPU.ML.Animation;

/// <summary>The rig is not a standing humanoid the mapper can find a spine, two legs, a chest with two arms and a neck in.</summary>
public sealed class NotAHumanoidException(string message) : InvalidOperationException(message);

/// <summary>Which SOMA joint each bone of a rig plays, and the way the character faces.</summary>
/// <param name="Soma">[bone] index into <see cref="SomaSkeleton"/>. Bones the body does not include (hair, a cape, a tail, a weapon) follow the nearest body bone above them.</param>
/// <param name="Frame">Columns are the character's (left, up, forward) in the rig's world axes.</param>
public sealed record HumanoidMapping(int[] Soma, Mat3 Frame);

/// <summary>
/// Finds the body of a generated humanoid by the SHAPE of its bone tree and names its bones after SOMA's joints, so a clip written for SOMA can drive any skeleton SkinTokens
/// invents. Nothing depends on bone numbers or one character's measurements:
/// <list type="bullet">
/// <item><description>under the root, the child reaching highest is the spine and the two reaching lowest are the legs;</description></item>
/// <item><description>facing comes from the legs: the knee bends forward of the hip-ankle line, or foot to toe points forward;</description></item>
/// <item><description>walking up the spine, the first bone that forks into a neck and two arms is the chest (arms are the two children reaching furthest to each side and
/// must have a shoulder, upper arm and forearm; arms that reach the floor are front legs and refuse a four-legged body);</description></item>
/// <item><description>an arm is its first four bones, a hand with five finger chains maps thumb, index, middle, ring and pinky by position, any other hand moves its fingers with
/// SOMA's middle finger; chains shorter or longer than SOMA's are stretched or squeezed onto its joints.</description></item>
/// </list>
/// The approach is that of image-to-3dlab's scripts/kimodo_retarget.py (Apache-2.0); this is an independent C# implementation checked against that script's numbers
/// (tools/unirig-harness/reference/golden_retarget.py).
/// </summary>
public static class HumanoidMapper
{
    private static readonly string[] Fingers = { "Index", "Middle", "Ring", "Pinky" };

    public static HumanoidMapping Map(int[] parents, Vec3[] heads, Vec3 up)
    {
        int n = parents.Length;
        if (heads.Length != n) throw new ArgumentException("one head per bone");
        var kids = new List<int>[n]; for (int i = 0; i < n; i++) kids[i] = new List<int>();
        int root = -1;
        for (int i = 0; i < n; i++) { if (parents[i] < 0) { if (root < 0) root = i; } else kids[parents[i]].Add(i); }
        if (root < 0) throw new NotAHumanoidException("the skeleton has no root");
        double H(int b) => Vec3.Dot(heads[b], up);

        double High(int b) { double v = H(b); foreach (int c in kids[b]) v = Math.Max(v, High(c)); return v; }
        double Low(int b) { double v = H(b); foreach (int c in kids[b]) v = Math.Min(v, Low(c)); return v; }
        List<int> Subtree(int b) { var o = new List<int> { b }; foreach (int c in kids[b]) o.AddRange(Subtree(c)); return o; }
        List<int> Chain(int start) { var o = new List<int> { start }; while (kids[o[^1]].Count == 1) o.Add(kids[o[^1]][0]); return o; }
        List<int> Limb(int start, Func<int, double> score, int length)
        {
            var o = new List<int> { start };
            while (kids[o[^1]].Count > 0)
            {
                var options = kids[o[^1]];
                if (options.Count == 1) o.Add(options[0]);
                else if (o.Count < length) o.Add(options.MaxBy(score));
                else break;
            }
            return o;
        }
        Vec3 Flat(Vec3 v) => v - up * Vec3.Dot(v, up);

        double rootH = H(root);
        var high = kids[root].ToDictionary(c => c, High); var low = kids[root].ToDictionary(c => c, Low);
        int spineStart = kids[root].Count == 0 ? -1 : kids[root].MaxBy(c => high[c]);
        var legs = kids[root].Where(c => c != spineStart && low[c] < rootH - 1e-6).OrderBy(c => low[c]).Take(2).ToList();
        if (spineStart < 0 || legs.Count != 2 || high[spineStart] <= rootH)
            throw new NotAHumanoidException($"expected 2 legs + 1 spine under the root, found legs [{string.Join(", ", legs)}] and spine {spineStart}");
        var legChains = legs.Select(c => Limb(c, x => -Low(x), 4)).ToList();
        if (legChains.Min(ch => ch.Count) < 2) throw new NotAHumanoidException("the legs are too short to find the feet");

        // facing: knee bend (three-bone legs) or foot -> toe (longer legs)
        var votes = new List<Vec3>();
        foreach (var ch in legChains)
        {
            if (ch.Count == 3)
            {
                Vec3 hip = heads[ch[0]], knee = heads[ch[1]], ankle = heads[ch[2]];
                var bend = Flat(knee - (hip + ankle) / 2);
                if (bend.Length >= 0.01 * (hip - ankle).Length) { votes.Add(bend); continue; }
            }
            votes.Add(heads[ch[^1]] - heads[ch[^2]]);
        }
        var fwd = Flat(new Vec3(votes.Average(v => v.X), votes.Average(v => v.Y), votes.Average(v => v.Z)));
        if (fwd.Length < 1e-12) throw new NotAHumanoidException("cannot tell which way the legs face");
        fwd = fwd.Normalized();
        var leftAxis = Vec3.Cross(up, fwd);
        var frame = Mat3.FromColumns(leftAxis, up, fwd);

        double floor = legs.Min(c => low[c]), height = high.Values.Max() - floor;
        var walkers = kids[root].Where(c => c != spineStart && low[c] <= floor + 0.03 * height && Subtree(c).Count >= 3).ToList();
        if (walkers.Count > 2) throw new NotAHumanoidException($"expected 2 legs + 1 spine under the root, found {walkers.Count} chains on the floor");
        double knee2 = floor + 0.25 * (rootH - floor), reach = 0.05 * height;

        // up the spine to the first bone that forks into a neck and two arms
        var path = new List<int> { spineStart }; int b0 = spineStart; int neck = -1; int[] arms = Array.Empty<int>();
        while (true)
        {
            if (kids[b0].Count >= 3)
            {
                double SideReach(int c, int sign) => Subtree(c).Max(x => sign * Vec3.Dot(heads[x] - heads[b0], leftAxis));
                int left = kids[b0].MaxBy(c => SideReach(c, 1)), right = kids[b0].MaxBy(c => SideReach(c, -1));
                var rest = kids[b0].Where(c => c != left && c != right).ToList();
                bool wide = left != right && SideReach(left, 1) > reach && SideReach(right, -1) > reach;
                if (wide && Math.Max(Low(left), Low(right)) <= knee2) throw new NotAHumanoidException($"bone {b0} stands on front legs {left}, {right}: a four-legged body");
                bool armsOk = wide && Math.Min(Subtree(left).Count, Subtree(right).Count) >= 3;
                if (armsOk && rest.Count > 0)
                {
                    int cand = rest.MaxBy(High);
                    if (High(cand) > H(b0)) { neck = cand; arms = new[] { left, right }; break; }
                }
            }
            if (kids[b0].Count == 0) throw new NotAHumanoidException($"no bone above the hips forks into a neck and two arms (walked [{string.Join(", ", path)}])");
            b0 = kids[b0].MaxBy(High); path.Add(b0);
        }
        var spine = new List<int> { root }; spine.AddRange(path);
        int chest = spine[^1];

        var m = new int[n]; Array.Fill(m, -1);
        void Spread(List<int> chain, params string[] names)
        {
            if (chain.Count == 0) return;
            if (chain.Count == 1) { m[chain[0]] = SomaSkeleton.IndexOf(names[^1]); return; }
            for (int i = 0; i < chain.Count; i++)                                    // numpy linspace + round (half to even)
            {
                int idx = (int)Math.Round((double)i * (names.Length - 1) / (chain.Count - 1), MidpointRounding.ToEven);
                m[chain[i]] = SomaSkeleton.IndexOf(names[idx]);
            }
        }
        Spread(spine, "Hips", "Spine1", "Spine2", "Chest");
        var neckChain = new List<int> { neck };
        while (kids[neckChain[^1]].Count == 1 && H(kids[neckChain[^1]][0]) > H(neckChain[^1])) neckChain.Add(kids[neckChain[^1]][0]);
        Spread(neckChain.Take(neckChain.Count - 1).ToList(), "Neck1", "Neck2");
        m[neckChain[^1]] = SomaSkeleton.IndexOf("Head");

        foreach (int arm in arms)
        {
            string lr = Vec3.Dot(heads[arm] - heads[chest], leftAxis) > 0 ? "Left" : "Right";
            var ch = Limb(arm, x => Math.Abs(Vec3.Dot(heads[x] - heads[chest], leftAxis)), 4);
            var pastHand = ch.Skip(4).ToList(); ch = ch.Take(4).ToList();           // a hand with one finger does not fork: anything past the 4th bone is that finger
            Spread(ch, $"{lr}Shoulder", $"{lr}Arm", $"{lr}ForeArm", $"{lr}Hand");
            int hand = ch[^1];
            var fingers = pastHand.Count > 0 ? new List<List<int>> { pastHand } : kids[hand].Select(Chain).ToList();
            if (fingers.Count != 5)
            {
                foreach (var f in fingers) Spread(f, $"{lr}HandMiddle2", $"{lr}HandMiddle3", $"{lr}HandMiddle4");
                continue;
            }
            var thumb = fingers.MinBy(f => (heads[f[0]] - heads[hand]).Length)!;
            var others = fingers.Where(f => !ReferenceEquals(f, thumb)).OrderBy(f => -Vec3.Dot(heads[f[0]], fwd)).ToList();     // front to back
            Spread(thumb, $"{lr}HandThumb1", $"{lr}HandThumb2", $"{lr}HandThumb3");
            for (int i = 0; i < 4; i++) Spread(others[i], $"{lr}Hand{Fingers[i]}2", $"{lr}Hand{Fingers[i]}3", $"{lr}Hand{Fingers[i]}4");
        }
        for (int i = 0; i < 2; i++)
        {
            string lr = Vec3.Dot(heads[legs[i]] - heads[root], leftAxis) > 0 ? "Left" : "Right";
            Spread(legChains[i], $"{lr}Leg", $"{lr}Shin", $"{lr}Foot", $"{lr}ToeBase");
        }
        var queue = new Queue<int>(); queue.Enqueue(root);                           // extras follow the nearest mapped bone above them (parents first)
        while (queue.Count > 0)
        {
            int b = queue.Dequeue();
            if (m[b] < 0) m[b] = m[parents[b]];
            foreach (int c in kids[b]) queue.Enqueue(c);
        }
        return new HumanoidMapping(m, frame);
    }
}
