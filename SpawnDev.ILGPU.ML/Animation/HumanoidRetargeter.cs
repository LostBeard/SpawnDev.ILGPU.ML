namespace SpawnDev.ILGPU.ML.Animation;

/// <summary>A clip played on a rig: the local rotation of every bone and the root's position at each frame, ready for a glTF animation or an engine's skeleton.</summary>
/// <param name="Times">Seconds of each frame.</param>
/// <param name="Rotations">[frames * bones * 4] local rotation quaternions (x, y, z, w) relative to the parent bone; the rig's bones have identity rest rotations.</param>
/// <param name="RootTranslation">[frames * 3] the root bone's translation (its rest position plus the hips' travel, scaled to the rig).</param>
public sealed record RetargetedClip(string Name, int Frames, int Bones, float[] Times, float[] Rotations, float[] RootTranslation)
{
    public double Seconds => Times.Length == 0 ? 0 : Times[^1];
}

/// <summary>
/// Plays a Kimodo SOMA <see cref="MotionClip"/> on a SkinTokens-rigged humanoid. A rig's bones are plain joints (identity rest rotation, <see cref="RiggedGlb"/>), so with the
/// bone-to-SOMA mapping of <see cref="HumanoidMapper"/> each bone's WORLD rotation is SOMA's world turn of its joint (re-expressed in the rig's axes) applied on top of
/// the rig's rest, after <i>rest alignment</i>: generated characters rarely stand in a clean T-pose (legs splay, arms droop), and copying SOMA's turns over that error keeps it in
/// every frame, so each limb bone with one child is first swung to point where SOMA's matching bone points in its T-pose (forks, spine, neck and head keep their rest; a hand
/// turns with its forearm; leaves turn with their parent). Because the turn is world-space, joints SOMA has and the rig lacks (neck 2, metacarpals) are absorbed. The hips' travel
/// is scaled by the ratio of hip heights so a short character does not moonwalk.
/// Approach and conventions are those of image-to-3dlab's scripts/kimodo_retarget.py (Apache-2.0); independent C#, gated against that script's numbers.
/// </summary>
public sealed class HumanoidRetargeter
{
    private readonly int[] _parents;
    private readonly Vec3[] _heads;
    private readonly Vec3 _up;

    /// <summary>Bone to SOMA joint and the character's facing.</summary>
    public HumanoidMapping Mapping { get; }

    /// <param name="parents">Parent of each bone (-1 for the root; parents precede children).</param>
    /// <param name="heads">Rest position of each bone's joint in the rig's world space.</param>
    /// <param name="up">The rig's up axis (default +Y).</param>
    /// <exception cref="NotAHumanoidException">The skeleton is not a standing humanoid.</exception>
    public HumanoidRetargeter(int[] parents, Vec3[] heads, Vec3? up = null)
    {
        _parents = parents; _heads = heads; _up = up ?? new Vec3(0, 1, 0);
        Mapping = HumanoidMapper.Map(parents, heads, _up);
    }

    private static double SomaHipHeight(Vec3[] joints0)
    {
        double feet = double.MaxValue;
        foreach (var n in new[] { "LeftToeBase", "RightToeBase", "LeftFoot", "RightFoot" }) feet = Math.Min(feet, joints0[SomaSkeleton.IndexOf(n)].Y);
        return joints0[0].Y - feet;
    }

    /// <summary>SOMA's standard T-pose joint positions (hips at the origin), recovered from frame 0: x_j = x_parent + G_parent^T (p_j - p_parent).</summary>
    private static Vec3[] SomaTpose(MotionClip clip)
    {
        var n = new Vec3[SomaSkeleton.Count];
        for (int j = 0; j < n.Length; j++)
        {
            int p = SomaSkeleton.Parents[j];
            if (p >= 0) n[j] = n[p] + clip.Rotation(0, p).Transpose().Mul(clip.Joints0[j] - clip.Joints0[p]);
        }
        return n;
    }

    private Mat3[] RestAlignment(Mat3 axes, Vec3[] tpose)
    {
        int n = _parents.Length; var kids = new List<int>[n]; for (int i = 0; i < n; i++) kids[i] = new List<int>();
        for (int i = 0; i < n; i++) if (_parents[i] >= 0) kids[_parents[i]].Add(i);
        var soma = Mapping.Soma; var align = new Mat3[n];
        bool Limb(int b) { var s = SomaSkeleton.Names[soma[b]]; return s.StartsWith("Left", StringComparison.Ordinal) || s.StartsWith("Right", StringComparison.Ordinal); }
        for (int b = 0; b < n; b++)
        {
            align[b] = Mat3.Identity;
            if (kids[b].Count != 1 || !Limb(b)) continue;                            // only limbs: a spine's curve and a neck's angle are the character's design
            int c = kids[b][0];
            var want = axes.Mul(tpose[soma[c]] - tpose[soma[b]]); var have = _heads[c] - _heads[b];
            if (want.Length > 1e-6 && have.Length > 1e-6) align[b] = Mat3.Between(have, want);
        }
        for (int b = 0; b < n; b++)                                                  // a hand forks into fingers: it has no direction of its own, it turns with its forearm
            if (kids[b].Count > 1 && _parents[b] >= 0 && SomaSkeleton.Names[soma[b]].EndsWith("Hand", StringComparison.Ordinal)) align[b] = align[_parents[b]];
        for (int b = 0; b < n; b++)                                                  // a leaf turns with its parent (parents precede children, so the parent is settled)
            if (kids[b].Count == 0 && _parents[b] >= 0) align[b] = align[_parents[b]];
        return align;
    }

    /// <param name="clip">A SOMA clip.</param>
    /// <param name="speed">Play faster (&gt; 1) or slower (&lt; 1); strikes gain snap at 1.25.</param>
    /// <param name="armClearanceDegrees">Swing both upper arms this far away from the body (about the torso's forward axis, whatever the arm is doing). Generated characters are often bulky (armour,
    /// a coat, a cape) and Kimodo's arms hang for a slim body, so a few degrees keep the arms out of the torso; 0 plays the clip as it is.</param>
    public RetargetedClip Retarget(MotionClip clip, double speed = 1.0, double armClearanceDegrees = 0)
    {
        if (speed <= 0) throw new ArgumentOutOfRangeException(nameof(speed));
        int J = _parents.Length, T = clip.Frames; int root = Array.IndexOf(_parents, -1);
        var axes = Mapping.Frame;                                                    // SOMA's frame is the identity (left, up, forward), so axes = the rig's frame
        double lowest = double.MaxValue; foreach (var h in _heads) lowest = Math.Min(lowest, Vec3.Dot(h, _up));
        double scale = (Vec3.Dot(_heads[root], _up) - lowest) / SomaHipHeight(clip.Joints0);
        var align = RestAlignment(axes, SomaTpose(clip)); var axesT = axes.Transpose();

        // the upper arms' extra swing: left arm +phi, right arm -phi about the forward axis, in the parent's frame (the torso), so it follows the body however it turns
        int leftArm = SomaSkeleton.IndexOf("LeftArm"), rightArm = SomaSkeleton.IndexOf("RightArm");
        double phi = armClearanceDegrees * Math.PI / 180;
        Mat3 Swing(double a) { double c = Math.Cos(a), s = Math.Sin(a); return axes * new Mat3(c, -s, 0, s, c, 0, 0, 0, 1) * axesT; }
        var swingLeft = Swing(phi); var swingRight = Swing(-phi);

        var rot = new float[T * J * 4]; var trans = new float[T * 3]; var times = new float[T];
        var world = new Mat3[J]; var prev = new (double X, double Y, double Z, double W)[J];
        for (int t = 0; t < T; t++)
        {
            for (int b = 0; b < J; b++) world[b] = axes * clip.Rotation(t, Mapping.Soma[b]) * axesT * align[b];
            for (int b = 0; b < J; b++)
            {
                var local = _parents[b] < 0 ? world[b] : world[_parents[b]].Transpose() * world[b];
                if (phi != 0 && _parents[b] >= 0) { int sj = Mapping.Soma[b]; if (sj == leftArm) local = swingLeft * local; else if (sj == rightArm) local = swingRight * local; }
                var q = local.ToQuaternion();
                if (t > 0 && q.X * prev[b].X + q.Y * prev[b].Y + q.Z * prev[b].Z + q.W * prev[b].W < 0) q = (-q.X, -q.Y, -q.Z, -q.W);   // the short way round between frames
                prev[b] = q;
                int o = (t * J + b) * 4; rot[o] = (float)q.X; rot[o + 1] = (float)q.Y; rot[o + 2] = (float)q.Z; rot[o + 3] = (float)q.W;
            }
            var travel = axes.Mul(clip.Hips[t] - clip.Hips[0]) * scale;
            var p = _heads[root] + travel;
            trans[t * 3] = (float)p.X; trans[t * 3 + 1] = (float)p.Y; trans[t * 3 + 2] = (float)p.Z;
            times[t] = (float)(t / (clip.Fps * speed));
        }
        return new RetargetedClip(clip.Name, T, J, times, rot, trans);
    }
}
