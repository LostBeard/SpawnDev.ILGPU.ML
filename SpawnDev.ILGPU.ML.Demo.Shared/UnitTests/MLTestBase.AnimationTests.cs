using System.Text.Json;
using SpawnDev.ILGPU.ML.Animation;
using SpawnDev.ILGPU.ML.UniRig;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Humanoid animation: Kimodo SOMA clips played on a SkinTokens-rigged character (<see cref="HumanoidMapper"/>, <see cref="HumanoidRetargeter"/>, <see cref="AnimatedGlb"/>).
/// The semantic tests run on a stick figure built in code; the reference tests compare with the numbers of image-to-3dlab's kimodo_retarget.py on a real SkinTokens rig
/// (tools/unirig-harness/reference/golden_retarget.py; skipped when ANIM_DIR / E:/spawndev-unirig/anim is absent).
/// </summary>
public abstract partial class MLTestBase
{
    // ───────────────────────────── a stick figure: 22 bones, Y up, facing +Z, character-left = +X ─────────────────────────────

    private sealed record Anim_Rig(int[] Parents, Vec3[] Heads, Dictionary<string, int> By);

    /// <param name="legs">number of legs under the hips (2 for a person, 4 for a creature)</param>
    private static Anim_Rig Anim_StickFigure(int legs = 2, bool splayed = false)
    {
        var parents = new List<int>(); var heads = new List<Vec3>(); var by = new Dictionary<string, int>();
        int Add(string name, int parent, Vec3 head) { parents.Add(parent); heads.Add(head); by[name] = parents.Count - 1; return parents.Count - 1; }
        int hips = Add("hips", -1, new Vec3(0, 1.0, 0));
        for (int i = 0; i < legs; i++)                                               // legs: leg, knee, ankle, toe (the knee a little forward)
        {
            double x = legs == 2 ? (i == 0 ? 0.1 : -0.1) : (i % 2 == 0 ? 0.1 : -0.1), z = legs == 2 ? 0 : (i < 2 ? 0.3 : -0.3), spl = splayed ? 0.05 : 0;
            int leg = Add($"leg{i}", hips, new Vec3(x, 0.95, z)); int knee = Add($"knee{i}", leg, new Vec3(x + (x > 0 ? spl : -spl), 0.5, z + 0.03));
            int ankle = Add($"ankle{i}", knee, new Vec3(x + (x > 0 ? 2 * spl : -2 * spl), 0.08, z)); Add($"toe{i}", ankle, new Vec3(x + (x > 0 ? 2 * spl : -2 * spl), 0.02, z + 0.12));
        }
        int spine1 = Add("spine1", hips, new Vec3(0, 1.1, 0)); int spine2 = Add("spine2", spine1, new Vec3(0, 1.25, 0)); int chest = Add("chest", spine2, new Vec3(0, 1.4, 0));
        Add("cape", chest, new Vec3(0, 1.3, -0.1)); int neck = Add("neck", chest, new Vec3(0, 1.5, 0)); int head = Add("head", neck, new Vec3(0, 1.6, 0)); Add("headTop", head, new Vec3(0, 1.8, 0));
        foreach (var (side, sx) in new[] { ("L", 1.0), ("R", -1.0) })                // arms: shoulder, upper arm, forearm, hand
        {
            int sh = Add($"shoulder{side}", chest, new Vec3(sx * 0.1, 1.45, 0)); int arm = Add($"arm{side}", sh, new Vec3(sx * 0.25, 1.45, 0));
            int fore = Add($"fore{side}", arm, new Vec3(sx * 0.55, 1.45, 0)); Add($"hand{side}", fore, new Vec3(sx * 0.8, 1.45, 0)); Add($"handTip{side}", by[$"hand{side}"], new Vec3(sx * 0.9, 1.45, 0));
        }
        return new Anim_Rig(parents.ToArray(), heads.ToArray(), by);
    }

    private static MotionClip Anim_ClipWith(int frames, Func<int, string, Mat3> rotation)
    {
        // SOMA's T-pose: hips at 1.0 with the standard layout; only frame-0 joints and hips matter for scale and rest alignment
        var tpose = new Vec3[SomaSkeleton.Count];
        for (int j = 1; j < tpose.Length; j++)
        {
            string n = SomaSkeleton.Names[j]; double side = n.StartsWith("Left") ? 1 : n.StartsWith("Right") ? -1 : 0;
            int p = SomaSkeleton.Parents[j]; var offset = new Vec3(0, 0.1, 0);
            if (n.Contains("Arm") || n.Contains("Hand") || n.Contains("Shoulder")) offset = new Vec3(side * 0.12, 0, 0);
            if (n.EndsWith("Leg")) offset = new Vec3(side * 0.1, -0.05, 0);
            if (n.EndsWith("Shin") || n.EndsWith("Foot")) offset = new Vec3(0, -0.45, 0);
            if (n.Contains("ToeBase") || n.Contains("ToeEnd")) offset = new Vec3(0, -0.05, 0.1);
            tpose[j] = tpose[p] + offset;
        }
        var joints0 = tpose.Select(v => v + new Vec3(0, 1.0, 0)).ToArray();
        var rot = new Mat3[frames * SomaSkeleton.Count];
        for (int t = 0; t < frames; t++) for (int j = 0; j < SomaSkeleton.Count; j++) rot[t * SomaSkeleton.Count + j] = rotation(t, SomaSkeleton.Names[j]);
        return new MotionClip(30, rot, joints0, Enumerable.Range(0, frames).Select(t => new Vec3(0, 1.0, 0.1 * t)).ToArray()) { Name = "test" };
    }

    private static Mat3 Anim_RotZ(double degrees) { double c = Math.Cos(degrees * Math.PI / 180), s = Math.Sin(degrees * Math.PI / 180); return new Mat3(c, -s, 0, s, c, 0, 0, 0, 1); }

    // ───────────────────────────── mapping ─────────────────────────────

    [TestMethod]
    public async Task Animation_Mapper_FindsTheBodyOfAStickFigure() => await RunPureTest(async () =>
    {
        var rig = Anim_StickFigure();
        var m = HumanoidMapper.Map(rig.Parents, rig.Heads, new Vec3(0, 1, 0));
        string S(string bone) => SomaSkeleton.Names[m.Soma[rig.By[bone]]];
        Ur_Expect(S("hips") == "Hips" && S("spine1") == "Spine1" && S("spine2") == "Spine2" && S("chest") == "Chest", $"spine: {S("hips")} {S("spine1")} {S("spine2")} {S("chest")}");
        Ur_Expect(S("neck") == "Neck1" && S("head") == "Neck2" && S("headTop") == "Head", $"neck and head: {S("neck")} {S("head")} {S("headTop")}");
        Ur_Expect(S("handL") == "LeftHand" && S("handR") == "RightHand" && S("armL") == "LeftArm" && S("foreR") == "RightForeArm" && S("shoulderL") == "LeftShoulder", $"arms: {S("shoulderL")} {S("armL")} {S("foreR")} {S("handL")} {S("handR")}");
        Ur_Expect(S("leg0") == "LeftLeg" && S("knee0") == "LeftShin" && S("ankle0") == "LeftFoot" && S("toe1") == "RightToeBase", $"legs: {S("leg0")} {S("knee0")} {S("ankle0")} {S("toe1")}");
        Ur_Expect(S("cape") == "Chest" && S("handTipL") == "LeftHandMiddle4", $"extras follow the bone above them: cape {S("cape")}, a single finger {S("handTipL")}");
        var f = m.Frame;                                                              // facing +Z, left +X, up +Y
        Ur_Expect(Math.Abs(f[0, 0] - 1) < 1e-9 && Math.Abs(f[1, 1] - 1) < 1e-9 && Math.Abs(f[2, 2] - 1) < 1e-9, "character frame (left, up, forward) = (X, Y, Z)");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Animation_Mapper_RefusesWhatIsNotAHumanoid() => await RunPureTest(async () =>
    {
        var four = Anim_StickFigure(legs: 4);
        bool refused = false;
        try { HumanoidMapper.Map(four.Parents, four.Heads, new Vec3(0, 1, 0)); } catch (NotAHumanoidException) { refused = true; }
        Ur_Expect(refused, "a four-legged body must be refused");
        refused = false;
        try { HumanoidMapper.Map(new[] { -1, 0, 1 }, new[] { new Vec3(0, 1, 0), new Vec3(0, 1.2, 0), new Vec3(0, 1.4, 0) }, new Vec3(0, 1, 0)); } catch (NotAHumanoidException) { refused = true; }
        Ur_Expect(refused, "a chain of three bones is not a humanoid");
        await Task.CompletedTask;
    });

    // ───────────────────────────── retargeting semantics ─────────────────────────────

    /// <summary>World joint positions by forward kinematics of a retargeted frame (rest rotations are identity).</summary>
    private static Vec3[] Anim_Fk(Anim_Rig rig, RetargetedClip clip, int frame)
    {
        int J = rig.Parents.Length; var world = new Mat3[J]; var pos = new Vec3[J];
        for (int b = 0; b < J; b++)
        {
            int o = (frame * J + b) * 4; var local = Mat3.FromQuaternion(clip.Rotations[o], clip.Rotations[o + 1], clip.Rotations[o + 2], clip.Rotations[o + 3]);
            world[b] = rig.Parents[b] < 0 ? local : world[rig.Parents[b]] * local;
            pos[b] = rig.Parents[b] < 0 ? new Vec3(clip.RootTranslation[frame * 3], clip.RootTranslation[frame * 3 + 1], clip.RootTranslation[frame * 3 + 2])
                                        : pos[rig.Parents[b]] + world[rig.Parents[b]].Mul(rig.Heads[b] - rig.Heads[rig.Parents[b]]);
        }
        return pos;
    }

    [TestMethod]
    public async Task Animation_Retarget_ARestClipKeepsAnAlignedRig() => await RunPureTest(async () =>
    {
        // a clip that never leaves SOMA's T-pose, on a rig that already stands like it: the arms stay where they are (hips travel moves the whole body)
        var rig = Anim_StickFigure();
        var clip = new HumanoidRetargeter(rig.Parents, rig.Heads).Retarget(Anim_ClipWith(3, (_, _) => Mat3.Identity));
        for (int t = 0; t < 3; t++)
        {
            var p = Anim_Fk(rig, clip, t); var move = p[0] - rig.Heads[0];
            foreach (var bone in new[] { "armL", "foreL", "handL", "armR", "foreR", "handR", "spine2", "chest", "neck" })
                Ur_Expect((p[rig.By[bone]] - rig.Heads[rig.By[bone]] - move).Length < 1e-4, $"frame {t}: {bone} left its rest position relative to the hips");
            Ur_Expect(Math.Abs(move.X) < 1e-9 && Math.Abs(move.Y) < 1e-9 && (t == 0 ? move.Z == 0 : move.Z > 0), $"frame {t}: the hips travel forward only ({move.X:F3}, {move.Y:F3}, {move.Z:F3})");
        }
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Animation_Retarget_AlignsASplayedRestPose() => await RunPureTest(async () =>
    {
        // legs splayed outward in the rig's rest pose (the knee -> ankle line leans 0.05 sideways); the clip holds SOMA's straight-legged T-pose, so alignment swings the shin down
        var rig = Anim_StickFigure(splayed: true);
        var rest = rig.Heads[rig.By["ankle0"]].X - rig.Heads[rig.By["knee0"]].X;
        var clip = new HumanoidRetargeter(rig.Parents, rig.Heads).Retarget(Anim_ClipWith(1, (_, _) => Mat3.Identity));
        var p = Anim_Fk(rig, clip, 0);
        double lean = p[rig.By["ankle0"]].X - p[rig.By["knee0"]].X;
        Ur_Expect(Math.Abs(rest) > 0.04 && Math.Abs(lean) < 1e-4, $"the shin leaned {rest:F3} at rest and {lean:F5} after alignment (it should stand straight like SOMA's)");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Animation_Retarget_RaisingTheLeftArmRaisesTheLeftArm() => await RunPureTest(async () =>
    {
        var rig = Anim_StickFigure();
        // SOMA: rotate the left upper arm 90 degrees about +Z (forward): the arm points to +Y (up) instead of +X (left); the fore arm and hand follow
        var clip = new HumanoidRetargeter(rig.Parents, rig.Heads).Retarget(Anim_ClipWith(2, (t, n) => t == 1 && n is "LeftArm" or "LeftForeArm" or "LeftHand" ? Anim_RotZ(90) : Mat3.Identity));
        var rest = Anim_Fk(rig, clip, 0); var posed = Anim_Fk(rig, clip, 1);
        var shoulder = rig.By["armL"]; var hand = rig.By["handL"];
        var v0 = rest[hand] - rest[shoulder]; var v1 = posed[hand] - posed[shoulder];
        Ur_Expect(Math.Abs(v1.X) < 1e-4 && v1.Y > 0.5 && Math.Abs(v1.Length - v0.Length) < 1e-4, $"the left hand should be straight above the upper arm now, got ({v1.X:F3}, {v1.Y:F3}, {v1.Z:F3}) from ({v0.X:F3}, {v0.Y:F3}, {v0.Z:F3})");
        var other = posed[rig.By["handR"]] - rest[rig.By["handR"]]; var travel = posed[0] - rest[0];
        Ur_Expect((other - travel).Length < 1e-4, "the right arm must not move relative to the body");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Animation_Retarget_ArmClearanceSwingsTheUpperArmsOutward() => await RunPureTest(async () =>
    {
        var rig = Anim_StickFigure();
        // both arms hanging straight down (SOMA: 90 degrees about +Z for the left arm chain, -90 for the right), then 12 degrees of clearance
        // (frame 0 is the T-pose every clip starts from; the arms hang at frame 1)
        var hang = Anim_ClipWith(2, (t, n) => t == 0 ? Mat3.Identity : n is "LeftArm" or "LeftForeArm" or "LeftHand" ? Anim_RotZ(-90) : n is "RightArm" or "RightForeArm" or "RightHand" ? Anim_RotZ(90) : Mat3.Identity);
        var retargeter = new HumanoidRetargeter(rig.Parents, rig.Heads);
        var plain = Anim_Fk(rig, retargeter.Retarget(hang), 1); var clear = Anim_Fk(rig, retargeter.Retarget(hang, 1.0, 12), 1);
        foreach (var (side, sign) in new[] { ("L", 1.0), ("R", -1.0) })
        {
            var arm = rig.By["arm" + side]; var hand = rig.By["hand" + side];
            var v0 = plain[hand] - plain[arm]; var v1 = clear[hand] - clear[arm];
            double out0 = v0.X * sign, out1 = v1.X * sign;
            Ur_Expect(v0.Y < -0.4 && Math.Abs(out0) < 1e-4, $"{side}: without clearance the arm hangs straight down ({v0.X:F3}, {v0.Y:F3})");
            Ur_Expect(Math.Abs(v1.Length - v0.Length) < 1e-4 && Math.Abs(out1 - v0.Length * Math.Sin(12 * Math.PI / 180)) < 1e-4, $"{side}: 12 degrees of clearance moves the hand {out1:F4} outward (expected {v0.Length * Math.Sin(12 * Math.PI / 180):F4})");
            Ur_Expect(Math.Abs(v1.Z) < 1e-4, $"{side}: the swing stays in the body's frontal plane");
        }
        Ur_Expect((plain[rig.By["spine2"]] - clear[rig.By["spine2"]]).Length < 1e-6 && (plain[rig.By["chest"]] - clear[rig.By["chest"]]).Length < 1e-6, "the clearance moves nothing but the arms");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task MeshRepair_AlignWindingToNormals_FlipsOnlyBackwardsTriangles() => await RunPureTest(async () =>
    {
        // a flat 6 x 6 sheet facing +Z; one triangle wound backwards must flip back, the rest stay
        const int n = 6;
        var p = new List<float>(); var f = new List<int>();
        for (int y = 0; y <= n; y++) for (int x = 0; x <= n; x++) { p.Add(x); p.Add(y); p.Add(0); }
        int V(int x, int y) => y * (n + 1) + x;
        for (int y = 0; y < n; y++) for (int x = 0; x < n; x++) f.AddRange(new[] { V(x, y), V(x + 1, y), V(x + 1, y + 1), V(x, y), V(x + 1, y + 1), V(x, y + 1) });
        var pa = p.ToArray(); var good = f.ToArray(); var fa = (int[])good.Clone();
        Ur_Expect(SpawnDev.ILGPU.ML.Trellis2.MeshRepair.AlignWindingToNormals(pa, fa) == 0 && fa.SequenceEqual(good), "a consistently wound sheet is left as it is");
        int t = 14;                                                     // a triangle in the middle of the sheet, backwards
        (fa[t * 3 + 1], fa[t * 3 + 2]) = (fa[t * 3 + 2], fa[t * 3 + 1]);
        int flipped = SpawnDev.ILGPU.ML.Trellis2.MeshRepair.AlignWindingToNormals(pa, fa);
        Ur_Expect(flipped == 1 && fa.SequenceEqual(good), $"the backwards triangle flips back ({flipped} flipped)");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task SkinTokens_RigCleanup_CutsOnlyTheBridgesBetweenFarParts() => await RunPureTest(async () =>
    {
        // a chain of 8 bones 0-1-2-3-4-5-6-7 (hops are index differences)
        var parents = new[] { -1, 0, 1, 2, 3, 4, 5, 6 }; int J = parents.Length;
        var hops = SpawnDev.ILGPU.ML.SkinTokens.RigCleanup.HopMatrix(parents);
        Ur_Expect(hops[0 * J + 7] == 7 && hops[2 * J + 5] == 3 && hops[4 * J + 4] == 0, "hop matrix of a chain");
        // vertices: 0 = bone 1 only, 1 = bone 1 (0.6) + bone 2 (0.4) (an ordinary joint blend), 2 = bone 1 (0.5) + bone 7 (0.5) (a hand on a hip), 3 = bone 7 only
        var w = new float[4 * J];
        w[0 * J + 1] = 1f;
        w[1 * J + 1] = .6f; w[1 * J + 2] = .4f;
        w[2 * J + 1] = .55f; w[2 * J + 7] = .45f;
        w[3 * J + 7] = 1f;
        int snapped = SpawnDev.ILGPU.ML.SkinTokens.RigCleanup.SnapFarInfluences(w, 4, J, hops, 4);
        Ur_Expect(snapped == 1, $"only the far blend is snapped ({snapped} vertices)");
        Ur_Expect(Math.Abs(w[1 * J + 1] - .6f) < 1e-6 && Math.Abs(w[1 * J + 2] - .4f) < 1e-6, "an ordinary joint blend is untouched");
        Ur_Expect(Math.Abs(w[2 * J + 1] - 1f) < 1e-6 && w[2 * J + 7] == 0f, "the far influence is dropped and the row renormalised");
        var dom = SpawnDev.ILGPU.ML.SkinTokens.RigCleanup.DominantBones(w, 4, J);
        // triangles: (0,1,1) within one part; (0,1,2) near; (2,3,3) joins bone 1 and bone 7 (6 hops); (1,1,3) joins bone 1 and bone 7
        var tri = new[] { 0, 1, 1, 0, 1, 2, 2, 3, 3, 1, 1, 3 };
        var bridge = SpawnDev.ILGPU.ML.SkinTokens.RigCleanup.BridgeTriangles(tri, dom, J, hops, 4);
        Ur_Expect(!bridge[0] && !bridge[1] && bridge[2] && bridge[3], $"bridges: {string.Join(",", bridge)}");
        Ur_Expect(SpawnDev.ILGPU.ML.SkinTokens.RigCleanup.SnapFarInfluences((float[])w.Clone(), 4, J, hops, 0) == 0 && !SpawnDev.ILGPU.ML.SkinTokens.RigCleanup.BridgeTriangles(tri, dom, J, hops, 0).Any(b => b), "0 turns the cleanup off");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Animation_Retarget_FollowsTheCharactersFacing() => await RunPureTest(async () =>
    {
        // the same figure turned to face +X (everything rotated 90 degrees about Y): SOMA's "left arm up" is still the character's left arm up
        var rig0 = Anim_StickFigure(); var turn = new Mat3(0, 0, 1, 0, 1, 0, -1, 0, 0);                  // +Z -> +X, +X -> -Z
        var rig = new Anim_Rig(rig0.Parents, rig0.Heads.Select(h => turn.Mul(h - new Vec3(0, 0, 0))).ToArray(), rig0.By);
        var r = new HumanoidRetargeter(rig.Parents, rig.Heads);
        var clip = r.Retarget(Anim_ClipWith(2, (t, n) => t == 1 && n is "LeftArm" or "LeftForeArm" or "LeftHand" ? Anim_RotZ(90) : Mat3.Identity));
        var rest = Anim_Fk(rig, clip, 0); var posed = Anim_Fk(rig, clip, 1);
        var v1 = posed[rig.By["handL"]] - posed[rig.By["armL"]];
        Ur_Expect(v1.Y > 0.5 && Math.Abs(v1.X) < 1e-4 && Math.Abs(v1.Z) < 1e-4, $"left arm up in a figure facing +X, got ({v1.X:F3}, {v1.Y:F3}, {v1.Z:F3})");
        await Task.CompletedTask;
    });

    // ───────────────────────────── against the reference script's numbers ─────────────────────────────

    private static string? AnimDir()
    {
        var dir = Environment.GetEnvironmentVariable("ANIM_DIR") ?? "E:/spawndev-unirig/anim";
        return File.Exists(Path.Combine(dir, "golden", "humanoid42_walk_quats.npy")) && File.Exists(Path.Combine(dir, "clips", "walk.npz")) ? dir : null;
    }

    private static (int[] Parents, Vec3[] Heads) Anim_GoldenRig(string dir, string name)
    {
        var (h, _) = Ur_Npy<float>(Path.Combine(dir, "golden", name + "_heads.npy")); var (p, _) = Ur_Npy<int>(Path.Combine(dir, "golden", name + "_parents.npy"));
        return (p, Enumerable.Range(0, p.Length).Select(i => new Vec3(h[i * 3], h[i * 3 + 1], h[i * 3 + 2])).ToArray());
    }

    [TestMethod]
    public async Task Animation_Mapper_MatchesTheReference() => await RunPureTest(async () =>
    {
        var dir = AnimDir() ?? throw new UnsupportedTestException("animation goldens not found (ANIM_DIR, default E:/spawndev-unirig/anim; tools/unirig-harness/reference/golden_retarget.py)");
        var (parents, heads) = Anim_GoldenRig(dir, "humanoid42_walk");
        var (soma, _) = Ur_Npy<int>(Path.Combine(dir, "golden", "humanoid42_walk_soma.npy")); var (frame, _) = Ur_Npy<double>(Path.Combine(dir, "golden", "humanoid42_walk_frame.npy"));
        var m = HumanoidMapper.Map(parents, heads, new Vec3(0, 1, 0));
        for (int b = 0; b < parents.Length; b++) Ur_Expect(m.Soma[b] == soma[b], $"bone {b} maps to {SomaSkeleton.Names[m.Soma[b]]}, the reference says {SomaSkeleton.Names[soma[b]]}");
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) Ur_Expect(Math.Abs(m.Frame[r, c] - frame[r * 3 + c]) < 1e-6, $"frame[{r},{c}] {m.Frame[r, c]} vs {frame[r * 3 + c]}");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Animation_Retarget_MatchesTheReference() => await RunPureTest(async () =>
    {
        var dir = AnimDir() ?? throw new UnsupportedTestException("animation goldens not found (ANIM_DIR, default E:/spawndev-unirig/anim; tools/unirig-harness/reference/golden_retarget.py)");
        var lines = new List<string>();
        foreach (var name in new[] { "walk", "wave", "run", "idle" })
        {
            var (parents, heads) = Anim_GoldenRig(dir, $"humanoid42_{name}");
            var (q, qs) = Ur_Npy<double>(Path.Combine(dir, "golden", $"humanoid42_{name}_quats.npy")); var (travel, _) = Ur_Npy<double>(Path.Combine(dir, "golden", $"humanoid42_{name}_travel.npy"));
            var clip = MotionClip.FromNpz(File.ReadAllBytes(Path.Combine(dir, "clips", name + ".npz")), name);
            var got = new HumanoidRetargeter(parents, heads).Retarget(clip);
            Ur_Expect(got.Frames == qs[0] && got.Bones == qs[1], $"{name}: {got.Frames} frames x {got.Bones} bones, the reference {qs[0]} x {qs[1]}");
            double worst = 0, worstT = 0; int root = Array.IndexOf(parents, -1);
            for (int t = 0; t < got.Frames; t++)
            {
                for (int b = 0; b < got.Bones; b++)
                {
                    int o = (t * got.Bones + b) * 4, r = (t * got.Bones + b) * 4;      // reference (w, x, y, z); ours (x, y, z, w)
                    double dot = got.Rotations[o] * q[r + 1] + got.Rotations[o + 1] * q[r + 2] + got.Rotations[o + 2] * q[r + 3] + got.Rotations[o + 3] * q[r];
                    worst = Math.Max(worst, 1 - Math.Abs(dot));
                }
                for (int k = 0; k < 3; k++) worstT = Math.Max(worstT, Math.Abs(got.RootTranslation[t * 3 + k] - (k == 0 ? heads[root].X : k == 1 ? heads[root].Y : heads[root].Z) - travel[t * 3 + k]));
            }
            Ur_Expect(worst < 1e-6, $"{name}: rotations differ from the reference by up to {worst:E1} (1 - |dot|)");
            Ur_Expect(worstT < 1e-4, $"{name}: root travel differs by up to {worstT:E1}");
            lines.Add($"{name} {got.Frames} frames, 1-|dot| {worst:E1}, travel {worstT:E1}");
        }
        Console.WriteLine("[Animation] retarget vs reference: " + string.Join("; ", lines));
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Animation_AnimatedGlb_IsWellFormed() => await RunPureTest(async () =>
    {
        var dir = AnimDir() ?? throw new UnsupportedTestException("animation clips not found (ANIM_DIR)");
        var rigged = Path.Combine(dir, "..", "tmp", "skintokens_rigged_humanoid.glb");
        if (!File.Exists(rigged)) throw new UnsupportedTestException("tmp/skintokens_rigged_humanoid.glb (written by SkinTokens_Rigger_RigsAGeneratedHumanoid) not found");
        var glb = File.ReadAllBytes(rigged);
        var sk = RiggedGlb.ReadSkeleton(glb);
        var heads = Enumerable.Range(0, sk.Count).Select(i => new Vec3(sk.Joints[i * 3], sk.Joints[i * 3 + 1], sk.Joints[i * 3 + 2])).ToArray();
        var rt = new HumanoidRetargeter(sk.Parents, heads);
        var clips = new[] { "walk", "wave" }.Select(n => rt.Retarget(MotionClip.FromNpz(File.ReadAllBytes(Path.Combine(dir, "clips", n + ".npz")), n))).ToList();
        var animated = AnimatedGlb.Add(glb, clips);
        var (root, bin) = RiggedGlb.Parse(animated);
        var anims = root["animations"]!.AsArray();
        Ur_Expect(anims.Count == 2 && anims[0]!["name"]!.GetValue<string>() == "walk", "two named animations");
        foreach (var a in anims)
        {
            var channels = a!["channels"]!.AsArray(); var samplers = a["samplers"]!.AsArray();
            Ur_Expect(channels.Count == sk.Count + 1, $"{sk.Count} rotation channels and one root translation, got {channels.Count}");
            int times = samplers[0]!["input"]!.GetValue<int>(); var acc = root["accessors"]![times]!;
            Ur_Expect(acc["min"]![0]!.GetValue<float>() == 0f && acc["max"]![0]!.GetValue<float>() > 1f, "times start at 0 and span the clip");
            foreach (var c in channels) Ur_Expect(c!["target"]!["node"]!.GetValue<int>() < root["nodes"]!.AsArray().Count, "channel targets exist");
        }
        Ur_Expect(RiggedGlb.ReadMesh(animated).Positions.SequenceEqual(RiggedGlb.ReadMesh(glb).Positions), "the mesh is untouched");
        File.WriteAllBytes(Path.Combine(dir, "..", "tmp", "skintokens_animated_humanoid.glb"), animated);      // for the glTF validator and a viewer
        await Task.CompletedTask;
    });
}
