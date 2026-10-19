using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.UniRig;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// UniRig auto-rigging (skeleton + skinning prior): the host-side pieces as pure tests on shapes whose answer is known, and the networks against goldens written by the
/// REAL reference classes (tools/unirig-harness, CPU torch) when the checkpoints are on this machine (UNIRIG_DIR, default E:/spawndev-unirig; skipped otherwise).
/// </summary>
public abstract partial class MLTestBase
{
    private static void Ur_Expect(bool ok, string what) { if (!ok) throw new Exception(what); }

    // ───────────────────────────────────────────── tokenizer ─────────────────────────────────────────────

    /// <summary>The skeleton language: grammar states and allowed sets, a walk with a branch decoded to the right tree, and refusals.</summary>
    [TestMethod]
    public async Task UniRig_Tokenizer_GrammarAndDetokenize() => await RunPureTest(async () =>
    {
        const int B = UniRigTokenizer.Branch, Bos = UniRigTokenizer.Bos, Eos = UniRigTokenizer.Eos, Cls = UniRigTokenizer.ClsArticulationXl, Spring = UniRigTokenizer.Spring;
        // root (10,20,30) -> (40,50,60) -> (70,80,90); a branch back to (40,50,60) -> (100,110,120)
        int[] seq = { Bos, Cls, Spring, 10, 20, 30, 40, 50, 60, 70, 80, 90, B, 40, 50, 60, 100, 110, 120, Eos };
        var sk = UniRigTokenizer.Detokenize(seq);
        Ur_Expect(sk.Count == 4, $"4 joints expected, got {sk.Count}");
        Ur_Expect(sk.Parents.SequenceEqual(new[] { -1, 0, 1, 1 }), $"parents [{string.Join(",", sk.Parents)}], expected [-1,0,1,1]");
        float[] want = { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 };
        for (int i = 0; i < want.Length; i++) Ur_Expect(Math.Abs(sk.Joints[i] - UniRigTokenizer.Undiscretize((int)want[i])) < 1e-6, $"joint coordinate {i}");
        Ur_Expect(Math.Abs(UniRigTokenizer.Undiscretize(0) + 255f / 256) < 1e-6 && Math.Abs(UniRigTokenizer.Undiscretize(255) - 255f / 256) < 1e-6, "bin centres");

        // allowed sets by state
        var atStart = UniRigTokenizer.Allowed(UniRigTokenizer.StateAfter(ReadOnlySpan<int>.Empty));
        Ur_Expect(atStart.Count == 1 && atStart[0] == Bos, "an empty sequence may only start with bos");
        var afterBos = UniRigTokenizer.Allowed(UniRigTokenizer.StateAfter(new[] { Bos }));
        Ur_Expect(afterBos.Count == 256 + 4 + 3 && afterBos.Contains(Cls) && afterBos.Contains(Spring) && !afterBos.Contains(Eos), $"after bos: {afterBos.Count} ids");
        var afterX = UniRigTokenizer.Allowed(UniRigTokenizer.StateAfter(new[] { Bos, Cls, 10 }));
        Ur_Expect(afterX.Count == 256 && afterX.All(t => t < 256), "mid-joint only coordinates may follow");
        var afterJoint = UniRigTokenizer.Allowed(UniRigTokenizer.StateAfter(new[] { Bos, Cls, 10, 20, 30 }));
        Ur_Expect(afterJoint.Contains(Eos) && afterJoint.Contains(B) && afterJoint.Contains(Spring) && afterJoint.Contains(7) && !afterJoint.Contains(Bos) && !afterJoint.Contains(Cls), "after a complete joint: eos, branch, part or a new joint");
        var afterCls = UniRigTokenizer.Allowed(UniRigTokenizer.StateAfter(new[] { Bos, Cls }));
        Ur_Expect(afterCls.Contains(Eos) && !afterCls.Contains(B), "after the class: a part, a joint or eos, never a branch");
        var afterBranch = UniRigTokenizer.Allowed(UniRigTokenizer.StateAfter(new[] { Bos, Cls, 1, 2, 3, B }));
        Ur_Expect(afterBranch.Count == 256, "after a branch: a coordinate");

        // a cut-off walk keeps only complete joints
        Ur_Expect(UniRigTokenizer.CompleteLength(new[] { Bos, Cls, 10, 20 }) == 0, "no complete joint yet");
        Ur_Expect(UniRigTokenizer.CompleteLength(new[] { Bos, Cls, 10, 20, 30, 40, 50 }) == 5, "one complete joint, the second unfinished");

        // refusals
        foreach (var bad in new[] { new[] { Cls, 1, 2, 3, Eos }, new[] { Bos, Cls, 1, 2, 3 } })
        {
            bool threw = false;
            try { UniRigTokenizer.Detokenize(bad); } catch (ArgumentException) { threw = true; }
            Ur_Expect(threw, "a sequence without bos / without eos must be refused");
        }
        await Task.CompletedTask;
    });

    // ─────────────────────────────────────────── voxel skin ───────────────────────────────────────────

    /// <summary>A closed rod along x with a joint a quarter in from each end: the ends follow their own joint, the middle shares, weights sum to 1; the voxel grid fills a sphere to its volume.</summary>
    [TestMethod(Timeout = 120000)]
    public async Task UniRig_VoxelSkin_TwoBonesOnARod() => await RunPureTest(async () =>
    {
        // closed cylinder along x, radius 0.2, x in [-0.8, 0.8], 40 rings x 24 around, a vertex at each end for the caps
        int rings = 40, around = 24; var v = new List<float>(); var f = new List<int>();
        for (int r = 0; r < rings; r++)
            for (int a = 0; a < around; a++)
            {
                double x = -0.8 + 1.6 * r / (rings - 1), t = 2 * Math.PI * a / around;
                v.Add((float)x); v.Add((float)(0.2 * Math.Cos(t))); v.Add((float)(0.2 * Math.Sin(t)));
            }
        int capA = v.Count / 3; v.Add(-0.8f); v.Add(0); v.Add(0); int capB = v.Count / 3; v.Add(0.8f); v.Add(0); v.Add(0);
        for (int r = 0; r + 1 < rings; r++)
            for (int a = 0; a < around; a++)
            {
                int i0 = r * around + a, i1 = r * around + (a + 1) % around, j0 = (r + 1) * around + a, j1 = (r + 1) * around + (a + 1) % around;
                f.Add(i0); f.Add(j0); f.Add(i1); f.Add(i1); f.Add(j0); f.Add(j1);
            }
        for (int a = 0; a < around; a++) { f.Add(capA); f.Add((a + 1) % around); f.Add(a); f.Add(capB); f.Add((rings - 1) * around + a); f.Add((rings - 1) * around + (a + 1) % around); }
        var vf = v.ToArray(); var ff = f.ToArray(); int n = vf.Length / 3;
        float[] joints = { -0.4f, 0f, 0f, 0.4f, 0f, 0f };
        var vox = UniRigVoxelSkin.Voxelize(vf, ff, 64);
        Ur_Expect(vox.Count > 2000, $"{vox.Count} voxels for a rod");
        var skin = UniRigVoxelSkin.Compute(vf, ff, joints, vox);
        double worstSum = 0; int checkedL = 0, checkedR = 0, checkedM = 0;
        for (int i = 0; i < n; i++)
        {
            double w0 = skin[i * 2], w1 = skin[i * 2 + 1]; worstSum = Math.Max(worstSum, Math.Abs(w0 + w1 - 1));
            float x = vf[i * 3];
            // the reference measures through voxels at three times their length, so even the tip of the rod keeps ~0.84 (inverse-quadratic weights): dominant and well above half
            if (x < -0.65f) { checkedL++; Ur_Expect(w0 > 0.75, $"vertex at x={x:F2}: weight {w0:F3} for the left joint"); }
            else if (x > 0.65f) { checkedR++; Ur_Expect(w1 > 0.75, $"vertex at x={x:F2}: weight {w1:F3} for the right joint"); }
            else if (Math.Abs(x) < 0.03f) { checkedM++; Ur_Expect(Math.Abs(w0 - 0.5) < 0.12, $"middle vertex at x={x:F2}: weights {w0:F3}/{w1:F3}"); }
        }
        Ur_Expect(worstSum < 1e-5, $"weights sum to 1 (worst {worstSum:E1})");
        Ur_Expect(checkedL > 50 && checkedR > 50 && checkedM > 20, $"checked {checkedL}/{checkedR}/{checkedM} vertices");

        // the visual hull of a convex shape is the shape: a sphere of radius 0.8 fills its own volume at grid 64
        var (sp, sf) = Mr_Sphere(4); for (int i = 0; i < sp.Length; i++) sp[i] *= 0.8f;
        var sv = UniRigVoxelSkin.Voxelize(sp, sf, 64);
        double expected = 4.0 / 3 * Math.PI * Math.Pow(0.8, 3) / Math.Pow(2.0 / 64, 3);
        Ur_Expect(Math.Abs(sv.Count / expected - 1) < 0.03, $"sphere filled {sv.Count} voxels, volume says {expected:F0}");

        // A convex shape is its own hull on every axis, so it cannot tell "inside along two axes" from "inside along one". Two boxes with a gap between them can: a ray along x through
        // both fills the gap with a vote from x alone, and the gap is NOT part of the shape (rays along y and z through it hit nothing). Two votes are required.
        var (bv, bf) = Ur_Box(-0.8f, -0.4f, -0.3f, 0.3f, -0.3f, 0.3f); var (cv, cf) = Ur_Box(0.4f, 0.8f, -0.3f, 0.3f, -0.3f, 0.3f);
        var gv = bv.Concat(cv).ToArray(); var gf = bf.Concat(cf.Select(i => i + bv.Length / 3)).ToArray();
        var gh = UniRigVoxelSkin.Voxelize(gv, gf, 64);
        int exact = 0, outside = 0;                                                      // the discretised truth: voxel centres inside either box
        bool InBoxes(double x, double y, double z) => Math.Abs(y) <= 0.3 && Math.Abs(z) <= 0.3 && ((x >= -0.8 && x <= -0.4) || (x >= 0.4 && x <= 0.8));
        for (int i = 0; i < 64; i++) for (int j = 0; j < 64; j++) for (int k = 0; k < 64; k++)
            if (InBoxes((i + 0.5) * 2.0 / 64 - 1, (j + 0.5) * 2.0 / 64 - 1, (k + 0.5) * 2.0 / 64 - 1)) exact++;
        for (int q = 0; q < gh.Count; q++) if (!InBoxes(gh.Coord(q, 0), gh.Coord(q, 1), gh.Coord(q, 2))) outside++;
        Ur_Expect(outside == 0, $"{outside} voxels outside the two boxes (the gap between them must stay empty)");
        Ur_Expect(gh.Count == exact, $"the two boxes fill {gh.Count} voxels, their voxel centres say {exact}");
        await Task.CompletedTask;
    });

    /// <summary>A closed box, 12 triangles, outward winding.</summary>
    private static (float[] V, int[] F) Ur_Box(float x0, float x1, float y0, float y1, float z0, float z1)
    {
        var v = new float[] { x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1 };
        var f = new[] { 0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5 };
        return (v, f);
    }

    // ─────────────────────────────────────────── mesh sampler ───────────────────────────────────────────

    /// <summary>The encoder's input: 65536 points in [-1,1] with unit normals, surface samples on the surface facing outward, 1024 well-spread distinct query rows, reproducible.</summary>
    [TestMethod(Timeout = 120000)]
    public async Task UniRig_MeshSampler_ProducesTheEncoderInput() => await RunPureTest(async () =>
    {
        var (sp, sf) = Mr_Sphere(5);
        for (int i = 0; i < sp.Length; i += 3) { sp[i] = sp[i] * 2 + 3; sp[i + 1] = sp[i + 1] * 2 - 2; sp[i + 2] = sp[i + 2] * 2 + 1; }   // radius 2 centred on (3, -2, 1)
        var s = UniRigMeshSampler.Sample(sp, sf, seed: 5);
        Ur_Expect(s.Points.Length == UniRigMeshSampler.NumSamples * 3 && s.Normals.Length == s.Points.Length, "65536 points and normals");
        Ur_Expect(Math.Abs(s.Center[0] - 3) < 1e-2 && Math.Abs(s.Center[1] + 2) < 1e-2 && Math.Abs(s.Center[2] - 1) < 1e-2 && Math.Abs(s.Scale - 2) < 1e-2, $"centre ({s.Center[0]:F2},{s.Center[1]:F2},{s.Center[2]:F2}) scale {s.Scale:F3}");
        double minNorm = 1e9, maxNorm = 0; int outward = 0;
        for (int i = 0; i < UniRigMeshSampler.NumSamples; i++)
        {
            double x = s.Points[i * 3], y = s.Points[i * 3 + 1], z = s.Points[i * 3 + 2], r = Math.Sqrt(x * x + y * y + z * z);
            Ur_Expect(Math.Abs(x) <= 1.0001 && Math.Abs(y) <= 1.0001 && Math.Abs(z) <= 1.0001, "points lie in [-1,1]");
            double nl = Math.Sqrt(s.Normals[i * 3] * s.Normals[i * 3] + s.Normals[i * 3 + 1] * s.Normals[i * 3 + 1] + s.Normals[i * 3 + 2] * s.Normals[i * 3 + 2]);
            minNorm = Math.Min(minNorm, nl); maxNorm = Math.Max(maxNorm, nl);
            if (i >= UniRigMeshSampler.VertexSamples && (s.Normals[i * 3] * x + s.Normals[i * 3 + 1] * y + s.Normals[i * 3 + 2] * z) / r > 0.95) outward++;   // a sphere: the normal is the radius
        }
        Ur_Expect(minNorm > 0.999 && maxNorm < 1.001, $"unit normals ({minNorm:F4}..{maxNorm:F4})");
        Ur_Expect(outward > (UniRigMeshSampler.NumSamples - UniRigMeshSampler.VertexSamples) * 0.99, $"surface samples facing outward: {outward}");
        Ur_Expect(s.QueryRows.Length == 1024 && s.QueryRows.Distinct().Count() == 1024 && s.QueryRows.All(r => r >= 0 && r < UniRigMeshSampler.NumSamples), "1024 distinct query rows");
        // farthest-point sampling spreads them: every point of a unit sphere is near a query (random choices leave bigger gaps)
        double cover = 0;
        for (int i = 0; i < UniRigMeshSampler.NumSamples; i += 61)
        {
            double best = 1e9;
            foreach (int q in s.QueryRows)
            {
                double dx = s.Points[i * 3] - s.Points[q * 3], dy = s.Points[i * 3 + 1] - s.Points[q * 3 + 1], dz = s.Points[i * 3 + 2] - s.Points[q * 3 + 2];
                best = Math.Min(best, dx * dx + dy * dy + dz * dz);
            }
            cover = Math.Max(cover, Math.Sqrt(best));
        }
        Ur_Expect(cover < 0.2, $"farthest-point query rows leave a gap of {cover:F3} (a unit sphere)");
        // what farthest-point sampling guarantees and a random subset does not: no two queries are close (1024 random points on a sphere have a pair ~0.01 apart)
        double closest = double.MaxValue;
        for (int a = 0; a < 1024; a++)
            for (int b = a + 1; b < 1024; b++)
            {
                int qa = s.QueryRows[a] * 3, qb = s.QueryRows[b] * 3;
                double dx = s.Points[qa] - s.Points[qb], dy = s.Points[qa + 1] - s.Points[qb + 1], dz = s.Points[qa + 2] - s.Points[qb + 2];
                closest = Math.Min(closest, Math.Sqrt(dx * dx + dy * dy + dz * dz));
            }
        Ur_Expect(closest > 0.05, $"two query points only {closest:F4} apart: the queries are not farthest-point sampled");
        var again = UniRigMeshSampler.Sample(sp, sf, seed: 5); var other = UniRigMeshSampler.Sample(sp, sf, seed: 6);
        Ur_Expect(again.QueryRows.SequenceEqual(s.QueryRows) && again.Points.AsSpan(0, 3000).SequenceEqual(s.Points.AsSpan(0, 3000)), "same seed, same input");
        Ur_Expect(!other.QueryRows.SequenceEqual(s.QueryRows), "another seed, another input");
        var back = new float[3]; s.ToMeshSpace(new[] { 0f, 0f, 0f }, back);
        Ur_Expect(Math.Abs(back[0] - 3) < 1e-2 && Math.Abs(back[1] + 2) < 1e-2 && Math.Abs(back[2] - 1) < 1e-2, "ToMeshSpace maps the origin to the mesh centre");
        await Task.CompletedTask;
    });

    // ───────────────────────────────────────────── rigged GLB ─────────────────────────────────────────────

    /// <summary>Adding a skin to a glTF: joints, hierarchy, inverse bind matrices, the four strongest influences, and the rest of the file untouched.</summary>
    [TestMethod]
    public async Task UniRig_RiggedGlb_AddsASkinAndKeepsTheRest() => await RunPureTest(async () =>
    {
        float[] pos = { 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0 }, nrm = { 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1 }, uv = { 0, 0, 1, 0, 0, 1, 1, 1 };
        int[] idx = { 0, 1, 2, 2, 1, 3 };
        var png = PngEncoder.EncodeRgba8(new uint[] { 0xFF8040C0u }, 1, 1);
        var glb = PbrGlbWriter.Write(pos, nrm, uv, idx, png, png);
        var before = RiggedGlb.ReadMesh(glb);
        // five joints in a chain with a side branch; positions in the file's own space
        var sk = new UniRigTokenizer.Skeleton(new float[] { 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 2, 0, -1, 2, 0 }, new[] { -1, 0, 1, 2, 2 });
        var w = new float[4 * 5];
        float[][] rows = { new[] { 0.1f, 0.2f, 0.3f, 0.25f, 0.15f }, new[] { 1f, 0, 0, 0, 0 }, new[] { 0, 0, 0, 0, 2f }, new[] { 0, 0.5f, 0, 0.5f, 0 } };
        for (int v = 0; v < 4; v++) for (int j = 0; j < 5; j++) w[v * 5 + j] = rows[v][j];
        var rigged = RiggedGlb.Add(glb, sk, w);

        var after = RiggedGlb.ReadMesh(rigged);
        Ur_Expect(after.Positions.SequenceEqual(before.Positions) && after.Indices.SequenceEqual(before.Indices), "the mesh itself is untouched");
        int jsonLen = (int)BitConverter.ToUInt32(rigged, 12), bin = 20 + jsonLen + 8;
        using var doc = JsonDocument.Parse(rigged.AsMemory(20, jsonLen));
        var root = doc.RootElement;
        Ur_Expect(root.GetProperty("images").GetArrayLength() == 2 && root.GetProperty("materials").GetArrayLength() == 1, "textures and material kept");
        var skin = root.GetProperty("skins")[0];
        var joints = skin.GetProperty("joints").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        Ur_Expect(joints.Length == 5 && joints.Distinct().Count() == 5, "five joint nodes");
        var nodes = root.GetProperty("nodes");
        // translations are relative to the parent; the chain 0->1->2 and 2->{3,4}
        float[] rel = { 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, -1, 0, 0 };
        for (int j = 0; j < 5; j++)
        {
            var t = nodes[joints[j]].GetProperty("translation").EnumerateArray().Select(e => e.GetSingle()).ToArray();
            Ur_Expect(t.SequenceEqual(rel.AsSpan(j * 3, 3).ToArray()), $"joint {j} translation [{string.Join(",", t)}]");
        }
        var kids = nodes[joints[2]].GetProperty("children").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        Ur_Expect(kids.SequenceEqual(new[] { joints[3], joints[4] }), "joint 2 has children 3 and 4");
        int armature = skin.GetProperty("skeleton").GetInt32();
        Ur_Expect(nodes[armature].GetProperty("children").EnumerateArray().Select(e => e.GetInt32()).SequenceEqual(new[] { joints[0] }), "the armature parents the root");
        Ur_Expect(root.GetProperty("scenes")[0].GetProperty("nodes").EnumerateArray().Select(e => e.GetInt32()).Contains(armature), "the armature is in the scene");
        Ur_Expect(nodes[0].GetProperty("skin").GetInt32() == 0, "the mesh node uses the skin");

        var prim = root.GetProperty("meshes")[0].GetProperty("primitives")[0].GetProperty("attributes");
        (int off, int count) Acc(int a) { var acc = root.GetProperty("accessors")[a]; var bv = root.GetProperty("bufferViews")[acc.GetProperty("bufferView").GetInt32()]; return (bin + bv.GetProperty("byteOffset").GetInt32(), acc.GetProperty("count").GetInt32()); }
        var (jo, jc) = Acc(prim.GetProperty("JOINTS_0").GetInt32()); var (wo, wc) = Acc(prim.GetProperty("WEIGHTS_0").GetInt32());
        Ur_Expect(jc == 4 && wc == 4, "per-vertex joints and weights");
        ushort J(int v, int k) => BitConverter.ToUInt16(rigged, jo + v * 8 + k * 2);
        float W(int v, int k) => BitConverter.ToSingle(rigged, wo + v * 16 + k * 4);
        // vertex 0: the four strongest of [.1 .2 .3 .25 .15] are joints 2, 3, 1, 4 (renormalised by 0.9)
        Ur_Expect(new[] { J(0, 0), J(0, 1), J(0, 2), J(0, 3) }.SequenceEqual(new ushort[] { 2, 3, 1, 4 }), $"vertex 0 joints {J(0, 0)},{J(0, 1)},{J(0, 2)},{J(0, 3)}");
        Ur_Expect(Math.Abs(W(0, 0) - 0.3f / 0.9f) < 1e-6 && Math.Abs(W(0, 3) - 0.15f / 0.9f) < 1e-6, "vertex 0 weights renormalised");
        for (int v = 0; v < 4; v++) Ur_Expect(Math.Abs(W(v, 0) + W(v, 1) + W(v, 2) + W(v, 3) - 1) < 1e-6, $"vertex {v} weights sum to 1");
        Ur_Expect(J(1, 0) == 0 && W(1, 0) == 1f && W(1, 1) == 0f, "vertex 1 follows joint 0 alone");
        Ur_Expect(J(2, 0) == 4 && W(2, 0) == 1f, "a weight of 2 renormalises to 1 on joint 4");
        Ur_Expect(J(3, 0) == 1 && J(3, 1) == 3 && Math.Abs(W(3, 0) - 0.5f) < 1e-6, "vertex 3 splits between joints 1 and 3");
        // inverse bind matrices: a translation by minus the joint position (column-major)
        var ibmAcc = root.GetProperty("accessors")[skin.GetProperty("inverseBindMatrices").GetInt32()];
        var ibmOff = bin + root.GetProperty("bufferViews")[ibmAcc.GetProperty("bufferView").GetInt32()].GetProperty("byteOffset").GetInt32();
        Ur_Expect(ibmAcc.GetProperty("type").GetString() == "MAT4" && ibmAcc.GetProperty("count").GetInt32() == 5, "five MAT4");
        for (int j = 0; j < 5; j++)
        {
            float M(int k) => BitConverter.ToSingle(rigged, ibmOff + (j * 16 + k) * 4);
            Ur_Expect(M(0) == 1 && M(5) == 1 && M(10) == 1 && M(15) == 1 && M(12) == -sk.Joints[j * 3] && M(13) == -sk.Joints[j * 3 + 1] && M(14) == -sk.Joints[j * 3 + 2], $"inverse bind matrix {j}");
        }
        var views = root.GetProperty("bufferViews"); var total = root.GetProperty("buffers")[0].GetProperty("byteLength").GetInt32();
        foreach (var bv in views.EnumerateArray()) Ur_Expect(bv.GetProperty("byteOffset").GetInt32() + bv.GetProperty("byteLength").GetInt32() <= total, "buffer views inside the buffer");
        bool again = false; try { RiggedGlb.Add(rigged, sk, w); } catch (InvalidOperationException) { again = true; }
        Ur_Expect(again, "a skinned mesh is refused a second skin");
        await Task.CompletedTask;
    });

    // ─────────────────────────────────── networks against the reference ───────────────────────────────────

    private static string? UniRigDir()
    {
        var dir = Environment.GetEnvironmentVariable("UNIRIG_DIR") ?? "E:/spawndev-unirig";
        return File.Exists(Path.Combine(dir, "weights", "skeleton.safetensors")) && File.Exists(Path.Combine(dir, "golden", "skeleton", "latents.npy")) ? dir : null;
    }

    private static (T[] Data, int[] Shape) Ur_Npy<T>(string path) where T : unmanaged
    {
        using var f = File.OpenRead(path);
        var magic = new byte[8]; f.ReadExactly(magic);
        int hlen; if (magic[6] == 1) { var b = new byte[2]; f.ReadExactly(b); hlen = BitConverter.ToUInt16(b); } else { var b = new byte[4]; f.ReadExactly(b); hlen = BitConverter.ToInt32(b); }
        var hb = new byte[hlen]; f.ReadExactly(hb); var header = System.Text.Encoding.ASCII.GetString(hb);
        var descr = System.Text.RegularExpressions.Regex.Match(header, @"'descr':\s*'([^']+)'").Groups[1].Value;
        var shape = System.Text.RegularExpressions.Regex.Match(header, @"'shape':\s*\(([^)]*)\)").Groups[1].Value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse).ToArray();
        string want = typeof(T) == typeof(float) ? "<f4" : typeof(T) == typeof(double) ? "<f8" : typeof(T) == typeof(int) ? "<i4" : "<i8";
        if (descr != want || header.Contains("'fortran_order': True")) throw new InvalidDataException($"{path}: {descr} / {header}");
        var data = new T[shape.Aggregate(1L, (a, b) => a * b)];
        f.ReadExactly(System.Runtime.InteropServices.MemoryMarshal.AsBytes(data.AsSpan()));
        return (data, shape);
    }

    private static string Ur_RequireDevice(Accelerator accelerator)
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("UniRig goldens run on the desktop GPU lanes (CUDA, OpenCL)");
        return UniRigDir() ?? throw new UnsupportedTestException("UniRig checkpoints and goldens not found (UNIRIG_DIR, default E:/spawndev-unirig; tools/unirig-harness)");
    }

    /// <summary>The perceiver encoder on the golden's 65536 points against the reference's latents (bf16 weights alone cost relRMS ~2e-3).</summary>
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task UniRig_Encoder_MatchesTheReferenceLatents() => await RunTest(async accelerator =>
    {
        var dir = Ur_RequireDevice(accelerator); var g = Path.Combine(dir, "golden", "skeleton");
        var (pts, ps) = Ur_Npy<float>(Path.Combine(g, "points.npy")); var (nrm, _) = Ur_Npy<float>(Path.Combine(g, "normals.npy"));
        var (pre, _) = Ur_Npy<long>(Path.Combine(g, "presample_idx.npy")); var (fps, _) = Ur_Npy<long>(Path.Combine(g, "fps_idx.npy")); var (gold, _) = Ur_Npy<float>(Path.Combine(g, "latents.npy"));
        using var weights = GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "skeleton.safetensors"), name => name.StartsWith("mesh_encoder.encoder.") || name.StartsWith("output_proj."));
        using var p = accelerator.Allocate1D(pts); using var q = accelerator.Allocate1D(nrm); using var enc = new UniRigEncoder(accelerator, weights);
        var o = enc.Encode(p.View, q.View, ps[0], fps.Select(i => (int)pre[i]).ToArray());
        var got = await o.View.CopyToHostAsync();
        double num = 0, den = 0; for (int i = 0; i < gold.Length; i++) { double d = got[i] - gold[i]; num += d * d; den += (double)gold[i] * gold[i]; }
        double rel = Math.Sqrt(num / den);
        Ur_Expect(rel < 0.02, $"encoder latents relRMS {rel:E2} vs the reference (expected ~2e-3 to 5e-3)");
        Console.WriteLine($"[UniRig] encoder relRMS {rel:E2} on {accelerator.AcceleratorType}");
    });

    /// <summary>The OPT decoder, teacher-forced on the reference's own latents and tokens: logits against the reference's, and the grammar-constrained argmax at every step.</summary>
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task UniRig_Decoder_TeacherForcedLogitsMatchTheReference() => await RunTest(async accelerator =>
    {
        var dir = Ur_RequireDevice(accelerator); var g = Path.Combine(dir, "golden", "skeleton");
        var (lat, _) = Ur_Npy<float>(Path.Combine(g, "latents.npy")); var (full64, _) = Ur_Npy<long>(Path.Combine(g, "full_ids.npy")); var (gold, gs) = Ur_Npy<float>(Path.Combine(g, "logits_teacher.npy"));
        var allowed = JsonSerializer.Deserialize<List<List<int>>>(File.ReadAllText(Path.Combine(g, "allowed.json")))!;
        var full = full64.Select(v => (int)v).ToArray(); int steps = gs[0], V = gs[1];
        using var weights = GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "skeleton.safetensors"), name => name.StartsWith("transformer."));
        using var cond = accelerator.Allocate1D(lat); using var dec = new UniRigDecoder(accelerator, weights);
        var got = await dec.PrefillAsync(cond.View, 1024, full, steps);
        double num = 0, den = 0; int agree = 0;
        for (int r = 0; r < steps; r++)
        {
            var ok = new HashSet<int>(allowed[r]); int best = -1; float bv = float.NegativeInfinity;
            for (int v = 0; v < V; v++)
            {
                num += (double)(got[r * V + v] - gold[r * V + v]) * (got[r * V + v] - gold[r * V + v]); den += (double)gold[r * V + v] * gold[r * V + v];
                if (ok.Contains(v) && got[r * V + v] > bv) { bv = got[r * V + v]; best = v; }
            }
            if (best == (r + 2 < full.Length ? full[r + 2] : UniRigTokenizer.Eos)) agree++;
        }
        double rel = Math.Sqrt(num / den);
        Ur_Expect(rel < 0.05, $"decoder logits relRMS {rel:E2} vs the reference");
        Ur_Expect(agree >= steps - 2, $"the grammar-constrained argmax reproduced {agree}/{steps} generated tokens (near-ties may flip one or two under bf16)");
        Console.WriteLine($"[UniRig] decoder relRMS {rel:E2}, {agree}/{steps} tokens on {accelerator.AcceleratorType}");
    });

    /// <summary>Encoder + decoder + grammar, greedy, on the golden's points: the SAME skeleton as the reference's (joints, tree, positions within a few coordinate bins).</summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task UniRig_Skeleton_EndToEndMatchesTheReference() => await RunTest(async accelerator =>
    {
        var dir = Ur_RequireDevice(accelerator); var g = Path.Combine(dir, "golden", "skeleton");
        var (pts, ps) = Ur_Npy<float>(Path.Combine(g, "points.npy")); var (nrm, _) = Ur_Npy<float>(Path.Combine(g, "normals.npy"));
        var (pre, _) = Ur_Npy<long>(Path.Combine(g, "presample_idx.npy")); var (fps, _) = Ur_Npy<long>(Path.Combine(g, "fps_idx.npy"));
        var (bones, bs) = Ur_Npy<float>(Path.Combine(g, "bones.npy")); var (refParents, _) = Ur_Npy<long>(Path.Combine(g, "parents.npy"));
        using var weights = GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "skeleton.safetensors"));
        using var p = accelerator.Allocate1D(pts); using var q = accelerator.Allocate1D(nrm); using var gen = new UniRigSkeletonGenerator(accelerator, weights);
        var r = await gen.GenerateAsync(p.View, q.View, ps[0], fps.Select(i => (int)pre[i]).ToArray());
        const double Bin = 2.0 / 256;
        Ur_Expect(r.Skeleton.Count == bs[0], $"{r.Skeleton.Count} joints, the reference found {bs[0]}");
        double sum = 0, worst = 0;
        for (int j = 0; j < bs[0]; j++)
        {
            Ur_Expect(r.Skeleton.Parents[j] == (int)refParents[j], $"joint {j} has parent {r.Skeleton.Parents[j]}, the reference {refParents[j]}");
            double dx = r.Skeleton.Joints[j * 3] - bones[j * 6 + 3], dy = r.Skeleton.Joints[j * 3 + 1] - bones[j * 6 + 4], dz = r.Skeleton.Joints[j * 3 + 2] - bones[j * 6 + 5];
            double d = Math.Sqrt(dx * dx + dy * dy + dz * dz); sum += d; worst = Math.Max(worst, d);
        }
        Ur_Expect(sum / bs[0] <= Bin && worst <= 6 * Bin, $"joint positions: mean error {sum / bs[0]:F4}, worst {worst:F4} (a coordinate bin is {Bin:F4})");
        Console.WriteLine($"[UniRig] skeleton: {r.Skeleton.Count} joints, mean error {sum / bs[0]:F4}, worst {worst:F4} on {accelerator.AcceleratorType}");
    });

    /// <summary>The whole rigger on a TRELLIS-generated humanoid GLB: a valid skinned glTF whose right arm follows a posed shoulder and whose other side stays still.</summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task UniRig_Rigger_RigsAGeneratedHumanoid() => await RunTest(async accelerator =>
    {
        var dir = Ur_RequireDevice(accelerator);
        var humanoid = Path.Combine(dir, "inputs", "humanoid.glb");
        if (!File.Exists(humanoid)) throw new UnsupportedTestException("inputs/humanoid.glb (a TRELLIS.2 generated T-pose humanoid) not found");
        var glb = File.ReadAllBytes(humanoid);
        using var weights = GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "skeleton.safetensors"));
        using var rigger = new UniRigRigger(accelerator, weights);
        var res = await rigger.RigGlbAsync(glb, seed: 0);
        var sk = res.Skeleton;
        Ur_Expect(sk.Count is >= 15 and <= 60, $"{sk.Count} joints for a humanoid");
        Ur_Expect(sk.Parents.Count(p => p < 0) == 1 && sk.Parents.Select((p, i) => p < i).All(b => b), "one root, parents before children");

        var mesh = RiggedGlb.ReadMesh(res.Glb); var orig = RiggedGlb.ReadMesh(glb);
        Ur_Expect(mesh.Positions.SequenceEqual(orig.Positions), "the rigged file keeps the mesh");
        int V = mesh.Positions.Length / 3, J = sk.Count;
        float lo = mesh.Positions.Where((_, i) => i % 3 == 1).Min(), hi = mesh.Positions.Where((_, i) => i % 3 == 1).Max();
        for (int j = 0; j < J; j++)
            Ur_Expect(sk.Joints[j * 3 + 1] > lo - 0.05f * (hi - lo) && sk.Joints[j * 3 + 1] < hi + 0.05f * (hi - lo), $"joint {j} lies within the model's height");

        // read the skin back out of the file and pose it: rotate the outermost shoulder on +x
        int jsonLen = (int)BitConverter.ToUInt32(res.Glb, 12), bin = 20 + jsonLen + 8;
        using var doc = JsonDocument.Parse(res.Glb.AsMemory(20, jsonLen)); var root = doc.RootElement;
        var prim = root.GetProperty("meshes")[0].GetProperty("primitives")[0].GetProperty("attributes");
        int Off(int a) { var acc = root.GetProperty("accessors")[a]; return bin + root.GetProperty("bufferViews")[acc.GetProperty("bufferView").GetInt32()].GetProperty("byteOffset").GetInt32(); }
        int jo = Off(prim.GetProperty("JOINTS_0").GetInt32()), wo = Off(prim.GetProperty("WEIGHTS_0").GetInt32());
        for (int v = 0; v < V; v++)
        {
            float s = 0; for (int k = 0; k < 4; k++) s += BitConverter.ToSingle(res.Glb, wo + v * 16 + k * 4);
            Ur_Expect(Math.Abs(s - 1) < 1e-4, $"vertex {v} weights sum to {s}");
        }
        float span = sk.Joints.Where((_, i) => i % 3 == 0).Max();
        int shoulder = -1;
        for (int j = 0; j < J && shoulder < 0; j++) if (sk.Joints[j * 3] > 0.2f * span && sk.Parents[j] >= 0 && sk.Joints[sk.Parents[j] * 3] <= 0.2f * span) shoulder = j;
        Ur_Expect(shoulder >= 0, "a joint out on the right arm");
        var follows = new bool[J]; follows[shoulder] = true; for (int j = 0; j < J; j++) if (sk.Parents[j] >= 0 && follows[sk.Parents[j]]) follows[j] = true;
        float height = hi - lo; int armVerts = 0, leftMoved = 0, bodyMoved = 0;
        for (int v = 0; v < V; v++)
        {
            float follow = 0; for (int k = 0; k < 4; k++) if (follows[BitConverter.ToUInt16(res.Glb, jo + v * 8 + k * 2)]) follow += BitConverter.ToSingle(res.Glb, wo + v * 16 + k * 4);
            float x = mesh.Positions[v * 3];
            if (follow > 0.5f) armVerts++;
            if (x < -0.1f * height && follow > 0.01f) leftMoved++;
            if (Math.Abs(x) < 0.08f * height && follow > 0.25f) bodyMoved++;
        }
        Ur_Expect(armVerts > V * 0.05, $"only {armVerts} of {V} vertices follow the right arm");
        Ur_Expect(leftMoved == 0, $"{leftMoved} vertices on the left half would move with the right shoulder");
        Ur_Expect(bodyMoved < V * 0.01, $"{bodyMoved} vertices of the torso, head and legs would move noticeably with the right shoulder");
        Console.WriteLine($"[UniRig] rigged humanoid: {J} joints, {armVerts} of {V} vertices follow the right arm, left half {leftMoved}, body {bodyMoved} on {accelerator.AcceleratorType}");
    });
}
