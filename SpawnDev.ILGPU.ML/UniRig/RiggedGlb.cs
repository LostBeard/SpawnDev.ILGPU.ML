using System.Buffers.Binary;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace SpawnDev.ILGPU.ML.UniRig;

/// <summary>
/// Reads the first mesh of a binary glTF and adds a skeleton to it: joint nodes in the bone hierarchy, a skin with inverse bind matrices, and JOINTS_0 / WEIGHTS_0 on the
/// mesh primitive. The rest of the file (textures, material, UVs, normals) is carried over byte for byte: the new data is appended to the binary chunk and the JSON patched.
/// Bones are plain joints (identity rotation, translation relative to the parent), so a pose is a rotation of a joint node; at most four influences per vertex, as glTF allows.
/// </summary>
public static class RiggedGlb
{
    public sealed record Mesh(float[] Positions, int[] Indices);

    // trim-safe numeric nodes (JsonArray.Add<T> and implicit conversions are not)
    private static JsonNode Num(int v) => JsonValue.Create(v)!;
    private static JsonNode Num(float v) => JsonValue.Create(v)!;

    private static (JsonObject Root, byte[] Bin) Parse(byte[] glb)
    {
        if (glb.Length < 20 || BinaryPrimitives.ReadUInt32LittleEndian(glb) != 0x46546C67) throw new InvalidDataException("not a binary glTF");
        int jsonLen = (int)BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(12));
        var root = JsonNode.Parse(glb.AsSpan(20, jsonLen))!.AsObject();
        int binAt = 20 + jsonLen;
        int binLen = binAt + 8 <= glb.Length ? (int)BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(binAt)) : 0;
        return (root, glb.AsSpan(binAt + 8, binLen).ToArray());
    }

    private static (int Offset, int Count, int ComponentType) Accessor(JsonObject root, int index)
    {
        var a = root["accessors"]![index]!.AsObject();
        var bv = root["bufferViews"]![a["bufferView"]!.GetValue<int>()]!.AsObject();
        int offset = (bv["byteOffset"]?.GetValue<int>() ?? 0) + (a["byteOffset"]?.GetValue<int>() ?? 0);
        return (offset, a["count"]!.GetValue<int>(), a["componentType"]!.GetValue<int>());
    }

    /// <summary>Positions (xyz floats) and triangle indices of the first primitive of the first mesh.</summary>
    public static Mesh ReadMesh(byte[] glb)
    {
        var (root, bin) = Parse(glb);
        var prim = root["meshes"]![0]!["primitives"]![0]!.AsObject();
        var (po, pc, _) = Accessor(root, prim["attributes"]!["POSITION"]!.GetValue<int>());
        var pos = new float[pc * 3];
        Buffer.BlockCopy(bin, po, pos, 0, pc * 12);
        var (io, ic, ict) = Accessor(root, prim["indices"]!.GetValue<int>());
        var idx = new int[ic];
        for (int i = 0; i < ic; i++)
            idx[i] = ict == 5123 ? BinaryPrimitives.ReadUInt16LittleEndian(bin.AsSpan(io + i * 2)) : (int)BinaryPrimitives.ReadUInt32LittleEndian(bin.AsSpan(io + i * 4));
        return new Mesh(pos, idx);
    }

    /// <param name="glb">The mesh file.</param>
    /// <param name="skeleton">Joint positions in the GLB's own space and the parent of each (parents precede their children).</param>
    /// <param name="weights">[V * J] weight of GLB vertex v for joint j (any non-negative numbers; the four largest are kept and renormalised).</param>
    public static byte[] Add(byte[] glb, UniRigTokenizer.Skeleton skeleton, float[] weights)
    {
        var (root, bin) = Parse(glb);
        var prim = root["meshes"]![0]!["primitives"]![0]!.AsObject();
        int V = Accessor(root, prim["attributes"]!["POSITION"]!.GetValue<int>()).Count, J = skeleton.Count;
        if (weights.Length != V * J) throw new ArgumentException($"weights hold {weights.Length} values, expected {V} vertices x {J} joints");
        if (prim["attributes"]!.AsObject().ContainsKey("JOINTS_0")) throw new InvalidOperationException("the mesh is already skinned");

        // four strongest influences per vertex, renormalised
        var joints = new byte[V * 8]; var w4 = new float[V * 4];
        Span<int> top = stackalloc int[4]; Span<float> topW = stackalloc float[4];
        for (int v = 0; v < V; v++)
        {
            top.Clear(); topW.Clear();
            for (int j = 0; j < J; j++)
            {
                float w = weights[v * J + j];
                for (int k = 0; k < 4; k++)
                    if (w > topW[k]) { for (int m = 3; m > k; m--) { top[m] = top[m - 1]; topW[m] = topW[m - 1]; } top[k] = j; topW[k] = w; break; }
            }
            float sum = topW[0] + topW[1] + topW[2] + topW[3];
            if (sum <= 0) { top[0] = 0; topW[0] = 1; sum = 1; }
            for (int k = 0; k < 4; k++)
            {
                BinaryPrimitives.WriteUInt16LittleEndian(joints.AsSpan(v * 8 + k * 2), (ushort)(topW[k] > 0 ? top[k] : 0));
                w4[v * 4 + k] = topW[k] / sum;
            }
        }
        // inverse bind matrices: the bind pose has no rotation, so each is a translation by minus the joint position (column-major)
        var ibm = new float[J * 16];
        for (int j = 0; j < J; j++)
        {
            ibm[j * 16] = ibm[j * 16 + 5] = ibm[j * 16 + 10] = ibm[j * 16 + 15] = 1f;
            ibm[j * 16 + 12] = -skeleton.Joints[j * 3]; ibm[j * 16 + 13] = -skeleton.Joints[j * 3 + 1]; ibm[j * 16 + 14] = -skeleton.Joints[j * 3 + 2];
        }

        var data = new MemoryStream(); data.Write(bin);
        int AddView(ReadOnlySpan<byte> bytes, int? target)
        {
            while (data.Length % 4 != 0) data.WriteByte(0);
            var views = root["bufferViews"]!.AsArray();
            var view = new JsonObject { ["buffer"] = 0, ["byteOffset"] = (int)data.Length, ["byteLength"] = bytes.Length };
            if (target.HasValue) view["target"] = target.Value;
            views.Add((JsonNode)view);
            data.Write(bytes);
            return views.Count - 1;
        }
        int AddAccessor(int view, int componentType, int count, string type)
        {
            var acc = root["accessors"]!.AsArray();
            acc.Add((JsonNode)new JsonObject { ["bufferView"] = view, ["componentType"] = componentType, ["count"] = count, ["type"] = type });
            return acc.Count - 1;
        }
        int jointsAcc = AddAccessor(AddView(joints, 34962), 5123, V, "VEC4");
        int weightsAcc = AddAccessor(AddView(System.Runtime.InteropServices.MemoryMarshal.AsBytes(w4.AsSpan()), 34962), 5126, V, "VEC4");
        int ibmAcc = AddAccessor(AddView(System.Runtime.InteropServices.MemoryMarshal.AsBytes(ibm.AsSpan()), null), 5126, J, "MAT4");
        while (data.Length % 4 != 0) data.WriteByte(0);
        root["buffers"]![0]!["byteLength"] = (int)data.Length;
        prim["attributes"]!["JOINTS_0"] = jointsAcc; prim["attributes"]!["WEIGHTS_0"] = weightsAcc;

        // joint nodes: translation relative to the parent; an "Armature" node parents the roots
        var nodes = root["nodes"]!.AsArray();
        int firstJoint = nodes.Count;
        var children = new List<int>[J]; for (int j = 0; j < J; j++) children[j] = new List<int>();
        var roots = new List<int>();
        for (int j = 0; j < J; j++) { int p = skeleton.Parents[j]; if (p < 0) roots.Add(firstJoint + j); else children[p].Add(firstJoint + j); }
        for (int j = 0; j < J; j++)
        {
            int p = skeleton.Parents[j];
            float tx = skeleton.Joints[j * 3] - (p < 0 ? 0 : skeleton.Joints[p * 3]), ty = skeleton.Joints[j * 3 + 1] - (p < 0 ? 0 : skeleton.Joints[p * 3 + 1]), tz = skeleton.Joints[j * 3 + 2] - (p < 0 ? 0 : skeleton.Joints[p * 3 + 2]);
            var node = new JsonObject { ["name"] = $"bone_{j}", ["translation"] = new JsonArray(Num(tx), Num(ty), Num(tz)) };
            if (children[j].Count > 0) node["children"] = new JsonArray(children[j].Select(Num).ToArray());
            nodes.Add((JsonNode)node);
        }
        int armature = nodes.Count;
        nodes.Add((JsonNode)new JsonObject { ["name"] = "Armature", ["children"] = new JsonArray(roots.Select(Num).ToArray()) });
        var skins = root["skins"]?.AsArray() ?? new JsonArray();
        skins.Add((JsonNode)new JsonObject { ["name"] = "skeleton", ["inverseBindMatrices"] = ibmAcc, ["skeleton"] = armature, ["joints"] = new JsonArray(Enumerable.Range(firstJoint, J).Select(Num).ToArray()) });
        root["skins"] = skins;
        // the node that carries the mesh gets the skin; the armature joins the scene
        foreach (var n in nodes) if (n is JsonObject o && o["mesh"] != null) { o["skin"] = skins.Count - 1; break; }
        root["scenes"]![0]!["nodes"]!.AsArray().Add(Num(armature));

        var json = new MemoryStream(); using (var w = new Utf8JsonWriter(json)) root.WriteTo(w);
        while (json.Length % 4 != 0) json.WriteByte(0x20);
        int total = 12 + 8 + (int)json.Length + 8 + (int)data.Length;
        var o2 = new MemoryStream(total); var u = new byte[4];
        void U32(uint x) { BinaryPrimitives.WriteUInt32LittleEndian(u, x); o2.Write(u); }
        U32(0x46546C67); U32(2); U32((uint)total);
        U32((uint)json.Length); U32(0x4E4F534A); o2.Write(json.GetBuffer().AsSpan(0, (int)json.Length));
        U32((uint)data.Length); U32(0x004E4942); o2.Write(data.GetBuffer().AsSpan(0, (int)data.Length));
        return o2.ToArray();
    }
}
