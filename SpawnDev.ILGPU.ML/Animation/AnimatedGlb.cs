using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using SpawnDev.ILGPU.ML.UniRig;

namespace SpawnDev.ILGPU.ML.Animation;

/// <summary>
/// Adds glTF animations to a rigged GLB (<see cref="RiggedGlb"/>): one animation per <see cref="RetargetedClip"/>, a rotation channel for every joint node and a translation
/// channel for the root, sampled at the clip's frames with linear interpolation (viewers slerp the rotations). The mesh, skin and everything else are carried over byte for byte.
/// </summary>
public static class AnimatedGlb
{
    private static JsonNode Num(int v) => JsonValue.Create(v)!;
    private static JsonNode Num(float v) => JsonValue.Create(v)!;

    public static byte[] Add(byte[] riggedGlb, IReadOnlyList<RetargetedClip> clips)
    {
        if (clips.Count == 0) return riggedGlb;
        var (root, bin) = RiggedGlb.Parse(riggedGlb);
        var skin = root["skins"]?[0]?.AsObject() ?? throw new InvalidOperationException("the GLB has no skin: rig it first");
        var joints = skin["joints"]!.AsArray().Select(j => j!.GetValue<int>()).ToArray();
        int J = joints.Length;
        var jointOf = new HashSet<int>(joints);
        // the root is the joint no other joint lists as a child
        var nodes = root["nodes"]!.AsArray();
        var childOf = new HashSet<int>();
        foreach (int n in joints) if (nodes[n]!["children"] is JsonArray ch) foreach (var c in ch) childOf.Add(c!.GetValue<int>());
        int rootJoint = Array.FindIndex(joints, n => !childOf.Contains(n));

        // Sized for everything that gets appended, so the stream never grows: a growing MemoryStream asks for double its size in one piece, and in the browser's
        // fragmented 32-bit heap a 90 MB binary chunk (a 1M-triangle model with a 4096 texture) could not get its 180 MB ("Out of memory" in MemoryStream.set_Capacity).
        long extra = 4;
        foreach (var clip in clips) extra += (long)clip.Frames * (4 + J * 16 + 12) + 4L * (J + 2);
        var data = new MemoryStream(checked((int)(bin.Length + extra))); data.Write(bin);
        int AddView(ReadOnlySpan<byte> bytes)
        {
            while (data.Length % 4 != 0) data.WriteByte(0);
            var views = root["bufferViews"]!.AsArray();
            views.Add((JsonNode)new JsonObject { ["buffer"] = 0, ["byteOffset"] = (int)data.Length, ["byteLength"] = bytes.Length });
            data.Write(bytes);
            return views.Count - 1;
        }
        int AddAccessor(int view, int count, string type, float[]? min = null, float[]? max = null)
        {
            var acc = new JsonObject { ["bufferView"] = view, ["componentType"] = 5126, ["count"] = count, ["type"] = type };
            if (min != null) acc["min"] = new JsonArray(min.Select(Num).ToArray());
            if (max != null) acc["max"] = new JsonArray(max.Select(Num).ToArray());
            root["accessors"]!.AsArray().Add((JsonNode)acc);
            return root["accessors"]!.AsArray().Count - 1;
        }

        var animations = root["animations"]?.AsArray() ?? new JsonArray();
        foreach (var clip in clips)
        {
            if (clip.Bones != J) throw new ArgumentException($"clip '{clip.Name}' animates {clip.Bones} bones, the skin has {J} joints");
            int T = clip.Frames;
            int time = AddAccessor(AddView(MemoryMarshal.AsBytes(clip.Times.AsSpan())), T, "SCALAR", new[] { clip.Times[0] }, new[] { clip.Times[^1] });
            var samplers = new JsonArray(); var channels = new JsonArray();
            var track = new float[T * 4];
            for (int j = 0; j < J; j++)
            {
                for (int t = 0; t < T; t++) Array.Copy(clip.Rotations, (t * J + j) * 4, track, t * 4, 4);
                int acc = AddAccessor(AddView(MemoryMarshal.AsBytes(track.AsSpan())), T, "VEC4");
                samplers.Add((JsonNode)new JsonObject { ["input"] = time, ["output"] = acc, ["interpolation"] = "LINEAR" });
                channels.Add((JsonNode)new JsonObject { ["sampler"] = samplers.Count - 1, ["target"] = new JsonObject { ["node"] = joints[j], ["path"] = "rotation" } });
            }
            int tr = AddAccessor(AddView(MemoryMarshal.AsBytes(clip.RootTranslation.AsSpan())), T, "VEC3");
            samplers.Add((JsonNode)new JsonObject { ["input"] = time, ["output"] = tr, ["interpolation"] = "LINEAR" });
            channels.Add((JsonNode)new JsonObject { ["sampler"] = samplers.Count - 1, ["target"] = new JsonObject { ["node"] = joints[rootJoint], ["path"] = "translation" } });
            animations.Add((JsonNode)new JsonObject { ["name"] = clip.Name, ["samplers"] = samplers, ["channels"] = channels });
        }
        root["animations"] = animations;
        while (data.Length % 4 != 0) data.WriteByte(0);
        root["buffers"]![0]!["byteLength"] = (int)data.Length;
        return RiggedGlb.Assemble(root, data);
    }
}
