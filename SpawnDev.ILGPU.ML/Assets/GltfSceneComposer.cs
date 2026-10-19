using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace SpawnDev.ILGPU.ML.Assets;

/// <summary>
/// Puts generated assets into one glTF scene: every object kind of a <see cref="SceneLayout"/> is copied in once (meshes, materials, textures, images, buffers, re-indexed) and referenced by
/// one node per instance, so a barrel that stands four times is stored once. Each instance is scaled so the asset is as tall as the layout says, stood on the ground (its lowest point at y = 0,
/// centred on its own footprint) and turned about the vertical axis. A flat ground quad of the layout's size and colour is added. Static meshes only (skins and animations of an asset are not carried over).
/// </summary>
public static class GltfSceneComposer
{
    /// <param name="layout">what stands where</param>
    /// <param name="assets">the generated GLB of each object, by <see cref="SceneObject.Name"/>; an object without one is skipped</param>
    public static byte[] Compose(SceneLayout layout, IReadOnlyDictionary<string, byte[]> assets)
    {
        var buffers = new JsonArray(); var bufferViews = new JsonArray(); var accessors = new JsonArray(); var meshes = new JsonArray();
        var materials = new JsonArray(); var textures = new JsonArray(); var images = new JsonArray(); var samplers = new JsonArray(); var nodes = new JsonArray();
        var sceneNodes = new JsonArray();
        var bin = new MemoryStream();

        int Align4() { while (bin.Length % 4 != 0) bin.WriteByte(0); return (int)bin.Length; }

        foreach (var obj in layout.Objects)
        {
            if (!assets.TryGetValue(obj.Name, out var glb)) continue;
            var (json, data) = ReadGlb(glb);
            int bvBase = bufferViews.Count, accBase = accessors.Count, matBase = materials.Count, texBase = textures.Count, imgBase = images.Count, smpBase = samplers.Count, meshBase = meshes.Count;
            int binBase = Align4(); bin.Write(data, 0, data.Length);

            foreach (var bv in Arr(json, "bufferViews"))
            {
                var c = bv!.DeepClone().AsObject();
                c["buffer"] = 0; c["byteOffset"] = binBase + (c["byteOffset"]?.GetValue<int>() ?? 0);
                Put(bufferViews, c);
            }
            foreach (var a in Arr(json, "accessors"))
            {
                var c = a!.DeepClone().AsObject();
                if (c.ContainsKey("sparse")) throw new NotSupportedException("A sparse accessor in an asset cannot be merged.");
                if (c.ContainsKey("bufferView")) c["bufferView"] = c["bufferView"]!.GetValue<int>() + bvBase;
                Put(accessors, c);
            }
            foreach (var s in Arr(json, "samplers")) Put(samplers, s!.DeepClone());
            foreach (var im in Arr(json, "images"))
            {
                var c = im!.DeepClone().AsObject();
                if (c.ContainsKey("uri")) throw new NotSupportedException("An asset with an external image cannot be merged.");
                if (c.ContainsKey("bufferView")) c["bufferView"] = c["bufferView"]!.GetValue<int>() + bvBase;
                Put(images, c);
            }
            foreach (var t in Arr(json, "textures"))
            {
                var c = t!.DeepClone().AsObject();
                if (c.ContainsKey("source")) c["source"] = c["source"]!.GetValue<int>() + imgBase;
                if (c.ContainsKey("sampler")) c["sampler"] = c["sampler"]!.GetValue<int>() + smpBase;
                Put(textures, c);
            }
            foreach (var m in Arr(json, "materials")) { var c = m!.DeepClone(); ShiftTextureIndices(c, texBase); Put(materials, c); }
            float minX = float.MaxValue, minY = float.MaxValue, minZ = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue, maxZ = float.MinValue;
            foreach (var m in Arr(json, "meshes"))
            {
                var c = m!.DeepClone().AsObject();
                foreach (var p in c["primitives"]!.AsArray())
                {
                    var prim = p!.AsObject();
                    var attrs = prim["attributes"]!.AsObject();
                    foreach (var key in attrs.Select(k => k.Key).ToList()) attrs[key] = attrs[key]!.GetValue<int>() + accBase;
                    if (prim.ContainsKey("indices")) prim["indices"] = prim["indices"]!.GetValue<int>() + accBase;
                    if (prim.ContainsKey("material")) prim["material"] = prim["material"]!.GetValue<int>() + matBase;
                    var pos = accessors[attrs["POSITION"]!.GetValue<int>()]!.AsObject();
                    var mn = pos["min"]?.AsArray(); var mx = pos["max"]?.AsArray();
                    if (mn == null || mx == null) throw new InvalidDataException("The POSITION accessor of an asset has no min / max.");
                    minX = Math.Min(minX, mn[0]!.GetValue<float>()); minY = Math.Min(minY, mn[1]!.GetValue<float>()); minZ = Math.Min(minZ, mn[2]!.GetValue<float>());
                    maxX = Math.Max(maxX, mx[0]!.GetValue<float>()); maxY = Math.Max(maxY, mx[1]!.GetValue<float>()); maxZ = Math.Max(maxZ, mx[2]!.GetValue<float>());
                }
                Put(meshes, c);
            }
            if (meshes.Count == meshBase) continue;
            float height0 = maxY - minY, cx = (minX + maxX) / 2f, cz = (minZ + maxZ) / 2f;
            int inst = 0;
            foreach (var i in obj.Instances)
            {
                float s = (height0 > 1e-6f ? obj.Height / height0 : 1f) * i.Scale;
                double th = i.RotationYDegrees * Math.PI / 180.0, cos = Math.Cos(th), sin = Math.Sin(th);
                // the footprint centre and the lowest point go to the origin of the node, then the node is turned and put on the ground at (x, z)
                double lx = -s * cx, ly = -s * minY, lz = -s * cz;
                double tx = i.X + lx * cos + lz * sin, ty = ly, tz = i.Z - lx * sin + lz * cos;
                var node = new JsonObject
                {
                    ["name"] = obj.Instances.Count > 1 ? $"{obj.Name} {++inst}" : obj.Name,
                    ["mesh"] = meshBase,
                    ["translation"] = new JsonArray((float)tx, (float)ty, (float)tz),
                    ["rotation"] = new JsonArray(0f, (float)Math.Sin(th / 2), 0f, (float)Math.Cos(th / 2)),
                    ["scale"] = new JsonArray(s, s, s),
                };
                Put(sceneNodes, nodes.Count); Put(nodes, node);
            }
        }

        // ---- the ground: one quad ----------------------------------------------------------------------------------------------------------------------
        {
            float hw = layout.Ground.Width / 2f, hd = layout.Ground.Depth / 2f;
            var verts = new[] { -hw, 0f, -hd, hw, 0f, -hd, hw, 0f, hd, -hw, 0f, hd };
            var norms = new[] { 0f, 1f, 0f, 0f, 1f, 0f, 0f, 1f, 0f, 0f, 1f, 0f };
            var idx = new ushort[] { 0, 2, 1, 0, 3, 2 };
            int pOff = Align4(); foreach (var f in verts) bin.Write(BitConverter.GetBytes(f));
            int nOff = Align4(); foreach (var f in norms) bin.Write(BitConverter.GetBytes(f));
            int iOff = Align4(); foreach (var u in idx) bin.Write(BitConverter.GetBytes(u));
            int bv0 = bufferViews.Count, ac0 = accessors.Count;
            Put(bufferViews, new JsonObject { ["buffer"] = 0, ["byteOffset"] = pOff, ["byteLength"] = 48, ["target"] = 34962 });
            Put(bufferViews, new JsonObject { ["buffer"] = 0, ["byteOffset"] = nOff, ["byteLength"] = 48, ["target"] = 34962 });
            Put(bufferViews, new JsonObject { ["buffer"] = 0, ["byteOffset"] = iOff, ["byteLength"] = 12, ["target"] = 34963 });
            Put(accessors, new JsonObject { ["bufferView"] = bv0, ["componentType"] = 5126, ["count"] = 4, ["type"] = "VEC3", ["min"] = new JsonArray(-hw, 0f, -hd), ["max"] = new JsonArray(hw, 0f, hd) });
            Put(accessors, new JsonObject { ["bufferView"] = bv0 + 1, ["componentType"] = 5126, ["count"] = 4, ["type"] = "VEC3" });
            Put(accessors, new JsonObject { ["bufferView"] = bv0 + 2, ["componentType"] = 5123, ["count"] = 6, ["type"] = "SCALAR" });
            var (r, g, b) = Hex(layout.Ground.Color);
            int mat = materials.Count;
            Put(materials, new JsonObject { ["name"] = "ground", ["doubleSided"] = true, ["pbrMetallicRoughness"] = new JsonObject { ["baseColorFactor"] = new JsonArray(r, g, b, 1f), ["metallicFactor"] = 0f, ["roughnessFactor"] = 1f } });
            int mesh = meshes.Count;
            Put(meshes, new JsonObject { ["name"] = "ground", ["primitives"] = new JsonArray(new JsonObject { ["attributes"] = new JsonObject { ["POSITION"] = ac0, ["NORMAL"] = ac0 + 1 }, ["indices"] = ac0 + 2, ["material"] = mat }) });
            Put(sceneNodes, nodes.Count); Put(nodes, new JsonObject { ["name"] = "ground", ["mesh"] = mesh });
        }

        Align4();
        Put(buffers, new JsonObject { ["byteLength"] = (int)bin.Length });
        var root = new JsonObject
        {
            ["asset"] = new JsonObject { ["version"] = "2.0", ["generator"] = "SpawnDev.ILGPU.ML GltfSceneComposer" },
            ["scene"] = 0,
            ["scenes"] = new JsonArray(new JsonObject { ["name"] = layout.Name, ["nodes"] = sceneNodes, ["extras"] = new JsonObject { ["layout"] = JsonNode.Parse(layout.ToJson()) } }),
            ["nodes"] = nodes, ["meshes"] = meshes, ["materials"] = materials, ["accessors"] = accessors, ["bufferViews"] = bufferViews, ["buffers"] = buffers,
        };
        if (textures.Count > 0) root["textures"] = textures;
        if (images.Count > 0) root["images"] = images;
        if (samplers.Count > 0) root["samplers"] = samplers;
        return WriteGlb(root.ToJsonString(), bin.ToArray());
    }

    // ---- GLB container ---------------------------------------------------------------------------------------------------------------------------------

    /// <summary>The JSON and the binary chunk of a GLB.</summary>
    public static (JsonObject Json, byte[] Bin) ReadGlb(byte[] glb)
    {
        if (glb.Length < 20 || BitConverter.ToUInt32(glb, 0) != 0x46546C67u) throw new InvalidDataException("Not a GLB file.");
        int jsonLen = (int)BitConverter.ToUInt32(glb, 12);
        var json = JsonNode.Parse(Encoding.UTF8.GetString(glb, 20, jsonLen))!.AsObject();
        int off = 20 + jsonLen;
        byte[] bin = Array.Empty<byte>();
        if (off + 8 <= glb.Length)
        {
            int binLen = (int)BitConverter.ToUInt32(glb, off);
            bin = new byte[binLen]; Buffer.BlockCopy(glb, off + 8, bin, 0, binLen);
        }
        return (json, bin);
    }

    public static byte[] WriteGlb(string json, byte[] bin)
    {
        var j = Encoding.UTF8.GetBytes(json);
        int jPad = (4 - j.Length % 4) % 4, bPad = (4 - bin.Length % 4) % 4;
        int total = 12 + 8 + j.Length + jPad + 8 + bin.Length + bPad;
        var o = new byte[total];
        BitConverter.GetBytes(0x46546C67u).CopyTo(o, 0); BitConverter.GetBytes(2u).CopyTo(o, 4); BitConverter.GetBytes((uint)total).CopyTo(o, 8);
        BitConverter.GetBytes((uint)(j.Length + jPad)).CopyTo(o, 12); BitConverter.GetBytes(0x4E4F534Au).CopyTo(o, 16);
        j.CopyTo(o, 20); for (int i = 0; i < jPad; i++) o[20 + j.Length + i] = 0x20;
        int p = 20 + j.Length + jPad;
        BitConverter.GetBytes((uint)(bin.Length + bPad)).CopyTo(o, p); BitConverter.GetBytes(0x004E4942u).CopyTo(o, p + 4);
        bin.CopyTo(o, p + 8);
        return o;
    }

    /// <summary>JsonArray.Add with the non-generic overload (the generic one is not trim safe).</summary>
    private static void Put(JsonArray a, JsonNode n) => a.Add(n);

    private static JsonArray Arr(JsonObject o, string name) => o[name] as JsonArray ?? new JsonArray();

    private static void ShiftTextureIndices(JsonNode node, int texBase)
    {
        if (node is JsonObject o)
        {
            foreach (var kv in o.ToList())
            {
                if (kv.Key.EndsWith("Texture", StringComparison.Ordinal) && kv.Value is JsonObject ti && ti.ContainsKey("index")) ti["index"] = ti["index"]!.GetValue<int>() + texBase;
                else if (kv.Value != null) ShiftTextureIndices(kv.Value, texBase);
            }
        }
        else if (node is JsonArray a) foreach (var e in a) if (e != null) ShiftTextureIndices(e, texBase);
    }

    private static (float R, float G, float B) Hex(string c)
    {
        float V(int i) => int.Parse(c.AsSpan(1 + i * 2, 2), NumberStyles.HexNumber, CultureInfo.InvariantCulture) / 255f;
        // the colour factor is linear in glTF: convert from sRGB
        static float Lin(float s) => s <= 0.04045f ? s / 12.92f : MathF.Pow((s + 0.055f) / 1.055f, 2.4f);
        try { return (Lin(V(0)), Lin(V(1)), Lin(V(2))); } catch { return (0.2f, 0.3f, 0.15f); }
    }
}
