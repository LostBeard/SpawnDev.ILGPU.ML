using System.Text.Json.Nodes;
using SpawnDev.ILGPU.ML.Assets;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// The layout-then-assets scene baseline: <see cref="SceneLayout"/> (what an LLM's answer is parsed into), <see cref="SceneLayoutPlanner"/> (asks again with the reason when the answer is unusable) and
/// <see cref="GltfSceneComposer"/> (assets into one glTF scene). Pure CPU, synthetic data, a fake LLM.
/// </summary>
public abstract partial class MLTestBase
{
    private static void Sc_Expect(bool ok, string what) { if (!ok) throw new Exception(what); }

    [TestMethod]
    public async Task SceneLayout_Parse_ReadsFencedJsonAndThinkBlocks_ClampsAndDefaults() => await RunPureTest(async () =>
    {
        var answer = "<think>I should use { braces } here</think>\nSure! Here it is:\n```json\n" + """
            {
              "name": "yard",
              "ground": { "width": 9999, "depth": 6, "color": "not-a-colour" },
              "objects": [
                { "name": "barrel", "prompt": "a wooden barrel", "height": 500, "instances": [ { "x": 1, "z": 2, "rotation": 30 }, { "x": "3.5", "z": -1 } ] },
                { "name": "no prompt here", "height": 1, "instances": [ { "x": 0, "z": 0 } ] },
                { "prompt": "a lantern", "height": 0.0001, "x": 2, "z": 2 },
                { "name": "empty", "prompt": "nothing placed", "instances": [] }
              ],
            }
            """ + "\n```\nHope that helps.";
        var l = SceneLayout.Parse(answer);
        Sc_Expect(l.Name == "yard", "name");
        Sc_Expect(l.Ground.Width == 500 && l.Ground.Depth == 6 && l.Ground.Color == "#7a8a5a", $"ground clamped, bad colour replaced ({l.Ground})");
        Sc_Expect(l.Objects.Count == 2, $"an object without a prompt and one without instances are dropped ({l.Objects.Count} kept)");
        var barrel = l.Objects[0];
        Sc_Expect(barrel.Height == SceneLayout.MaxHeight && barrel.Instances.Count == 2 && barrel.Instances[1].X == 3.5f && barrel.Instances[0].RotationYDegrees == 30, "height clamped, string numbers read, rotation kept");
        var lantern = l.Objects[1];
        Sc_Expect(lantern.Name == "a lantern" && lantern.Height == SceneLayout.MinHeight && lantern.Instances.Count == 1 && lantern.Instances[0].X == 2, "name defaults to the prompt; a single x/z becomes one instance");
        foreach (var bad in new[] { "no json at all", "{ \"objects\": [ ", "{ \"name\": \"x\" }", "{ \"objects\": [ { \"name\": \"a\" } ] }" })
        {
            try { SceneLayout.Parse(bad); throw new Exception($"'{bad}' must be refused"); } catch (FormatException) { }
        }
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task SceneLayout_RoundTripsThroughJson() => await RunPureTest(async () =>
    {
        var l = new SceneLayout
        {
            Name = "camp", Ground = new SceneGround(12, 8, "#3a5f2b"),
            Objects = new[] { new SceneObject("tent", "a canvas tent", 2.2f, new[] { new SceneInstance(1, 2, 45, 1.5f) }), new SceneObject("log", "a fallen log", 0.4f, new[] { new SceneInstance(-3, 0), new SceneInstance(-2, 1, 90) }) },
        };
        var back = SceneLayout.Parse(l.ToJson());
        Sc_Expect(back.ToJson() == l.ToJson(), "ToJson / Parse is a round trip");
        Sc_Expect(back.Objects[1].Instances.Count == 2 && back.Objects[0].Instances[0].Scale == 1.5f, "instances survive");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task SceneLayoutPlanner_AsksAgainWithTheReason_AndGivesUp() => await RunPureTest(async () =>
    {
        var asked = new List<string>();
        var good = SceneLayout.SchemaExample;
        var planner = new SceneLayoutPlanner((system, user) =>
        {
            asked.Add(user);
            return Task.FromResult(asked.Count == 1 ? "I cannot draw that, sorry." : good);
        });
        var layout = await planner.PlanAsync("a blacksmith's workshop", 5);
        Sc_Expect(asked.Count == 2 && layout.Objects.Count == 2, $"the second try gave a layout ({asked.Count} asks)");
        Sc_Expect(asked[0].Contains("blacksmith") && asked[0].Contains("about 5"), "the first ask names the scene and the object count");
        Sc_Expect(asked[1].Contains("could not be used") && asked[1].Contains("no JSON object"), "the second ask says why the first answer failed");
        Sc_Expect(layout.Name == "blacksmith workshop", "the model's name is kept");

        int calls = 0;
        var hopeless = new SceneLayoutPlanner((_, _) => { calls++; return Task.FromResult("nope"); }) { MaxAttempts = 2 };
        try { await hopeless.PlanAsync("x"); throw new Exception("must give up"); } catch (FormatException) { }
        Sc_Expect(calls == 2, $"it asks MaxAttempts times and no more ({calls})");
        await Task.CompletedTask;
    });

    private static byte[] Sc_BoxGlb(float w, float h, float d)
    {
        // an axis-aligned box centred on the origin, 8 shared corners, 12 triangles, one texture
        var p = new List<float>(); var n = new List<float>(); var uv = new List<float>();
        for (int i = 0; i < 8; i++)
        {
            p.Add((i & 1) == 0 ? -w / 2 : w / 2); p.Add((i & 2) == 0 ? -h / 2 : h / 2); p.Add((i & 4) == 0 ? -d / 2 : d / 2);
            n.AddRange(new[] { 0f, 1f, 0f }); uv.AddRange(new[] { (i & 1) * 1f, ((i >> 1) & 1) * 1f });
        }
        var idx = new[] { 0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6 };
        var png = AssetGenerator.EncodePng(new byte[] { 200, 50, 50, 255, 50, 200, 50, 255, 50, 50, 200, 255, 220, 220, 220, 255 }, 2, 2);
        return PbrGlbWriter.Write(p.ToArray(), n.ToArray(), uv.ToArray(), idx, png, png);
    }

    [TestMethod]
    public async Task GltfSceneComposer_StoresEachAssetOnce_AndPlacesInstancesOnTheGround() => await RunPureTest(async () =>
    {
        var layout = new SceneLayout
        {
            Name = "test", Ground = new SceneGround(10, 8, "#808080"),
            Objects = new[]
            {
                new SceneObject("crate", "a crate", 2f, new[] { new SceneInstance(3, 1, 90), new SceneInstance(-2, 0) }),
                new SceneObject("slab", "a slab", 0.5f, new[] { new SceneInstance(0, -3, 0, 2f) }),
            },
        };
        var glb = GltfSceneComposer.Compose(layout, new Dictionary<string, byte[]> { ["crate"] = Sc_BoxGlb(1, 1, 1), ["slab"] = Sc_BoxGlb(2, 0.25f, 2) });
        var (json, bin) = GltfSceneComposer.ReadGlb(glb);
        Sc_Expect(json["meshes"]!.AsArray().Count == 3, $"two assets and the ground are three meshes, not one per instance ({json["meshes"]!.AsArray().Count})");
        Sc_Expect(json["textures"]!.AsArray().Count == 4 && json["images"]!.AsArray().Count == 4, "each asset's two textures (base colour, metallic-roughness) are carried over once, not once per instance");
        Sc_Expect(json["buffers"]![0]!["byteLength"]!.GetValue<int>() == bin.Length, "the buffer length matches the binary chunk");
        var nodes = json["nodes"]!.AsArray();
        Sc_Expect(nodes.Count == 4 && json["scenes"]![0]!["nodes"]!.AsArray().Count == 4, $"three instances and the ground ({nodes.Count} nodes)");
        foreach (var bv in json["bufferViews"]!.AsArray()) Sc_Expect(bv!["byteOffset"]!.GetValue<int>() % 4 == 0 && bv["byteOffset"]!.GetValue<int>() + bv["byteLength"]!.GetValue<int>() <= bin.Length, "buffer views are aligned and inside the buffer");
        int accCount = json["accessors"]!.AsArray().Count;
        foreach (var m in json["meshes"]!.AsArray())
            foreach (var pr in m!["primitives"]!.AsArray())
            {
                foreach (var a in pr!["attributes"]!.AsObject()) Sc_Expect(a.Value!.GetValue<int>() < accCount, "attribute accessors are re-indexed into range");
                Sc_Expect(pr["indices"]!.GetValue<int>() < accCount && pr["material"]!.GetValue<int>() < json["materials"]!.AsArray().Count, "indices and material are re-indexed into range");
            }

        // world-space box of every instance node: bottom on the ground, as tall as asked, centred where asked
        (float[] Min, float[] Max) World(JsonNode node)
        {
            var mesh = json["meshes"]![node["mesh"]!.GetValue<int>()]!;
            var pos = json["accessors"]![mesh["primitives"]![0]!["attributes"]!["POSITION"]!.GetValue<int>()]!;
            var mn = pos["min"]!.AsArray().Select(v => v!.GetValue<float>()).ToArray(); var mx = pos["max"]!.AsArray().Select(v => v!.GetValue<float>()).ToArray();
            var s = node["scale"]!.AsArray().Select(v => v!.GetValue<float>()).ToArray();
            var t = node["translation"]!.AsArray().Select(v => v!.GetValue<float>()).ToArray();
            var q = node["rotation"]!.AsArray().Select(v => v!.GetValue<float>()).ToArray();
            double th = 2 * Math.Atan2(q[1], q[3]), c = Math.Cos(th), sn = Math.Sin(th);
            var lo = new[] { float.MaxValue, float.MaxValue, float.MaxValue }; var hi = new[] { float.MinValue, float.MinValue, float.MinValue };
            for (int i = 0; i < 8; i++)
            {
                double x = ((i & 1) == 0 ? mn[0] : mx[0]) * s[0], y = ((i & 2) == 0 ? mn[1] : mx[1]) * s[1], z = ((i & 4) == 0 ? mn[2] : mx[2]) * s[2];
                var w = new[] { (float)(x * c + z * sn + t[0]), (float)(y + t[1]), (float)(-x * sn + z * c + t[2]) };
                for (int k = 0; k < 3; k++) { lo[k] = Math.Min(lo[k], w[k]); hi[k] = Math.Max(hi[k], w[k]); }
            }
            return (lo, hi);
        }
        bool Near(float a, float b) => Math.Abs(a - b) < 1e-4f;
        var (a0, a1) = World(nodes[0]!);
        Sc_Expect(Near(a0[1], 0) && Near(a1[1], 2), $"the crate stands on the ground and is 2 m tall (y {a0[1]} .. {a1[1]})");
        Sc_Expect(Near((a0[0] + a1[0]) / 2, 3) && Near((a0[2] + a1[2]) / 2, 1), $"the crate is centred on (3, 1) after a quarter turn ({(a0[0] + a1[0]) / 2}, {(a0[2] + a1[2]) / 2})");
        var (b0, b1) = World(nodes[1]!);
        Sc_Expect(Near((b0[0] + b1[0]) / 2, -2) && Near(b0[1], 0), "the second instance of the same crate is at (-2, 0)");
        var (c0, c1) = World(nodes[2]!);
        Sc_Expect(Near(c0[1], 0) && Near(c1[1], 0.5f * 2) && Near((c0[2] + c1[2]) / 2, -3), $"the slab is 0.5 m x its scale 2 tall at (0, -3) (y top {c1[1]})");
        Sc_Expect(json["scenes"]![0]!["extras"]!["layout"]!["name"]!.GetValue<string>() == "test", "the layout travels with the scene");
        await Task.CompletedTask;
    });
}
