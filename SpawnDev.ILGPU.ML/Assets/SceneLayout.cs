using System.Globalization;
using System.Text;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Assets;

/// <summary>One placement of an object: where it stands on the ground (metres, x right, z toward the viewer), how it is turned about the vertical axis, and a size factor on top of the object's height.</summary>
public sealed record SceneInstance(float X, float Z, float RotationYDegrees = 0f, float Scale = 1f);

/// <summary>One kind of object in a scene: a name, the prompt its 3D asset is generated from, how tall it is in metres, and where it stands (several instances share one generated asset).</summary>
public sealed record SceneObject(string Name, string Prompt, float Height, IReadOnlyList<SceneInstance> Instances);

/// <summary>The ground the scene stands on: size in metres (centred on the origin) and a flat colour (hex, e.g. "#6b7f4e").</summary>
public sealed record SceneGround(float Width, float Depth, string Color);

/// <summary>
/// A scene as data: a name, a ground and objects with placements. The LLM planner (<see cref="SceneLayoutPlanner"/>) writes it as JSON, <see cref="GltfSceneComposer"/> turns it and the generated assets
/// into one glTF scene. Units are metres, Y is up, the ground is the plane y = 0 centred on the origin.
/// </summary>
public sealed class SceneLayout
{
    public string Name { get; init; } = "scene";
    public SceneGround Ground { get; init; } = new(10, 10, "#7a8a5a");
    public IReadOnlyList<SceneObject> Objects { get; init; } = Array.Empty<SceneObject>();

    public const float MinHeight = 0.05f, MaxHeight = 30f;
    public const int MaxObjects = 24, MaxInstancesPerObject = 64;

    /// <summary>The JSON the planner is asked for (and <see cref="ToJson"/> writes).</summary>
    public const string SchemaExample = """
        {
          "name": "blacksmith workshop",
          "ground": { "width": 10, "depth": 8, "color": "#8a7a62" },
          "objects": [
            { "name": "anvil", "prompt": "a heavy iron blacksmith anvil on a short oak stump", "height": 0.9,
              "instances": [ { "x": 0.5, "z": 0.0, "rotation": 20 } ] },
            { "name": "barrel", "prompt": "a wooden barrel with iron hoops", "height": 0.9,
              "instances": [ { "x": -3.2, "z": -2.5, "rotation": 0 }, { "x": -2.4, "z": -2.9, "rotation": 40 } ] }
          ]
        }
        """;

    /// <summary>
    /// Read a layout from the text an LLM wrote: it may be wrapped in a code fence or in prose, and a reasoning model's <c>&lt;think&gt;</c> block is dropped. Missing optional fields get defaults,
    /// heights and counts are clamped to sane limits. Throws <see cref="FormatException"/> with a message the model can act on when there is nothing usable.
    /// </summary>
    public static SceneLayout Parse(string text)
    {
        var json = ExtractJsonObject(text) ?? throw new FormatException("The answer holds no JSON object.");
        JsonDocument doc;
        try { doc = JsonDocument.Parse(json, new JsonDocumentOptions { AllowTrailingCommas = true, CommentHandling = JsonCommentHandling.Skip }); }
        catch (JsonException e) { throw new FormatException("The JSON is malformed: " + e.Message); }
        using (doc)
        {
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object) throw new FormatException("The JSON must be an object with an \"objects\" array.");
            string name = Str(root, "name") ?? "scene";
            var ground = new SceneGround(10, 10, "#7a8a5a");
            if (root.TryGetProperty("ground", out var g) && g.ValueKind == JsonValueKind.Object)
                ground = new SceneGround(Clamp(Num(g, "width", 10), 1, 500), Clamp(Num(g, "depth", 10), 1, 500), ValidColor(Str(g, "color")) ?? "#7a8a5a");
            if (!root.TryGetProperty("objects", out var objs) || objs.ValueKind != JsonValueKind.Array) throw new FormatException("The JSON has no \"objects\" array.");
            var list = new List<SceneObject>();
            foreach (var o in objs.EnumerateArray())
            {
                if (o.ValueKind != JsonValueKind.Object) continue;
                var prompt = Str(o, "prompt") ?? Str(o, "description");
                var oname = Str(o, "name") ?? prompt;
                if (string.IsNullOrWhiteSpace(prompt) || string.IsNullOrWhiteSpace(oname)) continue;
                float height = Clamp(Num(o, "height", 1f), MinHeight, MaxHeight);
                var inst = new List<SceneInstance>();
                if (o.TryGetProperty("instances", out var ia) && ia.ValueKind == JsonValueKind.Array)
                    foreach (var i in ia.EnumerateArray())
                    {
                        if (i.ValueKind != JsonValueKind.Object) continue;
                        inst.Add(new SceneInstance(Clamp(Num(i, "x", 0), -1000, 1000), Clamp(Num(i, "z", 0), -1000, 1000), Num(i, "rotation", Num(i, "rotationY", 0)), Clamp(Num(i, "scale", 1), 0.1f, 10f)));
                        if (inst.Count >= MaxInstancesPerObject) break;
                    }
                else inst.Add(new SceneInstance(Clamp(Num(o, "x", 0), -1000, 1000), Clamp(Num(o, "z", 0), -1000, 1000), Num(o, "rotation", 0)));
                if (inst.Count == 0) continue;
                list.Add(new SceneObject(oname.Trim(), prompt.Trim(), height, inst));
                if (list.Count >= MaxObjects) break;
            }
            if (list.Count == 0) throw new FormatException("The layout has no usable objects (each needs a \"prompt\" and at least one instance).");
            return new SceneLayout { Name = name, Ground = ground, Objects = list };
        }
    }

    /// <summary>The layout as JSON (the format <see cref="Parse"/> reads).</summary>
    public string ToJson()
    {
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = true }))
        {
            w.WriteStartObject();
            w.WriteString("name", Name);
            w.WriteStartObject("ground"); w.WriteNumber("width", Ground.Width); w.WriteNumber("depth", Ground.Depth); w.WriteString("color", Ground.Color); w.WriteEndObject();
            w.WriteStartArray("objects");
            foreach (var o in Objects)
            {
                w.WriteStartObject();
                w.WriteString("name", o.Name); w.WriteString("prompt", o.Prompt); w.WriteNumber("height", o.Height);
                w.WriteStartArray("instances");
                foreach (var i in o.Instances) { w.WriteStartObject(); w.WriteNumber("x", i.X); w.WriteNumber("z", i.Z); w.WriteNumber("rotation", i.RotationYDegrees); w.WriteNumber("scale", i.Scale); w.WriteEndObject(); }
                w.WriteEndArray();
                w.WriteEndObject();
            }
            w.WriteEndArray();
            w.WriteEndObject();
        }
        return Encoding.UTF8.GetString(ms.ToArray());
    }

    // ---- helpers ------------------------------------------------------------------------------------------------

    private static string? Str(JsonElement e, string name) => e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    private static float Num(JsonElement e, string name, float fallback)
    {
        if (!e.TryGetProperty(name, out var v)) return fallback;
        if (v.ValueKind == JsonValueKind.Number && v.TryGetSingle(out var f) && float.IsFinite(f)) return f;
        if (v.ValueKind == JsonValueKind.String && float.TryParse(v.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var s) && float.IsFinite(s)) return s;
        return fallback;
    }

    private static float Clamp(float v, float lo, float hi) => Math.Min(hi, Math.Max(lo, v));

    private static string? ValidColor(string? c)
    {
        if (c == null) return null;
        c = c.Trim();
        if (c.Length == 7 && c[0] == '#' && c.Skip(1).All(Uri.IsHexDigit)) return c.ToLowerInvariant();
        return null;
    }

    /// <summary>The first balanced <c>{ ... }</c> in the text, after dropping a reasoning model's think block.</summary>
    internal static string? ExtractJsonObject(string text)
    {
        if (string.IsNullOrEmpty(text)) return null;
        int think = text.LastIndexOf("</think>", StringComparison.OrdinalIgnoreCase);
        if (think >= 0) text = text[(think + 8)..];
        int start = text.IndexOf('{');
        while (start >= 0)
        {
            int depth = 0; bool inString = false, escape = false;
            for (int i = start; i < text.Length; i++)
            {
                char c = text[i];
                if (inString) { if (escape) escape = false; else if (c == '\\') escape = true; else if (c == '"') inString = false; continue; }
                if (c == '"') inString = true;
                else if (c == '{') depth++;
                else if (c == '}' && --depth == 0) return text.Substring(start, i - start + 1);
            }
            start = text.IndexOf('{', start + 1);           // an unbalanced brace in prose: try the next one
        }
        return null;
    }
}
