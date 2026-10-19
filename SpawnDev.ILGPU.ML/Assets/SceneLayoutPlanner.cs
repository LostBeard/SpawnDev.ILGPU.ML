using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;

namespace SpawnDev.ILGPU.ML.Assets;

/// <summary>
/// Turns a sentence ("a blacksmith's workshop") into a <see cref="SceneLayout"/> with a local LLM: the model is asked for the objects the scene needs, how tall each is, and where they stand,
/// as JSON; the answer is parsed tolerantly and, when it is not usable, the model is asked again with the reason. The chat function is a parameter (<c>(system, user) -> answer</c>) so any
/// local model works; <see cref="FromPipeline"/> wraps a <see cref="GgufTextGenerationPipeline"/> (Gemma, Qwen, Llama...).
/// </summary>
public sealed class SceneLayoutPlanner
{
    private readonly Func<string, string, Task<string>> _chat;

    public SceneLayoutPlanner(Func<string, string, Task<string>> chat) => _chat = chat;

    /// <summary>How many times the model is asked before giving up (the first try plus corrections).</summary>
    public int MaxAttempts { get; set; } = 3;

    /// <summary>A planner on a loaded GGUF model. Reasoning is switched off (the answer is a JSON object, not an essay) and the answer is greedy and capped.</summary>
    public static SceneLayoutPlanner FromPipeline(GgufTextGenerationPipeline pipeline, int maxNewTokens = 1800)
        => new((system, user) => pipeline.GenerateAsync(new List<(string, string)> { ("system", system), ("user", user) },
            new GenerationConfig { MaxNewTokens = maxNewTokens, Strategy = "greedy" }, thinking: false));

    public const string SystemPrompt =
        "You are a 3D scene layout planner. Given a short description of a scene, you decide which objects it contains and place them on the ground. " +
        "Reply with ONE JSON object and nothing else (no explanation, no code fence), in exactly this shape:\n" + SceneLayout.SchemaExample + "\n" +
        "Rules: units are metres; the ground is the plane y = 0 centred on the origin, x runs right and z toward the viewer; every object stands on the ground at (x, z); " +
        "\"height\" is the real-world height of the object in metres (a chair 0.9, a door 2.0, a barrel 0.9, a tree 6); \"rotation\" is degrees about the vertical axis; " +
        "\"prompt\" describes ONE object on its own, as if photographed alone on an empty floor (what it is, its material and style): it must not mention any other object, a table or bench it stands on, a room or a scene; " +
        "everything stands directly on the ground, nothing is placed on top of another object; " +
        "describe each object by what it LOOKS like in plain words (shape, material, colour, a few distinctive parts) and avoid rare or technical names a picture model may not know: write \"a hand-operated air pump shaped like an accordion, brown leather folds between two wooden boards with a long iron nozzle\", not \"a bellows\"; " +
        "an object that appears several times is ONE entry with several instances; keep objects inside the ground and do not overlap them; " +
        "group objects the way a real place would be arranged; use few distinct objects and several instances of the repeated ones; the ground colour is a hex colour that fits the place.";

    /// <param name="description">what the scene is</param>
    /// <param name="distinctObjects">about how many different kinds of object to use (each one costs a 3D generation)</param>
    public async Task<SceneLayout> PlanAsync(string description, int distinctObjects = 6)
    {
        string user = $"Scene: {description}\nUse about {distinctObjects} different kinds of object (more instances of the repeated ones are fine). Answer with the JSON object only.";
        string? problem = null;
        for (int attempt = 1; attempt <= Math.Max(1, MaxAttempts); attempt++)
        {
            string ask = problem == null ? user : user + $"\n\nYour previous answer could not be used: {problem} Answer again with ONLY the JSON object.";
            var answer = await _chat(SystemPrompt, ask);
            try
            {
                var layout = SceneLayout.Parse(answer);
                return layout.Name == "scene" ? new SceneLayout { Name = description.Length > 60 ? description[..60] : description, Ground = layout.Ground, Objects = layout.Objects } : layout;
            }
            catch (FormatException e) { problem = e.Message; }
        }
        throw new FormatException($"The model did not produce a usable layout after {MaxAttempts} attempts: {problem}");
    }
}
