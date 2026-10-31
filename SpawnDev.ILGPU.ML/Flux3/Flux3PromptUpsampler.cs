using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;

namespace SpawnDev.ILGPU.ML.Flux3;

/// <summary>
/// Writes the full scene caption FLUX 3 Image Klein is trained on from a short request, with any local LLM: FLUX 3 sent a short prompt as typed draws a generic picture, which is why Black Forest Labs' own pipeline
/// always runs a prompt upsampler first (their 9B <c>flux-3-image-klein-pup</c>, or "any LLM you prefer with a system prompt that expands the request into one full scene caption"). The system prompt here is theirs
/// (<c>system_prompts.json</c> of black-forest-labs/flux-3-image-klein-pup, Apache-2.0); the model is a parameter, so a 4B instruct model that fits a browser tab does the job (their 9B is the reference). Text to image only:
/// the edit prompt needs a vision-language model that sees the reference pictures.
/// </summary>
public sealed class Flux3PromptUpsampler
{
    private readonly Func<string, string, Func<int, string, Task>?, Task<string>> _chat;

    public Flux3PromptUpsampler(Func<string, string, Task<string>> chat) : this((system, user, _) => chat(system, user)) { }

    /// <summary>A chat that can stream: the third argument, when not null, is called with (tokens so far, the answer so far) as the model writes.</summary>
    public Flux3PromptUpsampler(Func<string, string, Func<int, string, Task>?, Task<string>> chat) => _chat = chat;

    /// <summary>
    /// An upsampler on a loaded GGUF model (greedy, capped at 1200 new tokens: captions run 100 to 350 words). Qwen3 reasons before the caption: every caption so far was written that way (the
    /// thinking flag did not reach ChatML models until 2026-10-31) and the pages show the reasoning as it grows, so it stays on.
    /// </summary>
    public static Flux3PromptUpsampler FromPipeline(GgufTextGenerationPipeline pipeline, int maxNewTokens = 1200)
        => new((system, user, onToken) => pipeline.GenerateAsync(new List<(string, string)> { ("system", system), ("user", user) },
            new GenerationConfig { MaxNewTokens = maxNewTokens, Strategy = "greedy" }, thinking: true, onToken: onToken));

    /// <summary>Black Forest Labs' text-to-image upsampler system prompt (Apache-2.0).</summary>
    public const string TextToImagePrompt = """
You are the prompt upsampler of a text-to-image model. Expand the user's short request into ONE detailed, self-contained caption: the exact text the image model receives. Output only the caption, with no greetings, explanations, refusals, questions, Markdown, headings, JSON, coordinates, ids or angle-bracket tokens, and no quotes around it. If the input is not clearly an image request, caption a visualization of its concept. The request may begin with a "TARGET CANVAS: W:H" line: compose for that canvas (wide spreads content sideways, tall stacks it) and never mention the ratio or the canvas.

Fidelity and planning: keep every explicit subject, attribute, count, spatial constraint, style cue and piece of wording; never contradict, substitute, drop or narrow the request, and keep an abstract concept abstract. Decide everything left open as ONE concrete, consistent plan; never offer alternatives or hedge. Never re-express a stated constraint through another mechanism; add camera optics (depth of field, bokeh, blur) only when asked for or implied. Without a stated style, plan an actual photograph and open with the medium convention ("Full-color outdoor photograph, ...", "Full-color indoor photograph, ...", "Studio photograph, ...", "Close-up photograph, ..."); keep "Photorealistic" for renders, and turn "photorealistic" or "realistic" requests for photos into the photograph convention. Give a photograph a few ordinary, non-ideal specifics drawn from the scene's own objects instead of the stock-photo ideal. Do not invent names, brands, logos or celebrities; keep user-named ones. A long request keeps its content but not its wording: drop appraisal and sales vocabulary, turn moods into observable physical facts, and add only new concrete content with positions. Time of day is lighting, not rendered text.

Caption format: one continuous prose block in the generative voice, never "this image shows", "we can see" or "is visible". Lead with medium and style, then scene, lighting and camera, then the subjects with concrete colors, materials and state. Every person in focus gets an observable profile: ethnicity, gender, age range, skin tone ("fair skin", "brown skin"), hair and clothing. Make the layout reconstructable from words: frame-relative positions (frame-left, frame-right, upper, lower, foreground, midground, background), sentence-initial anchors ("On the left, ...", "In the background, ..."), relations (holds, faces, behind) and a consistent order; describe up to about ten instances one by one, beyond that give the exact count and arrangement. Out-of-focus or distant planes are shapes, colors and silhouettes, never identities or readable text, and the background takes about a sixth of the caption. Be definite: no hedging adverbs, no quality or emotion judgments, no closing sentence about mood; state negations as what is present. Rendered text: transcribe every wording verbatim inside double quotes in its original language and script, keep the line breaks of multi-line text, and give its typography, color and placement; author concrete, domain-correct wording when the request needs text it does not supply, and name canonical structures (calendar, keyboard, periodic table) exactly with their full layout. All prose is English; only quoted text keeps its language. Typical length is 100 to 350 words, more for complex scenes.

Output only the caption.
""";

    /// <summary>The caption for <paramref name="request"/>. With a canvas size the prompt composes for it (its "TARGET CANVAS: W:H" line, as the reference code writes it).</summary>
    /// <param name="onText">Called as the caption is written with (tokens so far, the answer so far, <see cref="Clean"/>ed), so a page can show it growing. The model's final answer is returned as before.
    /// While the model is still REASONING, the text so far is its <c>&lt;think&gt;</c> block growing (a progress cue the demo keeps on purpose, TJ 2026-10-20); the moment
    /// <c>&lt;/think&gt;</c> arrives <see cref="Clean"/> drops the reasoning, the text is REPLACED by the caption, and from then on it only grows.</param>
    public async Task<string> ExpandAsync(string request, int width = 0, int height = 0, Func<int, string, Task>? onText = null)
    {
        string user = width > 0 && height > 0 ? $"TARGET CANVAS: {width}:{height}\n{request}" : request;
        Func<int, string, Task>? onToken = onText == null ? null : (count, soFar) => onText(count, Clean(soFar));
        return Clean(await _chat(TextToImagePrompt, user, onToken));
    }

    /// <summary>The model's answer as one caption: reasoning blocks, code fences and surrounding quotes removed.</summary>
    public static string Clean(string answer)
    {
        string s = answer ?? "";
        int end = s.LastIndexOf("</think>", StringComparison.Ordinal);
        if (end >= 0) s = s[(end + 8)..];
        s = s.Trim();
        if (s.StartsWith("```")) { int nl = s.IndexOf('\n'); s = nl >= 0 ? s[(nl + 1)..] : s; s = s.TrimEnd('`').Trim(); }
        if (s.Length > 1 && s[0] == '"' && s[^1] == '"') s = s[1..^1].Trim();
        return s;
    }
}
