using System.Text.RegularExpressions;
using SpawnDev.ILGPU.ML.Flux3;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;

namespace SpawnDev.ILGPU.ML.Ltx;

/// <summary>
/// Turns one idea into a shot list for a story video: a small local language model (Qwen3 4B, the file <see cref="Flux3CaptionWriter"/> already uses, so a browser that drew a FLUX 3 picture has it cached)
/// writes one LTX-Video prompt per shot. Each shot is generated on its own and then extends the clip from the last frames of the one before (<see cref="LtxRequest.Conditions"/> with a video), so every
/// line has to stand alone: the same character described the same way, the same place, and the action picking up where the last shot stopped.
/// The model is loaded for the call and freed after it, so it never shares the GPU with the video transformer.
/// </summary>
public static class LtxStoryWriter
{
    public const string Repo = Flux3CaptionWriter.Repo, File = Flux3CaptionWriter.File;
    public const string Label = Flux3CaptionWriter.Label, SizeText = Flux3CaptionWriter.SizeText;

    /// <summary>
    /// The instructions. LTX-Video reads one flowing paragraph per clip, literal and chronological (main action first, then movement, appearance, place, camera, light: the structure of LTX-Video's own
    /// prompt guide); the story parts are what make separately generated clips read as one video.
    /// </summary>
    public const string SystemPrompt = """
You write shot lists for a text-to-video model. The user gives an idea and a number of shots. Write that many shots, one per line, in story order. Each shot becomes a separate 4-second clip, and each clip starts from the last frames of the one before, so together they form one continuous video.

Every line is the complete prompt for its clip: the video model sees only that line. So in every line:
- Begin the line with the main character's full description, word for word the same in every line (like "A small rusty orange robot with round blue eyes and dented arms" in the example below), never just "the cat", "he", "it" or "the same".
- Name the place and its look again (setting, time of day, weather, light).
- Start with the action of this shot, then movements and gestures, then the environment, then the camera (angle and movement) and the lighting and colors.
- Pick up exactly where the previous shot ended; keep one camera style and one look for the whole story. The story has a beginning, a turn and an end.

Each line is one paragraph of 50 to 80 words in normal sentence case, literal and precise: only what can be seen moving on screen, no sounds, no dialogue, no thoughts, no text on screen, no cuts inside a shot.
Output only the shots: no title, no numbering, no "Shot 1:", no blank lines, no quotes, no other text.

Example for "a robot finds a flower", 2 shots:
A small rusty orange robot with round blue eyes and dented arms rolls slowly across a cracked grey desert at sunset, dust rising behind its treads. It stops and tilts its head toward a single red flower growing from a crack in the ground. The camera follows at a low angle from the side, warm orange light, long shadows.
A small rusty orange robot with round blue eyes and dented arms kneels beside the single red flower in the cracked grey desert at sunset, reaching out one dented hand and gently touching a petal. The flower sways in the wind. The camera slowly circles the robot at a low angle, warm orange light, long shadows, dust drifting in the air.
""";

    /// <summary>The user message for <paramref name="idea"/> in <paramref name="shots"/> shots. The idea's first letter is capitalised: Qwen3 4B copies its case, and "a cat ..." gave all-lowercase shots.</summary>
    public static string UserPrompt(string idea, int shots)
    {
        idea = idea.Trim();
        if (idea.Length > 0) idea = char.ToUpperInvariant(idea[0]) + idea[1..];
        return $"Idea: {idea}\nShots: {shots}";
    }

    /// <summary>
    /// The shots in the model's answer: the reasoning block, numbering ("1.", "Shot 2:", "- "), quotes and blank lines removed, and short lines (a title, "Here is your story:") dropped.
    /// At most <paramref name="shots"/>; fewer when the model wrote fewer.
    /// </summary>
    public static string[] Parse(string answer, int shots)
    {
        var text = Flux3PromptUpsampler.Clean(answer);
        var lines = new List<string>();
        foreach (var raw in text.Split('\n'))
        {
            var s = raw.Trim();
            // "**Shot 2:**" before the bullets: a bullet rule first takes the leading "*" of the bold
            s = Regex.Replace(s, @"^(?:\*{0,2}shot\s*\d+\s*[.:)\-–]*\*{0,2}\s*|[-*•]\s+|\d+\s*[.):]\s*)", "", RegexOptions.IgnoreCase).Trim();
            s = s.Trim('"', '“', '”', '*').Trim();
            if (s.Length < 30 || s.EndsWith(':')) continue;                 // a heading or an introduction, not a shot
            lines.Add(s);
        }
        return lines.Take(shots).ToArray();
    }

    /// <param name="open">where the model file comes from</param>
    /// <param name="idea">what the story is about, in a few words</param>
    /// <param name="shots">how many shots (lines) to write</param>
    /// <param name="seed">the sampling seed: the same idea and seed write the same story, another seed another one</param>
    /// <param name="status">receives progress text ("Qwen3 4B — downloading 12%") as the model loads and writes</param>
    /// <param name="progress">receives 0 to 100 while the model downloads and loads (100 when it is ready)</param>
    /// <param name="storyText">receives the answer as it is written (the whole text so far, at most every <see cref="Flux3CaptionWriter.CaptionTextIntervalMs"/> ms)</param>
    public static async Task<string[]> WriteAsync(global::ILGPU.Runtime.Accelerator accelerator, Flux3CaptionWriter.OpenModel open, string idea, int shots, int seed, Action<string> status,
        Action<double>? progress = null, Action<string>? storyText = null)
    {
        Stream? stream = null; GgufTextGenerationPipeline? llm = null;
        try
        {
            var download = new Progress<ModelDownloadProgress>(p =>
            {
                status(p.Fraction is { } f ? $"{Label} — downloading {f:P0}" : $"{Label} — downloading {p.BytesReceived / 1048576} MiB");
                if (p.Fraction is { } fraction) progress?.Invoke(Math.Min(100, fraction * 100));
            });
            SpawnDev.ILGPU.ML.GGUF.GGUFGraphBuilder.EnableLastPositionLogits = true;   // generation reads only the last position's logits
            stream = await open(Repo, File, download);
            llm = await GgufTextGenerationPipeline.CreateFromStreamAsync(accelerator, stream, maxSeqLen: 4096,
                onProgress: (stage, pct) => { status($"{Label} — {stage} {pct}%"); progress?.Invoke(Math.Min(100, pct)); });
            llm.EnableWebGPUDecodeCapture = true;
            SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.EnableWebGPUGemv = true;
            progress?.Invoke(100);
            status($"{Label} — writing the story");
            var clock = System.Diagnostics.Stopwatch.StartNew();
            long lastShown = -Flux3CaptionWriter.CaptionTextIntervalMs;
            // Qwen3's recommended sampling without reasoning (temperature 0.7, top-p 0.8) and no repetition penalty: every line repeats the character's description on purpose, and a penalty
            // over the last tokens also hit "." and "The" (measured: lowercase lines, then one run-on line with no punctuation at all)
            var answer = await llm.GenerateAsync(new List<(string, string)> { ("system", SystemPrompt), ("user", UserPrompt(idea, shots)) },
                new GenerationConfig { MaxNewTokens = 160 * shots + 64, Strategy = "top_p", Temperature = 0.7f, TopP = 0.8f, Seed = seed },
                thinking: false,
                onToken: (tokens, soFar) =>
                {
                    if (clock.ElapsedMilliseconds - lastShown >= Flux3CaptionWriter.CaptionTextIntervalMs)
                    {
                        lastShown = clock.ElapsedMilliseconds;
                        storyText?.Invoke(Flux3PromptUpsampler.Clean(soFar));
                        status($"{Label} — writing the story ({tokens} tokens)");
                    }
                    return Task.CompletedTask;
                });
            var lines = Parse(answer, shots);
            status($"{Label} — wrote {lines.Length} shots ({llm.LastStopReason}, {answer.Length} chars)");
            storyText?.Invoke(string.Join("\n", lines));
            return lines;
        }
        finally
        {
            await accelerator.SynchronizeAsync();
            llm?.Dispose(); stream?.Dispose();
        }
    }
}
