using System.Linq;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Ltx;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// The story writer's host-side parts: Qwen3's thinking switch in the ChatML template (thinking=false must emit the empty reasoning block, or the 4B model spends the
/// token budget reasoning and the shots come out cut off: measured 2026-10-31, 1 of 3 shots) and the shot parser. Synthetic vocab, no model, no GPU.
/// </summary>
public partial class MLTestBase
{
    [TestMethod]
    public async Task ChatML_ThinkingOff_EmitsQwen3EmptyThinkBlock() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.CPU) throw new UnsupportedTestException("host-only check: runs on the CPU lane");
        string[] toks = { "<unk>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "▁user", "▁assistant", "▁Hi", "\n" };
        var tok = new SentencePieceTokenizer(toks, new float[toks.Length]);
        int Idx(string s) => System.Array.IndexOf(toks, s);
        var msgs = new[] { ("user", "Hi") };

        var plain = ChatTemplates.BuildChatMLPromptTokens(tok, msgs);
        if (plain.Contains(Idx("<think>")) || plain.Contains(Idx("</think>"))) throw new Exception("the default prompt must not carry a reasoning block");
        var off = ChatTemplates.BuildChatMLPromptTokens(tok, msgs, emptyThinking: true);
        if (!off.Take(plain.Length).SequenceEqual(plain)) throw new Exception("the empty reasoning block must follow the unchanged prompt");
        var tail = off.Skip(plain.Length).ToArray();
        if (tail.Length < 2 || tail[0] != Idx("<think>")) throw new Exception($"after the generation prompt: <think> first, got [{string.Join(",", tail)}]");
        int close = System.Array.IndexOf(tail, Idx("</think>"));
        if (close < 1) throw new Exception($"</think> must follow <think> ([{string.Join(",", tail)}])");
        if (!ChatTemplates.HasThinkingSwitch("{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}"))
            throw new Exception("Qwen3's template has the switch");
        if (ChatTemplates.HasThinkingSwitch("{% for message in messages %}<|im_start|>{{ message.role }}") || ChatTemplates.HasThinkingSwitch(null))
            throw new Exception("a plain ChatML template (Qwen2.5) has no switch: it must not get the block");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task LtxStoryWriter_Parse_OneShotPerLine() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.CPU) throw new UnsupportedTestException("host-only check: runs on the CPU lane");
        const string a = "A small rusty orange robot with round blue eyes rolls across a cracked grey desert at sunset.";
        const string b = "A small rusty orange robot with round blue eyes kneels beside a red flower in the desert.";
        const string c = "A small rusty orange robot with round blue eyes watches the flower open as the sun sets.";
        var answer = $"<think>\nplanning the shots\n</think>\n\nHere is your story:\n\n1. {a}\n**Shot 2:** \"{b}\"\n- {c}\nThe End\n";
        var shots = LtxStoryWriter.Parse(answer, 3);
        if (!shots.SequenceEqual(new[] { a, b, c })) throw new Exception($"parsed [{string.Join(" | ", shots)}]");
        if (LtxStoryWriter.Parse(answer, 2).Length != 2) throw new Exception("at most the requested number of shots");
        if (LtxStoryWriter.Parse(a, 3).Length != 1) throw new Exception("fewer shots when the model wrote fewer");
        if (!LtxStoryWriter.UserPrompt("a cat in the rain", 3).StartsWith("Idea: A cat in the rain")) throw new Exception("the idea's first letter is capitalised (Qwen3 copies its case)");
        var narration = LtxStoryWriter.Parse("1. The robot wakes alone.\n2. It finds a flower.\nThe end?", 3, minLength: 8);
        if (!narration.SequenceEqual(new[] { "The robot wakes alone.", "It finds a flower.", "The end?" })) throw new Exception($"narration parsed [{string.Join(" | ", narration)}]");
        await Task.CompletedTask;
    });

    /// <summary>
    /// The narration track: each clip at its shot's time, a clip that would overlap the one before moved to after it (plus the gap), the track cut at the video's end with a fade,
    /// samples clamped. Rate 100 so positions are easy to read.
    /// </summary>
    [TestMethod]
    public async Task StoryNarrator_Mix_PlacesLinesWithoutOverlap() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.CPU) throw new UnsupportedTestException("host-only check: runs on the CPU lane");
        const int rate = 100;
        float[] Clip(int n, float v) => Enumerable.Repeat(v, n).ToArray();
        // clip 0 at 0.5 s (50 samples long, ends at 100); clip 1 asked for 0.8 s but clip 0 ends at 1.0 s: it starts at 1.0 + Gap; clip 2 runs past the 3 s end
        var track = StoryNarrator.Mix(new[] { Clip(50, 0.5f), Clip(40, 0.25f), Clip(100, 2f) }, new[] { 0.5, 0.8, 2.5 }, 3.0, rate);
        if (track.Length != 300) throw new Exception($"track length {track.Length}, expected 300");
        int gap = (int)Math.Round(StoryNarrator.Gap * rate);
        int first1 = Array.FindIndex(track, v => v == 0.25f);
        if (track[49] != 0f || track[50] != 0.5f || track[99] != 0.5f || track[100] != 0f) throw new Exception("clip 0 must fill [50, 100)");
        if (first1 != 100 + gap) throw new Exception($"clip 1 must start after clip 0 and the gap ({100 + gap}), started at {first1}");
        if (track[250] != 1f) throw new Exception($"clip 2 at 2.5 s, clamped to 1: got {track[250]}");
        if (track[299] != 0f) throw new Exception($"the last sample of a line cut by the end fades to 0: got {track[299]}");
        await Task.CompletedTask;
    });
}
