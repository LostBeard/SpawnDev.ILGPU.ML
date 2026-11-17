using System.Globalization;

namespace SpawnDev.ILGPU.ML.AceStep;

/// <summary>
/// The prompts ComfyUI's TextEncodeAceStepAudio1.5 builds, as measured from its token ids on SpawnGen's dumps (decoded with ACE-Step's own MIT tokenizer; re-encoded here they
/// give ComfyUI's ids exactly: AceStep_Prompts_MatchComfyUiIds). Tokenise with <see cref="Flux2.Qwen2Tokenizer"/> on ACE-Step's tokenizer.json (Qwen3-Embedding-0.6B's for the
/// caption and lyrics, acestep-5Hz-lm-1.7B's for the LM; the template text tokenises the same with either).
/// <para>Measured so far on one run (bpm, time signature, key all set); how ComfyUI prints a missing value and where it cuts long lyrics are still to be measured.</para>
/// </summary>
public static class AceStepPrompts
{
    /// <summary>The instruction ComfyUI puts in both the caption prompt and the LM's system turn.</summary>
    public const string Instruction = "Generate audio semantic tokens based on the given conditions:";

    /// <summary>The caption prompt for the 0.6B encoder (its states are the DiT's text conditioning).</summary>
    public static string Caption(string tags, int bpm, string timeSignature, string keyScale, int durationSeconds)
        => "# Instruction\n" + Instruction + "\n\n# Caption\n" + tags + "\n\n# Metas\n"
         + "- bpm: " + bpm.ToString(CultureInfo.InvariantCulture) + "\n"
         + "- timesignature: " + timeSignature + "\n"
         + "- keyscale: " + keyScale + "\n"
         + "- duration: " + durationSeconds.ToString(CultureInfo.InvariantCulture) + " seconds\n"
         + "<|endoftext|>\n<|endoftext|>";

    /// <summary>The lyric prompt (its tokens' embedding rows are the lyric conditioning).</summary>
    public static string Lyrics(string language, string lyrics) => "# Languages\n" + language + "\n\n# Lyric\n" + lyrics + "\n<|endoftext|><|endoftext|>";

    /// <summary>The code LM's positive prompt: the chat with the metas in the assistant's thinking block.</summary>
    public static string LmPositive(string tags, string lyrics, int bpm, string timeSignature, string keyScale, int durationSeconds)
        => LmUser(tags, lyrics) + "<|im_start|>assistant\n<think>\n"
         + "bpm: " + bpm.ToString(CultureInfo.InvariantCulture) + "\n"
         + "duration: " + durationSeconds.ToString(CultureInfo.InvariantCulture) + "\n"
         + "keyscale: " + keyScale + "\n"
         + "timesignature: " + timeSignature + "\n"
         + "</think>\n\n<|im_end|>\n";

    /// <summary>The code LM's negative (CFG) prompt: the same chat with an empty thinking block. ComfyUI left-pads it to the positive's length; <see cref="AceStepCodeLm"/> takes it unpadded.</summary>
    public static string LmNegative(string tags, string lyrics) => LmUser(tags, lyrics) + "<|im_start|>assistant\n<think>\n\n</think>\n\n<|im_end|>\n";

    private static string LmUser(string tags, string lyrics)
        => "<|im_start|>system\n# Instruction\n" + Instruction + "\n\n<|im_end|>\n<|im_start|>user\n# Caption\n" + tags + "\n\n# Lyric\n" + lyrics + "\n<|im_end|>\n";
}
