using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// ComfyUI's Generate Text node on its own text-encoder file: <c>qwen_3_4b.safetensors</c> (F2_DIR\comfy; Qwen3-4B, bf16, no head) run as a language model by the GGUF
/// batch decoder through <see cref="SafeTensorsGguf"/> (no conversion, no copy), against SpawnGen's black-box token ids of ComfyUI's run (F2_DIR\comfy\textgenerate:
/// greedy, the default template, CPU bf16): the prompt ids ComfyUI fed the model must be ours exactly, then the greedy tokens. ComfyUI never feeds back its last
/// token, so its recorded run is one token shorter than what it generated: for thinking off that last token is the end of the turn.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task TextGenerate_ComfyQwen3_MatchesComfyUiTokens() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads the files from disk: a desktop GPU");
        var dir = Environment.GetEnvironmentVariable("F2_DIR") ?? "E:/spawndev-flux2";
        string file = Path.Combine(dir, "comfy", "qwen_3_4b.safetensors"), tokDir = Path.Combine(dir, "klein4b", "tokenizer"), refs = Path.Combine(dir, "comfy", "textgenerate", "qwen3_4b_textgenerate_token_refs.json");
        if (!File.Exists(file) || !File.Exists(refs) || !File.Exists(Path.Combine(tokDir, "tokenizer.json"))) throw new UnsupportedTestException($"needs {file}, the Qwen3 tokenizer in {tokDir} and SpawnGen's references {refs}");
        await using var fs = File.OpenRead(file);
        var model = await SafeTensorsGguf.FromHfQwen3Async(fs, await File.ReadAllTextAsync(Path.Combine(tokDir, "tokenizer.json")), await File.ReadAllTextAsync(Path.Combine(tokDir, "chat_template.jinja")), "qwen_3_4b");
        var tok = SentencePieceTokenizer.FromGGUF(model) ?? throw new Exception("no tokenizer from the metadata");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var gen = await GgufGenerator.CreateAsync(accelerator, fs, maxSeqLen: 512, model: model, engine: GgufEngine.BatchDecoder);
        var report = new List<string> { $"{model.BlockCount} layers, engine {gen.Engine}, loaded in {sw.Elapsed.TotalSeconds:F1} s" };
        using var doc = JsonDocument.Parse(await File.ReadAllTextAsync(refs));
        foreach (var r in doc.RootElement.GetProperty("references").EnumerateArray())
        {
            string name = r.GetProperty("name").GetString()!, prompt = r.GetProperty("prompt").GetString()!;
            bool thinking = r.GetProperty("thinking").GetBoolean();
            int maxLength = r.GetProperty("max_length").GetInt32();
            int[] wantPrompt = r.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            int[] wantFed = r.GetProperty("fed_back_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var ids = ChatTemplates.BuildChatMLPromptTokens(tok, new[] { ("user", prompt) }, addGenerationPrompt: true, emptyThinking: !thinking);
            Ur_Expect(ids.SequenceEqual(wantPrompt), $"{name}: prompt ids differ from ComfyUI's: ours [{string.Join(",", ids)}] ComfyUI [{string.Join(",", wantPrompt)}]");
            sw.Restart();
            var res = await gen.GenerateAsync(ids, new GenerationConfig { MaxNewTokens = maxLength, Strategy = "greedy" });
            var got = res.GeneratedTokenIds;
            int same = 0; while (same < Math.Min(got.Length, wantFed.Length) && got[same] == wantFed[same]) same++;
            report.Add($"{name}: prompt {ids.Length} ids identical; {got.Length} tokens in {sw.Elapsed.TotalSeconds:F1} s, the first {same} of ComfyUI's {wantFed.Length} fed-back tokens identical; stop {res.Stop} ({res.StopTokenId})");
            Ur_Expect(same == wantFed.Length, $"{name}: diverged at token {same}: ours [{string.Join(",", got.Skip(Math.Max(0, same - 2)).Take(6))}] ComfyUI [{string.Join(",", wantFed.Skip(Math.Max(0, same - 2)).Take(6))}]; " + string.Join("; ", report));
            if (!thinking) Ur_Expect(got.Length == wantFed.Length && res.StopTokenId == 151645, $"{name}: ComfyUI's answer ends there (<|im_end|>): " + string.Join("; ", report));
        }
        foreach (var line in report) Console.WriteLine("[TextGenerate] " + line);
        return string.Join("; ", report);
    });
}
