using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Real Ollama exports keep some tensors in F16 (qwen3:1.7b's attn_v, smollm2:360m's token_embd), which the batch decoder could not load: every desktop
    /// Ollama model fell back to the session engine (Data, SpawnDev.AI, 2026-11-02). From this machine's Ollama cache (skipped when absent): the pipeline on
    /// Auto must choose the batch decoder, and its greedy answer must equal the session engine's, token for token. Desktop lanes (the browser cannot read the cache).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task GgufBatchDecoder_OllamaF16Tensors_MatchGenerator() => await RunTest(async accelerator =>
    {
        if (OperatingSystem.IsBrowser() || accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL))
            throw new UnsupportedTestException("reads the local Ollama cache (CUDA / OpenCL lanes)");
        string root = Environment.GetEnvironmentVariable("OLLAMA_MODELS") is { Length: > 0 } env ? env
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".ollama", "models");
        string? Blob(string model, string tag)
        {
            var manifest = Path.Combine(root, "manifests", "registry.ollama.ai", "library", model, tag);
            if (!File.Exists(manifest)) return null;
            using var doc = JsonDocument.Parse(File.ReadAllText(manifest));
            foreach (var layer in doc.RootElement.GetProperty("layers").EnumerateArray())
                if (layer.GetProperty("mediaType").GetString()!.EndsWith(".model"))
                    return Path.Combine(root, "blobs", layer.GetProperty("digest").GetString()!.Replace(':', '-'));
            return null;
        }
        var q = new List<(string, string)> { ("user", "Name three primary colors and say why they are called primary.") };
        var cfg = new GenerationConfig { MaxNewTokens = 32, Strategy = "greedy" };
        var notes = new List<string>();
        bool ok = true, any = false;
        foreach (var (model, tag) in new[] { ("qwen3", "1.7b"), ("smollm2", "360m") })
        {
            var path = Blob(model, tag);
            if (path == null || !File.Exists(path)) { notes.Add($"{model}:{tag} not in the Ollama cache"); continue; }
            any = true;
            string want, got; GgufEngine engine; string reason;
            using (var gen = await GgufTextGenerationPipeline.CreateFromFileAsync(accelerator, path, maxSeqLen: 1024, engine: GgufEngine.Generator))
                want = await gen.GenerateAsync(q, cfg, thinking: false);
            using (var auto = await GgufTextGenerationPipeline.CreateFromFileAsync(accelerator, path, maxSeqLen: 1024))
            {
                engine = auto.Engine; reason = auto.EngineReason;
                got = await auto.GenerateAsync(q, cfg, thinking: false);
            }
            bool pass = engine == GgufEngine.BatchDecoder && got == want;
            ok &= pass;
            notes.Add($"{model}:{tag}: Auto -> {engine}{(reason.Length > 0 ? " (" + reason + ")" : "")}, greedy {(got == want ? "= the session engine" : "DIFFERS")}" +
                (got == want ? "" : $" | session: {want[..Math.Min(70, want.Length)]} | batch: {got[..Math.Min(70, got.Length)]}"));
        }
        if (!any) throw new UnsupportedTestException("no qwen3:1.7b or smollm2:360m in the Ollama cache");
        var report = $"{accelerator.AcceleratorType}: " + string.Join(" | ", notes);
        Console.WriteLine("[OllamaF16] " + report);
        if (!ok) throw new Exception(report);
        return report;
    });
}
