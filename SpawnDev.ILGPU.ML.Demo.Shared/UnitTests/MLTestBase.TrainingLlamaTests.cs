using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Training milestone M1 (Plans/training-autograd-lora-2026-11-04.md): a tiny f32 llama GGUF with GQA and its float64 torch
/// reference (tools/gen_tiny_llama_train_reference.py -> wwwroot/test-train/). Forward parity first: gradients are only
/// meaningful against a forward that already matches.
/// </summary>
public abstract partial class MLTestBase
{
    private async Task<(byte[] Gguf, JsonDocument Ref)> LoadTinyLlamaTrainFixture()
    {
        var assets = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var gguf = await assets.GetByteArrayAsync("test-train/tiny-llama-train.gguf");
        var json = JsonDocument.Parse(await assets.GetStringAsync("test-train/tiny-llama-train-ref.json"));
        return (gguf, json);
    }

    /// <summary>
    /// The engine's llama forward on the tiny GGUF equals the float64 torch reference's base-model logits at every position
    /// (llama.cpp layout: permuted q/k with adjacent-pair rope, GQA 4:2, RMSNorm, SwiGLU, untied LM head). Added 2026-11-04.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task Training_TinyLlama_ForwardMatchesTorch() => await RunTest(async accelerator =>
    {
        var (gguf, refDoc) = await LoadTinyLlamaTrainFixture();
        using var _ = refDoc;
        var root = refDoc.RootElement;
        int vocab = root.GetProperty("config").GetProperty("vocab").GetInt32();
        var tokens = root.GetProperty("tokens").EnumerateArray().Select(t => (float)t.GetInt32()).ToArray();
        var want = root.GetProperty("logits_base").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        using var session = InferenceSession.CreateFromGGUF(accelerator, gguf);
        using var ids = accelerator.Allocate1D(tokens);
        var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["input_ids"] = new Tensor(ids.View, new[] { 1, tokens.Length }, "input_ids") });
        var logitsT = outs.TryGetValue("logits", out var lt) ? lt : outs.Values.First();
        int n = tokens.Length * vocab;
        if (logitsT.ElementCount < n) throw new Exception($"logits has {logitsT.ElementCount} values, expected {n} (all positions)");
        var got = await logitsT.Data.SubView(0, n).CopyToHostAsync();
        double worst = 0; int at = 0;
        for (int i = 0; i < n; i++)
        {
            double d = Math.Abs(got[i] - want[i]);
            if (d > worst) { worst = d; at = i; }
        }
        double scale = want.Max(Math.Abs);
        var line = $"{accelerator.AcceleratorType}: logits max |diff| {worst:E2} (max |logit| {scale:F2}) at position {at / vocab}, token {at % vocab}";
        Console.WriteLine("[Training] " + line);
        if (worst > 2e-4 * Math.Max(1, scale)) throw new Exception("forward does not match the torch reference: " + line);
        return line;
    });
}
