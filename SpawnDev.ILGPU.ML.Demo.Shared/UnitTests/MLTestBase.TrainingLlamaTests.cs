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

    /// <summary>
    /// THE M1 gate: a LoRA training step through the graph autograd (GraphAutograd + GgufLoraTrainer) on the tiny llama equals float64
    /// torch with the same adapters (rank 4 on q and v, PEFT semantics): the masked next-token loss, every adapter gradient, and the
    /// adapters after one AdamW step. Added 2026-11-04.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task Training_TinyLlama_LoraStepMatchesTorch() => await RunTest(accelerator => TinyLlamaLoraStepCheck(accelerator, checkpoint: false));

    /// <summary>The M1 gate with per-layer activation checkpointing (LoraOptions.CheckpointLayers): the recomputed backward must equal torch exactly as the kept one does.</summary>
    [TestMethod(Timeout = 300000)]
    public async Task Training_TinyLlama_LoraStepMatchesTorch_Checkpointed() => await RunTest(accelerator => TinyLlamaLoraStepCheck(accelerator, checkpoint: true));

    /// <summary>The M1 gate with the LM head and loss on the labelled positions only (LoraOptions.LossOnLabeledPositionsOnly) and checkpointing: same loss and gradients.</summary>
    [TestMethod(Timeout = 300000)]
    public async Task Training_TinyLlama_LoraStepMatchesTorch_LabeledOnly() => await RunTest(accelerator => TinyLlamaLoraStepCheck(accelerator, checkpoint: true, labeledOnly: true));

    private async Task<string> TinyLlamaLoraStepCheck(Accelerator accelerator, bool checkpoint, bool labeledOnly = false)
    {
        if (labeledOnly && accelerator.AcceleratorType == AcceleratorType.WebGL)
            throw new UnsupportedTestException("the labelled-positions loss scatters rows (GatherRowsGrad), which WebGL cannot");
        var (gguf, refDoc) = await LoadTinyLlamaTrainFixture();
        using var _ = refDoc;
        var root = refDoc.RootElement;
        var cfg = root.GetProperty("config");
        var tokens = root.GetProperty("tokens").EnumerateArray().Select(t => t.GetInt32()).ToArray();
        int lossFrom = root.GetProperty("loss_from").GetInt32();
        var labels = new int[tokens.Length];
        for (int t = 0; t < tokens.Length; t++) labels[t] = t >= lossFrom && t < tokens.Length - 1 ? tokens[t + 1] : -100;
        var lora = root.GetProperty("lora").EnumerateObject().ToDictionary(p => p.Name, p => p.Value);
        float[] Arr(JsonElement e, string key) => e.GetProperty(key).EnumerateArray().Select(v => v.GetSingle()).ToArray();
        var initial = lora.ToDictionary(kv => kv.Key, kv => Arr(kv.Value, "value"));
        var options = new SpawnDev.ILGPU.ML.Training.LoraOptions
        {
            Targets = new[] { "attn_q", "attn_v" },
            Rank = cfg.GetProperty("rank").GetInt32(), Alpha = cfg.GetProperty("alpha").GetSingle(),
            LearningRate = cfg.GetProperty("lr").GetSingle(), Beta1 = cfg.GetProperty("beta1").GetSingle(), Beta2 = cfg.GetProperty("beta2").GetSingle(),
            Epsilon = cfg.GetProperty("adam_eps").GetSingle(), WeightDecay = cfg.GetProperty("weight_decay").GetSingle(),
            Initial = initial,
            CheckpointLayers = checkpoint,
            LossOnLabeledPositionsOnly = labeledOnly,
        };
        using var trainer = SpawnDev.ILGPU.ML.Training.GgufLoraTrainer.Create(accelerator, gguf, options);
        if (trainer.Adapters.Count != lora.Count / 2) throw new Exception($"{trainer.Adapters.Count} adapters, expected {lora.Count / 2}");

        float loss = await trainer.StepAsync(tokens, labels, update: true);
        double wantLoss = root.GetProperty("loss").GetDouble();
        var notes = new List<string> { $"loss {loss:F6} (torch {wantLoss:F6})" };
        if (Math.Abs(loss - wantLoss) > 1e-4 * Math.Max(1, Math.Abs(wantLoss))) throw new Exception("loss differs from torch: " + notes[0]);

        var grads = await trainer.GetGradientsAsync();
        var after = await trainer.GetAdaptersAsync();
        double worstGrad = 0, worstAfter = 0; string atGrad = "", atAfter = "";
        foreach (var (name, e) in lora)
        {
            var wantG = Arr(e, "grad"); var g = grads[name];
            double scale = Math.Max(1e-3, wantG.Max(Math.Abs));
            for (int i = 0; i < wantG.Length; i++) { double d = Math.Abs(g[i] - wantG[i]) / scale; if (d > worstGrad) { worstGrad = d; atGrad = $"{name}[{i}] {g[i]:G6} vs {wantG[i]:G6}"; } }
            var wantA = Arr(e, "after_adamw"); var a = after[name];
            for (int i = 0; i < wantA.Length; i++) { double d = Math.Abs(a[i] - wantA[i]); if (d > worstAfter) { worstAfter = d; atAfter = $"{name}[{i}] {a[i]:G6} vs {wantA[i]:G6}"; } }
        }
        notes.Add($"adapter grads max relative diff {worstGrad:E2} ({atGrad})");
        notes.Add($"after AdamW max |diff| {worstAfter:E2} ({atAfter})");
        var line = $"{accelerator.AcceleratorType}{(checkpoint ? " checkpointed" : "")}{(labeledOnly ? " labeled-only" : "")}: " + string.Join("; ", notes);
        Console.WriteLine("[Training] " + line);
        if (worstGrad > 1e-3) throw new Exception("adapter gradients differ from torch: " + line);
        // One AdamW step moves each element by ~lr (1e-2) in the gradient's sign direction; a sign error or a wrong bias correction is ~1e-2.
        if (worstAfter > 1e-4) throw new Exception("AdamW step differs from torch: " + line);

        // PEFT files: save, read back with the engine's safetensors parser, equal to the live adapters; a trainer started from the file
        // reproduces the same forward loss (no update) as the trained one.
        var (st, configJson) = await trainer.SaveAdaptersAsync("tiny-llama-train");
        var loaded = SpawnDev.ILGPU.ML.Training.LoraAdapterFile.ReadSafeTensors(st);
        foreach (var (name, v) in after)
            if (!loaded.TryGetValue(name, out var l) || !l.SequenceEqual(v)) throw new Exception($"safetensors round trip changed {name}");
        using var cfgDoc = JsonDocument.Parse(configJson);
        if (cfgDoc.RootElement.GetProperty("r").GetInt32() != options.Rank || cfgDoc.RootElement.GetProperty("peft_type").GetString() != "LORA")
            throw new Exception("adapter_config.json: " + configJson);
        float lossTrained = await trainer.StepAsync(tokens, labels, update: false);
        options.Initial = loaded;
        using (var reloaded = SpawnDev.ILGPU.ML.Training.GgufLoraTrainer.Create(accelerator, gguf, options))
        {
            float lossReloaded = await reloaded.StepAsync(tokens, labels, update: false);
            if (lossReloaded != lossTrained) throw new Exception($"reloaded adapters give loss {lossReloaded}, the trained ones {lossTrained}");
            line += $"; saved {st.Length} B PEFT safetensors, reloaded loss {lossReloaded:F6} == trained";
        }
        return line;
    }

    /// <summary>
    /// Gradient accumulation (GgufLoraTrainer.AccumulateAsync / ApplyAsync): accumulating the same sequence twice and applying is one AdamW step on
    /// the mean of two identical gradients, which is the gradient itself exactly (2g * 0.5 = g in float), so the adapters after it must equal
    /// torch's single step, as StepAsync's do. Added 2026-11-04.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task Training_TinyLlama_AccumulateTwiceEqualsOneStep() => await RunTest(async accelerator =>
    {
        var (gguf, refDoc) = await LoadTinyLlamaTrainFixture();
        using var _ = refDoc;
        var root = refDoc.RootElement;
        var cfg = root.GetProperty("config");
        var tokens = root.GetProperty("tokens").EnumerateArray().Select(t => t.GetInt32()).ToArray();
        int lossFrom = root.GetProperty("loss_from").GetInt32();
        var labels = tokens.Select((_, t) => t >= lossFrom && t < tokens.Length - 1 ? tokens[t + 1] : -100).ToArray();
        var lora = root.GetProperty("lora").EnumerateObject().ToDictionary(p => p.Name, p => p.Value);
        float[] Arr(JsonElement e, string key) => e.GetProperty(key).EnumerateArray().Select(v => v.GetSingle()).ToArray();
        var options = new SpawnDev.ILGPU.ML.Training.LoraOptions
        {
            Targets = new[] { "attn_q", "attn_v" },
            Rank = cfg.GetProperty("rank").GetInt32(), Alpha = cfg.GetProperty("alpha").GetSingle(),
            LearningRate = cfg.GetProperty("lr").GetSingle(), Beta1 = cfg.GetProperty("beta1").GetSingle(), Beta2 = cfg.GetProperty("beta2").GetSingle(),
            Epsilon = cfg.GetProperty("adam_eps").GetSingle(), WeightDecay = cfg.GetProperty("weight_decay").GetSingle(),
            Initial = lora.ToDictionary(kv => kv.Key, kv => Arr(kv.Value, "value")),
        };
        using var trainer = SpawnDev.ILGPU.ML.Training.GgufLoraTrainer.Create(accelerator, gguf, options);
        float l1 = await trainer.AccumulateAsync(tokens, labels);
        float l2 = await trainer.AccumulateAsync(tokens, labels);
        if (l1 != l2) throw new Exception($"the same sequence gave two losses before any update: {l1} vs {l2}");
        if (trainer.AccumulatedSequences != 2) throw new Exception($"AccumulatedSequences {trainer.AccumulatedSequences}");
        await trainer.ApplyAsync();
        var after = await trainer.GetAdaptersAsync();
        double worst = 0;
        foreach (var (name, e) in lora)
        {
            var want = Arr(e, "after_adamw"); var got = after[name];
            for (int i = 0; i < want.Length; i++) worst = Math.Max(worst, Math.Abs(got[i] - want[i]));
        }
        var line = $"{accelerator.AcceleratorType}: loss {l1:F6} x2, after accumulate+apply max |diff| vs torch's step {worst:E2}";
        if (worst > 1e-4) throw new Exception("accumulated step differs from torch's single step: " + line);
        if (trainer.Step != 1 || trainer.AccumulatedSequences != 0) throw new Exception($"Step {trainer.Step}, AccumulatedSequences {trainer.AccumulatedSequences} after ApplyAsync");
        return line;
    });
}
