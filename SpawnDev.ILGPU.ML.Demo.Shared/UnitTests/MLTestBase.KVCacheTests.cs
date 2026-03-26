using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Tests for KVCacheAnalyzer — detects autoregressive KV cache patterns in model graphs.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task KVCache_DetectsExplicitPastKeyValues() => await RunTest(async accelerator =>
    {
        // Simulate DistilGPT-2 merged model inputs/outputs
        var inputNames = new[]
        {
            "input_ids", "attention_mask",
            "past_key_values.0.key", "past_key_values.0.value",
            "past_key_values.1.key", "past_key_values.1.value",
            "past_key_values.2.key", "past_key_values.2.value",
            "past_key_values.3.key", "past_key_values.3.value",
            "past_key_values.4.key", "past_key_values.4.value",
            "past_key_values.5.key", "past_key_values.5.value",
            "use_cache_branch"
        };
        var outputNames = new[]
        {
            "logits",
            "present.0.key", "present.0.value",
            "present.1.key", "present.1.value",
            "present.2.key", "present.2.value",
            "present.3.key", "present.3.value",
            "present.4.key", "present.4.value",
            "present.5.key", "present.5.value",
        };

        var info = KVCacheAnalyzer.Analyze(inputNames, outputNames);

        if (!info.HasExplicitKVCache)
            throw new Exception("Should detect explicit KV cache");

        if (info.NumLayers != 6)
            throw new Exception($"Expected 6 layers, got {info.NumLayers}");

        if (info.UseCacheBranchInput != "use_cache_branch")
            throw new Exception($"use_cache_branch not detected: {info.UseCacheBranchInput}");

        // Verify each layer is paired correctly
        for (int i = 0; i < 6; i++)
        {
            var layer = info.Layers[i];
            if (layer.LayerIndex != i)
                throw new Exception($"Layer {i} index={layer.LayerIndex}");
            if (!layer.PastKeyInput.Contains($"{i}.key"))
                throw new Exception($"Layer {i} past key: {layer.PastKeyInput}");
            if (!layer.PresentKeyOutput.Contains($"{i}.key"))
                throw new Exception($"Layer {i} present key: {layer.PresentKeyOutput}");
        }

        if (!info.ShouldQuantize)
            throw new Exception("ShouldQuantize should be true for explicit KV cache");
    });

    [TestMethod]
    public async Task KVCache_NoDetectionForInferenceOnly() => await RunTest(async accelerator =>
    {
        // Model with no KV cache (like SqueezeNet — classification only)
        var inputNames = new[] { "data" };
        var outputNames = new[] { "squeezenet0_flatten0_reshape0" };

        var info = KVCacheAnalyzer.Analyze(inputNames, outputNames);

        if (info.HasExplicitKVCache)
            throw new Exception("Should NOT detect KV cache for classification model");

        if (info.NumLayers != 0)
            throw new Exception($"Expected 0 layers, got {info.NumLayers}");

        if (info.ShouldQuantize)
            throw new Exception("ShouldQuantize should be false");
    });

    [TestMethod]
    public async Task KVCache_PartialPairsIgnored() => await RunTest(async accelerator =>
    {
        // Model with mismatched past/present (broken or partial)
        var inputNames = new[]
        {
            "input_ids",
            "past_key_values.0.key", "past_key_values.0.value",
            "past_key_values.1.key", // missing value!
        };
        var outputNames = new[]
        {
            "logits",
            "present.0.key", "present.0.value",
            // layer 1 present missing entirely
        };

        var info = KVCacheAnalyzer.Analyze(inputNames, outputNames);

        // Only layer 0 should be detected (complete pair)
        if (info.NumLayers != 1)
            throw new Exception($"Expected 1 complete layer, got {info.NumLayers}");

        if (info.Layers[0].LayerIndex != 0)
            throw new Exception($"Expected layer 0, got {info.Layers[0].LayerIndex}");
    });

    [TestMethod]
    public async Task KVCache_HeadDimFromShape() => await RunTest(async accelerator =>
    {
        var inputNames = new[]
        {
            "input_ids",
            "past_key_values.0.key", "past_key_values.0.value",
        };
        var outputNames = new[]
        {
            "logits",
            "present.0.key", "present.0.value",
        };

        // Provide shape info: [batch, heads, seq, head_dim]
        var shapes = new Dictionary<string, int[]>
        {
            ["past_key_values.0.key"] = new[] { 1, 12, 128, 64 },
        };

        var info = KVCacheAnalyzer.Analyze(inputNames, outputNames, shapes);

        if (info.Layers[0].HeadDim != 64)
            throw new Exception($"Expected head_dim=64, got {info.Layers[0].HeadDim}");
    });
}
