using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// <see cref="KVCacheAnalyzer"/> name-pairing tests. Pure string logic over the IO contract of real
/// exports - no model download, so these run in milliseconds on every backend and are the cheap gate that
/// the expensive model tests rest on.
/// </summary>
public abstract partial class MLTestBase
{
    // Real IO, MEASURED 2026-08-30 from the actual files rather than assumed:
    //   Xenova/distilgpt2 decoder_with_past_model.onnx        - 14 inputs / 13 outputs
    //   Xenova/distilgpt2 decoder_model.onnx                  -  2 inputs / 13 outputs (NO past)
    //   onnx-community/whisper-tiny decoder_with_past_model   - 17 inputs /  9 outputs
    private static string[] Gpt2WithPastInputs()
    {
        var names = new List<string> { "input_ids", "attention_mask" };
        for (int l = 0; l < 6; l++) { names.Add($"past_key_values.{l}.key"); names.Add($"past_key_values.{l}.value"); }
        return names.ToArray();
    }

    private static string[] Gpt2PresentOutputs()
    {
        var names = new List<string> { "logits" };
        for (int l = 0; l < 6; l++) { names.Add($"present.{l}.key"); names.Add($"present.{l}.value"); }
        return names.ToArray();
    }

    /// <summary>Plain decoder-only naming (<c>past_key_values.N.key</c>) pairs into a full cache.</summary>
    [TestMethod]
    public Task KVCacheAnalyzer_Gpt2Naming_DetectsAllLayers()
    {
        var info = KVCacheAnalyzer.Analyze(Gpt2WithPastInputs(), Gpt2PresentOutputs());
        if (!info.HasExplicitKVCache) throw new Exception("GPT-2 style past/present naming was not detected");
        if (info.NumLayers != 6) throw new Exception($"expected 6 layers, got {info.NumLayers}");
        for (int l = 0; l < 6; l++)
        {
            var pt = info.Layers[l];
            if (pt.PastKeyInput != $"past_key_values.{l}.key") throw new Exception($"layer {l} past key = '{pt.PastKeyInput}'");
            if (pt.PresentKeyOutput != $"present.{l}.key") throw new Exception($"layer {l} present key = '{pt.PresentKeyOutput}'");
        }
        Console.WriteLine("[KVAnalyzer] GPT-2 naming: 6 layers paired");
        return Task.CompletedTask;
    }

    /// <summary>
    /// The base decoder emits <c>present.*</c> but takes NO <c>past_key_values.*</c>, so there is no cache
    /// to manage and detection must say so.
    /// </summary>
    /// <remarks>
    /// This is the condition that made two DistilGPT-2 tests vacuous for months: they guarded their
    /// assertions behind <c>if (!HasKVCache) return;</c> while loading exactly this model, so they could
    /// never fail. Asserting it here means the fact is pinned in a test that costs nothing to run.
    /// </remarks>
    [TestMethod]
    public Task KVCacheAnalyzer_PresentOutputsWithoutPastInputs_IsNotACache()
    {
        var info = KVCacheAnalyzer.Analyze(new[] { "input_ids", "attention_mask" }, Gpt2PresentOutputs());
        if (info.HasExplicitKVCache)
            throw new Exception($"present.* alone must not count as a KV cache (got {info.NumLayers} layers)");
        if (info.ShouldQuantize) throw new Exception("ShouldQuantize must be false without past inputs");
        Console.WriteLine("[KVAnalyzer] present-without-past correctly reports no cache");
        return Task.CompletedTask;
    }

    /// <summary>
    /// Whisper-style encoder-decoder naming: only the <c>.decoder.</c> entries are the autoregressive
    /// cache; <c>.encoder.</c> is static cross-attention and must be excluded.
    /// </summary>
    /// <remarks>
    /// ⚠️ THE regression guard. Both forms used to parse to the same (layer, isKey) slot, so the encoder
    /// entry - second in the input list - overwrote the decoder one, and the analyzer happily paired
    /// ENCODER past against DECODER present and reported a healthy 4-layer cache. The executor would then
    /// append decoder keys into a cache addressed by the constant cross-attention inputs, with nothing
    /// failing loudly. Asserting `HasExplicitKVCache` alone would STILL PASS on the old code - the
    /// assertion that catches it is the PastKeyInput NAME.
    /// </remarks>
    [TestMethod]
    public Task KVCacheAnalyzer_WhisperEncoderDecoderNaming_ExcludesCrossAttention()
    {
        var inputs = new List<string> { "input_ids" };
        for (int l = 0; l < 4; l++)
        {
            inputs.Add($"past_key_values.{l}.decoder.key");
            inputs.Add($"past_key_values.{l}.decoder.value");
            inputs.Add($"past_key_values.{l}.encoder.key");     // cross-attention: constant, no present.*
            inputs.Add($"past_key_values.{l}.encoder.value");
        }
        var outputs = new List<string> { "logits" };
        for (int l = 0; l < 4; l++)
        {
            outputs.Add($"present.{l}.decoder.key");
            outputs.Add($"present.{l}.decoder.value");
        }

        var info = KVCacheAnalyzer.Analyze(inputs.ToArray(), outputs.ToArray());
        if (!info.HasExplicitKVCache) throw new Exception("whisper decoder self-attention cache was not detected");
        if (info.NumLayers != 4) throw new Exception($"expected 4 layers, got {info.NumLayers}");

        foreach (var pt in info.Layers)
        {
            if (pt.PastKeyInput.Contains(".encoder.") || pt.PastValueInput.Contains(".encoder."))
                throw new Exception(
                    $"layer {pt.LayerIndex} paired CROSS-ATTENTION past ('{pt.PastKeyInput}', " +
                    $"'{pt.PastValueInput}') - encoder KV is constant and has no present.* counterpart");
            if (pt.PastKeyInput != $"past_key_values.{pt.LayerIndex}.decoder.key")
                throw new Exception($"layer {pt.LayerIndex} past key = '{pt.PastKeyInput}'");
            if (pt.PresentKeyOutput != $"present.{pt.LayerIndex}.decoder.key")
                throw new Exception($"layer {pt.LayerIndex} present key = '{pt.PresentKeyOutput}'");
        }
        Console.WriteLine("[KVAnalyzer] whisper naming: 4 decoder layers paired, cross-attention excluded");
        return Task.CompletedTask;
    }

    /// <summary>Cross-attention entries ALONE are not a cache - there is nothing autoregressive there.</summary>
    [TestMethod]
    public Task KVCacheAnalyzer_CrossAttentionOnly_IsNotACache()
    {
        var inputs = new List<string> { "input_ids" };
        for (int l = 0; l < 4; l++)
        {
            inputs.Add($"past_key_values.{l}.encoder.key");
            inputs.Add($"past_key_values.{l}.encoder.value");
        }
        var outputs = new List<string> { "logits" };
        for (int l = 0; l < 4; l++)
        {
            outputs.Add($"present.{l}.encoder.key");
            outputs.Add($"present.{l}.encoder.value");
        }

        var info = KVCacheAnalyzer.Analyze(inputs.ToArray(), outputs.ToArray());
        if (info.HasExplicitKVCache)
            throw new Exception($"cross-attention-only naming must not count as a KV cache (got {info.NumLayers} layers)");
        Console.WriteLine("[KVAnalyzer] cross-attention-only correctly reports no cache");
        return Task.CompletedTask;
    }
}
