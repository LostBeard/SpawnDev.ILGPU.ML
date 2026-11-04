using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Training;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// LoRA at inference: GgufBatchDecoder with a LoraRuntime (adapters applied unmerged after each projection, including the decoder's fused
    /// q|k|v and gate|up outputs) must reproduce the TRAINING graph's forward (GraphAutograd.AddLora) with the same adapters, on every projection
    /// type, and the base decoder must differ from it (the adapters do something). Tiny F16 llama (tools/gen_tiny_llama_f16.py), head dim 64,
    /// GQA 2:1. The backends the batch decoder serves (CUDA, OpenCL, WebGPU). Added 2026-11-04.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task Training_DecoderLora_MatchesTrainingForward() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("the batch decoder's backends (CUDA, OpenCL, WebGPU)");
        var assets = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var gguf = await assets.GetByteArrayAsync("test-train/tiny-llama-f16.gguf");
        var targets = new[] { "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down" };
        const int rank = 4; const float alpha = 8f;
        using var trainer = GgufLoraTrainer.Create(accelerator, gguf, new LoraOptions { Targets = targets, Rank = rank, Alpha = alpha, Seed = 3 });
        // B nonzero (PEFT starts it at zero, which would make the adapters a no-op and the comparison vacuous)
        var rng = new Random(5);
        var adapters = await trainer.GetAdaptersAsync();
        foreach (var k in adapters.Keys.Where(k => k.EndsWith("lora_B.weight")).ToList())
            adapters[k] = adapters[k].Select(_ => (float)((rng.NextDouble() * 2 - 1) * 0.3)).ToArray();
        await trainer.SetAdaptersAsync(adapters);
        var ids = Enumerable.Range(0, 20).Select(_ => rng.Next(0, trainer.VocabSize)).ToArray();
        var labels = ids.Select((_, t) => t < ids.Length - 1 ? ids[t + 1] : -100).ToArray();
        await trainer.StepAsync(ids, labels, update: false);
        var want = await trainer.GetLogitsAsync(ids.Length - 1);

        using var dec = await GgufBatchDecoder.CreateAsync(accelerator, new MemoryStream(gguf), 64, 1, null);
        await dec.PrefillAsync(ids);
        var baseLogits = await dec.ReadLogitsAsync();
        using var lora = new LoraRuntime(accelerator, adapters, rank, alpha);
        dec.Lora = lora;
        await dec.PrefillAsync(ids);
        var got = await dec.ReadLogitsAsync();
        double diff = 0, moved = 0, scale = want.Max(Math.Abs);
        for (int i = 0; i < want.Length; i++) { diff = Math.Max(diff, Math.Abs(got[i] - want[i])); moved = Math.Max(moved, Math.Abs(baseLogits[i] - want[i])); }
        var line = $"{accelerator.AcceleratorType}: decoder + LoRA vs training forward max |diff| {diff:E2} (max |logit| {scale:F2}); base decoder differs by {moved:E2}; "
            + $"{lora.Targets.Count()} adapted projections";
        Console.WriteLine("[Training] " + line);
        if (moved < 0.05 * Math.Max(1, scale)) throw new Exception("the adapters barely change the logits: the comparison proves nothing: " + line);
        if (diff > 5e-3 * Math.Max(1, scale)) throw new Exception("the decoder's adapters do not reproduce the training forward: " + line);
        dec.Lora = null;
        await dec.PrefillAsync(ids);
        var back = await dec.ReadLogitsAsync();
        if (!back.SequenceEqual(baseLogits)) throw new Exception("removing the adapters did not restore the base logits");
        return line;
    });

    /// <summary>
    /// Training steps must not grow device memory: after two warm steps (planned reuse records, then plans), five more steps allocate nothing.
    /// MEASURED 2026-11-04 before the fix: +1 pool device allocation and +32 MiB per step - the graph executor rented outputs an operator
    /// produces but the node does not name (SoftmaxCrossEntropyLoss's optional log_prob) and never returned them. Added 2026-11-04.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task Training_StepsDoNotGrowThePool() => await RunTest(async accelerator =>
    {
        var assets = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var gguf = await assets.GetByteArrayAsync("test-train/tiny-llama-train.gguf");
        using var trainer = GgufLoraTrainer.Create(accelerator, gguf, new LoraOptions { Targets = new[] { "attn_q", "attn_v" }, Rank = 4, Alpha = 8, Seed = 2 });
        var rng = new Random(3);
        var ids = Enumerable.Range(0, 12).Select(_ => rng.Next(0, trainer.VocabSize)).ToArray();
        var labels = ids.Select((_, t) => t >= 6 && t < ids.Length - 1 ? ids[t + 1] : -100).ToArray();
        await trainer.StepAsync(ids, labels);
        await trainer.StepAsync(ids, labels);
        long allocs0 = SpawnDev.ILGPU.ML.Tensors.BufferPool.TotalDeviceAllocations;
        int live0 = trainer.Session.Executor.Pool.LiveNamedCount;
        for (int i = 0; i < 5; i++) await trainer.StepAsync(ids, labels);
        long grew = SpawnDev.ILGPU.ML.Tensors.BufferPool.TotalDeviceAllocations - allocs0;
        int liveGrew = trainer.Session.Executor.Pool.LiveNamedCount - live0;
        var line = $"{accelerator.AcceleratorType}: 5 steps after warm-up: pool device allocations +{grew}, live rents {liveGrew:+0;-0;+0}";
        if (grew > 0 || liveGrew > 0) throw new Exception("training steps grow device memory: " + line);
        return line;
    });

    /// <summary>
    /// GgufGenerator.CreateAsync(lora:) on BOTH engines: the session engine (adapters inserted into its graph at load) and the batch decoder
    /// (a LoraRuntime) generate the same greedy tokens with the same adapter set, and both differ from the base model. Added 2026-11-04.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task Training_GeneratorLora_BothEnginesAgree() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("the batch decoder's backends (CUDA, OpenCL, WebGPU), to compare the two engines");
        var assets = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var gguf = await assets.GetByteArrayAsync("test-train/tiny-llama-f16.gguf");
        const int rank = 4; const float alpha = 8f;
        Dictionary<string, float[]> adapters;
        int vocab;
        using (var trainer = GgufLoraTrainer.Create(accelerator, gguf, new LoraOptions { Targets = new[] { "attn_q", "attn_v", "ffn_up", "ffn_down" }, Rank = rank, Alpha = alpha, Seed = 9 }))
        {
            var rng0 = new Random(13);
            adapters = await trainer.GetAdaptersAsync();
            foreach (var k in adapters.Keys.Where(k => k.EndsWith("lora_B.weight")).ToList())
                adapters[k] = adapters[k].Select(_ => (float)((rng0.NextDouble() * 2 - 1) * 0.3)).ToArray();
            vocab = trainer.VocabSize;
        }
        var set = new LoraAdapterSet(adapters, rank, alpha);
        var rng = new Random(17);
        var prompt = Enumerable.Range(0, 12).Select(_ => rng.Next(0, vocab)).ToArray();
        var cfg = new SpawnDev.ILGPU.ML.Preprocessing.GenerationConfig { Strategy = "greedy", MaxNewTokens = 8 };
        async Task<int[]> Gen(GgufEngine engine, LoraAdapterSet? lora)
        {
            using var gen = await GgufGenerator.CreateAsync(accelerator, new MemoryStream(gguf), 64, engine: engine, lora: lora);
            if (gen.Engine != engine) throw new Exception($"asked for {engine}, got {gen.Engine}");
            return (await gen.GenerateAsync(prompt, cfg)).GeneratedTokenIds;
        }
        var sessionLora = await Gen(GgufEngine.Generator, set);
        var batchLora = await Gen(GgufEngine.BatchDecoder, set);
        var sessionBase = await Gen(GgufEngine.Generator, null);
        var line = $"{accelerator.AcceleratorType}: session+LoRA [{string.Join(",", sessionLora)}], batch+LoRA [{string.Join(",", batchLora)}], base [{string.Join(",", sessionBase)}]";
        Console.WriteLine("[Training] " + line);
        if (!sessionLora.SequenceEqual(batchLora)) throw new Exception("the two engines generate differently with the same adapters: " + line);
        if (sessionLora.SequenceEqual(sessionBase)) throw new Exception("the adapters do not change the generation: " + line);

        // GenerationResult.StopTokenId: make the 4th greedy token a stop id; both engines must stop there, report THAT id, and not include it.
        int stopId = sessionBase[3];
        foreach (var engine in new[] { GgufEngine.Generator, GgufEngine.BatchDecoder })
        {
            using var gen = await GgufGenerator.CreateAsync(accelerator, new MemoryStream(gguf), 64, engine: engine);
            var res = await gen.GenerateAsync(prompt, cfg, stopTokenIds: new[] { stopId });
            int firstStop = Array.IndexOf(sessionBase, stopId);
            if (res.StopTokenId != stopId || res.Stop != StopReason.StopToken || !res.GeneratedTokenIds.SequenceEqual(sessionBase.Take(firstStop)))
                throw new Exception($"{engine}: StopTokenId {res.StopTokenId?.ToString() ?? "null"} (want {stopId}), Stop {res.Stop}, ids [{string.Join(",", res.GeneratedTokenIds)}] (want [{string.Join(",", sessionBase.Take(firstStop))}])");
        }
        return line + $"; StopTokenId {stopId} reported by both engines";
    });
}
