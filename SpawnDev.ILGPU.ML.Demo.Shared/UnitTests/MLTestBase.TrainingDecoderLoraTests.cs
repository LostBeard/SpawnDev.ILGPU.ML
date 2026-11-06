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
        // the packed q|k|v / gate|up launch (FuseAdjacent) sums each column's own rank terms in the same order: bit-identical to per-projection
        bool savedFuse = LoraRuntime.FuseAdjacent;
        float[] unfused;
        try
        {
            LoraRuntime.FuseAdjacent = !savedFuse;
            await dec.PrefillAsync(ids);
            unfused = await dec.ReadLogitsAsync();
        }
        finally { LoraRuntime.FuseAdjacent = savedFuse; }
        if (!unfused.SequenceEqual(got))
            throw new Exception($"FuseAdjacent {savedFuse} vs {!savedFuse} differ: max |diff| {got.Zip(unfused, (a, b) => Math.Abs(a - b)).Max():E2}");
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
    /// Adapters changed IN PLACE on a decoder that has recorded its decode steps (CUDA graphs, WebGPU plans): each phase's greedy tokens must equal a fresh
    /// decoder's with that adapter set. Found by Data on SpawnDev.AI's lora-test (ML 5.8.4, CUDA): after <c>Lora = null</c> the recorded step kept
    /// replaying the old adapter kernels, 2 of 24 tokens right. Sequence: A, none, A, B (B: different buffers, so a stale recording cannot pass by luck).
    /// Tiny F16 llama, random B matrices (PEFT's zero B would make every adapter a no-op).
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task Training_DecoderLora_SwapInPlace_DecodeMatchesFresh() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("the batch decoder's backends (CUDA, OpenCL, WebGPU)");
        var assets = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var gguf = await assets.GetByteArrayAsync("test-train/tiny-llama-f16.gguf");
        var targets = new[] { "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down" };
        const int rank = 4; const float alpha = 8f;
        async Task<Dictionary<string, float[]>> Adapters(int seed)
        {
            using var trainer = GgufLoraTrainer.Create(accelerator, gguf, new LoraOptions { Targets = targets, Rank = rank, Alpha = alpha, Seed = seed });
            var rng = new Random(seed * 7 + 1);
            var a = await trainer.GetAdaptersAsync();
            foreach (var k in a.Keys.Where(k => k.EndsWith("lora_B.weight")).ToList())
                a[k] = a[k].Select(_ => (float)((rng.NextDouble() * 2 - 1) * 0.3)).ToArray();
            return a;
        }
        var adA = await Adapters(3);
        var adB = await Adapters(11);
        using var loraA = new LoraRuntime(accelerator, adA, rank, alpha);
        using var loraB = new LoraRuntime(accelerator, adB, rank, alpha);
        int vocab;
        using (var probe = await GgufBatchDecoder.CreateAsync(accelerator, new MemoryStream(gguf), 128, 1, null)) vocab = probe.Vocab;
        var prompt = Enumerable.Range(0, 12).Select(i => (i * 37 + 5) % vocab).ToArray();
        var noStops = new HashSet<int>();
        const int N = 24;
        async Task<int[]> Fresh(LoraRuntime? l)
        {
            using var d = await GgufBatchDecoder.CreateAsync(accelerator, new MemoryStream(gguf), 128, 1, null);
            d.Lora = l;
            return (await d.GenerateAsync(prompt, 1, N, 0f, 1f, 1, noStops))[0];
        }
        var wantBase = await Fresh(null); var wantA = await Fresh(loraA); var wantB = await Fresh(loraB);
        if (wantA.SequenceEqual(wantBase) || wantB.SequenceEqual(wantA))
            throw new Exception("the adapters do not change the greedy tokens: the comparison would prove nothing");
        using var dec = await GgufBatchDecoder.CreateAsync(accelerator, new MemoryStream(gguf), 128, 1, null);
        var notes = new List<string>();
        foreach (var (name, l, want) in new (string, LoraRuntime?, int[])[] { ("A", loraA, wantA), ("none", null, wantBase), ("A again", loraA, wantA), ("B", loraB, wantB) })
        {
            dec.Lora = l;
            var got = (await dec.GenerateAsync(prompt, 1, N, 0f, 1f, 1, noStops))[0];
            int same = got.Zip(want, (a, b) => a == b ? 1 : 0).Sum();
            notes.Add($"{name}: {same}/{N}");
            if (same != N) throw new Exception($"{accelerator.AcceleratorType}: after setting the adapters to {name} in place, {same} of {N} greedy tokens equal a fresh decoder's ({string.Join(", ", notes)}; captures {dec.CaptureStats.Captures}, replays {dec.CaptureStats.Replays})");
        }
        var line = $"{accelerator.AcceleratorType}: in-place adapter swaps match fresh decoders ({string.Join(", ", notes)}); captures {dec.CaptureStats.Captures}, replays {dec.CaptureStats.Replays}";
        Console.WriteLine("[Training] " + line);
        return line;
    });

    /// <summary>
    /// LoRA over Q8_0 weights regrouped in place (<see cref="GgufBatchDecoder.GroupQ8"/>, GGMLType.Q8_0G): Qwen3-0.6B Q8_0 (hidden 1,024 and FFN 3,072, so every
    /// matrix regroups) with a synthetic rank-8 adapter on q / k / v / o / gate / up / down, against the same model and adapter on the native layout. The last-row
    /// logits after prompts of 1, 5 and 40 tokens (the one-row vector GEMV, the 2-8 row word-wise GEMV, the prefill GEMM) agree within 2e-3 of the largest
    /// logit, and the adapter moves them (else the comparison would not cover it). Added 2026-11-06: the in-repo LoRA fixtures are F16, so no test paired LoRA
    /// with the regrouped layout.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task Training_DecoderLora_OverQ8Grouped_MatchesNative() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("the batch decoder's backends (CUDA, OpenCL, WebGPU)");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var m = await hub.OpenAsync("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf", deselect: false);
        await using (m.Stream)
        {
            var model = await SpawnDev.ILGPU.ML.GGUF.GGUFParser.ParseHeaderAsync(m.Stream);
            var peftName = new Dictionary<string, string> { ["attn_q"] = "self_attn.q_proj", ["attn_k"] = "self_attn.k_proj", ["attn_v"] = "self_attn.v_proj",
                ["attn_output"] = "self_attn.o_proj", ["ffn_gate"] = "mlp.gate_proj", ["ffn_up"] = "mlp.up_proj", ["ffn_down"] = "mlp.down_proj" };
            const int rank = 8; const float alpha = 16f;
            var rng = new Random(23);
            var tensors = new Dictionary<string, float[]>();
            for (int l = 0; l < (int)model.BlockCount; l++)
                foreach (var (t, peft) in peftName)
                {
                    var info = model.Tensors.First(x => x.Name == $"blk.{l}.{t}.weight");
                    int fin = (int)info.Dimensions[0], fout = (int)info.Dimensions[1];
                    string b = $"base_model.model.model.layers.{l}.{peft}";
                    tensors[b + ".lora_A.weight"] = Enumerable.Range(0, rank * fin).Select(_ => (float)((rng.NextDouble() * 2 - 1) / Math.Sqrt(fin))).ToArray();
                    tensors[b + ".lora_B.weight"] = Enumerable.Range(0, fout * rank).Select(_ => (float)((rng.NextDouble() * 2 - 1) * 0.05)).ToArray();
                }
            using var lora = new LoraRuntime(accelerator, tensors, rank, alpha);
            bool groupSaved = GgufBatchDecoder.GroupQ8;
            var notes = new List<string>();
            try
            {
                async Task<GgufBatchDecoder> Make(bool grouped)
                {
                    GgufBatchDecoder.GroupQ8 = grouped;
                    m.Stream.Seek(0, SeekOrigin.Begin);
                    return await GgufBatchDecoder.CreateAsync(accelerator, m.Stream, model, 256, 1);
                }
                using var g = await Make(true);
                using var n = await Make(false);
                foreach (int len in new[] { 1, 5, 40 })
                {
                    var ids = Enumerable.Range(0, len).Select(i => (i * 131 + 17) % g.Vocab).ToArray();
                    async Task<float[]> Logits(GgufBatchDecoder d, LoraRuntime? l) { d.Lora = l; await d.PrefillAsync(ids, 0f, false); return await d.ReadLogitsAsync(); }
                    var lg = await Logits(g, lora); var ln = await Logits(n, lora); var lb = await Logits(n, null);
                    float scale = ln.Max(Math.Abs), diff = 0, moved = 0;
                    for (int i = 0; i < ln.Length; i++) { diff = Math.Max(diff, Math.Abs(lg[i] - ln[i])); moved = Math.Max(moved, Math.Abs(ln[i] - lb[i])); }
                    notes.Add($"{len} tokens: grouped vs native max |diff| {diff:E2} (max |logit| {scale:F1}); the adapter moves them {moved:F2}");
                    if (moved < 0.05f * scale) throw new Exception($"{len} tokens: the adapter barely moves the logits ({moved:E2}): the comparison would prove nothing");
                    if (diff > 2e-3f * scale) throw new Exception($"{accelerator.AcceleratorType}: LoRA over Q8_0G differs from the native layout: " + string.Join("; ", notes));
                }
            }
            finally { GgufBatchDecoder.GroupQ8 = groupSaved; }
            var line = $"{accelerator.AcceleratorType}: " + string.Join("; ", notes);
            Console.WriteLine("[Training] " + line);
            return line;
        }
    });

    /// <summary>
    /// The fused LoraLinear / LoraLinearGrad ops (GraphAutograd.FusedLora, ranks that are a multiple of 8) against the four-op form (MatMul, MatMul,
    /// Mul, Add and its autograd): the same loss, logits and every adapter gradient, at ranks 8 and 16 on all seven projection types, with and
    /// without layer checkpointing. Both are fp32; only the summation order differs. Also GraphAutograd.FuseGradientSums against the Add
    /// chain, bitwise. Tiny F16 llama. Added 2026-11-05.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task Training_FusedLora_MatchesFourOpForm() => await RunTest(async accelerator =>
    {
        var assets = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var gguf = await assets.GetByteArrayAsync("test-train/tiny-llama-f16.gguf");
        var targets = new[] { "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down" };
        var notes = new List<string>();
        bool savedSums = GraphAutograd.FuseGradientSums;
        bool saved = GraphAutograd.FusedLora, savedTcFwd = GraphAutograd.TensorCoreForwardAttention, savedTcBwd = BackwardKernels.UseTensorCoreAttentionBackward;
        try
        {
            // fp32 attention on every backend: the bf16 tensor-core attention backward rounds dO, and a 1e-7 difference upstream (summation order)
            // flips some of those roundings, which reads as ~1e-3 in dV: the comparison would measure that, not the LoRA math
            GraphAutograd.TensorCoreForwardAttention = false;
            BackwardKernels.UseTensorCoreAttentionBackward = false;
            foreach (var (rank, checkpoint) in new[] { (8, false), (16, true) })
            {
                var rng = new Random(11 + rank);
                Dictionary<string, float[]>? adapters = null;
                int[] ids = Array.Empty<int>(), labels = Array.Empty<int>();
                async Task<(float Loss, float[] Logits, Dictionary<string, float[]> Grads)> Run(bool fused, bool fuseSums = true)
                {
                    GraphAutograd.FusedLora = fused;
                    GraphAutograd.FuseGradientSums = fuseSums;
                    using var trainer = GgufLoraTrainer.Create(accelerator, gguf, new LoraOptions { Targets = targets, Rank = rank, Alpha = 2 * rank + 3, Seed = 4, CheckpointLayers = checkpoint });
                    if (adapters == null)
                    {
                        // B nonzero (PEFT starts it at zero: the adapters would be a no-op and dA zero)
                        adapters = await trainer.GetAdaptersAsync();
                        foreach (var k in adapters.Keys.Where(k => k.EndsWith("lora_B.weight")).ToList())
                            adapters[k] = adapters[k].Select(_ => (float)((rng.NextDouble() * 2 - 1) * 0.3)).ToArray();
                        ids = Enumerable.Range(0, 37).Select(_ => rng.Next(0, trainer.VocabSize)).ToArray();
                        labels = ids.Select((_, t) => t < ids.Length - 1 ? ids[t + 1] : -100).ToArray();
                    }
                    await trainer.SetAdaptersAsync(adapters);
                    float loss = await trainer.StepAsync(ids, labels, update: false);
                    return (loss, await trainer.GetLogitsAsync(ids.Length - 1), await trainer.GetGradientsAsync());
                }
                var want = await Run(false);
                var got = await Run(true);
                // the gradient sums folded into MatMulGradInput / LoraLinearGrad (FuseGradientSums) write part + sum, the value the Add wrote: bitwise
                var adds = await Run(true, fuseSums: false);
                if (BitConverter.SingleToInt32Bits(adds.Loss) != BitConverter.SingleToInt32Bits(got.Loss) || !adds.Logits.SequenceEqual(got.Logits))
                    throw new Exception($"rank {rank}: FuseGradientSums changed the forward ({got.Loss:R} vs {adds.Loss:R})");
                foreach (var (k, g) in adds.Grads)
                    for (int i = 0; i < g.Length; i++)
                        if (BitConverter.SingleToInt32Bits(g[i]) != BitConverter.SingleToInt32Bits(got.Grads[k][i]))
                            throw new Exception($"rank {rank}: FuseGradientSums is not bit-identical: {k}[{i}] = {got.Grads[k][i]:R}, with Add nodes {g[i]:R}");
                double Rel(float[] a, float[] b) { double m = b.Max(x => Math.Abs((double)x)), d = 0; for (int i = 0; i < a.Length; i++) d = Math.Max(d, Math.Abs(a[i] - b[i])); return d / Math.Max(m, 1e-30); }
                double lossRel = Math.Abs(got.Loss - want.Loss) / Math.Abs(want.Loss), logitRel = Rel(got.Logits, want.Logits);
                double worst = 0; string worstKey = "";
                foreach (var (k, g) in want.Grads)
                {
                    if (!got.Grads.TryGetValue(k, out var gg)) throw new Exception($"rank {rank}: fused graph has no gradient for {k}");
                    if (g.All(v => v == 0f)) throw new Exception($"rank {rank}: the four-op gradient of {k} is all zero: the comparison proves nothing");
                    double r = Rel(gg, g);
                    if (!(r <= worst)) { worst = r; worstKey = k; }
                }
                notes.Add($"rank {rank}{(checkpoint ? " checkpointed" : "")}: loss {want.Loss:F5} (rel {lossRel:E1}), logits {logitRel:E1}, {want.Grads.Count} gradients worst {worst:E1} ({worstKey})");
                if (!(lossRel <= 1e-5 && logitRel <= 1e-5 && worst <= 1e-4)) throw new Exception("fused LoRA differs from the four-op form: " + string.Join("; ", notes));
            }
        }
        finally { GraphAutograd.FusedLora = saved; GraphAutograd.FuseGradientSums = savedSums; GraphAutograd.TensorCoreForwardAttention = savedTcFwd; BackwardKernels.UseTensorCoreAttentionBackward = savedTcBwd; }
        var line = $"{accelerator.AcceleratorType}: fused LoraLinear vs four-op form: " + string.Join("; ", notes);
        Console.WriteLine("[Training] " + line);
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
