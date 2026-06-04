using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Fast, deterministic unit tests for the token sampling primitives that drive text generation
/// (greedy / top-k / top-p / temperature / repetition penalty). These run on synthetic logits — no
/// model, no GPU, no network — so they are the PRIMARY correctness proof for the sampler and run in
/// the normal fast loop (not HeavyModel). The end-to-end pipeline wiring (config → sampler) is proven
/// separately by the HeavyModel TextGen_Sampling_EscapesGreedy test, which only needs to confirm the
/// config flows through, not re-verify the math here.
///
/// Why this matters: the /text-gen demo's Strategy/Temperature/Top-K/Top-P controls were wired into a
/// sampler that the pipeline never called (it hardcoded greedy argmax), so DistilGPT-2 degenerated into
/// "the first time I saw the first time I saw" loops. These tests guard the sampler the pipeline now uses.
/// </summary>
public abstract partial class MLTestBase
{
    // Greedy must return the index of the maximum logit, full stop.
    [TestMethod(Category = "Sampling")]
    public Task Sampler_Greedy_PicksArgmax()
    {
        var logits = new[] { 0.1f, 3.0f, -1.0f, 2.9f, -5.0f };
        int idx = TextGenerationSampler.Greedy(logits);
        if (idx != 1) throw new Exception($"Greedy returned {idx}, expected 1 (the argmax).");
        return Task.CompletedTask;
    }

    // Top-K with k=1 has exactly one candidate (the argmax) → it MUST collapse to greedy, regardless of
    // the RNG. Proves the top-k path degrades correctly and shares greedy's argmax.
    [TestMethod(Category = "Sampling")]
    public Task Sampler_TopK_K1_EqualsGreedy()
    {
        var logits = new[] { 1.0f, 5.0f, 2.0f, 3.0f };
        int greedy = TextGenerationSampler.Greedy(logits);
        for (int seed = 0; seed < 8; seed++)
        {
            int k1 = TextGenerationSampler.TopK(logits, k: 1, temperature: 0.8f, rng: new Random(seed));
            if (k1 != greedy) throw new Exception($"TopK(k=1) returned {k1} (seed {seed}), expected greedy {greedy}.");
        }
        return Task.CompletedTask;
    }

    // Top-P with a tiny p collapses the nucleus to just the single most-likely token → deterministic
    // argmax regardless of RNG. Proves nucleus selection is correct at the boundary.
    [TestMethod(Category = "Sampling")]
    public Task Sampler_TopP_TinyP_ReturnsArgmax()
    {
        var logits = new[] { 1.0f, 5.0f, 2.0f, 3.0f };
        int greedy = TextGenerationSampler.Greedy(logits); // index 1
        for (int seed = 0; seed < 8; seed++)
        {
            int tp = TextGenerationSampler.TopP(logits, p: 0.01f, temperature: 1.0f, rng: new Random(seed));
            if (tp != greedy) throw new Exception($"TopP(p=0.01) returned {tp} (seed {seed}), expected argmax {greedy}.");
        }
        return Task.CompletedTask;
    }

    // Same seed → same RNG stream → identical token sequence. This is the determinism that lets a seeded
    // generation be reproduced (the GenerationConfig.Seed contract the pipeline relies on for testing).
    [TestMethod(Category = "Sampling")]
    public Task Sampler_TopP_SameSeed_Reproducible()
    {
        var logits = new[] { 0.5f, 1.5f, 0.2f, 2.0f, 1.0f, -0.5f, 0.8f, 1.2f };
        var a = new Random(1234);
        var b = new Random(1234);
        for (int i = 0; i < 25; i++)
        {
            int ta = TextGenerationSampler.TopP(logits, p: 0.9f, temperature: 0.8f, rng: a);
            int tb = TextGenerationSampler.TopP(logits, p: 0.9f, temperature: 0.8f, rng: b);
            if (ta != tb) throw new Exception($"TopP not reproducible at step {i}: {ta} vs {tb} with identical seeds.");
        }
        return Task.CompletedTask;
    }

    // Top-P over a FLAT distribution with p=1.0 must actually SAMPLE (cover many tokens), not collapse to
    // a single index — otherwise the "sampling" is really just argmax and the UI controls would be dead.
    [TestMethod(Category = "Sampling")]
    public Task Sampler_TopP_FlatDistribution_SamplesMany()
    {
        var logits = new float[8]; // all zero → uniform after softmax
        var rng = new Random(7);
        var seen = new HashSet<int>();
        for (int i = 0; i < 300; i++)
            seen.Add(TextGenerationSampler.TopP(logits, p: 1.0f, temperature: 1.0f, rng: rng));
        if (seen.Count < 2)
            throw new Exception($"TopP over a flat distribution only ever returned {seen.Count} distinct token(s) — it is not sampling.");
        return Task.CompletedTask;
    }

    // PERF GUARD - the exact bug that hung the E2E sampling test. Top-p/top-k must NOT do a LINQ
    // OrderByDescending over the full vocabulary: in interpreted Blazor WASM (RunAOTCompilation=false)
    // that was ~tens of seconds PER call and hung multi-token sampling. At a realistic ~50k vocab, many
    // sampling calls must finish well inside this 60s timeout. If the slow path is reintroduced, THIS
    // test fails (timeout) in the fast loop - cheap to catch - instead of only surfacing as an 8-minute
    // HeavyModel E2E timeout. (This runs in the same interpreted-WASM runtime as production decode.)
    [TestMethod(Timeout = 60000, Category = "Sampling")]
    public Task Sampler_RealVocabSize_IsFast()
    {
        const int vocab = 50257; // GPT-2 / DistilGPT-2 vocabulary
        var logits = new float[vocab];
        for (int i = 0; i < vocab; i++) logits[i] = MathF.Sin(i * 0.001f); // deterministic, non-degenerate
        logits[12345] = 50f; // a clear probability peak

        var rng = new Random(99);
        for (int call = 0; call < 16; call++)
        {
            int tp = TextGenerationSampler.TopP(logits, p: 0.9f, temperature: 0.7f, rng: rng);
            int tk = TextGenerationSampler.TopK(logits, k: 50, temperature: 0.7f, rng: rng);
            if (tp < 0 || tp >= vocab) throw new Exception($"TopP returned out-of-range index {tp}.");
            if (tk < 0 || tk >= vocab) throw new Exception($"TopK returned out-of-range index {tk}.");
        }

        // Correctness alongside speed: with one logit dominating, a low-temperature nucleus collapses to
        // that peak, so the draw MUST be the peak index (also proves the argsort ordered correctly).
        int peak = TextGenerationSampler.TopP(logits, p: 0.5f, temperature: 0.1f, rng: new Random(1));
        if (peak != 12345) throw new Exception($"TopP with a dominant peak returned {peak}, expected 12345.");
        return Task.CompletedTask;
    }

    // Repetition penalty: divides positive logits and multiplies negative logits for already-seen tokens
    // (pushing both toward less-likely), and leaves unseen tokens untouched.
    [TestMethod(Category = "Sampling")]
    public Task Sampler_RepetitionPenalty_PenalizesSeenTokens()
    {
        var logits = new[] { 2.0f, -2.0f, 0.5f };
        TextGenerationSampler.ApplyRepetitionPenalty(logits, previousTokens: new[] { 0, 1 }, penalty: 1.5f);
        if (MathF.Abs(logits[0] - (2.0f / 1.5f)) > 1e-5f)
            throw new Exception($"positive seen logit not divided by penalty: got {logits[0]}, expected {2.0f / 1.5f}.");
        if (MathF.Abs(logits[1] - (-2.0f * 1.5f)) > 1e-5f)
            throw new Exception($"negative seen logit not multiplied by penalty: got {logits[1]}, expected {-2.0f * 1.5f}.");
        if (MathF.Abs(logits[2] - 0.5f) > 1e-6f)
            throw new Exception($"unseen logit was modified: got {logits[2]}, expected 0.5 (unchanged).");
        return Task.CompletedTask;
    }
}
