using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// End-to-end WebGPU (+ all-backend) gate for the NEW GGUF architectures being wired into the
/// SpawnDev.AI demo: qwen3 (standard transformer + useRMSNorm) and LFM2 (ShortConv hybrid).
/// Each loads a small instruct GGUF via the hub, runs a real greedy decode, asserts the factual
/// oracle (" Paris"), and then gates COHERENCE on a second, open-ended generation.
///
/// ⚠ READ BEFORE TRUSTING A PASS HERE (2026-07-16): this test being green NEVER proved the arch
/// works. Its oracle was Contains("Paris") on a 16-64 token greedy answer - and a broken decode
/// emits the strong factual association FIRST, then degenerates. LFM2 shipped twice on that green
/// (a missing BOS, then a WebGPU-only defect) while the demo emitted token soup for the Captain.
/// It is also Category=HeavyModel, i.e. EXCLUDED from the default sweep - so it usually does not
/// even run. A pass here is NOT a substitute for the kernel-level CPU-oracle tests
/// (MLTestBase.ShortConvTests) or the per-node cross-backend golden (MLTestBase.Lfm2TrajectoryTests),
/// which localize a defect instead of averaging over it. Do not write "WebGPU-verified" on the
/// strength of this file alone.
/// HeavyModel (excluded from the default CI sweep;
/// run manually with PMT_EXCLUDE_CATEGORIES=). The CPU-desktop lane is the unoptimized correctness
/// REFERENCE (~12s/token) and is killed by PMT's 10-min console cap on multi-hundred-M LLM decode -
/// an expected backend impracticality (same class as WasmHeavy), NOT a correctness gap; raise
/// PMT_CONSOLE_TIMEOUT_MS to run the CPU E2E deliberately.
/// </summary>
public abstract partial class MLTestBase
{
    // Shared driver: stream a GGUF instruct model from the hub, greedy-decode the France prompt,
    // assert the answer mentions Paris. Network/hub outage → UnsupportedTestException (not a failure).
    private async Task GgufLLM_AnswersParis(Accelerator accelerator, string repoId, string file, string tag,
        string userSuffix = "", int maxTokens = 64)
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var client = new SpawnDev.WebTorrent.WebTorrentClient();
        try
        {
            var hub = new SpawnDev.ILGPU.ML.Hub.HubModelStream(client, http) { PrepareTimeout = TimeSpan.FromMinutes(8) };
            using var cts = new System.Threading.CancellationTokenSource(TimeSpan.FromMinutes(9));

            var model = await hub.OpenAsync(repoId, file, deselect: false, cts.Token);
            if (model.Length < 100_000_000)
                throw new Exception($"[{tag}] hub GGUF length={model.Length}, expected a real model");

            await using (model.Stream)
            using (var pipe = await GgufTextGenerationPipeline.CreateFromStreamAsync(
                accelerator, model.Stream, maxSeqLen: 512, ct: cts.Token))
            {
                Console.WriteLine($"[{tag}] loaded arch={pipe.Architecture} format={pipe.ChatFormat}");
                var messages = new[] { ("user", "What is the capital of France? Answer in one short sentence." + userSuffix) };
                var answer = await pipe.GenerateAsync(messages,
                    config: new GenerationConfig { MaxNewTokens = maxTokens, Strategy = "greedy" }, ct: cts.Token);
                Console.WriteLine($"[{tag}] answer='{answer.Trim()}'");

                if (string.IsNullOrWhiteSpace(answer))
                    throw new Exception($"[{tag}] produced empty output");
                if (!answer.Contains("Paris", StringComparison.OrdinalIgnoreCase))
                    throw new Exception($"[{tag}] answer did not mention Paris (oracle): '{answer.Trim()}'");

                // ── COHERENCE gate ── Contains("Paris") passes on TOKEN SOUP: a broken decode emits the strong
                // factual association first, THEN degenerates. That hole shipped LFM2 twice (missing BOS, then a
                // WebGPU-only defect) while this test was green. The previous gate was inert: it required >=24
                // words but the ONLY prompt asked for "one short sentence" (maxTokens 48-64), so it never ran.
                // So: ask a SECOND, open-ended question that forces a long answer, on the already-loaded model,
                // and gate its trigram diversity. Degenerate text repeats trigrams; healthy prose stays >0.5.
                var longAnswer = await pipe.GenerateAsync(
                    new[] { ("user", "In one paragraph, explain what a chicken is and what it is farmed for.") },
                    config: new GenerationConfig { MaxNewTokens = 160, Strategy = "greedy" }, ct: cts.Token);
                Console.WriteLine($"[{tag}] longAnswer='{longAnswer.Trim()}'");
                var words = longAnswer.ToLowerInvariant()
                    .Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                // Too short to judge = the gate could not run. Fail rather than silently skip (the old bug).
                if (words.Length < 20)
                    throw new Exception($"[{tag}] coherence gate could not run: open-ended prompt produced only " +
                        $"{words.Length} words (need >=20). Answer: '{longAnswer.Trim()}'");
                var tri = new List<string>();
                for (int i = 0; i + 2 < words.Length; i++) tri.Add($"{words[i]} {words[i + 1]} {words[i + 2]}");
                double uniqueRatio = (double)tri.Distinct().Count() / tri.Count;
                // Stopword rate: token soup is mostly punctuation/fragments and loses normal English function
                // words, which trigram diversity alone does not catch (random fragments are all "unique").
                string[] stop = { "a", "an", "the", "is", "are", "of", "to", "and", "in", "for", "it", "that", "they" };
                double stopRate = (double)words.Count(w => stop.Contains(w.Trim('.', ',', ':', ';', '*', '#', '(', ')'))) / words.Length;
                Console.WriteLine($"[{tag}] words={words.Length} uniqueTrigramRatio={uniqueRatio:F2} stopwordRate={stopRate:F2}");
                if (uniqueRatio < 0.5)
                    throw new Exception($"[{tag}] degenerate output (unique-trigram ratio {uniqueRatio:F2} < 0.5) " +
                        $"- coherent-generation gate failed. Answer: '{longAnswer.Trim()}'");
                if (stopRate < 0.10)
                    throw new Exception($"[{tag}] degenerate output (stopword rate {stopRate:F2} < 0.10 - fragmented " +
                        $"token soup, not English) - coherent-generation gate failed. Answer: '{longAnswer.Trim()}'");
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network")
            || ex.Message.Contains("magnet") || ex.Message.Contains("preparing") || ex is TimeoutException)
        {
            throw new UnsupportedTestException($"[{tag}] hub/network unavailable: {ex.Message}");
        }
        finally { await client.DisposeAsync(); }
    }

    // qwen3 (0.6B, Q8_0) - standard transformer arch (shares qwen2.5's op set + the useRMSNorm fix).
    // qwen3 is a REASONING model (emits <think>…</think> first); "/no_think" gives the direct answer so the
    // oracle lands quickly. Verified: the WGSL forward is coherent + bit-consistent across all backends.
    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy,HeavyCpu", RetryCount = 2)]
    public async Task Pipeline_Qwen3_ViaHubStream_AnswersParis() => await RunTest(async accelerator =>
        await GgufLLM_AnswersParis(accelerator, "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf", "Qwen3",
            userSuffix: " /no_think", maxTokens: 48));

    // LFM2 (1.2B instruct, Q4_K_M) - ShortConv hybrid. The WebGPU pass proves ShortConv's WGSL in-browser.
    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy,HeavyCpu", RetryCount = 2)]
    public async Task Pipeline_Lfm2_ViaHubStream_AnswersParis() => await RunTest(async accelerator =>
        await GgufLLM_AnswersParis(accelerator, "LiquidAI/LFM2-1.2B-GGUF", "LFM2-1.2B-Q4_K_M.gguf", "LFM2"));
}
